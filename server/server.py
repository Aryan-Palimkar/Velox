from __future__ import annotations

import argparse
import logging
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, List, Optional

from fastapi import FastAPI, HTTPException, Request as HttpRequest
from fastapi.responses import JSONResponse, StreamingResponse

from engine.config import EngineConfig
from engine.engine import Engine
from engine.utils import SamplingParams
from server.engine_client import EngineClient
from server.protocol import (
    ChatCompletionChoice,
    ChatCompletionChunk,
    ChatCompletionChunkChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    CompletionChoice,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    DeltaMessage,
    ErrorDetail,
    ErrorResponse,
    ModelCard,
    ModelList,
    UsageInfo,
    _request_id,
)

logger = logging.getLogger(__name__)

client: Optional[EngineClient] = None
engine_config: Optional[EngineConfig] = None

_DONE = "data: [DONE]\n\n"


def _sse(payload) -> str:
    body = payload if isinstance(payload, str) else payload.model_dump_json(exclude_none=True)
    return f"data: {body}\n\n"


def _sampling_params(request, default_max_tokens: int) -> SamplingParams:
    return SamplingParams.from_optional(
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        presence_penalty=request.presence_penalty,
        frequency_penalty=request.frequency_penalty,
        repetition_penalty=request.repetition_penalty,
        max_tokens=request.max_tokens or default_max_tokens,
        seed=request.seed,
    )


def _reject_unsupported(request) -> None:
    unsupported = request.unsupported_fields()
    if unsupported:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported parameters: {', '.join(unsupported)}",
        )


def _require_client() -> EngineClient:
    if client is None:
        raise HTTPException(status_code=503, detail="engine is still starting")
    return client


def build_app(config: EngineConfig, served_model_name: Optional[str] = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        global client, engine_config
        from transformers import AutoTokenizer

        engine_config = config
        logger.info("loading %s", config.model)
        tokenizer = AutoTokenizer.from_pretrained(config.model, cache_dir=config.download_dir)
        engine = Engine.from_pretrained(config)
        client = EngineClient(
            engine=engine,
            tokenizer=tokenizer,
            model_name=served_model_name or config.model,
        )
        client.start()
        logger.info("velox is serving %s", client.model_name)
        try:
            yield
        finally:
            client.shutdown()
            client = None

    app = FastAPI(title="Velox Inference Server", version="1.0", lifespan=lifespan)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: HttpRequest, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=ErrorDetail(message=str(exc.detail))).model_dump(),
        )

    @app.get("/health")
    async def health():
        if client is None:
            raise HTTPException(status_code=503, detail="engine is still starting")
        return {"status": "ok", "model": client.model_name}

    @app.get("/v1/models")
    async def list_models():
        served = _require_client()
        return ModelList(data=[ModelCard(id=served.model_name)])

    @app.get("/metrics")
    async def metrics():
        served = _require_client()
        engine = served.engine
        snapshot = engine.snapshot()
        for key, values in engine.stats.items():
            if values:
                window = values[-256:]
                snapshot[f"{key}_mean"] = sum(window) / len(window)
        return snapshot

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest, http_request: HttpRequest):
        served = _require_client()
        _reject_unsupported(body)

        messages = [message.model_dump() for message in body.messages]
        try:
            prompt = served.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=body.add_generation_prompt
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"failed to apply chat template: {exc}")

        token_ids = served.tokenizer.encode(prompt)
        params = _sampling_params(body, default_max_tokens=512)
        _validate_limits(served, len(token_ids), params.max_tokens)

        request_id = _request_id("chatcmpl")
        stop_strings = body.stop_strings()
        include_usage = bool(body.stream_options and body.stream_options.include_usage)

        if not body.stream:
            final = await served.collect(request_id, token_ids, params, stop_strings)
            return ChatCompletionResponse(
                id=request_id,
                model=served.model_name,
                choices=[
                    ChatCompletionChoice(
                        message=ChatCompletionResponseMessage(content=final.text),
                        finish_reason=final.finish_reason,
                    )
                ],
                usage=_usage(len(token_ids), final),
            )

        async def stream() -> AsyncIterator[str]:
            created = int(time.time())

            def chunk(delta: DeltaMessage, finish_reason: Optional[str] = None) -> ChatCompletionChunk:
                return ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=served.model_name,
                    choices=[ChatCompletionChunkChoice(delta=delta, finish_reason=finish_reason)],
                )

            yield _sse(chunk(DeltaMessage(role="assistant")))

            final = None
            async for piece in served.generate(request_id, token_ids, params, stop_strings):
                if piece.text:
                    yield _sse(chunk(DeltaMessage(content=piece.text)))
                if piece.finished:
                    final = piece
                    yield _sse(chunk(DeltaMessage(), finish_reason=piece.finish_reason))

            if include_usage and final is not None:
                usage_chunk = ChatCompletionChunk(
                    id=request_id,
                    created=created,
                    model=served.model_name,
                    choices=[],
                    usage=_usage(len(token_ids), final),
                )
                yield _sse(usage_chunk)
            yield _DONE

        return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)

    @app.post("/v1/completions")
    async def completions(body: CompletionRequest, http_request: HttpRequest):
        served = _require_client()
        _reject_unsupported(body)
        try:
            prompt = body.single_prompt()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        token_ids = served.tokenizer.encode(prompt)
        params = _sampling_params(body, default_max_tokens=128)
        _validate_limits(served, len(token_ids), params.max_tokens)

        request_id = _request_id("cmpl")
        stop_strings = body.stop_strings()
        include_usage = bool(body.stream_options and body.stream_options.include_usage)

        if not body.stream:
            final = await served.collect(request_id, token_ids, params, stop_strings)
            text = (prompt + final.text) if body.echo else final.text
            return CompletionResponse(
                id=request_id,
                model=served.model_name,
                choices=[CompletionChoice(text=text, finish_reason=final.finish_reason)],
                usage=_usage(len(token_ids), final),
            )

        async def stream() -> AsyncIterator[str]:
            created = int(time.time())
            if body.echo:
                yield _sse(
                    CompletionChunk(
                        id=request_id,
                        created=created,
                        model=served.model_name,
                        choices=[CompletionChoice(text=prompt)],
                    )
                )

            final = None
            async for piece in served.generate(request_id, token_ids, params, stop_strings):
                if piece.text:
                    yield _sse(
                        CompletionChunk(
                            id=request_id,
                            created=created,
                            model=served.model_name,
                            choices=[CompletionChoice(text=piece.text)],
                        )
                    )
                if piece.finished:
                    final = piece
                    yield _sse(
                        CompletionChunk(
                            id=request_id,
                            created=created,
                            model=served.model_name,
                            choices=[
                                CompletionChoice(text="", finish_reason=piece.finish_reason)
                            ],
                        )
                    )

            if include_usage and final is not None:
                yield _sse(
                    CompletionChunk(
                        id=request_id,
                        created=created,
                        model=served.model_name,
                        choices=[],
                        usage=_usage(len(token_ids), final),
                    )
                )
            yield _DONE

        return StreamingResponse(stream(), media_type="text/event-stream", headers=_SSE_HEADERS)

    return app


_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _usage(num_prompt_tokens: int, final) -> UsageInfo:
    return UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=final.num_output_tokens,
        total_tokens=num_prompt_tokens + final.num_output_tokens,
        prompt_cached_tokens=final.num_cached_prompt_tokens,
    )


def _validate_limits(served: EngineClient, num_prompt_tokens: int, max_tokens: int) -> None:
    try:
        served.engine.check_limits(num_prompt_tokens, max_tokens)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a Qwen2 model with Velox")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--served-model-name", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-prefill-chunk-size", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-gpu-blocks", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8", "fp8_e5m2", "int8"])
    parser.add_argument("--quantization", default=None, choices=[None, "int8", "fp8"])
    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--no-cuda-graphs", action="store_true")
    parser.add_argument("--download-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-level", default="info")
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace) -> EngineConfig:
    return EngineConfig(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_prefill_chunk_size=args.max_prefill_chunk_size,
        block_size=args.block_size,
        num_gpu_blocks=args.num_gpu_blocks,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization=args.quantization,
        enable_prefix_caching=not args.no_prefix_caching,
        enable_cuda_graphs=not args.no_cuda_graphs,
        download_dir=args.download_dir,
        seed=args.seed,
    )


def main(argv: Optional[List[str]] = None) -> None:
    import uvicorn

    args = parse_args(argv)
    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = build_app(config_from_args(args), served_model_name=args.served_model_name)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
