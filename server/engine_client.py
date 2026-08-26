from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import AsyncIterator, List, Optional, Sequence

from engine.detokenizer import IncrementalDetokenizer, StopStringMatcher
from engine.engine import Engine
from engine.request import FinishReason, Request, RequestEvent
from engine.utils import SamplingParams

logger = logging.getLogger(__name__)

_SENTINEL = object()


@dataclass
class StreamDelta:
    text: str
    token_id: Optional[int]
    finished: bool = False
    finish_reason: Optional[str] = None
    num_output_tokens: int = 0
    num_cached_prompt_tokens: int = 0


class EngineClient:
    def __init__(self, engine: Engine, tokenizer, model_name: str, idle_sleep: float = 0.0005) -> None:
        self.engine = engine
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.idle_sleep = idle_sleep

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._counter = 0
        self._counter_lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="velox-engine", daemon=True)
        self._thread.start()
        logger.info("engine worker started")

    def shutdown(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("engine worker stopped")

    def _run(self) -> None:
        import torch

        with torch.no_grad():
            while not self._stop.is_set():
                if self.engine.has_work():
                    try:
                        self.engine.step()
                    except Exception:
                        logger.exception("engine step failed; aborting in-flight requests")
                        self._abort_all()
                else:
                    time.sleep(self.idle_sleep)

    def _abort_all(self) -> None:
        try:
            self.engine.abort_all()
        except Exception:
            logger.exception("failed to abort in-flight requests")

    def next_request_id(self, prefix: str) -> str:
        with self._counter_lock:
            self._counter += 1
            return f"{prefix}-{self._counter}"

    async def generate(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        sampling_params: SamplingParams,
        stop_strings: Optional[Sequence[str]] = None,
    ) -> AsyncIterator[StreamDelta]:
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        request = Request(
            request_id=request_id,
            prompt_token_ids=list(prompt_token_ids),
            sampling_params=sampling_params,
        )

        def on_event(_request: Request, event: RequestEvent) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        request.add_listener(on_event)
        self.engine.submit(request)

        detokenizer = IncrementalDetokenizer(self.tokenizer, prompt_token_ids)
        matcher = StopStringMatcher(stop_strings or [])
        finished = False

        try:
            while True:
                event: RequestEvent = await queue.get()

                if event.kind == "token":
                    delta = detokenizer.append(event.token_id)
                    text, hit_stop = matcher.feed(delta)
                    if text:
                        yield StreamDelta(text=text, token_id=event.token_id)
                    if hit_stop:
                        finished = True
                        self.engine.request_abort(request_id)
                        yield self._final(request, "stop")
                        return
                    continue

                finished = True
                pending = matcher.flush()
                if pending:
                    yield StreamDelta(text=pending, token_id=None)
                reason = event.finish_reason or FinishReason.STOP
                yield self._final(request, reason.value)
                return
        finally:
            if not finished:
                self.engine.request_abort(request_id)

    @staticmethod
    def _final(request: Request, finish_reason: str) -> StreamDelta:
        return StreamDelta(
            text="",
            token_id=None,
            finished=True,
            finish_reason=finish_reason,
            num_output_tokens=request.num_output_tokens,
            num_cached_prompt_tokens=request.num_cached_tokens,
        )

    async def collect(
        self,
        request_id: str,
        prompt_token_ids: Sequence[int],
        sampling_params: SamplingParams,
        stop_strings: Optional[Sequence[str]] = None,
    ) -> StreamDelta:
        chunks: List[str] = []
        final = StreamDelta(text="", token_id=None, finished=True, finish_reason="stop")
        async for delta in self.generate(
            request_id, prompt_token_ids, sampling_params, stop_strings
        ):
            if delta.text:
                chunks.append(delta.text)
            if delta.finished:
                final = delta
        final.text = "".join(chunks)
        return final
