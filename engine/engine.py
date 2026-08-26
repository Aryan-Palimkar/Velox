from __future__ import annotations

import logging
import queue
import time
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence

import torch

from src.block_manager import BlockSpaceManager
from src.paged_kv_cache import PagedKVCache, compute_num_blocks
from src.transformer import QwenForCausalLM

from .config import EngineConfig
from .model_runner import ModelRunner
from .request import FinishReason, Request
from .sampler import IndexStaging, Sampler, TokenHistory
from .scheduler import Scheduler, SchedulerConfig, SchedulerOutput
from .utils import SamplingParams

logger = logging.getLogger(__name__)

DEFAULT_EOS_TOKEN_IDS = frozenset({151643, 151645})


class Engine:
    def __init__(
        self,
        model: QwenForCausalLM,
        kv_cache: PagedKVCache,
        config: EngineConfig,
        eos_token_ids: Optional[Iterable[int]] = None,
    ) -> None:
        config.validate()
        self.config = config
        self.device = torch.device(config.device)
        self.model = model
        self.kv_cache = kv_cache
        self.eos_token_ids: FrozenSet[int] = frozenset(
            eos_token_ids if eos_token_ids is not None else DEFAULT_EOS_TOKEN_IDS
        )

        self.block_manager = BlockSpaceManager(
            kv_cache,
            enable_prefix_caching=config.enable_prefix_caching,
            max_model_len=config.max_model_len,
        )
        self.scheduler = Scheduler(
            kv_cache=kv_cache,
            config=SchedulerConfig(
                max_num_seqs=config.max_num_seqs,
                max_num_batched_tokens=config.max_num_batched_tokens,
                max_prefill_chunk_size=config.max_prefill_chunk_size,
                max_model_len=config.max_model_len,
                enable_prefix_caching=config.enable_prefix_caching,
            ),
            block_manager=self.block_manager,
        )
        self.max_model_len = self.scheduler.config.max_model_len

        self.runner = ModelRunner(
            model=model,
            kv_cache=kv_cache,
            block_manager=self.block_manager,
            max_num_seqs=config.max_num_seqs,
            max_num_batched_tokens=config.max_num_batched_tokens,
            device=self.device,
            enable_cuda_graphs=config.enable_cuda_graphs,
        )
        self.sampler = Sampler(device=self.device, max_num_seqs=config.max_num_seqs)
        self.token_history = TokenHistory(
            max_num_seqs=config.max_num_seqs,
            max_model_len=self.max_model_len,
            device=self.device,
        )

        self._logit_row_staging = IndexStaging(config.max_num_seqs, self.device)
        self._seq_index_staging = IndexStaging(config.max_num_seqs, self.device)

        self.scheduler.on_sequence_admitted = self._on_sequence_admitted
        self.scheduler.on_sequence_released = self._on_sequence_released

        if self.device.type == "cuda":
            self.prefill_stream = torch.cuda.Stream(device=self.device)
            self.decode_stream = torch.cuda.Stream(device=self.device)
        else:
            self.prefill_stream = None
            self.decode_stream = None

        self.stats: Dict[str, List[float]] = {
            "scheduler_ms": [],
            "prepare_ms": [],
            "prefill_gpu_ms": [],
            "decode_gpu_ms": [],
            "cpu_sync_wait_ms": [],
            "sampling_ms": [],
            "step_ms": [],
        }
        self.num_steps = 0
        self._graphs_ready = False

        self._pending_adds: "queue.Queue[Request]" = queue.Queue()
        self._pending_aborts: "queue.Queue[str]" = queue.Queue()

    @classmethod
    def from_pretrained(
        cls, config: Optional[EngineConfig] = None, **overrides
    ) -> "Engine":
        from src.config import ModelConfig
        from src.utils import seed_everything
        from src.weight_loader import load_qwen_weights

        config = config or EngineConfig(**overrides)
        config.validate()

        device = torch.device(config.device)
        dtype = config.torch_dtype()
        if config.seed is not None:
            seed_everything(config.seed)

        model_config = ModelConfig.from_hf(config.model, cache_dir=config.download_dir)
        max_model_len = min(config.max_model_len, model_config.max_position_embeddings)
        config.max_model_len = max_model_len

        logger.info("building %s on %s (%s)", config.model, device, dtype)
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            with torch.device(device):
                model = QwenForCausalLM(model_config, max_position=max_model_len)
        finally:
            torch.set_default_dtype(previous_dtype)
        load_qwen_weights(model, model_config, config.model, cache_dir=config.download_dir)
        model.eval()

        if config.quantization:
            from .quantize import quantize_model

            quantize_model(model, config.quantization)

        head_dim = model_config.hidden_size // model_config.num_attention_heads
        num_blocks = config.num_gpu_blocks or compute_num_blocks(
            num_layers=model_config.num_hidden_layers,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=head_dim,
            block_size=config.block_size,
            dtype=dtype,
            gpu_memory_utilization=config.gpu_memory_utilization,
            device=device,
            kv_cache_dtype=config.kv_cache_dtype,
            reserve_bytes=config.kv_cache_reserve_mb * 1024 * 1024,
        )

        kv_cache = PagedKVCache(
            num_layers=model_config.num_hidden_layers,
            num_kv_heads=model_config.num_key_value_heads,
            head_dim=head_dim,
            num_blocks=num_blocks,
            block_size=config.block_size,
            dtype=dtype,
            device=device,
            kv_cache_dtype=config.kv_cache_dtype,
        )

        engine = cls(
            model=model,
            kv_cache=kv_cache,
            config=config,
            eos_token_ids=_eos_token_ids(config.model, config.download_dir),
        )
        engine.warmup()
        return engine

    def warmup(self) -> None:
        if self.device.type != "cuda":
            return
        self._warmup_forward()
        self.runner.capture_decode_graphs()
        self._graphs_ready = self.runner.uses_cuda_graphs
        torch.cuda.synchronize()

    @torch.no_grad()
    def _warmup_forward(self) -> None:
        limit = self.max_model_len - 2
        lengths = sorted(
            {
                min(self.config.max_prefill_chunk_size, limit),
                min(64, limit),
                min(8, limit),
            }
        )
        params = SamplingParams(temperature=0.0, max_tokens=2)
        for index, length in enumerate(lengths):
            if length < 1:
                continue
            request = Request(
                request_id=f"__warmup_{index}__",
                prompt_token_ids=[1] * length,
                sampling_params=params,
            )
            self.scheduler.add_request(request)
            guard = 0
            while self.scheduler.has_work() and guard < 128:
                self.step()
                guard += 1
            self.scheduler.abort_request(request.request_id)

        self.block_manager.prefix_cache.clear()
        for values in self.stats.values():
            values.clear()
        self.num_steps = 0

    def add_request(self, request: Request) -> None:
        self.scheduler.add_request(request)

    def abort_request(self, request_id: str) -> bool:
        self.sampler.release(request_id)
        return self.scheduler.abort_request(request_id)

    def abort_all(self) -> str:
        self._drain_control_queue()
        for request_id in list(self.scheduler.requests):
            self.abort_request(request_id)
        return "aborted"

    def check_limits(self, num_prompt_tokens: int, max_tokens: int) -> None:
        limit = self.max_model_len
        if num_prompt_tokens >= limit:
            raise ValueError(
                f"prompt of {num_prompt_tokens} tokens does not fit in "
                f"max_model_len={limit} (at least one token must remain for generation)"
            )
        if num_prompt_tokens + max_tokens > limit:
            raise ValueError(
                f"prompt of {num_prompt_tokens} tokens plus max_tokens={max_tokens} "
                f"exceeds max_model_len={limit}"
            )

    def submit(self, request: Request) -> None:
        self.check_limits(request.num_prompt_tokens, request.sampling_params.max_tokens)
        self._pending_adds.put(request)

    def request_abort(self, request_id: str) -> None:
        self._pending_aborts.put(request_id)

    def _drain_control_queue(self) -> None:
        while True:
            try:
                request = self._pending_adds.get_nowait()
            except queue.Empty:
                break
            try:
                self.scheduler.add_request(request)
            except ValueError:
                logger.exception("rejecting request %s", request.request_id)
                request.finish(FinishReason.ABORT)

        while True:
            try:
                request_id = self._pending_aborts.get_nowait()
            except queue.Empty:
                break
            self.abort_request(request_id)

    def has_work(self) -> bool:
        return self.scheduler.has_work() or not self._pending_adds.empty()

    def _on_sequence_admitted(self, request: Request) -> None:
        self.runner.on_sequence_admitted(request)
        self.token_history.reset_sequence(request.seq_index, request.all_token_ids)

    def _on_sequence_released(self, request: Request) -> None:
        self.runner.on_sequence_released(request)

    @torch.no_grad()
    def step(self) -> None:
        step_start = time.perf_counter()
        self._drain_control_queue()

        t0 = time.perf_counter()
        scheduled = self.scheduler.schedule()
        self.stats["scheduler_ms"].append((time.perf_counter() - t0) * 1000)

        if scheduled.is_empty():
            return

        prefill_batch = scheduled.prefill_batch
        decode_batch = scheduled.decode_batch

        completing: List[Request] = []
        completing_rows: List[int] = []
        for row, req in enumerate(prefill_batch):
            if req.num_computed_tokens + req.current_chunk_size >= len(req.all_token_ids):
                completing.append(req)
                completing_rows.append(row)

        events = _StepEvents(self.device)
        prepare_start = time.perf_counter()

        prefill_hidden = None
        decode_hidden = None

        if prefill_batch:
            with self._stream(self.prefill_stream):
                events.record_prefill_start()
                hidden = self.runner.execute_prefill(prefill_batch)
                if completing:
                    rows = self._logit_row_staging.upload(completing_rows)
                    prefill_hidden = hidden.index_select(0, rows)
                events.record_prefill_end()

        if decode_batch:
            with self._stream(self.decode_stream):
                events.record_decode_start()
                decode_hidden = self.runner.execute_decode(decode_batch)
                events.record_decode_end()

        self.stats["prepare_ms"].append((time.perf_counter() - prepare_start) * 1000)

        current = torch.cuda.current_stream(self.device) if self.device.type == "cuda" else None
        if current is not None:
            if prefill_batch:
                current.wait_stream(self.prefill_stream)
                if prefill_hidden is not None:
                    prefill_hidden.record_stream(current)
            if decode_batch:
                current.wait_stream(self.decode_stream)
                if decode_hidden is not None and not self.runner.uses_cuda_graphs:
                    decode_hidden.record_stream(current)

        sample_start = time.perf_counter()
        sampled_requests = completing + decode_batch
        if not sampled_requests:
            self._advance_partial_prefills(prefill_batch)
            self.num_steps += 1
            self.stats["step_ms"].append((time.perf_counter() - step_start) * 1000)
            return

        pieces = [h for h in (prefill_hidden, decode_hidden) if h is not None]
        hidden_states = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=0)
        logits = self.model.compute_logits(hidden_states)

        seq_indices = self._seq_index_staging.upload(
            [req.seq_index for req in sampled_requests]
        )
        history_width = max(req.num_tokens for req in sampled_requests)
        tokens_gpu = self.sampler.sample(
            logits,
            [req.sampling_params for req in sampled_requests],
            request_ids=[req.request_id for req in sampled_requests],
            history=self.token_history,
            seq_indices=seq_indices,
            history_width=history_width,
        )
        self.token_history.append(seq_indices, tokens_gpu)

        sync_start = time.perf_counter()
        # The one place per step where the host waits on the device.
        tokens = tokens_gpu.tolist()
        self.stats["cpu_sync_wait_ms"].append((time.perf_counter() - sync_start) * 1000)
        self.stats["sampling_ms"].append((time.perf_counter() - sample_start) * 1000)

        events.collect(self.stats, bool(prefill_batch), bool(decode_batch))

        self._advance_partial_prefills(prefill_batch)
        self._apply_tokens(sampled_requests, tokens, num_prefill=len(completing))

        self.num_steps += 1
        self.stats["step_ms"].append((time.perf_counter() - step_start) * 1000)

    def _stream(self, stream):
        if stream is None:
            return _NullContext()
        return torch.cuda.stream(stream)

    def _advance_partial_prefills(self, prefill_batch: Sequence[Request]) -> None:
        for request in prefill_batch:
            request.num_computed_tokens += request.current_chunk_size
            self.scheduler.commit(request)
            request.refresh_status()

    def _apply_tokens(
        self, requests: Sequence[Request], tokens: List[int], num_prefill: int
    ) -> None:
        for position, (request, token) in enumerate(zip(requests, tokens)):
            if position >= num_prefill:
                request.num_computed_tokens += 1
                self.scheduler.commit(request)

            request.append_token(token)
            request.refresh_status()

            reason = request.should_stop(token, self.eos_token_ids)
            if reason is not None:
                request.finish(reason)
                self.sampler.release(request.request_id)

    @torch.no_grad()
    def generate(
        self, requests: Sequence[Request], max_steps: Optional[int] = None
    ) -> List[Request]:
        for request in requests:
            self.add_request(request)
        steps = 0
        while self.has_work():
            self.step()
            steps += 1
            if max_steps is not None and steps >= max_steps:
                break
        return list(requests)

    def snapshot(self) -> Dict[str, float]:
        stats = self.scheduler.stats()
        stats["num_steps"] = float(self.num_steps)
        stats["cuda_graphs"] = 1.0 if self._graphs_ready else 0.0
        stats["kv_cache_bytes"] = float(self.kv_cache.total_bytes())
        return stats


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False


class _StepEvents:
    def __init__(self, device: torch.device) -> None:
        self.enabled = device.type == "cuda"
        if not self.enabled:
            return
        self.prefill_start = torch.cuda.Event(enable_timing=True)
        self.prefill_end = torch.cuda.Event(enable_timing=True)
        self.decode_start = torch.cuda.Event(enable_timing=True)
        self.decode_end = torch.cuda.Event(enable_timing=True)

    def record_prefill_start(self) -> None:
        if self.enabled:
            self.prefill_start.record()

    def record_prefill_end(self) -> None:
        if self.enabled:
            self.prefill_end.record()

    def record_decode_start(self) -> None:
        if self.enabled:
            self.decode_start.record()

    def record_decode_end(self) -> None:
        if self.enabled:
            self.decode_end.record()

    def collect(self, stats: Dict[str, List[float]], had_prefill: bool, had_decode: bool) -> None:
        if not self.enabled:
            return
        if had_prefill:
            stats["prefill_gpu_ms"].append(self.prefill_start.elapsed_time(self.prefill_end))
        if had_decode:
            stats["decode_gpu_ms"].append(self.decode_start.elapsed_time(self.decode_end))


def _eos_token_ids(model_id: str, cache_dir: Optional[str]) -> FrozenSet[int]:
    try:
        from transformers import GenerationConfig

        generation_config = GenerationConfig.from_pretrained(model_id, cache_dir=cache_dir)
        eos = generation_config.eos_token_id
    except Exception:
        return DEFAULT_EOS_TOKEN_IDS

    if eos is None:
        return DEFAULT_EOS_TOKEN_IDS
    if isinstance(eos, int):
        return frozenset({eos})
    return frozenset(int(token) for token in eos)
