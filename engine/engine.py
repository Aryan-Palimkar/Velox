from dataclasses import dataclass
import torch
from .schedulerv2 import Scheduler
from src.transformer import QwenForCausalLM
from engine.sampler import Sampler
from engine.request import Request, RequestStatus
from engine.utils import BatchMetadata
import time

STOP_TOKENS = {151643, 151645}

class Engine:
    def __init__(self, model: QwenForCausalLM, scheduler: Scheduler, device="cuda"):
        self.model = model.to(device)
        self.scheduler = scheduler
        self.sampler = Sampler()
        self.device = device

        self.prefill_stream = torch.cuda.Stream(device=self.device)
        self.decode_stream = torch.cuda.Stream(device=self.device)

        self.stats = {
            "scheduler_ms": [],
            "prefill_gpu_ms": [],
            "decode_gpu_ms": [],
            "cpu_sync_wait_ms": [],
            "sampling_ms": []
        }

    def _prepare_inputs(self, batch: list['Request'], is_prefill: bool):
        input_ids = []
        positions = []
        cu_seq_lens_q = [0]
        cu_seq_lens_kv = [0]
        cache_slots = []
        seq_lens = []
        prompt_lens = []
        current_q_len = 0
        current_kv_len = 0

        for req in batch:
            cache_slots.append(req.cache_slot)

            if is_prefill:
                start_idx = req.num_computed_tokens
                end_idx = start_idx + req.current_chunk_size

                tokens = req.prompt_token_ids[start_idx:end_idx]
                seq_len = len(tokens)
                pos = list(range(start_idx, end_idx))

                current_q_len += seq_len
                current_kv_len += (start_idx + seq_len)

                seq_lens.append(start_idx)
                prompt_lens.append(seq_len)

                req.num_computed_tokens += seq_len

                input_ids.extend(tokens)
                positions.extend(pos)

            else:
                tokens = [req.output_token_ids[-1] if req.output_token_ids else req.prompt_token_ids[-1]]
                seq_len = 1
                total_history_len = req.num_prompt_tokens + len(req.output_token_ids)
                pos = [total_history_len - 1]

                current_q_len += seq_len
                current_kv_len += total_history_len

                seq_lens.append(total_history_len)
                prompt_lens.append(0)

                input_ids.extend(tokens)
                positions.extend(pos)

            cu_seq_lens_q.append(current_q_len)
            cu_seq_lens_kv.append(current_kv_len)

        def to_gpu(lst, dtype=torch.int32):
            return torch.tensor(lst, dtype=dtype).pin_memory().to(self.device, non_blocking=True)

        metadata = BatchMetadata(
            cu_seq_lens_q=to_gpu(cu_seq_lens_q),
            cu_seq_lens_kv=to_gpu(cu_seq_lens_kv),
            max_q_len=max((cu_seq_lens_q[i] - cu_seq_lens_q[i-1] for i in range(1, len(cu_seq_lens_q)))),
            max_kv_len=max((cu_seq_lens_kv[i] - cu_seq_lens_kv[i-1] for i in range(1, len(cu_seq_lens_kv)))),
            positions=to_gpu(positions, dtype=torch.long),
            cache_slots=to_gpu(cache_slots, dtype=torch.long),
            seq_lens=to_gpu(seq_lens),
            prompt_lens=prompt_lens,

            cu_seq_lens_q_cpu=cu_seq_lens_q,
            seq_lens_cpu=seq_lens,
            cache_slots_cpu=cache_slots
        )
        flat_input_ids = to_gpu(input_ids, dtype=torch.long)

        return flat_input_ids, metadata

    def _forward_pass(self, batch: list['Request'], is_prefill: bool):
        input_ids, metadata = self._prepare_inputs(batch, is_prefill)

        hidden_states = self.model(
            input_ids=input_ids,
            kv_cache=self.scheduler.kv_cache,
            prefill=is_prefill,
            metadata=metadata,
            positions=metadata.positions
        )

        last_token_indices = metadata.cu_seq_lens_q[1:] - 1
        last_token_logits = hidden_states[last_token_indices]

        return last_token_logits

    def _update_requests(self, batch: list['Request'], next_tokens_cpu: list[int]):
        for req, next_tok in zip(batch, next_tokens_cpu):
            if req.status == RequestStatus.RUNNING_PREFILL:
                if req.num_computed_tokens >= req.num_prompt_tokens:
                    req.output_token_ids.append(next_tok)
                    req.all_token_ids.append(next_tok)
                    req.status = RequestStatus.RUNNING_DECODE
                else:
                    self.scheduler.running_queue.remove(req)
                    self.scheduler.waiting_queue.appendleft(req)
            else:
                req.output_token_ids.append(next_tok)
                req.all_token_ids.append(next_tok)

                if len(req.output_token_ids) >= req.sampling_params.max_tokens or next_tok in STOP_TOKENS:
                    req.finished = True
                    req.stop_reason = "finished" if next_tok in STOP_TOKENS else "max_tokens"
                    req.status = RequestStatus.FINISHED

    @torch.no_grad()
    def step(self):
        t0 = time.perf_counter()
        prefill_batch, decode_batch, has_waiting = self.scheduler.schedule()
        self.stats["scheduler_ms"].append((time.perf_counter() - t0) * 1000)

        if not prefill_batch and not decode_batch:
            return

        prefill_logits = None
        decode_logits = None

        pf_start = torch.cuda.Event(enable_timing=True)
        pf_end   = torch.cuda.Event(enable_timing=True)
        dc_start = torch.cuda.Event(enable_timing=True)
        dc_end   = torch.cuda.Event(enable_timing=True)

        if prefill_batch:
            with torch.cuda.stream(self.prefill_stream):
                pf_start.record(self.prefill_stream)
                prefill_logits = self._forward_pass(prefill_batch, is_prefill=True)
                pf_end.record(self.prefill_stream)

        if decode_batch:
            with torch.cuda.stream(self.decode_stream):
                dc_start.record(self.decode_stream)
                decode_logits = self._forward_pass(decode_batch, is_prefill=False)
                dc_end.record(self.decode_stream)

        t_sync_start = time.perf_counter()
        current_stream = torch.cuda.current_stream()
        if prefill_batch:
            current_stream.wait_stream(self.prefill_stream)
        if decode_batch:
            current_stream.wait_stream(self.decode_stream)


        t_sample_start = time.perf_counter()

        prefill_tokens_gpu = None
        decode_tokens_gpu  = None

        if prefill_batch:
            prefill_tokens_gpu = self.sampler.sample(prefill_logits, [req.sampling_params for req in prefill_batch])
        if decode_batch:
            decode_tokens_gpu  = self.sampler.sample(decode_logits,  [req.sampling_params for req in decode_batch])

        all_tokens_gpu = torch.cat([
            t for t in [prefill_tokens_gpu, decode_tokens_gpu] if t is not None
        ])

        all_tokens_cpu = all_tokens_gpu.cpu().tolist()

        self.stats["cpu_sync_wait_ms"].append((time.perf_counter() - t_sync_start) * 1000)

        if prefill_batch:
            self.stats["prefill_gpu_ms"].append(pf_start.elapsed_time(pf_end))
        if decode_batch:
            self.stats["decode_gpu_ms"].append(dc_start.elapsed_time(dc_end))

        self.stats["sampling_ms"].append((time.perf_counter() - t_sample_start) * 1000)

        if prefill_batch:
            n = len(prefill_batch)
            self._update_requests(prefill_batch, all_tokens_cpu[:n])
        if decode_batch:
            n_pre = len(prefill_batch) if prefill_batch else 0
            self._update_requests(decode_batch, all_tokens_cpu[n_pre:])
