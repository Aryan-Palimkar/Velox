# Velox

Velox is a small, from-scratch LLM inference engine built around continuous batching, chunked prefill, and hand-written Triton attention kernels. It serves a Qwen2.5 model behind an OpenAI-compatible `/v1/chat/completions` endpoint and is meant as a deep dive into what actually makes an inference server fast.

## What it does

- **Continuous batching** — requests are admitted, prefilled, and decoded independently of each other on a per-step basis instead of running in fixed static batches.
- **Chunked prefill** — long prompts are split into bounded chunks (`max_prefill_chunk_size`) so a single huge prompt can't starve decode steps for other in-flight requests. Partially-prefilled requests are re-queued and resumed from `num_computed_tokens`.
- **Slot-based KV cache** — `BatchedKVCache` pre-allocates a fixed `[max_batch_size, max_seq_len, H_K, D]` tensor per layer and hands out integer "slots" to active requests, freeing them back to a pool on completion or abort.
- **Custom Triton attention kernels**:
  - A prefill kernel (`_attn_prefill_optimized`) that's flash-attention-style, varlen, causal, and aware of the split between cached history and the current chunk (so chunked prefill produces numerically identical results to a single full prefill).
  - A decode kernel (`decode_attention` / `_decode_attn_kernel`) specialized for the single-query-token case with GQA head grouping.
- **Dual-stream prefill/decode overlap** — prefill and decode forward passes are launched on separate CUDA streams (`prefill_stream`, `decode_stream`) so the prefill matmuls for one request can overlap with decode matmuls for others, instead of serializing two passes on the default stream.
- **Vectorized batched sampling** — `Sampler` does temperature scaling and top-p filtering via `topk(512)` instead of a full vocab sort, and deliberately returns a GPU tensor so the caller controls exactly where (and how often) the device-to-host sync happens.
- **OpenAI-compatible API** — a FastAPI server with a background worker thread continuously stepping the engine, and request objects polled until `finished` is set.


The engine's `step()` loop is intentionally structured so that every GPU op for a step is issued before there is exactly **one** point where the CPU blocks on the GPU.

## 📊 Performance Benchmarks

*Hardware: RTX 4050 Mobile (6GB VRAM) | 16GB RAM | 8-Core CPU*

Velox was benchmarked head-to-head against vanilla HuggingFace `transformers` (Batched and Sequential) using a mixed-workload prompt set.

| Metric | Velox | HF (baseline) |
| :--- | :--- | :--- |
| **Wall Time (s)** | **35.63** | 105.18 |
| **Decode Throughput** | **79.40 tok/s** | 27.00 tok/s |
| **Mean TTFT (s)** | **1.59** | 27.73 |
| **p99 Queue Wait** | **1.59s** | 84.84s |

## A Debugging Rabbit Hole

One of the more interesting parts of this project wasn't writing the Triton kernels, it was realizing they weren't actually the bottleneck.

The attention and sampling kernels both benchmarked well in isolation, but the engine's end-to-end throughput was consistently lower than those numbers suggested. At first, I assumed there was still room to optimize the kernels. After enough profiling, though, it became clear that the time was disappearing somewhere else.

This is why `Engine.stats` tracks metrics like `scheduler_ms`, `prefill_gpu_ms`, `decode_gpu_ms`, `cpu_sync_wait_ms`, and `sampling_ms`. Measuring the entire `step()` with `time.perf_counter()` wasn't enough because CUDA launches are asynchronous. Instead, I used CUDA events (`torch.cuda.Event(enable_timing=True)`) to separate actual GPU execution time from time the CPU spent waiting.

The biggest culprit turned out to be **implicit host-device synchronizations**. Operations like calling `.item()` inside a per-request loop or copying a tensor back to the CPU just to make a scheduling decision seem harmless, but each one forces the CPU to wait until all previously queued GPU work has completed. Individually they're tiny, but inside a decoding loop they add up surprisingly quickly.

The changes that ended up making the biggest difference were:

- **Keeping a single synchronization point per decoding step.** Everything inside `step()` stays on the GPU until `all_tokens_gpu.cpu().tolist()`, which is the only point where results actually need to leave the device.

- **Maintaining CPU-side shadow metadata.** `BatchMetadata` stores CPU copies (`cu_seq_lens_q_cpu`, `seq_lens_cpu`, and `cache_slots_cpu`) alongside the GPU tensors so that Python-side scheduling and launch configuration never have to read values back from device memory.

- **Reducing kernel launches where it made sense.** Fusing the QKV projections into a single `qkv_proj` GEMM wasn't a huge compute optimization by itself, but it reduced the number of launch boundaries where accidental synchronizations could sneak in.

- **Using CUDA events instead of wall-clock timers.** `time.perf_counter()` around an asynchronous CUDA call mostly measures how long it takes to enqueue work. CUDA events measure how long the GPU actually spent executing it, which made the real bottleneck much easier to spot.



## Quickstart

```bash
python -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m server.server
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "."}],
    "max_tokens": 128,
    "temperature": 0.7,
    "top_p": 0.9
  }'
```

## Known limitations / roadmap

Things that are explicitly cut for now, roughly in the order I'd tackle them:

- **CUDA graphs for the decode step.** Decode is launch-overhead-bound (lots of small kernels for a batch of single tokens); capturing the decode path as a CUDA graph should cut a meaningful chunk of per-step CPU overhead.
- **True paged attention.** The current KV cache reserves a contiguous `max_seq_len` per slot rather than allocating in fixed-size blocks, so memory is over-provisioned for short sequences. Moving to block-based paging (vLLM-style) would let `max_batch_size` scale with actual usage instead of worst-case usage.
- **Prefix / radix caching.** No sharing of KV across requests with a common prefix (e.g. shared system prompts) yet, every request pays for its own prefill.
- **Streaming responses.** The API currently polls `req.finished` and returns the full completion at once.
- **Speculative decoding.** The dual-stream split between prefill and decode is a natural place to slot in a draft model later.
- **Quantization (FP8 / INT8 weights or KV cache).** Would reduce memory pressure and likely raise the achievable `max_batch_size`.
- **Better preemption policy.** Preempted requests currently just go back to `waiting_queue.appendleft()`; there's no priority scheme or fairness guarantee under load.