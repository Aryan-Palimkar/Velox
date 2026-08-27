# Velox

Velox is a small, from-scratch LLM inference engine built around paged attention, continuous batching, chunked prefill, and hand-written Triton kernels. It serves a Qwen2.5 model behind an OpenAI-compatible `/v1/chat/completions` endpoint with streaming, and is meant as a deep dive into what actually makes an inference server fast.

Everything below runs on a single 6GB laptop GPU.

## What it does

- **Paged KV cache** — the cache is a pool of fixed-size blocks (`[num_blocks, block_size, H_K, D]` per layer) handed out by a reference-counted allocator. Each request addresses its history through a block table, so a sequence's KV doesn't have to be contiguous and a request only holds what it has actually written. The pool is sized from whatever GPU memory is left after the weights land.
- **Prefix caching over a radix tree** — full blocks get published into a token-prefix trie keyed by chained block hashes. A new request matches the longest cached prefix and skips prefilling it. Only *full* blocks are ever cached, which removes the need for copy-on-write entirely: a request inherits read-only blocks and always allocates a fresh private block for its partial tail. Eviction is LRU over the leaves the tree alone still holds.
- **Continuous batching with recompute preemption** — decode work has priority, then in-flight prefill chunks, then new admissions. When the pool can't cover the next step, the *newest* running request is preempted and rewound. Preempting from the back is what keeps it starvation-free — a request only ever yields to one that arrived before it.
- **Chunked prefill** — long prompts are split into bounded chunks (`max_prefill_chunk_size`) so one huge prompt can't starve everyone else's decode. Outstanding work is defined as `len(all_token_ids) - num_computed_tokens`, which makes prefill and decode the same operation at different chunk sizes, and means a preempted request recomputes the tokens it generated instead of silently losing their KV.
- **Custom Triton attention kernels**:
  - A prefill kernel that's flash-attention-style, varlen, causal, and resolves every K/V tile through the request's block table. It's chunked-prefill aware, so a chunked prefill produces the same result as a single full one.
  - A decode kernel with GQA head grouping — one program serves every query head sharing a KV head, so a page of K/V is read once per group instead of once per head. Optional split-K (flash-decoding) kicks in when the `(batch, kv_head)` grid is too small to fill the device.
- **CUDA graphs for decode** — the decode forward pass is captured once per padded batch size (1, 2, 4, … `max_num_seqs`), turning several hundred kernel launches into a single replay. Padded rows point at a reserved null block so they read and write somewhere harmless.
- **Quantization** — INT8 or FP8 KV cache with per-token, per-head scales, dequantized inside the attention kernel. The K scale folds in after the score dot and the V scale folds into the softmax weights, so neither costs an extra pass. Weight-only INT8/FP8 for every projection GEMM as well.
- **Dual-stream prefill/decode overlap** — prefill and decode forward passes are launched on separate CUDA streams (`prefill_stream`, `decode_stream`) so the prefill matmuls for one request can overlap with decode matmuls for others, instead of serializing two passes on the default stream. They get separate staging buffers, for reasons the debugging section gets into.
- **Batched sampling that respects each request** — per-request temperature, top-k, top-p and repetition/presence/frequency penalties, all vectorized. Nucleus filtering runs on a shortlist whose truncation is checked against the exact log-sum-exp of the full row, so a flat enough distribution falls back to a full sort instead of silently clipping the nucleus. The sampler returns a GPU tensor so the caller controls where the device-to-host sync happens.
- **OpenAI-compatible streaming API** — server-sent events with proper chunk framing, incremental detokenization that won't split a multi-byte character, stop strings matched across chunk boundaries, and usage on the final chunk. The engine runs on its own thread; results cross back through `loop.call_soon_threadsafe`, so a slow client can never stall the GPU loop.

The engine's `step()` loop is still structured so that every GPU op for a step is issued before there is exactly **one** point where the CPU blocks on the GPU.

## Performance Benchmarks

*Hardware: RTX 4050 Mobile (6GB VRAM) | 16GB RAM | Qwen2.5-1.5B-Instruct, bf16*

### Against HuggingFace `transformers`

16 concurrent requests, mixed short and long prompts, 128 max tokens. TTFT is measured under load — every request arrives at once, so anything queued behind others waits for them. That's the comparison continuous batching exists to win. Measured alone, HF reaches its first token in 45ms.

| Metric | Velox | HF (sequential) | |
| :--- | ---: | ---: | ---: |
| Wall time | **3.04 s** | 52.56 s | 17.2× |
| Decode throughput | **394 tok/s** | 26 tok/s | 15.0× |
| Mean TTFT under load | **0.16 s** | 22.87 s | 143× |
| p99 TTFT under load | **0.17 s** | 47.54 s | 297× |

### Concurrency

| Concurrent | Decode throughput | p50 TTFT | Blocks used | Preemptions |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 50 tok/s | 76.7 s | 69 | 0 |
| 8 | 226 tok/s | 20.8 s | 113 | 0 |
| 32 | 613 tok/s | 7.4 s | 269 | 0 |
| 64 | 680 tok/s | 4.4 s | 512 | 0 |
| **128** | **931 tok/s** | **3.9 s** | 611 | 0 |

18.7× throughput scaling from 1 to 128, using 611 of 2047 blocks at the top end — and that pool was deliberately undersized. Left to size itself it holds 46,896 tokens in bf16, or 163,472 with INT8 weights and an INT8 KV cache, against 16,384 for the fixed-slot design this started from.

### Under a realistic load spike

The benchmarks above all submit everything at once, which isn't what a server sees. This one drives the real HTTP + SSE path with Poisson arrivals across three traffic classes — short chat, long-context RAG at ~630 prompt tokens, long-output code gen — ramping 2 req/s → **15 req/s** → 2.5 req/s, with 5% of clients disconnecting mid-stream.

| | |
| :--- | ---: |
| Requests / errors | 267 / **0** |
| Output throughput, sustained | **530 tok/s** (peak 968) |
| Service rate for this mix | 4.49 req/s |
| p50 TTFT, steady phase | **0.07 s** |
| p50 TTFT, during the 15 req/s spike | 8.34 s |
| Inter-token latency at 32-way concurrency | p50 **32 ms**, p95 101 ms |
| Peak queue depth | 105 |
| Preemptions | **0** |

Two things worth pulling out of that. The 8.34s p50 TTFT during the spike isn't the engine failing — offering 15 req/s to a server whose measured service rate is 4.5 req/s means requests queue, and that's arithmetic. What matters is that inter-token latency held at 32ms while it happened: admitted requests kept streaming smoothly and the overload got absorbed by the queue instead of by making everyone stutter.

The other is that KV blocks hit **100% occupancy** with a 105-deep queue and still nothing was preempted. The admission watermark refused to admit rather than admitting and thrashing, and the prefix cache handed its retained blocks back under pressure instead of forcing anyone to recompute. That path is hard to exercise any way other than actually saturating the thing.

### Prefix caching

12 requests sharing one system prompt, replayed three times:

| Round | Prompt tokens from cache | Mean TTFT |
| ---: | ---: | ---: |
| 1 (cold) | 0% | 0.173 s |
| 2 (warm) | **91.7%** | **0.033 s** |
| 3 (warm) | 91.7% | 0.033 s |

5.2× lower time to first token once the shared prefix is resident. Throughput barely moves, which is right — prefix caching doesn't make decoding faster, it deletes prefill work. On traffic where prompts don't repeat it does nothing at all, and there's a flag to turn it off. (In the load simulation above, with a shared system prompt on 70% of traffic and otherwise varied prompts, it served 20% of prompt tokens from cache — a more honest number for real traffic.)

### Quantization

| 16 concurrent, 96 max tokens | bf16 | INT8 weights | |
| :--- | ---: | ---: | ---: |
| Decode step | 20.1 ms | **11.1 ms** | 1.81× |
| Decode throughput | 408 tok/s | **696 tok/s** | 1.71× |
| Mean TTFT | 0.162 s | 0.176 s | 1.09× worse |

Decode on this GPU is bandwidth-bound: 1.5B bf16 parameters is 3.1GB against ~192 GB/s, so a decode step can't beat ~16ms no matter how good the kernels are. Halving the weight width halves that floor, which is what the 1.81× is. Prefill is compute-bound instead and gets 8% slower, since a Triton dequant-GEMM isn't going to beat cuBLAS at large M. Fine trade — decode runs `max_tokens` times per request, prefill runs once.

**On FP8 vs INT8 for the KV cache:** both cost one byte per element and I assumed they'd be roughly equivalent(they aren't). Comparing teacher-forced logits against bf16, INT8 agrees on the argmax 94% of the time (cosine 0.998); FP8-e4m3 manages 72% (cosine 0.892). e4m3 keeps three mantissa bits, and because relative precision in a float format is independent of scale, finer-grained scaling can't buy it back — near the per-head maximum, which is where attention scores are dominated, its spacing is 9× coarser than INT8's. I checked the Triton conversion against PyTorch's round-to-nearest as well (18 tie differences in 4096 elements, identical mean error), so it's the format and not a rounding bug. Use `--kv-cache-dtype int8`. FP8 is fine for *weights*, where the error averages out over a 1536-deep dot product.

## A Debugging Rabbit Hole

Everything was built and the attention kernels checked out against `F.scaled_dot_product_attention`. Then I ran three prompts through the assembled engine:

```
'What is the capital of France?'      -> 'The capital of France is Paris.'
'List three prime numbers.'           -> 'The TheThe vast ocean covers over 70'
'Write one sentence about the ocean.' -> 'The vast ocean covers over 70% of the Earth'
```

The middle one isn't garbage and it isn't a crash. It's a grammatical, on-topic answer to the *third* prompt, produced by the second request. That tells you something before you know anything else: garbage means broken arithmetic, but coherent-and-wrong means the arithmetic is fine and the inputs belonged to somebody else.

So I started turning things off. Prefix caching was the obvious suspect since it shares blocks by design but the bug was still there with the cache disabled. What did change the outcome was **CUDA graphs**, which have nothing whatsoever to do with which KV a request reads. It also needed more than one request in the batch; running the same prompts with `max_num_seqs=1` was fine.

Then I added tracing to dump the block tables each step and the output was completely correct and so was the generated text. Printing a tensor calls `.tolist()`, which synchronizes. My debug statements were fixing the bug.

A bug that vanishes when you look at it and depends on an unrelated feature flag isn't an arithmetic bug. It's a race and the thing racing is a memory copy.

Block tables get pushed to the GPU incrementally, a decode step usually appends nothing and when it does append it's one integer — so I staged the new entries through a small pinned buffer:

```python
buffers.h_block_row[:count] = torch.tensor(request.block_table[synced:total], ...)
self.block_tables_by_seq[slot, synced:total].copy_(
    buffers.h_block_row[:count], non_blocking=True
)
```

That's correct for one request. Called in a loop over a batch it's a use-after-overwrite. `non_blocking=True` means the copy is *enqueued*, not performed; the CPU races on to the next request and overwrites `h_block_row` while the first copy is still sitting in the stream. When it finally runs it reads whatever is in the buffer now. Request 1's block table ends up pointing at request 2's blocks, and it attends over a perfectly coherent history that just isn't its own.

The fix is a pinned mirror of the device-side table with one row per sequence slot, so no two requests in a batch ever stage through the same bytes. Copies stay asynchronous, and the common case — nothing appended — is a comparison and a return. Costs 8KB.

What I took from it, beyond the one-liner:

- **A pinned buffer is a lease, not a variable.** Between `copy_(non_blocking=True)` and the copy actually running, those bytes belong to the DMA engine.
- Reuse across steps was already safe, but only because the engine synchronizes once per step, which orders the previous step's copies before this step's writes. That's a real invariant the code depends on and it was nowhere written down.
- PyTorch protects you for the idiom I *didn't* use: `torch.tensor(x).pin_memory().to(device, non_blocking=True)` is safe, because the caching host allocator records an event and won't recycle the block until the copy completes. Hand-managing a long-lived pinned buffer quietly opts out of that.
- Cross-request contamination is invisible to anything that only ever runs one request at a time. The check that catches it is boring: run a batch of prompts together, run each one alone, require identical tokens.

Two smaller ones from the same rebuild, both in the same family:

**Hidden syncs** `torch.tensor(some_list, device="cuda")` looks harmless and is a *pageable* host copy, which blocks until the stream drains. Two of those sat in the sampling path, so a full decode step's worth of GPU work had to finish before the sampler's own kernels were even enqueued. The giveaway was `sampling_ms` sitting at ~23ms — almost exactly `decode_gpu_ms` — while `cpu_sync_wait_ms`, the designated wait point, read 0.15ms. Wall time barely moved after fixing it, but the profile became honest, and a profile that attributes time to the wrong stage is worse than no profile because you act on it.

**A tuning regression that was a compiler.** With INT8 weights prefill measured 171ms against bf16's 91ms. I assumed my GEMM tiles were bad, swept 200 configurations, applied the winners, and prefill got *worse* — 632ms. That's not how tuning behaves. Triton compiles per constexpr signature, and the quantized GEMM picks its tile shape from the token count, so the warmup (a 16-token prompt) never touched the large-M bucket. The first real prefill was paying for JIT compilation, and the CUDA events bracketing the forward pass counted it as GPU time because the stream sat idle while the CPU compiled. Warming one prompt per tile bucket took TTFT from 0.599s to 0.176s without touching a kernel.

## Quickstart

```bash
python -m venv venv
source venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m server.server --model Qwen/Qwen2.5-1.5B-Instruct --port 8000
```

Streaming, with the official client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="not-needed")

for chunk in client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    messages=[{"role": "user", "content": "Explain paged attention in two sentences."}],
    max_tokens=128,
    stream=True,
):
    if chunk.choices:
        print(chunk.choices[0].delta.content or "", end="", flush=True)
```

Or with curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 128,
    "temperature": 0.7,
    "stream": true
  }'
```

The flags worth knowing: `--max-num-seqs` (concurrent sequences, and the largest CUDA graph bucket), `--max-model-len`, `--gpu-memory-utilization` (fraction of *free* memory given to the KV cache), `--kv-cache-dtype int8`, `--quantization int8`, and `--no-prefix-caching` / `--no-cuda-graphs` to turn either off. `GET /metrics` reports live block occupancy, prefix hit rate, preemption count and the per-stage step breakdown.

Offline, without the server:

```python
from engine.config import EngineConfig
from engine.engine import Engine
from engine.request import Request
from engine.utils import SamplingParams

engine = Engine.from_pretrained(EngineConfig(model="Qwen/Qwen2.5-1.5B-Instruct"))
request = Request("r0", prompt_token_ids=[...], sampling_params=SamplingParams(max_tokens=64))
engine.generate([request])
print(request.output_token_ids)
```

`benchmark.py` reproduces the numbers above — `python benchmark.py --compare-hf`, or `--sweep-batch-size 1,8,32,64,128`.

## Known limitations / roadmap

Things that are explicitly cut for now, roughly in the order I'd tackle them:

- **Overlapping scheduling with execution.** The per-step breakdown still shows ~1.5ms of batch preparation during which the GPU has nothing queued. Preparing step N+1 while step N runs would recover most of it; it needs the metadata double-buffered and the sampler's dependency on the previous step made explicit.
- **AWQ / GPTQ checkpoints.** Quantization is computed at load time by symmetric per-channel rounding. Good for INT8, adequate for FP8, and clearly behind a properly calibrated checkpoint that has seen activation statistics.
- **Swap-based preemption.** Recompute won here because this machine's PCIe is slow relative to its compute, and because prefix caching lets a preempted request recover most of its KV anyway. On different hardware the answer flips.
- **Speculative decoding.** The dual-stream split between prefill and decode is a natural place to slot a draft model in. It's also the only thing on this list that attacks the roofline directly — verifying *k* tokens per weight read is the one way to raise decode's arithmetic intensity without making the weights smaller.
- **Better admission policy under overload.** Requests are FCFS with no priority scheme, so a burst of long-context requests will queue short ones behind them. The spike above shows the queue absorbing overload gracefully, but it doesn't show it choosing *well*.
- **Single GPU, single process.** No tensor or pipeline parallelism.
- **Qwen2 only.** The weight loader fuses `q/k/v_proj` and `gate/up_proj` into the layouts Velox's GEMMs expect; another architecture needs its own loader.
