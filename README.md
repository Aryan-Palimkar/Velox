# Velox

Velox is a from-scratch LLM inference engine built around paged attention, continuous batching, and hand-written Triton kernels. It serves Qwen2 models behind an OpenAI-compatible API with streaming, and it is meant as a deep dive into what actually makes an inference server fast.

Everything below runs on a single 6 GB laptop GPU.

## What it does

**Paged KV cache.** KV lives in fixed-size blocks (`[num_blocks, block_size, num_kv_heads, head_dim]`) handed out by a reference-counted allocator. Each request addresses its history through a block table, so memory is provisioned for actual usage rather than for `max_seq_len` per sequence. The block pool is sized from whatever GPU memory is left after the weights land.

**Prefix caching over a radix tree.** Full blocks are published into a token-prefix trie keyed by chained block hashes. A new request matches the longest cached prefix and skips prefilling it entirely. Only *full* blocks are ever cached, which removes the need for copy-on-write: a request inherits read-only blocks and always allocates a fresh private block for its partial tail. Eviction is LRU over leaves the tree alone still holds.

**CUDA graphs for decode.** The decode forward pass is captured once per padded batch size (1, 2, 4, … `max_num_seqs`), turning hundreds of kernel launches into a single replay. Padded rows point at a reserved null block, so they read and write somewhere harmless.

**Custom Triton attention kernels.**
- A flash-attention prefill kernel that is varlen, causal, chunked-prefill aware, and resolves every K/V tile through the request's block table.
- A decode kernel with GQA head grouping — one program serves every query head sharing a KV head, so a page of K/V is read once per group instead of once per head — plus optional split-K (flash-decoding) when the `(batch, kv_head)` grid is too small to fill the device.

**Continuous batching with recompute preemption.** Decode work has priority, then in-flight prefill chunks, then new admissions. When the pool cannot cover the next step, the *newest* running request is preempted and rewound; preempting from the back is what keeps the policy starvation-free. A preempted request recomputes its generated tokens as well as its prompt, and often recovers most of them straight from the prefix cache.

**Quantization.** INT8 or FP8 KV cache with per-token, per-head scales, dequantized inside the attention kernel — the K scale folds in after the score dot, the V scale folds into the softmax weights, so neither costs an extra pass. Weight-only INT8/FP8 for every projection GEMM, with a Triton kernel that widens the weight tile in SRAM. INT8 is the right choice for the KV cache and the numbers below say why.

**Streaming OpenAI API.** Server-sent events with correct chunk framing, incremental detokenization that never splits a multi-byte character, stop strings matched across chunk boundaries, and usage on the final chunk.

**Batched sampling that respects each request.** Per-request temperature, top-k, top-p, and repetition/presence/frequency penalties, all vectorized. Nucleus filtering runs on a shortlist whose truncation is verified against the exact log-sum-exp of the full row, so a distribution flat enough to spill past it falls back to a full sort instead of silently clipping. Per-request seeds are honoured through exponential-noise argmax sampling.

The `step()` loop is structured so that every GPU op for a step is issued before there is exactly **one** point where the CPU blocks on the GPU.

## Benchmarks

*Hardware: RTX 4050 Mobile (6 GB VRAM), 16 GB RAM, WSL2. Model: Qwen2.5-1.5B-Instruct, bf16.*

### Against HuggingFace `transformers`

16 concurrent requests, mixed short/long prompts, 128 max tokens.

| Metric | Velox | HF (sequential) | Speedup |
| :--- | ---: | ---: | ---: |
| Wall time | **3.04 s** | 52.56 s | **17.2×** |
| Decode throughput | **394 tok/s** | 26 tok/s | **15.0×** |
| Mean TTFT under load | **0.16 s** | 22.87 s | **143×** |
| p99 TTFT under load | **0.17 s** | 47.54 s | **297×** |

TTFT is measured under load — every request arrives at once, so a request queued behind others waits for them. That is the comparison continuous batching exists to win; measuring each request alone would hide the queue entirely. In isolation HF reaches its first token in 0.045 s.

### Concurrency

The old slot-based cache reserved a contiguous `max_seq_len` per slot and topped out at 8 concurrent sequences. Paging removes that ceiling. 128 requests submitted at once, 96 max tokens, pool pinned at 2048 blocks so the only variable is concurrency:

| Concurrent | Decode throughput | p50 TTFT | Blocks used | Preemptions |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 50 tok/s | 76.7 s | 69 | 0 |
| 8 | 226 tok/s | 20.8 s | 113 | 0 |
| 32 | 613 tok/s | 7.4 s | 269 | 0 |
| 64 | 680 tok/s | 4.4 s | 512 | 0 |
| **128** | **931 tok/s** | **3.9 s** | 611 | 0 |

18.7× throughput scaling from 1 to 128 concurrent sequences, with 611 of 2047 blocks used at the top end — about 30% of a deliberately undersized pool. Left to size itself on this GPU the pool holds **46,896 tokens** versus 16,384 for the old fixed-slot layout, and it can spend them on any mix of sequence lengths rather than 8 worst-case slots.

TTFT is again measured under load: with `max_num_seqs=1` the 128th request waits for 127 others, which is exactly the queueing that higher concurrency removes.

### Prefix caching

12 requests sharing one system prompt, replayed three times:

| Round | Prompt tokens served from cache | Mean TTFT | Wall time |
| ---: | ---: | ---: | ---: |
| 1 (cold) | 0% | 0.173 s | 1.48 s |
| 2 (warm) | **91.7%** | **0.033 s** | 1.33 s |
| 3 (warm) | 91.7% | 0.033 s | 1.33 s |

**5.2× lower time to first token** once the shared prefix is resident.

### Weight quantization

16 concurrent requests, 96 max tokens.

| | bf16 | INT8 weights |
| :--- | ---: | ---: |
| Decode step | 20.1 ms | **11.1 ms** |
| Decode throughput | 408 tok/s | **696 tok/s** |
| Mean TTFT | 0.162 s | 0.176 s |

Decode on this GPU is bandwidth-bound: 1.5 B bf16 parameters is 3.1 GB, and the card has ~192 GB/s, so a decode step cannot beat ~16 ms no matter how good the kernels are. Halving the weight width halves that floor, which is exactly what the 1.81× shows. Prefill is compute-bound instead, and the Triton dequant-GEMM lands within 8% of cuBLAS there.

### What quantization costs

Measured, not assumed: teacher-forced logits over a 50-token prompt, compared position-by-position against bf16.

| Mode | Bytes/element | argmax agreement | mean cosine |
| :--- | ---: | ---: | ---: |
| INT8 weights | 1 | 94% | 0.999 |
| FP8 weights | 1 | 96% | 0.998 |
| **INT8 KV cache** | 1 | **94%** | **0.998** |
| FP8 (e4m3) KV cache | 1 | 72% | 0.892 |
| FP8 (e5m2) KV cache | 1 | 46% | 0.871 |

**Use `--kv-cache-dtype int8`, not `fp8`.** Both cost one byte per element, but e4m3 keeps only three mantissa bits, and because relative precision in a float format is independent of scale, no amount of finer-grained scaling buys it back. INT8 with a per-token, per-head scale spends its whole range on the values that actually dominate an attention score. The kernel's FP8 conversion was checked against PyTorch's round-to-nearest — 18 tie differences in 4096 elements and an identical mean error — so this is the format's precision, not a rounding bug. `tests/test_engine_e2e.py` asserts the ordering so the recommendation cannot silently rot.

FP8 is fine for *weights*, where the error averages out over a 1536-deep dot product and the model is far less sensitive.

### What it all adds up to

Running `--quantization int8 --kv-cache-dtype int8` on the same 6 GB card:

| | Original | Velox |
| :--- | ---: | ---: |
| KV cache capacity | 16,384 tokens | **163,472 tokens** |
| Shape of that capacity | 8 fixed slots of 2048 | any mix of lengths |
| Decode step (batch 16) | — | 10.8 ms |

**10× the KV capacity**, and it can be spent on 100 short conversations or a handful of long ones rather than 8 worst-case slots.

## A debugging rabbit hole

The most interesting bug in this rewrite was not in a kernel. It was a single line of staging code.

Block tables are pushed to the GPU incrementally — a decode step usually appends nothing, and when it does append it is one integer. The obvious implementation copies the new entries through a small pinned buffer:

```python
buffers.h_block_row[:count] = torch.tensor(request.block_table[synced:total])
self.block_tables_by_seq[slot, synced:total].copy_(
    buffers.h_block_row[:count], non_blocking=True
)
```

That is correct for one request and wrong for a batch. The copy is asynchronous, so when the loop reaches the next request it overwrites `h_block_row` while the previous DMA is still queued. The last writer wins, and one sequence ends up reading another's KV blocks.

What made it interesting is how it presented. Three prompts went in — "What is the capital of France?", "List three prime numbers.", "Write one sentence about the ocean." — and the middle one answered with the ocean. Not garbage, not a crash: a fluent, coherent answer to somebody else's question. Every kernel test passed, because every kernel *was* correct.

It also disappeared under instrumentation. Printing the metadata called `.tolist()`, which synchronizes, which let the pending copy land before the next overwrite. And it disappeared with CUDA graphs enabled, because capture changed the timing enough to hide it. A bug that vanishes when you look at it and depends on an unrelated feature flag is the signature of a race, not of arithmetic.

The fix is to give every sequence slot its own row in a pinned mirror of the device table, so no two requests in a batch ever stage through the same bytes. The test that would have caught it immediately now exists: run a batch of prompts together, run each one alone, and require identical tokens. Cross-request contamination is invisible to any test that only ever runs one request at a time.

Two smaller ones from the same rewrite:

**Hidden synchronizations.** `torch.tensor(some_list, device="cuda")` looks harmless and is a *pageable* host copy, which blocks until the stream drains. Two of those sat in the sampling path, so a full decode step's worth of GPU work had to finish before the sampler's own kernels were even enqueued. `Engine.stats` separates `decode_gpu_ms` (CUDA events) from `cpu_sync_wait_ms` (wall clock) precisely so this shows up: when they matched almost exactly, the sampler was not slow, it was waiting.

**Warmup that warmed the wrong shapes.** Triton compiles per constexpr signature, and the quantized GEMM picks its tile shape from the token count. A warmup that only ran a 16-token prompt left every real prefill shape uncompiled, so the *first user request* paid 600 ms of compilation inside its own time to first token. Warming one prompt per tile bucket moved it to startup: TTFT went from 0.599 s to 0.176 s without touching a single kernel.

## Quickstart

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Serve:

```bash
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
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"messages":[{"role":"user","content":"Hello"}],"max_tokens":64,"stream":true}'
```

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

## Configuration

| Flag | Default | Notes |
| :--- | :--- | :--- |
| `--max-model-len` | 4096 | Context limit; also caps the rotary table. |
| `--max-num-seqs` | 32 | Concurrent sequences, and the largest CUDA graph bucket. |
| `--max-num-batched-tokens` | 2048 | Token budget per step. |
| `--max-prefill-chunk-size` | 512 | Caps how much prefill one request can take in a step. |
| `--block-size` | 16 | KV page size, in tokens. Must be a power of two. |
| `--num-gpu-blocks` | auto | Overrides the memory-derived pool size. |
| `--gpu-memory-utilization` | 0.90 | Fraction of *free* memory given to the KV cache. |
| `--kv-cache-dtype` | `auto` | `int8` (recommended), `fp8`, or `fp8_e5m2`. FP8 needs compute capability 8.9+ and is measurably lossier — see above. |
| `--quantization` | none | `int8` or `fp8` weight-only quantization. |
| `--no-prefix-caching` | off | Disables the radix cache. |
| `--no-cuda-graphs` | off | Falls back to eager decode. |

`GET /metrics` reports live block occupancy, prefix hit rate, preemption count, and the per-stage step breakdown.

## Tests

```bash
pytest                      # everything
pytest -m "not slow"        # unit tests only, ~12 s
pytest -m gpu               # anything needing CUDA
```

The suite is 190 tests. The ones that matter most:

- **Kernel equivalence** (`test_paged_attention.py`) — paged prefill and decode against `F.scaled_dot_product_attention`, over deliberately fragmented block tables so a kernel that assumed contiguity fails rather than accidentally passes. Covers chunked prefill, cached prefixes, split-K, and both quantized KV dtypes.
- **Teacher-forced logits** (`test_engine_e2e.py`) — a whole sequence compared against HuggingFace in one pass, so every position is checked independently instead of only the first divergence.
- **Batched vs solo** — the same prompts run together and alone must produce identical tokens.
- **Reference counting** (`test_radix_cache.py`, `test_block_allocator.py`) — eviction never frees a live block, and churn never leaks one. Every scheduler test ends by asserting allocator and tree invariants.
- **Streaming** (`test_server_streaming.py`) — a real uvicorn process driven by the `openai` client: chunk framing, the `[DONE]` sentinel, usage placement, concurrent streams, and an assertion that the first frame arrives well before the last, which is what distinguishes streaming from buffering.
- **Long context** — ~900 tokens through a chunked prefill, so block-table traversal past the first few pages and the seam between chunks are checked at every position rather than only near the start.

Feature-flag tests (CUDA graphs, chunked prefill, preemption, prefix caching) assert that turning a feature on does not change what the model produces. Where two paths reduce in a different order — split-K versus a single pass — they are compared on argmax and cosine similarity rather than on a chained greedy continuation, because bf16 rounding can legitimately flip a near-tie and send two correct implementations to different sentences.

## Layout

```
src/
  paged_kv_cache.py     block-paged K/V storage and the write path
  block_allocator.py    reference-counted physical block pool
  block_manager.py      per-request block tables, growth, and publication
  radix_cache.py        token-prefix trie with LRU eviction
  attention_prefill.py  paged flash-attention prefill kernel
  decode_attention.py   GQA-grouped paged decode kernel with split-K
  quantization.py       FP8/INT8 KV cache and weight-only quantized GEMM
  rope.py, rope_inplace.py, rmsnorm.py, mlp.py, self_attention.py, transformer.py
engine/
  engine.py             the step loop
  scheduler.py          continuous batching, admission, preemption
  model_runner.py       batch staging, CUDA graph capture and replay
  sampler.py            batched sampling, penalties, token history
  detokenizer.py        incremental detokenization and stop strings
  quantize.py           swaps linear layers for quantized ones
server/
  server.py             FastAPI app, SSE streaming
  engine_client.py      async bridge over the engine thread
  protocol.py           OpenAI request/response models
benchmark.py            the benchmark used for the numbers above
```

## Limitations

- **Qwen2 architecture only.** The weight loader fuses `q/k/v_proj` and `gate/up_proj` into the layouts Velox's GEMMs expect; another architecture needs its own loader.
- **Weight quantization is computed at load time**, not read from an AWQ or GPTQ checkpoint. Accuracy is what symmetric per-output-channel rounding gives you, which is good for INT8 and adequate for FP8, but it will not match a properly calibrated checkpoint.
- **`n > 1`, `logprobs`, `logit_bias`, and tool calling are rejected**, not silently ignored.
- **Preemption recomputes rather than swapping to host memory.** With prefix caching on, a preempted request usually recovers most of its KV from the cache; without it, the recompute is real work.
- **Single GPU, single process.** No tensor or pipeline parallelism.
- **No speculative decoding.** The dual-stream split between prefill and decode is a natural place to slot a draft model in later.
