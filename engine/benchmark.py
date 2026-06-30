import argparse
import time
import torch
import numpy as np
import traceback

def stats(label: str, times_ms: list[float]):
    arr = np.array(times_ms)
    print(f"  {label:<35} mean={arr.mean():>8.3f}ms  "
          f"p50={np.percentile(arr,50):>8.3f}ms  "
          f"p90={np.percentile(arr,90):>8.3f}ms  "
          f"min={arr.min():>8.3f}ms")

def timeit_gpu(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times

def timeit_cpu(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000)
    return times

def bench_linear(hidden: int, seq_len: int, device="cuda", dtype=torch.float16):
    print(f"\n{'─'*60}")
    print(f" Raw Linear (GEMM) — hidden={hidden}, seq_len={seq_len}, dtype={dtype}")

    with torch.device(device):
        x   = torch.randn(seq_len, hidden, dtype=dtype)
        lin = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)

    times = timeit_gpu(lambda: lin(x))
    stats("Linear(hidden, hidden)", times)

def bench_sdpa(num_heads: int, head_dim: int, seq_len: int, device="cuda", dtype=torch.float16):
    print(f"\n{'─'*60}")
    print(f" F.scaled_dot_product_attention — H={num_heads}, D={head_dim}, S={seq_len}")

    with torch.device(device):
        q = torch.randn(1, num_heads, seq_len, head_dim, dtype=dtype)
        k = torch.randn(1, num_heads, seq_len, head_dim, dtype=dtype)
        v = torch.randn(1, num_heads, seq_len, head_dim, dtype=dtype)
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))

    import torch.nn.functional as F
    times = timeit_gpu(lambda: F.scaled_dot_product_attention(q, k, v, attn_mask=mask))
    stats("SDPA (causal, padded mask)", times)

def bench_single_layer(config, seq_len: int, device="cuda", dtype=torch.float16):
    print(f"\n{'─'*60}")
    print(f" Single Transformer Layer forward — seq_len={seq_len}")

    try:
        from src.decoder_layer import DecoderLayer
        from src.kv_cache_batched import BatchedKVCache
        from src.rope import RotaryEmbedding
        from engine.utils import BatchMetadata

        with torch.device(device):
            dummy_rope = RotaryEmbedding(config.hidden_size // config.num_attention_heads)

            layer = DecoderLayer(
                hidden_size=config.hidden_size,
                num_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                intermediate_size=config.intermediate_size,
                rms_norm_eps=config.rms_norm_eps,
                rope=dummy_rope
            ).to(dtype=dtype)

            hidden = torch.randn(seq_len, config.hidden_size, dtype=dtype)
            positions = torch.arange(seq_len, dtype=torch.int32)

        kv_cache = BatchedKVCache(
            num_layers=1, max_batch_size=4, max_seq_len=2048,
            num_kv_head=config.num_key_value_heads, head_dim=config.hidden_size // config.num_attention_heads
        )
        slot = kv_cache.allocate_slot()

        cu_q = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
        cu_kv = torch.tensor([0, seq_len], dtype=torch.int32, device=device)

        metadata = BatchMetadata(
            cu_seq_lens_q=cu_q, cu_seq_lens_kv=cu_kv, max_q_len=seq_len, max_kv_len=seq_len,
            positions=positions, cache_slots=torch.tensor([slot], dtype=torch.int32, device=device),
            seq_lens=torch.tensor([seq_len], dtype=torch.int32, device=device), prompt_lens=[seq_len],
        )

        @torch.no_grad()
        def fwd():
            return layer(hidden, positions=positions, kv_cache=kv_cache, layer_idx=0, prefill=True, metadata=metadata)

        times = timeit_gpu(fwd)
        stats(f"Single layer prefill (seq={seq_len})", times)
        print(f"    → Full model estimate: {np.mean(times) * config.num_hidden_layers:.1f} ms")
    except Exception:
        traceback.print_exc()

def bench_full_model(config, seq_len: int, device="cuda", dtype=torch.float16):
    print(f"\n{'─'*60}")
    print(f" Full Model forward — seq_len={seq_len}")

    try:
        from src.transformer import QwenForCausalLM
        from src.kv_cache_batched import BatchedKVCache
        from engine.utils import BatchMetadata

        kv_cache = BatchedKVCache(
            num_layers=config.num_hidden_layers, max_batch_size=4, max_seq_len=2048,
            num_kv_head=config.num_key_value_heads, head_dim=config.hidden_size // config.num_attention_heads
        )
        slot = kv_cache.allocate_slot()

        print("    Initializing model directly on GPU (fast)...")
        with torch.device(device):
            model = QwenForCausalLM(config).to(dtype=dtype)

            input_ids = torch.randint(0, config.vocab_size, (seq_len,), dtype=torch.int32)
            positions = torch.arange(seq_len, dtype=torch.int32)
            cu_q  = torch.tensor([0, seq_len], dtype=torch.int32)
            cu_kv = torch.tensor([0, seq_len], dtype=torch.int32)
            slots_t = torch.tensor([slot], dtype=torch.int32)
            seq_lens_t = torch.tensor([seq_len], dtype=torch.int32)

        metadata = BatchMetadata(
            cu_seq_lens_q=cu_q, cu_seq_lens_kv=cu_kv, max_q_len=seq_len, max_kv_len=seq_len,
            positions=positions, cache_slots=slots_t, seq_lens=seq_lens_t, prompt_lens=[seq_len],
        )

        @torch.no_grad()
        def fwd():
            return model(input_ids=input_ids, kv_cache=kv_cache, prefill=True, metadata=metadata, positions=positions)

        for _ in range(3):
            fwd()
        torch.cuda.synchronize()

        times = timeit_gpu(fwd)
        stats(f"Full model prefill (seq={seq_len})", times)

    except Exception:
        traceback.print_exc()


def bench_decode_kernel(num_heads_q: int, num_heads_k: int, head_dim: int,
                        batch: int, history_len: int, device="cuda", dtype=torch.float16):
    print(f"\n{'─'*60}")
    print(f" Triton Decode Attention kernel — "
          f"B={batch}, H_Q={num_heads_q}, H_K={num_heads_k}, D={head_dim}, hist={history_len}")

    try:
        from src.decode_attention import decode_attention

        q = torch.randn(batch, num_heads_q, head_dim, device=device, dtype=dtype)
        k_cache = torch.randn(batch, history_len, num_heads_k, head_dim, device=device, dtype=dtype)
        v_cache = torch.randn(batch, history_len, num_heads_k, head_dim, device=device, dtype=dtype)
        slots   = torch.arange(batch, dtype=torch.int32, device=device)
        seq_lens = torch.full((batch,), history_len, dtype=torch.int32, device=device)

        times = timeit_gpu(lambda: decode_attention(q, k_cache, v_cache, slots, seq_lens))
        stats(f"decode_attention (hist={history_len})", times)

    except Exception as e:
        print(f"    [SKIPPED] {e}")


def bench_sampler(vocab_size: int, batch: int, device="cuda", dtype=torch.float16):
    print(f"\n{'─'*60}")
    print(f" Sampler — vocab={vocab_size}, batch={batch}")

    try:
        from engine.sampler import Sampler
        from engine.utils import SamplingParams

        sampler = Sampler()
        logits  = torch.randn(batch, vocab_size, device=device, dtype=dtype)
        params  = [SamplingParams.from_optional(temperature=0.7, top_p=0.9, top_k=50)
                   for _ in range(batch)]

        times_cpu = timeit_cpu(lambda: sampler.sample(logits, params))
        stats("Sampler (CPU wall)", times_cpu)

        t0 = time.perf_counter()
        sampler.sample(logits, params)
        torch.cuda.synchronize()
        print(f"    Wall+sync time: {(time.perf_counter()-t0)*1000:.2f}ms  "
              f"(if >> CPU wall above, sampler is blocking GPU)")

    except Exception as e:
        print(f"    [SKIPPED] {e}")


def sanity_check(config, device="cuda"):
    print(f"\n{'─'*60}")
    print(" Sanity checks")
    try:
        from src.transformer import QwenForCausalLM
        with torch.device(device):
            model = QwenForCausalLM(config).to(dtype=torch.float16)
        p = next(model.parameters())
        print(f"    Model device : {p.device}")
        print(f"    Model dtype  : {p.dtype}")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"    Params       : {total_params/1e6:.1f}M")
    except Exception as e:
        print(f"    [SKIPPED] {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="Qwen/Qwen2.5-1.5B-Instruct")
    args = parser.parse_args()

    from src.config import ModelConfig
    config = ModelConfig.from_hf(args.model_path)

    H_Q  = config.num_attention_heads
    H_K  = config.num_key_value_heads
    D    = config.hidden_size // H_Q
    hidden = config.hidden_size

    print("=" * 60)
    print("  FAST ISOLATION MICROBENCHMARK SUITE")
    print("=" * 60)

    sanity_check(config)
    bench_linear(hidden, seq_len=50)
    bench_linear(hidden, seq_len=1)
    bench_sdpa(H_Q, D, seq_len=50)
    bench_sdpa(H_Q, D, seq_len=1)
    bench_single_layer(config, seq_len=50)
    bench_single_layer(config, seq_len=1)
    bench_full_model(config, seq_len=50)
    bench_full_model(config, seq_len=1)

if __name__ == "__main__":
    with torch.no_grad():
        main()
