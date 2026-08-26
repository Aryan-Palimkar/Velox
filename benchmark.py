from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from engine.config import EngineConfig
from engine.engine import Engine
from engine.request import Request
from engine.utils import SamplingParams

SHORT_PROMPTS = [
    "What is the capital of France?",
    "Name three prime numbers.",
    "Translate 'good morning' into Spanish.",
    "What colour is the sky on a clear day?",
    "Give one example of a mammal that lays eggs.",
    "What does CPU stand for?",
    "Who wrote Pride and Prejudice?",
    "What is the boiling point of water in Celsius?",
]

LONG_PROMPTS = [
    "Write a detailed essay on the history of the Roman Empire, focusing on the "
    "transition from Republic to Empire, the reign of Augustus, and the eventual "
    "fall of the Western Roman Empire. Include specific dates and figures.",
    "Explain how a modern GPU executes a matrix multiplication, covering warps, "
    "shared memory, tensor cores, and the memory hierarchy, then describe how a "
    "tiled algorithm maps onto that hardware.",
    "Describe the process of photosynthesis in detail, from light absorption in "
    "the thylakoid membrane through the Calvin cycle, and explain why C4 plants "
    "evolved a different pathway.",
    "Summarise the causes of the 2008 financial crisis, the regulatory response, "
    "and the lasting changes to how banks are supervised.",
]

SHARED_SYSTEM_PROMPT = (
    "You are a careful, concise technical assistant. Answer accurately, cite "
    "your reasoning briefly, and prefer concrete examples over generalities. "
    "If a question is ambiguous, state the interpretation you are using. "
    "Never invent facts you are not confident about."
)


@dataclass
class RequestTrace:
    request_id: str
    prompt_len: int
    arrival_time: float
    first_token_time: Optional[float] = None
    token_times: List[float] = field(default_factory=list)
    finish_time: Optional[float] = None
    num_cached_tokens: int = 0

    @property
    def ttft(self) -> Optional[float]:
        if self.first_token_time is None:
            return None
        return self.first_token_time - self.arrival_time

    @property
    def inter_token_latencies_ms(self) -> List[float]:
        return [
            (self.token_times[i] - self.token_times[i - 1]) * 1000
            for i in range(1, len(self.token_times))
        ]

    @property
    def num_generated(self) -> int:
        return len(self.token_times)

    @property
    def end_to_end(self) -> Optional[float]:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time


def percentile(values: Sequence[float], p: float) -> float:
    return float(np.percentile(values, p)) if len(values) else 0.0


def summarise(values: Sequence[float]) -> Dict[str, float]:
    if not len(values):
        return dict(n=0, mean=0.0, p50=0.0, p90=0.0, p99=0.0, min=0.0, max=0.0)
    array = np.asarray(values, dtype=float)
    return dict(
        n=len(array),
        mean=float(array.mean()),
        p50=percentile(array, 50),
        p90=percentile(array, 90),
        p99=percentile(array, 99),
        min=float(array.min()),
        max=float(array.max()),
    )


def rule(title: str = "") -> None:
    if title:
        print("\n" + "=" * 78)
        print(f"  {title}")
        print("=" * 78)
    else:
        print("-" * 78)


def build_prompts(workload: str, count: int, shared_prefix: bool) -> List[List[dict]]:
    if workload == "short":
        texts = SHORT_PROMPTS
    elif workload == "long":
        texts = LONG_PROMPTS
    else:
        texts = [
            text
            for pair in zip(SHORT_PROMPTS, LONG_PROMPTS * 2)
            for text in pair
        ]

    system = (
        [{"role": "system", "content": SHARED_SYSTEM_PROMPT}] if shared_prefix else []
    )
    return [
        system + [{"role": "user", "content": texts[index % len(texts)]}]
        for index in range(count)
    ]


def run_workload(
    engine: Engine,
    tokenizer,
    conversations: Sequence[Sequence[dict]],
    sampling_params: SamplingParams,
    label: str = "workload",
) -> Dict:
    requests: List[Request] = []
    traces: Dict[str, RequestTrace] = {}

    start = time.perf_counter()
    for index, messages in enumerate(conversations):
        prompt = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        token_ids = tokenizer.encode(prompt)
        request = Request(
            request_id=f"{label}-{index}",
            prompt_token_ids=token_ids,
            sampling_params=sampling_params,
        )
        requests.append(request)
        traces[request.request_id] = RequestTrace(
            request_id=request.request_id,
            prompt_len=len(token_ids),
            arrival_time=start,
        )
        engine.add_request(request)

    batch_log = []
    seen = {request.request_id: 0 for request in requests}

    while engine.has_work():
        engine.step()
        now = time.perf_counter()
        batch_log.append(
            {
                "running": engine.scheduler.num_running,
                "waiting": engine.scheduler.num_waiting,
                "blocks_used": engine.block_manager.allocator.num_used_blocks,
            }
        )
        for request in requests:
            trace = traces[request.request_id]
            produced = len(request.output_token_ids)
            if produced > seen[request.request_id]:
                trace.token_times.extend([now] * (produced - seen[request.request_id]))
                seen[request.request_id] = produced
                if trace.first_token_time is None:
                    trace.first_token_time = trace.token_times[0]
            if request.finished and trace.finish_time is None:
                trace.finish_time = now
                trace.num_cached_tokens = request.num_cached_tokens

    torch.cuda.synchronize()
    wall = time.perf_counter() - start

    prompt_tokens = sum(trace.prompt_len for trace in traces.values())
    generated_tokens = sum(trace.num_generated for trace in traces.values())
    cached_tokens = sum(trace.num_cached_tokens for trace in traces.values())

    all_ttft = [t.ttft for t in traces.values() if t.ttft is not None]
    all_itl = [value for t in traces.values() for value in t.inter_token_latencies_ms]

    return {
        "label": label,
        "wall_seconds": wall,
        "num_requests": len(requests),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "cached_prompt_tokens": cached_tokens,
        "decode_throughput": generated_tokens / wall if wall else 0.0,
        "total_throughput": (prompt_tokens + generated_tokens) / wall if wall else 0.0,
        "ttft": summarise(all_ttft),
        "itl_ms": summarise(all_itl),
        "steps": engine.num_steps,
        "preemptions": engine.scheduler.num_preemptions,
        "max_running": max((entry["running"] for entry in batch_log), default=0),
        "max_blocks_used": max((entry["blocks_used"] for entry in batch_log), default=0),
        "requests": [
            {
                "id": trace.request_id,
                "prompt_tokens": trace.prompt_len,
                "generated_tokens": trace.num_generated,
                "ttft": trace.ttft,
                "end_to_end": trace.end_to_end,
                "cached_prompt_tokens": trace.num_cached_tokens,
            }
            for trace in traces.values()
        ],
        "outputs": {
            request.request_id: tokenizer.decode(
                request.output_token_ids, skip_special_tokens=True
            )
            for request in requests
        },
    }


def report(result: Dict, engine: Engine, show_outputs: bool = False) -> None:
    rule(f"RESULTS: {result['label']}")
    print(f"  Requests              : {result['num_requests']}")
    print(f"  Wall time             : {result['wall_seconds']:.3f} s")
    print(f"  Engine steps          : {result['steps']}")
    print(f"  Prompt tokens         : {result['prompt_tokens']}")
    if result["cached_prompt_tokens"]:
        share = result["cached_prompt_tokens"] / max(1, result["prompt_tokens"])
        print(
            f"  Prompt tokens cached  : {result['cached_prompt_tokens']} ({share:.1%})"
        )
    print(f"  Generated tokens      : {result['generated_tokens']}")
    print(f"  Decode throughput     : {result['decode_throughput']:.2f} tok/s")
    print(f"  End-to-end throughput : {result['total_throughput']:.2f} tok/s")
    print(f"  Peak running requests : {result['max_running']}")
    print(f"  Peak KV blocks in use : {result['max_blocks_used']}")
    print(f"  Preemptions           : {result['preemptions']}")

    ttft, itl = result["ttft"], result["itl_ms"]
    print(
        f"\n  TTFT (s)   mean={ttft['mean']:.3f}  p50={ttft['p50']:.3f}  "
        f"p90={ttft['p90']:.3f}  p99={ttft['p99']:.3f}  max={ttft['max']:.3f}"
    )
    print(
        f"  ITL (ms)   mean={itl['mean']:.2f}  p50={itl['p50']:.2f}  "
        f"p90={itl['p90']:.2f}  p99={itl['p99']:.2f}  max={itl['max']:.2f}"
    )

    rule()
    print("  Per-step breakdown (ms)")
    header = f"  {'stage':<20}{'n':>7}{'mean':>10}{'p50':>10}{'p90':>10}{'p99':>10}{'total(s)':>11}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for stage, values in engine.stats.items():
        stats = summarise(values)
        print(
            f"  {stage:<20}{stats['n']:>7}{stats['mean']:>10.3f}{stats['p50']:>10.3f}"
            f"{stats['p90']:>10.3f}{stats['p99']:>10.3f}{sum(values) / 1000:>11.3f}"
        )

    gpu_ms = sum(engine.stats["prefill_gpu_ms"]) + sum(engine.stats["decode_gpu_ms"])
    busy = gpu_ms / 1000 / result["wall_seconds"] if result["wall_seconds"] else 0.0
    print(f"\n  GPU busy      : {gpu_ms / 1000:.3f} s  ({busy:.1%} of wall time)")
    print(f"  CUDA graphs   : {'on' if engine.runner.uses_cuda_graphs else 'off'}")
    print(f"  Peak GPU alloc: {torch.cuda.max_memory_allocated() / 1e9:.3f} GB")

    if show_outputs:
        rule("SAMPLE OUTPUTS")
        for request_id, text in list(result["outputs"].items())[:4]:
            print(f"\n  [{request_id}]\n  {text[:400]}")


def run_huggingface_baseline(
    model_id: str, tokenizer, conversations: Sequence[Sequence[dict]], max_tokens: int
) -> Dict:
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    start = time.perf_counter()
    generated = 0
    prompt_tokens = 0
    ttfts = []

    for messages in conversations:
        prompt = tokenizer.apply_chat_template(
            list(messages), tokenize=False, add_generation_prompt=True
        )
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        prompt_tokens += input_ids.shape[1]

        request_start = time.perf_counter()
        with torch.no_grad():
            first = model.generate(
                input_ids,
                max_new_tokens=1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()
        ttfts.append(time.perf_counter() - request_start)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        generated += output.shape[1] - input_ids.shape[1]
        del first

    torch.cuda.synchronize()
    wall = time.perf_counter() - start
    del model
    torch.cuda.empty_cache()

    return {
        "label": "huggingface-sequential",
        "wall_seconds": wall,
        "num_requests": len(conversations),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated,
        "decode_throughput": generated / wall if wall else 0.0,
        "ttft": summarise(ttfts),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end benchmark for the Velox engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--workload", default="mixed", choices=["short", "long", "mixed"])
    parser.add_argument("--num-requests", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)

    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-num-batched-tokens", type=int, default=2048)
    parser.add_argument("--max-prefill-chunk-size", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--num-gpu-blocks", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8", "fp8_e5m2", "int8"])
    parser.add_argument("--quantization", default=None, choices=[None, "int8", "fp8"])

    parser.add_argument("--no-prefix-caching", action="store_true")
    parser.add_argument("--no-cuda-graphs", action="store_true")
    parser.add_argument("--shared-system-prompt", action="store_true",
                        help="prepend one shared system prompt to every request, "
                             "which is what the prefix cache is meant to exploit")

    parser.add_argument("--compare-hf", action="store_true",
                        help="also run a sequential transformers baseline")
    parser.add_argument("--sweep-batch-size", default=None,
                        help="comma-separated max_num_seqs values to sweep")
    parser.add_argument("--show-outputs", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def config_from_args(args: argparse.Namespace, max_num_seqs: Optional[int] = None) -> EngineConfig:
    return EngineConfig(
        model=args.model,
        max_model_len=args.max_model_len,
        max_num_seqs=max_num_seqs or args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_prefill_chunk_size=args.max_prefill_chunk_size,
        block_size=args.block_size,
        num_gpu_blocks=args.num_gpu_blocks,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        quantization=args.quantization,
        enable_prefix_caching=not args.no_prefix_caching,
        enable_cuda_graphs=not args.no_cuda_graphs,
        seed=args.seed,
    )


def print_environment(config: EngineConfig) -> None:
    rule("VELOX BENCHMARK")
    print(f"  Model            : {config.model}")
    print(f"  PyTorch          : {torch.__version__}")
    try:
        import triton

        print(f"  Triton           : {triton.__version__}")
    except ImportError:
        pass
    print(f"  CUDA             : {torch.version.cuda}")
    print(f"  GPU              : {torch.cuda.get_device_name(0)}")
    properties = torch.cuda.get_device_properties(0)
    print(f"  GPU memory       : {properties.total_memory / 1e9:.1f} GB")
    print(f"  Compute cap.     : {properties.major}.{properties.minor}")
    print(f"  SMs              : {properties.multi_processor_count}")
    print(f"  Prefix caching   : {config.enable_prefix_caching}")
    print(f"  CUDA graphs      : {config.enable_cuda_graphs}")
    print(f"  KV cache dtype   : {config.kv_cache_dtype}")
    print(f"  Weight quant     : {config.quantization or 'none'}")


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("this benchmark requires a CUDA device")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    conversations = build_prompts(args.workload, args.num_requests, args.shared_system_prompt)
    sampling_params = SamplingParams(
        temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens
    )

    base_config = config_from_args(args)
    print_environment(base_config)

    results = []
    batch_sizes = (
        [int(value) for value in args.sweep_batch_size.split(",")]
        if args.sweep_batch_size
        else [args.max_num_seqs]
    )

    for max_num_seqs in batch_sizes:
        config = config_from_args(args, max_num_seqs=max_num_seqs)
        engine = Engine.from_pretrained(config)
        torch.cuda.reset_peak_memory_stats()

        label = f"velox(max_num_seqs={max_num_seqs})"
        result = run_workload(engine, tokenizer, conversations, sampling_params, label=label)
        result["kv_blocks"] = engine.block_manager.num_total_blocks
        result["kv_capacity_tokens"] = engine.kv_cache.num_slots
        result["prefix_hit_rate"] = engine.block_manager.prefix_cache.hit_rate
        report(result, engine, show_outputs=args.show_outputs)
        results.append(result)

        del engine
        torch.cuda.empty_cache()

    if args.compare_hf:
        baseline = run_huggingface_baseline(
            args.model, tokenizer, conversations, args.max_tokens
        )
        results.append(baseline)
        rule("HUGGINGFACE BASELINE (sequential)")
        print(f"  Wall time         : {baseline['wall_seconds']:.3f} s")
        print(f"  Generated tokens  : {baseline['generated_tokens']}")
        print(f"  Decode throughput : {baseline['decode_throughput']:.2f} tok/s")
        print(f"  Mean TTFT         : {baseline['ttft']['mean']:.3f} s")

        velox = results[0]
        rule("SPEEDUP")
        print(f"  Wall time         : {baseline['wall_seconds'] / velox['wall_seconds']:.2f}x")
        print(
            f"  Decode throughput : "
            f"{velox['decode_throughput'] / max(baseline['decode_throughput'], 1e-9):.2f}x"
        )
        print(
            f"  Mean TTFT         : "
            f"{baseline['ttft']['mean'] / max(velox['ttft']['mean'], 1e-9):.2f}x lower"
        )

    if len(batch_sizes) > 1:
        rule("BATCH SIZE SWEEP")
        print(f"  {'max_num_seqs':>14}{'wall(s)':>10}{'tok/s':>10}{'TTFT p50':>11}{'ITL p50':>10}{'preempt':>9}")
        for size, result in zip(batch_sizes, results):
            print(
                f"  {size:>14}{result['wall_seconds']:>10.2f}"
                f"{result['decode_throughput']:>10.2f}"
                f"{result['ttft']['p50']:>11.3f}{result['itl_ms']['p50']:>10.2f}"
                f"{result['preemptions']:>9}"
            )

    if args.output_json:
        for result in results:
            result.pop("outputs", None)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nWrote {args.output_json}")


if __name__ == "__main__":
    main()
