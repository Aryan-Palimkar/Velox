from __future__ import annotations

import gc

import pytest
import torch

from engine.config import EngineConfig
from engine.engine import Engine
from engine.request import Request
from engine.utils import SamplingParams

pytestmark = [pytest.mark.gpu, pytest.mark.slow]

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
PROMPTS = [
    "What is the capital of France?",
    "List three prime numbers.",
    "Write one sentence about the ocean.",
]


_SHARED: dict = {}


def _free_engine(engine: Engine) -> None:
    del engine
    gc.collect()
    torch.cuda.empty_cache()


def _release_shared() -> None:
    instance = _SHARED.pop("engine", None)
    if instance is not None:
        _free_engine(instance)


@pytest.fixture(scope="module")
def tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(MODEL_ID)


def _chat_tokens(tokenizer, prompt: str):
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(text)


def _build_engine(**overrides) -> Engine:
    config = EngineConfig(
        model=MODEL_ID,
        max_model_len=1024,
        max_num_seqs=overrides.pop("max_num_seqs", 8),
        max_num_batched_tokens=overrides.pop("max_num_batched_tokens", 1024),
        max_prefill_chunk_size=overrides.pop("max_prefill_chunk_size", 256),
        gpu_memory_utilization=overrides.pop("gpu_memory_utilization", 0.75),
        num_gpu_blocks=overrides.pop("num_gpu_blocks", 1024),
        enable_prefix_caching=overrides.pop("enable_prefix_caching", True),
        enable_cuda_graphs=overrides.pop("enable_cuda_graphs", True),
        kv_cache_dtype=overrides.pop("kv_cache_dtype", "auto"),
        quantization=overrides.pop("quantization", None),
        seed=0,
    )
    assert not overrides, f"unexpected overrides: {overrides}"
    return Engine.from_pretrained(config)


def assert_similar(actual, expected, tokenizer, label: str, min_ratio: float = 0.75) -> None:
    common = 0
    for a, b in zip(actual, expected):
        if a != b:
            break
        common += 1
    total = min(len(actual), len(expected))
    assert common >= max(1, int(total * min_ratio)), (
        f"{label}: outputs diverged after {common}/{total} tokens\n"
        f"  actual  ={tokenizer.decode(actual)!r}\n"
        f"  expected={tokenizer.decode(expected)!r}"
    )


def _greedy(engine: Engine, tokenizer, prompts, max_tokens=24, ignore_eos=False):
    requests = [
        Request(
            request_id=f"t{index}",
            prompt_token_ids=_chat_tokens(tokenizer, prompt),
            sampling_params=SamplingParams(
                temperature=0.0, max_tokens=max_tokens, ignore_eos=ignore_eos
            ),
        )
        for index, prompt in enumerate(prompts)
    ]
    engine.generate(requests)
    return [request.output_token_ids for request in requests]


@pytest.fixture
def engine(tokenizer):
    if "engine" not in _SHARED:
        _SHARED["engine"] = _build_engine()
    return _SHARED["engine"]


def _velox_prefill_logits(engine: Engine, token_ids):
    request = Request(
        request_id="logits-probe",
        prompt_token_ids=list(token_ids),
        sampling_params=SamplingParams(temperature=0.0, max_tokens=1),
    )
    engine.scheduler.add_request(request)
    scheduled = engine.scheduler.schedule()
    assert scheduled.prefill_batch == [request]
    assert request.current_chunk_size == len(token_ids), "prompt did not prefill in one chunk"

    with torch.no_grad():
        input_ids, metadata = engine.runner.prepare_prefill([request])
        hidden = engine.runner.forward(input_ids, metadata)
        logits = engine.model.compute_logits(hidden)

    engine.scheduler.abort_request(request.request_id)
    return logits.float()


def test_prefill_logits_match_huggingface(engine, tokenizer):
    from transformers import AutoModelForCausalLM

    text = (
        "The ocean covers most of the planet. Marine biologists study its "
        "ecosystems, from coral reefs to the deep abyssal plains, where "
        "pressure and darkness shape unusual forms of life."
    )
    token_ids = tokenizer.encode(text)
    assert len(token_ids) > 32, "the probe text should span several cache blocks"

    actual = _velox_prefill_logits(engine, token_ids)

    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).eval()
    with torch.no_grad():
        expected = reference_model(torch.tensor([token_ids])).logits[0].float()
    del reference_model
    gc.collect()

    assert actual.shape == expected.shape

    agreement = (actual.argmax(-1).cpu() == expected.argmax(-1)).float().mean().item()
    assert agreement >= 0.95, f"argmax agreed on only {agreement:.1%} of positions"

    cosine = torch.nn.functional.cosine_similarity(
        actual.cpu(), expected, dim=-1
    )
    assert cosine.min().item() >= 0.99, f"worst-position cosine similarity {cosine.min():.4f}"

    assert actual[-1].argmax().item() == expected[-1].argmax().item()


def test_first_generated_token_matches_huggingface(engine, tokenizer):
    from transformers import AutoModelForCausalLM

    velox_first = [output[0] for output in _greedy(engine, tokenizer, PROMPTS, max_tokens=1)]

    reference_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16
    ).eval()
    expected = []
    with torch.no_grad():
        for prompt in PROMPTS:
            token_ids = _chat_tokens(tokenizer, prompt)
            logits = reference_model(torch.tensor([token_ids])).logits[0, -1]
            expected.append(int(logits.argmax()))
    del reference_model
    gc.collect()

    assert velox_first == expected, (
        f"velox={[tokenizer.decode([t]) for t in velox_first]} "
        f"hf={[tokenizer.decode([t]) for t in expected]}"
    )


def test_batched_matches_one_at_a_time(engine, tokenizer):
    prompts = [
        "What is the capital of France?",
        "List three prime numbers.",
        "Write one sentence about the ocean.",
        "Name a colour.",
        "What is two plus two?",
    ]
    solo = [_greedy(engine, tokenizer, [prompt], max_tokens=20)[0] for prompt in prompts]
    batched = _greedy(engine, tokenizer, prompts, max_tokens=20)

    for prompt, alone, together in zip(prompts, solo, batched):
        assert alone == together, (
            f"batching changed the output of {prompt!r}\n"
            f"  alone   ={tokenizer.decode(alone)!r}\n"
            f"  batched ={tokenizer.decode(together)!r}"
        )


def test_repeated_prompt_is_deterministic(engine, tokenizer):
    first = _greedy(engine, tokenizer, PROMPTS)
    second = _greedy(engine, tokenizer, PROMPTS)
    assert first == second


def test_prefix_cache_does_not_change_output(engine, tokenizer):
    shared = "You are a helpful assistant. " * 12
    prompts = [shared + tail for tail in ("Say hello.", "Say goodbye.", "Say thanks.")]

    engine.block_manager.prefix_cache.clear()
    cold = _greedy(engine, tokenizer, prompts)
    cached = _greedy(engine, tokenizer, prompts)

    for first, second in zip(cold, cached):
        assert_similar(second, first, tokenizer, "warm vs cold prefix cache")
    assert engine.block_manager.prefix_cache.num_cached_tokens > 0


def test_chunked_prefill_does_not_change_output(tokenizer):
    _release_shared()
    long_prompt = "Count the words in this sentence. " * 20
    prompts = [long_prompt]

    coarse = _build_engine(max_prefill_chunk_size=512, enable_prefix_caching=False)
    reference = _greedy(coarse, tokenizer, prompts, max_tokens=16)
    _free_engine(coarse)

    fine = _build_engine(max_prefill_chunk_size=16, enable_prefix_caching=False)
    chunked = _greedy(fine, tokenizer, prompts, max_tokens=16)
    _free_engine(fine)

    assert_similar(chunked[0], reference[0], tokenizer, "chunked vs single-shot prefill")


def test_cuda_graph_decode_matches_eager(engine, tokenizer):
    from engine.request import RequestStatus

    assert engine.runner.uses_cuda_graphs

    requests = [
        Request(
            request_id=f"graph-{index}",
            prompt_token_ids=_chat_tokens(tokenizer, prompt),
            sampling_params=SamplingParams(temperature=0.0, max_tokens=64, ignore_eos=True),
        )
        for index, prompt in enumerate(PROMPTS)
    ]
    for request in requests:
        engine.add_request(request)

    for _ in range(24):
        engine.step()
    batch = [r for r in engine.scheduler.running if r.status is RequestStatus.RUNNING_DECODE]
    assert len(batch) == len(requests)

    try:
        with torch.no_grad():
            graph_logits = engine.model.compute_logits(
                engine.runner.execute_decode(batch)
            ).float()

            saved_graphs = engine.runner.graphs
            saved_buckets = engine.runner.graph_buckets
            engine.runner.graphs = {}
            engine.runner.graph_buckets = []
            try:
                eager_logits = engine.model.compute_logits(
                    engine.runner.execute_decode(batch)
                ).float()
            finally:
                engine.runner.graphs = saved_graphs
                engine.runner.graph_buckets = saved_buckets

        assert torch.equal(graph_logits.argmax(-1), eager_logits.argmax(-1))
        cosine = torch.nn.functional.cosine_similarity(graph_logits, eager_logits, dim=-1)
        assert cosine.min().item() >= 0.995, f"cosine similarity {cosine.min():.5f}"
    finally:
        for request in requests:
            engine.abort_request(request.request_id)
        engine.step()


def test_preemption_does_not_change_output(tokenizer):
    _release_shared()
    prompts = [f"Name a colour that starts with {letter}." for letter in "ABCDEFGH"]
    params = dict(max_tokens=48, ignore_eos=True)

    roomy = _build_engine(num_gpu_blocks=2048, enable_prefix_caching=False)
    expected = _greedy(roomy, tokenizer, prompts, **params)
    _free_engine(roomy)

    cramped = _build_engine(num_gpu_blocks=40, enable_prefix_caching=False)
    actual = _greedy(cramped, tokenizer, prompts, **params)
    preemptions = cramped.scheduler.num_preemptions
    _free_engine(cramped)

    assert preemptions > 0, "the constrained pool should have forced a preemption"
    for prompt, got, want in zip(prompts, actual, expected):
        assert len(got) == 48
        assert_similar(got, want, tokenizer, f"preempted vs roomy on {prompt!r}")


def test_engine_frees_all_blocks_after_completion(engine, tokenizer):
    engine.block_manager.prefix_cache.clear()
    before = engine.kv_cache.allocator.num_free_blocks
    _greedy(engine, tokenizer, PROMPTS, max_tokens=12)
    engine.block_manager.prefix_cache.clear()
    assert engine.kv_cache.allocator.num_free_blocks == before
    engine.kv_cache.allocator.assert_consistent()


def test_per_request_sampling_params_are_independent(engine, tokenizer):
    token_ids = _chat_tokens(tokenizer, "Name one colour.")
    greedy_reference = Request(
        "solo", token_ids, SamplingParams(temperature=0.0, max_tokens=12)
    )
    engine.generate([greedy_reference])

    batched = [
        Request("g", list(token_ids), SamplingParams(temperature=0.0, max_tokens=12)),
        Request("s1", list(token_ids), SamplingParams(temperature=1.5, top_p=0.95, max_tokens=12)),
        Request("s2", list(token_ids), SamplingParams(temperature=1.5, top_p=0.95, max_tokens=12)),
    ]
    engine.generate(batched)

    assert batched[0].output_token_ids == greedy_reference.output_token_ids


def test_max_tokens_is_respected(engine, tokenizer):
    token_ids = _chat_tokens(tokenizer, "Write a long story about a robot.")
    request = Request("len", token_ids, SamplingParams(temperature=0.0, max_tokens=7))
    engine.generate([request])
    assert len(request.output_token_ids) == 7
    assert request.finish_reason.value == "length"


def test_shared_engine_teardown():
    _release_shared()
