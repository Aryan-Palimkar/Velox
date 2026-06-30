import time
import asyncio
import threading
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import torch
from transformers import AutoTokenizer

from src.config import ModelConfig
from src.transformer import QwenForCausalLM
from src.kv_cache_batched import BatchedKVCache
from engine.schedulerv2 import Scheduler
from engine.engine import Engine
from engine.request import Request, RequestStatus
from engine.utils import SamplingParams

from run_engine import load_fused_qwen_weights

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    max_tokens: int = 150
    temperature: float = 0.7
    top_p: float = 0.9

app = FastAPI(title="Velox Inference API")
engine: Engine = None
tokenizer = None
request_counter = 0

@app.on_event("startup")
def startup_event():
    global engine, tokenizer
    device = "cuda"
    dtype = torch.bfloat16
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    print("Loading Velox Engine...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    config = ModelConfig.from_hf(model_id)

    with torch.device(device):
        model = QwenForCausalLM(config).to(dtype=dtype)

    load_fused_qwen_weights(model, config.num_hidden_layers, model_id)

    kv_cache = BatchedKVCache(
        num_layers=config.num_hidden_layers,
        max_batch_size=8,
        max_seq_len=2048,
        num_kv_head=config.num_key_value_heads,
        head_dim=config.hidden_size // config.num_attention_heads,
        dtype=dtype
    )

    scheduler = Scheduler(kv_cache=kv_cache, max_batch_size=8, max_prefill_chunk_size=256)
    engine = Engine(model=model, scheduler=scheduler, device=device)

    threading.Thread(target=engine_worker_loop, daemon=True).start()
    print("Velox Engine API Live. Listening for requests...")

def engine_worker_loop():
    global engine

    with torch.no_grad():
        while True:
            if engine.scheduler.waiting_queue or engine.scheduler.running_queue:
                engine.step()
            else:
                time.sleep(0.001)

@app.post("/v1/chat/completions")
async def chat_completions(req_body: ChatCompletionRequest):
    global engine, tokenizer, request_counter

    request_id = f"req_{request_counter}"
    request_counter += 1

    messages_dict = [{"role": msg.role, "content": msg.content} for msg in req_body.messages]
    prompt_text = tokenizer.apply_chat_template(messages_dict, tokenize=False, add_generation_prompt=True)
    token_ids = tokenizer.encode(prompt_text)

    sampling_params = SamplingParams.from_optional(
        temperature=req_body.temperature,
        top_p=req_body.top_p,
        max_tokens=req_body.max_tokens
    )
    req = Request(request_id=request_id, prompt_token_ids=token_ids, sampling_params=sampling_params)

    engine.scheduler.add_request(req)

    while not req.finished:
        await asyncio.sleep(0.05)

    output_text = tokenizer.decode(req.output_token_ids, skip_special_tokens=True)

    return {
        "id": request_id,
        "object": "chat.completion",
        "model": "velox-qwen-1.5b",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": output_text
            },
            "finish_reason": req.stop_reason
        }],
        "usage": {
            "prompt_tokens": req.num_prompt_tokens,
            "completion_tokens": len(req.output_token_ids),
            "total_tokens": len(req.all_token_ids)
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
