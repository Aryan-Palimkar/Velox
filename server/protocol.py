from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


def _request_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def _now() -> int:
    return int(time.time())


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class StreamOptions(BaseModel):
    include_usage: bool = False


class _SamplingFields(BaseModel):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    n: int = 1
    user: Optional[str] = None

    logprobs: Optional[Any] = None
    top_logprobs: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Any] = None
    functions: Optional[List[Any]] = None
    response_format: Optional[Any] = None

    def stop_strings(self) -> List[str]:
        if self.stop is None:
            return []
        if isinstance(self.stop, str):
            return [self.stop]
        return list(self.stop)

    def unsupported_fields(self) -> List[str]:
        unsupported = []
        if self.logprobs:
            unsupported.append("logprobs")
        if self.top_logprobs:
            unsupported.append("top_logprobs")
        if self.logit_bias:
            unsupported.append("logit_bias")
        if self.tools or self.functions or (self.tool_choice not in (None, "none")):
            unsupported.append("tools")
        if self.response_format and getattr(self.response_format, "get", lambda _k: None)("type") not in (
            None,
            "text",
        ):
            unsupported.append("response_format")
        if self.n != 1:
            unsupported.append("n>1")
        return unsupported


class ChatCompletionRequest(_SamplingFields):
    model: Optional[str] = None
    messages: List[ChatMessage]
    add_generation_prompt: bool = True


class CompletionRequest(_SamplingFields):
    model: Optional[str] = None
    prompt: Union[str, List[str]]
    echo: bool = False

    def single_prompt(self) -> str:
        if isinstance(self.prompt, str):
            return self.prompt
        if len(self.prompt) != 1:
            raise ValueError("batched prompt lists are not supported; send one prompt per request")
        return self.prompt[0]


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_cached_tokens: int = 0


class ChatCompletionResponseMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatCompletionChoice(BaseModel):
    index: int = 0
    message: ChatCompletionResponseMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _request_id("chatcmpl"))
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=_now)
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class DeltaMessage(BaseModel):
    role: Optional[Literal["assistant"]] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    index: int = 0
    delta: DeltaMessage
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    usage: Optional[UsageInfo] = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _request_id("cmpl"))
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=_now)
    model: str
    choices: List[CompletionChoice]
    usage: UsageInfo


class CompletionChunk(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Optional[UsageInfo] = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int = Field(default_factory=_now)
    owned_by: str = "velox"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: List[ModelCard]


class ErrorDetail(BaseModel):
    message: str
    type: str = "invalid_request_error"
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    error: ErrorDetail
