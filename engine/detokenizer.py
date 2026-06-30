from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

_REPLACEMENT = "�"


class IncrementalDetokenizer:
    def __init__(
        self,
        tokenizer,
        prompt_token_ids: Sequence[int],
        skip_special_tokens: bool = True,
        context_window: int = 8,
    ) -> None:
        self.tokenizer = tokenizer
        self.skip_special_tokens = skip_special_tokens
        self.context_window = context_window

        self.token_ids: List[int] = list(prompt_token_ids)
        self.read_offset = len(self.token_ids)
        self.prefix_offset = max(0, self.read_offset - context_window)
        self.text = ""

    def _decode(self, start: int, end: int) -> str:
        return self.tokenizer.decode(
            self.token_ids[start:end], skip_special_tokens=self.skip_special_tokens
        )

    def append(self, token_id: int) -> str:
        self.token_ids.append(token_id)

        prefix_text = self._decode(self.prefix_offset, self.read_offset)
        new_text = self._decode(self.prefix_offset, len(self.token_ids))

        if new_text.endswith(_REPLACEMENT):
            return ""

        if not new_text.startswith(prefix_text):
            delta = new_text
            self.prefix_offset = self.read_offset
        else:
            delta = new_text[len(prefix_text) :]
            self.prefix_offset = max(0, len(self.token_ids) - self.context_window)

        self.read_offset = len(self.token_ids)
        self.text += delta
        return delta

    def extend(self, token_ids: Sequence[int]) -> str:
        return "".join(self.append(token_id) for token_id in token_ids)


class StopStringMatcher:
    def __init__(self, stop_strings: Sequence[str], include_stop_str: bool = False) -> None:
        self.stop_strings = [s for s in stop_strings if s]
        self.include_stop_str = include_stop_str
        self.max_len = max((len(s) for s in self.stop_strings), default=0)
        self.buffer = ""
        self.matched: Optional[str] = None

    @property
    def active(self) -> bool:
        return bool(self.stop_strings)

    def feed(self, delta: str) -> Tuple[str, bool]:
        if not self.stop_strings:
            return delta, False
        if self.matched is not None:
            return "", True

        self.buffer += delta

        earliest = None
        for stop in self.stop_strings:
            index = self.buffer.find(stop)
            if index != -1 and (earliest is None or index < earliest[0]):
                earliest = (index, stop)

        if earliest is not None:
            index, stop = earliest
            self.matched = stop
            end = index + len(stop) if self.include_stop_str else index
            emit = self.buffer[:end]
            self.buffer = ""
            return emit, True

        hold = self._suffix_overlap()
        emit = self.buffer[: len(self.buffer) - hold] if hold else self.buffer
        self.buffer = self.buffer[len(emit) :]
        return emit, False

    def flush(self) -> str:
        pending, self.buffer = self.buffer, ""
        return pending

    def _suffix_overlap(self) -> int:
        limit = min(len(self.buffer), self.max_len - 1)
        for length in range(limit, 0, -1):
            suffix = self.buffer[-length:]
            if any(stop.startswith(suffix) for stop in self.stop_strings):
                return length
        return 0
