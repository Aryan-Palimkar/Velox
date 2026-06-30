from enum import Enum, auto
import time
from engine.utils import SamplingParams

class RequestStatus(Enum):
    WAITING_PREFILL = auto()
    RUNNING_PREFILL = auto()
    RUNNING_DECODE = auto()
    FINISHED = auto()
    ABORTED = auto()

class Request:
    def __init__(
        self,
        request_id: str,
        prompt_token_ids: list[int],
        sampling_params: SamplingParams,
        arrival_time: float | None = None,
    ):
        self.request_id = request_id

        self.arrival_time = arrival_time or time.time()
        self.status = RequestStatus.WAITING_PREFILL

        self.prompt_token_ids = prompt_token_ids
        self.num_prompt_tokens = len(prompt_token_ids)

        self.sampling_params = sampling_params
        self.max_tokens = sampling_params.max_tokens

        self.output_token_ids: list[int] = []

        self.all_token_ids = prompt_token_ids.copy()

        self.num_computed_tokens = 0

        self.finished = False
        self.stop_reason: str | int | None = None

        self.cache_slot: int | None = None

        self.num_preemptions = 0
