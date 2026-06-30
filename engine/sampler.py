import torch
from .utils import SamplingParams

class Sampler:
    def sample(self, logits: torch.Tensor, sampling_params: list[SamplingParams]) -> torch.Tensor:
        B, V = logits.shape

        temp = sampling_params[0].temperature
        top_p = sampling_params[0].top_p

        if temp == 0.0:
            return torch.argmax(logits, dim=-1)

        logits = logits / temp

        if top_p < 1.0:
            k = min(512, V)
            topk_logits, topk_indices = torch.topk(logits, k, dim=-1)

            topk_probs = torch.softmax(topk_logits, dim=-1)
            cumulative_probs = torch.cumsum(topk_probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False

            topk_probs = topk_probs.masked_fill(sorted_indices_to_remove, 0.0)
            topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

            sampled_local = torch.multinomial(topk_probs, num_samples=1).squeeze(-1)
            next_tokens = topk_indices.gather(1, sampled_local.unsqueeze(-1)).squeeze(-1)
        else:
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return next_tokens
