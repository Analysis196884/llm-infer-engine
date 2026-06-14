import torch
import torch.nn.functional as F

def sample(logit, temperature=0.7, top_p = 0.9):
    torch.cuda.nvtx.range_push("Sample")
    if logit.dim() > 2:
        logit = logit.view(-1, logit.size(-1))

    if temperature > 0:
        probs = F.softmax(logit / temperature, dim=-1)
        top_p = float(top_p)
        top_p = max(0.0, min(1.0, top_p))

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        to_remove = torch.zeros_like(probs, dtype=torch.bool)
        to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

        probs = probs.masked_fill(to_remove, 0.0)
        probs_sum = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = probs / probs_sum

        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(logit, dim=-1, keepdim=True)

    torch.cuda.nvtx.range_pop()
    return next_token