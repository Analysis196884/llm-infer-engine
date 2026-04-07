import torch
import torch.nn.functional as F

def sample(logits, temperature=0.7, top_p = 0.9):
    # only take the last token's logits
    logits = logits[:, -1, :]

    if temperature > 0:
        probs = F.softmax(logits / temperature, dim=-1)
        # top-p (Nucleus) sampling
        top_p = float(top_p)
        top_p = max(0.0, min(1.0, top_p))

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Remove tokens with cumulative probability above the threshold
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
        # Greedy sampling
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    
    return next_token