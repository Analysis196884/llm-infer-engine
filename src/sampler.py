import torch
import torch.nn.functional as F
import time

_sample_times = {"sort": 0, "multinomial": 0, "argmax": 0, "calls": 0}

def sample(logits, temperature=0.7, top_p = 0.9):
    """Sample next token with optional temperature and top-p sampling"""
    global _sample_times
    _sample_times["calls"] += 1
    
    # only take the last token's logits
    logits = logits[:, -1, :]

    if temperature > 0:
        probs = F.softmax(logits / temperature, dim=-1)
        # top-p (Nucleus) sampling
        top_p = float(top_p)
        top_p = max(0.0, min(1.0, top_p))

        # Sort and find top-p threshold
        sort_start = time.time()
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        _sample_times["sort"] += time.time() - sort_start

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False

        to_remove = torch.zeros_like(probs, dtype=torch.bool)
        to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)

        probs = probs.masked_fill(to_remove, 0.0)
        probs_sum = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        probs = probs / probs_sum
        
        multi_start = time.time()
        next_token = torch.multinomial(probs, num_samples=1)
        _sample_times["multinomial"] += time.time() - multi_start

    else:
        # Greedy sampling
        argmax_start = time.time()
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        _sample_times["argmax"] += time.time() - argmax_start
    
    return next_token

def get_sample_stats():
    """Get sampling statistics"""
    global _sample_times
    if _sample_times["calls"] == 0:
        return {}
    return {
        "calls": _sample_times["calls"],
        "sort_time": _sample_times["sort"],
        "multinomial_time": _sample_times["multinomial"],
        "argmax_time": _sample_times["argmax"],
        "total_time": _sample_times["sort"] + _sample_times["multinomial"] + _sample_times["argmax"],
        "avg_per_call": (_sample_times["sort"] + _sample_times["multinomial"] + _sample_times["argmax"]) / _sample_times["calls"],
    }

def reset_sample_stats():
    """Reset sampling statistics"""
    global _sample_times
    _sample_times = {"sort": 0, "multinomial": 0, "argmax": 0, "calls": 0}