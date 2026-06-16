from typing import Tuple

import torch
import torch.nn as nn


def quantize_weight_symmetric_per_channel(
    weight: torch.Tensor,
    quant_dtype: torch.dtype = torch.int8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Symmetric per-channel weight quantization.

    Args:
        weight: Float weight tensor of shape [out_features, in_features].
        quant_dtype: Target integer dtype. Defaults to int8.

    Returns:
        qweight: Quantized integer weight of the same shape as ``weight``.
        scales: Per-channel scale of shape [out_features, 1] in the original dtype.
    """
    orig_dtype = weight.dtype
    if orig_dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise ValueError(f"Unsupported weight dtype for quantization: {orig_dtype}")

    if quant_dtype != torch.int8:
        raise NotImplementedError("Only int8 symmetric per-channel quantization is supported.")

    max_int = 127
    # Work in float32 for stable scale computation.
    weight_f32 = weight.float()
    abs_max = weight_f32.abs().amax(dim=1, keepdim=True)
    abs_max = abs_max.clamp(min=1e-12)

    scales = (abs_max / max_int).to(orig_dtype)
    qweight = torch.clamp(torch.round(weight_f32 / scales), -max_int - 1, max_int).to(quant_dtype)
    return qweight, scales


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute mean cosine similarity over the last dimension."""
    a_flat = a.reshape(-1, a.shape[-1]).float()
    b_flat = b.reshape(-1, b.shape[-1]).float()
    a_norm = a_flat.norm(dim=1, keepdim=True)
    b_norm = b_flat.norm(dim=1, keepdim=True)
    denom = (a_norm * b_norm).clamp_min(eps)
    cos = (a_flat * b_flat).sum(dim=1, keepdim=True) / denom
    return cos.mean()


def logits_max_abs_error(a: torch.Tensor, b: torch.Tensor) -> float:
    """Maximum absolute error between two tensors."""
    return float((a.float() - b.float()).abs().max())


def logits_relative_error(
    a: torch.Tensor,
    b: torch.Tensor,
    eps: float = 1e-12,
) -> float:
    """Mean relative error over elements."""
    diff = (a.float() - b.float()).abs()
    denom = a.float().abs().clamp_min(eps)
    return float((diff / denom).mean())


def top1_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    """Fraction of positions where argmax(a) == argmax(b)."""
    a_top = a.argmax(dim=-1)
    b_top = b.argmax(dim=-1)
    return float((a_top == b_top).float().mean())


def perplexity(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute perplexity from logits and target token ids.

    Args:
        logits: [batch, seq_len, vocab_size] or [seq_len, vocab_size].
        labels: [batch, seq_len] or [seq_len] with token ids.

    Returns:
        Perplexity as a Python float.
    """
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
        labels = labels.unsqueeze(0)
    logits = logits.float()
    # Shift for next-token prediction: predict labels[:, 1:] from logits[:, :-1].
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = nn.functional.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="mean",
        ignore_index=-100,
    )
    return float(torch.exp(loss))


def get_transformer_linear_names() -> Tuple[str, ...]:
    """Names of transformer-block Linear layers subject to W8A16 quantization."""
    return (
        "attention.wq",
        "attention.wk",
        "attention.wv",
        "attention.wo",
        "feed_forward.w1",
        "feed_forward.w2",
        "feed_forward.w3",
    )


def should_quantize_key(model_key: str) -> bool:
    """Return True if a state-dict key belongs to a quantizable transformer Linear."""
    if not model_key.endswith(".weight"):
        return False
    # model_key looks like "layers.{idx}.attention.wq.weight"
    parts = model_key.split(".")
    if len(parts) < 5 or parts[0] != "layers":
        return False
    layer_module = ".".join(parts[2:-1])  # e.g. "attention.wq"
    return layer_module in get_transformer_linear_names()
