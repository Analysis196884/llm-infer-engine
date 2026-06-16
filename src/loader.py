from safetensors.torch import load_file
import time
import torch

from .quant_utils import quantize_weight_symmetric_per_channel, should_quantize_key


def _to_hf_name(model_key: str) -> str:
    if model_key == "tok_embeddings.weight":
        return "model.embed_tokens.weight"
    if model_key == "norm.weight":
        return "model.norm.weight"
    if model_key == "output.weight":
        return "lm_head.weight"

    if model_key.startswith("layers."):
        parts = model_key.split(".")
        layer_idx = parts[1]
        suffix = ".".join(parts[2:])

        attn_map = {
            "attention.wq.weight": "self_attn.q_proj.weight",
            "attention.wk.weight": "self_attn.k_proj.weight",
            "attention.wv.weight": "self_attn.v_proj.weight",
            "attention.wo.weight": "self_attn.o_proj.weight",
        }
        ffn_map = {
            "feed_forward.w1.weight": "mlp.gate_proj.weight",
            "feed_forward.w2.weight": "mlp.down_proj.weight",
            "feed_forward.w3.weight": "mlp.up_proj.weight",
        }
        norm_map = {
            "attention_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
        }

        if suffix in attn_map:
            return f"model.layers.{layer_idx}.{attn_map[suffix]}"
        if suffix in ffn_map:
            return f"model.layers.{layer_idx}.{ffn_map[suffix]}"
        if suffix in norm_map:
            return f"model.layers.{layer_idx}.{norm_map[suffix]}"

    return model_key


def _find_source_key(model_key: str, state_dict: dict) -> str | None:
    """Find the source safetensors key that matches a model state-dict key."""
    for candidate in (model_key, _to_hf_name(model_key)):
        if candidate in state_dict:
            return candidate

    # Output head is often tied to input embeddings in Llama checkpoints.
    if model_key == "output.weight" and "model.embed_tokens.weight" in state_dict:
        return "model.embed_tokens.weight"

    return None


def _prepare_weight(
    model_key: str,
    source_tensor: torch.Tensor,
    target_shape: tuple,
    device: torch.device,
    dtype: torch.dtype,
    quantization: str | None,
) -> dict[str, torch.Tensor]:
    """Prepare a model weight from the source checkpoint tensor.

    For W8A16 quantization this returns both the int8 quantized weight and the
    per-channel scale. Otherwise it simply casts/moves the tensor.
    """
    if source_tensor.shape != target_shape:
        raise ValueError(
            f"Shape mismatch for {model_key}: expected {tuple(target_shape)}, got {tuple(source_tensor.shape)}"
        )

    source_tensor = source_tensor.to(device=device)

    if quantization == "w8a16" and should_quantize_key(model_key): # quantized weights
        qweight, scales = quantize_weight_symmetric_per_channel(source_tensor)
        return {
            model_key: qweight,
            model_key.replace(".weight", ".scales"): scales.to(dtype=dtype),
        }

    return {model_key: source_tensor.to(dtype=dtype)} # normal weights


def load_weights(model, weights_path, device, dtype=torch.bfloat16, assign=True, quantization=None):
    """Load safetensors weights into a model.

    The model is expected to be already instantiated (typically on CPU). We
    avoid meta-tensor initialization here to keep the loading path simple and
    uniform across bf16 and W8A16 models.
    """
    torch.cuda.nvtx.range_push("WeightDiskLoad")
    stage1_start = time.time()
    state_dict = load_file(weights_path, device="cpu")
    stage1_time = time.time() - stage1_start
    print(f"    Load: {stage1_time:.4f}s ({len(state_dict)} tensors)")
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("WeightRemapTransfer")
    stage2_start = time.time()
    model_dict = model.state_dict()
    remapped_state = {}
    missing_keys = []

    for model_key, model_value in model_dict.items():
        source_key = _find_source_key(model_key, state_dict)
        if source_key is None:
            missing_keys.append(model_key)
            continue

        remapped_state.update(
            _prepare_weight(
                model_key,
                state_dict[source_key],
                model_value.shape,
                device,
                dtype,
                quantization,
            )
        )

    stage2_time = time.time() - stage2_start
    print(f"    Transfer: {stage2_time:.4f}s ({len(remapped_state)} tensors)")
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("WeightStateDictLoad")
    load_result = model.load_state_dict(remapped_state, strict=False, assign=assign)

    # Guard against any parameters that are still meta (should not happen when
    # the model was instantiated normally).
    if any(p is not None and p.device.type == "meta" for p in model.parameters()):
        model = model.to_empty(device=device)

    model = model.to(device)
    model.eval()
    torch.cuda.nvtx.range_pop()

    unresolved_missing = set(load_result.missing_keys) - set(missing_keys)
    if unresolved_missing:
        raise KeyError(f"Failed to load required model keys: {sorted(unresolved_missing)}")

    return {
        "loaded": len(remapped_state),
        "missing": sorted(missing_keys),
        "unexpected": sorted(load_result.unexpected_keys),
    }


def build_model_from_weights(model_cls, model_args, weights_path, device, dtype=torch.bfloat16, quantization=None):
    from src.model import precompute_freqs_cis

    # Pick the concrete model class based on the requested quantization scheme.
    if quantization == "w8a16":
        from src.model_quantized import Llama3Quantized
        model_cls = Llama3Quantized

    torch.cuda.nvtx.range_push("ModelInit")
    with torch.device("meta"):
        model = model_cls(model_args)
    torch.cuda.nvtx.range_pop()

    report = load_weights(model, weights_path, device, dtype=dtype, assign=True, quantization=quantization)

    torch.cuda.nvtx.range_push("FreqsCisPrecompute")
    model.freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        model_args.max_seq_len,
        theta=model_args.rope_theta,
        rope_scaling=model_args.rope_scaling,
    )
    torch.cuda.nvtx.range_pop()
    return model, report
