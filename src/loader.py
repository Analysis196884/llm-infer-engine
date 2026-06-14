from safetensors.torch import load_file
import torch
import time


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
def load_weights(model, weights_path, device, dtype=torch.float16, assign=True):
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
        candidate_keys = [model_key, _to_hf_name(model_key)]
        found_key = None
        for key in candidate_keys:
            if key in state_dict:
                found_key = key
                break

        if found_key is None:
            if model_key == "output.weight" and "model.embed_tokens.weight" in state_dict:
                found_key = "model.embed_tokens.weight"
            else:
                missing_keys.append(model_key)
                continue

        tensor = state_dict[found_key]
        if tensor.shape != model_value.shape:
            raise ValueError(
                f"Shape mismatch for {model_key}: expected {tuple(model_value.shape)}, got {tuple(tensor.shape)}"
            )

        remapped_state[model_key] = tensor.to(dtype=dtype, device=device)

    stage2_time = time.time() - stage2_start
    print(f"    Transfer: {stage2_time:.4f}s ({len(remapped_state)} tensors)")
    torch.cuda.nvtx.range_pop()

    torch.cuda.nvtx.range_push("WeightStateDictLoad")
    load_result = model.load_state_dict(remapped_state, strict=False, assign=assign)

    if next(model.parameters(), None) is not None and next(model.parameters()).device.type == "meta":
        model = model.to_empty(device=device)
    else:
        model.to(device=device, dtype=dtype)
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


def build_model_from_weights(model_cls, model_args, weights_path, device, dtype=torch.float16):
    from src.model import precompute_freqs_cis

    torch.cuda.nvtx.range_push("MetaInit")
    with torch.device("meta"):
        model = model_cls(model_args)
    torch.cuda.nvtx.range_pop()

    report = load_weights(model, weights_path, device, dtype=dtype, assign=True)

    torch.cuda.nvtx.range_push("FreqsCisPrecompute")
    model.freqs_cis = precompute_freqs_cis(
        model_args.dim // model_args.n_heads,
        model_args.max_seq_len,
        theta=model_args.rope_theta,
        rope_scaling=model_args.rope_scaling,
    )
    torch.cuda.nvtx.range_pop()
    return model, report