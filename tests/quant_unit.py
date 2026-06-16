import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.quant_linear import W8A16Linear
from src.quant_utils import quantize_weight_symmetric_per_channel


def _test_matches_reference(
    M: int,
    K: int,
    N: int,
    dtype: torch.dtype,
    weight_distribution: str = "normal",
    x_requires_grad: bool = False,
) -> float:
    """Compare Triton W8A16Linear against a Triton-equivalent PyTorch reference."""
    device = "cuda"
    torch.manual_seed(42)

    if weight_distribution == "normal":
        w = torch.randn(N, K, dtype=dtype, device=device)
    elif weight_distribution == "uniform":
        w = (torch.rand(N, K, dtype=torch.float32, device=device) - 0.5) * 2.0
        w = w.to(dtype)
    elif weight_distribution == "extreme":
        w = torch.randn(N, K, dtype=dtype, device=device) * 1e-2
        idx = torch.randint(0, N * K, (min(10, N * K),), device=device)
        w.view(-1)[idx] = torch.tensor([10.0, -10.0, 5.0, -5.0, 8.0, -8.0, 3.0, -3.0, 12.0, -12.0], dtype=dtype, device=device)[: idx.numel()]
    else:
        raise ValueError(weight_distribution)

    qweight, scales = quantize_weight_symmetric_per_channel(w)
    layer = W8A16Linear(K, N).to(device)
    layer.weight.data = qweight
    layer.scales.data = scales.to(dtype)

    x = torch.randn(M, K, dtype=dtype, device=device, requires_grad=x_requires_grad)
    # Test non-contiguous input: transpose a buffer so inner dim stride is not 1.
    x_large = torch.randn(K, M + 1, dtype=dtype, device=device)
    x_nc = x_large[:, :-1].t()
    assert not x_nc.is_contiguous()

    out = layer(x)
    out_nc = layer(x_nc)
    ref = _triton_like_reference(x, qweight, scales.to(dtype))
    ref_nc = _triton_like_reference(x_nc, qweight, scales.to(dtype))

    max_err = (out - ref).abs().max().item()
    max_err_nc = (out_nc - ref_nc).abs().max().item()
    return max(max_err, max_err_nc)


# Tolerance for comparisons against the fp32 reference. Outputs are bf16, so
# differences up to ~1.0 ulp at the maximum output magnitude are acceptable.
TEST_TOL = 1.5


def _triton_like_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """PyTorch fp32 reference: dequantize weight in fp32, matmul in fp32, scale in fp32.

    This matches the internal arithmetic of the Triton kernels (fp32 accumulator,
    per-channel scale applied in fp32) and gives a tighter correctness bound than
    a bf16 reference, whose own rounding errors can be larger than the Triton
    kernel's error.
    """
    w_deq = weight.float() * scales.float()
    return (x.float() @ w_deq.t()).to(x.dtype)


def test_basic_shapes():
    dtype = torch.bfloat16
    cases = [
        (1, 256, 256),
        (1, 2048, 2048),
        (1, 2048, 8192),
        (1, 8192, 2048),
        (8, 2048, 2048),
        (33, 2048, 2048),
        (65, 512, 1024),
    ]
    print("Testing basic shapes (bf16):")
    for M, K, N in cases:
        err = _test_matches_reference(M, K, N, dtype)
        status = "PASS" if err < TEST_TOL else "FAIL"
        print(f"  M={M:3d} K={K:4d} N={N:4d} max_err={err:.6f} {status}")
        assert err < TEST_TOL, f"Failed at M={M}, K={K}, N={N} with err={err}"


def test_3d_input():
    device = "cuda"
    dtype = torch.bfloat16
    torch.manual_seed(0)
    B, S, K, N = 2, 17, 512, 1024
    w = torch.randn(N, K, dtype=dtype, device=device)
    qweight, scales = quantize_weight_symmetric_per_channel(w)
    layer = W8A16Linear(K, N).to(device)
    layer.weight.data = qweight
    layer.scales.data = scales.to(dtype)

    x = torch.randn(B, S, K, dtype=dtype, device=device)
    out = layer(x)
    ref = _triton_like_reference(x.reshape(-1, K), qweight, scales.to(dtype)).reshape(B, S, N)
    err = (out - ref).abs().max().item()
    print(f"Testing 3D input B={B} S={S} K={K} N={N}: max_err={err:.6f} {'PASS' if err < TEST_TOL else 'FAIL'}")
    assert err < TEST_TOL


def test_weight_distributions():
    dtype = torch.bfloat16
    cases = ["normal", "uniform", "extreme"]
    print("Testing weight distributions:")
    for dist in cases:
        err = _test_matches_reference(1, 2048, 2048, dtype, weight_distribution=dist)
        status = "PASS" if err < TEST_TOL else "FAIL"
        print(f"  {dist:10s} max_err={err:.6f} {status}")
        assert err < TEST_TOL, f"Failed for distribution {dist} with err={err}"


def test_fp16():
    dtype = torch.float16
    err = _test_matches_reference(4, 1024, 1024, dtype)
    print(f"Testing fp16: max_err={err:.6f} {'PASS' if err < TEST_TOL else 'FAIL'}")
    assert err < TEST_TOL


def test_llama_like_layers():
    """Smoke test with Llama-3.2-1B-like shapes."""
    dtype = torch.bfloat16
    shapes = [
        (2048, 2048),  # q/k/v/o
        (8192, 2048),  # w1/w3
        (2048, 8192),  # w2
    ]
    print("Testing Llama-like layer shapes:")
    for K, N in shapes:
        err = _test_matches_reference(1, K, N, dtype)
        status = "PASS" if err < TEST_TOL else "FAIL"
        print(f"  K={K:4d} N={N:4d} max_err={err:.6f} {status}")
        assert err < TEST_TOL


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for W8A16 unit tests")

    test_basic_shapes()
    test_3d_input()
    test_weight_distributions()
    test_fp16()
    test_llama_like_layers()
    print("\nAll W8A16 unit tests passed.")


if __name__ == "__main__":
    main()
