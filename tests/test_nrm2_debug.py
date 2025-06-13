import torch
import pytest
import tritonblas as tb

def test_nrm2_small_deterministic():
    x = torch.randn(1024, dtype=torch.float32)

    expected_norm = torch.linalg.norm(x) # Use PyTorch's reference on CPU
    
    print(f"\nExpected norm: {expected_norm.item()}")

    triton_norm = tb.nrm2(x)
    print(f"Triton norm: {triton_norm.item()}")

    assert torch.allclose(triton_norm, expected_norm, atol=1e-5, rtol=1e-5), \
        f"Triton norm {triton_norm.item()} does not match expected {expected_norm.item()}"

    print("Triton norm matches expected value for small tensor.")
