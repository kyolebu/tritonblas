import torch
import triton
import tritonblas as tb
from triton.testing import assert_close
from .utils import get_rtol


DEVICE = triton.runtime.driver.active.get_current_target().backend


def test_add():
    size = 98432

    torch.manual_seed(0)
    x = torch.randn(size, device=DEVICE, dtype=torch.float32)

    triton_output = tb.add(x)
    torch_output = x.sum()

    rtol = get_rtol()
    assert_close(triton_output, torch_output, rtol=rtol)
