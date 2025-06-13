import triton
import triton.language as tl
import torch

@triton.jit
def add_kernel(
    x_ptr,  
    a_output_ptr,
    n_elements,  # Size of the vector
    BLOCK_SIZE: tl.constexpr,  # Optional meta-parameters for the kernel
):

    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    a_val = tl.sum(x, axis=0)
    tl.atomic_add(a_output_ptr, a_val)

def add(x: torch.Tensor):
    a_output = torch.zeros(1, device=x.device)
    assert x.device == a_output.device
    n_elements = x.shape[0]
    def grid(META): return (triton.cdiv(n_elements, META['BLOCK_SIZE']), )
    add_kernel[grid](x, a_output, n_elements, BLOCK_SIZE=128)
    return a_output