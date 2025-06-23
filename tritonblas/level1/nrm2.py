import torch
import triton
import triton.language as tl
import numpy as np


def get_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE": 1024}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 32}),
    ]


@triton.autotune(
    configs=get_autotune_config(),
    key=["n_elements"]
)
@triton.jit
def nrm2_partial(
    x_ptr,
    partial_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)

    exp = x * x
    sum_of_squares = tl.sum(exp)
    
    tl.store(partial_ptr + pid, sum_of_squares)


@triton.jit
def nrm2_final(
    partial_ptr,
    final_ptr,
    n_partial,
):
    total = tl.zeros((), dtype=tl.float32)
    
    # Process partial sums one by one
    for i in range(n_partial):
        partial_val = tl.load(partial_ptr + i)
        total += partial_val
    
    tl.store(final_ptr, total)


def nrm2(x: torch.Tensor):
    n_elements = x.numel()
    output = torch.zeros((1,))

    # allocate enough memory for the partial sums
    min_block_size = min(config.kwargs['BLOCK_SIZE'] for config in get_autotune_config())
    max_partial_programs = triton.cdiv(n_elements, min_block_size)
    partial_sums = torch.zeros(max_partial_programs)

    # Capture the actual grid size used by autotuning
    auto_grid = None
    
    def grid(META):
        nonlocal auto_grid
        auto_grid = triton.cdiv(n_elements, META['BLOCK_SIZE'])
        return (auto_grid,)
    
    nrm2_partial[grid](
        x,
        partial_sums,
        n_elements,
    )

    # Process exactly the number of partial sums that were written
    nrm2_final[(1,)](
        partial_sums,
        output,
        auto_grid
    )

    result = torch.sqrt(output)
    return result
