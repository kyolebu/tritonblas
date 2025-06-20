import torch
import triton
import triton.language as tl
import numpy as np


def get_autotune_config_partial():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_PARTIAL": 1024,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_PARTIAL": 512,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_PARTIAL": 256,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_PARTIAL": 128,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_PARTIAL": 64,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_PARTIAL": 32,
            },
        ),
    ]

def get_autotune_config_final():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_FINAL": 256,
            },
        ),
        triton.Config(
            {
                "BLOCK_SIZE_FINAL": 512,
            },
        ),
    ]


@triton.autotune(
    configs=get_autotune_config_partial(),
    key=["n_elements"]
)

@triton.jit
def nrm2_partial(
    x_ptr,
    partial_ptr,
    n_elements,
    BLOCK_SIZE_PARTIAL: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE_PARTIAL
    offsets = block_start + tl.arange(0, BLOCK_SIZE_PARTIAL)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    #tl.device_print("load x", x)

    exp = x * x
    #tl.device_print("exp", exp)

    sum_of_squares = tl.sum(exp)  # sum of squares for a program's assigned chunk
    #tl.device_print("sum_of_squares", sum_of_squares)
    
    tl.store(partial_ptr + pid, sum_of_squares)


@triton.autotune(
    configs=get_autotune_config_final(),
    key=["n_partial"]
)
@triton.jit
def nrm2_final(
    partial_ptr,
    final_ptr,
    n_partial,
    BLOCK_SIZE_FINAL: tl.constexpr
):
    thread_id = tl.program_id(axis=0) * BLOCK_SIZE_FINAL + tl.arange(0, BLOCK_SIZE_FINAL)
    mask = thread_id < n_partial
    chunk = tl.load(partial_ptr + thread_id, mask=mask)

    local_sum = tl.sum(chunk, axis=0)

    tl.atomic_add(final_ptr, local_sum)


def nrm2(x: torch.Tensor):
    n_elements = x.numel()

    output = torch.zeros((1,), device=x.device, dtype=torch.float32)

    buffer = min(config.kwargs['BLOCK_SIZE_PARTIAL'] for config in get_autotune_config_partial())
    partial_programs = triton.cdiv(n_elements, buffer)

    partial_sums = torch.empty(partial_programs)

    def grid_partial(META):
        return (triton.cdiv(n_elements, META['BLOCK_SIZE_PARTIAL']),)
    nrm2_partial[grid_partial](
        x,
        partial_sums,
        n_elements,
    )

    def grid_final(META):
        return (triton.cdiv(partial_programs, META['BLOCK_SIZE_FINAL']),)
    nrm2_final[grid_final](
        partial_sums,
        output,
        partial_programs,
    )

    output_val = output[0]
    output = torch.sqrt(output_val)
    return output
