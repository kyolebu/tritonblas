import torch
import triton
import triton.language as tl
import numpy as np


# def get_autotune_config():
#     return [
#         triton.Config(
#             {
#                 "BLOCK_SIZE": 1024,
#             },
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE": 512,
#             },
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE": 256,
#             },
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE": 128,
#             },
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE": 64,
#             },
#         ),
#         triton.Config(
#             {
#                 "BLOCK_SIZE": 32,
#             },
#         ),
#     ]


# @triton.autotune(
#     configs=get_autotune_config(),
#     key=["n_elements"]
# )


@triton.jit
def nrm2_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    #tl.device_print("load x", x)

    exp = x * x
    tl.device_print("exp", exp)

    sum_of_squares = tl.sum(exp)  # sum of squares for a program's assigned chunk
    tl.device_print("sum_of_squares", sum_of_squares)
    
    tl.atomic_add(output_ptr, sum_of_squares)


def nrm2(x: torch.Tensor):
    output = torch.empty(())  # output is a single scalar value
    assert x.device == output.device
    n_elements = x.numel()

    def grid(META): return (triton.cdiv(n_elements, META['BLOCK_SIZE']), )

    nrm2_kernel[grid](
        x,
        output,
        n_elements,
        BLOCK_SIZE=128
    )
    result = torch.sqrt(output)
    print(result)
    return result