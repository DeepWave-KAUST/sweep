import torch
from geophyai.scalars import staggered_grid_coes
from typing import List

@torch.jit.script
def laplace(u: torch.Tensor, 
            h: torch.Tensor, 
            kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    return torch.nn.functional.conv2d(u, kernel, padding=padding) / (h**2)

def forward_diff(x: torch.Tensor, 
                 axis: int, 
                 a: torch.Tensor, 
                 padding_value: int) -> torch.Tensor:
    """
    Compute the forward difference of an input tensor along a given dimension.
    """
    M = a.size(0)
    diff = torch.zeros_like(x)

    for i in range(1, M + 1):
        # Roll the tensor
        rolled_x = torch.roll(x, shifts=i, dims=axis)
        diff += a[i - 1] * (x - rolled_x)

        if axis == 2: # x
            diff[:, :, 0, :] = padding_value
        elif axis == 3: # x
            diff[..., 0] = padding_value

    return diff


def backward_diff(x: torch.Tensor, 
                  axis: int, 
                  a: torch.Tensor, 
                  padding_value: int) -> torch.Tensor:
    """
    Compute the backward difference of an input tensor along a given dimension.
    """
    M = a.size(0)
    diff = torch.zeros_like(x)

    for i in range(1, M + 1):
        # Roll the tensor
        rolled_x = torch.roll(x, shifts=-i, dims=axis)
        diff += a[i - 1] * (rolled_x - x)
        if axis == 2: # x
            diff[:, :, -1, :] = padding_value
        elif axis == 3: # z
            diff[..., -1] = padding_value

    return diff

def diff_using_roll(input: torch.Tensor, 
                    axis: int,
                    coes: torch.Tensor, 
                    forward: bool=True, 
                    padding_value:int =0) -> torch.Tensor:
    if forward:
        return forward_diff(input, axis, coes, padding_value)
    else:
        return backward_diff(input, axis, coes, padding_value)