
import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

class EdgePadding(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input_tensor, pad):
        ctx.pad = pad
        ctx.input_ndim = input_tensor.dim()

        # replicate padding
        out = F.pad(input_tensor.unsqueeze(0), pad=pad, mode='replicate')
        return out.squeeze(0)

    @staticmethod
    def backward(ctx, grad_output):
        pad = ctx.pad
        ndim = ctx.input_ndim

        slices = [slice(None)] * grad_output.dim()

        pad_pairs = len(pad) // 2

        for i in range(pad_pairs):
            left = pad[2 * i]
            right = pad[2 * i + 1]

            dim = - (i + 1)

            start = left
            end = -right if right > 0 else None

            slices[dim] = slice(start, end)

        grad_input = grad_output[tuple(slices)]

        return grad_input, None
