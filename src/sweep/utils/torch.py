
import torch
import torch.nn.functional as F
class EdgePadding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, pad):
        ctx.pad = pad
        return F.pad(input_tensor.unsqueeze(0), pad=pad, mode='replicate').squeeze(0)

    @staticmethod
    def backward(ctx, grad_output):
        pad_left, pad_right, pad_top, pad_bottom = ctx.pad
        grad_input = grad_output[..., pad_top:-pad_bottom, pad_left:-pad_right]
        return grad_input, None