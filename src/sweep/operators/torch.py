import torch
import torch.nn.functional as F

def laplace1d_sep(u, k1d, hz=1.0, hx=1.0):
    kz = k1d[None, None, :, None]  # (k,1,1,1)
    kx = k1d[None, None, None, :]  # (1,k,1,1)
    pad = k1d.shape[-1] // 2
    lapx = F.conv2d(u, kx, padding=(0, pad)) / (hx*hx)
    lapz = F.conv2d(u, kz, padding=(pad, 0)) / (hz*hz)
    return lapz, lapx

def apply_kernels_torch(u, kernels):
    # u: (B, 1, H, W), torch.Tensor
    # kernels: (K, kh, kw), torch.Tensor

    B, C, H, W = u.shape
    K, KH, KW = kernels.shape

    kernels_exp = kernels.unsqueeze(1)  # (K, 1, kh, kw)

    padding = (KH // 2, KW // 2) 

    conv_out = F.conv2d(u, kernels_exp, padding=padding)  # (B, K, H, W)

    out = conv_out.sum(dim=1, keepdim=True)  # (B, 1, H, W)

    return out

def laplace2d(u: torch.Tensor, 
            h: float | torch.Tensor, 
            kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    return torch.nn.functional.conv2d(u, kernel, padding=padding) / (h*h)

def gradient(u, h, axis):
    return torch.gradient(u, spacing=h, dim=axis)[0]
