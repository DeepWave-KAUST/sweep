import torch
import torch.nn.functional as F

def laplace1d_sep(u, k1d, hz=1.0, hx=1.0):
    kz = k1d[None, None, :, None]  # (k,1,1,1)
    kx = k1d[None, None, None, :]  # (1,k,1,1)
    pad = k1d.shape[-1] // 2
    lapx = F.conv2d(u, kx, padding=(0, pad)) / (hx*hx)
    lapz = F.conv2d(u, kz, padding=(pad, 0)) / (hz*hz)
    return lapz, lapx

def laplace3d_sep(u, k1d, hz=1.0, hy=1.0, hx=1.0):
    """
    u: (B, 1, nz, ny, nx)
    k1d: (k,)
    """

    pad = k1d.shape[-1] // 2

    # reshape kernels to 3D
    kz = k1d.view(1, 1, -1, 1, 1)  # (1,1,k,1,1)
    ky = k1d.view(1, 1, 1, -1, 1)  # (1,1,1,k,1)
    kx = k1d.view(1, 1, 1, 1, -1)  # (1,1,1,1,k)

    lapz = F.conv3d(u, kz, padding=(pad, 0, 0)) / (hz * hz)
    lapy = F.conv3d(u, ky, padding=(0, pad, 0)) / (hy * hy)
    lapx = F.conv3d(u, kx, padding=(0, 0, pad)) / (hx * hx)

    return lapz, lapy, lapx

@torch.jit.script
def apply_kernels_torch(u, kernels):
    # u: (B, 1, H, W), torch.Tensor
    # kernels: (K, kh, kw), torch.Tensor

    B, C, H, W = u.shape
    K, KH, KW = kernels.shape

    kernels_exp = kernels.flip(-1, -2).unsqueeze(1)  # (K, 1, kh, kw)

    padding = (KH // 2, KW // 2) 

    conv_out = F.conv2d(u, kernels_exp, padding=padding)  # (B, K, H, W)

    out = conv_out.sum(dim=1, keepdim=True)  # (B, 1, H, W)

    return out

@torch.jit.script
def apply_kernels_torch3d(u, kernels):
    # u: (B, 1, D, H, W), torch.Tensor
    # kernels: (K, kD, kH, kW), torch.Tensor
    B, C, D, H, W = u.shape
    K, KD, KH, KW = kernels.shape

    kernels_exp = kernels.flip(-1, -2, -3).unsqueeze(1)  # (K, 1, kD, kH, kW)

    padding = (KD // 2, KH // 2, KW // 2) 

    conv_out = F.conv3d(u, kernels_exp, padding=padding)  # (B, K, D, H, W)

    out = conv_out.sum(dim=1, keepdim=True)  # (B, 1, D, H, W)

    return out

def laplace2d(u: torch.Tensor, 
            h: float | torch.Tensor, 
            kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    return torch.nn.functional.conv2d(u, kernel, padding=padding) / (h*h)

def gradient(u, h, axis):
    return torch.gradient(u, spacing=h, dim=axis)[0]
