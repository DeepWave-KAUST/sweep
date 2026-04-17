import torch
import torch.nn.functional as F


def _to_nchw(u):
    if u.ndim == 2:
        return u.unsqueeze(0).unsqueeze(0), lambda x: x.squeeze(0).squeeze(0)
    if u.ndim == 3:
        return u.unsqueeze(1), lambda x: x.squeeze(1)
    if u.ndim == 4:
        return u, lambda x: x
    raise ValueError(f"Expected 2D/3D/4D input for gradient kernel, got shape {tuple(u.shape)}")


def _zero_halo(out, halo):
    if halo <= 0:
        return out
    out[..., :halo, :] = 0
    out[..., -halo:, :] = 0
    out[..., :, :halo] = 0
    out[..., :, -halo:] = 0
    return out

def laplace1d_sep(u, k1d, hz=1.0, hx=1.0):
    if isinstance(k1d, tuple):
        kz, kx = k1d
        pad = max(kz.shape[-3], kx.shape[-1]) // 2
    else:
        kz = k1d[None, None, :, None]  # (1,1,k,1)
        kx = k1d[None, None, None, :]  # (1,1,1,k)
        pad = k1d.shape[-1] // 2
    lapx = F.conv2d(u, kx, padding=(0, pad)) / (hx*hx)
    lapz = F.conv2d(u, kz, padding=(pad, 0)) / (hz*hz)
    return lapz, lapx

def laplace3d_sep(u, k1d, hz=1.0, hy=1.0, hx=1.0):
    """
    u: (B, 1, nz, ny, nx)
    k1d: (k,)
    """
    if isinstance(k1d, tuple):
        kz, ky, kx = k1d
        pad = max(kz.shape[-3], ky.shape[-2], kx.shape[-1]) // 2
    else:
        pad = k1d.shape[-1] // 2
        kz = k1d.view(1, 1, -1, 1, 1)
        ky = k1d.view(1, 1, 1, -1, 1)
        kx = k1d.view(1, 1, 1, 1, -1)

    lapz = F.conv3d(u, kz, padding=(pad, 0, 0)) / (hz * hz)
    lapy = F.conv3d(u, ky, padding=(0, pad, 0)) / (hy * hy)
    lapx = F.conv3d(u, kx, padding=(0, 0, pad)) / (hx * hx)

    return lapz, lapy, lapx

@torch.jit.script
def apply_kernels_torch(u, kernels):
    # u: (B, 1, H, W), torch.Tensor
    # kernels: (1, 1, kh, kw) or (K, 1, kh, kw), torch.Tensor
    _, _, KH, KW = kernels.shape

    padding = (KH // 2, KW // 2) 

    conv_out = F.conv2d(u, kernels, padding=padding)  # (B, K, H, W)
    return conv_out if conv_out.shape[1] == 1 else conv_out.sum(dim=1, keepdim=True)

@torch.jit.script
def apply_kernels_torch3d(u, kernels):
    # u: (B, 1, D, H, W), torch.Tensor
    # kernels: (1, 1, kD, kH, kW) or (K, 1, kD, kH, kW), torch.Tensor
    _, _, KD, KH, KW = kernels.shape

    padding = (KD // 2, KH // 2, KW // 2) 

    conv_out = F.conv3d(u, kernels, padding=padding)  # (B, K, D, H, W)
    return conv_out if conv_out.shape[1] == 1 else conv_out.sum(dim=1, keepdim=True)

def laplace2d(u: torch.Tensor, 
            h: float | torch.Tensor, 
            kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    return torch.nn.functional.conv2d(u, kernel, padding=padding) / (h*h)

def gradient(u, h, axis, kernels=None):
    if kernels is not None:
        if axis not in kernels:
            raise ValueError(f"No gradient kernel configured for axis={axis}.")
        kernel = kernels[axis]
        padding = (kernel.shape[-2] // 2, kernel.shape[-1] // 2)
        u_nchw, restore = _to_nchw(u)
        out = F.conv2d(u_nchw, kernel / h, padding=padding)
        out = _zero_halo(out, max(padding))
        return restore(out)
    return torch.gradient(u, spacing=h, dim=axis)[0]
