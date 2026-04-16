import jax
import jax.numpy as jnp
from jax import vmap, lax
from jax.scipy.signal import convolve2d as conv2d


def _to_nchw(u):
    if u.ndim == 2:
        return u[None, None, ...], lambda x: x[0, 0]
    if u.ndim == 3:
        return u[:, None, ...], lambda x: x[:, 0]
    if u.ndim == 4:
        return u, lambda x: x
    raise ValueError(f"Expected 2D/3D/4D input for gradient kernel, got shape {u.shape}")


def _zero_halo(out, halo):
    if halo <= 0:
        return out
    out = out.at[..., :halo, :].set(0)
    out = out.at[..., -halo:, :].set(0)
    out = out.at[..., :, :halo].set(0)
    out = out.at[..., :, -halo:].set(0)
    return out

@jax.jit
def laplace1d_sep(u, k1d, hz=1.0, hx=1.0):
    """
    Anisotropic spacing: Laplace = d2/dz2 / hz^2 + d2/dx2 / hx^2
    """
    # k1d = jnp.asarray(k1d, dtype=u.dtype)
    kz = k1d[:, None, None, None]  # (k,1,1,1)
    kx = k1d[None, :, None, None]  # (1,k,1,1)

    d2z = lax.conv_general_dilated(
        u, kz,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "HWIO", "NCHW"),
    )

    d2x = lax.conv_general_dilated(
        u, kx,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NCHW", "HWIO", "NCHW"),
    )

    return d2z / (hz * hz), d2x / (hx * hx)


def laplace3d_sep(u, k1d, hz, hy, hx):
    """ 3D Laplace operator using JAX.
     Args:
         u (jnp.ndarray): Input wavefield of shape (batch, 1, depth, height, width).
         h (float): Grid spacing.
         kernel (jnp.ndarray): 3D convolution kernel of shape (1, 1, kD, kH, kW).

     Returns:
         jnp.ndarray: Resulting wavefield after applying the Laplace operator.
     """
    kz = k1d[None, None, :, None, None]  # (1,1,k,1,1)
    ky = k1d[None, None, None, :, None]  # (1,1,1,k,1)
    kx = k1d[None, None, None, None, :]  # (1,1,1,1,k)
    dn = jax.lax.conv_dimension_numbers(u.shape, kz.shape,
                                        ('NCDHW', 'OIDHW', 'NCDHW'))
    kwargs = {'window_strides': (1,1,1),
              'padding': 'SAME',
              'lhs_dilation': (1,1,1),
              'rhs_dilation': (1,1,1),
              'dimension_numbers': dn}
    d2z = lax.conv_general_dilated(u, kz, **kwargs)/ (hz ** 2)
    d2y = lax.conv_general_dilated(u, ky, **kwargs)/ (hy ** 2)
    d2x = lax.conv_general_dilated(u, kx, **kwargs)/ (hx ** 2)
    return d2z, d2y, d2x


def _laplace(image, kernel):
    # Expected input shape: (height, width)
    return conv2d(image, kernel, mode='same')

batch_convolve2d = vmap(vmap(_laplace, in_axes=(0, None)), in_axes=(0, None))

@jax.jit
def laplace2d(u, h=1.0, kernel=None):
    return batch_convolve2d(u, kernel) / (h ** 2)

def laplace3d(u, kernel=None, h=1.0):
    """ 3D Laplace operator using JAX.
     Args:
         u (jnp.ndarray): Input wavefield of shape (batch, 1, depth, height, width).
         h (float): Grid spacing.
         kernel (jnp.ndarray): 3D convolution kernel of shape (1, 1, kD, kH, kW).

     Returns:
         jnp.ndarray: Resulting wavefield after applying the Laplace operator.
     """
    dn = jax.lax.conv_dimension_numbers(u.shape, kernel.shape,
                                        ('NCDHW', 'OIDHW', 'NCDHW'))
    out = jax.lax.conv_general_dilated(u,    # lhs = image tensor
                                       kernel,  # rhs = conv kernel tensor
                                       (1,1,1), # window strides
                                       'SAME',  # padding mode
                                       (1,1,1), # lhs/image dilation
                                       (1,1,1), # rhs/kernel dilation
                                       dn)      # dimension_numbers
    return out / (h ** 2)

def apply_kernels_jax(u, kernels):
    # u: (b, 1, h, w)
    # kernels: (k, kh, kw)
    B, C, H, W = u.shape
    K, KH, KW = kernels.shape
    kernels_exp = kernels[:, None, ::-1, ::-1]  # (K, 1, kh, kw), need reverse for lax conv
    def single_conv():
        return jax.lax.conv_general_dilated(
            lhs=u,  # (b, k, h, w)
            rhs=kernels_exp, # (1, k, kh, kw)
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW'), 
        )  # → (b, k, 1, h, w)
    conv_out = single_conv()  # → (b, k, h, w)
    return jnp.sum(conv_out, axis=1, keepdims=True)  # → (b, 1, h, w)

def gradient(u, h, axis, kernels=None):
    if kernels is not None:
        if axis not in kernels:
            raise ValueError(f"No gradient kernel configured for axis={axis}.")
        u_nchw, restore = _to_nchw(u)
        rhs = jnp.transpose(kernels[axis] / h, (2, 3, 1, 0))[::-1, ::-1, :, :]
        out = lax.conv_general_dilated(
            u_nchw,
            rhs,
            window_strides=(1, 1),
            padding='SAME',
            dimension_numbers=('NCHW', 'HWIO', 'NCHW'),
        )
        out = _zero_halo(out, max(kernels[axis].shape[-2] // 2, kernels[axis].shape[-1] // 2))
        return restore(out)
    return jnp.gradient(u, h, axis=axis)
