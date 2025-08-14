import jax
import jax.numpy as jnp

import torch
import torch.nn.functional as F

import numpy as np
import math

from .scalars import normal_grid_coes

def symbol_S(theta, a0, a_m):
    """
    Calculate the symbol S(theta) = a0 + 2 * sum_{m=1}^M a_m * cos(m*theta)
    """
    s = a0
    for m_idx, a in enumerate(a_m, start=1):
        s += 2.0 * a * math.cos(m_idx * theta)
    return s

def max_dx_from_error(f_max, c_min, so, error_tol=0.01, dx_upper=None):
    """
    Calculate the maximum stable spatial step dx based on frequency, minimum and maximum velocities,
    and the.
    """
    a_m = normal_grid_coes(so//2)
    a0 = -2.0 * np.sum(a_m)
    lambda_min = c_min / f_max
    k = 2.0 * math.pi / lambda_min

    if dx_upper is None:
        dx_upper = lambda_min * 0.9

    low = 1e-9
    high = dx_upper
    for _ in range(60):
        mid = 0.5 * (low + high)
        theta = k * mid
        S = symbol_S(theta, a0, a_m)
        k_num_sq = - S / (mid * mid)
        if k_num_sq <= 0:
            high = mid
            continue
        k_num = math.sqrt(k_num_sq)
        c_ratio = k_num / k
        err = abs(1.0 - c_ratio)
        if err <= error_tol:
            low = mid
        else:
            high = mid
    return low

def max_dt_from_dx_dy(dx, dy, so, c_max, C_safe=0.9, theta_samples=300):
    """
    Calculate the maximum stable time step dt based on dx, dy, and the coefficients a0, a_m.
    """
    a_m = normal_grid_coes(so//2)
    a0 = -2.0 * np.sum(a_m)
    thetas = np.linspace(0.0, math.pi, theta_samples)
    lambda_max = 0.0
    for tx in thetas:
        Sx = symbol_S(tx, a0, a_m)
        kx_sq = max(0.0, - Sx / (dx * dx))
        for ty in thetas:
            Sy = symbol_S(ty, a0, a_m)
            ky_sq = max(0.0, - Sy / (dy * dy))
            lam = kx_sq + ky_sq
            if lam > lambda_max:
                lambda_max = lam

    if lambda_max <= 0:
        return float('inf'), lambda_max

    dt_limit = 2.0 / (c_max * math.sqrt(lambda_max))
    return dt_limit * C_safe


def to_tensor(array):
    if isinstance(array, np.ndarray):
        return torch.from_numpy(array)
    elif isinstance(array, torch.Tensor):
        return array
    
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
    
def edge_pad_base(u, pad_width):
    """Pad the edges of the input data.

    Args:
        u (jnp.array): The input data with shape (batch_size, 1, nz, nx).
        pad_width (int): The width of the padding.

    Returns:
        jnp.array: Padded data.
    """
    return jnp.pad(u, pad_width, mode='edge')

def edge_pad_fwd(u, pad_width):
    """Forward function of edge_pad.

    Args:
        u (jnp.array): The input data with shape (batch_size, 1, nz, nx).
        pad_width (int): The width of the padding.

    Returns:
        jnp.array: Padded data.
    """
    u = jnp.pad(u, pad_width, mode='edge')
    return u, pad_width

def edge_pad_bwd(pad_width, g):
    """Backward function of edge_pad.

    Args:
        pad_width (jnp.array): The padding width for each dimension.
        g (jnp.array): The gradient.

    Returns:
        jnp.array: The gradient of the input data.
    """    
    slices = [
        slice(p0, g.shape[i] - p1)
        for i, (p0, p1) in enumerate(pad_width)
    ]
    return g[tuple(slices)], None

edge_pad = jax.custom_vjp(edge_pad_base)
edge_pad.defvjp(edge_pad_fwd, edge_pad_bwd)

def split_model(m, sources, receivers, one_side_expand=50):
    """Split the model based on the sources and receivers locations for saving memory and computational resources.

    Args:
        m (2d array): The model to be split, shape (nz, nx).
        sources (2d array): The source locations, shape (nshots, 2).
        receivers (3d array): The receiver locations, shape (nshots, nreceivers, 2).
        one_side_expand (int): The number of grids to expand for one side.

    Returns:
        tuple: A tuple containing:
            - m_split (array): The split velocity model, shape (nshots, nz, nx).
            - left (array): The leftmost index of the split model for each shot.
            - right (array): The rightmost index of the split model for each shot.
    """
    ori_domain = m[0].shape
    assert sources.shape[0] == receivers.shape[0], "Sources and receivers must have the same number of shots"

    m_split = []
    left = []
    right = []

    sources_moved = sources.copy()
    receivers_moved = receivers.copy()

    # Calculate the max length between the sources and receivers for each shot
    # and determine the leftmost and rightmost points
    # max_length = np.abs(np.max(receivers[..., 0], axis=1) - sources[..., 0]).max()
    max_length = np.max(np.abs(receivers[..., 0] - sources[..., 0][..., None]), axis=1).max()
    for shot in range(sources.shape[0]):
        srcx = sources[shot, ..., 0]
        recx = receivers[shot, ..., 0]

        leftmost_recx = np.min(recx)
        rightmost_recx = np.max(recx)

        leftmost = np.min(np.array([leftmost_recx, srcx]), axis=0)
        rightmost = np.max(np.array([rightmost_recx, srcx]), axis=0)

        # Expand the leftmost and rightmost points by the specified number of grids
        expand_left = one_side_expand

        if leftmost - expand_left < 0:
            left_start = 0
            expand_left = leftmost
            expand_right = one_side_expand*2 - expand_left
        else:
            left_start = leftmost - expand_left
            expand_right = one_side_expand

        right_end = left_start + expand_left + max_length + expand_right

        if right_end > ori_domain[1]: # if the right end exceeds the domain size, try to adjust

            right_end = ori_domain[1]
            expand_right = right_end - rightmost
            expand_left = one_side_expand * 2 - expand_right
            left_start = leftmost - expand_left
            assert left_start >= 0, f'Left start {left_start} is out of bounds for the domain size {ori_domain[1]}'
            assert right_end <= ori_domain[1], f'Right end {right_end} is out of bounds for the domain size {ori_domain[1]}'

        m_split.append([_m[:, left_start:right_end] for _m in m])
        left.append(left_start)
        right.append(right_end)

        sources_moved = sources_moved.at[shot, ..., 0].subtract(left_start)
        receivers_moved = receivers_moved.at[shot, ..., 0].subtract(left_start)
    try:
        m_split = np.stack(m_split, axis=0)
    except ValueError as e:
        print(f"Error stacking m_split: {e}")
    left = np.array(left)
    right = np.array(right)

    return m_split, left, right, sources_moved, receivers_moved

def split_by_lr(m, left, right):
    """Split the model by left and right indices.

    Args:
        m (2d array): The models to be split, shape (nz, nx).
        left (array): The leftmost index of the split model for each shot.
        right (array): The rightmost index of the split model for each shot.

    Returns:
        list: A list of split models.
    """
    assert len(left) == len(right), "Left and right indices must have the same length"
    return jnp.stack([m[:, l:r] for l, r in zip(left, right)])