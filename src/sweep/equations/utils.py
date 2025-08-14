import torch
import jax.numpy as jnp

def to_backend(arr, backend, device=None):
    if backend == 'torch':
        return torch.from_numpy(arr).float().to(device)
    else:
        return jnp.array(arr, dtype=jnp.float32).squeeze()