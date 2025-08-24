import torch
import jax.numpy as jnp
import numpy as np

def to_backend(arr, backend, device=None):
    if backend == 'torch':
        return torch.from_numpy(arr).float().to(device)
    elif backend == 'jax':
        return jnp.array(arr, dtype=jnp.float32).squeeze()
    elif backend == 'cuda':
        return arr.astype(np.float32)
    