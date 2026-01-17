import numpy as np

def to_backend(arr, backend, device=None):
    if backend == 'torch':
        import torch
        if isinstance(arr, list):
            return [torch.from_numpy(a).float().to(device) for a in arr]
        return torch.from_numpy(arr).float().to(device)
    elif backend == 'jax':
        import jax.numpy as jnp
        if isinstance(arr, list):
            return [jnp.array(a, dtype=jnp.float32) for a in arr]
        return jnp.array(arr, dtype=jnp.float32)
    elif backend == 'cuda':
        return arr.astype(np.float32)
    