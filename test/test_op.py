import sys
sys.path.append('../src')
import numpy as np
from sweep.operators.general import PartialDerivative
from sweep.equations.utils import to_backend

backend = 'jax'

if backend == 'jax':
    import jax
    import jax.numpy as jnp
if backend == 'torch':
    import torch
    torch.backends.cudnn.benchmark = True

op = PartialDerivative(spatial_order=4, backend=backend, device='cuda')
op.to_backend(to_backend)

shape = (1, 1, 3, 3)
np.random.seed(0)
u = np.random.rand(*shape).astype(np.float32)
print(u)
if backend == 'jax':
    u_forward_x = op.x_forward(jnp.array(u))
    u_backward_x = op.x_backward(jnp.array(u))
    u_forward_z = op.z_forward(jnp.array(u))
    u_backward_z = op.z_backward(jnp.array(u))
    np.save(f'./op_forward_x_{backend}.npy', u_forward_x)
    np.save(f'./op_backward_x_{backend}.npy', u_backward_x)
    np.save(f'./op_forward_z_{backend}.npy', u_forward_z)
    np.save(f'./op_backward_z_{backend}.npy', u_backward_z)
if backend == 'torch':
    u_forward_x = op.x_forward(torch.from_numpy(u).to('cuda'))
    u_backward_x = op.x_backward(torch.from_numpy(u).to('cuda'))
    u_forward_z = op.z_forward(torch.from_numpy(u).to('cuda'))
    u_backward_z = op.z_backward(torch.from_numpy(u).to('cuda'))
    np.save(f'./op_forward_x_{backend}.npy', u_forward_x.cpu().numpy())
    np.save(f'./op_backward_x_{backend}.npy', u_backward_x.cpu().numpy())
    np.save(f'./op_forward_z_{backend}.npy', u_forward_z.cpu().numpy())
    np.save(f'./op_backward_z_{backend}.npy', u_backward_z.cpu().numpy())

