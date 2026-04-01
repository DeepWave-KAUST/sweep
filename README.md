# SWEEP
Seismic Wave Equation Exploration Platform (SWEEP) is a Python package designed for seismic wave equation modeling and inversion.

** Note: From this version on, lazy imports are supported. You no longer need to install both JAX and PyTorch—you only need to install whichever backend you intend to use.

## Installation
```bash
python -m build
pip install dist/*.whl
```

## Usage
The following example shows how to compute the gradient of the a toy model with respect to the velocity model.
```python
import torch
# import jax
torch.backends.cudnn.benchmark = True
from sweep.propagator.torch import PropTorch
# from sweep.propagator.jax import PropJax
from sweep.equations import Acoustic
from sweep.signal import ricker
import numpy as np
import matplotlib.pyplot as plt

# Model parameters
nt = 1500
dt = 0.002
dh = 10
delay = 0.1
fm = 5
spatial_order = 8
shape = (100,100)

# Device
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Create a 2-layer model
true_model = np.ones(shape, dtype=np.float32)*1500
true_model[50:, :] = 2000

# Create a model
model = PropTorch(Acoustic(spatial_order=spatial_order, device=dev, backend='torch'), 
            shape=shape, 
            dev=dev, 
            dh=dh,
            dt=dt,
            source_type=['h1'],
            receiver_type=['h1'],
            pml_type='cpmlr',
            free_surface=False)
            
# Set the model parameters (Pytorch)
vp = torch.from_numpy(true_model).to(dev).requires_grad_(True)
# Set the model parameters (Jax)
# model.set_parameters([jnp.array(true_model)])
# Create a wavelet
t = np.arange(0, int(nt//2)*dt, dt)
wave = ricker(t-delay, f=fm)

# Acquicition geometry
sources = np.array([[1, 1]]) # in grid, shape=(nshots, 2)
receivers = np.array([[[99, 1]]]) # in grid, shape=(nshots, nreceivers, 2)

# Forward modeling
# Backward propagation (Pytorch)
obs = model.forward(wave, sources, receivers, models=[vp])
obs.pow(2).sum().backward()
# Backward propagation (Jax)
# def fwi(vp):
#     return (model(wave, sources, receivers, models=[vp])**2).sum()
# grad = jax.grad(fwi)(model.vp)

# Show the results
fig, axes=plt.subplots(1,3, figsize=(12,3))

axes[0].imshow(true_model, cmap='seismic', aspect='auto')
axes[0].set_title('True model')
axes[1].plot(obs.detach().cpu().numpy().squeeze(), label='Observed data')
grad = vp.grad.detach().cpu().numpy() # Pytorch
# grad = jax.device_get(grad) # Jax
vmin,vmax=np.percentile(grad, [1,99])
axes[1].set_title('Observed data')
axes[2].imshow(grad, cmap='seismic', aspect='auto', vmin=vmin, vmax=vmax)
axes[2].set_title('Gradient of vp')
fig.tight_layout()
fig.savefig('grad_vp.png', dpi=300, bbox_inches='tight')
plt.close()
```
The ground truth model, observed data and the gradient of the velocity model are shown below.
![grad_vp](figures/grad_vp.png)

# Examples
Some examples are provided in the `examples` folder (Still working on it since some of the APIs are changed). 

## License
This project is licensed under the MIT License.