# SWEEP
Seismic Wave Equation Exploration Platform (SWEEP) is a Python package designed for seismic wave equation modeling and inversion.

## Installation
```bash
python -m build
pip install dist/*.whl
```

## Usage
The following example shows how to compute the gradient of the a toy model with respect to the velocity model.
```python
import torch
torch.backends.cudnn.benchmark = True
from sweep.rnn import RNN
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
model = RNN(Acoustic(spatial_order=spatial_order, device=dev), 
            shape=shape, 
            dev=dev, 
            dh=dh,
            dt=dt,
            source_type=['h1'],
            receiver_type=['h1'],
            abcn=50, 
            free_surface=False)
            
# Set the model parameters (Pytorch)
model.set_parameters([torch.from_numpy(true_model).to(dev)])
# Set the model parameters (Jax)
model.set_parameters([jnp.array(true_model)])
# Create a wavelet
t = np.arange(0, int(nt//2)*dt, dt)
wave = ricker(t-delay, f=fm)

# Acquicition geometry
sources = np.array([[1, 1]]) # in grid, shape=(nshots, 2)
receivers = np.array([[[99, 1]]]) # in grid, shape=(nshots, nreceivers, 2)

# Forward modeling
# Backward propagation (Pytorch)
obs = model.forward(wave, sources, receivers)
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
grad = model.vp.grad.detach().cpu().numpy() # Pytorch
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
Three examples are provided in the `examples` folder. 
- The script `acoustic_fwi.py` shows an example of a simple acoustic inversion with Marmousi model. 
- The script `elastic_fwi.py` shows an example of a simple elastic multi-parameters inversion with Marmousi model.
- The script `acoustic_encoding_fwi.py` shows an example of the source encoding acoustic inversion with Marmousi model.
For running the examples, please install the package and type `python xxx.py` in the terminal.

# CLI
The following command lists all available equations in GeophyAI, it will also show the available parameters for each equation.
```
geophyai list equations
```
