import jax.numpy as jnp
from jax import vmap
from geophyai.scalars import staggered_grid_coes_torch
from jax.scipy.signal import convolve2d as conv2d

def _laplace(image, kernel):
    # Expected input shape: (height, width)
    return conv2d(image, kernel, mode='same')

batch_convolve2d = vmap(vmap(_laplace, in_axes=(0, None)), in_axes=(0, None))

def laplace(u, h=1, kernel=None):
    return batch_convolve2d(u, kernel) / (h ** 2)

class PartialDerivative:

    def __init__(self, spatial_order:int=4):
        self.coes = jnp.array(staggered_grid_coes_torch(int(spatial_order//2)), dtype=jnp.float32)
        num_kernels = spatial_order // 2 # max length is 2*num_kernels+1
        self.kxf = self.create_x_kernels(num_kernels) # x partial derivatives, forward mode
        self.kxb = self.create_x_kernels(num_kernels) # x partial derivatives, backward mode
        self.kzf = self.create_z_kernels(num_kernels) # z partial derivatives, forward mode
        self.kzb = self.create_z_kernels(num_kernels) # z partial derivatives, backward mode

        for i in range(num_kernels):
            self.kxf[i] = self.kxf[i].at[..., 0].set(-1)
            self.kxf[i] = self.kxf[i].at[..., -2].set(1)
            self.kxb[i] = self.kxb[i].at[..., 1].set(-1)
            self.kxb[i] = self.kxb[i].at[..., -1].set(1)
            self.kzf[i] = self.kzf[i].at[0,:].set(-1)
            self.kzf[i] = self.kzf[i].at[-2,:].set(1)
            self.kzb[i] = self.kzb[i].at[1,:].set(-1)
            self.kzb[i] = self.kzb[i].at[-1,:].set(1)
    
    def create_x_kernel(self, length: int):
        kernel = jnp.zeros((1,length), dtype=jnp.float32)
        return kernel
    
    def create_z_kernel(self, length: int):
        kernel = jnp.zeros((length,1), dtype=jnp.float32)
        return kernel

    def create_x_kernels(self, num_kernels: int):
        return [self.create_x_kernel(2*i+1) for i in range(1, num_kernels+1)]
    
    def create_z_kernels(self, num_kernels: int):
        return [self.create_z_kernel(2*i+1) for i in range(1, num_kernels+1)]
    
    def x_forward(self, u):
        results = jnp.zeros_like(u, dtype=jnp.float32)
        for i in range(len(self.kxf)):
            results = results + batch_convolve2d(u, self.kxf[i])*self.coes[i]
        return results
    
    def x_backward(self, u):
        results = jnp.zeros_like(u, dtype=jnp.float32)
        for i in range(len(self.kxb)):
            results = results + batch_convolve2d(u, self.kxb[i])*self.coes[i]
        return results
    
    def z_forward(self, u):
        results = jnp.zeros_like(u, dtype=jnp.float32)
        for i in range(len(self.kzf)):
            results = results + batch_convolve2d(u, self.kzf[i])*self.coes[i]
        return results
    
    def z_backward(self, u):
        results = jnp.zeros_like(u, dtype=jnp.float32)
        for i in range(len(self.kzb)):
            results = results + batch_convolve2d(u, self.kzb[i])*self.coes[i]
        return results