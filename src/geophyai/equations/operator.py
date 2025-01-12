import torch
from geophyai.scalars import staggered_grid_coes_torch
from typing import List

@torch.jit.script
def laplace(u: torch.Tensor, 
            h: torch.Tensor, 
            kernel: torch.Tensor) -> torch.Tensor:
    padding = kernel.shape[-1] // 2
    return torch.nn.functional.conv2d(u, kernel, padding=padding) / (h**2)

class PartialDerivative:

    def __init__(self, spatial_order:int=4, device:torch.device=torch.device('cpu')):
        self.dev = device
        self.coes = staggered_grid_coes_torch(int(spatial_order//2)).to(self.dev)
        num_kernels = spatial_order // 2 # max length is 2*num_kernels+1
        self.kxf = self.create_x_kernels(num_kernels) # x partial derivatives, forward mode
        self.kxb = self.create_x_kernels(num_kernels) # x partial derivatives, backward mode
        self.kzf = self.create_z_kernels(num_kernels) # z partial derivatives, forward mode
        self.kzb = self.create_z_kernels(num_kernels) # z partial derivatives, backward mode
        for i in range(num_kernels):
            self.kxf[i][..., 0] = -1
            self.kxf[i][..., -2] = 1
            self.kxb[i][..., 1] = -1
            self.kxb[i][..., -1] = 1
            self.kzf[i][:,:,0,:] = -1
            self.kzf[i][:,:,-2,:] = 1
            self.kzb[i][:,:,1,:] = -1
            self.kzb[i][:,:,-1,:] = 1

    def create_x_kernel(self, length: int) -> torch.Tensor:
        kernel = torch.zeros((1,1,1,length)).to(self.dev)
        return kernel
    
    def create_z_kernel(self, length: int) -> torch.Tensor:
        kernel = torch.zeros((1,1,length,1)).to(self.dev)
        return kernel

    def create_x_kernels(self, num_kernels: int) -> List[torch.Tensor]:
        return [self.create_x_kernel(2*i+1) for i in range(1, num_kernels+1)]
    
    def create_z_kernels(self, num_kernels: int) -> List[torch.Tensor]:
        return [self.create_z_kernel(2*i+1) for i in range(1, num_kernels+1)]
    
    def x_forward(self, u: torch.Tensor) -> torch.Tensor:
        results = torch.zeros_like(u)
        for i in range(len(self.kxf)):
            results += torch.nn.functional.conv2d(u, self.kxf[i], padding='same')*self.coes[i]
        return results
    
    def x_backward(self, u: torch.Tensor) -> torch.Tensor:
        results = torch.zeros_like(u)
        for i in range(len(self.kxb)):
            results += torch.nn.functional.conv2d(u, self.kxb[i], padding='same')*self.coes[i]
        return results
    
    def z_forward(self, u: torch.Tensor) -> torch.Tensor:
        results = torch.zeros_like(u)
        for i in range(len(self.kzf)):
            results += torch.nn.functional.conv2d(u, self.kzf[i], padding='same')*self.coes[i]
        return results
    
    def z_backward(self, u: torch.Tensor) -> torch.Tensor:
        results = torch.zeros_like(u)
        for i in range(len(self.kzb)):
            results += torch.nn.functional.conv2d(u, self.kzb[i], padding='same')*self.coes[i]
        return results