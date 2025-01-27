import torch
import numpy as np
import torch.nn as nn
import tinycudann as tc

class SineMLP(nn.Module):
    def __init__(self, 
                 layers: int = 4,
                 features: int = 64, 
                 out_dim: int = 1, 
                 in_dim: int = 2, 
                 omega: float = 30, 
                 use_bias=False, 
                 use_hash=False, 
                 hash_config=None):
        
        super().__init__()
        self.use_hash = use_hash

        if use_hash:
            # raise NotImplementedError('Hashing not implemented yet')
            # self.enc = MultiResHashGrid(in_dim, 16, 2, 18, 64, 256)

            self.enc = tc.Encoding(in_dim, hash_config, dtype=torch.float32)
            in_dim = self.enc.n_output_dims
            print('Reset in_dim to', in_dim)
        self.first_layer = nn.Linear(in_dim, 
                                     features, 
                                     bias=use_bias)
        self.last_layer = nn.Linear(features,
                                    out_dim,
                                    bias=use_bias)
        self.omega = omega
        # initialize the weights
        with torch.no_grad():
            self.first_layer.weight.uniform_(-1 / in_dim, 
                                             1 / in_dim)
            self.last_layer.weight.uniform_(-np.sqrt(6./features)/omega,
                                            np.sqrt(6./features)/omega)
            
        self.layers = []
        for _ in range(layers):
            layer = nn.Linear(features, features, bias=use_bias)
            with torch.no_grad():
                layer.weight.uniform_(-np.sqrt(6./features)/omega,
                                      np.sqrt(6./features)/omega)
            self.layers.append(layer)
        
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x, omega=None):
        shape = x.shape[:-1]
        omega = omega if omega is not None else self.omega
        if self.use_hash:
            x = self.enc(x.reshape(-1, 2)).view(*shape, -1)
            # x = self.enc(x)
        x = torch.sin(omega * self.first_layer(x))
        for layer in self.layers:
            x = torch.sin(omega * layer(x))
        x = self.last_layer(x)
        return x
    
class MLP(nn.Module):

    def __init__(self, 
                 layers: int, 
                 features: int, 
                 out_dim: int=1, 
                 in_dim: int=2, 
                 use_bias=True, 
                 use_hash=False, 
                 hash_config=None, 
                 activation=torch.tanh
                 ):
        super().__init__()

        self.use_hash = use_hash
        self.activation = activation
        if use_hash:
            # self.enc = MultiResHashGrid(in_dim, 16, 2, 18, 64, 256)
            self.enc = tc.Encoding(in_dim, hash_config, dtype=torch.float32)
            in_dim = self.enc.n_output_dims

        self.first_layer = nn.Linear(in_dim, 
                                     features, 
                                     bias=use_bias)
        self.last_layer = nn.Linear(features,
                                    out_dim,
                                    bias=use_bias)
        self.layers = []
        for _ in range(layers):
            layer = nn.Linear(features, features, bias=use_bias)
            self.layers.append(layer)
        
        self.layers = nn.ModuleList(self.layers)


    def forward(self, x):
        shape = x.shape[:-1]
        if self.use_hash:
            x = self.enc(x.reshape(-1, 2)).view(*shape, -1)
            # x = self.enc(x)
        x = self.activation(self.first_layer(x))
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.last_layer(x)
        return x
