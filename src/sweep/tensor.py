import jax
import jax.numpy as jnp
from typing import Callable

class JaxTensor:
    def __init__(self, value: jnp.ndarray):
        self.value = value
        self.grad = None
        self._id = id(self)

    def __repr__(self):
        return f"JaxTensor(value={self.value}, grad={self.grad})"
    
    def shape(self):
        return self.value.shape