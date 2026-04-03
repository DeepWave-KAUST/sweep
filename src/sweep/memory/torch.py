import torch

class Allocator:
    def __init__(self, device):
        self.device = device
        self.data = []

    def zeros(self, shapes, dtype=torch.float32, dev=None, pin_memory=False):
        device = self.device if dev is None else dev
        tensors = [
            torch.zeros(
                s,
                dtype=dtype,
                device=device,
                pin_memory=(pin_memory and str(device) == 'cpu')
            )
            for s in shapes
        ]
        self.data.extend(tensors)
        return tensors

    def zero_(self,):
        for d in self.data:
            d.zero_()
