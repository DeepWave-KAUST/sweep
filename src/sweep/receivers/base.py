class ReceiverBase:
    def __init__(self, **kwargs):
        pass

    def forward(self, wavefield):
        return wavefield[(self.bidx, slice(None), *self.coords_r)]