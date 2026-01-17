class ReceiverBase:
    def __init__(self, **kwargs):
        pass

    def forward(self, wavefield):
        return wavefield[self.bidx, :, *self.coords_r]