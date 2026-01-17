import numpy as np

class PropBase:

    def __init__(self,
                 equation, 
                 shape, 
                 source_type: list=[],
                 receiver_type: list=[],
                 abcn=50, 
                 free_surface=False, 
                 dh=10., 
                 dt=0.002, 
                 dev=None, 
                 use_ckpt=True,
                 ckpt_chunks=100,
                 pml_type='spml',
                 **kwargs):
        """Base class for the RNN

        Args:
            equation (class): The wave equation class from sweep.equations
            shape (tupel or list): The shape of the model
            source_type (list, optional): List of strings for the source type. Defaults to [].
            receiver_type (list, optional): List of strings for the receiver type. Defaults to [].
            abcn (int, optional): The number of layers of absorbing boundary conditions. Defaults to 50.
            free_surface (bool, optional): If the model has a free surface. Defaults to False.
            dh (float, optional): Grid spacing (meters). Defaults to 10..
            dt (float, optional): Time step (seconds). Defaults to 0.002.
            dev (str, optional): The device to run the simulation on. Defaults to None.
            use_ckpt (bool, optional): Use checkpointing to save memory. Defaults to True.
            ckpt_chunks (int, optional): The number of time steps to chunk for checkpointing. Defaults to 50.
            pml_type (str, optional): 
        """
        
        self.equation = equation
        if getattr(self.equation, 'setup_pml', None):
            self.equation.setup_pml(pml_type)
        self.wavefield_names = equation.wavefields
        self.model_names = equation.models
        self.shape = shape
        self.dev = dev
        self.abcn = abcn
        self.free_surface = free_surface
        self._dh = float(dh)
        self._dt = float(dt)
        self.use_ckpt = use_ckpt
        self.ckpt_chunks = ckpt_chunks
        self.ndim = len(shape)
        self.pml_type = pml_type

        self.source_type = source_type
        self.receiver_type = receiver_type

        if self.free_surface:
            self.padding_z = (0, self.abcn)
            shape_z = self.shape[0] + self.abcn
        else:
            self.padding_z = (self.abcn, self.abcn)
            shape_z = self.shape[0] + 2*self.abcn

        self.padding = (self.abcn,) * 2*(self.ndim-1) + self.padding_z
        self.shape_nopad = tuple([w+2*self.equation.so for w in self.shape])
        self.shape = (shape_z,) + tuple(s+2*self.abcn for s in self.shape[1:])

    def init_abc(self, **kwargs):
        self.equation.init_abc(
                type=self.pml_type,
                pml_width=[self.abcn if not self.free_surface else 0] + (2**self.ndim-1) * [self.abcn],
                accuracy=self.equation.so,
                fd_pad=[self.equation.so // 2, self.equation.so // 2] * self.ndim,#[self.equation.so//2 if not self.free_surface else 0] + (2**self.ndim-1) * [self.equation.so//2], #
                dt=self._dt, 
                grid_spacing=[self._dh]*self.ndim,
                max_vel=kwargs.get('max_vel', 4500.0),
                dtype=np.float32,
                pml_freq=kwargs.get('pml_freq', 25.0),
                shape=self.shape,
        )
        
        if getattr(self.equation, 'need_init', False):
            self.equation.init(self.shape, self.dev, self._dh)

    def crop(self, data):
        """Crop the data to the original shape

        Args:
            data (np.ndarray): The data to be cropped

        Returns:
            np.ndarray: The cropped data
        """
        if self.free_surface:
            return data[..., 0:-self.abcn, self.abcn:-self.abcn]
        else:
            s = slice(self.abcn, -self.abcn)
            return data[(...,) + (s,) * self.ndim]

    def get_parameters(self, key):
        assert key in self.model_names, f'Key must be in {self.model_names}, got {key}'
        yield getattr(self, key)

    def parameters(self, ):
        return [getattr(self, name) for name in self.model_names]