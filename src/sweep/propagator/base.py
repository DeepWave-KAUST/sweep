from collections.abc import Sequence

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
                 ckpt_mode="chunk",
                 ckpt_num=0,
                 pml_type='spml',
                 nt=-1,
                 B=1,
                 allow_growth=True,
                 full_mode="full",
                 boundary_saving_config=None,
                 **kwargs):
        """Base class for the RNN

        Args:
            equation (class): The wave equation class from sweep.equations
            shape (tupel or list): The shape of the model
            source_type (list, optional): List of strings for the source type. Defaults to [].
            receiver_type (list, optional): List of strings for the receiver type. Defaults to [].
            abcn (int, optional): The number of layers of absorbing boundary conditions. Defaults to 50.
            free_surface (bool, optional): If the model has a free surface. Defaults to False.
            dh (float or sequence, optional): Grid spacing in model-axis order.
                For 2D use ``(dz, dx)`` and for 3D use ``(dz, dy, dx)``.
                Defaults to 10..
            dt (float, optional): Time step (seconds). Defaults to 0.002.
            dev (str, optional): The device to run the simulation on. Defaults to None.
            use_ckpt (bool, optional): Use checkpointing to save memory. Defaults to True.
            ckpt_chunks (int, optional): The number of time steps to chunk for checkpointing. Defaults to 50.
            ckpt_mode (str, optional): Checkpointing mode. "chunk" stores periodic checkpoints and
                replays each chunk, while "recursive" stores a fixed number of checkpoints and
                recursively recomputes intermediate states. Defaults to "chunk".
            ckpt_num (int, optional): Number of persistent checkpoints to save when
                ckpt_mode="recursive". Defaults to 0.
            pml_type (str, optional): The type of PML to use. Defaults to 'spml'. Options include 'spml', 'cpml', 'cpmlr', etc.
            nt (int, optional): The number of time steps. Defaults to -1, which means it will be determined by the length of the source time function.
            B (int, optional): The batch size for the simulation. Defaults to 1.
            allow_growth (bool, optional): Whether to allow GPU memory growth. Defaults to True.
            boundary_saving_config (dict, optional): Configuration for boundary saving. Defaults to None, which means boundary saving is disabled. If provided, it should be a dictionary with the following keys:
                - enabled (bool): Whether to enable boundary saving. If True, the boundary wavefields will be saved and transferred to CPU for checkpointing. Defaults to False.
                - storage (str): Where to store the boundary wavefields. Options are 'gpu' and 'cpu'. If 'gpu', the boundary wavefields will be stored in GPU memory. If 'cpu', the boundary wavefields will be transferred to CPU memory. Defaults to 'gpu'.
                - transfer_interval (int): The interval (in time steps) at which to transfer the boundary wavefields to CPU memory if storage is 'cpu'. For example, if transfer_interval is 10, then every 10 time steps the boundary wavefields will be transferred to CPU memory. Defaults to 1.
                - pinned_memory (bool): Whether to use pinned memory for the boundary wavefields when storage is 'cpu'. Using pinned memory can speed up the transfer between GPU and CPU. Defaults to False.
        """
        
        self.equation = equation
        if getattr(self.equation, 'setup_pml', None):
            self.equation.setup_pml(pml_type)
        self.wavefield_names = equation.wavefields
        self.model_names = equation.models
        self.shape = shape
        self.ndim = len(shape)
        self.dev = dev
        self.abcn = abcn
        self.free_surface = free_surface
        if np.isscalar(dh):
            self._dh = float(dh)
            self._grid_spacing = tuple([self._dh] * self.ndim)
        else:
            if not isinstance(dh, Sequence) or isinstance(dh, (str, bytes)):
                raise TypeError(
                    "dh must be a float or a sequence ordered like shape "
                    "(2D: (dz, dx), 3D: (dz, dy, dx))."
                )
            if len(dh) != self.ndim:
                raise ValueError(
                    f"dh must have length {self.ndim} to match shape {shape}, "
                    f"got {len(dh)}."
                )
            if self.__class__.__name__ != "PropCUDA":
                raise ValueError(
                    "Anisotropic dh is currently supported only by the CUDA solver. "
                    "Use a scalar dh for non-CUDA propagators."
                )
            self._grid_spacing = tuple(float(v) for v in dh)
            self._dh = float(self._grid_spacing[-1])
        self._dt = float(dt)
        self.use_ckpt = use_ckpt
        self.ckpt_chunks = ckpt_chunks
        self.ckpt_mode = ckpt_mode
        self.ckpt_num = ckpt_num
        self.pml_type = pml_type

        self.nt = nt
        self.B = B
        self.allow_growth = allow_growth
        self.full_mode = full_mode
        legacy_boundary_config = {
            "transfer_interval": kwargs.pop("transfer_interval", 1),
            "storage": "cpu" if kwargs.pop("boundary_on_cpu", False) else "gpu",
            "pinned_memory": kwargs.pop("use_pinned_memory", False),
        }
        if boundary_saving_config is None:
            boundary_saving_config = legacy_boundary_config
        else:
            boundary_saving_config = {**legacy_boundary_config, **boundary_saving_config}

        self.boundary_saving_config = self._normalize_boundary_saving_config(boundary_saving_config)
        self.transfer_interval = self.boundary_saving_config["transfer_interval"]
        self.boundary_on_cpu = (self.boundary_saving_config["storage"] == "cpu")
        self.use_pinned_memory = self.boundary_saving_config["pinned_memory"]

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
        self.shape_cuda = tuple([s+self.equation.so for s in self.shape])

    def _normalize_boundary_saving_config(self, config):
        default = {
            "enabled": False,
            "storage": "gpu",
            "transfer_interval": 1,
            "pinned_memory": False,
        }

        if config is None:
            return default

        merged = default.copy()
        merged.update(config)

        if merged["storage"] not in {"gpu", "cpu"}:
            raise ValueError("boundary_saving_config['storage'] must be 'gpu' or 'cpu'")

        if merged["transfer_interval"] < 1:
            raise ValueError("boundary_saving_config['transfer_interval'] must be >= 1")

        if merged["storage"] == "gpu":
            merged["transfer_interval"] = 1
            merged["pinned_memory"] = False

        return merged

    def resolve_boundary_saving_config(self, override=None, use_boundary_saving=None):
        config = self.boundary_saving_config.copy()
        if override is not None:
            config = self._normalize_boundary_saving_config({**config, **override})
        if use_boundary_saving is not None:
            config["enabled"] = bool(use_boundary_saving)
        return config

    def init_abc(self, **kwargs):
        _padding = [self.equation.so // 2, self.equation.so // 2] * self.ndim
        self.equation.init_abc(
                type=self.pml_type,
                pml_width=[self.abcn if not self.free_surface else 0] + (2**self.ndim-1) * [self.abcn],
                accuracy=self.equation.so,
                fd_pad=kwargs.get('fd_pad', _padding),#[self.equation.so//2 if not self.free_surface else 0] + (2**self.ndim-1) * [self.equation.so//2], #
                dt=self._dt, 
                grid_spacing=list(self._grid_spacing),
                max_vel=kwargs.get('max_vel', 4500.0),
                dtype=np.float32,
                pml_freq=kwargs.get('pml_freq', 25.0),
                shape=kwargs.get('shape', self.shape)
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
