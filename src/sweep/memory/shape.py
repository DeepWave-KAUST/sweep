class Layout:

    def __init__(
        self,
        shape,
        nvar,
        nt,
        abcn,                   # number of ABC layers
        M,                      # stencil radius, default to 4 for 8th order, 5 for 10th order
        B=1,                    # Batchsize for this propagator, default to 1 (no batching)
        transfer_interval=1,    # number of time steps for each CPU-GPU transfer
        free_surface=False,     # whether the top boundary is free surface (stress=0) or not (absorbing)
        width=-1,               # width of the boundary to be saved, default to M (the stencil radius)
        tangent_pad=0,           # extra saved cells in tangential directions
        cut_mask=0,             # DD cut-face bitmask (x_lo=1,x_hi=2,z_lo=4,z_hi=8,y_lo=16,y_hi=32)
        **kwargs
    ):

        dim = len(shape)

        if dim == 3:
            self.nz, self.ny, self.nx = shape
        else:
            self.nz, self.nx = shape
            self.ny = 1

        self.shape = shape
        self.nt = nt
        self.nvar = nvar
        self.abcn = abcn
        self.M = M
        self.B = B
        self.dim = dim
        self.free_surface = free_surface
        self.transfer_interval = transfer_interval

        # -------------------------
        # physical domain (match SolverContext::phys_* exactly, incl. the
        # cut-aware shrink: a cut face carries only the M halo, not abcn+M)
        # -------------------------
        self.cut_mask = cut_mask
        cx_lo = bool(cut_mask & 1);  cx_hi = bool(cut_mask & 2)
        cz_lo = bool(cut_mask & 4);  cz_hi = bool(cut_mask & 8)
        cy_lo = bool(cut_mask & 16); cy_hi = bool(cut_mask & 32)

        self.phys_x0 = M if cx_lo else abcn + M
        self.phys_x1 = self.nx - (M if cx_hi else abcn + M)

        if dim == 3:
            self.phys_y0 = M if cy_lo else abcn + M
            self.phys_y1 = self.ny - (M if cy_hi else abcn + M)
        else:
            self.phys_y0 = 0
            self.phys_y1 = 1

        self.phys_z0 = M if free_surface else (M if cz_lo else abcn + M)
        self.phys_z1 = self.nz - (M if cz_hi else abcn + M)

        self.nx_phys = self.phys_x1 - self.phys_x0
        self.ny_phys = self.phys_y1 - self.phys_y0
        self.nz_phys = self.phys_z1 - self.phys_z0

        if (width < 0):
            self.width = M
        else:
            self.width = width
        self.tangent_pad = tangent_pad

        self.nx_boundary = self.nx_phys + 2 * tangent_pad
        self.ny_boundary = self.ny_phys + 2 * tangent_pad
        self.nz_boundary = self.nz_phys + 2 * tangent_pad


    # -------------------------
    # CPU boundary shapes
    # -------------------------

    @property
    def left_shape(self):

        if self.dim == 3:
            return (
                self.nvar * self.nt,
                self.B,
                self.nz_boundary,
                self.ny_boundary,
                self.width,
            )

        return (
            self.nvar,
            self.nt,
            self.B,
            self.nz_boundary,
            self.width,
        )

    @property
    def right_shape(self):
        return self.left_shape

    @property
    def front_shape(self):

        if self.dim == 2:
            return None

        return (
            self.nvar * self.nt,
            self.B,
            self.nz_boundary,
            self.width,
            self.nx_boundary,
        )

    @property
    def back_shape(self):
        return self.front_shape

    @property
    def bottom_shape(self):

        if self.dim == 3:
            return (
                self.nvar * self.nt,
                self.B,
                self.width,
                self.ny_boundary,
                self.nx_boundary,
            )

        return (
            self.nvar,
            self.nt,
            self.B,
            self.width,
            self.nx_boundary,
        )

    @property
    def top_shape(self):
        return self.bottom_shape

    # -------------------------
    # GPU staging buffer
    # -------------------------

    @property
    def left_gpu_shape(self):

        if self.dim == 3:
            return (
                self.nvar * self.transfer_interval,
                self.B,
                self.nz_boundary,
                self.ny_boundary,
                self.width,
            )

        return (
            self.nvar,
            self.transfer_interval,
            self.B,
            self.nz_boundary,
            self.width,
        )

    @property
    def right_gpu_shape(self):
        return self.left_gpu_shape

    @property
    def front_gpu_shape(self):

        if self.dim == 2:
            return None

        return (
            self.nvar * self.transfer_interval,
            self.B,
            self.nz_boundary,
            self.width,
            self.nx_boundary,
        )

    @property
    def back_gpu_shape(self):
        return self.front_gpu_shape

    @property
    def bottom_gpu_shape(self):

        if self.dim == 3:
            return (
                self.nvar * self.transfer_interval,
                self.B,
                self.width,
                self.ny_boundary,
                self.nx_boundary,
            )

        return (
            self.nvar,
            self.transfer_interval,
            self.B,
            self.width,
            self.nx_boundary,
        )

    @property
    def top_gpu_shape(self):
        return self.bottom_gpu_shape

    # -------------------------
    # last two wavefields
    # -------------------------

    @property
    def last_two_shape(self):

        if self.dim == 3:
            return (
                self.nvar,
                2,
                self.B,
                1,
                self.nz,
                self.ny,
                self.nx,
            )

        return (
            self.nvar,
            2,
            self.B,
            1,
            self.nz,
            self.nx,
        )
    
    @property
    def cpu_shapes(self,):
        shapes = (
            self.top_shape, 
            self.bottom_shape, 
            self.front_shape, 
            self.back_shape, 
            self.left_shape, 
            self.right_shape
        )
        return tuple(shape for shape in shapes if shape is not None)

    @property
    def gpu_shapes(self):
        shapes = (
            self.top_gpu_shape,
            self.bottom_gpu_shape,
            self.front_gpu_shape,
            self.back_gpu_shape,
            self.left_gpu_shape,
            self.right_gpu_shape
        )
        return tuple(shape for shape in shapes if shape is not None)

    @property
    def gpu_full_shapes(self):
        return self.cpu_shapes
