nt = 2500
dt = 0.002
delay = 0.256
fm = 5
true_path = "models/marmousi/true.npy"
smooth_path = "models/marmousi/smooth.npy"
# smooth_path = "models/marmousi/linear.npy"
dh = 25.0
spatial_order = 8

abcn = 20
free_surface = False
use_habc = False

src_step = 2
rec_step = 1
srcz = 1
recz = 18

lr = 25
epochs = 101

batchsize = 8
step_per_epoch = 4
batch_per_step = int(batchsize / step_per_epoch)
show_every = 10
source_encoding = True
wavelet_scale = 1e6


_SHARED_BASE = {
    "nt": nt,
    "dt": dt,
    "delay": delay,
    "dh": dh,
    "epochs": epochs,
    "show_every": show_every,
}

_FWI_2D_ACOUSTIC_COMMON = {
    **_SHARED_BASE,
    "fm": float(fm),
    "spatial_order": spatial_order,
    "abcn": abcn,
    "free_surface": free_surface,
    "src_step": src_step,
    "rec_step": rec_step,
    "srcz": srcz,
    "recz": recz,
    "lr": float(lr),
    "batchsize": batchsize,
    "true_model": true_path,
    "init_model": smooth_path,
}

_FWI_2D_ELASTIC_COMMON = {
    **_SHARED_BASE,
    "fm": float(fm),
    "wavelet_scale": wavelet_scale,
    "spatial_order": spatial_order,
    "abcn": abcn,
    "free_surface": free_surface,
    "src_step": src_step * 2,
    "rec_step": rec_step * 2,
    "srcz": srcz,
    "recz": recz,
    "lr_vp": float(lr),
    "lr_vs": float(lr) / 1.73,
    "lr_rho": 0.0,
    "batchsize": batchsize,
    "source_encoding": source_encoding,
    "true_model": true_path,
    "init_model": smooth_path,
}


_CONFIGS = {
    "fwi_2d_acoustic_jax": {
        **_FWI_2D_ACOUSTIC_COMMON,
        "output_dir": "acoustic_fwi_jax",
        "use_ckpt": False,
    },
    "fwi_2d_acoustic_torch_common": {
        **_FWI_2D_ACOUSTIC_COMMON,
    },
    "fwi_2d_acoustic_torch_eager": {
        "output_dir": "acoustic_fwi_torch",
        "use_compile": True,
        "use_ckpt": False,
        "transpose_shot": False,
    },
    "fwi_2d_acoustic_torch_cuda": {
        "output_dir": "acoustic_fwi_cuda",
        "transpose_shot": True,
        "boundary_saving_config": {
            "enabled": True,
            "storage": "gpu",
            "transfer_interval": 10,
            "pinned_memory": True,
        },
    },
    "fwi_2d_elastic_jax": {
        **_FWI_2D_ELASTIC_COMMON,
        "output_dir": "elastic_jax_marmousi",
        "use_ckpt": False,
        "ckpt_chunks": 128,
    },
    "fwi_2d_elastic_torch_common": {
        **_FWI_2D_ELASTIC_COMMON,
        "fix_top_layers": 0,
    },
    "fwi_2d_elastic_torch_eager": {
        "output_dir": "elastic_fwi_marmousi_torch",
        "use_compile": True,
        "use_ckpt": False,
        "ckpt_chunks": 128,
        "record_layout": "standard",
    },
    "fwi_2d_elastic_torch_cuda": {
        "output_dir": "elastic_fwi_marmousi_cuda",
        "record_layout": "cuda",
        "boundary_saving_config": {
            "enabled": True,
            "storage": "gpu",
            "transfer_interval": 10,
            "pinned_memory": True,
        },
    },
}


def get_config(name):
    if name not in _CONFIGS:
        raise KeyError(f"Unknown Marmousi config '{name}'. Available keys: {sorted(_CONFIGS)}")
    return _CONFIGS[name].copy()
