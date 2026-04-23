nt = 1500
dt = 0.002
delay = 0.3
fm = 8
true_path = "models/overthrust/true_2d.npy"
smooth_path = "models/overthrust/smooth_2d.npy"
obs_path = "models/overthrust/elastic_obs_ot.npy"
# smooth_path = "models/marmousi/linear.npy"
dh = 25.0
spatial_order = 4

abcn = 10
free_surface = False
use_habc = False

src_step = 2
rec_step = 1
srcz = 1
recz = 1

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

_FWI_3D_ACOUSTIC_COMMON = {
    **_SHARED_BASE,
    "fm": float(fm),
    "spatial_order": 2,
    "abcn": abcn,
    "free_surface": free_surface,
    "src_step": 16,
    "rec_step": 4,
    "srcz": srcz,
    "recz": recz,
    "src_margin": 8,
    "rec_margin": 4,
    "lr": 20.0,
    "batchsize": 4,
    "forward_batchsize": 1,
    "true_model": "models/overthrust/true_3d.npy",
    "init_model": "models/overthrust/smooth_3d.npy",
    "model_stride_z": 1,
    "model_stride_y": 4,
    "model_stride_x": 4,
}

_LSRTM_3D_ACOUSTIC_COMMON = {
    **_FWI_3D_ACOUSTIC_COMMON,
    "output_dir": "acoustic_3d_lsrtm_overthrust",
    "epochs": 51,
    "show_every": 10,
    "batchsize": 2,
    "forward_batchsize": 1,
    "lr_ref": 5e-2,
}


_CONFIGS = {
    "fwi_2d_elastic_jax": {
        **_FWI_2D_ELASTIC_COMMON,
        "output_dir": "elastic_jax_overthrust",
        "use_ckpt": True,
        "ckpt_chunks": 128,
    },
    "fwi_2d_elastic_torch_common": {
        **_FWI_2D_ELASTIC_COMMON,
        "fix_top_layers": 0,
    },
    "fwi_2d_elastic_torch_eager": {
        "output_dir": "elastic_fwi_overthrust_torch",
        "use_compile": True,
        "use_ckpt": True,
        "ckpt_chunks": 128,
        "record_layout": "standard",
    },
    "fwi_2d_elastic_torch_cuda": {
        "output_dir": "elastic_fwi_overthrust_cuda",
        "record_layout": "cuda",
        "boundary_saving_config": {
            "enabled": True,
            "storage": "gpu",
            "transfer_interval": 10,
            "pinned_memory": True,
        },
    },
    "fwi_3d_acoustic_jax": {
        **_FWI_3D_ACOUSTIC_COMMON,
        "output_dir": "acoustic_3d_jax_overthrust",
        "use_ckpt": True,
        "ckpt_chunks": 16,
    },
    "fwi_3d_acoustic_torch_common": {
        **_FWI_3D_ACOUSTIC_COMMON,
    },
    "fwi_3d_acoustic_torch_eager": {
        "output_dir": "acoustic_3d_fwi_overthrust_torch",
        "use_compile": True,
        "use_ckpt": True,
        "ckpt_chunks": 16,
    },
    "fwi_3d_acoustic_torch_cuda": {
        "output_dir": "acoustic_3d_fwi_overthrust_cuda",
        "boundary_saving_config": {
            "enabled": True,
            "storage": "gpu",
            "transfer_interval": 4,
            "pinned_memory": True,
        },
    },
    "lsrtm_3d_acoustic_jax": {
        **_LSRTM_3D_ACOUSTIC_COMMON,
        "output_dir": "acoustic_3d_lsrtm_overthrust_jax",
        "use_ckpt": True,
        "ckpt_chunks": 16,
    },
    "lsrtm_3d_acoustic_torch_common": {
        **_LSRTM_3D_ACOUSTIC_COMMON,
    },
    "lsrtm_3d_acoustic_torch_eager": {
        "output_dir": "acoustic_3d_lsrtm_overthrust_torch",
        "use_compile": True,
        "use_ckpt": True,
        "ckpt_chunks": 16,
    },
    "lsrtm_3d_acoustic_torch_cuda": {
        "output_dir": "acoustic_3d_lsrtm_overthrust_cuda",
        "boundary_saving_config": {
            "enabled": True,
            "storage": "gpu",
            "transfer_interval": 4,
            "pinned_memory": True,
        },
    },
}


def get_config(name):
    if name not in _CONFIGS:
        raise KeyError(f"Unknown Overthrust config '{name}'. Available keys: {sorted(_CONFIGS)}")
    return _CONFIGS[name].copy()
