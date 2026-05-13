import torch


PUBLIC_BACKENDS = ("torch", "jax")
TORCH_IMPLS = ("eager", "c")
DEVICES = ("auto", "cuda", "cpu")

LEGACY_BACKEND_ALIASES = {
    "pytorch": ("torch", "eager", None),
    "eager": ("torch", "eager", None),
    "c": ("torch", "c", None),
    "cuda": ("torch", "c", "cuda"),
    "cpu": ("torch", "c", "cpu"),
}

IMPL_ALIASES = {}


def resolve_backend_impl_device(backend="torch", impl=None, device="auto", *, allow_jax=False):
    backend = str(backend).lower()
    impl = None if impl is None else str(impl).lower()
    device = str(device).lower()

    if device not in DEVICES:
        raise ValueError(f"Unsupported device '{device}'. Expected one of {list(DEVICES)}.")

    if backend in LEGACY_BACKEND_ALIASES:
        legacy_backend = backend
        backend, legacy_impl, legacy_device = LEGACY_BACKEND_ALIASES[legacy_backend]
        if impl is not None and IMPL_ALIASES.get(impl, impl) != legacy_impl:
            raise ValueError(
                f"Legacy --backend {legacy_backend} implies --impl {legacy_impl}, got --impl {impl}."
            )
        impl = legacy_impl
        if legacy_device is not None:
            if device != "auto" and device != legacy_device:
                raise ValueError(
                    f"Legacy --backend {legacy_backend} implies --device {legacy_device}, got --device {device}."
                )
            device = legacy_device

    if backend == "pytorch":
        backend = "torch"
    if backend not in PUBLIC_BACKENDS:
        expected = list(PUBLIC_BACKENDS) + list(LEGACY_BACKEND_ALIASES)
        raise ValueError(f"Unsupported backend '{backend}'. Expected one of {expected}.")
    if backend == "jax" and not allow_jax:
        raise ValueError("This example is Torch-based. Use a JAX example for --backend jax.")

    if backend == "torch":
        impl = "eager" if impl is None else IMPL_ALIASES.get(impl, impl)
        if impl not in TORCH_IMPLS:
            raise ValueError(f"Unsupported torch impl '{impl}'. Expected one of {list(TORCH_IMPLS)}.")
    elif impl is None:
        impl = "eager"

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")

    return backend, impl, device


def add_backend_impl_device_args(parser, *, default_backend="torch", default_impl=None):
    parser.add_argument(
        "--backend",
        metavar="{torch,jax}",
        default=default_backend,
        help=(
            "Array/programming backend. Torch examples use backend='torch'. "
            "Legacy aliases 'pytorch', 'eager', 'c', 'cuda', and 'cpu' are still accepted."
        ),
    )
    parser.add_argument(
        "--impl",
        metavar="{eager,c}",
        default=default_impl,
        help=(
            "Torch implementation: 'eager' uses PyTorch ops, 'c' uses compiled C++/CUDA kernels. "
            "Defaults to 'eager' for --backend torch."
        ),
    )
    parser.add_argument(
        "--device",
        choices=DEVICES,
        default="auto",
        help="Execution device. 'auto' uses CUDA when available, otherwise CPU.",
    )
    return parser
