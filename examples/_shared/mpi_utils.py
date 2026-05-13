from dataclasses import dataclass
import os

import numpy as np
import torch


@dataclass(frozen=True)
class MPIContext:
    enabled: bool
    comm: object = None
    rank: int = 0
    size: int = 1
    MPI: object = None

    @property
    def is_root(self):
        return self.rank == 0


def _positive_int_env(name):
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None
    try:
        parsed = int(value.split(",")[0])
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def _configure_torch_cpu_threads(local_size=1):
    num_threads = _positive_int_env("OMP_NUM_THREADS")
    if num_threads is None:
        try:
            available = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available = os.cpu_count() or 1
        num_threads = max(1, available // max(1, int(local_size)))

    torch.set_num_threads(num_threads)
    torch.set_num_interop_threads(1)


def init_mpi(enabled, backend, device=None, impl=None):
    if not enabled:
        return MPIContext(enabled=False)
    legacy_cpu_binding = backend == "c" and device == "cpu"
    torch_cpu = backend == "torch" and impl in {"eager", "c"} and device == "cpu"
    if not (legacy_cpu_binding or torch_cpu):
        raise ValueError("--mpi is currently supported only for CPU Torch runs: impl='eager' or impl='c'.")
    try:
        from mpi4py import MPI
    except ImportError as exc:
        raise RuntimeError("--mpi requires mpi4py. Install mpi4py and launch with mpirun/mpiexec.") from exc

    comm = MPI.COMM_WORLD
    node_comm = comm.Split_type(MPI.COMM_TYPE_SHARED, 0, MPI.INFO_NULL)
    local_size = node_comm.Get_size()
    node_comm.Free()
    _configure_torch_cpu_threads(local_size)
    return MPIContext(
        enabled=True,
        comm=comm,
        rank=comm.Get_rank(),
        size=comm.Get_size(),
        MPI=MPI,
    )


def barrier(mpi):
    if mpi.enabled:
        mpi.comm.Barrier()


def rank_zero_print(mpi, *args, **kwargs):
    if mpi.is_root:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def split_indices(indices, mpi):
    indices = np.asarray(indices, dtype=np.int64)
    if not mpi.enabled:
        return indices
    return np.array_split(indices, mpi.size)[mpi.rank]


def reduce_scalar(value, mpi, op="sum"):
    value = float(value)
    if not mpi.enabled:
        return value
    if op == "sum":
        mpi_op = mpi.MPI.SUM
    elif op == "max":
        mpi_op = mpi.MPI.MAX
    else:
        raise ValueError(f"Unsupported MPI scalar reduction '{op}'.")
    return float(mpi.comm.allreduce(value, op=mpi_op))


def _allreduce_tensor_(tensor, mpi):
    if not mpi.enabled or tensor is None or tensor.numel() == 0:
        return

    array = tensor.detach().numpy()
    if array.flags.c_contiguous:
        mpi.comm.Allreduce(mpi.MPI.IN_PLACE, array, op=mpi.MPI.SUM)
        return

    contiguous = np.ascontiguousarray(array)
    mpi.comm.Allreduce(mpi.MPI.IN_PLACE, contiguous, op=mpi.MPI.SUM)
    with torch.no_grad():
        tensor.copy_(torch.from_numpy(contiguous).to(tensor.device))


def allreduce_gradients(parameters, mpi):
    if not mpi.enabled:
        return
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad = torch.zeros_like(parameter)
        _allreduce_tensor_(parameter.grad, mpi)


def allreduce_tensors(tensors, mpi):
    if not mpi.enabled:
        return
    for tensor in tensors:
        _allreduce_tensor_(tensor, mpi)


def _mpi_dtype(mpi, dtype):
    dtype = np.dtype(dtype)
    if dtype == np.dtype("float32"):
        return mpi.MPI.FLOAT
    if dtype == np.dtype("float64"):
        return mpi.MPI.DOUBLE
    if dtype == np.dtype("int32"):
        return mpi.MPI.INT
    if dtype == np.dtype("int64"):
        return mpi.MPI.LONG_LONG
    return None


def allgather_shot_records(local_indices, local_data, total_shots, shot_axis, mpi):
    if not mpi.enabled:
        return local_data

    local_indices = np.asarray(local_indices, dtype=np.int64)
    local_shape = None if local_data is None else tuple(local_data.shape)
    local_dtype = None if local_data is None else np.dtype(local_data.dtype).str
    metadata = mpi.comm.allgather((local_indices, local_shape, local_dtype))

    first_shape = next(shape for _, shape, _ in metadata if shape is not None)
    dtype = np.dtype(next(dtype for _, _, dtype in metadata if dtype is not None))
    mpi_dtype = _mpi_dtype(mpi, dtype)

    if mpi_dtype is None:
        gathered = mpi.comm.allgather((local_indices, local_data))
        full_shape = list(first_shape)
        full_shape[shot_axis] = int(total_shots)
        full = np.empty(full_shape, dtype=dtype)
        for indices, piece in gathered:
            if piece is None:
                continue
            slices = [slice(None)] * len(full_shape)
            slices[shot_axis] = indices
            full[tuple(slices)] = piece
        return full

    if local_data is None:
        send = np.empty(0, dtype=dtype)
    else:
        send = np.ascontiguousarray(local_data, dtype=dtype).ravel()

    counts = np.asarray(mpi.comm.allgather(send.size), dtype=np.int64)
    displacements = np.zeros_like(counts)
    displacements[1:] = np.cumsum(counts[:-1])
    received = np.empty(int(counts.sum()), dtype=dtype)
    mpi.comm.Allgatherv(send, (received, counts, displacements, mpi_dtype))

    full_shape = list(first_shape)
    full_shape[shot_axis] = int(total_shots)
    full = np.empty(full_shape, dtype=dtype)

    offset = 0
    for indices, shape, _ in metadata:
        count = int(np.prod(shape)) if shape is not None else 0
        if count > 0:
            piece = received[offset : offset + count].reshape(shape)
            slices = [slice(None)] * len(full_shape)
            slices[shot_axis] = indices
            full[tuple(slices)] = piece
        offset += count

    return full
