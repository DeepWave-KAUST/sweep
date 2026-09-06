import os, time, json
import numpy as np
import torch, torch.distributed as dist
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

SHAPE = (384, 512); DH, DT, NT = 6.0, 0.8e-3, 2000
FREQ, DELAY = 14.0, 0.08; NSHOTS = 8

dist.init_process_group(backend="nccl")
rank, world = dist.get_rank(), dist.get_world_size()
local = int(os.environ.get("LOCAL_RANK", rank))
torch.cuda.set_device(local)
dev = torch.device(f"cuda:{local}")

vp_true = np.full(SHAPE, 1500.0, dtype=np.float32); vp_true[SHAPE[0] // 2:, :] = 2500.0
vp_init = np.full(SHAPE, 1500.0, dtype=np.float32)
sx = np.linspace(20, SHAPE[1] - 21, NSHOTS, dtype=np.int64)
sources = np.stack([sx, np.full(NSHOTS, 2, dtype=np.int64)], axis=1)
rx = np.linspace(0, SHAPE[1] - 1, 128, dtype=np.int64)
receivers = np.broadcast_to(np.stack([rx, np.full_like(rx, 4)], axis=1),
                            (NSHOTS, rx.size, 2)).copy()
t = np.arange(NT, dtype=np.float32) * DT
wavelet = ricker(t - DELAY, f=FREQ).astype(np.float32)

solver = PropTorch(Acoustic(device=dev), shape=SHAPE, dh=DH, dt=DT, dev=dev,
                   use_ckpt=False, impl="c")
with torch.no_grad():
    obs = solver(wavelet, sources, receivers,
                 models=[torch.tensor(vp_true, device=dev)])
    _ = solver(wavelet, sources[:1], receivers[:1],
               models=[torch.tensor(vp_init, device=dev)])
torch.cuda.synchronize(dev); dist.barrier()

my_shots = np.array_split(np.arange(NSHOTS), world)[rank].tolist()
vp = torch.tensor(vp_init, device=dev, requires_grad=True)
t0 = time.perf_counter()
local_loss = 0.0
for s in my_shots:
    syn = solver(wavelet, sources[s:s+1], receivers[s:s+1], models=[vp])
    loss = 0.5 * (syn - obs[s:s+1]).pow(2).sum()
    loss.backward()
    local_loss += float(loss.detach())
torch.cuda.synchronize(dev)
dist.all_reduce(vp.grad, op=dist.ReduceOp.SUM)
loss_t = torch.tensor([local_loss], device=dev)
dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
dist.barrier()
elapsed = time.perf_counter() - t0

if rank == 0:
    print(json.dumps({"world": world,
                       "loss": float(loss_t.item()),
                       "elapsed": elapsed,
                       "grad_abs_sum": float(vp.grad.abs().sum().item())}))
dist.destroy_process_group()
