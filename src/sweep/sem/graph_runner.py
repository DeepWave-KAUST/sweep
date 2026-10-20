"""CUDA-graph replay for the SEM ``impl='c'`` time loop (opt-in).

Why this exists
---------------
The C++ drivers run the whole ``nt``-step loop in C++, but they still issue one
``cudaLaunchKernel`` per kernel per step, and the SEM step is made of many small
kernels. Measured on an idle RTX 6000 Ada (elastic C-PML, 1120 elements, 18193
GLL nodes, nt = 400):

===================  ==========  ==========  =====
mode                 wall        GPU busy    gap
===================  ==========  ==========  =====
forward              10.3 ms     7.0 ms      32 %
forward + backward   60.6 ms     17.3 ms     71 %
===================  ==========  ==========  =====

That gap is ~2.7 us of dispatch per launch over ~18k launches -- the loop is
launch-bound, not compute-bound. A CUDA graph records the whole launch sequence
once and replays it with a single submission, which removes the per-launch CPU
dispatch entirely.

What makes it possible here: the drivers are pure stream work. They never
synchronise, never call a blocking memcpy, and never allocate device memory --
every buffer is allocated Python-side (a standing rule of this backend). The one
thing that *did* break capture was a host sync hidden in the constants builder
(``torch.nonzero`` for the C-PML band indices); those constants are now cached,
so nothing on the hot path touches the host.

Constraints (read before using)
-------------------------------
A replay re-runs *exactly* the recorded launches against the *same* device
buffers. Therefore:

* **Geometry, ``nt``, source/receiver layout and the C-PML profile are frozen**
  at capture time. Changing any of them requires a new capture.
* **Model values may change** -- write them in place into the tensors the
  equation already holds (``eq._lam.copy_(...)``, etc.) and replay. This is the
  property FWI needs and it is verified in ``test_sem_graph_runner.py``.
* The wavelet is likewise a captured buffer: ``copy_`` into ``runner.wavelet``.
* **The returned tensors are the static output buffers.** They are overwritten by
  the next replay -- clone them if you need to keep them.
* Capture is **forward-only and no-grad**. Autograd cannot be captured this way;
  gradients still go through the normal path.

Numerics: replay is not bit-identical to a fresh call, because the assembly uses
``atomicAdd`` and the summation order is not reproducible -- but two ordinary
eager calls differ by the same amount. Measured drift is ~1e-12 on a ~2.5e-7
signal (relative ~8e-6), i.e. the atomic-ordering floor. Numerics are NOT a
reason to avoid graphing.

When NOT to turn this on
------------------------
Graphing is opt-in on purpose. On a cascaded multiscale iFWI it proved not to
be a free win:

* **Memory.** A graph pins a private memory pool for its lifetime, and
  ``make_graphed_callables`` additionally keeps static input/output buffers. There
  this cost several GB per band, enough to push a fine band into OOM on
  a V100. This is the single most common way graphing backfires -- check
  headroom before enabling.
* **It can be slower.** In the same study graphing removed a reparameterisation
  stall but *exposed* I/O that the slower compute had been hiding, and the
  iteration got slower overall. Graphs help a dispatch-bound loop; they do nothing
  for an I/O-bound one and can unmask it.
* **Frozen geometry.** A multiscale cascade changes grid, dt and record length per
  band, so every stage needs its own capture (and pays its own capture cost and
  memory pool).
* **Not combined with boundary saving.** ``memory='boundary'`` is a memory lever,
  this is a dispatch lever, and the band buffers would have to be static too.
* **No double backward.** ``SEMGraphGradRunner`` cannot provide Hessian-vector
  products; this backend keeps those on the eager path by design.
* **Capture is brittle.** Any host synchronisation anywhere on the call path --
  ``.item()``, ``.cpu()``, ``nonzero()``, printing a tensor -- breaks capture with
  a bare ``cudaErrorStreamCaptureInvalidated``. Keeping this opt-in means such a
  change breaks only the callers who asked for graphing.

Rough guide:

=========================================  ==========================
situation                                  enable?
=========================================  ==========================
fixed geometry, memory headroom,           yes -- 2.5-2.9x measured
  dispatch-bound
forward modelling / RTM, same geometry     yes
  reused many times
multiscale cascade (per-band grids)        re-capture per stage; check memory
memory already tight (e.g. domain          no
  decomposition across GPUs)
I/O-heavy encoded FWI                      profile first -- may be a net loss
Hessian-vector products needed             no
=========================================  ==========================
"""
from __future__ import annotations

import gc

import torch


class SEMGraphRunner:
    """Capture one :class:`SEMCudaPropagator` forward and replay it.

    Parameters
    ----------
    prop:
        A built ``SEMCudaPropagator``. Its geometry is frozen at capture.
    wavelet:
        ``(nt,)`` source wavelet. A private copy is captured; write new values
        with ``runner.wavelet.copy_(new_wavelet)``.
    memory:
        Passed through to ``prop.forward`` (e.g. ``'boundary'``). Only meaningful
        for gradient runs, so normally left ``None`` here.
    warmup:
        Forward calls on a side stream before capture. CUDA requires the work to
        have run at least once so that lazy module loading and the caching
        allocator have settled; 3 is the usual recommendation.

    Example
    -------
    >>> runner = SEMGraphRunner(prop, wavelet)        # doctest: +SKIP
    >>> for it in range(n_iter):                      # doctest: +SKIP
    ...     eq._lam.copy_(new_lam)                    # mutate the model in place
    ...     out = runner()                            # replay
    ...     seis = out[..., 0].clone()                # clone: buffers are reused
    """

    def __init__(self, prop, wavelet: torch.Tensor, *, memory=None, warmup: int = 3):
        if not torch.cuda.is_available():
            raise RuntimeError("SEMGraphRunner requires CUDA.")
        dev = torch.device("cuda")
        self.prop = prop
        self.memory = memory
        self.wavelet = wavelet.detach().to(device=dev, dtype=torch.float32).clone()

        def _run():
            with torch.no_grad():
                return prop.forward(self.wavelet, memory=memory)

        # Warm up on a side stream: capture must not be the first execution.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(max(1, warmup)):
                _run()
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(self.graph):
                self._out = _run()
        except Exception as exc:                       # pragma: no cover
            raise RuntimeError(
                "CUDA graph capture of the SEM time loop failed. The usual cause "
                "is a host synchronisation on the forward path (anything that "
                "moves a device value to the CPU -- .item(), .cpu(), nonzero(), "
                "an unguarded print). Capture requires the whole call to be pure "
                f"stream work. Original error: {exc}") from exc
        self.graph.replay()                            # fill the output buffers
        torch.cuda.synchronize()

    def __call__(self) -> torch.Tensor:
        """Replay the captured loop and return the (reused) output buffer.

        ``(nt, nrec, ncomp)``, the same contract as ``SEMPropagator.forward``.
        The buffer is STATIC and overwritten by the next replay -- clone it if
        you need to keep a result across calls."""
        self.graph.replay()
        return self._out

    def outputs(self) -> torch.Tensor:
        """The static output buffer, without replaying."""
        return self._out


# --------------------------------------------------------------------------- #
# Graphed GRADIENT path.
#
# SEMGraphRunner above only helps a forward-only workload. The measurement says
# the gradient is where the dispatch cost really lands: forward+backward spends
# 71 % of its wall time outside any kernel, ~43 ms per gradient on the reference
# case. torch.cuda.make_graphed_callables records a callable's forward AND its
# backward node, which is exactly what the autograd path needs.
#
# Measured on the same idle GPU (elastic C-PML, 1120 elements, nt=400):
# eager forward+backward 45.9 ms -> graphed 15.6 ms, i.e. 2.9x, with every
# gradient matching eager to within the run-to-run atomicAdd floor.
# --------------------------------------------------------------------------- #
def _graphable_call(prop, memory=None, batch=None):
    """``(callable, leaf_fn, receiver_field_names)`` for the propagator's C Function.

    ``leaf_fn(eq, wavelet)`` returns the tensors the Function consumes, read from
    the equation passed in. Taking ``eq`` as an argument (rather than closing over
    ``prop.eq``) is what makes an FWI loop work: every iteration REBUILDS the
    equation from the current vp/vs/rho, so ``eq._lam`` is a fresh tensor carrying
    a fresh ``grad_fn``. ``make_graphed_callables`` copies its inputs into static
    buffers, so fresh input tensors each call are fine -- only the C loop is
    graphed, the cheap ``build()`` chain stays eager and autograd threads through
    the graphed node normally.

    ``batch`` (fluid-solid only): a list of same-geometry per-shot propagators
    (``prop`` must be ``batch[0]``). The captured call is then the SHOT-BATCHED
    C loop -- one graph covers every shot, which is what makes graphing the
    coupled path practical (per-shot graphs would pin one memory pool each).
    """
    from .equations.acoustic_elastic2d import AcousticElasticSEM2D
    from .equations.acoustic_potential2d import AcousticPotentialSEM2D
    from .equations.elastic2d_deformed import ElasticSEM2DDeformed
    from .prop_c import _SEMAcousticCpmlC, _SEMElasticCpmlC

    eq = prop.eq
    if getattr(eq, "cpml", None) is None:
        raise NotImplementedError(
            "graphed gradients are wired for the C-PML paths; the bare paths are "
            "cheap enough that the dispatch cost is not the bottleneck.")
    if batch is not None and not isinstance(eq, AcousticElasticSEM2D):
        raise NotImplementedError(
            "batch= is wired for the coupled fluid-solid path; the acoustic "
            "batch already runs unamortised dispatch low enough that graphing "
            "has not been needed there.")

    if isinstance(eq, AcousticElasticSEM2D):
        from .prop_c import _SEMFluidSolidC, _fs_batch_consts
        c = _fs_batch_consts(list(batch) if batch else [prop], memory)
        f32 = lambda t: t.to(device="cuda", dtype=torch.float32)

        def call(air, aim, akp, lam, mu, l2m, eim, wrj, wav):
            return _SEMFluidSolidC.apply(air, aim, akp, lam, mu, l2m,
                                         eim, wrj, wav, c)

        def leaf_fn(e, wav):
            return (f32(e.ac._inv_rho), f32(e.ac._inv_M), f32(e.ac.kappa_node),
                    f32(e.el._lam), f32(e.el._mu), f32(e.el._lam2mu),
                    f32(e.el._inv_M), f32(e.el._w_rho_J), wav)

        return call, leaf_fn, ("chi", "u_x", "u_z")

    if isinstance(eq, ElasticSEM2DDeformed):
        c = prop._elastic_cpml_consts()
        c["bs"] = memory == "boundary"
        f32 = lambda t: t.to(device="cuda", dtype=torch.float32)

        def call(lam, mu, lam2mu, inv_M, w_rho_J, wav):
            return _SEMElasticCpmlC.apply(lam, mu, lam2mu, inv_M, w_rho_J, wav, c)

        def leaf_fn(e, wav):
            return (f32(e._lam), f32(e._mu), f32(e._lam2mu),
                    f32(e._inv_M), f32(e._w_rho_J), wav)

        return call, leaf_fn, ("u_x", "u_z")

    if isinstance(eq, AcousticPotentialSEM2D):
        c = prop._cpml_consts()
        c["bs"] = memory == "boundary"
        f32 = lambda t: t.to(device="cuda", dtype=torch.float32)

        def call(inv_rho, inv_M, kappa, wav):
            return (_SEMAcousticCpmlC.apply(inv_rho, inv_M, kappa, wav, c),)

        def leaf_fn(e, wav):
            return (f32(e._inv_rho), f32(e._inv_M), f32(e.kappa_node), wav)

        return call, leaf_fn, (eq.field_specs[0].name,)

    raise NotImplementedError(
        f"graphed gradients are not wired for {type(eq).__name__}")


class SEMGraphGradRunner:
    """Graph the forward AND backward of the SEM C++ loop, keeping autograd.

    Unlike :class:`SEMGraphRunner` this stays differentiable: call it, build a
    loss, call ``.backward()``, and the gradient reaches vp / vs / rho exactly as
    on the ordinary path -- only the recorded launch sequence is replayed instead
    of re-dispatched.

    Constraints, in addition to those of :class:`SEMGraphRunner`:

    * the model tensors **must require grad** at graph time and stay that way
      (``make_graphed_callables`` bakes in the ``requires_grad`` pattern)
    * **no double backward** -- ``make_graphed_callables`` does not support it.
      That matches this backend, whose Hessian-vector products deliberately stay
      on the eager path.
    * ``memory='boundary'`` IS supported -- pass it to the constructor. Boundary
      saving is the production memory lever, and graphing does not fight it:
      measured peak memory with graphing is 0.98-0.99x of the ungraphed
      boundary-saved run at nt = 300 / 900 / 1800, i.e. the graph adds nothing
      measurable, and boundary saving keeps its full 1.7-1.9x saving over the
      save-all tape. Gradients match the save-all path at cos = 1.0.

    Example
    -------
    >>> runner = SEMGraphGradRunner(prop, memory='boundary')   # doctest: +SKIP
    >>> for it in range(n_iter):                           # doctest: +SKIP
    ...     eq = rebuild_equation(vp, vs, rho)             # vp requires grad
    ...     out = runner(wavelet)
    ...     loss = misfit(out[..., 0], obs)     # (nt, nrec, ncomp)
    ...     loss.backward()                                # reaches vp/vs/rho
    """

    def __init__(self, prop, *, memory=None, warmup: int = 3, batch=None):
        if not torch.cuda.is_available():
            raise RuntimeError("SEMGraphGradRunner requires CUDA.")
        self.prop = prop
        self.memory = memory
        # batch (fluid-solid): list of same-geometry per-shot props, prop ==
        # batch[0]. ONE capture then covers the whole shot batch and __call__
        # returns (B, nt, nrec, ncomp) instead of (nt, nrec, ncomp).
        self.batch = list(batch) if batch is not None else None
        if self.batch is not None and self.batch[0] is not prop:
            raise ValueError("batch[0] must be the prop the runner is built on")
        self._call, self._leaf_fn, self._recv_names = _graphable_call(
            prop, memory=memory, batch=self.batch)

        # Sample args must mirror the real ones in shape, dtype and requires_grad.
        wav0 = torch.zeros(prop.nt, dtype=torch.float32, device="cuda")
        sample = tuple(t.detach().clone().requires_grad_(True)
                       for t in self._leaf_fn(prop.eq, wav0))
        # Drain the caching allocator before capture. Capture requires every
        # allocation on the path to be served from the graph's private pool; a
        # pool left fragmented by an earlier large run makes the backward's
        # first allocation raise cudaErrorStreamCaptureInvalidated instead
        # ("operation failed due to a previous error during capture", surfacing
        # at the grad_wavelet zeros in prop_c). Reproduced deterministically by
        # running a 3-D boundary-saved test immediately before this one; without
        # the drain that pair fails, with it it passes. Costs one drained pool
        # per runner, which is built once.
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        try:
            self._graphed = torch.cuda.make_graphed_callables(
                self._call, sample, num_warmup_iters=max(3, warmup))
        except Exception as exc:                            # pragma: no cover
            raise RuntimeError(
                "make_graphed_callables failed on the SEM C loop. Usual causes: a "
                "host synchronisation on the path (anything moving a device value "
                "to the CPU), or model tensors that do not require grad. Original "
                f"error: {exc}") from exc

    def __call__(self, wavelet: torch.Tensor, eq=None) -> torch.Tensor:
        """Run the graphed forward; the result is differentiable as usual.

        ``eq`` defaults to the propagator's equation. Pass a REBUILT equation
        (same mesh and C-PML profile, new model values) to run an FWI iteration:
        the gradient then flows back to that equation's vp / vs / rho. Rebuilding
        is in fact required if you want to call ``.backward()`` more than once,
        since ``build()`` computes lam/mu from vp/vs/rho eagerly and the first
        backward frees that part of the graph.
        """
        wav = wavelet.to(device="cuda", dtype=torch.float32)
        out = self._graphed(*self._leaf_fn(eq if eq is not None else self.prop.eq, wav))
        # (nrec, nt) per receiver field, in _recv_names order. Honour the
        # propagator's receiver_type: without this the runner returned the
        # equation's own field order and silently ignored the selection.
        at = {nm: i for i, nm in enumerate(self._recv_names)}
        want = getattr(self.prop, "receiver_type", None) or list(self._recv_names)
        missing = [nm for nm in want if nm not in at]
        if missing:
            raise ValueError(
                f"receiver_type {missing} is not available on the graphed path "
                f"for this equation; it produces {list(self._recv_names)}.")
        sel = torch.stack([out[at[nm]] for nm in want], -1)
        if self.batch is not None and len(self.batch) > 1:
            return sel.transpose(1, 2)        # (B, nt, nrec, ncomp)
        return sel.transpose(0, 1)            # (nt, nrec, ncomp)
