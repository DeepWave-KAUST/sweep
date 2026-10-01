# Installation

## From PyPI (recommended)

sweep works with any PyTorch version, but **the PyTorch build has to match your NVIDIA
driver**. A plain `pip install sweepx` pulls PyPI's default torch, which on Linux is
built for CUDA 13 and needs driver >= 580. With an older driver torch cannot start
CUDA (`UserWarning: ... The NVIDIA driver on your system is too old`) and
`impl='c'` reports no visible GPU. Install the matching torch first, then sweep:

<div class="sweep-install" id="sweep-install">
  <div class="sweep-install__row" data-key="pkg">
    <div class="sweep-install__label">Package</div>
    <div class="sweep-install__opts">
      <button data-v="sweepx">sweepx (solver + agent)</button>
      <button data-v="sweep-solver">sweep-solver (solver only)</button>
    </div>
  </div>
  <div class="sweep-install__row" data-key="tool">
    <div class="sweep-install__label">Installer</div>
    <div class="sweep-install__opts">
      <button data-v="pip">pip</button>
      <button data-v="uv">uv</button>
    </div>
  </div>
  <div class="sweep-install__row" data-key="gpu">
    <div class="sweep-install__label">Your GPU</div>
    <div class="sweep-install__opts">
      <button data-v="volta">V100</button>
      <button data-v="std">T4 · A100 · H100 · RTX 20–40</button>
      <button data-v="bw">B200 · RTX 50 (Blackwell)</button>
      <button data-v="none">No NVIDIA GPU</button>
    </div>
  </div>
  <div class="sweep-install__row" data-key="cuda">
    <div class="sweep-install__label">Compute Platform</div>
    <div class="sweep-install__opts">
      <button data-v="auto" data-only-tool="uv">Auto</button>
      <button data-v="cu121">CUDA 12.1</button>
      <button data-v="cu124">CUDA 12.4</button>
      <button data-v="cu126">CUDA 12.6</button>
      <button data-v="cu128">CUDA 12.8</button>
      <button data-v="cu129">CUDA 12.9</button>
      <button data-v="cu130">CUDA 13.0</button>
      <button data-v="cu132">CUDA 13.2</button>
      <button data-v="cpu">CPU</button>
    </div>
  </div>
  <div class="sweep-install__row sweep-install__row--cmd">
    <div class="sweep-install__label">Run this Command</div>
    <div class="sweep-install__cmdbox">
      <pre class="sweep-install__cmd"><code id="sweep-install-cmd"></code></pre>
    </div>
  </div>
  <div class="sweep-install__note" id="sweep-install-note"></div>
</div>

<style>
.sweep-install { margin: 1.2em 0 1.6em; font-size: .78rem; }
.sweep-install__row { display: flex; gap: .4em; margin-bottom: .4em; }
.sweep-install__label { flex: 0 0 9.5em; padding: .55em .7em; border-left: 3px solid var(--md-default-fg-color--lightest); color: var(--md-default-fg-color--light); display: flex; align-items: center; }
.sweep-install__opts { flex: 1; display: flex; gap: .4em; flex-wrap: wrap; }
.sweep-install__opts button { flex: 1 1 auto; padding: .55em .7em; border: 0; border-radius: 2px; cursor: pointer; font: inherit; background: var(--md-code-bg-color); color: var(--md-default-fg-color); }
.sweep-install__opts button:hover { outline: 1px solid var(--md-accent-fg-color); }
.sweep-install__opts button.is-on { background: var(--md-accent-fg-color); color: var(--md-accent-bg-color); }
.sweep-install__opts button:disabled { opacity: .35; cursor: not-allowed; outline: 0; }
.sweep-install__opts button[hidden] { display: none; }
.sweep-install__row--cmd .sweep-install__label { border-left-color: var(--md-default-fg-color); }
.sweep-install__cmdbox { flex: 1; position: relative; min-width: 0; }
.md-typeset .sweep-install__cmd { margin: 0; }
.sweep-install__cmd code { white-space: pre-wrap; overflow-wrap: anywhere; }
.sweep-install__note { padding: .3em .7em 0 calc(9.5em + .4em + 3px); color: var(--md-default-fg-color--light); line-height: 1.55; }
@media (max-width: 44em) {
  .sweep-install__row { flex-direction: column; }
  .sweep-install__label { flex: none; }
  .sweep-install__note { padding-left: .7em; }
}
</style>

<script>
(function () {
  var root = document.getElementById("sweep-install");
  if (!root) return;
  var state = { pkg: "sweepx", tool: "pip", gpu: "std", cuda: "cu126" };
  // Which compute platforms a card can use (torch.cuda.get_arch_list()):
  // cu121, cu124 and cu126 carry sm_70; cu128 (torch 2.11, what pip picks there),
  // cu129 and cu13x do not (nor does sweep's cu13 core). cu121, cu124 and cu126
  // carry no Blackwell kernels.
  var allowed = {
    volta: ["cu126", "cu121", "cu124"],  // first = default
    std:   ["auto", "cu126", "cu121", "cu124", "cu128", "cu129", "cu130", "cu132"],
    bw:    ["auto", "cu130", "cu132", "cu128", "cu129"],  // first = default
    none:  ["cpu"]
  };
  var notes = {
    cu121: "Driver >= 525. This index stops at <strong>torch 2.5</strong>; pip picks 2.5.1. " +
           "sweep loads its prebuilt <code>cu12</code> core &mdash; nothing compiles.",
    cu124: "Driver >= 525. This index stops at <strong>torch 2.6</strong>; pip picks 2.6.0. " +
           "sweep loads its prebuilt <code>cu12</code> core &mdash; nothing compiles.",
    cu126: "Driver >= 525 (<code>nvidia-smi</code> shows <em>CUDA Version: 12.x</em> or newer). " +
           "sweep loads its prebuilt <code>cu12</code> core &mdash; nothing compiles.",
    cu128: "Driver >= 525 (>= 570 on Blackwell). This index stops at <strong>torch 2.11</strong>, " +
           "which has no V100 kernels. sweep loads its prebuilt <code>cu12</code> core &mdash; nothing compiles.",
    cu129: "Driver >= 525 (>= 570 on Blackwell). This index stops at <strong>torch 2.13</strong>, " +
           "and has no V100 kernels. sweep loads its prebuilt <code>cu12</code> core &mdash; nothing compiles.",
    cu130: "Driver >= 580 (<code>nvidia-smi</code> shows <em>CUDA Version: 13.0</em> or newer). " +
           "sweep loads its prebuilt <code>cu13</code> core &mdash; nothing compiles.",
    cu132: "Driver >= 580 (<code>nvidia-smi</code> shows <em>CUDA Version: 13.x</em>). " +
           "sweep loads its prebuilt <code>cu13</code> core (built with CUDA 13.0; tested with torch cu130).",
    auto:  "uv reads your driver and picks the matching torch build; sweep then loads the " +
           "<code>cu12</code> or <code>cu13</code> core to match. Needs a recent uv (<code>pip install -U uv</code>).",
    cpu:   "No compiled backend: use <code>impl='eager'</code> (pure PyTorch) or the JAX backend. " +
           "On macOS a plain <code>pip install torch</code> is enough."
  };
  function pick(key, v) {
    state[key] = v;
    if (state.tool !== "uv" && state.cuda === "auto") state.cuda = "cu126";
    var ok = allowed[state.gpu].filter(function (c) { return c !== "auto" || state.tool === "uv"; });
    if (ok.indexOf(state.cuda) < 0) state.cuda = ok[0];
    render();
  }
  function command() {
    var p = state.pkg, c = state.cuda;
    if (state.tool === "uv") {
      return "uv pip install " + p + " --torch-backend=" + c;
    }
    return "pip install torch --index-url https://download.pytorch.org/whl/" + c + "\n" +
           "pip install " + p;
  }
  function render() {
    var ok = allowed[state.gpu];
    root.querySelectorAll(".sweep-install__row[data-key]").forEach(function (row) {
      var key = row.getAttribute("data-key");
      row.querySelectorAll("button").forEach(function (b) {
        var v = b.getAttribute("data-v");
        b.classList.toggle("is-on", state[key] === v);
        if (key === "cuda") {
          b.hidden = b.getAttribute("data-only-tool") === "uv" && state.tool !== "uv";
          b.disabled = ok.indexOf(v) < 0;
        }
      });
    });
    document.getElementById("sweep-install-cmd").textContent = command();
    document.getElementById("sweep-install-note").innerHTML = notes[state.cuda];
  }
  root.addEventListener("click", function (e) {
    var b = e.target.closest("button");
    if (!b || b.disabled) return;
    var row = b.closest(".sweep-install__row[data-key]");
    if (row) pick(row.getAttribute("data-key"), b.getAttribute("data-v"));
  });
  render();
})();
</script>

- **Which CUDA?** `nvidia-smi` shows the newest *CUDA Version* your driver supports;
  pick that or lower. Combinations your GPU cannot run are greyed out.
- **Older CUDA 12 builds come with an older torch** (cu121: 2.5, cu124: 2.6, cu128:
  2.11, cu129: 2.13). sweep runs on all of them. CUDA 11 supports `impl='eager'` only.
- **Wrong torch already installed?** Reinstall just torch with the command above plus
  `--force-reinstall`; sweep stays as is.

!!! note
    `sweepx` (Python >= 3.10) installs the solver plus the `sweep-agent` companion;
    `pip install sweep-solver` installs the solver alone (Python >= 3.9). Either way
    you `import sweep`: the bare name `sweep` is taken on PyPI.

## From source

Install torch first, as above, then:

```bash
git clone https://github.com/DeepWave-KAUST/sweep
cd sweep
pip install .
python -m sweep.build   # optional: build the CUDA core now, not on the first impl='c' call
```

A clone carries no prebuilt core: the CUDA core is compiled once for your card
(2–5 min, needs an nvcc of your torch's CUDA major) and cached.

## Verification

Run a small forward model:

```python
import numpy as np, torch, sweep
from sweep.equations import Acoustic
from sweep.propagator.torch import PropTorch
from sweep.signal import ricker

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
solver = PropTorch(Acoustic(device=dev), shape=(64, 64), dh=10.0, dt=1e-3, dev=dev)
wavelet = ricker(np.arange(500) * 1e-3 - 0.1, f=15.0).astype(np.float32)
gather = solver(wavelet, np.array([[32, 4]]), np.array([[[ix, 4] for ix in range(64)]]),
                models=[torch.full((64, 64), 2000.0, device=dev)])

print("sweep", sweep.__version__, "| torch", torch.__version__)
print("device:", dev, "| backend:", solver.impl)
print("gather:", tuple(gather.shape), "| finite:", bool(torch.isfinite(gather).all()))
```

On a GPU the output should look like:

```
sweep 0.3.1 | torch 2.13.0+cu129
device: cuda | backend: c
gather: (1, 500, 64, 1) | finite: True
```

`backend: c` means the compiled CUDA backend is working. If you see `device: cpu` on
a GPU machine, torch cannot use your GPU: reinstall it with the selector above. If you
see `device: cuda | backend: eager`, run `python -c "import sweep; sweep.precompile()"`:
its error says why `impl='c'` is unavailable. Without a GPU, `backend: eager` is expected. With
g++ older than 10, `torch.compile` cannot build the CPU step: sweep warns and runs it uncompiled
(slower, same result).

How the core is picked, when nvcc is needed, building without a GPU and developer
builds: [Building the CUDA core](../dev/building.md).
