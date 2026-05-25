<div class="sweep-qs">

<div class="sweep-qs__eyebrow">GETTING STARTED</div>

<h1 class="sweep-qs__title">Your first FWI
<span class="sweep-hero__gradient">in five steps.</span></h1>

<p class="sweep-qs__lede">
  From a fresh terminal to running gradient descent on a toy velocity model.
  By the end you'll be ready to open
  <a href="../../examples/">any of the notebooks</a>.
</p>

<nav class="sweep-qs__strip">
  <a href="#install">       <span>01</span> Install</a>
  <a href="#model">          <span>02</span> Build a velocity model</a>
  <a href="#forward">        <span>03</span> Run a forward shot</a>
  <a href="#gradient">       <span>04</span> Get gradients</a>
  <a href="#next">           <span>05</span> Try a real notebook</a>
</nav>

<section class="sweep-qs__step" id="install">
  <div class="sweep-qs__hd">
    <span class="sweep-qs__num">01</span>
    <h2>Install</h2>
  </div>

  <p>
    Pick whichever backend you already have. SWEEP supports <em>lazy imports</em> —
    you don't need both PyTorch and JAX.
  </p>

  <div class="sweep-qs__pair">
    <div class="sweep-qs__pill">
      <div class="sweep-qs__pill-hd">
        <span class="sweep-qs__pill-name">PyTorch backend</span>
        <span class="sweep-qs__pill-tag">recommended</span>
      </div>
      <code>$ pip install "sweep[torch] @ git+https://github.com/DeepWave-KAUST/sweep.git"</code>
    </div>
    <div class="sweep-qs__pill">
      <div class="sweep-qs__pill-hd">
        <span class="sweep-qs__pill-name">JAX backend</span>
      </div>
      <code>$ pip install "sweep[jax] @ git+https://github.com/DeepWave-KAUST/sweep.git"</code>
    </div>
  </div>

  <div class="sweep-qs__hint">
    <strong>tip</strong> &nbsp; Not on PyPI yet — install straight from GitHub. For the compiled CUDA backend (<code>impl="c"</code>) or a dev clone, see
    <a href="../installation/">Install from source</a>.
  </div>
</section>

<section class="sweep-qs__step" id="model">
  <div class="sweep-qs__hd">
    <span class="sweep-qs__num">02</span>
    <h2>Build a velocity model</h2>
  </div>

  <p>
    A simple 2-layer model: 1500 m/s in the top half, 2000 m/s below. 10 m grid spacing.
  </p>
</section>

```python title="quickstart.py"
import torch, numpy as np
from sweep.propagator.torch import PropTorch
from sweep.equations       import Acoustic
from sweep.signal          import ricker

# A 2-layer 100×100 model.
vp = np.ones((100, 100), dtype=np.float32) * 1500
vp[50:, :] = 2000
```

<div class="sweep-qs__figure">
  <img src="../../figures/quickstart/vp.png" alt="vp model">
  <div class="sweep-qs__caption">vp · 100 × 100 · 10 m grid</div>
</div>

<section class="sweep-qs__step" id="forward">
  <div class="sweep-qs__hd">
    <span class="sweep-qs__num">03</span>
    <h2>Run a forward shot</h2>
  </div>

  <p>
    Construct the equation, wrap it in a propagator, and call <code>forward()</code>.
  </p>
</section>

```python title="propagator"
eq   = Acoustic(spatial_order=8, device="cuda", backend="torch")
prop = PropTorch(eq, shape=(100, 100), dev="cuda", dh=10., dt=2e-3,
                 source_type=["h1"], receiver_type=["h1"], pml_type="cpmlr")

wave      = ricker(np.arange(0, 1.5, 2e-3) - 0.1, f=8)
sources   = np.array([[50, 2]])
receivers = np.array([[[ix, 2] for ix in range(10, 90)]])

obs = prop.forward(wave, sources, receivers, models=[torch.from_numpy(vp).to("cuda")])
```

<div class="sweep-qs__figure">
  <img src="../../figures/quickstart/record.png" alt="shot record">
  <div class="sweep-qs__caption">shot record · 1 src · 1 rec · 750 samples</div>
</div>

<section class="sweep-qs__step" id="gradient">
  <div class="sweep-qs__hd">
    <span class="sweep-qs__num">04</span>
    <h2>Get the gradient</h2>
  </div>

  <p>
    Tag <code>vp</code> with <code>requires_grad_</code>, run forward, and <code>backward()</code>. That's it.
  </p>
</section>

```python
vp_t = torch.from_numpy(vp).to("cuda").requires_grad_(True)
obs  = prop.forward(wave, sources, receivers, models=[vp_t])
obs.pow(2).sum().backward()

# vp_t.grad now holds the sensitivity kernel.
```

<div class="sweep-qs__figure">
  <img src="../../figures/quickstart/gradient.png" alt="vp gradient">
  <div class="sweep-qs__caption">∂L/∂vp · 100 × 100</div>
</div>

<section class="sweep-qs__step" id="next">
  <div class="sweep-qs__hd">
    <span class="sweep-qs__num">05</span>
    <h2>Where to go next</h2>
  </div>

  <div class="sweep-qs__cards">

    <a class="sweep-qs__card" href="../../notebooks/01_fwi_acoustic_marmousi/">
      <img src="../../figures/gallery/01_fwi_acoustic_marmousi.png" alt="Marmousi FWI">
      <div class="sweep-qs__card-body">
        <div class="sweep-qs__card-tag">NOTEBOOK</div>
        <div class="sweep-qs__card-name">Marmousi FWI</div>
        <div class="sweep-qs__card-desc">A real benchmark · 17 min on A100</div>
      </div>
    </a>

    <a class="sweep-qs__card" href="../../notebooks/12_multi_gpu/">
      <img src="../../figures/gallery/12_multi_gpu_ddp.png"
           onerror="this.src='../../assets/logo/sweep-icon-256.png'" alt="Multi-GPU DDP">
      <div class="sweep-qs__card-body">
        <div class="sweep-qs__card-tag">NOTEBOOK</div>
        <div class="sweep-qs__card-name">Multi-GPU DDP</div>
        <div class="sweep-qs__card-desc">Scales to 8× A100</div>
      </div>
    </a>

    <a class="sweep-qs__card" href="../../notebooks/10_wavefield_elastic_tti/">
      <img src="../../figures/gallery/10_wavefield_elastic_tti.png" alt="Anisotropic TTI">
      <div class="sweep-qs__card-body">
        <div class="sweep-qs__card-tag">DOCS</div>
        <div class="sweep-qs__card-name">Anisotropic TTI</div>
        <div class="sweep-qs__card-desc">When acoustic isn't enough</div>
      </div>
    </a>

  </div>
</section>

</div>
