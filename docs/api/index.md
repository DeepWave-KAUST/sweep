<div class="sweep-api">

<div class="sweep-api__eyebrow">API REFERENCE · V0.0.1</div>

<h1 class="sweep-api__title">The whole
<span class="sweep-hero__gradient">surface area.</span></h1>

<p class="sweep-api__lede">
  Every public symbol in <code>sweep</code>. Click any equation to see its
  constructor, fields, and example usage. All pages render directly from
  Python docstrings via <code>mkdocstrings</code>.
</p>

<div class="sweep-api__pills">
  <span class="sweep-api__pill"><span class="sweep-api__pill-dot teal"></span>33 equation classes</span>
  <span class="sweep-api__pill">2 propagators</span>
  <span class="sweep-api__pill">PML + sponge boundaries</span>
  <span class="sweep-api__pill">torch · jax</span>
</div>

<section class="sweep-api__sec">
  <div class="sweep-api__sec-hd">
    <span class="sweep-api__sec-dot teal"></span>
    <h2>Acoustic</h2>
    <span class="sweep-api__count">6 classes</span>
  </div>
  <p class="sweep-api__sec-lede">Constant-density acoustic wave equation. Most-used family; default for FWI.</p>
  <div class="sweep-api__grid">
    <a class="sweep-api__card" href="equations/acoustic/">
      <div class="sweep-api__card-hd"><code>Acoustic</code><div class="sweep-api__card-tags"><span>2D</span><span>2nd-order</span></div></div>
      <p>Default. Second-order in time, PML boundaries.</p>
      <div class="sweep-api__card-models">models: <span class="sweep-api__chip teal">vp</span></div>
    </a>
    <a class="sweep-api__card" href="equations/acoustic1st/">
      <div class="sweep-api__card-hd"><code>Acoustic1st</code><div class="sweep-api__card-tags"><span>2D</span><span>1st-order</span></div></div>
      <p>First-order velocity-stress formulation.</p>
      <div class="sweep-api__card-models">models: <span class="sweep-api__chip teal">vp</span> <span class="sweep-api__chip teal">rho</span></div>
    </a>
    <a class="sweep-api__card" href="equations/acoustic3d/">
      <div class="sweep-api__card-hd"><code>Acoustic3D</code><div class="sweep-api__card-tags"><span>3D</span><span>2nd-order</span></div></div>
      <p>3-D extension. Larger memory footprint.</p>
      <div class="sweep-api__card-models">models: <span class="sweep-api__chip teal">vp</span></div>
    </a>
    <a class="sweep-api__card" href="equations/acoustic_vrz/">
      <div class="sweep-api__card-hd"><code>AcousticVRZ</code><div class="sweep-api__card-tags"><span>2D</span><span>2nd-order</span></div></div>
      <p>Variable density via vertical impedance perturbation.</p>
      <div class="sweep-api__card-models">models: <span class="sweep-api__chip teal">vp</span> <span class="sweep-api__chip teal">z</span></div>
    </a>
    <a class="sweep-api__card" href="equations/acoustic_lsrtm/">
      <div class="sweep-api__card-hd"><code>AcousticLSRTM</code><div class="sweep-api__card-tags"><span>2D</span><span>Born</span></div></div>
      <p>LSRTM-aware acoustic; <code>mp</code> reflectivity perturbation field.</p>
      <div class="sweep-api__card-models">models: <span class="sweep-api__chip teal">vp</span> <span class="sweep-api__chip teal">mp</span></div>
    </a>
    <a class="sweep-api__card" href="equations/acoustic_lsrtm3d/">
      <div class="sweep-api__card-hd"><code>AcousticLSRTM3D</code><div class="sweep-api__card-tags"><span>3D</span><span>Born</span></div></div>
      <p>3-D extension of LSRTM-aware acoustic.</p>
      <div class="sweep-api__card-models">models: <span class="sweep-api__chip teal">vp</span> <span class="sweep-api__chip teal">mp</span></div>
    </a>
  </div>
</section>

<section class="sweep-api__sec">
  <div class="sweep-api__sec-hd">
    <span class="sweep-api__sec-dot orange"></span>
    <h2>Elastic</h2>
    <span class="sweep-api__count">2 classes</span>
  </div>
  <p class="sweep-api__sec-lede">Particle velocities and stresses. Required for converted-wave physics.</p>
  <div class="sweep-api__grid">
    <a class="sweep-api__card" href="equations/elastic/">
      <div class="sweep-api__card-hd"><code>Elastic</code><div class="sweep-api__card-tags"><span>2D</span><span>1st-order</span></div></div>
      <p>Staggered-grid velocity-stress.</p>
      <div class="sweep-api__card-models">models: <span class="sweep-api__chip orange">vp</span> <span class="sweep-api__chip orange">vs</span> <span class="sweep-api__chip orange">rho</span></div>
    </a>
    <a class="sweep-api__card" href="equations/elastic3d/">
      <div class="sweep-api__card-hd"><code>Elastic3D</code><div class="sweep-api__card-tags"><span>3D</span><span>1st-order</span></div></div>
      <p>3-D staggered grid.</p>
      <div class="sweep-api__card-models">models: <span class="sweep-api__chip orange">vp</span> <span class="sweep-api__chip orange">vs</span> <span class="sweep-api__chip orange">rho</span></div>
    </a>
  </div>
</section>

<section class="sweep-api__sec">
  <div class="sweep-api__sec-hd">
    <span class="sweep-api__sec-dot amber"></span>
    <h2>Anisotropic</h2>
    <span class="sweep-api__count">2 classes</span>
  </div>
  <p class="sweep-api__sec-lede">TTI elastic media with rotated symmetry axis. Required when tilted shales / layering matter.</p>
  <div class="sweep-api__grid">
    <a class="sweep-api__card" href="equations/elastic_tti/">
      <div class="sweep-api__card-hd"><code>ElasticTTI</code><div class="sweep-api__card-tags"><span>2D</span><span>rotated staggered</span></div></div>
      <p>Rotated staggered-grid TTI elastic.</p>
      <div class="sweep-api__card-models">models: <span class="sweep-api__chip amber">vp0</span> <span class="sweep-api__chip amber">vs0</span> <span class="sweep-api__chip amber">rho</span> <span class="sweep-api__chip amber">ε</span> <span class="sweep-api__chip amber">δ</span> <span class="sweep-api__chip amber">γ</span> <span class="sweep-api__chip amber">θ</span> <span class="sweep-api__chip amber">φ</span></div>
    </a>
    <a class="sweep-api__card" href="equations/elastic_tti_sg/">
      <div class="sweep-api__card-hd"><code>ElasticTTISG</code><div class="sweep-api__card-tags"><span>2D</span><span>staggered</span></div></div>
      <p>Axis-aligned staggered-grid TTI elastic.</p>
      <div class="sweep-api__card-models">models: <span class="sweep-api__chip amber">vp0</span> <span class="sweep-api__chip amber">vs0</span> <span class="sweep-api__chip amber">rho</span> <span class="sweep-api__chip amber">ε</span> <span class="sweep-api__chip amber">δ</span> <span class="sweep-api__chip amber">γ</span> <span class="sweep-api__chip amber">θ</span> <span class="sweep-api__chip amber">φ</span></div>
    </a>
  </div>
</section>

<section class="sweep-api__sec">
  <div class="sweep-api__sec-hd">
    <span class="sweep-api__sec-dot lime"></span>
    <h2>DAS family</h2>
    <span class="sweep-api__count">8 classes</span>
  </div>
  <p class="sweep-api__sec-lede">Distributed-acoustic-sensing strain / strain-rate operators on top of the elastic wavefield. Zhao, Mu, Elastic, and Modeler formulations; 2D + 3D.</p>
  <div class="sweep-api__grid">
    <a class="sweep-api__card" href="equations/das/">
      <div class="sweep-api__card-hd"><code>DAS · DASZhao · DASMu · DASElastic</code><div class="sweep-api__card-tags"><span>2D + 3D</span><span>strain</span></div></div>
      <p>All four DAS formulations on one page — same source field, different strain output.</p>
      <div class="sweep-api__card-models">models: <span class="sweep-api__chip lime">vp</span> <span class="sweep-api__chip lime">vs</span> <span class="sweep-api__chip lime">rho</span></div>
    </a>
  </div>
</section>

<section class="sweep-api__sec">
  <div class="sweep-api__sec-hd">
    <span class="sweep-api__sec-dot accent"></span>
    <h2>Propagators</h2>
    <span class="sweep-api__count">2 classes</span>
  </div>
  <p class="sweep-api__sec-lede">Wrap any equation into a callable solver. Equation-agnostic; same API for every physics family above.</p>
  <div class="sweep-api__grid">
    <a class="sweep-api__card" href="propagators/prop_torch/">
      <div class="sweep-api__card-hd"><code>PropTorch</code><div class="sweep-api__card-tags"><span>torch</span><span>eager + c</span></div></div>
      <p>PyTorch backend. Eager + compiled CUDA paths.</p>
      <div class="sweep-api__card-models">options: <span class="sweep-api__chip accent">EagerOptions</span> <span class="sweep-api__chip accent">CUDAOptions</span></div>
    </a>
    <a class="sweep-api__card" href="propagators/prop_jax/">
      <div class="sweep-api__card-hd"><code>PropJax</code><div class="sweep-api__card-tags"><span>jax</span><span>jit</span></div></div>
      <p>JAX backend. Functional, jit'd, pmap-friendly.</p>
      <div class="sweep-api__card-models">options: <span class="sweep-api__chip accent">JaxOptions</span></div>
    </a>
  </div>
</section>

<section class="sweep-api__sec">
  <div class="sweep-api__sec-hd">
    <span class="sweep-api__sec-dot accent"></span>
    <h2>Parallel</h2>
    <span class="sweep-api__count">2 classes</span>
  </div>
  <p class="sweep-api__sec-lede">Split one model across GPUs — a tile per rank, a halo exchange per step. Shot-parallel DDP needs nothing from here.</p>
  <div class="sweep-api__grid">
    <a class="sweep-api__card" href="parallel/">
      <div class="sweep-api__card-hd"><code>ModelParallel</code><div class="sweep-api__card-tags"><span>torch</span><span>nccl</span></div></div>
      <p>Wraps a PropTorch; tiles, halos and gradients are automatic.</p>
      <div class="sweep-api__card-models">helpers: <span class="sweep-api__chip accent">pad_to_mesh</span> <span class="sweep-api__chip accent">gather_record</span></div>
    </a>
    <a class="sweep-api__card" href="parallel/">
      <div class="sweep-api__card-hd"><code>MeshTopology</code><div class="sweep-api__card-tags"><span>py × px</span><span>shot groups</span></div></div>
      <p>The rank grid: tile coordinates, neighbours, shot groups.</p>
      <div class="sweep-api__card-models">guide: <span class="sweep-api__chip accent">Domain decomposition</span></div>
    </a>
  </div>
</section>

</div>
