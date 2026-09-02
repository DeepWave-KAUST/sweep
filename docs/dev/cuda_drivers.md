# CUDA driver 骨架（`eq_driver.cuh` / `sg_driver.cuh`）

面向要读懂或扩展 `impl="c"` 时间循环的开发者。用户侧的"加一个方程"流程见
[Extending](../user-guide/extending.md)；本页只讲 `src/sweep/csrc/cuda/` 里
driver 层的结构、钩子时序和验证门禁。`gate/DRIVER_TEMPLATE_DESIGN.md` 是模板化之前的
早期设计草图，已由本页取代。

## 1. 架构总览

**两个骨架，一份 traits，一层薄入口。** 过去每个方程目录手抄 ~300 行 forward driver 和
~1000 行四模式 backward driver，60–80% 逐行相同；stepped 区间、phase-split、
boundary-tail 截断等横切能力只存在于碰巧实现了它们的拷贝里。现在：

| 层 | 文件 | 内容 |
|---|---|---|
| 骨架（声学家族） | `common/eq_driver.cuh` | `template <class Eq>`：`generic_forward` / `generic_backward`（full）/ `generic_backward_bs` / `generic_backward_ckpt` / `generic_backward_recursive_ckpt`，以及 `GenericForwardRunner` / `GenericBackwardBsRunner`。二阶位移形：`u_prev/u_now/u_next` 缓冲轮转，源/检波器单场，BS 存单场，backward 算 `grad_wavelet` 与 illumination。成员：acoustic2d、acoustic3d、acoustic_vrz2d。 |
| 骨架（staggered 家族） | `common/sg_driver.cuh` | `sg_generic_*` 五个入口 + `SgForwardRunner` / `SgBackwardBsRunner`。一阶速度–应力形：场原地更新（无轮转），每步 = velocity 子步 + stress 子步，源/检波器按场索引循环，BS 每步存一个场列表，`last_two` 是末态场快照，backward 无 `grad_wavelet` 无 illumination。成员：elastic2d/3d、das_mu2d/3d、elastic_tti_sg2d/3d、elastic_vr2d。 |
| 每方程 traits | `equations/<eq>/driver_traits.cuh` | 一个 `struct Driver`：常量 + 类型别名 + 全 static 的复合启动钩子，按 [1] identity / [2] forward / [3] full / [4] bs / [5] ckpt+recursive 五节排列，节内按骨架调用序 —— 从上往下读 traits ≈ 读一遍执行流程。 |
| 薄入口 | `equations/<eq>/forward.cu`、`backward.cu` | 每个入口一行：`return eqdrv::generic_forward<Driver>(in);` 等，外加 `forward_runner` / `backward_bs_runner` 两个工厂。例外：elastic2d/3d 的 APM 入口（`apm_forward` / `apm_backward*`）仍手写在同一文件里；acoustic_vrz2d 的 chunk/recursive ckpt backward 是线性段式扫描而非声学二分，保留手写。 |

`eq_driver.cuh` 是 acoustic2d 手写 driver 的逐行转写，`sg_driver.cuh` 是 elastic2d 的；物理
kernel 一行未动。钩子粒度刻意**粗**（每步一个复合操作，不是每个 kernel 一个钩子）：家族成员在
步内**顺序**上互不相同（2-D 声学在源注入+swap 后成像，3-D 在注入前，VRZ 在 restore 前注入），
顺序位级承重，所以差异只进方程钩子，骨架里没有 per-equation 分支；骨架只拥有真正相同的部分
（输入校验、stepped/phase 记账、缓冲绑定与 legacy 回退分配、Boundary/Checkpoint runtime 编排、
时间循环、输出打包）。仍手写的方程：das2d/3d（导数缓冲形，第三种形状）、elastic_tti_2nd2d、
acoustic_lsrtm2d/3d、acoustic_vrz3d、acoustic_vti_1st_2d/3d。

### 持久 runner

`GenericForwardRunner` / `GenericBackwardBsRunner`（声学）和 `SgForwardRunner` /
`SgBackwardBsRunner`（staggered）实现 `shared/wavetypes.h` 的 `IForwardRunner` /
`IBackwardRunner`：**构造函数 = 整个序言只跑一次**（校验、绑定、`SolverContext`、CPML、
boundary saver/runtime、checkpoint runtime、`State`、workspace；声明顺序即构造顺序，析构逆序，
与手写函数的栈退出一致），`run(it_begin, it_end, step_phase)` / `run(bw_it_begin, bw_it_end,
step_phase)` **只剩时间循环**。首段专属动作（`seed_reconstruction` / `prep_adjoint_bs` /
`seed_recon` / 初始 prefetch）留在 `run()` 里按"该次区间是否首段"门控，与手写版一致。

单体入口就是"构造 + 一次 `run`"（`generic_forward<Eq>` ≡
`GenericForwardRunner<Eq>(in).run(in.it_begin, in.it_end, in.step_phase)`），所以位级门禁锤的
正是 runner 路径本身。复用契约（第二次 `run` 由 C++ 强制检查）：gpu-direct boundary 存储且无
checkpoint —— 只有这两种模式的跨调用状态完全活在 Python 绑定的缓冲里。动机：DD 每步重付
~1–2 ms host 序言（launch 地板的 30–100 倍），CUDA graph 治不了 host 逻辑；runner 化后 elastic
2-D DD 端到端 1.8–4.4×、Elastic3D ~1.5×、acoustic 1.05–1.8×（`PROGRESS.md`）。`module.cpp` 以
`py::class_` 绑 `ForwardRunner.run` / `BackwardRunner.run`，工厂按 `{C_NAME}_forward_runner` /
`{C_NAME}_backward_bs_runner` 导出；Python 侧 `WaveEquation._compiled_runner_factories()` 按同一
约定解析，缺失则回退逐调用 stepped 路径。

### stepped 区间与 DD

* **forward**：推进 `[it_begin, it_end)`（`it_end < 0` → `nt`）。`it_begin > 0` 的续段必须绑定
  Python 侧 `wavefields`、`record_out`、（`save_all_wavefields` 时）`u_allt_out`、
  （BS 时）`boundary_gpu` —— 否则内部 `allocate()` 会静默清零传播态。`save_last_state` 只在
  `it_end == nt` 的末段执行。声学 boundary-tail 截断（`boundary_tail_steps = K`）用全局 `it`
  做 `bs_it0` 平移，与分段透明组合。
* **backward**：反向从 `bw_it_begin`（开区间高端，`< 0` → `nt`）到 `bw_it_end`（闭区间低端）。
  stepped 时要求绑定 `adjoint_wavefields`（`ADJ_WF_COUNT` 个）、`grads_out`、`illum_out`，
  BS 模式再加 `forward_wavefields`（`RECON_WF_COUNT` 个重建列表）。DD（`cut_face_mask != 0`）
  **只支持 `backward_bs`**：full 路径 `set_cut_mask(0)`，ckpt 两模式拒绝 stepped/phase/cut。
  声学 DD 支持 gpu-direct 或 cpu 存储（disk 不支持）；staggered DD 只支持 gpu-direct。
* Python 侧 `equations/cuda_layout.py` 的 `stepped=True` 声明"forward 与 backward_bs 都遵守
  区间"（模板化后的方程才可置位 —— 不遵守的方程不会报错，而是每次 stepped 调用跑整条记录、
  返回全零），`dd_backward_phases=True` 声明 backward 实现了编号相位。`ModelParallel` 据此
  准入；调度本身在 `parallel/dd_spec.py` 里声明，`parallel/dd_propagator.py` 解释执行。

**两种不同的 phase split**（`step_phase`）：

| | 声学家族（`eq_driver`） | staggered 家族（`sg_driver`） |
|---|---|---|
| forward | **空间条带切分**。phase 1 = 仅切面相邻的 M 宽物理边缘条带（`cut_face_mask` bit0/bit1 = x_lo/x_hi，v1 只支持 x 切面），不做 BS/源/检波/swap/ckpt；phase 2 = 严格补集 + 整个尾巴（同一格点不能跑两次：CPML psi 双缓冲写会被推进两次）。要求 `it_end == it_begin + 1`、`cut_face_mask != 0`、tile 宽 ≥ 2M。用途：halo 交换与 phase 2 计算重叠（`ACOUSTIC_FWD_OVERLAP`）。 | **物理切分**。phase 1 = 全网格 velocity 子步；phase 2 = 全网格 stress 子步 + 源/ckpt/BS/检波尾巴。DD 在两相之间交换 v、phase 2 后交换 s，使切面相邻应力列读到交换来的（而非本地重算的）速度。无 cut 前置条件，`world_size == 1` 也合法。 |
| backward | **无相位**：`check_stepped_backward` 对 `step_phase != 0` 响亮拒绝。每反向步一次调用，之后交换 `(lambda, recon u)`；floor = 0（`it == 0` 的 adjoint-only 尾巴仍贡献 `grad_wavelet`，`HAS_BS_T0_TAIL`）。 | **bs 相位 3 → 1 → 2**（`sg_check_stepped_backward` 允许 0–3，phased 只限 backward_bs 且单步段）。3 = 只做注入（`undo_body_force` / `inject_residuals` / `uninject_forward_source`，首段的 `seed_recon` 也属于这一相）；1 = `bs_phase1`（应力 NOPML 重建 + 条带 restore + 成像 + receiver-rho + 应力伴随半步）；2 = `bs_phase2`（速度伴随半步 + 载体捕获 + 速度 NOPML 重建 + restore + prefetch）。单体 `step_phase = 0` 在循环头做注入，执行的算子序列与分相完全相同。floor = 1。 |

VRZ：3-D 兄弟的 backward 另有 4 相耦合交换（梯度是耦合场的散度，切缝处需要邻居值），仍手写；
acoustic_vrz2d 的 kernel 不 ranged（`launch_step_range` 对子区间响亮拒绝）、backward 无相位，
所以它是 stepped 但被 DD 拒绝。

**`cut_face_mask`**：`SolverContext::set_cut_mask` 的位定义 bit0..5 = x_lo, x_hi, z_lo, z_hi,
y_lo, y_hi；每方程用 `CUT_MASK_BITS` 限定合法位（2-D 0xF、acoustic3d 0x3F、elastic3d 0x33
仅 x/y、不支持 DD 的 0x0）。置位后 `phys_x0()/phys_x1()` 等物理边界在切面侧变成 stencil halo
（M）而非 pad+M，影响：BS 条带 restore 跳过切面、seed 时的 rim 清零、NOPML 排除带、fused
adjoint 的 `pure_interior` 判据、adjoint prepare 核的 `in_pml` 谓词，以及 band/strip 核的
`wxl/wxh/wzl/wzh`（切面侧为 0）。注意同一 `SolverContext` 里 free-surface 位掩码轴序相反
（bit0 = z_lo），`test_cut_face_mask.py` 钉死这一差异。

## 2. HOOK TIMING MAP（逐字摘自两个骨架文件头）

`common/eq_driver.cuh`：

```text
// ---------------------------------------------------------------------------
// HOOK TIMING MAP — read this before any equation's driver_traits.cuh.
// Per entry point, the traits hooks fire in exactly this order; everything
// not named here is shared runtime (checkpoint / boundary machinery).
// Prologue of every entry (in call order): validate_forward / (backward:
// check_stepped + validate_backward + bind_backward_outputs / alloc_grads +
// rtm gate), bind_or_alloc_* wavefields, alloc_cpml, setup_ctx,
// init_aux_slabs, make_state, make_bwd_workspace.
//
// generic_forward — per it in [it_begin, it_end):
//   launch_step_range          the whole per-range stencil step (air-clear
//                              prepass included); DD phase 1 = the cut-side
//                              M-wide strips, phase 2 = strict complement,
//                              unphased = (0, nx)
//   save_boundary_fwd          BS strips (when use_boundary_saving)
//   inject_source_fwd          source injection
//   record                     receiver sampling
//   end_of_step                u_pre/u_now buffer-role rotation
//   capture_allt               deferred u_allt snapshot (only 3-D uses it)
//   <checkpoint save>          shared runtime, not a hook
//   after the loop: save_last_state (final u pair for backward_bs)
//
// generic_backward (full storage) — per reverse it:
//   adjoint_step               adjoint stencil; with HAS_FUSED_FULL_IMG the
//                              imaging of u_forward[it+1] fuses into it
//   inject_adjoint_source      residual injection
//   post_adjoint               adjoint buffer-role rotation
//   accumulate_source_grad     grad_wavelet sampling
//   image_step                 standalone imaging / RTM+illumination taps
//                              (skipped when fused, except for RTM)
//   after the loop (fused only): one trailing image_step at it == 0.
//
// generic_backward_bs — per reverse it, floor max(max(it_lo, 1), bs_stop):
//   adjoint_step / inject_adjoint_source / post_adjoint /
//   accumulate_source_grad     same four as full mode
//   bs_reverse_step            reconstruction (un-inject, NOPML reverse,
//                              strip restore) + gradient imaging, in the
//                              equation's exact order
//   bs_image_step              RTM / illumination tap
//   before the loop (first segment): seed_reconstruction from u_last_two;
//   after the loop (HAS_BS_T0_TAIL): the four adjoint hooks once at it == 0.
//
// generic_backward_ckpt — per chunk: replay then reverse:
//   replay:  replay_step / inject_source_fwd / swap_recon
//   reverse: adjoint_step / inject_adjoint_source / post_adjoint /
//            accumulate_source_grad / image_step
//
// generic_backward_recursive_ckpt — bisection over each ckpt segment; a
//   leaf runs one replay triple, then the reverse-five of ckpt mode with
//   the imaging fed from the leaf's scratch u.
// ---------------------------------------------------------------------------
```

`common/sg_driver.cuh`：

```text
// ---------------------------------------------------------------------------
// HOOK TIMING MAP — read this before any equation's driver_traits.cuh.
// Per entry point, the traits hooks fire in exactly this order; everything
// not named here is shared runtime (checkpoint / boundary machinery).
// Prologue of every entry (in call order): validate_backward (backward only),
// parse_models, setup_ctx, bind_or_alloc_* wavefields, init_aux_slabs,
// alloc_cpml, bind_grads / alloc_grads, make_workspace, make_state,
// signed_adjoint_sources.
//
// sg_generic_forward — per it in [it_begin, it_end):
//   velocity_substep           v: t -> t+1/2           (DD step_phase 1)
//   stress_substep             s: t -> t+1, u_allt[it] (DD step_phase 2 from here)
//   inject_source              per source field
//   <checkpoint save>          shared runtime, not a hook
//   save_boundary_fields       BS strips (when use_boundary_saving)
//   record_field               per receiver field
//   after the loop: save_last_state (final 5-field snapshot for backward_bs)
//
// sg_generic_backward (full storage) — per reverse it:
//   undo_body_force            body-force rho correction (pre-residual)
//   inject_residuals           signed residuals into the adjoint fields
//   select_forward_velocities  v(it) / v(it+1) pointers from u_forward
//   it == 0: image_standalone + undo_receiver_rho, loop ends
//   it  > 0: full_fused_step   imaging + receiver-rho + adjoint step, in
//                              the equation's exact fused order
//
// sg_generic_backward_bs — per reverse it, floor max(it_lo, 1):
//   undo_body_force / inject_residuals / uninject_forward_source  [inject_step]
//   bs_phase1                  stress recon (NOPML) + strip restore +
//                              imaging + receiver-rho + stress-adjoint half
//   bs_phase2                  velocity-adjoint half + carrier capture +
//                              velocity recon (NOPML) + strip restore + prefetch
//   before the loop (first segment): seed_recon from u_last_two.
//   (DD runs step_phase 3 = injections, then 1, then 2 — same op order.)
//
// sg_generic_backward_ckpt — per chunk (sg_backward_segment):
//   replay:  velocity_substep / stress_substep / capture_seg /
//            inject_sources_fwd_bw
//   reverse: undo_body_force / inject_residuals / seg_vel_ptrs /
//            image_standalone / undo_receiver_rho /
//            (it > 0) plain_adjoint_step
//   after each chunk: store_prev_segment hands v(start+1) to the older chunk.
//
// sg_generic_backward_recursive_ckpt — per reverse it:
//   undo_body_force / inject_residuals
//   sg_replay_forward_to_time: velocity/stress substeps + capture_velocities
//                              (NEXT_V eqs also capture v at it+1)
//   carrier_vel_ptrs / image_standalone / undo_receiver_rho /
//   (it > 0) plain_adjoint_step
// ---------------------------------------------------------------------------
```

## 3. 钩子词汇表

模式缩写：F = forward，B = backward(full)，BS = backward_bs，CK = backward_ckpt，
RC = backward_recursive_ckpt，all = 五个入口。下表只列骨架调用的钩子；traits 内部为去重而抽出的
私有 helper（如 staggered 家族的 `stress_adjoint_prepare/apply`、`velocity_adjoint_half`、EVR 的
`momentum_adjoint_half`）不是钩子，骨架不认识它们。

### 声学家族（`eq_driver.cuh`）

常量：`NDIM`、`NAME`、`CKPT_NVAR`、`BS_NVAR`（saver 存几个场）、`BS_LAST_TWO_NVAR`、
`TANGENT_PAD`（条带切向 pad = TANGENT_PAD×M；VRZ 为 1）、`CUT_MASK_BITS`/`CUT_MASK_DESC`、
`ADJ_WF_COUNT`、`RECON_WF_COUNT`、`HAS_FUSED_FULL_IMG`（B 模式把滞后成像融进 adjoint 核）、
`ADCIG_IN_FULL_MODES`、`HAS_BS_T0_TAIL`（BS 循环后是否补 `it == 0` 的四个伴随钩子）。
类型：`Wavefield`、`CPML`、`State`、`BwdWorkspace`、`BsScratch`。

| 钩子 | 作用 | 模式 |
|---|---|---|
| `make_state(p, d, ctx, launch, src_cfg, rec_cfg)` | 循环外构造一次的模型指针 / 算子参数块 / launch 配置包 | all |
| `make_bwd_workspace(p, state, ctx, adjoint)` | 伴随 scratch（acoustic 为空；VRZ 在此清零 adjoint、建 C0/Cx/Cz 系数） | B, BS, CK, RC |
| `make_bs_scratch(p, vp)` | BS 每步 scratch（3-D 的 NOPML 输出场 `f_this`） | BS |
| `validate_forward(p)` / `validate_backward(p, need_recon)` | 方程自己的入口校验，保留手写文本 | F / B, BS |
| `setup_ctx(ctx, p)` | `SolverContext` 家族附加项：topo 行、per-edge FS 面、APM 标志 | all |
| `init_aux_slabs(ctx, wf)` | 安装 CPML aux 条带（slab）几何 | all |
| `alloc_cpml(cpml, p)` | 分配 CPML 剖面张量 | all |
| `allt_shape(d, nt)` | `u_allt` / ckpt chunk 缓冲形状 | F, CK |
| `save_width(abcn, M)` | BS 条带宽度 | F, BS |
| `bind_or_alloc_forward` / `_adjoint` / `_recon` / `_recon_ckpt`、`alloc_recursive_start_state` | 绑定 Python 波场列表，空则内部分配（ckpt 形态按 checkpoint 槽布局） | F / B,BS,CK,RC / BS / CK,RC / RC |
| `bind_backward_outputs(p, grads, illum, want_adcig)` | 绑定/分配 grads（slot 0 = `grad_wavelet`）与 illumination 输出；VRZ 自实现 | B, BS |
| `alloc_grads(p, grads)` / `pack_outputs(out, grads, illum)` | ckpt 内部梯度分配 / 打包 `BackwardOutput` | CK, RC / B, BS, CK, RC |
| `full_rtm_gate` / `bs_rtm_gate` | 是否打开 RTM/illumination/ADCIG（返回指针或 nullptr） | B, CK, RC / BS |
| `fused_grad_ptr(grads)` / `full_store_ptr(p, it)` | 融合成像的梯度目标 / 全存储前向场第 it 步指针 | B (fused) / B, CK |
| `launch_step_range(state, ctx, xb, xe, view, save_all, u_thist, cpml)` | x∈[xb,xe) 的整步 stencil（含 air-clear 前置）；phase 条带由骨架传区间 | F |
| `save_boundary_fwd(rt, state, ctx, view, it_shifted, nt_shifted, bs, w)` | 存 BS 条带（tail 截断后的平移坐标） | F |
| `inject_source_fwd(state, ctx, view, p, it, nsrc)` | 源注入；`ForwardInput` 与 `BackwardInput` 两个重载（后者供 ckpt 重放） | F, CK, RC |
| `record(state, ctx, view, record, p, it, nrec)` | 检波器采样 | F |
| `end_of_step(wf)` | 前向缓冲角色轮转（`swap_pml`：u 与 psi 双缓冲） | F |
| `capture_allt(u_allt, wf, it)` | swap 后的张量拷贝式历史捕获（VRZ 存 5 场；acoustic 为空） | F |
| `save_last_state(saver, wf)` | 末段后把 `u_prev/u_now` 存进 `last_two` | F |
| `adjoint_step(state, ctx, adj_view, cpml, ws, img_fwd, grad_out)` | 融合伴随 stencil；`img_fwd/grad_out` 非空时附带滞后成像 | B, BS, CK, RC |
| `inject_adjoint_source(state, ctx, adj_view, p, it, nsrc, ws)` | 残差注入（VRZ 注入取反残差） | B, BS, CK, RC |
| `post_adjoint(wf)` | 伴随缓冲轮转（`swap_aux` / VRZ `swap_pml`） | B, BS, CK, RC |
| `accumulate_source_grad(state, ctx, adjoint, p, grads, it, nsrc)` | 采样 `grad_wavelet`（VRZ 为空） | B, BS, CK, RC |
| `image_step(state, ctx, fwd_ptr, adjoint, grads*, rtm_out, ws)` | 独立成像 + RTM/illumination；`grads == nullptr` 表示成像已融合 | B, CK, RC |
| `seed_reconstruction(state, ctx, forward, p)` | 首段：从 `u_last_two` 播种重建场并清零吸收 rim（切面除外） | BS |
| `bs_reverse_step(state, ctx, forward, adjoint, rt, bs, w, cpml, p, grads, rtm, ws, scratch, it, bs_it0)` | 一步反向重建（NOPML、restore、成像、源注入、swap）—— 顺序是该方程的位级顺序 | BS |
| `bs_image_step(state, ctx, forward, adjoint, illum, compute_illum)` | prefetch 后的 RTM/illumination/ADCIG 采样 | BS |
| `replay_step(state, ctx, view, cpml, save_all, u_this)` / `swap_recon(wf)` | ckpt 重放的全域前向步 / 重放后的 swap | CK, RC |

### staggered 家族（`sg_driver.cuh`）

常量：`NDIM`、`NAME`、`CKPT_NVAR`、`CKPT_COUNT_MSG`、`CKPT_RECURSIVE_COUNT_MSG`、`BS_NVAR`、
`CUT_MASK_BITS`/`CUT_MASK_DESC`、`ADJ_WF_COUNT`、`RECON_WF_COUNT`、`RECON_LIST_DESC`、
`N_VEL`（速度分量数）、`NEXT_V`（成像是否消费 v(t+1) 载体；false 时 recursive 重放在目标步后
立即 break、不分配跨段载体）。类型：`Wavefield`、`WfView`、`CPML`、`Models`、`State`、
`Workspace`、`VelPtrs`、`ReconCarriers`。

| 钩子 | 作用 | 模式 |
|---|---|---|
| `parse_models(p)` | 从 `p.models` 派生模型（lambda/mu 等，struct 持有以保活） | all |
| `make_state(p, d, models, launch, src_cfg, rec_cfg)` / `make_workspace(p, vp)` | 循环外的参数包 / 伴随 workspace（`init_adjoint_workspace` 或内部 scratch） | all / B, BS, CK, RC |
| `validate_forward(p)` / `validate_backward(p, "full"\|"bs"\|"ckpt"\|"ckpt_recursive")` | 入口校验，逐模式保留手写文本，在 stepped 检查之前运行 | F / B, BS, CK, RC |
| `setup_ctx` / `init_aux_slabs` / `alloc_cpml` / `allt_shape` | 同声学家族（sg 的 `save_width` 固定为 `M + 1`，无钩子） | all |
| `field_ptr(wf, idx)` / `view(wf)` | 按场索引取指针（源/检波循环用）/ 取 `WfView` | all |
| `bind_or_alloc_forward` / `_adjoint` / `_recon`（返回 `ReconCarriers`）/ `_recon_ckpt`、`alloc_recursive_start_state`、`check_ckpt_aux_layout` | 绑定或分配各波场；recon 附带 v(t+1) 载体；ckpt 校验 aux 布局一致 | F / B,BS,CK,RC / BS / CK,RC / CK,RC / CK |
| `prep_adjoint(adjoint, first_segment)` / `prep_adjoint_bs(...)` | 首段清零伴随态（3-D 系需要；2-D 空） | B / BS |
| `bind_grads(p, grads)` / `alloc_grads(vp, grads)` | 绑定 `grads_out`（stepped）或分配，元素数 = 模型数 | B, BS, CK / RC |
| `signed_adjoint_sources(p, receiver_fields)` | 按检波场给残差加符号（应力检波取反；EVR 原样） | B, BS, CK, RC |
| `velocity_substep(state, wf, cpml, solver)` / `stress_substep(state, wf, cpml, solver, u_this)` | 两个半步 kernel（也用于 ckpt/recursive 重放） | F, CK, RC |
| `inject_source(state, solver, field, source, loc, it, nsrc)` | 单场源注入（骨架按 `source_field_indices` 循环） | F |
| `save_boundary_fields(rt, state, solver, wf, it, nt, bs, w)` / `record_field(...)` | 存 BS 场列表 / 单场检波采样 | F |
| `save_last_state(saver, wf)` | 末态各场快照进 `last_two` | F |
| `undo_body_force(state, solver, adj_view, p, src_fields, it, grads)` | 体力源格点的 rho 梯度修正，必须在本步残差注入**之前** | B, BS, CK, RC |
| `inject_residuals(state, solver, adj_view, p, rec_fields, signed, it, nsrc)` | 带符号残差注入伴随场（EVR 尾部再清零伴随应力表面行） | B, BS, CK, RC |
| `select_forward_velocities(p, it, zero_v)` / `seg_vel_ptrs(seg, now, next, next_seg_v)` / `carrier_vel_ptrs(cur_v, next_v)` | 三种来源的 v(it)/v(it+1) 指针：全存储 / ckpt 段缓冲 / recursive 载体 | B / CK / RC |
| `image_standalone(state, solver, adj_view, vptrs, grads)` | 独立梯度核（+EVR 链式规则核） | B(it==0), CK, RC |
| `undo_receiver_rho(state, solver, grads, vptrs, p, rec_fields, it, nsrc)` | 撤销刚注入残差对 rho 成像的污染（速度检波格点） | B, BS, CK, RC |
| `full_fused_step(state, solver, adjoint, ws, cpml, vptrs, grads, p, rec_fields, it, nsrc)` | full 模式一步：成像 + receiver-rho + 伴随步，方程自己的融合顺序 | B (it>0) |
| `plain_adjoint_step(state, solver, adjoint, ws, cpml)` | 无成像参数的四核伴随步 | CK, RC (it>0) |
| `seed_recon(forward, p)` | 首段从 `u_last_two` 播种重建场 | BS |
| `uninject_forward_source(state, solver, for_view, p, src_fields, neg_src, it, nsrc)` | 重建场反注入（-source） | BS |
| `bs_phase1(...)` / `bs_phase2(...)` | 见时序表；DD 相位 1/2 各调一个 | BS |
| `alloc_seg_buffers(vp, len)` / `capture_seg(seg, fwd, slot)` / `store_prev_segment(prev, seg)` | ckpt 段内速度缓冲：分配 / 逐步捕获 / 把 v(start+1) 交给更早的段 | CK |
| `inject_sources_fwd_bw(state, solver, for_view, p, src_fields, it)` | 重放时的前向源注入（`BackwardInput` 字段名） | CK, RC |
| `capture_velocities(v, forward)` | recursive 重放在目标步捕获 v(it)（`NEXT_V` 时再捕获 v(it+1)） | RC |

## 4. 各方程相对参考实现的差异

参考实现：**acoustic2d** 是 `eq_driver.cuh` 的转写来源，**elastic2d** 是 `sg_driver.cuh` 的
转写来源，两者的 traits 是家族内的"零差异"基线。其余摘自各 `driver_traits.cuh` 文件头。

| 方程 | 相对参考的差异 |
|---|---|
| acoustic2d | 声学家族参考实现（本身即基线）；其文件头逐条列出"与 acoustic2d 相同"的含义（常量、State、各钩子的具体行为、bs 顺序 NOPML → restore → band 成像 → 注入 → swap）。 |
| acoustic3d | 无 `ctx.set_per_edge`（per-edge 自由面仅 2-D）；融合伴随带 psi 与 zeta 三重双缓冲（15 个伴随张量，`adjoint_extra_nvar=3`）；BS 反向步在前向源注入**之前**成像（2-D 在注入+swap 之后），其 NOPML 核写每步 scratch 场（`BsScratch.f_this`）；ADCIG 只由 backward_bs 提供（full/ckpt 成像相关的是 vp²·Lap(u) 而非原始压力），且无 seed rim 清零；旧手写 backward_bs 的 `SolverContext` 用 nullptr lap/grad 系数指针，骨架处处传真实指针 —— bs 路径不解引用它们，位级惰性。 |
| acoustic_vrz2d | 模型 `[vp, z]`，`inv_z` 派生；`TANGENT_PAD=1`（条带位于 pad 内 M 处，offset −M）；`ADJ_WF_COUNT=9`（伴随经 `swap_pml` 轮转，无 zeta 双缓冲）；无 `grad_wavelet`（`accumulate_source_grad` 空钩子，`grads_out` slot 0 未用）、无 RTM/illumination/ADCIG（两个 gate 返回 nullptr，`ADCIG_IN_FULL_MODES=false`）；`HAS_FUSED_FULL_IMG=false`（每步独立 `CALCULATE_GRAD_VRZ2D_AUTO`）；`HAS_BS_T0_TAIL=false`（bs floor 为 it==1）；`BwdWorkspace` 持有取反残差、一次性 `BUILD_VRZ_ADJOINT_COEFFS` 的 C0/Cx/Cz 与分裂梯度 scratch，`make_bwd_workspace` 顺带清零伴随态；伴随注入**取反**残差；`u_allt` 存 5 场（u, psix, psiz, zetax, zetaz）由 `capture_allt` 张量拷贝完成，核内 `u_this` 关闭；`save_width` 恒为 M+1；无 `setup_ctx`/aux slab；BS 顺序 = NOPML → 源注入 → restore → swap → 在 swap 后的 `u_now` 上成像；seed 额外清零 `u_next`，rim 清零不带 cut 掩码；`launch_step_range` 拒绝子区间（不支持 phase-split）；chunk/recursive ckpt 保留手写线性段扫描（recursive 入口直接转调 chunk）。 |
| elastic2d | staggered 家族参考实现（本身即基线）；其文件头同样逐条列出"与 elastic2d 相同"的含义。APM 入口保持手写。 |
| elastic3d | 9 个物理场 / 36 个波场张量 / 18 张量伴随 workspace，三个速度载体（`N_VEL=3`）；DD 切面仅 x/y（掩码 0x33），forward 校验之；bound/snapshot 列表缺 `m_syzx` 记忆场时回填（历史布局怪癖）；full backward 只在首段清零伴随态（2-D 依赖 Python 清零的缓冲）；重建绑定接受 12 张量列表（9 场 + 3 载体）或宽松地接受任何带内部载体的完整列表。APM 入口保持手写。 |
| das_mu2d | 速度子步就是 elastic2d 的核，经波场的 `elastic_view()` 适配器到达；应力子步是自定义 stress+strain 核（应变积分在核内），因此每步视图是一**对**（das 视图 + elastic 视图）；CPML 记忆变量保持全域：任何核启动前**必须**安装恒等 aux slab（aux-slab 竞争事故）；8 场 BS 列表（5 弹性 + 3 应变；只恢复弹性 5 场 —— 应变只记录）、18 个波场张量、8 场 `last_two` 并宽松读取 5 场旧格式；full backward **无**梯度融合：独立成像 → receiver-rho 修正 → 然后伴随步；checkpoint 快照全域态（`allocate` 而非 `allocate_from_snapshots`），无 aux 布局检查；无 DD 切面支持（`CUT_MASK_BITS=0`：借用的核不 cut-aware）。骨架新增（旧调用休眠）：stepped 区间、Python 绑定的 record/波场/梯度缓冲、物理 phase split、响亮的 stepped/phase 校验；旧 backward_bs 的死 `f_this` scratch 分配被删。 |
| das_mu3d | 结构上是三维的 das_mu2d（家族差异见上）。3-D 成员自身的差异：15 个物理场（9 弹性 + 6 应变）/ 33 个波场张量 / 18 张量伴随 workspace，三个速度载体（`N_VEL=3`）；BS 存全部 15 场但只恢复弹性 9 场（应变只记录，且与 2-D 不同，从不从 `last_two` 播种 —— 手写 seed 拷 9 场）；重建波场绑定/分配**不带** CPML 记忆张量（`use_pml=false`；2-D 保留）；full backward 绑定后清零伴随态（2-D 依赖 Python 清零），映射为首段 `prep_adjoint`；full/ckpt 成像核是共享的 `LAUNCH_CALCULATE_GRAD_3DELASTIC_BS` 作用于纯速度视图（2-D 有专用 `_NOBS` 核）。 |
| elastic_tti_sg2d | 模型集 = rho + 15 个刚度张量（16 个梯度），核取 `StiffnessPointer`，按需从 `p.models`/grads 重建；2-D 网格上三个速度分量（TTI 耦合 vy），`N_VEL=3`，带符号伴随源用 3-D 场布局；伴随 workspace 是六个普通 scratch 张量，总是内部分配（手写 driver 从不读 `p.adjoint_workspace`）；`u_allt` 存全部 8 个物理场而非仅速度；BS 重建波场总是内部分配（从不绑定 `p.forward_wavefields`）；逐模式入口校验保留手写文本（`validate_backward`）；无 recursive checkpoint：forward 拒绝之，backward.cu 不实例化 recursive driver，recursive 专属钩子（`capture_velocities`、`carrier_vel_ptrs`、`CKPT_RECURSIVE_COUNT_MSG`）刻意缺席；无 DD 切面支持、无 aux slab（CPML 记忆在方程自己的波场张量里）。 |
| elastic_tti_sg3d | 相对 2-D 兄弟：模型集 = rho + 21 个刚度张量（22 个梯度），12 条 PML 剖面；波场/workspace 是**共享**弹性类型（`ElasticWavefieldTensor` 36 张量；`ElasticAdjointWorkspaceTensor` 经 `init_adjoint_workspace`）—— 与 elastic3d 不同，无 `m_syzx` 回填；`u_allt` 只存三个速度（2-D 存全部 8 场）；full **与** BS backward 绑定后都清零伴随态（`prep_adjoint` 与 `prep_adjoint_bs`；2-D 两者都不清零）；forward 直接拒绝 `free_surface`（各向异性介质拒绝镜像法）；速度核只取 `model.rho`，应力核取完整 `StiffnessPointer`；与 2-D 相同：重建波场总是内部分配、无 recursive checkpoint（recursive 专属钩子刻意缺席）、无 DD 切面支持、无 aux slab。 |
| elastic_vr2d | 六个原始模型 `{vp, vs, Rp_x, Rp_z, Rs_x, Rs_z}`、六个梯度、无 rho —— 所有 rho 钩子（`undo_body_force` / `undo_receiver_rho`）为空，成像无 v(t+1) 项（`NEXT_V=false`：recursive 重放在目标步 break，无跨段速度载体）；波场复用 `ElasticWavefieldTensor`（vx/vz 槽放动量 px/pz），15 张量绑定/checkpoint 布局与 5 场 BS 列表与 elastic2d 一致；每个 backward 模式在残差注入后立即清零伴随应力表面行（前向自由面 BC 的伴随），该核位于 `inject_residuals` 尾部；梯度核后每个模式都跟一个链式规则核（`LAUNCH_EVR_GRAD_CHAIN_APPLY`）—— 两者都在 `image_standalone`；14 槽伴随 workspace 池拆成伴随步半（槽 0–9，`Workspace`）与成像半（槽 10–13 + 零动量缓冲，挂在 `State` 上供 `image_standalone` 取用）；backward_bs 从不绑定 Python 重建波场（总是分配，无载体），ckpt/recursive 态用普通全形状 `allocate`（非 snapshot 驱动的 aux 布局）。 |

## 5. 实例：`backward_bs` 每个反向步的 kernel 启动序列

### elastic2d（`sg_generic_backward_bs`，单体 `step_phase = 0`；DD 分相 3 → 1 → 2 时序列完全相同）

`for it = it_hi-1 … max(it_lo, 1)`（首段循环前：`seed_recon` = 5 次 `copy_`，无 kernel）：

* `inject_step(it)`（DD phase 3）
    1. `add_body_force_rho_grad_correction` —— 每个属于 vx/vz 的源场一次（应力源跳过）[`undo_body_force`]
    2. `add_source`（带符号残差 → 伴随场）—— 每个检波场一次 [`inject_residuals`]
    3. `add_source`（`-forward_source` → 重建场）—— 每个源场一次 [`uninject_forward_source`]
* `bs_phase1`（DD phase 1）
    1. `elastic_stress_kernel_nopml<order>` —— 应力反向重建（NOPML）
    2. `boundary_kernel2d`（或 `_compact` / `_bf16` / int8 反量化后的 `boundary_kernel2d`，按存储 dtype）× 3 —— restore sxx、szz、sxz（field 2 先等 chunk）[`restore_backward_2d_field`]
    3. `elastic_stress_adjoint_prepare<order>` —— 带成像指针：vp/vs/rho 梯度融合于此（读 `for_view.v* = v(it)` 与载体 `fv*_prev = v(it+1)`）[helper `stress_adjoint_prepare`]
    4. `sub_receiver_rho_grad_correction` —— 每个速度检波场一次（应力检波无 rho 项）[`undo_receiver_rho`]
    5. `elastic_stress_adjoint_apply<order>` [helper `stress_adjoint_apply`]
* `bs_phase2`（DD phase 2）
    1. `elastic_velocity_adjoint_prepare<order>` [helper `velocity_adjoint_half`]
    2. `elastic_velocity_adjoint_apply<order>` [同上]
    3. `elastic_capture_strips_2d` —— 把 restore 条带上的 v(it) 先拷进载体（`n_strip == 0` 时跳过）
    4. `elastic_velocity_kernel_nopml<order>` —— 速度反向重建，核内 RMW 前把已加载值写入 `fvx_prev/fvz_prev`
    5. `boundary_kernel2d`（变体同上）× 2 —— restore vx、vz（vz 标记 done）[`restore_backward_2d_field`]
    6. `prefetch_next_backward_chunk_if_needed` —— host 侧；gpu-direct 下无 kernel

### acoustic2d（`generic_backward_bs`）

`for it = it_hi-1 … max(max(it_lo, 1), bs_stop)`（首段循环前：`seed_reconstruction` = 2 次 `copy_` + `set_boundary_zeros` × 2）：

1. `acoustic2nd_adjoint_fused<order>` —— 融合伴随（bs 模式不带成像指针）[`adjoint_step`]
2. `add_source`（残差 → `adj.u_next`）[`inject_adjoint_source`]
3. host：`adjoint.swap_aux()` —— u + psi + zeta 双缓冲轮转 [`post_adjoint`]
4. `accumulate_source_grad_2d` [`accumulate_source_grad`]
5. `acoustic2nd_nopml<order>` —— 反向重建，vp 梯度成像融合于此（restore 不覆盖的每个格点）[`bs_reverse_step`]
6. `boundary_kernel2d`（或 `_compact` / `_bf16` / 反量化变体）—— restore `u_next` [`restore_backward_2d`]
7. `calculate_grad_utt_band` —— 只对 restore 条带补成像（`n_strip == 0` 时跳过）
8. `add_source`（`forward_source` → `recon.u_next`）
9. host：`forward.swap()`
10. host：`prefetch_next_backward_chunk_if_needed`
11. `accumulate_rtm_image_2d` —— 仅 `compute_illumination` [`bs_image_step`]
12. `accumulate_adcig_2d` —— 仅请求了 ADCIG

循环后（`HAS_BS_T0_TAIL`，`it_lo == 0` 且无 tail 截断）：步骤 1–4 在 `it = 0` 再执行一次。

## 6. 新增方程清单

1. **选家族**：二阶位移形 + 缓冲轮转 → `eq_driver.cuh`；一阶速度–应力原地更新 → `sg_driver.cuh`；都不是（如 das2d/3d 的导数缓冲形）→ 手写 driver。
2. **kernels.cuh / kernels.cu**：整步 stencil、NOPML 反向步、伴随步、成像核。要走声学 phase-split 必须遵守 `ctx.x_base/x_limit` ranged 启动；要支持 DD 必须用 cut-aware 的 `in_pml` / `phys_*()` 谓词（P2 的共享 helper，见 `gate/in_pml_equiv.cpp`）。
3. **driver_traits.cuh**：从参考实现（acoustic2d 或 elastic2d）复制，保持 [1]–[5] 五节与节内调用序；填常量；钩子里只放启动，复合钩子内部的顺序是位级承重的；不需要的能力用空钩子/常量关闭（`HAS_*`、`NEXT_V`、`CUT_MASK_BITS = 0`），文件头写明相对参考的差异（第 4 节的来源）。
4. **forward.cu / backward.cu / `<eq>.h`**：五个一行入口 + `forward_runner` / `backward_bs_runner` 工厂；不提供 recursive 时不实例化（模板惰性实例化，钩子可缺席）。
5. **`bindings/module.cpp`**：`m.def` 五个 `{C_NAME}_*` 入口和两个 `{C_NAME}_*_runner` 工厂。
6. **Python**：`C_NAME`、`cuda_layout`（`base_nvar` / `pml_nvar` / `last_two_nvar` / `checkpoint_nvar` / `adjoint_extra_nvar` / `boundary_tangent_pad` / `slots` / `grads_out_has_wavelet` 等；模板化后置 `stepped=True`，实现了编号 backward 相位再置 `dd_backward_phases=True`）；无 recursive 时 `C_HAS_RECURSIVE_CKPT = False`；DD 还要在 `parallel/dd_spec.py` 选或声明调度。
7. **重建扩展**：`rm -rf $TORCH_EXTENSIONS_DIR` 后在**同一条命令**里 `SWEEP_JIT_FULL=1` 重编（`.staged` 哨兵按版本号且不分模式，改 `.cu` 会静默编旧码）。
8. **纳入门禁**：加进 `test/solver_gradient_mode_suite.py::SOLVERS` 与 `gate/bitgate.py::ALL_SOLVERS`，迁移前先录基线。

## 7. 验证

**原则**：位级门禁的判据是 `torch.equal`，先用 `--verify-reproducible` 证明可达再用；每配置独立子进程；`gate/run_gate.sh` 是唯一批准的运行方式（永不管道、无判决行即失败、`ran == PASS+FAIL+MISSING+NEW` 截断检查）。环境：`. gate/env.sh`（钉 `PY`、`PYTHONPATH=worktree/src`、专用 `TORCH_EXTENSIONS_DIR`）；门禁与 pytest 永不在同一 shell（`SWEEP_JIT_FULL` 泄漏会造成成片假红）。每步迁移的验收（`PROGRESS.md` 的记法）：A/C/T/dd1 位级全绿 + 全套 pytest 不新增失败，家族收官补 B tier；改 `csrc/` 后先 `rm -rf` 扩展目录再重编，并核对 `.so` 的 mtime 晚于改动、早于门禁日志。

| 工具 | 覆盖 | 用法 |
|---|---|---|
| `gate/bitgate.py` tier **A**（27 配置，~2 min） | acoustic2d/3d、elastic2d/3d、vrz2d、lsrtm2d：eager+c × full/bs_gpu/bs_cpu/bs_gpu_int8/ckpt_chunk/ckpt_recursive × interior/free_surface/free_surface_all4 × canon/phys 网格；每次提交后跑 | `gate/run_gate.sh A base_A.pt` |
| tier **C**（30，~2 min） | `ALL_SOLVERS` 每方程 eager full + c bs_gpu 各一次（浅而全） | `gate/run_gate.sh C base_C.pt` |
| tier **B**（~165，~12 min） | 每方程 × 7 种 c 内存模式 + phys 网格 + free surface；阶段收官跑。`gate/noise_floors.json` 记录已测的非确定配置（DAS、3-D ckpt、int8）的梯度容差，record 与 loss 永远严格；基线含一条预期的错误文本条目（`elastic_tti_sg2d\|c\|ckpt_recursive`） | `gate/run_gate.sh B base_B.pt`；子集 `$PY gate/bitgate.py --tier B --only elastic --compare gate/base_B.pt` |
| tier **T**（10） | 起伏地形（hill/stairs，image 与 APM），其他 tier 全是平地 | `gate/run_gate.sh T base_T.pt` |
| `--verify-reproducible` / `--self-test` / `--measure-noise` | 同码两遍逐位；1 ULP 扰动必须变红；多遍测本底 | `$PY gate/bitgate.py --tier A --verify-reproducible` |
| `gate/ddgate.py` world=1 | 单 tile `ModelParallel`：capture、lazy adjoint 提升、逐炮几何重绑、两家族 step 循环、缓冲角色轮转、持久 runner 路径；12 配置（Acoustic/3D、AcousticVRZ3D、Elastic/3D × fs，+ 两个 bodyforce） | `gate/run_gate.sh dd1 base_dd1.pt` |
| `gate/ddgate.py` world≥2 | 真实 tile、NCCL halo、`cut_face_mask` —— 唯一能抓错发场列表的 rung；ibex 上 `torchrun --nproc-per-node=2 gate/ddgate.py --ranks 2 …`，基线在同一作业里从 dev 重录 | `dd_reverify.sbatch` |
| `gate/evr_ab.py` | `ElasticVRR`（elastic_vr2d）不在 suite 里，A/B/C/T 全绿也不测它：4 backward 模式 × FS，record + 6 梯度共 56 张量逐位 | `$PY gate/evr_ab.py --out new.pt --compare gate/evr_base.pt` |
| `gate/check_equations_api.py` | 冻结 `sweep.equations` 公开面（名字数、注册方程数、别名同一性） | 触碰 `equations/` 时跑 |
| pytest | 全套 ~886 通过（`test_import_does_not_pull_optional_deps` 是已知顺序依赖噪声，单跑绿）。driver 相关：`test_stepped_forward{,_elastic}.py`、`test_stepped_backward{,_elastic}.py`、`test_dd_*two_tile*.py`、`test_dd_tiles_3d.py`、`test_cut_face_mask.py`、`test_slot_table_consistency.py`、`test_dd_supported_equations.py`、`test_boundary_tail_truncation.py`；C-vs-eager 梯度一致性用 `test/solver_gradient_mode_suite.py` | `SWEEP_JIT_FULL=1 $PY -m pytest test/` |
