# CUDA driver 骨架模板化 — 设计 (branch refactor/driver-template)

## 目标
18 个 equation dir 各自手抄的 forward.cu(~300-350行) + backward.cu(~830-1320行, 四模式)
收敛为 `cuda/common/eq_driver.cuh` 里的 `template<class Eq>` 五个 generic driver:
generic_forward / generic_backward / generic_backward_bs / generic_backward_ckpt /
generic_backward_recursive_ckpt。方程侧只剩一个 Traits 结构(常数 + kernel 启动 hook)
+ 5 个 1 行的入口函数(module.cpp 符号名不变)。

**由此免费获得**: stepped(it_begin/it_end 区间推进) = 骨架自带 → 迁移过的方程自动
具备 DD 前提; tail_steps/phase-split/ckpt 四模式一次实现全方程共享。

## 硬判据
物理 kernel 一行不动。逐方程迁移, 每迁一个: 全套 pytest 不新增失败 +
位级 gate (A/C 必跑; 涉及方程的 tier 全配置; T=topo; dd1=DD)。
模板从 acoustic2d 的两个 driver **逐行转写**而来(launch 序列逐条对应),
acoustic2d 迁移后的 gate 就是模板正确性的位级证明。

## Eq Traits (acoustic 家族第一版; hook 命名容 elastic)
- 常数: NDIM, CKPT_NVAR(=6), BOUNDARY_NVAR(=1), ADJ_TENSORS(2D=11), RECON_TENSORS(=3), NAME
- 类型: Wavefield(=AcousticWavefieldTensor, 自身已 ndim-generic), CPML
- Ops bundle: make_ops(ctx, p, vp) → {vp_ptr, LaplaceParam, GradParam×3}(构造一次, 循环外)
- hook(全 static, 模板体内零 if-方程分支):
  - init_aux_slabs(ctx, wf)
  - launch_step_range(order, ctx, xb, xe, view, save_all, u_this, ops, cpml)
      = air-clear prepass + 主 stencil (acoustic2d 的 launch_stencil lambda 原文)
  - step_plain(...) = 全域 forward step (ckpt 重放/advance 用)
  - step_nopml(...) = BS 逆重建 step
  - adjoint_step(..., img_fwd, grad_out) = fused 精确伴随(带可选滞后成像)
  - swap_forward(wf)=swap_pml / swap_recon(wf)=swap / swap_adjoint(wf)=swap_aux
  - seed_recon_last_two(fwd_wf, p) / store_last_two(saver, wf) / zero_recon_rims(...)
  - imaging: calculate_grad / calculate_grad_utt / rtm_image / adcig / source_grad
- 维度差: SolverContext ctor(ny,dy)、fdtd::Wave2D/3D::make、
  boundary_runtime.save_forward_2d/_3d 等 → `if constexpr (Eq::NDIM==3)`。

## 迁移顺序
1. acoustic2d (模板事实来源; gate 最厚)
2. acoustic3d (验证 dim-generic)
3. elastic2d (验证第二波场家族; hook 微调在此发生)
4. acoustic_vrz2d (**不 stepped → stepped 的第一个实证**; 之后 DD 准入可声明化)
5. 其余批量; CPML/nopml 拷贝谁迁移谁顺带收编(单独 gate)

## 已知风险
- csrc 变更 → 本地两个扩展目录(gate/.ext-dd-refactor 与 ~/.cache)必须 rm -rf 重建
- kernels.cu 里 calculate_grad 等 __global__ 无 namespace(历史 ODR 惯例), 模板不动它们
- 模板 per-TU 实例化(header-only), ODR 安全; 入口仍在各 equation namespace
