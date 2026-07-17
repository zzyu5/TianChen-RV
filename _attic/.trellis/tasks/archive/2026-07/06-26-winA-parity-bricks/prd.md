# Win-A parity 砖（board，非 byte-exact，invalidate 触及格）

> 父任务 [[06-26-compiler-maturity-retest]]；纪律护栏见父 PRD。**腿(1) MECHANISM——天花板 = 追平 ggml 自己的 `_vlN` 形状/regression-removal，不是 beat headline**（beat 见 [[06-26-row2-beat-levers]]）。

## Goal
把"窄-strip、没调 LMUL/shape"emitter 成熟度坑逐 kernel 修掉，接 gearbox 自动 stamp（`getRVVStripVLMAXElements` 单源 + I7 silent-wrong 守卫）。

## In-scope（7 原子项）
- **wa1 (q4_0/q4_1) nibble wide-LMUL**：mf4(¼-LMUL)→m1 block-dot（注19 标的 4 LOSS 0.21–0.29× 坑）。**第一增量砖**。
- **wa2 (q5_0/q5_1) nibble wide-LMUL**：同上；ceiling 可能仍 LOSS（compute-bound）。
- **wa3 IQ gather-LMUL**（iq1_s/iq1_m/iq3）：**先解 `keven_signs_q2xs`/signs64 op-attr 缺失**（没做#8，dep=op-attr 扩展），再接 gearbox。
- **wa4 repack half_lanes auto-select**：现 h16 手 stamp（注17/18），做 VLEN→half_lanes 自动选。**MECHANISM-only，无 beat/无 e2e/无第三类**。
- **wa5 iq4_xs fold-back 砖**（注14 留的独立砖，FP4 家）。
- **memoization-default (CM-2)**：live 路明确"实测记录优先、否则静态 argmin"；**保留诚实 caveat：是 default 非 active capability search**。
- **c3 silu/softmax vcpop**：给共享 `ggml_v_expf_m2` emit `vcpop`-gated 快路径短路（注15 root-cause，0.84×→approaches-parity）。

## DoD（每砖）
- **byte-exact gate 先过**：vs 独立 scalar oracle **且** vs ggml 出厂 `_vlN`（两板 rvv VLEN128 + k1 VLEN256，harts 0-3）；非-NULL VLEN-flip lit。
- perf 报 micro **且**（被触及格）e2e；每格证据状态标；**parity 是天花板时如实标 parity 非 beat**；宽形若不更快=NO claim（非 placeholder 成功）。
- 触及的表格交 [[06-26-table-retest-fill]] 按 per-cell L4-touch map 重测。

## Deps / Risk
不 hard-block on L1/L2（m1 in-tree）。wa3 dep op-attr 扩展。**must-NOT**：bank 推定 parity；声明 2–5×（真值 1.0–1.43×）。

> **执行注**：本 child 含 7 原子项（5 砖 + memoization + c3），start 后**很可能逐砖 sub-spawn**多个 implement+check 周期——别在一个 task cycle 里硬塞 7 项。
>
> **发射归属（见父 PRD"Body 发射归属规则"）**：wa 的手写 wide-LMUL body 是 fallback / 过渡 stop-gap；某 quant 一旦被 [[06-26-track-b-generic-lowering]] 的 `G` 覆盖封印，其权威发射路转 generic，wa 手写体退役或留作对照。别与 Track B 双拥有同一 quant 的 body。
