# Phase0：doc/spec 诚实闭环（no board）

> 父任务 [[06-26-compiler-maturity-retest]]；纪律护栏见父 PRD。无板、不动任何 perf 格、可立即起。

## Goal
关掉 doc/表里所有"推定/已知错"内容，并把 §9 构想纳入 spec 现状口径——零板风险，抗 infra timeout。

## In-scope（4 原子项）
- **doc-bank-null-audit (CM-1)**：审计 `doc/KERNEL-优化自查表.md` + 诚实版里所有"推定 parity/推定平/推定输"格子，标回 `presumed`/NO-claim；任何被当成功 bank 的 unmeasured cell 复原。
- **ime-prefill-corrigendum (N2-C)**：IME e2e 只测过 decode-heavy(≈1.0)，补 prefill-regime 口径说明（**保留 M=1 decode 对照**——证矩阵单元净贡献，doc §5 待重测#6）。
- **spec-status-update (§8⑥)**：把 §9.1–9.4 designed/scaffold/none 构想纳入 spec 现状口径。SC-list 须覆盖**全部** §9.1–9.4 行 + §9.1 三 qualifier（扁平单基类/2-core-行/IME-dialect-MMA-only）+ §9.2 三轴；填 SC-02/04/07 编号空洞，确保无构想静默 scaffold→in-scope-maturity。
- **tf-routing-audit**：**硬前置**（所有 e2e/presumed/memwall/prefill-reseal 依赖它）——逐 e2e 重测确认 ggml 在该板实际派发哪个 kernel（k1 派 `_vl256`、VLEN128 q4_0 无 repack=block-dot 才是真基线、q4_K@VLEN256 被 GEMM 绕过）。

## DoD
- doc/表里无 unmeasured-as-success；每格证据状态标正确。
- spec §9 构想口径落地（直接编辑 spec，spec=稳定契约非状态）。
- routing-audit 产出一张"板×算子→ggml 实派 kernel"表，供 table-retest 用。
- **注**：spec §8①-④ 重写已 done（commit bcde9b89/2361d4d8/5102eaa6，红队核实在 `.trellis/spec/`），本项只补 ⑤(表口径)/⑥(§9)。

## Deps / Risk
无 dep。低风险。**must-NOT**：把 spec 写成进度/状态机。
