# 表重测 + 填（deliverable closure）

> 父任务 [[06-26-compiler-maturity-retest]]；纪律护栏见父 PRD。**这是用户的终点目标——重测+填完 `doc/KERNEL-优化自查表.md`**。每格带证据状态标；**重构改变过的格子一律重测（micro+e2e），不准沿用旧数字**（用户 converge：重构改变过的就要全测）。[[kernel-wins-dont-transplant-to-e2e]]

## Goal
按 per-cell L4-touch map，在对应砖落地后重测所有触及格 + 填平所有 open/presumed/board-pending 格 + 跑最干净的内存墙判官。

## In-scope（7 原子项）
- **tf-micro-fills (A7+A10+A6)**：残留 micro 格（含 q4_K-repack k1 micro vs ggml 自己 VLEN256 repack=更强对手，注17 留的）。
- **tf-prefill-reseal (A4)**：q4_0 GEMM prefill 5.68× compiler-emitted 复测**盖章**（after `memoization-default`）。
- **tf-presumed-e2e (A2)**：q8_0/q4_K GEVM e2e·rvv + q4_K GEMM e2e·rvv（**预期 LOSS 非 parity**，别 bank 成功）。
- **tf-blockdot-e2e-column (A3)**：~24 block-dot e2e（只测过 q4_K，**其余非 default-NULL**）。重、board-budget。
- **tf-forward-e2e (A8)**：6 前向算子 e2e（after `c3`）。
- **tf-trackB-seal (A9)**：Track-B 板封印（iq4_nl/iq2_xxs VLEN256 k1）经 production 路（dep `trackB-production-export`）。
- **tf-memwall-8b (A1)**：**8B@VLEN128 on rvv = 内存墙真判官**（doc 称最干净 runnable 重测；板在线，"掉线"是历史瞬时态）。

## per-cell L4-touch 门控（必须在对应砖后重测）
q4_0/q4_1←wa1；q5_0/q5_1←wa2；iq1/iq3 gather←wa3；q4_K-repack k1 micro←wa4；选定 K-quant←cm5；VLEN256 wide-cover←cm4；silu/softmax←c3；q4_0 GEMM prefill reseal←memoization-default；q8_0/q5_0 prefill←build-q80/build-q50（否则 N/A-by-absence）。**无 L4-touch 的格并行填。**

## DoD
- 每格：byte-exact gate 先过；micro **且** e2e（被触及/被要求）；证据状态标 `{measured|presumed|board-pending|N/A-by-construction}`；命名板 + harness + 真 baseline（routing-audit 产出）。
- `doc/KERNEL-优化自查表.md` 填完——空格只剩真 N/A-by-construction / N/A-by-absence，无 unmeasured-as-success。
- micro/e2e 分开报；decode regime 标 per-block reduction 延迟受限。

## Deps / Risk
硬 dep：`tf-routing-audit`（baseline）；各砖（per-cell 门控）；build-op / production-export（解锁格）。重 e2e（memwall/blockdot-column）board-budget-gated + timeout-prone（k1 harts 0-3、forced clean rebuild、热插 .so md5-assert）。

> **执行注**：本 child 是收口、含 7 原子项（多个重 e2e），强依赖几乎所有上游砖；start 后**逐格/逐 harness sub-spawn**，别在一个 task cycle 里跑完。
