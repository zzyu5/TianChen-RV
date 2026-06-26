# Emitter L0–L3 基建（byte-exact，no board）

> 父任务 [[06-26-compiler-maturity-retest]]；纪律护栏见父 PRD。**byte-exact ⇒ 不改任何可测行为 = 基建欠债清理 ≠ 成熟**（红队硬话；价值是为 L4/cm 杠杆铺路 + 去玩具感）。

## Goal
把"一个 conversation pattern 拆 9 文件 28267 行、每 kernel 从头手拼 `__riscv_*`"的发射层，整理成单一发射词汇表 + 单源 widening-chain + per-family 表——全程 byte-exact。

## In-scope（6 原子项，L0→L3 + 两小项）
- **emitVCall-canary (L0a)**：`emitVCall(rewriter,loc,resultTy,mnemonic,suffix,operands,stepCtx)` 进 `RVVToEmitCSupport.cpp`（现 21 mangler/0 CallOpaqueOp）；把 `RVVToEmitCKQuant.cpp:49/60/69` 三 lambda 转用它（最硬 canary：KQuant 调 Support 0 次）。
- **emitVCall-consolidation (L0b/c)**：补 ~10–12 widening mangler（vwmul/vwmacc/vwredsum/vwadd/vwmulu/vwcvt/vfwmul/vfwcvt/vfwredusum/vncvt）+ 全量 sweep。**复核计数：764 bare `__riscv_` literal / 607 CallOpaqueOp**，趋零。XL。
- **derive-widening-chain (L1)**：单一纯 `deriveWideningChain(base)→{l8,l16,l32,strip,foldGroups}`，**修潜伏 m1/m2 bug seam**（`RVVToEmitCKQuant.cpp:686` 只 {mf2,m1} vs `:1597` 含 m2）。**需 m2 discriminating 单测**（{mf2,m1} byte-exact 不验 bug）。
- **blockdot-facts-handle (L2)**：emitter 侧 stamped-attr + capability-fact 读句柄；**legality 权威仍在上游**（L2 只读，不加决策逻辑——不得读成"修了 cost model"）。
- **per-family-emitter-tables (L3)**：forward/dequant/masked-store 12 个散落 `if(is*Body)`（`RVVToEmitC.cpp:253..646`）→ per-family 表（BlockDot 表已成 35 entry，顺序 load-bearing 勿动语义）。
- **rvv07-fractional-prune (CM-3)**：RVV0.7 无 fractional-LMUL 防御性剪枝（low-pri/defer-eligible，byte-exact 零回归）。

## DoD（每项）
- **byte-exact gate**：forced CLEAN rebuild（ODS .inc 每生、tcrv-opt 有时不重链、**绝不增量** [[build-incremental-unreliable]]）+ glob `.c` **BEFORE==AFTER** 全 RVV lit（≥277）；不用 stale 绝对指纹。
- L1 额外：`deriveWideningChain("m2")=={l8=m2,l16=m4,l32=m8,strip=32,foldGroups=4}` 的 discriminating 断言。
- 证据状态 = **N/A-by-construction**（无 perf/board 主张）。

## Deps / Risk
L1/L2 **不 hard-block** WA 砖（m1 是 in-tree 路；L1 仅在某砖 stamp m2 时相关）。L3 与 L0/L1/L2 正交。**must-NOT**：把"整理发射"读成性能/成熟提升。
