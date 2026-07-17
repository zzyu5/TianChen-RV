# Research: 现有 dequant emit 码 · codegen 抽签的分支点

- **Query**: lib/ 中 dequantize_row / dequant emit 在哪 · codegen-lottery（宿主 autovec）vs owned emit 的分叉点
- **Scope**: 内部（lib/Conversion/RVV）
- **Date**: 2026-07-18

## 结论（一句）

dequant emit 的**分叉点**在 `lib/Conversion/RVV/RVVToEmitC.cpp:906-924`（`isGgmlDequantizeRowBody`）：**q8_0 家族头**走 FRONT-DOOR CONSTRUCTED 真向量 typed region；**其余 22 格式**（含全部 dequant 标量类目标）走 **DISPATCH-WIRED 手写 per-format monolith decode**，emit 出的是**标量 C**（`dequantize_row_<format>` 的复现），真向量内容全靠宿主 clang autovec = 「codegen 抽签」（ISSUE-001）。

## 文件地图

| 文件 | 行数 | 职责 |
|---|---|---|
| `lib/Conversion/RVV/RVVToEmitC.cpp` | — | dequantize_row 家族**分叉点**（:906）+ dequant-contraction 家族 table（:957 `kDequantKernels`） |
| `lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp` | 4,572 | 前向 elementwise + dequant-support 分支（rms_norm/rope/scale/silu/exp strip） |
| `lib/Conversion/RVV/RVVToEmitCDeferredDequant.cpp` | 1,070 | deferred-wide / low-precision dequant 真向量体 + standalone-dequant body（`vluxei16` gather · lmul 读自 IR 类型 = fail-closed 范本） |
| `lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp` | 2,501 | grid-codebook 真向量 emit（iq2*/iq3* · C4a 复用机体） |
| `include/Weft/Dialect/RVV/IR/RVVDequantizeRowConstruction.h` | — | dequantize_row 构造契约 header |
| `lib/Plugin/RVV/FrontDoor/RVVDequantizeRowStreamFrontDoor.cpp` | — | dequantize_row 前门（构造 typed region） |

## 分叉点原文（`RVVToEmitC.cpp:906-924`）

```
// The dequantize_row family (block_qX -> f32 row): the FAMILY-HEAD q8_0 is
// FRONT-DOOR CONSTRUCTED -- constructOrEmitGgmlDequantizeRow rewrites the
// abstract weft_rvv.dequantize_row into the typed
// weft_rvv.typed_dequantize_row_loop_body region { dequantize_row_decode_core;
// typed_dequantize_row_loop_yield } and lowers it (emission DRIVEN by the typed
// region op-identity + decode_model, [L-6]/[L-8] construction, byte-exact to the
// retired q8_0 monolith). The other 22 formats stay DISPATCH-WIRED (the op
// identity + bounded `format` route to a hand-written per-format monolith decode
// reproducing ggml's reference dequantize_row_<format>). Void-return...
```

⟹ **owned emit（真向量·构造轴强义）= 目前只有 q8_0 家族头**（1 格）。**codegen 抽签 = 其余 22 DISPATCH-WIRED 手写 monolith decode**。dequant 真向量发射器（PR-31）= 把这 22 格从 DISPATCH-WIRED-scalar 升级为 FRONT-DOOR-CONSTRUCTED 真向量（复用 q8_0 家族头的 typed region 机制 + C4a grid-gather 机体）。

## codegen 抽签的机检铁证（ISSUE-001）

- dequant leaf 向量 intrinsic 恒 = 2，且**两条全是 vsetvl（死值）**，真向量运算 = 0。
- 同一字节源不同宿主编译器：向量指令数 **0 ↔ 256（iq1_m）**、**0 ↔ 1018（iq3_s）**。
- nvfp4 额外硬阻断：lowering 发 `ldexpf`×8（libm 不透明 call·autovec 也无从下手）。ISSUE-025 证实 nvfp4 dequant 我方核 rvv=0/gather=0/vset=0 = 纯标量。
- 悬置 dequant 格 = `dequantize_row | iq2_xs | @rvv`（ISSUE-011·note 自述「scalar 源 autovec」·随 ISSUE-001 处置）。

## dequant-contraction 家族 table（另一路 · 已真向量）

`RVVToEmitC.cpp:957` `kDequantKernels[]`（first-match · ORDER IS SEMANTIC）：
1. `isDeferredWideDotReduceBody` → `emitDeferredWideDotReduceBody`
2. `isDeferredWideDequantBody` → `emitDeferredWideDequantBody`
3. `isLowPrecisionDequantBody` → `emitLowPrecisionDequantBody`

这三条住在 `RVVToEmitCDeferredDequant.cpp`，是**已真向量**的 dequant-contraction 路（非 standalone dequantize_row leaf）。standalone dequantize_row leaf（36 board-cell 欠账轴）走的是上面的 DISPATCH-WIRED-scalar 路。

## 诚实边界 / 未找到

- 「22 格 DISPATCH-WIRED」= 注释自述数（22 formats stay dispatch-wired，加 q8_0 家族头 = 23 dequantize_row 格式）；ISSUE-001 影响面口径 = 主表 24 dequant 行 / 36 board-cell。两个数不同口径（格式数 vs 主表行数 vs board-cell）——**引用时须钉死口径**，未在本轮机核统一。
- 未逐一列出 22 个 DISPATCH-WIRED monolith decode 函数名（需读 RVVToEmitC.cpp 全表；本轮只定位分叉点，未穷举）。
