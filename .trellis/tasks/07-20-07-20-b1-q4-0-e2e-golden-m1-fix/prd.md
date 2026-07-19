# PRD — 修 B1 q4_0→m1 flip 的 2 个 Target/RVV e2e golden 漂移（golden-only·byte-exact 锚 B1 seal）

## 缘起（B1 集成漏项·supervisor 核出）
B1 [GAP-P1] repack 家族 sweep（commit `a230adc61`）的 **q4_0→m1 measured-gate flip**（board-verified byte-exact·decode 2.3-2.5×/prefill 1.24×·spill-free）把 q4_0 repack emit 从 **mf2-form（half_lanes=8）→ m1-form（half_lanes=16）**。B1 更新了 `Conversion/RVV` 的 q4_0 lit·但**漏了 2 个 `Target/RVV` full-pipeline-export-e2e golden**：
- `test/Target/RVV/q4-0-q8-0-repack-gemv-full-pipeline-export-e2e.mlir`（159 行·GEVM decode）
- `test/Target/RVV/q4-0-q8-0-repack-gemm-full-pipeline-export-e2e.mlir`（168 行·GEMM prefill）

两者 golden 仍写 mf2-form（`half_lanes=8` / 注释"lowers to the mf2" / "two 8-lane f32m2 accumulators" / "two disjoint i8mf2 sub-loads" / `__riscv_vle8_v_i8mf2`）·实际 emit 已是 m1-form（`half_lanes=16`·`integer_core_lmul="m1"`·`path_selection_reason="repack-kept-q4_0-vlen128-decode"`·`reason="measured"`）⟹ FileCheck 失败。

## 做（golden-only·零源改）
把这 2 个 e2e golden **更新为 m1-form**·反映 B1 已部署已验证的行为：
1. 每条 RUN 跑 pipeline（`weft-opt`）·捕获实际 emit·**逐 CHECK-prefix 更新 golden CHECK 行**到实际值（half_lanes 8→16·`i8mf2`→`i8m1`·累加器 LMUL·strip 结构 m1=1×16 vs mf2=2×8·导出 C 段·CONSTRUCT-SAME/CORE/EMIT 各 prefix）。
2. **同步更新语义注释**（"lowers to the mf2"→"m1"·"two 8-lane f32m2 accumulators / two disjoint i8mf2 sub-loads"→m1-form 对应描述·别留 mf2 陈述矛盾）。
3. 两 test 都更新（gemv + gemm）。

## ★正确性锚（不是盲配 weft-opt 输出·须确认 m1-form 是对的）
- **B1 board seal**：q4_0 m1 3-arm byte-exact `mism=0`（m1==mf2==oracle）·= m1-form 输出与 mf2-form **数值逐字节等价**·只是 emit 结构（LMUL/half_lanes/strip）变。⟹ 更新 golden 到 m1-form = 反映**已验证的部署真相**·非 mask bug。
- **确认 emit 是 intended m1**：`integer_core_lmul="m1"` + `repack_accumulator_lmul_selection_reason="measured"` + `path_selection_reason="repack-kept-q4_0-vlen128-decode"`（= B1 measured-gate·kNibbleQ40ScaleModel）。若 emit 出现**非** m1-measured 的意外（如 mf2 残留或错 reason）·**停·报 supervisor**（可能 flip 未正确应用·非 golden 问题）。
- ⚠ **别把 golden 改成掩盖真 bug**：更新的每行须能对上"q4_0 部署 m1"这个已验证事实·不是"weft-opt 吐啥写啥"。

## 门
- 2 test FileCheck 全 PASS · **全套 lit 无 NEW 失败**（跑前基线 = 3 个 pre-existing Scripts `widening-dot-reduce`/product-reduce 失败·这 3 个与本役无关·不许新增第 4 个）· 零源码改（`git diff --name-only` 仅这 2 个 .mlir）· 未 git commit。
- 🔴 禁碰任何 .cpp/.h/.td 源 · 禁碰其它 test。

## 汇报（自然）
2 test FileCheck PASS + 每 test 改了哪些 CHECK 行(half_lanes/LMUL/vle8/strip/导出C) + 确认 emit 是 intended m1-measured(reason 对) + 全套 lit 失败数(应仍=3 pre-existing·0 new) + git diff --name-only 仅 2 .mlir。final message 放结论。
