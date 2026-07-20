# B2：Bench cell 覆盖与正确性加固

## Goal

把现有 harness 从“已有五个脚本”推进到能支撑下一批正式实验的真实 coverage，并修复空心正确性臂。优先补 K-quant vec_dot、k1/标量路由和 q4_K min-term，不重复实现已覆盖的 dequant/grid/GEMM 资产。

## Scope

- `vec_dot`：为 q2_K/q3_K/q5_K 补 ZERO-MODEL oracle/driver/leaf contract；复核 q4_K/q6_K 现有路径。
- `vec_dot`：在 ISSUE-105 已解决后重新评估 k1 VLEN256 路，删除过时“仍 gated”诊断；只有 byte-exact 后才开放。
- ISSUE-114：标准 q4_K/q5_K harness 加 `dmin != 0` 的 min-term-active correctness arm。
- scalar：让 runner 能按 board/op 明确路由 `scalar_vec_dot.sh`，完成 ISSUE-061 family identity gate 后才允许真模式；保持 NON-Win。
- product_reduce：核实三格式 × rvv/k1 的 verify/measure parser 与 lineage；缺失才补。
- 所有 unsupported 组合继续在 ssh 前 fail-closed。

## Measurement Pre-registration

- 本任务主体是 harness/correctness；不把性能倍率作为验收。
- board：rvv、k1、scalar 仅按板册支持组合。
- engine/regime：每个 harness 在 PRD 执行阶段固定四元行键，不依赖默认猜测。
- opponent：部署符号/合法 scalar-ref 必须先过身份探针。
- correctness：ZERO-MODEL、oracle、anti-hollow、fault arm、CORE==PROD；整数 byte-exact。

## Primary Touch Set

- `tools/bench/cells/vec_dot.sh`
- `tools/bench/cells/scalar_vec_dot.sh`
- `tools/bench/cells/product_reduce.sh`
- 必要 driver/oracle/fixture 与 runner routing/parser tests
- 不改 master 结果值

## Dependencies

- harness 实现可与 B1 并行；首次正式写入依赖 B1。
- B3 依赖 K-quant coverage；B6 依赖可追溯 run。

## Acceptance Criteria

- [ ] q2/q3/q4/q5/q6_K vec_dot 的支持/不支持状态逐格明确，支持项具备非空心 oracle。
- [ ] q4_K/q5_K min-term-active arm 可杀死 min-term 错误。
- [ ] k1 VLEN256 不再沿用已解决 blocker；开放项 byte-exact，未开放项具名原因准确。
- [ ] scalar runner 路由不误进 RVV harness，且仍标 enablement/NON-Win。
- [ ] product_reduce contract 与 parser 对现有三格式一致。
- [ ] unsupported 在 ssh 前退出，harness 自身零持久写盘。
- [ ] runner self-test 增加新路由/coverage 负控并通过。

## Verification

- hermetic parser/routing tests;
- focused harness verify on permitted boards when task执行获板窗;
- byte-exact/anti-hollow/fault tests;
- no-write and fail-closed checks;
- no performance headline from correctness-only runs.

## Out of Scope

- K-quant 性能翻正、全部格式/板覆盖、scalar 进入性能头条、修改公式层。

## Issue Mapping

- ISSUE-061、ISSUE-099、ISSUE-104、ISSUE-114；已解决 ISSUE-105 只作事实前提，不重新开启。
