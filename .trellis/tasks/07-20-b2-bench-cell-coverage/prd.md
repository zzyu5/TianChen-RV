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

- [x] q2/q3/q4/q5/q6_K vec_dot 的支持/不支持状态逐格明确，支持项具备非空心 oracle。
- [x] q4_K/q5_K min-term-active arm 可杀死 min-term 错误。
- [x] k1 VLEN256 不再沿用已解决 blocker；开放项 byte-exact，未开放项具名原因准确。
- [x] scalar runner 路由不误进 RVV harness，且仍标 enablement/NON-Win。
- [x] 修复后的错误 dispatch/alias、临时 workaround、过期 blocker 与只保护错误旧行为的 golden 为 0；合法 scalar reference 不视为兼容路径。
- [x] product_reduce contract 与 parser 对现有三格式一致。
- [x] unsupported 在 ssh 前退出，harness 自身零持久写盘。
- [x] runner self-test 增加新路由/coverage 负控并通过。

## Completed Slice（2026-07-20）

- runner 原子切换为显式 `CELL_ROUTES` + 唯一 verify/cold parser registries。`roster key`、`route known`、`parser covered`、`run eligible` 分立；旧 `cells/<op>.sh` 猜路与未命中后 GEMM fallback 已删除。
- K-quant vec_dot driver 统一 q2_K–q6_K 五格式：五个 raw-byte oracle、五叶共链、`QK_K=256` 单 ABI、`K=2048` 多 block、三向固定 fixture byte-exact、三种结果故障、q8-bsums 输入故障；q2/q4/q5 另有 min-term-active + dmin=0 反事实。
- CORE==PROD：五个 leaf 均由当前 production front door + emission plan 重新生成并逐字节核对。发现 q5 历史 fixture 漂移后，直接再生为当前 emitter 输出并双板复验；未保留旧 q5 leaf 或兼容选择。
- 真板 correctness：当前 runner parser 实际消费 q2_K–q6_K × rvv(VLEN128)/k1(VLEN256)，10/10 返回结构化合格结果；product_reduce 三格式 × 两板，6/6 verify/parser 合格；GEMM `iq1_s` × 两板实际 parser 回归合格。
- 对手身份不再把“真实路径”与“强度”混写：q2/q3/q4/q6 记录板宽手调专化；q5 两板实际 exported path 记录为通用向量。两者都是真实部署 ggml 路径，档位只决定证据成色，不决定路径是否合法。
- scalar 仅完成 dormant route/parser contract；roster 没有 `engine=scalar` 行，`BOARDS[scalar]` 仍受 ISSUE-061/104 阻塞，未触发 SSH、未新增分母/主表列、仍是 enablement-NONWIN。
- 本任务只执行 correctness-only `verify` 与 hermetic cold-parser fixture；没有 cold campaign、性能倍率、official run、`runs.log`、master、roster 或论文数字改动。

### 语义边界

这里的三向 byte-exact 只陈述 B2 固定、受控、精确整数 fixture 上的事实，不外推“所有合法浮点输入都普遍 bit-identical”。后续 B3 性能攻坚必须另走正式 cold/T-N/qualification；B2 的 parser reachability 与 correctness 不自动构成性能资格。

## Verification

完整结果见 [`verification.md`](./verification.md)。收口包含 hermetic parser/routing tests、双板 focused correctness、byte-exact/anti-hollow/fault、CORE==PROD、no-write/fail-closed 与全仓回归；没有执行 cold campaign，也不产生性能头条。

## Out of Scope

- K-quant 性能翻正、全部格式/板覆盖、scalar 进入性能头条、修改公式层。

## Issue Mapping

- ISSUE-061、ISSUE-099、ISSUE-104、ISSUE-114；已解决 ISSUE-105 只作事实前提，不重新开启。
