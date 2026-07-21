# B3：K-quant 已证伪策略退役

## Goal

依据 ISSUE-109 已完成的五轮板测与明确 STOP 裁定，把仍残留在生产可达图中的四个 q4_K 实验策略原子退役：`fused`、`vwredsum`、`minterm-vec`、`mlp`。负结果继续由归档 task、issue 与 Git 历史保存；生产 emitter、typed attr 合法域和 active test 不再把已证伪实验当作可部署 candidate。

本任务不再发起性能攻坚。归档 task `07-18-q4k-mlp-attack-...` 已证明 clang 保住宽多流 MLP，但 m2 register pressure/spill 抵消收益，cold 0.155 未超过既有 best 0.186；ISSUE-109 随后明确“停、别找第六杠杆、禁 inline asm”。继续原 B3 会重复已经完成的负实验，并违反 [RET-1]。

## Facts and Boundary

- 退役对象仅是 `weft_rvv.q4_k_scaled_dot` 的 ISSUE-109 exact sentinels 及其专属 emission helpers。
- 保留合法 `integer_core_lmul` 域 `{mf2,m1,m2}`，保留默认 q4_K/q5_K typed body、scalar MIN term、正常 widening/dot/reduction 与 q2/q3/q5/q6 生产路径。
- `vwredsum`、fusion、MLP 等通用机制在其他 family 中不属于本任务；禁止按宽泛词语全仓删除。
- q4_K 的 MIN 数学语义必须保留；退役的是 vectorized MIN 实验策略，不是 `dmin * min * bsums` 本身。
- 历史 task/run/数字不改；master、roster、denominator、科研主张不改。

## Production Reachability Census

退役必须同时覆盖：

1. Dialect verifier 对四个 sentinel 的合法化；
2. `RVVToEmitCKQuant.cpp` 的 sentinel decode、bool propagation 与 dispatch branch；
3. `RVVToEmitCInternal.h` 中仅服务四策略的 helper 声明；
4. 四策略专属 helper 实现；
5. active positive-enable lit；
6. live tools/profile/env setter（若 census 命中）；
7. 生成产物中的 retired marker（clean regenerate 后检查）。

允许名称继续出现的位置只有：ISSUE-109 tombstone、immutable archive、本任务退役说明/防复发 gate 与 Git 历史。

## Scope

- 将四个 sentinel 从 `Q4KScaledDotOp` 的合法 `integer_core_lmul` 域移除；旧输入走既有通用 verifier unknown-value rejection，不设 deprecated/compat handler。
- 删除专属实现与所有 production branch，不以 `if (false)`、宏、环境开关或 fallback 保存。
- 保持 canonical q4_K/q5_K 生成 C byte-identical；用 before/after fixture 证明删除未改变默认路径。
- 保留 q4_K min-active、q5_K qh、高位/符号、multi-block/tail 与 CORE==PROD correctness 门。
- 在 ISSUE-109 写清 `RESOLVED — falsified and retired` 的边界与退役 commit，不把负结果改写成性能已修复。
- 同步 B 线当前规划：MLP 不是剩余实验，B4/B5 才是下一批真实性能工作。

## Primary Touch Set

- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp`
- `lib/Conversion/RVV/RVVToEmitCKQuant.cpp`
- `lib/Conversion/RVV/RVVToEmitCInternal.h`
- `test/Dialect/RVV/q4-k-scaled-dot-dataflow.mlir`
- focused canonical q4_K/q5_K conversion tests
- `.trellis/spec/issues/发射器与架构.md`
- B 线 parent/current-audit documents

## Dependencies and Parallelism

- 依赖 B2 已完成的 K-vec correctness/route/parser 面，但不产生新 official run。
- 与 A3 codebook dequant worktree 文件所有权不相交，可并行施工。
- 若未来重审不同硬件/编译器上的新结构，必须新建 issue/task/candidate；不得复活旧 sentinel。

## Acceptance Criteria

- [ ] active production roots 中四个 exact sentinel 的 producer/propagator/consumer/helper 为 0。
- [ ] active positive-enable tests 为 0；旧 sentinel 输入由通用 verifier fail-closed。
- [ ] `integer_core_lmul` 合法域唯一为 `{mf2,m1,m2}`，无兼容 alias。
- [ ] canonical q4_K 与 q5_K emitted C 在删除前后 byte-identical。
- [ ] q4_K min-active 与 q5_K qh 等 focused correctness 仍通过。
- [ ] clean build、focused lit 与全量 `check-weft` 无新增失败。
- [ ] live bench/profile/env 中不存在四策略 setter；历史 archive/issue 仍可追溯。
- [ ] ISSUE-109 明确标为 exact campaign 已证伪并退役，且不声称性能问题被修复。
- [ ] master、性能数字、板册、分母和科研主张零改动。

## Verification

- scoped `rg` caller census（production/include/live tests/tools）；
- before/after canonical emitted-C hash/diff；
- q4/q5 focused Dialect/Conversion lit；
- B2 vec_dot verify（不计时、不写正式 run）；
- clean build + `check-weft`；
- issue/task validation 与工作树 diff audit。

## Out of Scope

- 新的 MLP/m1-width/inline-asm attack；
- q4_K 性能翻 0.8；
- q6/q2/q3/q5 的未经证实 fanout；
- official cold campaign、master promotion 或 e2e 外推；
- 删除其他 family 的合法 fusion/vwredsum/MLP 实现。

## Issue Mapping

- ISSUE-109（exact negative campaign closure + production retirement）。
- ISSUE-112 的通用性能成熟度不由本任务关闭。
