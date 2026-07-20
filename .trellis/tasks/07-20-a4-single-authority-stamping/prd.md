# A4：单一 authority、stamping 与机械 emission

## Goal

把代表性 RVV 决策收口成不可绕过的单向链：formula/provider 构造 → legality 验证 → selector 选择 → selected typed result 落印 → body realization → emitter 机械消费。关闭 GridDecodePlan/DequantMechanismPlan、provider/verifier 与 selector/emitter 的重复 authority。

## Scope

- 以 A1 指出的双头路径和 A3 的 codebook/grid slice 为主标的。
- 把 A1 的 schedule 反例作为独立 declared slice：SP4 legality 不得纳入无 real body 的 candidate；`tiling_variant` / `loop_order` 成为 emission 前可验证的 required selected result；删除 `ABSENT => S6Tiled`、sibling `reason==measured` override、q4_K missing-stamp stride recompute。
- 为 selected decision 定义唯一 stamped representation；reason/provenance 只作 mirror。
- verifier 从 canonical formula 独立重算可验证关系并拒 stamped≠recomputed，或采用等价的单源校验方式；不得修正输入。
- emitter 不再直接查 measurement、board/march、format winner 或第二 registry 决策。
- 未落印、非法、unknown 与空合法集均具名 fail-closed/fallback。
- block-dot 仍直接消费 GridDecodePlan 的剩余路径须逐项迁移或明确分立 authority，不留同一语义双源。

## Primary Touch Set

- `include/Weft/Support/GridDecodePlan.h` 及实现
- dequant mechanism plan/provider
- RVV dialect verifier
- selected body/front door stamping
- `lib/Conversion/RVV/` 相关 emitters
- mutation/negative tests

## Dependencies

- 依赖 A2；A3 的代表性 plan 接口应先稳定。
- ISSUE-122 行为裁决是 verifier 升级前置；任务启动时复核当前用户/canon 是否已授权具体行为。
- A5、A6 依赖本任务的 stamped contract。

## Acceptance Criteria

- [ ] 每个承重 decision 只有一个构造 authority。
- [ ] verifier 能拒绝 forged/stale/不一致 stamp，且不会自行选择替代值。
- [ ] selector 无法选 illegal candidate。
- [ ] emitter 缺 stamp 即失败，不再重算或 fallback 到隐式默认。
- [ ] direct `lookupGridDecodePlan` 等旧入口的合法存活点有完整清单；同义重复点为零。
- [ ] mutation tests 分别杀死：跳 legality、伪造 stamp、emitter 重算、measurement 造 candidate。
- [ ] `rvv-to-emitc-repack-gemm-q6-K-q8-K-col-outer-prior-override.mlir` 不再固化 `selected col_outer/prior → realized row_outer`；改为证明 selected value 被机械实现，且删除 stamp 会 fail closed。
- [ ] min-fold `Plain` 要么有真实 realization，要么在 legality 阶段被排除；selector 返回后不得再由 emitter 报“registered but deferred”。
- [ ] 当前合法 fixtures byte-exact/ULP，deployed regression 无新增失败。

## Verification

- verifier positive/negative unit/lit;
- emitted-C golden/byte-exact;
- A1 authority census and direct-lookup census;
- focused plus relevant full RVV test suite;
- clean rebuild for shared plan/body layout changes.

## Rollback

按 mechanism/consumer 分 slice、小 commit 施工，但一个 declared slice 只有在全部 production caller 迁移、旧入口/bridge 删除后才可合入。某 consumer 尚无法迁移时，该 slice 保持未完成；rollback 仅指整笔 Git 回滚，禁止在 active tree 保留 fail-closed compatibility bridge、双 writer 或 shadow route。

## Out of Scope

- 合并 verifier/selector/emitter 成巨型对象；改变性能 winner；全仓一次删 GridDecodePlan；Formula IR。

## Issue Mapping

- ISSUE-122 为 dequant plan 主项；ISSUE-125 为 SP4/loop-order selected≠realized 主项；关联 ISSUE-034、ISSUE-118/119。
