# A4：单一 authority、stamping 与机械 emission

## Goal

把代表性 RVV 决策收口成不可绕过的单向链：formula/provider 构造 → legality 验证 → selector 选择 → selected typed result 落印 → body realization → emitter 机械消费。关闭 GridDecodePlan/DequantMechanismPlan、provider/verifier 与 selector/emitter 的重复 authority。

## Scope

- 以 A1 指出的双头路径和 A3 的 codebook/grid slice 为主标的。
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
- [ ] 当前合法 fixtures byte-exact/ULP，deployed regression 无新增失败。

## Verification

- verifier positive/negative unit/lit;
- emitted-C golden/byte-exact;
- A1 authority census and direct-lookup census;
- focused plus relevant full RVV test suite;
- clean rebuild for shared plan/body layout changes.

## Rollback

按 mechanism/consumer 分小 commit；若某 consumer 尚无法迁移，保持旧入口但显式标单一 owner 和 fail-closed bridge，禁止同时启用两个 writer。

## Out of Scope

- 合并 verifier/selector/emitter 成巨型对象；改变性能 winner；全仓一次删 GridDecodePlan；Formula IR。

## Issue Mapping

- ISSUE-122 为主；关联 ISSUE-118/119。
