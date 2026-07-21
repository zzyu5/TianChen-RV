# A6：IME 第二 family 的 decision 垂直切片

## Goal

选择一个现有真实 IME kernel slice，优先 `gemm_tile/q4_0/engine=ime`，使其复用 A2/A4 的最小 formula、legality、selection、stamping 与 mechanical emission contract，证明公式层是 family-neutral 的组织方式而不是 RVV 专用抽象。

## Existing Assets

- IME plugin family、capability/profile、typed body/emitter 与真实 vendor/board 证据已存在。
- T3 已有 q4_0@ime 行和 vendor IME 对手/负边界；本任务不从零创建 IME backend。

## Scope

- 启动时复核 q4_0 是最小且真实 deployed slice；若代码事实显示另一 slice 更小，记录替换理由但只选一个。
- family-local g、c、ω projection 与 plan/legality 留在 IME plugin。
- common contract 只抽取 RVV/IME 已共同存在的最小字段。
- capability absence、illegal format 与 fallback 有明确诊断。
- selected typed body 在 IME emitter 前落印；common/core 不新增 family-name branch。
- 结构闭环后由 B5/B6 决定是否做性能/e2e，不在 A6 承诺翻正。

## Primary Touch Set

- `lib/Plugin/IME/` 与对应 include/ODS/tests
- A2 common decision contract 的最小 extension point
- plugin registry/interface only if required
- IME focused contract tests

## Dependencies

- 依赖 A2 与 A4。
- A5 可选：若 slice 有合格 measured winner 则接入，否则先只走 analytic prior。
- B5/B6 在结构完成后做性能与 e2e。

## Acceptance Criteria

- [ ] 一个真实 IME slice 走完整 `g/c/ω → formula → legality → selector → stamp → body → emit`。
- [ ] common/core 零新增 `if IME`/`if RVV`。
- [ ] declared IME slice 的全部 production caller 走共同 contract；旧直达入口、一次性 selector/dispatcher 和兼容 adapter 为 0。
- [ ] family-specific compute 与 capability 仍在 IME plugin 内。
- [ ] capability missing/illegal candidate/fallback tests 齐全。
- [ ] 同一 contract 的 RVV 与 IME cross-family tests 通过。
- [ ] 无为了抽象而新增未消费字段；接口差异能由 family-local plan 表达。
- [ ] 现有 IME emitted behavior/correctness 无回退。

## Verification

- focused IME build/lit;
- cross-family contract tests;
- common family-name branch census;
- existing IME correctness/deployed regression where available.

## Out of Scope

- 迁移整个 IME family、K3 等新板采购、通用 backend API、性能 winner 承诺、runtime offload。

## Issue Mapping

- 结合现有 IME/能力 issue；不得把特定 SpacemiT profile 扩成项目 scope。
