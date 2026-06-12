# stage2 立真: typify capability as MLIR Attr (N1 first evidence)

Parent: 06-12-mlir-audit-refactor (ADR Stage 2). Created 2026-06-12.

## Goal

把 target capability 从"裸 std::string / map"典型化为**真正的 MLIR Attribute**，带
`provides` / `implies` / `conflicts` 关系，可被 C++ MLIR pass 和插件以 typed 方式查询。
这是 **N1**（RISC-V 扩展异构性作为 first-class capability IR）的第一份结构证据。

## Why (spec 依据)

- **I1**：capability 必须是 first-class、可查询对象，带 provides/implies/conflicts；
  **不是** prose / 裸字符串 / JSON-only / Python dict。它必须能影响：启用哪些插件、
  variant 提议/合法性/选择、tuning 空间、cost 输入、dispatch、emission 路径、fallback。
- 审计现状（dialect-audit / target-plugin-audit）：`CapabilityDescriptor` = std::string/map
  （lib/Support/CapabilityModel.{h,cpp}，~668 行），是 I1 的反例、N1 第一证据缺位。

## Current state（待 scoping 投资确认）

- `lib/Support/CapabilityModel.cpp` (538) + `include/TianChenRV/Support/CapabilityModel.h` (130)：
  `CapabilityDescriptor` + `makeDescriptor` + accessors，string/set 表达 provides/implies/conflicts。
- 已存在部分 capability 结构：`include/TianChenRV/Dialect/Exec/IR/ExecOps.td`、
  `include/TianChenRV/Dialect/Exec/IR/CapabilityProviderComposition.h` —— 需厘清哪些已 typed、
  哪些仍是字符串。
- ~15 个消费者跨 legality/cost/dispatch/plugins（见 orphan-sweep preservation boundary）。

## Approach（strangler-fig，additive-first）

1. **加真 Attr（additive，不破现状）**：在 ODS 定义 capability MLIR Attribute
   （provides/implies/conflicts），与现有 `CapabilityModel` 并存。→ N1 第一证据（一个 pass/插件
   以 typed attr 做真实决策）。
2. **迁消费者**：把 legality/cost/dispatch/plugin enablement 的查询从 string-keyed 改成 typed attr。
3. **删字符串模型**：消费者全迁后删 `CapabilityModel` 的 string/map 表达（去伪收尾）。

## Open design forks（scoping 投资要回答）

- capability Attr 的 ODS 形状：`provides/implies/conflicts` 用 `ArrayAttr<SymbolRefAttr>`？专用
  `CapabilityAttr` 带 nested params？枚举 vs 开放符号？
- 落在哪个 dialect / td（Exec dialect？新 capability dialect？复用 ExecOps.td 既有结构？）。
- 与 `CapabilityProviderComposition.h` 既有 typed 结构的关系（复用/替换/桥接）。
- N1 "第一证据"的最小可证形态：哪个 pass/插件先改成 query typed attr。

## Acceptance Criteria (evolving)

- [x] capability 是 ODS-defined MLIR Attribute，provides/implies/conflicts 结构化在 IR 里
      （`TCRVExec_CapabilityRelationsAttr`，commit ad013a9a）。
- [x] 至少一个真实决策点从 typed attr 读取 —— CheckCapabilityRequires 冲突合法性判定走 typed
      attr（e0a37f64，trellis-check neutralization 证明 load-bearing），lit+单测覆盖。
- [x] build 绿 + lit honest-green（3 environmental reds）；additive 不回归消费者。

## Progress — RELATIONS contract DONE; kind/status/properties = Phase B (still open)

完成（commits）：ad013a9a typed AttrDef · e0a37f64 N1 first evidence（typed-preferred
descriptor）· 431d43f3 Phase-A 迁 14 fixtures + 删 IR 字符串关系表示。**provides/implies/
conflicts 现在只走 typed `#tcrv.capability_relations<...>`。**

仍未做（本 task 续做或新 task）：
- capability 的 `kind`/`status`/`properties`（含 hart `count`、SEW/LMUL/march/mabi）仍是字符串，
  未 typed —— N1 证据可再加深。
- `CapabilityDescriptor`/`TargetCapabilitySet` 仍内部持 std::string（Phase B：等 id/kind/status/
  properties 全 typed 后整体删字符串模型；属更后期 stage）。

## Fable advisor review (2026-06-12) — 3 findings shaping Phase B

经 Fable-5 顾问（advisor 工具不可用，改用 Agent model=fable）评审，VERDICT =
sound-with-caveats，无 correctness 回归（删除确为死码、I7/I8 enforcement 在
RVVTargetSupportBundle.cpp 完好，且该结论 mirage-proof——基于读活代码而非 lit 计数）。
三条 Phase-B 必读 correction：

1. **【最高·设计决策】先定 `CapabilityModel` 字符串模型的去留，再 typed 更多字段。** 当前
   Stage-2 模式是「typed attr → `makeDescriptor` 立刻 restringify 成 `SmallVector<std::string>`
   → passes 全走 `std::string ==`」（CapabilityModel.cpp:101-127,205-217,272-287；消费者
   `classifyRVVSatisfaction` RVVEmitCRoutePlanning.cpp:81-90 仍 string-compare）。**若 kind/
   status/properties 照此再 typed，会付全部 ODS 成本却仍在 resolution 层不是 first-class——
   字符串模型永久藏在 typed 立面后，I1 变 cosmetic。** Phase-B 开工前必须承诺方向：
   要么退役 `CapabilityDescriptor`/`TargetCapabilitySet`（passes 直接查 typed model/attr，
   I1 落地），要么显式接受 typed-attr 只是序列化层。**推荐前者**（否则 N1 只是外壳）。
2. **【fail-closed 缺口】给悬空 relation id 加 pass-time 诊断。** verifier 仅 hygiene
   （ExecOps.cpp:184-211 无跨 op 解析），`conflicts=["typo.id"]` 解析不到任何 provider 也静默
   通过——capability 关系可悬空，I1/fail-closed-adjacent 漏洞。resolution 本是 pass-time 事，
   故在 pass 加「unresolved relation id」诊断（非 verify 时）。
3. **【已做 ✅】**把 38e8c2d0 重锁的 122 条 golden 全句 CHECK 放松成 token regex
   （commit edf02844）——顾问点名的 anti-golden 一致性已补。

诚实校准：N1 "first evidence" 应表述为「relations 的 IR 表示已 typed+verified，且一个真实
legality 决策（CheckCapabilityRequires 冲突路径）从 typed attr 取数」，**不是**「capability
已是 first-class queryable」——是 I1 的首付，不是 I1 达成。

## Out of Scope

- 不要求一次迁完所有消费者（strangler-fig 分步）。
- 不动 Stage 3 的 RVV 字符串机器换心。

## Notes

- 审计证据：parent task research/dialect-audit.md、target-plugin-audit.md。
- 删字符串模型时走 [dead-mirror-removal guide]，确认 CapabilityModel 消费者已全部迁移再删。
