# M-FLAT 闭合可穿透性探针 — 空跑量测冻结注册/validator 管线的真成本

**parent:** 07-03-g1-q8-0-strong-construct(M-FLAT 里程碑)· **base:** refactor/full-refactor-m1

## 为什么是这一步(读我)

砖①②③ 已落(3 个 typed 原语,byte-exact lit),但 **C_construct 仍 0/24——一寸没动**。砖④ STOP-report 证明:①②③ 组成的是**直线/共享-scale body,不是 per-block nb 循环**,且**唯一让 C_construct 动的"闭合"是一套冻结管线**,从未验证过是否可穿透。**在再建任何原语(4a/4b)前,先廉价量测这套闭合到底可不可穿——否则是在往未验证的冻结墙里灌注更多惰性原语。** 这是 de-risking 空跑,**不是建实现**。

## 唯一要回答的问题

**一个"原语组成的强构造 shape"(looped 或哪怕单块 composite)能否被登记进 + validator-认证过那套冻结注册管线,不破坏既有 3 单块强路?真实工作量(碰几个文件/enum/表)与回归风险各是多少?**

具体冻结点(砖④ STOP 已定位,你去量真成本):
1. `RVVConstructionProtocol.cpp` `kRetainedSelectedBodySpecializations`(:357)+ 硬 `.size()!=67` 门(`verifySelectedBodyRoutes`)+ 助记枚举(:932-981)+ 冻结 `kTypedRoleRealizationSummary`(:46-72)+ typed-role operationName 列表(:317-333,当前**排除** ①②③ ops)。
2. `RVVSelectedBodyOperationKind` —— 跨 25+ 文件 switch(是否 exhaustive/must-handle?加一个 enum 编译器会强制几处?)。
3. loop-aware validator：`rejectMixedPreRealizedContractionBody`(`RVVEmitCContractionRouteFamilyInternal.h:527-541`)**无独立调用点**,只在 `materializeSelectedLoweringBoundary` 里对 descriptor 触发 → **能否在没有完整 descriptor→realizer 路的情况下 exercise 一个新 validator?还是必须走全 descriptor 路才能测?**
4. route-identity(`RVVContractionRouteIdentity.cpp`)登记一个新 shape 要动什么。

## 方法(throwaway spike,量测优先于读)

先追一条**既有**单块强路(如 `widening_product_reduce_dequantize`)从 descriptor→realizer→emission-plan→registration **端到端**,列出它当年为登记这一个强 shape 触碰的每个文件/enum/表。然后做**最小 throwaway spike**:尝试登记一个 stub 新 composite 强-shape identity(bump `.size()` 门、加一个 enum、加 role/summary 条目、stub 各 switch 站点),`cmake --build build --target tcrv-opt`,**让编译器告诉你** exhaustive switch 强制几处、`.size()` 门是否级联、加 enum 是否炸构建;再跑既有 RVV lit(`test/Conversion/RVV` + `test/Target/RVV`)看 3 单块强路是否回归。

**量完立即 REVERT 全部源改动**(`git checkout -- <files>` / `git restore`),**工作树只留 `research/` 判决文件**,不 commit,不留半成品。收尾 `git status` 必须干净(仅 `research/` 新增)。

## 交付物(写进 `research/closing-penetrability-verdict.md`)

1. **既有强 shape 注册全清单**:登记一个单块强路需触碰的完整文件/enum/表列表(端到端)。
2. **spike 实测**:加一个新 composite 强-shape 时,编译器强制的 exhaustive-switch 站点数 · `.size()` 门是级联还是简单 bump · `kTypedRoleRealizationSummary` 是否必须手工扩 · route-identity 要动什么。
3. **validator 可 exercise 性**:loop-aware validator 能否脱离完整 descriptor 路被单测触发?(能=闭合可增量测;不能=必须先建 descriptor 路)。
4. **回归实测**:spike 期间既有 3 单块强路 + RVV lit 是否存活。
5. **判决(必须给)**:`tractable-1session` / `heavy-multisession` / `frozen-slog`,附**最险的一处耦合**与**登记一个新强 shape 的最小工作量估**(文件数量级 + 是否需真设计分叉)。

## 红线

- **不动 C_construct / coverage-sixstate / roster**(0/24 不动)。**不 wire q8_0**。**不建 4a/4b**,不删弱 body。
- spike 是**一次性**的:量完必 REVERT,工作树净留 `research/`。**绝不 commit spike 源改动。**
- 既有 3 单块强路 / rejectMixed / DequantizeOp 契约零永久改动。
- 数值主张一律 pending-hardware,不本轮碰。
- **这是探针不是实现**:目标是**判决 + 工作量数**,不是把闭合做出来。若中途发现连 stub 登记都要真设计分叉,**那就是答案**——写清并收尾(对 de-risking 探针,给出"这是多会话冻结机器苦役"的实证判决 = 成功)。

## 权威 spec

core-invariants(I4 evidence-mirror / I5 op-identity / I7 fail-closed)· 执行总纲 [K-4] 六态阶梯 · [PAT-1](科研总纲:118)· burn-down 纪律(进度只认 ΔC_construct + Δ手写 body-LOC)。
