# brainstorm: audit real-MLIR-ness and plan project refactor

## Goal

回答用户的核心怀疑：TianChen-RV 是真 MLIR 项目，还是"套着 MLIR 外壳、内在逻辑自写"的假 MLIR 项目。基于审计结论制定全项目重构计划（以 Trellis 为脚手架），让代码回到 MLIR 原生机制 + 项目自己的 core-invariants（I1–I9）。

## What I already know

- 工程骨架是真 MLIR：`find_package(MLIR)` + AddMLIR、34 处 `add_mlir_*`、9 个 ODS `.td`（7 个 dialect + Passes.td + 1 个 op interface）、GEN_PASS_DEF tablegen pass、tcrv-opt/tcrv-translate、371 个 `.mlir` lit 测试。
- 重大红旗：92 个 cpp 中 **0 处** `RewritePatternSet` / `ConversionPattern` / `applyPartialConversion` —— 全部变换走手写 `walk()` + OpBuilder（30 个文件），完全绕过 MLIR pattern/conversion 框架。
- 自造词汇巨型文件：`RVVTargetSupportBundle.cpp`（>1800 行）、`RVVTargetArtifactRouteFamilyValidation.cpp`（>1000 行，内含对 emitted 产物的字符串期望匹配）、`SourceFrontDoor`、`ConstructionProtocol`、`StatementPlanOwners` 等。
- RVV target 会 `findProgramByName("clang")` shell out 编译。
- spec I7 已承认：descriptor-driven computation 是非法架构，现存 descriptor / microkernel / direct-C 路径是"历史残留、删除目标或 fail-closed 债务"。
- Novelty 契约：N1 capability IR / N2 零-core-branch plugin 泛化 / N3 Gearbox 实测胜出。重构必须服务 N1–N3，不是为洁癖而重构。

## Assumptions (temporary)

- "假 MLIR"的判定标准 = 项目自己的 I1–I9 + MLIR 社区惯例（ODS 定义语义、verifier、pattern/conversion 框架、interface 驱动、lit 测 IR 而非测字符串）。
- 重构是多 PR 的长期工程，本 task 先产出审计报告 + 重构总纲 + 第一批子任务。

## Audit Verdict（四路审计综合，详证据见 research/）

**判定：骨架真、内核假——"MLIR 形状的容器 + 自写字符串机器"。**

真的部分（保留）：CMake/AddMLIR、7 个 TableGen dialect、GEN_PASS_DEF pass 框架、MlirOptMain/mlirTranslateMain、官方 emitc translateToCpp、common 路径零 family-name 分支（I3/N2 成立）、ExtensionPlugin 是合格 C++ 纯虚接口 + registry、lib/Support 无害（1.3k 行 typed 契约）、lib 不读 artifacts（I4 字面合规）。

假的部分（病灶，按严重度）：
1. **RVV codegen 实质是字符串机器**：lib/Plugin/RVV/EmitC/ ~93k 行（RVVEmitCRoutePlanning.cpp 单文件 44,304 行）按 route family 手写 C 语句字符串计划，intrinsic 名靠 Twine 拼接；emitc dialect 只是字符串的合法化载体（违反 I5 精神）。
2. **生产路径上的 golden 自检**：RVVTargetArtifactRouteFamilyValidation.cpp 17,502 行在 export 路径逐字重拼发射器字符串做等值断言（"n - i"、"out + i"、步骤数==3）——验证"发射器没改"而非语义（违反 I4 精神，纯刹车）。
3. **变换不变换**：13 个 pass 中 replaceOp/erase/RewritePatternSet 全仓 0 处；4 个真结构 pass 全是浅层 append-only；Gearbox（N3 核心）是 2,358 行属性记账查表器，不搜索不建结构。
4. **capability 不是 MLIR 对象**：CapabilityDescriptor = std::string/map（违反 I1，N1 第一证据缺位）；RVV ops 445 处 AnyType + verifier 462 处字符串比较，自定义 type 0 处用作 ODS 约束。
5. **测试锁字符串**：T1 真 IR 结构测试仅 ~23%；两个 C++ golden 测试 63,809 行 + 40,796 行 Python 平行 oracle 锁死字符串机器。
6. **链路头部断裂**：唯一通用 vector 入口 fail-closed（1,777 行僵尸前门），端到端只能从手工 typed fixture 起跑。

## Decision (ADR-lite)

**Context**：全量重写 ~210k 行病灶不可能一次完成；且 RVV 字符串机器是当前唯一能在真硬件出活的路径（gate4 实测产物依赖它），直接砍掉 = 截肢不是重构。用户已授权按审计 + 审美推进重构并要求持续执行。

**Decision**：strangler-fig 分期重构。
- **Stage 1（本 task 执行）：去伪。** 切除对系统行为无语义贡献、只锁死现状的部分：(a) 生产 export 路径上的 golden 字符串自检模块（17.5k 行 + 其 golden 测试）；(b) 僵尸 descriptor 前门及 direct-C 残留（I7 点名的删除目标）。验收 = build 绿 + lit 不差于基线（15 个预存失败已固化）。
- **Stage 2（子任务）：立真。** capability 典型化为 MLIR Attr（provides/implies/conflicts 进 ODS，N1 第一证据）；RVV op 用自定义 type 做 ODS 约束替代 AnyType + verifier 字符串比较。
- **Stage 3（子任务）：换心。** 把 RoutePlanning 字符串计划按 route family 逐个置换为细粒度 emitc IR 构造（或 typed 中间 op + 真 lowering），每换一个 family 删对应 golden 测试、补 T1 IR 测试。Gearbox 决策输入从字符串属性迁到 typed facts（N3）。

**Consequences**：Stage 1 后仓库行为不变但减重 ~20k 行死配重，后续改发射器不再被 golden 自检卡死；Stage 2/3 是论文证据工程，按子任务推进；风险 = golden 测试删除后过渡期回归覆盖变薄，靠 lit IR 测试 + ssh rvv e2e 兜底。

## Open Questions

（已由 ADR 收敛；执行中如遇 scope 变化再补）

## Requirements (evolving)

- 产出一份诚实的"真伪 MLIR 审计报告"（research/ 持久化，按维度给判定和证据）。
- 产出重构总纲：目标态架构（MLIR 原生机制 ↔ 自写机制的映射表）、分期计划、每期验收。

## Acceptance Criteria (evolving)

- [x] 审计覆盖：dialect ODS 深度、pass/conversion 机制、target/plugin 机制、测试有效性 4 个维度，每维度有文件级证据（research/*.md）。
- [x] 重构计划与 N1/N2/N3 对齐，每一期说明推进哪个 N（ADR + research/frontdoor-removal-plan.md + research/orphan-sweep-plan.md）。
- [x] 用户确认重构方向与第一批子任务（用户授权大刀阔斧删除/重构；Stage 2/3 子任务已建）。

## Definition of Done

- [x] 审计报告 + 重构总纲落在 task 目录，子任务建立（06-12-stage2-typify-capability-attr、06-12-stage3-replace-string-machine）。
- [x] spec 若需修订，走 trellis-update-spec（新增 guides/dead-mirror-removal-guide.md）。

## Execution Status — Stage 1 去伪 COMPLETE

分期：Stage 1 去伪（本 task 执行，已完成）→ Stage 2 立真（子任务 N1）→ Stage 3 换心（子任务 N3）。

Stage 1 commit ledger（全部 build 绿 + lit 不差于 baseline=3 reds + spec 校验）：
- `d0934210`（前序）excise golden-string validation + dead golden tests（−81,401）。
- `f418cdf9` sync stale lit checks to real verifier/emitter behavior。
- `8d022042` remove dead fail-closed RVV vector source front-door pass（−225）。
- `4740b7c2` delete zero-caller orphans left by the golden-validator excision（−139）。
- `43d44446` excise dead RVV route-validation/metadata-mirror contract cluster（−6,919）。
- `21dc35a9` dedupe the I7 direct-C forbidden-marker guard。
- `de948920` docs(spec): dead-mirror removal guide。

本会话净删 ~−7,300 行死配重，0 行为变化。Stage 1 纯删除已基本穷尽（sweep 确认 contract
cluster 是最后大块；剩余 Batch D 测试层删除涉活 I7 gate 覆盖权衡，已 DEFER 并记录）。
下一步：Stage 2（scoping 投资 wpla0gg0w 进行中 → research/capability-typification-plan.md）。

## Out of Scope (explicit)

- 本 task 不直接动 lib/ 代码；实施在子任务里做。
- 不重写 spec 层（已在 a9c8d9e1 收敛过）。

## Technical Notes

- spec 契约：`.trellis/spec/index.md`（N1–N3 表）、`architecture/core-invariants.md`（I1–I9）。
- 关键早期证据 grep 见 journal / 本 PRD "What I already know"。
