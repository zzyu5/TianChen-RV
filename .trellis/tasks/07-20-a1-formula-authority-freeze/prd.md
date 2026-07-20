# A1：公式 authority 冻结与 characterization tests

## Goal

在动结构前建立可执行的 authority 基线：逐项追踪五类 dequant plan、repack LMUL、SP4、loop-order 等承重决定的 typed 输入、公式/provider、legality、selector、stamping、consumer 与 emitter，找出重复推导、隐藏默认和绕过入口；以 characterization 与负控测试冻结当前合法行为。

## Existing Assets

- 五类 dequant plan 与 FormulaProvider 已存在，dequant-row head 5/5。
- `RVVGearboxSchedule.h`、`RVVToEmitCSupport.cpp`、front-door selector、typed body 和 emitters 已有大量局部契约。
- formula migration ledger/census 已提供历史 inventory，但不能替代 HEAD 数据流核查。

## Scope

- 生成 HEAD authority matrix：`decision → g/c/ω → provider → legal set → selection → stamp → consumer`。
- 对每一项标记 analytic、measured、constant、honest-null、fallback。
- 具名列出 provider/verifier 重算、selector/emitter 重决策、`value_or`/march/format-name shortcut。
- 为代表性 slice 增加现状 characterization tests：五类 dequant 各一、repack LMUL 一、SP4/loop-order 各一。
- 负控至少覆盖 missing capability、illegal candidate、missing stamp 与错误 emitter bypass。

## Primary Touch Set

- `experiments/active/formula-layer-migration/`
- `test/Conversion/RVV/` 与相关 unit tests
- 只读核查 `include/Weft/Plugin/RVV/`、`include/Weft/Support/*Plan.h`、`lib/Plugin/RVV/`、`lib/Conversion/RVV/`

生产代码只允许添加最小 test seam/diagnostic，不进行 contract 重构。

## Dependencies

- 无；A 线首任务。
- 输出阻塞 A2；B1 可并行。

## Acceptance Criteria

- [ ] HEAD matrix 覆盖全部具名决定且每项只有一个“当前实际 owner”描述。
- [ ] 五类 dequant plan 不被误写成待从零创建。
- [ ] 每个重复 authority 有符号级路径和后续 owner 建议。
- [ ] characterization tests 在未改生产语义时全绿，并能由至少一个 mutation/负控变红。
- [ ] ledger 更新为 defined/consumed/stamped/emitted/tested，而非单一 migrated 标记。
- [ ] `git diff` 不含性能策略或 winner 改判。

## Verification

- relevant `ninja` targets and focused `llvm-lit` tests;
- formula/provider symbol census;
- `git diff --check`;
- clean rebuild when shared C++ layouts are inspected via compiled tests.

## Out of Scope

- 新 decision API、性能翻正、板测、Formula IR、第二 family 接入。

## Issue Mapping

- ISSUE-117、ISSUE-118、ISSUE-119、ISSUE-122；发现新重复 authority 时新增独立 issue，不在报告里私设问题清单。
