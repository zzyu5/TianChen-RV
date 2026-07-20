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

- [x] HEAD matrix 覆盖全部具名决定且每项只有一个“当前实际 owner”描述。
- [x] 五类 dequant plan 不被误写成待从零创建。
- [x] 每个重复 authority 有符号级路径和后续 owner 建议。
- [x] 每个发现同时绑定后续 task owner、应删除的旧 symbol/caller 和可杀死它的验收；不允许 matrix/ledger 成为终点。
- [x] characterization tests 在未改生产语义时全绿，并能由至少一个 mutation/负控变红。
- [x] ledger 更新为 defined/consumed/stamped/emitted/tested，而非单一 migrated 标记。
- [x] `git diff` 不含性能策略或 winner 改判。

## Verification

- relevant `ninja` targets and focused `llvm-lit` tests;
- formula/provider symbol census;
- `git diff --check`;
- clean rebuild when shared C++ layouts are inspected via compiled tests.

## Out of Scope

- 新 decision API、性能翻正、板测、Formula IR、第二 family 接入。

## Issue Mapping

- ISSUE-117、ISSUE-118、ISSUE-119、ISSUE-122；本轮新发现的 SP4 假 legal candidate、missing-stamp 默认与 loop-order selected≠realized 已登记 ISSUE-125，并绑定 A4；不在报告里私设第二问题清单。

## A1 Deliverables

- `experiments/active/formula-layer-migration/AUTHORITY-MATRIX.md`：人读的 8-decision HEAD 数据流与切割面。
- `experiments/active/formula-layer-migration/authority-matrix.v1.json`：机器可读 owner/g/c/ω/provider/legal/select/stamp/consumer、debt→task/issue/killing-test 绑定与 source census。
- `test/Scripts/formula-authority-matrix.test`：矩阵、源符号计数、既有 characterization tests 与三类内存 mutation 的 hermetic gate。

## Completion Record

- clean configure + `ninja weft-opt`：GREEN；本任务未改生产 C++，但仍从当前 worktree 全量重编了真实 consumer/emitter。
- focused lit：12/12 GREEN（五 plan、LMUL、missing capability、SP4、loop-order、matrix/self-test）。
- full `check-weft`：977 total / 974 pass；仅 3 个仓内既有 generated-bundle ABI 失败，与 A7 基线同名同因；新增 matrix test 为新增 1 GREEN。
- issue census：ISSUE-001..125，零缺号、零重号；ISSUE-125 绑定 A4。
- experiments index：新增 formula cell MANIFEST 并重生成，index gate 的既有债从 A7 基线 1370 降到 1367；剩余均为本任务外既有 orphan/manifest debt。
