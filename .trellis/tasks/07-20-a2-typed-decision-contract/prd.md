# A2：最小 typed decision contract

## Goal

在普通 C++/MLIR 插件实现中建立一个最小、可组合的 decision contract，使承重决定显式接收 typed g/c/ω projection，返回 candidate/plan、legality/resource verdict、analytic prior、reason/fallback 与可选 measurement key；随后迁入两个完整 RVV vertical slice。每个 declared slice 必须迁移全部 production caller 并删除旧入口，证明它不是文档接口或包裹层。

## Design Constraints

- 复用现有 mechanism-specific plan 与 typed body；不建立 variant 万能对象。
- g/c/ω 各自有明确类型和 owner，禁止 `map<string, any>`、字符串 bag 或 JSON-only authority。
- contract 可由小型 struct/free function/interface 组合实现，不要求新目录或跨插件 ABI。
- formula 构造候选，legality 过滤，selector 只消费合法候选；emitter 不属于 decision contract。
- capability missing/default/conflict 必须显式，禁止 silent `value_or` 恢复板值。

## Scope

- 基于 A1 matrix 选定最小字段集和 API 住址。
- 第一 slice：一个现有 dequant FormulaProvider/plan 及其全部 production caller。
- 第二 slice：repack LMUL 或 reduction/loop-order 中一个真实非 dequant decision 及其全部 production caller。
- 输出显式 consumed-fields、reason、domain 和 fallback。
- 保持现有合法输入的 selected plan/emitted C byte-exact；只改变 authority/data flow。

## Primary Touch Set

- `include/Weft/Plugin/RVV/RVVGearboxSchedule.h`
- 必要的 plugin-local decision header/source
- 两个选定 consumer 的 front door/provider
- focused tests under `test/Conversion/RVV/`

避免修改 `ExtensionPlugin` 准 ABI；若不可避免，先单独 RFC/issue。

## Dependencies

- 依赖 A1 authority matrix 与 characterization tests。
- A3、A4、A5 依赖本 contract。

## Acceptance Criteria

- [x] API 字段可逐一映射到 architecture decision contract，不含未消费装饰字段。
- [x] 两个完整 vertical slice 通过新 contract 构造并消费 decision，各自全部 production caller 已切换。
- [x] 同输入确定性同结果；g/c/ω 各至少有一个 decisive 或 honest-null 测试。
- [x] illegal/unknown capability fail-closed；fallback total。
- [x] emitted output 对既有合法 fixtures 无非预期差异。
- [x] common/core 无 family-name branch；没有通用 DSL/AST。
- [x] A1 中对应旧入口、旧 overload、adapter/mirror 与 production caller 为 0；测试对照只能是 test-only oracle。

## Verification

- focused unit/lit + byte-exact golden;
- missing/unknown/mutation negative tests;
- full relevant plugin build;
- A1 authority census re-run.

## Out of Scope

- 全部 decision 一次迁入、measurement winner view、IME 迁移、板上性能改判。

## Issue Mapping

- ISSUE-117；新公共 ABI 需求如出现必须独立登记。

## Completion Record

- 新增 plugin-local `RVVFormulaDecision.h`，只承载 Nibble 与 repack
  accumulator-LMUL 两个机制专属 contract；未新增 Formula IR、通用 variant bag
  或跨插件 ABI。
- Nibble 的旧 provider、ignored `minimumVLEN` 与本 slice literal-128 caller 均为
  0；原有 plan reason 保持不变，合法 fixture 的 emitted output 不因 A2 改写。
- LMUL decision 在 `lowerOne` 构造一次，18 个互斥 builder 消费同一 typed
  decision；旧 choice/selector 和 builder-local 独立选择均为 0。reason/key 的实际
  attr writer 只有一个 helper body。
- C++ decision test 覆盖 g mutation、c/ω honest-null、qualified measurement、
  illegal winner、missing capability 和 empty legal set；生产 lit 另证 Unknown RVV
  generation fail closed。
- clean build 与显式 `weft-opt`/`weft-translate` 重链通过；focused lit 5/5；全量
  `check-weft` 975/978，三项失败与 A1/A7 基线完全同名，均为 ISSUE-057。
- authority matrix 8 decisions 全部 source/test contract 匹配，matrix self-test、
  zero-core-family、retired-index、monolith-retire 与 schema 21/21 self-test 全绿。
