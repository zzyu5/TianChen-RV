# A2：最小 typed decision contract

## Goal

在普通 C++/MLIR 插件实现中建立一个最小、可组合的 decision contract，使承重决定显式接收 typed g/c/ω projection，返回 candidate/plan、legality/resource verdict、analytic prior、reason/fallback 与可选 measurement key；随后迁入至少两个真实 RVV consumer，证明它不是文档接口。

## Design Constraints

- 复用现有 mechanism-specific plan 与 typed body；不建立 variant 万能对象。
- g/c/ω 各自有明确类型和 owner，禁止 `map<string, any>`、字符串 bag 或 JSON-only authority。
- contract 可由小型 struct/free function/interface 组合实现，不要求新目录或跨插件 ABI。
- formula 构造候选，legality 过滤，selector 只消费合法候选；emitter 不属于 decision contract。
- capability missing/default/conflict 必须显式，禁止 silent `value_or` 恢复板值。

## Scope

- 基于 A1 matrix 选定最小字段集和 API 住址。
- 第一 consumer：一个现有 dequant FormulaProvider/plan。
- 第二 consumer：repack LMUL 或 reduction/loop-order 中一个真实非 dequant decision。
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

- [ ] API 字段可逐一映射到 architecture decision contract，不含未消费装饰字段。
- [ ] 两个真实 consumer 通过新 contract 构造并消费 decision。
- [ ] 同输入确定性同结果；g/c/ω 各至少有一个 decisive 或 honest-null 测试。
- [ ] illegal/unknown capability fail-closed；fallback total。
- [ ] emitted output 对既有合法 fixtures 无非预期差异。
- [ ] common/core 无 family-name branch；没有通用 DSL/AST。
- [ ] A1 中对应旧入口被删除、封死或明确仅作兼容 mirror。

## Verification

- focused unit/lit + byte-exact golden;
- missing/unknown/mutation negative tests;
- full relevant plugin build;
- A1 authority census re-run.

## Out of Scope

- 全部 decision 一次迁入、measurement winner view、IME 迁移、板上性能改判。

## Issue Mapping

- ISSUE-117；新公共 ABI 需求如出现必须独立登记。
