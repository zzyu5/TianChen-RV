# A3：Dequant plan 的真实 c-driven 消费

## Goal

在五类现有 dequant plan 中选择确有硬件能力分叉的 small-codebook gather 轴，完成真实 `plan = f(g,c,ω)` 垂直切片：g 是现役四种 layout 的 canonical `{scale-model,qk,stride,scale-offset,quant-offset}`，c 是 typed `minimum_vlen`、SEW8/32 支持与实际 emitted LMUL chain 支持，ω honest-null。公式在 bounded declared `{mf2,m1,m2}` realization set 内构造合法域并选择最窄 anchor，selected plan 在 emission 前完整 stamping，emitter 只做机械消费。不要为满足文字制造未证实的 vreg/register-pressure c 旋钮。

## Existing Assets

- `CodebookGatherPlan`、四种 production layout facts、minimum-VLEN seam 已存在。
- capability model 已有 typed property、provider relation/conflict 与 availability 查询。
- 现有 VLEN128 codebook C golden 提供 byte-exact 基线；A3 新增 VLEN64/M2 与
  VLEN256/MF2 差异链。

## Scope

- 只做 small-codebook gather anchor 这一条完整垂直切片；grid 留给后续独立迁移。
- 公式从 typed g/c 推导 candidate/参数和合法性；禁止从 board 名、march 字符串或 format winner 表推导。
- `supported_sew` / `supported_lmul` 是 selected base-V provider 的可选限制表：缺席采用 base SEW8/32 与 whole-LMUL 语义，显式表权威，显式空值/未知 token 非法；fractional mf2 必须有 RVV1.0 或显式 token 正证。
- selected provider 的可选 tail/mask policy 是 typed 单值 enum；显式空、错类型、
  未知值不能退化为 absent。parent/core source provenance 必须一致，但不得成为
  Codebook emission/scale/header 的第二控制轴。
- VLEN64/128/256、SEW allow-list 或 emitted LMUL-chain 支持变化必须导致可解释的 plan/legal-set 变化。
- selected plan 在 emission 前落印，emitter 只读参数。
- 对当前 deployed fixtures 保持 byte-exact；性能是否更优留给 B 线配对测量。

## Primary Touch Set

- `include/Weft/Support/CodebookGatherPlan.h`
- `include/Weft/Plugin/RVV/RVVFormulaDecision.h`
- `include/Weft/Plugin/RVV/RVVSelectedTargetCapability.h`
- `include/Weft/Plugin/RVV/RVVGearboxSchedule.h`
- `lib/Conversion/RVV/RVVCodebookGatherPlanMaterialization.cpp`
- shared typed-backend preparation hook、RVV construction/ODS/verifier 与 codebook emitter
- 对应 front door/body/emitter 与 focused tests

## Dependencies

- 依赖 A2 contract。
- 与 A5 fixture/schema 工作可并行；与 A4 共享 stamping/emitter 文件时串行。
- B4 负责随后的真板性能 disposition。

## Acceptance Criteria

- [x] selected plan 同时真实消费 g 与 c：g 决定并校验 layout/scale 字段，c 改变
      legal anchor；不把生产域恒为 16 的 table cardinality 伪装成可变 g 旋钮。
- [x] VLEN/SEW/supported-LMUL decisive test 能翻 plan 或 legal set，不只是 reason 文本。
- [x] direct/registry 共用 canonical capability set、唯一 selected-provider collector
      与 token predicate；missing/empty/unknown/ambiguous/conflict 均 fail-closed，
      transitive provides/implies 可正确解析。
- [x] measurement 不参与 candidate/legality 构造。
- [x] emitter 中对应 board/march/format 参数重算被删除。
- [x] declared slice 的旧 plan helper/default、emitter direct lookup 与 capability parser caller 为 0；缺 stamp 不回历史默认。
- [x] 五类 plan 状态表区分 analytic、measured、constant、honest-null。
- [x] 当前合法 fixtures byte-exact/ULP 无回退。

## Verification

- VLEN64/128/256 decisive lit；direct/registry capability-gate parity lit；
- illegal gather span、缺失 SEW 与不完整 emitted-LMUL-chain negative test;
- emitted C byte comparison;
- focused build/lit;
- B4 后续同板 paired measurement（不在本任务执行）。

本轮验证：focused 13/13、`Conversion/RVV + Conversion/EmitC` 376/376；全量
978/981，三项失败均为已登记 ISSUE-057 同名基线，无新增失败。direct/registry
parity 单文件包含 26 个 RUN；四个现役 VLEN128 Codebook golden 保持 byte-exact。

## Out of Scope

- 对所有五类强制制造 c 分叉、在线 profile、性能 winner 扩表、GridDecodePlan block-dot 全面收口。
- 未经 objdump/真板证据定义 codebook peak-live/register-pressure 模型；该资源轴留给 B4 证成后再接。

## Issue Mapping

- ISSUE-117、ISSUE-119；ISSUE-122 的 Codebook forged/stale 子集在本任务关闭，
  其余 dequant/Grid registry 收口由 A4 承接。ISSUE-118 的 GridCodebook 几何不在
  本任务范围。
