# A3：Dequant plan 的真实 c-driven 消费

## Goal

在五类现有 dequant plan 中选择确有硬件能力分叉的 codebook/grid 轴，完成首个真实 `plan = f(g,c,ω)` 切片：g 提供 codebook/grid/layout 事实，c 提供 VLEN/SEW/vreg 等能力，公式产生合法 load/strip/LMUL 参数，typed plan 被 stamping 与 emitter 真消费。对 nibble 等 board-invariant 轴明确 honest-null，不制造假 c 旋钮。

## Existing Assets

- `getRVVCodebookGatherAnchorLMUL`、register-pressure legality、minimum VLEN seam 已存在。
- `CodebookGatherPlan` 与 `GridLookupPlan` 已存在并被 dequant emitter 使用。
- VLEN/version/vreg_count capability 管道已有 decisive tests。

## Scope

- 首选 codebook gather anchor 或 grid strip/load LMUL 作为垂直切片。
- 公式从 typed g/c 推导 candidate/参数和合法性；禁止从 board 名、march 字符串或 format winner 表推导。
- VLEN128/VLEN256 或资源能力变化必须导致可解释的 plan/legal-set 变化；若物理上不变则登记 honest-null。
- selected plan 在 emission 前落印，emitter 只读参数。
- 对当前 deployed fixtures 保持 byte-exact；性能是否更优留给 B 线配对测量。

## Primary Touch Set

- `include/Weft/Support/CodebookGatherPlan.h`
- `include/Weft/Support/GridLookupPlan.h`
- `include/Weft/Plugin/RVV/RVVGearboxSchedule.h`
- `lib/Conversion/RVV/RVVToEmitCSupport.cpp`
- 对应 front door/body/emitter 与 focused tests

## Dependencies

- 依赖 A2 contract。
- 与 A5 fixture/schema 工作可并行；与 A4 共享 stamping/emitter 文件时串行。
- B4 负责随后的真板性能 disposition。

## Acceptance Criteria

- [ ] 至少一个 plan 参数由真实 g+c 输入共同决定并被 consumer 使用。
- [ ] VLEN/resource decisive test 能翻 plan 或 legal set，不只是 reason 文本。
- [ ] capability missing/conflict fail-closed。
- [ ] measurement 不参与 candidate/legality 构造。
- [ ] emitter 中对应 board/march/format 参数重算被删除。
- [ ] 五类 plan 状态表区分 analytic、measured、constant、honest-null。
- [ ] 当前合法 fixtures byte-exact/ULP 无回退。

## Verification

- VLEN128/256 decisive lit;
- illegal gather span/register-pressure negative test;
- emitted C byte comparison;
- focused build/lit;
- B4 后续同板 paired measurement（不在本任务执行）。

## Out of Scope

- 对所有五类强制制造 c 分叉、在线 profile、性能 winner 扩表、GridDecodePlan block-dot 全面收口。

## Issue Mapping

- ISSUE-117、ISSUE-118、ISSUE-119；ISSUE-122 的 verifier/registry 收口由 A4 承接。
