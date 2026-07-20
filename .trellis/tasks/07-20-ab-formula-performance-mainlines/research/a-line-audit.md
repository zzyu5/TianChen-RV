# A 线只读审计

## Verified Current State

- 五类 dequant plan 已存在：`NibbleDecodePlan`、`CodebookGatherPlan`、`KQuantScaleMinPlan`、`GridLookupPlan`、`TernaryDecodePlan`。
- FormulaProvider 声明集中在 `RVVGearboxSchedule.h`，实现主要在 `RVVToEmitCSupport.cpp`；dequant-row head 已达到 plan 5/5。
- `selectRepackAccumulatorLMUL` 与 `kRepackMeasuredM1FasterMeasurements` 仍住 `RVVLowerQuantContraction.cpp`，说明 selection/data authority 尚未统一。
- `GridDecodePlan` 仍被 dequant provider、block-dot verifier/emitter 等路径直接查询，存在多头 authority。
- 当前 plan 多数先 reproduce-current；真实 c-driven θ 应优先选择 codebook/grid 等确有 VLEN/资源分叉的轴，禁止在 nibble board-invariant 轴制造假旋钮。

## Recommended Modules

1. A1：冻结全链 authority inventory，并以 characterization/negative tests 固定现状。
2. A2：建立最小 typed `g/c/ω → candidates/plan/legality/prior` C++ contract，并迁入两个真实 RVV consumer。
3. A3：让 codebook/grid 等 plan 真实消费 capability，完成首个 g+c 决定性切片。
4. A4：清除 provider/verifier、selector/emitter 双 authority，完成 selected stamping 与 mechanical emission。
5. A5：建立 qualified compiled winner view；hit/miss/stale/illegal winner 均有测试。
6. A6：让 IME 或另一 family 走同一最小 contract 的垂直切片。

## DAG

~~~text
A1 → A2 → A3 → A4 → A6
          └────→ A5 ─┘
~~~

A3 与 A5 的 schema/fixture 部分可并行；A4 涉及共享 stamping/emitter authority，应在接口稳定后串行收口。

## Guardrails

- 不从零重做五类 plan。
- 不建立 Formula dialect/DSL。
- 不把 measurement 当 candidate/legality authority。
- 不为增加公式覆盖制造 unused capability。
- 每刀 byte-exact/ULP、decisive test、fail-closed 和 emitter no-redecision。
