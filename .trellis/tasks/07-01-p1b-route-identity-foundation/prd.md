# P1b: ContractionRouteIdentity 基建(header + registry, 零 binary diff)

> 父 [[07-01-arch-refactor-noperand-core]] P1 第 2 子任务。设计在 [[07-01-p1a-noperand-route-identity-design]] `research/DESIGN-noperand-route-identity.md`(**实现前必读**)。**这是 FOUNDATION:零 binary diff**——建数据结构 + registry,但**无任何 validator 读它**,故 trivially byte-exact。
>
> ⚠ **REVISION(2026-07-01,P1c STOP 后)**:1b 首版(commit `8c8bc496`)给 descriptor 装了 form-owned 的投机 tail 字段(`accSpec/outSpec/nSpec/reduceOpName/leafProfile`),P1c prototype-first 证明它们对 bare `widening_product` 形**错**(4 form 共享 head、tail 各异,head-keyed descriptor 装不下)。**这些字段在 [[07-01-p1c-r1-route-family-derive]] step 0 被 strip**,descriptor 收窄到 **只 multiplicand/arity 轴**(见该任务 PRD + DESIGN 顶部 REVISION v2)。self-check 同时补 unsigned 路(首版只测 signed)。

## Goal

建 `include/TianChenRV/Plugin/RVV/RVVContractionRouteIdentity.h`:`SourceKind`{`PerIterInputBufferLoad`,`ConstantTableLoad`} + `ContractionSourceSpec` + `ContractionRouteIdentity` + `ConditionalStep`(结构见 DESIGN §1)+ 静态 registry + `getContractionRouteIdentity(StringRef mnemonic, bool isSigned) -> const ContractionRouteIdentity*`。**只填现有 N=2 路**:WideningProduct(signed + unsigned)+ PackedI4NibbleUnpack。

## DoD

- header 编译过(纳入 build;`cmake --build build` 绿)。
- registry 填 N=2 路,每 `ContractionSourceSpec` 的 role/abiRole/abiCName/abiCType/srcStripLabel/headOperandIndex/bodyStepPosition **精确对应**现有硬编码(为 1c 的逐字派生做准备)——尤其对齐 `RVVEmitCContractionRouteFamilyInternal.h:167-174` 的 2-entry multiplicand-roles 常量(signed/unsigned)、`RVVDequantDotSourceFrontDoor.cpp:910-915` 的 role stamp。
- **零 binary diff**:无 validator/consumer 引用 registry → 任何 emit 字节不变(429 RVV lit 不变、dequant e2e 不变)。这是"零输出变"foundation,不是行为改。
- 加一个**编译期/单测 self-check**(可选但建议):assert N=2 signed 路 join 出的 multiplicand-roles 串 == `kRVVLowPrecisionSignedWideningProductMultiplicandRoles` 逐字(证 registry 数据为 1c 派生就绪)——**但不接进任何 emit 路**。

## Out of Scope(留给 1c–1f)
- 任何 validator 改成从 identity 派生(= 1c R1 / 1d R2)。
- C3/C4 条目(= 1e/1f)。
- ConditionalStep 的 generator 实现(= 1d;1b 只定义结构)。

## 纪律
- **零 binary diff**:byte-exact gate = forced clean rebuild + 429 RVV lit BEFORE==AFTER(应完全不变,因无 consumer)。见 [[build-incremental-unreliable]]。
- 非推倒重写;不动任何现有 validator(1b 只**新增** header + registry)。
- 走 trellis-implement → trellis-check。
