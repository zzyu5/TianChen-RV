# P1b: ContractionRouteIdentity 基建(header + registry, 零 binary diff)

> 父 [[07-01-arch-refactor-noperand-core]] P1 第 2 子任务。设计在 [[07-01-p1a-noperand-route-identity-design]] `research/DESIGN-noperand-route-identity.md`(**实现前必读**)。**这是 FOUNDATION:零 binary diff**——建数据结构 + registry,但**无任何 validator 读它**,故 trivially byte-exact。

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
