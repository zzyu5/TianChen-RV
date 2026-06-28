# RVV Slice 模块化落点

一条 slice 要改哪些层、落在哪些文件。对任意一个 [12 个作业 slice](../assignments/rvv-slices-12.md) 都适用;`add` 范例([walkthrough](add-rvv-slice-walkthrough.md))是这张图的最小实例。

> 当前后端是 emission-registry 架构:核心 `BackendEmissionRegistry` + `TypedBackendEmissionDriver` 迭代各 family driver(RVV 是其一),common 路径**不含任何 family-name 分支**(I3 零-core-branch)。所有 dtype/SEW/LMUL/intrinsic 知识都在 RVV driver 内,从 **typed facts** 派生。

## 模块 1 — Typed IR Surface + Verifier
```text
include/TianChenRV/Dialect/RVV/IR/RVVOps.td                 # op 定义(采纳 TCRVEmitCLowerableOpInterface)
lib/Dialect/RVV/IR/RVVDialect{Arithmetic,Control,Memory,Store,Reduction,Widening}Ops.cpp  # 各 op verifier(fail-closed, I7)
lib/Dialect/RVV/IR/RVVConfigContract.{h,cpp}               # 合法 SEW/LMUL 组合(枚举,非自由) + ABI 形状
```
新能力若需新 op / 新 kind / 新 dtype 档:在此加 op + verifier 规则。证据:`test/Dialect/RVV/<cap>-dataflow.mlir`(正向) + `<cap>-negative.mlir`(拒非法)。

## 模块 2 — Selected-Body Realization（RVV plugin）
```text
lib/Plugin/RVV/RVV<Cap>SelectedBodyRealizationOwner.cpp     # 把 selected variant 实体化成 typed body
lib/Plugin/RVV/RVVSelectedBodyRealization.cpp               # getRVVSelectedBodyRealizationOwnerRegistry() 注册 owner
```
注册一个 `{name, isPreRealized…Op 谓词, realize…Owner}`。每个 owner 是独立符号,互不影响。

## 模块 3 — Provider EmitC Route（从 facts 派生路由）
```text
lib/Plugin/RVV/EmitC/RVVEmitC<Family>RouteFamilyPlanOwners.cpp  # 构建 TCRVEmitCLowerableRoute(语义角色/ABI 绑定/校验)
lib/Plugin/RVV/EmitC/RVVEmitCRouteFamilyDerivation.cpp          # 按 RVVSelectedBodyOperationKind enum 分派到 family
```
family:Base / ComputedMaskMemory / Contraction / Elementwise / MAcc / Segment2。route 只能从 typed facts 派生,不许读 route id/名字。

## 模块 4 — EmitC 发射
```text
lib/Conversion/RVV/RVVToEmitC.cpp                # VariantToEmitCFunc:matchAndRewrite + emitScopeForLoop per-op dispatch + kBlockDotKernels[] 表
lib/Conversion/RVV/RVVToEmitC{BlockQuantLinear,KQuant,CodebookFp4,GridCodebook,TernaryBinary,DeferredDequant,ForwardElementwise}.cpp  # 各 family emitter
lib/Conversion/RVV/RVVToEmitCSupport.cpp         # riscvIntrinsicName(...) 等纯 mangler(从 type facts 拼 intrinsic 名)
```
- 新 elementwise/memory kind:在 `emitScopeForLoop` 加一臂 + 一个 `emit*` helper。
- 整段新 kernel(forward / block-dot):在 `kBlockDotKernels[]` 或 matchAndRewrite 加项,family 文件写 emitter。前向核(activation/norm/rope/softmax)落在 `RVVToEmitCForwardElementwise.cpp`。
- C 向量类型经 TypeConverter(`populateRVVToEmitCTypeConversions`),非手拼。

## 模块 5 — Capability / Schedule（可选,N1/N3）
```text
lib/Plugin/RVV/RVVCapabilityProfile.cpp          # 从 -march 派生 RVV 版本/SEW/LMUL allow-list(能力分歧 N1)
include/TianChenRV/Conversion/EmitC/TunableScheduleOpInterface.td + lib/Plugin/RVV/RVVScheduleDescriptorRegistry.cpp  # 采纳接口→统一 autotuner 自动发现(零新 pass)
```
多数作业 slice 不碰这层。

## 模块 6 — 测试 / Proof
```text
test/Dialect/RVV/<cap>-dataflow.mlir         # verifier
test/Conversion/RVV/rvv-to-emitc-<cap>.mlir  # 发射 golden(FileCheck)
examples/qemu/harness_<cap>.cpp              # 验证 harness + scalar 参考实现
```
三层 proof 见 [`build-and-test.md`](build-and-test.md) 与 [`testcase-submission-format.md`](testcase-submission-format.md)。

## 禁止路线（I3/I4/I5,所有 slice 通用）
- dtype-prefixed helper(`tcrv_rvv.i8_*` / `getI32xxx`);
- source-front-door / source-artifact 正向路径;
- common EmitC 里硬编码 RVV dtype/SEW/LMUL/intrinsic;
- 从 route id / artifact 名 / C 参数名 / test 名反推计算语义;
- 只为过 harness 而绕过 typed body/provider。
