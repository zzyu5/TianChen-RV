# RVV Slice 贡献指南

本课程每个学生认领一条 [RVV slice](../assignments/rvv-slices-12.md) —— 一个 **TianChen-RV 当前做不到的新能力**,让编译器从 MLIR 生成对应的 RVV kernel。你交的是 **compiler 路径**,不是手写 RVV C。

## 目标链路
```text
test/Conversion/RVV/<cap>.mlir
  -> tcrv.exec selected RVV variant
  -> typed low-level tcrv_rvv body
  -> RVV verifier / legality
  -> selected-body realization
  -> RVV provider route planning
  -> RVVToEmitC 发射
  -> tcrv-translate -> 生成的 RVV C/C++
  -> harness 真机正确性 proof
```
route 必须从 typed facts 派生:element type / SEW / LMUL / policy / unit-stride 等 memory form / op-kind / runtime-ABI / target capability。**不许**从 route id、artifact 名、参数名、`c_type` 串、test 名、精确 intrinsic 拼写反推语义。

## 推荐实现顺序(以 `add` 范例为模板,见 [walkthrough](add-rvv-slice-walkthrough.md))
1. **先读范例 + 你 slice 的 [作业页](../assignments/rvv-slices-12.md)**:确认你加的能力当前确实没有(作业页给了缺席证据),并找到最近的**兄弟范例**(已实现、形状最像的能力)copy-then-adapt。
2. **typed surface + verifier**(模块 1):让 typed `tcrv_rvv` body 能表达你的能力;加 verifier 规则。`*-dataflow.mlir` 正向 + `*-negative.mlir` 负向。
3. **realization owner**(模块 2):把 selected variant 实体化成 typed body 并注册。
4. **provider route + emitter**(模块 3/4):构建 route,写 emitter 发射精确的 RVV intrinsic 序列;`rvv-to-emitc-<cap>.mlir` FileCheck。
5. **proof**(模块 6):写 `harness_<cap>.cpp` + scalar oracle,过三层(lit → 本地 object → k1 真机 `proof ok`)。

落点文件见 [module-map](rvv-slice-module-map.md);验收见 [submission-format](testcase-submission-format.md)。

## PR Checklist
- 我加的能力是什么、为什么当前系统没有(缺席证据)。
- 扩展了哪些 typed facts / op / kind。
- provider 如何从 typed facts 派生 route。
- 生成 RVV C 的关键 intrinsic。
- 用了哪个 harness/oracle,真机(k1)输出。
- 三层 proof 状态(lit / object / k1)。
- 未犯禁忌(dtype-prefixed helper / source-front-door 正向 / common-EmitC RVV 语义分支 / route-id 推语义)。

## 协作纪律
- 每条 slice 独立、互不依赖;一个能力一个学生。
- 改 ABI 要五处同步(MLIR runtime_abi_value / 生成签名 / harness extern "C" / 调用 / PR 说明)。
- 真硬件主张要有真机证据(`ssh k1`,SpacemiT X60;旧 `ssh rvv` 当前不可用)。
