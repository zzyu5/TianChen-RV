# RVV Slice 贡献指南

每位同学认领一条 [RVV slice](../assignments/rvv-slices-12.md) —— 一项 TianChen-RV 当前尚未支持的能力，让编译器从 MLIR 生成对应的 RVV kernel。提交的是一条编译器路径，而不是手写的 RVV C。

> 作业页与本指南中给出的 `文件:行号` 是发布时的锚点，代码演进后会漂移；以符号名（op、函数、owner 名）为准，行号不符时按符号名搜索。

## 目标链路

```text
test/Conversion/RVV/<cap>.mlir
  → tcrv.exec selected RVV variant
  → typed low-level tcrv_rvv body
  → RVV verifier / legality
  → selected-body realization
  → RVV provider route planning
  → RVVToEmitC 发射
  → tcrv-translate → 生成的 RVV C
  → harness 数值正确性验证
```

route 必须从 typed facts 派生：element type、SEW、LMUL、policy、unit-stride 等 memory form、op-kind、runtime-ABI、target capability。不得从 route id、artifact 名、参数名、`c_type` 串、test 名、精确 intrinsic 拼写反推语义。

## 推荐实现顺序（以 `add` 范例为模板，见 [walkthrough](add-rvv-slice-walkthrough.md)）

1. 先读范例与你 slice 的[作业页](../assignments/rvv-slices-12.md)：确认你要增加的能力当前确实没有，并找到形状最接近的兄弟范例 copy-then-adapt。
2. typed surface 与 verifier（模块 1）：让 typed `tcrv_rvv` body 能表达你的能力，加 verifier 规则。`*-dataflow.mlir` 正向，`*-negative.mlir` 负向。
3. realization owner（模块 2）：把 selected variant 实体化成 typed body 并注册。
4. provider route 与 emitter（模块 3 / 4）：构建 route，写 emitter 发射 RVV intrinsic 序列；`rvv-to-emitc-<cap>.mlir` FileCheck。
5. 验证（模块 6）：写 `harness_<cap>.cpp` 与 scalar 参考实现，过三层（lit → 本地 object → qemu 数值 `proof ok`）。

落点文件见 [module-map](rvv-slice-module-map.md)；验收见 [submission-format](testcase-submission-format.md)。

## PR 检查项

- 增加的能力是什么、为什么当前系统没有。
- 扩展了哪些 typed facts / op / kind。
- provider 如何从 typed facts 派生 route。
- 生成 RVV C 的关键 intrinsic。
- 用了哪个 harness / 参考实现，qemu 运行输出。
- 三层验证状态（lit / object / qemu）。
- 未触犯实现纪律（dtype 前缀 helper / source-front-door 正向 / 公共 EmitC 中的 RVV 语义分支 / route-id 推语义）。

## 协作

- 每条 slice 独立、互不依赖；一项能力一位同学。
- 改 ABI 需四处同步：MLIR 的 runtime_abi_value 绑定、生成签名、harness 的 `extern "C"` 与调用、PR 说明。
- 数值正确性主张需有 qemu 运行证据。
