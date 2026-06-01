# `q8_0_q8_0` RVV Slice 贡献指南

本轮课堂贡献只有一个主线：让 TianChenRV 能从 MLIR 生成 llama.cpp 风格 `q8_0_q8_0` RVV dot-product kernel。

教师分支提供的 `examples/qemu/llama_q8_0_q8_0_rvv.cpp` 是手写 baseline。学生 PR 不应提交“我也手写了一个 RVV C”。学生要补的是 compiler 路径。

## 目标链路

学生贡献应沿着下面这条链：

```text
test/Target/RVV/<q8-vdot>.mlir
  -> tcrv.exec selected RVV variant
  -> typed low-level tcrv_rvv body
  -> RVV verifier / legality
  -> optional selected-body realization
  -> RVV provider route planning
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> generated RVV C/C++
  -> q8 harness correctness proof
```

route 必须从 typed facts 推导：

```text
element type = i8 source, i16 product, i32 reduction
SEW / LMUL
policy
unit-stride memory form
operation kind = signed q8 dot / widening dot reduction
runtime ABI binding
target capability facts
```

不要从 route id、artifact name、parameter name、`c_type` 字符串、test name 或精确 intrinsic spelling 反推出计算语义。

## 推荐实现顺序

### 1. i8 typed vector 和 load

先让 typed `tcrv_rvv` surface 能表达 signed i8 source vector 和 unit-stride load。

应有测试：

```text
test/Dialect/RVV/<q8-i8-load>-dataflow.mlir
test/Target/RVV/<q8-i8-load>-artifact.mlir
```

target evidence 应能看到：

```text
__riscv_vle8_v_i8m*
```

### 2. i8 widening multiply

表达并 materialize：

```text
i8 vector x i8 vector -> i16 vector
```

target evidence 应能看到：

```text
__riscv_vwmul_vv_i16m*
```

### 3. i16 widening reduction to i32

表达并 materialize：

```text
i16 vector -> i32 scalar/lane0 result
```

target evidence 应能看到：

```text
__riscv_vwredsum_vs_i16m*_i32m1
```

### 4. q8 block ABI 和 selected body

把 `lhs/rhs/out/n` 或等价 ABI 显式绑定进 selected `tcrv_rvv` body。

允许两种 ABI 形状：

```text
struct block_q8_0 *x, struct block_q8_0 *y, float *out, size_t n
```

或拆分数组：

```text
int8_t *x_qs, int8_t *y_qs, float *x_d, float *y_d, float *out, size_t n
```

无论选择哪种，MLIR fixture、generated signature、harness call site 必须一致。

### 5. scale/dequant

最低版可以先输出 raw `int32_t` dot result。完整课堂目标要输出：

```text
sum_i32 * x.d * y.d
```

如果暂时不实现 float scale/dequant，PR 必须明确这是未完成项，不能声明完整 `q8_0_q8_0` correctness。

## 通常需要改哪里

### RVV IR Surface

```text
include/TianChenRV/Dialect/RVV/IR/RVVOps.td
lib/Dialect/RVV/IR/RVVDialect.cpp
lib/Dialect/RVV/IR/RVVConfigContract.cpp
```

优先扩展 generic typed surface。不要新增 `tcrv_rvv.i8_dot_q8_0` 这种绑定单个算法名字的 helper。

### Selected-Body Realization

```text
include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h
lib/Plugin/RVV/RVVSelectedBodyRealization.cpp
```

如果使用 compact pre-realized selected-body op，必须由 RVV plugin realize 成显式 `setvl`、load、widening multiply/reduction、store/output boundary。

### Provider Route

```text
include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h
lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp
lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp
```

provider 负责 C vector type、intrinsic、header 和 route payload。common EmitC 只消费 provider-built route。

### Tests

最低组合：

```text
test/Dialect/RVV/<q8-feature>-dataflow.mlir
test/Conversion/EmitC/rvv-<q8-feature>-materialization.mlir
test/Target/RVV/<q8-feature>-artifact.mlir
```

还应有至少一个 fail-closed negative case，例如：

- `i8`/`i16`/`i32` relation 不匹配；
- LMUL 不匹配；
- accumulator/output ABI role 缺失；
- unsupported memory form；
- 用 metadata 或 route id 伪装 dtype。

## PR Checklist

提交前说明：

- 你的 slice 是 i8 load、widening multiply、widening reduction、selected body、scale/dequant 中哪一段。
- 你新增/扩展了哪些 typed facts。
- provider 如何从 typed facts 推导 route。
- generated RVV C 中的关键 intrinsic。
- 使用了哪个 harness，输入输出是什么。
- QEMU 或 ssh-rvv 的实际输出。
- 没有新增 dtype-prefixed helper、source-front-door positive path 或 common EmitC RVV semantic branch。
