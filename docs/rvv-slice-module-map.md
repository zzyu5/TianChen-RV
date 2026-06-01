# `q8_0_q8_0` 模块化落点

本文只服务本轮课堂主线：llama.cpp 风格 `q8_0_q8_0` RVV dot-product。不要把它扩展成任意 RVV 特性清单。

## 目标模块图

```text
q8 block ABI
  -> i8 typed vector load
  -> i8*i8 widening multiply to i16
  -> i16 widening reduction to i32
  -> scale/dequant to float output
  -> generated RVV C/C++
  -> harness oracle
```

学生可以承担其中一段，也可以完成完整路径。无论哪种，都要保证 contribution 落在 compiler 路径上，而不是手写 RVV C。

## 模块 1：Typed RVV IR Surface

主要文件：

```text
include/TianChenRV/Dialect/RVV/IR/RVVOps.td
lib/Dialect/RVV/IR/RVVDialect.cpp
lib/Dialect/RVV/IR/RVVConfigContract.cpp
```

本轮需要关注：

- signed i8 source vector；
- i16 intermediate product vector；
- i32 reduction result；
- SEW/LMUL/policy 关系；
- unit-stride memory form；
- widening dot/reduction relation。

输出证据：

```text
test/Dialect/RVV/q8-*-dataflow.mlir
```

不要新增绑定算法名的 helper，例如 `tcrv_rvv.q8_0_dot`。如果需要 compact op，也必须能被 realization 展开成 typed vector-level body。

## 模块 2：Selected-Body Realization

主要文件：

```text
include/TianChenRV/Plugin/RVV/RVVSelectedBodyRealization.h
lib/Plugin/RVV/RVVSelectedBodyRealization.cpp
```

何时需要：

- 你设计了 `typed_q8_dot_pre_realized_body` 或等价 compact selected-body；
- 你需要把 `x/y/out/n` ABI 显式导入 `tcrv_rvv` body；
- 你需要把 `setvl`、load、widening multiply、reduction、store/output boundary 结构化生成出来。

输出证据：

```text
test/Target/RVV/pre-realized-selected-body-artifact-q8-dot.mlir
```

如果直接写显式 low-level `tcrv_rvv` body，可以不改 realization，但要解释 body 已经结构化表达了 route 所需 facts。

## 模块 3：Provider Route Planning

主要文件：

```text
include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h
lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp
lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp
```

本轮 provider 至少要能派生：

```text
__riscv_vle8_v_i8m*
__riscv_vwmul_vv_i16m*
__riscv_vwredsum_vs_i16m*_i32m1
```

这些 intrinsic 必须来自 typed body/config/capability/runtime facts，不允许从 `q8_0_q8_0` 字符串、route id、artifact name 或 C 参数名猜出来。

输出证据：

```text
test/Conversion/EmitC/rvv-q8-*-materialization.mlir
test/Target/RVV/q8-*-artifact.mlir
```

## 模块 4：Runtime ABI

主要文件：

```text
include/TianChenRV/Support/RuntimeABI.h
include/TianChenRV/Support/RuntimeABIContract.h
include/TianChenRV/Support/RuntimeABIMemWindow.h
include/TianChenRV/Support/RuntimeABIParam.h
```

推荐两种 ABI 方案。

方案 A：block struct：

```text
const block_q8_0 *x
const block_q8_0 *y
float *out
size_t n
```

方案 B：拆分数组：

```text
const int8_t *x_qs
const int8_t *y_qs
const float *x_d
const float *y_d
float *out
size_t n
```

方案 B 更容易映射到简单 MLIR fixture；方案 A 更接近 llama.cpp 数据结构。二者都可以，但 PR 必须保持 MLIR ABI、generated signature、harness call 一致。

`c_name` 和 `c_type` 只描述导出 ABI，不定义 dtype/compute。

## 模块 5：Harness 和 Proof

本轮统一参考：

```text
examples/qemu/llama_q8_0_q8_0.h
examples/qemu/harness_llama_q8_0_q8_0.cpp
```

学生可以复用这个 harness，也可以写等价 harness。必须保留：

- q8 block 输入；
- scalar oracle；
- got/expected/diff；
- mismatch 非零退出；
- 成功输出 `proof ok`。

如果 generated signature 不同，需要同步改 harness 并解释 ABI 差异。

## 进阶边界

基础目标是先把 `q8_0_q8_0` 做成可生成、可运行、可 review 的完整路径。完成后可以选择 `q4_0_q8_0` 作为进阶拓展，但它必须建立在已有 q8 dot path 上。

`q4_0_q8_0` 额外需要：

```text
packed u8 load
low/high nibble unpack
u8 -> i8 reinterpret or conversion boundary
zero-point subtract 8
two q8 half-vector loads
i8 widening multiply / widening accumulate
i16 widening reduction to i32
scale/dequant output
```

仍然不建议进入：

- K-quant / q2_K / q3_K；
- mxfp4 lookup/gather；
- segment2/segmentN route-family 泛化；
- source-front-door/source-artifact；
- Toy/Template/TensorExtLite；
- internal supervisor / Trellis / Codex automation。

这些不是没有价值，而是会让本轮统一比较目标失焦。`q4_0_q8_0` 是唯一推荐的近期拓展方向，因为它直接复用 q8 dot 的核心能力，并补充真实 LLM 4-bit weight unpack。
