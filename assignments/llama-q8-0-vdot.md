# 课程主线：llama.cpp `q8_0_q8_0` RVV VDot Slice

本轮课程围绕同一个有 LLM 应用价值的目标推进：

```text
llama.cpp q8_0_q8_0 quantized vector dot
  -> TianChenRV MLIR fixture
  -> selected typed tcrv_rvv body
  -> RVV provider route
  -> generated RVV C/C++
  -> harness correctness proof
```

教师分支只提供参考 RVV C baseline、输入输出 harness、构建流程和验收口径。**TianChenRV 的 MLIR 到 RVV C 路径由学生完成**，不能把本分支提供的 baseline C 当作最终答案提交。

## 目标算子

参考 llama.cpp 的 `ggml_vec_dot_q8_0_q8_0` RVV 路径。本课堂版本保留核心计算形状：

```c
vint8m2_t x = __riscv_vle8_v_i8m2(...);
vint8m2_t y = __riscv_vle8_v_i8m2(...);
vint16m4_t prod = __riscv_vwmul_vv_i16m4(x, y, vl);
vint32m1_t sum = __riscv_vwredsum_vs_i16m4_i32m1(prod, zero, vl);
```

每个 block 有 32 个 signed int8 quantized values 和一个 scale：

```text
block_q8_0:
  d:  float scale
  qs: int8[32]
```

真实 llama.cpp 使用 fp16 scale；课堂 baseline 使用 `float d`，避免把作业重点转移到 fp16 ABI 和转换细节上。学生完成主路径后，可以把 fp16 scale 作为扩展项。

数学语义：

```text
out = sum over blocks b:
        dot_i32(x[b].qs, y[b].qs) * x[b].d * y[b].d

dot_i32(a, b) = sum_{lane=0..31} int32(a[lane]) * int32(b[lane])
```

## 学生必须完成什么

最低交付不是手写一个 RVV C kernel，而是让 TianChenRV 能表达并生成这个 kernel 的核心路径。

必须提交：

- 一个 `test/Target/RVV/` 下的 MLIR fixture，表达 selected RVV variant 和 typed `tcrv_rvv` body。
- 一个或多个 `test/Dialect/RVV/` / `test/Conversion/EmitC/` 测试，覆盖 verifier 和 route materialization。
- generated RVV C/C++ evidence，能看到 q8 dot 的关键 intrinsic family。
- 一个 runtime harness，使用同一组输入和 scalar oracle 验证输出。
- 一段运行记录，说明在 QEMU 或等价 RVV 环境中的结果。

学生需要补齐或扩展的 compiler 能力通常包括：

- signed i8 vector value/config，例如 `!tcrv_rvv.vector<i8, "...">` 或等价结构。
- unit-stride signed i8 load，对应 `__riscv_vle8_v_i8m*`。
- signed i8 x signed i8 widening multiply，对应 `__riscv_vwmul_vv_i16m*`。
- signed i16 vector widening reduction to i32 scalar/vector lane，对应 `__riscv_vwredsum_vs_i16m*_i32m1`。
- scalar output ABI，至少能把 lane0 reduction result 写入 output。
- provider 从 typed body/config/capability/runtime facts 推导 C vector type、intrinsic、header 和 `TCRVEmitCLowerableRoute`。

禁止提交：

- 只复制 `examples/qemu/llama_q8_0_q8_0_rvv.cpp` 作为答案。
- 在 common EmitC 中硬编码 `q8_0_q8_0`。
- 使用 route id、artifact name、参数名或精确 intrinsic spelling 反推计算语义。
- 新增 `tcrv_rvv.i8_*` 这类 dtype-prefixed helper。
- 使用 source-front-door/source-artifact 正向路径绕过 typed body/provider。

## 实现自由度

所有学生都面向同一个 `q8_0_q8_0` 目标，这样结果才可比较。每个提交都应该说明自己怎样从 MLIR 走到 generated RVV C，并用同一类 harness 验证输出。

允许存在不同实现方案，例如：

- block struct ABI，接近 llama.cpp 的 `block_q8_0 *x, block_q8_0 *y, float *out, size_t n`。
- 拆分数组 ABI，例如 `x_qs/y_qs/x_d/y_d/out/n`，更容易先打通 compiler path。
- 先生成 raw int32 dot，再扩展到 `float scale` 输出；但只有包含 scale/dequant 的版本才能声明完整 `q8_0_q8_0` correctness。
- 选择不同 LMUL 或 loop shape，但必须解释 vector type、VL、reduction boundary 和 baseline 的等价性。
- 增加更强的 negative tests、tail case、不同 block 数、不同 scale 数据或真实 RVV 性能记录。

无论选择哪条路线，最终都要保持同一个语义：

```text
dot_i32(qs_x, qs_y) * x.d * y.d
```

评审时统一看：

- 是否走 TianChenRV compiler path；
- generated RVV C 是否包含 q8 dot 需要的 intrinsic family；
- harness 输出是否匹配 scalar oracle；
- 实现说明是否清楚解释了和 baseline 的差异。

## 进阶拓展：`q4_0_q8_0`

完成基础 `q8_0_q8_0` 后，可以把 llama.cpp 的 `ggml_vec_dot_q4_0_q8_0` 作为进阶拓展。它更接近常见 4-bit weight + 8-bit activation 的 LLM quantized matvec 路径，但不适合作为第一步，因为它在 q8 dot 之外还需要 q4 unpack。

核心形状：

```text
load packed q4 bytes
  -> low nibble  = byte & 0x0f
  -> high nibble = byte >> 4
  -> reinterpret / convert to signed i8
  -> subtract zero point 8
  -> load two q8 halves
  -> vwmul low half
  -> vwmacc high half
  -> vwredsum to i32
  -> scale by x.d * y.d
```

对应 RVV intrinsic family 通常包括：

```text
__riscv_vle8_v_u8m*
__riscv_vand_vx_u8m*
__riscv_vsrl_vx_u8m*
__riscv_vreinterpret_v_u8m*_i8m*
__riscv_vsub_vx_i8m*
__riscv_vle8_v_i8m*
__riscv_vwmul_vv_i16m*
__riscv_vwmacc_vv_i16m*
__riscv_vwredsum_vs_i16m*_i32m1
```

进阶 PR 必须仍然保持 TianChenRV 路径：

```text
typed tcrv_rvv body
  -> q4 unpack / q8 dot structure
  -> RVV provider route
  -> generated RVV C/C++
  -> q4/q8 harness oracle
```

不要把 `q4_0_q8_0` 写成一个单独的 C 字符串模板，也不要用算法名直接选择 intrinsic。q4 unpack、zero-point、widening multiply-accumulate 和 reduction 都应该能在 typed body/provider facts 中解释。

## Baseline 位置

本分支提供一个手写 RVV C baseline，用于说明目标输出形状和 runtime oracle：

```text
examples/qemu/llama_q8_0_q8_0.h
examples/qemu/llama_q8_0_q8_0_rvv.cpp
examples/qemu/harness_llama_q8_0_q8_0.cpp
examples/qemu/Makefile.rvv
```

运行方式见：

```text
examples/qemu/README.md
docs/build-and-rvv-proof.md
```

学生最终 generated C/C++ 不要求逐字符等同 baseline，但必须保持同一计算语义，并能解释与 baseline 的差异。
