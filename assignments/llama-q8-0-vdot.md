# 课程主线：llama.cpp `q8_0_q8_0` RVV VDot Slice

本分支不再随机分配 25 个互不相同的 RVV slice。所有学生围绕同一个有 LLM 应用价值的目标推进：

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

## 推荐拆分

虽然全班目标相同，但每个 PR 应保持边界清晰。可以按下面顺序推进：

### Slice A：i8 typed vector load

交付：

- typed i8 vector/config legality；
- unit-stride i8 load；
- target FileCheck 能看到 `__riscv_vle8_v_i8m*`；
- negative test 覆盖不支持的 dtype/LMUL 或 ABI role。

### Slice B：i8 widening multiply

交付：

- `i8 x i8 -> i16` widening multiply typed op 或扩展现有 widening op；
- provider route 到 `__riscv_vwmul_vv_i16m*`；
- FileCheck 覆盖 C vector type 和 intrinsic。

### Slice C：i16-to-i32 widening reduction

交付：

- `i16 vector -> i32 scalar/lane0` reduction relation；
- provider route 到 `__riscv_vwredsum_vs_i16m*_i32m1`；
- harness 或 FileCheck 证明 scalar result boundary。

### Slice D：`q8_0_q8_0` selected body

交付：

- selected-body 或显式 typed body 表达完整 q8 dot；
- `lhs/rhs/out/n` ABI 显式绑定；
- generated RVV C 与 baseline 同形；
- runtime harness 与 scalar oracle 通过。

### Slice E：scale/dequant

交付：

- 把 raw int32 dot 乘以 `x.d * y.d`；
- 明确 scale ABI 和 float output；
- QEMU 或真实 RVV 运行输出与 scalar oracle 匹配。

教师可根据班级人数把 Slice A-E 分给多个学生，也可以要求每个学生独立完成完整路径的不同实现方案。评审时统一用 `q8_0_q8_0` baseline 和同一套 harness 比较。

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

