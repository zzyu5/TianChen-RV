# `q8_0_q8_0` 测试用例收集格式

本文规定学生完成本轮 `q8_0_q8_0` RVV VDot slice 后，如何提交可复现、可 review 的测试证据。

教师需要明确看到：

```text
MLIR 输入是什么
compiler 生成的 RVV C/C++ kernel 是什么
harness 给 kernel 喂了什么 q8 runtime 输入
scalar oracle 如何计算 expected
QEMU 或 ssh-rvv 实际输出是什么
```

## 一、必须提交的 Compiler 测试

最低测试组合：

```text
test/Dialect/RVV/q8-<part>-dataflow.mlir
test/Conversion/EmitC/rvv-q8-<part>-materialization.mlir
test/Target/RVV/q8-<part>-artifact.mlir
```

`<part>` 可以是：

```text
i8-load
i8-widening-mul
i16-widening-reduce
selected-body
scale-dequant
full-vdot
```

最低内容：

- positive：typed `tcrv_rvv` body 能 parse/verify。
- negative：至少一个错误必须 fail closed，例如 dtype relation 错误、LMUL 不匹配、ABI role 缺失、unsupported memory form。
- target FileCheck：能看到本 slice 对应的关键 RVV intrinsic family。

完整 `q8_0_q8_0` 目标应能看到：

```text
__riscv_vle8_v_i8
__riscv_vwmul_vv_i16
__riscv_vwredsum_vs_i16
```

## 二、Runtime Proof Bundle

如果 PR 声明 runtime correctness，必须提供一组运行证据。建议目录：

```text
examples/qemu/cases/q8-<student-id-or-part>/
  README.md
  harness.cpp
```

默认不要提交 generated C++，因为它应从 MLIR fixture 生成。PR 文本中可以贴关键片段：

- generated function signature；
- `vle8`；
- `vwmul_vv_i16`；
- `vwredsum_vs_i16...i32`；
- output store 或 scalar return boundary。

## 三、Case README 模板

每个 runtime case README 建议包含：

```text
Source MLIR
Generate RVV C++
Generated Kernel Evidence
Runtime ABI
Runtime Inputs
Scalar Oracle
Run Command
Actual Output
Known Limitations
```

生成命令示例：

```bash
build/bin/tcrv-opt test/Target/RVV/q8-full-vdot-artifact.mlir \
  --tcrv-materialize-selected-lowering-boundaries \
  --tcrv-materialize-emission-plans \
| build/bin/tcrv-translate --tcrv-rvv-emitc-to-cpp \
> /tmp/tcrv-q8/generated.cpp
```

运行命令示例：

```bash
cp examples/qemu/llama_q8_0_q8_0.h /tmp/tcrv-q8/
cp examples/qemu/harness_llama_q8_0_q8_0.cpp /tmp/tcrv-q8/harness.cpp

make -C /tmp/tcrv-q8 \
  -f "$PWD/examples/qemu/Makefile.rvv" \
  run-rvv \
  RVV_GENERATED=generated.cpp \
  RVV_HARNESS=harness.cpp \
  RVV_OUTPUT=student_q8_case \
  RVV_CXX=/usr/lib/llvm-20/bin/clang++ \
  RVV_EXTRA_CXXFLAGS="--gcc-toolchain=/usr -I/usr/riscv64-linux-gnu/include/c++/14 -I/usr/riscv64-linux-gnu/include/c++/14/riscv64-linux-gnu -static" \
  RVV_EXTRA_LDFLAGS="-L/usr/lib/gcc-cross/riscv64-linux-gnu/14"
```

## 四、Harness 要求

`harness.cpp` 必须保持小而明确：

- 声明 generated kernel 的 `extern "C"` signature。
- 构造 q8 block 输入，或等价拆分数组输入。
- 说明 `n`，且 `n` 必须是 32 的整数倍。
- 初始化 output。
- 调用 generated kernel。
- 用 scalar oracle 计算 expected。
- 打印 got/expected/diff。
- mismatch 时返回非零。
- 成功时打印 `proof ok`。

如果学生改变 ABI，例如从 block struct 改成拆分数组，必须同步更新：

```text
MLIR fixture 的 ABI/runtime value binding
generated C++ function signature
harness.cpp 的 extern "C" 声明
harness.cpp 的调用参数
README 中的 runtime input/output 描述
```

## 五、教师收集口径

教师 review 时按这个顺序：

1. 跑 `cmake --build build --target check-tianchenrv`。
2. 从学生 MLIR 生成 `/tmp/tcrv-q8/generated.cpp`。
3. grep generated C++ 的 signature 和 q8 dot intrinsic。
4. 用学生 harness 或 baseline harness 编译运行 QEMU。
5. 保存 PR 链接、MLIR 路径、generated kernel evidence、harness 路径和运行输出。

如果一个 PR 只有手写 RVV C baseline，没有 MLIR/compiler 测试，不合格。如果一个 PR 只有 FileCheck，但声明 runtime correctness，却没有 harness proof，也不合格。
