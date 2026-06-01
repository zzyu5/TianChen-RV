# llama.cpp `q8_0_q8_0` RVV Baseline

这个目录提供本轮课程的 runtime proof 基准。它不是 TianChenRV generated C，也不是学生最终答案，而是用于说明目标 kernel 的输入、输出、RVV intrinsic 形状和 scalar oracle。

文件：

```text
llama_q8_0_q8_0.h
llama_q8_0_q8_0_rvv.cpp
harness_llama_q8_0_q8_0.cpp
harness_add.cpp
harness_xor.cpp
Makefile.rvv
```

`llama_q8_0_q8_0_*` 是本轮主任务的 baseline。`harness_add.cpp` 和 `harness_xor.cpp` 是参考例子，用来展示一个 generated kernel 如何被普通 C++ harness 调用和验证。

## Baseline 语义

每个 block：

```text
tcrv_classroom_block_q8_0:
  d:  float scale
  qs: int8[32]
```

计算：

```text
out = sum over blocks:
        sum_lanes(int32(x.qs[lane]) * int32(y.qs[lane])) * x.d * y.d
```

核心 RVV 形状：

```c
vint8m2_t vx = __riscv_vle8_v_i8m2(x[ib].qs, vl);
vint8m2_t vy = __riscv_vle8_v_i8m2(y[ib].qs, vl);
vint16m4_t prod = __riscv_vwmul_vv_i16m4(vx, vy, vl);
vint32m1_t reduced = __riscv_vwredsum_vs_i16m4_i32m1(prod, zero, vl);
```

学生后续生成的 RVV C 不需要逐字符相同，但必须能解释自己的 vector type、LMUL、reduction boundary 和 ABI 为什么等价。

## 本地 QEMU 运行

安装：

```bash
sudo apt update
sudo apt install -y qemu-user gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
```

通用命令：

```bash
make -C examples/qemu \
  -f Makefile.rvv \
  run-rvv \
  RVV_GENERATED=llama_q8_0_q8_0_rvv.cpp \
  RVV_HARNESS=harness_llama_q8_0_q8_0.cpp \
  RVV_OUTPUT=llama_q8_0_q8_0_case
```

如果需要显式指定 LLVM 20 clang 和 cross sysroot：

```bash
make -C examples/qemu \
  -f Makefile.rvv \
  run-rvv \
  RVV_GENERATED=llama_q8_0_q8_0_rvv.cpp \
  RVV_HARNESS=harness_llama_q8_0_q8_0.cpp \
  RVV_OUTPUT=llama_q8_0_q8_0_case \
  RVV_CXX=/usr/lib/llvm-20/bin/clang++ \
  RVV_EXTRA_CXXFLAGS="--gcc-toolchain=/usr -I/usr/riscv64-linux-gnu/include/c++/14 -I/usr/riscv64-linux-gnu/include/c++/14/riscv64-linux-gnu -static" \
  RVV_EXTRA_LDFLAGS="-L/usr/lib/gcc-cross/riscv64-linux-gnu/14"
```

成功时会看到：

```text
llama q8_0_q8_0 classroom baseline
n=160 blocks=5 qk=32
block[0]: dot_i32=...
got=... expected=... diff=...
rvv classroom llama q8_0_q8_0 proof ok
```

本分支提交前在真实 `ssh rvv` 主机上用 `clang++ -O2 -march=rv64gcv -mabi=lp64d` 验证过，输出为：

```text
llama q8_0_q8_0 classroom baseline
n=160 blocks=5 qk=32
block[0]: dot_i32=-268 x.d=0.03515625 y.d=0.04882812
block[1]: dot_i32=3134 x.d=0.03906250 y.d=0.05078125
block[2]: dot_i32=457 x.d=0.04296875 y.d=0.05273438
got=4.99381256 expected=4.99381256 diff=0.00000000
rvv classroom llama q8_0_q8_0 proof ok
```

## 真实 `ssh rvv` 运行

维护者可直接在真实 RVV 主机上运行：

```bash
ssh rvv 'mkdir -p /tmp/tcrv-classroom-q8'
scp examples/qemu/llama_q8_0_q8_0.h \
    examples/qemu/llama_q8_0_q8_0_rvv.cpp \
    examples/qemu/harness_llama_q8_0_q8_0.cpp \
    rvv:/tmp/tcrv-classroom-q8/

ssh rvv 'cd /tmp/tcrv-classroom-q8 && \
  clang++ -std=c++17 -O2 -march=rv64gcv -mabi=lp64d \
    llama_q8_0_q8_0_rvv.cpp harness_llama_q8_0_q8_0.cpp \
    -o llama_q8_0_q8_0_case && \
  ./llama_q8_0_q8_0_case'
```

## 学生如何使用这个目录

学生应该把这里当作三个东西：

1. 目标 kernel 形状：`vle8`、`vwmul_vv_i16`、`vwredsum_vs_i16...i32`。
2. 输入输出约定：q8 block、scale、`n`、float output。
3. 正确性 oracle：`harness_llama_q8_0_q8_0.cpp` 中的 scalar reference。

学生完成 compiler slice 后，应把自己的 generated C/C++ 放入临时目录，使用相同或等价 harness 运行：

```bash
mkdir -p /tmp/tcrv-student-q8
build/bin/tcrv-opt test/Target/RVV/<student-q8-vdot>.mlir \
  --tcrv-materialize-selected-lowering-boundaries \
  --tcrv-materialize-emission-plans \
| build/bin/tcrv-translate --tcrv-rvv-emitc-to-cpp \
> /tmp/tcrv-student-q8/generated.cpp

cp examples/qemu/llama_q8_0_q8_0.h /tmp/tcrv-student-q8/
cp examples/qemu/harness_llama_q8_0_q8_0.cpp /tmp/tcrv-student-q8/harness.cpp

make -C /tmp/tcrv-student-q8 \
  -f "$PWD/examples/qemu/Makefile.rvv" \
  run-rvv \
  RVV_GENERATED=generated.cpp \
  RVV_HARNESS=harness.cpp \
  RVV_OUTPUT=student_q8_case
```

如果学生生成的 function signature 与 baseline 不同，必须同步修改 harness 的 `extern "C"` 声明和 call site，并在 PR 中解释 ABI 差异。

## add/xor 参考 harness

`harness_add.cpp` 和 `harness_xor.cpp` 用来展示一个普通 generated kernel 怎么被 harness 调用。它们不是 `q8_0_q8_0` baseline，但很适合学生先看懂“MLIR 生成 kernel，harness 提供输入和 oracle”的工作方式。

add 示例：

```bash
mkdir -p /tmp/tcrv-rvv-add
build/bin/tcrv-opt test/Target/RVV/emitc-to-cpp-handoff.mlir \
  --tcrv-materialize-emission-plans \
| build/bin/tcrv-translate --tcrv-rvv-emitc-to-cpp \
> /tmp/tcrv-rvv-add/generated.cpp

cp examples/qemu/harness_add.cpp /tmp/tcrv-rvv-add/harness.cpp
make -C /tmp/tcrv-rvv-add \
  -f "$PWD/examples/qemu/Makefile.rvv" \
  run-rvv
```

xor 示例：

```bash
mkdir -p /tmp/tcrv-rvv-xor
build/bin/tcrv-opt test/Target/RVV/emitc-to-cpp-xor.mlir \
  --tcrv-materialize-emission-plans \
| build/bin/tcrv-translate --tcrv-rvv-emitc-to-cpp \
> /tmp/tcrv-rvv-xor/generated.cpp

cp examples/qemu/harness_xor.cpp /tmp/tcrv-rvv-xor/harness.cpp
make -C /tmp/tcrv-rvv-xor \
  -f "$PWD/examples/qemu/Makefile.rvv" \
  run-rvv
```

如果本机 RISC-V toolchain 需要显式参数，沿用前面 `RVV_CXX`、`RVV_EXTRA_CXXFLAGS` 和 `RVV_EXTRA_LDFLAGS` 的覆盖方式。
