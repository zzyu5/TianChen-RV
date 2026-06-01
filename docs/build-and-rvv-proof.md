# 构建与 RVV Proof 流程

本文说明两件事：

1. 如何编译 TianChenRV。
2. 如何运行本分支提供的 llama.cpp 风格 `q8_0_q8_0` RVV C baseline。

注意：baseline 是教师提供的正确性参照，不是学生的最终 compiler 贡献。学生最终应从 MLIR fixture 生成 RVV C/C++，再用同类 harness 验证。

## 构建 TianChenRV

### 1. 基础工具

Ubuntu / WSL 上建议安装：

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build python3 git wget
```

### 2. LLVM / MLIR

TianChenRV 走真实 MLIR compiler path，不是模拟 MLIR。课堂分支在 LLVM 20 上验证。

如果系统源提供 LLVM 20：

```bash
sudo apt install -y \
  llvm-20 llvm-20-dev llvm-20-tools llvm-20-runtime \
  clang-20 lld-20 \
  libmlir-20-dev mlir-20-tools
```

如果系统源没有 LLVM 20，可用官方 LLVM apt 仓库脚本：

```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 20

sudo apt install -y \
  llvm-20 llvm-20-dev llvm-20-tools llvm-20-runtime \
  clang-20 lld-20 \
  libmlir-20-dev mlir-20-tools
```

确认 CMake package 存在：

```bash
ls /usr/lib/llvm-20/lib/cmake/llvm
ls /usr/lib/llvm-20/lib/cmake/mlir
/usr/lib/llvm-20/bin/mlir-opt --version
```

### 3. 配置和编译项目

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir

cmake --build build
```

运行现有测试：

```bash
cmake --build build --target check-tianchenrv
```

学生完成 `q8_0_q8_0` compiler slice 后，应补充新的 MLIR/FileCheck 测试，并让这个目标通过。

## 运行 q8_0_q8_0 RVV C Baseline

本分支提供手写 RVV C baseline：

```text
examples/qemu/llama_q8_0_q8_0.h
examples/qemu/llama_q8_0_q8_0_rvv.cpp
examples/qemu/harness_llama_q8_0_q8_0.cpp
examples/qemu/Makefile.rvv
```

它验证的是 llama.cpp `q8_0_q8_0` 的核心 RVV 形状：

```text
vle8 i8
vwmul i8*i8 -> i16
vwredsum i16 -> i32
float scale accumulation
```

### 本地 QEMU 环境

安装 RISC-V userspace emulator 和 cross sysroot：

```bash
sudo apt install -y qemu-user gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
```

通用运行命令：

```bash
make -C examples/qemu \
  -f Makefile.rvv \
  run-rvv \
  RVV_GENERATED=llama_q8_0_q8_0_rvv.cpp \
  RVV_HARNESS=harness_llama_q8_0_q8_0.cpp \
  RVV_OUTPUT=llama_q8_0_q8_0_case
```

如果默认 `clang++` 找不到 RISC-V sysroot，可以覆盖变量。下面这组配置在课堂同学机器上验证过：

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

成功输出应包含：

```text
llama q8_0_q8_0 classroom baseline
rvv classroom llama q8_0_q8_0 proof ok
```

本分支提交前的真实 `ssh rvv` 运行输出：

```text
llama q8_0_q8_0 classroom baseline
n=160 blocks=5 qk=32
block[0]: dot_i32=-268 x.d=0.03515625 y.d=0.04882812
block[1]: dot_i32=3134 x.d=0.03906250 y.d=0.05078125
block[2]: dot_i32=457 x.d=0.04296875 y.d=0.05273438
got=4.99381256 expected=4.99381256 diff=0.00000000
rvv classroom llama q8_0_q8_0 proof ok
```

### 真实 `ssh rvv` 环境

维护者可以在真实 RVV 主机上跑同一份 baseline：

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

如果 `rvv` 主机没有 `clang++`，可尝试 `g++`：

```bash
ssh rvv 'cd /tmp/tcrv-classroom-q8 && \
  g++ -std=c++17 -O2 -march=rv64gcv -mabi=lp64d \
    llama_q8_0_q8_0_rvv.cpp harness_llama_q8_0_q8_0.cpp \
    -o llama_q8_0_q8_0_case && \
  ./llama_q8_0_q8_0_case'
```

## 学生最终 Proof 形状

学生 PR 最终应该把 baseline 中的手写 RVV C 替换为 TianChenRV 生成的 RVV C/C++：

```text
student MLIR fixture
  -> tcrv-opt / tcrv-translate
  -> generated.cpp
  + harness_llama_q8_0_q8_0.cpp 或学生等价 harness
  -> qemu-riscv64 或 ssh rvv 运行
```

生成命令形状应类似：

```bash
build/bin/tcrv-opt test/Target/RVV/<student-q8-vdot>.mlir \
  --tcrv-materialize-selected-lowering-boundaries \
  --tcrv-materialize-emission-plans \
| build/bin/tcrv-translate --tcrv-rvv-emitc-to-cpp \
> /tmp/tcrv-q8/generated.cpp
```

如果学生的 fixture 不使用 pre-realized selected-body，可以省略 selected-boundary materialization pass，但必须说明原因。

## Harness 要求

无论是 baseline 还是学生 generated kernel，runtime harness 都要说明：

- block 数量和 `n`；
- 每个 block 的 `d` 和 `qs[32]` 如何生成；
- scalar oracle；
- expected/got/diff；
- QEMU 或 `ssh rvv` 实际输出。

如果学生改变 kernel ABI，例如把 block struct 拆成 `int8_t *x_qs`、`int8_t *y_qs`、`float *x_d`、`float *y_d`，必须同步更新：

```text
MLIR runtime ABI value binding
generated function signature
harness extern "C" declaration
harness call site
PR 文档里的输入输出说明
```
