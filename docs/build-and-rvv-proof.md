# 构建与 RVV Proof 流程

本文记录本分支的构建方式，以及一个最小 RVV proof 流程。QEMU 是学生 slice 的默认运行证据，不是 slice 本身，也不需要接入项目默认测试目标。维护者也可以用真实 `ssh rvv` 环境跑同一份 generated C++ 和 harness；在运行 RVV 程序之前，编译器生成路径是一致的。

## 构建 TianChenRV

### 1. 安装基础工具

Ubuntu / WSL 上建议先装基础构建工具：

```bash
sudo apt update
sudo apt install -y build-essential cmake ninja-build python3 git wget
```

### 2. 安装 LLVM / MLIR

TianChenRV 是真实 MLIR compiler path，不是自己模拟 MLIR。学生需要安装 LLVM/MLIR 工具链和 CMake package。课堂分支已在 LLVM 20 上验证。

如果系统 apt 源已经提供 LLVM 20，可以直接尝试：

```bash
sudo apt install -y \
  llvm-20 llvm-20-dev llvm-20-tools llvm-20-runtime \
  clang-20 lld-20 \
  libmlir-20-dev mlir-20-tools
```

如果系统源没有 LLVM 20，使用官方 LLVM apt 仓库脚本。官方入口是 <https://apt.llvm.org/>：

```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 20

sudo apt install -y \
  llvm-20 llvm-20-dev llvm-20-tools llvm-20-runtime \
  clang-20 lld-20 \
  libmlir-20-dev mlir-20-tools
```

安装后确认路径存在：

```bash
ls /usr/lib/llvm-20/lib/cmake/llvm
ls /usr/lib/llvm-20/lib/cmake/mlir
/usr/lib/llvm-20/bin/mlir-opt --version
```

如果你使用的是源码编译 LLVM/MLIR，也可以，只要 `LLVM_DIR` 和 `MLIR_DIR` 指向对应的 CMake package 目录。

### 3. 配置项目

准备 LLVM/MLIR CMake package 后配置：

```bash
cmake -S . -B build -G Ninja \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
```

编译：

```bash
cmake --build build
```

运行本分支精简后的 RVV classroom 测试：

```bash
cmake --build build --target check-tianchenrv
```

如果 LLVM 安装路径不同，请把 `LLVM_DIR` 和 `MLIR_DIR` 指到对应位置。本项目的核心编译器实现使用 C++ / MLIR / LLVM / TableGen / CMake；Python 只应作为辅助脚本或外部 proof 工具，不应替代 compiler core。

## 生成 RVV Intrinsic C++

最短的 compiler-path proof 是把一个已有 RVV fixture materialize 成 RVV intrinsic C++：

```bash
build/bin/tcrv-opt test/Target/RVV/emitc-to-cpp-handoff.mlir \
  --tcrv-materialize-emission-plans \
| build/bin/tcrv-translate --tcrv-rvv-emitc-to-cpp \
> /tmp/tcrv-rvv-add.cpp
```

`/tmp/tcrv-rvv-add.cpp` 里应该能看到：

```text
#include <riscv_vector.h>
__riscv_vsetvl_...
__riscv_vle...
__riscv_vadd...
__riscv_vse...
```

对于 pre-realized selected-body fixture，需要先 materialize selected lowering boundary：

```bash
build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-add.mlir \
  --tcrv-materialize-selected-lowering-boundaries \
  --tcrv-materialize-emission-plans \
| build/bin/tcrv-translate --tcrv-rvv-emitc-to-cpp \
> /tmp/tcrv-rvv-pre-realized-add.cpp
```

## 可选 QEMU Proof

学生可以用任意等价的 RISC-V RVV toolchain / QEMU 设置来证明自己的 slice。推荐 proof 形状：

```text
generated RVV C/C++
  + 小型 slice-specific harness
  -> riscv64 executable
  -> qemu-riscv64 run
  -> PR 中记录命令和输出
```

本仓库提供一个可选 Makefile 片段：

```bash
cp examples/qemu/Makefile.rvv /tmp/Makefile.rvv
```

QEMU proof 需要 RISC-V userspace emulator 和 cross sysroot。Ubuntu / WSL 上通常需要：

```bash
sudo apt install -y qemu-user gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
```

本仓库也提供最小 add/xor harness 示例。add 示例：

```bash
build/bin/tcrv-opt test/Target/RVV/emitc-to-cpp-handoff.mlir \
  --tcrv-materialize-emission-plans \
| build/bin/tcrv-translate --tcrv-rvv-emitc-to-cpp \
> /tmp/generated.cpp

cp examples/qemu/harness_add.cpp /tmp/harness.cpp
```

在包含 `generated.cpp` 和 `harness.cpp` 的目录中可以这样运行：

```bash
make -f /tmp/Makefile.rvv run-rvv \
  RVV_CXX=/usr/lib/llvm-20/bin/clang++ \
  QEMU_RISCV64=qemu-riscv64 \
  SYSROOT=/usr/riscv64-linux-gnu
```

`examples/qemu/Makefile.rvv` 默认使用课堂验证过的 clang/riscv64 参数：

```make
RVV_CXX ?= /usr/lib/llvm-20/bin/clang++
RVV_CXXFLAGS ?= --target=riscv64-linux-gnu \
                -std=c++17 \
                -O2 \
                -march=rv64gcv \
                -mabi=lp64d \
                --sysroot=$(SYSROOT) \
                --gcc-toolchain=/usr \
                -I/usr/riscv64-linux-gnu/include/c++/14 \
                -I/usr/riscv64-linux-gnu/include/c++/14/riscv64-linux-gnu \
                -L/usr/lib/gcc-cross/riscv64-linux-gnu/14 \
                -static
```

如果本机工具链路径不同，可以替换 `RVV_CXX`、`QEMU_RISCV64`、`SYSROOT`，或者覆盖 `RVV_CXXFLAGS`。PR 只要声明 runtime correctness，就应该记录实际编译和 QEMU 运行命令。

更完整的 add/xor 示例见 [本地 RVV QEMU 示例](../examples/qemu/README.md)。

## 维护者 ssh rvv Proof

如果你有项目维护者提供的 `ssh rvv` 环境，可以把 QEMU 的最后一步替换成真实 RISC-V 主机运行。流程仍然是：

```text
本机 tcrv-opt / tcrv-translate 生成 RVV C++
  -> 复制 generated.cpp 和 harness.cpp 到 rvv 主机
  -> 在 rvv 主机用 clang++ 或 g++ 编译
  -> 运行 executable
```

示例：

```bash
build/bin/tcrv-opt test/Target/RVV/emitc-to-cpp-handoff.mlir \
  --tcrv-materialize-emission-plans \
| build/bin/tcrv-translate --tcrv-rvv-emitc-to-cpp \
> /tmp/tcrv-rvv-generated.cpp

scp /tmp/tcrv-rvv-generated.cpp rvv:/tmp/
scp examples/qemu/harness_add.cpp rvv:/tmp/tcrv-rvv-harness.cpp

ssh rvv 'clang++ -std=c++17 -O2 -march=rv64gcv -mabi=lp64d \
  /tmp/tcrv-rvv-generated.cpp /tmp/tcrv-rvv-harness.cpp \
  -o /tmp/tcrv-rvv-proof && /tmp/tcrv-rvv-proof'
```

这个 proof 是维护者参考路径。学生提交 PR 时仍然优先使用本地 QEMU，除非课程明确提供真实 RVV 主机。

## Harness 要求

一个基本 harness 应该：

- 分配输入/输出 buffer；
- 对 output、tail lane、inactive lane 使用 sentinel 初始化；
- 调用生成的 TianChenRV 函数；
- 用 scalar oracle 对比结果；
- mismatch 时返回非零状态。

masked、tail、indexed、segment、widening slice 应该包含能区分正确 RVV 行为和弱实现的测试数据。

## 最小 Harness 约定

生成的 RVV C++ 通常只包含 exported kernel 函数，不包含 `main`。harness 负责提供输入和 oracle。建议每个 slice 的 harness 只验证一个行为：

```text
准备输入
初始化 output sentinel
调用 generated kernel
用 scalar oracle 检查输出
返回 0 或非 0
```

不要把 QEMU harness 做成新的项目框架；它应该只是 PR 里的运行证据。

学生需要注意：generated RVV C++ 只包含 exported kernel 函数，不包含 `main`，也不会自己构造测试矩阵或数组。具体输入、输出、mask、index、segment 数据和 scalar oracle 都由 harness 提供。如果 slice 修改了 kernel ABI，MLIR fixture、generated function signature、harness 的 `extern "C"` 声明和调用参数必须一起更新。
