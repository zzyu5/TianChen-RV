# 构建与 RVV Proof 流程

本文记录本分支的构建方式，以及一个最小 RVV proof 流程。QEMU 是 slice 的运行证据，不是 slice 本身，也不需要接入项目默认测试目标。

## 构建 TianChenRV

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

在包含 `generated.cpp` 和 `harness.cpp` 的目录中可以这样运行：

```bash
make -f /tmp/Makefile.rvv run-rvv \
  RVV_CXX=/usr/lib/llvm-20/bin/clang++ \
  QEMU_RISCV64=qemu-riscv64 \
  SYSROOT=/usr/riscv64-linux-gnu
```

如果本机工具链路径不同，可以替换 `RVV_CXX`、`QEMU_RISCV64`、`SYSROOT`。PR 只要声明 runtime correctness，就应该记录实际编译和 QEMU 运行命令。

## Harness 要求

一个基本 harness 应该：

- 分配输入/输出 buffer；
- 对 output、tail lane、inactive lane 使用 sentinel 初始化；
- 调用生成的 TianChenRV 函数；
- 用 scalar oracle 对比结果；
- mismatch 时返回非零状态。

masked、tail、indexed、segment、widening slice 应该包含能区分正确 RVV 行为和弱实现的测试数据。
