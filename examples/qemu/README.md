# 本地 RVV QEMU 示例

这个目录不是 lit test 的替代品，而是给学生看的最小 runtime proof 模板。它展示一条真实路径：

```text
MLIR fixture
  -> tcrv-opt / tcrv-translate
  -> generated.cpp，里面只有 exported RVV kernel
  -> harness.cpp，提供具体输入、输出 buffer、oracle 和 main
  -> riscv64 executable
  -> qemu-riscv64 运行
```

`generated.cpp` 来自 TianChenRV 编译器。`harness.cpp` 是学生为自己的 slice 写的运行证明。MLIR 输入一定是 `.mlir`，编译器输出一定是 RVV C/C++；具体输入数组、输出 buffer 和检查逻辑不在 generated kernel 里，而在 harness 里。

## 环境

Ubuntu / WSL 上建议安装：

```bash
sudo apt update
sudo apt install -y qemu-user gcc-riscv64-linux-gnu g++-riscv64-linux-gnu
```

本目录的 Makefile 默认使用 LLVM 20 的 clang：

```make
RVV_CXX ?= /usr/lib/llvm-20/bin/clang++
```

如果你的 clang 或 sysroot 不在默认位置，运行 `make` 时覆盖变量即可：

```bash
make -f /path/to/examples/qemu/Makefile.rvv print-rvv-config \
  RVV_CXX=/your/clang++ \
  SYSROOT=/your/riscv64/sysroot
```

## 运行 add 示例

从仓库根目录执行：

```bash
mkdir -p /tmp/tcrv-rvv-qemu-add

build/bin/tcrv-opt test/Target/RVV/emitc-to-cpp-handoff.mlir \
  --tcrv-materialize-emission-plans \
| build/bin/tcrv-translate --tcrv-rvv-emitc-to-cpp \
> /tmp/tcrv-rvv-qemu-add/generated.cpp

cp examples/qemu/harness_add.cpp /tmp/tcrv-rvv-qemu-add/harness.cpp

make -C /tmp/tcrv-rvv-qemu-add \
  -f "$PWD/examples/qemu/Makefile.rvv" \
  run-rvv
```

你应该能看到类似输出：

```text
lane[0]: -17 + 41 = 24
lane[1]: -14 + 39 = 25
...
rvv classroom proof ok: 37 lanes checked
```

这里的实际 kernel 是 `generated.cpp` 里的：

```text
tcrv_emitc_rvv_i32_add_kernel_rvv_i32_add(...)
```

`harness_add.cpp` 只负责构造 `lhs/rhs/out/n`，调用这个 kernel，然后用 scalar oracle 检查结果。

## 运行 xor 示例

`xor` 是本课堂分支已经完成的参考 slice。它使用不同的 MLIR fixture 和 harness：

```bash
mkdir -p /tmp/tcrv-rvv-qemu-xor

build/bin/tcrv-opt test/Target/RVV/emitc-to-cpp-xor.mlir \
  --tcrv-materialize-emission-plans \
| build/bin/tcrv-translate --tcrv-rvv-emitc-to-cpp \
> /tmp/tcrv-rvv-qemu-xor/generated.cpp

cp examples/qemu/harness_xor.cpp /tmp/tcrv-rvv-qemu-xor/harness.cpp

make -C /tmp/tcrv-rvv-qemu-xor \
  -f "$PWD/examples/qemu/Makefile.rvv" \
  run-rvv
```

你应该能在 `generated.cpp` 中看到：

```text
__riscv_vxor_vv_i32m1
```

运行输出会打印前几个 lane 的输入和结果，最后给出：

```text
rvv classroom xor proof ok: 41 lanes checked
```

## 学生需要自己写什么

学生做一个新 slice 时通常需要自己写或修改三类东西：

1. MLIR fixture：描述 `tcrv.exec` envelope、selected RVV variant、typed `tcrv_rvv` body、ABI/runtime value binding 和具体 op。
2. compiler/provider 改动：让 RVV plugin 从 typed body/config/capability/runtime facts 推导 route、C vector type、header 和 intrinsic。
3. harness：为 generated kernel 提供具体输入、输出 buffer、`n`、mask/index/segment 等 runtime 数据，并写 scalar oracle。

如果只是想换一组输入数据，改 `harness_*.cpp` 里的 `make_lhs`、`make_rhs`、`TCRV_N` 和 oracle 即可。如果改变 kernel ABI，例如多一个 scalar、mask、index buffer、accumulator 或输出 buffer，那么 MLIR fixture 的 ABI role、generated function 的 C signature、harness 的 `extern "C"` 声明和调用参数都要同步修改。

不要把 QEMU harness 写成新的 compiler pass，也不要让 common EmitC 因为某个 harness 需求去理解 RVV 语义。QEMU proof 只证明 generated kernel 在一组具体 runtime 输入上行为正确。
