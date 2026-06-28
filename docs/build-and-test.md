# 构建、测试与验证流程

本文说明如何构建 TianChen-RV、运行 lit 测试，以及对一条 slice 做三层正确性验证。

## 1. 前置

- CMake ≥ 3.22 与 Ninja。
- LLVM + MLIR 20（本环境位于 `/usr/lib/llvm-20`，含 `cmake/llvm`、`cmake/mlir`、`bin/{clang,clang++,mlir-translate,FileCheck}`）。
- `lit`（`pip install lit`，或随 LLVM 提供）。
- tier 3 数值验证：`qemu-riscv64` 与一份 riscv64 sysroot。

## 2. 构建编译器

```bash
cmake -G Ninja -S . -B build \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
ninja -C build tcrv-opt tcrv-translate
```

`build/` 已在 `.gitignore` 中忽略。`tcrv-opt`（pass driver）与 `tcrv-translate`（导出 / 翻译 driver）是后续命令所依赖的两个二进制。

## 3. 运行 lit 测试（tier 1）

```bash
ninja -C build check-tianchenrv
# 或：  llvm-lit -sv build/test
```

lit 递归发现 `test/**/*.mlir`（`ShTest` + FileCheck）。一条 slice 至少需要让以下三类测试通过：

- `test/Dialect/RVV/<cap>-dataflow.mlir` —— typed body 合法（verifier 正向）。
- `test/Conversion/RVV/rvv-to-emitc-<cap>.mlir` —— 发射的 RVV C 结构（FileCheck 关键 intrinsic）。
- 至少一个 `*-negative.mlir` —— 非法形状被 fail-closed 拒绝（`--verify-diagnostics`）。

## 4. 生成一条 slice 的 RVV C

```bash
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-<cap>.mlir \
  --tcrv-rvv-lower-to-emitc | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp > gen.cpp
```

入口名为 `tcrv_emitc_<kernelSym>_<variantSym>`，参数顺序为 `tcrv_rvv.runtime_abi_value` 的声明顺序（见 [`add-rvv-slice-walkthrough.md`](add-rvv-slice-walkthrough.md)）。

## 5. 编译与数值验证（tier 2 / tier 3）

tier 2，本地 object 编译（不需要 sysroot 或 qemu），验证生成的 C 是合法的 RVV C：

```bash
make -C examples/qemu -f Makefile.rvv object RVV_GENERATED=/abs/path/gen.cpp
# -> OK: ... compiles to a valid rv64gcv object
```

tier 3，数值验证，将生成的 kernel 与 harness 编成 riscv64 可执行文件，用 `qemu-riscv64` 运行，对 harness 内的 scalar 参考实现逐元素比对：

```bash
make -C examples/qemu -f Makefile.rvv qemu \
  RVV_GENERATED=/abs/path/gen.cpp RVV_HARNESS=examples/qemu/harness_<cap>.cpp \
  SYSROOT=/path/to/riscv64-sysroot
# 通过时打印： <cap> proof ok
```

## 6. 日常循环

改代码 → `ninja -C build tcrv-opt tcrv-translate check-tianchenrv`（tier 1）→ 生成 C 并 `make object`（tier 2）。tier 3 数值验证在数值有疑问或提交前运行。

> 提示：本树 ODS 的 `.inc` 偶尔需要重新生成、`tcrv-opt` 偶尔不重新链接；做 byte-exact 断言时用彻底 / 强制重建较为稳妥。
