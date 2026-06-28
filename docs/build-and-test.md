# 构建、测试与 Proof 流程

这篇说明怎么把 TianChen-RV 编译出来、跑 lit 测试、以及对一条 slice 跑三层正确性 proof。命令均已在本环境验证。

## 1. 前置
- CMake ≥ 3.22 + Ninja。
- **LLVM + MLIR 20**(本环境在 `/usr/lib/llvm-20`,含 `cmake/llvm`、`cmake/mlir`、`bin/{clang,clang++,mlir-translate,FileCheck}`)。
- `lit`(`pip install lit` 或随 LLVM)。
- 真机 proof(可选但推荐):一台 RVV 1.0 板,ssh 别名 `k1`(SpacemiT X60;带 `clang++`)。注:旧的 `ssh rvv` 主机当前不可用,真机走 `k1`。

## 2. 构建编译器
```bash
cmake -G Ninja -S . -B build \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
ninja -C build tcrv-opt tcrv-translate     # 两个 driver
```
`build/` 已被 `.gitignore` 忽略。`tcrv-opt`(pass driver)和 `tcrv-translate`(导出/翻译 driver)是后续所有命令要用的。

## 3. 跑 lit 测试（tier 1）
```bash
ninja -C build check-tianchenrv
# 或直接:  llvm-lit -sv build/test
```
lit 递归发现 `test/**/*.mlir`(`ShTest` + FileCheck)。你的 slice 至少要让以下三类绿:
- `test/Dialect/RVV/<cap>-dataflow.mlir` —— typed body 合法(verifier 正向)。
- `test/Conversion/RVV/rvv-to-emitc-<cap>.mlir` —— 发射的 RVV C 结构(FileCheck 关键 intrinsic)。
- 至少一个 `*-negative.mlir` —— 非法形状被 fail-closed 拒(`--verify-diagnostics`)。

## 4. 生成一条 slice 的 RVV C
```bash
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-<cap>.mlir \
  --tcrv-rvv-lower-to-emitc | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp > gen.cpp
```
入口名 = `tcrv_emitc_<kernelSym>_<variantSym>`,参数序 = `tcrv_rvv.runtime_abi_value` 声明序(详见 [`add-rvv-slice-walkthrough.md`](add-rvv-slice-walkthrough.md))。

## 5. 编译 + 跑 proof（tier 2 / tier 3）
**tier 2 本地 object 编译**(人人可做,不需 sysroot/qemu):
```bash
make -C examples/qemu -f Makefile.rvv object RVV_GENERATED=/abs/path/gen.cpp
# -> OK: ... compiles to a valid rv64gcv object
```
**tier 3 真机数值证明**(`ssh k1`):
```bash
examples/qemu/run-on-k1.sh /abs/path/gen.cpp examples/qemu/harness_<cap>.cpp <cap>
# -> ... proof ok
```
没有 k1 的同学,若装了 `qemu-riscv64` + riscv64 sysroot:
```bash
make -C examples/qemu -f Makefile.rvv qemu \
  RVV_GENERATED=/abs/path/gen.cpp RVV_HARNESS=examples/qemu/harness_<cap>.cpp \
  SYSROOT=/path/to/riscv64-sysroot
```

## 6. 日常循环建议
本地循环 = 改代码 → `ninja -C build tcrv-opt tcrv-translate check-tianchenrv`(tier 1)→ 生成 C + `make object`(tier 2)。**tier 3(k1)是正确性的最终封印**,不必每次都跑;数值有疑问或提交前再上真机。

> 提示:本树 ODS `.inc` 偶尔需重生、`tcrv-opt` 偶尔不重链;做 byte-exact 断言时用干净/强制重建最稳。
