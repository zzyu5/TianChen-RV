# TianChen-RV 课堂分支：RVV 编译器 Slice 作业

TianChen-RV 是一个面向 RISC-V 的 MLIR 编译框架，从 typed 的 `tcrv_rvv` 中间表示生成 RISC-V Vector（RVV）kernel 的 C 代码。本分支用于课程：每位同学认领一条 **slice**，为编译器增加一项它当前尚未支持的 RVV kernel 生成能力。

编译链路：

```text
typed tcrv_rvv body (.mlir)
  → selected-body realization
  → provider EmitC route
  → RVVToEmitC emitter
  → 生成的 RVV C
  → harness 对 scalar 参考实现做数值验证
```

一条 slice 是一次完整的编译器改动，而不是手写一个 RVV C 函数：需要扩展 typed IR 与 verifier、realization、provider route 与 emitter，并给出测试与数值验证。12 条 slice 各自对应一项当前编译器无法生成的能力，彼此独立。

## 阅读顺序

1. [Worked example：完整 slice `add`](docs/add-rvv-slice-walkthrough.md) —— 已实现的最小 slice，端到端走通整条链路，作为所有作业的模板。
2. [构建、测试与验证流程](docs/build-and-test.md) —— 如何构建编译器、运行 lit 测试、对一条 slice 做三层验证。
3. [RVV slice 的模块落点](docs/rvv-slice-module-map.md) —— 一条 slice 涉及哪些层、落在哪些文件。
4. [RVV slice 贡献指南](docs/rvv-slice-contribution-guide.md) —— 推荐实现顺序与 PR 检查项。
5. [12 个 slice 作业](assignments/rvv-slices-12.md) —— 任务清单，每条一页（目标、当前实现现状、涉及文件、scalar oracle、验收）。
6. [测试用例与验收格式](docs/testcase-submission-format.md) —— 提交格式与验收口径。
7. [验证 harness 目录](examples/qemu/README.md) —— `add` harness、Makefile 与运行说明。

## 构建

```bash
cmake -G Ninja -S . -B build \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm \
  -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
ninja -C build tcrv-opt tcrv-translate
```

依赖 LLVM/MLIR 20（本环境位于 `/usr/lib/llvm-20`）。

## 测试

```bash
ninja -C build check-tianchenrv     # lit / FileCheck
```

`add` 范例端到端：生成 C，再本地编成 rv64gcv object（tier 2）：

```bash
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-add-cpp-golden.mlir \
  --tcrv-rvv-lower-to-emitc | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp \
  > examples/qemu/add_generated.reference.cpp
make -C examples/qemu -f Makefile.rvv object RVV_GENERATED=add_generated.reference.cpp
```

数值验证（tier 3）通过 `qemu-riscv64` 运行生成的 kernel 与 harness，对 scalar 参考实现逐元素比对，详见 [build-and-test.md](docs/build-and-test.md)。

## 仓库结构

```text
include/TianChenRV/   公共头文件与 TableGen dialect / op 定义
lib/                  编译器实现（Dialect/RVV、Plugin/RVV、Conversion/RVV+EmitC、Target、Transforms、Support）
tools/tcrv-opt/       pass driver
tools/tcrv-translate/ translation / export driver
test/                 lit / FileCheck 测试
docs/                 walkthrough、构建测试、模块落点、贡献指南、验收格式
assignments/          12 个 slice 作业
examples/qemu/        add 范例 harness、Makefile 与运行说明
```

## 实现纪律（所有 slice 通用，对应 I3/I4/I5）

语义只能来自 op identity 与 typed facts（element type、SEW、LMUL、policy、unit-stride 等 memory form、op-kind、runtime-ABI、target capability）。不得：

- 以 dtype 前缀命名 helper（如 `tcrv_rvv.i8_*`）；
- 从 route id、artifact 名、C 参数名、test 名反推计算语义；
- 在公共 EmitC 路径中硬编码 RVV 的 dtype / SEW / LMUL / intrinsic；
- 为通过 harness 而绕过 typed `tcrv_rvv` body 或 provider。

每个 PR 需说明：typed facts 在何处、ABI / runtime value 如何绑定、RVV provider 如何据此推导 route。
