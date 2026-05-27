# TianChen-RV MLIR

TianChen-RV 是一个基于 MLIR 的 RISC-V 执行层项目。本分支面向学生和外部贡献者，目标是围绕 RVV 做一个个边界清晰的 slice 贡献。

当前贡献主线是：

- 低层、typed 的 `tcrv_rvv` IR；
- RVV plugin 负责 legality、selected-body realization、route derivation；
- common EmitC 只负责 materialization；
- 最终生成 RVV intrinsic C/C++；
- 如声明运行正确性，再用本地 RVV QEMU 或等价环境做证明。

本分支已经移除内部自动化和 steering 文件，只保留学生写代码、写 MLIR、跑测试、生成 RVV C++ 所需的工程面。

## 阅读顺序

建议按下面顺序读文档。每份文档只负责一件事，先理解路径，再选题，再补测试和 runtime proof。

1. [构建与 RVV Proof 流程](docs/build-and-rvv-proof.md)
   先把项目编译起来，确认 LLVM/MLIR、`tcrv-opt`、`tcrv-translate`、lit/FileCheck 和可选 QEMU proof 路径。
2. [RVV Slice 贡献指南](docs/rvv-slice-contribution-guide.md)
   理解一个 slice 从 typed `tcrv_rvv` body 到 RVV provider route、EmitC、RVV C++ 的完整贡献边界。
3. [从零新增一个 RVV Slice：xor 示例](docs/add-rvv-xor-slice-workflow.md)
   对照本分支已经完成的 `xor` slice，看一次真实代码改动、测试改动和 generated C++ 证据。
4. [25 个 RVV Slice 任务](assignments/rvv-slices-25.md)
   查看可分配题目。题目按随机分配准备，每个题都要求完整 compiler slice。
5. [RVV Slice 模块化落点](docs/rvv-slice-module-map.md)
   开始动手前，用它判断应该改 IR surface、selected-body realization、provider route、ABI role 还是测试。
6. [测试用例收集格式](docs/testcase-submission-format.md)
   提 PR 前按统一格式整理 MLIR、generated kernel evidence、harness 输入输出和 QEMU 输出。
7. [本地 RVV QEMU 示例](examples/qemu/README.md)
   需要 runtime proof 时，参考 add/xor 的 generated kernel + harness + Makefile 路径。

## 当前 RVV 路径

一个合格的 RVV slice 应该沿着下面这条链路推进：

```text
tcrv.exec envelope
  -> selected RVV variant
  -> typed low-level tcrv_rvv body
  -> RVV plugin legality / selected-body realization / route provider
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> RVV intrinsic C/C++ 或等价后端表示
  -> 可选本地 RVV QEMU proof
```

`tcrv.exec` 只负责 kernel 组织、variant、dispatch/fallback、capability scope、diagnostic 和 ABI role declaration。真正的 RVV 计算、dtype、SEW/LMUL/policy、load/store、mask、reduction、intrinsic 选择，都必须来自 typed `tcrv_rvv` body 和 RVV plugin。

不要新增这些旧路线：

- `tcrv_rvv.i32_*` 这种 dtype-prefixed helper；
- `RVVI32M1*` / `rvv-i32m1` route-id 驱动的语义；
- source-front-door / source-artifact 正向路径；
- common EmitC 中硬编码 RVV dtype、SEW/LMUL、intrinsic 或 schedule。

## 仓库结构

```text
include/TianChenRV/        Public headers 和 TableGen dialect/op 定义
lib/                       编译器实现
tools/tcrv-opt/            pass driver
tools/tcrv-translate/      translation/export driver
test/                      精简后的 RVV lit/FileCheck 测试
docs/                      贡献说明
assignments/               25 个 RVV slice 任务清单
examples/qemu/             可选本地 QEMU proof Makefile 片段
```

## 构建

配置 LLVM/MLIR：

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

更多命令见 [构建与 RVV Proof 流程](docs/build-and-rvv-proof.md)。

## 看一个真实 Kernel 运行

lit/FileCheck 只能证明 compiler 输出形状正确；学生还需要理解 generated kernel 怎么被普通程序调用。本分支提供了一个本地 RVV QEMU 示例：

- MLIR 输入：`test/Target/RVV/emitc-to-cpp-handoff.mlir` 或 `test/Target/RVV/emitc-to-cpp-xor.mlir`
- compiler 输出：`generated.cpp`，里面只有 `extern "C"` RVV kernel，没有 `main`
- runtime 输入/输出：`examples/qemu/harness_add.cpp` 或 `examples/qemu/harness_xor.cpp`
- 编译运行：`examples/qemu/Makefile.rvv`

快速路径：

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

学生要改具体输入矩阵、数组、mask、index、`n` 或 oracle，改 harness。学生要改 kernel ABI 或 RVV 行为，先改 MLIR fixture 和 compiler/provider 路径，再同步 harness 里的 `extern "C"` 声明和调用参数。完整说明见 [本地 RVV QEMU 示例](examples/qemu/README.md)。

## 贡献边界

每个 PR 应该只负责一个 RVV slice。一个合格的 slice 通常包含：

- typed `tcrv_rvv` IR 或对已有 generic op 的小范围扩展；
- verifier / negative coverage；
- RVV plugin 从 typed body/config/runtime/capability facts 推导 route；
- EmitC / RVV C++ FileCheck；
- 如声明 runtime correctness，则给出本地 RVV QEMU 命令和输出。

保持改动窄而清楚。不要把 frontend、source-front-door、Toy/Template/TensorExtLite、远程测试基础设施或大 runtime 工程塞进一个 RVV slice。

## 当前不建议学生触碰的区域

主开发分支仍在持续推进 Stage2 RVV coverage。课堂任务暂时不要选择：

- `segment2` route-family planning owner 或 route-entry registry；
- `segment3` / `segment4` / `segmentN` 泛化；
- source-front-door / source-artifact positive route；
- common EmitC RVV semantic branch；
- Toy / Template / TensorExtLite / Offload 相关工作。

如果一个 PR 需要先重构这些区域，说明它不适合作为本轮课堂 slice。
