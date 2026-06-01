# TianChen-RV Classroom：LLM `q8_0_q8_0` RVV VDot

这是 TianChen-RV 的课堂贡献分支。本分支不再分配 25 个互不相同的 RVV 小题，而是让所有学生围绕同一个真实 LLM 算子推进：

```text
llama.cpp q8_0_q8_0 quantized vector dot
```

目标不是手写一个 RVV C 函数，而是为 TianChen-RV 补齐一条可 review 的 compiler slice：

```text
MLIR fixture
  -> selected typed tcrv_rvv body
  -> RVV plugin route
  -> generated RVV C/C++
  -> harness correctness proof
```

教师分支提供：

- 一个参考的 llama.cpp 风格 RVV C baseline；
- 一个可运行 harness，说明输入、输出和 scalar oracle；
- 构建、QEMU/ssh-rvv 运行流程；
- 学生 PR 的测试和验收格式。

教师分支**不提前实现** TianChenRV 的 `q8_0_q8_0` MLIR 到 RVV C 路径；这正是学生要完成的贡献。

## 阅读顺序

按下面顺序读，不要从源码里随机找入口。

1. [课程主线：llama.cpp `q8_0_q8_0` RVV VDot Slice](assignments/llama-q8-0-vdot.md)
   明确本轮作业目标、数学语义、学生需要补的 compiler 能力和禁止路线。
2. [构建与 RVV Proof 流程](docs/build-and-rvv-proof.md)
   编译 TianChenRV，运行参考 RVV C baseline，并了解 QEMU/ssh-rvv proof 如何收集。
3. [本地 RVV QEMU/ssh-rvv 示例](examples/qemu/README.md)
   直接运行 `llama_q8_0_q8_0_rvv.cpp + harness_llama_q8_0_q8_0.cpp`，看真实 kernel 输入输出。
4. [RVV Slice 贡献指南](docs/rvv-slice-contribution-guide.md)
   说明学生怎样把 baseline 变成 TianChenRV 的 MLIR/compiler slice。
5. [RVV Slice 模块化落点](docs/rvv-slice-module-map.md)
   判断应该改 typed IR、verifier、selected-body realization、provider route、EmitC 还是测试。
6. [测试用例收集格式](docs/testcase-submission-format.md)
   提 PR 前按统一格式提交 MLIR、generated kernel evidence、harness 输入输出和运行输出。

## 当前参考 Baseline

参考 RVV C baseline 在：

```text
examples/qemu/llama_q8_0_q8_0.h
examples/qemu/llama_q8_0_q8_0_rvv.cpp
examples/qemu/harness_llama_q8_0_q8_0.cpp
examples/qemu/Makefile.rvv
```

核心计算形状：

```c
vint8m2_t x = __riscv_vle8_v_i8m2(...);
vint8m2_t y = __riscv_vle8_v_i8m2(...);
vint16m4_t prod = __riscv_vwmul_vv_i16m4(x, y, vl);
vint32m1_t sum = __riscv_vwredsum_vs_i16m4_i32m1(prod, zero, vl);
```

课堂版 block：

```text
block_q8_0:
  d:  float scale
  qs: int8[32]
```

真实 llama.cpp 使用 fp16 scale；课堂 baseline 使用 `float d`，先把重点放在 i8 load、i8 widening multiply、i16 widening reduction、scalar output 和 provider route 上。

## 快速运行 Baseline

在本地 QEMU 路径：

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

在维护者真实 RVV 主机上可以复制同一组文件后编译运行：

```bash
scp examples/qemu/llama_q8_0_q8_0.* \
    examples/qemu/harness_llama_q8_0_q8_0.cpp \
    rvv:/tmp/tcrv-classroom-q8/

ssh rvv 'cd /tmp/tcrv-classroom-q8 && \
  clang++ -std=c++17 -O2 -march=rv64gcv -mabi=lp64d \
    llama_q8_0_q8_0_rvv.cpp harness_llama_q8_0_q8_0.cpp \
    -o llama_q8_0_q8_0_case && \
  ./llama_q8_0_q8_0_case'
```

成功输出应包含：

```text
rvv classroom llama q8_0_q8_0 proof ok
```

## 学生交付边界

学生最终 PR 应该让 TianChenRV 生成与 baseline 同语义的 RVV C/C++。最低交付包括：

- `test/Target/RVV/` 下的 MLIR fixture；
- `test/Dialect/RVV/` 或 `test/Conversion/EmitC/` 下的 verifier/route 测试；
- generated RVV C/C++ evidence，能看到 `vle8`、`vwmul_vv_i16`、`vwredsum_vs_i16...i32` 相关路线；
- runtime harness，使用同样的 q8 block 输入和 scalar oracle；
- QEMU 或真实 RVV 运行输出。

不要提交只手写 baseline C 的 PR。baseline C 只是目标形状和正确性对照。

## 仓库结构

```text
include/TianChenRV/        Public headers 和 TableGen dialect/op 定义
lib/                       编译器实现
tools/tcrv-opt/            pass driver
tools/tcrv-translate/      translation/export driver
test/                      学生要补的 MLIR/FileCheck 测试位置
docs/                      贡献说明和测试收集格式
assignments/               本轮 q8_0_q8_0 主线说明
examples/qemu/             baseline RVV C、harness、QEMU Makefile
```

本分支不包含 `.trellis`、`.codex`、内部 supervisor loop 或远程自动化 steering 文件。

## 禁止路线

不要新增这些东西：

- `tcrv_rvv.i8_*`、`tcrv_rvv.i32_*` 这种 dtype-prefixed helper；
- `RVVI32M1*` / `rvv-i32m1` route-id 驱动语义；
- source-front-door / source-artifact 正向路径；
- common EmitC 中硬编码 RVV dtype、SEW/LMUL、intrinsic 或 schedule；
- 只为通过 harness 而绕过 typed `tcrv_rvv` body/provider 的实现。

每个 PR 应该能解释：typed facts 在哪里、ABI/runtime value 如何绑定、RVV provider 如何从这些 facts 推导 route。
