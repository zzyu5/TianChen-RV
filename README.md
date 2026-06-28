# TianChen-RV Classroom：12 个 RVV 编译器 Slice

这是 TianChen-RV 的课堂分支 —— 一个**干净、可直接上手**的 RISC-V Vector(RVV)/MLIR 编译器仓库。课程目标:每个学生认领一条 **slice**,让编译器**新增一项它现在做不到的 RVV 能力**,做出来即真实拓宽系统边界、构成一次真实贡献。

```text
MLIR fixture
  -> selected typed tcrv_rvv body
  -> RVV plugin route
  -> generated RVV C/C++
  -> harness 正确性证明(真机 k1 byte-exact)
```

这**不是**"手写一个 RVV C 函数",而是补齐一条可 review 的 **compiler slice**。也**不是**重做已有功能 —— 12 条任务都是当前编译器发不出来的新能力。

## 阅读顺序(别从源码里随便找入口)
1. [Worked Example:一条完整 slice `add`](docs/add-rvv-slice-walkthrough.md) —— 已实现、端到端在真机验证过的样板,所有 slice 的模板。
2. [构建、测试与 Proof 流程](docs/build-and-test.md) —— 编译器怎么构建、lit 怎么跑、三层 proof 怎么做。
3. [RVV Slice 模块化落点](docs/rvv-slice-module-map.md) —— 一条 slice 改哪些层、落哪些文件。
4. [RVV Slice 贡献指南](docs/rvv-slice-contribution-guide.md) —— 推荐实现顺序 + PR checklist。
5. [**12 个 slice 作业**](assignments/rvv-slices-12.md) —— 你的任务清单,每条一页(目标/缺席证据/改哪些文件/oracle/验收)。
6. [测试用例与验收格式](docs/testcase-submission-format.md) —— 提交格式与三层验收口径。
7. [Proof Harness 目录](examples/qemu/README.md) —— `add` harness + Makefile + 真机 run 脚本。

## 快速上手
```bash
# 1) 构建编译器(LLVM/MLIR 20)
cmake -G Ninja -S . -B build \
  -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir
ninja -C build tcrv-opt tcrv-translate

# 2) 跑 lit 测试(tier 1,人人可做)
ninja -C build check-tianchenrv

# 3) 看 add 范例端到端:生成 C -> 本地 rv64gcv object 编译(tier 2)
build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-add-cpp-golden.mlir \
  --tcrv-rvv-lower-to-emitc | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp \
  > examples/qemu/add_generated.reference.cpp
make -C examples/qemu -f Makefile.rvv object RVV_GENERATED=add_generated.reference.cpp
```
**日常循环** = 改代码 → `ninja … check-tianchenrv`(tier 1)→ 生成 C + `make object`(tier 2)。**tier 3(真机 k1)是正确性的最终封印**:`examples/qemu/run-on-k1.sh`,详见 [build-and-test.md](docs/build-and-test.md)。

## 真机
真机 proof 用 SpacemiT X60 板,ssh 别名 `k1`(RVV 1.0 + Zvfh + IME;`clang++ 18`)。注:旧的 `ssh rvv` 主机当前不可用,真机一律走 `k1`;没有真机时本地 `lit + object-compile` 已能覆盖大部分验收,或用 `qemu-riscv64` 代替 tier 3。

## 仓库结构
```text
include/TianChenRV/   Public headers + TableGen dialect/op 定义
lib/                  编译器实现(Dialect/RVV, Plugin/RVV, Conversion/RVV+EmitC, Target, Transforms, Support)
tools/tcrv-opt/       pass driver
tools/tcrv-translate/ translation/export driver
test/                 lit/FileCheck 测试(你的 slice 在这里加 dataflow/materialization/negative)
docs/                 walkthrough / 构建测试 / 模块落点 / 贡献指南 / 验收格式
assignments/          12 个 slice 作业
examples/qemu/        add worked-example harness + Makefile + k1 run 脚本
```
本分支不含 `.trellis`、`.codex`、内部 supervisor / 远程自动化文件。

## 禁止路线(所有 slice 通用,I3/I4/I5)
- `tcrv_rvv.i8_*` 这种 dtype-prefixed helper;
- route-id / artifact 名 / C 参数名 / test 名驱动语义;
- source-front-door / source-artifact 正向路径;
- common EmitC 里硬编码 RVV dtype/SEW/LMUL/intrinsic;
- 只为过 harness 而绕过 typed `tcrv_rvv` body / provider。

每个 PR 都要能解释:typed facts 在哪、ABI/runtime value 如何绑定、RVV provider 如何从这些 facts 推导 route。
