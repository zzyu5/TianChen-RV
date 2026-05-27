# 测试用例收集格式

本文规定学生完成一个 RVV slice 后，如何提交可收集、可复现、可 review 的测试用例。目标不是堆材料，而是让教师能明确看到：

```text
MLIR 输入是什么
compiler 生成的 RVV C/C++ kernel 是什么
C/C++ harness 给 kernel 喂了什么 runtime 输入
期望输出是什么
实际 QEMU 或等价 RVV 运行输出是什么
```

## 一、必须提交的 Compiler 测试

每个 slice 至少应该补下面三类测试。文件名可以按 slice 调整，但目录不要随意新建。

```text
test/Dialect/RVV/<slice>-dataflow.mlir
test/Conversion/EmitC/rvv-<slice>-materialization.mlir
test/Target/RVV/emitc-to-cpp-<slice>.mlir
```

最低内容：

- dialect positive：typed `tcrv_rvv` body 能 parse/verify。
- negative：至少一个会误导 route authority 的错误必须 fail closed，例如 dtype/LMUL 不匹配、mask 类型不匹配、unsupported immediate、ABI role 缺失。
- EmitC/Target FileCheck：能看到 expected intrinsic family、C vector type 或 exported kernel signature。

## 二、Runtime Proof Bundle

如果 PR 声明 runtime correctness，建议额外提交一个小目录：

```text
examples/qemu/cases/<slice-id>/
  README.md
  harness.cpp
```

`<slice-id>` 用短名，例如：

```text
bitwise-and
slide-down
mask-vcpop
saturating-add
```

默认不要提交 generated C++，因为它应该从 MLIR fixture 可复现生成。只有课程要求离线审阅时，才额外附 `generated.snippet.txt`，且只放 kernel signature 和关键 intrinsic 片段。

## 三、Runtime Case README 模板

每个 `examples/qemu/cases/<slice-id>/README.md` 建议包含下面这些小节：

- `Source MLIR`：指向 `test/Target/RVV/emitc-to-cpp-<slice>.mlir`。
- `Generate RVV C++`：记录从 MLIR 生成 `generated.cpp` 的完整命令。
- `Generated Kernel Evidence`：贴出 exported kernel signature 和关键 intrinsic。
- `Runtime Inputs`：说明 `n`、输入数组、mask/index/scalar 和 expected output 的生成方式。
- `Run`：记录 QEMU 编译运行命令。
- `Actual Output`：贴出实际运行输出。

生成命令示例：

```bash
build/bin/tcrv-opt test/Target/RVV/emitc-to-cpp-<slice>.mlir \
  --tcrv-materialize-emission-plans \
| build/bin/tcrv-translate --tcrv-rvv-emitc-to-cpp \
> /tmp/tcrv-<slice>/generated.cpp
```

运行命令示例：

```bash
cp examples/qemu/cases/<slice-id>/harness.cpp /tmp/tcrv-<slice>/harness.cpp
make -C /tmp/tcrv-<slice> \
  -f "$PWD/examples/qemu/Makefile.rvv" \
  run-rvv \
  RVV_CXX=/usr/lib/llvm-20/bin/clang++ \
  RVV_EXTRA_CXXFLAGS="--gcc-toolchain=/usr -I/usr/riscv64-linux-gnu/include/c++/14 -I/usr/riscv64-linux-gnu/include/c++/14/riscv64-linux-gnu -static" \
  RVV_EXTRA_LDFLAGS="-L/usr/lib/gcc-cross/riscv64-linux-gnu/14"
```

## 四、Harness 要求

`harness.cpp` 应该保持小而明确：

- 声明 generated kernel 的 `extern "C"` signature。
- 分配输入和输出 buffer。
- 用函数或常量清楚描述输入数据，例如 `make_lhs(i)`、`make_rhs(i)`、`make_mask(i)`。
- 初始化 output sentinel，避免没有写出的 lane 被误判。
- 调用 generated kernel。
- 用 scalar oracle 检查输出。
- mismatch 时打印 index、got、expected 并返回非零。
- 成功时打印前几个 lane 的输入/输出和 `proof ok`。

如果 slice 改变 ABI，例如增加 scalar、mask buffer、index buffer、accumulator 或第二个输出，必须同时更新：

```text
MLIR fixture 的 ABI/runtime value binding
generated C++ function signature
harness.cpp 的 extern "C" 声明
harness.cpp 的调用参数
README 中的 runtime input/output 描述
```

## 五、教师收集口径

教师 review 时按这个顺序收集：

1. 先跑 `cmake --build build --target check-tianchenrv`，确认 compiler tests。
2. 再用学生 README 的 generate 命令生成 `/tmp/tcrv-<slice>/generated.cpp`。
3. grep generated C++ 的 kernel signature 和 expected intrinsic。
4. 用学生 harness 编译运行 QEMU。
5. 保存 PR 链接、slice 名称、MLIR 路径、harness 路径、QEMU 输出。

如果一个 PR 只有 QEMU 输出但没有 MLIR/EmitC/Target 测试，不合格。如果一个 PR 只有 FileCheck 但声明 runtime correctness，也不合格。
