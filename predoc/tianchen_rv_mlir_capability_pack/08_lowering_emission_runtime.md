# 08. Lowering, Emission and Runtime 边界

## 8.1 模块定位

本模块定义 TianChen-RV variants 最终如何生成可执行代码或 runtime calls。

核心原则：

```text
core pass 不处理 extension-specific emission 细节；
每个 plugin 负责自己的 lowering/emission/runtime glue；
工具链 patch 或 vendor workaround 应被封装在 plugin adapter 中；
variant 选择和 emission path 必须被 capability model 记录和验证。
```

## 8.2 Emission provider

每个 extension plugin 应实现 emission provider。

职责：

```text
注册 lowering patterns；
声明 emission path；
声明需要的 compiler flags、headers、libraries；
生成 runtime glue；
报告 unsupported path；
提供 diagnostics。
```

## 8.3 RVV emission

RVV plugin 支持两类 emission。

### MLIR vector / LLVM scalable vector path

适合普通 vector 模式。

```text
tcrv.rvv ops
  -> MLIR vector dialect / LLVM dialect
  -> LLVM scalable vector
  -> LLVM RISC-V backend
  -> RVV machine code
```

优点：

```text
更接近 upstream 工具链；
可维护性好；
适合一般 arithmetic/load/store/reduction。
```

限制：

```text
某些 RVV-specific policy 控制可能不稳定；
难以精确控制 vsetvl；
复杂 mask/tail/segment 模式可能需要 intrinsic path。
```

### RVV intrinsic / inline asm path

适合需要显式控制的关键 kernel。

```text
tcrv.rvv ops
  -> LLVM RVV intrinsic / compiler builtin / inline asm
  -> native compile
```

优点：

```text
控制力强；
便于对齐手写 RVV kernel；
适合性能关键路径。
```

限制：

```text
可移植性较弱；
需要更多工具链和 ABI 管理；
诊断和验证更重要。
```

## 8.4 IME emission

IME emission 取决于后续 K3/IME 工具链。

可能路径：

```text
vendor intrinsic
compiler builtin
inline asm
external assembly stub
patched LLVM/backend adapter
```

设计要求：

```text
IME-specific emission 全部封装在 IME plugin；
core pass 只知道该 variant 有 emission path；
capability model 记录工具链支持情况；
如果 IME emission 不可用，variant 应被 verifier 拒绝或保留为 disabled diagnostic。
```

## 8.5 Offload emission

Offload emission 不是机器指令生成，而是 runtime glue 生成。

路径：

```text
tcrv.offload ops
  -> C ABI call / vendor runtime call
  -> link vendor runtime library
  -> runtime dispatch / sync / buffer management
```

需要生成或声明：

```text
runtime handle initialization
buffer binding or allocation
host-device copy
async call
wait/sync
error handling
resource release
fallback path
```

## 8.6 Runtime dispatch generation

`tcrv.exec.dispatch` 应 lower 到 host-side decision logic。

输入：

```text
capability availability
runtime probe
shape/dtype guard
cost threshold
user policy
```

输出：

```text
call selected variant
or fallback
```

dispatch 逻辑必须保持可诊断：

```text
为什么选择 offload；
为什么选择 RVV；
为什么 runtime unavailable；
为什么 fallback。
```

## 8.7 Fallback lowering

Fallback 是系统完整性的一部分。

Fallback 可以来自：

```text
MLIR default lowering
scalar/scf lowering
LLVM auto-vectorization
portable C/C++ kernel
RVV conservative kernel
```

Fallback 的目标是正确性和覆盖，不是主性能路径。

## 8.8 Toolchain boundary

某些扩展可能需要工具链 patch。系统应将其作为 plugin 内部 adapter，而不是核心依赖。

设计要求：

```text
plugin 声明 toolchain requirement；
capability verifier 检查 requirement；
核心 pass 不直接调用 vendor-specific compiler path；
build system 通过 plugin metadata 选择 flags/libraries；
unsupported toolchain path 输出清晰 diagnostics。
```

## 8.9 Build and deployment profiles

建议定义 deployment profile：

```text
rvv-native
  target: ssh rvv
  emission: LLVM/RVV intrinsic/native compile
  runtime: OpenMP/pthread optional

k3-ime
  target: K3/IME environment
  emission: IME plugin selected path
  runtime: native or vendor support

riscv-sophgo-offload
  target: RISC-V host + Sophgo runtime
  emission: C ABI/runtime glue
  runtime: vendor accelerator runtime
```

## 8.10 Diagnostics and reproducibility

每个 emission path 应输出：

```text
selected variant；
required capabilities；
selected compiler flags；
selected runtime libraries；
lowering path；
fallback status；
unsupported reason if failed。
```

这有助于调试，也有助于论文实验记录。

