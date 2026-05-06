# 05. IME Plugin 设计

## 5.1 模块定位

IME plugin 是 TianChen-RV MLIR 后续接入 K3/IME 环境时的关键 extension plugin。

它的主要价值不是单纯获得 matrix 性能，而是验证 TianChen-RV 的核心主张：

```text
原系统支持 RVV；
后续通过新增 IME plugin 添加新的 matrix-extension 能力；
同一个 high-level op 自动多出 IME execution variant；
核心 pass 不硬编码 IME；
核心 pass 的修改量应保持很小或为零。
```

## 5.2 IME 的定位

IME 不应被粗暴归类为普通外部 accelerator。它更接近：

```text
ISA-level matrix-like extension
vector-register-backed matrix execution
与 RVV resource model 高度相关
```

因此，IME plugin 需要同时理解：

```text
IME 自己的 matrix/dot instruction；
RVV vector register resource；
VLEN 对 fragment shape 的影响；
数据类型和 accumulator policy；
vendor intrinsic / inline asm / toolchain path。
```

## 5.3 IME capability

IME plugin 应注册：

```text
spacemit.ime or target-specific IME capability
ime.version
ime.supported_dtype
ime.vector_register_backed
ime.vlen_range
ime.fragment_shapes
ime.accumulator_types
ime.toolchain_path
```

示意：

```mlir
#tcrv.ext<"spacemit.ime",
          kind = "isa-matrix-vector-backed",
          status = "available",
          register_model = "rvv-vector-register-backed",
          dtype = ["int8", "fp16", "bf16"],
          vlen_dependent = true>
```

## 5.4 IME extension dialect

建议 dialect 名称：

```text
tcrv.ime
```

### Types

```text
!tcrv.ime.frag<dtype, shape, layout>
!tcrv.ime.accfrag<dtype, shape>
!tcrv.ime.config<...>
```

由于 IME 依赖 vector registers，type 设计需要显式表达它和 RVV resource 的关系，例如：

```text
register_model = rvv-vector-register-backed
vlen_dependent = true
```

### Ops

```text
tcrv.ime.config
tcrv.ime.load_frag
tcrv.ime.store_frag
tcrv.ime.pack
tcrv.ime.unpack
tcrv.ime.mma
tcrv.ime.dot
tcrv.ime.accumulate
tcrv.ime.convert
```

这些 op 是 IME execution ops，不是 high-level matmul op。

## 5.5 IME variant 示例

```mlir
tcrv.exec.variant @ime
  requires = #tcrv.requires<["rvv", "spacemit.ime"]>
  origin = "ime-plugin" {

  tcrv.exec.hart_parallel %tid in %num_harts {
    %cfg = tcrv.ime.config {
      a_type = f16,
      b_type = f16,
      acc_type = f32,
      register_model = "rvv-vector-register-backed"
    }

    %af = tcrv.ime.load_frag %A[%i, %k], %cfg
          : memref<?x?xf16> -> !tcrv.ime.frag<f16, shape = "target_selected">
    %bf = tcrv.ime.load_frag %B[%k, %j], %cfg
          : memref<?x?xf16> -> !tcrv.ime.frag<f16, shape = "target_selected">
    %cf = tcrv.ime.mma %af, %bf, %acc, %cfg
    tcrv.ime.store_frag %cf, %C[%i, %j], %cfg
  }
}
```

## 5.6 IME variant generation

IME plugin 应优先支持：

```text
matmul
batched matmul
attention 中的 qk/av matrix block
MLP dense block
int8/fp16/bf16 dot-like kernels
```

IME plugin 不需要覆盖所有算子。它应覆盖能体现 matrix-extension 价值的算子。

## 5.7 IME legality rules

IME plugin 应检查：

```text
target 是否声明 IME capability；
RVV/vector-register-backed 前置能力是否满足；
所需 dtype 是否被 IME 支持；
fragment shape 是否符合 VLEN 和指令限制；
accumulator type 是否支持；
memory layout 是否需要 pack；
工具链是否支持 IME emission path；
inline asm 或 vendor intrinsic 是否可用。
```

## 5.8 IME tuning space

IME 的调优空间不同于普通 RVV：

```text
fragment shape
K blocking
accumulator residency
packing format
register pressure
thread partition
IME instruction selection
fallback to RVV for unsupported shapes or dtype
```

调优结果应作为 variant metadata，不应污染 high-level op 语义。

## 5.9 IME emission path

IME emission 取决于后续硬件和工具链可用性。应允许多种后端：

```text
vendor intrinsic
compiler builtin
inline asm
patched LLVM/backend adapter
external assembly stub
```

这些路径都应封装在 IME plugin 内，不应散落到核心 pass。

## 5.10 IME 作为插件化验证实验

IME plugin 应作为新增 extension 的验证对象。记录：

```text
新增 IME plugin 后，核心 pass 修改量；
新增 capability 数量；
新增 ops/types 数量；
新增 variant generator 覆盖哪些 high-level ops；
同一 high-level op 是否自动产生 IME variant；
IME variant 是否和 RVV variant 共存；
selection/dispatch 是否复用 core 逻辑；
emission 是否局部封装在 plugin。
```

这个实验比单纯 IME 性能更能证明 TianChen-RV 的科研价值。

## 5.11 和未来 AME/custom ISA 的关系

IME plugin 不应被写死成唯一 matrix extension 形态。

未来如果接入 AME 或其他 matrix/custom ISA，应新增新的 extension plugin，而不是强行复用 IME dialect。

系统核心只要求新插件实现通用接口：

```text
capability provider
variant builder
legality verifier
tuning space provider
cost model provider
emission provider
```

