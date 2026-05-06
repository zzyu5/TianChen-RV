# 04. RVV Plugin 设计

## 4.1 模块定位

RVV plugin 是 TianChen-RV MLIR 的当前主硬件路径。

它负责把 high-level MLIR 算子实现为 RVV execution variants，并提供 RVV capability、RVV-specific ops/types、合法性检查、调优空间、cost model 和 lowering/emission。

当前实验环境：

```text
access: ssh rvv
hardware: RISC-V CPU, 64 cores
vector: RVV 1.0
permission: sudo available
role: 主开发环境、主性能实验环境、主 correctness path
```

## 4.2 RVV plugin 的职责

RVV plugin 提供：

```text
RVV capability registration
RVV dialect registration
RVV variant generation
RVV legality verification
RVV tuning space
RVV cost model
RVV emission path
RVV runtime/threading integration
```

它不负责：

```text
定义高层 matmul 语义；
定义通用 tensor/tile IR；
处理 IME/offload/custom ISA 的内部细节。
```

## 4.3 RVV capability

RVV plugin 应注册并查询：

```text
rvv
rvv.version
vlen
elen
supported SEW
supported LMUL
mask support
tail policy support
Zvfh / Zvfbfmin / other vector dtype extensions
LLVM scalable vector support
RVV intrinsic support
inline asm policy
thread runtime availability
```

Capability 示例：

```mlir
#tcrv.ext<"rvv",
          kind = "isa-vector",
          version = "1.0",
          vlen = 128,
          supports_mask = true,
          supports_tail_policy = true>
```

## 4.4 RVV extension dialect

建议 dialect 名称：

```text
tcrv.rvv
```

### Types

```text
!tcrv.rvv.vreg<dtype, lmul>
!tcrv.rvv.mask<lmul>
!tcrv.rvv.vl
!tcrv.rvv.policy<tail, mask>
```

### Core ops

```text
tcrv.rvv.setvl
tcrv.rvv.load
tcrv.rvv.store
tcrv.rvv.masked_load
tcrv.rvv.masked_store
tcrv.rvv.broadcast
tcrv.rvv.fma
tcrv.rvv.add / mul / max / min
tcrv.rvv.reduce
tcrv.rvv.slide / gather / compress, as needed
tcrv.rvv.convert
```

这些 op 是 RVV execution ops，不是 high-level tensor ops。

## 4.5 RVV variant 示例

```mlir
tcrv.exec.variant @rvv
  requires = #tcrv.requires<["rvv", "zvfh"]>
  origin = "rvv-plugin" {

  tcrv.exec.hart_parallel %tid in %num_harts {
    %vl = tcrv.rvv.setvl %n {sew = 16, lmul = 4, tail = "agnostic"}
    %a  = tcrv.rvv.load %A[%i, %k], %vl
          : memref<?x?xf16>, !tcrv.rvv.vl -> !tcrv.rvv.vreg<f16, lmul = 4>
    %b  = tcrv.rvv.load %B[%k, %j], %vl
          : memref<?x?xf16>, !tcrv.rvv.vl -> !tcrv.rvv.vreg<f16, lmul = 4>
    %c1 = tcrv.rvv.fma %a, %b, %c0
    tcrv.rvv.store %c1, %C[%i, %j], %vl
  }
}
```

## 4.6 RVV variant generation

RVV plugin 应支持从以下 high-level op class 生成 variants：

```text
matmul / batched matmul
softmax
layernorm / rmsnorm
rope
elementwise + reduction fusion
attention micro-kernel fragments
```

它的输出不是 generic `tcrv.matmul`，而是包含 `tcrv.rvv.*` 的 `tcrv.exec.variant`。

## 4.7 RVV legality rules

RVV plugin 应检查：

```text
target 是否支持 RVV；
需要的 dtype 是否有对应扩展或 fallback path；
SEW/LMUL 是否合法；
VL policy 是否可表达；
mask/tail policy 是否完整；
load/store pattern 是否能由 RVV 表达；
reduction 是否能正确处理 tail/mask；
工具链是否支持所选 emission path。
```

## 4.8 RVV tuning space

RVV tuning 是系统能力，不作为孤立理论点。它主要服务于 variant quality。

可调参数包括：

```text
SEW
LMUL
VL policy
unroll factor
K blocking
packing of B or weights
thread partition across harts
prefetch or software pipelining option
boundary handling strategy
```

Tuning 结果应记录在 variant metadata 中：

```mlir
#tcrv.tuning<
  lmul = 4,
  sew = 16,
  k_block = 64,
  unroll = 2,
  thread_partition = "row_block"
>
```

## 4.9 RVV emission paths

RVV plugin 应支持两类 emission path。

### Path A: MLIR vector / LLVM scalable vector

适合：

```text
普通 vector arithmetic
普通 load/store
常规 reduction
可由 LLVM RISC-V backend 可靠生成的 RVV 模式
```

路径：

```text
tcrv.rvv
  -> MLIR vector dialect / LLVM dialect
  -> LLVM scalable vector
  -> LLVM RISC-V backend
```

### Path B: RVV intrinsic / inline asm / builtin

适合：

```text
需要精确控制 vsetvl；
需要明确 mask/tail policy；
需要 segment/strided/special RVV op；
LLVM vector lowering 难以稳定生成的关键 kernel。
```

路径：

```text
tcrv.rvv
  -> LLVM RVV intrinsic / compiler builtin / inline asm
  -> native compile
```

## 4.10 多核执行

RVV plugin 本身不定义 thread/block 模型。多核由 `tcrv.exec.hart_parallel` 组织。

RVV plugin 应给出：

```text
每个 hart 内部如何使用 RVV；
hart 间如何分割 M/N/batch/row；
OpenMP/pthread/runtime thread pool 的 lowering preference；
single-thread 和 multi-thread path 的 metadata。
```

## 4.11 Correctness and diagnostics

RVV plugin 需要生成可诊断信息：

```text
非法 LMUL/SEW 组合；
不支持 dtype；
mask/tail policy 缺失；
工具链无法 emission；
memory pattern 不适合 RVV；
variant 不满足 capability。
```

这些诊断应统一回传给 core diagnostic system。

## 4.12 RVV plugin 的系统角色

RVV plugin 是第一个完整插件，应承担以下角色：

```text
验证 TianChen-RV capability model 可以驱动真实 codegen；
验证 tcrv.exec variant IR 能表达真实 RVV kernel；
验证 plugin protocol 的基本接口是否充分；
提供性能主线和 correctness 主线；
为后续 IME/offload plugin 提供对照。
```

