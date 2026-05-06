# 01. Capability Model 设计

## 1.1 模块定位

Capability model 是 TianChen-RV MLIR 的第一核心。

它负责把 RISC-V target 的 ISA 扩展、微架构信息、运行时 offload 能力、工具链能力转化为 MLIR 中可查询、可验证、可参与 pass 决策的对象。

TianChen-RV 不应把 capability 当成普通字符串属性。Capability 必须决定：

```text
哪些 extension plugins 可用；
某个 high-level op 可以生成哪些 variants；
某个 variant 是否 legal；
某个 variant 的调优空间是什么；
是否需要 runtime dispatch；
某个 emission path 是否可用；
fallback 条件是什么。
```

## 1.2 Capability 的来源

Capability 不是只来自 `-march`。它至少来自四类来源。

### 1.2.1 ISA capabilities

来自 RISC-V ISA 字符串、profile、编译器 target feature 或硬件探测结果。

示例：

```text
rv64
I / M / A / F / D / C
V
Zvl128b
Zvfh
Zvfbfmin
Zvfbfwma
vendor custom opcode
SpacemiT IME extension
future matrix extension
```

### 1.2.2 Microarchitecture capabilities

同样的 ISA 扩展在不同实现上的性能和限制可能不同，因此系统需要记录微架构相关信息。

示例：

```text
core count
VLEN
cache size
memory bandwidth
preferred LMUL
supported dtype throughput
NUMA / memory topology
OpenMP or thread runtime availability
```

### 1.2.3 Runtime/offload capabilities

RISC-V AI 系统可能使用外部或片上 accelerator。它们不一定通过 RISC-V instruction stream 暴露，而是通过 runtime、driver、C API、PCIe 或 SoC 内部队列调用。

示例：

```text
Sophgo accelerator present
TPU runtime available
PCIe mode
SoC mode
supported offload operator set
supported model format
host-device transfer cost
async execution support
```

### 1.2.4 Toolchain capabilities

某些 hardware feature 即使硬件存在，也可能因为工具链不支持而不能直接生成。

示例：

```text
LLVM RVV scalable vector support
RVV intrinsic support
compiler builtin support
inline asm allowed
vendor header available
patched compiler available
runtime library linkable
```

## 1.3 Capability object 的逻辑结构

建议将 capability 表示为 target-level attribute 或 module-level attribute。

示意：

```mlir
#tcrv.target<
  arch = "riscv64",
  isa = ["i", "m", "a", "f", "d", "c", "v", "zvl128b", "zvfh"],
  uarch = {
    cores = 64,
    vlen = 128,
    has_openmp = true,
    cache_model = "target_specific"
  },
  extensions = [
    #tcrv.ext<"rvv", kind = "isa-vector", status = "available">
  ],
  accelerators = [
    #tcrv.accel<"sophgo.bm1684x", kind = "runtime-offload",
                mode = "pcie", runtime = "vendor-c-abi">
  ],
  toolchain = {
    llvm_rvv = true,
    rvv_intrinsic = true,
    inline_asm = true,
    vendor_runtime_link = true
  }
>
```

K3/IME 到位后可以增加：

```mlir
#tcrv.ext<"spacemit.ime",
          kind = "isa-matrix-vector-backed",
          status = "available",
          register_model = "rvv-vector-register-backed",
          dtype = ["int8", "fp16", "bf16"]>
```

## 1.4 Capability kind 是开放集合

系统不能把扩展类别限制死。建议使用开放的 `kind` 字段，并允许插件注册新 kind。

内置常见 kind：

```text
isa-scalar
isa-vector
isa-matrix-vector-backed
isa-matrix-separate-register
isa-custom-instruction
runtime-offload
toolchain
uarch
memory
thread-runtime
```

这些 kind 只是初始集合，不构成上限。

例如，未来可以新增：

```text
isa-sparse
isa-dma
isa-cluster
isa-crypto-ai
runtime-remote-device
runtime-shared-memory-accelerator
```

核心 pass 不应依赖 kind 的穷举分支，而应通过 plugin 注册的 interface 处理。

## 1.5 Capability relation

Capability model 需要支持以下关系。

### require

Variant 声明自身需要哪些能力：

```mlir
tcrv.exec.variant @rvv_fp16
  requires = #tcrv.requires<["rvv", "zvfh", "zvl128b"]> { ... }
```

### provide

Plugin 声明它能提供什么能力：

```text
RVV plugin provides: rvv, vector-stripe, scalable-vl, rvv-load-store, rvv-reduce
IME plugin provides: ime, vector-register-backed-matrix, ime-frag-mma
Offload plugin provides: sophgo-runtime, async-offload, runtime-buffer
```

### imply

一些能力蕴含其他能力，例如：

```text
rv64gcv implies rvv
zvfh implies fp16 vector arithmetic support, subject to toolchain support
spacemit.ime implies vector-register-backed matrix capability, subject to vendor toolchain support
```

### conflict

某些能力或变体可能互斥，例如：

```text
某 variant 需要 vendor runtime，但 target 没有 runtime library。
某 variant 需要 inline asm，但 build policy 禁止 inline asm。
某 offload variant 需要固定 shape，但当前 input shape 不满足。
```

### dispatch condition

一些能力需要运行时确认，或不同 shape 下选择不同 variant：

```text
if shape is large and sophgo runtime is available -> offload variant
else -> RVV variant
```

## 1.6 Capability verifier

Capability verifier 负责检查：

```text
variant 的 requires 是否被 target capability 满足；
extension op 是否出现在相容的 variant 中；
tcrv.rvv op 是否只在 RVV-capable variant 中出现；
tcrv.ime op 是否只在 IME-capable variant 中出现；
tcrv.offload op 是否只在 runtime-offload-capable variant 中出现；
工具链 path 是否存在；
runtime ABI 是否声明完整；
dispatch/fallback 是否覆盖不可用条件。
```

Verifier 不负责证明数值正确性，但它必须阻止明显非法的 target-feature 使用。

## 1.7 当前目标 profile

### Profile: RVV 主开发环境

```text
name: rvv-main
access: ssh rvv
hardware: RISC-V CPU, 64 cores
vector: RVV 1.0
permission: sudo available
role: 主开发、主性能实验、主 correctness path
```

Capability 示例：

```text
rv64
rvv
zvl128b or actual probed minimum VLEN
fp32/fp64 depending on hardware
thread-runtime: OpenMP or pthread
native compile support
```

### Profile: K3/IME 后续环境

```text
name: k3-ime
hardware: SpacemiT/K3-class RISC-V system
role: IME extension plugin 验证
status: 后续到位
```

Capability 示例：

```text
rvv
spacemit.ime
vector-register-backed matrix capability
vendor intrinsic or inline asm path
```

### Profile: RISC-V + Sophgo offload

```text
name: riscv-sophgo-offload
hardware: RISC-V host + Sophgo accelerator path
role: runtime-offload capability case
```

Capability 示例：

```text
rvv or scalar CPU fallback
sophgo runtime available
C ABI call path
PCIe or SoC mode
async call if available
```

## 1.8 Capability model 的科研价值

Capability model 的价值不在于记录硬件信息，而在于它改变了 pass 的结构：

```text
没有 capability model：
  lowering pass 里充满 target-specific if/else。
  新增扩展时要改 high-level lowering、legalization、cost、dispatch、emission。

有 capability model：
  core pass 只查询 target capability 和 plugin registry。
  新增扩展通过 plugin 注册 capability 和 variant generator。
  同一个 high-level op 在不同 capability 下产生不同 variants。
  非法 variant 在 MLIR 层被 verifier 拒绝。
```

这也是 TianChen-RV MLIR 区别于普通 RVV lowering 的核心。

