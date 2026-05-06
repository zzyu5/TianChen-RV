# 06. Runtime Offload Plugin 设计

## 6.1 模块定位

Runtime offload plugin 用于表达 RISC-V host + external/SoC accelerator 的执行能力。

以 Sophgo 这类环境为例，能力通常不是 RISC-V instruction stream 中的 custom opcode，而是：

```text
vendor runtime
C API
PCIe or SoC data transfer
compiled accelerator model or kernel
async execution queue
host-device synchronization
```

因此，这个模块不应被称为 custom ISA extension。准确名称是：

```text
runtime-offload capability
```

它可以模拟或代表厂商加速能力接入 TianChen-RV 的系统级情况，但论文和系统设计中必须诚实区分：

```text
custom ISA extension: 通过 RISC-V 指令流、intrinsic、inline asm、custom opcode 执行。
runtime offload: 通过 runtime/driver/C ABI 调用 accelerator。
```

## 6.2 为什么仍然纳入 TianChen-RV

RISC-V AI 系统不一定只依赖 CPU 内部扩展。实际系统可能是：

```text
RISC-V CPU + RVV
RISC-V CPU + matrix extension
RISC-V CPU + PCIe accelerator
RISC-V SoC + on-chip accelerator runtime
```

TianChen-RV MLIR 的目标是 capability-driven execution layer，因此 runtime-offload 能力也应该被 capability model 管理。

它回答的问题是：

```text
当前 high-level op 是否可以 offload？
offload 是否比 RVV CPU variant 更合适？
offload 需要哪些 runtime 和 buffer 条件？
如何在同一个 kernel 中保留 RVV fallback？
```

## 6.3 Offload capability

Offload plugin 应注册：

```text
accelerator name
runtime availability
C ABI availability
PCIe or SoC mode
supported operator set
supported dtype
supported shape constraints
async execution support
transfer model
compiled artifact format
```

示意：

```mlir
#tcrv.accel<"sophgo.runtime",
            kind = "runtime-offload",
            mode = "pcie",
            abi = "c",
            async = true,
            supported_ops = ["matmul", "conv", "transformer_block"]>
```

## 6.4 Offload extension dialect

建议 dialect 名称：

```text
tcrv.offload
```

### Types

```text
!tcrv.offload.buffer<device, dtype, shape>
!tcrv.offload.event
!tcrv.offload.handle
!tcrv.offload.runtime
```

### Ops

```text
tcrv.offload.bind
tcrv.offload.alloc
tcrv.offload.copy_to_device
tcrv.offload.copy_from_device
tcrv.offload.async_call
tcrv.offload.call
tcrv.offload.wait
tcrv.offload.release
tcrv.offload.shape_guard
```

## 6.5 Offload variant 示例

```mlir
tcrv.exec.variant @sophgo_offload
  requires = #tcrv.requires<["sophgo.runtime", "vendor_c_abi"]>
  origin = "offload-plugin" {

  %devA = tcrv.offload.bind %A {direction = "input"}
  %devB = tcrv.offload.bind %B {direction = "input"}
  %devC = tcrv.offload.bind %C {direction = "output"}

  %ev = tcrv.offload.async_call @vendor_matmul(%devA, %devB, %devC)
    { abi = "c", runtime = "sophgo", artifact = "compiled_kernel_or_model" }

  tcrv.offload.wait %ev
}
```

## 6.6 Offload variant generation

Offload plugin 应优先处理粗粒度算子：

```text
matmul / batched matmul
conv
large transformer subgraph
attention block if runtime supports
MLP block
```

不建议把细粒度 elementwise op 作为主线 offload 对象，因为传输和 runtime overhead 可能超过收益。

## 6.7 Offload legality rules

Offload plugin 应检查：

```text
runtime 是否可用；
C ABI 或 vendor API 是否可 link；
输入输出 dtype 是否支持；
shape 是否满足 runtime/operator 限制；
buffer ownership 是否明确；
同步边界是否完整；
是否需要 host-device copy；
是否有 fallback variant；
是否允许异步调用。
```

## 6.8 Offload cost model

Offload cost 不应只估计 accelerator compute time。它至少包括：

```text
host-device transfer cost
runtime launch overhead
device compute cost
sync/wait cost
buffer reuse benefit
batching benefit
RVV fallback cost
```

变体选择应基于：

```text
shape size
dtype
operator type
data already on device or not
runtime availability
batching opportunity
```

## 6.9 Offload tuning space

Offload tuning 主要包括：

```text
offload threshold
batch size
transfer granularity
async overlap
buffer reuse policy
host fallback threshold
```

它不是普通 tile tuning，而是 runtime/offload 决策调优。

## 6.10 和 custom ISA plugin 的区别

Runtime offload plugin 不应替代 custom ISA plugin。

未来如果有厂商通过 RISC-V custom opcode、intrinsic 或 inline asm 暴露加速能力，应新增 custom ISA plugin，例如：

```text
tcrv.vendor_x
```

custom ISA plugin 的 emission path 是：

```text
custom intrinsic / inline asm / backend patch / object stub
```

runtime offload plugin 的 emission path 是：

```text
C ABI / runtime library / driver call
```

二者都可以被 capability model 管理，但系统不能混淆它们。

## 6.11 Sophgo 环境中的角色

Sophgo 路线在系统中的角色是：

```text
证明 TianChen-RV 能管理 RISC-V host + accelerator runtime；
证明 offload 能力可以作为 execution variant；
证明 RVV CPU variant 和 offload variant 可以共存并通过 dispatch 选择；
证明 runtime capability 也能纳入统一 capability model。
```

它不是“RISC-V custom ISA extension”的直接证据。

