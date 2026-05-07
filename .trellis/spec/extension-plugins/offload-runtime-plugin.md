# Runtime Offload Plugin

## Role

Runtime offload plugin represents RISC-V host plus external or SoC accelerator execution.

For Sophgo-like environments, execution usually uses:

```text
vendor runtime
C API
PCIe or SoC data transfer
compiled accelerator model or kernel
async execution queue
host-device synchronization
```

This is a `runtime-offload capability`, not a custom RISC-V ISA extension.

## First C++ Plugin Slice

The first concrete C++ runtime-offload slice is intentionally bounded. It proves
plugin identity, explicit capability gating, proposal metadata, legality
routing, conservative preference metadata, selected lowering-boundary
materialization, plugin-owned runtime ABI handoff metadata, and emission
manifest serialization through the existing generic registry interfaces.

Stable first-slice names:

```text
plugin name: offload-plugin
plugin version: 0.1.0
plugin capability id: offload.runtime
plugin capability kind: runtime-offload
preferred kernel capability symbol: @offload_runtime
first-slice proposal / variant symbol: @offload_runtime_first_slice
variant origin: offload-plugin
required capability id: offload.runtime
materialized requires form: requires = [@offload_runtime]
runtime ABI handoff id: generic-runtime-offload-c-abi-handoff.v1
handoff kind: runtime-offload
```

The first slice may propose `@offload_runtime_first_slice` only when the kernel
declares an available structured capability with id `offload.runtime`, kind
`runtime-offload`, and bounded generic properties:

```text
runtime_abi = "generic-runtime-offload-c-abi-handoff.v1"
handoff_kind = "runtime-offload"
```

If the capability is missing or generically unavailable, the plugin proposes no
variant. If the explicit offload capability is present but required generic
handoff metadata is missing, malformed, unbounded, secret-like, or not the
generic first-slice handoff contract, proposal collection records a recoverable
plugin-local decline and produces no offload proposal. Vendor names, Sophgo
strings, RVV probe facts, architecture names, and unrelated runtime-offload
capabilities must not enable the generic offload slice.

The first slice carries generic decision metadata (`condition`, `guard`, and
`policy`) and plugin-owned discardable string metadata
`tcrv_offload.runtime_abi` plus `tcrv_offload.handoff_kind` on the materialized
variant. Plugin legality checks that the selected variant requires
`offload.runtime`, that the capability remains available, and that the
plugin-owned variant metadata matches the preserved capability properties. This
metadata is compiler handoff metadata only; it is not executable runtime glue.

## Why It Belongs In TianChen-RV

TianChen-RV is a capability-driven execution layer. RISC-V AI systems may include CPU extensions and accelerator runtimes:

```text
RISC-V CPU + RVV
RISC-V CPU + matrix extension
RISC-V CPU + PCIe accelerator
RISC-V SoC + on-chip accelerator runtime
```

Runtime offload answers:

- can this high-level op be offloaded?
- is offload better than RVV CPU variant?
- what runtime and buffer conditions are required?
- how does RVV or scalar fallback remain available?

## Capability Fields

Offload plugin should register:

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

Reference attribute:

```mlir
#tcrv.accel<"sophgo.runtime",
            kind = "runtime-offload",
            mode = "pcie",
            abi = "c",
            async = true,
            supported_ops = ["matmul", "conv", "transformer_block"]>
```

## Dialect

Architectural family:

```text
tcrv.offload
```

Concrete MLIR namespace:

```text
tcrv_offload
```

Current first-slice op:

```mlir
tcrv_offload.lowering_boundary {
  source_kernel = "kernel_symbol",
  selected_variant = @offload_runtime_first_slice,
  origin = "offload-plugin",
  role = "dispatch case",
  status = "metadata-only",
  required_capabilities = [@offload_runtime],
  runtime_abi = "generic-runtime-offload-c-abi-handoff.v1",
  handoff_kind = "runtime-offload",
  handoff_reason = "runtime-offload boundary is plugin-owned handoff metadata only"
}
```

This op is a selected-path handoff boundary. It is not a high-level compute op,
custom RISC-V ISA op, vendor runtime call, DMA operation, accelerator kernel,
object-generation step, correctness result, or performance result.

Future types may include:

```text
!tcrv.offload.buffer<device, dtype, shape>
!tcrv.offload.event
!tcrv.offload.handle
!tcrv.offload.runtime
```

Future ops may include:

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

## Variant Generation Scope

Prioritize coarse-grained operators:

```text
matmul / batched matmul
conv
large transformer subgraph
attention block if runtime supports
MLP block
```

Fine-grained elementwise offload should not be the mainline because transfer and launch overhead may dominate.

## Legality Rules

Offload plugin checks:

- runtime is available;
- C ABI or vendor API can link;
- input/output dtypes are supported;
- shapes meet runtime/operator constraints;
- buffer ownership is explicit;
- synchronization boundary is complete;
- host-device copy requirements are represented;
- fallback variant exists;
- async call is allowed when used.

## Cost Model

Offload cost includes:

```text
host-device transfer cost
runtime launch overhead
device compute cost
sync/wait cost
buffer reuse benefit
batching benefit
RVV fallback cost
```

Selection depends on:

```text
shape size
dtype
operator type
whether data is already on device
runtime availability
batching opportunity
```

## Tuning Space

Offload tuning includes:

```text
offload threshold
batch size
transfer granularity
async overlap
buffer reuse policy
host fallback threshold
```

This is runtime/offload decision tuning, not ordinary tile-size tuning.

## Difference From Custom ISA Plugin

Custom ISA plugin:

```text
custom intrinsic / inline asm / backend patch / object stub
```

Runtime offload plugin:

```text
C ABI / runtime library / driver call
```

Both can be capability-managed, but they must not be conflated.

## Sophgo Role

Sophgo route may prove:

- RISC-V host plus accelerator runtime can be capability-managed;
- offload can be represented as execution variant;
- RVV CPU and offload variants can coexist under dispatch;
- runtime capability fits unified capability model.

It does not prove RISC-V custom ISA support.
