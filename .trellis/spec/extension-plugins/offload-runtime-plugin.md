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
materialization, plugin-owned runtime ABI handoff metadata, emission manifest
serialization, and target-owned descriptor artifact export through the existing
generic registry and target artifact interfaces.

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
materialized requires form: requires = [@offload_runtime] for an exact
  runtime-offload capability, or a relation-provider symbol such as
  [@module_offload_profile] when that capability structurally `provides` or
  `implies` id `offload.runtime`
runtime ABI handoff id: generic-runtime-offload-c-abi-handoff.v1
handoff kind: runtime-offload
descriptor route id: tcrv-export-offload-runtime-descriptor
descriptor artifact kind: runtime-offload-handoff-descriptor
```

The first slice may propose `@offload_runtime_first_slice` only when the kernel
declares an available structured capability provider for id `offload.runtime`.
That provider may be an exact capability with id `offload.runtime`, kind
`runtime-offload`, or an explicit relation-provider profile whose `provides` or
`implies` list satisfies id `offload.runtime`. The provider must carry bounded
generic properties:

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
capabilities must not enable the generic offload slice unless they explicitly
provide or imply `offload.runtime` and preserve the required generic handoff
properties.

The first slice carries generic decision metadata (`condition`, `guard`, and
`policy`) and plugin-owned discardable string metadata
`tcrv_offload.runtime_abi` plus `tcrv_offload.handoff_kind` on the materialized
variant. Plugin legality checks that the selected variant requires an exact or
relation-provider capability satisfying `offload.runtime`, that the provider
remains available, and that the plugin-owned variant metadata matches the
preserved capability properties. This metadata is compiler handoff metadata
only; it is not executable runtime glue.

The supported first-slice emission plan must also preserve bounded
`selected_plan_metadata` for the descriptor handoff. At minimum, it records the
generic runtime-offload capability id, the selected runtime-offload handoff
kind, and the descriptor-only evidence scope. Generic target artifact bundle
export and the offload runtime descriptor exporter may serialize these fields
as deterministic handoff metadata, but they remain compiler metadata and do not
claim vendor runtime execution.

The same supported descriptor plan must expose deterministic runtime ABI role
metadata derived from typed `tcrv.exec.mem_window` host-buffer declarations and
typed `tcrv.exec.runtime_param` scalar/control declarations. For the current
bounded i32-vadd callable ABI, the descriptor route requires the
`lhs-input-buffer`, `rhs-input-buffer`, `output-buffer`, and
`runtime-element-count` roles
to be mirrored as `runtime_abi_parameters` with stable C names, C types, roles,
and ownership. Descriptor-local metadata may describe the compiler handoff
contract, binding, access, memory space, and selected plan owner, but it must
not embed sample runtime values, tensor sizes, RVV guard values, hardware facts,
or vendor execution data.
The offload descriptor target exporter must register this required typed role
contract with the generic target artifact exporter registry. Before descriptor
text or bundle files are emitted, the generic target-artifact/front-door
preflight must reject missing roles or metadata that no longer mirrors the
IR-backed `mem_window` / `runtime_param` ABI plan. The offload descriptor
exporter still repeats route-local validation as the final safety net, but
shared target artifact code must not gain offload policy branches.

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

When the selected offload path has this matching lowering boundary and the
plugin-owned emission plan records the supported descriptor route, the target
artifact exporter may emit a deterministic text descriptor for runtime-offload
handoff metadata. The descriptor may contain only sanitized compiler-visible
fields such as source kernel, selected variant, origin plugin, required
capability refs, descriptor schema version, descriptor kind/status, adapter
contract, runtime ABI kind/name, emission kind, artifact kind,
lowering-boundary op name/status, runtime-offload handoff kind, handoff reason,
selected-plan handoff metadata, artifact component role/evidence role, and
explicit non-claim metadata. The generic target artifact bundle exporter may
materialize the selected non-fallback offload path as one descriptor artifact
with an index entry carrying route, owner/origin, runtime ABI, selected-plan
metadata, and handoff kind metadata. It is not runtime execution, vendor
integration, DMA, object generation, correctness evidence, or performance
evidence.

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
