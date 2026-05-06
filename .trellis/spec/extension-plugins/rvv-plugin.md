# RVV Plugin

## Role

RVV plugin is the current primary real hardware path for TianChen-RV MLIR.

Environment:

```text
access: ssh rvv
hardware: RISC-V CPU, 64 cores
vector: RVV 1.0
permission: sudo available
role: primary development, primary performance, primary correctness
```

## Responsibilities

RVV plugin provides:

- RVV capability registration;
- `tcrv.rvv` dialect registration;
- RVV variant generation;
- RVV legality verification;
- RVV tuning space;
- RVV cost model;
- RVV emission paths;
- RVV runtime/threading integration.

It does not define high-level matmul semantics, generic tensor/tile IR, IME internals, offload internals, or custom ISA internals.

## First C++ Plugin Slice

The first concrete C++ RVV slice is intentionally narrower than the full RVV
plugin above. It proves plugin identity, capability participation, proposal
metadata, materialization, legality routing, and selection consumption through
`ExtensionPluginRegistry`.

Stable first-slice names:

```text
plugin name: rvv-plugin
plugin version: 0.1.0
plugin capability id: rvv
plugin capability kind: isa-vector
preferred kernel capability symbol: @rvv
first-slice proposal / variant symbol: @rvv_first_slice
variant origin: rvv-plugin
required capability id: rvv
materialized requires form: requires = [@rvv]
typed policy attr name: tcrv_rvv.policy
typed policy attr value: #tcrv_rvv.policy<tail = agnostic, mask = agnostic>
```

The first-slice RVV plugin may propose `@rvv_first_slice` only when the request
contains a real high-level MLIR operation, a `tcrv.exec.kernel`, and a
`TargetCapabilitySet` where capability id `rvv` is explicitly available. If
the capability is missing or generically unavailable (`status = "disabled"`,
`"missing"`, or `"unavailable"`), the plugin proposes no variant.

The first slice carries generic decision metadata (`condition`, `guard`, and
`policy`), one typed non-compute RVV policy attribute
(`tcrv_rvv.policy = #tcrv_rvv.policy<tail = agnostic, mask = agnostic>`), and a
plugin-owned neutral cost estimate. The generic string policy remains the input
for core selection/dispatch; the typed `tcrv_rvv.policy` attribute is
plugin-local metadata preserved by the generic proposal/materialization path and
validated by `RVVExtensionPlugin`. These fields are compiler-visible metadata
for the existing generic materialization, legality, capability, and selection
helpers. They are not RVV lowering, code emission, runtime ABI, correctness
evidence, or performance evidence.

The RVV first slice participates in the generic emission-readiness protocol
only to report an explicit unsupported status. Because this slice has no RVV
ops, lowering patterns, runtime ABI, executable kernel, toolchain invocation,
or runtime proof, `RVVExtensionPlugin` must not return a supported emission
path for `@rvv_first_slice`. The unsupported reason is a diagnostic boundary
that prevents accidental RVV lowering/runtime claims until a later slice adds
real lowering and `ssh rvv` evidence.

`registerDialects` now registers the minimal RVV dialect skeleton through the
RVV plugin path. The default `registerAllDialects` path remains core-only; RVV
dialect availability is proven by populating an `ExtensionPluginRegistry` with
the RVV plugin and calling `registerPluginDialects`.

The first RVV dialect slice is still metadata/control-plane only. It introduces
the vector-length token type `!tcrv_rvv.vl` and the finite policy attribute
`#tcrv_rvv.policy<tail = agnostic|undisturbed, mask =
agnostic|undisturbed>`. These surfaces do not introduce RVV ops, lowering,
emission, runtime ABI, executable behavior, correctness claims, or performance
claims. `tcrv_rvv` is the concrete MLIR dialect namespace because MLIR dialect
namespaces cannot contain `.` characters; the architectural extension family
remains `tcrv.rvv`.

## Capability Fields

RVV plugin should register and query:

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

Reference attribute:

```mlir
#tcrv.ext<"rvv",
          kind = "isa-vector",
          version = "1.0",
          vlen = 128,
          supports_mask = true,
          supports_tail_policy = true>
```

## Current Dialect Skeleton

Architectural family:

```text
tcrv.rvv
```

Concrete MLIR namespace:

```text
tcrv_rvv
```

Current first-slice type:

```text
!tcrv_rvv.vl
```

Current first-slice policy attribute:

```text
#tcrv_rvv.policy<tail = agnostic, mask = agnostic>
```

The type is a non-compute vector-length token used to prove plugin-local dialect
registration, parser/printer ownership, and enabled-plugin registry behavior.
The policy attribute is finite non-compute metadata for proposal preservation
and RVV plugin-local legality. Neither surface is `vsetvl`, a vector register,
a mask, a memory operation, lowering, runtime ABI, correctness evidence, or
performance evidence.

## Future Dialect Surface

Future RVV execution dialect work may add richer types:

```text
!tcrv.rvv.vreg<dtype, lmul>
!tcrv.rvv.mask<lmul>
!tcrv.rvv.vl
!tcrv.rvv.policy<tail, mask>
```

Future RVV execution dialect work may add ops:

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
tcrv.rvv.slide / gather / compress
tcrv.rvv.convert
```

These are RVV execution ops, not high-level tensor ops.

## Variant Generation

RVV plugin may support high-level op classes such as:

```text
matmul / batched matmul
softmax
layernorm / rmsnorm
rope
elementwise + reduction fusion
attention micro-kernel fragments
```

Output must be a `tcrv.exec.variant` containing `tcrv.rvv.*` ops, not a generic `tcrv.matmul`.

## Legality Rules

RVV plugin checks:

- target supports RVV;
- required dtype has corresponding extension or fallback path;
- SEW/LMUL combination is legal;
- VL policy is expressible;
- mask/tail policy is complete;
- load/store pattern is RVV-expressible;
- reduction handles tail/mask correctly;
- selected toolchain supports the emission path.

## Tuning Space

RVV tuning is variant quality metadata:

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

Reference metadata:

```mlir
#tcrv.tuning<
  lmul = 4,
  sew = 16,
  k_block = 64,
  unroll = 2,
  thread_partition = "row_block"
>
```

## Emission Paths

Current first slice:

```text
rvv-plugin @rvv_first_slice -> unsupported emission readiness
reason: metadata/control-plane only; no RVV lowering/runtime/executable path
```

This unsupported readiness result is required. It is not a failure of the
architecture and is not RVV hardware, toolchain, runtime, correctness, or
performance evidence.

Public `tcrv-opt` registers the built-in RVV plugin at the tool boundary, so
materialized variants with `origin = "rvv-plugin"` can route through
`RVVExtensionPlugin` for emission-readiness and emission-plan diagnostics. The
result remains explicitly unsupported metadata: the plugin reports no RVV
lowering pipeline, runtime ABI, artifact contract, executable emission path,
hardware execution, correctness result, or performance result. Unknown origins
must still fail through the generic unregistered-origin registry diagnostic.
Tests that need the historical empty-registry parser surface should pass
`--tcrv-disable-builtin-plugins`.

### MLIR vector / LLVM scalable vector

Use for ordinary vector arithmetic, load/store, and reductions that LLVM reliably lowers.

### RVV intrinsic / inline asm / builtin

Use when precise `vsetvl`, mask/tail policy, segment/strided ops, or key-kernel stability requires explicit control.

## Hart Parallelism

RVV plugin does not define a thread/block model. Multi-core execution is organized by `tcrv.exec.hart_parallel`.

RVV plugin provides per-hart RVV execution behavior and lowering preferences for OpenMP, pthread, runtime thread pool, or single-thread paths.

## Diagnostics

Diagnostics must report:

- illegal LMUL/SEW;
- unsupported dtype;
- missing mask/tail policy;
- unavailable emission path;
- unsuitable memory pattern;
- unsatisfied capability.
