# Scalar Fallback Plugin

## Role

The scalar fallback plugin provides the portable fallback slot for TianChen-RV
execution variants. It is a coverage and correctness-oriented path used when a
target profile explicitly exposes a fallback capability.

It is not the primary performance path and must not be reported as RVV, IME,
offload, or hardware-specific evidence.

## First C++ Plugin Slice

The first scalar fallback C++ slice is intentionally bounded. It proves that a
fallback implementation can participate through the same plugin protocol as RVV
without adding extension-specific branches to core orchestration.
Legacy descriptor-shaped field names are deletion residue or fail-closed
implementation debt, not the architecture for adding new computation.

Stable first-slice names:

```text
plugin name: scalar-plugin
plugin version: 0.1.0
plugin capability id: scalar.fallback
plugin capability kind: fallback
preferred kernel capability symbol: @scalar_fallback
first-slice proposal / variant symbol: @scalar_fallback_first_slice
variant origin: scalar-plugin
required capability id: scalar.fallback
materialized requires form: requires = [@scalar_fallback] for an exact scalar
  fallback capability, or a relation-provider symbol such as
  [@module_fallback_profile] when that capability structurally `provides` or
  `implies` id `scalar.fallback`
generic policy: portable_scalar_fallback_first_slice
generic fallback role: fallback_role = "conservative"
deleted legacy selected-path element attr: tcrv_scalar.element_count
```

The first-slice scalar plugin may propose `@scalar_fallback_first_slice` only
when the request contains a real high-level MLIR operation, a
`tcrv.exec.kernel`, and a `TargetCapabilitySet` where capability id
`scalar.fallback` is available either as an exact capability id or through an
explicit structured relation-provider capability whose `provides` or `implies`
list satisfies id `scalar.fallback`. Missing or unavailable fallback capability
must produce no proposal rather than an implicit always-available variant.

The default proposal for the scalar fallback source slice is descriptorless and
must not derive a finite i32/i64 add/sub/mul family from frontend metadata,
bridge metadata, default family tables, absent body state, or hand-authored
microkernel names. There is no current typed scalar selected-path compute
authority. Historical markers such as `tcrv_frontend_lowering` and
`tcrv_scalar.element_count` are deleted-route residue only; the active scalar
plugin does not consume them as proposal, legality, selection, or boundary
authority.

## Capability And Legality

Scalar fallback is still capability-driven:

- target profiles must declare `scalar.fallback` as a structured exact
  `tcrv.exec.capability` or explicitly provide/imply it from a structured
  capability-provider `tcrv.exec.target`;
- generated variants must require that capability through `requires`;
- plugin legality must reject variants with missing origin, missing fallback
  capability requirement, or unavailable fallback capability;
- core passes must not special-case scalar fallback by name.

## Cost And Selection

The first slice uses plugin-owned preference/cost metadata to mark scalar
fallback as a conservative coverage path. Its explicit preference is deliberately
weaker than plugin-specific legal variants such as RVV, so it wins only when no
better legal plugin-owned path is selected. It should not be treated as a
performance baseline unless a later task adds real lowering, runtime, input
metadata, and measured evidence.

Selection and dispatch must reuse the existing generic variant-selection
protocol. Scalar fallback does not introduce a new core fallback selector: the
plugin marks the proposal/cost estimate with the abstract conservative fallback
role, and the core consumes only that generic role.

## Unsupported Emission Boundary

Scalar fallback is currently a fail-closed compiler-planning route. A selected
scalar fallback proposal may remain visible to generic selection, dispatch, and
coherence checks, but it must not become scalar compute, a metadata-only
emission route, a runtime ABI, or target artifact authority:

```text
readiness status: unsupported
emission kind: scalar-fallback-unsupported-emission
lowering pipeline: scalar-fallback-no-materialized-emitc-route
runtime ABI: scalar-fallback-no-runtime-abi
runtime ABI kind: unsupported-plugin-runtime-abi
runtime ABI name: unsupported-emission-runtime-abi
runtime glue role: no-runtime-glue-unsupported
required capabilities: selected scalar fallback variant required capability refs
artifact kind: unsupported-emission-diagnostic
```

This unsupported emission-plan diagnostic records a missing rebuild route only.
It does not mean that TianChen-RV emitted LLVM IR, generated an object, linked a
runtime, executed a scalar kernel, proved correctness, or measured performance.
It also does not authorize metadata-alone selected lowering-boundary
materialization.

## Selected Lowering Boundary

Scalar fallback currently has no active selected lowering-boundary operation.
Generic fallback-only and dispatch-fallback paths may keep selected `tcrv.exec`
metadata, but they must not ask the scalar plugin to synthesize a scalar
plugin-local boundary, runtime ABI operations, or a typed scalar microkernel
from descriptorless no-body state, kernel frontend markers, bridge metadata, or
a default family.

Architectural family:

```text
tcrv.scalar
```

Reserved MLIR namespace:

```text
tcrv_scalar
```

The reserved namespace is for a later scalar rebuild slice. It is not a current
boundary surface and must not be treated as executable lowering, metadata-only
emission, runtime ABI glue, object generation, correctness evidence, or
performance evidence. Generic fallback-only and dispatch-fallback paths must not
cause a missing-plugin diagnostic merely because no scalar boundary is
materialized.

The scalar plugin must not create any finite-family scalar microkernel from
defaults, RVV-scalar bridge metadata, hand-authored deleted-op syntax, or any
equivalent metadata selector. Absence of a body is an unsupported fallback
state, not a compute authority.

Downstream emission planning may emit only the unsupported diagnostic above for
selected scalar fallback. It must not consume a scalar boundary as route
authority because no active scalar boundary route exists.

Real scalar fallback lowering must be added by a later plugin-local rebuild
slice through extension-family ops and a materialized MLIR EmitC route, then
validated with compiler-generated artifacts and runtime evidence appropriate to
that path.

## Deleted Finite-Family Scalar Microkernel Ops

Historical scalar extension-dialect microkernel source-export surfaces are
deleted. Their old finite dtype/arithmetic spelling set must not remain as
active dialect ops, EmitC-lowerable ops, inert future-route IR,
selected-boundary authority, runtime ABI authority, generated-artifact
authority, correctness evidence, or performance evidence.

Scalar fallback direct runtime-callable C source/header/object routes are
deleted production routes. Generic target artifact front doors must fail closed
for historical scalar direct route ids until a rebuilt path materializes a real
MLIR EmitC module and emits C/C++ through the MLIR emitter.
