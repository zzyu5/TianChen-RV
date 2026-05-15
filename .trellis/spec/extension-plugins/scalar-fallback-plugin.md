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
selected-path element attr: tcrv_scalar.element_count
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
bridge metadata, default family tables, or absent body state. Only explicit
typed scalar microkernel family ops that already exist on the selected path can
act as typed selected-path compute authority for future lowering work.
`tcrv_scalar.element_count` is bounded selected-path metadata only; it is not
high-level shape, problem size, AVL, vl, runtime loop trip count, or performance
evidence. Kernel metadata such as `tcrv_frontend_lowering = "i32-vsub"` is a
deleted old authority: a scalar plugin must reject or ignore non-empty
frontend-lowering markers rather than using them to select vadd/vsub/vmul,
i32/i64 families, route ids, ABI names, runtime glue roles, operation labels,
C types, or emitted arithmetic.

## Capability And Legality

Scalar fallback is still capability-driven:

- target profiles must declare `scalar.fallback` as a structured exact
  `tcrv.exec.capability` or explicitly provide/imply it from a structured
  capability-provider `tcrv.exec.target`;
- generated variants must require that capability through `requires`;
- plugin legality must reject variants with missing origin, missing fallback
  capability requirement, unavailable fallback capability, malformed
  `tcrv_scalar.element_count`, or selected-path element counts outside the
  bounded first-slice range `[1, 64]`;
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

## Metadata-Only Emission Boundary

Scalar fallback is currently a metadata-only route for compiler decisions:

```text
readiness status: metadata-only
emission kind: portable-scalar-fallback-metadata-route
lowering pipeline: none-executable-metadata-only
runtime ABI: none-metadata-only
runtime ABI kind: host-scalar-fallback-metadata
runtime ABI name: portable-scalar-fallback-metadata-abi.v1
runtime glue role: metadata-only-host-fallback-boundary
required capabilities: selected scalar fallback variant required capability refs
artifact kind: metadata-diagnostic
```

This metadata-only emission-plan diagnostic records plugin-owned intent only.
It applies even if a hand-authored scalar microkernel attachment is present in
the selected kernel. Such attachments remain dialect/future-route IR only; the
scalar plugin must not interpret them as current callable microkernel family
authority, direct-C source authority, runtime ABI authority, generated-artifact
authority, correctness evidence, or performance evidence. It does not mean that
TianChen-RV emitted LLVM IR, generated an object, linked a runtime, executed a
scalar kernel, proved correctness, or measured performance.
It also does not authorize metadata-alone selected lowering-boundary
materialization: a selected scalar fallback variant carrying only
`tcrv_scalar.element_count` metadata must fail closed before
`tcrv_scalar.lowering_boundary` creation.

## Selected Lowering Boundary

Scalar fallback participates in the generic selected lowering-boundary registry
path. Its first selected-boundary slice validates the selected scalar fallback
variant through the same plugin-local legality rules, then materializes a
plugin-local scalar metadata operation. A descriptorless no-body scalar fallback
path may materialize this metadata-only boundary, but it must not synthesize a
typed scalar microkernel body, derive family authority from a typed scalar
microkernel body, or insert runtime ABI `tcrv.exec.mem_window` /
`tcrv.exec.runtime_param` operations from scalar/RVV-scalar bridge metadata.

```mlir
tcrv_scalar.lowering_boundary {
  source_kernel = "kernel_symbol",
  selected_variant = @scalar_fallback_first_slice,
  origin = "scalar-plugin",
  role = "direct variant",
  status = "metadata-only",
  required_capabilities = [@scalar_fallback],
  fallback_reason = "portable scalar fallback metadata boundary"
}
```

Architectural family:

```text
tcrv.scalar
```

Concrete MLIR namespace:

```text
tcrv_scalar
```

The boundary op is a direct child of a `tcrv.exec.kernel`, references a direct
sibling selected scalar fallback `tcrv.exec.variant`, and carries the selected
variant's required capability symbol references. It must not materialize
`tcrv_rvv` operations, must not be treated as executable lowering, and must not
cause a missing-plugin diagnostic when selected as a fallback-only or dispatch
fallback path.

When the selected variant is descriptorless and no explicit typed scalar body is
present, the scalar plugin must not create `tcrv_scalar.i32_vadd_microkernel` or
any other finite-family microkernel from defaults, RVV-scalar bridge metadata,
`tcrv_frontend_lowering`, `tcrv_scalar.element_count`, or any equivalent
metadata selector. Absence of a body is a metadata-only fallback state, not a
compute authority.

When an explicit scalar microkernel body already exists in the selected kernel,
lowering-boundary materialization may leave that operation as inert input IR,
but must not use it to select a finite family, validate selected element-count
mirrors, synthesize ABI windows/params, or switch emission readiness/planning to
a direct-C unsupported microkernel branch. A selected scalar fallback variant
carrying `tcrv_scalar.element_count` metadata still fails closed before
`tcrv_scalar.lowering_boundary` creation; the metadata alone is not a current
boundary authority.

Downstream emission planning may consume this boundary only to materialize the
metadata-only diagnostic above. The lowering boundary itself still records
selected-path metadata only. It is not LLVM lowering, object generation, linked
runtime glue, hardware execution, correctness evidence, or performance
evidence.

Real scalar fallback lowering must be added by a later plugin-local rebuild
slice through extension-family ops and a materialized MLIR EmitC route, then
validated with compiler-generated artifacts and runtime evidence appropriate to
that path.

## Explicit I32/I64 Vector Add/Sub/Mul Microkernel Ops

`tcrv_scalar.i32_vadd_microkernel` and
`tcrv_scalar.i32_vsub_microkernel` and
`tcrv_scalar.i32_vmul_microkernel` and
`tcrv_scalar.i64_vadd_microkernel` and
`tcrv_scalar.i64_vsub_microkernel` and
`tcrv_scalar.i64_vmul_microkernel` are scalar extension-dialect typed
microkernel attachment ops for future scalar lowering work. They are not active
callable bodies for the current scalar fallback plugin. The ops are plugin-local
under the `tcrv_scalar` dialect and must carry
only selected-path metadata: source kernel, selected variant, origin, selected
role, required capability refs, and a tiny element count. They must reject
generic tensor/tile/benchmark attributes, unbounded or secret-like strings,
invalid element counts, stale selected variants, missing or unavailable scalar
fallback capability refs, and required-capability mismatches.

Scalar fallback direct runtime-callable C source/header/object routes are
deleted production routes. A selected scalar fallback path may still materialize
typed scalar extension ops for future lowering work, but the scalar plugin must
not advertise selected metadata, typed scalar ops, or family records as
supported or specially unsupported executable C artifacts. Generic target
artifact front doors must fail closed for historical scalar direct route ids
until a rebuilt path materializes a real MLIR EmitC module and emits C/C++
through the MLIR emitter.
