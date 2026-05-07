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
materialized requires form: requires = [@scalar_fallback]
generic policy: portable_scalar_fallback_first_slice
generic fallback role: fallback_role = "conservative"
```

The first-slice scalar plugin may propose `@scalar_fallback_first_slice` only
when the request contains a real high-level MLIR operation, a
`tcrv.exec.kernel`, and a `TargetCapabilitySet` where capability id
`scalar.fallback` is explicitly available. Missing or unavailable fallback
capability must produce no proposal rather than an implicit always-available
variant.

## Capability And Legality

Scalar fallback is still capability-driven:

- target profiles must declare `scalar.fallback` as a structured
  `tcrv.exec.capability`;
- generated variants must require that capability through `requires`;
- plugin legality must reject variants with missing origin, missing fallback
  capability requirement, or unavailable fallback capability;
- core passes must not special-case scalar fallback by name.

## Cost And Selection

The first slice uses plugin-owned cost metadata to mark scalar fallback as a
conservative coverage path. It should not be treated as a performance baseline
unless a later task adds real lowering, runtime, input metadata, and measured
evidence.

Selection and dispatch must reuse the existing generic variant-selection
protocol. Scalar fallback does not introduce a new core fallback selector: the
plugin marks the proposal/cost estimate with the abstract conservative fallback
role, and the core consumes only that generic role.

## Emission Boundary

The first scalar fallback emission readiness path is a metadata-only route for
compiler decisions:

```text
readiness status: metadata-only
emission kind: portable-scalar-fallback-metadata-route
lowering pipeline: none-executable-metadata-only
runtime ABI: none-metadata-only
artifact kind: metadata-diagnostic
```

This metadata-only emission-plan diagnostic records plugin-owned intent only. It
does not mean that TianChen-RV emitted LLVM IR, generated an object, linked a
runtime, executed a scalar kernel, proved correctness, or measured performance.

Real scalar fallback lowering must be added by a later plugin-local lowering
slice and validated with compiler-generated artifacts and runtime evidence
appropriate to that path.
