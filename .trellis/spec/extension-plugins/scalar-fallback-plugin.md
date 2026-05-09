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
materialized requires form: requires = [@scalar_fallback] for an exact scalar
  fallback capability, or a relation-provider symbol such as
  [@module_fallback_profile] when that capability structurally `provides` or
  `implies` id `scalar.fallback`
generic policy: portable_scalar_fallback_first_slice
generic fallback role: fallback_role = "conservative"
lowering descriptor attr: tcrv_scalar.lowering_descriptor
lowering descriptor value: i32-vadd-microkernel.v1
descriptor-local element attr: tcrv_scalar.element_count
default descriptor-local element count: 16
```

The first-slice scalar plugin may propose `@scalar_fallback_first_slice` only
when the request contains a real high-level MLIR operation, a
`tcrv.exec.kernel`, and a `TargetCapabilitySet` where capability id
`scalar.fallback` is available either as an exact capability id or through an
explicit structured relation-provider capability whose `provides` or `implies`
list satisfies id `scalar.fallback`. Missing or unavailable fallback capability
must produce no proposal rather than an implicit always-available variant.

The proposal also carries a finite plugin-owned lowering descriptor for the
bounded i32 vector-add fallback source slice:

```text
tcrv_scalar.lowering_descriptor = "i32-vadd-microkernel.v1"
tcrv_scalar.element_count = 16 : i64
```

The descriptor is a compiler decision handle for one tiny plugin-owned
microkernel attachment. `tcrv_scalar.element_count` is descriptor-local bounded
metadata only; it is not high-level shape, problem size, AVL, vl, runtime loop
trip count, or performance evidence.

## Capability And Legality

Scalar fallback is still capability-driven:

- target profiles must declare `scalar.fallback` as a structured exact
  `tcrv.exec.capability` or explicitly provide/imply it from a structured
  capability-provider `tcrv.exec.target`;
- generated variants must require that capability through `requires`;
- plugin legality must reject variants with missing origin, missing fallback
  capability requirement, unavailable fallback capability, malformed
  `tcrv_scalar.lowering_descriptor`, malformed `tcrv_scalar.element_count`, or
  descriptor-local element counts outside the bounded first-slice range
  `[1, 64]`;
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

Scalar fallback without a matching executable microkernel attachment is a
metadata-only route for compiler decisions:

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
It applies when the selected scalar fallback path has no validated matching
`tcrv_scalar.i32_vadd_microkernel`. It does not mean that TianChen-RV emitted
LLVM IR, generated an object, linked a runtime, executed a scalar kernel,
proved correctness, or measured performance.

## Selected Lowering Boundary

Scalar fallback participates in the generic selected lowering-boundary registry
path. Its first selected-boundary slice validates the selected scalar fallback
variant through the same plugin-local legality rules, then materializes the
plugin-local scalar metadata operation:

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

When the selected variant carries
`tcrv_scalar.lowering_descriptor = "i32-vadd-microkernel.v1"` and a valid
descriptor-local `tcrv_scalar.element_count`, the same plugin-local
materialization step must also create exactly one matching direct-child
`tcrv_scalar.i32_vadd_microkernel`. A pre-existing matching scalar microkernel
for that selected path is rejected during descriptor materialization so that the
descriptor has a single owner and stale hand-authored fallback bodies cannot
silently replace the plugin-owned proposal.

Downstream emission planning may consume this boundary as the validated
selected-path attachment point before materializing either metadata-only
diagnostics or, when the matching microkernel exists, a supported scalar C
source-export plan. The lowering boundary itself still records selected-path
metadata only. It is not LLVM lowering, object generation, linked runtime glue,
hardware execution, correctness evidence, or performance evidence.

Real scalar fallback lowering beyond the bounded i32 vector-add C source
microkernel must be added by later plugin-local lowering slices and validated
with compiler-generated artifacts and runtime evidence appropriate to those
paths.

## Explicit I32 Vector-Add Microkernel Export

`tcrv_scalar.i32_vadd_microkernel` is the first scalar extension-dialect
executable source-export microkernel op. It represents exactly one bounded i32
vector-add callable body for a selected scalar fallback path. The op is
plugin-local under the `tcrv_scalar` dialect and must carry only selected-path
metadata: source kernel, selected variant, origin, selected role, required
capability refs, and a tiny element count. It must reject generic
tensor/tile/benchmark attributes, unbounded or secret-like strings, invalid
element counts, stale selected variants, missing or unavailable scalar fallback
capability refs, and required-capability mismatches.

When the selected scalar fallback path has exactly one matching
`tcrv_scalar.lowering_boundary` and exactly one matching
`tcrv_scalar.i32_vadd_microkernel`, including the microkernel materialized from
the finite descriptor above, the scalar plugin may return a supported
runtime-callable C source export route:

```text
status: supported
emission kind: scalar-explicit-i32-vadd-microkernel-c-source
lowering pipeline: tcrv-export-scalar-microkernel-c
runtime ABI: scalar-i32-vadd-runtime-callable-c-abi.v1
runtime ABI kind: scalar-runtime-callable-c-abi
runtime ABI name: scalar-i32-vadd-runtime-callable-c-function.v1
runtime glue role: runtime-callable-i32-vadd-fallback-function
artifact kind: runtime-callable-c-source
```

The exported source must contain a deterministic callable C function with
`const int32_t *lhs`, `const int32_t *rhs`, `int32_t *out`, and `size_t n`
parameters. The callable ABI plan must be built from direct
`tcrv.exec.mem_window` IR for lhs/rhs/out buffer roles plus direct
`tcrv.exec.runtime_param` IR for runtime `n`; supported emission-plan
`runtime_abi_parameters` entries are validated mirrors of that IR-backed plan.
The generic target-artifact source front door must also validate those
compiler-owned runtime ABI parameter roles through the typed scalar callable
route contract before invoking the scalar source exporter, so stale or
conflicting C type, C name, ownership, or role metadata fails before source
output rather than slipping to late exporter behavior.
For this first slice those parameters are target/export ABI-owned, not
IR-modeled scalar operands or high-level tensor shape. The default artifact has
no hidden `main`, stdio-only self-check machinery, or success marker. The source
may include bounded metadata comments for selected kernel, selected variant,
selected role/fallback role, artifact kind, element count, required
capabilities, runtime ABI kind/name, runtime glue role, and ABI parameter roles.
Scalar fallback selected paths without a valid descriptor or explicit matching
microkernel remain metadata-only. The supported route does
not add generic scalar lowering, arbitrary scalar source export, object
generation, linking, full runtime dispatch integration, broad correctness
coverage, or performance evidence.
