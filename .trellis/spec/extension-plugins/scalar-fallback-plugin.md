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
Its descriptor fields are bounded implementation debt, not the architecture for
adding new computation.

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
lowering descriptor values:
  i32-vadd-microkernel.v1
  i32-vsub-microkernel.v1
  i32-vmul-microkernel.v1
  i64-vadd-microkernel.v1
  i64-vsub-microkernel.v1
  i64-vmul-microkernel.v1
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
bounded i32/i64 vector add/sub/mul fallback source slice:

```text
tcrv_scalar.lowering_descriptor = "i32-vadd-microkernel.v1"
tcrv_scalar.lowering_descriptor = "i32-vsub-microkernel.v1"
tcrv_scalar.lowering_descriptor = "i32-vmul-microkernel.v1"
tcrv_scalar.lowering_descriptor = "i64-vadd-microkernel.v1"
tcrv_scalar.lowering_descriptor = "i64-vsub-microkernel.v1"
tcrv_scalar.lowering_descriptor = "i64-vmul-microkernel.v1"
tcrv_scalar.element_count = 16 : i64
```

The descriptor is a compiler decision handle for one tiny plugin-owned
microkernel attachment. `tcrv_scalar.element_count` is descriptor-local bounded
metadata only; it is not high-level shape, problem size, AVL, vl, runtime loop
trip count, or performance evidence.
Bounded frontend lowering may preserve
`tcrv_frontend_lowering = "i32-vadd"`, `"i32-vsub"`, `"i32-vmul"`,
`"i64-vadd"`, `"i64-vsub"`, or `"i64-vmul"` on the generated kernel. The
scalar plugin must select the matching descriptor and must not use another
family descriptor, route id, ABI name, runtime glue role, operation label, C
type, or emitted arithmetic for subtract or multiply.

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
family-specific `tcrv_scalar.i32_*_microkernel` or
`tcrv_scalar.i64_*_microkernel`. It does not mean that
TianChen-RV emitted LLVM IR, generated an object, linked a runtime, executed a
scalar kernel, proved correctness, or measured performance.

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
`tcrv_scalar.lowering_descriptor = "i32-vadd-microkernel.v1"` or
`"i32-vsub-microkernel.v1"` or `"i32-vmul-microkernel.v1"` or
`"i64-vadd-microkernel.v1"` or `"i64-vsub-microkernel.v1"` or
`"i64-vmul-microkernel.v1"` and a valid descriptor-local
`tcrv_scalar.element_count`, the same plugin-local materialization step must
also create exactly one matching direct-child
`tcrv_scalar.i32_vadd_microkernel` or `tcrv_scalar.i32_vsub_microkernel` or
`tcrv_scalar.i32_vmul_microkernel` or `tcrv_scalar.i64_vadd_microkernel` or
`tcrv_scalar.i64_vsub_microkernel` or
`tcrv_scalar.i64_vmul_microkernel`. A pre-existing matching scalar microkernel
for that selected path is rejected during descriptor materialization so that
the descriptor has a single owner and stale hand-authored fallback bodies
cannot silently replace the plugin-owned proposal.

Downstream emission planning may consume this boundary as the validated
selected-path attachment point before materializing either metadata-only
diagnostics or, when the matching microkernel exists, a supported scalar C
source-export plan. The lowering boundary itself still records selected-path
metadata only. It is not LLVM lowering, object generation, linked runtime glue,
hardware execution, correctness evidence, or performance evidence.

Real scalar fallback lowering beyond the bounded i32/i64 vector add/sub/mul C
source microkernel family must be added by later plugin-local lowering slices
and validated with compiler-generated artifacts and runtime evidence
appropriate to those paths.

## Explicit I32/I64 Vector Add/Sub/Mul Microkernel Export

`tcrv_scalar.i32_vadd_microkernel` and
`tcrv_scalar.i32_vsub_microkernel` and
`tcrv_scalar.i32_vmul_microkernel` and
`tcrv_scalar.i64_vadd_microkernel` and
`tcrv_scalar.i64_vsub_microkernel` and
`tcrv_scalar.i64_vmul_microkernel` are scalar extension-dialect executable
source-export microkernel ops for the bounded i32/i64 add/sub/mul family. Each
represents exactly one bounded callable body for a selected scalar fallback
path. The ops are plugin-local under the `tcrv_scalar` dialect and must carry
only selected-path metadata: source kernel, selected variant, origin, selected
role, required capability refs, and a tiny element count. They must reject
generic tensor/tile/benchmark attributes, unbounded or secret-like strings,
invalid element counts, stale selected variants, missing or unavailable scalar
fallback capability refs, and required-capability mismatches.

When the selected scalar fallback path has exactly one matching
`tcrv_scalar.lowering_boundary` and exactly one matching
`tcrv_scalar.i32_vadd_microkernel` or
`tcrv_scalar.i32_vsub_microkernel` or
`tcrv_scalar.i32_vmul_microkernel` or `tcrv_scalar.i64_vadd_microkernel` or
`tcrv_scalar.i64_vsub_microkernel` or `tcrv_scalar.i64_vmul_microkernel`,
including the microkernel materialized from the finite descriptor above, the
scalar plugin may return a supported runtime-callable C source export route:

```text
add emission kind: scalar-explicit-i32-vadd-microkernel-c-source
add lowering pipeline: tcrv-export-scalar-microkernel-c
add runtime ABI: scalar-i32-vadd-runtime-callable-c-abi.v1
add runtime ABI name: scalar-i32-vadd-runtime-callable-c-function.v1
add runtime glue role: runtime-callable-i32-vadd-fallback-function
sub emission kind: scalar-explicit-i32-vsub-microkernel-c-source
sub lowering pipeline: tcrv-export-scalar-i32-vsub-microkernel-c
sub runtime ABI: scalar-i32-vsub-runtime-callable-c-abi.v1
sub runtime ABI name: scalar-i32-vsub-runtime-callable-c-function.v1
sub runtime glue role: runtime-callable-i32-vsub-fallback-function
mul emission kind: scalar-explicit-i32-vmul-microkernel-c-source
mul lowering pipeline: tcrv-export-scalar-i32-vmul-microkernel-c
mul runtime ABI: scalar-i32-vmul-runtime-callable-c-abi.v1
mul runtime ABI name: scalar-i32-vmul-runtime-callable-c-function.v1
mul runtime glue role: runtime-callable-i32-vmul-fallback-function
i64 add emission kind: scalar-explicit-i64-vadd-microkernel-c-source
i64 add lowering pipeline: tcrv-export-scalar-i64-vadd-microkernel-c
i64 add runtime ABI: scalar-i64-vadd-runtime-callable-c-abi.v1
i64 add runtime ABI name: scalar-i64-vadd-runtime-callable-c-function.v1
i64 add runtime glue role: runtime-callable-i64-vadd-fallback-function
i64 sub emission kind: scalar-explicit-i64-vsub-microkernel-c-source
i64 sub lowering pipeline: tcrv-export-scalar-i64-vsub-microkernel-c
i64 sub runtime ABI: scalar-i64-vsub-runtime-callable-c-abi.v1
i64 sub runtime ABI name: scalar-i64-vsub-runtime-callable-c-function.v1
i64 sub runtime glue role: runtime-callable-i64-vsub-fallback-function
i64 mul emission kind: scalar-explicit-i64-vmul-microkernel-c-source
i64 mul lowering pipeline: tcrv-export-scalar-i64-vmul-microkernel-c
i64 mul runtime ABI: scalar-i64-vmul-runtime-callable-c-abi.v1
i64 mul runtime ABI name: scalar-i64-vmul-runtime-callable-c-function.v1
i64 mul runtime glue role: runtime-callable-i64-vmul-fallback-function
runtime ABI kind: scalar-runtime-callable-c-abi
status: supported
artifact kind: runtime-callable-c-source
```

The exported source must contain a deterministic callable C function with
family-derived pointer types (`int32_t` for i32 families, `int64_t` for i64
families) and `size_t n`. The production function body must be rendered from
the typed scalar microkernel op's common `TCRVEmitCLowerableRoute`: the route
carries the ABI value mappings, runtime-element-count loop boundary, and
ordered `emitc.call_opaque` compute/store steps. Scalar target code may emit
bounded target-owned inline helper definitions for callees such as
`tcrv_scalar_i32_add`, `tcrv_scalar_i32_sub`, `tcrv_scalar_i32_mul`, and their
i64 counterparts, but the callable body must invoke the route-authored callee
sequence rather than reconstructing `out[index] = lhs[index] <op> rhs[index]`
from descriptor or family metadata. No route may inherit stale route, C type,
or ABI metadata from another family. The callable ABI plan must be built from
direct
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

The same validated callable source candidate may also feed two bounded scalar
target artifact helpers. The i32-vadd route keeps the legacy helper names for
compatibility, while the other finite i32/i64 add/sub/mul families use
family-derived route ids:

```text
legacy add header route: tcrv-export-scalar-microkernel-header
header artifact kind: runtime-callable-c-header
legacy add object route: tcrv-export-scalar-microkernel-object
object artifact kind: riscv-elf-relocatable-object
family header route pattern:
  tcrv-export-scalar-<family>-microkernel-header
family object route pattern:
  tcrv-export-scalar-<family>-microkernel-object
```

The header route emits only the single external C prototype derived from the
same IR-backed callable ABI plan. It must not embed a body, RVV intrinsics,
`main`, self-check helpers, runtime probing, artifact paths, credentials, or
performance text.

The object route emits the default scalar library-style source internally and
uses local `clang` to compile a RISC-V ELF relocatable object. It is selected
only through the generic artifact front door after the scalar callable source
candidate has passed the typed ABI preflight. The route also requires
structured target/toolchain facts: an available `rv64` capability provider with
`architecture = "riscv64"` or `arch = "riscv64"`, plus available selected march
metadata from `riscv.toolchain.march.value`,
`rvv.probe.compile_run.selected_march`, or `rvv.toolchain.march.value`.
Optional selected MABI metadata may come from `riscv.toolchain.mabi.value`,
`rvv.probe.compile_run.selected_mabi`, or `rvv.toolchain.mabi.value`.

When source/header/object helpers all match the same scalar callable candidate,
target artifact bundle export may emit a deterministic scalar bundle containing
the C source, C header, and RISC-V relocatable object with matching selected
path and runtime ABI metadata. These helpers still do not add generic scalar
lowering, arbitrary scalar source export, linking, runtime dispatch
integration, broad correctness coverage, RVV hardware evidence, or performance
evidence. Scalar fallback selected paths without a valid descriptor or explicit
matching microkernel remain metadata-only. Header/object helper routes are
bounded to the same finite add/sub/mul i32/i64 scalar family set as the source
routes; a helper route must reject stale source candidates from a different
family before header/object output.

The finite scalar source/header/object target artifact routes are contributed
through the `scalar-plugin` plugin-owned target exporter bundle. Central
built-in target exporter composition may install the active plugin bundle, but
must not directly publish scalar fallback route ids or duplicate scalar family
validation. If `scalar-plugin` is missing or disabled, scalar microkernel
source/header/object routes must be absent and later artifact export must fail
closed through the generic route selection diagnostics.
