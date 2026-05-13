# Emission Runtime Contract

## Core Principle

Core pass does not implement extension-specific emission. Each plugin owns its lowering/emission/runtime glue.

Variant selection and emission path must be represented in capability model and verified before final lowering.

The current main route is:

```text
TCRV extension family ops
  -> EmitC ops
  -> C/C++ emitter
  -> intrinsic / vendor builtin / runtime C ABI
  -> native compiler
```

The default native compiler is clang/LLVM. GCC is a compatibility path.
Vendor compilers are extension-specific compatibility paths.

Direct descriptor-to-C string export is not the architecture. Existing
descriptor-backed source/object/bundle helpers are bounded implementation debt
that must not be used as the template for new extension work.

## Emission Readiness First Slice

The first compiler-visible emission slice is a target-neutral readiness check.
It does not lower IR, generate runtime ABI glue, invoke a toolchain, run a
kernel, or prove correctness/performance. It verifies that current
`tcrv.exec` structure can be routed to the origin plugin that owns each
materialized variant.

Rules:

- route by the materialized `tcrv.exec.variant` `origin` attribute;
- carry the variant, enclosing `tcrv.exec.kernel`, generic
  `TargetCapabilitySet`, and target-neutral role (`direct variant`,
  `dispatch case`, or `dispatch fallback`);
- require supported plugin results to contain a non-empty plugin-owned emission
  path identifier or description;
- require unsupported plugin results to contain a non-empty reason;
- diagnose missing/empty origin, unknown plugin, disabled plugin, malformed
  plugin result, missing kernel/variant, and non-direct variant/kernel
  relationships generically;
- validate `tcrv.exec.dispatch` case/fallback targets structurally before
  routing to plugins: every reference must resolve to a direct sibling
  `tcrv.exec.variant` in the same kernel and duplicate references are invalid;
- when no dispatch is present, consume a direct selected-path
  `tcrv.exec.diagnostic` marker before falling back to all variants; the marker
  must carry a direct sibling variant `target` and a generic `selection_kind`
  such as `static-variant` or `fallback-only`;
- avoid core branches on RVV, IME, offload, scalar, vendor, dtype, shape,
  runtime, toolchain, or microarchitecture details.

## Emission Plan First Slice

After readiness and selected-path traversal, the compiler may collect
plugin-owned emission plans. An emission plan is a target-neutral compiler
decision object, not an executable artifact.

Rules:

- route by the same materialized variant `origin` plugin as readiness;
- carry kernel symbol, variant symbol, selected-path role, support status,
  origin plugin, runtime ABI ownership metadata, required capability refs, and
  diagnostic/explanation metadata;
- carry bounded runtime ABI kind/name metadata and a bounded required runtime
  glue role chosen by the origin plugin;
- for supported `runtime-callable-c-source` paths, carry structured
  `runtime_abi_parameters` metadata for every exported C ABI parameter. Each
  entry records the C parameter name, C type spelling, semantic role, and
  ownership (`ir-modeled` or `target-export-abi-owned`). For the bounded
  i32 binary RVV/scalar callable source routes, these entries must be derived
  from and validated against direct `tcrv.exec.mem_window` /
  `tcrv.exec.runtime_param` IR boundaries rather than acting as an independent
  parameter truth source. Add/sub/mul ABI identity fields must be derived from
  the selected finite binary runtime ABI contract keyed by selected family id;
- carry required capability symbol refs that are a safe subset of the selected
  variant `requires` metadata;
- for supported paths, require non-empty emission kind, lowering pipeline
  identifier, runtime ABI identifier, artifact kind, and explanation;
- for unsupported paths, require a non-empty diagnostic reason;
- reject plugin-returned empty runtime ABI kind/name, empty runtime glue role,
  missing required capability refs, or unbounded/secret-like diagnostic and
  explanation text before materialization;
- diagnose missing origin, unregistered origin, disabled origin plugin,
  malformed plugin result, mismatched variant symbol, mismatched kernel symbol,
  mismatched selected-path role, duplicate selected markers, and missing
  dispatch targets generically before treating the plan as usable;
- for selected dispatch or selected-marker paths after lowering-boundary
  materialization, require exactly one matching plugin-owned boundary operation
  before emission-plan diagnostics are materialized;
- validate selected-boundary metadata generically before plugin plan routing:
  `source_kernel` must match the enclosing kernel, `selected_variant` and
  `role` must match the selected path, `origin` must match the selected
  variant origin, `required_capabilities` must be a safe subset of the selected
  variant `requires`, and stale or duplicate competing boundaries are fatal;
- recognize selected lowering-boundary candidates only when they are actual
  plugin-local `*.lowering_boundary` operations or explicit lowering-boundary
  diagnostic metadata. A plugin-local executable attachment may also carry
  selected-path fields, but it is not the generic lowering-boundary operation
  consumed by emission planning;
- allow public tools to populate a deterministic built-in plugin registry before
  constructing registry-dependent passes, while keeping the traversal and
  selected-path routing target-neutral in shared pass code;
- keep the boundary clear: readiness says whether a selected path is
  supportable, while the plan describes the plugin-owned lowering/runtime route
  or the structured unsupported reason;
- do not claim generated code, runtime ABI glue, linked artifacts, RVV hardware
  execution, correctness, or performance from an emission plan alone.

## Target Artifact Export Route

Public artifact export tools may add a generic routing front door after
selected-path, lowering-boundary, and emission-plan metadata already exist in
MLIR. The generic layer consumes only target-neutral emission-plan fields such
as selected variant, role, origin, support status, lowering pipeline route id,
emission kind, artifact kind, lowering-boundary reference, runtime ABI metadata,
and required capability refs. It must fail closed for missing selected paths,
missing emission-plan diagnostics, unsupported or metadata-only plans, unknown
route ids, artifact-kind mismatch, origin/emission-kind mismatch, stale selected
paths, duplicate or ambiguous supported artifacts, and malformed bounded text.

Concrete artifact generation remains target-owned. Built-in tools may register
target exporters, but the registration is where extension-specific route facts
belong. Exporters that declare required runtime ABI parameter roles also verify
that any selected emission-plan diagnostic parameter metadata is an exact mirror
of the executable IR-backed ABI boundary consumed by the target-owned exporter,
including role, C name, C type, and ownership. Missing, spoofed, stale, or
extra required roles fail before target-owned C source, header, object, or
bundle emission. Composite helper routes that are matched from callable source
candidates, such as RVV header/object helpers and RVV+scalar dispatch
source/header/object helpers, must register route-local candidate preflight
callbacks when their component callable routes carry compiler-owned ABI role
contracts. Those callbacks must reuse the same typed `TargetArtifactExporter`
candidate validation surface rather than duplicating a string-only ABI model.
Standalone target exporters may also register a route-local candidate
validation callback when a static role list is not expressive enough to check
the full compiler-owned runtime ABI contract. The generic target artifact
front door and the execution-plan coherence gate must invoke that callback
after checking route id, artifact kind, origin, emission kind, export callback,
and required typed ABI roles, and before calling the route-specific exporter.
Such callbacks may validate generic compiler-owned ABI surfaces such as
`tcrv.exec.mem_window` and `tcrv.exec.runtime_param` role/name/type/purpose/
ownership/kind consistency for the selected candidate, but extension-specific
descriptor body policy remains target/export-local.
Shared generic routing must not branch on RVV, IME, offload, scalar, vendor,
dtype, shape, runtime, toolchain, or microarchitecture semantics. The
currently supported source routes are bounded explicit target/export-owned
artifacts: the RVV standalone smoke-probe C source exporter registered by RVV
target/export code for an explicitly planned smoke-probe path, the RVV i32
vector-add runtime-callable library C exporter registered by RVV target/export
code, the scalar fallback i32 vector add/sub portable runtime-callable C
source exporters contributed by scalar target/export code through the
`scalar-plugin` target exporter bundle, and the RVV+scalar
i32 vector-add host dispatch C composite exporter registered by RVV+scalar
target/export code. The
composite exporter consumes the selected RVV dispatch-case callable candidate
plus selected scalar dispatch-fallback callable candidate and validates the
target-owned dispatch availability guard ABI before source output. This does
not add generic RVV or scalar lowering, full runtime ABI integration, object
generation, linking, arbitrary source export, correctness evidence, or
performance evidence.

The artifact-kind-aware generic route may also select target-owned bounded
RISC-V ELF relocatable object exporters for the same validated direct RVV
i32 binary microkernel path or RVV+scalar i32 binary dispatch path. These
library-object routes reuse the selected callable or dispatch source
validation, emit the default library-style source internally, and then use
structured RVV architecture capability metadata, selected RVV compile
capability metadata, and local `clang` to produce a RISC-V ELF relocatable
object with no hidden `main` or self-check harness. A distinct
runtime-callable C header route may be selected through the generic
`--tcrv-export-target-header-artifact` front door and must use artifact kind
`runtime-callable-c-header`. Header selection is source-like only for its
external C caller surface: it must not make
`--tcrv-export-target-source-artifact` ambiguous with
`runtime-callable-c-source`, and it must not make
`--tcrv-export-target-artifact` choose the header instead of a matching
library-object route. When both source and non-source target exporters match
the same selected plan, `--tcrv-export-target-artifact` must prefer the
non-source runtime-callable object route while
`--tcrv-export-target-source-artifact` remains source-only. The object and
header routes are still bounded target artifacts; they do not link, run
hardware, perform automatic probing, prove correctness, or measure performance.
The self-check object route remains an explicit target-owned helper command for
evidence collection, not the generic artifact front door.
For the direct RVV i32/i64 add/sub/mul microkernel paths, the runtime-callable C
header route is matched from the same validated callable source candidate as the
object route. It derives its single prototype from the same selected path,
microkernel body, callable ABI plan, mem_window/runtime_param boundaries, and
capability metadata as the source/object routes. For the i64 source/header/
object routes, the selected typed `tcrv_rvv.i64_vadd_microkernel`,
`tcrv_rvv.i64_vsub_microkernel`, or `tcrv_rvv.i64_vmul_microkernel` body is the
authority for family, dtype, arithmetic op, intrinsic config, callable ABI,
route id, artifact kind, component group, function stem, and generated C body.
If `tcrv_rvv.lowering_descriptor` is present beside that typed body, target
export validates it only as non-authoritative legacy mirror metadata; a stale
mirror or descriptor-only i64 path fails before source/header/object/bundle
output. The header must not embed callable bodies, RVV intrinsics, `main`,
self-check helpers, runtime probing, evidence logs, credentials, artifact
paths, or performance text.

The artifact-kind-aware generic route may also select scalar fallback
runtime-callable C header and RISC-V ELF relocatable object helpers for the same
validated scalar i32 binary callable source candidate. These scalar helpers are
contributed by scalar target/export code through the `scalar-plugin`
plugin-owned target exporter bundle, not by core orchestration. When a
single scalar header/object helper route is shared by add/sub/mul, its runtime
ABI kind/name must be derived from the matched source candidate rather than a
vadd-only route default. The header
route is a declaration-only external C surface for the same IR-backed
`tcrv.exec.mem_window` plus `tcrv.exec.runtime_param` callable ABI plan. The
object route emits the validated scalar library-style C source internally and
compiles it with local `clang` using structured RISC-V target/toolchain
capability metadata: an available `rv64` capability provider with `riscv64`
architecture metadata and selected march metadata from
`riscv.toolchain.march`, `rvv.probe.compile_run`, or `rvv.toolchain.march`.
Optional MABI metadata may be supplied by the matching `riscv.toolchain.mabi`,
`rvv.probe.compile_run`, or `rvv.toolchain.mabi` facts. If those facts or local
`clang` are absent, object export fails with a bounded diagnostic instead of
silently emitting a host object or a source-only substitute. The scalar helpers
do not add generic scalar lowering, linked runtime glue, automatic dispatch,
runtime execution, RVV evidence, correctness coverage, or performance evidence.

When a selected dispatch contains a primary supported non-fallback route plus a
supported `dispatch fallback` route, generic single-artifact export must choose
the primary non-fallback route and ignore the fallback candidate for ambiguity
purposes. This keeps `--tcrv-export-target-source-artifact` and
`--tcrv-export-target-artifact` deterministic for RVV/offload primary paths
while preserving the scalar fallback candidate for the specialized host
dispatch exporter. A scalar-only selected fallback remains exportable through
the generic route when it is the only supported candidate. If multiple
non-fallback candidates remain supported for one generic export request, the
request is still ambiguous and must fail closed.

### Parameter Claim Boundary

Lowering boundaries, emission plans, manifests, and target artifacts must
preserve parameter layering:

- hardware facts / target capability such as VLEN, vlenb-derived capacity,
  hart count, toolchain availability, selected march/mabi capability
  properties, and remote probe provenance constrain route legality and
  selection;
- compile-time variant config such as SEW, LMUL, tail/mask policy, unroll, and
  selected lowering strategy may be serialized as compiler decision metadata
  only when it was actually selected or proposed by the plugin and checked
  against capabilities. For RVV i32 binary paths, selected vector-shape
  metadata may include shape id, SEW, LMUL, tail/mask policy, vector type,
  intrinsic suffix, and setvl suffix, but those fields remain target/plugin
  compile-time config rather than runtime ABI values or `tcrv.exec` compute
  semantics. The RVV arithmetic family registry owns only the suffix-free
  arithmetic intrinsic prefix; target source emission forms the full RVV
  arithmetic intrinsic name by appending the selected vector-shape suffix after
  selected-shape validation;
- runtime SSA values / runtime control values such as AVL, vl, pointer
  arguments, length `n`, `rvv_available`, and dispatch guards may be emitted
  only as real IR/control fields or generated ABI parameters;
- generated ABI parameters must state whether they are actually IR-modeled or
  target/export ABI-owned. The current bounded i32 binary RVV and scalar source
  exports pass `lhs`, `rhs`, `out`, and runtime `n` as C ABI parameters, but the
  callable parameter plan must be built from real `tcrv.exec.mem_window` IR for
  lhs/rhs/out buffer meanings and real direct `tcrv.exec.runtime_param` IR for
  runtime-element-count. Candidate/emission-plan parameter metadata may only
  mirror that IR-backed plan, while runtime ABI strings and glue roles come
  from the selected finite binary runtime ABI contract. Runtime `n` remains a
  runtime ABI/control value, not descriptor-local element count;
- emission-plan-backed RVV+scalar dispatch export must resolve callable
  parameters from the same IR-backed callable ABI plan for both the selected RVV
  candidate and the selected scalar fallback candidate. The bounded dispatch
  exporter must reject callable candidate metadata that disagrees with the
  `mem_window` / runtime-element-count `runtime_param` boundaries, then append
  exactly one target/export-owned `dispatch-availability-guard` parameter
  resolved through a selected dispatch case that carries typed
  `runtime_guard_required = true` and has a `runtime_guard` symbol reference to
  direct `tcrv.exec.runtime_param` IR. Detached role lookup, printable
  condition/guard/policy strings, or stale dispatch metadata must not become
  the executable branch-control source. The
  default guard C name is `rvv_available`; an explicit runtime_param may use
  another valid C name without changing callable role order, adding the guard to
  callable microkernel signatures, or introducing automatic hardware probing;
- descriptor-local bounded values such as `tcrv_rvv.element_count` or
  `tcrv_scalar.element_count` describe a finite descriptor or fixture slice only.
  An extension plugin may choose such a sample size from validated structured
  capability facts, such as RVV i32 M1 lane capacity, but the value still must
  not be reported as tensor shape, global problem size, AVL, vl, runtime loop
  trip count, correctness coverage, or performance evidence.

Generated C may contain target-owned local variables such as a local `vl`
computed by RVV intrinsics or ABI parameters such as `n` and `rvv_available`.
That does not imply those values were modeled in MLIR unless the input IR has
the corresponding attribute, type, SSA value, region argument, or ABI/control
surface. The current bounded `tcrv_rvv.setvl` and `tcrv_rvv.with_vl` surfaces
model only runtime AVL/VL control-plane IR when they appear in the input. They
are not emitted runtime ABI evidence by themselves.

## Scenario: Dynamic Vector Source Tail Authority

### 1. Scope / Trigger

This scenario applies to the bounded dynamic vector/SCF i32-vadd source front
door that lowers through `--tcrv-lower-source-rvv-binary-to-exec` or its
explicit vector adapter alias. It is required when a source `%n` runtime extent
can be smaller than, equal to, or not a multiple of the selected finite
`vector<16xi32>` chunk.

### 2. Signatures

- Source wrapper shape: three `memref<?xi32>` buffers plus one `%n: index`,
  one `scf.for` from zero to `%n` in step `16`, zero i32 read padding, two
  `vector.transfer_read` operations at the induction variable, one
  `arith.addi` over `vector<16xi32>`, one `vector.transfer_write` at the same
  induction variable, and `func.return`.
- Source transfer-tail metadata on both `tcrv.exec.kernel` and the direct
  runtime-element-count `tcrv.exec.runtime_param`:
  `tcrv_frontend_source_kind =
  "mlir-vector-scf-runtime-i32-vadd.v1"`,
  `tcrv_frontend_source_authority =
  "source-scf-for-runtime-upper-bound"`,
  `tcrv_frontend_runtime_extent_arg = "n"`,
  `tcrv_frontend_source_loop_step = 16 : i64`,
  `tcrv_frontend_source_vector_chunk_extent = 16 : i64`,
  `tcrv_frontend_active_lane_authority =
  "mlir-vector-transfer-tail-active-lanes"`,
  `tcrv_frontend_source_tail_policy =
  "runtime-n-bounded-transfer-tail-padding-and-store"`, and
  `tcrv_frontend_runtime_element_count_constraint =
  "source-runtime-extent"`.
- Selected-plan metadata names:
  `tcrv_frontend.active_lane_authority` and
  `tcrv_frontend.source_tail_policy`.

### 3. Contracts

- Dynamic source transfer ops must not assert `in_bounds = [true]`. The source
  active lanes for tail iterations are defined by MLIR transfer tail behavior:
  reads use the i32 padding value for inactive lanes and writes do not store
  inactive lanes beyond `%n`.
- `%n` is the source `scf.for` upper bound, the runtime element-count ABI
  parameter, and the AVL source consumed by `tcrv_rvv.setvl`.
- Selected RVV `tcrv_rvv.selected_tail_policy` and
  `tcrv_rvv.selected_mask_policy` remain compile-time vector-shape policy.
  They are not the MLIR source active-lane authority.
- `tcrv_rvv.descriptor_element_count` remains descriptor-local chunk capacity
  metadata. It is not the dynamic source extent, runtime trip count, or
  correctness evidence.
- Direct source/header/object export and plan-and-export bundle export must
  consume the same neutral source-frontdoor route and the same IR-backed
  runtime ABI contract.

### 4. Validation & Error Matrix

- Dynamic transfer read/write carries `in_bounds = [true]` -> reject during
  source lowering before `tcrv.exec` is materialized.
- Kernel and runtime-element-count `runtime_param` do not both carry the full
  dynamic source-tail metadata -> fail before artifact output.
- Any dynamic source-tail metadata field is stale between kernel and
  runtime_param -> fail before artifact output.
- Selected-plan metadata for active-lane authority or source-tail policy is
  missing, duplicated, or has the wrong value/role/note -> fail before source,
  header, object, or bundle output.
- Fixed-vector source extent metadata and dynamic source-tail metadata appear
  together -> fail as mutually exclusive source authority.

### 5. Good/Base/Bad Cases

- Good: dynamic vector source omits `in_bounds` on transfer read/write, records
  the source-tail authority fields on kernel/runtime_param, and emits selected
  metadata that names both source active-lane authority and selected RVV policy.
- Base: fixed vector `vector<16xi32>` keeps its `n == 16` source-extent
  constraint and may use fixed in-bounds transfer assertions.
- Bad: dynamic source relies on `setvl(n)` alone while the MLIR source transfer
  ops still assert all 16 lanes are in bounds.

### 6. Tests Required

- Positive lit for dynamic `VectorToExec` showing no source transfer ops
  remain, dynamic source-tail metadata is present on kernel/runtime_param, and
  selected-plan metadata carries active-lane and source-tail fields.
- Negative lit for dynamic source `in_bounds = [true]`, missing kernel/param
  source-tail metadata, and stale selected-plan active-lane metadata.
- Direct artifact source/header/object checks exposing source-tail authority,
  selected RVV tail/mask policy, descriptor-local element count, runtime `n`,
  and `tcrv_rvv.setvl`.
- Bundle export checks proving the same neutral front door and selected-plan
  metadata reach the RVV+scalar dispatch source/header/object records.
- Fresh `ssh rvv` evidence for runtime counts 7, 16, and 23 when the generated
  artifact contract changes.

### 7. Wrong vs Correct

Wrong:

```text
dynamic %n loop + transfer_read/write in_bounds=true
  -> rely on downstream setvl(n) to imply source tail correctness
```

Correct:

```text
dynamic %n loop + MLIR transfer tail semantics
  -> tcrv.exec runtime_param @abi_runtime_element_count
  -> selected-plan active-lane/source-tail metadata
  -> tcrv_rvv.setvl(n) and generated artifact comments
```

### Execution-Plan / Export Preflight Coherence

Before a module is handed to a generic target artifact export route, the public
`tcrv-translate --tcrv-export-target-source-artifact` and
`tcrv-translate --tcrv-export-target-artifact` front doors must run the
target-neutral preflight verifier to check that all compiler-visible handoff
metadata still describes the same selected execution path. This check is a
metadata coherence gate only. It must not export artifacts, lower to
LLVM/RISC-V, emit extension instructions, create runtime ABI glue, run hardware,
or claim correctness or performance.

The verifier must fail closed when selected-path, dispatch/fallback,
lowering-boundary, runtime ABI ownership, emission-plan, and artifact route
metadata are stale or contradictory. Required failure cases include selected
variant references that no longer resolve to the current direct variant, origin
plugins that are missing or unregistered, dispatch or selected-marker origin
mismatch, lowering-boundary source kernel / selected variant / origin mismatch,
emission-plan target / role / origin / lowering-boundary mismatch, missing
runtime ABI ownership fields for plans that require them, unsupported artifact
route kind or emission kind, unknown target artifact route id, and multiple
ambiguous supported artifact candidates.

The generic preflight verifier may use the existing `ExtensionPluginRegistry`
to validate origin ownership and the generic target artifact exporter registry
to validate route metadata. Target-specific proof of a concrete microkernel,
descriptor body, toolchain, or runtime remains target-owned and must not move
into the shared transform.
Registry-owned route metadata is not limited to standalone routes. Composite
target artifact exporters may also carry `TargetArtifactRouteMetadata` for
route-level claim fields and bounded metadata-shape validation. Generic
registry code may validate the shape of those fields, copy route claim fields
into bundle records, and let extension bundles require metadata for composite
route ids, but it must not interpret RVV/scalar/offload family semantics from
that metadata.
If a route metadata requirement depends on an additional enabled plugin, such
as RVV-owned RVV+scalar dispatch routes that also require `scalar-plugin`, the
requirement must declare that plugin dependency and is enforced only when the
dependency is enabled.
When a registered target artifact route declares required runtime ABI roles or
a route-local ABI validation callback, this same preflight verifier must reject
missing or inconsistent compiler-owned ABI contracts before descriptor, source,
header, object, or bundle materialization. For the offload descriptor route,
that means malformed or stale `runtime_abi_parameters` fail at the selected
target-artifact/front-door boundary before descriptor text is emitted, while
the descriptor exporter keeps its own final validation safety net. RVV and
scalar callable routes use the same boundary for their compiler-owned
lhs/rhs/out/runtime-element-count role contracts: direct callable source routes
declare the typed callable role requirements, RVV header/object composite
helpers preflight the selected RVV callable source candidate through that same
route contract, and RVV+scalar dispatch composite helpers preflight both the
RVV dispatch-case callable candidate and scalar dispatch-fallback callable
candidate before deriving the dispatcher ABI.

The canonical `tcrv-opt --tcrv-execution-planning-pipeline` must run this same
preflight verifier as its final gate after emission-plan materialization when a
target artifact exporter registry is available at the tool boundary. The
pipeline builder must inject the active plugin registry and target artifact
exporter registry into the existing coherence pass instead of implementing a
second coherence check. Builders that only receive plugins may use an explicit
empty target artifact exporter registry, which preserves fail-closed diagnostics
for supported artifact front doors until the caller supplies registered target
exporters.

### Built-In Target Artifact Exporter Registration Boundary

#### 1. Scope / Trigger

Trigger: a public tool needs the complete built-in target artifact route set
for generic artifact export after selected-path, lowering-boundary, and
emission-plan metadata have already been materialized.

This boundary is a Target-layer registration composition helper only. It does
not inspect MLIR modules, selected plans, capabilities, lowering boundaries, or
runtime ABI metadata.

#### 2. Signatures

Header:

```cpp
#include "TianChenRV/Target/BuiltinTargetArtifactExporters.h"
```

C++ API:

```cpp
llvm::Error registerBuiltinTargetArtifactExporters(
    TargetArtifactExporterRegistry &registry);
```

#### 3. Contracts

- The caller owns the `TargetArtifactExporterRegistry`.
- The helper registers every currently supported built-in target artifact route
  by delegating to target-owned registration functions.
- The current non-plugin single-candidate route set is:
  - RVV standalone smoke-probe C source, artifact kind
    `standalone-c-source`, selected only when a plugin-owned smoke-probe
    emission plan names `tcrv-export-rvv-smoke-probe-c`.
- The current plugin-owned single-candidate route set includes:
  - RVV selected binary microkernel runtime-callable C source routes for the
    finite add/sub/mul i32/i64 families, registered by the `rvv-plugin`
    target-exporter bundle and emitted by RVV target/export code.
  - Offload runtime handoff descriptor route, registered by the
    `offload-plugin` target-exporter bundle and emitted by offload
    target/export code.
  - Scalar selected fallback microkernel runtime-callable C source routes for
    the finite add/sub/mul i32/i64 families, registered by the `scalar-plugin`
    target-exporter bundle and emitted by scalar target/export code.
  - Toy metadata diagnostic artifact route, registered by the `toy-plugin`
    target-exporter bundle and emitted by Toy target/export code.
- The current plugin-owned single-candidate composite route set includes:
  - RVV selected binary microkernel runtime-callable C header routes, matched
    from the same selected callable source candidates and emitted by RVV
    target/export code as external caller ABI surfaces with artifact kind
    `runtime-callable-c-header`. Registration must include route-local
    candidate preflight that validates the matched RVV callable source
    candidate against the RVV direct callable runtime ABI role contract before
    header output.
  - RVV selected binary microkernel runtime-callable RISC-V ELF relocatable
    library object routes, matched from the same selected callable source
    candidates and emitted by RVV target/export code without a hidden
    self-check harness or `main`. Registration must include the same
    route-local candidate preflight before object compilation.
- The current plugin-owned single-candidate composite route set also includes:
  - Scalar selected fallback microkernel runtime-callable C header routes for
    the finite add/sub/mul i32/i64 families, matched from the same selected
    scalar callable source candidate and emitted by scalar target/export code.
    The legacy i32-vadd header route remains
    `tcrv-export-scalar-microkernel-header`; non-add routes use
    `tcrv-export-scalar-<family>-microkernel-header`.
  - Scalar selected fallback microkernel runtime-callable RISC-V ELF
    relocatable object routes for the same finite families, matched from the
    same selected scalar callable source candidate and emitted by scalar
    target/export code. The legacy i32-vadd object route remains
    `tcrv-export-scalar-microkernel-object`; non-add routes use
    `tcrv-export-scalar-<family>-microkernel-object`.
  - Scalar header/object matching must be family-specific: a selected
    `i32-vadd` scalar source candidate must not satisfy the `i32-vmul` or i64
    scalar header/object helper routes, and stale ABI family metadata must fail
    before output.
- The current plugin-owned multi-candidate composite route set includes:
  - RVV+scalar explicit finite binary host dispatch runtime-callable C source,
    matched by target-owned RVV+scalar dispatch exporter code from the selected
    RVV dispatch-case callable route and scalar dispatch-fallback callable
    route. These RVV-primary routes are registered by the `rvv-plugin`
    target-exporter bundle and require an enabled `scalar-plugin`; disabled or
    missing RVV/scalar plugins must not publish the dispatch route group.
    Registration must preflight both callable component candidates against
    their direct route ABI role contracts before deriving the dispatcher ABI.
  - RVV+scalar explicit finite binary host dispatch runtime-callable RISC-V
    ELF relocatable library object, matched from the same selected callable
    candidates and emitted by the target-owned RVV+scalar dispatch exporter
    code without a hidden self-check harness or `main`. Its registration must
    include the same callable component candidate preflight.
  - RVV+scalar explicit finite binary host dispatch runtime-callable C header,
    matched from the same selected callable candidates and emitted by the
    target-owned RVV+scalar dispatch exporter code as an external caller ABI
    surface with artifact kind `runtime-callable-c-header`. Its registration
    must include the same callable component candidate preflight.
- The helper may include RVV/scalar/offload target headers and call their
  target-owned registration functions, but it must not duplicate route
  semantics or artifact validation.
- Generic public translate helpers should call this helper once and then call
  `exportTargetSourceArtifact`, `exportTargetArtifact`, or
  `exportTargetHeaderArtifact`.
- Source-only filtering remains the responsibility of the generic exporter via
  artifact-kind validation, not by omitting non-source built-ins from the
  registration helper.
- Extension/plugin-owned artifact routes may be registered through a
  target-layer plugin-exporter bundle registry keyed by extension plugin name.
  Public tools that already own an `ExtensionPluginRegistry` must pass that same
  active registry into built-in target exporter registration so enabled plugins
  can contribute their plugin-owned target artifact exporters through the
  generic boundary. A plugin-owned exporter bundle may additionally declare
  required enabled extension plugins for composite routes whose selected-plan
  contract spans more than one plugin-owned component. This applies to real
  target-owned RVV selected binary microkernel source/header/object exporters,
  offload runtime handoff descriptor exporters,
  scalar fallback source/header/object exporters, RVV+scalar dispatch
  source/header/object composite exporters, and metadata-only Toy exporters. A
  single extension plugin may contribute more than one bundle when route groups
  have different dependency requirements, such as scalar standalone routes and
  RVV-dependent RVV+scalar dispatch routes. Disabled or missing plugins must
  not silently publish their plugin-owned target routes. Later selected-plan
  export then fails closed as an unknown or unavailable route/origin instead of
  falling back to a central extension-specific exporter branch.
  For the finite RVV binary route set, the central built-in extension bundle
  composition must delegate RVV direct microkernel route metadata and
  RVV+scalar dispatch route metadata to RVV target-support bundle code. It must
  not iterate RVV direct or RVV+scalar dispatch manifests as central built-in
  route truth.

#### 4. Validation & Error Matrix

- Empty route id, empty artifact kind, or null callback from any target-owned
  exporter -> generic `TargetArtifactExporterRegistry` registration failure.
- Duplicate route id, including calling the built-in helper twice on the same
  registry -> generic duplicate route failure.
- Duplicate plugin-exporter bundle keys, null plugin-exporter callbacks, and
  duplicate route ids produced while populating enabled plugin-owned exporters
  -> generic registration failure before target artifact output.
- Missing built-in route registration in a tool -> route lookup fails closed as
  an unknown target artifact route or no supported artifact route.
- Offload descriptor selected through source-only command -> source artifact
  filtering must fail closed without descriptor output.
- Route spoofing across RVV/scalar/offload origins or artifact kinds -> generic
  exporter metadata validation must fail before target-owned output.

#### 5. Good/Base/Bad Cases

- Good: `tcrv-translate --tcrv-export-target-artifact` creates a registry,
  calls `registerBuiltinTargetArtifactExporters`, and exports a legal offload
  descriptor through the offload target-owned exporter.
- Good: `tcrv-translate --tcrv-export-target-artifact` selects the non-source
  RVV+scalar dispatch runtime-callable library object composite route when the
  selected plan has both supported callable sides and local RVV object
  compilation support.
- Good: `tcrv-translate --tcrv-export-target-header-artifact` selects the
  direct RVV selected binary microkernel runtime-callable C header route from
  the same selected callable source candidate after enabled `rvv-plugin`
  exporter-bundle registration, without changing source-only or default object
  artifact selection.
- Good: `tcrv-translate --tcrv-export-target-header-artifact` selects the
  RVV+scalar dispatch runtime-callable C header composite route from the same
  selected callable sides without changing the source or object front-door
  selection result.
- Base: `tcrv-translate --tcrv-export-target-source-artifact` uses the same
  built-in registry but filters to a legal RVV runtime-callable C source,
  scalar runtime-callable C source route, or target-owned RVV+scalar dispatch
  composite source route when the selected plan contains both callable sides.
- Bad: each generic translate helper manually repeats
  `registerRVVMicrokernelTargetExporters`,
  `registerScalarMicrokernelTargetExporters`, and
  `registerOffloadRuntimeDescriptorTargetExporters`; adding a target route then
  requires broad hand-editing in generic tool code.
- Bad: central built-in target exporter composition registers the selected RVV
  binary microkernel routes directly even when `rvv-plugin` is missing or
  disabled.

#### 6. Tests Required

- C++ registry tests must prove the built-in helper registers all current
  route ids with deterministic generic metadata and rejects duplicate
  registration.
- lit/FileCheck route tests must continue to cover RVV source export, scalar
  source export, direct RVV microkernel header export through the generic
  header front door, RVV+scalar composite dispatch source export, RVV+scalar
  composite dispatch object export when local RVV object compilation is
  available, offload descriptor export, source-only offload rejection, and
  RVV/scalar/offload route spoofing failures.
- CMake checks must include the built-in Target support library in the tool and
  C++ test link graph.

#### 7. Wrong vs Correct

Wrong:

```cpp
// In each generic translate helper:
registerRVVMicrokernelTargetExporters(registry);
registerScalarMicrokernelTargetExporters(registry);
registerOffloadRuntimeDescriptorTargetExporters(registry);
```

Correct:

```cpp
registerBuiltinTargetArtifactExporters(registry);
```

The correct shape keeps target-specific route registration in target-owned
modules plus one Target-layer built-in bundle, while shared generic artifact
routing remains target-neutral and fail-closed.

### Built-In Target Translate Route Registration Boundary

#### 1. Scope / Trigger

Trigger: a public translate tool needs direct route-family helper commands for
target-owned artifact routes, such as bounded RVV direct binary microkernel
source/header/object helpers or RVV+scalar dispatch source/header/object/
self-check helpers.

This boundary is a Target-layer translation route contribution helper only. It
does not inspect MLIR modules, select routes, validate emission plans, generate
artifacts, or run hardware.

#### 2. Signatures

Header:

```cpp
#include "TianChenRV/Target/BuiltinTargetTranslateRoutes.h"
#include "TianChenRV/Target/TargetTranslateRegistration.h"
```

C++ API:

```cpp
using TargetTranslateExportFn =
    std::function<llvm::Error(mlir::ModuleOp, llvm::raw_ostream &)>;

class TargetTranslateRoute {
public:
  llvm::StringRef getRouteID() const;
  llvm::StringRef getDescription() const;
  const TargetTranslateExportFn &getExportFn() const;
  bool requiresBinaryStdout() const;
};

class TargetTranslateRouteRegistry {
public:
  llvm::Error registerRoute(const TargetTranslateRoute &route);
  const TargetTranslateRoute *lookup(llvm::StringRef routeID) const;
  llvm::ArrayRef<TargetTranslateRoute> getRoutes() const;
};

llvm::Error
registerBuiltinTargetTranslateRoutes(TargetTranslateRouteRegistry &registry);
```

#### 3. Contracts

- The caller owns the `TargetTranslateRouteRegistry`.
- A target translate route contribution records only the direct helper command
  route id, help description, target-owned export callback, and whether stdout
  must be switched to binary mode before callback execution.
- The public translate tool owns the MLIR `TranslateFromMLIRRegistration`
  object lifetime and attaches its dialect-registration hook generically while
  iterating `TargetTranslateRouteRegistry::getRoutes()`.
- Built-in target translate route registration delegates to target-owned
  registration functions. Current route-family contributors are RVV direct
  binary microkernel routes and RVV+scalar dispatch routes.
- The RVV route-family contributor is the RVV target-support bundle. It may
  aggregate RVV direct binary helper routes and RVV+scalar dispatch helper
  routes, while public `tcrv-translate` registration still iterates only the
  generic `TargetTranslateRouteRegistry`.
- `tcrv-translate` must not manually loop over RVV direct or RVV+scalar
  dispatch manifests. It should call `registerBuiltinTargetTranslateRoutes`
  once and then register each contributed route generically.
- Legacy standalone helpers that are not route-family artifact helpers, such as
  the RVV smoke probe helper or the current standalone RVV microkernel
  self-check C helper, may remain direct tool registrations until their owner
  boundary is explicitly changed.
- Target translate route registration does not change `TargetArtifactExport`
  semantics, exporter matching, route-local ABI validation, generated artifact
  contents, object compilation, bundle export, or evidence claims.

#### 4. Validation & Error Matrix

- Empty route id -> generic target translate route registry failure.
- Empty description -> generic target translate route registry failure.
- Null export callback -> generic target translate route registry failure.
- Duplicate route id, including collisions between target contributors or
  calling the built-in helper twice on one registry -> generic duplicate route
  failure.
- Missing built-in target translate route registration in a tool -> direct
  helper route is absent from that tool's command surface; generic target
  artifact front doors remain governed by `TargetArtifactExporterRegistry`.

#### 5. Wrong vs Correct

Wrong:

```cpp
// In tcrv-translate:
for (const RVVMicrokernelDirectRouteManifestEntry &route :
     getRVVMicrokernelDirectRouteManifest())
  registerTranslateRoute(route);
for (const RVVScalarDispatchRouteManifestEntry &route :
     getRVVScalarDispatchRouteManifest())
  registerTranslateRoute(route);
```

Correct:

```cpp
TargetTranslateRouteRegistry routes;
registerBuiltinTargetTranslateRoutes(routes);
for (const TargetTranslateRoute &route : routes.getRoutes())
  registerTranslateRoute(route, registerTianChenRVTranslateDialects);
```

The correct shape keeps route-family facts and direct helper callbacks in
target-owned support modules, while the public tool supplies only the generic
MLIR translate registration and dialect hook.

The artifact-kind aware generic route may also dispatch supported non-source
artifacts through target-owned exporters. The first such route is the offload
runtime handoff descriptor, registered by offload target/export code and
selected by the offload plugin's supported descriptor emission plan. Shared
generic routing still validates only route id, artifact kind, origin, emission
kind, selected path, lowering-boundary reference, runtime ABI metadata, and
required capability refs; offload-specific descriptor content stays in the
offload target exporter.

The target artifact bundle export is a directory materialization layer over the
same registry-derived artifact records that the emission manifest serializes.
It may iterate `collectTargetArtifactBundleRecords`, call the registered
source/header/object/descriptor exporter callbacks, and write a deterministic
bundle index plus the selected artifact files under an explicit existing output
directory. The generic bundle layer must not branch on RVV, scalar, IME,
Sophgo, offload, vendor, target family, dtype, shape, runtime, toolchain, or
microarchitecture semantics to decide which files exist; route availability
comes from selected-path/emission-plan metadata and the target artifact
exporter registry. Before writing any artifact file or complete index, bundle
export must validate the selected target-artifact front door for each kernel:
registered composite routes may consume their complete component set, otherwise
there must be exactly one selected standalone front door after excluding a
selected dispatch fallback behind a supported non-fallback path. Multiple
non-fallback standalone candidates without a registered composite route,
unknown routes, or route/exporter metadata mismatches are selected-plan
coherence failures, not late file-emission choices. Bundle file names and the
index must be deterministic, safe, and metadata-derived, and must not contain
local absolute paths, credentials, timestamps, random values, `artifacts/tmp`
paths, raw logs, or secret-like text. Unsupported or metadata-only selected
paths must not produce a fake complete bundle. If artifact materialization
fails, the exporter must fail before writing a complete index and must either
remove partial outputs or otherwise avoid claiming a complete bundle. The
bundle index is build handoff metadata only: it may record file names, artifact
kind, route, owner, runtime ABI kind/name, component selected paths, and
conservative evidence roles, but it does not claim link success, runtime
success, RVV execution, correctness, or performance.
When a matched route, including a composite route, has registry-owned route
claim fields, the bundle index must preserve those fields on the corresponding
artifact record. This allows downstream evidence runners to consume the
compiler-emitted no-claim boundary instead of inferring it from file names or
route strings.
When a selected dispatch has a supported primary non-fallback route plus a
supported dispatch fallback route and no target-owned composite bundle route
matches, the bundle layer follows the single-artifact front-door rule and emits
only the selected non-fallback artifact record. A selected runtime-offload
descriptor path therefore produces a one-artifact descriptor bundle, while the
scalar fallback remains selected-path metadata rather than a second descriptor
bundle artifact.
Dispatch-capable external ABI bundles must additionally expose a typed
component contract in the compiler-emitted bundle index. For the bounded
RVV+scalar i32-vadd dispatch bundle, the generated source, generated header,
and generated relocatable object records must share one stable
`component_group`, carry distinct `component_role` values (`source`, `header`,
and `object`), record the same `external_abi_name`, and preserve matching
runtime ABI kind/name plus selected dispatch component variants/roles. The
generic bundle layer must validate grouped records before writing a complete
index: duplicate component roles, missing source/header/object records, missing
external ABI identity, mismatched runtime ABI metadata, or mismatched selected
component paths are coherence failures. Python evidence runners may consume
these compiler-emitted fields, but must not define or infer the bundle contract
from file names.

`tcrv-translate --tcrv-export-target-artifact-bundle` remains the
coherence-gated exporter for already planned MLIR. The separate
`tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` entry may first
run the bounded marked-linalg RVV binary frontend lowering slice, then run the
existing execution planning pipeline with built-in plugin and target artifact
exporter registries, and finally call the same bundle exporter. The frontend
step is limited to creating the already specified `tcrv.exec.kernel` plus
`mem_window` / `runtime_param` ABI boundary from explicitly marked test or
hand-written finite RVV binary linalg input and preserving the bounded frontend
family marker that lets plugins choose the existing add, subtract, or multiply
microkernel descriptor; it must not become generic linalg lowering or bypass
plugin-owned realization.
It must fail before printing bundle completion if frontend lowering, planning,
execution-plan coherence, route validation, or artifact materialization fails,
and it must not weaken the bundle component contract or runtime ABI signature
validation.

## RVV+Scalar I32/I64 Dispatch Route Manifest

### 1. Scope / Trigger

Trigger: target/export code or public tools add, expose, or validate a
RVV+scalar i32/i64 binary dispatch route. The route source of truth is the
bounded target-owned dispatch route manifest for the existing finite families
`i32-vadd`, `i32-vsub`, `i32-vmul`, `i64-vadd`, `i64-vsub`, and `i64-vmul`.

This manifest is a target/export contract over already selected RVV dispatch
case and scalar dispatch fallback callable artifacts. It must not become a
generic arithmetic registry, a new high-level compute IR, or Python-owned
compiler semantics.

### 2. Signatures

Public manifest API:

```cpp
enum class RVVScalarDispatchRouteKind {
  Source,
  Header,
  Object,
  SelfCheckSource,
  SelfCheckObject,
};

struct RVVScalarDispatchRouteManifestEntry {
  const rvv_scalar::DispatchBinaryFamilyDescriptor *family;
  RVVScalarDispatchRouteKind routeKind;
  llvm::StringRef routeID;
  llvm::StringRef description;
  llvm::StringRef artifactKind;
  llvm::StringRef runtimeABIKind;
  llvm::StringRef runtimeABIName;
  llvm::StringRef componentGroup;
  llvm::StringRef externalABIName;
  llvm::StringRef selfCheckSuccessMarker;
  bool requiresBinaryStdout;
};

llvm::ArrayRef<RVVScalarDispatchRouteManifestEntry>
getRVVScalarDispatchRouteManifest();

llvm::Error exportRVVScalarDispatchRoute(
    mlir::ModuleOp module, const RVVScalarDispatchRouteManifestEntry &route,
    llvm::raw_ostream &os);
```

Public command surface:

```text
tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c
tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-header
tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-object
tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c
tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-object
```

The same five route kinds must exist for `i32-vsub`, `i32-vmul`, `i64-vadd`,
`i64-vsub`, and `i64-vmul` with the same stable naming pattern.

### 3. Contracts

- The manifest covers exactly the current registry families `i32-vadd`,
  `i32-vsub`, `i32-vmul`, `i64-vadd`, `i64-vsub`, and `i64-vmul`;
  unsupported families are absent rather than partially represented.
- Each manifest entry derives family facts from
  `RVVScalarBinaryFamily` and adds only dispatch-route-specific stable fields
  such as route kind, artifact kind, description, and binary stdout mode.
- Source, header, and library-object entries are registered as target artifact
  composite exporters by iterating the manifest. Self-check entries remain
  explicit direct evidence-helper translate routes and are not generic bundle
  front-door records.
- Direct `tcrv-translate` dispatch routes are registered by iterating the same
  manifest. Public tool code must not keep an independent hand-written add/sub
  /mul route list.
- A direct route must require the selected RVV callable family and selected
  scalar callable family to match the route family before emitting source,
  header, object bytes, or self-check harness code.
- Source/header/object records in one dispatch external ABI group must agree on
  runtime ABI kind/name, component group, external ABI name, selected component
  paths, and ordered runtime ABI parameters.
- Python evidence helpers may consume the compiler-emitted route and bundle
  metadata, but they must not decide route selection, runtime ABI shape,
  artifact kind, source generation, or family semantics.

### 4. Validation & Error Matrix

- Manifest has fewer or more than five routes per finite family -> C++ registry
  coverage fails.
- Duplicate manifest route ids -> C++ registry coverage fails.
- Missing source/header/object route for any finite family -> built-in target
  exporter registration fails coverage.
- Direct source/header/object/self-check route family does not match the
  selected dispatch pair -> fail before artifact output with a diagnostic that
  names the expected family and selected family.
- Selected RVV callable family differs from selected scalar fallback callable
  family -> fail before dispatch artifact output.
- Composite bundle records in the same component group disagree on runtime ABI,
  external ABI name, component paths, or ABI signature -> fail before complete
  bundle index output.
- Object or self-check-object routes must switch stdout to binary mode only for
  object output and must still report bounded diagnostics on failure.

### 5. Good/Base/Bad Cases

- Good: a selected `i32-vmul` dispatch pair exported through
  `--tcrv-export-rvv-scalar-i32-vmul-dispatch-c` emits vmul RVV intrinsic,
  scalar `lhs * rhs`, vmul dispatcher ABI, and vmul self-check marker.
- Base: `--tcrv-export-target-artifact-bundle` selects source/header/object
  records for the same finite dispatch family by registry metadata and writes
  one coherent external ABI group.
- Bad: a vmul direct route accepts selected vadd artifacts and emits add route
  ids, add ABI names, or `lhs + rhs`. It must fail before output.
- Bad: generic tool registration manually lists add/sub/mul direct routes while
  target/export registration uses a different route list. Both surfaces must
  iterate the same manifest-backed API.

### 6. Tests Required

- C++ registry tests must prove the manifest covers exactly add/sub/mul, five
  route kinds per family, distinct route ids, distinct component groups,
  distinct runtime ABI names, distinct self-check markers, and family-derived
  operator/intrinsic metadata.
- C++ target exporter tests must prove source/header/object composite routes
  for add/sub/mul are registered with runtime ABI callbacks, route-local
  candidate validation, direct helper metadata, component group, and external
  ABI name.
- lit/FileCheck tests must prove public `tcrv-translate --help` exposes all
  add/sub/mul direct routes through manifest registration.
- lit/FileCheck tests must prove direct source/header/object/self-check routes
  fail closed on stale-family mismatches and that generic routes still emit the
  family-selected source/header/bundle records.
- Python runner tests may cover self-test, local dry-run, and bundle dry-run
  consumption of emitted metadata. Passing those tests is tooling evidence only.
- Full `check-tianchenrv` must pass before the task is archived.

### 7. Wrong vs Correct

Wrong:

```cpp
static TranslateFromMLIRRegistration addRoute(...);
static TranslateFromMLIRRegistration subRoute(...);
static TranslateFromMLIRRegistration mulRoute(...);
```

Correct:

```cpp
for (const RVVScalarDispatchRouteManifestEntry &route :
     getRVVScalarDispatchRouteManifest()) {
  registerTranslateRoute(route);
}
```

The correct shape keeps finite dispatch family facts in the target-owned
registry/manifest boundary, keeps generic front doors route-id driven, and makes
stale-family mismatches fail before artifacts are emitted.

## Selected Lowering Boundary First Slice

Before executable lowering exists, the compiler may materialize selected-path
lowering-boundary metadata through the generic extension plugin registry. RVV is
the first plugin that creates a plugin-local `tcrv_rvv.lowering_boundary`
operation for selected RVV direct variants or dispatch cases. Scalar fallback
creates a plugin-local `tcrv_scalar.lowering_boundary` operation for selected
portable fallback paths. These structures are attachment points for future
lowering work, not executable lowering products.

Rules:

- materialization consumes selected `tcrv.exec` dispatch or selected-marker
  structure and generic variant `origin` metadata;
- ownership and legality are routed through the `ExtensionPluginRegistry` and
  the selected variant's origin plugin before any plugin-local metadata is
  created;
- scalar or other fallback references remain selected `tcrv.exec` metadata and
  are routed to their origin plugin; scalar fallback materializes scalar
  plugin-local metadata and must not receive RVV ops;
- RVV boundary ops must carry `status = "unsupported"` and a non-empty
  unsupported reason;
- RVV and scalar fallback boundary ops must carry the generic selected-boundary
  contract fields needed by downstream emission planning: `source_kernel`,
  `selected_variant`, `origin`, `role`, `status`, and
  `required_capabilities`;
- scalar fallback boundary ops must carry `status = "metadata-only"` and
  selected variant, origin, role, and required capability reference metadata;
- for the bounded scalar i32/i64 add/sub/mul source slice, scalar
  plugin-local boundary materialization must identify typed selected-path
  authority before creating the boundary: either a matching explicit
  `tcrv_scalar.*_microkernel` direct child or a descriptorless typed default
  materialization selected from the kernel frontend marker/default family;
- a selected scalar fallback variant carrying only legacy
  `tcrv_scalar.lowering_descriptor` and/or `tcrv_scalar.element_count`
  metadata must fail closed before `tcrv_scalar.lowering_boundary` creation.
  Those fields are optional mirror metadata only after typed scalar
  microkernel authority exists, and stale mirrors must fail as mirror
  mismatches;
- selected lowering-boundary metadata must not claim intrinsics, LLVM/RISC-V
  lowering, runtime ABI glue, generated objects, hardware execution,
  correctness, or performance.

## EmissionProvider Responsibilities

Each plugin emission provider:

- registers lowering patterns;
- declares emission path;
- declares compiler flags, headers, libraries;
- generates runtime glue;
- reports unsupported path;
- emits diagnostics.

## RVV Emission

### Current EmitC RVV intrinsic route

Path:

```text
TCRV RVV family ops
  -> EmitC ops
  -> RVV intrinsic C/C++
  -> clang default, gcc compatible
```

Use this as the current main RVV route.

### Optional future routes

Possible future routes:

```text
MLIR vector dialect / LLVM dialect
LLVM scalable vector
LLVM RVV intrinsic IR
compiler builtin
inline asm
```

These are optional future routes. They are not the current default system
definition and should not be used to justify bypassing the EmitC route.

## IME Emission

Possible paths:

```text
TCRV IME family ops
  -> EmitC
  -> IME/vendor intrinsic C/C++
```

Rules:

- IME-specific emission stays inside IME extension family plugin.
- Core only knows that the variant has an emission path.
- Capability model records toolchain support.
- If emission is unavailable, verifier rejects the variant or preserves it as disabled diagnostic.

## Offload Emission

Offload emission is runtime glue generation, not instruction generation.

Path:

```text
TCRV Offload family ops
  -> EmitC
  -> runtime C ABI / vendor runtime call
  -> link vendor runtime library
  -> runtime dispatch / sync / buffer management
```

Generated or declared runtime pieces:

- runtime handle initialization;
- buffer binding/allocation;
- host-device copy;
- async call;
- wait/sync;
- error handling;
- resource release;
- fallback path.

## Runtime Dispatch Lowering

`tcrv.exec.dispatch` lowers to host-side decision logic.

Inputs:

```text
capability availability
runtime probe
shape/dtype guard
cost threshold
user policy
```

Output:

```text
call selected variant or fallback
```

Dispatch logic must diagnose why a variant was selected or skipped.

## Fallback Lowering

Fallback is required for system completeness. Sources may include:

```text
MLIR default lowering
scalar/scf lowering
LLVM auto-vectorization
portable C/C++ kernel
RVV conservative kernel
```

Fallback targets correctness and coverage, not the main performance story.

## Toolchain Boundary

Rules:

- plugin declares toolchain requirement;
- capability verifier checks the requirement;
- core pass does not call vendor-specific compiler path directly;
- build system selects flags/libraries through plugin metadata;
- unsupported path produces clear diagnostics.

## Deployment Profiles

Reference profiles:

```text
rvv-native
  target: ssh rvv
  emission: LLVM/RVV intrinsic/native compile
  runtime: OpenMP/pthread optional

k3-ime
  target: K3/IME environment
  emission: IME plugin selected path
  runtime: native or vendor support

riscv-sophgo-offload
  target: RISC-V host + Sophgo runtime
  emission: C ABI/runtime glue
  runtime: vendor accelerator runtime
```

## Reproducibility Output

Each emission path should record:

- selected variant;
- required capabilities;
- selected compiler flags;
- selected runtime libraries;
- lowering path;
- fallback status;
- unsupported reason if failed.

Before real lowering/runtime work exists, plugin emission plans may be
materialized as `tcrv.exec.diagnostic {reason = "emission_plan"}` metadata.
These diagnostics are reproducibility and compiler-decision records only. A
supported or metadata-only diagnostic records plugin-owned intent such as
emission kind, lowering pipeline id, runtime ABI id, runtime ABI kind/name,
runtime glue role, required capability refs, and artifact kind; it does not
mean that the compiler emitted LLVM/RISC-V/RVV IR, assembled an object, linked
a runtime, ran hardware, proved correctness, or measured performance.
Unsupported diagnostics should carry explicit plugin diagnostic text and are
valid evidence of a boundary, not of executable support. Unsupported
diagnostics may still carry plugin-owned runtime ABI kind/name and runtime glue
role metadata so the unsupported boundary is explicit; those fields are not
runtime ABI glue implementation or executable evidence. When a selected-path
lowering boundary was consumed, the diagnostic should also carry a generic
`lowering_boundary` metadata field naming the boundary operation used by the
plan. That field is a diagnostic link only and does not imply executable
lowering.
Plugins may additionally attach bounded `selected_plan_metadata` dictionaries
to emission-plan diagnostics when the selected plan needs durable
self-description across the lowering/export boundary. Each entry must have
non-empty name, value, role, and note fields. The generic layer validates and
serializes these entries but does not interpret plugin-specific names such as
RVV capacity facts. For RVV, any capacity metadata must already have been
validated by the RVV plugin against the selected variant and target
capabilities; generic emission/export code must not turn it into runtime ABI,
shape, VL/AVL, or performance evidence.

The current public `tcrv-opt` built-in registry includes the RVV first-slice
plugin. Therefore an `origin = "rvv-plugin"` selected path can materialize an
unsupported plugin-owned emission-plan diagnostic instead of a generic
unregistered-origin failure. This remains compiler metadata only and must not be
reported as RVV lowering, runtime, correctness, or performance evidence. The
tool-level `--tcrv-disable-builtin-plugins` option preserves an explicit
empty-registry surface for negative parser/diagnostic tests.

The `tcrv_rvv.lowering_boundary` first slice is one step more concrete than an
emission-plan diagnostic because it is an RVV extension-dialect op, but it is
still pre-executable and unsupported. It is valid evidence that selected-path
metadata reached a plugin-local RVV boundary. It is not evidence that the
compiler emitted LLVM/RISC-V/RVV IR, assembled an object, linked runtime glue,
ran hardware, proved correctness, or measured performance.

## RVV Probe Evidence Boundary

`scripts/rvv_remote_probe.py` may produce `ssh rvv` evidence under
`artifacts/tmp/rvv_probe/<run-id>/`. That artifact is a remote
hardware/toolchain evidence record, not an emission artifact. Even when its
minimal hand-written RVV intrinsic program compiles and runs, the result only
proves that the remote host can compile and execute that probe program.

Do not reinterpret an RVV probe artifact as:

- plugin-supported emission readiness;
- compiler lowering to LLVM, RISC-V, or RVV intrinsics;
- generated object, executable, runtime ABI glue, or dispatch lowering;
- TianChen-RV compiler correctness;
- RVV kernel performance.

Future supported RVV emission must add plugin-local lowering/runtime work and
then cite both compiler-generated artifact evidence and separate `ssh rvv`
hardware/toolchain evidence.

## RVV Smoke-Probe Target Export Boundary

### 1. Scope / Trigger

Trigger: post-planning MLIR contains a selected RVV path and a matching
plugin-owned `tcrv_rvv.lowering_boundary`, but RVV kernel lowering is still
unsupported. A target/export tool may emit a deterministic standalone C smoke
program to exercise the RVV host toolchain and `riscv_vector.h` path.

This export is hardware/toolchain smoke evidence only. It is not RVV lowering,
LLVM/RISC-V/RVV IR emission, runtime ABI glue, generated kernel object,
TianChen-RV kernel correctness, or performance evidence.

### 2. Signatures

Public command:

```text
tcrv-translate --tcrv-export-rvv-smoke-probe-c
```

C++ entry point:

```cpp
llvm::Error exportRVVSmokeProbeC(mlir::ModuleOp module,
                                 llvm::raw_ostream &os);
```

Expected pipeline use:

```bash
tcrv-opt input.mlir --tcrv-execution-planning-pipeline \
  | tcrv-translate --tcrv-export-rvv-smoke-probe-c > rvv_smoke_probe.c
```

### 3. Contracts

- Input must be real post-planning MLIR with one or more `tcrv.exec.kernel`
  operations.
- Each exported selected path must be a selected direct variant or dispatch case
  whose variant carries `origin = "rvv-plugin"`.
- The selected variant must require an available `rvv` capability through
  structured `requires = [@...]` metadata.
- The selected variant must carry bounded single-line
  `tcrv_rvv.required_march` metadata containing RVV vector evidence.
- The kernel capability set must preserve matching selected march metadata
  through available `rvv.probe.compile_run.selected_march` or
  `rvv.toolchain.march.value`.
- A matching direct child `tcrv_rvv.lowering_boundary` must identify the same
  `source_kernel`, `selected_variant`, `origin`, `role`, `status`, and
  `required_capabilities`.
- RVV smoke export must stay under RVV target/plugin-specific code. It must not
  add RVV branches to core orchestration, generic manifest export, or
  `tcrv.exec`.
- Output is deterministic standalone C using `riscv_vector.h` and tiny RVV
  intrinsics. It may include bounded comments naming selected kernel, variant,
  role, selected march, optional selected ABI, and required capability refs.
- Output must not serialize raw probe logs, credentials, absolute build paths,
  timestamps, benchmark sizes, latency/throughput numbers, manifest success, or
  runtime-success claims.

### 4. Validation & Error Matrix

- No `tcrv.exec.kernel` -> export fails before source output.
- Missing selected surface -> export fails before source output.
- Scalar-only, offload-only, or fallback-only selected paths -> export fails as
  not an RVV smoke-probe input.
- Selected RVV-like origin other than `rvv-plugin` -> export fails through a
  bounded unknown-origin diagnostic.
- Missing, malformed, unavailable, or non-symbol `requires` metadata -> export
  fails before source output.
- Selected variant does not require available capability id `rvv` -> export
  fails.
- Missing, non-RVV, secret-like, newline-containing, or unbounded
  `tcrv_rvv.required_march` -> export fails.
- Missing or mismatched preserved selected march capability metadata -> export
  fails.
- Missing, duplicate, stale, role-mismatched, status-mismatched, or
  required-capability-mismatched `tcrv_rvv.lowering_boundary` -> export fails.
- Any validation failure must leave stdout without partial C source.

### 5. Good/Base/Bad Cases

- Good: `tcrv-opt --tcrv-execution-planning-pipeline` selects an RVV dispatch
  case, materializes `tcrv_rvv.lowering_boundary`, preserves
  `tcrv_rvv.required_march = "rv64gcv"`, and the exporter emits stable C with
  `#include <riscv_vector.h>`.
- Base: multiple kernels with selected RVV paths are exported in deterministic
  symbol order with one bounded probe function per selected RVV path.
- Bad: an offload dispatch case plus scalar fallback is selected and the RVV
  smoke exporter silently emits a generic probe. It must fail instead because no
  selected `rvv-plugin` path exists.

### 6. Tests Required

- lit/FileCheck positive coverage for pipeline-to-export C source.
- Positive checks for deterministic multi-kernel ordering, bounded probe names,
  `riscv_vector.h`, selected kernel/variant comments, selected march metadata,
  and absence of manifest/runtime-success/raw-log/performance claims.
- Negative checks for scalar-only and offload-only selection.
- Negative checks for stale selected boundary, missing/malformed selected march,
  secret-like or newline metadata, unknown RVV-like origin, and unavailable RVV
  requirement.
- Full project check must still pass through `check-tianchenrv`.
- If an RVV runtime/toolchain claim is made, record separate `ssh rvv`
  compile/run evidence under `artifacts/tmp/...` and state that it proves only
  the generated smoke program compiled and ran.

### 7. Wrong vs Correct

Wrong:

```text
Generated RVV smoke probe compiled, therefore RVV kernel emission is supported.
```

Correct:

```text
Generated standalone RVV smoke probe C compiled and ran on ssh rvv; RVV kernel
emission remains unsupported/deferred until plugin-local kernel lowering and
runtime evidence are implemented.
```

## RVV I32 VAdd Microkernel Target Export Boundary

### 1. Scope / Trigger

Trigger: post-planning MLIR contains one selected `rvv-plugin` path, a matching
plugin-owned `tcrv_rvv.lowering_boundary`, preserved selected march metadata,
and exactly one `tcrv_rvv.i32_vadd_microkernel` op for that selected
kernel/variant. The microkernel op may be an explicit fixture attachment or an
RVV-plugin materialization from the finite selected-variant descriptor
`tcrv_rvv.lowering_descriptor = "i32-vadd-microkernel.v1"`. The microkernel
op must carry the structured RVV control/dataflow body for this bounded slice:
one runtime index body argument for target/export-owned `n`/AVL, one
`tcrv_rvv.setvl`, one matching `tcrv_rvv.with_vl`, and one nested finite
`tcrv_rvv.i32_load`, `tcrv_rvv.i32_load`, `tcrv_rvv.i32_add`,
`tcrv_rvv.i32_store` sequence. The load/store ops reference the
target/export-owned lhs input, rhs input, and output buffer ABI roles consumed
by this exporter. Runtime element count remains a direct
`tcrv.exec.runtime_param` ABI boundary, not an RVV dataflow operand. Concrete C
parameter names are resolved from structured runtime ABI parameter metadata by
role when a supported emission plan is present.

This export is the first bounded RVV executable microkernel slice. It emits a
deterministic library-style C source file whose primary behavior is a stable
runtime-callable C ABI function:

```c
void <generated_name>(const int32_t *lhs, const int32_t *rhs,
                      int32_t *out, size_t n);
```

The callable function computes finite family-selected i32 vector add, subtract,
or multiply using RVV intrinsics. The
default artifact has no embedded `main` or self-check harness; later runtime
glue can embed the source and call the ABI boundary directly. A separate
  explicit harness export may add bounded local arrays and `main` only for bounded
evidence collection. This is not generic high-level MLIR lowering, arbitrary
RVV kernel emission, full runtime integration, benchmarking, or performance
evidence.

### 2. Signatures

Public command:

```text
tcrv-translate --tcrv-export-rvv-microkernel-c
tcrv-translate --tcrv-export-rvv-microkernel-self-check-c
```

C++ entry point:

```cpp
llvm::Error exportRVVMicrokernelC(mlir::ModuleOp module,
                                  llvm::raw_ostream &os);
llvm::Error exportRVVMicrokernelSelfCheckC(mlir::ModuleOp module,
                                           llvm::raw_ostream &os);
```

### 3. Contracts

- Input must be real post-planning MLIR with one selected RVV path.
- The selected variant must be owned by `origin = "rvv-plugin"` and require an
  available capability whose id is `rvv`.
- The selected variant must carry bounded `tcrv_rvv.required_march`, and the
  enclosing kernel capability set must preserve matching selected march metadata
  through exact providers or explicit relation-provider target profile scope
  for `rvv.probe.compile_run.selected_march` or `rvv.toolchain.march.value`.
- If the selected variant asks the RVV plugin to materialize the microkernel,
  it must carry exactly the finite descriptor
  `tcrv_rvv.lowering_descriptor = "i32-vadd-microkernel.v1"` and a bounded
  descriptor-local integer `tcrv_rvv.element_count`. When structured RVV i32 M1
  lane capacity is present, that element count may be a bounded plugin-selected
  sample size derived from capacity; no generic tensor, dtype family, shape,
  AVL/VL, runtime trip count, correctness coverage, performance, or broad
  microarchitecture semantics are implied.
- A matching direct child `tcrv_rvv.lowering_boundary` must identify the same
  source kernel, selected variant, origin, role, status, and required
  capability refs.
- A matching direct child `tcrv_rvv.i32_vadd_microkernel` must identify the same
  selected path, required capability refs, required march, optional selected
  mabi, bounded element count, and structured `setvl` / `with_vl` /
  explicit load/add/store body. The body runtime index argument is
  runtime/control-plane/ABI state; `element_count` remains descriptor-local
  metadata.
- If a supported matching emission-plan diagnostic is present, the exporter
  must resolve exactly the lhs-input-buffer, rhs-input-buffer, output-buffer,
  and runtime-element-count runtime ABI parameters by role, validate their
  target/export ABI ownership and C type spelling, and use the resolved C names
  in the generated callable source.
- Output must be deterministic library-style C with `riscv_vector.h`, a stable
  runtime-callable i32 vadd C ABI function, RVV i32 load/add/store intrinsics
  inside that callable function, control-flow derived only after validating the
  structured control/dataflow body, and no default embedded `main` or
  self-check harness.
- The explicit self-check harness export must call the same callable ABI from a
  bounded helper over bounded local arrays and is evidence tooling, not the
  default target artifact contract. Descriptor-local `element_count` may bound
  harness capacity, but the harness must pass explicit runtime `n` values
  through the generated ABI and must not treat `element_count` as shape, AVL,
  VL, or the only runtime trip count.
- Output must not include timestamps, absolute paths, raw logs, credentials,
  benchmark sizes, latency/throughput numbers, or performance claims.

### 4. Validation & Error Matrix

- Missing selected RVV path, scalar-only path, offload-only path, or
  fallback-only path -> export fails before source output.
- Missing, duplicate, stale, role-mismatched, status-mismatched, or
  required-capability-mismatched `tcrv_rvv.lowering_boundary` -> export fails.
- Missing, duplicate, stale, selected-variant-mismatched,
  required-capability-mismatched, invalid-element-count, malformed-march,
  malformed structured control/dataflow body, setvl/with_vl policy mismatch,
  missing or mismatched finite RVV i32 load/add/store sequence, or secret-like
  `tcrv_rvv.i32_vadd_microkernel` -> export fails.
- Missing, unavailable, non-symbol, non-RVV, or unknown selected variant
  requirements -> export fails.
- Missing or mismatched preserved selected march capability metadata -> export
  fails.
- Any validation failure must leave stdout without partial C source and must not
  fall back to `--tcrv-export-rvv-smoke-probe-c`.

### 5. Evidence Interpretation

Real `ssh rvv` compile/run evidence for the generated self-check harness source
proves only that the explicit generated i32 vector-add callable ABI source
compiled and that its harness passed on that host with the selected compiler
flags. It does not prove generic TianChen-RV lowering correctness, supported
arbitrary RVV kernel emission, full runtime integration, or performance.

### 6. Emission Plan / Manifest Handoff

When the explicit microkernel op matches the selected RVV path, the RVV plugin
may return a supported emission plan for the runtime-callable C source export
route.
If the selected variant carries the finite
`tcrv_rvv.lowering_descriptor = "i32-vadd-microkernel.v1"` descriptor, the plan
must also validate that the attached microkernel's `element_count` matches the
selected variant's bounded `tcrv_rvv.element_count` descriptor metadata:

```text
status: supported
emission kind: rvv-explicit-i32-vadd-microkernel-c-source
lowering pipeline: tcrv-export-rvv-microkernel-c
runtime ABI: rvv-i32-vadd-runtime-callable-c-abi.v1
runtime ABI kind: rvv-runtime-callable-c-abi
runtime ABI name: rvv-i32-vadd-runtime-callable-c-function.v1
runtime glue role: runtime-callable-i32-vadd-function
artifact kind: runtime-callable-c-source
```

This supported plan is a plugin-owned compiler handoff to the existing RVV
microkernel exporter and names the callable C ABI emitted in that source. It is
more concrete than the default unsupported RVV first-slice plan, but it is
still not generic RVV lowering, full runtime integration, object generation,
arbitrary kernel emission, correctness evidence, or performance evidence by
itself. The generic emission manifest may serialize the record after validating
the selected surface, the `tcrv_rvv.lowering_boundary` link, and the
emission-plan diagnostic. It must not reinterpret the adjacent
`tcrv_rvv.i32_vadd_microkernel` attachment as a second lowering boundary.

## Scalar Fallback Metadata Boundary

The first scalar fallback plugin slice may return a metadata-only emission
readiness result and materialize a metadata-only emission-plan diagnostic for
the portable fallback route:

```text
status: metadata-only
emission kind: portable-scalar-fallback-metadata-route
lowering pipeline: none-executable-metadata-only
runtime ABI: none-metadata-only
runtime ABI kind: host-scalar-fallback-metadata
runtime ABI name: portable-scalar-fallback-metadata-abi.v1
runtime glue role: metadata-only-host-fallback-boundary
artifact kind: metadata-diagnostic
```

This is still compiler-decision metadata. It does not prove that TianChen-RV
emitted LLVM IR, generated an object, linked a runtime, executed a scalar
kernel, proved correctness, or measured performance. Later scalar fallback
lowering must add plugin-local lowering code and validation artifacts before
reporting executable support. This metadata-only readiness/plan result also
does not license descriptor-only selected-boundary materialization: legacy
`tcrv_scalar.lowering_descriptor` / `tcrv_scalar.element_count` metadata must
not create `tcrv_scalar.lowering_boundary` unless a typed scalar microkernel
body or descriptorless typed default materialization is already the
selected-path authority.

The selected scalar fallback boundary is slightly more concrete than an
emission-plan diagnostic because it is a scalar extension-dialect op:
`tcrv_scalar.lowering_boundary`. It records selected fallback metadata and
required capability references, but it remains metadata-only and non-executable.
It is not evidence that the compiler emitted LLVM IR, assembled an object,
linked runtime glue, ran a scalar kernel, proved correctness, or measured
performance.

## Scalar Explicit Microkernel Target Export Boundary

Trigger: post-planning MLIR contains one selected `scalar-plugin` fallback
path, a matching plugin-owned `tcrv_scalar.lowering_boundary`, and exactly one
explicit `tcrv_scalar.i32_vadd_microkernel` or
`tcrv_scalar.i32_vsub_microkernel` or `tcrv_scalar.i32_vmul_microkernel` op for
that selected kernel/variant/role.

This export is the first bounded scalar fallback executable source slice. It
may emit a deterministic portable runtime-callable C library source artifact
that computes finite scalar i32 vector add, subtract, or multiply through a
pointer-plus-length C ABI. It is not generic scalar lowering, arbitrary scalar
kernel emission, generic runtime dispatch glue, object generation, linking,
benchmarking, or performance evidence.

Public route:

```text
tcrv-translate --tcrv-export-target-source-artifact
tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-c
```

Route metadata:

```text
status: supported
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
runtime ABI kind: scalar-runtime-callable-c-abi
artifact kind: runtime-callable-c-source
```

Contracts:

- Input must be real post-planning MLIR with one selected scalar fallback path.
- The selected variant must be owned by `origin = "scalar-plugin"` and require
  an available capability whose id is `scalar.fallback`.
- A matching direct child `tcrv_scalar.lowering_boundary` must identify the
  same source kernel, selected variant, origin, role, metadata-only status, and
  required capability refs.
- A matching direct child `tcrv_scalar.i32_vadd_microkernel` or
  `tcrv_scalar.i32_vsub_microkernel` or `tcrv_scalar.i32_vmul_microkernel` must
  identify the same selected path and required capability refs with a bounded
  element count.
- Output must be deterministic portable C with a callable function
  `void <name>(const int32_t *lhs, const int32_t *rhs, int32_t *out, size_t n)`
  that performs scalar i32 add, subtract, or multiply over the provided arrays
  according to the selected scalar microkernel family.
- The default artifact must not include a hidden `main`, stdio-only self-check
  machinery, or a self-check success marker.
- Output must not include RVV headers, RVV intrinsics, route-spoof claims,
  timestamps, absolute paths, raw logs, credentials, benchmark sizes,
  latency/throughput numbers, or performance claims.

Missing selected scalar path, missing or stale scalar lowering boundary,
missing or stale scalar microkernel, unavailable fallback capability, unknown
route id, unsupported artifact kind, route spoofing, offload-only paths, and
ambiguous multiple supported artifacts must fail before source output.

## Support-Layer Finite Binary Runtime ABI Contract

The current bounded finite binary executable slice must have one compiler-owned
C++ runtime ABI contract shape in the support layer. The stable support-layer
planning entry points are descriptorless:

```cpp
llvm::Expected<tianchenrv::support::FiniteBinaryCallableABIPlan>
tianchenrv::support::buildFiniteBinaryCallableABIPlan(
    tcrv::exec::KernelOp kernel,
    const tianchenrv::support::FiniteBinaryRuntimeABIContract &contract);

const tianchenrv::support::I32BinaryRuntimeABIContract &
tianchenrv::support::getI32BinaryRuntimeABIContract(llvm::StringRef familyID);
```

The contract owns only reusable runtime ABI metadata for the bounded i32 binary
callable shape and selected-family identity fields:

- ordered callable C parameters:
  `const int32_t *lhs`, `const int32_t *rhs`, `int32_t *out`, `size_t n`;
- callable role requirements used by target artifact route validation;
- `tcrv.exec.mem_window` specs for lhs/rhs/out buffer roles;
- `tcrv.exec.runtime_param` specs for runtime element count and the optional
  dispatch availability guard;
- stable runtime ABI identity strings for the selected add/sub/mul RVV
  callable, scalar callable, and RVV+scalar dispatch callable surfaces, keyed by
  the selected family id such as `i32-vadd`, `i32-vsub`, or `i32-vmul`.

Support headers for callable planning must not import RVV, scalar, or i32
descriptor/registration headers. Target-owned RVV, scalar, dispatch, and
offload code may adapt their selected typed family/registration records into a
`FiniteBinaryRuntimeABIContract`, but that adaptation is outside the Support
default API and must occur only after typed selected-path authority has been
established.

The temporary add-only runtime ABI compatibility APIs have been retired. Active
RVV, scalar, dispatch, offload descriptor, and target artifact exporter code
that handles i32 add/sub/mul must consume the selected-family-id or generic
finite-binary contract APIs directly when it needs ABI shape, ABI identity,
mem-window specs, runtime-param specs, dispatch guard specs, role binding, or
callable-plan validation. Reintroducing `I32VAdd*` ABI wrappers or
descriptor-shaped Support overloads would reopen obsolete compatibility
surfaces and should be treated as a descriptor-exit regression unless a future
spec explicitly re-establishes such an API for a non-compatibility purpose.

The contract must not own target artifact route ids, artifact kinds, source vs
header vs object selection policy, bundle metadata, descriptor-local metadata,
evidence paths, ssh facts, target capabilities, selected march/mabi, or
performance/correctness claims. It also must not turn runtime SSA/control values
into compile-time facts: the dispatch availability guard remains a
`tcrv.exec.runtime_param` role and the selected dispatch case must still link to
that guard through typed `runtime_guard_required = true` plus `runtime_guard`.

Compiler-owned dispatch runtime-guard materialization happens in the transform
layer before selected lowering-boundary and emission-plan materialization. The
generic materializer may inspect `TargetCapabilitySet` availability/conflict
facts and the typed `tcrv.exec.case runtime_guard_required` marker, but it must
create only compute-free exec IR: one same-kernel dispatch-availability
`runtime_param` and selected `tcrv.exec.case runtime_guard` symbol references.
Generic `condition`, `guard`, or `policy` strings may be retained as printable
annotations, but they are not semantic runtime ABI guard triggers. RVV, scalar, and
future plugin lowering code may consume or validate those links, but must not
privately invent the dispatch guard parameter or attach dispatch-case links as a
plugin-local side effect.

Required tests for changes to this contract:

- C++ tests must prove the callable parameter order, roles, C spellings, and
  ownership match the contract for add/sub/mul;
- C++ tests must prove add/sub/mul expose distinct selected-family-derived RVV,
  scalar, and dispatch runtime ABI identities;
- C++ tests must prove Support callable-planning headers do not require target
  descriptor headers or descriptor-shaped planning overloads;
- at least one active target exporter or validation path must reject metadata
  that disagrees with the contract;
- existing lit/FileCheck coverage must continue proving that RVV, scalar, and
  RVV+scalar dispatch exports consume the same `tcrv.exec.mem_window` and
  `tcrv.exec.runtime_param` IR-backed ABI plan.

## Target-Layer RVV Binary Runtime ABI Contract

The selected RVV binary microkernel route has a target-owned runtime ABI
contract over the finite RVV family registry:

```cpp
const tianchenrv::target::rvv::RVVBinaryRuntimeABIContract &
tianchenrv::target::rvv::getRVVBinaryRuntimeABIContract(
    const tianchenrv::target::rvv::RVVBinaryFamilyDescriptor &family);
```

This contract covers exactly the currently supported direct RVV binary
families `i32-vadd`, `i32-vsub`, `i32-vmul`, `i64-vadd`, `i64-vsub`, and
`i64-vmul`. It derives from the selected `RVVBinaryFamilyDescriptor`:

- ordered callable C parameters:
  `lhs`, `rhs`, `out`, and runtime element count `n`;
- C pointer spellings from the selected dtype, for example
  `const int32_t *` / `int32_t *` for i32 families and
  `const int64_t *` / `int64_t *` for i64 families;
- target-export ABI ownership and callable role requirements;
- `tcrv.exec.mem_window` specs for lhs/rhs/out buffer roles;
- the direct runtime element-count `tcrv.exec.runtime_param` spec;
- RVV runtime ABI kind/name, runtime glue role, and external ABI component
  group for source/header/object bundle coherence.

The older i32 support-layer contract remains the scalar/dispatch compatibility
owner for bounded i32 shared surfaces. Direct RVV microkernel planning,
readiness, emission-plan diagnostics, target-artifact exporter registration,
route-local candidate preflight, header/object helper metadata, and manifest or
bundle serialization that handle direct RVV i32/i64 add/sub/mul must consume
the selected RVV binary contract when a concrete selected RVV family is
available. `i32-vadd` compatibility wrappers may stay as wrappers only; they
must not be the active source of truth for `i32-vsub`, `i32-vmul`, or any i64
RVV selected family.

Direct RVV binary source/header/object route ownership is target-local. The RVV
target support layer must expose one route contribution source of truth for the
finite i32/i64 add/sub/mul families, and both public direct helper registration
and `TargetArtifactExporterRegistry` population must iterate that contribution
data rather than hand-maintaining duplicate per-family route lists in tools or
generic target front doors. Missing contribution leaves the generic registry
without RVV direct routes, and duplicate contribution must fail through the
generic duplicate-route diagnostics. The legacy convenience route ids
`tcrv-export-rvv-microkernel-c`, `tcrv-export-rvv-microkernel-header`, and
`tcrv-export-rvv-microkernel-object` remain compatibility aliases that follow
the selected RVV binary family; family-specific route ids must reject selected
records from a different RVV binary family before emitting source, header, or
object bytes.

Required tests for changes to this contract:

- C++ tests must prove all direct RVV i32/i64 add/sub/mul families expose the
  selected-family runtime ABI parameter order, roles, C spellings, and
  ownership;
- C++ tests must prove i64 add/sub/mul selected plans and supported emission
  plans expose family-specific runtime ABI identity and `int64_t` callable
  parameters rather than stale i32-vadd defaults;
- target artifact export tests must prove an i64 route rejects stale i32-vadd
  runtime ABI metadata before source/header/object or bundle output;
- registry tests must prove RVV direct route contribution is explicit,
  duplicate contribution fails, and source/header/object direct route ids remain
  unique across the finite RVV binary family manifest.

## Host RVV + Scalar I32 VAdd Dispatch C Export Boundary

Trigger: post-planning MLIR contains one selected `rvv-plugin` dispatch case
and one selected `scalar-plugin` dispatch fallback for the same
`tcrv.exec.kernel`. The RVV path must have a matching
`tcrv_rvv.lowering_boundary`, matching `tcrv_rvv.i32_vadd_microkernel`, and a
supported runtime-callable C source emission-plan route. The scalar fallback
path must have a matching `tcrv_scalar.lowering_boundary`, matching
`tcrv_scalar.i32_vadd_microkernel`, and a supported runtime-callable C source
emission-plan route.

This export is the first bounded host-side runtime dispatch glue slice. It may
emit one deterministic C source file that embeds the existing RVV callable
function and scalar callable fallback function, then adds a dispatcher
function:

```c
void tcrv_dispatch_i32_vadd_<kernel>(const int32_t *lhs,
                                     const int32_t *rhs,
                                     int32_t *out, size_t n,
                                     int rvv_available);
```

The dispatcher must call the RVV callable function when `rvv_available` is
non-zero and the scalar fallback callable function otherwise. The guard is an
explicit host-provided parameter. Before source output, the exporter must
validate real direct `tcrv.exec.runtime_param` IR for both the
runtime-element-count parameter and the dispatch-availability-guard parameter,
and it must reject stale or detached ABI metadata that does not agree with that
IR. The same selected `tcrv.exec.dispatch` is also the fallback branch source
of truth: the exporter must resolve exactly one direct `tcrv.exec.fallback`,
validate that its target resolves to a direct same-kernel `tcrv.exec.variant`,
and reject any selected scalar callable route whose selected variant does not
match that `tcrv.exec.fallback` target before source or object output. This
slice does not implement automatic hardware probing, dynamic loading, object
generation, linking, arbitrary dispatch lowering, benchmarking, correctness
evidence, or performance evidence.

An explicit self-check harness export may wrap the same generated dispatcher in
a bounded `main` that invokes both branches over bounded local arrays with
explicit runtime element-count ABI inputs:

```text
tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c
tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-self-check-c
tcrv-translate --tcrv-export-rvv-scalar-i32-vmul-dispatch-self-check-c
```

That harness is target-owned runtime invocation evidence tooling for the
finite i32 add/sub/mul dispatch family only. The command route must match the
selected callable family: the vadd self-check route must reject vsub artifacts,
and the vsub/vmul self-check routes must reject artifacts from other families.
It may emit one bounded family-specific success marker after both scalar
fallback and RVV branch checks pass. It must not replace the default library-style dispatch
artifact, broaden selected-path validation, perform automatic hardware probing,
generate objects, link a runtime library, report benchmarks, or claim generic
RVV lowering correctness. Real correctness claims for the RVV branch still
require separate `ssh rvv` compile/run evidence for the generated harness
source and selected flags.

### Dispatch Self-Check Harness Export

#### 1. Scope / Trigger

Trigger: a post-planning module already satisfies the host RVV+scalar finite
i32 add/sub/mul dispatch C export boundary, and the caller wants explicit runtime
invocation evidence for the generated dispatcher rather than the default
library-style source artifact.

This is a target-owned evidence export mode for the finite i32 add/sub/mul
dispatcher family. It must not become a generic object/link/runtime pipeline.

#### 2. Signatures

Public command:

```text
tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c
tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-self-check-c
tcrv-translate --tcrv-export-rvv-scalar-i32-vmul-dispatch-self-check-c
```

C++ entry point:

```cpp
llvm::Error exportRVVScalarI32VAddDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os);
llvm::Error exportRVVScalarI32VSubDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os);
llvm::Error exportRVVScalarI32VMulDispatchSelfCheckC(mlir::ModuleOp module,
                                                     llvm::raw_ostream &os);
```

#### 3. Contracts

- The export reuses the same selected-path, lowering-boundary,
  emission-plan, route-id, artifact-kind, and structured runtime ABI parameter
  validation as `--tcrv-export-rvv-scalar-i32-vadd-dispatch-c`.
- The default dispatcher export remains library-style and must not contain a
  hidden `main`, self-check helper, or success marker.
- The self-check source embeds the same RVV callable source, scalar callable
  source, and dispatcher function, then appends a bounded harness.
- The harness calls the dispatcher with `rvv_available = 0` to exercise the
  scalar fallback branch and with `rvv_available = 1` to exercise the RVV
  branch.
- The harness expected arithmetic must come from the selected family: vadd
  checks `lhs + rhs`, vsub checks `lhs - rhs`, and vmul checks `lhs * rhs`. A
  stale route, ABI name, intrinsic, callable symbol stem, or success marker from
  another family is invalid.
- The harness must exercise more than one target/export-owned runtime `n` ABI
  value. The current bounded slice uses `n = 7` and `n = 16` for each branch.
- The harness uses bounded local arrays and a target/export-owned runtime `n`
  ABI argument. Descriptor-local `element_count` remains metadata only and
  must not become the callable runtime loop bound.
- The harness may print one bounded success marker after both checks pass.
- Output must not include benchmark sizes, throughput, latency, raw logs,
  credentials, URLs, absolute artifact paths, or performance claims.

#### 4. Validation & Error Matrix

- Missing selected RVV dispatch case -> same fail-before-source diagnostic as
  the default dispatch export.
- Missing selected scalar dispatch fallback -> same fail-before-source
  diagnostic as the default dispatch export.
- Missing or stale RVV/scalar lowering boundary -> fail before source output.
- Missing or unsupported callable emission-plan metadata for either branch ->
  fail before source output.
- Missing structured `lhs`, `rhs`, `out`, or runtime `n` ABI parameter metadata
  on either callable route -> fail before source output.
- Route spoofing, wrong origin, wrong role, or unsupported artifact kind ->
  fail before source output.
- Self-check route family mismatch, such as vsub artifacts routed through the
  vadd self-check command or vadd artifacts routed through the vsub self-check
  command, or vmul artifacts routed through an add/sub self-check command ->
  fail before source output.
- Remote compile/run failure of generated harness -> evidence collection
  failure only; it must not be converted into a compiler-side success claim.

#### 5. Good / Base / Bad Cases

- Good: execution-planning pipeline materializes RVV and scalar callable paths;
  the self-check export compiles on `ssh rvv`, runs both branch checks, and
  prints the bounded success marker.
- Base: local lit/FileCheck checks prove the harness structure without
  contacting `ssh rvv`.
- Bad: the default dispatcher export silently includes a `main`, or the
  self-check command bypasses runtime ABI metadata validation.

#### 6. Tests Required

- lit/FileCheck must prove the default dispatcher export still lacks `main`,
  self-check helpers, and runtime success markers.
- lit/FileCheck must prove the self-check export appends a harness that calls
  the dispatcher with both `rvv_available = 0` and `rvv_available = 1`.
- lit/FileCheck must prove the harness passes explicit runtime element-count
  ABI values through the dispatcher rather than hard-coding descriptor-local
  `element_count` as the runtime loop bound.
- Pipeline-to-self-check coverage must use
  `tcrv-opt --tcrv-execution-planning-pipeline | tcrv-translate
  --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c` for vadd and the
  matching `--tcrv-export-rvv-scalar-i32-vsub-dispatch-self-check-c` route for
  vsub or `--tcrv-export-rvv-scalar-i32-vmul-dispatch-self-check-c` route for
  vmul.
- Any RVV runtime/correctness claim for the self-check source must include
  separate `ssh rvv` compile/run evidence and must name the selected compile
  flags.

#### 7. Wrong vs Correct

Wrong:

```text
The dispatch self-check passed, so TianChen-RV supports generic RVV lowering
and runtime integration.
```

Correct:

```text
The generated bounded RVV+scalar i32 add/sub/mul dispatch self-check source
compiled and ran on ssh rvv with selected flags; this proves only that the
finite family-selected dispatcher harness invoked both callable branches
correctly for the explicit runtime element-count ABI values in the harness.
```

### Dispatch Library Object Export

#### 1. Scope / Trigger

Trigger: the same post-planning module satisfies the default host RVV+scalar
i32 family-selected dispatch C export boundary, and the caller requests a bounded
object-file artifact for that exact generated library-style dispatch source.

This is the generic target-owned object-generation boundary for the finite
RVV+scalar i32 add/sub/mul dispatcher. It compiles the runtime-callable dispatch
library source without adding a hidden `main`, self-check helper, or success
marker. It must not become a generic object/link/runtime pipeline.

#### 2. Signatures

Public command:

```text
tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-object
tcrv-translate --tcrv-export-target-artifact
```

The direct command remains the first finite convenience alias; generic
artifact-kind-aware export must select the family-specific add/sub/mul object
route from registry metadata.

C++ entry point:

```cpp
llvm::Error exportRVVScalarI32VAddDispatchObject(
    mlir::ModuleOp module, llvm::raw_ostream &os);
```

Explicit evidence-helper command:

```text
tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-object
tcrv-translate --tcrv-export-rvv-scalar-i32-vsub-dispatch-self-check-object
tcrv-translate --tcrv-export-rvv-scalar-i32-vmul-dispatch-self-check-object
```

C++ entry point:

```cpp
llvm::Error exportRVVScalarI32VAddDispatchSelfCheckObject(
    mlir::ModuleOp module, llvm::raw_ostream &os);
llvm::Error exportRVVScalarI32VSubDispatchSelfCheckObject(
    mlir::ModuleOp module, llvm::raw_ostream &os);
llvm::Error exportRVVScalarI32VMulDispatchSelfCheckObject(
    mlir::ModuleOp module, llvm::raw_ostream &os);
```

#### 3. Contracts

- The object route must first reuse the same selected-path,
  lowering-boundary, emission-plan, route-id, artifact-kind, and structured
  runtime ABI parameter validation as the dispatch source exporter.
- The generic object route must compile the exact generated default
  library-style dispatch C source for the selected dispatcher. It must not
  silently compile a different standalone probe, omit the scalar fallback
  branch, omit the RVV branch, or add evidence-only harness code.
- The object route must derive the compile target from structured RVV
  architecture capability metadata. The first supported value is `riscv64`;
  host-default target compilation must not be used as a substitute.
- The explicit self-check object helper may compile the exact generated
  self-check C source, but that helper must remain outside the generic
  `--tcrv-export-target-artifact` route.
- The selected compile facts are target-owned RVV dispatch facts. The route may
  pass the architecture-derived target, selected `-march`, and optional
  `-mabi` values already preserved by exact RVV capabilities, explicit
  relation-provider target profile metadata, or microkernel metadata. Shared
  generic routing must not learn RVV, scalar, dtype, vendor, runtime, or
  toolchain semantics for this.
- Direct binary stdout is the public output contract: success writes only the
  generated object bytes to stdout. Diagnostics go through the normal
  `tcrv-translate` error path. Textual metadata comments remain in the
  generated temporary source and are not prepended to the object stream.
- The route must use LLVM/C++ support facilities for temporary files, process
  execution, diagnostics, and binary output, and must pass compiler arguments
  as structured argv entries rather than shell-concatenated command text.
- If local/native RISC-V clang, RVV headers, selected flags, or target sysroot
  support are unavailable, the route must fail closed with an exact diagnostic.

#### 4. Validation & Error Matrix

- Missing selected RVV dispatch case, missing scalar dispatch fallback, missing
  or stale lowering boundaries, missing supported callable emission plans,
  route spoofing, wrong origin/role/artifact kind, or missing structured
  `lhs`/`rhs`/`out`/runtime `n` ABI parameters -> fail before object creation.
- Missing, duplicate, unknown, non-variant, or scalar-callable-mismatched
  `tcrv.exec.fallback` target in the selected dispatch -> fail before source
  or object creation with a diagnostic that names the selected scalar callable
  route and fallback target symbols.
- Missing selected RVV `tcrv_rvv.required_march` or missing preserved selected
  march capability metadata -> fail before object creation.
- Missing, malformed, unavailable, or unsupported RVV architecture capability
  metadata -> fail before object creation.
- Conflicting selected MABI metadata -> fail before object creation.
- Missing `clang`, unsupported local target, missing `riscv_vector.h`, missing
  target libc headers, unsupported `-march`/`-mabi`, or other compile failure ->
  fail with a bounded object-route diagnostic and no object claim.
- Successful generic object emission only proves that the bounded generated
  library-style dispatch source was compiled into a non-empty object-file
  artifact with the selected flags and no hidden self-check entry point.
  Runtime/correctness still requires separate `ssh rvv` compile/run evidence
  when claimed.

Contracts:

- Input must be real post-planning MLIR with one selected RVV dispatch case and
  one selected scalar dispatch fallback in the same kernel.
- The exporter must validate selected-path roles, lowering-boundary links,
  executable microkernel attachments, runtime ABI kind/name, runtime glue role,
  artifact kind, required capability refs, and export route ids for both paths
  before source output.
- Output must preserve bounded metadata comments for the kernel, RVV selected
  variant, scalar fallback variant, roles, runtime ABI fields, runtime glue
  roles, artifact kinds, route ids, and required capability refs.
- Output must preserve RVV intrinsic code in the embedded RVV callable function
  and scalar i32 addition in the embedded scalar callable fallback function.
- Generic library-object output must define the dispatcher and embedded
  callable symbols but must not define `main` or self-check symbols.
- Explicit self-check-object output may define `main` and the self-check helper
  because that route is an evidence helper.
- Missing RVV callable metadata, missing scalar callable fallback metadata,
  stale lowering boundaries, unsupported artifact kinds, unsupported origins,
  wrong roles, and ambiguous duplicate paths must fail before source output.

### Dispatch Self-Check Executable Evidence Bridge

The repo-owned executable evidence bridge is
`scripts/rvv_scalar_dispatch_e2e.py`. It is Python runner/evidence tooling
only. It must not implement compiler IR, plugin decisions, target selection,
capability modeling, lowering, emission, runtime ABI, correctness logic, or
performance measurement.

The bridge must consume compiler-generated artifacts from the planned dispatch
pipeline:

```bash
tcrv-opt input.mlir --tcrv-execution-planning-pipeline
tcrv-translate --tcrv-export-target-source-artifact post_planning.mlir
tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c \
  post_planning.mlir
```

For a bounded linalg frontend input, the bridge may run the existing add/sub/mul
frontend lowering pass before execution planning:

```bash
scripts/rvv_scalar_dispatch_e2e.py \
  --arithmetic-family i32-vsub \
  --lower-linalg-frontend \
  --input test/Target/RVVScalarDispatch/rvv-scalar-i32-vsub-dispatch-generic-route.mlir
```

When invoked with `--use-target-artifact-bundle`, the same bridge consumes the
registry-derived target artifact bundle instead of the direct self-check source
route:

```bash
scripts/rvv_scalar_dispatch_e2e.py \
  --use-target-artifact-bundle \
  --use-plan-and-export-bundle-front-door \
  --arithmetic-family i32-vsub \
  --input test/Target/TargetArtifactBundleExport/plan-linalg-i32-vsub-and-export-target-artifact-bundle.mlir
```

The underlying compiler command for the bundle front door is:

```bash
tcrv-translate --tcrv-export-target-artifact-bundle \
  --tcrv-target-artifact-bundle-output-dir=<fresh-dir> input.mlir
```

Bundle mode must parse only `tianchenrv-target-artifact-bundle.index` as the
stable file name, discover generated source/header/object file names from that
index, validate artifact kind, route, owner, runtime ABI kind/name, component
selected paths, explicit external ABI component metadata, the ordered
`runtime_abi_parameter[index]` signature (`c_name`, `c_type`, `role`, and
`ownership`), and evidence role, then generate a small external C caller from
the emitted header prototype and the compiler-emitted signature metadata.
Dry-run bundle mode records bundle export, index parsing, file discovery,
caller generation, hashes, command logs, and evidence JSON under
`artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/<run-id>/`; it is not runtime
or correctness evidence. Real ssh bundle mode may copy the generated source,
generated header, generated relocatable object, and generated external caller
to `ssh rvv`, compile the generated dispatch source and caller object on that
host, link and run the caller against the source-built object, then also link
and run the same caller against the generated bundle object. It must require
the bounded bundle external ABI success marker and record sanitized
host/toolchain facts such as `uname`, architecture, and clang path/version.
Successful ssh bundle evidence proves only that the finite family-selected
RVV+scalar i32 add/sub/mul compiler-generated bundle source/header/object external
ABI caller executed both `rvv_available = 0` and `rvv_available = 1` branches
on that host for the explicit runtime `n = 7` and `n = 16` caller inputs. For
vsub, the generated external caller must check `lhs - rhs`; for vmul, it must
check `lhs * rhs`; neither may use stale vadd semantics. It is not generic lowering, arbitrary RVV support, dynamic runtime
integration, performance evidence, or broad correctness coverage.

Dry-run mode records sanitized post-planning MLIR, emission manifest, generated
library dispatch source, generated self-check dispatch source, hashes, command
logs, and evidence JSON under
`artifacts/tmp/rvv_scalar_dispatch_e2e/<run-id>/`. It proves only that the
existing compiler tools can produce the planned dispatch handoff and bounded
source artifacts.

Real ssh mode may copy the generated self-check source to `ssh rvv`, compile it
with selected `-march` and optional `-mabi` metadata into a relocatable object,
link an executable, run that executable, and require the bounded success marker
from the harness. This is a runtime/evidence bridge over target-owned generated
C. The bridge must fail closed on ssh, scp, compiler, header, object, link,
run, timeout, or success-marker failure and must record sanitized command logs
rather than synthesizing runtime evidence.

Successful ssh evidence proves only that the finite generated family-selected
RVV+scalar i32 add/sub/mul dispatcher self-check executable compiled, linked, and
ran both the scalar fallback and RVV dispatch branches on the selected RVV host
flags for the explicit runtime `n = 7` and `n = 16` ABI inputs. It is not
generic high-level lowering correctness, arbitrary RVV emission support,
object-route proof for unrelated kernels, dynamic runtime integration,
performance evidence, or broad correctness coverage.

### Dispatch ABI Header Export

Trigger: the same post-planning module satisfies the host RVV+scalar i32-vadd
or registry-selected i32-vsub/i32-vmul dispatch C export boundary and carries
the selected RVV compile metadata needed by the matching library-object route.
A target/export tool may emit a bounded C header for an external runtime caller:

```text
tcrv-translate --tcrv-export-rvv-scalar-i32-vadd-dispatch-header
tcrv-translate --tcrv-export-target-header-artifact
```

The direct header command may remain a convenience alias, but the generic
header command must select the header through the target artifact exporter
registry using artifact kind `runtime-callable-c-header` and family-specific
route metadata. The header exporter
must reuse the same selected-path, lowering-boundary, callable route metadata,
scalar `tcrv.exec.fallback` link, `tcrv.exec.case runtime_guard` link, selected
RVV compile metadata, and structured runtime ABI parameter validation as the
dispatch source/object exporters. It must build the prototype from the same
exec-IR-backed `DispatchABIPlan` consumed by the dispatch source and object
exporters, not by parsing generated C comments or trusting detached candidate
metadata. The `lhs`, `rhs`, and `out` parameters come from direct
`tcrv.exec.mem_window` ABI roles; runtime `n` and the dispatch availability
guard come from direct `tcrv.exec.runtime_param` ABI/control roles, with the
guard linked through the selected RVV case's `runtime_guard` symbol.

Output is limited to a deterministic include guard, required standard
integer/size headers, an `extern "C"` guard for C++ callers, and exactly one
dispatcher function prototype whose parameter order and C type spellings match
the generated dispatch definition. The header must not embed computation,
callable source bodies, a hidden `main`, self-check code, runtime probing,
linking commands, correctness claims, performance claims, raw logs, credentials,
or artifact paths. Successful header export proves only that the compiler can
materialize the bounded external C ABI handoff for this dispatcher. Linking or
runtime claims require separate object/link or real `ssh rvv` evidence with
the claim scope stated explicitly.

## Runtime Offload Metadata Boundary

The first runtime-offload plugin slice may return a metadata-only emission
readiness result and materialize a supported descriptor emission-plan
diagnostic for a generic runtime ABI handoff artifact:

```text
status: supported
emission kind: runtime-offload-handoff-descriptor
lowering pipeline: tcrv-export-offload-runtime-descriptor
runtime ABI: generic-runtime-offload-c-abi-handoff.v1
runtime ABI kind: runtime-offload-c-abi-handoff
runtime ABI name: generic-runtime-offload-c-abi-handoff.v1
runtime glue role: plugin-owned-runtime-offload-glue-boundary
artifact kind: runtime-offload-handoff-descriptor
```

This supported plan means only that the compiler can export a deterministic
target-owned handoff descriptor for downstream runtime integration. It does not
emit vendor runtime calls, allocate or copy device buffers, compile accelerator
kernels, generate objects, link runtime libraries, run hardware, prove
correctness, or measure performance. The selected
`tcrv_offload.lowering_boundary` op records the plugin-local runtime-offload
handoff boundary, but it remains metadata-only and non-executable. Its
availability depends on explicit `offload.runtime` capability metadata and
plugin legality; it cannot resurrect unavailable, malformed, illegal, or
unselected variants.

The descriptor export route must validate that the selected offload path has a
matching `tcrv_offload.lowering_boundary`, runtime ABI kind/name, required
capability refs, emission kind, artifact kind, route id, and bounded handoff
reason. Unsafe strings, URLs, raw credentials, stale selected variants, stale
lowering boundaries, route spoofing, unknown route ids, and unsupported artifact
kinds must fail before descriptor output.
The emitted descriptor should expose a stable schema version, descriptor kind,
descriptor status, external adapter contract id, lowering-boundary metadata
status, runtime ABI, runtime-offload handoff kind, artifact component role,
evidence role, and explicit non-claim metadata stating that no vendor runtime
call, DMA/buffer management, accelerator kernel, object generation, hardware
execution, correctness proof, or performance claim was produced. The offload
emission plan may also carry bounded `selected_plan_metadata` for the
runtime-offload capability id, handoff kind, and descriptor-only scope; the
descriptor exporter and bundle index must preserve that metadata when present.
The descriptor exporter must also require deterministic `runtime_abi_parameters`
for the selected offload plan and verify them against the IR-backed
`tcrv.exec.mem_window` and `tcrv.exec.runtime_param` ABI role declarations before
writing output. The descriptor must serialize role, C name, C type, purpose,
ownership, source symbol, and buffer-only binding/access/memory-space fields as
compiler handoff contract metadata. Missing ABI roles, duplicate roles,
malformed C names or C type spellings, metadata that does not mirror the
IR-backed ABI plan, missing selected-plan handoff metadata, or descriptor-local
metadata that embeds sample runtime values or hardware facts must fail closed.
When the descriptor is emitted through the target artifact bundle route, the
bundle index must carry the descriptor route, owner/origin, runtime ABI,
component role, runtime ABI role signature, selected-plan metadata, and handoff
kind metadata and must remain a build handoff index only.

## Emission Manifest Export Boundary

After planning has materialized an explicit selected surface and
`tcrv.exec.diagnostic {reason = "emission_plan"}` metadata, a target/export
tool may serialize a deterministic emission handoff manifest. The manifest is a
compiler handoff artifact for downstream lowering/runtime-glue integration. It
may include module and kernel symbols, selected variant symbols, origin plugin,
selected-path role, dispatch cases and fallback targets, lowering-boundary
diagnostic links, runtime ABI kind/name, runtime glue role, emission status,
required capability refs, preference metadata, and bounded plugin
explanation/diagnostic text.

The exporter must validate the post-planning MLIR before writing any manifest.
It must reject missing selected surfaces, stale selected references, selected
paths without exactly one complete emission-plan diagnostic, duplicate runtime
ABI ownership diagnostics, malformed runtime ABI ownership fields, unknown
diagnostic targets, and unbounded or credential-like explanatory text. Export
failure must not produce a partial manifest.

The manifest does not perform plugin registry routing, legality recovery,
variant selection, lowering-boundary materialization, LLVM lowering, RVV
intrinsic emission, scalar code generation, object generation, runtime library
linking, hardware execution, correctness validation, or performance
measurement. It must remain target-neutral: it serializes generic selected-path
and diagnostic metadata without interpreting RVV, scalar fallback, IME, offload,
Sophgo, AME, vendor, dtype, shape, microarchitecture, probe evidence, runtime,
correctness, or performance semantics.

When a target artifact exporter registry knows additional supported artifacts
for the selected surface, the manifest may also serialize a deterministic
`target_artifacts` handoff bundle. These records are registry-derived compiler
handoff metadata, not proof that the artifact was emitted, linked, run, correct,
or performant. Each record should name the artifact kind, route/exporter id,
target or plugin owner, whether the generic front door can select it, the
command-facing generic selector, whether a direct helper route exists, the
matching runtime ABI kind/name when applicable, and a bounded evidence role such
as `compiler-artifact`, `header-declaration`, or `relocatable-object`. When a
record belongs to a non-empty external ABI `component_group`, the bundle/index
contract must also publish an ordered runtime ABI signature with
`runtime_abi_parameter[index].c_name`, `.c_type`, `.role`, and `.ownership`.
All source/header/object records in that external ABI group must agree on
runtime ABI kind/name, external ABI name, component selected variants/roles,
the full ordered runtime ABI signature, and any selected-plan metadata entries.
Missing signatures, duplicate roles, mismatched name/type/ownership for the
same role, reordered parameters, malformed selected-plan metadata, or
mismatched selected-plan metadata fail closed before bundle export. Source,
header, and object routes must remain separate records. Composite dispatch
records may be attached to the selected dispatch surface and must preserve the
component selected variants/roles rather than moving RVV/scalar branch
semantics into `tcrv.exec` or generic manifest code. For RVV+scalar dispatch
records, the target-owned dispatch exporter must consume the validated direct
RVV component selected-config/runtime AVL contract before deriving dispatch
source/header/object or bundle records. The composite bundle record must carry
dispatch-specific selected-plan metadata that exposes the consumed RVV
component contract fields: runtime element-count C name, selected vector
config, selected dispatch role, and descriptor-local element count as bounded
component capacity metadata only. Missing or mismatched direct RVV selected
config contract fields must fail before bundle output. Unsupported or
metadata-only selected paths must omit target artifact records instead of
fabricating route data.
