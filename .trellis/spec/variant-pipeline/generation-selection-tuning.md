# Generation, Selection, And Tuning

## Core Principle

Do not generate generic `tcrv` compute ops first. Extension plugins propose execution variants directly, and core passes organize, verify, select, dispatch, and lower them.

Current TianChen-RV work may start from hand-written or test TianChen-RV MLIR,
already materialized `tcrv.exec.variant`, selected-boundary IR,
`tcrv.exec.mem_window`, `tcrv.exec.runtime_param`, typed extension-family
bodies, or selected-path metadata. High-level MLIR op analysis is a future
extension point for `linalg`, `stablehlo`, `tosa`, and similar inputs; it is
not a precondition for integrating a new extension plugin today. When the
frontend lowering owner is selected, the first high-level path may start from
hand-written or test `linalg` inputs and lower them into TianChen-RV surfaces
that the backend/plugin pipeline can consume.

Correct:

```text
high-level op -> plugin-proposed execution variants -> capability-driven selection/lowering
```

Wrong:

```text
high-level op -> generic tcrv op -> backend-specific lowering
```

## Input

Inputs may include:

```text
hand-written TianChen-RV MLIR
test TianChen-RV MLIR
tcrv.exec.kernel with capabilities
already materialized tcrv.exec.variant
selected lowering-boundary IR
tcrv.exec.mem_window
tcrv.exec.runtime_param
typed extension-family body
bounded selected-path metadata
linalg.matmul
linalg.generic reduction
stablehlo dot_general
stablehlo softmax-like region
tosa ops
custom high-level kernel dialect
```

When the input is high-level MLIR, TianChen-RV assumes the high-level op already
expresses computation semantics. It does not recover or reinvent algorithm
semantics. When the input is already TianChen-RV MLIR, plugin work should focus
on capability, variant, selected-boundary, lowering, emission, runtime ABI, and
artifact routes rather than inventing a high-level lowering layer.
The backend-first path and the `linalg` frontend path are compatible: backend
plugin integration may proceed from hand-written TianChen-RV MLIR while a
frontend owner later introduces `linalg` tests that feed those same backend
surfaces.

Future high-level frontend work is semantic-preserving conversion, not semantic
recognition. It maps semantics already expressed by the source dialect into
TianChen-RV execution surfaces; it must not infer computation from route ids,
descriptors, artifact names, old microkernel family metadata, or other deleted
source-authority residue.

## Deleted Core RVV Source Frontend Slice

### 1. Scope / Trigger

The former bounded linalg/vector RVV binary frontend pass family is deleted as
a core semantic branch. Core transforms must not inspect finite RVV
add/sub/mul linalg or vector source bodies, consult RVV binary family records,
or materialize `tcrv.exec` kernels from source arithmetic.

The deleted public option family includes the old source, linalg RVV, linalg
i32 compatibility, and vector i32 add/sub/mul adapters. Those historical option
spellings must not remain as wrappers, delegating aliases, quarantine modes,
renamed compatibility paths, or active negative fixture contracts.

### 2. Contracts

- High-level MLIR source-to-`tcrv.exec` lowering is a future extension point,
  not current core transform behavior.
- Any rebuild must be plugin/interface-owned and follow the extension-family
  construction and common EmitC route specs.
- Core planning still consumes already materialized `tcrv.exec.kernel`,
  `tcrv.exec.variant`, selected-boundary IR, `tcrv.exec.mem_window`,
  `tcrv.exec.runtime_param`, and plugin-local extension-family ops.
- Source markers such as `tcrv_frontend_lowering` are historical deleted-route
  residue only; core transforms must not use them to infer RVV arithmetic
  semantics from high-level source bodies.

### 3. Validation & Error Matrix

- Invoking any deleted source-to-exec option through `tcrv-opt` must fail as an
  unknown command-line option or equivalent deleted-route diagnostic.
- The deleted `tcrv-translate` plan-and-export target artifact bundle wrapper
  must stay absent. Translate-side target artifact bundle export may consume
  already materialized TianChen-RV execution surfaces, but must not hide
  source-to-exec lowering or execution planning behind a second production
  bundle front door.
- Tests whose only purpose was to prove linalg/vector source lowering success,
  semantic validation diagnostics, compatibility alias delegation, or named
  deleted-pass absence must be deleted instead of kept alive as production
  coverage.
- Deletion gaps in future high-level frontend support must be reported as
  missing rebuild architecture, not patched by restoring the core source pass.

### 4. Wrong vs Correct

Wrong:

```text
linalg/vector source body -> core RVV family lookup -> tcrv.exec.kernel
```

Correct:

```text
already materialized TianChen-RV execution surfaces
  -> plugin registry proposal/legality/selection/lowering-boundary path

future rebuild:
plugin/interface-owned frontend construction
  -> extension family ops / execution surfaces
  -> common EmitC route
```

## Scenario: RVV-Owned Bounded Vector I32 Add Source Pattern

### 1. Scope / Trigger

This scenario applies to the bounded RVV-owned source materializer that accepts
one structurally recognized MLIR vector/arithmetic i32 add source shape and
materializes the existing RVV i32m1 selected-boundary route. It is a
plugin-owned RVV entry point, not a restoration of the deleted core
linalg/vector RVV source frontend family.

### 2. Signatures

- Public pass option:
  `--tcrv-rvv-materialize-i32m1-vector-source-front-door`.
- The pass option names the bounded production source front door. It does not
  make a `seed` attribute, seed metadata, route id, or artifact name the route
  authority.
- No positive source marker is required for the RVV i32 add slice.
- Stale RVV source marker metadata such as
  `tcrv_rvv.lowering_seed = "i32m1_add"` must not create a selected RVV route
  and may be rejected before materialization.
- Required source function ABI:
  `func.func @name(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index)`.
- Required source body shape:
  one `scf.for` from constant index `0` to `%n` with fixed vector i32 chunk
  step, two `vector.load` operations from `%lhs` and `%rhs`, one `arith.addi`
  over `vector<4xi32>`, one `vector.store` to `%out`, and an empty return.

### 3. Contracts

- The pass may materialize exactly the bounded RVV add selected-boundary form:
  `tcrv.exec.kernel` -> `origin = "rvv-plugin"` variant -> explicit
  `tcrv_rvv.runtime_abi_value` operands for `lhs`, `rhs`, `out`, and `n` ->
  `tcrv_rvv.setvl` -> selected `tcrv_rvv.with_vl` -> RVV
  `i32_load` / `i32_add` / `i32_store`.
- The accepted source function ABI is positional for this first slice:
  source arguments 0, 1, 2, and 3 map to `lhs`, `rhs`, `out`, and `n`. The
  produced `tcrv_rvv.runtime_abi_value` ops must preserve that source-derived
  mapping in bounded provenance, using purpose strings
  `source-arg-0:lhs`, `source-arg-1:rhs`, `source-arg-2:out`, and
  `source-arg-3:n`. These purpose strings are provenance only; computation
  semantics still come from the accepted source body and the materialized
  `tcrv_rvv` extension-family ops.
- The produced variant must require `@rvv` and must keep computation semantics
  in `tcrv_rvv` extension-family ops. `tcrv.exec` remains the execution
  envelope and selection surface only.
- The route must be consumed by the existing RVV construction/EmitC/target path;
  it must not introduce descriptor-driven computation, direct C semantic
  export, Python compiler-core logic, or a common/core RVV semantic branch.
- The pass is source-only. Pre-existing `tcrv.exec` or `tcrv_rvv` operations in
  the input are stale selected-boundary or variant residue for this pass and
  must fail closed instead of being merged with source lowering.
- The bounded source body is the positive authority for this slice. Route ids,
  descriptors, artifact names, stale seed metadata, deleted family records, and
  common/core source scans must not authorize RVV materialization.

### 4. Validation & Error Matrix

- Missing `lhs`/`rhs`/`out`/`n` ABI operands -> fail before materialization.
- Non-`memref<?xi32>` buffers, non-`index` `n`, non-i32 vector arithmetic,
  wrong rank, or wrong vector chunk shape -> fail before materialization.
- Missing `scf.for`, extra source compute, wrong operation order, wrong load or
  store operands, or source store not using the `arith.addi` result -> fail.
- `scf.for` with loop-carried `iter_args`, yielded values, unsupported lower
  bound, unsupported upper bound, or unsupported step -> fail.
- Pre-existing `tcrv.exec`/`tcrv_rvv` operations in the same source input ->
  fail as stale selected-boundary or unselected variant residue.
- Stale `tcrv_rvv.lowering_seed` metadata alone -> fail or be ignored in a way
  that cannot create a selected route. It must not silently fall back to a
  generic RVV lowering path.

### 5. Good/Base/Bad Cases

- Good: an unmarked `func.func` with the exact i32 add vector source shape
  materializes a selected RVV i32m1 variant that the existing EmitC route
  accepts.
- Base: hand-written TianChen-RV MLIR with explicit RVV variant bodies remains
  a valid backend-first input and does not require this source materializer.
- Bad: a common transform scans arbitrary linalg/vector/add/sub/mul source
  bodies and chooses RVV semantics through route ids, descriptors, artifact
  names, stale seed metadata, or deleted finite-family metadata.

### 6. Tests Required

- Positive lit/FileCheck coverage for unseeded source pattern -> selected
  `tcrv.exec.variant` containing explicit runtime ABI bindings,
  source-argument provenance purpose strings, `tcrv_rvv.with_vl`, and
  `tcrv_rvv.i32_add`.
- Positive route-consumption coverage proving the materializer output reaches
  existing RVV emission-plan and EmitC materialization.
- Negative lit/FileCheck coverage for missing ABI operands, unsupported
  dtype/rank/shape, malformed source body, stale seed metadata, and stale
  pre-existing `tcrv.exec`/`tcrv_rvv` residue.
- Negative source-shape coverage must include wrong arithmetic op, wrong
  buffer role/use, missing or extra runtime `n`, unsupported loop bounds/step,
  loop-carried values, missing store, extra loop ops, and unrelated body.
- Existing explicit RVV i32m1 construction/EmitC/target tests must continue to
  pass so the source materializer is proven to feed the current route instead
  of replacing it.

### 7. Wrong vs Correct

Wrong:

```text
arbitrary vector/linalg source -> core RVV arithmetic recognizer
  -> route id or descriptor chooses RVV operation
```

Correct:

```text
exact bounded RVV-owned source shape
  -> RVV-owned materialization pass
  -> plugin-owned typed RVV selected-boundary body
  -> existing construction/EmitC/target route
```

## Pipeline

```text
High-level MLIR op
    |
    v
Collect target capabilities
    |
    v
Query extension plugins
    |
    v
Plugins propose execution variants
    |
    v
Capability and legality verification
    |
    v
Variant-local tuning and cost estimation
    |
    v
Select variant or generate dispatch
    |
    v
Materialize dispatch runtime guard parameters and case links
    |
    v
Lower selected variants through plugin emission paths
```

## Public Execution Planning Pipeline

The public `tcrv-opt` front door exposes the bounded metadata planning pipeline
as:

```text
--tcrv-execution-planning-pipeline
```

Public target artifact exporters do not invoke this planning pipeline
in-process. They consume already planned/materialized MLIR. Source-level
one-command bundle export is handled by plugin source-front-door registration
and the source artifact bundle front door, not by a generic translate-side
plan-and-export wrapper.

This pipeline is a named MLIR pass pipeline, not a monolithic pass. It composes
existing pass factories in this order:

```text
tcrv-materialize-plugin-variants
  -> tcrv-check-hart-parallel-capabilities
  -> tcrv-verify-plugin-variant-legality
  -> tcrv-select-variants
  -> tcrv-materialize-dispatch-runtime-guards
  -> tcrv-check-capability-requires
  -> tcrv-materialize-selected-lowering-boundaries
  -> tcrv-materialize-emission-plans
  -> tcrv-check-execution-plan-coherence
```

The pipeline consumes existing `tcrv.exec.kernel` anchors and each kernel's
explicit capability-provider scope: direct `tcrv.exec.capability` providers,
kernel-local capability-provider `tcrv.exec.target` anchors, and the one
module-level capability-provider `tcrv.exec.target` explicitly referenced by
`target = @profile` plus that target profile's explicit
`capability_providers` composition. It routes proposal/cost/lowering/emission-plan queries
through an injected `ExtensionPluginRegistry`, checks final selected artifact
route id, artifact kind, origin/emission identifiers, callbacks, and runtime
ABI parameter contracts through an injected `TargetArtifactExporterRegistry`,
and materializes only compiler-visible planning metadata, including
plugin-local selected-boundary ops only when a plugin has an active boundary
surface. RVV and scalar fallback currently remain no-boundary/fail-closed
routes until explicit extension-family IR plus a materialized EmitC route
exists. In `tcrv-opt`, the tool boundary may inject the
deterministic built-in plugin registry plus built-in target artifact exporter
registry; embeddable library builders must remain usable with explicitly
supplied registries and must not create hidden target-specific global state.
Compatibility builders that only receive plugins use an explicit empty target
artifact exporter registry, so supported artifact front-door validation remains
a clear fail-closed diagnostic until a populated exporter registry is supplied.

The hart-parallel capability check is target-neutral. It checks only
`tcrv.exec.hart_parallel` requests against generic capability id
`target.hart_count` and property `count`; plugin profiles may provide that id
from extension-local facts, but the pipeline must not gain concrete RVV/IME/
offload/scalar branches to interpret hart count.

The `tcrv-select-variants` stage is the capability-aware selection and dispatch
planning stage for this pipeline. The older order-based
`tcrv-synthesize-variant-dispatch` pass remains a separate bounded helper and
is not inserted before selection, because selection owns cost-aware dispatch or
selected-marker materialization and must not compete with a pre-existing
dispatch surface.
The generic `tcrv-check-capability-requires` gate runs after this selection
surface is materialized so conflict-aware planning can turn intended
runtime-dispatched conflicts into explicit guarded cases before the unchanged
legality gate rejects unprotected static variants or fallbacks.

The `tcrv-materialize-dispatch-runtime-guards` stage is the compiler-owned
runtime-control linkage stage for selected dispatch surfaces. It scans direct
`tcrv.exec.dispatch` operations under each kernel, identifies guarded
`tcrv.exec.case` paths from the typed generic
`runtime_guard_required = true` marker and validates generic capability
availability/conflict state fail-closed, materializes one direct same-kernel
`tcrv.exec.runtime_param` with ABI role `dispatch-availability-guard`, and
attaches `runtime_guard = @...` links to the selected cases that need that
runtime-control parameter. Non-empty `condition`, `guard`, or `policy` strings
may remain printable annotations, but they are not sufficient semantic evidence
for creating a runtime ABI guard. This stage is target-neutral: it must not branch on
RVV, scalar fallback, IME, offload, Sophgo, AME, vendor, dtype, shape,
microkernel semantics, runtime probes, or generated C. `tcrv.exec.fallback`
does not receive `runtime_guard_required` or `runtime_guard` case metadata.

The `tcrv-verify-plugin-variant-legality` stage is the materialized-variant
legality boundary before selection. It builds the same generic
`TargetCapabilitySet` from the kernel capability-provider scope and routes each
direct `tcrv.exec.variant` only to the plugin named by its `origin` attribute.
Plugin-local legality failures stop the pipeline before selection, dispatch,
selected lowering-boundary materialization, or emission-plan diagnostics are
created. This stage complements, and does not replace,
`tcrv-check-capability-requires`.

Emission readiness is not part of this public planning pipeline while the RVV
first slice reports unsupported readiness as a fatal boundary. Instead, the
pipeline materializes plugin-owned emission-plan diagnostics after selected
lowering-boundary materialization. Emission-plan materialization must validate
the selected plugin-owned boundary surface before producing diagnostics and
should record a generic `lowering_boundary` diagnostic metadata field naming
the boundary operation used by each selected plan. These diagnostics are
reproducibility metadata only: they do not lower IR, emit LLVM/RISC-V/RVV code,
create runtime ABI glue, generate artifacts, run hardware, prove correctness,
or measure performance.

After emission-plan materialization, the canonical pipeline runs the existing
execution-plan coherence gate over the same selected-path metadata. This final
gate validates selected-path, plugin origin, lowering-boundary, runtime ABI
ownership, emission-plan, and concrete target artifact route fields against the
active plugin and target artifact exporter registries before downstream
target/export front doors consume the planned IR. It reuses the shared
`tcrv-check-execution-plan-coherence` pass and must not duplicate target route
logic in the pipeline builder.

The pipeline is deterministic but not allowed to paper over stale or competing
selected surfaces. Re-running on IR that already contains a direct dispatch,
pre-existing lowering-boundary metadata, mismatched materialized variants, or
emission-plan diagnostics must produce the existing bounded diagnostics rather
than duplicating symbols or silently appending stale metadata.

## Public Source Artifact Front-Door Pipeline

Bounded plugin-owned source materialization has a separate explicit `tcrv-opt`
front door:

```text
--tcrv-source-artifact-front-door-pipeline
```

This pipeline is for plugin-owned source materializers that already
materialize a selected extension-family boundary. It is not a replacement for
the ordinary execution-planning pipeline and must not be invoked silently by
default planning/export commands.

The pipeline composes existing registered pass factories in this order:

```text
all enabled plugin-registered source front-door materialization passes
  -> tcrv-check-hart-parallel-capabilities
  -> tcrv-verify-plugin-variant-legality
  -> tcrv-check-capability-requires
  -> tcrv-materialize-emission-plans
  -> tcrv-check-execution-plan-coherence
```

The source front-door pass list comes from
`ExtensionPluginRegistry::collectSourceFrontDoorPasses` in deterministic
enabled plugin order. Common/tool code may collect, validate, register, and
sequence those pass factories, but it must not inspect plugin source marker names,
source operation shapes, arithmetic, template semantics, route ids, runtime ABI
names, intrinsic names, artifact kinds, or target-family details.

This pipeline intentionally does not run
`tcrv-materialize-plugin-variants`, `tcrv-select-variants`, or unconditional
selected lowering-boundary materialization. Current bounded source front-door
materializers produce selected `tcrv.exec.variant` surfaces and selected
extension-family boundary IR themselves. Re-running proposal/selection would
compete with that selected surface, and unconditionally materializing
boundaries would duplicate plugin-owned source boundaries such as RVV
`tcrv_rvv.with_vl`. Plugins without an active source materializer do not
contribute source boundaries through this front door.

The artifact front door stops at emission-plan/coherence-checked TianChen-RV
IR. Existing target translate routes, such as
`tcrv-translate --tcrv-export-target-artifact` or route-specific artifact
translations, remain responsible for consuming the selected emission plan and
materializing target artifacts. If a caller wants textual EmitC evidence, it
may explicitly append `--tcrv-materialize-emitc-lowerable-routes` after the
front-door pipeline; this is a separate lowering step and must not replace
target artifact route validation.

Disabled built-in plugins leave the source front-door pass list empty. Source
inputs must then fail closed in the generic gates rather than being lowered
through hidden global plugin state.

The pipeline must preserve the fail-closed behavior of each plugin-owned source
materializer: stale marker metadata, malformed source shapes, stale
pre-existing `tcrv.exec`/extension-boundary residue, and mixed incompatible
source inputs must not be merged into one source program or repaired by common
code.

## Plugin-Driven Proposal

For each high-level op, core asks enabled plugins:

```text
RVV plugin: can it generate an RVV variant?
IME plugin: can it generate an IME variant?
Offload plugin: can it generate an offload variant?
Scalar/default plugin: can it generate fallback?
Future plugin: can it generate another variant?
```

The first C++ interface slice may collect proposal metadata objects before
materializing textual IR. Each proposal must still be directly mappable to
`tcrv.exec.variant` and dispatch metadata: variant symbol/name, origin plugin,
required capability ids or symbol references, and optional generic
condition/guard/policy metadata. A proposal may also carry plugin-owned MLIR
attributes as a generic named-attribute bag, but those names must be
dialect-qualified/discardable and must not collide with required
`tcrv.exec.variant` attributes. Proposal collection is orchestration only; it
does not select variants, run tuning, lower extension ops, emit runtime glue, or
rewrite IR. Registry orchestration must still validate that any proposal
required capability ids or symbol references are non-empty, known to the request
capability set, and available before the proposal can proceed to later legality
or materialization slices.

## Generic Proposal Materialization

After proposal collection and generic capability validation, a core C++/MLIR
materialization helper may turn validated `VariantProposal` metadata into real
`tcrv.exec.variant` operations under the relevant `tcrv.exec.kernel`.

The materialization slice is intentionally bounded:

- it uses MLIR builders and ODS-generated `tcrv.exec` op APIs, not textual
  string rewriting;
- it preserves proposal order when creating variants;
- it maps required capability ids through the generic `TargetCapabilitySet`
  into `FlatSymbolRefAttr` references to `tcrv.exec.capability` symbols. The
  mapping is relation-aware: an exact capability id wins when present, otherwise
  an available capability whose structured `provides` or `implies` relation
  satisfies the requested id may be used as the required symbol;
- it preserves required capability symbol references after checking they resolve
  in the kernel capability scope;
- it attaches only generic metadata supported by the current `tcrv.exec` ODS
  contract, such as variant symbol/name, origin plugin, `requires`, and
  non-empty generic `condition`/`guard`/`policy` decision metadata;
- it preserves plugin-owned dialect-qualified named attributes from the proposal
  onto the materialized `tcrv.exec.variant` as opaque MLIR attributes, after
  generic validation and without interpreting target-family semantics;
- it diagnoses malformed input before mutating IR when practical, including
  missing kernel anchors, duplicate variant symbols, invalid variant names,
  empty origins, unresolved capability ids or symbols, duplicate plugin-owned
  attributes, malformed plugin-owned attribute names, and collisions with core
  variant attributes.

This helper does not select variants, build dispatch, run tuning, lower
extension families, emit runtime glue, or implement extension-specific
semantics. Core materialization must remain free of target-family branches.

The public compiler pass surface for this slice is
`--tcrv-materialize-plugin-variants`. It scans existing `tcrv.exec.kernel`
anchors, builds each kernel's generic `TargetCapabilitySet` from direct
kernel-local capability providers plus the explicitly referenced module-level
`tcrv.exec.target` profile named by `target = @profile` and that profile's
explicit `capability_providers` composition, routes proposal
collection through an injected `ExtensionPluginRegistry`, and materializes the
validated proposal set through the shared helper above. Public tools may inject
the deterministic built-in registry at the tool boundary; the default factory
uses an empty registry for negative/embedded tests and must not invent
core-owned variants.

Recoverable plugin-local proposal declines are diagnostics, not variants. The
pass must keep collecting later plugins after such a decline so explicitly
available scalar/offload/fallback plugins can still contribute coverage. If
collection finishes with no viable proposals, the pass must fail with a generic
no-viable-proposals diagnostic that preserves bounded decline reasons in
registration order. This must not soften fatal malformed proposals, malformed
core IR, duplicate symbols, rerun mismatches, or typed materialization errors.

The pass must be safely repeatable: if a proposed variant symbol already exists,
it may skip only an existing direct `tcrv.exec.variant` whose origin, requires,
generic decision metadata, fallback role, plugin-owned attributes, and empty
materialized body exactly match the current proposal. Mismatched symbols must be
diagnosed rather than overwritten or duplicated.

## Generic Dispatch Synthesis

After variants are already materialized, a bounded core C++/MLIR dispatch
synthesis helper may organize sibling `tcrv.exec.variant` operations into a
single `tcrv.exec.dispatch`. This slice consumes real MLIR ops and the generic
`TargetCapabilitySet`; it does not collect plugin proposals, materialize new
variants, tune, lower, emit, probe hardware, or interpret target-family
semantics.

The deterministic first-slice policy is:

- scan direct `tcrv.exec.variant` children of each `tcrv.exec.kernel` in IR
  order;
- leave kernels with an existing direct `tcrv.exec.dispatch` unchanged rather
  than creating competing dispatch ops;
- choose the first direct conservative-fallback variant whose `requires` are
  generically available and conflict-free as the `tcrv.exec.fallback` target;
- emit `tcrv.exec.case` entries for the remaining variants in original variant
  order;
- inherit any present generic `condition`, `guard`, or `policy` metadata from
  the target variant onto the generated `tcrv.exec.case`;
- attach a synthesized non-empty generic guard policy only when a case target has
  unavailable or conflicting required capabilities and no inherited generic
  decision metadata, without overwriting plugin-proposed metadata;
- attach typed `runtime_guard_required = true` when a case target has
  unavailable or conflicting required capabilities, so the downstream
  runtime-guard materializer and capability check consume a structured
  compiler-owned contract rather than printable strings;
- diagnose and leave IR unmodified when no direct variant is conflict-free and
  generically available as an executable fallback.

Dispatch synthesis must stay target-neutral. It must not branch on RVV, IME,
offload, scalar fallback, vendors, accelerators, or any other target family.

Example without IME:

```text
linalg.matmul
  -> @rvv variant
  -> @sophgo_offload variant
  -> @fallback variant
```

Example with IME available:

```text
linalg.matmul
  -> @rvv variant
  -> @ime variant
  -> @sophgo_offload variant
  -> @fallback variant
```

## Generic Materialized Variant Legality Routing

After variants are materialized and before cost/tuning/lowering/emission, the
core registry may orchestrate plugin-local legality verification for real
`tcrv.exec.variant` IR.

The first C++/MLIR legality routing contract is:

- build or receive the generic `TargetCapabilitySet` from the enclosing
  `tcrv.exec.kernel`;
- create a `VariantLegalityRequest` containing the materialized
  `tcrv.exec.variant`, its enclosing `tcrv.exec.kernel`, and that capability
  set;
- route the request only to the plugin named by the variant `origin` attribute;
- reject missing variants, missing kernels, missing or empty origins, unknown
  origin plugins, and disabled origin plugins with generic diagnostics;
- propagate plugin-local legality failures while wrapping them with plugin,
  variant, and kernel context;
- verify direct variants in kernel IR order when checking a whole kernel;
- leave proposal collection, materialization, dispatch synthesis, selection,
  tuning, lowering, and emission as separate pipeline responsibilities.

This routing layer complements `--tcrv-check-capability-requires`. The generic
capability pass continues to check availability and dispatch/fallback guarding;
plugin legality checks extension-owned body details through plugin interfaces.
Core legality orchestration must not hard-code RVV, IME, offload, scalar
fallback, vendor, dtype, shape, layout, or target-family semantics.

## Generic Materialized Variant Cost Routing

After variants are materialized and legality can be routed plugin-locally, the
core registry may also orchestrate plugin-local cost estimation for real
`tcrv.exec.variant` IR. This is an input to later selection, not selection
itself.

The first C++/MLIR cost routing contract is:

- build or receive the generic `TargetCapabilitySet` from the enclosing
  `tcrv.exec.kernel`;
- create a `VariantCostRequest` containing the materialized
  `tcrv.exec.variant`, its enclosing `tcrv.exec.kernel`, and that capability
  set;
- route the request only to the plugin named by the variant `origin` attribute;
- reject missing variants, missing kernels, variants not enclosed by the
  requested kernel, missing or empty origins, unknown origin plugins, and
  disabled origin plugins with generic diagnostics;
- propagate plugin-local cost failures while wrapping them with plugin,
  variant, and kernel context;
- reject invalid estimates, including missing scores, non-finite scores,
  negative scores, mismatched origin/variant identity, and present-but-empty
  generic explanation or policy text;
- collect direct kernel variant costs in original IR order without mutating IR;
- return deterministic rankings by explicit plugin-preference availability,
  ascending generic score, generic fallback role, original kernel IR order, and
  finally symbol name;
- leave proposal collection, materialization, legality, dispatch synthesis,
  full selection, tuning, lowering, emission, runtime ABI, and hardware
  probing as separate pipeline responsibilities.

The default plugin cost hook is deterministic and safe: it returns a neutral
finite score marked as no explicit preference for the request variant and
origin plugin. Concrete analytic, empirical, profile-guided, shape-aware,
dtype-aware, runtime-aware, or hardware-specific cost/preference semantics
belong inside plugins and later selection slices, not in core registry
orchestration.

Core cost orchestration must not hard-code RVV, IME, offload, scalar fallback,
vendor, dtype, shape, layout, runtime ABI, microarchitecture, or target-family
semantics.

## Generic Cost-Aware Selection Planning

### 1. Scope / Trigger

After materialized variant costs are ranked, a bounded core C++/MLIR selection
planner may turn direct `tcrv.exec.variant` children into an explicit compiler
decision plan. This planner consumes real MLIR ops, a `tcrv.exec.kernel`, the
generic `TargetCapabilitySet`, and `ExtensionPluginRegistry` ranking output. It
does not collect proposals, materialize variants, verify plugin-local legality,
lower extension families, emit runtime glue, run hardware probes, or implement
target-specific selection semantics.

### 2. Signatures

The durable C++ API shape is:

```cpp
enum class VariantSelectionKind {
  StaticVariant,
  RuntimeDispatch,
  FallbackOnly,
  NoViableVariant,
};

struct VariantSelectionCase {
  tcrv::exec::VariantOp variant;
  plugin::VariantCostEstimate cost;
  std::size_t originalIndex;
  bool genericallyAvailable;
  bool conflictFree;
  bool hasGenericDecisionMetadata;
  bool requiresRuntimeCapabilityGuard;
};

struct VariantSelectionPlan {
  VariantSelectionKind kind;
  tcrv::exec::KernelOp kernel;
  tcrv::exec::VariantOp selectedVariant;
  tcrv::exec::VariantOp fallback;
  bool missingFallbackCoverage;
  SmallVector<VariantSelectionCase> dispatchCases;
  SmallVector<VariantSelectionCase> rankedVariants;
};

Expected<VariantSelectionPlan> planKernelVariantSelection(
    tcrv::exec::KernelOp kernel,
    const TargetCapabilitySet &capabilities,
    const ExtensionPluginRegistry &registry);

Error materializeRuntimeDispatchPlan(OpBuilder &builder,
                                     const VariantSelectionPlan &plan,
                                     tcrv::exec::DispatchOp *createdDispatch);

Error materializeSelectedVariantMarker(
    OpBuilder &builder,
    const VariantSelectionPlan &plan,
    tcrv::exec::DiagnosticOp *createdMarker);
```

### 3. Contracts

The generic selection-planning contract is:

- collect only direct `tcrv.exec.variant` children of the request
  `tcrv.exec.kernel`;
- obtain score order through `ExtensionPluginRegistry::rankKernelVariantsByCost`
  and preserve its target-neutral tie-break order: explicit plugin preference
  availability, score, fallback role, original IR order, then symbol name;
- require every selected or dispatched variant to have structured generic
  `requires` metadata and origin-owned cost information;
- treat a variant as generically available only when all required capability
  symbols resolve in the supplied `TargetCapabilitySet` and are available;
- treat a generically available variant whose required capability conflicts
  with another available capability as not conflict-free for static selection
  or fallback selection;
- retain a conflicting available non-fallback variant as a runtime dispatch
  case only when a conflict-free conservative fallback exists and the
  materialized case carries typed `runtime_guard_required = true`; printable
  `condition`, `guard`, or `policy` metadata may be inherited from the variant
  or synthesized as a target-neutral annotation;
- reject unavailable variants unless selection can materialize an explicit
  dispatch case with typed `runtime_guard_required = true` and a distinct
  conflict-free conservative fallback, rather than silently selecting them;
- do not turn printable `condition`, `guard`, or `policy` metadata by itself
  into a runtime dispatch plan when the selected path is already
  conflict-free and generically available;
- choose the best selected variant by generic availability and cost ranking;
- materialize generic preference metadata on selected markers, dispatch cases,
  and fallbacks so diagnostics expose origin plugin, explicit-preference
  availability, preference score, rank, policy/explanation when present,
  fallback role when present, and the target-neutral tie-break reason;
- choose a `tcrv.exec.fallback` only from a generically available variant that a
  plugin marked with an abstract conservative fallback role in proposal,
  materialized metadata, or cost-estimate metadata, and whose required
  capabilities are conflict-free under the supplied `TargetCapabilitySet`;
- do not infer fallback coverage from arbitrary available variants, origin
  strings, capability IDs, target families, dtypes, shapes, or runtime identities;
- produce an explicit no-variant plan when a kernel has no direct variants;
- diagnose runtime-dispatch situations that have guarded or conflicting
  candidates but no plugin-provided conflict-free conservative fallback
  candidate;
- produce a static plan when the lowest ranked executable path is available and
  no lower-cost guarded candidate must be retained;
- produce a runtime dispatch plan when capability-guarded unavailable or
  conflicting candidates must be retained and a conflict-free generically
  available fallback exists.

### 4. Validation & Error Matrix

- missing kernel or missing kernel body -> return an `llvm::Error`;
- no direct variants -> return a `NoViableVariant` plan, not a crash;
- cost ranking failure from unknown origin, disabled plugin, plugin-local
  failure, or invalid estimate -> propagate the registry error with plugin,
  variant, and kernel context;
- selected/dispatched variant without structured `requires` metadata -> return
  an `llvm::Error`;
- unavailable or conflicting variants may be retained only when the planner can
  materialize an explicit dispatch case with typed
  `runtime_guard_required = true` and a plugin-provided conflict-free
  conservative fallback; when no printable decision metadata exists, selection
  may synthesize a target-neutral annotation such as
  `policy = "capability_dispatch_guard"`;
- runtime-dispatch plan without a plugin-provided conflict-free conservative
  fallback candidate -> return an `llvm::Error`;
- dispatch materialization for a non-runtime-dispatch plan -> return an
  `llvm::Error`;
- dispatch materialization when the kernel already contains a direct
  `tcrv.exec.dispatch` -> return an `llvm::Error` and leave IR unchanged;
- dispatch materialization with case/fallback variants not directly enclosed by
  the plan kernel -> return an `llvm::Error`.
- selected-path marker materialization for a non-`StaticVariant` /
  non-`FallbackOnly` plan -> return an `llvm::Error`;
- selected-path marker materialization when the kernel already contains a direct
  `tcrv.exec.dispatch` or a different direct selected marker -> return an
  `llvm::Error`.

### 5. Good / Base / Bad Cases

- Good: lower-cost guarded variants carry typed `runtime_guard_required = true`
  and may also carry printable generic metadata; conflicting variants may
  receive synthesized generic guard policy as annotation, a conflict-free
  generically available fallback exists, and materialization creates ordered
  `tcrv.exec.case` ops plus one `tcrv.exec.fallback`.
- Base: all candidates are generically available and unguarded; the planner
  chooses the lowest-cost static variant and preserves stable equal-score order.
- Bad: an unavailable or conflicting variant has no structured runtime
  capability guard route and no conflict-free conservative fallback; the planner
  rejects it instead of silently selecting it or inventing a fallback.

### 6. Tests Required

- C++ tests must parse or build real MLIR modules with `tcrv.exec.kernel`,
  `tcrv.exec.capability`, and materialized `tcrv.exec.variant` ops.
- Mock plugins must provide costs through the real
  `ExtensionPluginRegistry::rankKernelVariantsByCost` path.
- Positive tests must cover static lowest-cost selection, equal-cost stable
  ties, guarded runtime dispatch, fallback-by-cost rather than IR order, typed
  dispatch materialization, metadata copying, and verifier/capability-check
  acceptance.
- Negative tests must cover no direct variants, no available fallback,
  unavailable unguarded variants, existing direct dispatch rejection, ranking
  failure propagation, missing kernels, and cross-kernel case/fallback variants.

### 7. Wrong vs Correct

Wrong:

```cpp
if (target.hasRVV()) choose(rvvVariant);
else if (target.hasIME()) choose(imeVariant);
```

Correct:

```cpp
Expected<VariantSelectionPlan> plan =
    planKernelVariantSelection(kernel, capabilities, registry);
```

A selection materialization helper may turn a runtime dispatch plan into typed
`tcrv.exec.dispatch`, `tcrv.exec.case`, and `tcrv.exec.fallback` operations. The
helper preserves planned case order, copies non-empty generic
`condition`/`guard`/`policy` metadata verbatim, rejects kernels that already
contain a direct dispatch, and does not erase variants. It must not parse,
switch on, or infer target-family meaning from generic metadata strings.

Static and fallback-only selection plans may instead materialize one direct
`tcrv.exec.diagnostic` selected-path marker:

```mlir
tcrv.exec.diagnostic {
  reason = "variant-selected",
  message = "static variant selected by generic cost and capability planning",
  severity = "note",
  status = "selected",
  target = @selected_variant,
  selection_kind = "static-variant"
}
```

The marker is generic control metadata. `target` references the selected direct
variant, and `selection_kind` is `static-variant` or `fallback-only` in the
first slice. It is not lowering IR, runtime ABI glue, or target-family logic.
Re-running selection must reuse an equivalent direct marker rather than
duplicating it.

If a static selected variant has no plugin-provided conservative fallback
candidate, the pass must also materialize a structured diagnostic with
`reason = "fallback-coverage-missing"`,
`selection_kind = "missing-conservative-fallback"`, and `status = "missing"`.
This diagnostic is the explicit no-fallback contract; it is not a failed
lowering/runtime/correctness/performance claim.

Selection planning must stay target-neutral. It must not branch on RVV, IME,
offload, scalar fallback, vendors, accelerators, dtype, shape, layout, runtime
ABI, microarchitecture, or any other target family. Preference metadata is
heuristic compiler ordering input only; it is not performance truth and cannot
weaken capability or plugin-legality failures.

### 8. Pass Integration and Registry Injection

The generic selection planner may be exposed through an MLIR module pass, but the
pass boundary must preserve plugin-owned cost semantics:

- `createSelectVariantsPass(const ExtensionPluginRegistry &registry)` is the
  production construction path. Tooling or future plugin loaders inject the
  populated registry before pass execution.
- `createSelectVariantsPass()` owns only an empty registry when used directly.
  Public tools such as `tcrv-opt` should register registry-dependent passes with
  a deterministic tool-owned registry populated by the built-in plugin helper
  set. Unknown `origin` plugins must still diagnose generically instead of
  inventing core-side attribute costs, target-family fallbacks, or Python-only
  ranking. Empty-registry public-tool coverage should use
  `--tcrv-disable-builtin-plugins`.
- The pass must build `TargetCapabilitySet` from each `tcrv.exec.kernel`, call
  the existing selection planner, materialize runtime-dispatch plans with typed
  `tcrv.exec.dispatch`, `tcrv.exec.case`, and `tcrv.exec.fallback`, and
  materialize static/fallback-only plans with one generic selected-path
  `tcrv.exec.diagnostic` marker plus a missing-fallback diagnostic when the plan
  intentionally has no conservative fallback candidate.
- Static, fallback-only, and no-direct-variant plans do not erase variants,
  lower extension families, or inject target-specific IR.
- Tests should cover injected-registry pass execution, public `tcrv-opt`
  built-in plugin routing, and generic diagnostics for missing origin plugins.

After selected-path planning and before executable lowering, a bounded
materialization step may collect plugin-owned emission plans and emit
`tcrv.exec.diagnostic {reason = "emission_plan"}` metadata. This step must keep
the selected-path order deterministic: dispatch cases in order, fallback after
cases, a single selected-path marker target when present, or all direct variants
in IR order for conservative mode. It must not reinterpret target families or
turn emission-plan metadata into runtime/correctness/performance evidence.

## Variant IR Required Fields

Each variant must include:

```text
variant name
origin plugin
required capabilities
shape/dtype/layout preconditions
extension family ops
cost/tuning metadata
emission path
fallback or dispatch relation
```

Reference shape:

```mlir
tcrv.exec.variant @rvv
  requires = #tcrv.requires<["rvv", "zvfh"]>
  origin = "rvv-plugin"
  tuning = #tcrv.tuning<...>
  cost = #tcrv.cost<...> {
  ... tcrv.rvv ops ...
}
```

## Legality

### Core legality

Core checks:

- variant declares `requires`;
- `requires` is satisfied or guarded by runtime dispatch;
- variant has origin plugin;
- body does not contain unknown/unregistered extension op;
- dispatch/fallback is complete.

### Plugin legality

Plugins check extension-specific details:

- RVV: SEW/LMUL/VL/mask/tail/memory/toolchain legality;
- IME: VLEN-backed fragment shape, dtype, accumulator, layout, toolchain;
- offload: runtime ABI, transfer, shape, sync, fallback;
- future plugin: its own extension rules.

Core verifier orchestrates plugin verifiers. It must not hard-code extension internals.

## Selection

Selection chooses:

```text
single static variant
runtime dispatch set
fallback-only path
```

Inputs:

```text
target capability
shape/dtype/layout
cost model
profile data
runtime availability
tuning result
user policy: performance / portability / debug / no-offload
```

## Runtime Dispatch

Dispatch conditions come from capability object, runtime probe, shape guard, dtype guard, cost threshold, or user policy.

Reference:

```mlir
tcrv.exec.dispatch {
  tcrv.exec.case @sophgo_offload {
    condition = "runtime_available",
    guard = "large_shape",
    policy = "prefer_offload"
  }
  tcrv.exec.case @ime {condition = "extension_capability_available"}
  tcrv.exec.case @rvv {condition = "vector_capability_available"}
  tcrv.exec.fallback @scalar_or_default
}
```

The structured `tcrv.exec.case` form is a compiler-visible reference to a
declared variant. The core verifier checks dispatch/fallback completeness and
symbol resolution only; it does not implement selection, cost modeling, tuning,
or extension-specific condition semantics.

Capability legality may treat a `tcrv.exec.case` as a runtime guard for an
unavailable required capability only when the case carries typed generic
`runtime_guard_required = true`. The core pass does not parse printable
`condition`, `guard`, or `policy` strings and must not switch on target families. A
`tcrv.exec.fallback` target is different: it must be generically available under
the same target capability set, so runtime dispatch always has an executable
fallback path when guarded cases are unavailable.

## Emission Readiness After Selection

After materialization, legality, cost ranking, selection, and dispatch
synthesis, the compiler may run a generic emission-readiness check before final
lowering. This pass does not create lowering IR or runtime glue. It verifies
that the current execution path can be routed back to the origin plugin that
owns each selected materialized variant.

First-slice behavior:

- kernels with a direct `tcrv.exec.dispatch` check every `tcrv.exec.case`
  target and the `tcrv.exec.fallback` target;
- kernels without a dispatch but with one direct selected-path
  `tcrv.exec.diagnostic` marker check only that marker target;
- kernels without a dispatch or selected-marker surface conservatively check all
  direct `tcrv.exec.variant` children;
- selected-path marker targets must resolve to direct sibling variants in the
  same kernel before plugin routing, and duplicate direct selected markers are
  invalid;
- dispatch and fallback targets must resolve to direct sibling variants in the
  same kernel before plugin routing;
- missing, duplicate, non-variant, or non-sibling dispatch references are
  diagnosed generically before plugin emission-readiness hooks are called;
- routing is by the generic variant `origin` attribute through
  `ExtensionPluginRegistry`, with no target-family branches in core code.

The same selected-path traversal may be reused to collect plugin-owned emission
plans after readiness. Emission-plan collection must keep dispatch cases and
fallbacks in dispatch body order, consume a single selected-path marker before
falling back to conservative all-direct-variant traversal, and diagnose the
same malformed selected-path structure before invoking plugin hooks. For
dispatch or selected-marker surfaces that have passed through selected
lowering-boundary materialization, collection must first validate that each
selected reference has exactly one matching plugin-owned boundary. Missing,
stale, duplicate, origin-mismatched, selected-variant-mismatched, or
required-capability-mismatched boundaries are deterministic failures and must
not result in appended emission-plan diagnostics. Each collected plan is a
structured compiler decision object for the selected path: it records origin
plugin, kernel symbol, variant symbol, selected-path role, support status, the
validated lowering-boundary operation when present, and either supported route
metadata or an unsupported diagnostic. It does not lower IR, generate
executable code, prove runtime correctness, or justify performance claims.

## Parameter Flow Boundary

Variant generation, legality, selection, tuning, and lowering-boundary
materialization must preserve the project parameter layering rule:

- hardware facts / target capability such as VLEN, raw VLENB bytes, ISA
  facts, hart count, toolchain availability, probe evidence, and capability
  provenance constrain proposal, legality, cost, and selection;
- compile-time variant config such as SEW, LMUL, tail/mask policy, unroll, and
  selected lowering strategy is plugin-proposed metadata that must be checked
  against those capabilities;
- runtime SSA values / runtime control values such as AVL, vl, pointer
  arguments, length `n`, `rvv_available`, and dispatch guards must be modeled
  as real SSA, region/block arguments, explicit ABI/control attributes, or
  generated C ABI parameters before any pass describes them as modeled;
- deleted legacy RVV local element-count markers identify historical or
  fail-closed residue only. They are not tensor shape, global problem size,
  AVL, vl, source authority, or production route input.

Selection and lowering-boundary metadata may name the parameter layer they
consume, but must not promote a descriptor-local or runtime value into target
capability, nor freeze hardware facts as per-variant runtime data.

## Capability-Aware Tuning

Tuning has two layers.

Variant selection tuning:

```text
choose RVV vs IME vs offload vs fallback
```

Variant-local tuning:

```text
RVV: LMUL, SEW, VL policy, unroll, packing, thread partition
IME: fragment shape, K-block, accumulator policy, packing
Offload: transfer threshold, batch size, async overlap, buffer reuse
```

Tuning must remain part of selection/variant quality. Do not present ordinary parameter search as the main theory.

## Cost Model Boundary

Cost model may be analytic, empirical, profile-guided, or hybrid. It must explain:

- why RVV is selected;
- why IME is selected;
- why offload is selected;
- why dispatch is retained;
- why fallback is used.

## Diagnostics

Structured diagnostics should cover:

- no plugin supports the high-level op;
- plugin supports op but capability is absent;
- plugin proposes a variant with an empty, unknown, or unavailable required
  capability id or symbol reference;
- plugin generated variant but legality failed;
- variant legal but emission path missing;
- multiple variants have close cost and dispatch is retained;
- offload runtime unavailable and RVV fallback selected.
