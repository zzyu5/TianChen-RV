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

Descriptor-driven C string export is not the architecture. C compute-body
printers that synthesize source from selected metadata, family records, or
route records are removed or fail closed until a real materialized MLIR EmitC
module route exists. Existing descriptor-backed source/object/bundle helpers
are historical residue, deletion targets, or fail-closed implementation debt.
They must not be used as transition architecture, production input, evidence
authority, or the template for new extension work.

## Scenario: Metadata-Driven C Exporter Erasure

### 1. Scope / Trigger

This applies when RVV, scalar fallback, or RVV+scalar dispatch selected paths
would historically have emitted runtime-callable C source, C headers,
relocatable objects, self-check sources, or artifact bundles from target-owned
records rather than from a materialized MLIR EmitC module.

### 2. Signatures

- Legacy translate options for standalone smoke-probe, microkernel,
  self-check, scalar fallback, and RVV+scalar dispatch C outputs are not
  current target artifact route identifiers.
- Legacy production route ids for generated source, header, object,
  self-check, and bundle-derived C outputs are not current target artifact
  route identifiers.
- There is no standalone source-output exception in the built-in target
  artifact exporter set. Future source output requires a real
  extension-family IR plus materialized MLIR EmitC module route.

### 3. Contracts

- Plugin emission plans for source-like outputs without a materialized EmitC
  route must return `status = "unsupported"`, `runtime_abi_kind =
  "unsupported-plugin-runtime-abi"`, `runtime_abi_name =
  "unsupported-emission-runtime-abi"`, and `runtime_glue_role =
  "no-runtime-glue-unsupported"`.
- Built-in target artifact exporter registration must publish only currently
  supported production routes. It must not use historical RVV/scalar/dispatch
  source, header, object, self-check, or bundle identifiers as current route
  authority.
- Public target source/header/object front doors must fail closed when no
  supported emission-plan route remains.

### 4. Validation & Error Matrix

- Removed direct translate option is invoked -> command-line parser reports the
  unknown option and no C source is printed.
- Generic source/header/object front door sees only unsupported plans ->
  reports no supported route and prints no C body, header, object, or bundle
  index.
- Plugin-selected path reaches emission planning without a materialized EmitC
  route -> emits a generic unsupported diagnostic for the missing current
  route, not supported artifact metadata.

### 5. Good/Base/Bad Cases

- Good: tests assert unsupported diagnostics and generic fail-closed registry
  behavior without treating old route names as current API.
- Base: stale standalone RVV smoke-probe route fixtures must not remain as
  active compiler inputs and must not emit C.
- Bad: selected metadata, family records, route records, or descriptor mirrors
  synthesize raw C loops, RVV intrinsics, scalar arithmetic bodies, dispatcher
  branches, headers, objects, or bundles.

### 6. Tests Required

- C++ registry tests proving the built-in target artifact route set contains
  only currently supported routes and preserves generic duplicate/invalid
  registration diagnostics.
- lit tests proving removed direct options and generic front doors fail closed
  without emitted C text.
- Full `check-tianchenrv` must not rely on stale positive direct-C source,
  header, object, bundle, or e2e dry-run fixtures.

### 7. Wrong vs Correct

Wrong:

```text
selected route metadata -> raw C body/header/object/bundle
```

Correct:

```text
selected route metadata -> unsupported missing-materialized-EmitC diagnostic
future rebuild -> materialized MLIR EmitC module -> C/C++ emitter
```

## Scenario: Common EmitC Source-Authority Exporter Deleted

### 1. Scope / Trigger

This applies to common Conversion/EmitC APIs that previously used
`TCRVEmitCLowerableRoute` metadata to build C++ source, wrapper functions,
runtime-control loops, or public source text from shared conversion code.

### 2. Signatures

Deleted public common APIs:

- `TCRVEmitCSourceAuthorityOptions`;
- `materializeTCRVEmitCLowerableRouteSourceAuthority`;
- `emitTCRVEmitCLowerableRouteAsCppSource`;
- `TCRVLowerToEmitCSourceOptions`;
- `TCRVLowerToEmitCSourceResult`;
- `lowerTCRVEmitCLowerableToEmitCSource`.

Deleted marker:

```text
tcrv_emitc.source_authority=mlir_emitc_cpp_emitter
```

Still-allowed common APIs may verify route shape and materialize an in-memory
MLIR `emitc` module, such as `materializeTCRVEmitCLowerableRoute` and
`verifyTCRVEmitCLowerableRouteMaterializesToEmitC`.

### 3. Contracts

- Common route metadata may validate route structure and materialize in-memory
  MLIR EmitC IR only.
- Common Conversion/EmitC code must not translate route metadata into C/C++
  source text, public wrapper function text, runtime loops, dispatch source, or
  extension intrinsic source.
- Common tests must not make RVV intrinsic names, `tcrv_rvv.*` op names,
  runtime-avl-to-vl loops, scalar-loop source output, or public wrapper C text
  the authority for shared conversion correctness.
- If a user-facing source bridge is needed after this deletion, that is a
  future rebuild gap requiring an explicit extension-family IR plus materialized
  EmitC route contract, not a reason to restore common source-authority APIs.

### 4. Validation & Error Matrix

- Code references a deleted common source-authority symbol -> compile failure;
  remove the caller or rebuild the feature under a new explicit contract.
- Common tests assert C/C++ source output from route metadata -> invalid test;
  rewrite to extension-agnostic in-memory EmitC materialization or delete.
- Common tests assert RVV intrinsic names or `tcrv_rvv.*` names as shared
  conversion authority -> invalid test; move such coverage to plugin-local
  syntax/metadata tests if still relevant.
- In-memory route materialization sees malformed route metadata -> emit generic
  materializer diagnostics and no C/C++ source text.

### 5. Good/Base/Bad Cases

- Good: generic route metadata -> in-memory MLIR `emitc.func` plus
  `emitc.call_opaque` materialization for verification.
- Base: route provenance comments in in-memory EmitC IR may record generic
  source-op/interface provenance without claiming source-export authority.
- Bad: route metadata -> C++ source string, public wrapper, RVV intrinsic source
  body, runtime-avl-to-vl loop source, scalar-loop source, or dispatch source.

### 6. Tests Required

- Focused C++ common test proving extension-agnostic route construction and
  in-memory EmitC materialization still work.
- Ref-scan proving deleted source-authority symbols and marker strings are not
  present under common Conversion/EmitC code and tests.
- Focused common EmitC build/test target plus `git diff --check`.

### 7. Wrong vs Correct

Wrong:

```text
TCRVEmitCLowerableRoute metadata
  -> common C++ source emitter
  -> public wrapper / intrinsic / runtime-loop source authority
```

Correct:

```text
TCRVEmitCLowerableRoute metadata
  -> generic verification
  -> in-memory MLIR EmitC materialization only
future rebuild:
explicit extension-family ops -> materialized MLIR EmitC route -> source contract
```

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
  origin plugin, runtime ABI ownership metadata, and diagnostic/explanation
  metadata;
- carry bounded runtime ABI kind/name metadata and a bounded required runtime
  glue role chosen by the origin plugin;
- reject `runtime-callable-c-source` as the artifact kind for every plugin
  emission plan. Structured runtime ABI parameter
  metadata must not make a direct C source artifact plan legal. Future source
  output requires a materialized MLIR EmitC module route with a new explicit
  source artifact contract; until then source-like mentions may appear only in
  unsupported missing-materialized-EmitC diagnostics or tests proving generic
  fail-closed behavior;
- for supported paths, carry required capability symbol refs that are a safe
  subset of the selected variant `requires` metadata;
- for supported paths, require non-empty emission kind, lowering pipeline
  identifier, runtime ABI identifier, artifact kind, and explanation;
- for unsupported paths, require a non-empty diagnostic reason;
- reject plugin-returned empty runtime ABI kind/name, empty runtime glue role,
  missing required capability refs on supported plans, or
  unbounded/secret-like diagnostic and explanation text before materialization;
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
  missing emission-plan diagnostics, unsupported plans, unknown
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
dtype, shape, runtime, toolchain, or microarchitecture semantics. The deleted
RVV standalone smoke-probe C exporter, runtime-callable RVV microkernel, scalar
microkernel, and RVV+scalar dispatch C source/object/header composite routes
must fail closed or be absent and must not synthesize compute bodies until the
rebuild supplies a materialized MLIR EmitC module route. This does not add
generic RVV or scalar lowering, full runtime ABI integration, object generation,
linking, arbitrary source export, correctness evidence, or performance
evidence.

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
`--tcrv-export-target-artifact` choose the header instead of a matching
library-object route. The generic source-only front door is deleted; source
artifacts are not generic-front-door selectable until a future materialized
EmitC route rebuild defines a new contract. The object and header routes are
still bounded target artifacts; they do not link, run hardware, perform
automatic probing, prove correctness, or measure performance.
Historical direct RVV microkernel library-object provenance sections are
deleted as active route authority. Future object provenance must be derived
from explicit extension-family IR, the materialized EmitC/runtime route, and
the ordered runtime ABI role contract rather than selected config descriptors
or runtime-length metadata helpers. Such provenance must not contain runtime
logs, hardware success text, artifact paths, credentials, correctness claims,
or performance claims.
The self-check object route remains an explicit target-owned helper command for
evidence collection, not the generic artifact front door.
For historical direct RVV i32/i64 add/sub/mul microkernel paths, the
runtime-callable C header/object route is deleted as production authority.
Selected RVV metadata, microkernel op names, or descriptor-shaped mirrors are
not authority for family, dtype, arithmetic op, intrinsic config, callable ABI,
route id, artifact kind, component group, function stem, or generated C body.
Target export must fail closed when a selected path lacks future explicit
extension-family ops plus a materialized EmitC route before
source/header/object/bundle output. The header must not embed
callable bodies, RVV intrinsics, `main`,
self-check helpers, runtime probing, evidence logs, credentials, artifact
paths, or performance text.

### Scenario: Deleted RVV Metadata-Only Default Artifacts

#### 1. Scope / Trigger

This scenario is historical. Generic source/header/object export must not
consume deleted RVV metadata fields as a supported runtime-callable artifact
route after direct C semantic exporter deletion.

#### 2. Signatures

- Deleted selected-boundary surface: the former RVV plugin-local selected
  lowering-boundary op.
- Deleted historical route families: RVV direct microkernel source, header,
  object, self-check, and bundle-style artifact routes.

#### 3. Contracts

- The former i32-vadd default artifact route is deleted as active source
  authority. Generic target artifact candidate preflight must not accept
  selected-boundary metadata, descriptor mirrors, or selected-plan metadata as
  sufficient to emit RVV source/header/object artifacts.
- Descriptor-selected route fields such as frontend lowering or default
  typed-body materialization are fail-closed historical inputs until an
  explicit extension-family IR plus materialized MLIR EmitC module route
  exists.
- Descriptor fields, route ids, intrinsic names, and selected
  microkernel-op names must not be validated as production compute authority.

#### 4. Validation & Error Matrix

- Any descriptor-selected RVV microkernel candidate -> fail before source,
  header, object, or bundle output.
- Selected-boundary route metadata, descriptor-only i32-vadd candidates, or
  hand-authored microkernel attachments -> fail as deleted-route inputs rather
  than re-enable artifact output.

#### 5. Good/Base/Bad Cases

- Good: RVV selected-path handling produces no plugin-local boundary, no
  emission plan, and fail-closed diagnostics without a materialized
  `tcrv_rvv.*_microkernel` body or callable ABI.
- Base: direct hand-authored RVV microkernel fixtures may remain parseable only
  for syntax or fail-closed negative coverage.
- Bad: an emission plan with typed selected-plan metadata, route identity, or
  descriptor mirrors emits RVV source/header/object artifacts.

#### 6. Tests Required

- Positive selected-boundary coverage proving no descriptor-selected
  microkernel body is materialized.
- Negative FileCheck/C++ coverage proving descriptor-selected RVV source,
  header, object, and bundle paths fail before generated bytes appear.
- Negative coverage for stale/wrong route identity, explicit microkernel
  attachments, and descriptor-derived ABI metadata as deleted-route inputs.

#### 7. Wrong vs Correct

Wrong:

```text
selected emission plan metadata + descriptor mirror -> emit i32-vadd artifact
```

Correct:

```text
selected RVV metadata boundary
  -> unsupported missing-materialized-EmitC diagnostic
future rebuild:
explicit extension-family ops -> materialized MLIR EmitC module -> artifact
```

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
purposes. This keeps `--tcrv-export-target-artifact` deterministic for
RVV/offload primary paths while preserving the scalar fallback candidate for
the specialized host dispatch exporter. The deleted generic source-only front
door must remain absent. A scalar-only selected fallback remains exportable
through the generic default route when it is the only supported non-source
candidate. If multiple non-fallback candidates remain supported for one generic
export request, the request is still ambiguous and must fail closed.

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
  against capabilities. RVV paths must derive executable intrinsic spelling and
  VL setup from explicit `tcrv_rvv` IR plus a materialized EmitC/runtime route,
  not from target-owned selected-shape descriptors, suffix tables, or comment
  metadata;
- runtime SSA values / runtime control values such as AVL, vl, pointer
  arguments, length `n`, and dispatch guards may be emitted only as real
  IR/control fields or generated ABI parameters;
- generated ABI parameters must state whether they are actually IR-modeled or
  target/export ABI-owned. The current bounded i32 binary RVV and scalar source
  exports pass `lhs`, `rhs`, `out`, and runtime `n` as C ABI parameters, but the
  callable parameter plan must be built from real `tcrv.exec.mem_window` IR for
  lhs/rhs/out buffer meanings and real direct `tcrv.exec.runtime_param` IR for
  runtime-element-count. Candidate/emission-plan parameter metadata may only
  mirror that IR-backed plan, while runtime ABI strings and glue roles come
  from plugin/target-owned route construction. Runtime `n` remains a runtime
  ABI/control value, not descriptor-local element count;
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
  the executable branch-control source. The guard C name must be explicit and
  caller-owned; changing it must not change callable role order, add the guard
  to callable microkernel signatures, or introduce automatic hardware probing;
- legacy bounded values such as `tcrv_rvv.element_count` or the deleted scalar
  element-count marker describe historical selected-path metadata at most. An
  extension plugin may choose bounded diagnostic sample sizes from validated
  structured capability facts, such as RVV i32 M1 lane capacity, but those
  values still must not be reported as tensor shape, global problem size, AVL,
  vl, runtime loop trip count, source authority, correctness coverage, or
  performance evidence. The current scalar fallback plugin does not consume its
  deleted element-count marker as active legality or boundary authority.

Generated C may contain target-owned local variables such as a local `vl`
computed by RVV intrinsics or ABI parameters such as `n` and an explicit
dispatch guard C name. That does not imply those values were modeled in MLIR
unless the input IR has the corresponding attribute, type, SSA value, region
argument, or ABI/control surface. The current bounded `tcrv_rvv.setvl` and
`tcrv_rvv.with_vl` surfaces model only runtime AVL/VL control-plane IR when
they appear in the input. They are not emitted runtime ABI evidence by
themselves.

## Scenario: Deleted Dynamic Vector Source Front Door

### 1. Scope / Trigger

The former bounded dynamic vector/SCF i32 add/sub/mul source front doors are
deleted with the core RVV source-to-exec pass family. Core code must not parse
vector transfer/SCF source shapes, inspect source arithmetic, or query the RVV
binary family registry to materialize `tcrv.exec`.

### 2. Signatures

The deleted public option family includes the historical generic source adapter
and vector i32 arithmetic adapters. Active specs and tests must not preserve
those option spellings as durable route contracts.

### 3. Contracts

- Invoking a deleted vector source pass must fail closed as an absent/deleted
  option and must not create a `tcrv.exec.kernel`.
- `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` must not run a
  vector or linalg source-frontdoor pass before planning.
- Existing source-tail metadata may remain as bounded metadata on already
  materialized execution/plugin surfaces where current target-owned validation
  consumes it, but the core transform layer must not produce it from high-level
  vector source bodies in this deleted route.
- Future high-level vector frontend reconstruction must be plugin/interface
  owned and must not restore this core RVV semantic branch.

### 4. Tests Required

- Current lit/FileCheck coverage should prove fail-closed behavior through the
  active plugin/interface-owned pipeline surfaces, not by preserving historical
  vector/source option spellings as named absence fixtures.
- Delete tests whose only purpose was to validate the old vector source wrapper
  shapes, tail-in-bounds diagnostics, family marker cross-checks, or
  compatibility alias behavior.
- Fresh `ssh rvv` evidence for runtime counts 7, 16, and 23 when the generated
  artifact contract changes.

### 7. Wrong vs Correct

Wrong:

```text
dynamic %n loop + transfer_read/write in_bounds=true
  -> rely on downstream setvl(n) to imply source tail correctness
```

Correct future rebuild:

```text
dynamic %n loop + MLIR transfer tail semantics
  -> plugin/interface-owned frontend construction
  -> tcrv.exec runtime_param @abi_runtime_element_count
  -> selected-plan active-lane/source-tail metadata
  -> tcrv_rvv.setvl(n) and generated artifact comments
```

### Execution-Plan / Export Preflight Coherence

Before a module is handed to a generic target artifact export route, the public
`tcrv-translate --tcrv-export-target-artifact` and
`tcrv-translate --tcrv-export-target-header-artifact` front doors must run the
target-neutral preflight verifier to check that all compiler-visible handoff
metadata still describes the same selected execution path. The deleted generic
source-only front door must fail at command-line registration instead of
running export preflight. This check is a metadata coherence gate only. It must
not export artifacts, lower to LLVM/RISC-V, emit extension instructions, create
runtime ABI glue, run hardware, or claim correctness or performance.

The verifier must fail closed when selected-path, dispatch/fallback,
lowering-boundary, runtime ABI ownership, emission-plan, and concrete artifact
route fields are stale or contradictory. Required failure cases include selected
variant references that no longer resolve to the current direct variant, origin
plugins that are missing or unregistered, dispatch or selected-marker origin
mismatch, lowering-boundary source kernel / selected variant / origin mismatch,
emission-plan target / role / origin / lowering-boundary mismatch, missing
runtime ABI ownership fields for plans that require them, unsupported artifact
route kind or emission kind, unknown target artifact route id, and multiple
ambiguous supported artifact candidates.

The generic preflight verifier may use the existing `ExtensionPluginRegistry`
to validate origin ownership and the generic target artifact exporter registry
to validate concrete route id, artifact kind, origin/emission identifiers,
export callbacks, and runtime ABI parameter contracts. Target-specific proof
of a concrete microkernel, descriptor body, toolchain, or runtime remains
target-owned and must not move into the shared transform. Extension bundles
must not require standalone or composite exporters to publish selected-plan
metadata descriptors as target artifact route authority.
When a registered target artifact route declares required runtime ABI roles or
a route-local ABI validation callback, this same preflight verifier must reject
missing or inconsistent compiler-owned ABI contracts before source, header,
object, or bundle materialization. RVV and scalar callable routes use this
boundary for their compiler-owned
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
  by delegating to target-owned registration functions or, for route groups
  with an extension plugin manifest hook, by asking that plugin to configure
  its target-support `ExtensionBundle` contribution.
- The current non-plugin single-candidate route set is empty until a
  materialized source route is rebuilt.
- The current plugin-owned executable runtime-callable C route set is empty
  until a materialized source route is rebuilt. RVV selected microkernel
  source/header/object routes, scalar selected fallback source/header/object
  routes, and RVV+scalar dispatch source/header/object routes must not be
  registered as supported target artifact routes.
- The scalar plugin may still contribute its extension bundle so selected
  scalar fallback intent and dialect ownership remain visible, but its
  target-artifact exporter bundle must not
  publish `tcrv-export-scalar-*` source/header/object routes or
  `scalar-*-runtime-callable-c-abi.v1` identities as executable artifact
  authority.
- The current plugin-owned supported target artifact route set is empty until a
  future Common EmitC rebuild materializes explicit extension-family routes.
  Template, Toy, TensorExtLite, and Offload metadata artifact exporters must not
  publish route ids or stand in for compiler-owned lowering.
- The helper may include scalar/offload/Toy/template target headers for
  current legacy bundle composition, but target-support-enabled extensions must
  activate any supported route contribution through the plugin/manifest hook.
  In particular, central built-in code must not include the RVV target-support
  bundle header, directly call the RVV target-support helper, iterate RVV
  direct/RVV+scalar dispatch manifests as central route truth, or keep scalar
  route strings as compatibility authority.
- Generic public translate helpers should call this helper once and then call
  `exportTargetArtifact` or `exportTargetHeaderArtifact`. The former
  source-specific export API is deleted with the source-only front door.
- Source artifacts are not generic-front-door selectable until a materialized
  EmitC-backed source contract exists. Source route ids may remain only as
  exact-route or bundle-component packaging metadata where a target-owned
  caller has already proven a non-semantic packaging need.
- Extension/plugin-owned artifact routes may be registered through a
  target-layer plugin-exporter bundle registry keyed by extension plugin name.
  Public tools that already own an `ExtensionPluginRegistry` must pass that same
  active registry into built-in target exporter registration so enabled plugins
  can contribute their plugin-owned target artifact exporters through the
  generic boundary. A plugin-owned exporter bundle may additionally declare
  required enabled extension plugins for composite routes whose selected-plan
  contract spans more than one plugin-owned component. Template, Toy,
  TensorExtLite, Offload, RVV, scalar, and dispatch families without current
  materialized artifact routes must contribute no
  route authority. Disabled
  or missing plugins must not silently publish their plugin-owned target routes.
  Later selected-plan export then fails closed as an unknown or unavailable
  route/origin instead of falling back to a central extension-specific exporter
  branch. Central built-in composition may perform generic built-in plugin
  registration/linkage, but it must not know which RVV-specific or
  scalar-specific target-support helper to call.

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
- Offload selected without a supported target artifact route -> default/header/
  bundle artifact front doors fail closed without output.
- Route spoofing across RVV/scalar/offload origins or artifact kinds -> generic
  exporter metadata validation must fail before target-owned output.

#### 5. Good/Base/Bad Cases

- Good: the generic source-only front door is absent, while
  `--tcrv-export-target-header-artifact` and the generic object front door fail
  closed for families without current materialized artifact routes and emit no
  source, header, object, or bundle bytes.
- Base: historical `tcrv-export-scalar-*`, RVV direct microkernel, and
  RVV+scalar dispatch route strings are not supported artifact selection.
- Bad: each generic translate helper manually repeats
  `registerRVVMicrokernelTargetExporters`,
  `registerScalarMicrokernelTargetExporters`, and similar target-owned route
  helpers; adding a target route then
  requires broad hand-editing in generic tool code.
- Bad: central built-in target exporter composition registers the selected RVV
  binary microkernel routes directly even when `rvv-plugin` is missing or
  disabled.

#### 6. Tests Required

- C++ registry tests must prove the built-in helper registers only currently
  supported target artifact routes and keeps duplicate/invalid custom
  registration diagnostics.
- lit/FileCheck route tests must continue to cover generic front-door
  fail-closed behavior for missing materialized artifact routes and route
  spoofing failures that produce no executable artifact.
- CMake checks must include the built-in Target support library in the tool and
  C++ test link graph.

#### 7. Wrong vs Correct

Wrong:

```cpp
// In each generic translate helper:
registerRVVMicrokernelTargetExporters(registry);
registerScalarMicrokernelTargetExporters(registry);
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

Trigger: a public translate tool needs route-family helper commands for
target-owned artifact routes. Generic target artifact front doors still use
`TargetArtifactExporterRegistry`; this contract is only for optional public
command aliases and their route metadata. Deleted RVV/scalar/RVV+scalar direct
C helper routes are not current alias candidates. Future helper commands for
executable artifacts must be rebuilt on a materialized EmitC-module route
rather than restoring descriptor/family-record printers.

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
- A target translate route contribution records only the helper command route
  id, help description, target-owned export callback, and whether stdout must
  be switched to binary mode before callback execution.
- The public translate tool owns the MLIR `TranslateFromMLIRRegistration`
  object lifetime and attaches its dialect-registration hook generically while
  iterating `TargetTranslateRouteRegistry::getRoutes()`.
- Built-in target translate route registration iterates enabled extension
  plugins and asks each plugin's target-support manifest hook to contribute any
  currently supported helper routes.
- `tcrv-translate` must not manually loop over deleted RVV direct,
  scalar direct, or RVV+scalar dispatch manifests. It should call
  `registerBuiltinTargetTranslateRoutes` once and then register each supported
  contributed route generically.
- Historical standalone C helpers, including RVV smoke probe, RVV
  microkernel, scalar microkernel, and RVV+scalar dispatch helpers, must remain
  absent or fail closed until a rebuilt EmitC-module route owns them.
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
registerDeletedRVVDirectCRoutes();
registerDeletedScalarOrDispatchDirectCRoutes();
```

Correct:

```cpp
TargetTranslateRouteRegistry routes;
registerBuiltinTargetTranslateRoutes(routes);
for (const TargetTranslateRoute &route : routes.getRoutes())
  registerTranslateRoute(route, registerTianChenRVTranslateDialects);
```

The correct shape keeps supported route-family facts and callbacks in
target-owned support modules, while the public tool supplies only the generic
MLIR translate registration and dialect hook. Today, historical metadata-driven
C exporters contribute no route-family callbacks.

The artifact-kind aware generic route may also dispatch supported non-source
artifacts through target-owned exporters. Shared generic routing still validates
only route id, artifact kind, origin, emission kind, selected path,
lowering-boundary reference, runtime ABI metadata, and required capability refs;
target-specific artifact body content stays in target-owned exporters.

The target artifact bundle export is a directory materialization layer over the
same registry-derived artifact records that the emission manifest serializes.
It may iterate `collectTargetArtifactBundleRecords`, call the registered
source/header/object exporter callbacks, and write a deterministic
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
  paths, raw logs, or secret-like text. Unsupported selected
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
only the selected non-fallback artifact record. An Offload selected path with an
unsupported emission plan produces no bundle artifact and fails closed at the
front door.
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
`tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` entry may run
the existing execution planning pipeline with built-in plugin and target
artifact exporter registries, and finally call the same bundle exporter. It
must not first run the deleted bounded marked-linalg/vector RVV binary frontend
lowering slice or otherwise create `tcrv.exec.kernel`, `mem_window`, or
`runtime_param` ABI boundaries from high-level source bodies in the tool
front door. Inputs to this entry must already contain the execution anchors
needed by planning until a future plugin/interface-owned frontend rebuild
exists.
It must fail before printing bundle completion if planning, execution-plan
coherence, route validation, or artifact materialization fails, and it must not
weaken the bundle component contract or runtime ABI signature validation.

## Deleted RVV+Scalar Dispatch Direct Route Surface

Historical RVV+scalar dispatch source/header/object/self-check manifests and
route-family helper APIs are deleted as production route authority. Public
tools and generic target-artifact front doors must not register these route
families as supported executable C artifact paths. Historical add/sub/mul route
names must not be used as active target artifact selection authority, test API,
or diagnostic API.

Future dispatch executable artifact support must be rebuilt through
extension-family ops, a materialized common EmitC module, and the MLIR C/C++
emitter. Python evidence helpers may consume compiler-emitted metadata from
that rebuilt path, but they must not decide route selection, runtime ABI shape,
artifact kind, source generation, or family semantics.

Tests for the current deletion state must prove that RVV+scalar dispatch
source/header/object/self-check route ids are absent from active target
artifact exporter registration and absent from built-in translate route
registration. Tests must not protect no-op route-shell registration APIs.

## Deleted RVV Selected Lowering Boundary Route

The compiler may materialize selected-path lowering-boundary metadata through
the generic extension plugin registry only when a plugin has an active boundary
surface. RVV no longer has an active plugin-local selected-boundary operation:
selected RVV direct variants or dispatch cases return fail-closed no-boundary
behavior until explicit extension-family IR plus a materialized EmitC route
exists. Scalar fallback remains a no-boundary generic fallback envelope.

Rules:

- materialization consumes selected `tcrv.exec` dispatch or selected-marker
  structure and generic variant `origin` metadata;
- ownership and legality are routed through the `ExtensionPluginRegistry` and
  the selected variant's origin plugin before any plugin-local metadata is
  created;
- scalar or other fallback references remain selected `tcrv.exec` metadata and
  are routed through their generic fallback envelope; scalar fallback must not
  receive RVV ops and must not synthesize a scalar plugin-local boundary;
- RVV must not materialize an unsupported boundary placeholder;
- downstream emission planning must treat a selected RVV path without a future
  materialized EmitC route as fail-closed, not as a metadata-only artifact
  candidate;
- the bounded scalar source slice has no active typed scalar microkernel
  authority; historical scalar microkernel syntax must fail closed and scalar
  fallback must not synthesize a microkernel from descriptorless no-body state,
  kernel frontend markers, bridge metadata, or a default family;
- deleted scalar element-count metadata is not selected-boundary authority and
  must not authorize scalar boundary materialization;
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
  supported diagnostic records plugin-owned intent such as
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
self-description across the lowering boundary. Each entry must have non-empty
name, value, role, and note fields. The generic emission-planning layer may
serialize these entries as diagnostic self-description, but target artifact
export must not use plugin-specific names such as RVV capacity facts as route
selection, route preflight, runtime ABI, shape, VL/AVL, or performance
evidence. For RVV, any capacity metadata must already have been validated by
the RVV plugin against the selected variant and target capabilities before it
is serialized as diagnostic metadata.

The current public `tcrv-opt` built-in registry includes the RVV first-slice
plugin. Therefore an `origin = "rvv-plugin"` selected path can route through
RVV plugin diagnostics instead of a generic unregistered-origin failure. RVV
emission planning still fails closed until explicit extension-family IR plus a
  materialized EmitC route exists; no unsupported route metadata may be promoted
to runtime ABI, artifact, correctness, or performance evidence. The tool-level
`--tcrv-disable-builtin-plugins` option preserves an explicit empty-registry
surface for negative parser/diagnostic tests.

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

## Deleted RVV Smoke-Probe Target Export Boundary

### 1. Scope / Trigger

Trigger: historical post-planning MLIR contained a selected RVV path and a
  matching plugin-owned selected-boundary record, then used a
target/export tool to emit a deterministic standalone C smoke program with
`riscv_vector.h` and RVV intrinsics.

This direct source frontdoor is deleted. RVV hardware/toolchain smoke evidence
belongs in explicit probe tooling and separate `ssh rvv` artifacts; it must not
be exposed as a compiler source artifact route.

### 2. Signatures

Deleted C++ entry point:

```cpp
llvm::Error exportRVVSmokeProbeC(mlir::ModuleOp module,
                                 llvm::raw_ostream &os);
```

### 3. Contracts

- The former standalone smoke-probe source front door must be absent before
  printing C source.
- Built-in target artifact exporter registration must not publish the former
  smoke-probe route identity or any standalone direct C source artifact kind as
  a supported RVV source route.
- `RVVExtensionPlugin` must not turn plugin-local smoke-probe metadata or route
  records into supported emission readiness or a supported emission plan.
- Historical standalone smoke-probe metadata must not remain as active
  code/spec/test fixtures or plugin special-case legality input.
- No output may contain `#include <riscv_vector.h>`, `__riscv_` intrinsic
  compute, probe functions, or a `main` from this compiler frontdoor.

### 4. Validation & Error Matrix

- Generic target artifact export sees a stale smoke-probe source route identity
  -> the generic source-only front door is absent and no C source is printed;
  coherence must still fail closed if stale source-route metadata reaches a
  target-owned helper.
- A selected RVV variant carries stale standalone smoke-probe route metadata ->
  plugin legality/emission must not report a supported standalone source
  artifact.

### 5. Good/Base/Bad Cases

- Good: lit coverage proves selected RVV metadata remains unsupported and emits
  no C text.
- Base: selected RVV metadata now reaches no-boundary/fail-closed diagnostics
  and does not materialize plugin-local boundary records for non-executable
  planning evidence.
- Bad: selected RVV metadata or stale standalone smoke-probe route metadata
  emits standalone C, `riscv_vector.h`, `__riscv_` intrinsics, or a probe
  `main`.

### 6. Tests Required

- lit/FileCheck coverage for generic source frontdoor fail-closed behavior when
  no materialized EmitC route exists.
- C++ registry coverage proving built-in target artifact exporters only expose
  the current allowed route set.
- Plugin legality/emission coverage proving historical standalone smoke-probe
  metadata is not a supported artifact source.
- Full project check must still pass through `check-tianchenrv`.

### 7. Wrong vs Correct

Wrong:

```text
selected RVV metadata -> standalone RVV smoke-probe C source
```

Correct:

```text
selected RVV metadata -> unsupported missing-materialized-EmitC diagnostic
future rebuild:
explicit extension-family ops -> materialized MLIR EmitC module -> artifact
```

## Deleted RVV Direct Microkernel Target Export Boundary

### 1. Scope / Trigger

Trigger: historical post-planning MLIR would previously have selected an RVV
path and a bounded RVV wrapper attachment, then exported executable C directly
from selected metadata and target records.

This boundary is deleted. RVV direct microkernel source/header/object and
self-check exports must fail closed until a future route materializes a real
MLIR EmitC module and emits C/C++ through the MLIR emitter.

### 2. Signatures

Deleted public command:

```text
none
```

Historical C++ entry point:

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
- If the selected variant carries legacy bounded `tcrv_rvv.element_count`,
  that field is metadata-only diagnostic input. It must not ask the RVV plugin
  to materialize a wrapper, callable ABI, generated C source, correctness
  coverage, performance, or broad
  microarchitecture semantics.
- The deleted RVV selected-boundary route must not be required or accepted as
  source/export authority for the selected path.
- Deleted direct child `tcrv_rvv.*_microkernel` wrappers are no longer
  parseable active RVV dialect inputs. Historical references to those wrappers
  are fail-closed evidence only and must not be treated as
  selected compute truth, descriptor/config validation authority, or a source
  of callable ABI parameters.
- RVV selected emission planning must not resolve lhs-input-buffer,
  rhs-input-buffer, output-buffer, or runtime-element-count ABI parameters from
  `selectedPlan.descriptor` or finite family records.
- Output must be no executable C source, no header, no object, and no
  self-check harness.
- Historical selected metadata may remain parseable only as fail-closed input
  for unsupported missing-materialized-EmitC diagnostics.
- Output must not include timestamps, absolute paths, raw logs, credentials,
  benchmark sizes, latency/throughput numbers, or performance claims.

### 4. Validation & Error Matrix

- Missing selected RVV path, scalar-only path, offload-only path, or
  fallback-only path -> export fails before source output.
- Missing future EmitC route authority, stale selected-boundary metadata, or
  mismatched selected-path metadata -> export fails.
- Deleted RVV wrapper references, missing or stale selected-path metadata,
  malformed structured control/dataflow body, setvl/with_vl policy mismatch,
  or missing future EmitC route authority -> deleted-route/fail-closed
  behavior; no source/header/object bytes are emitted.
- Missing, unavailable, non-symbol, non-RVV, or unknown selected variant
  requirements -> export fails.
- Missing or mismatched preserved selected march capability metadata -> export
  fails.
- Any validation failure must leave stdout without partial C source and must not
  fall back to a legacy standalone smoke-probe source front door.

### 5. Evidence Interpretation

This deleted route produces no source, self-check harness, object, runtime
execution, correctness evidence, or performance evidence. Future RVV evidence
must start from a materialized MLIR EmitC module route and then record separate
real `ssh rvv` compile/run evidence for the concrete artifact under test.

### 6. Emission Plan / Manifest Handoff

When historical selected RVV metadata reaches emission planning, the RVV plugin
must return an unsupported deleted-route plan:

```text
status: unsupported
emission kind: rvv-runtime-callable-direct-c-source-exporter-deleted
lowering pipeline: deleted-direct-c-source-route
runtime ABI: unsupported-emission-runtime-abi
runtime ABI kind: unsupported-plugin-runtime-abi
runtime ABI name: unsupported-emission-runtime-abi
runtime glue role: no-runtime-glue-unsupported
artifact kind: unsupported-deleted-direct-c-route
```

This plan is a deletion diagnostic only. It is not a compiler handoff to a C
source exporter.

## Scalar Fallback Unsupported Boundary

The first scalar fallback plugin slice may keep the portable fallback proposal
visible to generic planning, but selected scalar fallback has no active EmitC
lowering, runtime ABI, target artifact route, or legacy metadata emission route.
It must therefore return an unsupported emission readiness result and materialize
only a fail-closed emission-plan diagnostic:

```text
status: unsupported
emission kind: scalar-fallback-unsupported-emission
lowering pipeline: scalar-fallback-no-materialized-emitc-route
runtime ABI: scalar-fallback-no-runtime-abi
runtime ABI kind: unsupported-plugin-runtime-abi
runtime ABI name: unsupported-emission-runtime-abi
runtime glue role: no-runtime-glue-unsupported
artifact kind: unsupported-emission-diagnostic
```

This is still compiler-decision metadata. It does not prove that TianChen-RV
emitted LLVM IR, generated an object, linked a runtime, executed a scalar
kernel, proved correctness, or measured performance. Later scalar fallback
lowering must add plugin-local lowering code and validation artifacts before
reporting executable support. This unsupported readiness/plan result also does
not license metadata-alone selected-boundary materialization.

## Deleted Scalar Explicit Microkernel Target Export Boundary

Trigger: historical post-planning MLIR would previously have selected a scalar
fallback path and a typed scalar microkernel attachment, then exported
executable portable C directly from selected metadata and family records.

This boundary is deleted. Scalar direct source/header/object exports must fail
closed until a future route materializes a real MLIR EmitC module and emits
C/C++ through the MLIR emitter.

Deleted public route:

```text
none
```

Deleted-route metadata:

```text
status: unsupported
emission kind: scalar-runtime-callable-direct-c-source-exporter-deleted
lowering pipeline: deleted-direct-c-source-route
runtime ABI kind: unsupported-plugin-runtime-abi
runtime ABI name: unsupported-emission-runtime-abi
runtime glue role: no-runtime-glue-unsupported
artifact kind: unsupported-deleted-direct-c-route
```

Contracts:

- Input must be real post-planning MLIR with one selected scalar fallback path.
- The selected variant must be owned by `origin = "scalar-plugin"` and require
  an available capability whose id is `scalar.fallback`.
- Deleted finite-family scalar microkernel syntax must not be accepted as
  active route authority; if any historical fixture still contains that syntax,
  it must fail closed before source output.
- Output must be no executable portable C source, no header, and no object.
- Historical selected metadata may remain parseable only as fail-closed input
  for unsupported missing-materialized-EmitC diagnostics.

Missing selected scalar path, deleted scalar microkernel syntax, unavailable
fallback capability, unknown route id, unsupported artifact kind, route
spoofing, offload-only paths, and ambiguous multiple supported artifacts must
fail before source output.

## Support-Layer Runtime ABI Shape Primitives

The support layer may keep C++ helpers for extension-agnostic runtime ABI shape
validation. These helpers are descriptorless and do not own selected route
identity, artifact identity, runtime evidence, or callable emission authority.

`FiniteBinaryRuntimeABIContract` owns only reusable ABI shape primitives:

- ordered parameter roles for lhs/rhs/out/runtime element count;
- role requirements used by target or plugin validation;
- `tcrv.exec.mem_window` specs for lhs/rhs/out buffer roles;
- `tcrv.exec.runtime_param` specs for runtime element count and an explicitly
  named dispatch availability guard.

It must not own runtime ABI kind/name strings, runtime glue role strings,
target artifact route ids, artifact kinds, source/header/object selection
policy, bundle metadata, descriptor-local metadata, evidence paths, ssh facts,
target capabilities, selected march/mabi, or performance/correctness claims.
It also must not publish static selected-family contracts for RVV/scalar
families.

### Scenario: Explicit ABI Shape Primitive

#### 1. Scope / Trigger

- Trigger: plugin/target-owned code needs reusable parameter-role,
  mem-window, or runtime-param shape helpers.
- The caller must provide the contract; Support must not infer one from RVV,
  scalar, dtype, arithmetic family, route id, selected artifact metadata, or
  descriptor metadata.

#### 2. Signatures

Deleted support signatures:

- no public support-layer callable-plan struct or builder;
- no support-layer metadata mirror validator for an IR-backed callable plan;
- no direct/dispatch invocation-comment contract struct, builder, or
  formatter.

Still-allowed support signatures are limited to neutral role/shape primitives
for runtime ABI parameters, mem-window specs, runtime-param specs, and bounded
role binding.

#### 3. Contracts

- The caller-owned contract may name a family identifier for diagnostics only;
  Support must not interpret it as a selected RVV/scalar route.
- Runtime element-count and dispatch guard C names are explicit caller-selected
  ABI spellings, not Support defaults, descriptor element counts, or
  compile-time vector shape facts.
- Support must not provide selected-family i32 helper overloads or no-argument
  defaults equivalent to an i32 family.
- Support must not publish `rvv_available` as a default dispatch guard C name.
  A transform or plugin that needs a guard name must pass one explicitly.
- Runtime ABI identity strings for RVV, scalar, dispatch, offload, or future
  families must come from the owning plugin/target route construction, not from
  Support.
- Support must not build an IR-backed callable plan object from plugin family
  contracts, validate detached metadata mirrors of that object, or format
  direct/dispatch invocation evidence as comment-body metadata.

#### 4. Validation & Error Matrix

- Missing plugin/target ABI construction -> fail closed as a missing
  plugin-owned ABI construction gap; do not recreate Support defaults.
- Missing callable role -> role binding fails before export or invocation.
- Duplicate callable role -> role binding fails before export or invocation.
- Wrong C type or ownership for any callable role -> role binding fails before
  export or invocation.
- Dispatch ABI metadata attached as detached records instead of direct
  `tcrv.exec.runtime_param` IR -> dispatch export fails.
- Stale selected runtime-count metadata that disagrees with typed exec IR ->
  target/export validation fails.
- A generated artifact needs invocation evidence -> it must wait for a rebuilt
  materialized EmitC-module route and plugin-owned runtime ABI surface, not a
  support-layer comment formatter.

#### 5. Good/Base/Bad Cases

- Good: a plugin-owned route constructs an explicit contract and passes
  explicit runtime-count and dispatch-guard C names into role/shape validation.
- Base: a missing plugin-owned ABI route remains unsupported/fail-closed.
- Bad: Support publishes selected RVV/scalar i32 family contracts, dispatch C
  function names, callable-plan metadata mirrors, comment-body invocation
  contracts, or `rvv_available` defaults.

#### 6. Tests Required

- C++ support tests may cover neutral role binding and shape primitives without
  hard-coded RVV/scalar selected-family identity.
- Target artifact tests must not assert RVV/scalar dispatch C function identity
  strings as Support-owned contract output.
- Generated source/header comment tests must wait for a rebuilt EmitC-module
  route; support tests must not preserve direct/dispatch invocation comments as
  compiler-owned runtime ABI evidence.
- Focused scans must show deleted Support selected-family helper surfaces and
  callable-plan/comment-contract surfaces remain absent.

#### 7. Wrong vs Correct

Wrong:

```text
selected-family metadata -> support-owned callable evidence comment
```

Correct:

```text
plugin-owned runtime ABI surface -> explicit role/shape validation helper
future executable evidence -> materialized EmitC module and runtime artifact
```

The contract also must not turn runtime SSA/control values into compile-time
facts: the dispatch availability guard remains a `tcrv.exec.runtime_param` role
and the selected dispatch case must still link to that guard through typed
`runtime_guard_required = true` plus `runtime_guard`.

Compiler-owned dispatch runtime-guard materialization happens in the transform
layer before selected lowering-boundary and emission-plan materialization. The
generic materializer may inspect `TargetCapabilitySet` availability/conflict
facts and the typed `tcrv.exec.case runtime_guard_required` marker, but it must
create only compute-free exec IR: one same-kernel dispatch-availability
`runtime_param` and selected `tcrv.exec.case runtime_guard` symbol references.
Generic `condition`, `guard`, or `policy` strings may be retained as printable
annotations, but they are not semantic runtime ABI guard triggers. RVV, scalar,
and future plugin lowering code may consume or validate those links, but must
not privately invent the dispatch guard parameter or attach dispatch-case links
as a plugin-local side effect.

## Deleted Target-Layer RVV Binary Runtime ABI Contract

The target-layer RVV binary runtime ABI contract and finite family registry are
deleted as active planning/emission authority. RVV selected-path code must not
derive callable parameters, C pointer spellings, runtime ABI kind/name, runtime
glue role, component group, function stem, route id, or artifact kind from
finite add/sub/mul family records or selected-binary metadata.

The older i32 support-layer contract remains the scalar/dispatch compatibility
owner for bounded i32 shared surfaces only. It must not be repurposed as RVV
binary family authority, and it must not synthesize direct RVV source/header/
object routes.

Direct RVV binary source/header/object route ownership is deleted as production
route authority. Historical RVV direct microkernel route families must not be
registered as supported target artifact exporters until a future EmitC-module
source route replaces direct C semantic export.

Required tests for changes to this contract:

- C++ tests must prove deleted RVV direct route ids are absent from active
  target artifact exporter registration;
- lit tests must prove generic source/header/object front doors fail closed
  without printing direct RVV C source, headers, objects, or bundle outputs.

## Deleted Host RVV + Scalar Dispatch Direct C Artifact Boundary

Historical post-planning RVV+scalar dispatch inputs once attempted to export
source, headers, objects, self-check harnesses, and self-check objects through
direct target-owned C printers. That direct C artifact boundary is deleted.
Selected `tcrv.exec.dispatch`, `tcrv.exec.fallback`, runtime guard metadata,
RVV/scalar lowering boundaries, or typed microkernel attachments do not by
themselves provide a supported runtime-callable C artifact route.

The current valid behavior is fail-closed route absence. Future dispatch
source/header/object support must be rebuilt from extension-family ops through
a materialized common EmitC module and the MLIR C/C++ emitter. Runtime,
correctness, or performance claims must wait for that rebuilt artifact plus
separate real `ssh rvv` evidence when RVV execution is claimed.

### Deleted Dispatch Self-Check Harness Export

#### 1. Scope / Trigger

Trigger: historical post-planning input would previously have satisfied the
RVV+scalar finite dispatch C export boundary, and the caller wanted explicit
runtime invocation evidence for the generated dispatcher.

This export mode is deleted. It must not be restored as a direct C evidence
helper.

#### 2. Signatures

Deleted public command:

```text
none
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

- The export must fail closed before printing direct C source.
- The default dispatcher direct C export is also deleted.
- No harness code, success marker, embedded RVV/scalar callable source, or
  dispatcher C body may be emitted by this deleted route.

#### 4. Validation & Error Matrix

- Any attempt to use historical self-check route metadata -> fail before
  direct C source output.
- Runtime evidence must wait for a rebuilt EmitC-module route and real `ssh rvv`
  compile/run evidence over that rebuilt artifact.

#### 5. Good / Base / Bad Cases

- Good: fail-closed tests prove no self-check C body is printed.
- Base: historical metadata can remain as negative input.
- Bad: a direct self-check command emits a harness or success marker.

#### 6. Tests Required

- lit/FileCheck must prove deleted dispatch self-check routes fail closed.
- Generic target artifact tests must prove deleted dispatch self-check route
  ids are absent from active route registration.
- Any future RVV runtime/correctness claim for a rebuilt self-check artifact
  must include separate `ssh rvv` compile/run evidence and must name the
  selected compile flags.

#### 7. Wrong vs Correct

Wrong:

```text
The dispatch self-check passed, so TianChen-RV supports generic RVV lowering
and runtime integration.
```

Correct after future rebuild:

```text
A rebuilt EmitC-module-derived dispatch self-check artifact compiled and ran on
ssh rvv with selected flags; this proves only the bounded rebuilt artifact under
the stated selected runtime ABI values.
```

### Deleted Dispatch Library Object Export

Historical RVV+scalar dispatch object and self-check-object helpers compiled
direct-printer generated C. Those object helpers are deleted with the direct C
semantic exporter path. The generic `--tcrv-export-target-artifact` front door
must not select a dispatch object route until a future EmitC-module source
route supplies the artifact authority. Historical object route ids must not be
used as active selection authority, test API, or diagnostic API.

### Deleted Dispatch Self-Check Executable Evidence Bridge

The former repo-owned dispatch e2e runner was removed with the direct C
semantic exporter deletion campaign. It is no longer a supported evidence path
because it consumed selected metadata, route records, generated direct C
source, and bundle outputs as executable proof.

Future dispatch executable evidence must consume artifacts rebuilt from
extension-family ops through a materialized common EmitC module and the MLIR
C/C++ emitter. Runtime/correctness claims still require separate real `ssh rvv`
evidence over that rebuilt artifact.

### Deleted Dispatch ABI Header Export

Historical RVV+scalar dispatch header helpers are deleted with the direct C
semantic exporter route. The generic `--tcrv-export-target-header-artifact`
front door must not select a dispatch header route until a future EmitC-module
route supplies source/header authority. Historical header route ids must not
be used as active selection authority, test API, or diagnostic API.

## Runtime Offload No-Route Boundary

The first runtime-offload plugin slice may return an unsupported emission
readiness result. Production boundary materialization must return no selected
`tcrv_offload.lowering_boundary` until a real materialized EmitC/runtime/artifact
route exists. After descriptor-route deletion, its emission plan must be
unsupported:

```text
status: unsupported
runtime ABI kind: unsupported-plugin-runtime-abi
runtime ABI name: unsupported-emission-runtime-abi
runtime glue role: no-runtime-glue-unsupported
diagnostic: runtime-offload has no active executable lowering or target artifact route
```

This unsupported plan means only that the compiler recognized a selected
Offload-capable variant and deliberately found no active materialized route. It
does not emit vendor runtime calls, allocate or copy device buffers, compile
accelerator kernels, generate objects, link runtime libraries, run hardware,
prove correctness, or measure performance. Production lowering-boundary
materialization for Offload now returns no boundary; any stale
`tcrv_offload.lowering_boundary` op is a no-active-route validation surface, not
route authority. Availability depends on explicit `offload.runtime` capability
metadata and plugin legality; it cannot resurrect unavailable, malformed,
illegal, or unselected variants.

Public source/default/header/bundle artifact front doors must ignore unsupported
Offload emission plans as artifact candidates and fail closed if no other
supported route exists. Reintroducing a metadata artifact exporter, selected-plan
export scope, or descriptor-shaped ABI mirror is a descriptor-exit regression
unless a future spec explicitly rebuilds Offload through the common runtime C ABI
route.

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
semantics into `tcrv.exec` or generic manifest code. Deleted direct RVV
selected-config/runtime AVL contracts are not active bundle authority. A future
RVV+scalar dispatch exporter must consume explicit extension-family IR plus a
materialized EmitC/runtime route before deriving dispatch source/header/object
or bundle records. Unsupported selected paths must omit target artifact records
instead of fabricating route data.
