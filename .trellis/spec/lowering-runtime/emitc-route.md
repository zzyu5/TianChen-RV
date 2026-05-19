# Unified EmitC Route

## Current Main Route

The current TianChen-RV lowering route is:

```text
TCRV extension family ops
  -> EmitC ops
  -> C/C++ emitter
  -> intrinsic / vendor builtin / runtime C ABI
  -> native compiler
```

The default native compiler is clang/LLVM. GCC is a compatibility path.
Vendor compilers are extension-specific compatibility paths.

This route applies across extension families:

```text
TCRV RVV family ops
  -> EmitC
  -> RVV intrinsic C/C++
  -> clang default, gcc compatible

TCRV IME family ops
  -> EmitC
  -> IME/vendor intrinsic C/C++

TCRV TensorExt family ops
  -> EmitC
  -> vendor intrinsic C/C++

TCRV Offload family ops
  -> EmitC
  -> runtime C ABI
```

MLIR vector, LLVM scalable vector, LLVM RVV intrinsic IR, inline assembly, and
backend patches are future optional routes. They are not the current default
system definition.

EmitC and `TCRVEmitCLowerableRoute` are faithful emission mechanics, not the
place where performance structure, scheduling, or missing computation
semantics are invented. For RVV, the route consumes the selected vector-level
`tcrv_rvv` body after any RVV plugin-local selected-body realization. The
route builder may map that body to includes, C vector types, intrinsic/runtime
callee names, ABI bindings, loops, and `emitc.call_opaque` calls. It must not
derive RVV dtype, SEW/LMUL policy, mask/tail behavior, reduction layout,
unroll/prefetch structure, or operation semantics from route ids, artifact
metadata, test names, descriptor residue, i32m1 helper names, or common target
export code.

A route is not a semantic decorator over an existing op name or route id. The
provider-built route is valid only after the selected family body already
expresses the body/config facts that make the route legal. Common EmitC lowering
then materializes that payload as MLIR EmitC; it must not inspect legacy
`tcrv_rvv.i32_*` names, descriptors, or artifact labels to decide which RVV
intrinsic, dtype, memory form, or runtime schedule should exist.

Common EmitC lowering remains a neutral materialization shell. It must not
branch on RVV, IME, TensorExt, Offload, scalar, vendor names, dtype names,
kernel names, or intrinsic spellings to choose extension semantics. Unsupported
or unrealized extension-family body shapes must fail closed in the plugin-owned
route provider or selected-body realization boundary before C/C++ or target
artifacts become authoritative.

## Common EmitC Lowering Template

All extension families should reuse common lowering infrastructure:

```text
header/include builder
function boundary builder
emitc.call_opaque builder
C type mapping helper
ABI argument mapping helper
pointer/buffer mapping helper
intrinsic name resolver
generated C compile test harness
```

Each extension family contributes only the mapping:

```text
selected and, when needed, realized extension-family body structure
  -> intrinsic / runtime call name
  -> header
  -> operand mapping
  -> result mapping
```

The common lowering pass consumes `TCRVEmitCLowerableInterface` and emits
EmitC ops. This is the key difference from one hardware target owning one
separate lowering pass.

## Scenario: Generated EmitC Lowerable Op Interface Boundary

### 1. Scope / Trigger

Use this contract when a typed extension-family operation participates in the
common EmitC route. The operation must carry a generated MLIR op interface, and
the target/export path must query that generated interface before constructing
the C++ route payload.

### 2. Signatures

- Generated ODS interface:
  `TCRVEmitCLowerableOpInterface`.
- Generated interface methods:
  `llvm::StringRef getTCRVEmitCLowerableSourceOpName()`;
  `llvm::StringRef getTCRVEmitCLowerableSourceRole()`.
- C++ route payload:
  `TCRVEmitCLowerableRoute`, built through the hand-written
  `TCRVEmitCLowerableInterface` adapter API.
- Route provenance field:
  `TCRVEmitCSourceOpProvenance::opInterface`.

### 3. Contracts

- `TCRVEmitCLowerableOpInterface` is the IR-modeled source-op boundary.
- `TCRVEmitCLowerableInterface` remains the C++ adapter for building and
  verifying a `TCRVEmitCLowerableRoute`.
- The generated op interface must expose bounded provenance only: source op
  name and source role. It must not expose descriptor-selected computation,
  target hardware facts, runtime results, correctness evidence, performance
  evidence, or family-specific intrinsic spelling to common code.
- Target/plugin-owned code may map the interface-backed provenance to
  family-owned intrinsic or runtime names, then store the interface class name
  in route provenance for source/export evidence.
- Descriptor metadata remains selected-config, ABI identity, legacy id, and
  mismatch cross-check data only.

### 4. Validation & Error Matrix

- Typed family op lacks `TCRVEmitCLowerableOpInterface` before route
  construction -> fail before artifact export.
- Interface source op name disagrees with the selected family operation ->
  fail before artifact export.
- Interface source role is unsupported for the bounded route -> fail before
  artifact export.
- Descriptor family disagrees with the typed body op -> fail before artifact
  export.
- Route provenance text is empty, unbounded, multiline, or unsafe -> route
  verification fails before source emission.

### 5. Correct/Base/Bad Cases

- Correct Stage 1: bounded i32 arithmetic ops such as `tcrv_rvv.i32_add`,
  `tcrv_rvv.i32_sub`, and `tcrv_rvv.i32_mul` are not kept as positive
  compatibility routes. Their route-interface use is deleted or fail-closed
  until a corrected vector-level route surface exists.
- Base: a target-specific bounded route still owns intrinsic names, vector
  suffixes, C type spellings, and ABI details locally.
- Bad: descriptor strings alone choose the arithmetic intrinsic, common code
  adds an `if RVV` branch to recover family semantics, or a new dtype-prefixed
  `tcrv_rvv.i32_*` op family is added and treated as dtype propagation.

### 6. Tests Required

- TableGen/CMake build coverage for generated op-interface declarations and
  definitions.
- C++ coverage proving the bounded typed ops implement the generated interface
  and that route provenance records the interface-backed source op.
- lit/FileCheck coverage proving exported source contains the interface-backed
  route evidence for each affected family op.
- Negative coverage proving descriptor/body mismatch still fails before source
  export.

### 7. Wrong vs Correct

Wrong:

```text
descriptor id -> choose arithmetic intrinsic -> emit C source
```

Correct:

```text
typed extension op implements generated op interface
  -> target/plugin queries interface-backed source provenance
  -> target/plugin maps to family-owned intrinsic/runtime call
  -> common TCRVEmitCLowerableRoute carries bounded route payload
  -> EmitC/C source evidence records generated-interface provenance
```

## Descriptor Boundary

Descriptor-driven computation is invalid as long-term architecture.

A descriptor must not:

- define computation semantics;
- decide which microkernel or extension operation represents the computation;
- serve as the way to add a new op;
- become the RAG template for generating a new extension family;
- replace extension family ops;
- justify direct C string emission as the primary route.

Existing fields and paths such as descriptor-based microkernel dispatch and
descriptor-to-C exporters are historical residue, deletion targets, or
fail-closed implementation debt. They must not be treated as transition
architecture, compatibility aids, semantic sources, or production inputs.
Executable rebuild work must use:

```text
extension family ops
  -> EmitC lowering
```

Direct handwritten C string emission is not the architecture. Generated C
should come from EmitC operations or a clearly marked bounded legacy helper
that is not used as the template for new extension work.

## Scenario: Selected Emission-Plan To Target Artifact Handoff

### 1. Scope / Trigger

Use this contract when a target artifact exporter materializes an object,
header, bundle, or metadata artifact from a selected execution path. This
applies to both direct selected-path diagnostics and selected
`tcrv.exec.dispatch` surfaces.

### 2. Signatures

- Selected path surfaces:
  `tcrv.exec.diagnostic {reason = "variant-selected", target = @variant}` or
  `tcrv.exec.dispatch` with `tcrv.exec.case` / `tcrv.exec.fallback` targets.
- Emission-plan diagnostic reason:
  `reason = "emission_plan"`.
- Supported emission-plan fields:
  `target`, `role`, `origin`, `status = "supported"`, `plan_kind`,
  `lowering_pipeline`, `emission_kind`, `artifact_kind`,
  `lowering_boundary`, `runtime_abi`, `runtime_abi_kind`,
  `runtime_abi_name`, `runtime_glue_role`, `runtime_abi_parameters`, and
  `required_capabilities`.
- Target exporter candidate fields:
  `kernel`, `selectedVariant`, `role`, `origin`, `routeID`,
  `emissionKind`, `artifactKind`, `loweringBoundary`, `runtimeABI*`, and
  structured runtime ABI parameters.

### 3. Contracts

- Target artifact exporters must select from supported emission-plan
  candidates that are selected by the current dispatch or selected-path
  diagnostic surface.
- Exporters must resolve the materialized variant from the selected candidate's
  `selectedVariant` and `role`; they must not infer the target by scanning for
  exactly one direct `tcrv.exec.variant` in the module.
- Non-selected sibling variants must not affect target artifact
  materialization.
- A selected path may hand off through a supported emission-plan diagnostic
  without a separate materialized lowering-boundary op when the plugin route
  consumes explicit typed extension-family IR directly. In that case,
  `lowering_boundary` remains bounded route metadata owned by the emission
  plan. If a materialized lowering-boundary op exists, the emission-plan
  `lowering_boundary` value must match that op name and capability metadata.
- Runtime ABI ownership metadata and ordered structured runtime ABI parameters
  must be checked before object/header/bundle output.
- A selected artifact route that emits C/C++ or packages an object from EmitC
  must materialize the selected extension-family route through the common
  EmitC materialization helper and verify the materialized module before
  invoking the MLIR EmitC C/C++ emitter or native compiler. The verified
  materialized module must contain an EmitC function boundary plus route
  source-op and call source-op provenance derived from the selected
  `TCRVEmitCLowerableRoute`.

### 4. Validation & Error Matrix

- No selected dispatch or selected-path diagnostic -> fail before artifact
  output.
- Selected path lacks exactly one emission-plan diagnostic -> fail before
  artifact output.
- Emission-plan target/role is stale relative to current selection -> fail.
- Multiple supported standalone artifact candidates and no registered
  composite route -> fail as ambiguous.
- Unknown route id, artifact kind mismatch, origin mismatch, emission kind
  mismatch, or runtime ABI mismatch -> fail before artifact output.
- Materialized lowering-boundary op present but not selected, duplicated,
  stale, origin-mismatched, or capability-mismatched -> fail before artifact
  output.
- Selected artifact route materializes an EmitC module without route source-op
  provenance, without call source-op provenance, without an EmitC function
  boundary, or with non-EmitC residue -> fail before C/C++ emission or object
  packaging.

### 5. Good/Base/Bad Cases

- Good: a rebuilt extension target artifact route has a supported
  emission-plan candidate selected by dispatch; fallback paths have
  unsupported diagnostics; artifact output materializes only from the selected
  non-fallback candidate.
- Base: RVV explicit typed bodies may materialize through the common EmitC
  route and then the MLIR EmitC C/C++ emitter. The bounded rebuilt RVV target
  artifact path may export a RISC-V object, a declaration-only header, and a
  coherent object+header bundle only from the selected materialized EmitC
  candidate.
- Base: A bounded RVV multi-VL claim is valid only when the selected
  extension-family route owns a structured EmitC loop payload that materializes
  as `emitc.for`, derives per-iteration remaining AVL through EmitC values, and
  keeps pointer/index advancement and intrinsic calls inside the materialized
  loop. Target artifact export may validate route provenance and package the
  materialized result, but it must not print loop source, choose RVV intrinsics,
  or infer route semantics from metadata.
- Base: TensorExtLite explicit typed bodies may materialize an ordered
  `configure -> load_frag -> tile_mma -> store_frag` role sequence through the
  common `TCRVEmitCLowerableRoute` materializer and produce an MLIR EmitC
  module. The rebuilt first target artifact slice may publish exactly one
  selected `riscv-elf-relocatable-object` route derived from that materialized
  EmitC handoff and the MLIR EmitC C/C++ emitter. The declaration-only
  `runtime-callable-c-header` surface is a composite derived from the same
  selected object candidate. Neither route may publish metadata-only artifact
  authority, direct C/source-export bodies, TensorExtLite runtime execution,
  correctness, or performance evidence.
- Base: TensorExtLite source-front-door materialization may also create a
  direct `tcrv_tensorext_lite.lowering_boundary` marker so generic
  emission-plan and coherence gates can validate the selected path. The ordered
  role sequence remains the TensorExtLite EmitC route provenance and artifact
  evidence; the lowering-boundary marker must not become descriptor, direct-C,
  source-export, or compute-body authority.
- Bad: two direct RVV variants exist and the exporter chooses whichever direct
  variant happens to be first or only after test reduction.

### 6. Tests Required

- Positive lit coverage for a retained selected-path target artifact route with
  a non-selected sibling variant when such a route exists.
- Positive lit coverage for current RVV and TensorExtLite explicit typed
  bodies reaching materialized EmitC and the MLIR EmitC C/C++ emitter, with
  object/header artifact route authority only where a route has been explicitly
  rebuilt.
- Negative lit coverage for ambiguous supported multi-variant candidates.
- Negative lit coverage for unselected multi-variant input.
- Negative lit coverage for unsupported selected RVV target artifact export
  after descriptor-route erasure.
- C++ or lit coverage proving selected artifact export fails closed when the
  common materialized EmitC handoff is missing required route provenance.

### 7. Wrong vs Correct

Wrong:

```text
module has exactly one direct variant
  -> target exporter assumes that variant is selected
  -> object/header route materializes from module shape
```

Correct:

```text
selected dispatch or selected-path diagnostic
  -> supported emission-plan diagnostic for the selected path
  -> TargetArtifactCandidate with selected variant, role, route, ABI metadata
  -> plugin-owned target exporter materializes object/header/bundle
```

## Scenario: Common Source-Artifact Bundle Front Door

### 1. Scope / Trigger

Use this contract when a user-facing translate command starts from supported
source MLIR and directly exports a target artifact bundle. The command is a
common orchestration bridge over already registered plugin source front doors,
shared planning/coherence, selected EmitC route materialization, and target
artifact bundle export. It is not a new frontend semantic recognizer and not a
family-specific target exporter.

### 2. Signatures

- Translate command:
  `tcrv-translate --tcrv-source-artifact-bundle-front-door`.
- Required bundle output option:
  `--tcrv-target-artifact-bundle-output-dir=<directory>`.
- Diagnostic/negative test switch:
  `--tcrv-disable-builtin-plugins`.
- The front door consumes
  `plugin::SourceFrontDoorPassRegistration` values collected from the enabled
  `ExtensionPluginRegistry`.
- The front door must invoke the shared
  `buildSourceArtifactFrontDoorPipeline` before calling
  `target::exportTargetArtifactBundle`.

### 3. Contracts

- The common translate layer may collect enabled plugin source-front-door pass
  registrations, run the shared source-artifact front-door pipeline, and invoke
  the existing target artifact bundle exporter.
- Source materialization remains plugin-owned. The common translate layer must
  not inspect source op bodies, route ids, artifact names, descriptor text, or
  family markers to infer computation semantics.
- Target artifact output must still flow through selected emission-plan
  diagnostics, plugin-owned `TCRVEmitCLowerableRoute` builders, the common
  materialized EmitC helper, MLIR EmitC C/C++ emission, and registered target
  artifact exporters.
- The command must use enabled plugin registrations. If plugins are disabled or
  no source-front-door pass is registered, it must fail before bundle output.
- The command must not bypass existing target bundle validation, output
  directory checks, overwrite refusal, candidate ambiguity checks, runtime ABI
  signature checks, materialized EmitC provenance checks, or forbidden metadata
  checks.

### 4. Validation & Error Matrix

- No registered source-front-door pass -> fail with a source-artifact bundle
  front-door diagnostic and write no bundle index.
- No source-front-door pass materializes a `tcrv.exec.kernel` or selected
  execution surface -> fail during the source-artifact front-door pipeline.
- Source-front-door materialization rejects stale seed, descriptor, selected
  boundary, or unselected variant residue -> propagate the plugin-owned
  fail-closed diagnostic and write no bundle index.
- The selected path has no supported target artifact candidate -> fail before
  bundle records or artifact files become authoritative.
- Multiple supported non-fallback candidates without a registered composite
  route -> fail as ambiguous before bundle output.
- Materialized EmitC handoff lacks route/call source provenance, function
  boundary, runtime ABI coherence, or uses forbidden descriptor/direct-C/
  source-export metadata -> fail before object/header/bundle output.
- Output directory is missing, not a directory, or already contains a derived
  bundle file/index -> fail through the target bundle exporter.

### 5. Good/Base/Bad Cases

- Good: the bounded RVV vector-source front door is deleted or fails closed
  before i32 add/sub/mul source can materialize a selected RVV boundary or
  export a RISC-V object/header bundle.
- Good: Toy or TensorExtLite source-front-door input reaches the same common
  translate command through plugin registrations, proving the command is not an
  RVV-specific source bridge.
- Base: The two-step `tcrv-opt --tcrv-source-artifact-front-door-pipeline |
  tcrv-translate --tcrv-export-target-artifact-bundle` chain may remain useful
  for debugging intermediate IR, but it is not the only production path for
  supported source-artifact bundle export.
- Bad: common translate code branches on RVV, Toy, TensorExtLite, dtype, shape,
  route id, descriptor string, or artifact filename to decide source semantics
  or target artifact behavior.
- Bad: a failed source-front-door route restores direct C/source-export or
  descriptor-driven compatibility output to make bundle checks pass.

### 6. Tests Required

- Positive lit coverage for at least one RVV vector-source case invoking
  `--tcrv-source-artifact-bundle-front-door` and checking selected variant
  identity, materialized EmitC provenance, runtime ABI metadata, object/header
  bundle records, and unmangled callable object symbol.
- Positive lit coverage for at least one non-RVV plugin source-front-door case
  through the same translate command.
- Negative lit coverage for disabled/unregistered plugins or empty
  source-front-door registration.
- Negative lit coverage for no matching source-front-door materialization and
  for no supported selected target artifact route.
- Focused scans over touched common translate/front-door code must show no
  descriptor route authority, direct C semantic exporter, source-export route,
  or family-specific semantic branch.

### 7. Wrong vs Correct

Wrong:

```text
source MLIR
  -> common translate code recognizes RVV/Toy/TensorExt source shape
  -> direct C/source-export or descriptor-compatible artifact output
```

Correct:

```text
source MLIR
  -> enabled plugin source-front-door pass registration
  -> shared source-artifact planning/coherence pipeline
  -> selected emission-plan candidate
  -> plugin-owned EmitC route builder
  -> common materialized EmitC helper
  -> MLIR EmitC C/C++ emitter
  -> registered target artifact bundle exporter
```

## Scenario: Common Materialized EmitC Header Artifact Foundation

### 1. Scope / Trigger

Use this contract when multiple extension-family target exporters need the same
declaration-only runtime-callable C header artifact shape from a selected
materialized EmitC route. A family may consume it as a standalone header route
only when the selected materialized EmitC header candidate is itself the
explicit artifact handoff authority. When the selected materialized EmitC object
candidate is the production handoff authority, the family must consume the
header helper as an object-backed composite header route with a route-local
candidate preflight that preserves the plugin-owned runtime ABI contract.

### 2. Signatures

- Common config:
  `target::MaterializedEmitCHeaderArtifactConfig`.
- Common object/header bundle config:
  `target::MaterializedEmitCObjectBundleArtifactConfig`.
- Common metadata evidence entry:
  `target::MaterializedEmitCHeaderArtifactMetadataEvidence`.
- Common validator:
  `validateMaterializedEmitCHeaderArtifactCandidate(candidate, config)`.
- Common exporter:
  `exportMaterializedEmitCHeaderArtifact(module, os, config)`.
- Common object/header bundle construction helper:
  `registerMaterializedEmitCObjectBundleArtifactExporters(registry, config)`.
- The config wraps a `SelectedEmitCArtifactRouteConfig` plus plugin-supplied
  header guard, evidence prefix, includes, optional expected selected variant,
  emission kind, lowering boundary, runtime ABI fields or dynamic runtime ABI
  identity mode, ordered runtime ABI parameters, and required artifact metadata
  evidence.
- The object/header bundle config wraps the header config plus plugin-supplied
  object/header export callbacks, header route id, component group, optional
  external ABI name, object handoff kind, and a bounded selected-object
  description for diagnostics.
- Dynamic runtime ABI identity flag:
  `MaterializedEmitCHeaderArtifactConfig::allowDynamicRuntimeABIIdentity`.
- Construction-template adapter dynamic runtime ABI identity wrapper:
  `ConstructionTemplateArtifactAdapterConfig::allowDynamicRuntimeABIIdentity`.

### 3. Contracts

- Common target code may validate extension-agnostic selected candidate fields
  supplied through the config: route id, selected candidate artifact kind,
  origin plugin,
  selected variant identity when configured, emission kind, lowering boundary,
  runtime ABI, runtime ABI kind/name, runtime glue role, ordered ABI parameter
  signature, and required artifact metadata values.
- A header helper may consume either a selected header candidate or a selected
  object candidate, but the generated artifact remains a header declaration.
  Object-backed header routes require a route-local candidate validation
  callback; the common layer must not infer object/header coherence from route
  names alone.
- If runtime ABI name is static for the route, the config must provide
  `runtimeABI` and `runtimeABIName` and the common validator must require exact
  equality with the selected candidate. If runtime ABI name is candidate-owned
  or route-family-owned, `allowDynamicRuntimeABIIdentity` may be enabled only
  with a route-local candidate validation callback. The common helper still
  requires non-empty candidate runtime ABI identity, a static runtime ABI kind,
  a static runtime glue role, and an exact ordered ABI parameter signature.
- Construction-template artifact adapter configs may leave
  `selectedVariant`, `runtimeABI`, `runtimeABIName`, or `externalABIName`
  empty only when the selected candidate owns that identity and a route-local
  candidate validation callback proves it. The adapter must forward dynamic
  runtime ABI mode into the common header/object bundle helper; it must not
  replace candidate-owned identity with a static legacy-route name or route
  family placeholder.
- Common target code may reject forbidden artifact metadata containing
  descriptor, metadata-diagnostic, direct-C, source-export, or compute-body
  residue.
- Common target code must still use the plugin-owned
  `SelectedEmitCArtifactRouteBuilderFn` to build the
  `TCRVEmitCLowerableRoute`; it must not recover family semantics from plugin
  names, route ids, metadata keys, or header evidence labels.
- The exported header must be declaration-only: configured includes, bounded
  evidence comments, and one function declaration whose argument list is the
  selected ordered runtime ABI parameter signature. Zero-argument headers must
  use `(void)`.
- Plugin-local code remains responsible for construction protocol metadata,
  source-op/source-role/source-interface provenance keys, semantic role graph,
  typed role evidence, and any family-specific route mapping.
- Common object/header bundle construction may register the object exporter and
  object-backed declaration-only header composite from the same config, and may
  own the reusable selected-object candidate selection, composite match,
  composite validation, runtime ABI parameter extraction, component-group
  metadata, external ABI name propagation, and object handoff metadata.
- Common object/header bundle construction must still call the plugin-supplied
  route-local candidate validation and export callbacks. It must not choose
  extension semantics, build intrinsic/vendor payloads, select compiler flags,
  infer role graphs, or branch on RVV, TensorExtLite, Toy, IME, Offload,
  scalar, vendor, dtype, shape, runtime, toolchain, or microarchitecture names.

### 4. Validation & Error Matrix

- Config lacks route id, artifact kind, origin plugin, route builder, header
  guard, evidence prefix, emission kind, lowering boundary, required runtime
  ABI identity fields, or runtime glue role -> fail before output.
- Config selected candidate artifact kind is neither
  `runtime-callable-c-header` nor `riscv-elf-relocatable-object` -> fail
  before output.
- Object-backed header config lacks a route-local candidate validation callback
  -> fail before output.
- Object/header bundle config lacks object/header callbacks, header route id,
  component group, handoff kind, owner plugin, or a header artifact kind of
  `runtime-callable-c-header` -> fail before registration.
- Dynamic runtime ABI identity mode lacks a route-local candidate validation
  callback, or the selected candidate lacks non-empty runtime ABI identity ->
  fail before output.
- Construction-template adapter dynamic mode lacks the route-local candidate
  validator, selected candidate runtime ABI identity, runtime ABI kind,
  runtime glue role, ordered ABI parameter signature, or required metadata
  evidence -> fail before object, header, or bundle output.
- Static runtime ABI identity mode lacks configured runtime ABI/name -> fail
  before output.
- Selected candidate has wrong route id, artifact kind, origin, selected
  variant, emission kind, lowering boundary, runtime ABI field, or runtime glue
  role -> fail before output.
- Candidate ordered runtime ABI parameters differ from the configured signature
  by role, type, ownership, name, arity, or order -> fail before output.
- Required artifact metadata key is missing or has a stale value -> fail before
  output.
- Candidate metadata contains descriptor/direct-C/source-export/compute-body
  residue -> fail before output.
- Object/header bundle construction sees no selected object candidate -> no
  composite match; sees multiple selected object candidates, mixed candidate
  groups, stale routes, wrong origin, mismatched runtime ABI signature, or a
  selected candidate rejected by the route-local preflight -> fail before
  bundle record or artifact output.
- Materialized EmitC handoff lacks route/call source provenance, has non-EmitC
  residue, lacks the expected `emitc.func`, has multiple `emitc.func` roots, or
  has function arity different from the selected ABI parameter signature ->
  fail before output.

### 5. Good/Base/Bad Cases

- Good: Toy and TensorExtLite provide only local config and metadata evidence
  requirements, then call the common header validator/exporter through
  object-backed header composites after validating their selected object
  candidates. The common helper renders the declaration header; the
  family-owned object route remains the artifact handoff authority.
- Good: RVV provides local object route config, RVV-specific candidate
  preflight, dynamic runtime ABI identity mode, and required RVV provenance
  evidence; the production header exporter calls the common declaration-only
  header helper while RVV keeps object packaging and bundle metadata local.
- Good: RVV and TensorExtLite provide local object/header bundle config and
  consume the common object/header bundle construction helper for their
  materialized EmitC object exporter plus object-backed header composite
  registration, while object packaging, route payloads, typed role validation,
  and extension evidence remain plugin-owned.
- Good: RVV may consume the construction-template artifact adapter with dynamic
  selected variant and runtime ABI identity, so source-front-door selected
  variants such as add/sub/mul keep their candidate-owned callable ABI names
  while the adapter owns only generic object/header/bundle wiring.
- Good: Toy may consume the same common object/header bundle construction
  helper once its selected typed role-op path materializes through the
  plugin-owned EmitC route. Toy may carry a target-export-owned runtime element
  count parameter, and the object/header/bundle records must preserve the same
  ordered parameter signature.
- Good: Template may also consume the common object/header bundle construction
  helper as a construction-template example after its selected typed role-op
  path materializes through the plugin-owned EmitC route. Its local object
  callback may only prove generated C++ relocatable packaging; it must not
  claim runtime, hardware, intrinsic, correctness, or performance behavior.
- Base: TensorExtLite may carry an ordered source-op/source-role evidence
  sequence because its selected route contains multiple typed role ops.
- Base: Toy may carry one target-export-owned runtime element count parameter;
  TensorExtLite may carry an empty ABI parameter signature and therefore export
  `void function(void);`.
- Bad: common target code branches on Toy, TensorExtLite, RVV, operation names,
  role graph strings, intrinsic names, or metadata keys to decide computation
  semantics.
- Bad: an object-backed header helper has no route-local candidate validation
  callback, or derives runtime ABI identity from a route name instead of the
  selected candidate and plugin-owned preflight.
- Bad: a construction-template adapter forces a static selected variant or
  static runtime ABI identity for a route family whose selected candidate owns
  the concrete operation ABI.
- Bad: a header helper prints C/C++ compute bodies, direct source exporters,
  descriptor-derived loops, or runtime execution code.

### 6. Tests Required

- C++ coverage proving at least two production extension-family target
  exporters consume the common materialized EmitC header validator/exporter.
- C++ negative coverage for missing/stale route metadata, wrong runtime ABI
  fields, unexpected or misordered ABI parameters, forbidden metadata, and
  missing plugin registration.
- lit coverage for Toy positive object-backed header and bundle output, and
  TensorExtLite positive object-backed header output, through the target
  artifact front doors.
- lit coverage proving header output includes origin plugin, selected variant,
  selected route, runtime ABI kind/name, ordered ABI parameters when present,
  source-op/interface provenance, construction protocol, semantic role graph,
  and typed role evidence.
- Focused RVV target artifact tests proving RVV object/header/bundle behavior
  is not regressed by the common header helper, including evidence that the RVV
  header production path is emitted by the common declaration-only helper while
  object packaging and bundle component metadata remain RVV-owned.
- Focused common target artifact tests proving the object/header bundle
  construction helper is code-consumed by at least RVV, Toy, TensorExtLite,
  and Template when those routes opt in, accepts zero-argument and non-empty
  ordered runtime ABI signatures when both components agree, and fails closed
  for missing
  materialized EmitC provenance, mismatched ABI signatures, mixed or ambiguous
  candidates, stale descriptor/direct-C/source-export residue, and unsupported
  plugin routes.

### 7. Wrong vs Correct

Wrong:

```text
Toy target file / TensorExtLite target file
  -> copied candidate validators
  -> copied EmitC function-boundary checks
  -> copied declaration header renderers
```

Correct:

```text
plugin-local config and evidence keys
  -> common materialized EmitC header candidate validator
  -> common selected EmitC materialization and function-boundary check
  -> common declaration-only header renderer
```

## Scenario: Runtime-Callable C ABI Linkage

### 1. Scope / Trigger

Use this contract when a selected materialized EmitC target artifact route
publishes a runtime-callable C header, a RISC-V relocatable object, or a
coherent object+header bundle for external consumption.

This scenario is triggered by any route whose artifact metadata or bundle
metadata claims a callable C ABI, including names such as
`runtime-callable-c-header` or `*-callable-c-abi.v1`.

### 2. Signatures

- Common materializer option:
  `TCRVEmitCMaterializationOptions::emitExternC`.
- EmitC function linkage surface:
  `emitc.func ... attributes {specifiers = ["extern", "\"C\""]}`.
- Generated C header C++ compatibility wrapper:
  `#ifdef __cplusplus` / `extern "C" {` / `#endif`.
- Expected exported object symbol:
  the unmangled selected function name, for example
  `tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add`.

### 3. Contracts

- Runtime-callable C headers must be usable from both C and C++ callers.
- A generated header must wrap public declarations in an `extern "C"` guard
  when included from C++, while remaining valid C when included from C.
- A matching relocatable object must define the same public function with C
  linkage. It must not rely on C++ name mangling for the callable ABI.
- Target artifact materialization may still use the MLIR EmitC C/C++ emitter
  and a C++ source suffix internally, but callable boundary functions intended
  for object/header export must carry `emitExternC` into the materialized
  `emitc.func` specifiers.
- Runtime ABI name, ordered ABI parameters, header declaration, bundle index,
  and object symbol table must all describe the same callable boundary.
- C linkage does not authorize direct C semantic exporters, handwritten compute
  bodies, descriptor-driven route selection, source-export routes, or common
  code branches on RVV/IME/offload semantics.

### 4. Validation & Error Matrix

- Header declares a runtime-callable C ABI but lacks C++ `extern "C"` guards
  -> reject in review or update header tests before accepting the route.
- Object symbol table contains only a C++-mangled selected function symbol for
  a callable C ABI route -> route is not externally consumable as C ABI.
- Header declaration name and object global symbol disagree -> fail the
  external ABI proof before claiming runtime/correctness evidence.
- Bundle index runtime ABI parameter order disagrees with the header
  declaration or harness call order -> fail before remote execution evidence.
- External C harness cannot compile and link against the generated header and
  generated object on the target machine -> no callable C ABI claim.

### 5. Good/Base/Bad Cases

- Good: RVV vector-source front-door selected add no longer exports a positive
  header/object/harness artifact through the legacy i32 route. Tests assert
  unsupported/no route until the corrected vector-level route surface supplies a
  new non-legacy artifact authority.
- Base: C++ callers may include the same generated header; the guard preserves
  the same C ABI symbol rather than choosing a C++ ABI.
- Bad: the generated object exposes only a symbol such as
  `_Z57tcrv_emitc_vector_source_kernel_vector_source_rvv_i32_add...` while the
  route calls itself `runtime-callable-c-header` or `callable-c-abi`.

### 6. Tests Required

- Focused local target artifact tests must inspect the generated object symbol
  table with `llvm-readobj --symbols` or equivalent and assert the unmangled
  selected function name is present.
- Header lit/C++ tests must assert the `extern "C"` guard is present around the
  public declaration for runtime-callable C header routes.
- Bundle tests must continue to check object/header component coherence,
  external ABI name, runtime ABI kind/name, and ordered ABI parameters.
- Any runtime/correctness claim must include real `ssh rvv` evidence where a C
  harness includes the generated header, links the generated object, executes,
  and observes a bounded `PASS`.

### 7. Wrong vs Correct

Wrong:

```text
EmitC C++ emitter output
  -> C++-mangled object symbol
  -> header called runtime-callable C
  -> only C++ harness can link
```

Correct:

```text
selected materialized EmitC artifact route
  -> emitc.func with extern "C" specifier
  -> generated header with C++ extern guard
  -> object with unmangled C symbol
  -> external C harness links and runs on ssh rvv
```
