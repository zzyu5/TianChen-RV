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
extension op
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

### 5. Good/Base/Bad Cases

- Good: `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, and `tcrv_rvv.i32_mul`
  implement `TCRVEmitCLowerableOpInterface`; RVV target code queries the
  interface, then builds `TCRVEmitCLowerableRoute` with
  `opInterface = "TCRVEmitCLowerableOpInterface"`.
- Base: a target-specific bounded route still owns intrinsic names, vector
  suffixes, C type spellings, and ABI details locally.
- Bad: descriptor strings alone choose the arithmetic intrinsic, or common code
  adds an `if RVV` branch to recover family semantics.

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
  selected `runtime-callable-c-header` route derived from that materialized
  EmitC handoff. This header route is declaration-only: it must not publish
  metadata-only artifact authority, direct C/source-export bodies, TensorExtLite
  object/bundle target artifact export, hardware execution, correctness, or
  performance evidence.
- Bad: two direct RVV variants exist and the exporter chooses whichever direct
  variant happens to be first or only after test reduction.

### 6. Tests Required

- Positive lit coverage for a retained selected-path target artifact route with
  a non-selected sibling variant when such a route exists.
- Positive lit coverage for current RVV explicit typed bodies reaching
  materialized EmitC and the MLIR EmitC C/C++ emitter without object/header
  artifact route authority.
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

## Scenario: Common Materialized EmitC Header Artifact Foundation

### 1. Scope / Trigger

Use this contract when multiple extension-family target exporters need the same
declaration-only runtime-callable C header artifact shape from a selected
materialized EmitC route. The first required consumers are Toy and
TensorExtLite. RVV object/header/bundle behavior may keep RVV-specific object
packaging and composite bundle logic unless it explicitly opts into the common
header helper.

### 2. Signatures

- Common config:
  `target::MaterializedEmitCHeaderArtifactConfig`.
- Common metadata evidence entry:
  `target::MaterializedEmitCHeaderArtifactMetadataEvidence`.
- Common validator:
  `validateMaterializedEmitCHeaderArtifactCandidate(candidate, config)`.
- Common exporter:
  `exportMaterializedEmitCHeaderArtifact(module, os, config)`.
- The config wraps a `SelectedEmitCArtifactRouteConfig` plus plugin-supplied
  header guard, evidence prefix, includes, optional expected selected variant,
  emission kind, lowering boundary, runtime ABI fields, ordered runtime ABI
  parameters, and required artifact metadata evidence.

### 3. Contracts

- Common target code may validate extension-agnostic selected candidate fields
  supplied through the config: route id, artifact kind, origin plugin,
  selected variant identity when configured, emission kind, lowering boundary,
  runtime ABI, runtime ABI kind/name, runtime glue role, ordered ABI parameter
  signature, and required artifact metadata values.
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

### 4. Validation & Error Matrix

- Config lacks route id, artifact kind, origin plugin, route builder, header
  guard, evidence prefix, emission kind, lowering boundary, runtime ABI fields,
  or runtime glue role -> fail before output.
- Config artifact kind is not `runtime-callable-c-header` -> fail before
  output.
- Selected candidate has wrong route id, artifact kind, origin, selected
  variant, emission kind, lowering boundary, runtime ABI field, or runtime glue
  role -> fail before output.
- Candidate ordered runtime ABI parameters differ from the configured signature
  by role, type, ownership, name, arity, or order -> fail before output.
- Required artifact metadata key is missing or has a stale value -> fail before
  output.
- Candidate metadata contains descriptor/direct-C/source-export/compute-body
  residue -> fail before output.
- Materialized EmitC handoff lacks route/call source provenance, has non-EmitC
  residue, lacks the expected `emitc.func`, has multiple `emitc.func` roots, or
  has function arity different from the selected ABI parameter signature ->
  fail before output.

### 5. Good/Base/Bad Cases

- Good: Toy and TensorExtLite provide only local config and metadata evidence
  requirements; both call the same common header validator/exporter through
  their production target artifact exporters.
- Base: TensorExtLite may carry an ordered source-op/source-role evidence
  sequence because its selected route contains multiple typed role ops.
- Base: Toy may carry one target-export-owned runtime element count parameter;
  TensorExtLite may carry an empty ABI parameter signature and therefore export
  `void function(void);`.
- Bad: common target code branches on Toy, TensorExtLite, RVV, operation names,
  role graph strings, intrinsic names, or metadata keys to decide computation
  semantics.
- Bad: a header helper prints C/C++ compute bodies, direct source exporters,
  descriptor-derived loops, or runtime execution code.

### 6. Tests Required

- C++ coverage proving at least two production extension-family target
  exporters consume the common materialized EmitC header validator/exporter.
- C++ negative coverage for missing/stale route metadata, wrong runtime ABI
  fields, unexpected or misordered ABI parameters, forbidden metadata, and
  missing plugin registration.
- lit coverage for Toy and TensorExtLite positive header output through the
  default target artifact front door.
- lit coverage proving header output includes origin plugin, selected variant,
  selected route, runtime ABI kind/name, ordered ABI parameters when present,
  source-op/interface provenance, construction protocol, semantic role graph,
  and typed role evidence.
- Focused RVV target artifact tests proving RVV object/header/bundle behavior
  is not regressed by the common header helper.

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
