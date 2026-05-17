# Extension Family Plugin Template

This is the durable template for adding a new RISC-V extension family such as
IME, TensorExt, or a vendor/custom extension. A plugin is not an independent
backend. It contributes one extension family to the unified TCRV RISC-V MLIR
system.

## Extension-Family Plugin Construction Protocol

The construction protocol is the durable shape for adding a new RISC-V
extension to TCRV. It is stronger than a file checklist: it defines how an
extension becomes a TCRV extension family that common passes, interfaces, and
EmitC routes can consume.

```text
extension archetype
  -> semantic role graph
  -> extension family declaration
  -> common interface realization
  -> EmitC route mapping
  -> evidence profile
```

### 1. Extension Archetype

The archetype classifies the structural kind of extension being added. It
guides the expected roles, ops, interface coverage, EmitC mapping, and tests.
Examples include:

- `rvv-finite-binary` or broader vector-binary/vector-dataflow extensions;
- `fragment-mma-like` extensions such as IME or TensorExt;
- `runtime-offload` extensions whose hardware work is reached through a
  runtime ABI;
- `custom-riscv-extension` families that expose vendor intrinsics or builtins.

The archetype is not the implementation. It is the first design commitment that
prevents every new extension from re-litigating the same architecture choices.

### 2. Semantic Role Graph

The semantic role graph names the family-local execution roles and their
ordering. It must describe extension execution roles, not high-level tensor
semantics and not descriptor fields.

Examples:

```text
RVV: configure/setvl -> load -> compute -> store
IME/TensorExt: config -> load_frag -> mma -> store_frag
Offload: bind -> call -> wait
```

Each role should map to one or more extension family ops and to the common
interfaces those ops implement. Core passes should see role/interface data, not
family-specific operation names.

### 3. Extension Family Declaration

The family declaration records the capability ids, family namespace, ops,
types, attributes, required runtime/toolchain facts, and any target-owned
artifact routes. The declaration contributes a family to the unified TCRV
system; it does not create a separate backend.

Implementation may organize files under family-specific dialect/plugin/target
directories, but those files remain part of one TCRV dialect suite and one
TCRV orchestration model.

### 4. Common Interface Realization

Extension family ops must realize the common TCRV interfaces needed by their
roles, such as extension/config/resource/memory/compute/EmitC-lowerable
interfaces. Common passes must query those interfaces rather than branching on
`RVV`, `IME`, `TensorExt`, or vendor names.

Family-local verifier, legality, canonicalization, and lowering hooks are
allowed. They must attach through shared interfaces or plugin registry hooks
instead of rewriting core orchestration passes for one extension.

### 5. EmitC Route Mapping

The current production route is:

```text
extension family ops
  -> EmitC ops
  -> C/C++ emitter
  -> intrinsic / vendor builtin / runtime C ABI
  -> native compiler
```

Each family provides only the local mapping from family ops and roles to
headers, intrinsic/runtime-call names, operand/result mapping, ABI mapping, and
compile requirements. The common route should remain reusable by RVV, IME,
TensorExt, Offload, scalar fallback, and future vendor/custom families.

### 6. Evidence Profile

The evidence profile states how a new extension proves it is really integrated.
The minimum profile should include parse/verify, capability recognition and
rejection, interface verification, selected boundary or route production,
EmitC lowering, generated C/C++ compile when the route claims source output,
and runtime evidence when real hardware/runtime claims are made.

RVV runtime/correctness/performance evidence requires `ssh rvv`. Other
families require target-appropriate hardware/toolchain evidence before making
runtime or performance claims.

## Required Contributions

A new extension family plugin must provide:

1. Extension Manifest.
2. Capability declarations.
3. Extension family ops, types, and attributes.
4. TCRV interface implementations.
5. Local legality and verifier hooks.
6. Local canonicalization or legalization hooks when needed.
7. EmitC lowering mapping.
8. Tests.

These are construction-protocol outputs. They are not a license to skip the
archetype, role graph, interface, EmitC route, or evidence-profile decisions.

The plugin must not modify these core orchestration surfaces for a
family-specific special case:

- core capability verification pass;
- core variant selection pass;
- core dispatch/fallback pass;
- core ABI envelope logic;
- core EmitC lowering framework.

Core changes are allowed only when they extend a public interface or generic
orchestration surface for all families.

## TCRV Common Interfaces

Common passes should operate through interfaces rather than family-name
branches. The long-term interface set includes:

```text
TCRVExtensionOpInterface
TCRVCapabilityOpInterface
TCRVConfigOpInterface
TCRVResourceOpInterface
TCRVMemoryOpInterface
TCRVComputeOpInterface
TCRVEmitCLowerableInterface
```

Interface responsibilities:

- expose required capability ids and capability predicates;
- expose config scope and selected extension state;
- expose resource kind, memory role, and compute primitive kind;
- expose runtime ABI and buffer/pointer mapping needs;
- expose EmitC lowering metadata without exposing family-specific logic to core
  passes.

## Common Passes

The TCRV pass framework should favor common passes that consume these
interfaces:

```text
tcrv-verify-capability
tcrv-verify-extension-config
tcrv-verify-resource-legality
tcrv-canonicalize-extension-state
tcrv-select-route
tcrv-lower-extension-to-emitc
```

These passes must not implement:

```text
if RVV
if IME
if TensorExt
if Offload
```

They query:

```text
required capability
config scope
resource kind
memory role
compute primitive kind
EmitC lowering mapping
```

Extension families may provide local verification, canonicalization,
legalization, and mapping hooks, but those hooks are attached through the
shared interfaces and registry.

## Extension Manifest

Each new family should define a machine-readable or semi-machine-readable
manifest. The manifest is the machine-readable entry point for the construction
protocol and the source for future RAG-assisted skeleton generation;
descriptors are not.

Example:

```yaml
extension: tensorext
kind: tensorext
archetype: fragment-mma-like

capabilities:
  - tensorext
  - tensorext.fp16
  - tensorext.int8

family:
  name: tensorext
  namespace: tcrv_tensorext
  required_toolchain:
    - clang
  required_runtime:
    - tensorext_runtime_abi

types:
  - name: frag
    role: tensor-fragment
  - name: accfrag
    role: accumulator-fragment

attributes:
  - name: layout
    role: fragment-layout
  - name: shape
    role: intrinsic-tile-shape

semantic_roles:
  - role: config
    order: 0
    description: configure extension state
  - role: load_frag
    order: 1
    description: memory to fragment
  - role: mma
    order: 2
    description: fragment compute
  - role: store_frag
    order: 3
    description: fragment to memory

ops:
  - name: config
    role: config
    interfaces:
      - TCRVExtensionOpInterface
      - TCRVConfigOpInterface
      - TCRVEmitCLowerableInterface
  - name: load_frag
    role: load_frag
    interfaces:
      - TCRVExtensionOpInterface
      - TCRVMemoryOpInterface
      - TCRVResourceOpInterface
      - TCRVEmitCLowerableInterface
  - name: mma
    role: mma
    interfaces:
      - TCRVExtensionOpInterface
      - TCRVComputeOpInterface
      - TCRVResourceOpInterface
      - TCRVEmitCLowerableInterface
  - name: store_frag
    role: store_frag
    interfaces:
      - TCRVExtensionOpInterface
      - TCRVMemoryOpInterface
      - TCRVResourceOpInterface
      - TCRVEmitCLowerableInterface

route_boundary:
  route_id: tensorext-no-active-emitc-route
  emission_kind: no-active-route
  artifact_kind: unsupported-emission-diagnostic
  runtime_abi: unsupported-emission-runtime-abi
  runtime_glue_role: no-runtime-glue-unsupported

local_passes:
  - canonicalize-config
  - verify-fragment-layout
  - legalize-intrinsic-shape

evidence:
  - parse_verify
  - capability
  - interface
  - selected_boundary_or_route
  - emitc_route_mapping
```

RAG-assisted extension addition should use this manifest to generate:

```text
ODS skeleton
C++ family registration
plugin class
capability provider
interface implementations
verifier skeleton
local pass skeleton
fail-closed descriptor-erasure checks
tests
```

The manifest should be sufficient for tooling or RAG-assisted generation to
create a reviewable skeleton, but the generated skeleton is not success by
itself. A family is accepted only when the evidence profile proves the family
declaration and interface realization. Executable routes require separate
materialized EmitC module and runtime ABI evidence; construction metadata must
not stand in for that route.

## Scenario: Construction Template Manifest After Source-Skeleton Deletion

### 1. Scope / Trigger

Use this contract when the repository exposes a compiler-owned construction
template for a future extension family after deletion of construction
metadata-derived source skeletons. The template must be consumed by C++ plugin
and test code; a prose checklist or passive metadata file is not enough.

This scenario is a construction-surface contract only. It does not authorize a
construction manifest to synthesize C/C++ source, fake intrinsic headers,
runtime-call skeletons, source-like call lines, target artifacts, or executable
plugin output.

### 2. Signatures

- C++ manifest entry:
  `const TemplateConstructionManifest &getTemplateConstructionManifest()`.
- C++ verifier:
  `llvm::Error verifyTemplateConstructionManifest(const
  TemplateConstructionManifest &manifest)`.
- Manifest payload:
  protocol version, archetype, semantic role graph, family declaration,
  semantic roles with common-interface realization, fail-closed descriptor
  erasure checks,
  and evidence profile.
- Typed role/interface realization:
  `const TemplateTypedRoleGraphRealization
  &getTemplateTypedRoleGraphRealization()`.
- Typed realization verifier:
  `llvm::Error verifyTemplateTypedRoleGraphRealization(const
  TemplateConstructionManifest &manifest, const
  TemplateTypedRoleGraphRealization &realization)`.
- Minimal ODS role-op boundary:
  `tcrv_template.compute_skeleton`.
- Role-op interface:
  `TCRVEmitCLowerableOpInterface`, exposing source op name and source role.
- ODS/interface-backed role verifier:
  `llvm::Error verifyTemplateComputeRoleOpInterface(const
  TemplateConstructionManifest &manifest, const
  TemplateTypedRoleGraphRealization &realization, mlir::Operation
  *computeRoleOp)`.
- Role-op payload:
  selected kernel and variant identity, origin plugin, selected-path role,
  role-op status, required capability refs, typed role id, role order, source
  role, and role-specific interface.

### 3. Contracts

- Plugin proposal materialization must attach manifest-derived construction
  fields to the materialized variant.
- Plugin legality must validate those fields against the C++ manifest before
  readiness or emission planning accepts the variant.
- Emission planning may serialize construction protocol identity only through
  typed emission-plan fields and plugin-local typed role operations for the
  protocol, archetype, role graph, common-interface realization, fail-closed
  descriptor-erasure checks, typed role/interface realization, and evidence
  profile.
- The compute role-op must implement `TCRVEmitCLowerableOpInterface`;
  construction validation must cross-check the source op and source role
  against the typed compute role realization.
- Construction protocol code must not expose generated-output route structs,
  generated-output route builders, source printers, fake required headers,
  role-to-call maps, direct `__tcrv_*` call names, or source-line fields as
  active API.
- Tests must prove the manifest and typed role/interface realization validate
  as non-source construction declarations. They must not assert source-like
  skeleton output or generated C/C++ text from construction metadata.

### 4. Validation & Error Matrix

- Empty protocol, archetype, role graph, family fields, route fields, or
  evidence profile -> manifest verifier error.
- Non-contiguous role ordering or duplicate roles -> manifest verifier error.
- Role missing `TCRVExtensionOpInterface` or `TCRVEmitCLowerableInterface` ->
  manifest verifier error.
- Role missing its role-specific common interface, such as config, memory, or
  compute interface -> manifest verifier error.
- Role common-interface data disagreeing with the common-interface realization
  summary -> manifest verifier error.
- Typed role realization missing a role object, reordering role objects,
  duplicating typed role ids, using stale family operation identity, or using a
  stale role-specific interface -> typed-realization verifier error.
- Compute role op missing `TCRVEmitCLowerableOpInterface` -> construction
  verifier error.
- Interface source op name or source role disagreeing with the typed compute
  role realization -> construction verifier error.
- Compute role op stale typed role id, role-specific interface, source role, or
  selected variant -> dialect verifier or construction verifier error.
- Materialized variant missing construction metadata -> plugin legality error.

### 5. Good/Base/Bad Cases

- Good: Template plugin proposal, legality, emission plan, and typed
  role/interface tests consume one C++ manifest without constructing source
  output.
- Base: Existing RVV-specific route mappings remain plugin/target-owned and may
  be used as reference evidence without expanding RVV coverage.
- Bad: A YAML/Markdown-only manifest, a generated source skeleton from
  construction metadata, a fake intrinsic header/call list, or a core pass
  branch on a concrete extension name.

### 6. Tests Required

- C++ test asserting the manifest shape and agreement with plugin capability,
  variant, emission-plan typed fields, and plugin-local typed role operations.
- C++ test asserting typed role/interface realization agreement with the
  manifest and fail-closed behavior for missing, reordered, stale, or
  mismatched typed role/interface data.
- C++ test asserting ODS role-op interface identity against the manifest,
  typed role realization, `TCRVEmitCLowerableOpInterface` source op and source
  role, role-specific interface, and missing-interface failure.
- lit/FileCheck test proving `tcrv_template.compute_skeleton` parses/verifies
  and rejects stale source role, stale typed role, stale role-specific
  interface, stale selected variant, or generic compute attributes.
- Focused ref-scan proving generated-output route APIs, fake intrinsic headers,
  role-to-call maps, direct `__tcrv_*` construction call names, and source-line
  skeleton output are absent from active construction protocol code and tests.
- Focused regression that existing built-in plugin registration still reaches
  RVV plugin routes when the Template path changes.

### 7. Wrong vs Correct

Wrong:

```text
construction manifest -> role-to-call map -> source-like C/C++ skeleton
```

Correct:

```text
C++ construction manifest
  -> plugin proposal metadata
  -> plugin legality validation
  -> ODS role-op boundary with interface provenance
  -> emission-plan selected metadata
  -> fail closed until a future materialized EmitC module route exists
```

## Scenario: Executable EmitC Construction Template

### 1. Scope / Trigger

Use this contract when a construction-template extension family graduates one
bounded selected path from passive construction metadata to a materialized common
EmitC module. The route must remain plugin-owned and must consume typed
role/interface realization before route construction.

This scenario stops at a verified common EmitC module unless target/export work
is explicitly part of the task. A materialized EmitC module is not a runtime,
correctness, performance, object-export, or hardware execution claim.

### 2. Signatures

- Plugin route hook:
  `llvm::Error buildVariantEmitCLowerableRoute(const
  VariantEmitCLowerableRequest &, TCRVEmitCLowerableRoute &)`.
- Selected role-op boundary:
  an ODS extension-family operation implementing
  `TCRVEmitCLowerableOpInterface`.
- Route payload:
  `TCRVEmitCLowerableRoute` with route id, source-op provenance, headers, and
  bounded `emitc.call_opaque` steps.
- Route verifier/materializer:
  `verifyTCRVEmitCLowerableRouteMaterializesToEmitC(...)` and the common
  EmitC lowerable materialization pass.

### 3. Contracts

- The materialized variant must carry construction protocol, archetype, semantic
  role graph, common-interface realization, typed-role realization, route
  mapping, and evidence-profile metadata.
- Plugin legality must reject missing or stale construction metadata before
  readiness, boundary materialization, emission planning, or route building.
- The selected lowering boundary must be a typed plugin-local role op with
  selected kernel/variant identity, selected-path role, required capabilities,
  typed role id, source role, role order, and role-specific interface.
- Common/core boundary discovery may recognize the selected role op only through
  `TCRVEmitCLowerableOpInterface` plus generic selected-path attrs. It must not
  branch on a concrete extension family.
- The plugin route builder must validate the typed role/interface realization
  and selected boundary before constructing `TCRVEmitCLowerableRoute`.
- Unsupported legacy template boundaries, stale route mappings, wrong typed
  roles, wrong interfaces, duplicate selected boundaries, or missing selected
  boundaries must fail closed.

### 4. Validation & Error Matrix

- Missing construction metadata on selected variant -> plugin legality error.
- Stale route mapping -> selected-boundary or route-construction error before
  EmitC materialization.
- Selected role op missing, duplicated, or mismatched by selected variant/role ->
  route-construction error.
- Selected role op stale typed role, source role, role-specific interface, role
  order, or required capabilities -> dialect/construction validation error.
- Legacy `*.lowering_boundary` for the template family without the typed role op
  -> unsupported route error, not a compatibility fallback.
- Route materializes to EmitC but no target artifact exporter exists -> target
  front-door/coherence must fail at the target-export boundary.

### 5. Good/Base/Bad Cases

- Good: selected variant -> typed role-op boundary ->
  `TCRVEmitCLowerableRoute` -> common EmitC module.
- Base: route omits target object export and hardware claims when the bounded
  task only proves common EmitC materialization.
- Bad: construction metadata directly prints C/C++, core code checks `if Toy` or
  `if RVV`, or a legacy no-active-route boundary is treated as executable.

### 6. Tests Required

- C++ test proving the plugin manifest, typed-role realization, selected role
  op, route mapping, and route materializer agree.
- lit/FileCheck positive proving the selected path reaches a materialized EmitC
  module through the common materializer.
- lit/FileCheck negatives for missing boundary, wrong typed role/interface,
  stale route mapping, unsupported legacy boundary, and target/export boundary
  absence when export is intentionally out of scope.
- Changed-surface scan proving descriptor-driven computation, direct C source
  printers, Python compiler-core logic, and family-name branches in common/core
  orchestration were not introduced.

### 7. Wrong vs Correct

Wrong:

```text
construction metadata -> direct C/source exporter -> target claim
```

Correct:

```text
construction manifest
  -> plugin legality
  -> typed role-op boundary via common interface
  -> plugin-owned TCRVEmitCLowerableRoute
  -> common EmitC module
  -> separate target/export validation if required
```

## Scenario: Reusable Executable Construction Conformance Surface

### 1. Scope / Trigger

Use this contract when an executable extension-family construction route needs
common validation for selected role-sequence materialization, selected
lowering-boundary coherence, and construction artifact metadata. The surface is
for generic construction-template invariants only; family semantics, intrinsic
names, role callees, ABI identity, and target export callbacks remain
plugin-owned.

### 2. Signatures

- Executable role model:
  `plugin::construction::ExecutableRoleStep`.
- Selected role-sequence request:
  `plugin::construction::SelectedExecutableRoleSequenceSpec`.
- Selected role-sequence inspection:
  `plugin::construction::inspectSelectedExecutableRoleSequence(...)`.
- Complete sequence collection:
  `plugin::construction::collectSelectedExecutableRoleSequence(...)`.
- Selected lowering-boundary request:
  `plugin::construction::SelectedLoweringBoundaryConformanceSpec`.
- Selected lowering-boundary verifier:
  `plugin::construction::verifySelectedLoweringBoundaryConformance(...)`.
- Artifact metadata verifier:
  `plugin::construction::verifyConstructionArtifactMetadata(...)`.

### 3. Contracts

- `ExecutableRoleStep` may name role id, typed operation name,
  role-specific common interface, EmitC lowerable interface, EmitC callee, and
  order. It must not encode descriptor fields, direct C source text, runtime
  results, hardware evidence, or family compute semantics.
- `inspectSelectedExecutableRoleSequence` may inspect a plugin-owned selected
  variant block using generic `selected_variant` and `role` attributes plus
  plugin-supplied role steps. It may detect zero, partial, complete, duplicate,
  or reordered role materialization without knowing the concrete extension
  family.
- `collectSelectedExecutableRoleSequence` is the fail-closed route-builder
  entry: it succeeds only for exactly one selected op per role in role order.
- `SelectedLoweringBoundaryConformanceSpec` validates only generic selected
  boundary coherence: source kernel, selected variant, origin, role, status,
  required capabilities, and plugin-supplied extra string attributes. Extra
  attributes may preserve ABI or handoff identity, but the common verifier must
  not interpret those values as compute semantics.
- `verifyConstructionArtifactMetadata` checks exact ordered metadata evidence
  supplied by the plugin. The common verifier may check equality and structured
  error reporting, but it must not infer source ops, routes, ABI, or artifact
  meaning from metadata keys.

### 4. Validation & Error Matrix

- Missing selected variant role block -> selected role-sequence inspection
  error.
- Duplicate selected role op for the same role step -> selected role-sequence
  inspection error.
- Partial selected role sequence -> plugin readiness must fail before adding a
  second replacement sequence or proceeding to route construction.
- Missing selected role op during route build -> complete sequence collection
  error.
- Reordered selected role ops -> complete sequence collection error.
- Missing or stale boundary source kernel, selected variant, origin, role,
  status, required capabilities, or plugin-supplied extra attribute ->
  selected-boundary conformance error.
- Missing, extra, reordered, stale-key, or stale-value artifact metadata ->
  construction artifact metadata error.

### 5. Good/Base/Bad Cases

- Good: TensorExtLite provides local role steps and ABI/handoff expectations,
  then consumes the common role-sequence, boundary, and artifact metadata
  validators on its production selected path.
- Base: RVV, Toy, Template, IME, Offload, or future plugins may provide their
  own local role steps and target route data while using the same conformance
  surface.
- Bad: common construction code branches on TensorExtLite/RVV/IME/Offload,
  interprets fragment semantics, maps role names to intrinsic calls, emits
  source text, or treats metadata as executable compute authority.

### 6. Tests Required

- C++ coverage proving at least one production plugin consumes the common
  selected role-sequence and selected-boundary validators, not only a test
  fixture.
- C++ negative coverage for duplicate, partial, missing, or reordered selected
  role sequences.
- C++ or lit coverage proving selected-boundary conformance rejects stale
  source kernel, selected variant, origin, role, status, required capability,
  or plugin-owned ABI/handoff attribute.
- C++ coverage proving construction artifact metadata rejects stale keys and
  stale values through the common verifier.
- Focused changed-surface scan proving no descriptor-driven computation,
  direct C/source-export route, Python compiler-core path, or common/core
  family semantic branch was introduced.

### 7. Wrong vs Correct

Wrong:

```text
common construction code sees TensorExtLite role name
  -> chooses fragment-MMA semantics or callee
  -> emits/exports source or target artifact
```

Correct:

```text
plugin-owned role steps and route data
  -> common selected role-sequence and boundary conformance
  -> plugin-owned TCRVEmitCLowerableRoute
  -> common EmitC materialization and separate target/export validation
```

## Scenario: Template Object And Bundle Packaging Bridge

### 1. Scope / Trigger

Use this contract when the construction-template family demonstrates the target
artifact packaging side of an already selected materialized EmitC route.
Template may be a reusable object/header/bundle construction example only after
the selected path has passed through the typed role-op boundary and the
plugin-owned `TCRVEmitCLowerableRoute`.

This is a local compile/package proof for generated C++ and a manifest bridge.
It is not a Template runtime, correctness, performance, RISC-V hardware,
intrinsic, vendor, or offload execution claim.

### 2. Signatures

- Template route payload:
  `TemplateEmitCConstructionRoute` with object route id, header route id,
  object artifact kind, header artifact kind, bundle component group,
  object handoff kind, runtime ABI kind/name, lowering boundary, and
  EmitC-to-C++ translate route id.
- Template target route accessors:
  `getTemplateMaterializedEmitCTargetArtifactRouteID()`,
  `getTemplateMaterializedEmitCHeaderArtifactRouteID()`, and
  `getTemplateEmitCToCppTranslateRouteID()`.
- Template target support registration:
  `registerTemplateTargetSupportPluginTargetExporterBundles(...)`.
- Common target helper:
  `registerMaterializedEmitCObjectBundleArtifactExporters(...)` over a
  `MaterializedEmitCObjectBundleArtifactConfig`.

### 3. Contracts

- The standalone Template target artifact route may publish exactly one
  selected `riscv-elf-relocatable-object` candidate derived from the selected
  materialized EmitC module and the MLIR EmitC C/C++ emitter.
- Template object packaging may invoke a local native `clang++` compile of the
  generated C++ source to prove relocatable object packaging. This does not
  imply RISC-V target code, remote hardware execution, or runtime correctness.
- The declaration-only `runtime-callable-c-header` route must be an
  object-backed composite derived from the same selected object candidate. It
  must not become a separate metadata-only header authority.
- The bundle manifest must tie object and header records to the same selected
  variant, origin plugin, object route id, header route id, runtime ABI
  kind/name, ordered ABI parameters, component group, lowering boundary,
  construction protocol, source-op/interface provenance, semantic role graph,
  typed role evidence, and object handoff kind.
- Template target support remains plugin-local. Common target code may consume
  the common object/header bundle helper, but it must not add a Template branch
  to infer computation semantics or route provenance.

### 4. Validation & Error Matrix

- Missing selected path, missing materialized EmitC route provenance, stale
  route id, wrong origin plugin, wrong artifact kind, stale lowering boundary,
  or missing runtime ABI identity -> fail before generated bytes are emitted.
- Fallback-only, unsupported, duplicate, or ambiguous selected candidates ->
  fail before object, header, or bundle output.
- Header/object route identity mismatch, component-group mismatch, runtime ABI
  parameter mismatch, missing header route id, missing object/header callback,
  or missing object handoff kind -> fail before registration or bundle output.
- Descriptor, metadata-diagnostic, direct-C, source-export, compute-body, fake
  intrinsic, or handwritten source residue in selected artifact metadata ->
  fail closed before output.
- Missing local native `clang++` for Template object packaging -> emit a
  bounded diagnostic and do not substitute metadata, source-only, or hardware
  evidence.

### 5. Good/Base/Bad Cases

- Good: selected Template variant -> `tcrv_template.compute_skeleton` typed
  role op -> Template-owned EmitC route -> materialized EmitC module -> MLIR
  EmitC C++ source -> local relocatable object -> declaration header -> bundle
  index tying both components to one selected object candidate.
- Base: the Template runtime ABI may have an empty ordered parameter signature;
  header and object bundle records must agree on that zero-argument boundary.
- Bad: construction metadata prints a C body, the header route validates
  without the selected object candidate, object packaging claims `ssh rvv`
  correctness, or common target code branches on Template to choose semantics.

### 6. Tests Required

- C++ tests proving Template target support registers one object exporter and
  one object-backed header composite through the common object/header bundle
  helper.
- C++ negative coverage for stale route metadata, fallback-only candidates,
  duplicate/ambiguous candidates, wrong artifact kind, missing or extra runtime
  ABI parameters, and descriptor/direct-C/source-export residue.
- lit/FileCheck coverage proving generated C++ remains available, header output
  is declaration-only, object output is relocatable when local `clang++` is
  available, and the bundle index records coherent object/header metadata.
- Focused scans proving no descriptor-driven compute authority, source-export
  route, direct C semantic exporter, Python compiler-core path, or
  extension-specific core branch was introduced.

### 7. Wrong vs Correct

Wrong:

```text
construction metadata -> source/export printer -> object/header/bundle claim
```

Correct:

```text
selected Template role op
  -> plugin-owned EmitC route
  -> common materialized EmitC module
  -> MLIR C/C++ emitter
  -> plugin-local object packaging callback
  -> common object-backed header and bundle construction
```

## Fast Plugin Addition

"Fast" plugin addition means reducing architecture decision search. It does
not mean a new extension requires little code.

New extension families still normally need family-specific ops, types, attrs,
verifiers, local legality hooks, EmitC mappings, CMake wiring, and tests. They
should not redesign or fork these core TCRV mechanisms:

- capability collection and verification;
- variant orchestration and selection;
- dispatch/fallback organization;
- runtime ABI envelope;
- route selection;
- common EmitC lowering framework.

Expected primary edits for a new family are:

- extension family files;
- plugin or bundle files;
- EmitC mapping files;
- tests;
- manifest.

Unexpected edits are family-specific semantic changes in core orchestration
passes. If a core edit is needed, it must extend a shared interface or generic
orchestration surface for all families.

## RVV Exemplar

RVV is the first construction-protocol exemplar, not the final upper bound.
The current finite/vector binary path should converge toward:

```text
archetype: rvv-finite-binary
semantic role graph: setvl -> load -> compute -> store
family ops: typed TCRV RVV family ops
route: EmitC RVV intrinsic C/C++
evidence: generated C compile, and ssh rvv for runtime/correctness claims
```

The existing descriptor and direct-export microkernel surfaces are historical
residue, deletion targets, or fail-closed implementation debt. They must not be
used as migration evidence, compatibility architecture, semantic source, or
production route authority. Rebuild work moves through extension family ops,
common interfaces, and EmitC route mapping.

## Descriptor Boundary

Descriptor-driven computation is not part of the Extension-Family Plugin
Construction Protocol. A new extension must not add computation semantics by
adding descriptor fields, descriptor catalogs, or descriptor-to-C exporters.
Computation semantics belong in extension family ops and flow through common
interfaces and the EmitC route.

## Recommended Directory Layout

```text
include/TianChenRV/Plugin/<Ext>/
lib/Plugin/<Ext>/

include/TianChenRV/Dialect/<Ext>/IR/
lib/Dialect/<Ext>/IR/

include/TianChenRV/Target/<Ext>/        optional
lib/Target/<Ext>/                       optional

test/Plugin/<Ext>/
test/Dialect/<Ext>/
test/Target/<Ext>/
```

These directories are implementation organization for one unified TCRV system.
They do not imply independent backends.

## Test Template

New extension family work should include focused tests for:

- capability recognized and rejected when unavailable;
- plugin can register the family ops and capabilities;
- variant or selected boundary appears only with matching capability;
- illegal config fails closed;
- common interfaces are implemented and queried by common passes;
- selected boundary materialization is family-owned;
- EmitC lowering route is family-owned mapping through the common pass;
- generated C compiles with clang/LLVM by default when the route claims it;
- GCC is accepted only as a compatibility path where supported;
- no extension-specific semantic branch is added to core passes.
