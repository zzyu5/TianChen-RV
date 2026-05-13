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
    - tensorext_intrinsics

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

emitc_route:
  headers:
    - tensorext_intrinsics.h
  intrinsic_map:
    config: __tensorext_config
    load_frag: __tensorext_load_frag
    mma: __tensorext_mma
    store_frag: __tensorext_store_frag

local_passes:
  - canonicalize-config
  - verify-fragment-layout
  - legalize-intrinsic-shape

evidence:
  - parse_verify
  - capability
  - interface
  - selected_boundary_or_route
  - emitc_lowering
  - generated_c_compile
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
EmitC lowering skeleton
tests
```

The manifest should be sufficient for tooling or RAG-assisted generation to
create a reviewable skeleton, but the generated skeleton is not success by
itself. A family is accepted only when the evidence profile proves the family
declaration, interface realization, and EmitC route are executable.

## Scenario: Executable Construction Template Manifest

### 1. Scope / Trigger

Use this contract when the repository exposes a compiler-owned construction
template for a future extension family. The template must be consumed by C++
plugin, target/export, and test code; a prose checklist or passive metadata
file is not enough.

### 2. Signatures

- C++ manifest entry:
  `const TemplateConstructionManifest &getTemplateConstructionManifest()`.
- C++ verifier:
  `llvm::Error verifyTemplateConstructionManifest(const
  TemplateConstructionManifest &manifest)`.
- Manifest payload:
  protocol version, archetype, semantic role graph, family declaration,
  semantic roles with common-interface realization, EmitC route mapping, and
  evidence profile.
- Generated-output route builder:
  `llvm::Expected<TemplateGeneratedOutputRoute>
  buildTemplateGeneratedOutputRoute(const TemplateConstructionManifest
  &manifest)`.
- Generated-output payload:
  deterministic function name, required header, ordered role steps, each
  step's semantic role, role order, family operation name, common-interface
  realization, EmitC call name, and source-like call line.

### 3. Contracts

- Plugin proposal materialization must attach manifest-derived construction
  fields to the materialized variant.
- Plugin legality must validate those fields against the C++ manifest before
  readiness or emission planning accepts the variant.
- Emission planning must serialize construction selected-plan metadata for the
  protocol, archetype, role graph, common-interface realization, EmitC route,
  and evidence profile.
- Target artifact validation must consume the same manifest-derived route
  metadata and fail closed for stale selected-plan fields.
- Generated construction output must print the archetype, role graph, family
  declaration, interface realization, EmitC mapping, and evidence profile.
- Generated construction output must also include a deterministic source-like
  role-graph-to-EmitC skeleton derived from the manifest's role-to-call
  mapping. The skeleton is construction evidence only; it is not runtime ABI
  glue, linked code, hardware execution, correctness, or performance evidence.

### 4. Validation & Error Matrix

- Empty protocol, archetype, role graph, family fields, route fields, or
  evidence profile -> manifest verifier error.
- Non-contiguous role ordering or duplicate roles -> manifest verifier error.
- Role missing `TCRVExtensionOpInterface` or `TCRVEmitCLowerableInterface` ->
  manifest verifier error.
- Role missing its role-specific common interface, such as config, memory, or
  compute interface -> manifest verifier error.
- Role-to-EmitC mapping missing a role, reordering roles, duplicating roles, or
  using a non-C-identifier call name -> generated-output route builder error.
- Role common-interface data disagreeing with the common-interface realization
  summary -> manifest verifier error.
- Materialized variant missing construction metadata -> plugin legality error.
- Selected-plan metadata value disagrees with route registration metadata ->
  target artifact preflight error before generated output.

### 5. Good/Base/Bad Cases

- Good: Template plugin proposal, legality, emission plan, target exporter,
  and generated artifact all consume one C++ manifest.
- Base: Existing RVV-specific route mappings remain plugin/target-owned and may
  be used as reference evidence without expanding RVV coverage.
- Bad: A YAML/Markdown-only manifest, a metadata-only artifact that is not
  validated by code, or a core pass branch on a concrete extension name.

### 6. Tests Required

- C++ test asserting the manifest shape and agreement with plugin capability,
  variant, emission-plan, and selected-plan metadata.
- C++ test asserting the generated-output route's ordered role steps,
  role-specific common-interface realization, EmitC call mapping, and
  fail-closed behavior for reordered, missing, or mismatched route data.
- lit/FileCheck test proving generated artifact output includes construction
  protocol, archetype, role graph, family, interface, EmitC route, and evidence
  fields.
- lit/FileCheck test proving generated artifact output includes a deterministic
  source-like role-graph-to-EmitC skeleton derived from the manifest mapping,
  including at least the compute role.
- Target artifact registry/preflight test proving selected-plan route metadata
  requirements are registered and reject stale values.
- Focused regression that existing built-in plugin registration still reaches
  RVV plugin routes when the Template path changes.

### 7. Wrong vs Correct

Wrong:

```text
spec checklist -> passive manifest text -> target artifact prints comments
```

Correct:

```text
C++ construction manifest
  -> plugin proposal metadata
  -> plugin legality validation
  -> emission-plan selected metadata
  -> target route preflight
  -> generated construction artifact
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

The existing descriptor and microkernel surfaces are bounded implementation
debt. They may be used only as migration evidence or compatibility diagnostics
while the production route moves to extension family ops, common interfaces,
and EmitC route mapping.

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
