# Interfaces And Registry

## Required Plugin Interfaces

Each extension family plugin should provide the following interface set. A
plugin contributes a family to the unified TCRV RISC-V MLIR system; it is not
an independent backend.

These interfaces are compiler interfaces. They must be implemented in C++ against MLIR/LLVM APIs and wired through CMake. Python may launch or audit plugin runs, but it must not implement plugin availability, dialect registration, variant generation, legality, cost, tuning, lowering, or emission.

### CapabilityProvider

```cpp
class CapabilityProvider {
public:
  virtual void registerCapabilities(CapabilityRegistry &registry) = 0;
  virtual LogicalResult detectCapabilities(TargetCapability &target) = 0;
};
```

Responsibilities:

- declare provided capabilities;
- fill target capability from profile, compiler options, and runtime probes;
- declare implication and conflict relations;
- expose target availability.

### FamilyProvider / DialectProvider

```cpp
class DialectProvider {
public:
  virtual void registerDialects(DialectRegistry &registry) = 0;
};
```

Responsibilities:

- RVV plugin registers the RVV extension family. The current concrete
  MLIR namespace is `tcrv_rvv` because MLIR dialect namespaces cannot contain
  `.` characters; the architectural family remains `tcrv.rvv`;
- scalar fallback registers the scalar extension metadata family. The
  current concrete MLIR namespace is `tcrv_scalar`; the architectural family
  remains `tcrv.scalar`. First-slice scalar operations are selected-boundary
  metadata only, and future scalar execution ops must stay plugin-local rather
  than being added to `tcrv.exec`;
- IME plugin registers the `tcrv.ime` extension family;
- TensorExt plugin registers the `tcrv.tensorext` extension family;
- offload plugin registers the `tcrv.offload` extension family;
- future plugins register their own TCRV extension families.

Concrete ODS/C++ directory splits are implementation organization only. They
must share TCRV capability, registry, common interfaces, common orchestration
passes, and common EmitC route framework.

### SourceFrontDoorPassProvider

Bounded source-to-selected-boundary front doors are plugin-owned entry points.
A plugin may expose one or more public MLIR pass factories through the common
extension plugin registry when the source front door is part of that extension
family's durable contract.

```cpp
class SourceFrontDoorPassRegistration {
public:
  StringRef getOwnerPlugin() const;
  StringRef getArgument() const;
  StringRef getDescription() const;
  std::function<std::unique_ptr<mlir::Pass>()> getFactory() const;
};

class ExtensionPlugin {
public:
  virtual Error registerSourceFrontDoorPasses(
      SmallVectorImpl<SourceFrontDoorPassRegistration> &out) const;
};
```

The registration object is pass plumbing only. It may name the owning plugin,
the public pass argument, a bounded description, and a factory for the
plugin-owned pass. It must not carry computation semantics, target-family
semantic branches, descriptor fields, route ids, runtime ABI identities,
artifact kinds, tuning state, or source-export payloads.

Public tools such as `tcrv-opt` may collect source front-door pass
registrations from enabled plugins and register those pass factories at the
tool/front-door boundary. If built-in plugins are disabled or a plugin is not
registered, that plugin's source front-door passes are not public command-line
options. Default empty registries remain fail-closed for embedded users and
tests.

The common registry validates registration shape and deterministic ordering,
but it does not inspect the source dialect body or decide RVV, IME, offload,
scalar, vendor, dtype, shape, runtime, toolchain, or microarchitecture
semantics. The concrete plugin pass remains responsible for its own narrow
source contract, fail-closed diagnostics, and selected-boundary materialization.

## Scenario: Toy Construction-Template Source Front Door

### 1. Scope / Trigger

Use this contract when proving that the source front-door registration surface
is reusable by a non-RVV extension family. Toy is a bounded construction
template, not a runtime performance backend, high-level tensor IR, descriptor
adapter, or source-export path.

### 2. Signatures

- Public pass option:
  `--tcrv-toy-materialize-template-source-front-door`.
- Common pipeline:
  `--tcrv-source-artifact-front-door-pipeline`.
- Toy source module attributes:
  `tcrv_toy.source_front_door = "template_compute"` and optional
  `tcrv_toy.source_kernel = "<valid-symbol>"`.
- Materialized extension-family boundary op:
  `tcrv_toy.compute_skeleton`.

### 3. Contracts

- The Toy plugin owns the source marker interpretation, Toy capability
  materialization, selected Toy variant metadata, `tcrv_toy.compute_skeleton`
  boundary materialization, Toy runtime ABI metadata, and Toy EmitC route
  provenance.
- The common registry and common source-artifact pipeline may only collect and
  run registered source-front-door pass factories, then run generic legality,
  capability, emission-plan, and coherence checks.
- The Toy source front door must materialize a selected Toy path with
  `origin = "toy-plugin"`, `requires = [@toy_template]`, a selected diagnostic
  targeting `@toy_template_first_slice`, and a `tcrv_toy.compute_skeleton`
  boundary whose `selected_variant`, `source_kernel`, `source_role`, typed role,
  role order, and required-capability metadata match the selected variant.
- The resulting selected path must be consumable by the Toy-owned EmitC route
  provider and by the Toy target header artifact exporter through the existing
  runtime ABI and artifact metadata contracts.
- The source marker and optional source-kernel attribute are front-door input
  syntax only. They must be removed or cease to be route authority after
  materialization.

### 4. Validation & Error Matrix

- Built-in plugins disabled -> Toy source-front-door pass option is not
  registered, and the common source-artifact pipeline remains fail-closed with
  an empty plugin registry.
- Unknown `tcrv_toy.source_front_door` value -> fail before materialization.
- Empty or invalid `tcrv_toy.source_kernel` symbol -> fail before
  materialization.
- Stale `tcrv_toy.lowering_seed` metadata -> fail before materialization.
- Pre-existing `tcrv.exec`, `tcrv_toy`, or `tcrv_rvv` selected-boundary or
  variant residue in the Toy source input -> fail before materialization.
- Missing or stale Toy boundary provenance after materialization -> fail in Toy
  boundary validation, EmitC route construction, emission planning, or target
  artifact export before output.

### 5. Good/Base/Bad Cases

- Good: a module with the Toy source marker materializes one selected Toy
  variant, one `tcrv_toy.compute_skeleton` boundary, a selected diagnostic, a
  supported Toy emission plan, Toy EmitC route source-op provenance, and the
  materialized Toy header artifact.
- Base: already materialized Toy execution surfaces remain valid backend-first
  inputs through the existing execution-planning and target artifact paths.
- Bad: common code branches on Toy/RVV names, a Toy seed option aliases the new
  pass, descriptor strings choose Toy computation, or Toy source marker
  metadata directly exports C/source artifacts without a selected Toy
  extension-family boundary and EmitC route.

### 6. Tests Required

- C++ registry coverage proving Toy and RVV source-front-door passes are
  collected through the same `collectSourceFrontDoorPasses` interface without
  duplicate or stale registration.
- lit/FileCheck positive coverage for the Toy pass and for
  `--tcrv-source-artifact-front-door-pipeline` showing selected Toy variant,
  `origin = "toy-plugin"`, `tcrv_toy.compute_skeleton`, runtime ABI ownership,
  supported emission plan, and Toy EmitC route provenance.
- lit/FileCheck target artifact coverage piping the Toy source-front-door
  pipeline to `--tcrv-export-target-header-artifact`.
- Negative coverage for disabled built-ins, stale Toy seed entry points,
  malformed Toy source marker, stale pre-materialized selected-boundary
  residue, missing boundary provenance, and absence of descriptor/direct-C/
  source-export/RVV-only residue in Toy source-front-door outputs.

### 7. Wrong vs Correct

Wrong:

```text
Toy source marker -> common Toy/RVV branch or descriptor metadata
  -> direct C/source artifact export
```

Correct:

```text
Toy source marker
  -> Toy-owned source front-door pass
  -> selected Toy variant + tcrv_toy.compute_skeleton boundary
  -> Toy-owned EmitC route provenance
  -> common emission-plan/coherence checks
  -> Toy target header artifact validation
```

## Scenario: TensorExtLite Construction-Template Source Front Door

### 1. Scope / Trigger

Use this contract when proving that the source-front-door construction template
is reusable by TensorExtLite. TensorExtLite source materialization is a bounded
fragment-MMA construction proof, not a general TensorExt frontend, linalg
lowering, high-level tensor/tile IR, descriptor adapter, runtime execution
path, source-export path, correctness claim, or performance path.

### 2. Signatures

- Public pass option:
  `--tcrv-tensorext-lite-materialize-fragment-mma-source-front-door`.
- Common pipeline:
  `--tcrv-source-artifact-front-door-pipeline`.
- TensorExtLite source module attributes:
  `tcrv_tensorext_lite.source_front_door = "fragment_mma_template"` and
  optional `tcrv_tensorext_lite.source_kernel = "<valid-symbol>"`.
- Materialized selected TensorExtLite role ops:
  `tcrv_tensorext_lite.config_skeleton`,
  `tcrv_tensorext_lite.load_frag_skeleton`,
  `tcrv_tensorext_lite.tile_mma_skeleton`, and
  `tcrv_tensorext_lite.store_frag_skeleton`.
- Materialized direct selected-boundary marker for generic emission gates:
  `tcrv_tensorext_lite.lowering_boundary`.

### 3. Contracts

- The TensorExtLite plugin owns the source marker interpretation, capability
  materialization, selected variant metadata, ordered role-op materialization,
  selected lowering-boundary marker, runtime ABI metadata, EmitC route
  provenance, relocatable object route configuration, and object-backed
  declaration-only header composite configuration.
- The selected variant must use `origin = "tensorext-lite-plugin"`, require the
  `tensorext_lite.tile_mma` capability, preserve the existing TensorExtLite
  construction protocol metadata, and contain the role sequence in
  `configure -> load_frag -> tile_mma -> store_frag` order.
- The direct `tcrv_tensorext_lite.lowering_boundary` marker is the generic
  selected-boundary surface consumed by emission-plan and coherence gates. It
  is not computation authority; the TensorExtLite EmitC route still derives
  source-op/source-role provenance from the ordered role ops.
- The common registry and common source-artifact pipeline may only collect and
  run registered source-front-door pass factories, then run generic legality,
  capability, emission-plan, and coherence checks.
- The source marker and optional source-kernel attribute are front-door input
  syntax only. They must be removed or cease to be route authority after
  materialization.
- The resulting selected path may export the TensorExtLite
  materialized-EmitC-derived relocatable object artifact, plus the
  declaration-only header composite derived from the same selected object
  candidate. Bundle, runtime execution, correctness, and performance claims
  remain out of scope.

### 4. Validation & Error Matrix

- Built-in plugins disabled -> TensorExtLite source-front-door pass option is
  not registered, and the common source-artifact pipeline remains fail-closed
  with an empty plugin registry.
- Unknown `tcrv_tensorext_lite.source_front_door` value -> fail before
  materialization.
- Empty or invalid `tcrv_tensorext_lite.source_kernel` symbol -> fail before
  materialization.
- Stale `tcrv_tensorext_lite.lowering_seed` metadata -> fail before
  materialization.
- Pre-existing `tcrv.exec`, `tcrv_tensorext_lite`, `tcrv_rvv`, or `tcrv_toy`
  selected-boundary or variant residue in the TensorExtLite source input ->
  fail before materialization.
- Missing or reordered TensorExtLite role ops -> fail in TensorExtLite EmitC
  route construction before emission-plan or target artifact output.
- Missing selected lowering-boundary marker -> fail in generic emission-plan
  validation before plugin emission routing.
- Missing route provenance, stale artifact metadata, descriptor/direct-C/
  source-export residue, or missing materialized EmitC handoff -> fail before
  object or header artifact output.

### 5. Good/Base/Bad Cases

- Good: a module with the TensorExtLite source marker materializes one selected
  TensorExtLite variant, the ordered role sequence, one direct
  `tcrv_tensorext_lite.lowering_boundary`, a selected diagnostic, supported
  TensorExtLite emission-plan metadata, TensorExtLite EmitC route provenance,
  the relocatable object artifact, and the declaration-only header composite
  derived from the same selected object candidate.
- Base: already materialized TensorExtLite role-sequence inputs remain valid
  backend-first inputs when they satisfy selected-path, emission-plan, and
  target artifact contracts.
- Bad: common code branches on TensorExtLite/Toy/RVV names, a TensorExtLite
  seed option aliases the new pass, descriptor strings choose fragment-MMA
  computation, or the source marker exports C/source artifacts without a
  selected TensorExtLite extension-family boundary and EmitC route.

### 6. Tests Required

- C++ registry coverage proving RVV, Toy, and TensorExtLite source-front-door
  passes are collected through the same `collectSourceFrontDoorPasses`
  interface without duplicate or stale registration.
- lit/FileCheck positive coverage for the TensorExtLite pass and for
  `--tcrv-source-artifact-front-door-pipeline` showing selected TensorExtLite
  variant, `origin = "tensorext-lite-plugin"`, ordered role ops,
  `tcrv_tensorext_lite.lowering_boundary`, runtime ABI ownership metadata,
  supported emission plan, and TensorExtLite EmitC route provenance.
- lit/FileCheck target artifact coverage piping the TensorExtLite
  source-front-door pipeline to `--tcrv-export-target-artifact`, proving a
  relocatable object with `llvm-readobj`, and to
  `--tcrv-export-target-header-artifact`, proving the declaration-only header
  stays derived from the selected materialized EmitC object candidate.
- Negative coverage for disabled built-ins, malformed TensorExtLite source
  marker, stale pre-materialized selected-boundary residue, missing/reordered
  role ops, missing boundary/route provenance, and absence of descriptor/
  direct-C/source-export/RVV-only/Toy-only residue in TensorExtLite
  source-front-door outputs.

### 7. Wrong vs Correct

Wrong:

```text
TensorExtLite source marker
  -> common TensorExtLite branch or descriptor metadata
  -> direct C/source artifact export
```

Correct:

```text
TensorExtLite source marker
  -> TensorExtLite-owned source front-door pass
  -> selected TensorExtLite variant
  -> ordered TensorExtLite role ops + TensorExtLite lowering-boundary marker
  -> TensorExtLite-owned EmitC route provenance
  -> common emission-plan/coherence checks
  -> TensorExtLite relocatable object artifact validation
  -> TensorExtLite declaration-only header composite validation
```

### TCRV Common Operation Interfaces

Long-term extension family ops should implement shared interfaces where
applicable:

```text
TCRVExtensionOpInterface
TCRVCapabilityOpInterface
TCRVConfigOpInterface
TCRVResourceOpInterface
TCRVMemoryOpInterface
TCRVComputeOpInterface
TCRVEmitCLowerableInterface
```

Core/common passes use these interfaces to query required capability, config
scope, resource kind, memory role, compute primitive kind, ABI mapping, and
EmitC lowering mapping. They must not branch on RVV, IME, TensorExt, Offload,
scalar fallback, vendor, intrinsic name, or fragment layout.

### VariantBuilder

```cpp
class VariantBuilder {
public:
  virtual bool supportsOperation(const VariantProposalRequest &request) = 0;

  virtual Error proposeVariants(const VariantProposalRequest &request,
                                SmallVectorImpl<VariantProposal> &out) = 0;
};
```

Responsibilities:

- decide whether the current planning anchor can be implemented by the plugin;
- receive the relevant execution envelope, materialized variant,
  selected-boundary surface, extension family ops, or future high-level
  `Operation *`, together with the generic `TargetCapabilitySet`;
- propose compiler-visible variant metadata before IR materialization;
- declare variant name, `origin`, required capability ids or symbol
  references, optional generic guard/policy metadata, and optional abstract
  fallback role metadata such as `ConservativeFallback`;
- later generation slices may materialize `tcrv.exec.variant`, place extension
  family ops in the variant body, and attach tuning/emission metadata.

### LegalityVerifier

```cpp
class LegalityVerifier {
public:
  virtual LogicalResult verifyVariant(TCRVVariant variant,
                                      const TargetCapability &target,
                                      DiagnosticEmitter &diag) = 0;
};
```

Responsibilities:

- check requires satisfaction;
- check ops/types against target capability;
- validate dtype, shape, layout, runtime ABI, and toolchain path;
- emit structured diagnostics.

The current registry-level C++ slice exposes materialized-variant legality
through the existing `ExtensionPlugin` protocol:

```cpp
class VariantLegalityRequest {
public:
  tcrv::exec::VariantOp getVariant() const;
  tcrv::exec::KernelOp getKernel() const;
  const TargetCapabilitySet &getCapabilities() const;
};

class ExtensionPlugin {
public:
  virtual Error verifyVariantLegality(
      const VariantLegalityRequest &request) const;
};
```

This request carries only compiler objects that already exist in the current
IR: the materialized `tcrv.exec.variant`, its enclosing `tcrv.exec.kernel`, and
the generic `TargetCapabilitySet` built from that kernel. It is not a Python
schema, pseudo-IR, tuning object, lowering plan, emission object, or runtime
probe result.

### TuningSpaceProvider

```cpp
class TuningSpaceProvider {
public:
  virtual void populateTuningSpace(TCRVVariant variant,
                                   TuningSpace &space) = 0;
};
```

Examples:

```text
RVV: LMUL, SEW, VL policy, unroll, packing, thread partition
IME: fragment shape, K blocking, accumulator residency, packing
Offload: batch size, transfer threshold, async overlap
```

### CostModelProvider

```cpp
class CostModelProvider {
public:
  virtual Cost estimateCost(TCRVVariant variant,
                            const TargetCapability &target,
                            const ShapeContext &shape) = 0;
};
```

Responsibilities:

- rank variants for selection;
- combine shape, dtype, capability, runtime transfer cost, and profile data;
- explain selection decisions.

The registry-level first C++ cost slice exposes materialized-variant cost
estimation through the existing `ExtensionPlugin` protocol:

```cpp
class VariantCostRequest {
public:
  tcrv::exec::VariantOp getVariant() const;
  tcrv::exec::KernelOp getKernel() const;
  const TargetCapabilitySet &getCapabilities() const;
};

class VariantCostEstimate {
public:
  bool hasScore() const;
  double getScore() const;
  bool hasExplicitPreference() const;
  StringRef getOriginPlugin() const;
  StringRef getVariantSymbol() const;
  bool hasExplanation() const;
  StringRef getExplanation() const;
  bool hasPolicy() const;
  StringRef getPolicy() const;
  VariantFallbackRole getFallbackRole() const;
};

class ExtensionPlugin {
public:
  virtual Error estimateVariantCost(const VariantCostRequest &request,
                                    VariantCostEstimate &out) const;
};
```

This request carries only compiler objects already present in IR: the
materialized `tcrv.exec.variant`, its enclosing `tcrv.exec.kernel`, and the
generic `TargetCapabilitySet` built from that kernel. The estimate is generic
selection-preference metadata: explicit-preference availability, a finite
non-negative score, origin/variant identity, fallback role metadata, and
optional non-empty generic explanation or policy text. It is not a selection
decision, tuning space, lowering plan, emission object, runtime ABI,
target-family object, hardware performance claim, correctness result, or
benchmark. The default plugin hook returns a deterministic neutral
no-explicit-preference estimate for the request variant and origin plugin.

### EmissionReadinessProvider

Before final lowering, the core may ask the selected variant's origin plugin
whether a compiler-visible emission path exists. This is a readiness check, not
lowering or runtime generation.

```cpp
enum class VariantEmissionRole {
  DirectVariant,
  DispatchCase,
  DispatchFallback,
};

class VariantEmissionRequest {
public:
  tcrv::exec::VariantOp getVariant() const;
  tcrv::exec::KernelOp getKernel() const;
  const TargetCapabilitySet &getCapabilities() const;
  VariantEmissionRole getRole() const;
};

class VariantEmissionStatus {
public:
  bool isSupported() const;
  bool isUnsupported() const;
  StringRef getOriginPlugin() const;
  StringRef getVariantSymbol() const;
  StringRef getEmissionPath() const;
  StringRef getReason() const;
};

class ExtensionPlugin {
public:
  virtual Error checkVariantEmissionReadiness(
      const VariantEmissionRequest &request,
      VariantEmissionStatus &out) const;
};
```

Registry routing is by the generic `origin` string on the materialized
`tcrv.exec.variant`. The request carries only the materialized variant, its
enclosing kernel, generic `TargetCapabilitySet`, and target-neutral role. A
supported result must include a non-empty plugin-owned emission path identifier
or description. An unsupported result must include a non-empty reason. The core
validates result shape, origin/variant identity, plugin enablement, and sibling
structure generically, but does not interpret RVV, IME, offload, scalar,
vendor, dtype, shape, runtime, toolchain, or microarchitecture semantics.

If variant selection materialized a direct selected-path `tcrv.exec.diagnostic`
marker for a static or fallback-only plan, the emission readiness pass resolves
that marker to the target `tcrv.exec.variant` before registry routing. The
plugin still receives the same `VariantEmissionRequest` with role
`DirectVariant`; the marker is core control metadata, not plugin-specific
lowering or runtime input.

### SelectedLoweringBoundaryMaterializer

The selected-path lowering-boundary interface is the generic handoff between
core selected `tcrv.exec` structure and plugin-local pre-lowering metadata. It
is still not executable lowering or runtime generation.

```cpp
class VariantLoweringBoundaryRequest {
public:
  tcrv::exec::VariantOp getVariant() const;
  tcrv::exec::KernelOp getKernel() const;
  const TargetCapabilitySet &getCapabilities() const;
  VariantEmissionRole getRole() const;
  OpBuilder &getBuilder() const;
};

class VariantLoweringBoundaryResult {
public:
  bool isMaterialized() const;
  bool isNoBoundary() const;
  bool isUnsupported() const;
  StringRef getOriginPlugin() const;
  StringRef getKernelSymbol() const;
  StringRef getVariantSymbol() const;
  VariantEmissionRole getRole() const;
  Operation *getMaterializedOperation() const;
  StringRef getReason() const;
};
```

Registry routing is by the selected variant `origin` string. The registry must
reject missing variants, missing kernels, non-direct variant/kernel
relationships, missing or empty origins, unknown origin plugins, disabled
origin plugins, plugin failures, unsupported plugin responses, and malformed or
mismatched results with generic diagnostics. A materialized result must identify
the request origin/kernel/variant/role and return a direct child operation in
the request kernel. A no-boundary result is valid only for an origin plugin
whose durable selected-path contract explicitly says no plugin-local boundary
operation is needed; the current scalar fallback route is such a generic
fallback envelope, and RVV currently uses fail-closed no-boundary behavior
until a materialized EmitC route exists.

The generic pass `--tcrv-materialize-selected-lowering-boundaries` traverses
selected dispatch cases followed by fallback, or a direct selected-path marker
when no dispatch exists, and delegates each reference to the registry. The pass
must not branch on RVV, scalar, IME, offload, vendor, runtime, dtype, shape, or
microarchitecture semantics. Plugin-local implementations decide whether to
create extension-dialect metadata when an active boundary surface exists; RVV
and scalar fallback currently report no plugin-local boundary for their deleted
or generic no-boundary routes.

### EmissionProvider

```cpp
class EmissionProvider {
public:
  virtual void populateLoweringPatterns(RewritePatternSet &patterns) = 0;
  virtual LogicalResult emitRuntimeGlue(ModuleOp module,
                                        const TargetCapability &target) = 0;
};
```

Responsibilities:

- provide extension-local EmitC lowering mappings through
  `TCRVEmitCLowerableInterface`;
- declare required headers, intrinsic/runtime call names, operand/result
  mapping, C type mapping, compiler flags, libraries, and runtime glue only for
  a materialized extension-local lowering/runtime ABI route, not through
  construction-manifest source skeleton fields;
- keep MLIR vector, LLVM scalable vector, LLVM RVV intrinsic IR, inline asm,
  vendor backend patches, and backend adapters as optional future routes unless
  promoted by a family spec and implementation evidence.

## Core Pass Usage

Core pass flow:

```text
1. Read target capability.
2. Load and register extension plugins.
3. For each current planning anchor, iterate plugin registry.
4. Each plugin checks support under target capability.
5. Plugins propose one or more tcrv.exec.variant values.
6. Core verifier orchestrates plugin verifier calls.
7. Core selector chooses a static variant or dispatch set.
8. Core emission-readiness check routes selected/direct/dispatch/fallback
   variants to their origin plugin and rejects missing or unsupported paths.
9. Emission stage calls common EmitC lowering over extension family interfaces,
   then target-owned emission/export helpers.
```

Current planning anchors may be hand-written or test execution envelopes,
materialized `tcrv.exec.variant` operations, selected-boundary IR,
`tcrv.exec.mem_window`, `tcrv.exec.runtime_param`, or extension family ops.
Future high-level `Operation *` analysis is an additional frontend integration
surface, not a precondition for plugin integration today. When that frontend
surface is selected as the owner, `linalg` is the preferred first high-level
input family for handwritten tests. Existing descriptor-shaped residues are
deletion targets or fail-closed implementation debt, not an architecture input
for new families.

Correct core shape:

```cpp
for (auto *plugin : pluginRegistry.enabledPlugins()) {
  if (!plugin->supportsOperation(request)) continue;
  plugin->proposeVariants(request, proposals);
}
```

Wrong core shape:

```cpp
if (target.hasRVV()) { ... }
if (target.hasIME()) { ... }
if (target.hasSophgo()) { ... }
```

The registry-level first slice provides deterministic proposal orchestration:

- iterate plugins in registration order;
- skip disabled plugins before support queries;
- skip enabled unsupported plugins before proposal generation;
- collect proposals only from enabled supported plugins;
- preserve recoverable plugin-local decline diagnostics when a supported plugin
  cannot propose because extension-owned evidence is missing, unavailable,
  malformed, or otherwise not satisfied; such declines must not prevent later
  enabled plugins from contributing valid proposals;
- reject malformed proposals with `llvm::Error`, including empty variant names
  or empty origin/plugin ownership;
- validate each proposal's required capability ids and symbol references against
  the request `TargetCapabilitySet` before IR materialization;
- reject empty, unknown, or unavailable proposal requirements with structured
  generic diagnostics that name the plugin, variant, requirement, and capability
  status;
- reject malformed plugin-owned proposal attributes before materialization,
  including empty names, non-dialect-qualified/discardable names, duplicate
  names, null values, and collisions with required `tcrv.exec.variant`
  attributes such as `sym_name`, `origin`, `requires`, `condition`, `guard`,
  `policy`, or `fallback_role`;
- reject duplicate proposal symbols within a single plugin's proposal output as
  plugin contract violations rather than treating them as recoverable declines;
- preserve an abstract proposal fallback role such as
  `ConservativeFallback` as generic `fallback_role = "conservative"` metadata
  when a plugin explicitly marks the variant as a fallback candidate;
- preserve valid plugin-owned MLIR attributes as opaque named attributes for
  materialization without interpreting extension-family semantics in the
  registry;
- keep proposal collection generic and free of target-family branches.

The registry-level legality slice provides deterministic materialized-variant
verification orchestration:

- validate that the request contains a materialized `tcrv.exec.variant`, an
  enclosing `tcrv.exec.kernel`, and a variant whose `origin` is a non-empty
  string attribute;
- build or receive the generic `TargetCapabilitySet` from the kernel without
  target-family logic;
- look up exactly the origin plugin named by the variant `origin` attribute;
- reject unknown origin plugins before invoking plugin legality;
- reject disabled origin plugins for materialized variant legality, because a
  disabled plugin cannot own fresh legality decisions for existing IR;
- call only the origin plugin's `verifyVariantLegality` hook, never every
  plugin in the registry;
- wrap plugin-local failures with generic context naming the plugin, variant,
  and kernel;
- when verifying a kernel, visit direct `tcrv.exec.variant` children in IR order
  and do not mutate materialization, dispatch, selection, lowering, or emission
  structures;
- keep legality orchestration generic and free of RVV, IME, offload, scalar,
  vendor, dtype, shape, layout, or target-family branches.

The registry-level cost slice provides deterministic materialized-variant cost
orchestration:

- validate that the request contains a materialized `tcrv.exec.variant`, an
  enclosing `tcrv.exec.kernel`, and a variant whose `origin` is a non-empty
  string attribute;
- build or receive the generic `TargetCapabilitySet` from the kernel without
  target-family logic;
- look up exactly the origin plugin named by the variant `origin` attribute;
- reject unknown origin plugins before invoking plugin cost estimation;
- reject disabled origin plugins for materialized variant cost estimation,
  because a disabled plugin cannot own fresh cost decisions for existing IR;
- call only the origin plugin's `estimateVariantCost` hook, never every plugin
  in the registry;
- validate returned estimates by requiring a present finite non-negative score,
  non-empty origin plugin, non-empty variant symbol, matching request
  origin/variant identity, and non-empty explanation/policy strings when those
  optional fields are present;
- distinguish explicit plugin preference from the default no-preference hook so
  selection can prefer real plugin-provided ranking metadata before applying
  target-neutral tie-breaks;
- allow plugins to return an abstract fallback role in the cost estimate so
  generic selection can choose a `tcrv.exec.fallback` without inspecting plugin
  names, target families, capability ids, dtypes, shapes, or runtime identities;
- wrap plugin-local failures and invalid estimates with generic context naming
  the plugin, variant, and kernel;
- when collecting costs for a kernel, visit direct `tcrv.exec.variant` children
  in IR order and do not mutate materialization, dispatch, selection, lowering,
  or emission structures;
- when ranking collected estimates, sort by explicit-preference availability,
  generic score, generic fallback role, original kernel IR order, then symbol
  name;
- keep cost orchestration generic and free of RVV, IME, offload, scalar,
  vendor, dtype, shape, layout, runtime ABI, microarchitecture, or
  target-family branches.

The registry-level emission-plan slice provides plugin-owned selected-path
planning after readiness, without generating executable artifacts:

- create a `VariantEmissionRequest` containing the materialized
  `tcrv.exec.variant`, its enclosing `tcrv.exec.kernel`, the generic
  `TargetCapabilitySet`, and the selected-path role (`direct variant`,
  `dispatch case`, or `dispatch fallback`);
- route the request only to the plugin named by the variant `origin` attribute;
- reject missing variants, missing kernels, variants not directly enclosed by
  the requested kernel, missing or empty origins, unknown origin plugins, and
  disabled origin plugins with generic diagnostics;
- call only the origin plugin's emission-plan hook, never every plugin in the
  registry and never a core target-family branch;
- validate returned plans by requiring a present status, non-empty origin
  plugin, non-empty kernel symbol, non-empty variant symbol, matching
  origin/kernel/variant identity, matching selected-path role, non-empty
  runtime ABI kind/name metadata, non-empty runtime glue role metadata, and
  non-empty required capability symbol refs that are a safe subset of the
  selected variant `requires` metadata;
- when selected lowering-boundary metadata has been materialized for the
  selected path, validate it before plugin plan routing and require exactly one
  matching direct kernel-child boundary for each selected reference;
- treat selected lowering-boundary candidates as actual plugin-local
  `*.lowering_boundary` operations, explicit lowering-boundary diagnostic
  metadata, or typed plugin-local role operations implementing
  `TCRVEmitCLowerableOpInterface`. Interface-backed role ops may carry
  `selected_variant`, `role`, and `source_kernel` as selected-path boundary
  identity, but core emission planning must recognize them through the common
  interface and generic attrs only, never through Toy/RVV/family-name branches;
- reject selected-boundary absence, stale selected-variant references, role
  mismatches, origin mismatches, required-capability mismatches, and duplicate
  competing boundaries generically before materializing emission-plan
  diagnostics;
- require supported plans to carry non-empty generic emission kind, lowering
  pipeline identifier, runtime ABI identifier, artifact kind, and explanation;
- reject direct runtime-callable C output artifact kinds for supported or
  unsupported plans. Runtime ABI parameter metadata is bounded plan metadata
  only; it cannot legalize a direct C output route before a future materialized
  MLIR EmitC route rebuild;
- require unsupported plans to carry a non-empty diagnostic string;
- require plugin-returned route, ABI, glue-role, diagnostic, and explanation
  text to be bounded single-line metadata, not raw logs, credentials, or
  unbounded hardware output;
- treat plans as plugin-owned compiler metadata/intent only; a supported plan is
  not proof that code was generated, linked, executed, correct, or performant;
- keep readiness and planning separate: readiness answers whether the selected
  path is supportable, while the plan describes the plugin-owned
  lowering/runtime route or structured unsupported reason.

The compiler may materialize collected `VariantEmissionPlan` results as
structured `tcrv.exec.diagnostic` metadata with `reason = "emission_plan"`.
Materialization is a core orchestration surface only:

- it reuses the same selected-path traversal and registry routing as plan
  collection;
- it creates diagnostics only after plan collection and generic validation have
  succeeded;
- it copies plugin-owned generic fields such as origin, role, support status,
  emission kind, lowering pipeline id, runtime ABI id, runtime ABI kind/name,
  runtime glue role, required capability refs, artifact kind, and
  diagnostic/explanation text;
- for selected paths with a validated plugin boundary, it also records a generic
  `lowering_boundary` diagnostic metadata field naming the boundary operation
  used for that plan;
- it rejects pre-existing materialized emission-plan diagnostics instead of
  appending stale duplicates;
- it never fills in RVV, IME, offload, scalar, vendor, dtype, shape,
  toolchain, runtime, or target-family semantics in core code.

Cost ranking is an input to generic selection planning, not a plugin-side
selection decision. A core selection planner may consume
`rankKernelVariantsByCost` results together with the same generic
`TargetCapabilitySet` to choose a static variant, a fallback-only path, a
runtime dispatch plan, or a no-viable-variant result. The planner must continue
to route cost/preference ownership through the variant `origin` plugin and must
preserve the registry ranking tie-break order: explicit-preference
availability, score, fallback role, original kernel IR order, then symbol name.
Plugins may provide scores and optional explanatory metadata through their cost
hook, but they do not directly mutate dispatch IR or encode core selection
policy through target-family branches. If runtime dispatch requires a fallback,
the planner may use only a plugin-provided generic fallback role from
proposal/materialized metadata or cost-estimate metadata; it must not invent
fallback coverage from an arbitrary available variant.

When legality verification, selection, emission-readiness, or emission-plan
collection is run as an MLIR pass, the pass must receive the same registry
object explicitly from the owning tool or plugin loader. A public tool such as
`tcrv-opt` may own a deterministic process-local registry and register built-in
plugins before pass registration; this is a tool/front-door concern and must
not move concrete extension branches into core orchestration. Default factory
constructors with no injected plugins remain an honest diagnostic surface for
C++ tests and embedded users: they must fail on unregistered variant `origin`
plugins rather than falling back to core-side string attributes, target-family
switches, or Python legality/cost/readiness models. This keeps the dependency
direction `tool/plugin loader -> registry -> core orchestration -> abstract
plugin interface -> concrete extension implementation`.

## Registration Metadata

Each plugin registers:

- plugin name;
- plugin version;
- provided capabilities;
- required external toolchain/runtime;
- extension family ops/types/attrs and concrete MLIR namespace;
- implemented TCRV common interfaces;
- types, attributes, and operations;
- supported current planning anchors and future high-level op classes;
- variant generation hooks;
- legality rules;
- tuning parameters;
- cost model;
- emission paths;
- target-support extension bundle and target translate route contribution hooks
  when the plugin owns target artifact or direct helper routes;
- fallback behavior.

## Scenario: Extension Bundle Interface Ownership

### 1. Scope / Trigger

This applies when code registers built-in or external extension-family bundles,
configures plugin-owned target-support hooks, or connects plugin catalogs to
target artifact exporter registries. The extension bundle surface is a plugin
construction contract, not a target artifact export contract.

### 2. Signatures

The generic bundle API lives under the plugin/common interface:

```cpp
#include "TianChenRV/Plugin/ExtensionBundle.h"

namespace tianchenrv::plugin {
using ExtensionPluginRegistrationFn =
    llvm::Error (*)(ExtensionPluginRegistry &registry);

class ExtensionBundle;
class ExtensionBundleRegistry;

class ExtensionPlugin {
public:
  virtual llvm::Error
  configureTargetSupportExtensionBundle(ExtensionBundle &bundle) const;
};
} // namespace tianchenrv::plugin
```

Target artifact export may expose target-specific consumer APIs and exporter
registries, but `TargetArtifactExport.h` must not define the generic
`ExtensionBundle`, `ExtensionBundleRegistry`, or extension plugin registration
callback types.

### 3. Contracts

- `ExtensionBundle` records bundle id, owner plugin name, plugin registration
  callback, required dialect names, selected lowering-boundary op names, and
  optional plugin-owned target artifact exporter bundle registration callback.
- `ExtensionBundleRegistry` owns generic bundle validation, bundle lookup, and
  extension plugin registration. It may collect target artifact exporter bundle
  callbacks into a target-owned `PluginTargetArtifactExporterRegistry`, but it
  must not validate target artifact route semantics itself.
- Target artifact exporter registries remain target-layer code. They validate
  route ids, artifact kinds, candidate preflight callbacks, composite routes,
  dependency enablement, and exporter invocation.
- Built-in plugin catalogs register concrete plugin bundles from the plugin
  layer. Target built-in artifact aggregation consumes those registries
  generically and must not own concrete per-extension bundle lists.
- Target translate route aggregation remains a generic consumer of enabled
  plugin hooks; it must not regain family-specific orchestration branches.

### 4. Validation & Error Matrix

- Empty bundle id or plugin name -> extension bundle registry rejects bounded
  text.
- Null extension plugin registration callback -> extension bundle registry
  rejects the bundle.
- Duplicate bundle id or duplicate owner plugin -> extension bundle registry
  rejects registration.
- Duplicate required dialect or lowering-boundary op entry -> extension bundle
  registry rejects registration.
- Target artifact exporter bundle references a missing or disabled required
  plugin -> target exporter bundle registry fails closed before route
  registration.
- Target artifact exporter bundle registers malformed route metadata -> target
  artifact exporter registry rejects the route, not the plugin bundle registry.

### 5. Good/Base/Bad Cases

- Good: `Plugin/BuiltinExtensionPlugins` owns the concrete built-in bundle
  catalog and fills a `plugin::ExtensionBundleRegistry`; target artifact export
  consumes that registry through target exporter bundle registries.
- Base: a plugin with no current artifact route may still configure required
  dialect metadata or no target-support metadata, and target artifact export
  publishes no route for that plugin.
- Bad: `TargetArtifactExport.h` defines generic extension bundles, or
  `lib/Target/Builtin` includes concrete RVV/IME/Offload/TensorExt/Toy plugin
  headers to own the extension-family catalog.

### 6. Tests Required

- C++ coverage proving built-in bundle catalog registration reaches all
  built-in plugins through the plugin/common bundle registry.
- C++ coverage proving target artifact exporter routes contributed by bundles
  are still registered only for enabled plugins with satisfied dependencies.
- Negative C++ coverage for duplicate bundle ids, duplicate plugin ownership,
  invalid bundle text, invalid target exporter bundles, and duplicate target
  artifact routes.
- Ref-scan coverage proving `TargetArtifactExport.h` does not define generic
  bundle classes and `lib/Target/Builtin` does not own concrete plugin bundle
  manifests.

### 7. Wrong vs Correct

Wrong:

```text
target artifact export header
  -> generic ExtensionBundle / ExtensionBundleRegistry ownership
  -> target built-in layer owns concrete extension family catalog
```

Correct:

```text
plugin/common ExtensionBundleRegistry
  -> plugin-owned built-in or external extension catalog
  -> target-owned exporter registries consume enabled plugin bundle hooks
  -> target artifact export validates and emits only selected target artifacts
```

## Built-In Registration API

Concrete built-in plugins may expose small C++ helpers that populate an
`ExtensionPluginRegistry` without changing core pass orchestration. Aggregate
built-in registration must keep `ExtensionBundleRegistry` visible in the public
front door so target support bundles, plugin-owned exporter bundles, and plugin
registrations are derived from the same enabled extension bundle catalog. The
current first built-in helper set is:

```cpp
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/ExtensionBundle.h"
#include "TianChenRV/Plugin/Offload/OffloadExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"

llvm::Error registerRVVExtensionPlugin(ExtensionPluginRegistry &registry);
llvm::Error registerOffloadExtensionPlugin(ExtensionPluginRegistry &registry);
llvm::Error registerScalarExtensionPlugin(ExtensionPluginRegistry &registry);
llvm::Error registerBuiltinExtensionBundles(ExtensionBundleRegistry &registry);
llvm::Error registerBuiltinExtensionBundlePlugins(
    ExtensionBundleRegistry &bundles, ExtensionPluginRegistry &registry);
```

The registry stores non-owning plugin references, so built-in registration
helpers must register objects with safe process lifetime and must never register
temporaries or stack objects that go out of scope after the helper returns.
Duplicate registration continues to be rejected by the generic registry
diagnostic path. `tcrv-opt` and `tcrv-translate` register built-ins through the
bundle front door at the tool boundary and then register registry-dependent
passes or translations with that owned registry, so public `rvv-plugin` and
`scalar-plugin` origins route through their plugins instead of the
empty-registry path. Do not add a plugin-only built-in compatibility wrapper
that hides the bundle registry; tests that need all built-ins should construct
an `ExtensionBundleRegistry` and call the canonical bundle front door directly.
The same front-door rule applies to the generic `offload-plugin`: offload
dialect and handoff behavior are registered through the plugin registry, while
shared orchestration continues to route only by generic `origin` lookup.
`--tcrv-disable-builtin-plugins` is the public tool escape hatch for tests that
must exercise a completely empty plugin registry, including unregistered plugin
dialect diagnostics. Unknown origins still diagnose generically as unregistered
plugins, and direct factory tests can still construct empty-registry passes when
that negative behavior is required.

`registerPluginDialects` delegates to
`ExtensionPluginRegistry::registerDialectsForEnabledPlugins`. Therefore a
disabled plugin must not make enabled-only plugin dialects available. The RVV
first dialect slice relies on this contract: `!tcrv_rvv.vl` parses only when
the RVV plugin has been registered and enabled through the plugin registry path,
not through default core dialect registration.
