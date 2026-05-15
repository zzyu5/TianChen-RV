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
  mapping, C type mapping, compiler flags, libraries, and runtime glue;
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
  `*.lowering_boundary` operations or explicit lowering-boundary diagnostic
  metadata only. Other plugin-local selected-path attachments, such as an RVV
  executable microkernel op, may carry `selected_variant`, `role`, and
  `source_kernel` for their own validation, but core emission planning must not
  count them as competing lowering boundaries;
- reject selected-boundary absence, stale selected-variant references, role
  mismatches, origin mismatches, required-capability mismatches, and duplicate
  competing boundaries generically before materializing emission-plan
  diagnostics;
- require supported and metadata-only plans to carry non-empty generic emission
  kind, lowering pipeline identifier, runtime ABI identifier, artifact kind, and
  explanation;
- reject `runtime-callable-c-source` as the artifact kind for supported or
  metadata-only plans. Runtime ABI parameter metadata is bounded plan metadata
  only; it cannot legalize a direct C source artifact route before a future
  materialized MLIR EmitC source route rebuild;
- require unsupported plans to carry a non-empty diagnostic string;
- require plugin-returned route, ABI, glue-role, diagnostic, and explanation
  text to be bounded single-line metadata, not raw logs, credentials, or
  unbounded hardware output;
- treat plans as plugin-owned compiler metadata/intent only; a supported or
  metadata-only plan is not proof that code was generated, linked, executed,
  correct, or performant;
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

## Built-In Registration API

Concrete built-in plugins may expose small C++ helpers that populate an
`ExtensionPluginRegistry` without changing core pass orchestration. The current
first built-in helper set is:

```cpp
#include "TianChenRV/Plugin/BuiltinExtensionPlugins.h"
#include "TianChenRV/Plugin/Offload/OffloadExtensionPlugin.h"
#include "TianChenRV/Plugin/RVV/RVVExtensionPlugin.h"
#include "TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h"

llvm::Error registerRVVExtensionPlugin(ExtensionPluginRegistry &registry);
llvm::Error registerOffloadExtensionPlugin(ExtensionPluginRegistry &registry);
llvm::Error registerScalarExtensionPlugin(ExtensionPluginRegistry &registry);
llvm::Error registerBuiltinExtensionPlugins(ExtensionPluginRegistry &registry);
```

The registry stores non-owning plugin references, so built-in registration
helpers must register objects with safe process lifetime and must never register
temporaries or stack objects that go out of scope after the helper returns.
Duplicate registration continues to be rejected by the generic registry
diagnostic path. `tcrv-opt` registers the built-in helper set at the tool
boundary and then registers registry-dependent passes with that owned registry,
so public `rvv-plugin` and `scalar-plugin` origins route through their plugins
instead of the empty-registry path. The same front-door rule applies to the
generic `offload-plugin`: offload dialect and handoff behavior are registered
through the plugin registry, while shared orchestration continues to route only
by generic `origin` lookup. `--tcrv-disable-builtin-plugins` is the public tool
escape hatch for tests that must exercise a completely empty plugin registry,
including unregistered plugin dialect diagnostics. Unknown origins still
diagnose generically as unregistered plugins, and direct factory tests can still
construct empty-registry passes when that negative behavior is required.

`registerPluginDialects` delegates to
`ExtensionPluginRegistry::registerDialectsForEnabledPlugins`. Therefore a
disabled plugin must not make enabled-only plugin dialects available. The RVV
first dialect slice relies on this contract: `!tcrv_rvv.vl` parses only when
the RVV plugin has been registered and enabled through the plugin registry path,
not through default core dialect registration.
