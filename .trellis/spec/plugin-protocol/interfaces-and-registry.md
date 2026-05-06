# Interfaces And Registry

## Required Plugin Interfaces

Each extension plugin should provide the following interface set.

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

### DialectProvider

```cpp
class DialectProvider {
public:
  virtual void registerDialects(DialectRegistry &registry) = 0;
};
```

Responsibilities:

- RVV plugin registers the RVV extension dialect family. The current concrete
  MLIR namespace is `tcrv_rvv` because MLIR dialect namespaces cannot contain
  `.` characters; the architectural family remains `tcrv.rvv`;
- IME plugin registers `tcrv.ime`;
- offload plugin registers `tcrv.offload`;
- future plugins register their own extension dialects.

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

- decide whether a high-level op can be implemented by the plugin;
- receive the high-level `Operation *`, the relevant `tcrv.exec.kernel`
  anchor, and the generic `TargetCapabilitySet`;
- propose compiler-visible variant metadata before IR materialization;
- declare variant name, `origin`, required capability ids or symbol
  references, and optional generic guard/policy metadata;
- later generation slices may materialize `tcrv.exec.variant`, place extension
  dialect ops in the variant body, and attach tuning/emission metadata.

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
  StringRef getOriginPlugin() const;
  StringRef getVariantSymbol() const;
  bool hasExplanation() const;
  StringRef getExplanation() const;
  bool hasPolicy() const;
  StringRef getPolicy() const;
};

class ExtensionPlugin {
public:
  virtual Error estimateVariantCost(const VariantCostRequest &request,
                                    VariantCostEstimate &out) const;
};
```

This request carries only compiler objects already present in IR: the
materialized `tcrv.exec.variant`, its enclosing `tcrv.exec.kernel`, and the
generic `TargetCapabilitySet` built from that kernel. The estimate is a generic
finite non-negative score plus origin/variant identity and optional non-empty
generic explanation or policy text. It is not a selection decision, tuning
space, lowering plan, emission object, runtime ABI, target-family object, or
hardware performance claim. The default plugin hook returns a deterministic
neutral estimate for the request variant and origin plugin.

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

- RVV emission to MLIR vector, LLVM scalable vector, RVV intrinsic, builtin, or inline asm;
- IME emission to vendor intrinsic, inline asm, external stub, or backend adapter;
- offload emission to vendor C ABI or runtime library calls;
- future custom ISA emission to custom intrinsic, inline asm, backend patch, or object stub.

## Core Pass Usage

Core pass flow:

```text
1. Read target capability.
2. Load and register extension plugins.
3. For each high-level op, iterate plugin registry.
4. Each plugin checks support under target capability.
5. Plugins propose one or more tcrv.exec.variant values.
6. Core verifier orchestrates plugin verifier calls.
7. Core selector chooses a static variant or dispatch set.
8. Core emission-readiness check routes selected/direct/dispatch/fallback
   variants to their origin plugin and rejects missing or unsupported paths.
9. Emission stage calls each selected plugin emission provider.
```

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
  attributes such as `sym_name`, `origin`, `requires`, `condition`, `guard`, or
  `policy`;
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
- wrap plugin-local failures and invalid estimates with generic context naming
  the plugin, variant, and kernel;
- when collecting costs for a kernel, visit direct `tcrv.exec.variant` children
  in IR order and do not mutate materialization, dispatch, selection, lowering,
  or emission structures;
- when ranking collected estimates, sort by generic score and keep equal-score
  ties stable by original kernel IR order;
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
  origin/kernel/variant identity, and matching selected-path role;
- require supported plans to carry non-empty generic emission kind, lowering
  pipeline identifier, runtime ABI identifier, artifact kind, and explanation;
- require unsupported plans to carry a non-empty diagnostic string;
- treat plans as plugin-owned compiler metadata/intent only; a supported plan is
  not proof that code was generated, linked, executed, correct, or performant;
- keep readiness and planning separate: readiness answers whether the selected
  path is supportable, while the plan describes the plugin-owned
  lowering/runtime route or structured unsupported reason.

Cost ranking is an input to generic selection planning, not a plugin-side
selection decision. A core selection planner may consume
`rankKernelVariantsByCost` results together with the same generic
`TargetCapabilitySet` to choose a static variant, a fallback-only path, a
runtime dispatch plan, or a no-viable-variant result. The planner must continue
to route cost ownership through the variant `origin` plugin and must preserve
the registry ranking tie-break by original kernel IR order. Plugins may provide
scores and optional explanatory metadata through their cost hook, but they do
not directly mutate dispatch IR or encode core selection policy through
target-family branches.

When selection is run as an MLIR pass, the pass must receive the same registry
object explicitly from the owning tool or plugin loader. A default public pass
registration with no injected plugins is allowed only as an honest diagnostic
surface: it must fail on unregistered variant `origin` plugins rather than
falling back to core-side string attributes, target-family switches, or Python
cost models. This keeps the dependency direction
`core orchestration -> abstract plugin interface -> concrete extension
implementation`.

## Registration Metadata

Each plugin registers:

- plugin name;
- plugin version;
- provided capabilities;
- required external toolchain/runtime;
- extension dialects;
- types, attributes, and operations;
- supported high-level op classes;
- variant generation hooks;
- legality rules;
- tuning parameters;
- cost model;
- emission paths;
- fallback behavior.

## Built-In Registration API

Concrete built-in plugins may expose small C++ helpers that populate an
`ExtensionPluginRegistry` without changing core pass orchestration. The current
first built-in helper set is:

```cpp
llvm::Error registerRVVExtensionPlugin(ExtensionPluginRegistry &registry);
llvm::Error registerBuiltinExtensionPlugins(ExtensionPluginRegistry &registry);
```

The registry stores non-owning plugin references, so built-in registration
helpers must register objects with safe process lifetime and must never register
temporaries or stack objects that go out of scope after the helper returns.
Duplicate registration continues to be rejected by the generic registry
diagnostic path. Core passes must still receive a populated registry explicitly;
the default public `tcrv-select-variants` pass remains honest when no plugins
are injected.

`registerPluginDialects` delegates to
`ExtensionPluginRegistry::registerDialectsForEnabledPlugins`. Therefore a
disabled plugin must not make enabled-only plugin dialects available. The RVV
first dialect slice relies on this contract: `!tcrv_rvv.vl` parses only when
the RVV plugin has been registered and enabled through the plugin registry path,
not through default core dialect registration.
