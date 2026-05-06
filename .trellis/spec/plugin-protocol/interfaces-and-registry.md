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

- RVV plugin registers `tcrv.rvv`;
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
8. Emission stage calls each selected plugin emission provider.
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
