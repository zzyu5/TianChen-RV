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
  virtual bool supports(Operation *highLevelOp,
                        const TargetCapability &target) = 0;

  virtual LogicalResult proposeVariants(Operation *highLevelOp,
                                        const TargetCapability &target,
                                        SmallVectorImpl<TCRVVariant> &out) = 0;
};
```

Responsibilities:

- decide whether a high-level op can be implemented by the plugin;
- generate `tcrv.exec.variant`;
- place extension dialect ops in the variant body;
- declare `requires`, `origin`, shape guards, tuning metadata, and emission metadata.

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
  if (!plugin->supports(op, target)) continue;
  plugin->proposeVariants(op, target, variants);
}
```

Wrong core shape:

```cpp
if (target.hasRVV()) { ... }
if (target.hasIME()) { ... }
if (target.hasSophgo()) { ... }
```

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
