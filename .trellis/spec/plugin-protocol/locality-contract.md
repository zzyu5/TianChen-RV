# Plugin Locality Contract

## Meaning Of Pluginization

Pluginization does not mean new hardware costs zero work.

New IME, offload, or future custom ISA support still requires:

- ops, types, attributes;
- capability definitions;
- variant generator;
- legality rules;
- cost model;
- tuning rules;
- lowering patterns;
- runtime or toolchain adapter.

These pieces are compiler implementation work and belong in C++/MLIR/TableGen/CMake, with lit/FileCheck and C++ tests as appropriate.

Pluginization means:

- new capability code is concentrated in a plugin;
- core passes do not hard-code concrete extension names;
- core passes interact through registry and interfaces;
- high-level MLIR semantic conversion is not rewritten for each extension.

## Plugin-Owned Responsibilities

Plugin owns:

- extension-specific types and ops;
- extension-specific lowering;
- extension-specific tuning;
- extension-specific runtime ABI;
- selected-body realization when code-affecting hints/config/profile must
  become concrete body structure;
- extension-specific route provider output;
- extension-specific emission diagnostics/mirrors for selected paths;
- extension-specific legality details;
- extension-specific toolchain workarounds.

Plugin-owned does not mean Python-owned. Plugin compiler behavior remains C++/MLIR implementation.

## Core-Owned Responsibilities

Core owns:

- capability registry;
- plugin registry;
- variant container;
- variant selection orchestration;
- dispatch/fallback structure;
- common verifier orchestration;
- common diagnostics format.

Core may orchestrate selected-body realization and route-provider calls through
generic interfaces, but the semantics stay plugin-owned. Core must not fill in
RVV/IME/offload lowering pipelines, runtime ABIs, artifact kinds, intrinsic
names, dtype choices, or supported-path claims on behalf of plugins.

Emission-plan metadata is not executable route authority. Core may materialize
optional plugin-owned mirrors into
`weft.exec.diagnostic {reason = "emission_plan"}` only after selected-path
collection and provider route construction. That materialization is limited to
copying generic mirror fields and validating symbol structure. It is not
lowering, runtime glue, artifact generation, RVV support, correctness evidence,
performance evidence, or progress.

Public tool integration is a front-door/plugin-loader responsibility, not a core
target-family branch. For example, `weft-opt` may construct a deterministic
`ExtensionPluginRegistry`, populate it with built-in plugins such as the RVV
first-slice plugin, register plugin dialects, and pass that registry into
registry-dependent passes. The shared pass logic must still route only through
generic `origin` lookup and plugin interfaces.

## [F-3] Change Containment

When family code is consolidated under one directory, a family integration PR is
**contained to the family territory + table rows + docs + one shared registration row**:

```text
lib/{Dialect,Plugin,Target}/<Fam>/   family ops/types/attrs, legality, realization, route provider, emission patterns
  + include/Weft/{Dialect,Plugin,Target}/<Fam>/  header mirrors
                     (this is the real 6-directory-root shape — NOT a literal `plugins/<family>/`;
                      the plugin lib transitively links the family Dialect + Target libs, 见 [GAP-P4-TOUCHSET])
tables               capability fact rows + relation rows (schema/**) + ledger row (docs/method/C2_marginal_cost_ledger.md)
tests                test/**  ([P-2] piece ④)
docs                 integration doc ([P-4])
registration         lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp — ONE kBuiltinExtensionBundles[] row
                     (family name as data in a for-iterated table, no branch; 见 [GAP-P4-REGISTER].
                      ⚠ proposed shared_allowances addition to schema/family-manifest.v1.json — main-session/recon)
```

`plugins/<family>/` above is a generic stand-in; the code actually lives under
`lib/{Dialect,Plugin,Target}/<Fam>/` + `include/` mirrors (the shape
`schema/family-manifest.v1.json` `source_ranges` already declares as one family's
owned territory). This is the containment form of [P-1] (no core dispatch/selection/
lowering edits — the registration-table row is the sole intended shared touch) and
the spatial counterpart to the five-piece set ([P-2], 见
[extension-plugin-integration.md](./extension-plugin-integration.md)). It is
conditional on the structural premise that family code is actually gathered under
`plugins/<family>/`; where a family's code is still scattered, consolidating it is
the prerequisite work, not a reason to spill the integration into core. Combined
with zero family-name branches (I3) and the [F-1] grep-clean check, containment is
the auditable evidence for C2 (marginal-cost) generalization.

## When Core May Change

The system must not promise that every future extension needs zero core changes.

Acceptable statement:

```text
If a new extension maps to existing capability, variant, resource, and emission interfaces, it should be added as a plugin.
If it introduces genuinely new execution semantics, the core interfaces may need extension.
```

Examples that may require core interface extension:

- distributed multi-hart collective semantics;
- nonstandard consistency model;
- device-side scheduling queue;
- asynchronous side effects not modelable by existing dispatch/runtime contracts.

## Evaluation Metrics

For any new plugin, record:

- core pass modified LOC;
- plugin LOC;
- new capabilities;
- new ops/types;
- new variant builders;
- supported high-level op classes;
- selected-body realization hooks;
- route provider implementation;
- whether core contains extension-specific branch;
- whether `weft.exec.variant`, dispatch, verifier orchestration, and emission interfaces are reused.
