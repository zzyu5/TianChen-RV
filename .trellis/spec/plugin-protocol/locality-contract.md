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
- extension-specific emission plans for selected paths;
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

Core may route and validate emission-plan objects through the registry, but it
must not fill in RVV/IME/offload lowering pipelines, runtime ABIs, artifact
kinds, or supported-path claims on behalf of plugins. A plan is plugin-owned
metadata/intent and remains separate from executable code generation evidence.
Core may also materialize those plugin-owned plans into
`tcrv.exec.diagnostic {reason = "emission_plan"}` metadata after selected-path
collection succeeds. That materialization is limited to copying generic plan
fields, validating symbol structure, and validating the selected plugin-owned
lowering-boundary metadata when the selected path has such a boundary. It is
not lowering, runtime glue, artifact generation, RVV support, correctness
evidence, or performance evidence.

Public tool integration is a front-door/plugin-loader responsibility, not a core
target-family branch. For example, `tcrv-opt` may construct a deterministic
`ExtensionPluginRegistry`, populate it with built-in plugins such as the RVV
first-slice plugin, register plugin dialects, and pass that registry into
registry-dependent passes. The shared pass logic must still route only through
generic `origin` lookup and plugin interfaces.

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
- whether core contains extension-specific branch;
- whether `tcrv.exec.variant`, dispatch, verifier orchestration, and emission interfaces are reused.
