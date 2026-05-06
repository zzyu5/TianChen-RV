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
- extension-specific legality details;
- extension-specific toolchain workarounds.

## Core-Owned Responsibilities

Core owns:

- capability registry;
- plugin registry;
- variant container;
- variant selection orchestration;
- dispatch/fallback structure;
- common verifier orchestration;
- common diagnostics format.

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
