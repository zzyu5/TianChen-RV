# Extension Plugin Integration Contract

## Scope

This contract describes how a new extension family integrates with TianChen-RV
without rewriting core passes or turning metadata into executable authority.

RVV is the first and broadest real reference family; per-family build/maturity
status lives in tasks/journal, not here
(见 [../guides/trunk-discipline.md](../guides/trunk-discipline.md)).

## Standard Flow

The standard executable flow is:

```text
selected tcrv.exec variant
  -> typed extension-family body or selected boundary
  -> plugin legality
  -> optional plugin selected-body realization
  -> plugin-built TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> target artifact packaging
  -> runtime/hardware evidence when claimed
```

For RVV, the typed body is low-level `tcrv_rvv` vector-level structure.

## Plugin Adds

A plugin may add:

- capability ids/properties and relation contracts;
- extension dialect ops/types/attrs;
- verifier and legality hooks;
- variant builders;
- selected-body realization;
- route provider;
- cost/tuning hooks;
- EmitC route payload mapping;
- runtime ABI adapters;
- focused tests.

Compiler behavior remains C++/MLIR/TableGen/CMake/lit/FileCheck. Python may
only orchestrate, probe, or parse artifacts.

## Optional Planning Artifacts

Extension Manifest, semantic role graph, construction template, and
source-front-door examples are optional planning/provenance surfaces. They
cannot be source, route, dtype, compute, artifact, or progress authority.

Core may see route ids, artifact kinds, manifests, or selected-path metadata
only as validated mirrors after plugin decisions. Core must not use them to
construct or infer routes.

## Current Valid Inputs

Current plugin work may start from:

- hand-written or generated TianChen-RV MLIR;
- selected `tcrv.exec` variant;
- typed extension-family body;
- selected boundary that the origin plugin can legally consume;
- plugin-owned capability/profile data.

Selected-path metadata alone is diagnostic/control mirror only. It is not a
valid compute or route input.

## Source Front Doors

Source-front-door defaults must be explicit-only or disabled. Positive
source-front-door examples are future work unless a task explicitly enables a
mature typed-body route for that family.

Source-front-door/source-artifact RVV paths fail closed (见 core-invariants
I7). No current test should require positive RVV artifact generation from
source-only metadata.

## Integration Checklist

- [ ] Does the plugin declare structured capabilities and requirements?
- [ ] Does executable work start from a typed body or legal selected boundary?
- [ ] Does plugin legality run before route provider output?
- [ ] Does selected-body realization consume code-affecting hints/config into body structure?
- [ ] Does the plugin build `TCRVEmitCLowerableRoute`?
- [ ] Does common EmitC only materialize provider output?
- [ ] Are manifests/templates/source markers optional provenance only?
- [ ] Are route ids and artifact kinds mirrors only?
- [ ] Are tests attached to production compiler behavior?

## Good / Bad Cases

Good:

```text
selected RVV variant
  -> typed tcrv_rvv body
  -> RVV legality / realization
  -> RVV route provider
  -> common EmitC
```

Bad:

```text
Extension Manifest / semantic role graph
  -> executable route construction
```

Bad:

```text
selected-path metadata
  -> common lower-to-EmitC without typed body or plugin route provider
```

Bad:

```text
core branch recognizes RVV/Toy/TensorExtLite route id
```
