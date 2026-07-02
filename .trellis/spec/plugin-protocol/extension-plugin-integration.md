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

## [P-2] Integration Five-Piece Set

A family integration is **one PR series**, and the series is not accepted until
all five pieces are present (缺一不收):

```text
① fact + relation table rows         (capability facts + implies/conflicts)
② legality predicate                 (plugin-owned, runs before route provider)
③ emission pattern                   (or an explicit "reuse existing pattern" declaration)
④ tests                              (lit byte-exact + per-board objdump golden)
⑤ ledger entry                       (cost/attribution ledger row, 见 C2 automation)
```

The five-piece set is the concrete per-family form of [P-1]: each piece flows
through the frozen `ExtensionPlugin` interface and touches only family-local
surfaces plus table rows (change containment is [F-3], 见
[locality-contract.md](./locality-contract.md)). Piece ④ is what upgrades a claim
from compile-time to silicon-sealed: lit proves byte-exact emission
(compile-lit), the per-board objdump golden is the silicon-facing artifact
(编译≠硅封, 见 [../guides/trunk-discipline.md](../guides/trunk-discipline.md) and
core-invariants I8). Runtime/correctness/performance claims still need real
hardware evidence (I8).

## [P-4] External Integrability

The integration documentation must be complete enough that a **non-core author
can complete one family integration from the docs alone** — this is the criterion
that upgrades C1 from a single in-house demonstration to a repeatable *protocol*
(总纲 [P-4], M4). "External-integrable" means the five-piece set ([P-2]), the
frozen interface ([P-1]), the typed-body/route flow (Standard Flow above), and
the change-containment rule ([F-3]) are each documented as a followable
procedure, not reconstructable only by reading core source.

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
- [ ] Does the PR series carry all five pieces ([P-2]: facts+relations, legality predicate, emission pattern, tests, ledger)?
- [ ] Does the integration touch only `plugins/<family>/` + table rows + docs ([F-3])?
- [ ] Is the interface used as-is without editing core files ([P-1])?
- [ ] Is the integration doc followable by a non-core author ([P-4])?

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
