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

### [P-2] is the DELIVERY axis — three views of "five pieces" [GAP-P4-FIVEPIECE-COLLISION]

The phrase "five-piece set" (五件套) is used on three different axes across the
docs. **This `[P-2]` set (① facts+relations ② legality ③ emission pattern ④ tests
⑤ ledger) is the authoritative *acceptance* definition** — it is what "缺一不收"
means for a PR series. The other two "five" enumerations are **not competing
definitions of [P-2]**; they are the *file axis* and *repository axis* views of
the same integration. Cross-map:

| Axis | Enumeration | Authority |
|---|---|---|
| **Delivery / PR-series ([P-2], authoritative)** | ① facts+relations rows · ② legality predicate · ③ emission pattern · ④ tests · ⑤ ledger row | this section (acceptance gate) |
| **File (per-family)** — where the deliverables land | ① `<Fam>ExtensionPlugin.cpp` (entry+facts) · ② `<Fam>VariantLegality.cpp` · ③ `<Fam>ConstructionProtocol.cpp` + `<Fam>EmitCRouteProvider.cpp` · ④ `test/**` · ⑤ `<Fam>BackendEmissionDriver.cpp` | `REPOSITORY-MAP-五大件.md` §2 (a *realization* view, not a second [P-2]) |
| **Repository (whole stack)** — 五大件/6 components | schema · plugin-five-piece · front-door · selector · falsifier · measurement | `REPOSITORY-MAP-五大件.md` §1 (orthogonal: the whole template, not one family) |

The delivery pieces do **not** map one-to-one onto files (③ emission pattern
spans two files; ⑤ ledger is a `docs/` row, not the `BackendEmissionDriver`). The
file-axis "五件套" names the 5 `lib/Plugin/<Fam>/` `.cpp` files; the real linking
touch-set is wider (6 roots, 见 [GAP-P4-TOUCHSET] in
[extension-family-plugin-template.md](./extension-family-plugin-template.md)).

### Landing points — where each deliverable actually lives [GAP-P4-FACT-LOCATION]

`schema/*.json` shape files declare *shape*, not instances. Where the concrete
per-family rows/facts land (2026-07-12, code-verified):

| Deliverable | Landing file | Notes |
|---|---|---|
| ① capability **fact instances** | `lib/Plugin/<Fam>/<Fam>ExtensionPlugin.cpp` — `PluginCapability(...)` pushed in `getCapabilities()` | C++, **not** a schema JSON (e.g. `IMEExtensionPlugin.cpp:675`, `ScalarExtensionPlugin.cpp:112`). `schema/capability.schema.v1.json` is aspirational shape only, 见 [GAP-P4-SCHEMA-DIVERGENCE] |
| ① relation rows (`implies`/`conflicts`) | same `getCapabilities()` array + capability-model tests | |
| ④ tests | `test/**` (lit + gtest) | [F-3] shared allowance |
| ⑤ C2 ledger row | `docs/method/C2_marginal_cost_ledger.md` | core LOC / plugin LOC / new caps / new ops (见 [locality-contract.md](./locality-contract.md) §Evaluation Metrics) |
| coverage cell | `schema/coverage-roster.v1.json` (denominator) + `schema/coverage-sixstate.v1.json` (certified source) | |
| family territory + dispatch key | `schema/family-manifest.v1.json` (`source_ranges`) · `schema/family-regex.v1.json` (dispatch keys / core_scope) · `schema/family-dirs.v1.json` (C2-cost families) | the [F-1]/[F-3]/[F-6] falsifier rows |

## Registration (the shared step) [GAP-P4-REGISTER]

A family does not become discoverable until its `register<Fam>ExtensionPlugin`
free function is entered into the built-in catalog. This is a **real edit to a
shared file** and is easy to miss because it is not one of the five delivery
pieces. Concretely (2026-07-12, code-verified against
`lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp`):

1. Declare the entry point in the family header:
   `llvm::Error register<Fam>ExtensionPlugin(ExtensionPluginRegistry &registry);`
   in `include/TianChenRV/Plugin/<Fam>/<Fam>ExtensionPlugin.h` (definition in
   `<Fam>ExtensionPlugin.cpp`).
2. In `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp` add **two lines**:
   an `#include "TianChenRV/Plugin/<Fam>/<Fam>ExtensionPlugin.h"`, and one row in
   the `kBuiltinExtensionBundles[]` table:
   `{"<fam>-extension-bundle", register<Fam>ExtensionPlugin},`.

### Reconciliation with [F-3] no-core-edit

`BuiltinExtensionPlugins.cpp` lives under `lib/Plugin/Builtin/` (a `core_subdir`,
not a family directory), so it is **core-source** by
`schema/family-manifest.v1.json`'s definition — and editing it in a family PR is
*not* currently listed in that manifest's `shared_allowances`. So as written, the
[F-3] containment gate would flag this edit. Two things resolve the tension:

- **It is not an I3/[F-1] zero-branch violation.** `kBuiltinExtensionBundles[]` is
  a uniform registration *table* (family name as a string literal + a function
  pointer, iterated with a `for` loop, **no `if`/`switch` on the family name**) —
  structurally identical to `lib/Conversion/EmitC/Builtin/BuiltinBackendEmitters.cpp`'s
  `kBuiltinBackendEmitters[]`, which `schema/family-regex.v1.json` already blesses
  as "INTENTIONALLY in scope … zero-core-branch backend registration table." Adding
  a data row to a family-agnostic registration table is the [F-3] "+ table rows"
  spirit, not a core dispatch edit.
- **The containment allowance is LANDED** (`e20bd0f8`·2026-07-12·[裁三.2]·[NEW-STALE-P4-REGISTER-PROSE] 正名).
  `BuiltinExtensionPlugins.cpp`'s registration-table line is the **one legitimate
  shared touch-point** of a family integration. It is now registered in
  `schema/family-manifest.v1.json` `shared_allowances` (alongside `table_rows` /
  `tests` / `docs`) so the [F-3] CI check treats a registration-table row like a
  table row. **⚠ FLAG (schema domain — not edited by this docs pass; owner: main
  session / recon line):** add a `plugin_registration` allowance keyed to
  `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp` (`kBuiltinExtensionBundles[]`
  line only), mirroring the `kBuiltinBackendEmitters[]` precedent. Until then, a
  family PR that touches this file is a *documented, intended* exception, not a
  violation.

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
- [ ] Is the register step done — `register<Fam>ExtensionPlugin` declared in the family header and added to `kBuiltinExtensionBundles[]` in `lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp` ([GAP-P4-REGISTER])?
- [ ] Does the integration stay within the real family touch-set — `lib/{Dialect,Plugin,Target}/<Fam>/` + `include/` mirrors + table rows + docs + the one shared registration-table row ([F-3]; 6 roots, 见 [GAP-P4-TOUCHSET])?
- [ ] Is the interface used as-is without editing core dispatch/selection/lowering files ([P-1]; the registration-table row is the sole intended shared edit)?
- [ ] Is the integration doc followable by a non-core author ([P-4]; use the template in `docs/method/P4-family-integration-doc-TEMPLATE.md`)?

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
