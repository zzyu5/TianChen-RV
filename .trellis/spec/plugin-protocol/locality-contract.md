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
                     (+ for own-backend families, ONE kBuiltinBackendEmitters[] row in
                      lib/Conversion/EmitC/Builtin/BuiltinBackendEmitters.cpp; family name as data in
                      for-iterated tables, no branch; 见 [GAP-P4-REGISTER]. Both are declared in
                      schema/family-manifest.v1.json shared_allowances as plugin_registration /
                      backend_emitter_registration; ⚠ the [F-3] checker does not yet READ shared_allowances,
                      so they are reviewer-waved, not machine-waved — main-session/recon)
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

### `Family-Onboarding:` trailer & the [F-2'] schema-shape freeze

An onboarding PR must edit several `schema/**` **table-row** files — its own family
block in `schema/family-manifest.v1.json` (+ a `baseline_range_count` bump), its
dispatch keys in `schema/family-regex.v1.json`, and its coverage rows in
`schema/coverage-*.json`. Those edits are exactly the [F-3] "+ 表行" allowance
(above), **not** edits to the frozen schema *shape*. The distinction is enforced by
the [F-2'] gate, whose **authoritative implementation is
`.trellis/scripts/check_schema_gate.py`** (this spec only indexes it, per [P-1] —
it does not redefine the rule):

- **Marker.** An onboarding PR is identified by a `Family-Onboarding:` commit
  trailer in `base..head` (constant `ONBOARDING_TRAILER`; interim until the
  `plugins/<family>/` layout lands). A PR carrying the trailer is graded under the
  onboarding rule; one without it is graded as a core-author *evolution* PR (RFC +
  `classify_schema_change`).
- **Rule ([F-2']).** `is_f2prime_violation = is_onboarding AND (any changed file is
  under the schema governance path)`. The *intent* is to freeze the schema **shape**
  (`schema/capability.schema.v1.json` + its `VERSIONLOG.md`): an onboarding author
  reaching into the schema.def — *even to add a field* — is the falsifier firing,
  never waved through as "additive". Onboarding must integrate through the existing
  frozen shape, adding only table-row data.
- **⚠ FLAG (checker semantics — main-session / recon line; NOT changed by this docs
  pass).** As currently coded, `SCHEMA_GATE_PREFIX` is the **whole `schema/`
  prefix**, so [F-2'] fires on *any* `schema/` file — including the family
  table-row files (`family-manifest.v1.json` etc.) that [F-3] *requires* an
  onboarding PR to edit. So a single onboarding PR that both carries the trailer
  **and** adds its `family-manifest` block would currently trip [F-2'] RED — a
  direct collision with [F-3]. Reconciling this is a `check_schema_gate.py` behavior
  decision (narrow `SCHEMA_GATE_PREFIX` to the schema *shape* — `capability.schema.v1.json`
  + `VERSIONLOG.md` — and exempt the family table-row schemas; **or** require the
  onboarding PR to split its `schema/**` table-row edits into a separate,
  non-`Family-Onboarding:`-trailered commit). Until then, treat the table-row schema
  edits as the [F-3] allowance and keep the trailer off the commit that carries
  them.

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
