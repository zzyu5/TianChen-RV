# Extension Family Plugin Template

## Scope

This template is a compact checklist for adding a future extension family. It
is not an executable manifest system, semantic role graph engine, construction
template artifact pipeline, readiness dashboard, or source-front-door default.

RVV is the first and broadest real reference family; per-family build/maturity
status lives in tasks/journal, not here
(见 [../guides/trunk-discipline.md](../guides/trunk-discipline.md)).

## Real Touch-Set (Reference Family = `Template/`) [GAP-P4-TOUCHSET]

> **Accuracy note (2026-07-12, [GOV-8] code-verified).** The "Required Plugin
> Sections" below are the *conceptual* pieces. Do **not** read them as "a family
> = 5 files under one directory." The real, linking touch-set of the reference
> `Template` family spans **6 source directory roots, 20 source files, 6
> `CMakeLists.txt`, plus 1 shared registration file** — because the plugin library
> transitively links a family `Dialect` library and a family `Target` library
> (verified from `lib/Plugin/Template/CMakeLists.txt`: `WeftTemplatePlugin`
> `LINK_LIBS … WeftTemplateDialect WeftTemplateTarget`). Copying only
> the 5 `lib/Plugin/<Fam>/` files and running `cmake --build` fails at link with
> missing `Weft<Fam>Dialect` / `Weft<Fam>Target` targets.

```text
lib/Plugin/<Fam>/                      5 .cpp + CMakeLists  → 4 libs
  <Fam>ExtensionPlugin.cpp               plugin entry: getCapabilities(), registerDialects(),
                                         free fn register<Fam>ExtensionPlugin(registry)
  <Fam>VariantLegality.cpp               legality predicate
  <Fam>ConstructionProtocol.cpp          abstract op → in-compiler typed body (front-door family side)
  <Fam>EmitCRouteProvider.cpp            selected variant → WEFTEmitCLowerableRoute
  <Fam>BackendEmissionDriver.cpp         backend body emission
include/Weft/Plugin/<Fam>/       4 headers (mirror the 4 non-legality .cpp; legality is internal)
lib/Dialect/<Fam>/                     CMakeLists (add_subdirectory IR)                     ← MANDATORY link dep
lib/Dialect/<Fam>/IR/                  <Fam>Dialect.cpp + CMakeLists → lib Weft<Fam>Dialect
include/Weft/Dialect/<Fam>/      CMakeLists
include/Weft/Dialect/<Fam>/IR/   <Fam>Dialect.h + <Fam>Ops.td + CMakeLists (ODS/TableGen)
lib/Target/<Fam>/                      <Fam>TargetSupportBundle.cpp + CMakeLists → lib Weft<Fam>Target  ← MANDATORY link dep
include/Weft/Target/<Fam>/       <Fam>TargetSupportBundle.h
lib/Plugin/Builtin/BuiltinExtensionPlugins.cpp   +1 SHARED registration (见 [GAP-P4-REGISTER] in
                                       [extension-plugin-integration.md](./extension-plugin-integration.md))
```

The 6 `CMakeLists.txt` are: `lib/Plugin/<Fam>/`, `lib/Dialect/<Fam>/`,
`lib/Dialect/<Fam>/IR/`, `include/Weft/Dialect/<Fam>/`,
`include/Weft/Dialect/<Fam>/IR/`, `lib/Target/<Fam>/`. The `Plugin/` and
`Target/` include mirrors carry no `CMakeLists` (their headers are pulled by the
`lib/` targets); only the `Dialect/` side has extra `CMakeLists` for ODS
TableGen (`MLIR<Fam>OpsIncGen`).

**Relation to [F-3] containment.** All 6 roots follow the `lib/{Dialect,Plugin,Target}/<Fam>/`
+ `include/` mirror shape that `schema/family-manifest.v1.json` already declares as
one family's owned territory, so they are *family-local*, not core — the wide
touch-set is not an [F-3] violation. The one genuinely shared file is the Builtin
registration (see [GAP-P4-REGISTER]). [F-3]'s prose "`plugins/<family>/`" is a
generic stand-in for this real 6-root shape (the code lives under
`lib/{Dialect,Plugin,Target}/<Fam>/`, not a literal `plugins/` directory).

## Required Plugin Sections

### 1. Family Identity

Document:

```text
family name
concrete MLIR namespace
plugin name/version
target capability ids
toolchain/runtime assumptions
maturity status (real / not yet built)
```

### 2. Capability Contract

Define structured capability ids, properties, relations, and unavailable
diagnostics. Capability facts constrain legality/realization; they do not
create route ids, dtype authority, or artifact paths.

### 3. Typed Body / Boundary

Define the typed extension-family body or selected boundary consumed by the
plugin. The body must own extension compute/config/control/dataflow. For RVV,
the exemplar surface is:

```text
!weft_rvv.vector<elem = i32, lmul = m1>
weft_rvv.setvl
weft_rvv.load
weft_rvv.binary {kind = add}
weft_rvv.store
```

Avoid finite dtype-prefixed namespaces such as `rvv-finite-binary` or
`weft_rvv.i32_*` as architectural templates.

### 4. Legality

State plugin-owned legality rules and generic core prerequisites. Core may
check generic structure and capability presence; extension legality remains in
the plugin.

### 5. Selected-Body Realization

If hints/config/profile affect generated code, add a plugin-local realization
hook:

```text
selected pre-realized body
  + target capability
  + runtime SSA / ABI values
  + hints / policy / profile
    -> realized selected body
```

Do not persist this as a readiness state machine.

### 6. Route Provider

Build a provider-owned `WEFTEmitCLowerableRoute` from the legal typed/realized
body. Common EmitC materializes this route; it does not invent extension
semantics.

### 7. Runtime / Artifact Evidence

Define when runtime, correctness, or performance claims require hardware or
runtime evidence. For RVV, such claims require real `ssh rvv` evidence.

### 8. Tests

Attach tests to production-path changes:

- lit/FileCheck for syntax, verification, pass behavior, diagnostics;
- C++ tests for registry/provider APIs;
- negative tests for metadata-only/source-front-door/legacy-route authority;
- runtime evidence when making runtime/correctness/performance claims.

## Optional Provenance

Manifests, semantic role graphs, construction templates, and example
source-front-door snippets may be kept as planning/provenance notes only. They
cannot be executable source authority, route authority, dtype authority,
artifact authority, or progress proof.

## Common Pass Names

Use generic orchestration names that reflect ownership:

```text
weft-select-variants
weft-realize-selected-bodies
weft-materialize-emitc-lowerable-routes
weft-lower-routes-to-emitc
weft-export-target-artifact
```

Avoid:

```text
weft-select-route
```

because route construction is plugin-owned and common/core only orchestrate
variant selection and route materialization.

## Source Front-Door Policy

Default policy is explicit-only or disabled (source-front-door routes fail
closed, 见 core-invariants I7). Positive source-front-door workflows for Toy,
Template, TensorExtLite, IME, Offload, or future plugins are future examples.
They must not be copied into current RVV work.

## Good / Bad Template Use

Good:

```text
plugin-owned typed body
  -> plugin legality
  -> plugin route provider
  -> common EmitC
```

Bad:

```text
construction manifest
  -> object/header bundle bridge
  -> supported route
```

Bad:

```text
semantic role graph
  -> compute/dtype/source authority
```

Bad:

```text
Template/Toy/TensorExtLite source marker
  -> current positive workflow for a not-yet-built family
```
