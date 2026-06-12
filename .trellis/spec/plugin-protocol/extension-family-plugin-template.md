# Extension Family Plugin Template

## Scope

This template is a compact checklist for adding a future extension family. It
is not an executable manifest system, semantic role graph engine, construction
template artifact pipeline, readiness dashboard, or source-front-door default.

RVV is the current real/mature family. Template, Toy, TensorExtLite, IME,
Offload, and future plugin positive examples are not built yet
(见 [../guides/trunk-discipline.md](../guides/trunk-discipline.md)).

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
!tcrv_rvv.vector<elem = i32, lmul = m1>
tcrv_rvv.setvl
tcrv_rvv.load
tcrv_rvv.binary {kind = add}
tcrv_rvv.store
```

Avoid finite dtype-prefixed namespaces such as `rvv-finite-binary` or
`tcrv_rvv.i32_*` as architectural templates.

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

Build a provider-owned `TCRVEmitCLowerableRoute` from the legal typed/realized
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
tcrv-select-variants
tcrv-realize-selected-bodies
tcrv-materialize-emitc-lowerable-routes
tcrv-lower-routes-to-emitc
tcrv-export-target-artifact
```

Avoid:

```text
tcrv-select-route
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
