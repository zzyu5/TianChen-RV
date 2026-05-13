# Extension Plugin Integration Contract

This document is a durable integration contract for adding a new extension
plugin to TianChen-RV. It is not a current progress report and it is not a
step-by-step task script.

## Meaning

An extension plugin is an extension family realization provider inside the
unified TCRV RISC-V MLIR system. It is not a completely independent backend,
not a separate backend dialect family, and it must not rewrite TianChen-RV core
orchestration.

A new extension plugin normally provides:

- capability provider;
- extension family operation, type, and attribute registration;
- TCRV interface implementations;
- variant proposal or selected-boundary materialization;
- legality verification;
- cost or preference hook when supported by the interface;
- plugin-owned EmitC lowering mapping and emission route;
- optional target artifact route;
- tests.

The durable construction sequence for those contributions is defined by the
Extension-Family Plugin Construction Protocol:

```text
extension archetype
  -> semantic role graph
  -> extension family declaration
  -> common interface realization
  -> EmitC route mapping
  -> evidence profile
```

The Extension Manifest is the machine-readable entry point to this protocol.
It must describe the family archetype, semantic roles, capabilities,
ops/types/attrs, interface realization, EmitC mapping, and evidence profile.

The plugin may own extension-specific lowering, but it should reuse the common
TianChen-RV capability, variant, selection, dispatch, fallback, runtime ABI,
and artifact-route organization whenever the existing interfaces are
sufficient.

## Current Input Boundary

Current plugin work does not require a high-level MLIR lowering contract before
the plugin can be integrated and tested. This does not ban frontend lowering:
when a frontend owner is chosen, high-level lowering may start from `linalg`
and use hand-written or test `linalg` inputs to drive TianChen-RV MLIR.

Valid current inputs include:

- hand-written TianChen-RV MLIR;
- test TianChen-RV MLIR;
- `tcrv.exec.kernel` with capability anchors;
- already materialized `tcrv.exec.variant`;
- selected lowering-boundary IR;
- `tcrv.exec.mem_window`;
- `tcrv.exec.runtime_param`;
- bounded selected-boundary surfaces.

Existing descriptor-based slices may remain as implementation debt for current
tests and evidence, but a descriptor is not a valid architecture input for new
extension-family design.

High-level op analysis becomes necessary for frontend lowering from `linalg`,
`stablehlo`, `tosa`, or other high-level MLIR. Backend and extension plugin
integration should not wait for that phase: extension plugins may be validated
from TianChen-RV MLIR and selected-boundary IR first, while `linalg` can be the
first high-level frontend path once the backend surfaces are ready enough to
consume it.

## Standard Integration Flow

For a future `TensorExt`, `IME`, or similar extension, the standard path is:

```text
define extension capability ids, such as tensorext or tensorext.fp16
  -> define extension family ops/types/attrs, such as tcrv_tensorext.*
  -> implement TCRV extension/config/resource/memory/compute/EmitC interfaces
  -> register family ops and capabilities through the plugin
  -> connect variant proposal, legality, selected-boundary materialization,
     cost, lowering, and emission hooks through the plugin registry
  -> materialize a plugin-owned lowering boundary or extension ops when selected
  -> generate EmitC, then intrinsic, vendor builtin, runtime C ABI, or object
     artifact through plugin-owned mapping and target-owned emission
  -> reuse tcrv.exec.variant, requires, dispatch, fallback, mem_window,
     runtime_param, runtime ABI, and artifact-route mechanisms
```

The core should see generic route ids, artifact kinds, selected variant
references, and plugin interface results. It should not learn extension
intrinsic names, fragment layouts, custom tile semantics, or runtime-call
details.

## Lowering And Emission Template

A plugin-owned lowering path should fit this template:

```text
tcrv.exec selected variant
  -> plugin-owned lowering boundary or extension op family
  -> plugin-owned config attrs and runtime ABI metadata
  -> plugin-owned emission plan route
  -> common tcrv-lower-extension-to-emitc pass through TCRVEmitCLowerableInterface
  -> EmitC ops
  -> intrinsic / vendor builtin / runtime C ABI / object artifact
```

The selected-boundary operation is a handoff from core selection to plugin
lowering. It is not proof that executable code was generated. The emission
route becomes evidence only when the corresponding target/exporter/conversion
path actually produces the claimed artifact, and RVV runtime or correctness
claims still require `ssh rvv` evidence.

## Core And Plugin Boundary

Core passes may know about:

- `tcrv.exec.kernel`;
- `tcrv.exec.capability`;
- `tcrv.exec.variant`;
- `requires`;
- selected, fallback, and dispatch structure;
- `tcrv.exec.mem_window`;
- `tcrv.exec.runtime_param`;
- generic route ids;
- artifact kind;
- plugin registry interfaces.

Core passes should not know about:

- RVV intrinsic names;
- tensor extension intrinsic names;
- scalar loop semantics;
- offload runtime-call semantics;
- extension-specific fragment layouts;
- extension-specific dtype, tile, or microkernel legality details.

Those details belong in the extension family plugin, family ops/interfaces, or
target-owned lowering/export path.

## Expected File Changes For A New Plugin

Usually allowed:

- new plugin directories;
- new extension family op/type/attr directories;
- new target/exporter directories when needed;
- CMake registration and link entries;
- built-in plugin registration aggregation if the current architecture needs it;
- tests.

Usually not expected:

- extension-specific semantic branches in the core variant selection pass;
- extension-specific branches in the core capability checker;
- extension-specific branches in the core dispatch planner;
- extension-specific branches in the core selected-boundary pass;
- extension-specific semantic switches in the core artifact front door.

If core changes are required, they must extend a public extension interface or a
generic orchestration surface. They must not add a one-off branch for a concrete
extension.

## Recommended Directory Layout

```text
include/TianChenRV/Plugin/<Ext>/
lib/Plugin/<Ext>/

include/TianChenRV/Dialect/<Ext>/IR/
lib/Dialect/<Ext>/IR/

include/TianChenRV/Target/<Ext>/        optional
lib/Target/<Ext>/                       optional

test/Plugin/<Ext>/
test/Dialect/<Ext>/
test/Target/<Ext>/
```

The exact layout should follow existing project conventions, but new extension
code should remain discoverable as plugin, dialect, target/exporter, and test
surfaces rather than being folded into generic core files.

## Recommended Test Template

When adding a plugin, prefer a small set of tests that proves real integration:

- capability recognized and not recognized;
- plugin can register dialects and capabilities;
- variant or selected-boundary is produced only when capability is present;
- illegal config fails closed;
- selected-boundary materialization is plugin-owned;
- lowering or emission route is plugin-owned;
- fallback or dispatch reuses existing core mechanisms when applicable;
- no extension-specific semantic branch was added to core passes.

These tests should validate plugin integration and compiler behavior. They
should not turn into broad smoke matrices or helper-only progress.

See [Extension Family Plugin Template](./extension-family-plugin-template.md)
for the Extension-Family Plugin Construction Protocol, manifest, interface,
common pass, EmitC mapping, directory, and evidence profile expected for new
families.
