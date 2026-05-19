# Plugin Protocol Specs

This layer defines how hardware/runtime extensions integrate with TianChen-RV.

## Pre-Development Checklist

- [ ] Is new extension-specific logic placed inside a plugin?
- [ ] Does the plugin contribute an extension family to the unified TCRV system rather than an independent backend dialect?
- [ ] Does the plugin follow the Extension-Family Plugin Construction Protocol without treating manifests, semantic role graphs, or templates as executable authority?
- [ ] If an Extension Manifest exists, is it optional scaffolding/provenance rather than source, route, dtype, or compute authority?
- [ ] Does the plugin register capabilities, family ops/types/attrs, interfaces, variant builders, legality, tuning, cost, and EmitC emission mapping?
- [ ] Does core code call registry/interface APIs rather than `hasRVV`/`hasIME`/`hasSophgo` branches?
- [ ] If core interface changes are needed, are they justified by a genuinely new execution semantic?
- [ ] Can current plugin work start from TianChen-RV MLIR, selected-boundary IR, or typed extension-family bodies without requiring high-level op lowering first?
- [ ] Is selected-path metadata limited to diagnostic/control mirrors, never compute or route input?
- [ ] Does the work avoid descriptor-driven computation and descriptor-driven
      C/source export as a long-term path?
- [ ] Does documentation state that pluginization is local work, not zero work?
- [ ] Are plugin interfaces, registries, dialect registrations, and lowering hooks implemented in C++/MLIR rather than Python?

## Guidelines Index

| Spec | Description |
|---|---|
| [Interfaces And Registry](./interfaces-and-registry.md) | Required provider interfaces and core pass usage |
| [Locality Contract](./locality-contract.md) | Plugin/core boundaries and extension-locality evaluation |
| [Extension Plugin Integration Contract](./extension-plugin-integration.md) | Template for adding future extension plugins without rewriting core passes |
| [Extension Family Plugin Template](./extension-family-plugin-template.md) | Extension-Family Plugin Construction Protocol, optional manifest/provenance, common interfaces, common passes, EmitC mapping, and evidence profile for new families |

## Quality Check

- Adding a plugin should mainly add family/plugin files, interface
  implementations, EmitC mapping, registrations, and tests.
- Core pass diffs should show generic orchestration, not extension-specific lowering logic.
- Any new core branch mentioning a concrete extension must be reviewed as a likely violation.
- Plugin protocol behavior should be covered by lit/FileCheck for IR behavior and C++ tests for registry/interface APIs where useful.
- A new plugin should be testable from hand-written or test TianChen-RV MLIR before high-level `linalg`/`stablehlo`/`tosa` lowering exists.
- Descriptor-driven computation must not be used as the template for adding a
  new extension family.
- Source-front-door defaults must be explicit-only or disabled. `Eligible` is
  never a safe default for current RVV Stage1/Stage2 work.
- A new extension's "fast path" should reduce architecture decision search,
  not avoid family-specific ops/types/attrs/verifiers/EmitC mapping/tests.
