# Common Executable Plugin Construction Template From Materialized EmitC Bundles

## Goal

Extract the repeated RVV and TensorExtLite materialized EmitC object/header/bundle construction pattern into one common C++ construction surface that production plugin target-support code consumes. The common surface should own generic selected-artifact envelope validation, selected materialized EmitC provenance checks, runtime ABI signature checks, object/header bundle component identity, target artifact registration shape, and fail-closed behavior. RVV and TensorExtLite must remain responsible for their extension semantics, typed role validation, EmitC route builders, intrinsic/vendor/runtime payloads, compiler flags, and package/export callbacks.

This is a migration/refactor task, not a new extension-family feature.

## Current Repository Facts

- Current HEAD before this task is `49c771e tensorext: add materialized emitc artifact bundle`, with a clean worktree.
- No `.trellis/.current-task` existed at session start; this task was created from the Hermes Direction Brief as `05-17-common-executable-plugin-construction-template`.
- RVV and TensorExtLite both have materialized EmitC object routes plus object-backed declaration-only header composite routes.
- RVV and TensorExtLite currently duplicate the same target-support glue shape:
  - classify a selected object candidate as owned by the plugin/materialized route;
  - require exactly one selected supported object candidate for the header/bundle composite path;
  - validate the candidate before header/bundle acceptance;
  - provide a composite match callback;
  - provide a composite candidate-validation callback;
  - provide bundle metadata with component group and handoff identity;
  - register matching object and header target artifact exporters.
- The existing common `TargetArtifactExport` layer already has reusable selected EmitC route materialization, declaration-only header export, target artifact candidate validation, composite exporter registration, bundle record collection, component-group contract validation, and zero-argument bundle ABI support.
- The existing `Common Materialized EmitC Header Artifact Foundation` spec already authorizes common target code to validate generic selected candidate fields and materialized EmitC provenance, while keeping plugin-local construction protocol metadata and route mapping in plugin code.

## Requirements

- Add a common C++ construction interface/profile for materialized EmitC object-backed header/bundle registrations.
- The common profile must let plugin code provide only local facts:
  - owner plugin;
  - selected object route id and artifact kind;
  - object emission kind;
  - header route id and artifact kind;
  - bundle component group;
  - optional external ABI name;
  - object handoff kind;
  - route-local object candidate preflight callback;
  - route-local object export callback;
  - route-local header export callback;
  - runtime ABI parameter source, including valid zero-argument signatures.
- The common profile must provide reusable callbacks or registration helpers for:
  - exactly-one selected object candidate selection;
  - composite header match;
  - composite candidate validation;
  - runtime ABI signature extraction;
  - bundle metadata construction;
  - object/header exporter registration.
- RVV must consume the common construction surface for its materialized EmitC object/header/bundle route registration while preserving its RVV-specific arithmetic candidate validation, dynamic runtime ABI identity, RVV route builder, RVV intrinsic payload, `rv64gcv` object packaging callback, and RVV provenance evidence.
- TensorExtLite must consume the common construction surface for its materialized EmitC object/header/bundle route registration while preserving its TensorExtLite-specific role-sequence validation, selected lowering-boundary checks, TensorExtLite route builder, `rv64gc` object packaging callback, zero-argument ABI, object-backed declaration-only header behavior, and TensorExtLite provenance evidence.
- Remove or reduce duplicated plugin-local selected-candidate/composite/header/bundle registration glue where the common interface now owns it.
- Keep fail-closed behavior for stale descriptor/direct-C/source-export route metadata, missing materialized EmitC provenance, unsupported artifact kinds, mismatched runtime ABI parameter signatures, mixed component groups, ambiguous candidates, and unsupported plugin routes.
- Do not add descriptor tables, direct C semantic exporters, source-export routes, Python compiler-core behavior, high-level frontend lowering, new extension families, TensorExt runtime execution, RVV performance claims, or common branches on RVV/TensorExtLite semantics.

## Acceptance Criteria

- [x] A common C++ surface in the target artifact export layer is code-consumed by both RVV and TensorExtLite materialized EmitC object/header/bundle registrations.
- [x] RVV still exports its expected materialized EmitC object, declaration-only header, and object/header bundle artifacts through materialized EmitC authority.
- [x] TensorExtLite still exports its expected materialized EmitC object, declaration-only header, and object/header bundle artifacts through materialized EmitC authority.
- [x] Focused C++ tests prove the common construction surface registers object/header routes and produces fail-closed diagnostics for missing materialized EmitC provenance, mismatched runtime ABI signatures including zero-argument cases, mixed component groups, stale descriptor/direct-C/source-export route metadata, and unsupported plugin routes.
- [x] Focused lit/FileCheck tests continue to cover one RVV and one TensorExtLite materialized EmitC object/header/bundle path.
- [x] Targeted scans over changed common/plugin target surfaces and tests show no descriptor route authority, no direct C semantic exporter, no source-export route resurrection, and no common RVV/TensorExtLite semantic branch.
- [x] `git diff --check` passes.
- [x] Focused build/tests for changed target support and target artifact export pass; full `check-tianchenrv` is run if practical.

## Completion Evidence

- Added `target::MaterializedEmitCObjectBundleArtifactConfig` plus common
  selected object candidate selection, header composite match/validation,
  runtime ABI parameter extraction, bundle metadata construction, and object/
  header exporter registration helpers.
- Migrated RVV target support so `registerRVVI32M1ArithmeticTargetArtifactExporter`
  calls `registerMaterializedEmitCObjectBundleArtifactExporters` while keeping
  RVV-specific arithmetic validation, dynamic runtime ABI identity, EmitC route
  builder, object packaging, and provenance evidence local.
- Migrated TensorExtLite target support so
  `registerTensorExtLiteTargetArtifactExporter` calls the same common helper
  while keeping TensorExtLite role-sequence validation, selected lowering
  boundary checks, zero-argument ABI, object/header export callbacks, and
  provenance evidence local.
- Updated `TargetArtifactExportTest.cpp` with common construction-surface
  coverage and production RVV/TensorExtLite assertions, including TensorExtLite
  zero-argument runtime ABI extraction through the common path.
- Updated `.trellis/spec/lowering-runtime/emitc-route.md` only to document the
  code-owned interface that is now wired into RVV and TensorExtLite.
- Checks passed:
  `cmake --build build --target tianchenrv-target-artifact-export-test -j2`;
  `./build/bin/tianchenrv-target-artifact-export-test`;
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`;
  focused lit filter
  `vector-source-target-artifact-(object|header)|tensorext-lite-source-front-door-target-artifact-(object|header|bundle)|tensorext-lite-target-artifact-header`
  from `build/test`: 6/6 selected tests passed;
  `cmake --build build --target check-tianchenrv -j2`: 112/112 lit tests
  passed;
  `git diff --check`;
  final `./build/bin/tianchenrv-target-artifact-export-test`.
- Targeted scans:
  common `TargetArtifactExport` surfaces contain descriptor/direct-C/
  source-export strings only in fail-closed forbidden-metadata logic and no
  RVV/TensorExtLite semantic branch;
  plugin/test hits are plugin-local validation or `CHECK-NOT`/negative
  assertions.
- `ssh rvv` evidence was not run because this round refactored local
  object/header/bundle construction registration and did not change RVV
  runtime execution or performance claims.

## Out Of Scope

- No new extension family, TensorExt runtime execution, RVV SEW/LMUL/dtype expansion, high-level frontend lowering, RVV correctness/performance claim, or new `ssh rvv` runtime evidence unless object/header/bundle behavior changes in a way that affects executable RVV proof.
- No descriptor-driven computation, descriptor compatibility layer, historical route ID restoration, direct C semantic exporter, source-export route, artifact ledger protocol, checkpoint protocol, or common/core semantic branch on RVV/TensorExtLite semantics.
- No docs-only template or passive manifest without production RVV/TensorExtLite callers.
- No broad test matrix beyond the changed module behavior unless focused checks expose cross-module fallout.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous PRD read:
  `.trellis/tasks/archive/2026-05/05-17-tensorext-lite-materialized-emitc-artifact-bundle-bridge/prd.md`.
- Current code/test surfaces inspected:
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `include/TianChenRV/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.h`,
  `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `test/Target/RVV/`, and
  `test/Target/TensorExtLite/`.

## Finish Criteria

- Implementation is wired into production RVV and TensorExtLite target-support registration paths.
- Focused validation passes or failures are explained as concrete deletion/rebuild gaps without restoring wrong logic.
- Task status, journal, and archive state are truthful.
- One coherent commit is created if the task is complete.
