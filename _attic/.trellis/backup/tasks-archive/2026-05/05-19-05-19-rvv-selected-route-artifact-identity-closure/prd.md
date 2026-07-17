# RVV selected-route artifact identity closure

## Goal

Close the remaining RVV selected-route identity authority gap. Route IDs,
target artifact route IDs, runtime ABI labels, intrinsic spellings, and
retained i32m1 names must be provider-derived specialization facts after an
explicit typed `tcrv_rvv` selected body has been validated. Construction,
emission planning, and target/export code may mirror those facts, but must not
infer RVV semantics from manifest route family names, stale artifact metadata,
source-front-door labels, or exact `__riscv_*_i32m1` strings.

## Background

- Commit `557d5ab` made construction metadata mirror selected provider route
  facts instead of accepting stale construction metadata as export authority.
- Current bounded inspection still shows RVV route identity labels spread
  across provider, construction, source-front-door, and target bundle code:
  `kRVVSelectedBodyRouteFamilyID`, `kSelectedBodyRoutes`,
  `getRVVConstructionManifest().emitcRoute.routeID`, `getRVVI32M1*` helpers,
  and exact `__riscv_*_i32m1` intrinsic spellings.
- Some i32m1 strings are valid as the retained bounded specialization, but only
  after provider analysis has validated selected typed body structure, config,
  runtime ABI roles, memory form, operation kind, and policy.

## Requirements

- Add or centralize a provider-owned selected-body route identity fact set that
  contains the current target artifact route ID/kind and per-operation
  specialization facts.
- `describeRVVSelectedBodyEmitCRoute` must derive the final route/artifact/ABI
  labels and exact intrinsic spellings from validated typed-body/config/runtime
  facts, then verify the assembled description before consumers see it.
- Construction metadata must remain a mirror of the provider description; any
  construction-side route table is allowed only as retained specialization
  validation, not as target/export route authority.
- Target/export static registration may use a selected-body target artifact
  route label for route lookup, but candidate validation must still rebuild the
  provider description before consuming route, artifact, runtime ABI, or
  metadata labels.
- Source-front-door i32m1 helpers may remain only as an explicit plugin-owned
  typed-body seed. They must not be default source-artifact authority and must
  not authorize export without the downstream selected typed-body provider
  validation.
- Negative tests must prove stale route ID, target artifact route ID, runtime
  ABI label, intrinsic/config mismatch, or source-front-door-only labels fail
  closed before route/export.

## Acceptance Criteria

- [x] Provider description exposes one coherent selected-body route identity
  fact set used by construction and target code for route/artifact/runtime
  labels.
- [x] Any remaining `getRVVConstructionManifest().emitcRoute.routeID` or
  `kRVVSelectedBodyRouteFamilyID` use is not a route decision, or is replaced by
  an explicitly derived selected-body target artifact route label.
- [x] Exact `__riscv_*_i32m1` strings are reachable only through
  selected-body specialization mapping after SEW, LMUL, policy, memory form,
  operation, and runtime ABI validation.
- [x] Positive focused coverage still proves the explicit typed i32m1 selected
  body materializes/export-validates through provider-derived facts.
- [x] Negative focused coverage proves stale route/artifact/runtime ABI labels
  and mismatched selected-body config/intrinsic specialization fail closed
  before target artifact export.
- [x] Bounded residue scan over RVV provider/construction/source/target files
  shows no active compiler path treating route IDs, i32m1 helpers, exact
  intrinsic spelling, descriptor/direct-C/source-export residue, or artifact
  metadata as independent RVV route authority.

## Non-Goals

- Do not add RVV coverage such as broadcast expansion, compare/select expansion,
  reduction, conversion, new dtype, new LMUL, new source shape, or new
  intrinsic cases.
- Do not generalize Linalg, Vector, StableHLO, or frontend lowering.
- Do not move RVV semantic choice into common EmitC/export.
- Do not preserve old i32 route authority through compatibility wrappers.
- Do not do aesthetic-only renames; rename or factor only where it removes
  authority confusion or makes specialization provenance fail-closed.
- Do not work on Scalar, IME, Offload, TensorExt, autotuning, dashboards, broad
  smoke matrices, report/status machinery, or artifact-index-only evidence.
- Do not claim runtime, correctness, or performance without real `ssh rvv`
  evidence.

## Relevant Specs And Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-construction-manifest-route-authority-demotion/prd.md`

## Initial Code Surface

- `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- Focused tests in `test/Plugin`, `test/Conversion/EmitC`, `test/Target`,
  and `test/Transforms/RVV` that exercise selected-body route identity.

## Validation Plan

- Build and run focused C++ tests:
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`, and
  `tianchenrv-target-artifact-export-test`.
- Build `tcrv-opt` and `tcrv-translate`.
- Run focused lit tests for RVV selected-body EmitC materialization, RVV target
  artifact export, and RVV source-front-door negative/positive route identity
  cases touched by the change.
- Run bounded ref-scans over the listed RVV files for `manifest.emitcRoute.routeID`,
  `kRVVSelectedBodyRouteFamilyID`, `kSelectedBodyRoutes`, `getRVVI32M1`,
  exact `__riscv_*_i32m1`, descriptor/direct-C/source-export residue, and
  metadata-as-authority.
- Run `python3 ./.trellis/scripts/task.py validate` for this task and
  `git diff --check`.

## Implementation Notes

- Added `getRVVSelectedBodyTargetArtifactRouteID()` and
  `getRVVSelectedBodyTargetArtifactKind()` so provider/target code no longer
  scatters direct manifest route-family reads when it needs selected-body
  artifact identity labels.
- Renamed the construction-local retained i32m1 table from an active
  route-table-shaped name to `kRetainedSelectedBodySpecializations` and marked
  it as post-validation specialization facts, not public route authority.
- Added `verifyRVVSelectedBodyEmitCRouteDescription`, and the provider now runs
  it after selected typed body/config/runtime ABI/role validation assembles the
  route description. The verifier checks target artifact route/kind, EmitC route
  ID, runtime ABI label/contract, boundary, vector/mask C types, exact RVV
  intrinsic spellings, and runtime ABI parameters against the selected-body
  specialization mapping.
- Updated RVV target support and focused tests to consume the explicit
  selected-body target artifact label instead of re-reading the construction
  manifest route ID.
- Added provider-level negative checks for stale intrinsic and stale target
  artifact route description labels.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` to record the new
  provider description verifier signature and fail-closed identity contract.

## Checks Run

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-construction-protocol-common-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-first-slice|vector-materialized-target-artifact-exporters|emitc-to-cpp|rvv-i32m1-vector-source-front-door'` from `build/test`: 21/21 selected tests passed.
- Bounded residue scan over RVV provider/construction/source/target files for
  old route-family symbols, manifest route reads, retained i32m1 helpers,
  exact RVV intrinsic strings, descriptor/direct-C/source-export residue, and
  metadata-as-authority.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-19-05-19-rvv-selected-route-artifact-identity-closure`
- `git diff --check`
