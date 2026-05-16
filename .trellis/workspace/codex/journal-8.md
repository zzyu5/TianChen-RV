# Journal - codex (Part 8)

> Continuation from `journal-7.md` (archived at ~2000 lines)
> Started: 2026-05-16

---



## Session 93: RVV construction-backed hardware artifact proof

**Date**: 2026-05-16
**Task**: RVV construction-backed hardware artifact proof
**Branch**: `main`

### Summary

Refreshed current-HEAD proof that RVV i32m1 add/sub/mul selected paths cross construction-checked EmitC, target object/header export, bundle metadata, and real ssh rvv correctness evidence without requiring source rewiring.

### Main Changes

- Created and archived Trellis task `05-16-rvv-construction-backed-hardware-artifact-proof` from the Hermes direction brief.
- Wrote the PRD around current-HEAD proof of the existing RVV i32m1 add/sub/mul route instead of expanding RVV coverage.
- Confirmed no source-code repair was required: RVV construction mapping, route provider validation, selected `tcrv_rvv.with_vl`, common selected EmitC artifact materialization, object/header export, and translate routes are already wired.
- Generated current add/sub/mul default and exact object/header artifacts plus bundle indexes under `artifacts/tmp/rvv_construction_backed_hardware_artifact_proof/20260516T125646Z`.
- Bundle metadata records route ids, callable component groups, `plugin-owned-runtime-abi`, and ordered `lhs`, `rhs`, `out`, `n` target-export ABI parameters.
- Linked and ran the generated add/sub/mul objects on `ssh rvv` through `rvv_i32m1_arithmetic_harness.c`; remote output was `tcrv_rvv_i32m1_arithmetic_current_head status=PASS n=4 add=[12,6,16,12] sub=[2,-12,24,0] mul=[35,-27,-80,36]`.
- Focused build, RVV plugin/unit tests, target artifact export test, construction/dialect tests, focused RVV lit set, full `check-tianchenrv`, `git diff --check`, and legacy/common-core scans passed.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | task(rvv): record construction-backed hardware proof |

### Testing

- [OK] Focused build for RVV construction protocol, RVV EmitC route provider,
  RVV plugin, RVV target support, target artifact export, `tcrv-opt`, and
  `tcrv-translate`.
- [OK] RVV plugin/unit tests, target artifact export test, construction
  protocol test, and RVV dialect test.
- [OK] Focused lit filter for RVV EmitC materialization, Target/RVV i32m1, and
  selected with_vl boundary tests: 17/17 passed.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 102/102 passed.
- [OK] `ssh rvv` add/sub/mul external ABI harness passed.
- [OK] `git diff --check`.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 97: Registry-composed source-seed-to-artifact front door

**Date**: 2026-05-16
**Task**: Registry-composed source-seed-to-artifact front door
**Branch**: `main`

### Summary

Added an explicit `tcrv-opt` source-seed artifact front-door pipeline. The new
pipeline collects plugin-registered source-seed materialization passes in
registry order, then runs generic legality/capability, emission-plan, and
execution-plan coherence gates so bounded RVV and Toy source seeds can feed the
existing target artifact routes without direct Toy/RVV wiring in public tools or
common transforms.

### Main Changes

- Created Trellis task
  `05-16-registry-composed-source-seed-front-door` from the supplied Direction
  Brief and wrote the PRD around a registry-composed front door.
- Added `--tcrv-source-seed-artifact-front-door-pipeline`.
- The pipeline runs enabled plugin source-seed passes, then
  `tcrv-check-hart-parallel-capabilities`,
  `tcrv-verify-plugin-variant-legality`,
  `tcrv-check-capability-requires`, `tcrv-materialize-emission-plans`, and
  `tcrv-check-execution-plan-coherence`.
- Kept the ordinary `tcrv-execution-planning-pipeline` unchanged; source-seed
  outputs already contain selected variant/boundary surfaces and should not be
  sent through proposal/selection again.
- Added RVV/Toy positive lit coverage proving both source seeds exercise the
  same pipeline and feed the existing target artifact routes.
- Added negative lit coverage for disabled built-ins, unsupported source seed
  markers, stale residue, and mixed incompatible RVV+Toy seed inputs.
- Updated the variant-pipeline spec with the durable public contract for the
  source-seed artifact front-door pipeline.

### Testing

- [OK] Focused build:
  `cmake --build build --target TianChenRVTransforms TianChenRVPlugin
  TianChenRVRVVPlugin TianChenRVToyPlugin tcrv-opt tcrv-translate
  tianchenrv-plugin-registry-test tianchenrv-rvv-extension-plugin-test
  tianchenrv-toy-extension-plugin-test -j2`.
- [OK] C++ tests:
  `tianchenrv-plugin-registry-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-toy-extension-plugin-test`.
- [OK] Focused lit from `build/test`:
  `source-seed-artifact-front-door|rvv-i32m1-selected-boundary-seed|toy-template-selected-boundary-seed|i32m1-add-object-artifact`,
  8/8 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 109/109
  passed.
- [OK] `git diff --check`.
- [OK] Tool direct seed-factory scan returned no matches under
  `tools/tcrv-opt` and `tools/tcrv-translate`.
- [OK] Common/core source-seed semantic branch scan returned no matches under
  `include/TianChenRV/Transforms`, `lib/Transforms`,
  `include/TianChenRV/Plugin/ExtensionPlugin.h`, and
  `lib/Plugin/ExtensionPlugin.cpp`.
- [OK] RVV front-door artifacts generated under
  `artifacts/tmp/rvv_selected_boundary_seed_frontdoor/20260516T143133Z`.
- [OK] `ssh rvv` link/run printed
  `tcrv_rvv_i32m1_selected_boundary_seed status=PASS n=4 add=[12,6,16,12]`.

### Self-Repair

- Avoided composing source seeds with `tcrv-execution-planning-pipeline` after a
  live probe showed the proposal stage correctly rejects already selected seed
  variants.
- Avoided unconditional selected lowering-boundary materialization after a live
  probe showed it duplicates the Toy seed boundary.
- Repaired the disabled-builtins negative test to expect the final generic
  coherence fail-closed diagnostic.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 95: Plugin-owned source seed registration interface

**Date**: 2026-05-16
**Task**: Plugin-owned source seed registration interface
**Branch**: `main`

### Summary

Moved the RVV bounded selected-boundary source seed pass behind a common
extension-plugin source-seed registration surface. `tcrv-opt` now registers the
seed by collecting enabled plugin registrations instead of directly including
or invoking the RVV pass factory.

### Main Changes

- Created and archived Trellis task
  `05-16-plugin-owned-source-seed-registration-interface` from the supplied
  Direction Brief.
- Added `SourceSeedPassRegistration` and
  `ExtensionPluginRegistry::collectSourceSeedPasses` as common pass
  descriptor/factory plumbing.
- Added `ExtensionPlugin::registerSourceSeedPasses` with default no-op
  behavior, and moved the existing RVV i32m1 selected-boundary seed factory
  behind `RVVExtensionPlugin`.
- Removed direct `RVVSelectedBoundarySeed.h` include and direct RVV seed pass
  factory registration from `tools/tcrv-opt/tcrv-opt.cpp`.
- Removed the direct `TianChenRVRVVPlugin` link from `tcrv-opt`; the tool now
  reaches built-in RVV behavior through built-in plugin/target registration.
- Added registry and RVV plugin C++ coverage for source-seed pass registration,
  plus lit coverage proving `--tcrv-disable-builtin-plugins` leaves the RVV
  seed pass unregistered.
- Promoted the source-seed pass provider interface rules into
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | plugin(registry): route source seeds through plugins |

### Testing

- [OK] Focused build:
  `cmake --build build --target TianChenRVPlugin TianChenRVRVVPlugin tcrv-opt
  tcrv-translate tianchenrv-plugin-registry-test
  tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] C++ tests:
  `tianchenrv-plugin-registry-test`,
  `tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused lit from `build/test`:
  `rvv-i32m1-selected-boundary-seed|rvv-first-slice-materialization|Target/RVV/i32m1|rvv-with-vl-selected-boundary`,
  19/19 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 104/104
  passed.
- [OK] `git diff --check`.
- [OK] Public-tool direct RVV seed wiring scan returned no matches for
  `RVVSelectedBoundarySeed` or
  `createMaterializeRVVI32M1SelectedBoundarySeedPass` under `tools/`.

### Hardware Evidence

Not rerun. This round changed pass registration ownership only; selected
boundary output, emission-plan/EmitC route semantics, target artifact semantics,
and runtime ABI shape are unchanged and covered by focused lit. The previous
seed task already produced `ssh rvv` correctness evidence for the same route.

### Self-Repair

- First focused build exposed an incomplete `mlir::Pass` type in
  `RVVExtensionPlugin.cpp`; added `mlir/Pass/Pass.h` and reran successfully.
- First lit attempt used `lit.py build/test` from repo root and hit this repo's
  generated relative `lit.site.cfg.py` path. Reran from `build/test`, which is
  the project convention, and the focused set passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 94: Bounded MLIR-to-RVV selected-boundary seed

**Date**: 2026-05-16
**Task**: Bounded MLIR-to-RVV selected-boundary seed
**Branch**: `main`

### Summary

Added an RVV-owned bounded vector i32 add source seed pass that materializes selected RVV i32m1 boundary IR, covered positive/negative lit, generated target artifacts, and passed ssh rvv correctness plus full check-tianchenrv.

### Main Changes

- Created and archived Trellis task
  `05-16-bounded-mlir-to-rvv-selected-boundary-seed` from the supplied
  Direction Brief.
- Added RVV plugin-owned pass
  `--tcrv-rvv-materialize-i32m1-selected-boundary-seed`, registered through
  `tcrv-opt`, for one explicitly marked `func`/`scf`/`vector`/`arith` i32 add
  source shape.
- The pass materializes `tcrv.exec.kernel`, a selected `origin = "rvv-plugin"`
  variant, explicit `lhs`/`rhs`/`out`/`n` runtime ABI bindings,
  `tcrv_rvv.setvl`, selected `tcrv_rvv.with_vl`, and RVV
  `i32_load` / `i32_add` / `i32_store`.
- Added positive and fail-closed lit coverage for selected-boundary output,
  emission-plan/EmitC route consumption, missing ABI operands, unsupported
  dtype/rank/vector shape, malformed source body, and stale
  `tcrv.exec`/`tcrv_rvv` residue.
- Promoted the bounded seed command/input/output/error contract into
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
- Generated seed object/header artifacts under
  `artifacts/tmp/rvv_selected_boundary_seed/20260516T132508Z` and ran a real
  `ssh rvv` link/run harness.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] Focused build for RVV construction protocol, RVV EmitC route provider,
  RVV plugin, transforms, `tcrv-opt`, `tcrv-translate`, and focused C++ tests.
- [OK] C++ tests:
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-construction-protocol-common-test`,
  `tianchenrv-rvv-dialect-test`.
- [OK] Focused lit filter:
  `rvv-i32m1-selected-boundary-seed|rvv-first-slice-materialization|Target/RVV/i32m1|rvv-with-vl-selected-boundary`,
  19/19 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 104/104
  passed.
- [OK] `ssh rvv` seed harness printed
  `tcrv_rvv_i32m1_selected_boundary_seed status=PASS n=4 add=[12,6,16,12]`.
- [OK] `git diff --check`.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 96: Toy source-seed plugin-template proof

**Date**: 2026-05-16
**Task**: Toy source-seed plugin-template proof
**Branch**: `main`

### Summary

Added Toy as the second plugin-owned source-seed registry consumer. The new
bounded Toy seed is registered through `ToyExtensionPlugin::registerSourceSeedPasses`,
materializes the existing Toy selected `tcrv_toy.compute_skeleton` boundary,
and feeds the existing Toy EmitC and target artifact route without direct
`tcrv-opt` wiring or common/core Toy semantic branches.

### Main Changes

- Created Trellis task `05-16-toy-source-seed-plugin-template-proof` from the
  supplied Direction Brief and wrote the PRD around a Toy-only second consumer.
- Added `--tcrv-toy-materialize-template-selected-boundary-seed` under the Toy
  plugin.
- The seed accepts one explicit source-only marker,
  `tcrv_toy.lowering_seed = "template_compute"`, on a zero-argument,
  zero-result `func.func` with one empty return.
- The seed materializes a Toy `tcrv.exec.kernel`, available `toy.template`
  capability, selected `origin = "toy-plugin"` variant with the existing Toy
  construction metadata, selected diagnostic, and `tcrv_toy.compute_skeleton`.
- Added Toy plugin/unit coverage for source-seed registration metadata and
  factory construction.
- Added lit coverage for Toy seed positive boundary, EmitC, target artifact,
  built-in-disabled fail-closed behavior, unsupported marker/shape/body, stale
  `tcrv.exec`, and stale `tcrv_toy` residue.

### Testing

- [OK] Focused build:
  `cmake --build build --target TianChenRVToyPlugin tcrv-opt tcrv-translate
  tianchenrv-toy-extension-plugin-test tianchenrv-plugin-registry-test -j2`.
- [OK] C++ tests:
  `tianchenrv-toy-extension-plugin-test`,
  `tianchenrv-plugin-registry-test`.
- [OK] Focused lit from `build/test`:
  `toy-template-selected-boundary-seed|toy-template-target-artifact|toy-template-materialization|rvv-i32m1-selected-boundary-seed`,
  14/14 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 106/106
  passed.
- [OK] `git diff --check`.
- [OK] Tool direct seed-factory scan returned no matches under
  `tools/tcrv-opt` and `tools/tcrv-translate`.
- [OK] Common/core source-seed semantic branch scan returned no matches under
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/ExtensionPlugin.cpp`, `include/TianChenRV/Transforms`, and
  `lib/Transforms`.

### Self-Repair

- Fixed zero-argument MLIR source syntax in the new Toy lit input.
- Repaired Toy seed variant metadata to use the existing Toy legality
  attributes (`tcrv_toy.*`) rather than construction-protocol helper metadata
  names.
- Replaced an invalid unknown-op Toy residue test with a legal
  `tcrv_toy.compute_skeleton` residue.

### Spec Update Judgment

No spec update was needed. Existing plugin-protocol specs already define the
source-seed registry rule; this round proved a second consumer without adding a
new durable contract.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 98: RVV selected-boundary runtime-VL contract

**Date**: 2026-05-16
**Task**: RVV selected-boundary config and runtime-VL contract
**Branch**: `main`

### Summary

Defined and enforced one explicit RVV i32m1 config/runtime-VL contract across
the existing source-seed add route, selected emission-plan diagnostics, target
artifact candidates, generated headers, and bundle metadata. The RVV-specific
contract stays in RVV-owned code; common emission readiness, coherence, and
target export only carry generic bounded key/value artifact metadata.

### Main Changes

- Created Trellis task
  `05-16-rvv-selected-boundary-runtime-vl-contract` from the supplied
  Direction Brief and wrote the PRD around the existing RVV i32m1 source-seed
  proof path.
- Added a small generic `ArtifactMetadataEntry` support type and threaded
  artifact metadata through `VariantEmissionPlan`, `tcrv.exec.diagnostic`,
  target artifact candidates, and target artifact bundle records.
- Extended `RVVConfigContract` with the exact i32m1 metadata contract:
  SEW32, LMUL m1, tail/mask agnostic policy, runtime AVL from ABI `n`,
  `setvl`/`with_vl` boundary identity, same-VL dataflow uses, and callable ABI
  order `lhs,rhs,out,n`.
- Made RVV emission plans emit the metadata and made RVV target artifact export
  validate it before producing object/header/bundle artifacts.
- Updated RVV source-seed, selected-dispatch, target artifact, bundle, and C++
  target export tests for positive propagation plus fail-closed missing or
  mismatched metadata.

### Testing

- [OK] Focused build:
  `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin
  TianChenRVTarget tcrv-opt tcrv-translate
  tianchenrv-target-artifact-export-test -j2`.
- [OK] Focused RVV/plugin build:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  tianchenrv-construction-protocol-common-test
  tianchenrv-rvv-dialect-test -j2`.
- [OK] C++ tests:
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`, and
  `tianchenrv-rvv-dialect-test`.
- [OK] Focused lit from `build/test`:
  `rvv-i32m1-selected-boundary-seed|i32m1-add-object-artifact|i32m1-selected-dispatch-artifact|i32m1-sub-selected-dispatch-artifact|i32m1-mul-selected-dispatch-artifact|i32m1-object-stale-route-op|i32m1-object-missing-contract-metadata|i32m1-artifact-ambiguous-selected|source-seed-artifact-front-door`,
  12/12 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 110/110
  passed.
- [OK] `git diff --check`.
- [OK] Generated RVV source-seed artifacts under
  `artifacts/tmp/rvv_selected_boundary_runtime_vl_contract/20260516T150644Z`.
- [OK] `ssh rvv` link/run passed:
  `tcrv_rvv_i32m1_selected_boundary_seed status=PASS n=4 add=[12,6,16,12]`.
- [OK] Changed-surface scans showed no new descriptor/direct-C/Python
  compiler-core route and no RVV semantic branch in common/core orchestration.

### Self-Repair

- Kept common metadata parsing and printing extension-neutral after rejecting
  RVV-specific handling in common transforms/target export as the wrong
  ownership boundary.
- Updated existing hand-authored RVV emission-plan fixtures so they model the
  same explicit contract as plugin-generated emission plans.
- Added a missing-contract RVV target export negative test to prove fail-closed
  behavior for stale or incomplete artifact candidates.

### Spec Update Judgment

No spec update was needed. Existing RVV plugin, EmitC route, variant pipeline,
and MLIR testing specs already cover this boundary and evidence policy.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 95: Bounded source-vector-to-RVV selected-boundary lowering

**Date**: 2026-05-16
**Task**: Bounded source-vector-to-RVV selected-boundary lowering
**Branch**: `main`

### Summary

Made the RVV i32m1 add source seed consume validated source shape and source-derived ABI provenance before materializing the selected boundary.

### Main Changes

### Main Changes

- Created and archived Trellis task `05-16-05-16-bounded-source-vector-to-rvv-selected-boundary-lowering` from the supplied Direction Brief.
- Reworked the RVV selected-boundary source seed matcher to return a recognized `BoundedI32AddSourceSeed` before materialization.
- Materialization now consumes source-derived ABI argument mapping and emits `tcrv_rvv.runtime_abi_value` purpose provenance: `source-arg-0:lhs`, `source-arg-1:rhs`, `source-arg-2:out`, and `source-arg-3:n`.
- Tightened source-shape validation to reject loop-carried `scf.for` values before RVV boundary creation.
- Expanded negative lit coverage for unsupported marker, missing/extra `n`, wrong arithmetic op, wrong output buffer use, unsupported loop bounds/step, loop-carried values, unrelated/empty body, and stale `tcrv.exec`/`tcrv_rvv` residue.
- Updated the variant-pipeline spec with the source-argument provenance and loop-carried rejection contract.

### Testing

- [OK] Focused build for RVV dialect/plugin/target, `tcrv-opt`, `tcrv-translate`, and RVV/target C++ tests.
- [OK] C++ tests: `tianchenrv-rvv-extension-plugin-test`, `tianchenrv-construction-protocol-common-test`, `tianchenrv-rvv-dialect-test`, `tianchenrv-target-artifact-export-test`.
- [OK] Focused lit filter: `rvv-i32m1-selected-boundary-seed|source-seed-artifact-front-door|i32m1-add-object-artifact`, 6/6 passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 110/110 passed.
- [OK] `git diff --check`.
- [OK] Changed-surface scans: no common/core/tooling files changed; no descriptor/direct-C/source-export/Python compiler-core route terms in changed RVV seed/test diff.
- [OK] New artifacts under `artifacts/tmp/bounded_source_vector_to_rvv_selected_boundary/20260516T153540Z`.
- [OK] Real `ssh rvv` link/run: `tcrv_rvv_i32m1_selected_boundary_seed status=PASS n=4 add=[12,6,16,12]`.

### Self-Repair

- Fixed a const erase compile error in the matched seed cleanup loop.
- Fixed the loop-carried negative test to reach the intended `scf.for iter_args` source-shape gate.

### Status

[OK] Completed and archived.


## Session 97: Common materialized EmitC handoff for RVV source dispatch

**Date**: 2026-05-17
**Task**: `rvv-source-dispatch-emitc-handoff`
**Branch**: `main`

### Summary

Made the existing bounded RVV i32m1 add source-seed selected-dispatch path
prove a common materialized MLIR EmitC handoff before MLIR EmitC C/C++ emission
and RVV target artifact packaging.

### Main Changes

- Created and archived Trellis task
  `05-17-rvv-source-dispatch-emitc-handoff` from the supplied Direction Brief.
- Added target-side validation in `lib/Target/TargetArtifactExport.cpp` so
  selected EmitC artifact routes must materialize an EmitC function boundary
  with route source-op and call source-op provenance before C/C++ emission or
  object packaging.
- Kept the common helper extension-neutral; RVV intrinsic/header/runtime details
  remain in RVV-owned route and target-support code.
- Extended `test/Target/TargetArtifactExportTest.cpp` to prove common selected
  EmitC source keeps route provenance and rejects route builders that omit route
  source-op provenance.
- Extended the RVV source-seed selected-dispatch fixture to run through
  materialized EmitC into the MLIR EmitC C/C++ emitter before object/header/
  bundle export checks.
- Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the executable
  selected artifact handoff contract.

### Testing

- [OK] Focused build:
  `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`.
- [OK] C++ tests:
  `tianchenrv-emission-readiness-test`,
  `tianchenrv-emitc-lowerable-interface-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-rvv-dialect-test`, and
  `tianchenrv-target-artifact-export-test`.
- [OK] Focused lit filter from `build/test`:
  `rvv-i32m1-selected-boundary-seed|emitc-to-cpp-handoff|i32m1-add-object-artifact|i32m1-selected-dispatch-artifact|i32m1-object-stale-route-op|i32m1-object-missing-contract-metadata|i32m1-artifact-ambiguous-selected|source-seed-artifact-front-door`,
  11/11 selected tests passed.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 110/110 lit
  tests passed.
- [OK] `git diff --check`.
- [OK] Changed-surface scan: no Python files changed; no descriptor/direct-C/
  source-export route was added; common target code only adds generic EmitC
  handoff validation.
- [OK] Refreshed artifacts under
  `artifacts/tmp/source_seed_emitc_handoff/20260516T163726Z`.
- [OK] Real `ssh rvv` link/run:
  `tcrv_rvv_i32m1_source_emitc_handoff status=PASS n=4 add=[12,6,16,12]`.

### Self-Repair

- `clang-format` was unavailable, so the touched C++ line wrap was manually
  normalized.
- Re-ran the final full check with the clean target command after an earlier
  redundant build-tool suffix.

### Spec Update Judgment

Spec updated because this round changed the executable target/export handoff
contract: selected artifact routes that emit/package from EmitC must validate
materialized route provenance before emitter/toolchain steps.

### Status

[OK] Completed and archived.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 96: RVV source-seed selected-dispatch artifact path

**Date**: 2026-05-16
**Task**: `rvv-source-seed-selected-dispatch-artifact`
**Branch**: `main`

### Summary

Carried the bounded RVV i32m1 add source seed through a selected
`tcrv.exec.dispatch` envelope and into the existing RVV EmitC/object/header/
bundle artifact path.

### Main Changes

- Created Trellis task `05-16-rvv-source-seed-selected-dispatch-artifact` from
  the supplied Direction Brief and repaired its PRD before implementation.
- Updated the RVV source-seed materializer to emit `@rvv`,
  `@scalar_fallback`, source-derived `@seed_rvv_i32_add`, conservative
  `@seed_scalar_fallback`, `tcrv.exec.case`, and `tcrv.exec.fallback`.
- Kept fallback as selection/envelope metadata only; no scalar fallback compute
  semantics were added.
- Generalized emission readiness to allow optional lowering-boundary candidates
  for dispatch-case references while keeping direct static selected markers
  boundary-required.
- Updated EmitC lowerable materialization to select the supported
  selected-emission diagnostic when the module also contains an unsupported
  fallback variant.
- Extended the RVV source-seed fixture to prove dispatch/case/fallback
  organization, emission-plan metadata, EmitC route materialization,
  object/header export, and bundle index selection.

### Testing

- [OK] Focused build for `tcrv-opt`, `tcrv-translate`, and touched
  RVV/plugin/target C++ test executables.
- [OK] Focused lit filter:
  `rvv-i32m1-selected-boundary-seed|source-seed-artifact-front-door|i32m1-(add-object-artifact|selected-dispatch-artifact|sub-selected-dispatch-artifact|mul-selected-dispatch-artifact)|rvv-first-slice-materialization|emitc-to-cpp-handoff|toy-template-selected-boundary-seed`,
  18/18 passed.
- [OK] C++ tests:
  `tianchenrv-emission-readiness-test`,
  `tianchenrv-emitc-lowerable-interface-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-rvv-dialect-test`,
  `tianchenrv-target-artifact-export-test`.
- [OK] Full `cmake --build build --target check-tianchenrv -j2`, 110/110 lit
  tests passed.
- [OK] `git diff --check`.
- [OK] Changed-surface scans: no Python files changed; common/core transform
  diff has no RVV/scalar/descriptor/direct-C/source-export/Python compiler-core
  terms; RVV fixture intrinsic header assertion is the expected EmitC route
  evidence.
- [OK] New artifacts under
  `artifacts/tmp/source_seed_selected_dispatch_artifact/20260516T161432Z`.
- [OK] Real `ssh rvv` link/run:
  `tcrv_rvv_i32m1_source_selected_dispatch status=PASS n=4 add=[12,6,16,12]`.

### Self-Repair

- Removed an accidental scalar-plugin static link dependency from the RVV seed
  implementation and replaced it with local fallback identity constants.
- Fixed FileCheck ordering in the RVV source-seed fixture.
- Relabeled the temporary RVV harness output to name this round's
  source-selected-dispatch evidence path.

### Spec Update Judgment

No spec update was needed. Existing variant-pipeline, EmitC route,
plugin-protocol, and MLIR testing specs already cover this bounded path.

### Status

[OK] Completed and archived.
