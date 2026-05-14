# Descriptor Erasure: RVV Selected Microkernel Descriptor Materialization Deletion

## Goal

Delete one coherent Wrong Logic Deletion Campaign slice: the RVV selected
microkernel descriptor/materialization authority that still derives RVV
microkernel/dataflow/runtime ABI behavior from finite binary family records,
`RVVBinaryIntrinsicRoute`, and `selectedPlan.descriptor`.

This round must remove or fail-close the active path where plugin/target code
materializes `tcrv_rvv.*_microkernel` bodies, builds selected emission runtime
ABI parameters, or validates RVV body/intrinsic truth from descriptor/config
mirrors. Future rebuild work must start from plugin-owned extension-family IR
and explicit EmitC lowering, not descriptor tables.

## Why Now

Commit `b2ea361` deleted and archived the core RVV source-to-exec pass family.
Core transforms/tools no longer own RVV finite-family source semantics, but
bounded inspection still finds descriptor-driven compute authority in RVV
plugin/target code:

- `include/TianChenRV/Target/RVV/RVVBinaryFamily.h` defines finite family route
  records with microkernel op names, arithmetic op names, intrinsic prefixes,
  legacy lowering tokens, route IDs, and ABI strings.
- `include/TianChenRV/Target/RVV/RVVBinaryRoute.h` wraps those records as
  `RVVBinaryIntrinsicRoute` and derives intrinsic names and runtime ABI helper
  data.
- `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp` validates
  `selectedPlan.descriptor` against selected config and materializes
  `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, load/arithmetic/store bodies.
- `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp` creates default typed
  selected plans, ensures runtime ABI mem windows/params from descriptor route
  data, and invokes `materializeRVVBinaryMicrokernelOp`.
- `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp` discovers explicit
  microkernel bodies and builds emission-plan runtime ABI parameters from
  `selectedPlan.descriptor`.
- `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp` validates structured
  bodies against descriptor/config fields and intrinsic names.

These are descriptor-driven compute semantics, not the accepted long-term
architecture.

## Scope

- Delete or fail-close plugin auto-materialization of RVV binary microkernel
  dataflow ops from selected descriptor/family records.
- Delete or fail-close selected emission runtime ABI parameter construction from
  `selectedPlan.descriptor`.
- Delete or fail-close body verifier logic that treats descriptor/config fields
  as compute truth for RVV op sequences, vector types, intrinsic names, or ABI
  roles.
- Remove declarations, definitions, tests, and active call sites for:
  `buildRVVBinaryVLDataflowMaterialization`,
  `materializeRVVBinaryMicrokernelOp`,
  `buildRVVBinaryCallableRuntimeABIParameters`, and active
  `selectedPlan.descriptor` materialization/planning use.
- Rewrite tests that protect descriptor-driven microkernel materialization or
  descriptor/config body validation into deleted-route/fail-closed coverage, or
  delete them when they only protect old positive behavior.
- Keep this round deletion-only. If deleting the descriptor chain exposes a
  missing extension-family EmitC architecture gap, report that gap instead of
  rebuilding it here.

## Acceptance Criteria

- [x] RVV plugin code no longer materializes RVV microkernel/load/arithmetic/
      store ops from selected descriptor/family records.
- [x] RVV selected lowering-boundary code no longer uses
      `selectedPlan.descriptor` to create callable microkernel IR, runtime ABI
      mem windows, or runtime ABI params.
- [x] RVV selected emission planning no longer builds runtime ABI parameter
      metadata from `selectedPlan.descriptor`.
- [x] RVV body verification no longer treats descriptor/config fields or
      intrinsic names as the active compute truth for selected RVV op sequences.
- [x] Tests that protected descriptor-driven microkernel materialization,
      descriptor/config body verification, or descriptor-derived callable ABI
      emission planning are deleted or rewritten to assert unsupported/deleted
      route behavior.
- [x] If direct RVV microkernel source/object/header/bundle paths remain
      fail-closed, tests assert unsupported behavior and no generated compute
      source bytes.
- [x] No Common Lower-To-EmitC rebuild, new RVV extension-family lowering,
      replacement microkernel materializer, descriptor wrapper, compatibility
      alias, direct C exporter, or Python compiler semantics is added.
- [x] Bounded ref-scan reports residuals for target terms and classifies each
      as active authority, fail-closed metadata, historical wording, or next
      deletion owner.
- [x] Focused affected build/test targets run first; full
      `cmake --build build --target check-tianchenrv -j2` runs if the deletion
      is coherent.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation before
      finish and after archive, final clean `git status --short`, and one
      coherent commit are produced.

## Non-Goals

- No Common Lower-To-EmitC implementation.
- No new RVV extension-family lowering.
- No replacement microkernel materializer.
- No new selected-boundary implementation beyond fail-closed demotion required
  by this deletion.
- No descriptor wrapper, compatibility mode, legacy alias, direct C exporter,
  finite RVV family expansion, or plugin construction template.
- No `ssh rvv` evidence campaign.
- No broad repo audit beyond bounded owner/ref-scan evidence.
- No preserving descriptor-positive tests merely to keep stale RVV microkernel
  behavior passing.

## Minimal Evidence

- Focused inventory of RVV descriptor/family-record materialization owners and
  active call sites.
- Bounded scans over touched include/lib/test/spec areas for:
  `RVVBinaryFamilyRecord`, `RVVBinaryIntrinsicRoute`,
  `selectedPlan.descriptor`, `buildRVVBinaryVLDataflowMaterialization`,
  `materializeRVVBinaryMicrokernelOp`,
  `buildRVVBinaryCallableRuntimeABIParameters`,
  `getRVVBinaryFamilyRegistrationRecords`,
  `getRVVBinaryMicrokernelFamilyForOp`, `legacyLoweringToken`,
  `i32-vadd-microkernel.v1`, `arithmeticIntrinsicPrefix`,
  `descriptor-family`, `descriptor-element-count`, and descriptor/config body
  verifier wording.
- Focused C++ build targets for affected RVV plugin/target libraries and tests.
- Focused lit/C++ tests that cover deleted-route or fail-closed behavior for
  affected RVV plugin/target/export paths.
- Full `check-tianchenrv` when coherent.

## Completion Evidence

Completed one coherent deletion-only slice.

- Deleted `include/TianChenRV/Plugin/RVV/RVVBinaryMicrokernelMaterialization.h`
  and `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`; removed the CMake
  source entry and selected-boundary include.
- Rewired `RVVBinarySelectedLoweringBoundary.cpp` so selected RVV planning emits
  only metadata-only `tcrv_rvv.lowering_boundary` with unsupported status; it no
  longer materializes `tcrv_rvv.*_microkernel`, `setvl`, `with_vl`, load,
  arithmetic, or store ops.
- Rewrote `RVVBinarySelectedEmissionPlanning.cpp` as a fail-closed public surface
  that returns no selected emission plan/readiness/variant emission plan instead
  of building runtime ABI parameters from `selectedPlan.descriptor`.
- Removed the selected-plan `RVVBinaryIntrinsicRoute descriptor` mirror from
  `RVVBinarySelectedPlan`; selected-plan route, artifact, runtime ABI, and
  intrinsic accessors now fail closed.
- Replaced `RVVBinaryMicrokernelBodyVerifier.cpp` with a fail-closed verifier
  error; descriptor/config-driven RVV body validation is no longer an active
  compute authority.
- Deleted stale verifier-negative lit tests and rewrote plugin/variant/probe/
  manifest/export tests to assert metadata-only unsupported RVV emission and no
  generated RVV compute source bytes.
- Updated RVV plugin, emission-runtime, and MLIR testing specs to record that
  finite descriptor selected microkernel materialization is deleted and that
  future rebuild must use explicit extension-family IR plus EmitC lowering.

Bounded ref-scan classification:

- No code/test hits remain for `selectedPlan.descriptor`, `plan.descriptor`,
  `buildRVVBinaryVLDataflowMaterialization`,
  `materializeRVVBinaryMicrokernelOp`, or
  `buildRVVBinaryCallableRuntimeABIParameters`. The only
  `selectedPlan.descriptor` hits are deletion wording in the RVV plugin and
  emission-runtime specs.
- `RVVBinaryFamilyRecord`, `RVVBinaryIntrinsicRoute`, finite route ids, legacy
  lowering tokens, and intrinsic prefixes remain active in the target registry,
  RVV selected-config contract, plugin planning/variant legality, and target
  `RVVMicrokernel.cpp` direct artifact/export code. Those are not consumed by
  the deleted selected-boundary materializer or selected-emission planning path
  in this round; they are the next deletion owners.
- `RVVBinaryMicrokernelBodyVerifier.h` still exposes the old request shape, but
  the implementation is fail-closed and no longer validates descriptor/config
  truth.
- `descriptor-element-count` and explicit microkernel fixtures remain in target
  artifact export tests as next-owner direct-artifact/export residue, not as a
  restored selected materialization path.

Checks completed:

- `cmake --build build --target TianChenRVRVVPlugin TianChenRVRVVTarget -j2`
- `cmake --build build --target tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-selected-lowering-boundary-test
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-rvv-binary-planning-test`
- `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build build --target check-tianchenrv -j2`: 125/125 passed.
- `git diff --check`
- `git diff --cached --check`

## Technical Notes

- `.trellis/spec/index.md` rejects descriptor-driven computation and requires
  C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck for compiler behavior.
- `.trellis/spec/architecture/design-boundaries.md` lists descriptor-driven
  microkernel/exporter frameworks as a non-goal.
- `.trellis/spec/lowering-runtime/emitc-route.md` defines the future accepted
  path as extension family ops -> EmitC -> intrinsic/vendor/runtime C/C++.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` says direct
  descriptor-to-C source/exporter routes must be deleted or fail-closed until a
  materialized MLIR EmitC module route exists.
- `.trellis/spec/testing/mlir-testing-contract.md` requires deleting or
  rewriting tests that only protect old direct-C or descriptor-driven positive
  behavior.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-15-core-rvv-source-to-exec-erasure/prd.md`
  confirms the core source frontend path was already deleted and this round is
  the next plugin/target descriptor authority owner.
