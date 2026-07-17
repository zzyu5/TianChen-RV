# Delete RVV body-verifier descriptor/config surface

## Goal

Delete the remaining RVV descriptor/config body-verifier production surface:
`RVVBinaryMicrokernelBodyVerifier` header/source/build entry, its dataflow and
intrinsic-config request/result structs, and adjacent wording that still makes
descriptor/config validation look like RVV compute authority.

This is a Wrong Logic Deletion Campaign round. Future RVV executable output
must be rebuilt from typed RVV extension-family ops through a materialized
MLIR EmitC route. This task does not rebuild that route.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial repo state for this round was clean at HEAD `043c7c6`.
- `.trellis/.current-task` did not exist at session start; this task was
  created from the Hermes Direction Brief.
- The previous archived RVV direct-route deletion task removed
  `RVVMicrokernel.h` / `RVVMicrokernel.cpp`, stale direct callable route
  artifact fixtures, and passed `check-tianchenrv`.
- `include/TianChenRV/Target/RVV/RVVBinaryMicrokernelBodyVerifier.h` still
  exposes public RVV dataflow step/plan structs, `RVVIntrinsicConfig`,
  validation request/result structs, and
  `validateRVVBinaryMicrokernelBody`.
- `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp` is still compiled by
  `TianChenRVRVVTarget` and returns a fail-closed diagnostic for
  descriptor/config-driven body validation.
- Bounded ref-scan found no external production or test call site for
  `validateRVVBinaryMicrokernelBody`; the verifier surface is self-contained
  plus its CMake entry.
- `RVVSelectedConfigContract.h` still has at least one diagnostic phrase,
  "finite binary family descriptor", that should be narrowed to family record
  / selected config language.
- `RVVBinaryFamily.h` uses local variable names such as `descriptor` for
  family records; these should be renamed where they imply descriptor-era
  authority rather than harmless data records.
- Relevant specs define descriptor-driven microkernel/exporter frameworks as
  invalid and define the accepted future route as extension-family ops ->
  EmitC -> C/C++ emitter.

## Requirements

- Delete `include/TianChenRV/Target/RVV/RVVBinaryMicrokernelBodyVerifier.h`.
- Delete `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp`.
- Remove `RVVBinaryMicrokernelBodyVerifier.cpp` from the
  `TianChenRVRVVTarget` build.
- Ensure no production target library exposes or builds
  `RVVBinaryMicrokernelBodyVerifier`.
- Ensure no production call site or test depends on
  `validateRVVBinaryMicrokernelBody`,
  `RVVBinaryDataflowEmissionPlan`, `RVVBinaryDataflowStep`,
  `RVVIntrinsicConfig`, or related request/result structs.
- Narrow adjacent selected-config/family wording that still calls selected
  RVV family records "descriptors" where that could imply compute authority.
- Preserve selected-boundary/plugin configuration facts that are already
  typed, bounded, metadata-only, and non-authoritative for generated compute.
- Do not add a replacement verifier, compatibility shim, descriptor wrapper,
  new direct C route, new RVV lowering, Common EmitC rebuild, ssh RVV evidence
  campaign, or Python compiler semantics.

## Acceptance Criteria

- [x] No production header exposes `RVVBinaryMicrokernelBodyVerifier`,
      `validateRVVBinaryMicrokernelBody`,
      `RVVBinaryDataflowEmissionPlan`, `RVVBinaryDataflowStep`,
      `RVVIntrinsicConfig`, or the old body validation request/result structs.
- [x] `TianChenRVRVVTarget` no longer builds
      `RVVBinaryMicrokernelBodyVerifier.cpp`.
- [x] Ref-scans over touched RVV target/plugin/test files show verifier and
      dataflow/config body-validation APIs are gone.
- [x] Remaining RVV selected config / family records are described as
      selected-boundary or plugin configuration facts, not descriptor-to-C or
      body-validation authority.
- [x] No replacement descriptor/direct-C/wrapper/compatibility path is added.
- [x] Focused RVV target build and affected RVV plugin/target tests pass, or
      failures are recorded as truthful deletion gaps without restoring old
      logic.
- [x] Full `check-tianchenrv` is run when coherent.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation
      before finish and after archive, final clean `git status --short`, and
      one coherent commit are produced if complete.

## Non-goals

- No new RVV lowering.
- No Common Lower-To-EmitC pass.
- No replacement body verifier.
- No compatibility shim, legacy mode, or descriptor wrapper.
- No descriptor-to-C exporter restoration.
- No extension-family rebuild.
- No finite-family expansion.
- No ssh RVV runtime/correctness/performance evidence campaign.
- No Python compiler-core semantics.
- No broad repo audit beyond bounded RVV verifier/descriptor scans.

## Minimal Evidence

- Before/after scans over touched RVV target/plugin/test files for:
  `RVVBinaryMicrokernelBodyVerifier`,
  `validateRVVBinaryMicrokernelBody`,
  `RVVBinaryDataflowEmissionPlan`,
  `RVVBinaryDataflowStep`,
  `RVVIntrinsicConfig`,
  `descriptor/config-driven`,
  `finite binary family descriptor`,
  `lowering_descriptor`, and `microkernel.v1`.
- Focused build target:
  `cmake --build build --target TianChenRVRVVTarget -j2`.
- Focused RVV plugin/target lit or C++ tests when affected by deletion.
- Full `cmake --build build --target check-tianchenrv -j2` when coherent.
- `git diff --check`.
- `git diff --cached --check`.
- `python3 ./.trellis/scripts/task.py validate`.
- Trellis validation after archive.

## Technical Notes

- `.trellis/spec/architecture/design-boundaries.md` rejects
  descriptor-driven microkernel/exporter frameworks.
- `.trellis/spec/lowering-runtime/emitc-route.md` defines the current/future
  accepted compiler route as extension family ops -> EmitC -> C/C++ emitter.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` requires
  deleted direct descriptor/source exporter paths to be absent or fail closed
  until a materialized MLIR EmitC module route exists.
- `.trellis/spec/testing/mlir-testing-contract.md` requires deleted
  descriptor/direct-source behavior to remain fail-closed and prohibits tests
  that preserve old generated source/body authority.

## Completion Evidence

- Deleted public production header
  `include/TianChenRV/Target/RVV/RVVBinaryMicrokernelBodyVerifier.h`.
- Deleted production implementation
  `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp`.
- Removed `RVVBinaryMicrokernelBodyVerifier.cpp` from
  `lib/Target/RVV/CMakeLists.txt`, so `TianChenRVRVVTarget` no longer builds
  the body verifier.
- Narrowed `RVVSelectedConfigContract.h` invalid-contract diagnostic from
  "finite binary family descriptor" to "RVV binary family record".
- Renamed local `RVVBinaryFamily.h` family-record variables from
  `descriptor` to `record`, leaving only selected-plan metadata descriptor
  type names as harmless structured metadata terminology.
- Bounded post-change ref-scan over `include/TianChenRV/Target/RVV`,
  `lib/Target/RVV`, `test/Plugin`, `test/Target/RVVMicrokernel`, and
  `test/Target/ArtifactExport` found no hits for:
  `RVVBinaryMicrokernelBodyVerifier`,
  `validateRVVBinaryMicrokernelBody`,
  `RVVBinaryDataflowEmissionPlan`,
  `RVVBinaryDataflowStep`,
  `RVVIntrinsicConfig`,
  `descriptor/config-driven`,
  `finite binary family descriptor`,
  `lowering_descriptor`, or `microkernel.v1`.
- Remaining `descriptor` hits under RVV target headers are
  `RVVVectorShapeSelectedPlanMetadataDescriptor` type names and their users;
  they are selected-plan metadata container names, not body validation,
  descriptor-to-C, or compute authority.
- Focused build passed:
  `cmake --build build --target TianChenRVRVVTarget -j2`.
- Focused C++ tests passed:
  `tianchenrv-rvv-binary-planning-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-rvv-selected-lowering-boundary-test`,
  `tianchenrv-rvv-binary-variant-legality-test`, and
  `tianchenrv-target-artifact-export-test`.
- Focused lit passed:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv .
  --filter='(ArtifactExport|RVVMicrokernel)'` from `build/test`, 5/5 selected
  tests passed.
- Full evidence passed:
  `cmake --build build --target check-tianchenrv -j2`, 114/114 tests passed.
- `git diff --check` passed before finish.
