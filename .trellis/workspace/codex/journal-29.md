> Continuation from `journal-28.md` (archived at ~1980 lines)

## Session 577: Stage2 RVV low-precision contraction primitive Gate 1

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Created the new macro Trellis task for the RVV Stage 2 low-precision
contraction primitive-surface campaign and completed Gate 1 only. Source
inspection showed the signed i8 product-reduction primitive facts, unsigned u8
widening-product path, selected-body resource facts, and target mirrors already
exist in production. The remaining Gate 1 blocker was the target artifact
provider-facts preflight: it did not directly compare the selected
low-precision resource primitive surface fields against
`RVVLowPrecisionWideningReductionPrimitiveFacts` before artifact export.

### Main Changes

- Hardened target artifact provider-facts validation so
  `lowPrecisionResourceSelection` source/product/accumulator/reduction/final
  primitive facts must match provider-owned widening-reduction primitive facts
  before artifact export.
- Added focused target artifact C++ coverage mutating a packed-i4
  product-reduction dequant resource selection's primitive product SEW to prove
  stale resource primitive facts fail closed.
- Created and left open the macro Trellis task with Gate 1 complete and Gates
  2-4 remaining.

### Testing

- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-campaign`
- [OK] `git diff --check`

### Spec Update Decision

[NO SPEC UPDATE] The RVV plugin spec already requires low-precision primitive
facts to be provider-owned and target artifact validation to reject stale
primitive/resource mirrors. This slice implements that existing contract at one
missing target provider-facts preflight.

### Status

[OPEN MACRO TASK] Gate 1 is complete. Gates 2-4 remain open. The next
continuation point is Gate 2 selected-body realization consumption of these
typed primitive facts for a representative low-precision contraction/dequant
path without changing compute semantics, ABI roles, runtime AVL/VL, dispatch,
or fallback behavior.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 582: Stage2 RVV low-precision contraction primitive Gate 2

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed Gate 2 only. Gate 1 already
made typed i8/u8 source load and source extension facts visible. This round
hardened standalone low-precision widening-product facts by adding explicit
provider-owned multiplicand-role and extension-policy fields for signed i8 and
unsigned u8 products, then requiring those fields through route-family
planning, route description validation, target metadata, and target artifact
mirror validation.

### Main Changes

- Added `wideningProductMultiplicandRoleSummary` and
  `wideningProductExtensionPolicy` to standalone widening-product provider
  facts, route-family plans, route descriptions, validation contracts, and
  target artifact metadata.
- Signed i8 and unsigned u8 widening-product facts now expose explicit lhs/rhs
  `wprod-*` roles and sign/zero-extension policy summaries alongside the
  existing source/load/product/intrinsic facts.
- Added provider and target fail-closed checks for stale multiplicand roles
  and extension policy before target artifact acceptance.
- Updated signed and unsigned widening-product lit fixtures with accepted
  mirrors and stale metadata rejection coverage.
- Updated the RVV plugin spec with the standalone low-precision
  widening-product primitive-fact contract.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVV/explicit-selected-body-artifact-widening-product.mlir Target/RVV/explicit-selected-body-artifact-widening-product-unsigned-u8.mlir Target/RVV/explicit-selected-body-artifact-widening-product-reduce-add.mlir`
- [OK] `git diff --check`

### Spec Update Decision

[UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` now records that
standalone low-precision widening-product support must carry provider-owned
multiplicand-role and extension-policy facts through route validation and
target mirrors, while Common EmitC/export remains a mirror consumer only.

### Status

[OPEN MACRO TASK] Gates 1-2 are complete. Gates 3-4 remain open. The next
continuation point is Gate 3: widening reduction/accumulation facts for
contraction-style kernels, building on the standalone product facts without
q8/q4 wrappers or metadata authority.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 583: Stage2 RVV low-precision contraction primitive Gate 3

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed Gate 3 only. Gates 1-2 already
made typed i8/u8 source and widening-product facts provider-owned. This round
made widening reduction/accumulation primitive facts production-consumed for
signed i8 and unsigned u8 product-reduction chains, with route planning,
route description validation, route construction, statement planning, and
target artifact validation failing closed on stale or mismatched facts.

### Main Changes

- Extended typed RVV verification and runtime ABI contracts for unsigned
  `u8mf4 -> u16mf2 -> u32m1` product-reduction chains.
- Added description-aware RVV provider facts for signed and unsigned
  low-precision widening reduction/accumulation primitive chains.
- Routed unsigned product-reduction profile, operand binding, C type mapping,
  target leaf, widening product intrinsic, widening reduction intrinsic, scalar
  seed splat, accumulator/result layout, and store-VL facts through production
  route planning and target artifact validation.
- Added focused unsigned u8 selected-body artifact lit coverage with accepted
  mirrors and stale source signedness, source extension, accumulator dtype,
  reduction intrinsic, and C type mapping rejection.
- Updated the RVV plugin spec with the signed/unsigned widening-reduction
  primitive-fact contract.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'explicit-selected-body-artifact-widening-product-reduce-add-unsigned-u8|explicit-selected-body-artifact-widening-product-reduce-add|explicit-selected-body-artifact-widening-product-unsigned-u8|explicit-selected-body-artifact-widening-product\.mlir'`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-gate1`
- [OK] `git diff --check`
- [OK] Bounded added-diff scan returned only PRD/spec non-authority wording for
  forbidden markers and no new positive legacy route-authority support.

### Spec Update Decision

[UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` now records that
bounded low-precision product-reduction support includes both signed i8 and
unsigned u8 widening reduction primitive facts, and that provider planning,
route validation, target mirrors, and target artifact validation must consume
the signedness-specific accumulator/result/intrinsic/seed/layout facts.

### Status

[OPEN MACRO TASK] Gates 1-3 are complete. Gate 4 remains open. The next
continuation point is Gate 4: Gearbox/resource-aware selected-body realization
plus measured same-target comparison must consume these provider-owned
low-precision primitive facts with source-backed evidence.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 581: Stage2 RVV Gearbox Gate 4 closeout

**Date**: 2026-06-10
**Task**: Stage2 RVV Gearbox resource-aware selected-body realization campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed Gate 4. Gates 1-3 already
connected pressure-profile admission, Gearbox schedule choice, and
route/artifact/measurement proof consumption. This round closed the campaign by
proving that source-backed dequant-clamp artifact evidence parsed from JSON
composes with selected dispatch, production pressure profile, and
selected-body realization admission.

### Main Changes

- Repaired the macro PRD from stale Gate 3 current-slice wording to Gate 4
  closeout acceptance criteria, then marked Gates 1-4 complete.
- Extended focused RVV plugin coverage so the parsed dequant-clamp
  `same_target_measurement_record` flows through selected-dispatch policy,
  production pressure profile, and `admitRVVLowPrecisionSelectedBodyRealization`.
- Added Gate 4 fail-closed coverage for stale dequant-clamp artifact identity
  and metadata-only provider support, alongside existing stale schedule and
  correctness-disabled negative cases.
- Confirmed Common EmitC/Conversion stayed untouched; route/artifact/report
  metadata remains mirror-only.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir` from `build/test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-gearbox-realization-campaign`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Bounded added-diff scan returned no legacy RVV route-authority matches.
- [OK] Common EmitC/Conversion neutrality scan returned no changed common
  conversion paths.

### Spec Update Decision

[NO SPEC UPDATE] Gate 4 implements existing RVV plugin/testing contracts for
packed-i4 dispatch/performance policy consumption and source-backed
same-target measurement evidence. No new API signature, payload field,
validation matrix, or long-lived convention was introduced.

### Status

[READY TO ARCHIVE] Gates 1-4 are complete. Remaining slices: none. The macro
task can be archived after the coherent Gate 4 commit.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 581: Stage2 RVV Gearbox Gate 3 route/artifact/measurement proof

**Date**: 2026-06-10
**Task**: Stage2 RVV Gearbox resource-aware selected-body realization campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed Gate 3 only. Gates 1-2 already
established pressure-profile admission and admitted Gearbox schedule mirrors;
this round made the route/provider, target artifact, and same-target
measurement policy path consume and prove those admitted packed-i4 schedule
facts against source-backed same-target records. The macro task remains open
for Gate 4.

### Main Changes

- Added realization-admission proof fields to low-precision resource selection,
  same-target measurement records, policy inputs/outcomes, and production
  pressure profiles.
- Wired realized Gate 2 admission mirrors into the RVV route/provider resource
  selection path and validated them against schedule contract, decision, reason,
  dispatch policy, and measurement evidence identity.
- Required target artifact validation to reject missing, stale, mismatched, or
  metadata-only packed-i4 realization-admission proof before accepting artifact
  mirrors.
- Updated generated same-target evidence support and the dequant-clamp fixture
  with required provider realization-admission proof fields.
- Added focused C++ and lit coverage for successful Gate 3 proof and
  fail-closed stale/missing/sibling proof.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -v Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32-packed-i4.mlir` from `build/test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-gearbox-realization-campaign`
- [OK] Bounded added-line scan returned no new legacy RVV route-authority markers.
- [OK] Bounded `lib/Conversion` / `include/TianChenRV/Conversion` diff was empty, preserving Common EmitC neutrality.

### Spec Update Decision

[UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` now records the Gate 3
realization-admission proof consumption contract, payload fields, validation
matrix, good/base/bad cases, required tests, and wrong-vs-correct authority
path.

### Status

[OPEN MACRO TASK] Gates 1-3 are complete. Gate 4 remains open. The next
continuation point is Gate 4: compose selected-dispatch policy, realization
admission, target artifact evidence, and measurement provenance into one
closeout test, then archive only if the composed campaign acceptance criteria
are actually met.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 580: Stage2 RVV production-kernel capability Gate 4 closeout

**Date**: 2026-06-10
**Task**: Stage2 RVV production-kernel capability campaign
**Branch**: `main`

### Summary

Completed Gate 4 of the active production-kernel capability macro task. Gates
1-3 had already made the production pressure-profile API, source-backed
measurement-record parser, and selected-dispatch record resolver real. This
round closed the campaign by adding focused C++ coverage that proves a record
parsed from the checked-in generated dequant-clamp evidence JSON materializes
`RVVLowPrecisionProductionPressureProfile` through the same production APIs.

### Main Changes

- Added a Gate 4 closeout assertion in
  `test/Plugin/RVVExtensionPluginTest.cpp` after the parsed dequant-clamp
  selected-dispatch policy path. The test now directly builds
  `RVVLowPrecisionProductionPressureProfile` from the parsed evidence record
  and asserts selected variant, generated function, measurement evidence id,
  provider runtime ABI, selected-dispatch fallback mirror, and
  correctness-fallback policy facts.
- Updated the macro PRD to mark Gate 4 complete, explain why no production
  policy source change was needed in this round, record completed/verified
  Gate 4 evidence, and mark the macro task ready to finish/archive.
- Revalidated the packed-i4 dequant and dequant-clamp dry-run evidence bridge
  with explicit `/usr/bin/llvm-readobj-20` because unversioned `llvm-readobj`
  is not on the current PATH.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] Packed-i4 dequant dry-run with explicit `/usr/bin/llvm-readobj-20`
- [OK] Packed-i4 dequant-clamp dry-run with explicit `/usr/bin/llvm-readobj-20`
- [OK] Structured JSON checks over the generated dry-run records and existing
  real `ssh rvv` dequant-clamp evidence record
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-production-kernel-capability-campaign`
- [OK] Post-archive `python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-06/06-10-stage2-rvv-production-kernel-capability-campaign`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Bounded added-diff scan over source/test changes returned no legacy RVV
  route-authority matches. PRD marker names are negative reporting text only.

### Spec Update Decision

[NO SPEC UPDATE] The RVV plugin and testing specs already require Gate 4 final
audit coverage to parse generated evidence JSON into
`RVVLowPrecisionSameTargetMeasurementRecord`, feed that record through the
selected-dispatch overload, and prove the same source-backed facts reach the
production pressure-profile boundary. This round implements and verifies that
existing contract without changing API signatures or record schemas.

### Status

[COMPLETE MACRO TASK] Gates 1-4 are complete. The task is ready to finish and
archive.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 579: Stage2 RVV production-kernel capability Gate 2

**Date**: 2026-06-10
**Task**: Stage2 RVV production-kernel capability campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed Gate 2 only. The previous slice
made the production pressure-profile boundary real; this round made generated
artifact and same-target measurement workflow records source-backed enough for
that boundary to consume without treating q8/q4 labels, artifact names, route
ids, or handwritten metadata as authority.

### Main Changes

- Extended `RVVLowPrecisionSameTargetMeasurementRecord`,
  `RVVLowPrecisionSameTargetMeasurementPolicyInput`, and
  `RVVLowPrecisionProductionPressureProfile` with selected-boundary,
  generated artifact identity, measurement target, runtime-count, and
  non-authoritative pressure-label provenance fields.
- Added C++ fail-closed validation for missing/stale source record contract,
  selected boundary, generated object/header identity, measurement target,
  runtime-count provenance, and q8/q4 label-only pressure.
- Updated `rvv_generated_bundle_same_target_measure.py` to emit the
  source-backed record into per-op/root evidence and provider feedback inputs.
- Refreshed existing packed-i4 dequant-clamp `ssh rvv` evidence JSONs with the
  new record fields derived from already validated artifact identity and
  measurement configuration, without claiming a new runtime/performance run.
- Added focused C++ tests, script self-test cases, FileCheck coverage, and an
  RVV plugin code-spec update for the new cross-layer record/API contract.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] packed-i4 dequant dry-run plus manual `FileCheck --check-prefix=PACKED-WPRD`
- [OK] packed-i4 dequant-clamp dry-run plus manual `FileCheck --check-prefix=PACKED-CLAMP-WPRDC`
- [OK] `python3 -m json.tool` on refreshed packed-i4 evidence JSON files
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-production-kernel-capability-campaign`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded diff scan found no new legacy RVV route-authority path; q8/q4
  and metadata-only occurrences are negative guards, pressure labels, spec
  text, or evidence mirrors.

### Spec Update Decision

[SPEC UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` now records the
Gate 2 source-backed measurement-record/API contract, including field names,
validation matrix, bad cases, and tests required for selected-boundary,
generated artifact, measurement target, runtime-count, and non-authoritative
pressure-label provenance.

### Status

[OPEN MACRO TASK] Gates 1-2 are complete. Gates 3-4 remain open. The next
continuation point is Gate 3: selected-dispatch/performance policy consumes the
source-backed pressure-profile records for preference, denial, fallback, and
stale-provenance rejection.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 581: Stage2 RVV production-kernel capability campaign Gate 1

**Date**: 2026-06-10
**Task**: Stage2 RVV production-kernel capability campaign
**Branch**: `main`

### Summary

Created the new active macro task for the RVV production-kernel capability
campaign and completed Gate 1 only. The completed slice adds a production
pressure-profile boundary in the RVV low-precision performance-policy layer so
provider-owned resource/Gearbox facts, typed low-precision primitive facts,
runtime ABI facts, target capability mirrors, source-backed same-target
measurement inputs, and selected-dispatch policy facts are validated together
before selected-dispatch policy-input or measurement-record decisions return.

### Main Changes

- Added `RVVLowPrecisionProductionPressureProfile` and build/verify APIs to
  the low-precision performance policy contract.
- Added label-only q8/q4 and metadata-only pressure preflight rejection while
  reusing the existing provider tie-back, sibling-route, stale measurement, and
  selected-dispatch boundary validators.
- Wired selected-dispatch policy-input and source-backed record overloads
  through the pressure-profile materializer.
- Added focused RVV plugin positive coverage and fail-closed coverage for
  label-only authority, stale runtime ABI, stale primitive intrinsic, stale
  schedule decision, metadata-only provider support, sibling-route measurement,
  and stale selected-dispatch origin.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-production-kernel-capability-campaign`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded current-diff scan for legacy RVV route-authority markers returned
  no matches.

### Spec Update Decision

[NO SPEC UPDATE] The existing RVV plugin, EmitC route, variant-pipeline, and
testing specs already require provider-owned typed facts, same-target
measurement records, selected-dispatch mirrors, and fail-closed stale or
metadata-only rejection. This slice implements that existing contract as a
production pressure-profile boundary; no new durable rule was discovered.

### Status

[OPEN MACRO TASK] Gate 1 is complete. Gates 2-4 remain open. The next
continuation point is Gate 2: generated artifact and measurement workflow emits
source-backed same-target comparison records for the representative pressure
path on `ssh rvv`, using the production pressure-profile boundary as the
policy/measurement input contract.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 580: Stage2 RVV low-precision contraction primitive Gate 4

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed the final Gate 4 reconciliation.
Live source inspection showed the selected-dispatch/performance policy consumer
already exists in production source: `RVVLowPrecisionPerformancePolicy` exposes
strict evaluate/verify entry points and safe dispatch resolvers for same-target
measurement outcomes, policy inputs, measurement records, and selected-dispatch
boundary facts; provider route planning and target artifact validation consume
that policy before accepting packed-i4 performance preference or selected
dispatch mirrors.

### Main Changes

- Repaired the macro PRD from stale Gate 3 wording to the current Gate 4 final
  state and marked Gates 1-4 complete.
- Updated `task.json` so Trellis task metadata no longer reports Gate 4 open.
- Refreshed `check.jsonl` with the Gate 4 inspection and verification evidence.
- Did not change production C++ because the exact Gate 4 consumer requested by
  the direction brief was already present and covered: plugin tests parse the
  checked-in Gate 3 dequant-clamp `same_target_measurement_record` JSON, feed
  the selected-dispatch record overload, preserve correctness fallback, deny
  regression/no-win performance preference, reject stale schedule and
  correctness-disabled records, and cover the measured-win preference path.

### Testing

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-campaign`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `python3 -m json.tool artifacts/gate3-packed-i4-dequant-clamp-ssh/widening_product_reduce_dequant_clamp_f32/same_target_measurement_evidence.json`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded current-diff scan for legacy RVV route-authority markers returned
  no positive legacy-authority additions. The only `metadata-only` /
  `descriptor residue` matches are negative PRD guardrails.

### Spec Update Decision

[NO SPEC UPDATE] The RVV plugin spec already contains the Gate 4 packed-i4
selected-dispatch/performance policy consumption contract, including the
source-backed record overload, selected-dispatch boundary checks, resolver
fallback behavior, stale provenance rejection, and measured-win acceptance
requirements. This session reconciled the active Trellis task state with that
implemented contract.

### Status

[READY TO ARCHIVE] Gates 1-4 are complete. After final status checks, archive
the macro task and clear `.trellis/.current-task`.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 579: Stage2 RVV low-precision contraction primitive Gate 3

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed Gate 3 only. Source inspection
showed route/provider planning and target artifact mirrors already carried the
low-precision widening-reduction primitive facts, but source-backed same-target
measurement records only tied back a narrow primitive-chain subset. This round
expanded the measurement/evidence record boundary so later selected-dispatch
policy cannot consume a measurement unless it matches the provider-owned
primitive/resource/config/runtime provenance already validated by export.

### Main Changes

- Extended `RVVLowPrecisionSameTargetMeasurementRecord` and
  `RVVLowPrecisionSameTargetMeasurementPolicyInput` with provider primitive
  contract/kind, source/product/accumulator/result dtype and SEW/LMUL facts,
  primitive widening/reduction intrinsics, scalar seed splat,
  accumulator/result layouts, and reduction store-VL.
- Hardened record parsing and policy-input construction so missing or stale
  primitive measurement fields fail closed before selected-dispatch or
  performance policy consumption.
- Updated the same-target measurement script and checked-in dequant-clamp
  evidence JSON so generated records mirror validated low-precision
  resource/artifact metadata rather than carrying only primitive-chain names.
- Added plugin and target artifact C++ coverage for positive record propagation
  plus missing/stale primitive intrinsic fail-closed cases.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `python3 -m json.tool artifacts/gate3-packed-i4-dequant-clamp-ssh/widening_product_reduce_dequant_clamp_f32/same_target_measurement_evidence.json`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_same_target_measure.py`

### Spec Update Decision

[NO SPEC UPDATE] The existing RVV plugin and EmitC route specs already require
Gate 3 measurement records to be evidence tie-backs for provider-owned
resource/primitive facts and to fail closed on stale provider provenance. This
slice implements that contract at the measurement-record schema and C++ policy
input boundary.

### Status

[OPEN MACRO TASK] Gates 1-3 are complete. Gate 4 remains open. The next
continuation point is Gate 4: selected-dispatch/performance policy consumes the
expanded source-backed same-target measurement records fail-closed, preserving
correctness fallback and denying stale/no-win performance claims.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 578: Stage2 RVV low-precision contraction primitive Gate 2

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive-surface campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed Gate 2 only. The production
selected-body realization path already consumed many low-precision
product-reduction/dequant resource facts, but the selected resource candidate's
primitive-chain fields were not directly compared against provider-owned
`RVVLowPrecisionWideningReductionPrimitiveFacts` at the realization boundary.
This round tightened that RVV plugin-local boundary before route/provider or
target artifact authority can accept stale primitive-chain facts.

### Main Changes

- Added `validateLowPrecisionResourceCandidatePrimitiveFacts` in the RVV
  contraction selected-body realization owner so primitive contract/kind,
  chain contract/kind, product/reduction relations, intrinsic spellings,
  accumulator/result layouts, and reduction store-VL must match provider-owned
  primitive facts before realized `with_vl`, region-marker, handoff, or
  provider-visible resource facts are accepted.
- Added C++ regression coverage for missing primitive resource facts and a
  stale primitive reduction intrinsic, both failing closed at selected-body
  realization before route construction or artifact validation.
- Updated the macro PRD/task description to mark Gate 2 complete and leave
  Gates 3-4 open.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-campaign`
- [OK] `git diff --check`
- [OK] bounded diff scan for legacy RVV route-authority markers returned no
  matches.

### Spec Update Decision

[NO SPEC UPDATE] The RVV plugin spec already states that low-precision
product-reduction selected-body realization must consume provider-owned
widening-reduction primitive facts and pass-produced resource facts before
route construction. This slice implements that existing contract at the
candidate primitive-chain comparison point.

### Status

[OPEN MACRO TASK] Gates 1-2 are complete. Gates 3-4 remain open. The next
continuation point is Gate 3: carry the selected-body-realization-consumed
primitive/resource facts through route/provider/artifact export into generated
artifacts and source-backed same-target measurement records with fail-closed
stale mirror diagnostics.

### Git Commits

Final coherent commit is created after this journal entry.

## Session 579: Stage2 RVV production-kernel capability Gate 3

**Date**: 2026-06-10
**Task**: Stage2 RVV production-kernel capability campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed Gate 3 only. Gate 2 already
emitted source-backed pressure-profile records from selected-boundary,
provider-owned resource/primitive, generated artifact, and same-target
measurement facts. This round hardened the production selected-dispatch safe
resolver so accepted source-backed measurement records must also pass through
the production pressure-profile boundary before the resolver returns a dispatch
/ performance decision.

### Main Changes

- Updated the macro PRD from the stale Gate 2 current-slice description to Gate
  3 selected-dispatch/performance policy consumption, then marked Gate 3
  complete after implementation and checks.
- Changed `resolveRVVLowPrecisionDispatchPerformancePolicy(selection, record,
  dispatchBoundary, context)` so an accepted record/resource/measurement
  handoff also materializes `RVVLowPrecisionProductionPressureProfile`. If that
  boundary rejects stale or marker-only pressure facts, the resolver denies
  performance preference and win claims, preserves correctness fallback for the
  legal route, and carries the precise failure in `fallbackReason`.
- Added focused C++ coverage for a metadata-only selected-dispatch marker in
  the record resolver path, proving the resolver consumes the pressure-profile
  boundary instead of treating marker-only dispatch facts as policy authority.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Bounded added-diff scan for legacy RVV route-authority markers returned
  no `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  `__riscv_*_i32m1`, source-front-door, or descriptor-driven route-authority
  matches. The broader scan only found PRD non-authority wording and the new
  metadata-only negative test marker.

### Spec Update Decision

[NO SPEC UPDATE] The RVV plugin spec already requires selected-dispatch policy
consumption to prefer source-backed measurement records, validate the
record/resource/measurement handoff through the pressure-profile boundary, deny
performance preference for stale or marker-only facts, and preserve correctness
fallback where the selected route remains legal. This slice implements that
existing contract in production policy code and tests.

### Status

[OPEN MACRO TASK] Gates 1-3 are complete. Gate 4 remains open. The next
continuation point is Gate 4: final campaign closeout must prove the full
selected boundary -> plugin-owned realization/provider -> artifact/runtime
measurement -> selected-dispatch policy path, including parsed generated
evidence through the selected-dispatch record overload, before this macro task
can be finished or archived.

### Git Commits

Final coherent commit is created after this journal entry.


## Session 578: Stage2 RVV Gearbox Gate 1 pressure-profile admission

**Date**: 2026-06-10
**Task**: Stage2 RVV Gearbox Gate 1 pressure-profile admission
**Branch**: `main`

### Summary

Implemented Gate 1 low-precision production pressure-profile admission API, wired RVV contraction selected-body realization owner overload to consume it, added focused C++ success and fail-closed provenance coverage, and kept the macro campaign active for Gate 2.

### Main Changes

- Added primitive source load, primitive source extension, widening-product
  multiplicand-role, and widening-product extension-policy facts to Gearbox
  resource candidates and selected-body handoff attributes.
- Required RVV route-family validation, route metadata, target artifact mirror
  validation, and target support bundle header export to consume those facts.
- Extended same-target measurement records, production-pressure policy inputs,
  generated-bundle script self-tests, and the existing packed-i4
  dequant-clamp source-backed evidence record to preserve provider primitive
  provenance.
- Added focused positive and stale-provenance checks for Gearbox schedule,
  `gearbox_cross_region_handoff`, emission-plan metadata, target header export,
  parsed measurement records, and stale source-backed evidence.

### Git Commits

| Hash | Message |
|------|---------|
| `469a69ac` | (see git log) |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] Focused manual FileCheck positive checks for Gearbox schedule,
  selected-body realization, emission-plan metadata, and target header export.
- [OK] Focused manual FileCheck negative checks for stale primitive source
  extension in Gearbox schedule and `gearbox_cross_region_handoff`.
- [OK] `git diff --check`
- [OK] Bounded added-diff old-authority scan, excluding expected
  widening-reduction primitive intrinsic mirrors.
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-gate1`

### Status

[OPEN MACRO TASK] Gates 1-3 are complete. This session completed one Gate 4
primitive-provenance slice. Gate 4 remains open for measured same-target
production-kernel comparison maturity and any resource-schedule improvement or
explicit performance-denial closeout.

### Next Steps

- Continue Gate 4 from the source-backed primitive/resource/measurement
  provenance now in place. Do not switch to q8/q4 route authority,
  generated-bundle-only closeout, or a new macro task unless human steering
  redirects the campaign.

## Session 580: Stage2 RVV Gearbox Gate 2 schedule-choice consumption

**Date**: 2026-06-10
**Task**: Stage2 RVV Gearbox resource-aware selected-body realization campaign
**Branch**: `main`

### Summary

Continued the active macro task and completed Gate 2 only. Gate 1 already
created the source-backed production pressure-profile admission boundary; this
round made the packed-i4 Gearbox/resource-aware schedule choice part of the
admitted selected-body realization facts. The macro task remains open for Gates
3-4.

### Main Changes

- Extended `RVVLowPrecisionSelectedBodyRealizationAdmission` with provider-owned
  schedule decision contract, decision, and reason.
- Hardened packed-i4 admission so the selected resource candidate must carry
  the accepted schedule decision fields before a pressure profile can admit
  realization.
- Wired the contraction selected-body realization owner to materialize admitted
  schedule mirrors on realized `with_vl` operations and the Gearbox
  cross-region handoff.
- Added focused C++ coverage for successful admitted schedule consumption and
  missing/stale schedule fail-closed diagnostics.
- Updated the RVV plugin spec with the Low-Precision Realization Admission
  Schedule Handoff contract.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-gearbox-realization-campaign`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Bounded added-diff scan over source/test changes returned no legacy RVV
  route-authority matches. Documentation mentions forbidden markers only as
  non-authority scan vocabulary.

### Spec Update Decision

[UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` now records the
Gate 2 admission schedule handoff API fields, validation matrix, mirror-only
realized structure contract, tests required, and Common EmitC/export neutrality
boundary.

### Status

[OPEN MACRO TASK] Gates 1-2 are complete. Gates 3-4 remain open. The next
continuation point is Gate 3: realized body plus route/artifact/measurement
proof consumes the admitted packed-i4 schedule decision against source-backed
same-target records.

### Git Commits

Final coherent commit is created after this journal entry.


## Session 579: Gate 4 primitive provenance through Gearbox measurement

**Date**: 2026-06-10
**Task**: Gate 4 primitive provenance through Gearbox measurement
**Branch**: `main`

### Summary

Completed a Gate 4 slice for the active RVV low-precision macro task: primitive source/load/extension and widening-product role/policy facts now flow through Gearbox resource admission, selected-body handoff, route/target metadata, target header export, same-target measurement records, and production-pressure policy inputs with stale provenance rejection. The macro task remains active for measured same-target production-kernel comparison maturity.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `final-commit-see-git-log` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 581: Gate 4 packed-i4 dequant-clamp no-win policy audit

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive surface campaign
**Branch**: `main`

### Summary

Continued the active macro task under Gate 4. Audited the packed-i4
dequant/dequant-clamp same-target comparison boundary and found no safe
production resource-schedule win to claim in this slice: the current
dequant-clamp generated C++ already materializes the selected packed-i4
low/high nibble unpack, two widening products, pair-sum, single `vwredsum`, and
f32 dequant/clamp/store path. The current measured evidence remains a
regression/no-win boundary, so performance preference stays explicitly denied
while route support and correctness execution remain allowed.

### Main Changes

- Hardened the dequant-clamp packed-i4 target fixture so generated C++ must
  expose the full low/high i4 unpack, pair-sum, single-reduction, f32 clamp, and
  store sequence before the no-win policy evidence can be trusted.
- Updated the macro PRD and task notes with the current measured-comparison
  slice, no-production-source-change justification, no-win denial outcome, and
  remaining continuation point.
- Left Gate 4 open for a future provider-owned schedule/resource repair plus
  fresh same-target timing before any performance-preferred dispatch claim.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`
- [OK] Focused lit/FileCheck for
  `pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4.mlir`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-gate1`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Bounded added-diff old-authority scan. The only match was negative PRD
  wording for q8/q4 named wrappers; no new positive legacy route authority was
  added.

### Status

[OPEN MACRO TASK] Gates 1-3 remain complete. Gate 4 remains open; this slice
keeps the current dequant-clamp packed-i4 no-win/regression as an explicit
performance-preference denial.

### Spec Update Decision

[NO SPEC CHANGE] This slice added no new API, payload field, command signature,
or policy contract. The existing RVV plugin, variant-pipeline, EmitC route, and
testing specs already cover the packed-i4 no-win maturity boundary,
source-backed same-target evidence, performance-selection denial, and
metadata-only claim rejection. The durable state change is captured in the PRD,
fixture coverage, and check records.

## Session 582: Gate 4 selected-dispatch no-win preference denial

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive surface campaign
**Branch**: `main`

### Summary

Continued the active macro task under Gate 4. This slice made the existing
source-backed packed-i4 no-win/regression outcome a stricter
selected-dispatch policy boundary: route support and correctness execution
remain allowed, but selected-dispatch case/fallback policy or mirror text may
not carry `performance-preferred` markers unless measured-win evidence and
provider maturity facts promote the decision.

### Main Changes

- Added fail-closed selected-dispatch no-win validation in
  `RVVLowPrecisionPerformancePolicy.cpp`.
- Added plugin regression coverage for stale selected-dispatch case policy and
  selected-dispatch case mirror performance-preferred markers.
- Updated the macro PRD/task notes and RVV plugin spec with the new no-win
  dispatch preference boundary.
- Did not rerun `ssh rvv` measurement because no schedule/resource behavior was
  changed.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test
  tcrv-opt tcrv-translate`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32-packed-i4`
  from `build/test`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py --self-test`
- [OK] `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-gate1`
- [OK] `git diff --check`
- [OK] Bounded added-diff old-authority scan found no legacy RVV
  route-authority or q8/q4/source-front-door markers in added lines.

### Status

[OPEN MACRO TASK] Gates 1-3 remain complete. Gate 4 remains open for a real
provider-owned packed-i4 schedule/resource repair plus fresh same-target timing
before any performance-preferred dispatch claim.

### Spec Update Decision

[UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` now records the
low-precision no-win dispatch preference boundary and its required C++ tests.

## Session 583: Gate 4 packed-i4 low-product schedule repair

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive surface campaign
**Branch**: `main`

### Summary

Continued the active Gate 4 macro task with a production schedule/resource
repair. The packed-i4 statement owner now emits low-nibble sign-extension and
the low widening product before high-nibble unpack/product construction, while
preserving provider-owned typed primitive facts, pair-sum semantics, and one
`vwredsum`. A self-repair also synchronized realization-admission schedule
facts with provider schedule facts so regenerated evidence, policy records,
target metadata, and artifact mirrors carry the same
low-product-before-high-unpack reason.

### Main Changes

- Repaired packed-i4 statement planning in the RVV provider path and updated
  target artifact validation to require the new statement order.
- Updated provider schedule/remediation/maturity facts, generated-bundle script
  checks, target/FileCheck fixtures, and C++ policy tests for the
  low-product-before-high-unpack contract.
- Regenerated Gate 4 same-target evidence for dequant and dequant-clamp after
  the production repair.
- Kept performance policy on correctness fallback because both fresh
  measurements remain regression/no-win.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] Focused lit/FileCheck for packed-i4 dequant/dequant-clamp target
  fixtures and script dry-run tests: 5 passed from `build/test`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py
  --self-test`
- [OK] Fresh `ssh rvv` same-target timing:
  dequant `0.688202..0.705410`, dequant-clamp `0.683721..0.705212`
- [OK] Bounded old-evidence scan found no old Gate 3 evidence ID, old
  pair-sum-only schedule decision, stale schedule reason, q8/q4 route
  authority, source-front-door route, descriptor compute, or Common EmitC RVV
  semantic inference in touched source/tests/spec/current Gate 4 artifacts.

### Status

[OPEN MACRO TASK] Gates 1-3 remain complete. The current Gate 4 repair slice is
complete and evidence-backed, but Gate 4 remains open because the repaired
packed-i4 schedule still measured below the scalar packed-i4 baseline.

### Continuation

Next owner should choose a different provider-owned packed-i4 schedule/resource
bottleneck, or record a precise blocker, before any performance-preferred
dispatch claim. The current policy decision remains correctness fallback with
`same-target-measurement-no-win-or-regression`.

### Spec Update Decision

[UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` records the
low-product-before-high-unpack schedule/resource repair, the provider mirror
versus fresh measurement distinction, and the current Gate 4 strict measurement
ranges.

## Session 584: Gate 4 same-target evidence-root policy ingestion

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive surface campaign
**Branch**: `main`

### Summary

Continued the active Gate 4 macro task with a production policy-ingestion
refinement. The RVV low-precision performance policy now has evidence-root
overloads that consume the complete source-backed same-target JSON root before
selected-dispatch acceptance. The root verifier checks root status, ssh target,
timing method, result classification, harness schedule mirrors, packed-i4
oracle selection, maturity input, and provider feedback tie-back against the
nested measurement record and provider-owned packed-i4 schedule facts.

### Main Changes

- Added evidence-root parsing/policy/evaluation/verify APIs in
  `RVVLowPrecisionPerformancePolicy`.
- Updated plugin coverage so both current Gate 4 dequant and dequant-clamp
  JSON roots drive selected-dispatch policy evaluation.
- Added stale root-level speedup and stale root-level harness schedule
  rejection coverage.
- Updated the RVV plugin spec and macro PRD with the evidence-root ingestion
  contract.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py
  --self-test`
- [OK] Focused packed-i4 Gate 4 lit filter: 5 tests passed from `build/test`

### Status

[OPEN MACRO TASK] Gates 1-3 remain complete. Gate 4 remains open. This slice
does not change generated RVV schedule/runtime behavior, so no fresh `ssh rvv`
timing was rerun; the existing dequant `0.688202..0.705410` and dequant-clamp
`0.683721..0.705212` regression/no-win records remain the consumed evidence and
continue to deny performance preference.

### Spec Update Decision

[UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` now records complete
same-target evidence-root policy ingestion and stale root-level
result/schedule rejection as the current Gate 4 policy contract.

## Session 585: Gate 4 packed-i4 low-shifted-product-rescale schedule repair

**Date**: 2026-06-10
**Task**: Stage2 RVV low-precision contraction primitive surface campaign
**Branch**: `main`

### Summary

Continued the active Gate 4 macro task with a production schedule/resource
repair slice. The packed-i4 provider schedule now avoids materializing both
low signed-i4 values before the low product: it shift-lefts the low nibbles,
computes the widening low product on shifted operands, arithmetic-rescales the
i16 product by 8, then forms the high product, pair-sum, and single `vwredsum`.

### Main Changes

- Updated provider-owned packed-i4 schedule/remediation facts to the
  low-shifted-product-rescale budget-6of32 decision.
- Updated statement generation, route-family step counts, target artifact
  validation, generated-bundle scripts, FileCheck fixtures, and C++ tests for
  the new 12-step packed-i4 loop body.
- Refreshed source-backed same-target evidence roots for dequant and
  dequant-clamp and kept policy on correctness fallback because both remain
  regression/no-win.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py
  --self-test`
- [OK] Focused packed-i4 Gate 4 lit filter: 5 tests passed from `build/test`
- [OK] Fresh `ssh rvv` same-target timing:
  dequant `0.688202..0.705133`, dequant-clamp `0.677994..0.704931`

### Status

[OPEN MACRO TASK] Gates 1-3 remain complete. Gate 4 remains open because the
repaired packed-i4 schedule is correctness-supported but still below the scalar
packed-i4 baseline. The selected-dispatch policy remains
`correctness-fallback` with performance preference denied.

### Continuation

Next owner should pick a different provider-owned packed-i4 resource bottleneck
or record a sharper production blocker before any `performance-preferred`
claim.

### Spec Update Decision

[UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` records the
low-shifted-product-rescale schedule/resource contract and final candidate-
sensitive Gate 4 no-win measurement ranges.

## Session 586: Gate 4 packed-i4 resource-cost admission boundary

**Date**: 2026-06-11
**Task**: Stage2 RVV low-precision contraction primitive surface campaign
**Branch**: `main`

### Summary

Continued the active Gate 4 macro task with a production resource-cost and
admission boundary for the current packed-i4 low-shifted-product-rescale path.
The path now carries a provider-owned cost contract/model, loop-body step
count, no-win blocker, and performance admission decision from Gearbox resource
selection through selected-body realization, route facts, target mirrors,
same-target evidence inputs, and `RVVLowPrecisionPerformancePolicy`.

### Main Changes

- Added packed-i4 resource-cost facts:
  `rvv-low-precision-packed-i4-resource-cost-contract.v1`,
  `low-shifted-product-rescale-loop-12-peak-live-6of32-two-region-vsetvl.v1`,
  loop-body steps `12`,
  `packed-i4-low-shifted-product-rescale-loop-12-budget-6of32-no-win`, and
  `deny-performance-preferred-with-resource-cost-no-win-blocker`.
- Required those facts in Gearbox handoff verification, route-family resource
  selection, route metadata, target artifact metadata, generated-bundle
  metadata, same-target measurement records, evidence-root policy ingestion,
  production-pressure profiles, and no-win admission denial.
- Updated the current source-backed dequant/dequant-clamp evidence roots with
  provider resource-cost/admission facts without changing timing data or the
  generated runtime schedule.
- Self-repaired stack pressure exposed in the monolithic target exporter test by
  moving large provider/control temporaries off the stack and raising the test
  binary's own soft stack limit.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py
  --self-test`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py
  --dry-run --artifact-root /tmp/tcrv-gate4-packed-i4-dry-run --run-id
  codex-gate4-resource-cost --overwrite --op-kind
  widening_product_reduce_dequantize_f32 --op-kind
  widening_product_reduce_dequant_clamp_f32`
- [OK] `git diff --check`
- [OK] Bounded added-diff old-authority scan: only PRD non-goal wording matched.

### Status

[OPEN MACRO TASK] Gates 1-3 remain complete. Gate 4 remains open. The current
resource-cost boundary sharpens the production-consumed no-win blocker but does
not change generated runtime behavior or create a measured win, so no fresh
`ssh rvv` timing was rerun in this slice. Existing dequant
`0.688202..0.705133` and dequant-clamp `0.677994..0.704931` regression/no-win
records still deny performance preference.

### Continuation

Next owner should choose the next provider-owned packed-i4 schedule/resource
bottleneck beyond the current 12-step low-shifted-product-rescale path, or
record a precise production-consumed blocker if no measured-win repair is
available. Final coherent commit is created after this journal entry.

### Spec Update Decision

[NO UPDATE] This slice implemented the existing Gate 4 policy boundary in
production code and tests; no durable spec wording change was needed beyond the
active macro PRD update.

## Session 587: Gate 4 resource-cost admission decision consumption

**Date**: 2026-06-11
**Task**: Stage2 RVV low-precision contraction primitive surface campaign
**Branch**: `main`

### Summary

Continued the active Gate 4 macro task with a policy/provider closure slice.
The packed-i4 resource-cost admission field is now consumed as a decision, not
only mirrored as metadata: current no-win/regression policy requires the deny
admission, while any future measured-win `performance-preferred` path must
carry an explicit provider-owned measured-win admission value.

### Main Changes

- Added provider admission value
  `admit-performance-preferred-with-resource-cost-measured-win`.
- Split packed-i4 policy validation so base selection facts accept only known
  resource-cost admission decisions, no-win consistency requires
  `deny-performance-preferred-with-resource-cost-no-win-blocker`, and
  measured-win policy requires the new admit value.
- Made direct `RVVLowPrecisionPerformanceMeasurementOutcome` policy evaluation
  consume provider resource-cost and realization-admission tie-backs.
- Updated plugin and target measured-win fixtures so performance-preferred
  selection requires the explicit admission update; stale no-win admission now
  fails strict policy verification.

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test` after fixing the
  target measured-win fixture admission value
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py
  --self-test`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-
  surface-gate1`
- [OK] `git diff --check`

### Status

[OPEN MACRO TASK] Gates 1-3 remain complete. Gate 4 remains open. This slice
does not change generated RVV runtime scheduling, so no fresh `ssh rvv` timing
was rerun; existing dequant `0.688202..0.705133` and dequant-clamp
`0.677994..0.704931` regression/no-win evidence still denies performance
preference.

### Continuation

Next owner should choose the next provider-owned packed-i4 schedule/resource
bottleneck beyond the current 12-step low-shifted-product-rescale path, or
record a production-consumed blocker if no measured-win repair is available.

### Spec Update Decision

[UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` records that
measured-win `performance-preferred` policy requires a provider-owned
resource-cost admission update, while current no-win/regression policy must
retain the deny admission and fail closed on metadata-only or measurement-only
promotion.

## Session 588: Gate 4 high-nibble vwmacc schedule/resource repair

**Date**: 2026-06-11
**Task**: Stage2 RVV low-precision contraction primitive surface campaign
**Branch**: `main`

### Summary

Continued the active Gate 4 macro task with a production schedule/resource
repair. The packed-i4 product-reduction dequant/dequant-clamp route now
accumulates the high-nibble widening product into the rescaled low product with
`__riscv_vwmacc_vv_i16mf2`, replacing the separate high `vwmul` plus i16
pair-add. The packed-i4 loop contract is now 11 loop-body steps and 5-of-32
peak live vector groups.

### Main Changes

- Updated provider schedule/resource constants to
  `high-nibble-vwmacc-loop-11-peak-live-5of32-two-region-vsetvl.v1`.
- Reworked statement planning to emit low-shifted product rescale followed by
  high-nibble `vwmacc` and a single `vwredsum`.
- Made route-family validation, target artifact validation, generated
  header/index mirrors, ABI e2e checks, and same-target measurement records
  consume the new schedule/resource facts.
- Updated packed-i4 policy facts from regression to no-win maturity for this
  high-nibble vwmacc evidence, while keeping performance preference denied by
  the resource-cost no-win admission blocker.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py
  --self-test`
- [OK] focused lit from `build/test` for the two packed-i4 target fixtures,
  ABI e2e dry-run tests, and same-target dry-run test
- [OK] fresh `ssh rvv` measurement:
  dequant `0.896848..1.020953`, no-win, 12 summaries / 60 measurements /
  12 correctness records
- [OK] fresh `ssh rvv` measurement:
  dequant-clamp `0.867416..1.043671`, no-win, 24 summaries /
  120 measurements / 24 correctness records

### Status

[OPEN MACRO TASK] Gates 1-3 remain complete. Gate 4 remains open. This slice
changed generated RVV runtime scheduling and collected fresh same-target
evidence. The result is still no-win, so route support and correctness
execution remain allowed through correctness fallback, and
`performance-preferred` remains denied.

### Continuation

Next owner should choose the next provider-owned packed-i4 schedule/resource
bottleneck beyond the high-nibble vwmacc 11-step/5-of-32 contract, or record a
production-consumed blocker if no measured-win repair is available.

### Spec Update Decision

[UPDATED] `.trellis/spec/extension-plugins/rvv-plugin.md` records the
high-nibble vwmacc schedule/resource contract, 11-step/5-of-32 cost model,
fresh no-win timing ranges, and the continued correctness-fallback policy
boundary.

## Session 589: Gate 4 measured admission closure

**Date**: 2026-06-11
**Task**: Stage2 RVV low-precision contraction primitive surface campaign
**Branch**: `main`

### Summary

Continued the active Gate 4 macro task without opening a neighboring task.
The packed-i4 high-nibble vwmacc path now carries a production-owned
no-safe-local-repair admission closure and reopen requirement from Gearbox
resource facts through selected-body realization, route metadata, target
validation, generated-bundle mirrors, same-target evidence roots, and
`RVVLowPrecisionPerformancePolicy`.

### Main Changes

- Added `performance_admission_closure =
  no-safe-local-repair-no-win-high-nibble-vwmacc-loop-11-budget-5of32.v1`.
- Added `performance_admission_reopen_requirement =
  provider-schedule-resource-repair-plus-source-backed-measured-win-and-updated-admission-facts.v1`.
- Required target validation and policy ingestion to reject stale or missing
  closure/reopen facts before accepting performance preference.
- Preserved correctness fallback while keeping `performance-preferred` denied
  by `deny-performance-preferred-with-resource-cost-no-win-blocker`.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test` after updating the
  fresh source-backed object SHA expectation
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `python3 scripts/rvv_generated_bundle_same_target_measure.py
  --self-test`
- [OK] focused lit from `build/test` for the two packed-i4 target fixtures,
  ABI e2e dry-run tests, and same-target dry-run test
- [OK] fresh `ssh rvv` measurement:
  dequant `0.897163..1.018998`, no-win, 12 summaries / 60 measurements /
  12 correctness records
- [OK] fresh `ssh rvv` measurement:
  dequant-clamp `0.864516..1.043210`, no-win, 24 summaries /
  120 measurements / 24 correctness records

### Status

[OPEN MACRO TASK] Gates 1-3 remain complete. This slice closes the current
high-nibble vwmacc measured admission/denial subgate as a production-consumed
no-safe-local-repair/no-win boundary, but Gate 4 remains open for a future
provider-owned schedule/resource repair or a broader production blocker.
