# Journal - codex (Part 23)

> Continuation from `journal-22.md` (archived at ~2000 lines)
> Started: 2026-06-05

---



## Session 458: Stage2 RVV dequant-clamp ABI closure

**Date**: 2026-06-05
**Task**: Stage2 RVV dequant-clamp ABI closure
**Branch**: `main`

### Summary

Closed generated dequant-clamp f32 epilogue executable ABI evidence with dry-run and ssh rvv correctness; production route code unchanged.

### Main Changes

- Repaired the two pre-realized strided memory target fixture header checks by
  treating target metadata ordering as non-authority while still asserting all
  provider-derived mirrors.
- Added fail-closed target-export mutations for stale runtime ABI order, route
  operand binding, provider support mirror, strided memory layout, and
  source/destination stride source.
- Kept production provider/materializer/target C++ unchanged because current
  base memory provider and target validators already derive and reject the
  required route facts.

### Git Commits

- included-in-this-commit

### Testing

- [OK] Focused lit reproduction failed before the fixture repair and passed
  after the repair for both strided memory target fixtures.
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Focused lit filter for strided memory verifier, generated-bundle
  dry-run, and direct pre-realized route-entry fail-closed tests.
- [OK] Generated-bundle dry-runs for explicit/pre-realized
  `strided_load_unit_store` and `unit_load_strided_store`.
- [OK] Real `ssh rvv` generated-bundle runs for explicit/pre-realized
  `strided_load_unit_store` and `unit_load_strided_store`, each passing
  counts `0,1,16,23,257` with stride bytes `4,8`.
- [WARN] `ninja -C build check-tianchenrv` is 491/494: prior strided memory
  failures are fixed; remaining failures are unrelated existing runtime scalar
  splat-store negative, construction protocol common, and template extension
  plugin tests.
- [OK] `git diff --check`
- [OK] Trellis validation and added-line old-authority scan

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 479: Stage2 RVV widening-dot reduce selected-body realization

**Date**: 2026-06-06
**Task**: Stage2 RVV widening dot reduce selected-body realization
**Branch**: `main`

### Summary

Closed the focused plain `widening_dot_reduce_add` pre-realized selected-body
realization evidence path. The existing RVV contraction realization owner
already materializes
`tcrv_rvv.typed_widening_dot_reduce_pre_realized_body` into a realized typed
`setvl`/`with_vl`/load/load/`tcrv_rvv.widening_dot_reduce`/store body. This
round added the missing local stale-authority negative for the plain fixture,
aligned the pre-realized generated-bundle dry-run to the `17` tail case, and
proved real `ssh rvv` generated-bundle execution for counts
`0,1,16,17,257` and patterns `0,1`.

### Main Changes

- Created and completed the Trellis task for the Stage2 RVV plugin-local
  selected-body realization owner path for plain widening-dot reduce.
- Added a `STALE-AUTH` negative to
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-dot-reduce-add.mlir`
  showing a stale `route_id = "rvv-i32m1"` on the pre-realized typed body is
  rejected before route/provider facts.
- Updated
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-dot-reduce-add-dry-run.test`
  from tail count `23` to `17`, matching the current brief and explicit ABI
  evidence set.

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | `rvv: close widening dot reduce selected-body realization` |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] pre-realized target fixture direct FileCheck for `REALIZED`, `PLAN`,
  `HEADER`, and `STALE-AUTH`
- [OK] explicit target fixture direct FileCheck for `PLAN` and `HEADER`
- [OK] pre-realized generated-bundle dry-run for counts `0,1,16,17,257`
- [OK] direct FileCheck of the pre-realized dry-run `ROOT`, `WDOT`, and
  `HARNESS` prefixes
- [OK] explicit generated-bundle dry-run regression for counts `0,1,16,17,257`
- [OK] real `ssh rvv` generated-bundle compile/run:
  `PASS op=widening_dot_reduce_add counts=0,1,16,17,257 patterns=0,1`
- [OK] runtime evidence confirmed signed horizontal widening dot products,
  seed-added behavior, add-only/mul-only distinguishing cases, scalar
  `out[0]`, source preservation, accumulator preservation, and tail
  preservation
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] bounded added-line old-authority scan; only the intentional
  `STALE-AUTH` negative route-id injection matched
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Trellis context validation
- [REPAIR] `FileCheck`, `not`, and `llvm-readobj` were not in PATH; reran
  direct commands with `/usr/lib/llvm-20/bin/*`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 475: Stage2 RVV explicit computed masked indexed scatter store ABI closure

**Date**: 2026-06-06
**Task**: Stage2 RVV explicit computed masked indexed scatter store ABI closure
**Branch**: `main`

### Summary

Closed explicit `computed_masked_indexed_scatter_store_unit_load`
generated-bundle executable ABI evidence on ssh rvv for counts
`0,1,16,17,257` with patterns `0,1`, active/inactive compare-produced mask
lanes, unique noncontiguous indexed destinations, inactive indexed-destination
preservation, source preservation, and tail preservation; no production code
change required.

### Main Changes

- Created and archived the Trellis task for the explicit
  `computed_masked_indexed_scatter_store_unit_load` executable ABI closure.
- Proved the existing explicit selected-body generated bundle is executable on
  real `ssh rvv` without production compiler, script, fixture, or test changes.
- Recorded completion evidence for provider-derived ABI order, route operand
  binding, compare-produced mask facts, runtime index source, unique index
  facts, masked indexed destination store facts, inactive-lane contract,
  required header/type mirrors, and target validation coverage.

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] explicit generated-bundle dry-run for counts `0,1,16,17,257`
- [OK] direct `FileCheck` of the dry-run test `STDOUT`, `ROOT`,
  `CMISCATTER`, and `HARNESS` prefixes
- [OK] real `ssh rvv` generated-bundle compile/run:
  `PASS op=computed_masked_indexed_scatter_store_unit_load counts=0,1,16,17,257 patterns=0,1`
- [OK] focused explicit target fixture `PLAN` and `HEADER` FileCheck commands
- [OK] focused pre-realized target fixture `REALIZED`, `PLAN`, and `HEADER`
  FileCheck commands
- [OK] adjacent pre-realized generated-bundle dry-run regression
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] bounded old-authority scan over production/source diff lines
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Trellis context validation
- [SKIP] script self-test was not required because
  `scripts/rvv_generated_bundle_abi_e2e.py` was not changed

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 470: Stage2 RVV f32 clamp/select executable ABI closure

**Date**: 2026-06-06
**Task**: Stage2 RVV f32 clamp/select executable ABI closure
**Branch**: `main`

### Summary

Closed the bounded executable ABI evidence gap for pre-realized RVV f32 clamp/select. The generated bundle ran on real `ssh rvv` with scalar-oracle checks for counts 0,1,16,17,257, two bound pairs, patterns 0/1, source preservation, and output tail sentinel preservation.

### Main Changes

- Added and archived the Trellis task PRD/context with the generated bundle path, remote evidence path, remote compile/run summary, and focused check list.
- No compiler, fixture, script, or spec source files were changed.

### Git Commits

included-in-this-commit

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate`
- [OK] focused `tcrv-opt` materialization/emission plan command for f32 clamp/select
- [OK] focused `tcrv-translate --tcrv-export-target-header-artifact`
- [OK] negative verifier fixture for pre-realized f32 clamp/select
- [OK] generated-bundle dry-run with counts 0,1,16,17,257
- [OK] `ssh rvv` generated-bundle execution with final PASS marker
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [WARN] `python3 -m lit` unavailable locally: `No module named lit`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 467: Stage2 RVV computed-mask select/merge selected-body route foundation

**Date**: 2026-06-06
**Task**: Stage2 RVV computed-mask select/merge selected-body route foundation
**Branch**: `main`

### Summary

Closed computed-mask vector select/merge route foundation with focused
target-artifact stale mirror checks, explicit/pre-realized route/header
evidence, generated-bundle dry-runs, and real `ssh rvv` correctness for counts
`0,1,16,17,257`.

### Main Changes

- Added target artifact fail-closed checks for stale computed-mask select
  predicate, runtime ABI order, route operand binding plan/summary, provider
  mirror, target leaf profile, mask role/source/form, and select layout.
- Kept production C++ provider/materializer/target semantics unchanged because
  the current provider-owned computed-mask select contracts and target
  validation already support the bounded route.

### Git Commits

- included-in-this-commit

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] Direct FileCheck equivalents for pre-realized `computed_mask_select`
  realization/plan/header and stale mirror fail-closed paths
- [OK] Direct FileCheck equivalents for explicit `computed_mask_select_sle`
  plan/header paths
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Pre-realized generated-bundle dry-run for `computed_mask_select`
- [OK] Explicit generated-bundle dry-run for `computed_mask_select_sle`
- [OK] Pre-realized real `ssh rvv` generated-bundle run:
  `PASS op=computed_mask_select counts=0,1,16,17,257 compare_data_patterns=2`
- [OK] Explicit real `ssh rvv` generated-bundle run:
  `PASS op=computed_mask_select_sle counts=0,1,16,17,257 compare_data_patterns=2`
- [OK] `git diff --check`
- [OK] Trellis task validation and code/test added-line authority scan

### Status

[OK] **Completed** as executable computed-mask select/merge selected-body route
foundation. No `.trellis/spec/` update was needed because the existing RVV
plugin, EmitC-route, and testing contracts already captured provider-owned
computed-mask select behavior.

### Next Steps

- None - task complete


## Session 463: Stage2 RVV Gearbox two-candidate dequant route realization

**Date**: 2026-06-05
**Task**: Stage2 RVV Gearbox two-candidate dequant route realization
**Branch**: `main`

### Summary

Expanded the existing typed RVV `dequantize_i32_to_f32` Gearbox route from a
single legal u1 candidate to a bounded two-candidate contract with deterministic
u2 selection and real selected-route materialization.

### Main Changes

- Added the legal u2 Gearbox candidate while keeping the positive proof route
  bounded to the existing typed `dequantize_i32_to_f32` body.
- Made the selected u2 schedule change production route planning: the loop
  step is `full_chunk_vl * 2` and the loop body contains a second
  runtime-VL setvl/load/convert/scale/store slice.
- Extended provider validation to require selected-candidate membership,
  supported unroll 1/2, u2 route-plan materialization, and second-slice
  statement facts before route construction.
- Extended target validation and tests so provider-derived candidate and
  selected-schedule mirrors reject stale metadata and stale u2 route-plan
  structure.

### Git Commits

(Commit created after journal entry as part of the final task commit.)

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test`
- [OK] `lit -sv . --filter 'rvv-gearbox-dequantize-i32-to-f32|explicit-selected-body-artifact-dequantize-i32-to-f32'` from `build/test`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded old-authority/q-name scan; added hits are negative fixtures only
- [N/A] `ssh rvv`; no executable correctness, runtime, or performance claim

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 464: Stage2 RVV Gearbox bounded candidate-selection contract

**Date**: 2026-06-05
**Task**: `06-05-06-05-rvv-gearbox-bounded-candidate-selection-contract`
**Branch**: `main`

### Summary

Expanded the existing RVV Gearbox dequant MVP from a single static schedule
annotation into a bounded candidate-set plus selected-candidate contract for
the existing typed `dequantize_i32_to_f32` route. This remains a
pass/provider/target contract slice; no runtime benchmark, tuning cache, or
ssh-rvv executable claim was made.

### Main Changes

- Added provider-owned Gearbox candidate inventory, selected candidate,
  selection reason, and legality scope constants for the bounded
  `dequantize_i32_to_f32` `e32,m1,unroll=1` candidate.
- Updated `--tcrv-rvv-materialize-gearbox-schedules` to materialize the new
  candidate-selection facts on both `tcrv_rvv.with_vl` and
  `tcrv_rvv.dequantize`.
- Extended RVV dialect verification, dequant route facts, route-family plan,
  route description, conversion dtype-policy validation contract, target
  route-family validation, and header metadata mirrors.
- Added provider-side membership validation so the selected candidate must
  belong to the pass-produced legal candidate set before route construction.
- Added focused positive and negative lit/C++ coverage for candidate inventory,
  selected candidate, stale metadata, unsupported unroll, missing candidate
  set, and selected-candidate membership failure.

### Git Commits

(Recorded in the final commit for this session.)

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test`
- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-gearbox-dequantize-i32-to-f32|explicit-selected-body-artifact-dequantize-i32-to-f32'`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `git diff --check`
- [OK] Bounded old-authority/q-name scan over touched diff. Hits were limited
  to negative `artifact-name-derived-gear` and `route-string-derived-gear`
  guardrail strings.
- [N/A] `ssh rvv` was not run because this task makes no executable
  correctness/runtime/performance claim.

### Self-Repair

- Added missing RVV dialect verifier allow-list entries for new Gearbox attrs
  on `tcrv_rvv.with_vl` and `tcrv_rvv.dequantize`.
- Corrected the C++ target test expectation to match the target validator's
  provider-facts diagnostic boundary.

### Status

[OK] **Completed** as a pass/provider/target contract foundation.

### Next Steps

- Finish/archive the Trellis task and create one coherent commit.


## Session 463: Stage2 RVV Gearbox pass MVP for typed conversion route control

**Date**: 2026-06-05
**Task**: `06-05-rvv-gearbox-autotuning-pass-reference`
**Branch**: `main`

### Summary

Narrowed the existing Gearbox reference intake into a concrete Stage2 RVV
plugin-local pass/provider slice for the existing typed
`dequantize_i32_to_f32` route. The v3 Gearbox artifact was used only as
reference input, not as an implementation contract.

### Main Changes

- Added `--tcrv-rvv-materialize-gearbox-schedules` as a production MLIR pass
  registered in `tcrv-opt`.
- Implemented the first bounded static Gearbox schedule for selected typed RVV
  dequant bodies, deriving schedule facts from structural body/config/runtime
  evidence rather than route strings, artifact names, q-names, ABI strings, or
  intrinsic spelling.
- Extended RVV provider dequant route planning and validation so pass-produced
  Gearbox facts are required and consumed before route construction.
- Extended conversion dtype-policy target validation and target metadata/header
  output so Gearbox fields appear only as provider-derived mirrors, with stale
  mirror rejection.
- Added positive and negative lit/FileCheck tests for pass materialization,
  provider missing-fact failure, stale Gearbox attrs, unsupported runtime AVL
  facts, and artifact-name-derived stale target metadata.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test`
- [OK] `cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-gearbox-dequantize-i32-to-f32|explicit-selected-body-artifact-dequantize-i32-to-f32'`
- [OK] `./build/bin/tianchenrv-target-artifact-export-test`
- [OK] `./build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `git diff --check`
- [OK] Bounded old-authority/q-name scan over touched files. Expected hits were
  limited to PRD/checklist non-goals, negative `artifact-name-derived-gear`
  tests, and pre-existing legacy negative/compatibility text in touched files.
- [N/A] `ssh rvv` was not run because this pass/provider/metadata slice makes
  no executable correctness or performance claim.

### Status

[OK] **Completed** as a pass-plus-provider-consumption MVP. Ready for Trellis
finish/archive and one coherent commit.

### Next Steps

- Later Gearbox work can broaden the same pass/provider fact surface to another
  conversion family or add runtime/performance evidence in a separate Hermes
  round.


## Session 461: Stage2 RVV computed-mask select executable policy closure

**Date**: 2026-06-05
**Task**: Stage2 RVV computed-mask select executable policy closure
**Branch**: `main`

### Summary

Closed the executable evidence loop for the pre-realized RVV
computed-mask select route. Production C++ did not change; the source change is
neutral harness/evidence support that exposes provider-derived
`mask_tail_policy_boundary` facts for the existing generated-bundle path.

### Main Changes

- Added computed-mask select mask/tail policy boundary evidence to
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- The evidence now records typed-body authority, compare-produced mask role,
  tail/mask policy, mask/tail provider plan and owner, emitted compare/select
  structure, route metadata, runtime counts, and artifact ABI.
- Created and completed Trellis task
  `06-05-stage2-rvv-computed-mask-select-executable-policy-closure`.

### Git Commits

(Pending commit in this session)

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind computed_mask_select --run-id codex-cms-dryrun-20260605-policy --overwrite`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_mask_select --run-id codex-cms-ssh-20260605 --overwrite`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] New-diff old-authority scan over touched files

### Status

[OK] **Completed**

### Next Steps

- Archive the Trellis task and create the coherent commit.

---

## Session 459: Stage2 RVV contraction-dequant-clamp composition

**Date**: 2026-06-05
**Task**: `stage2-rvv-contraction-dequant-clamp-composition`
**Branch**: `main`

### Summary

Implemented a route-supported typed RVV selected-body composition foundation for
widening product/reduction -> f32 dequant -> lower/upper f32 clamp/select ->
f32 store. This is a provider/target-artifact foundation only; no executable
correctness or performance claim was made.

### Main Changes

- Added `tcrv_rvv.typed_widening_product_reduce_dequant_clamp_f32_pre_realized_body`
  with verifier checks for runtime ABI roles, dtype/config chain facts, clamp
  roles, policy, and stale authority metadata.
- Extended RVV selected-body realization, contraction route family planning,
  control-policy validation, statement planning, route construction metadata,
  runtime ABI contracts, and target artifact validation for the composed chain.
- Added positive generated-artifact lit coverage and negative diagnostics for
  missing scale, missing/swapped bounds, missing reduction facts, dtype-chain
  mismatch, unsupported policy, stale route-id authority, and stale provider
  mirrors.

### Testing

- [OK] `ninja -C build tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `lit -sv . --filter=pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32`
- [OK] `lit -sv . --filter=pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32`
- [OK] `lit -sv . --filter=pre-realized-selected-body-artifact-dequant-clamp-f32-epilogue`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `git diff --check`
- [OK] bounded old-authority/q-name scan over touched source/test files

### Status

[OK] **Completed** as route-supported plus target-validation foundation.

### Next Steps

- Executable ABI/harness and `ssh rvv` correctness evidence remain a later task
  only if the next direction asks for executable closure.


## Session 459: Stage2 RVV contraction-dequant-clamp executable ABI closure

**Date**: 2026-06-05
**Task**: Stage2 RVV contraction-dequant-clamp executable ABI closure
**Branch**: `main`

### Summary

Closed executable evidence for the pre-realized RVV widening-product reduce dequant clamp route with dry-run bundle validation and ssh rvv correctness; changes are neutral harness/evidence support only.

### Main Changes

- Repaired the stale `const float *` verifier expectation in the runtime
  scalar splat-store negative fixture.
- Added fail-closed Conversion/EmitC slices for runtime ABI operation order,
  SEW/config mismatch, and store/VL ownership mismatch.
- Strengthened pre-realized target artifact and generated-bundle dry-run
  checks for provider support, target leaf profile, required headers, and
  route type mapping mirrors.
- Archived the completed Trellis task.

### Git Commits

(No commits - planning session)

### Testing

- [OK] Focused lit for repaired runtime scalar splat-store negative fixture.
- [OK] Focused lit for runtime scalar splat-store materialization.
- [OK] Focused lit for explicit/pre-realized target artifact fixtures.
- [OK] Focused lit for pre-realized generated-bundle dry-run.
- [OK] `tianchenrv-target-artifact-export-test`.
- [OK] `tianchenrv-rvv-extension-plugin-test`.
- [EXPECTED UNRELATED FAIL] `check-tianchenrv`: 492/494 passed; remaining
  failures are `Plugin/construction-protocol-common.test` and
  `Plugin/template-extension-plugin.test`.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 460: Stage2 RVV mask/tail policy route foundation

**Date**: 2026-06-05
**Task**: Stage2 RVV mask/tail policy route foundation
**Branch**: `main`

### Summary

Implemented computed-mask select provider and target validation for explicit tail/mask policy mirrors, added focused negative coverage, archived the Trellis task.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `9b079872` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 461: Stage2 RVV runtime-strided memory-window route foundation

**Date**: 2026-06-05
**Task**: Stage2 RVV runtime-strided memory-window route foundation
**Branch**: `main`

### Summary

Implemented provider-derived strided memory route mirrors for base-memory movement, exported source/destination stride mirrors in RVV header artifacts, added focused plugin and target fixture coverage; route-supported only, no ssh rvv executable claim.

### Main Changes

- Added explicit strided memory mirrors to the RVV base-memory route provider
  plan: strided layout, source stride source, and destination stride source.
- Required the migrated base-memory statement-plan owner to consume those
  provider-plan mirrors before constructing strided load/store statement plans.
- Exported provider-derived source/destination stride mirrors through the RVV
  materialized header artifact metadata evidence table.
- Added focused plugin and target fixture coverage for strided-load/unit-store
  and unit-load/strided-store route-supported artifacts.

### Git Commits

(Recorded in the final commit for this session.)

### Testing

- [OK] `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- [OK] `cmake --build build --target tcrv-translate tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] Manual `tcrv-opt` materialization checks for both runtime-strided memory fixtures
- [OK] Manual `tcrv-opt | tcrv-translate --tcrv-export-target-header-artifact` header mirror checks for both fixtures
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Bounded old-authority/q-name scan over new source/test diff

### Status

[OK] **Completed** as route-supported plus target-validation foundation.

### Next Steps

- Executable ABI/harness and `ssh rvv` correctness evidence remain out of
  scope unless a later task asks for executable closure.


## Session 462: Stage2 RVV runtime-strided memory-window executable ABI closure

**Date**: 2026-06-05
**Task**: Stage2 RVV runtime-strided memory-window executable ABI closure
**Branch**: `main`

### Summary

Closed executable ABI evidence for the pre-realized RVV runtime-strided
base-memory routes. The generated external C ABI harness now proves read-only
source preservation in addition to host/reference output comparison,
noncontiguous memory behavior, destination sentinel preservation, and real
`ssh rvv` execution for both strided-load/unit-store and
unit-load/strided-store proof shapes.

### Main Changes

- Added source-buffer preservation checks to the generated
  `strided_load_unit_store` and `unit_load_strided_store` ABI harnesses in
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Updated the RVV plugin spec so executable base-memory route harnesses must
  verify host/reference behavior, multiple runtime counts and positive strides,
  destination sentinel preservation, and read-only source preservation without
  treating harness checks as route authority.
- Created and completed the Trellis task for this executable closure.

### Git Commits

(Recorded in the final commit for this session.)

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Generated-bundle dry-run for both pre-realized base-memory proof shapes
  with counts `1,17,257` and byte strides `4,8,12`
- [OK] `ssh rvv` generated-bundle ABI run for both proof shapes with PASS
  output and `source_preserved` markers
- [OK] Focused per-op evidence inspection for
  `base_memory_movement_boundary`, provider-derived memory forms, stride-source
  mirrors, and ordered statement-plan callees
- [OK] Bounded old-authority/q-name diff scan
- [OK] `git diff --check`

### Status

[OK] **Completed** as executable ABI closure with real RVV evidence.

### Next Steps

- None - task complete.


## Session 462: Stage2 RVV typed conversion SEW-policy route foundation

**Date**: 2026-06-05
**Task**: Stage2 RVV typed conversion SEW-policy route foundation
**Branch**: `main`

### Summary

Added provider-contract target validation coverage for typed RVV dequantization conversion facts, repaired conversion dtype-policy spec, archived the Trellis task, and kept ssh rvv out of scope because no executable correctness/performance claim was made.

### Main Changes

- Implemented target-side dequantization conversion dtype-policy validation coverage for provider-derived route facts, runtime ABI order, scale role/type/name, typed compute op, relation, intrinsic family, and stale mirror rejection.
- Repaired runtime ABI count diagnostics to report the provider-contract parameter list instead of hard-coded widening-only names.
- Updated the EmitC route spec with dequantization conversion facts and required positive/negative target validation coverage.
- Checks run: target artifact export test, RVV extension plugin test, focused tcrv-opt/translate dequant route pipeline, focused widening conversion diagnostics, git diff --check, and bounded old-authority scan.


### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 463: Stage2 RVV standalone reduction kind mirror closure

**Date**: 2026-06-05
**Task**: Stage2 RVV standalone reduction kind mirror closure
**Branch**: `main`

### Summary

Added provider-derived standalone reduction kind mirrors, target artifact stale-mirror validation, focused reduction fixture coverage, EmitC route spec update, and archived the Trellis task.

### Main Changes

- Added provider-derived `tcrv_rvv.reduction_kind` mirrors for standalone
  reduction routes and carried them through route facts, route-family plans,
  route descriptions, target validation contracts, emission-plan metadata, and
  target-header metadata.
- Tightened target artifact validation so stale standalone reduction kind
  mirrors fail closed before bundle acceptance.
- Added representative FileCheck coverage for positive standalone reduce-add
  kind mirrors and stale kind/accumulator/binding/type mirrors.
- Updated the EmitC route spec with the standalone reduction mirror contract.
- Archived the Trellis task for this round.

### Git Commits

| Hash | Message |
|------|---------|
| `final-report` | Commit hash is recorded in the final report. |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate`
- [OK] Manual FileCheck for PLAN, HEADER, STALE-REDUCTION-KIND,
  STALE-REDUCTION-ACC, STALE-REDUCTION-BINDING, and STALE-REDUCTION-TYPE
- [OK] `build/bin/tcrv-opt test/Dialect/RVV/standalone-reduction-dataflow.mlir --split-input-file --verify-diagnostics`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Bounded old-authority/q-name diff scan; only added match was a stale
  negative `artifact-name-derived-vector` test string.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 464: Stage2 RVV standalone reduce-add executable ABI closure

**Date**: 2026-06-05
**Task**: Stage2 RVV standalone reduce-add executable ABI closure
**Branch**: `main`

### Summary

Closed standalone_reduce_add executable ABI evidence with generated bundle dry-run and real ssh rvv correctness over counts 0,1,16,17,257.

### Main Changes

- Closed executable ABI evidence for the existing typed RVV
  `standalone_reduce_add` route. No compiler/source edit was needed after live
  inspection: provider facts, target validation, generated-bundle checks, and
  the external ABI harness already carried the required standalone reduction
  contract.
- Generated-bundle dry-run evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/standalone-reduce-add-dryrun/standalone_reduce_add/evidence.json`.
- Real RVV evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/standalone-reduce-add-ssh/standalone_reduce_add/evidence.json`.
- Runtime command used explicit counts `0,1,16,17,257` with seeds `-11,17`
  and patterns `0,1`.
- Final remote marker:
  `PASS op=standalone_reduce_add counts=0,1,16,17,257 seeds=-11,17 patterns=0,1`.
- Harness verified scalar add-reduction reference, `acc[0]` preservation,
  source preservation, and scalar-output sentinel preservation.
- Archived the Trellis task for this round.

### Git Commits

(Recorded in the final commit for this session.)

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate`
- [OK] Manual FileCheck for `PLAN`, `HEADER`, `STALE-REDUCTION-KIND`,
  `STALE-REDUCTION-ACC`, `STALE-REDUCTION-BINDING`, and
  `STALE-REDUCTION-TYPE`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Generated-bundle dry-run for `standalone_reduce_add`
- [OK] Real `ssh rvv` run for `standalone_reduce_add`
- [OK] Direct pre-realized route-entry negative for `standalone_reduce_add`
- [OK] `build/bin/tcrv-opt test/Dialect/RVV/standalone-reduction-dataflow.mlir --split-input-file --verify-diagnostics`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`

### Status

[OK] **Completed** as executable ABI closure with real RVV correctness
evidence. No `.trellis/spec/` update was needed because existing specs already
covered the executable contract.

### Next Steps

- None - task complete


## Session 465: Stage2 RVV computed-mask standalone reduce-add closure

**Date**: 2026-06-05
**Task**: Stage2 RVV computed-mask standalone reduce-add closure
**Branch**: `main`

### Summary

Closed computed-mask standalone reduce-add route evidence with provider-derived reduction_kind checks, stale mirror rejection, generated-bundle dry-run evidence, and ssh rvv correctness.

### Main Changes

- Created and archived Trellis task
  `06-06-stage2-rvv-dequant-clamp-f32-epilogue-executable-abi`.
- Proved the existing `dequant_clamp_f32_epilogue` generated-bundle path on
  real `ssh rvv` with runtime ABI order `lhs, scale, lower_bound, upper_bound,
  out, n`.
- Recorded executable evidence at
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260605T184905Z`.
- No production compiler or script change was required.

### Git Commits

- rvv: close dequant clamp epilogue executable abi evidence

### Testing

- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind dequant_clamp_f32_epilogue --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --dry-run`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind dequant_clamp_f32_epilogue --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-selected-body-artifact-dequant-clamp-f32-epilogue` from `build/test`
- [OK] `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter pre-realized-dequant-clamp-f32-epilogue-negative` from `build/test`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] task context validation

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 466: Stage2 RVV plain MAcc selected-body route foundation

**Date**: 2026-06-05
**Task**: `06-05-rvv-stage2-plain-macc-selected-body-route`
**Branch**: `main`

### Summary

Closed a bounded plain `macc_add` route-contract evidence gap. The existing
production RVV provider/target path already carried the plain MAcc selected
body, provider validation contract, target validation, and generated-bundle
route. This round pinned the provider-derived arithmetic kind in focused
route/generator evidence and documented the plain MAcc generated-bundle
testing contract.

### Main Changes

- Added `tcrv_rvv.macc_arithmetic_kind = add` FileCheck coverage in explicit
  and pre-realized plain MAcc emission-plan fixtures.
- Added generated-bundle evidence extraction and dry-run checks for plain
  MAcc `macc_arithmetic_kind = add` under `provider_route_facts`.
- Added the `Plain MAcc Generated-Bundle Evidence` section to the MLIR testing
  contract.
- Kept C++ provider/target route semantics unchanged.

### Git Commits

(Commit created after journal entry as part of the final task commit.)

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] focused lit from `build/test` for explicit/pre-realized plain MAcc
  target fixtures and generated-bundle dry-runs
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-05-rvv-stage2-plain-macc-selected-body-route`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] bounded added-line authority scan; hits were negative guardrail text in
  the testing spec only
- [N/A] `ssh rvv`; no route emission/runtime ABI/correctness behavior changed

### Status

[OK] **Completed** as route-contract evidence closure.

### Next Steps

- Archive the Trellis task and create one coherent commit.


## Session 467: Stage2 RVV plain MAcc executable ABI closure

**Date**: 2026-06-05
**Task**: `06-05-rvv-stage2-plain-macc-executable-abi-closure`
**Branch**: `main`

### Summary

Closed the current plain `macc_add` generated-bundle executable ABI path with
real `ssh rvv` correctness evidence for both explicit selected-body and
pre-realized selected-body routes. The generated header/source/object and
external ABI harness agree with provider-derived route facts for
`lhs,rhs,acc,out,n`, arithmetic kind `add`, accumulator/result layout, and the
plain MAcc operand-binding plan.

### Main Changes

- Aligned plain MAcc generated-bundle dry-run tests and script self-test
  boundary to the executable closure count set `0,1,16,17,257`.
- Added FileCheck coverage for `multiply_accumulate_boundary.runtime_counts`
  and generated harness `counts[]` in explicit and pre-realized dry-runs.
- Kept production C++ provider/materializer/target semantics unchanged because
  live generated-bundle and RVV evidence showed the current route is executable.

### Runtime Evidence

- Explicit artifact:
  `/tmp/tcrv-plain-macc-executable-abi/explicit-macc-add-ssh/macc_add`
- Pre-realized artifact:
  `/tmp/tcrv-plain-macc-executable-abi/pre-realized-macc-add-ssh/macc_add`
- Both remote runs passed with:
  `PASS op=macc_add counts=0,1,16,17,257 patterns=0,1`
- Harness per-case output included:
  `explicit_accumulator signed_products source_preserved tail_preserved`

### Self-Repair

- Initial manual dry-run failed because bare `llvm-readobj` was not on PATH;
  re-ran with `/usr/bin/llvm-readobj-20`.
- Initial lit update placed the `counts[]` FileCheck before the earlier
  generated function call; reordered the check to match harness source order.

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] explicit and pre-realized plain MAcc dry-runs with counts
  `0,1,16,17,257`
- [OK] focused lit from `build/test` for explicit/pre-realized plain MAcc
  generated-bundle dry-runs
- [OK] explicit real `ssh rvv` generated-bundle run for `macc_add`
- [OK] pre-realized real `ssh rvv` generated-bundle run for `macc_add`
- [OK] direct pre-realized route-entry fail-closed reproduction for `macc_add`
- [OK] `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] focused lit from `build/test` for explicit/pre-realized plain MAcc
  Target/RVV fixtures
- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-05-rvv-stage2-plain-macc-executable-abi-closure`
- [OK] `git diff --check`
- [OK] bounded added-line authority scan; no new positive legacy authority hits

### Status

[OK] **Completed** as executable ABI closure with real RVV correctness
evidence. No `.trellis/spec/` update was needed because the existing Plain
MAcc generated-bundle evidence contract already covered this behavior.

### Next Steps

- Archive the Trellis task and create one coherent commit.


## Session 466: Stage2 RVV computed-mask MAcc selected-body route foundation

**Date**: 2026-06-05
**Task**: Stage2 RVV computed-mask MAcc selected-body route foundation
**Branch**: `main`

### Summary

Closed computed-mask `macc_add` selected-body route foundation with
provider/header mirror evidence, stale mask/arithmetic fail-closed checks,
explicit and pre-realized generated-bundle dry-runs, and real `ssh rvv`
correctness for counts `0,1,16,17,257`.

### Main Changes

- Added explicit and pre-realized target fixture checks for provider-derived
  mask role/source/form, MAcc arithmetic kind, accumulator/result layout, and
  header mirrors.
- Added pre-realized target artifact stale-mirror negative checks for mask
  role, mask memory form, and MAcc arithmetic kind.
- Kept production C++ provider/materializer/target semantics unchanged because
  the current provider-owned MAcc contracts and target validation already
  support `computed_masked_macc_add`.

### Git Commits

- included-in-this-commit

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] Direct FileCheck equivalents for explicit/pre-realized computed-mask
  MAcc plan/header/realization paths and stale mask/arithmetic mirrors
- [OK] `build/bin/tcrv-opt test/Dialect/RVV/masked-macc-dataflow.mlir --split-input-file --verify-diagnostics`
- [OK] `build/bin/tcrv-opt test/Dialect/RVV/pre-realized-computed-mask-macc-negative.mlir --split-input-file --verify-diagnostics`
- [OK] Explicit and pre-realized generated-bundle dry-runs for
  `computed_masked_macc_add`
- [OK] Explicit and pre-realized real `ssh rvv` generated-bundle runs for
  `computed_masked_macc_add`, both passing
  `PASS op=computed_masked_macc_add counts=0,1,16,17,257 patterns=0,1`
- [OK] `git diff --check`
- [OK] Trellis task validation and added-line authority scan

### Status

[OK] **Completed** as executable computed-mask MAcc selected-body route
foundation. No `.trellis/spec/` update was needed because the existing RVV
plugin, EmitC-route, and testing contracts already captured the provider-owned
computed-mask MAcc behavior.

### Next Steps

- None - task complete


## Session 467: Stage2 RVV typed widening conversion route foundation

**Date**: 2026-06-06
**Task**: Stage2 RVV typed widening conversion route foundation
**Branch**: `main`

### Summary

Closed bounded widening conversion route evidence: added i16->i32 target mirror fail-closed checks, verified provider/target support, generated-bundle dry-runs, and ssh rvv correctness for supported explicit/pre-realized widening conversion modes.

### Main Changes

- Created and archived the Trellis task for the explicit
  `widening_macc_add` executable ABI closure.
- Proved the existing explicit selected-body generated bundle is executable on
  real `ssh rvv` without production compiler, script, fixture, or test changes.
- Recorded completion evidence for provider-derived ABI order, route operand
  binding, i16mf2 source facts, i32m1 accumulator/result facts, widening-MAcc
  relation, required header/type/intrinsic mirrors, and target/header
  fail-closed negatives.

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] explicit generated-bundle dry-run for counts `0,1,7,16,23,257`
- [OK] direct `FileCheck` of the explicit dry-run test `ROOT`, `WMACC`, and
  `HARNESS` prefixes
- [OK] real `ssh rvv` generated-bundle compile/run:
  `PASS op=widening_macc_add counts=0,1,7,16,23,257 patterns=0,1`
- [OK] focused explicit target fixture `PLAN` and `HEADER` FileCheck commands
- [OK] adjacent pre-realized generated-bundle dry-run regression
- [OK] direct `FileCheck` of the pre-realized dry-run test `ROOT`, `WMACC`,
  and `HARNESS` prefixes
- [OK] focused pre-realized target fixture `REALIZED`, `PLAN`, and `HEADER`
  FileCheck commands
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] bounded old-authority scan over production/source diff lines
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Trellis context validation
- [SKIP] script self-test was not required because
  `scripts/rvv_generated_bundle_abi_e2e.py` was not changed

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 468: Stage2 RVV strided memory selected-body route foundation

**Date**: 2026-06-06
**Task**: Stage2 RVV strided memory selected-body route foundation
**Branch**: `main`

### Summary

Closed bounded strided memory route evidence: repaired pre-realized target header drift, added stale target mirror fail-closed checks, verified explicit/pre-realized generated-bundle dry-runs, and passed ssh rvv correctness for strided load/store counts 0,1,16,23,257 with stride bytes 4,8.

### Main Changes

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 469: Stage2 RVV runtime scalar splat-store route repair

**Date**: 2026-06-06
**Task**: Stage2 RVV runtime scalar splat-store route repair
**Branch**: `main`

### Summary

Repaired runtime scalar splat-store negative fixture drift, added fail-closed Conversion evidence, strengthened pre-realized target/generated-bundle provider mirror checks, and archived the Trellis task.

### Main Changes

(Add details)

### Git Commits

included-in-this-commit

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 470: Stage2 RVV dequant-clamp f32 epilogue artifact foundation

**Date**: 2026-06-06
**Task**: Stage2 RVV dequant-clamp f32 epilogue artifact foundation
**Branch**: `main`

### Summary

Closed bounded dequant_clamp_f32_epilogue route-supported artifact foundation: proved existing typed body, RVV selected-body realization, provider route-family plan, target header mirror validation, stale metadata fail-closed checks, and generated-bundle dry-run evidence; no ssh rvv executable correctness claimed.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 471: Stage2 RVV dequant clamp f32 epilogue executable ABI closure

**Date**: 2026-06-06
**Task**: Stage2 RVV dequant clamp f32 epilogue executable ABI closure
**Branch**: `main`

### Summary

Closed dequant_clamp_f32_epilogue generated-bundle executable evidence on ssh rvv for counts 0,1,16,17,257 with scalar oracle, source preservation, and output tail preservation; no production code change required.

### Main Changes

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 472: Stage2 RVV explicit fused dequant-clamp executable ABI closure

**Date**: 2026-06-06
**Task**: Stage2 RVV explicit fused dequant-clamp executable ABI closure
**Branch**: `main`

### Summary

Closed explicit selected-body generated-bundle executable ABI evidence for
`widening_product_reduce_dequant_clamp_f32`. The explicit compound
`tcrv_rvv` body now enters the generated-bundle harness through plugin-local
selected-body realization, and the resulting bundle passed real `ssh rvv`
correctness with scalar oracle, signed source patterns, scale/bound coverage,
source preservation, accumulator preservation, and output tail preservation.

### Main Changes

- Added explicit fused dequant-clamp expectation support in
  `scripts/rvv_generated_bundle_abi_e2e.py`, including the explicit fixture,
  selected variant, generated C ABI function, and realization-before-emission
  bridge.
- Added evidence checks that the explicit compound body is consumed before
  provider route construction while the pre-realized body flag remains false.
- Added focused explicit generated-bundle dry-run test coverage for the
  explicit fixture/function, provider-derived route mirrors, target artifact
  facts, and harness oracle coverage.

### Git Commits

included-in-this-commit

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] explicit generated-bundle dry-run for counts `0,1,16,17,257`
- [OK] direct `FileCheck` of the new dry-run test's `ROOT`, `WPRDC`, and
  `HARNESS` prefixes
- [OK] focused explicit `tcrv-opt --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans`
- [OK] pre-realized fused generated-bundle dry-run regression
- [OK] real `ssh rvv` generated-bundle compile/run:
  `PASS op=widening_product_reduce_dequant_clamp_f32 counts=0,1,16,17,257 patterns=0,1 scales=-0.125,0.375 bound_pairs=-1.5:2.25,-8:-0.75 tolerance=1e-05`
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Trellis context validation
- [OK] bounded old-authority scan over touched files and added diff lines
- [WARN] `python3 -m lit` unavailable locally: `No module named lit`

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 473: Stage2 RVV explicit computed masked strided dot ABI closure

**Date**: 2026-06-06
**Task**: Stage2 RVV explicit computed masked strided dot ABI closure
**Branch**: `main`

### Summary

Closed explicit computed_masked_strided_input_widening_dot_reduce_add generated-bundle executable ABI evidence on ssh rvv for counts 0,1,16,17,257 with stride pairs 2:3 and 3:2, mask/input patterns, scalar oracle, inactive-lane skipping, source preservation, and tail preservation; no production code change required.

### Main Changes

- Created and archived the Trellis task for the explicit
  `computed_masked_strided_input_widening_dot_reduce_add` executable ABI
  closure.
- Proved the existing explicit selected-body generated bundle is executable on
  real `ssh rvv` without production compiler or script changes.
- Recorded completion evidence for provider-derived ABI order, route operand
  binding, compare-produced mask facts, strided source facts, inactive-lane
  zeroing, widening product/reduction facts, and target header/type mirrors.

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] explicit generated-bundle dry-run for counts `0,1,16,17,257`
- [OK] real `ssh rvv` generated-bundle compile/run:
  `PASS op=computed_masked_strided_input_widening_dot_reduce_add counts=0,1,16,17,257 stride_pairs=2:3,3:2 mask_patterns=2 input_patterns=2`
- [OK] direct `FileCheck` of the dry-run test's `ROOT`, `MSDOT`, and
  `HARNESS` prefixes
- [OK] focused explicit target fixture `PLAN` and `HEADER` FileCheck commands
- [OK] stale-mirror negatives for provider mirror, operand binding, ABI order,
  required headers, type mapping, contraction plan, widening relation,
  reduction store VL, strided load intrinsic, and masked widening product
  intrinsic
- [OK] adjacent pre-realized generated-bundle dry-run regression
- [OK] bounded old-authority scan over touched files and added diff lines
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Trellis context validation

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 474: Stage2 RVV explicit computed masked segment2 update ABI closure

**Date**: 2026-06-06
**Task**: Stage2 RVV explicit computed masked segment2 update ABI closure
**Branch**: `main`

### Summary

Closed explicit computed_masked_segment2_update_unit_load generated-bundle executable ABI evidence on ssh rvv for counts 0,1,16,17,257 with patterns 0,1, active/inactive mask lanes, segment update oracle, inactive destination preservation, source preservation, and tail preservation; no production code change required.

### Main Changes

- Created and archived the Trellis task for the explicit
  `computed_masked_segment2_update_unit_load` executable ABI closure.
- Proved the existing explicit selected-body generated bundle is executable on
  real `ssh rvv` without production compiler, script, fixture, or test changes.
- Recorded completion evidence for provider-derived ABI order, route operand
  binding, compare-produced mask facts, segment2 field/update facts,
  inactive-lane contract, required header/type mirrors, and target/header
  fail-closed negatives.

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] explicit generated-bundle dry-run for counts `0,1,16,17,257`
- [OK] direct `FileCheck` of the dry-run test `STDOUT`, `ROOT`, `CMSEG`, and
  `HARNESS` prefixes
- [OK] real `ssh rvv` generated-bundle compile/run:
  `PASS op=computed_masked_segment2_update_unit_load counts=0,1,16,17,257 patterns=0,1`
- [OK] focused explicit target fixture `PLAN` and `HEADER` FileCheck commands
- [OK] stale-mirror negatives for route id, provider mirror, operand binding,
  ABI order, required headers, type mapping, mask source, inactive-lane
  contract, field role, and update arithmetic
- [OK] adjacent pre-realized generated-bundle dry-run regression
- [OK] bounded old-authority scan over production/source diff lines
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Trellis context validation
- [SKIP] script self-test was not required because
  `scripts/rvv_generated_bundle_abi_e2e.py` was not changed

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 476: Stage2 RVV explicit computed masked indexed gather load ABI closure

**Date**: 2026-06-06
**Task**: Stage2 RVV explicit computed masked indexed gather load ABI closure
**Branch**: `main`

### Summary

Closed explicit computed_masked_indexed_gather_load_unit_store generated-bundle executable ABI evidence on ssh rvv for counts 0,1,16,17,257 and patterns 0,1; verified active/inactive compare mask lanes, noncontiguous indexed source reads, inactive old-destination passthrough, source preservation, tail preservation, provider/target facts, explicit/pre-realized FileCheck, generated-bundle dry-runs, and target/plugin C++ tests; no production code change required.

### Main Changes

- Created and archived the Trellis task for the explicit
  `computed_masked_indexed_gather_load_unit_store` executable ABI closure.
- Proved the existing explicit selected-body generated bundle is executable on
  real `ssh rvv` without production compiler, script, fixture, or test changes.
- Recorded completion evidence for provider-derived ABI order, route operand
  binding, compare-produced mask facts, runtime index source, masked indexed
  source load, old-destination passthrough, unit-stride destination store,
  inactive-lane contract, required header/type mirrors, and target/header
  fail-closed negatives.

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | (see git log) |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] explicit generated-bundle dry-run for counts `0,1,16,17,257`
- [OK] direct `FileCheck` of the explicit dry-run test `STDOUT`, `ROOT`,
  `CMIDX`, and `HARNESS` prefixes
- [OK] real `ssh rvv` generated-bundle compile/run:
  `PASS op=computed_masked_indexed_gather_load_unit_store counts=0,1,16,17,257 patterns=0,1`
- [OK] focused explicit target fixture `PLAN` and `HEADER` FileCheck commands
- [OK] focused pre-realized target fixture `REALIZED`, `PLAN`, and `HEADER`
  FileCheck commands
- [OK] adjacent pre-realized generated-bundle dry-run regression
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] bounded old-authority scan over production/source diff lines
- [OK] `git diff --check`
- [OK] `git diff --cached --check`
- [OK] Trellis context validation
- [SKIP] script self-test was not required because
  `scripts/rvv_generated_bundle_abi_e2e.py` was not changed

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 477: Stage2 RVV explicit widening-MAcc ABI closure

**Date**: 2026-06-06
**Task**: Stage2 RVV explicit widening-MAcc ABI closure
**Branch**: `main`

### Summary

Closed explicit widening_macc_add generated-bundle executable ABI evidence on ssh rvv for counts 0,1,7,16,23,257 and patterns 0,1; verified signed widening products, source preservation, accumulator preservation, tail preservation, provider/target facts, explicit/pre-realized FileCheck, and target/plugin C++ tests; no production code change required.

### Main Changes

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 478: Stage2 RVV explicit widening-dot reduce ABI closure

**Date**: 2026-06-06
**Task**: Stage2 RVV explicit widening-dot reduce executable ABI closure
**Branch**: `main`

### Summary

Closed explicit `widening_dot_reduce_add` generated-bundle executable ABI
evidence on real `ssh rvv` for counts `0,1,16,17,257` and patterns `0,1`.
Fixed the route-specific generated harness so source and accumulator snapshot
preservation is actual evidence, not an allocated-but-unused buffer. The final
remote run validated signed horizontal widening dot products, seed-added
behavior, add-only and mul-only distinguishing cases, scalar `out[0]` output,
source preservation, accumulator preservation, and output tail preservation.

### Main Changes

- Created and archived the Trellis task for the explicit
  `widening_dot_reduce_add` executable ABI closure.
- Updated `scripts/rvv_generated_bundle_abi_e2e.py` only for the plain
  `widening_dot_reduce_add` harness path:
  - compare `lhs_before`, `rhs_before`, and `acc_before` after execution;
  - free snapshot buffers on all touched return paths;
  - iterate patterns `0,1` and pass `pattern` into `run_case`;
  - print `patterns=0,1` and preservation markers in the success output.
- Updated explicit and pre-realized widening-dot reduce dry-run HARNESS
  FileCheck contracts to assert the preservation diagnostic, pattern loop, and
  `source_preserved accumulator_preserved` success marker.
- Updated `.trellis/spec/testing/mlir-testing-contract.md` with a generated
  harness snapshot-preservation contract.

### Git Commits

| Hash | Message |
|------|---------|
| `included-in-this-commit` | `rvv: close explicit widening dot reduce abi` |

### Testing

- [OK] `cmake --build build --target tcrv-opt tcrv-translate -j2`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] explicit generated-bundle dry-run for counts `0,1,16,17,257`
- [OK] direct `FileCheck` of the explicit dry-run test `ROOT`, `WDOT`, and
  `HARNESS` prefixes
- [OK] focused explicit target fixture `PLAN` and `HEADER` FileCheck commands
- [OK] adjacent pre-realized generated-bundle dry-run regression for counts
  `0,1,16,23,257`
- [OK] direct `FileCheck` of the pre-realized dry-run test `ROOT`, `WDOT`, and
  `HARNESS` prefixes
- [OK] real `ssh rvv` generated-bundle compile/run:
  `PASS op=widening_dot_reduce_add counts=0,1,16,17,257 patterns=0,1`
- [OK] remote compile evidence:
  `remote_arch=riscv64`,
  `clang_path=/usr/bin/clang`,
  `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] bounded old-authority scan over touched diff lines
- [OK] `git diff --check`
- [OK] Trellis context validation
- [WARN] `python3 -m lit` unavailable locally: `No module named lit`; direct
  RUN-equivalent commands and FileCheck prefixes were executed instead

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 478: Stage2 RVV computed-masked widening-dot selected-body realization

**Date**: 2026-06-06
**Task**: Stage2 RVV computed-masked widening-dot selected-body realization
**Branch**: `main`

### Summary

Strengthened pre-realized computed-mask widening-dot target evidence with inactive-lane zeroing checks and stale authority/mask provenance fail-closed guards; verified dry-run and ssh rvv generated-bundle execution.

### Main Changes

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 479: Stage2 RVV contraction fact contract

**Date**: 2026-06-06
**Task**: Stage2 RVV contraction fact contract
**Branch**: `main`

### Summary

Added a direct contraction provider-facts verifier before TCRVEmitCLowerableRoute construction; verified RVV plugin and target artifact tests plus focused computed-mask strided widening-dot pipelines.

### Main Changes

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
