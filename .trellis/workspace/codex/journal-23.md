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

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

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

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

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

(Add details)

### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

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
