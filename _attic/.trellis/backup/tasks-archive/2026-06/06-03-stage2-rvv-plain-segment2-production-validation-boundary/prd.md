# Stage2 RVV plain segment2 production validation boundary

## Goal

Close the production provider-to-target validation boundary for the plain
segment2 RVV memory routes:
`segment2_interleave_unit_load` and `segment2_deinterleave_unit_store`.

Target artifact validation must consume a provider-owned plain segment2 fact
surface for runtime ABI order and parameter roles, SEW/LMUL/policy, typed
operation kind, segment lane/field roles, interleave versus deinterleave
direction, source/destination memory forms, segment2 layout, route-family
plan, operand binding/header participation, headers, C type mapping, target
leaf profile, and explicit `provider_supported_mirror` labels. It must not
reconstruct those facts from route ids, artifact names, fixture names,
candidate metadata mirrors, descriptors, common EmitC/export code, or exact
intrinsic spelling.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV plain segment2 production validation boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` returned clean short status through RTK.
* Initial `git log --oneline -8` started at
  `c61860fd rvv: validate computed masked segment2 route facts`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* The immediately previous task archived at
  `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-computed-masked-segment2-production-validation-boundary/`
  added provider-owned computed-mask segment2 route facts and rewired target
  validation to consume those facts instead of target-local duplicate truth.
* Archived plain segment2 ABI tasks already proved executable behavior and
  `ssh rvv` correctness for `segment2_interleave_unit_load` and
  `segment2_deinterleave_unit_store`. This round should not rerun hardware
  evidence unless route emission, generated bundle behavior, or runtime ABI
  semantics change.

## Current Repository Evidence To Confirm

Live inspection must confirm the current production shape before editing:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` exposes
  `RVVComputedMaskSegment2MemoryRouteFacts`, but the plain segment2 routes do
  not yet have an equivalent explicit provider-owned fact accessor.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` already has
  plain segment2 predicates and header-binding checks, but the broader plain
  segment2 validation may still encode expected route-family facts directly in
  target-local helpers.
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` already populates segment2
  route descriptions and metadata for both plain and computed-mask segment2
  forms.
* Existing explicit and pre-realized plain segment2 MLIR fixtures and
  generated-bundle script tests exist under `test/Target/RVV/` and
  `test/Scripts/`.

## Requirements

* Keep scope to `segment2_interleave_unit_load` and
  `segment2_deinterleave_unit_store`.
* Provide a production provider-owned plain segment2 route fact surface or
  equivalent accessor in the RVV provider layer.
* The fact surface must record, for each route:
  * operation and memory form;
  * runtime ABI order and exported runtime ABI parameters;
  * SEW, LMUL, tail policy, and mask policy;
  * typed compute op name;
  * segment lane/field roles and field names;
  * interleave versus deinterleave direction;
  * source and destination memory forms, including field source/destination
    forms;
  * segment count, tuple C type, and segment2 memory layout;
  * plain segment2 route-family plan;
  * route operand binding plan and exact summary with `abi` and `hdr`
    participation for every exported runtime ABI parameter;
  * required headers and C type mapping summary;
  * target leaf profile and explicit `provider_supported_mirror` label.
* Rewire target artifact validation to compare rebuilt provider descriptions
  and candidate metadata mirrors against the provider-owned facts.
* Reject stale interleave facts on deinterleave, stale deinterleave facts on
  interleave, missing segment lane facts, stale typed-compute facts, stale
  route-family plan, stale binding plan/summary, stale header/type mapping,
  stale target profile, stale provider mirror, computed-mask residue on plain
  segment2 routes, and accidental scalar, unit, strided, indexed, descriptor,
  direct-C, source-front-door, or legacy `i32m1` fallback.
* Keep common EmitC/export neutral. It may carry provider-built payloads and
  mirrors unchanged, but must not infer segment2 semantics.
* Preserve existing explicit and pre-realized generated-bundle support.

## Acceptance Criteria

* [x] Production RVV provider/planning and target validation expose and consume
      provider-owned plain segment2 route facts for interleave load and
      deinterleave store.
* [x] Target validation checks provider-derived runtime ABI order/parameters,
      SEW/LMUL/policy, typed operation, lane/field roles, direction, source and
      destination memory forms, segment2 layout, route-family plan, binding
      plan/summary, header/type mapping, target profile, and provider mirror.
* [x] Target validation rejects stale plain segment2 cross-contamination,
      missing facts, computed-mask residue on plain routes, stale candidate
      metadata mirrors, and accidental scalar/unit/strided/indexed fallback.
* [x] Focused C++ target validation tests prove the new fail-closed boundary.
* [x] Existing lit/script dry-runs for explicit and pre-realized plain
      segment2 interleave/deinterleave still pass.
* [x] No source-front-door positive route, descriptor-driven computation,
      common EmitC semantic inference, route-id authority, artifact-name
      authority, exact intrinsic spelling authority, or legacy `i32m1` route
      authority is introduced.
* [x] Focused checks, `git diff --check`, and bounded old-authority scan
      complete this round.
* [x] Trellis finish/archive and one coherent commit complete this round if
      the module behavior is complete.

## Technical Approach

1. Use `c61860fd` and the archived computed-mask segment2 task as the provider
   fact-surface pattern.
2. Add `RVVPlainSegment2MemoryRouteFacts` or an equivalent provider-owned
   accessor in the RVV route provider header/implementation.
3. Populate the facts from the same constants and typed route description
   values used by provider planning for interleave and deinterleave.
4. Rewire target plain segment2 validation to consume this accessor rather than
   target-local operation switches for route-family truth.
5. Add focused C++ mutation tests in the target artifact export test for stale
   interleave/deinterleave facts, missing lane/binding/header/type/profile
   facts, computed-mask residue, and stale candidate mirrors.
6. Run the smallest lit/script/build checks that exercise the changed surface
   plus generated-bundle dry-runs for explicit and pre-realized plain segment2
   forms.

## Out Of Scope

* Segment width greater than 2.
* Computed-mask segment2 expansion or redo.
* Indexed or strided route expansion.
* Reductions, MAcc, dot/contraction, standalone compare/select, dtype/LMUL
  clone batches, high-level frontend/source-front-door positive routes, or
  global tuning/database/dashboard work.
* Moving RVV semantics into common EmitC/export.
* Evidence-only packaging as the main deliverable.

## Evidence Plan

* Build the focused target artifact export test target and any needed RVV
  provider target.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* Run lit filters for `segment2-interleave-unit-load` and
  `segment2-deinterleave-unit-store` target fixtures.
* Run generated-bundle dry-run checks for explicit and pre-realized plain
  segment2 forms.
* Run direct fail-closed checks through C++ mutation tests for stale or missing
  plain segment2 provider facts and candidate mirrors.
* Run a bounded old-authority scan over touched files for legacy `i32m1`,
  descriptor, source-front-door/source-artifact, direct-C/source-export,
  route-id, artifact-name, and mirror-only route authority drift.
* Run `rtk git diff --check`.
* Do not rerun `ssh rvv` unless route emission, generated bundle behavior, or
  runtime ABI semantics change. If this round only tightens production
  validation, reuse archived RVV correctness evidence from the individual
  plain segment2 interleave/deinterleave ABI tasks and state that no new
  runtime/correctness claim changed.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Relevant archived tasks read:

* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-computed-masked-segment2-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-segment2-interleave-unit-load-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-segment2-deinterleave-unit-store-artifact-abi/prd.md`

Initial implementation focus:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* `test/Target/RVV/*segment2-interleave*.mlir`
* `test/Target/RVV/*segment2-deinterleave*.mlir`
* `test/Scripts/*segment2-interleave*.test`
* `test/Scripts/*segment2-deinterleave*.test`

## Completed Behavior

* Added provider-owned `RVVPlainSegment2MemoryRouteFacts` and
  `getRVVPlainSegment2MemoryRouteFacts` for
  `segment2_deinterleave_unit_store` and `segment2_interleave_unit_load`.
* Built the plain segment2 route operand binding summary from provider facts,
  including exported ABI/header `abi` and `hdr` participation for every
  runtime ABI parameter.
* Rewired target artifact validation so plain segment2 header binding,
  runtime ABI, route payload, candidate metadata mirrors, header/type mapping,
  target profile, provider mirror, lane roles, direction facts, and
  interleave/deinterleave segment facts consume the provider-owned fact
  surface instead of target-local duplicate constants.
* Added focused C++ target artifact checks proving the provider facts accessor
  matches rebuilt route descriptions and that stale plain segment2 provider
  facts, header/type facts, target profiles, route-family plans, binding
  summaries, candidate mirrors, and computed-mask residue fail closed.
* Updated the pre-realized plain segment2 interleave negative target fixture to
  match the new provider-owned plain segment2 destination-memory diagnostic.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the concrete
  plain segment2 memory fact-surface contract.

## Validation Evidence

Commands run:

```bash
rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2
rtk build/bin/tianchenrv-target-artifact-export-test
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'segment2-interleave-unit-load'   # workdir: build/test
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'segment2-deinterleave-unit-store' # workdir: build/test
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/plain-segment2-interleave-explicit-dry-run --run-id explicit-segment2-interleave-unit-load --overwrite --op-kind segment2_interleave_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/plain-segment2-interleave-pre-realized-dry-run --run-id pre-realized-segment2-interleave-unit-load --overwrite --op-kind segment2_interleave_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/plain-segment2-deinterleave-explicit-dry-run --run-id explicit-segment2-deinterleave-unit-store --overwrite --op-kind segment2_deinterleave_unit_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/plain-segment2-deinterleave-pre-realized-dry-run --run-id pre-realized-segment2-deinterleave-unit-store --overwrite --op-kind segment2_deinterleave_unit_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-03-stage2-rvv-plain-segment2-production-validation-boundary
rtk git diff -U0 -- include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp .trellis/tasks/06-03-stage2-rvv-plain-segment2-production-validation-boundary/prd.md | rtk rg -n "^[+].*(RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|__riscv_.*_i32m1|source-front-door|source-artifact|emission_plan|descriptor|selected route|route id|artifact name)"
rtk git diff -U0 -- include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp test/Target/RVV/pre-realized-selected-body-artifact-segment2-interleave-unit-load.mlir | rtk rg -n "^[+].*(RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|__riscv_.*_i32m1|source-front-door|source-artifact|emission_plan|descriptor|selected route|route id|artifact name)"
rtk git diff --check
```

`task.py validate` passed with 21 `implement.jsonl` entries and 17
`check.jsonl` entries. A rerun of the interleave lit filter initially exposed
one stale negative FileCheck diagnostic in the pre-realized interleave fixture;
the fixture was updated to match the new provider-owned plain segment2
destination-memory diagnostic, and the filter then passed. The code-only
bounded old-authority scan returned no added-line hits. The broader task/spec
scan hit only newly added negative spec language that forbids route-id,
artifact-name, descriptor, source-front-door, source-export/direct-C, and
legacy `i32m1` authority.

`ssh rvv` evidence was not rerun because this round only tightened provider
and target validation; route emission, generated bundle behavior, runtime ABI
order, and runtime semantics did not change. The prior archived
`segment2_interleave_unit_load` and `segment2_deinterleave_unit_store` tasks
remain the runtime correctness evidence for counts `0,1,16,17,257`.
