# Stage2 RVV Computed-Masked Segment2 Update Executable Artifact ABI Boundary

## Goal

Prove the existing computed-masked segment2 update selected-body route is a real
generated RVV artifact boundary, not a dry-run-only or metadata-authorized
claim. The bounded owner is the path from selected/pre-realized typed
`tcrv_rvv` segment2 update body through RVV provider-owned segment2/mask/update
facts, `TCRVEmitCLowerableRoute`, common EmitC materialization, target artifact
validation, generated bundle ABI, and `ssh rvv` correctness evidence.

## What I Already Know

* The live repository is on `main`, clean, with recent runtime-scalar masked
  memory artifact contract hardening at commit `ee35e3fb`.
* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes/user Direction Brief.
* The relevant specs require RVV Stage2 executable support to start from typed
  low-level `tcrv_rvv` bodies and RVV plugin-owned route facts, with Common
  EmitC remaining neutral.
* Segment2 target export must rebuild the provider route and consume
  `RVVSegment2MemoryRouteValidationContract` before accepting route payload,
  header/type/ABI mappings, mask facts, update facts, statement steps, or
  metadata mirrors.
* Existing production code already has a segment2 target validator registry,
  computed-mask segment2 update provider contract access, route header/type/ABI
  validation, runtime AVL/VL validation, statement-plan validation, and
  metadata mirror validation.
* Existing dry-run script tests cover explicit and pre-realized
  `computed_masked_segment2_update_unit_load` evidence shape, including ABI
  order, route operand binding summary, computed-mask provenance, inactive-lane
  contract, update arithmetic, field roles, header/type mirrors, generated
  harness expectations, and source/tail preservation checks.

## Requirements

* Audit the production computed-mask segment2 update route before editing code.
* If the production seam is under-validated, harden the exact segment2 artifact
  boundary fact in C++ production code and add focused negative coverage.
* If the production seam is already contract-owned, do not invent an unrelated
  source change. Instead, justify the no-source-change path, collect non-dry-run
  generated-bundle and `ssh rvv` evidence, and add focused route-payload
  fail-closed coverage if that evidence is missing.
* Positive evidence must cover pre-realized computed-mask segment2 update
  through materialized selected boundary, emission plan, target artifact export,
  generated bundle compile, and `ssh rvv` correctness when runtime behavior is
  claimed.
* Fail-closed evidence must cover at least one stale or missing executable
  segment2 artifact-boundary fact such as route header, C type mapping, ABI
  mapping, computed-mask provenance, inactive-lane policy, source/destination
  channel, or update statement fact.
* Preserve ownership boundaries: RVV provider owns segment2/mask/update facts;
  Common EmitC only materializes provider-built route payloads; target artifact
  validation treats metadata as mirrors after provider route validation.

## Acceptance Criteria

* [x] PRD records the audit finding and exact selected contract for this round.
* [x] Existing explicit and pre-realized computed-mask segment2 update dry-run
      bundle tests still pass.
* [x] A non-dry-run generated bundle for pre-realized computed-mask segment2
      update compiles and passes correctness on `ssh rvv`.
* [x] Focused fail-closed evidence proves stale/missing route-level
      executable-boundary facts are rejected before artifact acceptance.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant generated-bundle dry-run tests pass.
* [x] Bounded old-authority scan over touched files and added diff lines shows
      no new legacy `i32m1`, descriptor, source-front-door, artifact-name, or
      exact-intrinsic-as-authority route drift.
* [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are clean after commit.

## Out of Scope

* No broad segment2 matrix.
* No dtype/LMUL clone batch.
* No high-level Linalg/Vector/StableHLO frontend.
* No per-Linalg route authority.
* No performance tuning database, dashboard, or report-only closure.
* No source-front-door positive route.
* No common EmitC invention of RVV semantics.
* No mass rewrite of unit-stride masked memory, product/dequant, contraction,
  standalone reduction, compare/select, conversion, or unrelated route families.

## Technical Notes

* Direction source: Hermes/user Direction Brief on 2026-06-06.
* Relevant specs:
  * `.trellis/spec/index.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/lowering-runtime/emitc-route.md`
* Prior task:
  * `.trellis/tasks/archive/2026-06/06-06-06-06-stage2-rvv-runtime-scalar-masked-memory-production-contract-hardening/`
* Primary production files audited:
  * `include/TianChenRV/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.h`
  * `lib/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.cpp`
  * `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`
  * `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`
  * `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  * `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  * `scripts/rvv_generated_bundle_abi_e2e.py`
  * relevant generated-bundle Script tests and Target/RVV fixtures

## Audit Finding And Selected Contract

The initial audit found that the current production path already routes
computed-mask segment2 update through the required owner chain:

```text
typed/pre-realized tcrv_rvv body
  -> segment2 route-family planning owner
  -> computed-mask memory provider facts
  -> segment2 statement-plan boundary
  -> provider-built TCRVEmitCLowerableRoute
  -> segment2 target artifact validator
  -> RVVSegment2MemoryRouteValidationContract
```

The target validator already checks the provider route id, runtime AVL/VL
contract, route-local runtime mirrors, computed-mask plan, mask role/source/form,
inactive-lane contract, segment layout, field roles/names, update arithmetic,
route headers, route type mappings, route ABI mappings, statement counts, loop
setvl, field loads, compare/mask, update arithmetic, tuple creation, masked
segment store, source provenance, and candidate metadata mirrors.

Therefore this round did not invent an unrelated production source change. The
bounded remaining executable-boundary work selected for implementation/check
was:

1. add focused target C++ route-payload negative evidence for computed-mask
   segment2 update route headers/type mappings/ABI mappings if not already
   present;
2. run non-dry-run pre-realized generated-bundle execution on `ssh rvv` and
   record the exact correctness evidence;
3. keep the dry-run script tests and focused C++ tests green.

## Evidence

* Added focused C++ route-payload negative checks in
  `test/Target/TargetArtifactExportTest.cpp` for computed-mask segment2 update:
  missing rebuilt route header `riscv_vector.h`, stale mask type mapping, and
  stale `src0` ABI value mapping now fail through the production
  `RVVSegment2MemoryRouteValidationContract` target validator before artifact
  acceptance.
* `cmake --build build --target tianchenrv-target-artifact-export-test`
  completed successfully.
* `build/bin/tianchenrv-target-artifact-export-test` passed.
* `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
  completed successfully.
* `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
* From `build/test`,
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'computed-masked-segment2-update'` passed 5 selected tests.
* Non-dry-run generated-bundle evidence passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py
  --pre-realized-selected-body --artifact-root
  artifacts/tmp/stage2-rvv-computed-masked-segment2-update-artifact-abi
  --run-id pre-realized-computed-mask-segment2-update-ssh --overwrite
  --op-kind computed_masked_segment2_update_unit_load --runtime-count 0
  --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23
  --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate
  build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
  --ssh-target rvv --timeout 240`.
* The generated bundle evidence recorded `dry_run: false`,
  `pre_realized_selected_body: true`, `ssh_evidence: true`, `status:
  success`, remote `riscv64`, remote clang `18.1.3`, and generated header
  prototype
  `void tcrv_emitc_pre_realized_body_cmseg_update_kernel_pre_realized_body_rvv_cmseg_update(const int32_t *cmp_lhs, const int32_t *cmp_rhs, const int32_t *src0, const int32_t *src1, int32_t *dst, size_t n);`.
* Runtime PASS covered counts `0,1,7,16,23,257` and patterns `0,1`, with
  active lanes, inactive lane preservation, field-distinguishing lanes,
  source preservation, and tail preservation.

## Spec Update Decision

No `.trellis/spec/` update is needed. This round implements and verifies the
already documented segment2 target export consumer contract: target validation
consumes provider-owned route validation contracts and treats metadata as
mirrors only. The new checks are focused regression evidence for an existing
contract, not a new architectural rule.

## Current Phase

Finish.
