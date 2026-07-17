# Stage2 RVV widening product-reduce dequant-clamp f32 executable artifact ABI boundary

## Goal

Make the existing widening product-reduce dequant-clamp f32 selected-body RVV
route truthfully executable as a generated RVV target artifact, or fail closed
at the exact missing executable artifact/ABI boundary. The owned boundary is
the typed `tcrv_rvv` product/reduction/dequant/clamp body facts, widening
multiply and accumulation roles, reduction accumulator/result roles, runtime
dequant scale parameter, Gearbox schedule facts, clamp predicate/select facts,
dtype/SEW/LMUL/config/policy, runtime AVL/VL, per-operand ABI/header bindings,
RVV plugin-owned route validation, common EmitC materialization, target
artifact export, generated bundle ABI, and `ssh rvv` correctness evidence.

## Why Now

Commit `5ea9bcbb` proved the standalone explicit selected-body
`dequantize_i32_to_f32` Gearbox artifact ABI with dry-run, fail-closed, and
`ssh rvv` evidence. The next bounded Stage 2 bottleneck should compose the
already-proven conversion, compare/select, and accumulation surfaces into the
more realistic product-reduction plus dequant/clamp path, while staying at the
low-level `tcrv_rvv` selected-body route boundary.

## What I Already Know

- There was no active `.trellis/.current-task`; this task was created from the
  Hermes Direction Brief.
- This is Stage 2 RVV selected-body coverage/realization work, not Stage 1
  legacy `i32m1` route-authority growth.
- The authority chain remains `tcrv.exec` selected RVV variant -> typed or
  realized `tcrv_rvv` body -> RVV plugin legality/selected-body
  realization/route provider -> `TCRVEmitCLowerableRoute` -> common EmitC ->
  target artifact -> generated bundle -> `ssh rvv` evidence when runtime
  correctness is claimed.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI/runtime
  roles only. Product input roles, reduction accumulator/result semantics,
  dequant scale semantics, clamp predicate/select semantics, dtype/config,
  runtime AVL/VL, intrinsic spelling, ABI order, route support, and generated
  C type facts must come from RVV provider facts derived from selected
  typed-body/config/runtime facts.
- Common EmitC/export may carry provider-built payloads and mirrors, but must
  not infer RVV product, reduction, dequantization, clamp, dtype, runtime AVL,
  ABI, header, or route-support semantics from route ids, metadata, helper
  names, test names, artifact names, or exact intrinsic spellings.
- The previous dequantization task found the standalone dequantize Gearbox
  seam production-valid and closed the missing executable evidence gap. This
  task must not close as a report-only or Trellis-only follow-on; it must find
  and close the exact product-reduce dequant-clamp executable blocker if one
  exists.

## Requirements

- Preserve RVV plugin ownership for widening product/reduction/dequant/clamp
  family/provider facts, dtype policy, SEW/LMUL/policy config, runtime scale
  role/type/name, Gearbox candidate/schedule facts, clamp compare/select
  roles, operand binding, statement planning, runtime AVL/VL, header/type
  summaries, and target validation contracts.
- Prove or repair the pre-realized and/or explicit selected-body
  product-reduce dequant-clamp f32 path from selected-body realization through
  provider route facts, common EmitC materialization, target artifact export,
  generated bundle compile, and `ssh rvv` correctness when executable
  behavior is claimed.
- Harden production code if the audit shows dry-run-only support, stale route
  validation, missing provider preflight, missing target validation, wrong
  product input role, wrong widening multiply fact, wrong accumulator/result
  role, wrong runtime scale binding, stale Gearbox schedule, stale clamp
  predicate/select binding, wrong ABI/header binding, wrong generated C type,
  incorrect loop/VL mapping, or metadata/route-id/common-EmitC semantic
  authority.
- Add or retain focused fail-closed evidence for at least one stale or missing
  executable-boundary fact such as product role, reduction accumulator/result,
  dequant scale/runtime binding, Gearbox schedule, clamp predicate/select,
  header/prototype binding, route-family validation contract, generated C
  type, ABI value mapping, or runtime AVL/VL.
- Keep support levels separate: parseable/verifier-legal is not
  route-supported; route-supported is not executable without complete
  ABI/runtime/export support and real `ssh rvv` evidence for runtime claims.

## Acceptance Criteria

- [x] PRD and Trellis context identify the widening product-reduce
  dequant-clamp f32 executable artifact ABI boundary, non-goals, and relevant
  specs.
- [x] Repository audit records whether current product-reduce dequant-clamp is
  dry-run-only, stale, under-validated, already production-valid, or too broad
  for a single round.
- [x] Production code is changed when the executable boundary is incomplete or
  under-validated; otherwise the PRD records a precise no-source-change
  justification backed by focused evidence.
- [x] Positive evidence covers the selected scope through selected-body
  realization or materialized selected boundary, emission plan, target
  artifact export, generated bundle compile, and `ssh rvv` correctness if
  runtime behavior is claimed.
- [x] Focused fail-closed evidence rejects at least one stale/missing
  product-reduce dequant-clamp executable-boundary fact before artifact
  acceptance or executable-route claim.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test` pass.
- [x] Relevant generated-bundle dry-run tests or underlying commands pass for
  pre-realized and/or explicit selected-body product-reduce dequant-clamp.
- [x] Bounded old-authority scan over touched files and added diff lines shows
  no new positive `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*` route
  authority, source-front-door route, descriptor compute path, or exact
  intrinsic spelling as route authority.
- [x] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are clean after commit.

## Out of Scope

- No broad product/reduction/dequant matrix, dtype/LMUL clone batch, standalone
  reduction family expansion except as bounded reference, high-level
  Linalg/Vector/StableHLO frontend, per-Linalg route authority, source-front
  door positive route, performance tuning database, dashboard, or report-only
  closeout.
- No dequant-only or clamp-only rework except as reference for this composed
  executable seam.
- No mass rewrite of memory, segment2, MAcc, compare/select, widening
  conversion, or unrelated mask routes.
- No common EmitC invention of RVV product, reduction, dequantization, clamp,
  dtype/config, Gearbox schedule, runtime AVL/VL, ABI, intrinsic spelling, or
  route support semantics.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, emission, or route semantics. Python
  may support generated-bundle tooling and evidence collection only.

## Technical Approach

1. Audit the archived dequant task, RVV plugin spec, EmitC route spec, selected
   body realization owner, contraction route family plan owners, reduction
   accumulation statement plan owners, Gearbox schedules, compare/select
   statement plan owners, route provider, target validator, generated-bundle
   script, and named dry-run/MLIR fixtures.
2. Determine whether the existing product-reduce dequant-clamp route is
   dry-run-only, under-validated, blocked by generated bundle ABI/runtime
   support, or already production-valid but missing executable evidence.
3. Patch production C++/MLIR/test/tooling only where the executable
   artifact/ABI seam is missing or stale.
4. Run focused positive dry-run and non-dry-run generated-bundle evidence for
   the selected scope, including counts that exercise zero, one, tail, and
   multi-VL behavior when runtime execution is claimed.
5. Run focused fail-closed validation for at least one stale executable-boundary
   fact.
6. Finish/archive the Trellis task and commit one coherent change.

## Technical Notes

- Read specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/index.md`
- Reference archive:
  - `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-dequantize-i32-to-f32-gearbox-artifact-abi/`
- Bounded source/test targets from the Direction Brief:
  - `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`
  - `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCReductionAccumulationStatementPlanOwners.cpp`
  - `lib/Plugin/RVV/RVVGearboxSchedules.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequant-clamp-f32-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-widening-product-reduce-dequant-clamp-f32-dry-run.test`
  - `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32.mlir`
  - `test/Target/RVV/explicit-selected-body-realization-widening-product-reduce-dequant-clamp-f32.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`
  - `test/Target/RVV/explicit-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`

## Audit Conclusion

The production widening product-reduce dequant-clamp f32 executable artifact
ABI seam already exists and is provider-owned/fail-closed. No compiler source
change was needed in this round. The exact missing blocker was executable
`ssh rvv` evidence for the composed product-reduction + dequant + clamp path,
not a production C++ validation gap.

- `RVVContractionSelectedBodyRealizationOwner.cpp` realizes both
  pre-realized and explicit
  `typed_widening_product_reduce_dequant_clamp_f32*` selected bodies into
  explicit `setvl/with_vl/load/load/widening_product/standalone_reduce/
  dequantize/splat/splat/compare/select/compare/select/store` structure before
  route planning.
- `RVVEmitCContractionRouteFamilyPlanOwners.cpp` validates the selected body
  signature, source/product/accumulator/result dtype/config, product-reduction
  chain relation, accumulator/result layout, runtime dequant scale role, lower
  and upper bound roles, clamp relation, select layout, and route operand
  binding plan for `widening_product_reduce_dequant_clamp_f32`.
- `RVVEmitCRouteProvider.cpp` adds the product vector and i32 accumulator type
  mappings and constructs the lowerable route only after selected-body route
  analysis, materialization facts, statement-plan owner selection, and provider
  route facts are available.
- `RVVTargetArtifactRouteFamilyValidation.cpp` consumes the product-reduction
  dequant-clamp validator kind before artifact export. It checks runtime ABI
  order, runtime n/AVL binding, dequant scale, lower/upper bounds, provider
  local carry, loop VL, source loads, widening product, widening reduction,
  post-loop i32-to-f32 conversion, scale, lower/upper clamp splats,
  compares/selects, final store, selected-body provenance, and route metadata
  mirrors.
- `scripts/rvv_generated_bundle_abi_e2e.py` already supports dry-run and
  non-dry-run `widening_product_reduce_dequant_clamp_f32` for explicit and
  pre-realized selected-body modes. The generated harness checks signed
  positive/negative products, source-width-overflowing widening products,
  accumulator preservation, source preservation, output-tail preservation, two
  nonzero runtime scale values, two ordered bound pairs, and host clamp-oracle
  results.
- The explicit and pre-realized lit fixtures already include focused
  fail-closed checks for stale or missing op kind, product-reduction relation,
  dtype/config, scale role/type, lower/upper bound roles, policy, legacy
  route-authority metadata, provider mirror, ABI order, operand binding,
  header declarations, C type mapping, leaf profile, accumulator/result layout,
  product type, product/reduction/dequant/scale/bound intrinsics, and clamp
  bound facts.

## Evidence

Positive generated-bundle dry-run evidence:

- Explicit selected body:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/stage2-rvv-widening-product-reduce-dequant-clamp-f32-audit --run-id explicit-wprdc-dry-run --overwrite --op-kind widening_product_reduce_dequant_clamp_f32 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/bin/llvm-readobj-20`
- Result: `rvv_generated_bundle_abi_e2e: dry_run_success`.
- Artifact root:
  `artifacts/tmp/stage2-rvv-widening-product-reduce-dequant-clamp-f32-audit/explicit-wprdc-dry-run`.
- Pre-realized selected body:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-widening-product-reduce-dequant-clamp-f32-audit --run-id pre-realized-wprdc-dry-run --overwrite --op-kind widening_product_reduce_dequant_clamp_f32 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/bin/llvm-readobj-20`
- Result: `rvv_generated_bundle_abi_e2e: dry_run_success`.
- Artifact root:
  `artifacts/tmp/stage2-rvv-widening-product-reduce-dequant-clamp-f32-audit/pre-realized-wprdc-dry-run`.

Positive executable `ssh rvv` evidence:

- Explicit selected body:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/stage2-rvv-widening-product-reduce-dequant-clamp-f32-audit --run-id explicit-wprdc-ssh --overwrite --op-kind widening_product_reduce_dequant_clamp_f32 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/bin/llvm-readobj-20`
- Result: `rvv_generated_bundle_abi_e2e: success`.
- Artifact root:
  `artifacts/tmp/stage2-rvv-widening-product-reduce-dequant-clamp-f32-audit/explicit-wprdc-ssh`.
- Pre-realized selected body:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2-rvv-widening-product-reduce-dequant-clamp-f32-audit --run-id pre-realized-wprdc-ssh --overwrite --op-kind widening_product_reduce_dequant_clamp_f32 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/bin/llvm-readobj-20`
- Result: `rvv_generated_bundle_abi_e2e: success`.
- Artifact root:
  `artifacts/tmp/stage2-rvv-widening-product-reduce-dequant-clamp-f32-audit/pre-realized-wprdc-ssh`.
- Both remote runs compiled on `ssh rvv` with `remote_arch=riscv64`,
  `/usr/bin/clang`, and `Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Both remote runs covered counts `0,1,16,17,257`, patterns `0,1`, scales
  `-0.125,0.375`, and bound pairs `-1.5:2.25,-8:-0.75`.
- Both remote outputs ended with:
  `tcrv_rvv_generated_bundle_abi_widening_product_reduce_dequant_clamp_f32_ok counts=0,1,16,17,257 patterns=0,1 scales=-0.125,0.375 bound_pairs=-1.5:2.25,-8:-0.75 tolerance=1e-05 source_preserved accumulator_preserved tail_preserved`
  and
  `PASS op=widening_product_reduce_dequant_clamp_f32 counts=0,1,16,17,257 patterns=0,1 scales=-0.125,0.375 bound_pairs=-1.5:2.25,-8:-0.75 tolerance=1e-05`.

Focused fail-closed evidence:

- Command:
  `build/bin/tcrv-opt test/Target/RVV/explicit-selected-body-realization-widening-product-reduce-dequant-clamp-f32.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | sed '0,/lower_bound=lower-bound-scalar-value:lower_bound:abi|lo|splat|cmp|sel|hdr/s//lower_bound=lower-bound-scalar-value:lower_bound:abi|lo|splat|cmp|sel/' | build/bin/tcrv-translate --tcrv-export-target-header-artifact`
- Result: failed as expected before artifact acceptance with:
  `candidate tcrv_rvv.route_operand_binding_operands provenance must mirror selected typed RVV body binding summary`.
- The diagnostic included the expected provider binding with
  `lower_bound=lower-bound-scalar-value:lower_bound:abi|lo|splat|cmp|sel|hdr`
  and rejected the stale candidate without `hdr`.

Build, test, and local checks:

- `ninja -C build tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate` passed.
- `./build/bin/tianchenrv-rvv-extension-plugin-test` passed with
  `RVV extension plugin smoke test passed`.
- `./build/bin/tianchenrv-target-artifact-export-test` passed with exit code
  `0`.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed with
  `rvv_generated_bundle_abi_e2e self-test passed`.
- `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter widening-product-reduce-dequant-clamp-f32`
  passed `4` selected tests from `build/test`.
- `git diff --check` passed.
- `git diff --cached --check` passed.

Old-authority scan:

- Current touched source set is empty; this round added only Trellis task
  files.
- A bounded scan over the added task diff for `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_`, source-front-door/source-artifact, descriptor,
  direct-C/source-export, and exact `__riscv_*_i32m1` route-authority strings
  found only PRD negative-inventory/non-goal text and the context JSONL reason
  describing common-EmitC/descriptor semantic authority as forbidden. It found
  no positive production route-authority addition.
- Existing generated evidence and fixture metadata contain provider-derived
  intrinsic spellings and the dequant relation string as mirrors after route
  construction; this round did not add a positive legacy route-authority
  surface.

## Spec Update Judgment

No `.trellis/spec/` update was needed. The current RVV plugin, EmitC route,
target artifact, and testing specs already cover the exact contract exercised
here: provider-owned selected-body facts, direct contraction owner routing,
product-reduction/dequant/clamp artifact validation, common EmitC neutrality,
generated-bundle evidence, fail-closed stale mirrors, and `ssh rvv` proof for
runtime correctness claims.

## Implementation Summary

- Created and started the Trellis task from the Direction Brief.
- Audited the relevant specs, previous dequant archive, selected-body
  realization owner, contraction family plan owners, route provider, target
  artifact validator, generated-bundle script, dry-run tests, and selected-body
  fixtures.
- Made no production source changes because the product-reduce dequant-clamp
  artifact/ABI seam was already production-valid and fail-closed.
- Produced explicit and pre-realized `ssh rvv` executable evidence for the
  generated RVV artifacts.
