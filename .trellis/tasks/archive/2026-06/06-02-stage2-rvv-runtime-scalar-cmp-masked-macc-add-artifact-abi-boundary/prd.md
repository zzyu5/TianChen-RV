# Stage2 RVV runtime-scalar-cmp masked macc-add artifact ABI boundary

## Goal

Make one bounded Stage 2 RVV end-to-end artifact/runtime ABI path executable
for an existing `runtime_scalar_cmp_masked_macc_add` selected body. The path
must prove that a runtime scalar ABI value is structurally splatted into the
compare mask producer while vector-vector MAcc payload, accumulator, output,
and runtime control facts survive RVV plugin realization, route planning,
common EmitC materialization, target artifact validation, generated bundle
export, and `ssh rvv` correctness evidence.

## Direction Source

This PRD is derived from the Hermes Direction Brief:

`Stage2 RVV runtime-scalar-cmp masked macc-add artifact ABI boundary`

## What I Already Know

* The previous completed task proved `computed_masked_macc_add` with a vector
  compare-produced mask, vector-vector MAcc, target-artifact validation,
  generated-bundle harness support, stale/fail-closed checks, and `ssh rvv`
  evidence.
* This task targets the next MAcc bottleneck: the compare RHS is a runtime
  scalar ABI value that must be splatted structurally before compare.
* The selected path is:
  `selected tcrv.exec RVV variant -> RVV plugin realized tcrv_rvv
  load+splat+compare+masked_macc+store body -> route-family provider facts ->
  TCRVEmitCLowerableRoute -> common EmitC materialization -> RVV target
  artifact bundle -> ssh rvv correctness evidence`.
* The module owner includes RVV plugin MAcc realization/route planning,
  target artifact validation, generated bundle ABI e2e tooling, remote probe
  support, and focused tests only.

## Assumptions To Validate

* The repository already contains fixtures named around
  `runtime-scalar-cmp-masked-macc-add`; the round should fill only missing
  production validation, route/provider/target consumers, generated bundle
  support, fail-closed checks, or evidence.
* Existing computed-masked, plain, and scalar-broadcast MAcc tests should be
  reused only as patterns. They are not part of this task's completion scope.
* Runtime correctness and ABI behavior can be tested with the existing
  generated bundle script and `ssh rvv` route after focused updates.

## Requirements

* The RVV plugin must realize or validate the selected
  `runtime_scalar_cmp_masked_macc_add` body before route construction.
* Route planning/provider facts must carry, at minimum:
  `cmp_lhs`, `rhs_scalar`, `lhs`, `rhs`, `acc`, `out`, and `n` ABI roles.
* `rhs_scalar` must remain a typed runtime scalar ABI value and compare RHS
  splat source. It must not become a name-derived route hint, artifact hint,
  descriptor residue, or mirror-only fact.
* Predicate kind, compare mask source, lhs/rhs payload loads, accumulator
  load/pass-through, masked MAcc add, inactive-lane accumulator preservation,
  output store, runtime `n`/AVL, SEW/LMUL/policy, accumulation plan,
  provider-supported mirror, C type mapping, required headers, and exported
  header/prototype facts must survive structurally.
* Target artifact validation must actively consume the provider-derived facts
  and fail closed on stale or missing provider mirror, binding plan, ABI order,
  RHS scalar/splat mask source, header, type mapping, accumulation plan,
  accumulator layout, inactive-lane contract, or operand-binding facts.
* Generated bundle ABI e2e support must prove the exported header/prototype and
  runtime argument order for this route family.
* Runtime evidence must use real `ssh rvv` compile/run correctness when
  claiming executable behavior.

## Acceptance Criteria

* [x] Focused route/provider tests show the selected body is realized or
      validated before route construction by RVV plugin-local owners.
* [x] Focused EmitC/export and target-artifact tests show
      `cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n` ABI order and operand binding
      facts survive into route diagnostics, generated header/prototype, and
      target artifact validation.
* [x] Tests show `rhs_scalar` is structurally splatted into the compare RHS,
      not inferred from route id, artifact name, test name, C string, manifest,
      or descriptor metadata.
* [x] Tests show masked MAcc facts, inactive-lane accumulator preservation,
      runtime `n`/AVL, C type mapping, required headers, accumulation plan, and
      provider-supported mirror validation survive into the generated bundle.
* [x] Fail-closed evidence covers direct pre-realized shortcut or stale/missing
      provider mirror, binding plan, ABI order, RHS scalar/splat mask source,
      header, type mapping, accumulation plan, accumulator layout,
      inactive-lane contract, or operand-binding facts as applicable to the
      changed production path.
* [x] Generated bundle e2e dry-run tests cover explicit selected-body and
      pre-realized selected-body forms for this route family.
* [x] Real `ssh rvv` evidence covers counts `0`, `1`, one VL-boundary count,
      one tail count, and one larger count, with at least two scalar/mask and
      lhs/rhs/accumulator patterns proving scalar compare behavior,
      vector-vector multiply, accumulator contribution, inactive-lane
      accumulator preservation, output tail sentinel preservation, source
      preservation, and runtime `n` behavior.
* [x] A bounded old-authority scan over touched files finds no new production
      reliance on legacy `i32m1` route authority, descriptor-driven compute,
      source-front-door positive route authority, or common EmitC RVV semantics.
* [x] Focused build/test/script commands pass and `git status --short` is
      clean before final report.
* [x] The task is finished/archived and one coherent commit is created if the
      acceptance criteria are met.

## Completion Evidence

* Added a runtime-scalar computed-mask MAcc generated-bundle harness pattern
  dimension. The harness now runs every runtime count through rhs scalar values
  `-37` and `91` and data patterns `0` and `1`, so scalar/splat compare facts
  and lhs/rhs/accumulator payload facts are independently exercised.
* Added a target artifact validator regression in
  `TargetArtifactExportTest.cpp` that mutates the
  `runtime_scalar_cmp_masked_macc_add` route operand-binding summary by
  removing the `rhs_scalar` `hdr` marker. The provider-facts validation now
  proves exact `rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr`
  consumption before artifact acceptance.
* Updated explicit and pre-realized generated-bundle dry-run tests to require
  counts `0,1,16,17,257`, rhs scalars `-37,91`, and `patterns=0,1`, including
  harness FileCheck coverage for the generated function call and expected
  `acc + lhs * rhs` expression under `cmp_lhs <= rhs_scalar`.
* Direct pre-realized route-entry remains fail-closed for
  `runtime_scalar_cmp_masked_macc_add` with the same runtime evidence counts.
* Real `ssh rvv` compile/run correctness passed for explicit and pre-realized
  selected-body generated bundles. Evidence artifacts were written under:
  `/tmp/tcrv-rt-scalar-macc-explicit-rvv.artifacts/runtime-scalar-cmp-masked-macc-add`
  and
  `/tmp/tcrv-rt-scalar-macc-pre-rvv.artifacts/pre-realized-runtime-scalar-cmp-masked-macc-add`.
  Both runs reported `ssh_evidence=true`, `status=success`, runtime counts
  `[0,1,16,17,257]`, rhs scalar values `[-37,91]`, remote `riscv64`, clang
  `Ubuntu clang version 18.1.3`, active/inactive lane coverage,
  inactive accumulator preservation, add-only/mul-only distinction, and tail
  preservation, ending with
  `PASS op=runtime_scalar_cmp_masked_macc_add counts=0,1,16,17,257 rhs_scalars=-37,91 patterns=0,1`.
* Updated `.trellis/spec/testing/mlir-testing-contract.md` to make the
  runtime-scalar computed-mask MAcc scalar-value plus data-pattern harness
  requirement durable.
* Focused checks run:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`;
  explicit and pre-realized dry-run script commands with manual FileCheck for
  ROOT/MACC/HARNESS prefixes;
  direct pre-realized fail-closed script command with FileCheck;
  `cmake --build build --target tianchenrv-target-artifact-export-test`;
  `./build/bin/tianchenrv-target-artifact-export-test`;
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`;
  explicit and pre-realized real `ssh rvv` generated-bundle runs; and
  `git diff --check`.
* Bounded added-line authority scan found only the newly added negative
  `implicit-check-not` guards for `descriptor`, `direct-C`, and
  `source-export`; no new positive legacy authority was added.

## Out Of Scope

* LMUL `m2`, widening MAcc, reductions, matmul, more arithmetic kinds, dtype
  clone batches, high-level frontend lowering, source-front-door positive
  routes, tuning/performance databases, and broad Stage 2 coverage expansion.
* Reworking computed-masked, plain, or scalar-broadcast MAcc except to reuse
  validation lessons in this bounded runtime-scalar compare task.
* Moving RVV semantics into common EmitC/export or treating route ids, artifact
  names, manifests, tests, descriptors, exact intrinsic spellings, or mirror
  metadata as authority.

## Technical Notes

Read first:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-macc-add-artifact-abi-boundary/`
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `scripts/rvv_remote_probe.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-macc-add-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-macc-add-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-macc-add-fail-closed.test`

## Current Phase

Finish / archive. Source implementation, focused evidence, and quality checks
are complete; this task is ready for archive and one coherent commit.
