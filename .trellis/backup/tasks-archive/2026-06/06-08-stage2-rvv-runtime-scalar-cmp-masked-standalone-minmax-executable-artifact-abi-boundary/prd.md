# Stage2 RVV runtime-scalar-cmp masked standalone minmax executable artifact ABI boundary

## Goal

Make or prove the production workflow for pre-realized
`runtime_scalar_cmp_masked_standalone_reduce_min` and
`runtime_scalar_cmp_masked_standalone_reduce_max` selected RVV bodies from typed
`tcrv_rvv` facts through RVV-owned realization, route planning,
`TCRVEmitCLowerableRoute`, common EmitC materialization, target artifact export,
generated bundle ABI, and real `ssh rvv` correctness evidence. If current HEAD
already owns the seam correctly, this task may finish without production source
changes only after focused positive, fail-closed, and executable evidence proves
the boundary.

## What I Already Know

* The Hermes direction selects this as the next bottleneck after runtime-scalar
  computed-mask segment2 load/store evidence converged in commit `4ccf4ec6`.
* `.trellis/spec/index.md` keeps the RVV-first authority chain at selected
  typed `tcrv_rvv` body -> RVV plugin legality/realization/provider -> common
  EmitC -> target artifact -> `ssh rvv` evidence when runtime correctness is
  claimed.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires standalone
  reduction route metadata to carry provider-derived reduction kind,
  accumulator/result layout, reduction store VL, route operand binding plan,
  route operand binding operands, and runtime ABI/header facts.
* The same spec requires standalone reduction facts for add/min/max, including
  runtime-scalar computed-mask variants with `rhs_scalar`, runtime-scalar mask
  producer facts, operation-specific inactive neutral literals, seed/result
  layout, header/type mapping, target leaf profile, and provider-supported
  mirror.
* `.trellis/spec/testing/mlir-testing-contract.md` requires generated-bundle
  evidence for computed-mask standalone reductions to check reduction kind,
  compare predicate, runtime ABI order, inactive neutral behavior, seed/result
  oracle, runtime AVL/VL, generated C/C++ shape, and `ssh rvv` compile/run when
  runtime correctness is claimed.
* The archived segment2 task shows the expected closure pattern: bounded PRD,
  focused dry-run/lit/C++ checks, generated bundle execution on `ssh rvv`, old
  authority scan, clean status, archive, and one coherent commit.
* Current HEAD already contains dry-run script tests for pre-realized
  runtime-scalar-cmp masked standalone min/max, including LMUL m2 references,
  and a fail-closed test for direct pre-realized route-entry shortcut use.
* Current HEAD has target fixtures for pre-realized runtime-scalar-cmp masked
  standalone reduce min/max that check realization to `masked_standalone_reduce`,
  emission plan facts, route operand binding, runtime ABI order, and header
  prototype.

## Assumptions

* Base scope is SEW32 LMUL m1 min/max. LMUL m2 may be included only if it is
  already part of the same coherent seam and the same evidence run covers it
  without broad matrix expansion.
* If source code changes are needed, they should be local to the standalone
  reduction realization/provider/statement-plan/target-validation/generated
  bundle seam.
* If no source changes are needed, the task still must record concrete evidence
  and an exact no-source-change justification rather than closing as a
  Trellis-only note.

## Requirements

* Keep authority in the typed selected `tcrv_rvv` body, RVV provider facts, and
  target validator. Do not infer reduction kind, runtime scalar binding,
  inactive neutral policy, ABI order, or route support from route ids, artifact
  names, helper names, test names, metadata mirrors, or common EmitC code.
* Verify or repair pre-realized runtime-scalar-cmp masked standalone min/max
  selected-body realization so runtime ABI operands bind as
  `cmp_lhs,rhs_scalar,src,acc,out,n`, compare uses the runtime scalar splat,
  and the realized body carries `masked_standalone_reduce` with `kind = min` or
  `kind = max`.
* Verify or repair RVV route/provider facts so min/max carry provider-derived
  reduction kind, operation-specific inactive neutral literal, inactive-lane
  neutral requirement, scalar seed/result layout, runtime AVL/VL, type/header
  mapping, target leaf profile, provider-supported mirror, and full operand
  binding summary.
* Verify or repair target artifact validation so stale/missing runtime scalar,
  reduction kind, inactive neutral, seed/result, ABI/header, type mapping, or
  statement facts fail closed before target artifact acceptance.
* Verify or repair generated bundle ABI evidence so dry-run output checks
  provider facts and generated C/C++ shape, and non-dry-run execution on
  `ssh rvv` checks mixed active/inactive lanes, tail preservation, source
  preservation, all-inactive seed preservation, and min/max expected results.
* Keep common EmitC/export neutral: it may carry provider payloads but must not
  choose RVV semantics, infer dtype/config, or synthesize runtime ABI.

## Acceptance Criteria

* [x] PRD, `implement.jsonl`, and `check.jsonl` describe the bounded module
      owner and relevant specs/files.
* [x] Focused dry-run generated-bundle evidence passes for pre-realized
      `runtime_scalar_cmp_masked_standalone_reduce_min` and
      `runtime_scalar_cmp_masked_standalone_reduce_max`; LMUL m2 is included
      only if it remains the same seam.
* [x] Focused fail-closed evidence rejects at least one stale or unsupported
      executable-boundary shortcut/fact, such as direct pre-realized route-entry
      shortcut or stale runtime-scalar/reduction/inactive-neutral facts.
* [x] Focused target/RVV lit evidence checks materialized selected boundary,
      emission plan, target header artifact, runtime ABI order, route operand
      binding, reduction kind, inactive neutral policy, and header prototype.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Generated bundle execution on `ssh rvv` passes before claiming runtime
      correctness, with counts including `0`, tail cases, and mixed
      active/inactive masks for min and max.
* [x] Bounded old-authority scan over touched files and added diff lines finds
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C, source-front-door, or
      mirror-only authority path.
* [x] `git diff --check`, `git diff --cached --check`, and focused Trellis task
      validation pass.
* [x] Task status, journal, archive state, commit, and final worktree status are
      truthful.

## Out Of Scope

* No broad reduction matrix beyond the base min/max seam.
* No dtype/LMUL clone batch; LMUL m2 only if already coherent and low risk.
* No segment2, indexed, strided, product-reduction, dequant, clamp, MAcc,
  contraction, compare/select, widening conversion, or unrelated route rework.
* No high-level Linalg/Vector/StableHLO frontend work.
* No source-front-door positive route and no descriptor-driven/source-export
  route authority.
* No common EmitC semantic invention for RVV reduction behavior.

## Technical Notes

Read and constrain against:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-memory-executable-artifact-abi-boundary/`
* `include/TianChenRV/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.h`
* `include/TianChenRV/Plugin/RVV/RVVReductionSelectedBodyRealizationOwner.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/RVVStandaloneReductionSelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCReductionAccumulationStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-standalone-minmax-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-standalone-minmax-fail-closed.test`
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-min.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-max.mlir`

Initial code inspection shows the standalone reduction realization owner
already validates runtime-scalar min/max op kinds, runtime scalar ABI role,
SEW-derived ABI types, runtime AVL/VL, compare-produced mask facts, and
pre-realized/realized body exclusivity. Target artifact validation already has
runtime-scalar computed-mask standalone reduction consumers that require
provider-built loop setvl, compare lhs load, RHS scalar splat, source load,
compare, inactive neutral splat, merge, scalar seed, reduction, and scalar
result store statements.

## Completion Evidence

No production source change was required. The existing production seam already
derives runtime-scalar standalone min/max executable facts from the typed
selected RVV body and validates them through RVV provider planning, target
artifact validation, generated bundle ABI, and `ssh rvv` execution.

Checks run:

* [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-08-stage2-rvv-runtime-scalar-cmp-masked-standalone-minmax-executable-artifact-abi-boundary`
* [x] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'runtime-scalar-cmp-masked-standalone-minmax'` from `build/test`: 2/2 passed.
* [x] `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-(min|max)'` from `build/test`: 4/4 passed.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
* [x] `build/bin/tianchenrv-target-artifact-export-test`
* [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [x] Direct generated-bundle dry-run for pre-realized runtime-scalar-cmp masked standalone min/max m1 and m2:
      `rvv_generated_bundle_abi_e2e: dry_run_success`.
* [x] Non-dry-run generated bundle execution on `ssh rvv` for
      `runtime_scalar_cmp_masked_standalone_reduce_min`,
      `runtime_scalar_cmp_masked_standalone_reduce_min_lmul_m2`,
      `runtime_scalar_cmp_masked_standalone_reduce_max`, and
      `runtime_scalar_cmp_masked_standalone_reduce_max_lmul_m2`: all reported
      `PASS`, `remote_compile_succeeded: true`, `remote_run_succeeded: true`,
      `dry_run: false`, and `ssh_evidence: true`.
* [x] Evidence JSON spot checks found provider-derived
      `runtime-scalar-splat-compare-rhs`, `cmp_lhs,rhs_scalar,src,acc,out,n`,
      `reduction_kind` min/max, `masked-standalone-reduction-neutral-inactive-lanes-before-reduction`,
      all-inactive seed preservation, source preservation, and tail preservation.
* [x] `git diff --check`
* [x] Bounded task-directory old-authority scan found only negative/no-authority
      PRD language and no source-file diff.
