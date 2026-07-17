# Stage2 RVV composite gather-MAcc-scatter ssh rvv runtime boundary

## Goal

Carry the just-added runtime-scalar computed-mask indexed gather + MAcc +
indexed scatter composite route from selected or pre-realized `tcrv_rvv` bodies
through the generated target artifact bundle to real `ssh rvv` execution, or
fail closed with a precise continuation point if the existing runtime boundary
is not yet executable.

This task is the runtime evidence continuation after commit `b2770dbe`, which
made the production composite route export through provider route facts,
statement planning, target artifact validation/export, and generated-bundle
dry-run evidence without claiming runtime correctness.

## What I Already Know

- The repository had no active Trellis task for this continuation, so this task
  records the Hermes direction brief before source changes.
- The current HEAD at task creation is `b2770dbe rvv: export composite gather
  macc scatter artifacts`.
- The previous archived task
  `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-composite-gather-macc-scatter-artifact-abi/`
  completed provider route facts, artifact mirrors, generated-bundle dry-run
  evidence, and a focused fail-closed check, but explicitly did not claim
  runtime correctness.
- Runtime/correctness claims for RVV require real `ssh rvv` execution evidence.
- The production route authority must remain:
  `tcrv.exec` envelope -> selected typed/realized `tcrv_rvv` body -> RVV plugin
  legality/realization/route provider -> `TCRVEmitCLowerableRoute` -> neutral
  common EmitC -> target artifact -> remote execution evidence.
- The composite route contract requires runtime ABI order
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`, provider plan
  `rt_scmp_gather_macc_scatter.v1`, runtime scalar compare, shared computed
  mask, indexed gather/scatter index use, masked MAcc, inactive lane
  preservation, and tail preservation.

## Assumptions To Validate

- The generated bundle script can already emit the composite bundle and harness
  in dry-run mode.
- Non-dry-run execution may require only harness/runtime-boundary repairs rather
  than route-provider architecture changes.
- If the route artifact is already sufficient, this task may close with focused
  evidence and no source changes beyond task bookkeeping.
- If `ssh rvv` fails because of a production seam bug in generated C/C++,
  header/prototype binding, harness construction, ABI order, mask/index binding,
  or preservation checks, the task should repair that seam in the owning module.

## Requirements

- Inspect the current generated-bundle script, composite fixtures, provider
  route planning/statement owners, target artifact validation, and previous
  task context before editing production source.
- Preserve RVV plugin ownership of operation, dtype, SEW/LMUL, policy, mask,
  memory-form, MAcc, and ABI facts. Common EmitC/export may only carry or
  validate provider-built facts.
- Produce non-dry-run generated-bundle `ssh rvv` evidence for the explicit
  selected-body composite route if executable in this round.
- Produce non-dry-run generated-bundle `ssh rvv` evidence for the pre-realized
  selected-body composite route if executable in this round.
- Evidence must cover active lanes, inactive destination preservation,
  accumulator preservation, signed products, indexed gather/scatter index use,
  runtime scalar compare, tail preservation, ABI order
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`, and provider mirror
  plan id `rt_scmp_gather_macc_scatter.v1`.
- Add or retain a focused fail-closed check for a stale or missing executable
  boundary fact, such as ABI order, runtime scalar binding, index binding, mask
  policy/source, inactive-lane preservation, header/prototype, or provider
  mirror.
- If runtime execution exposes a bug, fix the owning production seam rather
  than moving route authority into metadata, route ids, helper names, test
  names, or common EmitC semantic inference.
- Keep the validation focused; do not expand into a broad composite matrix,
  dtype/LMUL clone batch, unrelated route expansion, frontend work, or
  performance tuning.

## Acceptance Criteria

- [x] Explicit selected-body composite generated bundle executes on `ssh rvv`
  and reports route-specific correctness success, or the task records the exact
  production blocker and continuation point.
- [x] Pre-realized selected-body composite generated bundle executes on
  `ssh rvv` and reports route-specific correctness success, or the task records
  the exact production blocker and continuation point.
- [x] Runtime evidence checks active lanes, inactive destination preservation,
  accumulator/source preservation, signed products, indexed gather/scatter index
  use, runtime scalar compare, tail preservation, ABI order, and provider mirror
  plan id.
- [x] At least one focused fail-closed check rejects a stale or missing runtime
  executable-boundary fact with a targeted diagnostic.
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passes.
- [x] Script self-test is not required because
  `scripts/rvv_generated_bundle_abi_e2e.py` was not modified.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] Focused lit/generated-bundle checks for the composite route pass.
- [x] Bounded old-authority scan over touched files and added diff lines finds
  no new positive legacy route authority.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] Final `git status --short` is clean after commit if the task completes.

## Completion Notes

- No production source change was needed. The existing production
  artifact/runtime seam from `b2770dbe` already produced executable generated
  bundles for both explicit and pre-realized composite routes.
- Collected non-dry-run `ssh rvv` evidence for explicit selected-body route:
  `artifacts/tmp/stage2-composite-gms-ssh-rvv/explicit-composite-gms-ssh-rvv/evidence.json`.
- Collected non-dry-run `ssh rvv` evidence for pre-realized selected-body
  route:
  `artifacts/tmp/stage2-composite-gms-ssh-rvv/pre-composite-gms-ssh-rvv/evidence.json`.
- Both evidence runs used runtime counts `0,1,16,17,257`, runtime scalars
  `-37,91`, and patterns `0,1`. The remote harness reported active and
  inactive lane coverage, noncontiguous indexed gather/scatter lanes, signed
  product lanes, source preservation, payload/accumulator preservation, and
  tail preservation.
- Both evidence JSON roots recorded `status = success`, `ssh_evidence = true`,
  and remote compile/run success for
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`.
- Focused fail-closed evidence remains in the explicit and pre-realized target
  artifact lit tests via stale provider mirror and stale runtime ABI order
  rejection.
- Script self-test was not rerun because `scripts/rvv_generated_bundle_abi_e2e.py`
  was not modified in this task; `py_compile` and the focused dry-run lit test
  still covered the script surface used for this evidence.
- Spec update review found no new long-term executable contract to add: the
  existing RVV plugin, EmitC route, emission runtime, and testing specs already
  state the runtime evidence and composite route contracts exercised here.

## Definition Of Done

- A truthful runtime evidence artifact exists for the composite generated
  bundle path, or the task remains open with a precise blocker.
- Any source changes are limited to the generated-bundle/runtime/artifact seam
  or directly related fail-closed tests.
- Task context is updated, the task is finished/archived if complete, and one
  coherent commit records the work.

## Out Of Scope

- No broad composite matrix.
- No dtype/LMUL clone batch.
- No unrelated gather/scatter/memory route expansion.
- No high-level Linalg, Vector, StableHLO, source-front-door, or per-Linalg
  route authority.
- No performance tuning database, dashboard, index, or report-only closure.
- No common EmitC invention of RVV semantics.
- No mass rewrite of memory, segment2, reduction, compare/select, widening
  conversion, or standalone MAcc routes outside this composite runtime boundary.

## Technical Notes

- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Previous task:
  - `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-composite-gather-macc-scatter-artifact-abi/prd.md`
- Read-first source/test paths from the brief:
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-indexed-gather-macc-scatter-dry-run.test`
  - `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`
