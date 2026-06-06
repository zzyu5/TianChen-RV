# Stage2 RVV widening dot-reduce executable artifact ABI boundary

## Goal

Prove, or fail closed precisely, the end-to-end executable artifact ABI boundary
for the existing pre-realized selected-body `widening_dot_reduce_add` RVV route.
The route must flow from typed `tcrv_rvv` body facts through RVV-owned provider
validation, `TCRVEmitCLowerableRoute`, neutral Common EmitC materialization,
target artifact export, generated bundle ABI, and real `ssh rvv` correctness
evidence for one bounded direct-contraction kernel.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV widening dot-reduce executable artifact ABI boundary`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `a8a01f02 rvv: own residual memory provider facts`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created from the Hermes brief before source edits.
- Local memory warns that recent Stage2 RVV work drifted into evidence-only
  commits; this round must not close by only recording evidence if the
  production artifact/ABI seam is stale or under-validated.
- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous archived task
  `.trellis/tasks/archive/2026-06/06-06-06-06-stage2-rvv-residual-memory-provider-fact-surface-closure/prd.md`
  closed residual memory provider-fact ownership and explicitly did not cover
  widening-dot artifact/export/runtime.
- `RVVEmitCRouteProvider.cpp` already obtains a direct contraction provider
  plan before constructing `TCRVEmitCLowerableRoute`, passes that plan into the
  selected statement-plan owner selection, and verifies direct contraction
  provider facts before route construction.
- `RVVEmitCContractionRouteFamilyPlanOwners.cpp` already exposes
  `getRVVWideningDotReduceRouteFacts(...)` and
  `getRVVWideningDotReduceRouteValidationContract(...)` for widening-dot
  route facts including runtime ABI order, route operand binding summary,
  source/result type facts, accumulator/result layout, scalar seed/store facts,
  required headers, type mappings, and provider-supported mirror labels.
- `RVVTargetArtifactRouteFamilyValidation.cpp` already has a
  `widening-dot-reduction` target route-family validator that consumes the
  provider validation contract, validates rebuilt route headers, type mappings,
  ABI mappings, provider-built statement shape, scalar result store VL, and
  mirror-only candidate metadata.
- Current dry-run evidence for
  `scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind widening_dot_reduce_add`
  succeeds with generated bundle/header/harness evidence, but reports
  `ssh_evidence: false`.
- The generated harness for this route validates `lhs,rhs,acc,out,n` ABI order,
  `out[0]` scalar result, untouched non-scalar/tail sentinels, untouched input
  and accumulator buffers, and multiple runtime counts including a multi-VL
  sentinel.

## Requirements

- Keep the route authority chain structural:
  typed/pre-realized `tcrv_rvv` body facts ->
  direct contraction provider facts ->
  `TCRVEmitCLowerableRoute` ->
  Common EmitC materialization ->
  target artifact export ->
  generated bundle ABI ->
  `ssh rvv` correctness evidence.
- If the current executable path works, add focused evidence or harness
  assertions only where they directly prove the production artifact/ABI seam;
  do not add a report-only or journal-only achievement.
- If the current executable path fails, make the precise missing boundary fail
  closed with a targeted diagnostic before an executable route/artifact claim.
- Preserve RVV plugin ownership of widening-dot dtype/config/runtime/header/
  intrinsic/ABI facts. Common EmitC/export must stay neutral and may only
  consume provider-built route payloads and mirror metadata.
- Keep the scope to one route:
  pre-realized selected-body `widening_dot_reduce_add`.
- Keep direct-pre-realized route-entry mode fail-closed; this task must not
  revive route-entry realization, source-front-door export, descriptor-driven
  computation, or helper-name authority.
- Ensure evidence distinguishes local dry-run bundle generation from real
  remote executable correctness. Runtime/correctness claims require
  `ssh rvv`.
- Keep task status/context truthful and do not archive until the acceptance
  criteria are met or the exact unfinished continuation point is recorded.

## Acceptance Criteria

- [x] A Trellis PRD and context files exist for this bounded task before source
      edits.
- [x] Focused production code, test, or harness changes are made if current
      execution/validation is stale, under-validated, or dry-run-only. This
      condition did not apply: current production provider and target artifact
      validators already cover the route/ABI seam, and real `ssh rvv` evidence
      passed.
- [x] If no production behavior change is needed, the conclusion is supported
      by current production validator code and real executable `ssh rvv`
      evidence, not by dry-run evidence alone.
- [x] Positive evidence proves pre-realized selected-body
      `widening_dot_reduce_add` flows through materialized selected boundary,
      emission plan, target artifact export, generated bundle compile, and
      runtime correctness.
- [x] `ssh rvv` correctness evidence passes before any runtime/correctness
      claim is made.
- [x] Focused fail-closed evidence covers at least one stale or missing
      artifact/ABI/runtime fact such as wrong runtime ABI order, missing
      provider header/intrinsic, missing scalar result channel, wrong C type,
      stale route-family validation contract, or unsupported executable route
      claim.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] The relevant generated-bundle dry-run test passes.
- [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive legacy `i32m1`, `tcrv_rvv.i32_*`, descriptor,
      source-front-door/source-export, route-id, helper-name, artifact-name, or
      mirror-only authority drift.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation, and
      clean final git status pass.
- [x] If the task is complete, finish/archive the Trellis task and create one
      coherent commit.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
- Python remains evidence tooling only; it does not implement compiler core,
  dialect, pass, provider, target export, or lowering semantics.
- RVV widening-dot route facts remain provider-owned and target artifact
  metadata remains mirror-only after route construction.
- Common EmitC/export do not infer RVV semantics, dtype, ABI order, intrinsic
  spelling, scalar result layout, or support status.
- No new route family, dtype/LMUL clone batch, high-level frontend, dashboard,
  broad generated-bundle matrix, performance tuning DB, source-front-door
  positive route, or mass rewrite of unrelated route owners is introduced.

## Technical Approach

Use the existing production chain as the candidate executable seam:

```text
pre-realized selected tcrv_rvv widening-dot body
  -> selected lowering-boundary materialization
  -> RVV direct contraction provider facts and statement owner
  -> TCRVEmitCLowerableRoute
  -> Common EmitC materialization
  -> target artifact route-family validation
  -> generated object/header bundle
  -> generated external C ABI harness
  -> ssh rvv compile/run correctness
```

First run the non-dry-run generated-bundle script for only
`widening_dot_reduce_add`. If it passes, preserve the evidence in the task
completion notes and add/adjust focused test coverage only for the stale
artifact/ABI boundary that is not already covered. If it fails, patch the
production provider/target/export/harness boundary that produced the failure,
then rerun the focused checks.

## Out Of Scope

- New route families.
- New dtype or LMUL clone batches.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Source-front-door or source-artifact positive RVV routes.
- Descriptor-driven or direct-C computation export.
- Common EmitC invention of RVV semantics.
- Direct pre-realized route-entry support.
- Performance tuning, autotuning databases, dashboards, or broad generated
  bundle matrices.
- Mass rewrite of memory, compare/select, conversion/dequantization,
  standalone reduction, MAcc, segment2, elementwise, or unrelated contraction
  owners.

## Technical Notes

- Relevant production files:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`.
- Relevant evidence/tooling files:
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-dot-reduce-add-dry-run.test`,
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-dot-reduce-add.mlir`.
- Probe artifact generated during planning:
  `artifacts/tmp/codex-stage2-wdot-abi-probe/pre-realized-widening-dot-reduce-add`.
  This is temporary evidence and should not be committed.

## Completion Evidence

- Production artifact/ABI seam inspected:
  `RVVEmitCRouteProvider.cpp` obtains direct contraction provider facts before
  `TCRVEmitCLowerableRoute` construction; `RVVEmitCContractionRouteFamilyPlanOwners.cpp`
  owns widening-dot canonical facts and validation contracts; and
  `RVVTargetArtifactRouteFamilyValidation.cpp` consumes the provider contract
  to validate rebuilt route headers, type mappings, ABI mappings, scalar result
  store VL, provider-built statement shape, and mirror-only candidate metadata.
- No production code change was required because the current production path is
  not dry-run-only or under-validated for this bounded plain
  `widening_dot_reduce_add` route. Existing target artifact tests already
  reject stale ABI order, stale binding summaries, stale store pointer/VL,
  stale seed/reduction operands, missing provider mirrors, non-family residue,
  and strided/computed-mask residue at the target route-family validator.
- Real executable evidence passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/codex-stage2-wdot-abi-ssh --run-id pre-realized-widening-dot-reduce-add --overwrite --op-kind widening_dot_reduce_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv --connect-timeout 10 --timeout 120`
  returned `rvv_generated_bundle_abi_e2e: success` and
  `PASS op=widening_dot_reduce_add counts=0,1,16,17,257 patterns=0,1`.
- The generated harness validates the exported prototype
  `void tcrv_emitc_pre_realized_body_widening_dot_reduce_add_kernel_pre_realized_body_rvv_widening_dot_reduce_add(const int16_t *lhs, const int16_t *rhs, const int32_t *acc, int32_t *out, size_t n);`,
  `out[0]` scalar result, untouched non-scalar/tail sentinels, untouched
  input/accumulator buffers, and both single-VL and multi-VL runtime counts.
- Checks run:
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  `build/bin/tianchenrv-target-artifact-export-test`;
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -v . --filter rvv-generated-bundle-abi-e2e-pre-realized-widening-dot-reduce-add-dry-run`
  from `build/test` with `PATH` including `build/bin` and
  `/usr/lib/llvm-20/bin`;
  direct dry-run script smoke;
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-stage2-rvv-widening-dot-reduce-executable-artifact-abi-boundary`.
- Spec update judgment:
  no `.trellis/spec/` change is needed. The relevant specs already state the
  provider-owned widening-dot validation contract, Common EmitC neutrality, and
  the rule that runtime/correctness claims require real `ssh rvv` evidence.
