# Stage2 RVV selected-body-to-artifact runtime ABI evidence boundary

## Goal

Close one bounded Stage 2 executable boundary for an existing contraction
selected body: `typed_widening_macc_pre_realized_body` in a selected
`tcrv.exec` RVV variant must be consumed by the RVV plugin-local selected-body
realization path into a realized `tcrv_rvv.widening_macc` body, then flow
through the RVV provider-owned `TCRVEmitCLowerableRoute`, neutral common EmitC
materialization, RVV target artifact bundle validation, and runtime ABI
correctness evidence on `ssh rvv`.

The module goal is not to add a contraction matrix. It is to repair or prove
the single missing production/export/runtime boundary for
`widening_macc_add`, preserving ABI roles `lhs,rhs,acc,out,n` and avoiding any
semantic authority from route ids, artifact names, manifests, diagnostics, C
strings, exact intrinsic spellings, descriptors, source-front-door markers, or
common EmitC.

## What I Already Know

* The Hermes Direction Brief is the source for this task.
* The repository began clean on `main` at
  `62e77a73 rvv: prove contraction realization boundary`.
* No `.trellis/.current-task` existed, so this task was created as
  `.trellis/tasks/06-01-stage2-rvv-selected-body-artifact-runtime-abi-boundary`.
* `.trellis/spec/index.md` requires the RVV-first chain:
  selected RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality / selected-body realization / route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence for runtime/correctness claims.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires target artifact
  export to rebuild and validate provider-owned route facts, with metadata as
  mirrors only.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires common EmitC to
  consume provider-built routes without choosing RVV semantics, runtime ABI,
  dtype/config, operation kind, or intrinsic mapping.
* The immediately archived contraction realization task proves that contraction
  pre-realized bodies are realized before route construction and that direct
  pre-realized route entry fails closed.
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` defines
  `typed_widening_macc_pre_realized_body` and the realized
  `tcrv_rvv.widening_macc` op.
* `scripts/rvv_generated_bundle_abi_e2e.py` already exposes
  `--pre-realized-selected-body --op-kind widening_macc_add`, verifies
  selected-boundary materialization, validates generated bundle metadata, and
  can run an external ABI harness on `ssh rvv`.
* The same script treats `--direct-pre-realized-route-entry` as retired and
  fail-closed for selected pre-realized op kinds. This task follows that
  current RVV plugin boundary and the Direction Brief, not the older direct
  route-entry positive wording that still appears in part of the testing spec.

## Requirements

* Use exactly one positive contraction-selected body:
  `typed_widening_macc_pre_realized_body` / `widening_macc_add`.
* The positive path must be:
  selected `tcrv.exec` RVV variant -> RVV selected-body realization ->
  realized `tcrv_rvv.widening_macc` body -> RVV provider-owned route facts ->
  `TCRVEmitCLowerableRoute` -> common EmitC materialization -> RVV target
  artifact bundle -> external runtime ABI harness.
* The generated bundle must preserve provider-derived runtime ABI order and
  roles for `lhs`, `rhs`, `acc`, `out`, and `n`.
* The target artifact validator must consume provider-derived
  widening-MAcc/contraction facts and treat candidate metadata as mirror-only.
* Direct pre-realized route/artifact export must fail closed for this selected
  body before bundle generation.
* If a real runtime/correctness claim is made, collect `ssh rvv` compile/run
  evidence across multiple runtime counts including empty, one-vector, tail,
  and non-one-vector cases.
* If `ssh rvv` is unavailable for non-semantic infrastructure reasons, record
  the exact blocker and do not claim runtime correctness.
* Keep common EmitC/export neutral. Do not add semantic RVV inference to common
  materialization or target bridge code.

## Acceptance Criteria

* [x] `implement.jsonl` and `check.jsonl` contain the relevant spec and prior
  task context.
* [x] Focused evidence shows the pre-realized selected body is consumed by RVV
  selected-body realization before provider route construction.
* [x] A generated bundle or target artifact test validates that `lhs`, `rhs`,
  `acc`, `out`, and `n` survive as provider-derived runtime ABI roles into the
  exported bundle.
* [x] Direct pre-realized route/artifact export for `widening_macc_add` fails
  closed with the retired direct route-entry diagnostic.
* [x] If production/export/runtime validation is missing or stale, repair the
  narrow owner: RVV contraction realization, RVV EmitC contraction plan/provider,
  common materializer neutrality, target artifact validator, target support
  bundle, or evidence script as required by repository evidence.
* [x] `ssh rvv` compile/run correctness evidence is collected for
  `widening_macc_add` pre-realized selected-body mode, or one exact
  non-semantic infrastructure blocker is recorded.
* [x] A bounded authority scan over touched plugin/provider/materializer/target/
  script/test/spec files classifies remaining hits for `RVVI32M1`,
  `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
  `emission_plan`, `descriptor`, and `selected route`.
* [x] Focused build/test/script checks for the touched path pass.
* [x] `git diff --check` passes. Final git status is intentionally not clean
  before handoff because the repo finish skill forbids this worker from
  creating the requested commit.

## Completion Notes

* The existing production path already carried the chosen pre-realized
  `widening_macc_add` selected body through RVV selected-body realization,
  provider-owned contraction route facts, `TCRVEmitCLowerableRoute`, common
  EmitC, target artifact bundle validation, and an external runtime ABI harness.
* Final dry-run evidence:
  `artifacts/tmp/06-01-stage2-rvv-selected-body-artifact-runtime-abi-boundary/final-dry-run/20260601T133109Z`.
  The materialized selected body records `pre_realized_body_consumed: true`,
  `typed_compute_op: tcrv_rvv.widening_macc`, runtime AVL source
  `runtime_abi:n`, and runtime ABI order `lhs,rhs,acc,out,n`.
* Final direct route-entry negative evidence:
  `artifacts/tmp/06-01-stage2-rvv-selected-body-artifact-runtime-abi-boundary/direct-fail-final/20260601T133110Z`.
  The command exited with status 1 and reported that
  `--direct-pre-realized-route-entry` is unsupported for
  `widening_macc_add` because the direct shortcut is retired.
* Final `ssh rvv` evidence:
  `artifacts/tmp/06-01-stage2-rvv-selected-body-artifact-runtime-abi-boundary/final-ssh-rvv/20260601T133133Z`.
  The generated bundle compiled and ran on `ssh rvv` for counts
  `0,1,16,17,257`, with signed widening product accumulation and tail
  preservation checks passing.
* Added a script self-test regression for the exact selected
  `widening_macc_add` direct-pre-realized fail-closed diagnostic.
* Updated `.trellis/spec/testing/mlir-testing-contract.md` to align the
  generated-bundle evidence contract with current RVV plugin behavior:
  selected-boundary materialization is the positive pre-realized path, while
  direct pre-realized route entry is a negative fail-closed mode.
* Bounded changed-line authority scan found no new source-code old-authority
  hits. The only new scan hit is the testing spec's negative requirement for
  absence of `descriptor` / direct-C / source-export / source-front-door
  authority. PRD hits are task guardrails.

## Checks

* [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-stage2-rvv-selected-body-artifact-runtime-abi-boundary`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* [x] `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* [x] `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind widening_macc_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-selected-body-artifact-runtime-abi-boundary/final-dry-run`
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --op-kind widening_macc_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-selected-body-artifact-runtime-abi-boundary/direct-fail-final` exited 1 with the expected retired direct route-entry diagnostic.
* [x] `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind widening_macc_add --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --artifact-root artifacts/tmp/06-01-stage2-rvv-selected-body-artifact-runtime-abi-boundary/final-ssh-rvv`
* [x] Bounded changed-line authority scan over touched script/spec/task files.
* [x] `rtk git diff --check`
* [x] `rtk cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passed 465/465.

## Definition of Done

* The bounded `widening_macc_add` pre-realized selected-body-to-generated-bundle
  path is either repaired or proven already implemented with focused evidence.
* Runtime correctness is claimed only with real `ssh rvv` evidence.
* Trellis status, context files, workspace journal, archive state, and commit
  truthfully record the completed work or the exact continuation point.

## Out of Scope

* New contraction families or broad route-family matrices.
* High-level Linalg, matmul, tensor frontend, or source-front-door authority.
* Dtype/LMUL clone batches, dashboards, broad smoke matrices, performance
  claims, or global tuning/profile systems.
* Common EmitC semantic inference, descriptor-driven computation, direct-C or
  source-export paths, or positive legacy `i32m1` routes.
* Treating route ids, artifact names, manifests, test names, C strings, or
  exact intrinsic spellings as operation, dtype, layout, policy, route, ABI, or
  artifact authority.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/testing/mlir-testing-contract.md`.
* Read: archived task
  `.trellis/tasks/archive/2026-06/06-01-06-01-stage2-rvv-contraction-realization-boundary/prd.md`.
* Primary files to inspect or repair if evidence exposes a gap:
  `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `scripts/rvv_remote_probe.py`, and focused tests.
