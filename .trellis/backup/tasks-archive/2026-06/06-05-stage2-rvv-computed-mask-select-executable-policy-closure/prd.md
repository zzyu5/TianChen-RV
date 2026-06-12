# Stage2 RVV computed-mask select executable policy closure

## Goal

Close one bounded executable evidence loop for the existing Stage 2 RVV
computed-mask select route: selected `tcrv.exec` RVV variant with typed
computed-mask select `tcrv_rvv` body/config -> RVV provider-derived policy and
statement plan -> common EmitC materialization -> generated header/object
bundle -> external ABI harness -> real `ssh rvv` correctness evidence.

This round proves that mask/tail policy facts validated in `cd9eaabb` survive
generation and execution. It must not add new route coverage or move RVV
semantics into common EmitC/export or the Python harness.

## What I Already Know

- The repository starts on `main` with latest commit `cd9eaabb rvv: validate
  computed-mask select policy mirrors`; the worktree was clean before creating
  this task.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief.
- The archived task
  `.trellis/tasks/archive/2026-06/06-05-06-05-stage2-rvv-mask-tail-policy-route-foundation/`
  completed provider and target fail-closed validation for computed-mask select
  mask/tail mirrors, but explicitly did not claim runtime correctness.
- `.trellis/spec/index.md` requires RVV executable claims to follow the
  authority chain: selected typed `tcrv_rvv` body/config -> RVV plugin
  legality/realization/provider -> `TCRVEmitCLowerableRoute` -> common EmitC ->
  target artifact -> `ssh rvv` evidence.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires compare/select
  provider preflight to consume typed config, route-family plan,
  materialization facts, operand bindings, runtime AVL/VL facts, and
  mask/tail statement-plan provider facts before route construction.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires compare/select
  target artifact validation to consume the provider-owned
  `RVVCompareSelectRouteValidationContract` and embedded
  `RVVRuntimeAVLVLSelectedBoundaryContract` before accepting route payloads or
  metadata mirrors.
- `.trellis/spec/testing/mlir-testing-contract.md` requires real `ssh rvv`
  output for RVV runtime correctness claims and requires generated-bundle
  evidence for masked behavior to expose provider-derived `mask_tail_policy`
  boundary facts and inactive/tail preservation checks.
- `scripts/rvv_generated_bundle_abi_e2e.py` already supports
  `--op-kind computed_mask_select` and related typed variants, but inspection
  shows its top-level `mask_tail_policy_boundary` summary currently appears to
  be emitted for other masked routes and not yet for `computed_mask_select`.

## Requirements

- Use the existing typed computed-mask select fixture and generated route as
  authority. The selected body/config must carry compare operands, true/false
  select operands, output, `n`/AVL, predicate kind, SEW/LMUL, memory form, and
  mask/tail policy.
- Prove the policy-aware route is selected before emission by checking
  provider-derived route-family plan, mask/tail policy plan and owner,
  operand-binding summary, target leaf profile, provider support mirror, and
  runtime AVL/VL boundary.
- Generate the computed-mask select bundle in dry-run mode before any remote
  execution.
- Compile and run the generated external ABI harness on `ssh rvv` for
  executable correctness.
- Compare against a host/reference calculation over multiple runtime counts,
  including tail cases crossing VL boundaries.
- Exercise both true and false select lanes, mask boundary cases, edge values,
  and output tail sentinel preservation.
- If the script needs changes, keep them as neutral evidence/harness support:
  it may validate compiler-produced artifacts and compute a host oracle, but it
  must not choose RVV mask/tail semantics from route ids, artifact names, ABI
  strings, metadata mirrors, FileCheck labels, intrinsic spellings, or op-kind
  strings.

## Acceptance Criteria

- [x] A focused generated-bundle dry run for
      `--pre-realized-selected-body --op-kind computed_mask_select` succeeds
      and records artifact/header/source evidence.
- [x] Evidence exposes selected typed-body materialization, emitted C++ compare
      mask/select/store structure, provider-derived route metadata, runtime
      AVL/VL boundary, and `mask_tail_policy_boundary` for computed-mask
      select.
- [x] `ssh rvv` compile/run succeeds for the generated bundle and external ABI
      harness.
- [x] Runtime output or evidence JSON shows exact host/reference comparison
      over counts including tail cases, true/false lanes, mask boundary cases,
      edge values, and output tail sentinel preservation.
- [x] Focused fixture checks were not separately run because the fixture files
      were not touched and no `llvm-lit` entry was available in this build tree;
      the dry-run still exercised the relevant `tcrv-opt`/`tcrv-translate`
      pipeline on
      `test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-select.mlir`.
- [x] `tianchenrv-rvv-extension-plugin-test` and
      `tianchenrv-target-artifact-export-test` were not required because
      production C++ provider/target code did not change.
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and the
      script self-test pass if the harness changes.
- [x] Bounded old-authority scan over touched files shows no new legacy i32,
      q-name, descriptor, source-front-door, route-id, artifact-name,
      ABI-string, or intrinsic-spelling authority.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] The Trellis task status and journal accurately record the evidence and
      whether production C++ changed.

## Technical Approach

1. Run the existing computed-mask select generated-bundle dry run and inspect
   its evidence JSON for policy boundary, provider-route, runtime AVL/VL, and
   harness coverage gaps.
2. If evidence is incomplete, add the smallest neutral support to
   `scripts/rvv_generated_bundle_abi_e2e.py`, likely a computed-mask select
   branch in the mask/tail policy boundary summary and any missing harness
   checks for multi-pattern true/false lanes and tail sentinels.
3. Re-run dry-run and script self-test, then run the focused `ssh rvv`
   execution for `computed_mask_select`.
4. Run only focused lit/C++ checks required by files changed.
5. Record the resulting artifact path, command, and evidence summary in the
   task PRD/journal before finish/archive and commit.

## Out Of Scope

- New route coverage beyond computed-mask select.
- q8/q4/llama benchmark routes.
- ProviderSpec/model-name route authority.
- New contraction, dequant, clamp, MAcc, reduction, memory, segment, or dtype
  work as the primary result.
- High-level Linalg, Vector, StableHLO, or source-front-door frontend work.
- dtype/LMUL clone batches or one-intrinsic wrapper dialects.
- Compatibility wrappers preserving legacy i32m1 authority.
- Broad smoke matrices, dashboards, report-only tasks, or evidence-only
  changes that do not exercise the generated bundle and route path.
- Common EmitC/export logic that invents RVV mask semantics, tail behavior,
  dtype, schedule, policy, or body shape.

## Technical Notes

- Read first list satisfied so far:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  archived task
  `.trellis/tasks/archive/2026-06/06-05-06-05-stage2-rvv-mask-tail-policy-route-foundation/`,
  `test/Dialect/RVV/computed-mask-select-dataflow.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-select.mlir`,
  and `scripts/rvv_generated_bundle_abi_e2e.py`.
- Current likely owner is neutral evidence tooling, not production C++, unless
  focused dry-run or runtime evidence shows provider/materializer defects.
- Historical memory notes warn about recent evidence-only drift. This task is
  permitted to be a runtime/evidence closure only because the immediately
  previous commit changed production provider/target validation for this same
  route.

## Evidence Recorded

- Production C++ did not change. The only source change is neutral evidence
  support in `scripts/rvv_generated_bundle_abi_e2e.py`.
- Added computed-mask select mask/tail evidence summary support:
  `mask_tail_policy_boundary` now records provider-derived `tail_policy =
  agnostic`, `mask_policy = agnostic`,
  `mask_tail_policy_route_family_plan =
  rvv-mask-tail-policy-route-family-plan.v1`, and
  `mask_tail_policy_owner = computed-mask select mask/tail policy`.
- The new summary is built from existing verified materialized
  compare/select checks, emitted C++ checks, and object/header bundle metadata.
  It does not change compiler route construction, common EmitC materialization,
  or harness semantics.
- Dry-run command passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind computed_mask_select --run-id codex-cms-dryrun-20260605-policy --overwrite`.
- Dry-run artifact:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-cms-dryrun-20260605-policy`.
- `ssh rvv` command passed:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind computed_mask_select --run-id codex-cms-ssh-20260605 --overwrite`.
- Runtime artifact:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-cms-ssh-20260605`.
- Remote output covered counts `1,7,16,17,257` and two compare patterns. It
  reported both true and false select lanes for multi-lane counts and
  `tail_preserved` for every case, ending with
  `PASS op=computed_mask_select counts=1,7,16,17,257 compare_data_patterns=2`.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- `git diff --check` and `git diff --cached --check` passed.
- New-diff old-authority scan passed with no matches for legacy i32, q-name,
  descriptor, source-front-door, source-seed, route-id, artifact-name,
  ABI-string, or intrinsic-spelling authority.

## Spec Update Decision

No `.trellis/spec/` update is needed. The specs already require generated
mask/tail policy evidence to expose provider-derived typed body/config/runtime
facts and to treat artifact metadata as mirrors only. This round implements
that evidence reporting contract for the already-supported computed-mask select
route.
