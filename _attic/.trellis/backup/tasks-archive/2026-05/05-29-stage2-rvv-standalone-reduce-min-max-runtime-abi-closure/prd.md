# Stage2 RVV standalone reduce_min/reduce_max runtime ABI closure

## Goal

Close the plain standalone signed min/max reduction pair as one bounded
runtime-ABI and artifact evidence module. The executable path must be proven
from selected `tcrv.exec` RVV variant through typed or pre-realized
`tcrv_rvv.standalone_reduce` body facts, RVV-owned family/provider facts,
target-owned artifact validation, generated RVV C harnesses, and real `ssh rvv`
correctness evidence.

## Direction Source

- Direction title: `Switch: Stage2 RVV standalone reduce_min/reduce_max runtime
  ABI closure`.
- Module owner: selected-boundary/runtime-ABI-to-artifact path for
  `RVVSelectedBodyOperationKind::StandaloneReduceMin` and
  `RVVSelectedBodyOperationKind::StandaloneReduceMax`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `90554378 rvv: close plain segment2 runtime abi evidence`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes direction brief before source edits.
- Serial worker constraint from the brief: no subagents, spawned agents,
  parallel agents, or multi-agent workflows.

## What I Already Know

- Commit `90554378` closed the plain segment2 runtime ABI evidence task and the
  worktree began this round clean.
- Archived `05-29-stage2-rvv-vector-reduction-artifact-runtime-abi-closure`
  closed `reduce_add` with target validator, generated-bundle dry-run, `ssh rvv`
  runtime evidence, and authority scans.
- Current specs require standalone reductions to preserve a typed source/work
  channel and a typed scalar accumulator/result channel. Provider facts must
  derive seed splat, signed min/max intrinsic, result store, source/scalar
  vector type relation, runtime ABI order, and route operand binding from the
  typed body/config/runtime facts.
- Existing selected-body and pre-realized fixtures for `standalone_reduce_min`
  and `standalone_reduce_max` already materialize `tcrv_rvv.standalone_reduce`
  with `kind = "min"` / `kind = "max"` and expose provider mirrors.
- Existing `scripts/rvv_generated_bundle_abi_e2e.py` can generate dry-run
  bundles and harnesses for min/max, but the plain standalone reduction
  boundary summary is add-only, so min/max evidence does not yet expose the
  full validator-backed reduction/accumulation boundary.
- Existing min/max dry-run script test covers counts `7,16,23`, but this task
  requires counts including `0`, `1`, exact, tail, and stress cases.
- `RVVTargetArtifactRouteFamilyValidation.cpp` has a standalone
  reduction/accumulation route-family consumer, but inspection shows it mostly
  validates generic standalone reduction facts and mirrors. This round must
  harden plain min/max acceptance so stale operation kind, signed min/max
  intrinsic relation, typed compute op, memory form, source/result type policy,
  provider mirror, route operand binding, and ABI order fail closed before
  artifact acceptance.
- `test/Target/TargetArtifactExportTest.cpp` currently has focused stale-fact
  tests for vector reduction and several other families, but no focused plain
  standalone min/max target-artifact stale-fact negatives.

## Scope

- Production files may change if inspection or tests prove min/max artifact
  acceptance is stale or incomplete:
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- Expected production change from current inspection:
  - harden `RVVTargetArtifactRouteFamilyValidation.cpp` for plain standalone
    signed min/max operation/intrinsic/typed-body/memory-form/provider fact
    validation.
- Expected evidence/tooling changes:
  - `scripts/rvv_generated_bundle_abi_e2e.py`, so min/max expose the same
    reduction/accumulation boundary summary and hardware harness contracts as
    the closure requires.
  - `test/Target/TargetArtifactExportTest.cpp`, adding focused stale-fact
    negatives for both min and max.
  - directly related standalone min/max generated-bundle script tests.

## Requirements

- Plain `standalone_reduce_min` and `standalone_reduce_max` must remain
  selected-boundary/provider routes; direct pre-realized route-entry must stay
  fail-closed.
- `tcrv.exec` remains the execution/ABI envelope only. It must not define
  reduction operation, dtype, SEW/LMUL, signed min/max relation, seed role,
  source/result channel, or intrinsic spelling.
- RVV provider facts must preserve:
  - operation kind `StandaloneReduceMin` or `StandaloneReduceMax`;
  - typed compute op `tcrv_rvv.standalone_reduce`;
  - memory form `unit-stride-standalone-reduction`;
  - source role `lhs` / `lhs-input-buffer`;
  - scalar seed role `acc` / `accumulator-input-buffer`;
  - scalar result role `out` / `output-buffer`;
  - runtime count role `n` / `runtime-element-count`;
  - runtime ABI order `lhs,acc,out,n`;
  - route operand binding plan and operand summary for each min/max route;
  - provider mirror `provider_supported_mirror:rvv-standalone-reduction-plan-validated`;
  - source vector type/C type and scalar-result vector type/C type;
  - scalar accumulator layout, scalar result layout, and store VL `1`;
  - signed `vredmin` / signed `vredmax` intrinsic relation and seed splat/store
    facts derived from typed RVV provider facts.
- Target artifact validation must rebuild the provider route and reject stale
  provider facts or candidate mirrors. It must not accept route ids, artifact
  names, ABI strings, exact intrinsic spellings, script assumptions, descriptor
  residue, common EmitC inference, source-front-door state, direct route-entry
  state, pre-realized fixtures alone, or legacy i32-derived facts as authority.
- Generated harnesses must distinguish min from max and seed-dominant from
  vector-dominant cases using positive, negative, mixed-sign, tail, and empty
  counts. Runtime/correctness claims require real `ssh rvv`.

## Acceptance Criteria

- [ ] Current production selected-body/provider/target path for both plain
      standalone min and max is inspected and either changed if stale, or
      explicitly recorded as already correct.
- [ ] Target artifact validator fails closed for stale plain standalone min/max
      operation kind or signed min/max intrinsic relation.
- [ ] Target artifact validator fails closed for stale typed compute op, memory
      form, provider mirror, route operand binding, source/scalar-result type
      relation, scalar seed/result channel, runtime ABI order, and candidate
      metadata mirrors for both min and max.
- [ ] Generated-bundle explicit selected-boundary dry-runs pass for
      `standalone_reduce_min` and `standalone_reduce_max` over counts `0`, `1`,
      exact-VL, tail, and stress cases.
- [ ] Generated-bundle pre-realized selected-boundary dry-runs pass for both
      min and max over the same counts and prove `pre_realized_body_consumed:
      true`.
- [ ] Direct pre-realized route-entry remains fail-closed for both min and max,
      with no `route_entry_realization` shortcut used.
- [ ] `ssh rvv` explicit and pre-realized runs pass for both min and max over
      counts including `0`, `1`, exact-VL, tail, and stress cases, with
      positive, negative, mixed-sign, seed-dominant, vector-dominant, tail, and
      empty-count cases.
- [ ] `reduce_add` and recently closed segment2 paths have bounded
      non-regression coverage.
- [ ] Bounded touched-file authority scan finds no central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
      executable authority.
- [ ] `git diff --check` passes.
- [ ] Focused checks pass; `check-tianchenrv` passes, or an exact blocker is
      recorded.
- [ ] Trellis task status, check notes, archive state, journal, and commit are
      truthful at the end of the round.

## Non-Goals

- Do not reopen plain or computed-mask segment2 work.
- Do not reopen direct route-entry demotion.
- Do not reopen `reduce_add` closure except for bounded non-regression.
- Do not implement computed-mask or runtime-scalar masked min/max closure.
- Do not implement widening dot/MAcc/conversion, compare/select, scalar splat
  store, dtype/LMUL clone batches, high-level Linalg/frontend lowering,
  one-intrinsic wrapper dialects, dashboards, reports, or broad smoke matrices.
- Do not move RVV reduction semantics into common EmitC/export or Python
  compiler-core structures.

## Technical Approach

1. Harden the target artifact standalone reduction validator if the current
   generic checks do not explicitly prove min/max operation-kind, signed
   intrinsic, typed compute op, memory form, type policy, provider mirror, route
   operand binding, and ABI order relation.
2. Add focused C++ target-artifact stale-fact negatives for min and max.
3. Extend generated-bundle evidence so plain min/max get a full
   reduction/accumulation boundary summary, not only add.
4. Expand focused min/max dry-run tests to include `0`, `1`, exact, tail, and
   stress counts and assert explicit/pre-realized selected-boundary evidence.
5. Run direct pre-realized route-entry fail-closed checks for min and max.
6. Run `ssh rvv` explicit and pre-realized min/max evidence.
7. Run bounded non-regressions, authority scans, `git diff --check`, and
   `check-tianchenrv` if focused checks pass.

## Validation Plan

1. `cmake --build build --target TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
2. `build/bin/tianchenrv-target-artifact-export-test`
3. Focused lit/script tests for standalone min/max generated bundle dry-runs.
4. Generated-bundle explicit dry-runs:
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind standalone_reduce_min --op-kind standalone_reduce_max --dry-run --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`
5. Generated-bundle pre-realized dry-runs:
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --op-kind standalone_reduce_min --op-kind standalone_reduce_max --pre-realized-selected-body --dry-run --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`
6. Direct route-entry fail-closed dry-runs for both min and max.
7. `ssh rvv` generated-bundle runs for explicit and pre-realized min/max over
   the same counts.
8. Bounded reduce_add and segment2 non-regressions.
9. Bounded authority scan over touched RVV planning/provider/target/script/test
   files.
10. `git diff --check`
11. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- Archived context read:
  - `.trellis/tasks/archive/2026-05/05-29-stage2-rvv-vector-reduction-artifact-runtime-abi-closure/prd.md`
  - `.trellis/tasks/archive/2026-05/05-29-05-29-stage2-rvv-plain-segment2-runtime-abi-closure/prd.md`
  - `.trellis/workspace/codex/journal-18.md`
- Current inspection points:
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Target/TargetArtifactExportTest.cpp`
  - explicit/pre-realized standalone min/max target fixtures and generated
    bundle script tests.
