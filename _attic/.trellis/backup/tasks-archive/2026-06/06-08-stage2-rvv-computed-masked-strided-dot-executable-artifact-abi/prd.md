# Stage2 RVV computed-masked strided dot executable artifact ABI boundary

## Goal

Prove the existing
`computed_masked_strided_input_widening_dot_reduce_add` route family as a real
generated RVV artifact ABI workflow after the accepted provider-owned semantic
contract, or fail it closed at the RVV plugin / target artifact boundary if any
required executable artifact fact is missing or stale.

The focused production path is:

```text
selected or pre-realized tcrv_rvv body
  -> RVV provider-owned contract facts
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target object/header bundle
  -> generated C ABI consumer
  -> ssh rvv correctness evidence
```

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* Repository inspection started from clean `main` at
  `1bf69314 rvv: accept masked strided dot semantics`.
* The immediately previous archived task
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-computed-masked-strided-dot-acceptance-boundary/`
  accepted the computed-mask strided widening-dot semantics in production
  provider/target code.
* That previous task added the named contract
  `computed-mask-strided-source-before-skipped-source-ignored;inactive-products-zero-before-reduction;accumulator-out0-seed-carry;scalar-output-only-tail-preserve.v1`
  and wired it through route facts, route validation, target metadata/header
  evidence, generated bundle ABI dry-run evidence, stale mirror rejection, and
  focused tests.
* Its check context explicitly did not rerun `ssh rvv`; this round is therefore
  not allowed to close on dry-run evidence alone if a fresh runtime correctness
  claim is made.
* Current dry-run lit fixtures already cover both explicit and pre-realized
  selected bodies for the computed-mask strided-input widening dot route, with
  generated harness checks for counts `0,1,16,17,257`, stride pairs `2:3` and
  `3:2`, two mask patterns, two input patterns, source preservation,
  accumulator preservation, scalar output, skipped source ignoring, and tail
  preservation.

## Requirements

* Stay on the existing
  `computed_masked_strided_input_widening_dot_reduce_add` route family. Do not
  choose another route family.
* The production seam must remain RVV provider-owned and target-consumed. Common
  EmitC may only materialize the provider-built route payload; it must not infer
  mask, stride, dot, accumulator, scalar-output, dtype, policy, ABI, or header
  semantics.
* The executable artifact proof must use the generated target object/header
  bundle and generated external C ABI consumer. It must not use a hand-written
  direct runtime wrapper as the authority.
* Positive runtime evidence must be real `ssh rvv` evidence, not dry-run or
  local host execution. The evidence must show `dry_run: false` and
  `ssh_evidence: true`.
* Runtime evidence must compare against an oracle over varied runtime counts,
  signed input values, accumulator seeds, mask patterns, skipped/inactive lanes,
  strided source addresses, and output tail regions.
* The generated harness must keep source-before snapshots in the expected-value
  computation, not merely compare sources after the call.
* Explicit and pre-realized selected-body artifact paths are both in scope if
  both non-dry-run workflows are already bounded and executable. If one is
  blocked, close the working path and leave the exact continuation point.
* Negative evidence must be covered by existing or new focused tests that fail
  closed for stale or missing semantic contract, metadata mirror, ABI order,
  header/prototype binding, strided source fact, computed mask fact,
  inactive-product policy, accumulator seed/carry fact, scalar-output/tail
  fact, dtype/config fact, or runtime AVL/VL fact.
* If executable evidence exposes a production seam defect, fix the production
  seam or add a fail-closed guardrail at the RVV plugin/target artifact
  boundary before accepting any generated bundle.

## Acceptance Criteria

* [x] Positive generated-bundle ABI e2e for the explicit selected-body
  `computed_masked_strided_input_widening_dot_reduce_add` path compiles and
  runs on `ssh rvv`, reports `dry_run: false`, `ssh_evidence: true`, and passes
  oracle checks for counts `0,1,16,17,257`, multiple stride pairs, mask
  patterns, signed values, accumulator seeds, skipped source elements, scalar
  output, source preservation, accumulator preservation, and tail preservation;
  or the PRD records the precise executable blocker.
* [x] Positive generated-bundle ABI e2e for the pre-realized selected-body path
  compiles and runs on `ssh rvv` with the same correctness scope, or the PRD
  records the precise executable blocker.
* [x] Existing dry-run lit checks for explicit and pre-realized computed-mask
  strided-input widening dot artifacts still pass and still show the provider
  contract, runtime ABI order, route operand binding summary, strided source
  facts, mask facts, inactive-lane zeroing, low-precision resource selection,
  target validator consumed facts, and generated harness oracle.
* [x] Existing C++ target artifact / plugin tests still fail closed for stale
  provider facts, stale target metadata mirrors, ABI/header/prototype mismatch,
  route operand binding mismatch, and unsupported stale route-family facts for
  this route family.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passes.
* [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passes if
  the evidence script is touched or if the generated-bundle harness path needs
  validation before `ssh rvv`.
* [x] Bounded old-authority scan over touched files and added diff lines finds
  no new positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor/direct-C, source-front-door, or mirror-only
  route authority.
* [x] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are clean or explicitly reported.
* [x] Trellis task context, workspace journal, finish/archive status, and one
  coherent commit are completed if the task finishes.

## Out Of Scope

* No broad dot/MAcc/reduction matrix.
* No dtype/LMUL clone batch.
* No new route family.
* No source-front-door positive route.
* No high-level Linalg/Vector/StableHLO frontend.
* No per-Linalg route authority.
* No performance tuning database, dashboard, or index/report-only closeout.
* No common EmitC invention of RVV semantics.
* No unrelated memory, elementwise, base MAcc, compare/select, segment2, or
  standalone-reduction churn except stale-contract fail-closed guardrails
  directly required by this artifact boundary.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
* Archived task read:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-computed-masked-strided-dot-acceptance-boundary/`.
* Primary files from the brief:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  explicit/pre-realized generated-bundle ABI lit tests,
  explicit/pre-realized target artifact MLIR fixtures, and
  `test/Target/TargetArtifactExportTest.cpp`.

## Results

No compiler source change was required. The accepted production provider/target
contract from `1bf69314` generated executable object/header bundles for both
bounded selected-body paths.

Positive `ssh rvv` evidence:

* Explicit selected-body:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-explicit-computed-masked-strided-dot-ssh/computed_masked_strided_input_widening_dot_reduce_add/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.
* Pre-realized selected-body:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-prerealized-computed-masked-strided-dot-ssh/computed_masked_strided_input_widening_dot_reduce_add/evidence.json`
  reports `status: success`, `dry_run: false`, `ssh_evidence: true`,
  `remote_compile_succeeded: true`, and `remote_run_succeeded: true`.

Both remote runs printed:

```text
PASS op=computed_masked_strided_input_widening_dot_reduce_add counts=0,1,16,17,257 stride_pairs=2:3,3:2 mask_patterns=2 input_patterns=2 source_preserved accumulator_preserved tail_preserved
```

The evidence boundary records provider-derived authority, runtime ABI order
`cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n,lhs_stride,rhs_stride`, active-lane signed
i16*i16 widening products, inactive-lane zero-before-reduction behavior,
`acc[0]` seed source, and `out[0]` loop carry source.

Negative/fail-closed evidence was revalidated by the focused lit filter and the
C++ target/plugin regression tests already covering stale provider facts, stale
metadata mirrors, stale ABI/header/prototype binding, stale route operand
binding, and unsupported direct pre-realized route-entry shortcuts for this
route family.

Spec update judgment: no `.trellis/spec/` change is needed in this round. The
durable executable contract already existed in the RVV and EmitC route specs;
this task refreshed runtime artifact evidence without discovering a new
contract, signature, validation matrix, or reusable convention.
