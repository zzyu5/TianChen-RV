# Stage2 RVV computed-mask segment2 update executable artifact closeout

## Goal

Close one executable RVV generated-bundle path for
`computed_masked_segment2_update_unit_load`, starting from the existing
explicit and pre-realized selected `tcrv.exec` RVV segment2 fixtures and ending
in real `ssh rvv` correctness evidence. The intended execution chain is:

```text
selected typed tcrv_rvv computed-mask segment2 update body
  -> RVV selected-body realization/provider route
  -> TCRVEmitCLowerableRoute
  -> common EmitC/materialized target artifact
  -> generated header/object/harness bundle
  -> ssh rvv correctness evidence
```

This is an executable closeout for one contract-backed segment2 route. It is
not another target validation ownership cleanup and not a broad segment2 test
matrix.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-mask segment2 update executable artifact closeout`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short`: clean through RTK.
* Initial `git log --oneline -8` started at
  `441fe236 rvv: close segment2 target contract consumer`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflow.

## What I Already Know

* `.trellis/spec/index.md` requires the RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/provider -> `TCRVEmitCLowerableRoute` ->
  common EmitC materializer -> target artifact -> `ssh rvv` evidence for
  runtime/correctness/performance claims.
* `.trellis/spec/extension-plugins/rvv-plugin.md` says segment2 production
  planning is selected-body route-family provider planning, not route-entry
  ownership. `computed-mask segment2 update` must flow through the segment2
  route-family planning owner and segment2 statement-plan boundary.
* `.trellis/spec/lowering-runtime/emitc-route.md` already defines provider-
  owned computed-mask segment2 update facts and
  `RVVSegment2MemoryRouteValidationContract`. Target validation must consume
  those facts and must not infer semantics from route ids, artifact names,
  fixture names, descriptors, scripts, common EmitC, metadata mirrors, or exact
  RVV intrinsic spellings.
* `.trellis/spec/testing/mlir-testing-contract.md` requires real `ssh rvv`
  output for RVV runtime/correctness claims. Generated-bundle memory-writing
  evidence must check active lanes, inactive/passthrough preservation, runtime
  `n`/AVL/VL behavior, and tail sentinel preservation.
* The archived task
  `.trellis/tasks/archive/2026-06/06-04-06-04-stage2-rvv-segment2-memory-target-contract-closeout/`
  completed target consumer closeout and intentionally did not run `ssh rvv`,
  because that round changed only validation ownership and stale diagnostics.
* Live inspection shows `scripts/rvv_generated_bundle_abi_e2e.py` already has
  an `OpExpectation` and harness branch for
  `computed_masked_segment2_update_unit_load`, including two compare-mask
  patterns, independent scalar reference logic, source preservation checks,
  inactive lane preservation, and tail sentinel checks.
* Existing dry-run lit coverage covers both
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-segment2-update-dry-run.test`
  and
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-computed-masked-segment2-update-dry-run.test`
  with runtime counts `0,1,7,16,23,257`, but both currently assert
  `ssh_evidence=false`.

## Requirements

* Produce generated artifact/header/object/harness bundles for
  `computed_masked_segment2_update_unit_load` from the existing selected RVV
  body path.
* Run non-dry-run generated-bundle evidence on `ssh rvv` for representative
  runtime counts including `0`, `1`, small multi-lane counts, a VL-boundary
  count, and a tail/sentinel stress count.
* The harness must link against the generated object/header artifact and use an
  independent scalar reference for compare-produced masks, segment2 field
  update behavior, inactive lane preservation, runtime `n`/AVL/VL behavior,
  and tail preservation.
* Evidence JSON must record `ssh_evidence=true`, runtime ABI order, selected
  variant, provider-supported mirror, generated artifact paths, remote compile
  and run summaries, and PASS output.
* Keep existing dry-run lit coverage passing for explicit and pre-realized
  computed-mask segment2 update fixtures.
* If the script cannot run this route live, extend only the necessary
  generated-bundle harness/runner support for this route.
* Keep provider and target validation semantics unchanged unless live execution
  exposes a real compiler bug.

## Acceptance Criteria

* [x] A focused pre-realized selected-body generated-bundle command for
      `computed_masked_segment2_update_unit_load` produces a local bundle and
      passes `ssh rvv` compile/run.
* [x] A focused explicit selected-body generated-bundle command for
      `computed_masked_segment2_update_unit_load` is run if needed to prove the
      existing explicit fixture remains executable or to debug any shared
      runner issue.
* [x] Per-op evidence JSON records `ssh_evidence=true`, `status=success`,
      `input_mode`, selected variant, runtime counts, generated object/header/
      harness paths, runtime ABI order, provider-supported mirror, and PASS
      output from the remote run.
* [x] The generated harness links against the generated artifact instead of
      reimplementing RVV semantics, and its scalar oracle covers compare-
      produced active/inactive mask lanes, segment2 field update, source
      preservation, inactive lane preservation, runtime count boundaries, and
      tail sentinel preservation.
* [x] Existing dry-run lit coverage for pre-realized and explicit computed-mask
      segment2 update remains passing.
* [x] `tianchenrv-target-artifact-export-test` and
      `tianchenrv-rvv-extension-plugin-test` are built/run if provider or
      target source changes are made. No provider or target source changed in
      this round, so these were not required.
* [x] Focused segment2 lit/generated-bundle filters pass after any edits.
* [x] Bounded artifact inspection confirms no descriptor/direct-C/source-export/
      source-front-door residue and no legacy i32/source-front-door authority
      in generated evidence for this route.
* [x] Bounded old-authority scan over touched files finds no new positive
      dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door,
      mirror-only authority, route-id/artifact-name authority, or exact
      `__riscv_*_i32m1` intrinsic authority.
* [x] `rtk git diff --check` passes.
* [x] If complete, finish/archive the Trellis task and create one coherent
      commit.

## Technical Approach

1. Reproduce the existing dry-run bundle for pre-realized and explicit
   `computed_masked_segment2_update_unit_load` to establish the baseline.
2. Run the non-dry-run generated-bundle command on `ssh rvv` using a bounded
   artifact root under `artifacts/tmp/06-04-computed-mask-segment2-update-executable-closeout/`.
3. Inspect `evidence.json`, generated C/header/object/harness, and remote
   stdout/stderr for the exact runtime ABI, selected variant, provider mirror,
   artifact paths, and PASS marker.
4. If live execution fails because the runner or harness has a route-local gap,
   patch only `scripts/rvv_generated_bundle_abi_e2e.py` and the focused
   dry-run lit expectations needed to cover that gap.
5. If live execution exposes a compiler/provider/target bug, stop and repair
   only the bounded production path required for this route, then run the
   provider/target C++ checks named in the brief.
6. Record evidence results in this PRD, validate/archive the Trellis task, and
   commit if the acceptance criteria are satisfied.

## Evidence Plan

* Dry-run pre-realized baseline:
  `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/06-04-computed-mask-segment2-update-executable-closeout --run-id pre-realized-computed-mask-segment2-update-dry --overwrite --op-kind computed_masked_segment2_update_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`
* Live pre-realized evidence:
  `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-04-computed-mask-segment2-update-executable-closeout --run-id pre-realized-computed-mask-segment2-update-ssh-rvv --overwrite --op-kind computed_masked_segment2_update_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`
* Dry-run explicit baseline:
  `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/06-04-computed-mask-segment2-update-executable-closeout --run-id explicit-computed-mask-segment2-update-dry --overwrite --op-kind computed_masked_segment2_update_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`
* Live explicit evidence if needed:
  `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/06-04-computed-mask-segment2-update-executable-closeout --run-id explicit-computed-mask-segment2-update-ssh-rvv --overwrite --op-kind computed_masked_segment2_update_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate`
* Focused lit:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv build/test --filter 'computed-masked-segment2-update|pre-realized-selected-body-artifact-computed-masked-segment2-update|explicit-selected-body-artifact-computed-masked-segment2-update'`
* If provider or target code changes:
  `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
  `rtk build/bin/tianchenrv-target-artifact-export-test`
  `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* Bounded generated artifact residue scan over this task's artifact root for
  `descriptor`, `direct-C`, `source-export`, `source-front-door`, and
  positive legacy `tcrv_rvv.i32_` authority.
* Bounded old-authority scan over touched files.
* `rtk git diff --check`
* `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-computed-mask-segment2-update-executable-closeout`

## Out Of Scope

* No expansion to all segment2 routes, segment3+, new operations, dtype/LMUL
  clone batches, compare/select expansion, reduction/MAcc work, source-front-
  door routes, high-level frontend lowering, dashboards, broad smoke matrices,
  or report-only evidence.
* No weakening of the segment2 provider contract, target consumer closeout, or
  stale mirror diagnostics.
* No harness semantics derived from route ids, artifact names, C strings,
  descriptor residue, metadata mirrors, or test names.
* No provider/target validation source changes unless live evidence exposes a
  real compiler bug.

## Definition Of Done

* `computed_masked_segment2_update_unit_load` has at least one successful
  non-dry-run generated-bundle `ssh rvv` evidence record with `ssh_evidence`
  true and PASS output.
* Existing dry-run lit coverage remains green.
* The task PRD records artifact paths, remote command evidence, checks, scans,
  self-repair if any, and final status.
* The Trellis task is finished/archived and one coherent commit is created if
  all acceptance criteria pass.

## Implementation Results

* No compiler, provider, target, runner, or harness source changes were
  required.
* The existing pre-realized selected-body path already materialized the
  computed-mask segment2 update body through the public selected lowering
  boundary, rebuilt provider route facts, exported a header/object bundle, and
  linked a generated harness against that bundle on the RVV target.
* The existing explicit selected-body fixture also ran live through the same
  generated-bundle path.
* The generated harness checks two compare data patterns over runtime counts
  `0,1,7,16,23,257`. It verifies compare-produced active/inactive mask lanes,
  field update behavior (`src0[index] + src1[index]` for field0 and `src1`
  for field1 on active lanes), inactive interleaved destination preservation,
  source preservation, field-distinguishing coverage, and tail sentinel
  preservation past runtime `2*n`.

## Evidence Results

### Generated Bundle / `ssh rvv`

Pre-realized selected body:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-04-computed-mask-segment2-update-executable-closeout --run-id pre-realized-computed-mask-segment2-update-ssh-rvv --overwrite --op-kind computed_masked_segment2_update_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate
```

Result:

* Status: success.
* Evidence root:
  `artifacts/tmp/06-04-computed-mask-segment2-update-executable-closeout/pre-realized-computed-mask-segment2-update-ssh-rvv`.
* Per-op evidence:
  `artifacts/tmp/06-04-computed-mask-segment2-update-executable-closeout/pre-realized-computed-mask-segment2-update-ssh-rvv/computed_masked_segment2_update_unit_load/evidence.json`.
* Generated bundle:
  `artifacts/tmp/06-04-computed-mask-segment2-update-executable-closeout/pre-realized-computed-mask-segment2-update-ssh-rvv/computed_masked_segment2_update_unit_load/generated_bundle/`.
* Generated object:
  `generated_bundle/artifact-0-riscv-elf-relocatable-object-rvv-generic-typed-body-emitc-route-family.o`.
* Generated header:
  `generated_bundle/artifact-1-runtime-callable-c-header-rvv-generic-typed-body-emitc-route-family.header.h`.
* Harness:
  `rvv_generated_bundle_abi_computed_masked_segment2_update_unit_load_harness.c`.
* `ssh_evidence`: true.
* `input_mode`: `pre-realized-selected-body`.
* Selected variant: `pre_realized_body_rvv_cmseg_update`.
* Runtime ABI order: `cmp_lhs,cmp_rhs,src0,src1,dst,n`.
* Provider-supported mirror:
  `provider_supported_mirror:rvv-computed-mask-segment2-update-add-plan-validated`.
* Remote PASS output ended with:

```text
tcrv_rvv_generated_bundle_abi_computed_masked_segment2_update_unit_load_ok counts=0,1,7,16,23,257 patterns=0,1
PASS op=computed_masked_segment2_update_unit_load counts=0,1,7,16,23,257 patterns=0,1
```

Explicit selected body:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/06-04-computed-mask-segment2-update-executable-closeout --run-id explicit-computed-mask-segment2-update-ssh-rvv --overwrite --op-kind computed_masked_segment2_update_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 7 --runtime-count 16 --runtime-count 23 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate
```

Result:

* Status: success.
* Evidence root:
  `artifacts/tmp/06-04-computed-mask-segment2-update-executable-closeout/explicit-computed-mask-segment2-update-ssh-rvv`.
* `ssh_evidence`: true.
* `input_mode`: `explicit-selected-body`.
* Remote PASS output ended with:

```text
tcrv_rvv_generated_bundle_abi_computed_masked_segment2_update_unit_load_ok counts=0,1,7,16,23,257 patterns=0,1
PASS op=computed_masked_segment2_update_unit_load counts=0,1,7,16,23,257 patterns=0,1
```

Dry-run baselines also passed:

* `pre-realized-computed-mask-segment2-update-dry`
* `explicit-computed-mask-segment2-update-dry`

### Focused Lit

First attempted from the repository root:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv build/test --filter 'computed-masked-segment2-update|pre-realized-selected-body-artifact-computed-masked-segment2-update|explicit-selected-body-artifact-computed-masked-segment2-update'
```

That failed before test execution because `build/test/lit.site.cfg.py` resolves
`../../test/lit.cfg.py` relative to the working directory. This was a command
working-directory issue, not a test failure.

Rerun from `build/test`:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'computed-masked-segment2-update|pre-realized-selected-body-artifact-computed-masked-segment2-update|explicit-selected-body-artifact-computed-masked-segment2-update'
```

Result:

```text
Total Discovered Tests: 477
  Excluded: 472
  Passed  :   5
```

### Scans

Generated artifact residue scan:

```text
rtk rg -n "descriptor|direct-C|source-export|source-front-door|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|RVVI32M1|rvv-i32m1" artifacts/tmp/06-04-computed-mask-segment2-update-executable-closeout
```

Result: no matches.

Touched task-file old-authority scan:

```text
rtk rg -n "RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|route-id|artifact-name|mirror-only" .trellis/tasks/06-04-stage2-rvv-computed-mask-segment2-update-executable-closeout
```

Result: matches are only negative guardrails, scan terms, and non-authority
requirements in this PRD/context. No positive legacy authority was introduced.

### Not Run

* `tianchenrv-target-artifact-export-test`
* `tianchenrv-rvv-extension-plugin-test`

Rationale: no provider, target, compiler, script, or harness source changed.
This round only created the Trellis task/context and produced executable
generated-bundle evidence using the existing route.

## Spec Update Judgment

No `.trellis/spec/` update is needed. This round did not introduce a new route
contract, runner convention, target validation rule, or architectural boundary.
It executed an already specified and already implemented generated-bundle path
and recorded real RVV evidence for the bounded computed-mask segment2 update
route.

## Final Status

Completed. The task is ready for Trellis finish/archive and commit.
