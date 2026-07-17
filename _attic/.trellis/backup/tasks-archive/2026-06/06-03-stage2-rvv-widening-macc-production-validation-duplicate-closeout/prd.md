# Stage2 RVV widening MAcc production validation duplicate closeout

## Goal

Validate the Hermes direction brief against current repository state for the
production provider-to-target validation surface of the existing
`widening_macc_add` route.

If live code shows a missing provider-owned widening MAcc fact, target
artifact consumer check, stale-mirror fail-closed behavior, or focused test
coverage, this task should implement only that bounded missing behavior. If
live code, committed history, and archive evidence already cover the requested
boundary, this task is a duplicate/stale brief closeout and should finish with
no production source changes.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV widening MAcc production validation boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` returned no dirty file list through RTK.
* Initial `git log --oneline -8` started at
  `e96627e1 trellis: close duplicate macc validation brief`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* The brief says no completed widening-MAcc production-validation owner was
  visible in the current archive search.
* Live repository evidence contradicts that premise:
  * `91881603 rvv: validate widening macc route facts` is present in history.
  * The archived same-title task at
    `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-widening-macc-production-validation-boundary/prd.md`
    records completion of the requested provider-to-target validation boundary.
  * Adjacent archive evidence shows the previous unit-stride MAcc stale brief
    was closed after verifying `8c819ae0` and `78a38162`.

## Current Repository Evidence

Live inspection before implementation showed:

* `.trellis/spec/index.md` keeps the RVV-first authority chain at selected
  `tcrv.exec` variant -> typed low-level `tcrv_rvv` body -> RVV provider ->
  provider-built `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` names
  `widening_macc_add` as an active direct-provider contraction route and
  requires direct contraction provider plans to consume same-analysis
  contraction family, materialization, math operand-binding, control, ABI, and
  typed leaf facts before route construction.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires
  `widening_macc_add` provider facts for `lhs,rhs,acc,out,n`,
  `tcrv_rvv.widening_macc`, `i16mf2` sources, `i32m1` accumulator/result,
  signed widening relation, accumulator/result layouts, route operand
  binding, header/type summaries, target leaf profile, and explicit
  `provider_supported_mirror`.
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` exposes
  `RVVWideningMAccRouteFacts` and `getRVVWideningMAccRouteFacts(...)`.
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` exposes direct
  contraction provider/statement plan surfaces and widening MAcc flags.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` contains
  widening MAcc target artifact validation that consumes provider facts and
  validates provider/candidate mirrors.
* `test/Target/TargetArtifactExportTest.cpp` contains widening MAcc canonical
  fact checks and fail-closed mutations for provider facts, route statements,
  and candidate metadata mirrors.
* `scripts/rvv_generated_bundle_abi_e2e.py` already emits and self-checks the
  `widening_macc_boundary` evidence surface for explicit and pre-realized
  `widening_macc_add`.

## Requirements

* Do not redo unit-stride MAcc, computed-mask MAcc, widening dot-reduce,
  standalone reductions, conversions, memory movement, source-front-door
  routes, high-level frontend lowering, new dtype/LMUL clone batches, global
  tuning, or broad smoke matrices.
* Check whether current HEAD already satisfies the requested
  provider-to-target widening MAcc validation boundary.
* If a live gap is found, implement only the missing `widening_macc_add`
  provider/target validation behavior and focused coverage.
* If current HEAD already satisfies the requested boundary, finish this task as
  a duplicate/stale brief closeout with no production source edits.
* Keep common EmitC/export neutral: common code may carry provider-built
  payloads and mirrors, but must not infer widening MAcc semantics.

## Acceptance Criteria

* [x] Current HEAD is checked for provider-owned `widening_macc_add` facts
      covering operation kind, typed compute op, source and accumulator/result
      element types, source and accumulator/result SEW/LMUL, widening relation,
      accumulator/result layouts, memory forms, runtime ABI order/roles,
      tail/mask policy, intrinsic leaves, header/type summaries, target leaf
      profile, `provider_supported_mirror`, and route-family mirror labels.
* [x] Target validation is checked to consume provider-owned widening MAcc
      facts and fail closed on stale unit-stride MAcc, computed-mask MAcc,
      widening dot-reduce, conversion, reduction, memory, scalar-broadcast,
      compare/mask, dtype, widening relation, layout, intrinsic, ABI,
      header/type, target-profile, provider mirror, and route-family mirrors.
* [x] Focused C++ tests are checked for provider-fact and candidate-mirror
      fail-closed coverage for `widening_macc_add`.
* [x] Focused build/test checks pass.
* [x] Bounded old-authority scan over inspected/touched files finds no new
      positive dependency on legacy `i32m1`, descriptor, source-front-door,
      route-id, artifact-name, exact intrinsic spelling, or mirror-only
      authority.
* [x] No production gap was found; this task is archived as a duplicate
      stale-brief validation closeout with no production source edits.

## Out Of Scope

* Widening dot-reduce and computed-mask widening dot-reduce.
* Unit-stride, scalar-broadcast, computed-mask, or runtime-scalar computed-mask
  MAcc redo.
* Standalone reductions, compare/select expansion, segment2, indexed,
  strided, or base memory redo.
* New dtype/LMUL clone batches.
* Source-front-door positive routes or high-level frontend lowering.
* Global tuning, dashboards, broad smoke matrices, or evidence-only packaging.
* Runtime `ssh rvv` reruns unless route emission, runtime ABI, generated C,
  widening relation, accumulation behavior, tail behavior, or destination
  preservation changes.

## Evidence Plan

* Validate the current Trellis task shape.
* Build and run `tianchenrv-target-artifact-export-test`.
* Build `tcrv-opt` and `tcrv-translate` if route fixtures/scripts are touched
  or lit/script checks require fresh tools.
* Run focused lit filter for `widening-macc-add`.
* Run generated-bundle dry-runs for explicit and pre-realized
  `widening_macc_add` where script-supported.
* Run the direct pre-realized route-entry fail-closed script check if covered
  by the focused lit filter.
* Run bounded old-authority scans over the inspected provider/target/test
  files and diff-only scans over this task's changes.
* Run `rtk git diff --check`.
* Do not run `ssh rvv` if no production route emission or runtime behavior
  changes are made; reuse archived signed widening MAcc runtime evidence and
  state the rationale.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant archived tasks read:

* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-widening-macc-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-unit-stride-macc-production-validation-boundary/prd.md`

Relevant live files inspected:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/TargetArtifactExportTest.cpp`
* `test/Target/RVV/*widening-macc-add*.mlir`
* `test/Scripts/*widening-macc-add*.test`

## Final Outcome

Completed as a duplicate/stale-brief validation closeout.

Live HEAD already contains the requested production provider-to-target
`widening_macc_add` validation boundary:

* `91881603 rvv: validate widening macc route facts` is present in current
  history.
* The same-title archived task
  `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-widening-macc-production-validation-boundary/prd.md`
  records the production changes and focused evidence for this boundary.
* Current provider, planning, target validation, script, and C++ test surfaces
  still expose and validate the required widening MAcc facts.

No production source changes were needed. This task only records the current
duplicate closeout and verification evidence.

Checks run:

* `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-03-06-03-stage2-rvv-widening-macc-production-validation-boundary`
* `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk cmake --build build --target tcrv-opt tcrv-translate -j2`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter widening-macc-add`
  from `build/test`, which selected and passed 5 tests.
* Explicit selected-body generated-bundle dry-run for `widening_macc_add`
  over counts `0,1,16,17,257`: `dry_run_success`.
* Pre-realized selected-body generated-bundle dry-run for
  `widening_macc_add` over counts `0,1,16,17,257`: `dry_run_success`.
* Direct pre-realized route-entry fail-closed probe for
  `widening_macc_add` over counts `0,1,16,17,257` exited 1 with the expected
  retired direct route-entry diagnostic.
* Bounded old-authority scan over inspected provider/target/test files found
  only pre-existing negative fixture, fail-closed guard, or stale-mutation
  hits; diff-only scan over this task's changes found no matches.
* `rtk git diff --check`

Runtime evidence:

* No new `ssh rvv` run was required because this round made no production
  route-emission, generated C, runtime ABI, widening relation, accumulation,
  tail, or destination-preservation behavior changes.
* The archived signed widening MAcc runtime evidence remains the executable
  correctness evidence for the unchanged runtime path.
