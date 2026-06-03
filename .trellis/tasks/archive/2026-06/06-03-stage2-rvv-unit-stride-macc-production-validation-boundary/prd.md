# Stage2 RVV unit-stride MAcc production validation boundary

## Goal

Validate the Hermes direction brief against current repository state for the
production provider-to-target validation surface of the existing unit-stride
multiply-accumulate routes:

* `macc_add`
* `scalar_broadcast_macc_add`
* `computed_masked_macc_add`
* `runtime_scalar_cmp_masked_macc_add`

If live code shows any missing provider-owned MAcc facts, target-consumer
checks, stale-mirror fail-closed behavior, or focused test coverage for these
four routes, this task should implement that bounded missing behavior. If live
code and committed history already cover the requested boundary, this task is a
duplicate/stale brief closeout and should finish with no production source
changes.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV unit-stride MAcc production validation boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` returned clean status through RTK.
* Initial `git log --oneline -8` started at
  `230cb756 rvv: validate segment2 memory provider facts`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* The direction brief says no completed MAcc production-validation owner was
  visible in the current archive search.
* Live archive and git evidence contradict that premise:
  * `8c819ae0 rvv: validate unit-stride macc route facts` is present in
    current history.
  * `78a38162 rvv: validate computed-mask macc route facts` is present in
    current history.
  * Archived
    `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-unit-stride-macc-production-validation-boundary/prd.md`
    records provider-owned unit-stride MAcc facts and target validation for
    `macc_add` and `scalar_broadcast_macc_add`.
  * Archived
    `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-computed-mask-macc-production-validation-boundary/prd.md`
    records the follow-on computed-mask MAcc boundary for
    `computed_masked_macc_add` and `runtime_scalar_cmp_masked_macc_add`.

## Current Repository Evidence

Live inspection before implementation showed:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` exposes:
  * `RVVUnitStrideMAccRouteFacts`;
  * `RVVComputedMaskMAccRouteFacts`;
  * `RVVRuntimeScalarComputedMaskMAccRouteFacts`;
  * provider accessors for the unit-stride, computed-mask, and runtime-scalar
    computed-mask MAcc route facts.
* The fact surfaces carry provider-owned operation kind, memory form,
  SEW/LMUL, tail/mask policy, runtime control plan, runtime ABI order,
  runtime ABI parameters and roles, target leaf profile,
  `providerSupportedMirror`, header/type summaries, operand-binding plan and
  summary, typed compute op, arithmetic kind, MAcc accumulator/result layout,
  source/destination memory facts, and the scalar-broadcast or computed-mask
  sub-family facts where applicable.
* `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp` populates those
  facts from the RVV MAcc owner-local constants and builds provider-owned
  operand-binding summaries.
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` consumes these
  provider facts for MAcc payload validation and candidate mirror validation.
  It rejects stale unit-stride/computed-mask/widening/segment/indexed/base
  memory/reduction/contraction facts where they appear on the wrong MAcc
  sub-family.
* `test/Target/TargetArtifactExportTest.cpp` includes positive provider-fact
  mirror checks and fail-closed mutations for the four in-scope MAcc routes,
  including stale provider mirror, binding plan/summary, header/type mapping,
  target profile, accumulator role/layout, computed-mask mask/passthrough,
  runtime-scalar predicate, source-memory facts, candidate metadata mirrors,
  and provider-built statement operands.

## Requirements

* Do not expand beyond the four in-scope existing routes.
* Do not include widening MAcc, dot-reduce, standalone reductions,
  conversions, memory-movement redo, source-front-door routes, high-level
  frontend lowering, or new route families.
* If current HEAD already satisfies the requested provider-to-target MAcc
  boundary, do not duplicate the implementation.
* If a live gap is found, implement only that missing MAcc provider/target
  validation behavior and focused coverage.
* Keep common EmitC/export neutral: it may carry provider-built payloads and
  mirrors, but must not infer MAcc semantics.

## Acceptance Criteria

* [ ] Current HEAD is checked for provider-owned facts covering
      `macc_add`, `scalar_broadcast_macc_add`,
      `computed_masked_macc_add`, and
      `runtime_scalar_cmp_masked_macc_add`.
* [ ] Target validation is checked to consume provider-owned facts for runtime
      ABI order and parameter roles, SEW/LMUL/policy, typed operation,
      arithmetic kind, accumulator/result layout, scalar-broadcast RHS facts,
      computed-mask/runtime-scalar mask facts, operand binding, headers/types,
      target profile, provider mirror, and candidate mirrors.
* [ ] Focused tests are checked for provider-fact mirrors and fail-closed
      stale/missing fact coverage for the in-scope MAcc forms.
* [ ] Small focused checks pass.
* [ ] Bounded old-authority scan over inspected/touched files finds no new
      positive dependency on legacy `i32m1`, descriptor, source-front-door,
      route-id, artifact-name, or mirror-only authority.
* [ ] If no production gap is found, this task is archived as a duplicate
      stale-brief validation closeout with no production source commit.

## Out Of Scope

* Widening MAcc and widening dot-reduce.
* Standalone reductions, conversions, compare/select expansion, segment2,
  indexed, strided, or base memory redo.
* New dtype/LMUL clone batches.
* Source-front-door positive routes or high-level frontend lowering.
* Global tuning or broad smoke matrices.
* Runtime `ssh rvv` reruns unless route emission, runtime ABI, generated C
  behavior, accumulation semantics, scalar broadcast semantics, or mask/
  passthrough behavior changes.

## Evidence Plan

* Inspect provider fact surfaces and target artifact validation code for the
  four MAcc routes.
* Build and run `tianchenrv-target-artifact-export-test`.
* Run focused lit filters for explicit and pre-realized MAcc route fixtures
  where available.
* Run generated-bundle dry-run coverage for the in-scope MAcc forms if scripts
  are supported and build tools are current.
* Run bounded old-authority scan over the relevant provider/target/test files.
* Run `rtk git diff --check`.
* Do not run `ssh rvv` if no production route emission or runtime behavior
  changes are made; rely on the archived runtime evidence from the previous
  MAcc ABI/validation tasks.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant archived tasks read:

* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-unit-stride-macc-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-computed-mask-macc-production-validation-boundary/prd.md`

Initial implementation focus, if a gap is found:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* `test/Target/RVV/*macc-add*.mlir`
* `test/Scripts/*macc-add*.test`

## Final Outcome

Completed as a duplicate/stale-brief validation closeout.

Live HEAD already contains the requested production provider-to-target MAcc
validation boundary:

* `8c819ae0 rvv: validate unit-stride macc route facts` covers
  `macc_add` and `scalar_broadcast_macc_add`.
* `78a38162 rvv: validate computed-mask macc route facts` covers
  `computed_masked_macc_add` and
  `runtime_scalar_cmp_masked_macc_add`.

No production source changes were needed. The inspected implementation already
exposes provider-owned fact surfaces, populates them in the RVV MAcc
route-family owner, consumes them in the target artifact validator, and has
focused positive/fail-closed C++ coverage for all four in-scope route forms.

Checks run:

* `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-03-stage2-rvv-unit-stride-macc-production-validation-boundary`
* `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk cmake --build build --target tcrv-opt tcrv-translate -j2`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter macc-add` from `build/test`, 28/28 selected tests passed.
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter scalar-broadcast-macc-add` from `build/test`, 4/4 selected tests passed.
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter computed-masked-macc-add` from `build/test`, 6/6 selected tests passed.
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter runtime-scalar-cmp-masked-macc-add` from `build/test`, 8/8 selected tests passed.
* Explicit selected-body generated-bundle dry-run for all four MAcc forms over counts `0,1,16,17,257` and RHS scalars `-3,0,5`: `dry_run_success`.
* Pre-realized selected-body generated-bundle dry-run for all four MAcc forms over counts `0,1,16,17,257` and RHS scalars `-3,0,5`: `dry_run_success`.
* Direct pre-realized route-entry fail-closed probes for all four MAcc forms. A first single-count `macc_add` probe failed the script's evidence-count gate; it was rerun with counts `0,1,16,17,257` and then failed closed with the expected retired direct route-entry diagnostic.
* Bounded old-authority scan over production provider/target files found no matches for legacy i32/source-front-door/descriptor/direct-C/source-export authority strings.
* Bounded old-authority scan over the MAcc validation region of `test/Target/TargetArtifactExportTest.cpp` found no matches.

Runtime evidence:

* No new `ssh rvv` run was required because this round made no production
  route-emission, generated C, runtime ABI, accumulation, scalar RHS, mask,
  passthrough, tail, or destination-preservation behavior changes.
* Existing archived MAcc ABI/validation runtime evidence remains the
  executable correctness evidence for the unchanged runtime paths.
