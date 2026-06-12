# Stage2 RVV widening dot-reduce production validation boundary

## Goal

Close the production provider-to-target validation boundary for existing
`widening_dot_reduce_add` and
`strided_input_widening_dot_reduce_add` selected-body RVV routes. Include
computed-mask widening dot-reduce variants where they already share the same
coherent provider fact surface.

This round must make narrow source type, widened accumulator/result type,
reduction seed/result roles, widening product/reduction facts,
unit-stride/strided-input memory forms, optional mask facts, runtime ABI roles,
route operand binding, header/type participation, target leaf profile, and
fail-closed diagnostics come from provider-owned typed `tcrv_rvv`
body/config/runtime facts. Target artifact validation must not accept these
routes from route ids, artifact names, fixture names, candidate mirrors,
descriptors, C strings, exact intrinsic spellings, common EmitC inference, or
stale metadata.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV widening dot-reduce production validation boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` returned a clean worktree.
* Initial `git log --oneline -8` started at
  `91881603 rvv: validate widening macc route facts`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* Specs require the route authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned realization/provider facts -> provider-built
  `TCRVEmitCLowerableRoute` -> common EmitC materializer -> RVV target
  artifact validation.
* Archived ABI tasks already proved executable runtime behavior for
  `widening_dot_reduce_add`,
  `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add` with real `ssh rvv`
  evidence for counts including `0,1,16,17,257`.
* The immediately previous widening MAcc production validation task added a
  provider-owned fact accessor and rewired target validation to consume it.
* Live code inspection shows `RVVComputedMaskStridedInputWideningDotReduceRouteFacts`
  already exists, but plain, strided, and computed-mask unit-stride
  widening-dot target validation still carries target-local ABI/binding
  constants and mirror checks.

## Requirements

* Keep scope to existing signed widening dot-reduce relations:
  `i16mf2 lhs x i16mf2 rhs -> i32m1 reduction/result`, seeded from the
  accumulator scalar lane and stored to an `i32` scalar output.
* Preserve the production chain:
  selected `tcrv.exec` RVV variant -> typed/pre-realized widening dot-reduce
  body -> RVV plugin-local realization -> provider-built
  `TCRVEmitCLowerableRoute` -> target artifact validation.
* Production provider facts for each included route must expose and target
  validation must consume:
  * runtime ABI order and exported parameter roles;
  * source element type and widened accumulator/result element type;
  * source SEW/LMUL, accumulator/result SEW/LMUL, tail policy, mask policy,
    runtime AVL/VL control, setvl placement, and with-vl scope;
  * typed compute op: `tcrv_rvv.widening_dot_reduce` or
    `tcrv_rvv.masked_widening_dot_reduce`;
  * source/RHS/reduction accumulator/result/runtime-count logical roles;
  * widening product arithmetic relation and reduction intrinsic/result store
    facts;
  * scalar seed splat, reduction store VL, seed/carry/result layout;
  * unit-stride versus strided-input source memory form and stride-source
    facts where active;
  * computed-mask producer/source/form, predicate kind, inactive-lane zeroing,
    mask type/C type, masked product, and merge facts where active;
  * route operand binding plan and exact summary with `abi` and `hdr`
    participation for every exported runtime ABI parameter;
  * required header declarations and C type mapping for narrow inputs and wide
    result;
  * target leaf profile and explicit `provider_supported_mirror`.
* Validation must reject stale widening MAcc facts on dot-reduce, stale
  strided facts on unit-stride dot-reduce, stale unit-stride facts on
  strided-input dot-reduce, missing reduction facts, missing widened result
  facts, stale source/result type mapping, stale arithmetic/reduction kind,
  stale operand binding, stale header/type mapping, stale target profile,
  stale provider mirror, stale computed-mask facts on non-computed-mask routes,
  stale non-mask facts on computed-mask routes, and accidental MAcc,
  standalone-reduction, segment, indexed, or source-front-door fallback.
* Common EmitC/export must remain neutral. It may carry provider-built payloads
  and metadata mirrors unchanged, but must not infer RVV dot-reduce semantics.
* Preserve existing explicit and pre-realized generated-bundle support.

## Acceptance Criteria

* [x] Production RVV provider exposes a provider-owned widening dot-reduce
      route facts accessor covering `widening_dot_reduce_add`,
      `strided_input_widening_dot_reduce_add`, and computed-mask variants that
      are already in the coherent route fact surface.
* [x] Provider-side route-family verification compares selected route
      descriptions against the provider-owned dot-reduce facts before route
      construction.
* [x] Target validation consumes those provider facts and removes or bypasses
      duplicate target-local widening-dot ABI/binding/header/type/profile
      constants for shared provider/target fields.
* [x] Target validation rejects stale unit/strided cross-contamination,
      missing reduction/result facts, stale source/result type mapping, stale
      mask facts, stale binding summary, stale header/type facts, stale target
      profile, stale provider mirror, and stale candidate metadata mirrors.
* [x] Focused C++ target artifact tests prove the new fail-closed boundary.
* [x] Existing lit/script dry-runs for explicit and pre-realized
      widening-dot-reduce routes still pass for touched variants.
* [x] No source-front-door positive route, descriptor-driven computation,
      common EmitC semantic inference, route-id authority, artifact-name
      authority, exact intrinsic spelling authority, or legacy i32 route
      authority is introduced.
* [x] Focused checks, `git diff --check`, and bounded old-authority scan
      complete this round.
* [x] Trellis finish/archive and one coherent commit complete this round if
      the module behavior is complete.

## Technical Approach

1. Replace the narrow
   `RVVComputedMaskStridedInputWideningDotReduceRouteFacts` surface with a
   provider-owned widening dot-reduce facts surface that can represent all
   active dot-reduce variants.
2. Populate route facts from the same typed config/runtime constants already
   used by the contraction route-family plan, including unit-stride, strided,
   computed-mask, and computed-mask-strided distinctions.
3. Rewire provider-side contraction verification to compare dot-reduce route
   descriptions against the new facts surface before route construction.
4. Rewire target artifact validation and candidate mirror validation to use
   provider-owned facts instead of target-local dot-reduce constants.
5. Add focused C++ mutations in `TargetArtifactExportTest.cpp` for stale or
   missing dot-reduce provider facts and candidate mirrors.
6. Run the smallest focused build, C++ target test, lit filters, generated
   bundle dry-runs, bounded old-authority scan, and diff check required by the
   changed validation boundary.

## Out Of Scope

* Standalone reductions, non-widening reductions, dtype/LMUL clone batches,
  segment/indexed routes, high-level frontend lowering, source-front-door
  routes, global performance tuning, dashboards, or evidence-only packaging.
* Redoing widening MAcc production validation as this round's owner.
* Changing generated bundle runtime ABI semantics or route emission unless
  required by a focused validation defect.
* Moving RVV semantics into common EmitC/export.
* Treating route ids, artifact names, manifests, descriptors, C strings, test
  names, spec prose, or mirror metadata as route authority.

## Evidence Plan

* Build `tianchenrv-target-artifact-export-test`.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* Run lit filters for `widening-dot-reduce-add` and
  `strided-input-widening-dot-reduce-add`, plus computed-mask variants if
  touched.
* Run generated-bundle dry-runs for explicit/pre-realized dot-reduce forms
  covered by the changed validation surface.
* Run direct fail-closed C++ checks for stale or missing provider facts and
  candidate mirrors.
* Run bounded old-authority scans over touched source/test files for legacy
  i32 route authority, descriptor/source-front-door/source-artifact/direct-C
  residue, route-id/artifact-name authority, and mirror-only authority.
* Run `rtk git diff --check`.
* Do not rerun `ssh rvv` unless route emission, generated bundle behavior, or
  runtime ABI semantics change. If this round only tightens production
  provider-to-target validation, reuse archived dot-reduce runtime evidence
  and state that no new runtime claim changed.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant archived tasks read:

* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-widening-macc-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-widening-dot-reduce-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-strided-input-widening-dot-reduce-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-widening-dot-reduce-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-strided-input-widening-dot-reduce-artifact-abi/prd.md`

Initial implementation focus:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/RVV/*widening-dot-reduce-add*.mlir`
* `test/Scripts/*widening-dot-reduce-add*.test`
* `test/Target/TargetArtifactExportTest.cpp`

## Implementation Outcome

* Replaced the narrow computed-mask-strided-only dot-reduce facts surface with
  `RVVWideningDotReduceRouteFacts`, covering the plain, strided-input,
  computed-mask, and computed-mask-strided widening dot-reduce route variants.
* Populated the provider-owned facts with operation/memory form, source and
  widened result type facts, SEW/LMUL, tail/mask/runtime control, ABI order and
  parameters, operand binding plan/summary, header/type mapping, target leaf
  profile, stride facts, reduction/product facts, and computed-mask facts only
  for computed-mask variants.
* Rewired provider-side contraction verification so widening dot-reduce
  selected route descriptions must match the canonical provider facts before
  route construction.
* Rewired RVV target artifact validation to consume the provider facts for
  dot-reduce ABI, route payload, target profile, header/type, binding, and
  candidate mirror checks instead of target-local dot-reduce constants.
* Extended `TargetArtifactExportTest.cpp` to compare all four dot-reduce route
  descriptions against the canonical provider facts and preserve fail-closed
  mutation coverage.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the durable
  `RVVWideningDotReduceRouteFacts` provider/target contract.
* Repaired the explicit strided-input generated-bundle dry-run FileCheck
  harness assertions to match the generated harness shape: stride/data case
  tables are in `main`, while kernel invocation and scalar reference checks are
  in `run_case`.

## Evidence Results

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
  passed.
* `rtk build/bin/tianchenrv-target-artifact-export-test` passed.
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter widening-dot-reduce-add`
  from `build/test` passed: 20 selected tests passed, 457 excluded.
* `rtk git diff --check` passed.
* Touched-file old-authority scan found only existing negative tests and
  pre-existing legacy/intrinsic coverage in `TargetArtifactExportTest.cpp`.
* Diff-only old-authority scan over the changed files found no newly introduced
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-artifact, descriptor, direct-C/source-export, or
  exact `__riscv_*_i32m1` authority strings.
* No new `ssh rvv` run was performed because this round tightened
  provider-to-target validation and corrected a stale dry-run assertion; route
  emission, generated runtime ABI semantics, and correctness behavior were not
  changed. This round reuses archived dot-reduce runtime evidence for counts
  including `0,1,16,17,257`.
