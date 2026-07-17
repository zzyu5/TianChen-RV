# Stage2 RVV widening MAcc production validation boundary

## Goal

Close the production provider-to-target validation boundary for the existing
`widening_macc_add` route.

This round must make source element type, widened accumulator/result type,
widening arithmetic kind, accumulator/source/RHS roles, SEW/LMUL/policy,
runtime ABI roles, route-family plan, type/header participation, target leaf
profile, and fail-closed diagnostics come from provider-owned typed
`tcrv_rvv` body/config/runtime facts. Target artifact validation must not
accept widening MAcc from route ids, artifact names, test fixtures, candidate
metadata mirrors, descriptors, C strings, scripts, exact intrinsic spellings,
or common EmitC/export inference.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV widening MAcc production validation boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` returned a clean short status through RTK.
* Initial `git log --oneline -8` started at
  `78a38162 rvv: validate computed-mask macc route facts`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* The immediately previous committed task closed computed-mask MAcc production
  validation, including parameterized runtime-scalar computed-mask MAcc facts.
* The archived signed widening MAcc ABI task already implemented the bounded
  executable relation and recorded real `ssh rvv` correctness for counts
  `0,1,7,16,23,257` over two input patterns.
* The current specs already name `widening_macc_add` in the RVV route-family
  fact surface contract and require provider-owned facts for the i16mf2 source
  and i32m1 accumulator/result relation.

## Requirements

* Keep scope to the existing signed widening relation:
  `i16mf2 lhs x i16mf2 rhs + i32m1 accumulator -> i32m1 result`.
* Preserve the production chain:
  selected `tcrv.exec` RVV variant -> typed/pre-realized `widening_macc_add`
  body -> RVV plugin-local realization -> provider-built
  `TCRVEmitCLowerableRoute` -> target artifact validation.
* Production provider facts for `widening_macc_add` must expose and target
  validation must consume:
  * runtime ABI order and exported parameter roles for `lhs`, `rhs`, `acc`,
    `out`, and `n`;
  * source element type and accumulator/result widened element type;
  * source SEW/LMUL, accumulator/result SEW/LMUL, tail policy, mask policy,
    runtime AVL/VL control, setvl placement, and with-vl scope;
  * typed compute op `tcrv_rvv.widening_macc`;
  * widening multiply-add arithmetic kind and signed widening relation;
  * accumulator/source/RHS/output/runtime-count logical roles;
  * source, RHS, accumulator, and destination memory forms;
  * accumulator layout and result layout;
  * direct contraction or widening MAcc route-family plan as provider output;
  * route operand binding plan and exact summary with `abi` and `hdr`
    participation for every exported runtime ABI parameter;
  * required header declarations and C type mapping for narrow inputs and
    wide accumulator/result;
  * target leaf profile and explicit `provider_supported_mirror`.
* Validation must reject stale unit-stride MAcc facts on widening MAcc, stale
  computed-mask MAcc facts, missing widened accumulator facts, stale
  source/result type mapping, stale arithmetic kind, stale operand binding,
  stale header/type mapping, stale target profile, stale provider mirror, and
  accidental scalar-broadcast, computed-mask, segment, indexed, reduction, or
  widening-dot fallback.
* Keep common EmitC/export neutral. It may carry provider-built payloads and
  mirrors unchanged, but must not infer RVV widening semantics.
* Preserve existing explicit and pre-realized generated-bundle support.

## Acceptance Criteria

* [x] Production RVV provider exposes provider-owned `widening_macc_add`
      route facts for runtime ABI roles, source/wide types, SEW/LMUL/policy,
      typed compute op, arithmetic kind, memory forms, accumulator/result
      layouts, route-family plan, binding summary, headers/types, target
      profile, and provider mirror.
* [x] Target validation consumes those provider facts and removes or bypasses
      duplicate target-local widening MAcc truth for shared provider/target
      fields.
* [x] Target validation rejects stale unit-stride MAcc, computed-mask MAcc,
      scalar-broadcast, segment/indexed, reduction, and widening-dot residue
      on `widening_macc_add`.
* [x] Target validation rejects missing widened accumulator/result facts,
      stale source/result type mapping, stale arithmetic kind, stale binding
      summary, stale header/type facts, stale target profile, stale provider
      mirror, and stale candidate metadata mirrors.
* [x] Focused C++ target artifact tests prove the new fail-closed boundary.
* [x] Existing lit/script dry-runs for explicit and pre-realized
      `widening_macc_add` still pass.
* [x] No source-front-door positive route, descriptor-driven computation,
      common EmitC semantic inference, route-id authority, artifact-name
      authority, exact intrinsic spelling authority, or legacy i32 route
      authority is introduced.
* [x] Focused checks, `git diff --check`, and bounded old-authority scan
      complete this round.
* [x] Trellis finish/archive and one coherent commit complete this round if
      the module behavior is complete.

## Technical Approach

1. Inspect the current `RVVWideningMAccRouteFacts` surface and the widening
   MAcc plan/target validation consumers.
2. Extend or normalize the provider-owned fact surface only where live code
   lacks fields required by this PRD.
3. Rewire target validation to compare rebuilt route descriptions and
   candidate mirrors against provider-owned widening MAcc facts.
4. Add focused C++ target validation tests that mutate stale/missing provider
   facts and candidate metadata mirrors to prove fail-closed behavior.
5. Run only the focused build, C++ target test, lit filters, generated-bundle
   dry-runs, bounded authority scan, and diff check required by this boundary.

## Implementation Outcome

Completed on 2026-06-03.

* Extended `RVVWideningMAccRouteFacts` with provider-owned element type names,
  tail/mask/runtime-control facts, arithmetic kind, logical operand roles,
  source/RHS/accumulator/destination memory forms, exported runtime ABI
  parameter facts, and binding/header participation facts.
* Added `wideningMAccArithmeticKind` to the contraction route-family plan and
  derived/applied it from the typed `tcrv_rvv.widening_macc` body, alongside
  provider-owned unit-stride source and destination memory forms.
* Kept widening MAcc distinct from strided-input dot reductions in the
  route-control provider plan: widening MAcc may carry unit-stride source/store
  facts, while strided layout and stride-source facts remain rejected.
* Rewired target artifact validation to compare widening MAcc runtime ABI
  parameters, runtime control/policy, arithmetic kind, memory forms,
  header/type facts, target leaf profile, and candidate mirrors against the
  provider-owned facts.
* Added fail-closed C++ coverage for stale accumulator ABI C type, stale
  accumulator/result LMUL, stale arithmetic kind, stale source memory form,
  stale header facts, stale arithmetic/source/header candidate mirrors, and
  computed-mask residue mirrors.
* Updated generated route metadata for widening MAcc with
  `tcrv_rvv.widening_macc_arithmetic_kind`,
  `tcrv_rvv.source_memory_form`, and
  `tcrv_rvv.destination_memory_form`.

## Evidence Results

Focused checks passed:

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter widening-macc-add`
  from `build/test`, which selected and passed 5 widening-MAcc target/script
  tests, including explicit and pre-realized generated-bundle dry-runs and the
  direct pre-realized fail-closed script.
* `rtk git diff --check`
* Bounded old-authority scan over touched source/test files. The scan found
  pre-existing legacy/negative fixtures in `test/Target/TargetArtifactExportTest.cpp`
  and the existing fail-closed `tcrv_rvv.i32_` guard in
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`; a diff-only scan found no
  newly added `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-artifact, or exact i32m1
  intrinsic authority.

No new `ssh rvv` run was required. This round tightened production
provider-to-target validation and added route metadata mirrors, but did not
change generated C runtime ABI semantics, generated bundle execution behavior,
or the signed widening MAcc computation path. The archived signed widening
MAcc ABI task remains the runtime evidence source for counts `0,1,7,16,23,257`
over two input patterns.

## Out Of Scope

* Computed-mask widening MAcc, widening dot/reduction routes, standalone
  reductions, standalone compare/select expansion, segment/indexed routes,
  dtype/LMUL clone batches, source-front-door positive routes, high-level
  frontend lowering, dashboards, autotuning databases, or evidence-only
  packaging.
* Re-implementing selected-body realization or changing generated bundle
  runtime ABI semantics unless a focused check exposes a live defect.
* Redoing unit-stride or computed-mask MAcc production validation as this
  round's owner.
* Moving RVV semantics into common EmitC/export.
* Treating route ids, artifact names, manifests, descriptors, C strings, test
  names, spec prose, or mirror metadata as route authority.

## Evidence Plan

* Build `tianchenrv-target-artifact-export-test`.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* Run lit filters for `widening-macc-add` target/script tests.
* Run generated-bundle dry-runs for explicit and pre-realized
  `widening_macc_add` forms.
* Run direct fail-closed checks for stale or missing widening MAcc provider
  facts and candidate mirrors through focused C++ mutations.
* Run a bounded old-authority scan over touched files for legacy i32 route
  authority, descriptor/source-front-door/source-artifact/direct-C/source
  export residue, route-id/artifact-name authority, and mirror-only authority.
* Run `rtk git diff --check`.
* Do not rerun `ssh rvv` unless route emission, generated bundle behavior, or
  runtime ABI semantics change. If this round only tightens production
  provider-to-target validation, reuse archived signed widening MAcc runtime
  evidence and state that no new runtime claim changed.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant archived tasks read:

* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-computed-mask-macc-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-signed-widening-macc-add-artifact-abi-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-unit-stride-macc-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-macc-realization-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-computed-mask-macc-realization-boundary/prd.md`

Initial implementation focus:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* `test/Target/RVV/*widening-macc-add*.mlir`
* `test/Scripts/*widening-macc-add*.test`
