# Stage2 RVV widening conversion production validation boundary

## Goal

Close the production provider-to-target validation boundary for the existing
selected-body RVV widening conversion routes `widen_i32_to_i64` and
`widen_i16_to_i32`.

The RVV provider must expose canonical widening conversion route facts derived
from typed `tcrv_rvv` body/config/runtime facts. Target artifact validation
must consume those provider-owned facts for source/result dtype policy,
conversion relation, runtime ABI, memory form, policy, headers/types, target
profile, route operand binding, and explicit provider mirrors instead of
rebuilding or trusting route ids, artifact names, fixture names, C strings,
intrinsic spellings, descriptor residue, or metadata mirrors.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV widening conversion production validation boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no file entries, so the worktree started
  clean.
* Initial `git log --oneline -8` started at
  `441bbb85 rvv: validate standalone reduction route facts`.
* There was no `.trellis/.current-task`, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md` requires the RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned realization/provider facts -> provider-built
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence when runtime/correctness/performance is claimed.
* `.trellis/spec/extension-plugins/rvv-plugin.md` already requires widening
  conversion provider preflight for both `widen_i32_to_i64` and
  `widen_i16_to_i32`.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines
  `RVVWideningConversionRouteFacts` as the provider-owned fact surface, but
  the current durable text focuses on `widen_i32_to_i64`.
* Archived widening conversion artifact/realization tasks already proved:
  selected-body realization into setvl/with_vl/load/widen/store for both
  supported cases, generated bundle dry-runs, direct pre-realized fail-closed
  behavior, and real `ssh rvv` correctness evidence for counts
  `0,1,16,23,257`.
* Archived `widen_i32_to_i64` production work added
  `RVVWideningConversionRouteFacts` and target validation consumption, but live
  inspection shows the fact surface does not yet directly expose all facts named
  by this brief for both routes: source/result element type, tail/mask policy,
  source/destination memory forms, and conversion kind.
* Live target validation already calls
  `getRVVWideningConversionRouteFacts(description.operation)`, so the correct
  implementation path is to complete that provider-owned fact surface and make
  validation compare against it.

## Requirements

* Keep scope to the existing `widen_i32_to_i64` and `widen_i16_to_i32`
  selected-body widening conversion routes.
* Preserve the production chain:
  selected `tcrv.exec` RVV variant -> typed/pre-realized widening conversion
  body -> RVV plugin-local realization -> realized `tcrv_rvv` body -> RVV
  route/provider facts -> `TCRVEmitCLowerableRoute` -> target artifact
  validation.
* The provider-owned widening conversion fact surface must expose and provider
  planning / target validation must consume:
  * operation and typed compute op;
  * conversion kind and conversion relation;
  * source element type, result element type, source/result SEW, source/result
    LMUL, source/result vector type names, and source/result C vector types;
  * tail policy, mask policy, runtime AVL/VL control plan, runtime ABI order,
    and runtime ABI parameter roles for `lhs`, `out`, and `n`;
  * source memory form and destination memory form for the unit-stride
    conversion body;
  * source-load, conversion, store, setvl, VL C type, result name, required
    headers, and C type mapping facts;
  * route-family plan, route operand binding plan/summary, target leaf profile,
    and explicit `provider_supported_mirror`.
* Provider-side route-family plan derivation must copy or validate shared
  widening conversion constants through `getRVVWideningConversionRouteFacts`.
* Target artifact validation must reject stale or missing provider-owned facts:
  stale reduction/MAcc/dot-reduce/memory/scalar-splat route-family facts,
  stale i32-to-i64 facts on i16-to-i32 and vice versa, stale source/result
  element type, stale SEW or LMUL, stale tail/mask policy, stale conversion
  kind/relation, stale source/destination memory form, stale operand binding,
  stale header/type mapping, stale target profile, stale provider mirror, and
  accidental segment/indexed/source-front-door fallback.
* Common EmitC/export remains neutral. It may carry provider-built payloads and
  metadata mirrors, but must not infer RVV conversion semantics.

## Acceptance Criteria

* [x] `RVVWideningConversionRouteFacts` exposes the complete provider-owned
      widening conversion validation surface for both `widen_i32_to_i64` and
      `widen_i16_to_i32`.
* [x] Provider route-family plan derivation/validation copies or validates
      element type, SEW/LMUL, policy, conversion, memory form, runtime ABI,
      binding, header/type, target profile, and provider mirror facts from the
      canonical fact accessor.
* [x] Target artifact validation consumes the same accessor and rejects stale
      cross-case, cross-family, policy, memory-form, dtype, conversion,
      binding, header/type, target profile, and provider mirror facts before
      accepting a bundle.
* [x] Candidate metadata mirror validation includes provider-derived conversion
      source/destination memory form and remains fail-closed for non-conversion
      route-family mirrors.
* [x] Focused C++ target artifact tests prove the new fail-closed boundary for
      both supported widening conversion routes, including stale
      `widen_i32_to_i64` facts on `widen_i16_to_i32` and vice versa.
* [x] Existing lit/script dry-runs for explicit and pre-realized
      `widen_i32_to_i64` and `widen_i16_to_i32` still pass.
* [x] No source-front-door positive route, descriptor-driven computation,
      common EmitC semantic inference, route-id authority, artifact-name
      authority, exact intrinsic spelling authority, or legacy i32 route
      authority is introduced.
* [x] Focused checks, `git diff --check`, and bounded old-authority scan
      complete this round.
* [x] Trellis finish/archive and one coherent commit complete this round if
      the module behavior is complete.

## Technical Approach

1. Extend `RVVWideningConversionRouteFacts` with the missing provider-owned
   fields named by this PRD: source/result element type, tail/mask policy,
   conversion kind, source memory form, destination memory form, and runtime ABI
   parameter facts.
2. Populate those fields in the existing canonical fact builder for both
   `widen_i32_to_i64` and `widen_i16_to_i32`.
3. Extend the widening conversion route-family plan/description propagation so
   target validation can compare source/result element type and memory-form
   facts directly instead of inferring them from vector type strings.
4. Rewire target artifact validation and candidate mirror checks to compare the
   added provider facts and reject stale cross-case or cross-family conversion
   payloads.
5. Add focused C++ mutations in `TargetArtifactExportTest.cpp` for the added
   validation boundary.
6. Update the durable EmitC route spec only if the implementation establishes a
   stronger contract that was not already documented.

## Out Of Scope

* Narrowing conversions, saturating conversions, float conversions, unsigned
  conversion batches, new dtype/LMUL clone families, reductions, dot-reduce
  changes, MAcc changes, segment/indexed route changes, source-front-door
  routes, high-level frontend lowering, global tuning, dashboards, or
  evidence-only packaging.
* Redoing standalone reduction production validation as this round's owner.
* Changing generated bundle runtime ABI order, conversion semantics, or emitted
  route behavior unless required by a focused validation defect.
* Moving RVV conversion semantics into common EmitC/export.
* Treating route ids, artifact names, manifests, descriptors, C strings, test
  names, spec prose, intrinsic strings, or mirror metadata as route authority.

## Evidence Plan

* Build `tianchenrv-target-artifact-export-test`.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* Run lit filters for `widen-i32-to-i64` and `widen-i16-to-i32`.
* Run generated-bundle dry-runs for explicit and pre-realized widening
  conversion forms if fixtures are touched or metadata mirrors change.
* Run direct fail-closed C++ checks for stale or missing
  widening-conversion/provider facts and candidate mirrors.
* Run bounded old-authority scans over touched source/test/spec/task files for
  legacy i32 route authority, descriptor/source-front-door/source-artifact/
  direct-C residue, route-id/artifact-name authority, and mirror-only
  authority.
* Run `rtk git diff --check`.
* Do not rerun `ssh rvv` unless route emission, generated bundle behavior,
  runtime ABI order, or conversion semantics change. If this round only
  tightens production provider-to-target fact ownership and validation, reuse
  archived widening conversion runtime evidence and state that no new runtime
  claim changed.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant archived tasks read:

* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-standalone-reduction-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-widening-conversion-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-01-stage2-rvv-widening-conversion-realization-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-widen-i32-to-i64-conversion-artifact-abi-boundary/prd.md`

Initial implementation focus:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* `test/Target/RVV/*widen-i32-to-i64*.mlir`
* `test/Target/RVV/*widen-i16-to-i32*.mlir`
* `test/Scripts/*widen-i32-to-i64*.test`
* `test/Scripts/*widen-i16-to-i32*.test`

## Implementation Outcome

Completed on 2026-06-03.

Production changes:

* Extended `RVVWideningConversionRouteFacts` with source/result element type,
  tail/mask policy, runtime control plan, conversion kind, source/destination
  memory form, and canonical runtime ABI parameter facts for both supported
  widening conversion routes.
* Rewired widening conversion route-family plan derivation, validation,
  description propagation, construction metadata validation, and artifact
  metadata emission to consume the completed provider-owned fact surface.
* Rewired target artifact validation so runtime ABI order/roles/C types,
  source/result dtype policy, conversion kind/relation, policy, memory forms,
  target profile, header/type mapping, binding summary, and provider mirrors
  are compared against `getRVVWideningConversionRouteFacts(...)`.
* Added conversion candidate mirror checks for source/result element type,
  conversion kind, source/destination memory form, and tail/mask policy.
* Strengthened `TargetArtifactExportTest.cpp` with canonical fact assertions
  for both supported conversions and fail-closed mutations for stale
  cross-case facts, stale policy, stale memory forms, stale element types,
  stale conversion kind/relation, and stale candidate mirrors.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` to record the
  widened conversion fact-surface contract.

Self-repair:

* Fixed an initial header propagation miss where `runtimeControlPlanID` was not
  added to the widening conversion plan struct.
* Fixed a target validation Twine concatenation error caught by the focused
  build.
* Fixed construction metadata validation so widening conversion source and
  destination memory forms are checked against provider facts instead of an
  empty route-profile default.
* Updated cross-case test expectations to match the validator's intended
  fail-closed order: stale conversion kind is rejected before the later dtype
  comparison.

## Evidence Results

Focused checks:

* [x] `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-03-stage2-rvv-widening-conversion-production-validation-boundary`
* [x] `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
* [x] `rtk build/bin/tianchenrv-target-artifact-export-test`
* [x] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter widen-i32-to-i64` from `build/test`: 5 selected tests passed, 472 excluded.
* [x] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter widen-i16-to-i32` from `build/test`: 3 selected tests passed, 474 excluded.
* [x] `rtk git diff --check`
* [x] Added-diff old-authority scan over touched files found no newly
      introduced `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, exact `__riscv_*_i32m1`,
      source-front-door/source-artifact, descriptor/direct-C/source-export,
      route-id, or artifact-name authority. Full-file hits in specs/tests are
      existing prohibitions, negative tests, legacy guardrails, or
      provider-derived emitted-C evidence checks.

Real RVV evidence:

* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-03-stage2-rvv-widening-conversion-production-validation-boundary/ssh-rvv --run-id pre-realized-widen-i16-to-i32 --overwrite --op-kind widen_i16_to_i32 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 900 --connect-timeout 30`
  passed. Remote output reported all counts and both patterns OK with
  `sign_extension_checked`, `two_input_patterns_checked`, and
  `tail_preserved`, ending with
  `PASS op=widen_i16_to_i32 counts=0,1,16,17,257`.
* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/06-03-stage2-rvv-widening-conversion-production-validation-boundary/ssh-rvv --run-id pre-realized-widen-i32-to-i64 --overwrite --op-kind widen_i32_to_i64 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 900 --connect-timeout 30`
  passed. Remote output reported all counts and both patterns OK with
  `sign_extension_checked`, `wide_magnitude_checked`,
  `two_input_patterns_checked`, and `tail_preserved`, ending with
  `PASS op=widen_i32_to_i64 counts=0,1,16,17,257`.

Runtime note:

* This round changed provider-to-target validation and metadata mirror
  coverage, not the emitted conversion loop, runtime ABI order, or conversion
  semantics. `ssh rvv` was rerun anyway because generated bundle metadata
  behavior changed, and both supported routes passed the requested counts.
