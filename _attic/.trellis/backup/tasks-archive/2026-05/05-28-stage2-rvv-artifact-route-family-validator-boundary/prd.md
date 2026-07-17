# Stage2 RVV artifact route-family validator boundary

## Goal

Introduce an explicit RVV target-owned artifact route-family validation boundary and migrate the widening-dot reduction artifact family onto it. The artifact bridge must dispatch widening-dot artifact candidates through this new family validator instead of growing monolithic ad hoc checks in `RVVTargetSupportBundle.cpp`, while preserving the provider-derived route fact authority closed by commit `38573fbd`.

## What I already know

* Current HEAD is `38573fbd rvv: close widening dot artifact authority`; the worktree was clean at task creation.
* The previous task closed widening-dot artifact authority by requiring executable widening-dot artifact acceptance to depend on rebuilt `TCRVEmitCLowerableRoute` provider facts and mirror validation, not artifact metadata, route-entry, descriptor, intrinsic spelling, ABI strings, or artifact names.
* The current bottleneck is boundary shape: widening-dot target artifact consumer logic currently lives in a growing central `lib/Target/RVV/RVVTargetSupportBundle.cpp` implementation.
* This task should migrate one real family first: widening-dot reduction. Other route families may remain on existing code if the migrated boundary has a real production consumer and truthful continuation notes.
* Runtime/correctness/performance claims still require real `ssh rvv` evidence. Generated dry-run artifact evidence is expected for the four widening-dot cases and must still show `route_entry_realization: false`.

## Assumptions

* A small target-owned validator registry/dispatch layer inside the RVV target implementation is sufficient for this round; it does not need to rewrite every target artifact validator family.
* Existing widening-dot tests and generated-bundle evidence should remain the acceptance baseline unless live repository evidence shows a safer narrower check set.
* The bridge may keep neutral candidate discovery, artifact loading, and dispatch mechanics, but RVV semantic family checks for widening-dot should move behind the new family validator interface.

## Requirements

* Add or repair a production RVV target-owned route-family artifact validation boundary with a clear registry/dispatch surface.
* Invoke the widening-dot validator from artifact export/bridge production code for widening-dot candidates.
* Preserve all widening-dot authority guarantees from `38573fbd`: validation must consume rebuilt provider route facts and treat artifact metadata only as mirrors.
* Fail closed when provider support labels, binding plan, ABI order/mappings, headers, C type mappings, widening-dot relation, reduction store VL, computed-mask facts, stride facts, or provider-built statement plan are missing or mismatched.
* Keep common EmitC/export neutral; do not move RVV semantic validation into common EmitC/export code.
* Do not add new RVV op coverage, dtype/LMUL clone batches, high-level frontend lowering, source-front-door routes, descriptor-driven computation, direct route-entry compatibility, or one-intrinsic wrapper dialects.

## Acceptance Criteria

* [x] `RVVTargetSupportBundle.cpp` no longer owns the migrated widening-dot family as duplicated central ad hoc checks; it dispatches through a target-owned route-family validator boundary.
* [x] The new boundary has a real production consumer and fail-closed diagnostics for unsupported/mismatched widening-dot provider facts.
* [x] The widening-dot validator checks route id mirror, provider support mirror, ABI order/mappings, required headers, VL/source/result/mask C type mappings, i16mf2-to-i32m1 widening-dot relation, accumulator/result layout, reduction-store VL, computed-mask facts, stride facts, and provider-built pre-loop/loop statement plan using rebuilt route/provider facts.
* [x] Existing stale mirror fail-closed coverage still covers provider support, binding plan, ABI order, headers, C type mapping, contraction/widening-dot relation, reduction store VL, strided load intrinsic, and masked widening product intrinsic.
* [x] Generated-bundle dry-runs for all four widening-dot cases still report `route_entry_realization: false`.
* [x] Deprecated direct shortcut remains fail-closed.
* [x] `ssh rvv` evidence is collected for `computed_masked_strided_input_widening_dot_reduce_add` and plain `widening_dot_reduce_add`, or an exact blocker is recorded.
* [x] Focused target artifact export and RVV plugin tests pass; `check-tianchenrv` passes or an exact blocker is recorded.
* [x] Authority-leak scan confirms no executable artifact claim depends on central ad hoc, name-derived, metadata-derived, descriptor-derived, ABI-string-derived, script-derived, artifact-name-derived, common-EmitC-derived, source-front-door-derived, route-id-derived, exact-intrinsic-derived, direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived authority.
* [x] `git diff --check` passes and final `git status --short` is clean after commit.

## Out of Scope

* Rewriting all RVV target artifact validators.
* Adding Stage2 coverage, new widening-dot variants, dtype/LMUL clone batches, Linalg/frontend lowering, IME/Offload/TensorExt work, dashboard/reporting, broad smoke matrices, or evidence-only changes.
* Treating manifests, semantic role graphs, construction templates, route ids, artifact metadata, or intrinsic spellings as executable authority.

## Technical Notes

* Required specs to read before implementation:
  * `.trellis/spec/index.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/lowering-runtime/emitc-route.md`
  * `.trellis/spec/core-dialect/tcrv-exec-contract.md`
* Required prior task/code context:
  * archived task `05-28-05-28-stage2-rvv-widening-dot-artifact-authority`
  * `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  * `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  * `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  * `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  * directly related widening-dot target tests and generated-bundle evidence

## Round Result

Implemented a target-owned RVV route-family artifact validation boundary in:

* `include/TianChenRV/Target/RVV/RVVTargetArtifactRouteFamilyValidation.h`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`

`RVVTargetSupportBundle.cpp` now keeps generic target artifact bridge work and
dispatches provider-fact validation plus candidate mirror validation through
the route-family boundary. The first registered production family is
`widening-dot-reduction`.

Moved into the widening-dot family validator:

* route id mirror and provider-supported mirror checks;
* route operand binding, ABI order/mapping, headers, and C type mapping checks;
* i16mf2 source to i32m1 result relation checks;
* accumulator/result layout, widening-dot relation, and reduction store VL
  checks;
* computed-mask and strided-input subfamily fact checks;
* provider-built pre-loop and loop statement-plan checks;
* candidate metadata mirror checks for contraction family plan, memory form,
  runtime control, ABI order, headers, type mapping, target leaf profile,
  widening-dot relation/layout/store VL, strided load intrinsic, and masked
  widening product intrinsic.

Intentionally kept in the bridge:

* selected variant discovery and typed body route rebuild;
* generic candidate/exporter shape checks;
* route metadata/body mismatch checks;
* source-op provenance and descriptor/direct-C/source-export residue rejection;
* candidate runtime ABI parameter consistency with rebuilt route ABI mappings;
* neutral target artifact packaging/export mechanics.

Evidence:

* `git diff --check`: passed.
* `cmake --build build --target TianChenRVTarget tcrv-translate tcrv-opt`:
  passed.
* `cmake --build build --target tianchenrv-target-artifact-export-test &&
  build/bin/tianchenrv-target-artifact-export-test`: passed.
* `cmake --build build --target tianchenrv-rvv-extension-plugin-test &&
  build/bin/tianchenrv-rvv-extension-plugin-test`: passed.
* Four-case dry-run evidence:
  `artifacts/tmp/05-28-stage2-rvv-artifact-route-family-validator-boundary/widening-dot-dry-run/pre-realized-widening-dot-four-cases/`
  with `local_bundle_generation.route_entry_realization = false` for all four
  widening-dot cases.
* Deprecated direct pre-realized shortcut fail-closed:
  `artifacts/tmp/05-28-stage2-rvv-artifact-route-family-validator-boundary/widening-dot-dry-run/direct-pre-realized-widening-dot-fail-closed`
  with the expected unsupported direct-route-entry error.
* `ssh rvv` evidence:
  `artifacts/tmp/05-28-stage2-rvv-artifact-route-family-validator-boundary/widening-dot-ssh/pre-realized-widening-dot-runtime/`
  passed for `computed_masked_strided_input_widening_dot_reduce_add` and
  `widening_dot_reduce_add`.
* `cmake --build build --target check-tianchenrv`: passed 456/456 after
  cleaning a concurrent lit-output collision caused by accidentally starting
  duplicate check instances.
* Bounded authority scan over the touched target files and task PRD found only
  expected PRD guardrail text, rebuilt-provider route-id mirror diagnostics,
  and existing descriptor/direct-C rejection code; no executable widening-dot
  claim was moved to metadata, descriptor, ABI string, route-entry, source
  front-door, exact intrinsic, common EmitC, artifact name, script, or legacy
  i32 route authority.

Spec update:

* Added `.trellis/spec/extension-plugins/rvv-plugin.md` target artifact
  route-family validator boundary guidance as a durable code-spec rule.
