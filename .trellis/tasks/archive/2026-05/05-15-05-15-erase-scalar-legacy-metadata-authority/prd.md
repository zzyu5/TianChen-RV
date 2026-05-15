# Erase scalar legacy metadata authority

## Goal

Remove the remaining active scalar fallback dependence on deleted legacy
metadata markers so the scalar plugin is driven only by capability facts and
plugin-local selected-variant metadata. The round is deletion-only: no new
scalar ops, no compatibility wrapper, and no replacement runtime route.

## What I already know

* The current scalar plugin still inspects `tcrv_frontend_lowering` on kernels
  and `tcrv_scalar.element_count` on variants.
* The focused scalar plugin test still contains negative cases that expect those
  markers to block proposal or boundary materialization.
* The scalar fallback spec still describes those markers as part of the active
  proposal and legality story instead of pure deleted-route residue.
* The broader lowering-runtime and variant-pipeline specs still mention the old
  markers, so the wording needs to be checked for active authority leaks.
* The plugin already has a capability-driven first slice, metadata-only
  emission, and a selected lowering boundary path that can survive without the
  deleted markers.

## Assumptions

* Scalar fallback should continue to propose and materialize based on
  `scalar.fallback` capability availability and the existing plugin-local
  metadata contract.
* Any reference to the deleted markers should be historical or fail-closed
  residue only, not a live gating rule.

## Open Questions

* None. The direction brief is specific enough to implement directly.

## Requirements

* Remove active code-path dependence on `tcrv_frontend_lowering`.
* Remove active code-path dependence on `tcrv_scalar.element_count`.
* Keep scalar fallback capability-driven and metadata-only.
* Update or delete tests that assert behavior on the deleted markers.
* Update directly related scalar spec text so it no longer describes those
  markers as active authority.
* Keep the change scoped to the scalar plugin, its scalar-specific tests, and
  directly related scalar/variant-pipeline/lowering-runtime spec text.

## Acceptance Criteria

* [ ] `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp` no longer reads or rejects
  `tcrv_frontend_lowering`.
* [ ] `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp` no longer reads or rejects
  `tcrv_scalar.element_count`.
* [ ] `test/Plugin/ScalarExtensionPluginTest.cpp` no longer depends on those
  deleted markers as gating authority.
* [ ] Scalar fallback still proposes, materializes, and emits metadata-only
  results from capability-driven input.
* [ ] Directly related scalar and variant-pipeline spec text no longer treats
  the deleted markers as active authority.
* [ ] Focused scalar plugin build/test targets pass.
* [ ] A focused ref-scan shows no active scalar code/test/spec dependence on the
  deleted markers outside deleted-route or historical wording.

## Definition of Done

* Focused implementation is complete.
* Relevant checks pass.
* Trellis task state is truthful.
* Commit is created.

## Out of Scope

* New scalar execution lowering or runtime ABI work.
* Compatibility wrappers or legacy marker support.
* Broad repo-wide refactors unrelated to scalar fallback authority cleanup.

## Technical Notes

* Relevant spec sources:
  * `.trellis/spec/index.md`
  * `.trellis/spec/extension-plugins/index.md`
  * `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  * `.trellis/spec/lowering-runtime/index.md`
  * `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  * `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  * `.trellis/spec/testing/index.md`
* Relevant code and tests:
  * `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  * `test/Plugin/ScalarExtensionPluginTest.cpp`
  * `include/TianChenRV/Dialect/Scalar/IR/ScalarOps.td`
