# Stage2 RVV pre-realized composite family-fact fail-closed owner boundary

## Goal

Harden the RVV plugin-local selected-body realization owner for the
pre-realized runtime-scalar computed-mask indexed gather-MAcc-scatter
composite so stale family-body facts fail closed before provider route
analysis, `TCRVEmitCLowerableRoute` construction, Common EmitC materialization,
target artifact export, or executable ABI claims.

This round's bounded owner path is:

```text
selected pre-realized gather/MAcc/scatter family bodies
  -> RVV composite selected-body realization owner
  -> realized explicit tcrv_rvv.with_vl body
  -> RVVCompositeGatherMAccScatterRouteFamilyPlan
  -> TCRVEmitCLowerableRoute
```

## What I Already Know

* Current HEAD starts at `1f2da1ac` with a clean worktree.
* The previous archived task proved the positive pre-realized composite
  executable ABI path and updated the RVV plugin spec to make this owner
  positive instead of unsupported.
* `.trellis/spec/extension-plugins/rvv-plugin.md` now requires missing,
  duplicate, incomplete, stale, or unsupported pre-realized gather/MAcc/scatter
  family facts to fail closed with named owner-boundary diagnostics.
* The current owner already validates many stale cross-family facts after the
  composite owner is selected, including stale gather/scatter index value.
* The current owner candidate gate only selects the composite owner when one
  gather, one MAcc, and one scatter body are all present. Partial clusters such
  as gather+MAcc without scatter can fall through to the generic selected-body
  registry error instead of the named composite owner boundary.

## Requirements

* Route any selected variant containing a multi-body cluster of composite
  gather/MAcc/scatter pre-realized family bodies into the composite owner, even
  when the cluster is incomplete or has duplicates.
* Preserve standalone single-family pre-realized bodies, such as a single
  computed-mask memory or computed-mask MAcc body, so they still use their
  existing family owner.
* Make incomplete or duplicate composite family-body clusters fail closed with a
  targeted diagnostic that names the composite owner boundary and identifies
  gather/MAcc/scatter counts.
* Keep the positive pre-realized and explicit composite paths unchanged:
  exactly one gather, one MAcc, and one scatter family body must still realize
  into the explicit `setvl`/`with_vl` body and then reach provider route facts.
* Do not infer compute semantics from route ids, artifact names, ABI strings,
  helper names, Common EmitC, target metadata, or scripts.

## Acceptance Criteria

* [x] Production owner-boundary code selects the composite owner for
      incomplete or duplicate multi-family composite clusters before provider
      route construction.
* [x] Focused C++ fail-closed coverage proves a missing family body is rejected
      by the named composite owner boundary.
* [x] Focused C++ fail-closed coverage proves a duplicate family body is
      rejected by the named composite owner boundary.
* [x] Existing stale cross-family fact coverage, including stale index between
      gather and scatter, remains in place.
* [x] Positive pre-realized composite realization still produces an explicit
      body and reaches the same provider-owned route contract as the explicit
      selected body.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant lit filter for
      `runtime-scalar-cmp-masked-indexed-gather-macc-scatter` passes.
* [x] `git diff --check` and `git diff --cached --check` pass.
* [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor-driven compute, source-front-door, or
      Common EmitC semantic authority.

## Completion Notes

* Updated the composite selected-body realization owner to count gather, MAcc,
  and scatter pre-realized family bodies separately.
* Changed the composite owner candidate gate so variants with more than one
  bounded gather/MAcc/scatter pre-realized family body are routed to the
  composite owner. Single-family bodies remain available to the existing
  standalone family owners.
* Incomplete or duplicate multi-body clusters now fail with a named
  `Stage2 RVV composite gather-MAcc-scatter selected-body realization owner`
  diagnostic that reports gather/MAcc/scatter counts before route/provider,
  Common EmitC, or target export can claim support.
* Added C++ regressions for missing scatter and duplicate gather clusters, and
  preserved the existing stale-index owner-boundary regression and positive
  pre-realized-to-explicit route contract regression.
* Updated the RVV plugin spec with the executable owner-gate rule and required
  missing/duplicate count diagnostics.

## Validation Evidence

* `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
  passed.
* `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
* `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
  passed.
* `build/bin/tianchenrv-target-artifact-export-test` passed.
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='runtime-scalar-cmp-masked-indexed-gather-macc-scatter'`
  from `build/test` passed: 3 selected, 3 passed.
* `git diff --check` passed.
* `git diff --cached --check` passed.
* Bounded old-authority scan over touched files found only pre-existing
  negative legacy tests and this PRD's explicit non-goal/prohibition text;
  added source diff lines contain no new positive legacy authority.

## Out Of Scope

* New positive evidence-only tasks, broad smoke matrices, or ssh runtime claims.
* New composite route families, unrelated memory/segment/reduction/dtype/LMUL
  expansion, high-level frontend work, source-front-door positives, scalar
  fallback implementation, IME/offload/TensorExt work, or descriptor-driven
  computation.
* Weakening provider route-family plan, Common EmitC, or target artifact
  validation in order to accept stale metadata.

## Technical Notes

* Direction source: Hermes/Codex worker direction brief supplied on 2026-06-07.
* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/variant-pipeline/index.md`,
  `.trellis/spec/testing/index.md`.
* Previous task read:
  `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-pre-realized-composite-executable-abi-closure/prd.md`.
* Primary files:
  `include/TianChenRV/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`,
  `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`.
