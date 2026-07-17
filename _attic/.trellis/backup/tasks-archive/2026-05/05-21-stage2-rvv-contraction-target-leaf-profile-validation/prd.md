# Stage2 RVV contraction target-leaf/profile validation closure

## Goal

Close the target-leaf/profile validation boundary for the existing Stage2 RVV
contraction route family. The five existing contraction routes must carry
target profile, source load leaf, widening product leaf, reduction or macc
leaf, mask/zeroing requirements, store leaf, header/type mapping, and mirror
metadata facts through the validated RVV plugin-owned contraction family plan
before provider emission.

This is a production-path validation and ownership repair. It is not a new
operation, coverage expansion, frontend/linalg task, or metadata-only cleanup.

Covered routes:

- `widening_macc_add`
- `widening_dot_reduce_add`
- `computed_masked_widening_dot_reduce_add`
- `strided_input_widening_dot_reduce_add`
- `computed_masked_strided_input_widening_dot_reduce_add`

## Direction Source

- Hermes Direction Brief: `Stage2 RVV contraction target-leaf/profile validation closure`.
- Module owner: RVV plugin-owned contraction route-family target profile and
  intrinsic-leaf validation boundary for the existing widening macc/dot
  variants.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `1d1c96fa rvv: consolidate contraction route family planning`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.

## Current Repository Facts

- Commit `1d1c96fa` added
  `RVVSelectedBodyContractionRouteFamilyPlan` and migrated the five named
  contraction routes into a shared family plan consumed by the provider.
- The family plan currently carries operation, memory form, mask/stride/seed
  flags, ABI order, source config, mask provenance, stride source, and
  reduction/accumulator relation facts.
- Target leaf/profile validation still lives partly outside the contraction
  family owner: `RVVSelectedBodyTargetLeafProfile` is derived from the generic
  route description/config profile, while exact load/product/reduction/store
  leaf compatibility is still validated later in route-description checks or
  consumed through `RVVSelectedBodyEmitCRouteDescription`.
- Target/header metadata is mirror-only and must stay mirror-only. It may
  reflect provider-derived target leaves/profile facts after route validation,
  but it must not define route support.
- Common EmitC/export must remain neutral. The RVV plugin provider owns
  intrinsic leaves, C/RVV type mappings, ABI ordering, and fail-closed
  diagnostics.

## Requirements

1. Extend the contraction family plan, or an equivalent RVV-plugin-local owner,
   so target profile and leaf facts are explicit validated plan outputs for the
   five named routes.
2. The validated family boundary must include at least:
   target profile label, source load leaf, optional strided source load leaf,
   widening product leaf, optional masked widening product leaf, reduction or
   widening macc compute leaf, scalar seed splat leaf for dot reductions,
   scalar/vector store leaf, inactive-lane zeroing requirement for masked dot
   reductions, header declarations, C type mapping facts, and
   `provider_supported_mirror` metadata.
3. Unsupported target/profile/leaf combinations must fail closed before
   provider emission with targeted diagnostics for invalid target profile,
   unsupported source load leaf, unsupported product leaf, unsupported
   reduction/macc leaf, unsupported store leaf, invalid masked zeroing
   requirement, invalid strided load leaf, and stale route-id authority.
4. Provider emission must consume the validated contraction family plan for
   contraction route leaves and type/header facts. It must not re-derive route
   authority from route ids, helper names, artifact strings, descriptors, exact
   intrinsic strings in common code, or target metadata.
5. Existing executable behavior for the five named routes must remain intact.
6. Keep common EmitC/export and target artifact export as neutral consumers and
   mirror validators only.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe one bounded
      target-leaf/profile validation closure task.
- [x] `RVVSelectedBodyContractionRouteFamilyPlan` or its local equivalent
      explicitly records and validates required target profile and target leaf
      facts for all five named contraction routes.
- [x] Invalid target profile, unsupported source load/product/reduction or
      macc/store leaves, invalid mask zeroing requirement, invalid strided
      leaf, and stale route-id authority fail closed before provider emission
      with targeted diagnostics.
- [x] Provider emission consumes the validated family plan for contraction
      route leaves, type/header facts, and route support mirror fields.
- [x] Target/header metadata remains mirror-only and uses explicit mirror names
      where new route-support fields are exposed.
- [x] Common EmitC/export does not infer RVV contraction semantics, dtype,
      memory form, mask/zeroing behavior, target leaves, or route support.
- [x] Generated-bundle dry-runs pass for all five contraction routes.
- [x] If provider emission or generated C changes, real `ssh rvv` evidence
      remains PASS for `widening_macc_add`, `widening_dot_reduce_add`, and
      `computed_masked_strided_input_widening_dot_reduce_add`.
- [x] Focused build/lit/C++/script checks pass for touched planning/provider,
      target metadata, and generated-bundle paths.
- [x] Active-authority scan confirms no positive legacy i32/source-front-door,
      descriptor/direct-C/source-export, route-id, artifact-name, exact
      intrinsic, or common/export RVV semantic authority is reintroduced.
- [x] `git diff --check`, Trellis validation, finish/archive, and one coherent
      commit are completed if the task finishes.

## Non-Goals

- No new RVV operation coverage, matmul/Linalg/frontend lowering, gather or
  scatter/indexed memory, unsigned variants, dtype/LMUL clone batches,
  source-front-door positive routes, one-intrinsic wrapper dialects,
  dashboards, broad reports, or helper-only cleanup.
- No move of RVV semantic authority into common EmitC/export, target metadata,
  route ids, artifact names, descriptors, helper names, or exact intrinsic
  spelling.
- No new dtype-prefixed helper op families such as
  `tcrv_rvv.i32_reduction_*`, `tcrv_rvv.i32_accumulator_*`, or
  `tcrv_rvv.i32_macc`.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, or emission. Python changes, if any,
  are limited to generated-bundle tooling/evidence.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-extension-plugin-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused lit/FileCheck tests for the five existing contraction target
   artifact routes and any new fail-closed diagnostics.
4. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
5. Run generated-bundle dry-runs for all five named contraction routes.
6. Run real `ssh rvv` for `widening_macc_add`, `widening_dot_reduce_add`, and
   `computed_masked_strided_input_widening_dot_reduce_add` if provider
   emission or generated C changes.
7. Run active-authority scans over active RVV include/lib/script/test paths.
8. Run `git diff --check`.
9. Run `cmake --build build --target check-tianchenrv -j2` if shared behavior
   changes.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-contraction-route-family-planning/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-macc-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-dot-reduction/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-masked-widening-dot-reduction/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-strided-input-widening-dot-reduction/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-masked-strided-widening-dot-reduction/prd.md`
- `.trellis/workspace/codex/journal-12.md`

Initial implementation surface inspected:

- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Focused tests under `test/Plugin`, `test/Target/RVV`, `test/Target`, and
  `test/Scripts` for the existing contraction routes.

## Completion Evidence

Implementation completed in this round:

- Extended `RVVSelectedBodyContractionRouteFamilyPlan` with target leaf/profile,
  provider support mirror, required headers, abstract C type mapping, RVV type
  facts, setvl/load/strided-load/product/masked-product/reduction-or-macc/store
  leaves, and inactive-lane zeroing requirement.
- Added plan-level validation before provider emission, including targeted
  failure checks for invalid target profile, provider support mirror, source
  load, strided load, widening product, masked widening product, reduction/macc
  compute leaf, store leaf, inactive-lane zeroing requirement, and stale
  `rvv-i32m1` route-id authority.
- Rewired contraction provider emission to consume the validated plan for
  leaves, types, headers, and mirror support fields instead of re-deriving those
  facts from route ids, helper names, artifact strings, descriptors, common
  export, or target metadata.
- Kept target artifact metadata mirror-only under explicit keys:
  `tcrv_rvv.target_leaf_profile`, `tcrv_rvv.provider_supported_mirror`,
  `tcrv_rvv.required_header_declarations`, `tcrv_rvv.c_type_mapping`, and
  `tcrv_rvv.inactive_lane_zeroing_requirement`.

Validation completed in this round:

- `cmake --build build --target tcrv-opt tcrv-translate -j2` passed after
  rebuilding tools for the new metadata.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2` passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `build/bin/tianchenrv-target-artifact-export-test` passed.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- Focused lit over the five contraction artifact tests and four existing
  generated-bundle dry-run lit tests passed: 9/9.
- Manual generated-bundle dry-run for all five covered routes passed:
  artifact root
  `artifacts/tmp/20260521-contraction-leaf-profile-dryrun/contraction-leaf-profile-routes`.
- Real `ssh rvv` representative evidence passed for `widening_macc_add`,
  `widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`, each at runtime
  counts 7, 16, and 23.
- `cmake --build build --target check-tianchenrv -j2` passed 245/245.
- `git diff --check` passed.
- Active-authority diff scan found only the intentional negative
  `rvv-i32m1-stale-route` diagnostic test; no positive legacy route authority
  was introduced.

Self-repair performed:

- Initial generated-bundle dry-run failed because `llvm-readobj` was not on
  PATH; reran with `/usr/lib/llvm-20/bin/llvm-readobj`.
- Exact RVV C type names in metadata caused header-forbidden-token validation
  friction; replaced the mirror summary with abstract profile facts:
  `vl:size_t,source:signed-e16mf2,result:signed-e32m1,mask:b32`.
- Focused lit initially saw stale object-record metadata because `tcrv-opt` and
  `tcrv-translate` had not been rebuilt after the self-repair; rebuilt both and
  reran the focused lit successfully.

## Definition Of Done

- Target-leaf/profile compatibility for the existing contraction route family
  is validated by the RVV plugin-owned family plan before provider emission.
- Fresh focused positive, negative, generated-bundle, and runtime evidence is
  current to this task when claimed.
- No legacy/source/descriptor/common-export route authority is reintroduced.
- The task is finished/archived only if production code/tests/evidence are
  present in this round; otherwise it remains open with an exact continuation
  point.
