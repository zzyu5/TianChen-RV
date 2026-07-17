# Stage2 RVV Computed-Masked Segment2 Selected-Body Realization Migration

## Goal

Move the generated artifact path for the computed-masked segment2 memory
movement family behind the RVV plugin-local selected lowering-boundary
realization producer and remove active direct pre-realized route-entry shortcut
authority for:

- `computed_masked_segment2_load_unit_store`
- `computed_masked_segment2_store_unit_load`
- `computed_masked_segment2_update_unit_load`

The production path for this task must be:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized tcrv_rvv computed-mask segment2 body
  -> RVV plugin-local selected-body realization producer
  -> realized typed tcrv_rvv body preserving mask construction/use,
     segment2 tuple lane roles, unit and segment memory forms,
     load/store/update memory roles, update/pass-through source binding,
     runtime n/AVL/VL values, setvl placement, loop relation,
     mask/tail policy, SEW/LMUL/config facts, selected requires,
     provider facts, and artifact ABI order
  -> computed-mask memory route-family facts
  -> segment2 route-family facts / statement-plan boundary
  -> route materialization facts
  -> memory operand-binding facts
  -> route-control provider plan
  -> migrated segment2 statement-plan owner
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI validation
  -> generated RVV C artifact and harness
  -> ssh rvv correctness evidence
```

The generated artifact must not be accepted from a direct route-entry shortcut,
route id, artifact name, script option, ABI string, exact intrinsic spelling,
common EmitC behavior, descriptor residue, source-front-door metadata,
pre-realized fixture status, stale mirror metadata, or legacy i32 helper
authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV computed-masked segment2 selected-body
  realization migration`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `cb60e33d rvv: demote standalone reduce route entry`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint: one Codex worker; no subagents, spawned agents,
  parallel agents, or multi-agent workflows.

## What I Already Know

- The global RVV-first chain is `tcrv.exec` envelope -> selected RVV variant ->
  typed low-level `tcrv_rvv` body -> RVV plugin-owned selected-body
  realization / route provider -> `TCRVEmitCLowerableRoute` -> neutral common
  EmitC -> target artifact -> `ssh rvv` evidence for executable claims.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` bind ABI/runtime roles
  only. They do not define compute, dtype, shape, memory form, segment lane
  role, mask policy, or RVV route identity.
- Common EmitC must only materialize provider-built routes. It must not infer
  RVV dtype, SEW, LMUL, operation kind, mask/tail policy, memory form,
  intrinsic spelling, ABI order, or route support from route ids, artifact
  metadata, ABI strings, or test names.
- The RVV selected-body realization owner registry owns pre-realized body
  classification and realization dispatch. Route-entry eligibility is narrower
  than selected-body realization support.
- The current spec still describes computed-mask segment2 load/store/update as
  active direct route-entry families. This task intentionally migrates that
  production authority to the public selected lowering-boundary realization
  path for the bounded computed-masked segment2 family.
- The immediately previous archived task demoted `standalone_reduce_add` from
  direct route-entry support while preserving selected-boundary realization and
  generated-bundle / `ssh rvv` evidence. This task should follow the same
  migration pattern but for computed-masked segment2 memory movement.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python changes are allowed only for evidence tooling, harnesses,
   generated-bundle guardrails, and artifact parsing.
2. Keep computed-masked segment2 bodies under the RVV `segment2 memory`
   selected-body realization owner. The public selected lowering-boundary
   producer must realize the typed pre-realized body before provider route
   facts are collected.
3. Demote or delete direct pre-realized route-entry support for
   `computed_masked_segment2_load_unit_store`,
   `computed_masked_segment2_store_unit_load`, and
   `computed_masked_segment2_update_unit_load`. Direct shortcut requests must
   fail closed before route-entry materialization, bundle generation, or
   provider route construction.
4. Preserve typed facts for operation kind, compare-produced mask source and
   use, same-VL scope, mask/tail policy, segment2 tuple lane roles, field0 and
   field1 payload roles, unit-load / unit-store / segment-load / segment-store
   memory forms, load/store/update memory roles, update arithmetic kind and
   pass-through binding, source/destination `mem_window` roles, runtime count
   `n`, AVL/VL relation, `setvl`/`with_vl` placement, SEW/LMUL/config facts,
   selected capability requirements, provider route facts, and artifact ABI
   order.
5. Fail closed for unsupported or inconsistent selected-boundary input,
   including missing runtime_param, missing mem_window/runtime ABI value, wrong
   mask binding, stale mask producer, wrong segment lane role, wrong memory
   role, wrong update/pass-through binding, wrong dtype/config/policy, wrong
   AVL/VL relation, wrong setvl placement, missing selected capability, stale
   route id, stale mirror metadata, direct-route-entry-only authority,
   artifact-name or script-derived authority, exact-intrinsic-as-authority, and
   common-EmitC semantic invention.
6. Reuse existing computed-mask memory route-family, segment2 route-family,
   route materialization, route-control provider, memory operand-binding,
   segment2 statement-plan, target artifact, and generated-bundle boundaries
   where they already express the required facts. Do not add a new central
   route table, descriptor path, common EmitC semantic branch, source-front-door
   route, or one-intrinsic wrapper dialect.
7. Do not start unmasked segment2 deinterleave/interleave migration,
   compare/select, strided load/store, standalone reduction, widening dot,
   widen conversion, high-level Linalg/frontend lowering, selected-body
   realization framework rewrites, dashboard/report work, broad smoke matrices,
   or evidence-only tasks.

## Acceptance Criteria

- [ ] Production code no longer treats the computed-masked segment2
      load/store/update family as direct pre-realized route-entry eligible,
      while the `segment2 memory` selected-body realization owner still
      realizes the corresponding typed pre-realized bodies through the public
      selected lowering-boundary producer.
- [ ] C++ tests prove the computed-masked segment2 load/store/update bodies
      belong to the segment2 realization owner, are not direct route-entry
      consumers, preserve load/store/update owner separation where needed, and
      fail closed when `realizePreRealizedRVVRouteEntrySelectedBody` is used as
      a direct route-entry shortcut.
- [ ] C++ or lit tests prove the selected-boundary path consumes realized typed
      facts, including computed mask producer/use, segment2 tuple roles, field
      payload and pass-through/update bindings, source/destination ABI roles,
      runtime `n`/AVL/VL, setvl/with_vl placement, computed-mask memory family
      plan, segment2 statement-plan facts, memory operand bindings,
      route-control provider facts, provider-supported mirrors, and artifact
      ABI order.
- [ ] The generated-bundle script rejects
      `--direct-pre-realized-route-entry` for each migrated computed-masked
      segment2 op before route-entry materialization or bundle generation.
- [ ] Generated-bundle dry-runs for all three migrated ops pass through
      `--tcrv-materialize-selected-lowering-boundaries`, record
      `route_entry_realization: false`, record selected-body producer evidence,
      record realized computed-mask segment2/provider facts, and record no
      direct route-entry materializer.
- [ ] Real `ssh rvv` generated-bundle execution covers counts including `0`,
      `1`, exact/full-chunk, tail, and stress cases with mask patterns and
      segment lane checks for the migrated family.
- [ ] Focused non-regression covers adjacent selected-body realization paths
      risked by this demotion, including at minimum the RVV extension plugin
      test and at least one remaining direct route-entry family that should
      stay direct, if any remains in scope.
- [ ] A bounded touched-file authority scan finds no new executable or route
      claim depending on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [ ] `git diff --check` passes.
- [ ] Focused build/tests pass, and `check-tianchenrv` passes or an exact
      blocker is recorded with the task left open.
- [ ] Trellis status, journal/archive, and final commit are truthful.

## Validation Plan

1. Validate Trellis task context.
2. Inspect current selected-body realization, route planning/provider,
   target-support bundle, generated-bundle script, focused plugin tests, and
   target/script tests for active direct computed-masked segment2 route-entry
   authority.
3. Narrow computed-masked segment2 route-entry eligibility so the family
   remains selected-boundary capable but is not direct route-entry eligible.
4. Update generated-bundle tooling/lit coverage so direct pre-realized
   route-entry mode fails closed for each migrated computed-masked segment2 op.
5. Strengthen selected-boundary C++/lit/script coverage so evidence explicitly
   checks `route_entry_realization: false`, selected-body producer evidence,
   realized computed-mask segment2 facts, provider mirrors, memory operand
   bindings, segment2 statement-plan facts, and artifact ABI order.
6. Run focused C++ plugin tests and focused lit/script tests.
7. Run generated-bundle selected-boundary dry-runs for the three migrated ops.
8. Run real `ssh rvv` generated-bundle evidence for representative counts and
   mask/lane cases through the selected lowering-boundary producer path.
9. Run focused adjacent non-regression for explicit selected-body segment2
   artifacts and any remaining direct route-entry families sharing the owner
   registry boundary.
10. Run authority scan, `git diff --check`, task validation, and
    `check-tianchenrv` or record the exact blocker.

## Out Of Scope

- Plain segment2 deinterleave/interleave migration.
- Compare/select, strided load/store, standalone reduction, widening dot,
  MAcc, widen conversion, unrelated base-memory cleanup, additional dtype or
  LMUL clone batches, high-level Linalg/Vector/StableHLO frontend work,
  source-front-door positive RVV routes, descriptor-driven computation,
  one-intrinsic wrapper dialects, dashboard/report work, or broad smoke
  matrices.

## Technical Notes

- Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous migration pattern:
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-standalone-reduce-selected-realization-migration/`.
- Production files named by the Hermes brief for inspection:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.

## Completion Results

- Completed production demotion of computed-masked segment2 load/store/update
  direct pre-realized route-entry authority. The segment2 route-entry registry
  now retains only plain segment2 deinterleave/interleave direct entries.
- Preserved positive selected-boundary realization for
  `computed_masked_segment2_load_unit_store`,
  `computed_masked_segment2_store_unit_load`, and
  `computed_masked_segment2_update_unit_load` through the RVV plugin-local
  selected-body realization owner registry.
- Removed provider planning dependence on selected-body route-entry family
  owners. Computed-mask segment2 provider plans now rely on selected route
  analysis, typed config/runtime facts, computed-mask memory facts, segment2
  family facts, materialization facts, route-control facts, and operand-binding
  facts.
- Updated generated-bundle tooling so `--direct-pre-realized-route-entry`
  fails closed for the migrated computed-masked segment2 family before route
  entry materialization or bundle generation.
- Added/updated focused C++ and lit coverage proving selected-boundary-only
  behavior, `route_entry_realization: false`, selected-body producer evidence,
  direct shortcut rejection, and generated artifact ABI evidence for all three
  migrated ops.
- Real `ssh rvv` evidence passed for the migrated family with counts
  `0,1,16,17,23,257`, including active/inactive mask lanes, tail preservation,
  source preservation, and segment-lane distinction checks.
- `git diff --check` passed.
- Focused checks passed:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`,
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`,
  `build/bin/tianchenrv-rvv-extension-plugin-test`, and focused lit for the
  six migrated computed-masked segment2 direct-fail/selected-boundary dry-runs.
- Full `check-tianchenrv` passed 403/403 after rerunning without the earlier
  self-induced duplicate command interference.
- Bounded touched-file authority scan found no positive computed-masked
  segment2 route/executable claim depending on direct-route-entry-only,
  descriptor, source-front-door, common-EmitC, route-id, script, artifact-name,
  stale metadata, exact-intrinsic, or legacy-i32 authority. Remaining hits are
  negative fail-closed tests, direct support for still-in-scope plain segment2
  entries, or provider-derived intrinsic leaves.
