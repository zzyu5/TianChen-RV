# Stage2 RVV runtime-strided memory-window route foundation

## Goal

Implement one bounded production route-supported Stage 2 RVV memory-movement
foundation for explicit runtime-strided memory windows. The route must start
from a selected `tcrv.exec` RVV variant that binds source/destination memory
windows, runtime `n`/AVL, and runtime stride facts; consume those values in a
typed low-level `tcrv_rvv` body carrying element type, SEW, LMUL, policy,
memory form, stride unit, stride role, and source/destination roles; and let
the RVV plugin derive the EmitC statement plan and target artifact mirrors from
those typed facts or fail closed.

This is a production route foundation. It is not an executable correctness
closure unless the provider/target route support is finished and a later
focused harness proves runtime behavior on `ssh rvv`.

## What I Already Know

- The repository starts on `main` with latest commit
  `b1baa654 rvv: close computed-mask select policy executable`; the worktree
  was clean before this task was created.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief.
- The previous archived task
  `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-computed-mask-select-executable-policy-closure/`
  was an evidence closure for computed-mask select and intentionally did not
  move production provider/target code.
- `.trellis/spec/index.md` requires the RVV authority chain to flow through a
  selected typed `tcrv_rvv` body, RVV plugin legality/realization/provider,
  common `TCRVEmitCLowerableRoute`, neutral EmitC materialization, and target
  artifact mirrors.
- `.trellis/spec/extension-plugins/rvv-plugin.md` says `tcrv.exec` binds
  runtime ABI roles only; it must not own RVV dtype, memory form, schedule,
  stride semantics, intrinsic spelling, or selected route authority.
- `.trellis/spec/lowering-runtime/emitc-route.md` says Common EmitC/export may
  carry provider payloads unchanged but must not infer RVV memory semantics,
  dtype, SEW/LMUL, policy, schedule, ABI roles, or intrinsic family.
- Repository inspection shows existing typed strided memory skeletons:
  `typed_strided_memory_pre_realized_body`,
  `typed_strided_store_memory_pre_realized_body`, `tcrv_rvv.strided_load`, and
  `tcrv_rvv.strided_store`. This round should rewire/validate those existing
  production surfaces rather than adding a new one-intrinsic or dtype-prefixed
  op family.
- Existing route code already has strided-input contraction and computed-mask
  strided slices. This task is scoped to the memory movement owner:
  byte-strided source load to unit-stride destination store and/or unit-stride
  source load to byte-strided destination store, with one coherent proof
  instance if full bidirectional support is too large.

## Requirements

- The selected envelope must bind/import runtime memory and stride facts into
  the typed RVV body; route construction must not infer stride facts from ABI
  order, C names, artifact names, route ids, descriptors, intrinsic spellings,
  or test fixture names.
- The typed body/config must structurally carry:
  source/destination roles, memory form, stride role, stride unit, element type,
  SEW, LMUL, policy, runtime `n`/AVL, and whether the strided side is source or
  destination.
- The RVV plugin provider must validate legal combinations and derive:
  statement-plan owner, load/store intrinsic family, required headers,
  C type mapping, operand-binding summary, provider support mirror, and route
  metadata mirrors.
- Target artifact validation must compare provider-derived memory-form and
  stride facts against candidate mirrors and fail closed on stale or missing
  mirrors.
- Focused positive tests must prove route support through dialect/plugin/target
  artifact generation for one explicit runtime-strided memory movement fixture.
- Focused negative tests must cover missing/incorrect stride role, ambiguous or
  unsupported stride unit, missing runtime mem_window/runtime ABI binding,
  missing `n`/AVL fact, dtype/config mismatch, unsupported policy, stale
  provider memory-form or stride mirrors, and route-string/artifact-name/
  ABI-string/intrinsic-spelling authority.
- Production code changes are required. Evidence-only, helper-only, prompt-only,
  broad smoke, or dashboard/report changes are not sufficient.

## Acceptance Criteria

- [x] A selected `tcrv.exec` RVV fixture imports or binds source, destination,
      `n`/AVL, and runtime byte-stride values into the typed `tcrv_rvv` body.
- [x] The typed strided memory body/config structurally carries memory form,
      stride side/role, stride unit, source/destination memory roles, element
      type, SEW/LMUL, and policy.
- [x] RVV provider validation derives the route statement plan and fails closed
      for unsupported or stale typed facts.
- [x] Target artifact validation mirrors only provider-derived memory-form and
      stride facts and rejects stale/missing mirrors.
- [x] Focused positive dialect/plugin/target fixture proves route support for
      the runtime-strided memory-window route.
- [x] Focused negative tests cover the failure modes listed in Requirements.
- [x] `tianchenrv-rvv-extension-plugin-test` and
      `tianchenrv-target-artifact-export-test` pass if provider/target C++ code
      changes.
- [x] A generated artifact or lit check proves the route-supported fixture.
- [x] `ssh rvv` is run only if executable correctness is claimed; otherwise the
      final report explicitly says it was not applicable for route support.
- [x] Bounded old-authority and q-name scan over touched files shows no new
      legacy i32, q8/q4, descriptor, source-front-door, route-id,
      artifact-name, ABI-string, or intrinsic-spelling authority.
- [x] `git diff --check`, `git diff --cached --check`, and clean final
      `git status --short` are recorded.

## Completion Evidence

- Production owner implemented:
  `RVVSelectedBodyBaseMemoryMovementRouteProviderPlan` now carries explicit
  provider-plan mirrors for strided layout, source stride source, and
  destination stride source. The migrated base-memory statement-plan owner
  requires those provider-plan facts to match the provider-built route
  description before constructing route statements.
- Target artifact owner implemented:
  the RVV materialized header evidence whitelist now exports provider-derived
  `source_stride_source` and `destination_stride_source` mirrors when the
  provider emitted them, without letting target export infer RVV memory
  semantics.
- Positive route support:
  both `strided_load_unit_store` and `unit_load_strided_store` selected-body
  fixtures materialize to typed RVV load/move/store structure, produce
  provider-derived memory-form and stride-source metadata, and export callable
  header artifacts.
- Negative coverage:
  plugin tests reject stale strided route-description mirrors through the
  validated base-memory family-plan check; existing target artifact tests cover
  stale source/destination stride mirror rejection.
- Checks run:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`;
  `cmake --build build --target tcrv-translate tianchenrv-target-artifact-export-test`;
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  `build/bin/tianchenrv-target-artifact-export-test`;
  `build/bin/tcrv-opt ... --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans`
  for both strided memory fixtures;
  `build/bin/tcrv-opt ... | build/bin/tcrv-translate --tcrv-export-target-header-artifact`
  for both fixtures, with runtime ABI order, route-family plan, strided layout,
  stride source, memory forms, and callable ABI confirmed;
  `git diff --check`;
  `git diff --cached --check`;
  bounded old-authority/q-name scan over the new diff.
- `ssh rvv` was not run because this round claims route-supported target
  artifact capability, not executable correctness.

## Technical Approach

1. Inspect the existing typed strided memory ops, verifier rules,
   runtime ABI contracts, provider route planning, statement-plan owners, and
   target artifact validation.
2. Choose the smallest coherent production owner:
   prefer byte-strided source load to unit-stride destination store if it is the
   least disruptive route-supported proof; include unit-load to strided-store
   only if existing code already makes it a small paired validation extension.
3. Repair the selected-body, provider plan, statement-plan, and target mirror
   contracts so the route derives facts from typed body/config/runtime ABI
   values and fails closed otherwise.
4. Add/update focused positive and negative fixtures instead of broad smoke
   matrices.
5. Run focused lit/C++ checks for the changed module, self-repair failures, then
   update task notes/journal and archive if complete.

## Out Of Scope

- q8/q4/llama benchmark routes.
- ProviderSpec/model-name route authority.
- New computed-mask, contraction, dequant, clamp, MAcc, or reduction executable
  evidence as the primary deliverable.
- High-level Linalg, Vector, StableHLO, or source-front-door frontend work.
- Broad gather/scatter matrices or dtype/LMUL clone batches.
- One-intrinsic wrapper dialects or dtype-prefixed helper families.
- Compatibility wrappers preserving legacy i32m1 authority.
- Common EmitC/export logic that chooses RVV memory semantics, dtype, stride
  unit, schedule, policy, or intrinsic family.
- Broad smoke matrices, dashboards, report-only tasks, or helper-only changes.

## Technical Notes

- Read first list satisfied so far:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  archived task
  `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-computed-mask-select-executable-policy-closure/`,
  and initial targeted searches across the RVV dialect, provider, statement
  plan, target validation, and RVV target fixtures.
- Memory-derived guardrails used for scoping: recent evidence-only drift must
  not repeat here; dtype/config authority must remain structural; old i32m1 or
  q-name authority must not become a positive route.
