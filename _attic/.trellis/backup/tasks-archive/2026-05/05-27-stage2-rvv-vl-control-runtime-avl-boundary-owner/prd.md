# Stage2 RVV VL/control Runtime-AVL Boundary Owner

## Goal

Introduce or extract one production RVV plugin-local VL/control runtime-AVL
owner boundary over active selected-body runtime control routes. The boundary
must carry runtime `n`/AVL facts from selected `tcrv.exec` runtime ABI imports
into typed or realized `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` structure,
provider-built `TCRVEmitCLowerableRoute` facts, neutral common EmitC
materialization, target artifact ABI validation, generated RVV C artifacts, and
real `ssh rvv` evidence for the executable claim.

The primary migrated consumer is an existing runtime-AVL route. The preferred
primary consumer for this round is `runtime_i32_splat_store` / runtime scalar
splat-store because current code already treats it as route-control adopted but
not as a migrated statement-plan owner. One adjacent existing masked or
conversion consumer, preferably `widen_i32_to_i64` or `widen_i16_to_i32`, must
prove that runtime n, AVL, VL, setvl placement, loop bounds, policy, ABI order,
operand bindings, header/type facts, provider-supported mirrors, and artifact
ABI claims are shared typed facts rather than route-name, script, metadata,
artifact-name, or common-EmitC conventions.

The completed path must remain:

```text
selected tcrv.exec RVV variant runtime_param / mem_window bindings
  -> explicit typed tcrv_rvv runtime AVL / VL / setvl facts
  -> RVV plugin-owned runtime AVL/VL control owner
  -> route-control provider facts and statement plans
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI consumer
  -> generated RVV C artifact / harness
  -> ssh rvv evidence for runtime correctness claims
```

## Direction Source

- Direction title: `Stage2 RVV VL/control runtime-AVL boundary owner`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: untracked
  `.trellis/tasks/05-27-classroom-bitwise-xor-slice-workflow/`.
- Initial HEAD: `f0d1c632 rvv: add conversion dtype policy route owner`.
- `.trellis/.current-task` pointed to the classroom XOR slice task, but that
  PRD explicitly targets a classroom worktree and says not to touch the main
  Stage2 loop. This task was created from the supplied Hermes Direction Brief
  instead. The classroom task directory is left intact.
- Serial worker constraint from the brief: do not use subagents, spawned
  agents, parallel agents, or multi-agent workflows for the implementation.

## What I Already Know

- Current specs define `RVVRuntimeAVLVLControlPlan` and a
  `RVVSelectedBodyRouteControlProviderPlan` boundary for runtime AVL/VL,
  SEW/LMUL, tail/mask policy, runtime ABI order, and selected capability facts.
  That boundary is RVV-plugin-owned and sits before route statement
  construction.
- Current code already has a route-control provider owner registry with
  ordinary elementwise, masked elementwise, scalar-broadcast elementwise,
  compare/select, widening conversion, computed-mask memory, segment2, base
  memory, standalone reduction, MAcc, runtime scalar splat-store,
  computed-mask accumulation, and contraction owners.
- Current code also has a separate migrated statement-plan owner registry.
  Runtime scalar splat-store is explicitly outside that migrated registry,
  even though it is route-control adopted. That is the main production gap this
  round should close instead of adding wrapper-only validation.
- Target artifact validation already checks compare/select and conversion
  family mirror fields, and general runtime AVL/VL metadata validation exists.
  This task should extend or tighten target validation only where provider-built
  runtime-control facts are not consumed strongly enough for the migrated
  consumers.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` bind ABI/runtime roles
  only. They do not define RVV compute, dtype, SEW/LMUL, schedule, intrinsic
  spelling, route support, or executable authority.
- Common EmitC/export must remain neutral. It may materialize provider-built
  route payloads and carry mirror metadata after route construction, but it
  must not choose RVV VL/control semantics or infer runtime AVL from parameter
  names, ABI strings, route ids, artifacts, scripts, descriptors, comments, or
  exact intrinsic spellings.

## Requirements

1. Add or extract one explicit plugin-local runtime AVL/VL owner boundary, or
   promote the existing route-control boundary so it is the active production
   owner for the selected runtime-AVL consumer and the adjacent conversion or
   masked consumer.
2. Migrate the primary runtime-AVL consumer, preferably runtime scalar
   splat-store, behind the shared owner through provider construction and
   statement-plan ownership. Do not leave it on a central provider-local
   statement assembly path when an owner path exists.
3. Keep at least one adjacent masked or conversion consumer, preferably
   widening conversion, actively consuming the same runtime AVL/VL owner facts.
   The adjacent consumer must prove that the shared boundary is not a
   route-specific helper.
4. The owner must derive or validate runtime `n`, AVL source, VL definition and
   use, `setvl` placement, loop lower/upper/step relationship, remaining AVL
   expression, pointer advance metadata, tail/mask policy, ABI order, operand
   binding, required headers, C/RVV type mapping, provider-supported mirrors,
   selected capability mirrors, and target artifact ABI claims from typed
   selected-body/config/runtime facts and provider-built route facts.
5. Unsupported or inconsistent family, stale route id, stale mirror metadata,
   missing runtime binding, wrong AVL source, wrong VL value use, wrong setvl
   placement, wrong loop bound relation, policy mismatch, wrong ABI order,
   missing operand binding, wrong type/header mapping, artifact-name authority,
   script-derived authority, common-EmitC-derived authority, source-front-door
   authority, exact-intrinsic-derived authority, or legacy-i32-derived
   authority must fail closed with targeted diagnostics before common EmitC or
   target export accepts an executable artifact.
6. Existing completed segment2, MAcc, standalone reduction/accumulation,
   compare/select, computed-mask memory, conversion dtype-policy,
   scalar-broadcast, contraction, and base memory routes must not be weakened
   or reclassified as evidence-only work.
7. The task must not add broad route coverage, dtype/LMUL clone batches,
   high-level Linalg/frontend routes, source-front-door positive routes,
   descriptor/direct-C/source-export paths, dashboards, report-only changes,
   one-intrinsic wrapper dialects, or unrelated route-family follow-on polish.

## Acceptance Criteria

- [ ] A production runtime AVL/VL owner boundary exists in RVV plugin-local C++
      and is consumed by active route/provider construction for the migrated
      runtime-AVL consumer.
- [ ] Runtime scalar splat-store or an equivalent runtime-AVL representative is
      migrated behind the owner as an active provider/statement-plan consumer.
- [ ] One adjacent masked or conversion route remains an active consumer of
      the same owner boundary, with focused checks proving the shared typed
      runtime control facts.
- [ ] Provider-built routes for both consumers prove runtime control,
      route-control, operand binding, materialization, statement-plan,
      header/type/intrinsic, ABI order, provider-supported mirror, and selected
      capability agreement before materialization.
- [ ] Target artifact ABI validation consumes provider-built runtime AVL/VL
      facts for the migrated consumers before accepting generated artifact or
      header claims.
- [ ] Focused C++ or lit tests cover positive owner membership/dispatch,
      provider/statement-plan consumption, and fail-closed diagnostics for
      stale route id, stale mirror metadata, missing runtime binding, wrong AVL
      source, wrong VL use, wrong setvl placement, wrong loop relation, policy
      mismatch, wrong ABI order, missing operand binding, wrong type/header
      mapping, artifact-name authority, and script-derived authority where the
      current test surface exposes them.
- [ ] Generated-bundle dry-run covers both migrated consumers.
- [ ] Real `ssh rvv` generated-bundle execution covers the runtime-AVL
      representative over counts including `0`, small, exact, tail, and stress
      cases.
- [ ] Focused non-regression covers completed segment2 update/store/load,
      segment2 deinterleave/interleave, computed-mask MAcc,
      scalar-broadcast MAcc, runtime-scalar MAcc, standalone
      reduction/accumulation, compare/select, computed-mask memory, conversion
      dtype-policy, scalar-broadcast, contraction, and base memory paths.
- [ ] Bounded touched-file authority scan finds no new executable or route
      support depending on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, or legacy-i32-derived
      authority.
- [ ] `git diff --check` passes.
- [ ] Focused build/tests pass, and `check-tianchenrv` passes or an exact
      blocker is recorded with the task left open.
- [ ] Trellis task status, context, journal, and archive state are truthful,
      and one coherent commit is created if all acceptance criteria are
      satisfied.

## Validation Plan

1. Start and validate this Trellis task after PRD/context setup.
2. Inspect runtime scalar splat-store, widening conversion, route-control
   provider, migrated statement-plan, route materialization facts,
   target artifact ABI validation, generated-bundle, and focused tests.
3. Promote or extract the runtime AVL/VL owner boundary in the smallest
   production RVV plugin surface that removes duplicated or central one-off
   statement/control logic.
4. Migrate runtime scalar splat-store or the chosen runtime-AVL representative
   behind the owner path and keep a conversion or masked adjacent route behind
   the same owner.
5. Add focused C++ and/or lit tests for owner membership, provider/target
   consumption, statement-plan migration, and fail-closed stale or missing
   runtime-control facts.
6. Run focused build/tests for touched RVV plugin and target surfaces.
7. Run generated-bundle dry-runs for both consumers.
8. Run real `ssh rvv` evidence for the runtime-AVL representative over counts
   `0`, small, exact, tail, and stress cases.
9. Run required focused non-regression dry-runs/lit coverage.
10. Run bounded authority scan, `git diff --check`, Trellis validation, and
    `check-tianchenrv`.

## Out Of Scope

- Broad Stage2 coverage expansion or clone batches.
- New high-level Linalg, Vector, StableHLO, source-front-door, Toy,
  TensorExtLite, Template, IME, Offload, or future plugin routes.
- New dtype-prefixed RVV helper ops, legacy `RVVI32M1` / `rvv-i32m1` positive
  executable routes, descriptor-driven compute, direct-C/source-export routes,
  or exact intrinsic spelling as route authority.
- Dashboard, report-only, prompt-only, helper-only, or broad smoke-only work as
  the main achievement.
- Weakening completed segment2, MAcc, standalone reduction/accumulation,
  compare/select, conversion dtype-policy, computed-mask memory, runtime-scalar,
  contraction, scalar-broadcast, or base-memory owners.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Guides read:
  `.trellis/spec/guides/index.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Predecessor archived task read:
  `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-conversion-dtype-policy-route-family-owner/prd.md`,
  `implement.jsonl`, and `check.jsonl`.
- Initial code surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Open Questions

None blocking. The repo already exposes a route-control owner boundary; the
implementation should move a runtime-AVL representative, preferably runtime
scalar splat-store, behind the migrated/provider owner path and use a
conversion or masked route as the adjacent proof of shared runtime AVL/VL
typed authority.

## Definition Of Done

The runtime AVL/VL owner is the active production boundary for the migrated
runtime-AVL consumer and one adjacent masked or conversion consumer,
provider-built routes and target artifacts consume owner-derived facts,
focused positive and fail-closed tests pass, generated-bundle and `ssh rvv`
evidence are recorded, authority scans are clean, Trellis state is truthful,
and one coherent commit records the work.
