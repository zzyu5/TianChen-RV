# Stage2 RVV standalone reduction route-family boundary reuse

## Goal

Prove that the RVV plugin-local route-family boundary recently closed for
`scalar_broadcast_macc_add` is reusable for the existing executable
`standalone_reduce_add` reduction/accumulation family. The production route must
flow through typed or pre-realized `tcrv_rvv` facts, the standalone reduction
family plan, provider-owned materialization/statement planning, common EmitC,
target artifact mirrors, generated-bundle evidence, and real `ssh rvv`
correctness without moving reduction authority into common EmitC, metadata,
route ids, descriptors, ABI strings, artifacts, or harness constants.

## Direction Source

- Direction title: `Stage2 RVV standalone reduction route-family boundary reuse`.
- Module owner: RVV plugin-local route-family module and provider registry
  ownership for already executable `standalone_reduce_add`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `b3c31ef8 rvv: close scalar broadcast macc route family boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  provided Direction Brief before source edits.

## Current Repository Facts

- The current RVV authority chain required by specs is selected `tcrv.exec`
  envelope -> typed low-level `tcrv_rvv` body -> RVV plugin-owned legality,
  selected-body realization, route provider, and family plans ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence for correctness/runtime claims.
- `standalone_reduce_add` already has current-head production surfaces:
  `RVVSelectedBodyStandaloneReductionRouteFamilyPlan`, provider-plan
  verification, math operand-binding facts, a standalone reduction statement
  plan, and aggregate migrated statement-plan consumption before the provider
  returns a `TCRVEmitCLowerableRoute`.
- Current target fixtures already expose mirror labels such as
  `tcrv_rvv.standalone_reduction_route_family_plan`,
  `tcrv_rvv.provider_supported_mirror`, runtime ABI order,
  `RouteOperandBindingPlan`, accumulator/result layout, and reduction store VL.
- Current generated-bundle evidence for pre-realized `standalone_reduce_add`
  already records a `reduction_accumulation_boundary`, but it does not expose
  the statement-plan family and ordered pre-loop/loop callee sequence as
  explicitly as the latest `scalar_broadcast_macc_add` evidence.
- `--direct-pre-realized-route-entry` is currently bounded to selected
  pre-realized fixtures and rejects `standalone_reduce_add` before exercising
  the production route-entry selected-body realization bridge.

## Scope

1. Keep this round bounded to `standalone_reduce_add`.
2. Reuse the existing standalone reduction family plan and migrated
   statement-plan boundary; do not add a new reduction family or operation.
3. If current production route-entry realization supports the pre-realized
   standalone reduction body, admit `standalone_reduce_add` in the
   generated-bundle direct pre-realized route-entry evidence path.
4. Make generated-bundle evidence explicitly show the family-owned ordered
   statement plan for `standalone_reduce_add`: standalone reduction family,
   pre-loop setvl/seed-store sequence, loop setvl/load/seed-splat/reduce/store
   sequence, accumulator/result roles, reduction operand order, and scalar
   output store VL.
5. Keep common EmitC and target artifact code as neutral consumers and mirrors
   only.

## Requirements

1. `standalone_reduce_add` provider support must still require a validated
   standalone reduction route-family plan and `RouteOperandBindingPlan` closure
   before materialization.
2. The standalone reduction family plan must continue to own operation kind,
   memory form, runtime AVL/VL control, SEW/LMUL/policy/type/header/intrinsic
   facts, accumulator seed layout, scalar result layout, reduction store VL,
   provider-supported mirror, and runtime ABI parameters.
3. The migrated statement-plan boundary must remain the provider path for the
   ordered route sequence. Provider fallback/ad hoc assembly must not be the
   active path for `standalone_reduce_add` when the family plan is present.
4. The generated-bundle direct pre-realized route-entry mode must only be
   enabled for `standalone_reduce_add` if the production emission-plan
   route-entry actually consumes the pre-realized body before route provider
   construction.
5. Evidence metadata may mirror family, provider, ABI, type, layout, and
   statement-plan facts only after provider route construction. It must not
   become route authority.
6. No positive legacy `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
   `!tcrv_rvv.i32m*`, descriptor, direct-C, source-export, source-front-door,
   name-derived, route-id-derived, artifact-derived, ABI-string-derived, or
   common-EmitC-derived route authority may be introduced.

## Acceptance Criteria

- [x] Task PRD and context files reference the RVV plugin, EmitC route, testing
      contract, previous scalar-broadcast MAcc route-family boundary task, and
      relevant standalone reduction prior tasks.
- [x] Current standalone reduction provider/family code is inventoried against
      current HEAD before implementation.
- [x] `--direct-pre-realized-route-entry` accepts pre-realized
      `standalone_reduce_add` and generated-bundle evidence records
      `route_entry_realization = true`, or the exact production blocker is
      recorded without claiming support.
- [x] Generated-bundle `reduction_accumulation_boundary` evidence explicitly
      exposes the standalone reduction statement-plan family, ordered pre-loop
      callees, ordered loop callees, accumulator seed source, reduction operand
      order, store pointer, and reduction store VL.
- [x] Focused FileCheck coverage proves the new statement-plan evidence and
      existing explicit mirror labels for `standalone_reduce_add`.
- [x] C++ provider tests continue to prove family ownership, missing/stale plan
      fail-closed diagnostics, math operand-binding closure, ordered statement
      planning, and provider route consumption.
- [x] Generated-bundle dry-run passes for `standalone_reduce_add` across counts
      `7,16,23`, including direct pre-realized route-entry if enabled.
- [x] Real `ssh rvv` correctness passes for `standalone_reduce_add` across at
      least counts `7,16,23` if executable behavior is claimed.
- [x] Bounded scan over touched RVV plugin/planning/provider/target/script/test
      paths finds no central ad hoc standalone reduction route authority,
      metadata/name-derived authority, descriptor/source-front-door authority,
      common-EmitC semantic ownership, or legacy i32 route authority.
- [x] Focused checks, `git diff --check`, and `check-tianchenrv` pass, or an
      exact blocker is recorded.
- [x] Task status, journal, archive, clean final git status, and one coherent
      commit complete if this task finishes.

## Non-Goals

- No new RVV operation families, new reductions, min/max expansion work,
  dtype/LMUL batches, high-level Linalg/Vector/StableHLO frontend lowering,
  source-front-door positive routes, global autotuning, dashboards, broad smoke
  matrices, compatibility wrappers for legacy i32 authority, or one-op-per-
  intrinsic wrapper growth.
- No migration of reduction semantics into common EmitC, target artifact
  plumbing, `tcrv.exec`, route ids, metadata fields, manifests, artifact names,
  ABI strings, descriptors, scripts, tests, or harness constants.
- No performance claim.
- No subagents or parallel-agent workflow.

## Validation Plan

1. Validate and start the Trellis task.
2. Run focused generated-bundle dry-run for pre-realized
   `standalone_reduce_add` direct route-entry mode with counts `7,16,23`.
3. Run the focused script FileCheck test for pre-realized
   `standalone_reduce_add`.
4. Build and run `tianchenrv-rvv-extension-plugin-test` if RVV planning/provider
   C++ files are changed, or run it as regression evidence for the existing
   provider boundary.
5. Run real `ssh rvv` generated-bundle evidence for `standalone_reduce_add`
   counts `7,16,23`.
6. Run bounded authority scans over touched and relevant files.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`, or record the
   exact blocker.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-route-family-boundary/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-reduction-accumulation-route-closure/prd.md`,
  `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-standalone-reduction-provider-validation/prd.md`,
  and
  `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-reduce-add-executable-slice/prd.md`.
- Initial code/test surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/RVV/explicit-selected-body-artifact-standalone-reduce-add.mlir`,
  `test/Target/RVV/pre-realized-selected-body-artifact-standalone-reduce-add.mlir`,
  and
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-standalone-reduce-add-dry-run.test`.

## Completion Evidence

- Production route-entry bridge:
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` now admits
  `TypedStandaloneReducePreRealizedBodyOp` only when the pre-realized body
  structurally carries a supported standalone-reduction op kind and memory
  form, then delegates to the existing RVV plugin-owned selected-body
  realization path before route facts are collected.
- C++ provider route-path evidence:
  `build/bin/tianchenrv-rvv-extension-plugin-test` passed after adding a
  `rvv_pre_route_standalone_reduce_add` production route-entry case. The test
  proves the pre-realized standalone body is consumed, the realized route
  exposes `rvv-standalone-reduction-route-family-plan.v1`, and the non-
  standalone `typed_reduce_pre_realized_body` route-entry case still fails
  closed as an unsupported family.
- Generated-bundle dry-run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run
  --pre-realized-selected-body --direct-pre-realized-route-entry
  --op-kind standalone_reduce_add --runtime-count 7 --runtime-count 16
  --runtime-count 23 ...` passed with artifact root
  `artifacts/tmp/stage2_standalone_reduction_family_boundary_reuse/direct-pre-realized-standalone-reduce-add-dry`.
- Focused FileCheck:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv .
  --filter=rvv-generated-bundle-abi-e2e-pre-realized-standalone-reduce-add-dry-run`
  passed from `build/test`, proving direct route-entry evidence,
  route-entry-selected-body materializer evidence, statement-plan family and
  ordered callee mirrors, and absence of the explicit selected-lowering
  materializer path in the per-op evidence.
- Real RVV correctness:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py
  --pre-realized-selected-body --direct-pre-realized-route-entry
  --op-kind standalone_reduce_add --runtime-count 7 --runtime-count 16
  --runtime-count 23 --ssh-target rvv ...` passed with artifact root
  `artifacts/tmp/stage2_standalone_reduction_family_boundary_reuse/direct-pre-realized-standalone-reduce-add-ssh`;
  stdout included `PASS op=standalone_reduce_add counts=7,16,23 seeds=-11,17`.
- Authority scans:
  touched source/test scans found only existing negative/fail-closed references
  for descriptor/direct-C/source-export/source-front-door/legacy i32 terms;
  generated dry-run and ssh evidence/harness scans found no descriptor,
  direct-C, source-export, `tcrv_rvv.i32_`, or explicit
  `--tcrv-materialize-selected-lowering-boundaries` residue.
- Full quality gate:
  `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`,
  `git diff --check`, and `cmake --build build --target check-tianchenrv -j2`
  passed. Local `clang-format` was unavailable in PATH and under
  `/usr/lib/llvm-20/bin`; formatting-sensitive checks still passed.
- Spec sync:
  `.trellis/spec/extension-plugins/rvv-plugin.md` now records standalone
  reduction as a supported route-entry realization family only when matching
  RVV-owned family plans, materialization facts, math operand-binding facts,
  migrated statement plans, and tests are present.

## Definition Of Done

`standalone_reduce_add` visibly reuses the route-family/provider/statement-plan
boundary through direct or ordinary pre-realized generated-bundle evidence,
focused checks and real `ssh rvv` evidence pass, authority scans stay clean,
the Trellis task is finished and archived, and one coherent commit records the
round.
