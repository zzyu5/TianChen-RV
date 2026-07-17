# Stage2 RVV MAcc Selected-Body Realization Provider Handoff Closure

## Goal

Recover the already-started dirty MAcc selected-body realization/provider
handoff as one coherent compiler-path task. The module goal is to make plain
MAcc, scalar-broadcast MAcc, computed-mask MAcc, and runtime-scalar
computed-mask MAcc flow through owner-local selected-body validation /
realization hooks and owner-derived provider handoff facts, while removing dead
duplicate MAcc-specific typed-fact branches from generic central selected-body
realization code.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial inspected state for this round had three dirty compiler files:
  `include/TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`, and
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`.
- Initial inspected state also had an untracked archived Trellis task:
  `.trellis/tasks/archive/2026-05/05-30-05-30-05-30-stage2-rvv-macc-selected-body-realization-provider-handoff/`.
- That archived task described this same MAcc handoff but was already marked
  completed with no commit, which is not truthful while source files remain
  dirty. This active task supersedes that stale archive and must resolve it
  before final commit.
- Commit `ddb3adf9` completed non-segment computed-mask memory owner boundary;
  this round must not start another RVV family while MAcc changes are dirty.
- The long-term path remains:
  `tcrv.exec envelope -> selected RVV variant -> typed low-level tcrv_rvv body
  -> RVV plugin-owned legality / selected-body realization / route provider ->
  TCRVEmitCLowerableRoute -> common EmitC materializer -> target artifact`.

## Requirements

1. Keep the change bounded to the MAcc selected-body realization/provider
   handoff already present in the dirty worktree.
2. Move active pre-realized MAcc validation for plain MAcc,
   scalar-broadcast MAcc, computed-mask MAcc, and runtime-scalar computed-mask
   MAcc into MAcc owner-local surfaces.
3. Preserve validation for op kind, memory form, ABI roles, operand binding,
   SEW/LMUL/policy, accumulator/result layout, computed-mask provenance,
   runtime `n`/AVL/VL, route-description mirrors, provider-support mirrors,
   runtime ABI order, and fail-closed diagnostics.
4. Keep central selected-body realization code limited to owner dispatch and
   neutral shared helpers. It must not rebuild MAcc semantics or route/provider
   authority through duplicate branches.
5. Preserve production provider handoff through owner-derived MAcc
   route-family/provider-plan facts and neutral common EmitC materialization.
6. Repair stale inherited dirty code inside this MAcc owner boundary rather
   than carrying it forward as an unresolved partial migration.
7. Add or update focused C++ tests proving the four MAcc variants select the
   owner-local path and fail closed on mismatched typed facts.
8. Run focused build/test/evidence checks, including at least one
   representative generated-bundle dry-run or `ssh rvv` run for an MAcc
   variant. Do not claim runtime correctness unless real `ssh rvv` evidence is
   produced.
9. Resolve the inherited untracked stale archive so final Trellis state is
   truthful: this active task is finished and archived, and the final worktree
   is clean after a coherent commit.

## Acceptance Criteria

- [x] MAcc validation declarations/definitions live in
      `RVVEmitCMAccRouteFamilyPlanOwners` and are consumed by selected-body
      owner hooks.
- [x] `RVVSelectedBodyRealization.cpp` has no dead duplicate central branches
      for plain MAcc, scalar-broadcast MAcc, computed-mask MAcc, or
      runtime-scalar computed-mask MAcc inside the generic central realization
      helper.
- [x] The selected-body realization owner registry still exposes active MAcc
      and computed-mask MAcc owners with non-null consumer and realization
      hooks.
- [x] All four MAcc variants still materialize realized typed `tcrv_rvv`
      structure from the selected pre-realized body before provider facts are
      collected.
- [x] MAcc route-family provider-plan verification and route operand-binding
      authority remain owner-local and fail closed on stale/mismatched facts.
- [x] Central provider/planning code consumes owner APIs and neutral shared
      facts; it does not locally reintroduce MAcc owner registries, route ids,
      descriptor/source-front-door authority, exact intrinsic authority, or
      mirror-only authority.
- [x] Focused C++ tests cover positive owner-local selection for plain,
      scalar-broadcast, computed-mask, and runtime-scalar computed-mask MAcc,
      plus mismatched typed-fact diagnostics.
- [x] Representative generated-bundle dry-run or `ssh rvv` evidence is recorded
      for at least one MAcc variant, or an exact blocker is reported.
- [x] `git diff --check` and focused build/test targets pass, or exact blockers
      are reported.
- [x] The stale untracked archived task is removed or superseded by the final
      archived task for this round; final `git status --short` is clean after
      commit.

## Completion Evidence

- Focused build: `ninja -C build tianchenrv-rvv-extension-plugin-test`.
- Focused C++ test: `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- Generated-bundle evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind computed_masked_macc_add --artifact-root artifacts/tmp/rvv_macc_handoff_evidence --run-id codex-stage2-macc-handoff --overwrite`
  returned `dry_run_success`.
- Generated-bundle artifact:
  `artifacts/tmp/rvv_macc_handoff_evidence/codex-stage2-macc-handoff`.
- Full project check: `ninja -C build check-tianchenrv` passed 464/464 tests.
- Whitespace check: `git diff --check` passed.
- Authority scan:
  the generic selected-body helper has no plain/scalar/computed-mask MAcc
  duplicate branches; `RVVEmitCRouteProvider.cpp` has no MAcc-specific branch
  or exact-intrinsic matches; exact intrinsic strings for this handoff are
  concentrated in the MAcc owner/provider-plan source.
- Runtime note: no runtime correctness or performance claim is made; evidence
  is a local generated-bundle dry-run, not `ssh rvv`.

## Out Of Scope

- No computed-mask memory follow-up, segment2, base memory, widening dot/reduce,
  widening conversion, elementwise, standalone reduction, high-level
  frontend/Linalg work, new route-entry table growth, broad cleanup, dashboard
  or report work.
- No compatibility glue that preserves duplicate central MAcc authority.
- No descriptor-driven, source-front-door, artifact-name-derived, script-derived,
  common-EmitC-derived, route-id-derived, exact-intrinsic-derived, or legacy
  i32-derived authority.
- No runtime correctness or performance claim without real `ssh rvv` evidence.

## Technical Notes

- Specs read for this task:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  and `.trellis/spec/guides/plugin-locality-review-guide.md`.
- The existing untracked archive is treated as stale context, not current task
  truth.
- The production files to inspect are the MAcc owner header/source,
  selected-body realization, route provider/planning code, focused plugin C++
  tests, and the generated-bundle script for representative evidence.

## Validation Plan

1. Validate this task context.
2. Inspect the inherited dirty diff and current production code.
3. Implement the bounded MAcc owner-local handoff repair.
4. Run focused C++ build/test targets for the RVV plugin tests.
5. Run representative generated-bundle dry-run or `ssh rvv` evidence for one
   MAcc variant.
6. Run bounded authority scans over touched MAcc/selected-body/provider/test
   files.
7. Run `git diff --check`.
8. Finish/archive this task and create one coherent commit with clean final
   status.
