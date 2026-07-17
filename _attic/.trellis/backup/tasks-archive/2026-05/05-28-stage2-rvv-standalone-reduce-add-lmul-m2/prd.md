# Stage2 RVV plain standalone reduce-add LMUL m2 selected-boundary route member

## Goal

Add `standalone_reduce_add_lmul_m2` as the missing plain standalone
reduction LMUL m2 add member.

The production chain for this task is:

```text
selected tcrv.exec RVV variant
  -> typed/pre-realized tcrv_rvv standalone reduce-add body
  -> RVV plugin-local selected-body realization
  -> realized SEW32 LMUL m2 source/work channel
  -> realized SEW32 LMUL m1 scalar accumulator/result channel
  -> standalone-reduction route-family facts
  -> route materialization facts
  -> math operand-binding facts
  -> route-control provider plan
  -> standalone-reduction statement plan
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC
  -> generated RVV C artifact and ABI mirrors
  -> ssh rvv correctness evidence for runtime/correctness claims
```

The route must derive operation kind, signed i32 dtype, SEW32, LMUL m2
source/work channel, LMUL m1 scalar accumulator/result channel, accumulator
seed binding, lane-0 scalar output, runtime n/AVL/VL, policy, ABI order,
provider mirror facts, C type mapping, and generated artifact facts from typed
RVV body/config/runtime facts. It must not depend on script names, artifact
names, route ids, exact intrinsic spellings, direct pre-realized route-entry
acceptance, common EmitC semantic choices, descriptors, source-front-door
residue, ABI-string guessing, or legacy i32 helper authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV plain standalone reduce-add LMUL m2 selected-boundary route member`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `05bc92f0 rvv: add computed mask reduce add m2 route`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint: one Codex worker; no spawned agents or multi-agent
  workflow.

## What I Already Know

- Commit `05bc92f0` completed
  `computed_mask_standalone_reduce_add_lmul_m2` through selected-boundary
  dry-run, direct route-entry fail-closed behavior, ssh rvv correctness,
  `check-tianchenrv`, archived Trellis state, and a clean worktree.
- The standalone reduction scalar-channel spec requires a distinct source/work
  channel and scalar accumulator/result channel when source LMUL differs from
  RVV scalar reduction result LMUL.
- For SEW32 LMUL m2 standalone reduce-add, the source/work channel is
  `!tcrv_rvv.vector<i32, "m2">` / `vint32m2_t`; the scalar accumulator/result
  channel is `!tcrv_rvv.vector<i32, "m1">` / `vint32m1_t`.
- The archived standalone min/max task completed plain and computed-mask
  standalone min/max LMUL m2 scalar-channel evidence, but the current missing
  family member is the plain unmasked add witness.
- The current task must prove generic production reuse if the selected-body,
  planning/provider, and target/script surfaces already support the member; it
  must not add fake production churn only to create a diff.

## Requirements

1. Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python changes may only update generated-bundle evidence tooling,
   artifact parsing, and small harness support.
2. Add or repair one coherent module for
   `standalone_reduce_add_lmul_m2`.
3. Positive support must flow through the public selected lowering-boundary
   producer with `route_entry_realization=false`.
4. Direct pre-realized route-entry must remain fail-closed for the plain
   standalone reduce-add LMUL m2 member.
5. The selected/pre-realized body must structurally carry signed i32 source
   dtype, SEW32, LMUL m2 source/work config, LMUL m1 scalar accumulator/result
   config, reduction kind `add`, runtime `n`, policy, accumulator seed role,
   scalar output role, and selected ABI order.
6. RVV selected-body realization must consume the pre-realized body into
   realized `setvl` / `with_vl` / load / standalone reduction / scalar lane-0
   store structure before route-family/provider facts are collected.
7. Route planning/provider facts must expose provider-derived source vector
   type/C type, scalar result vector type/C type, route-control facts,
   accumulator seed binding, lane-0 scalar output binding, runtime AVL/VL,
   policy, ABI order, and provider-supported mirrors after route construction.
8. Common EmitC/export must remain neutral and may only materialize the
   provider-built route and mirror provider facts.
9. Runtime/correctness claims require real `ssh rvv` generated-bundle evidence
   for counts `0`, `1`, `16`, `23`, and `257` with signed i32 sources/seeds
   and tail preservation.

## Acceptance Criteria

- [x] A pre-realized selected-body MLIR fixture or equivalent focused existing
      fixture for `standalone_reduce_add_lmul_m2` materializes to
      `setvl`/`with_vl` using SEW32 LMUL m2, loads source vectors as m2, emits
      standalone reduction kind `add`, returns a scalar-result m1 vector, and
      stores the lane-0 scalar result.
- [x] Emission-plan/provider diagnostics for the member include typed
      route-family facts for source vector type/C type
      `!tcrv_rvv.vector<i32, "m2">` / `vint32m2_t`, scalar result vector
      type/C type `!tcrv_rvv.vector<i32, "m1">` / `vint32m1_t`,
      route-control facts, accumulator seed binding, scalar output lane-0
      binding, provider-supported mirror, and ABI order.
- [x] Target header artifact mirrors the rebuilt provider route and does not
      use route id, artifact name, status, script option, exact intrinsic
      spelling, or common EmitC behavior as authority.
- [x] `scripts/rvv_generated_bundle_abi_e2e.py` supports
      `standalone_reduce_add_lmul_m2` in pre-realized selected-body mode and
      records explicit source/scalar-channel/provider mirror fields in dry-run
      evidence.
- [x] Generated-bundle dry-run passes for `standalone_reduce_add_lmul_m2` and
      checks `route_entry_realization=false`, `pre_realized_body_consumed=true`,
      source m2, scalar-result m1, runtime AVL/VL, route-control mirrors,
      accumulator seed, lane-0 scalar output, ABI order, and mirror-only
      artifact fields.
- [x] Direct pre-realized route-entry dry-run rejects
      `standalone_reduce_add_lmul_m2` before provider/common route
      construction.
- [x] Focused C++ provider or lit coverage proves the plain standalone
      reduce-add LMUL m2 route consumes typed source m2 and scalar-result m1
      facts, or explicitly proves existing generic production code already
      handles the member without source churn.
- [x] Non-regression coverage includes current standalone reduction m1,
      standalone min/max m2, computed-mask add/min/max m2, and runtime-scalar
      standalone add m2 paths where focused commands can cover them.
- [x] Bounded touched-file scan finds no new central ad hoc, name-derived,
      metadata-derived, descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived route authority.
- [x] `git diff --check` passes.
- [x] Focused lit/C++/script checks pass.
- [x] `ssh rvv` generated-bundle correctness passes for counts `0`, `1`,
      `16`, `23`, and `257` with signed i32 sources/seeds and tail
      preservation.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded with the task
      left open.
- [x] If complete, the Trellis task is finished, archived, and committed
      coherently.

## Completion Evidence

- Production owner/generic reuse: no RVV C++ production branch was needed; the
  existing standalone reduction family provider consumes the typed realized
  body generically. Focused C++ coverage now proves the plain add LMUL m2 source
  channel (`!tcrv_rvv.vector<i32, "m2">` / `vint32m2_t`) and scalar-result
  channel (`!tcrv_rvv.vector<i32, "m1">` / `vint32m1_t`) facts.
- Selected-boundary target fixture:
  `test/Target/RVV/pre-realized-selected-body-artifact-standalone-reduce-add-lmul-m2.mlir`.
- Generated-bundle dry-run artifact:
  `artifacts/tmp/standalone-add-m2-local/pre-realized-standalone-reduce-add-lmul-m2`.
- `ssh rvv` artifact:
  `artifacts/tmp/standalone-add-m2-ssh/pre-realized-standalone-reduce-add-lmul-m2-ssh`.
- Direct route-entry fail-closed diagnostic covers
  `standalone_reduce_add` and `standalone_reduce_add_lmul_m2`.
- `check-tianchenrv` passed with 456/456 tests.

## Non-goals

- Do not start computed-mask, runtime-scalar-mask, min/max, widening dot, MAcc,
  segment, memory movement, frontend/Linalg, broad dtype/LMUL clone batches,
  route dashboards, broad smoke matrices, evidence-only bookkeeping, or future
  plugin work.
- Do not reintroduce direct route-entry support for standalone reductions.
- Do not add new dtype-prefixed RVV helper op families, one-intrinsic wrapper
  dialects, source-front-door positive routes, descriptor-driven computation,
  or common EmitC semantic inference.
- Do not make route ids, artifact names, script options, test names, exact
  intrinsic spellings, ABI strings, mirror metadata, or helper-only reports the
  executable authority for this member.

## Relevant Specs And Prior Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/capability-first-design-guide.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`
- `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-standalone-reduction-scalar-channel/prd.md`
- `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-standalone-minmax-scalar-channel/prd.md`
- `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-computed-mask-standalone-reduce-add-lmul-m2/prd.md`

## Technical Notes

- Primary production/evidence surfaces to inspect or change:
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
  - existing standalone-reduce MLIR fixtures and script tests
- Current starting hypothesis is that C++ selected-body and provider code may
  already support this member generically through the scalar-channel work; the
  implementation must verify that through focused production/provider evidence
  and only add production changes if repo evidence shows an actual rejection or
  special-case gap.
