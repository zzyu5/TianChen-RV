# Stage2 RVV computed-mask standalone reduce-add LMUL m2 typed route-family

## Goal

Add `computed_mask_standalone_reduce_add_lmul_m2` as one bounded typed RVV
selected-boundary route-family witness.

The production chain for this task is:

```text
selected tcrv.exec RVV variant
  -> typed/pre-realized tcrv_rvv computed-mask standalone reduce-add body
  -> RVV plugin-local selected-body realization
  -> realized setvl / with_vl / compare lhs/rhs loads / source load
     / compare-produced mask / masked_standalone_reduce / scalar store facts
  -> standalone-reduction route-family facts
  -> computed-mask accumulation shared facts
  -> route materialization facts
  -> math operand-binding facts
  -> route-control provider plan
  -> standalone-reduction statement plan
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC
  -> generated target artifact and ABI mirrors
  -> ssh rvv correctness evidence for runtime/correctness claims
```

The route must derive operation kind, signed i32 source/result dtype, SEW32,
LMUL m2 source/work channel, LMUL m1 scalar accumulator/result channel,
computed mask producer/use, tail/mask policy, runtime n/AVL/VL, accumulator
seed layout, lane-0 scalar output layout, ABI order, provider facts, C type
mapping, and generated artifact facts from typed RVV body/config/runtime facts.
It must not depend on generated-bundle names, route ids, artifact names, scripts,
exact intrinsic spellings, direct route-entry acceptance, common EmitC semantic
choices, descriptors, source-front-door residue, ABI-string guessing, or legacy
i32 helper authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV computed-mask standalone reduce-add LMUL m2 typed route-family`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `63c19470 rvv: derive computed mask macc m2 route`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint: one Codex worker; no spawned agents or multi-agent
  workflow.

## Current Repository Facts

- Existing C++ verifier/selected-body realization helpers already express
  bounded standalone-reduction source/work configs as SEW32 LMUL m1 or m2 with
  a separate LMUL m1 scalar result channel.
- Existing computed-mask standalone min/max m2 fixtures and generated-bundle
  entries prove the adjacent m2 skeleton for min/max, not the reduce-add m2
  witness.
- Existing runtime-scalar computed-mask standalone reduce-add m2 support is a
  different route family member and must remain non-regressed.
- Existing pre-realized `computed_mask_standalone_reduce_add` target fixture is
  m1-only.
- `scripts/rvv_generated_bundle_abi_e2e.py` currently lists
  `computed_mask_standalone_reduce_add`, `computed_mask_standalone_reduce_min_lmul_m2`,
  and `computed_mask_standalone_reduce_max_lmul_m2`, but not
  `computed_mask_standalone_reduce_add_lmul_m2`.
- The focused missing peer is therefore the computed-mask standalone reduce-add
  LMUL m2 selected-boundary/generated-bundle witness, plus C++ provider evidence
  that the add route consumes source m2 and scalar-result m1 typed facts.

## Requirements

1. Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python changes may only update generated-bundle evidence tooling
   and artifact parsing/harness expectations.
2. Preserve the existing baseline `computed_mask_standalone_reduce_add` m1 path.
3. Add one bounded non-baseline witness:
   `computed_mask_standalone_reduce_add_lmul_m2`.
4. Do not introduce a new dtype-prefixed op, a one-intrinsic wrapper, a
   descriptor-driven route, a source-front-door positive path, or a direct
   pre-realized route-entry shortcut.
5. Positive support must flow through selected-boundary realization with
   `route_entry_realization=false`.
6. The realized route must preserve source/work `!tcrv_rvv.vector<i32, "m2">`
   and mask `!tcrv_rvv.mask<i32, "m2">`, while the scalar accumulator/result
   channel is `!tcrv_rvv.vector<i32, "m1">` / `vint32m1_t`.
7. Provider facts must expose the source vector type/C type, scalar result
   vector type/C type, route operand binding, route-control mirrors, computed
   mask facts, zero-inactive add contract, and ABI order as provider-derived
   mirrors after route construction.
8. Direct pre-realized route-entry must remain fail-closed for
   `computed_mask_standalone_reduce_add_lmul_m2`.
9. Runtime/correctness claims require real `ssh rvv` evidence for counts
   including `0`, `1`, exact-VL, tail, and stress cases with signed compare
   inputs, signed source values, mask-active and mask-inactive lanes, and
   accumulator seed contribution.

## Acceptance Criteria

- [ ] A pre-realized selected-body MLIR fixture for
      `computed_mask_standalone_reduce_add_lmul_m2` materializes to
      `setvl`/`with_vl` using SEW32 LMUL m2, loads compare/source vectors as
      m2, produces an m2 compare mask, emits `masked_standalone_reduce` kind
      `add`, returns a scalar-result m1 vector, and stores the lane-0 scalar
      result.
- [ ] Emission-plan/provider diagnostics for the new fixture include typed
      route-family facts for source vector type/C type `!tcrv_rvv.vector<i32,
      "m2">` / `vint32m2_t`, scalar result vector type/C type
      `!tcrv_rvv.vector<i32, "m1">` / `vint32m1_t`, route-control facts,
      computed-mask facts, zero-inactive add contract, provider-supported
      mirror, and ABI order `cmp_lhs,cmp_rhs,src,acc,out,n`.
- [ ] Target header artifact mirrors the rebuilt provider route and does not
      use route id, artifact name, status, script option, exact intrinsic
      spelling, or common EmitC behavior as authority.
- [ ] `scripts/rvv_generated_bundle_abi_e2e.py` supports
      `computed_mask_standalone_reduce_add_lmul_m2` in pre-realized
      selected-body mode and records explicit typed-vector/typed-mask/provider
      mirror fields in dry-run evidence.
- [ ] Generated-bundle dry-run passes for
      `computed_mask_standalone_reduce_add_lmul_m2` and checks
      `route_entry_realization=false`, `pre_realized_body_consumed=true`,
      source m2, scalar-result m1, runtime AVL/VL, route-control mirrors, and
      zero-inactive add semantics.
- [ ] Direct pre-realized route-entry dry-run rejects
      `computed_mask_standalone_reduce_add_lmul_m2` before provider/common
      route construction.
- [ ] Focused C++ provider coverage proves the computed-mask standalone
      reduce-add m2 route-family analysis consumes typed source m2 and scalar
      result m1 facts, and retains fail-closed checks for stale mask/source,
      stale inactive-lane, stale scalar-result, and wrong scalar output ABI
      mirrors.
- [ ] Non-regression coverage includes the completed
      `computed_masked_macc_add_lmul_m2` path, existing runtime-scalar
      standalone reduce-add m2 path, and existing computed-mask standalone
      min/max m2 paths.
- [ ] Bounded touched-file scan finds no new central ad hoc, name-derived,
      metadata-derived, descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived route authority.
- [ ] `git diff --check` passes.
- [ ] Focused lit/C++ checks pass, and `check-tianchenrv` passes or an exact
      blocker is recorded with the task left open.
- [ ] If complete, the Trellis task is finished, archived, and committed
      coherently.

## Non-goals

- Do not expand arbitrary LMUL values beyond the existing bounded m1/m2 witness.
- Do not add plain `standalone_reduce_add_lmul_m2` unless it is proven to be the
  single blocking dependency; current scope is computed-mask reduce-add m2.
- Do not add i64 clone batches, widening dot, MAcc, segment2, compare/select,
  memory movement, high-level Linalg/frontend lowering, dashboards, broad smoke
  matrices, one-intrinsic wrapper dialects, or evidence-only changes.
- Do not add proof-only coverage for completed computed-mask MAcc m2,
  runtime-scalar MAcc m2, standalone min/max m2, compare/select m2, or previous
  direct-route demotions except as explicit non-regression checks.

## Relevant Specs And Prior Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-vector-computed-mask-macc-typed-route-family/prd.md`

## Technical Notes

- Primary production/evidence surfaces to inspect or change:
  - `lib/Dialect/RVV/IR/RVVDialect.cpp`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Target/RVV`
  - `test/Scripts`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
- Current inspection suggests C++ compiler route skeletons already support the
  necessary m2 typed facts through generic standalone-reduction helpers; the
  implementation should update C++ tests and target/generated-bundle consumers
  rather than adding redundant semantic branches.
