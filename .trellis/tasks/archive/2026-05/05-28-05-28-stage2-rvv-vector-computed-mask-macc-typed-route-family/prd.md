# Stage2 RVV vector computed-mask MAcc typed route-family derivation

## Goal

Extend the production `computed_masked_macc_add` selected-boundary path from
its current SEW32/LMUL m1-only vector computed-mask MAcc shape into a bounded
typed RVV route-family derivation with a SEW32/LMUL m2 witness.

The bounded production chain for this task is:

```text
selected tcrv.exec RVV variant
  -> typed/pre-realized tcrv_rvv vector computed_masked_macc_add body
  -> RVV plugin-local selected-body realization
  -> realized typed tcrv_rvv setvl / with_vl / compare lhs/rhs loads
     / payload loads / accumulator load / compare / masked_macc / store body
  -> computed-mask accumulation route-family facts
  -> route materialization facts
  -> math operand-binding facts
  -> route-control provider plan
  -> computed-mask accumulation statement plan
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI mirrors
  -> generated RVV C artifact and harness
  -> ssh rvv correctness evidence for runtime/correctness claims
```

The route must be supported because the typed body/config/runtime facts carry
operation kind, predicate, mask producer/use, compare lhs/rhs bindings, payload
lhs/rhs bindings, accumulator/result channel, SEW, LMUL, mask type, policy,
runtime n/AVL/VL, ABI order, inactive/merge semantics, and provider route
facts. It must not be supported because of op names, route ids, artifact names,
scripts, descriptors, exact intrinsic spellings, source-front-door residue, ABI
strings, common EmitC behavior, direct route-entry acceptance, or legacy i32
helper authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV vector computed-mask MAcc typed route-family derivation`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `d71d3db8 rvv: derive runtime scalar masked macc m2 route`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint: one Codex worker, no spawned agents, no parallel
  agent workflows.

## Current Repository Facts To Verify

- The immediately preceding runtime-scalar compare-masked MAcc task completed a
  SEW32/LMUL m2 selected-boundary derivation while keeping direct pre-realized
  route-entry unsupported.
- Bounded inspection is expected to show the adjacent vector
  `computed_masked_macc_add` owner still has m1-only gates:
  - `TypedComputedMaskMAccPreRealizedBodyOp::verify()` accepts only
    SEW32/LMUL m1.
  - Selected-body realization validation for computed-mask MAcc accepts only
    SEW32/LMUL m1.
  - `deriveRVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan(...)` treats
    vector masked MAcc as SEW32/LMUL m1-only.
  - Generated-bundle or target fixture coverage lacks a
    `computed_masked_macc_add` SEW32/LMUL m2 target witness.
- These facts must be rechecked against live source before implementation.

## Requirements

1. Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python may only update evidence tooling, generated-bundle
   harnesses, artifact parsing checks, or small support scripts.
2. Preserve the existing baseline SEW32/LMUL m1 selected-boundary
   `computed_masked_macc_add` path.
3. Add one bounded non-baseline typed witness for vector
   `computed_masked_macc_add`: SEW32/LMUL m2.
4. Extend RVV dialect/pre-realized body verification and selected-body
   realization validation so vector computed-mask MAcc accepts only
   structurally valid typed configs. Unsupported SEW64 and unsupported LMULs
   must fail closed with targeted diagnostics.
5. Extend route planning/provider facts so computed-mask MAcc route-family
   support derives vector type, mask type, setvl/load/compare/MAcc/store leaves,
   target leaf profile, C type mapping, runtime ABI order, route-control facts,
   materialization facts, math operand-binding facts, and statement plan from
   typed SEW/LMUL/body facts.
6. Update target artifact/generated-bundle boundary code as needed so m1/m2
   mirrors are consumed only after provider route construction and are never
   route authority.
7. Keep direct pre-realized route-entry unsupported for baseline m1 and the new
   m2 witness. Positive support must flow through selected-boundary realization
   with `route_entry_realization=false`.
8. Preserve non-regression for the completed runtime-scalar MAcc m2 path and
   runtime-scalar standalone min/max scalar-channel paths.
9. Do not expand unrelated families, add dtype/LMUL clone matrices, introduce
   one-intrinsic wrapper dialects, create high-level Linalg/frontend lowering,
   revive source-front-door or descriptor-driven compute paths, or move RVV
   semantics into common EmitC/export.

## Acceptance Criteria

- [ ] `TypedComputedMaskMAccPreRealizedBodyOp` accepts SEW32/LMUL m1 and
      SEW32/LMUL m2 only when op kind, predicate, mask role, compare input
      roles, payload input roles, accumulator role/layout, result layout,
      policy, pointer C types, runtime n binding, and selected-boundary shape
      are structurally valid.
- [ ] RVV selected-body realization validates the same typed config facts and
      realizes m2 bodies into `setvl`, `with_vl`, compare lhs/rhs loads,
      payload lhs/rhs loads, accumulator load, compare mask, `masked_macc`, and
      store with m2 vector/mask types.
- [ ] `deriveRVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan(...)`,
      route materialization facts, route-control provider plan, math
      operand-binding facts, and computed-mask accumulation statement plan
      accept the m2 typed vector MAcc body and expose provider-derived facts
      for dtype, SEW, LMUL, vector type/C type, mask type/C type, compare
      binding, payload binding, accumulator/result binding, MAcc operation,
      inactive/merge semantics, VL/AVL, policy, ABI order, and intrinsic
      mapping.
- [ ] Target support and generated-bundle evidence consume provider-derived m1
      and m2 mirrors after route construction, including target leaf profile, C
      type mapping, header declarations, ABI order, route operand binding,
      provider-supported mirror, route-control fields, and statement leaves.
- [ ] Direct pre-realized route-entry remains fail-closed for
      `computed_masked_macc_add`, including the m2 witness.
- [ ] Focused fail-closed coverage rejects wrong dtype/config, unsupported LMUL
      or SEW64 MAcc, stale e32m1 mirror, wrong mask producer/use, wrong operand
      binding, wrong accumulator/result channel, wrong AVL/VL relation, missing
      capability, direct-route-entry-only authority, artifact-name/script-derived
      authority, exact-intrinsic-as-authority, and common-EmitC semantic
      invention where current hooks expose them.
- [ ] Generated-bundle dry-run passes for baseline m1 and non-baseline m2
      selected-boundary `computed_masked_macc_add`, with
      `route_entry_realization=false` and selected-body realization/provider
      facts recorded.
- [ ] Real `ssh rvv` generated-bundle compile/run/correctness passes for the
      changed owner over counts including `0`, `1`, exact-VL, tail, and stress
      cases with signed i32 inputs, active/inactive masks, accumulator
      preservation, multiply-add distinguishing data, and tail preservation.
- [ ] Non-regression passes for completed runtime-scalar MAcc m2 and
      runtime-scalar standalone min/max scalar-channel paths.
- [ ] Bounded touched-file authority scan finds no new route or executable
      claim depending on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, direct-route-entry-only,
      pre-realized-fixture-only, or legacy-i32-derived authority.
- [ ] `git diff --check` passes.
- [ ] Focused lit/C++ checks pass, and `check-tianchenrv` passes or an exact
      blocker is recorded with the task left open.
- [ ] Trellis task status is truthful; if complete, the task is finished,
      archived, and committed coherently.

## Non-goals

- No runtime-scalar expansion beyond completed m1/m2 witnesses.
- No standalone reduction, min/max reduction, select, memory, segment2,
  widening conversion, widening dot, scalar, TensorExt, IME, Offload, frontend
  generalization, or Stage3 work.
- No dtype/LMUL clone matrix. This task proves exactly one bounded typed-family
  movement for vector computed-mask MAcc.
- No direct route-entry shortcut restoration.
- No descriptor-driven computation, source-front-door positive route,
  direct-C/source exporter authority, route-id authority, artifact-name
  authority, exact-intrinsic-as-authority, or common EmitC semantic invention.

## Relevant Specs And Prior Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-runtime-scalar-cmp-masked-macc-typed-route-family/prd.md`

## Technical Notes

- Primary production surfaces to inspect/change:
  - `lib/Dialect/RVV/IR/RVVDialect.cpp`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - focused `test/Plugin` and `test/Target/RVV` consumers.
- Implementation should reuse existing typed config profile helpers rather than
  adding route-id, ABI-string, exact-intrinsic, script, or artifact-name special
  cases.
