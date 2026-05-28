# Stage2 RVV runtime-scalar cmp-masked MAcc typed route-family derivation

## Goal

Generalize the production `runtime_scalar_cmp_masked_macc_add` selected-boundary
path from the current baseline SEW32/LMUL m1 shape into a typed RVV
route-family derivation with at least one non-baseline SEW32/LMUL m2 witness.

The bounded production chain for this task is:

```text
selected tcrv.exec RVV variant
  -> typed/pre-realized tcrv_rvv runtime_scalar_cmp_masked_macc_add body
  -> RVV plugin-local selected-body realization
  -> realized typed tcrv_rvv setvl / with_vl / compare lhs load
     / runtime scalar splat / payload loads / accumulator load / compare
     / masked_macc / store body
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
operation kind, dtype, SEW, LMUL, policy, runtime scalar threshold, mask
producer/use, accumulator/result channels, runtime n/AVL/VL, ABI order, and
inactive-lane merge semantics. It must not be supported because of op names,
route ids, artifact names, scripts, descriptors, exact intrinsic spellings,
source-front-door residue, ABI strings, common EmitC behavior, direct
route-entry acceptance, or legacy i32 helper authority.

## Direction Source

- Direction title: `Switch: Stage2 RVV runtime-scalar compare-masked MAcc typed route-family derivation`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `f48aa5c8 rvv: add runtime scalar masked minmax reductions`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint: one Codex worker, no subagents or parallel agent
  workflows.

## Current Repository Facts

- The preceding min/max standalone scalar-channel task completed
  runtime-scalar compare-masked standalone min/max reductions through typed
  selected-body realization, provider-derived route facts, generated artifacts,
  `ssh rvv` evidence, and archived Trellis state.
- The archived runtime-scalar MAcc selected-realization task completed the
  selected-boundary `runtime_scalar_cmp_masked_macc_add` path and demoted direct
  pre-realized route-entry authority.
- Current code still carries baseline e32m1-specific residue for this owner:
  - `TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp::verify()` requires
    SEW32/LMUL m1.
  - `validatePreRealizedRVVSelectedRuntimeScalarComputedMaskMAccBody(...)`
    requires SEW32/LMUL m1.
  - `deriveRVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan(...)`
    accepts computed-mask MAcc only for SEW32/LMUL m1, while adjacent
    runtime-scalar select, memory, and standalone-reduction families already
    accept typed m2 or other explicitly bounded typed configs.
  - Generated-bundle constants and target leaf/profile mirrors still say e32m1
    for `runtime_scalar_cmp_masked_macc_add`.
- The selected-body realization implementation already threads
  `runtimeControlPlan->sew` and `runtimeControlPlan->lmul` into realized
  `setvl`, `with_vl`, loads, splat, compare, masked MAcc, and store. The main
  missing production movement is validation/provider/artifact boundary support
  for non-baseline typed facts, not a new direct route-entry path.
- The planning layer already has typed config profiles and intrinsic/type
  helpers for SEW32/LMUL m2 (`vint32m2_t`, `vbool16_t`,
  `__riscv_vsetvl_e32m2`, `__riscv_vle32_v_i32m2`,
  `__riscv_vmv_v_x_i32m2`, `__riscv_vmsle_vv_i32m2_b16`,
  `__riscv_vmacc_vv_i32m2`, and `__riscv_vse32_v_i32m2`).
- Direct pre-realized route-entry for this owner must remain unsupported. The
  positive path is selected-boundary realization with
  `route_entry_realization=false`.

## Requirements

1. Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python may only update evidence tooling, generated-bundle
   harnesses, and artifact parsing checks.
2. Preserve the existing baseline SEW32/LMUL m1 selected-boundary path.
3. Add at least one non-baseline typed witness for
   `runtime_scalar_cmp_masked_macc_add`, specifically SEW32/LMUL m2 unless live
   evidence proves it structurally illegal.
4. Extend RVV dialect/pre-realized body verification and selected-body
   realization validation so the runtime-scalar computed-mask MAcc body accepts
   only structurally valid typed configs. SEW64 or unsupported LMULs must fail
   closed with targeted diagnostics instead of being cloned forward.
5. Extend route planning/provider facts so computed-mask MAcc route-family
   support derives vector type, mask type, setvl/load/splat/compare/MAcc/store
   leaves, target leaf profile, C type mapping, runtime ABI order, route-control
   facts, and statement plan from typed SEW/LMUL/body facts.
6. Update target artifact/generated-bundle boundary code so e32m1/e32m2 mirrors
   are consumed only after provider route construction and are never route
   authority.
7. Preserve direct pre-realized route-entry fail-closed behavior for
   `runtime_scalar_cmp_masked_macc_add`.
8. Preserve adjacent runtime-scalar standalone reduction min/max behavior and
   baseline runtime-scalar MAcc behavior.
9. Do not expand unrelated families, add dtype/LMUL clone batches, introduce
   one-intrinsic wrapper dialects, create high-level Linalg/frontend lowering,
   revive source-front-door or descriptor-driven compute paths, or move RVV
   semantics into common EmitC/export.

## Acceptance Criteria

- [ ] `TypedRuntimeScalarComputedMaskMAccPreRealizedBodyOp` accepts
      SEW32/LMUL m1 and SEW32/LMUL m2 only when op kind, predicate, mask role,
      mask source/form, accumulator role/layout, result layout, policy,
      runtime scalar binding, pointer C types, and runtime n binding are
      structurally valid.
- [ ] RVV selected-body realization validates the same typed config facts and
      realizes m2 bodies into `setvl`, `with_vl`, compare lhs load, runtime
      scalar splat, payload loads, accumulator load, compare mask,
      `masked_macc`, and store with m2 vector/mask types.
- [ ] `deriveRVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan(...)`,
      route materialization facts, route-control provider plan, math
      operand-binding facts, and computed-mask accumulation statement plan
      accept the m2 typed MAcc body and expose provider-derived facts for
      dtype, SEW, LMUL, vector type/C type, mask type/C type, runtime scalar
      threshold type, accumulator/result binding, MAcc operation,
      inactive/merge semantics, VL/AVL, policy, ABI order, and intrinsic
      mapping.
- [ ] Target support and generated-bundle evidence consume provider-derived m1
      and m2 mirrors after route construction, including target leaf profile,
      C type mapping, header declarations, ABI order, route operand binding,
      provider-supported mirror, route-control fields, and statement leaves.
- [ ] Direct pre-realized route-entry remains fail-closed for
      `runtime_scalar_cmp_masked_macc_add`, including the m2 witness.
- [ ] Focused fail-closed coverage rejects wrong dtype/config, unsupported LMUL
      or SEW64 MAcc, stale e32m1 mirror, wrong runtime scalar binding, wrong
      mask producer/use, wrong accumulator/result channel, wrong AVL/VL
      relation, missing capability, direct-route-entry-only authority,
      artifact-name/script-derived authority, exact-intrinsic-as-authority, and
      common-EmitC semantic invention where current hooks expose them.
- [ ] Generated-bundle dry-run passes for baseline m1 and non-baseline m2
      selected-boundary `runtime_scalar_cmp_masked_macc_add`, with
      `route_entry_realization=false` and selected-body realization facts
      recorded.
- [ ] Real `ssh rvv` generated-bundle runs pass for the changed owner over
      counts including `0`, `1`, exact-VL, tail, and stress cases with signed
      inputs, at least two runtime scalar thresholds, active/inactive masks,
      accumulator preservation, and tail preservation.
- [ ] Non-regression passes for baseline `runtime_scalar_cmp_masked_macc_add`
      and the completed runtime-scalar standalone min/max scalar-channel path.
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

- No standalone reduction expansion, min/max reduction work, select, memory,
  segment2, widening conversion, widening dot, non-runtime MAcc, scalar,
  TensorExt, IME, Offload, or frontend generalization work.
- No dtype/LMUL clone matrix. This task proves exactly the bounded typed-family
  movement needed for runtime-scalar computed-mask MAcc.
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
- `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-runtime-scalar-cmp-masked-standalone-minmax-scalar-channel/prd.md`
- `.trellis/tasks/archive/2026-05/05-27-05-27-stage2-rvv-runtime-scalar-macc-selected-realization/prd.md`
- `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-runtime-scalar-computed-mask-macc-boundary/prd.md`

## Technical Notes

- Primary production surfaces to inspect/change:
  - `lib/Dialect/RVV/IR/RVVDialect.cpp`
  - `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
  - `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - focused `test/Plugin` and `test/Target/RVV` consumers.
- Existing code search found the main m1-only gates in the RVV dialect
  verifier, selected-body realization validator, and computed-mask
  accumulation family plan derivation.
- The implementation should reuse existing typed config profile helpers rather
  than adding route-id or artifact-name special cases.
