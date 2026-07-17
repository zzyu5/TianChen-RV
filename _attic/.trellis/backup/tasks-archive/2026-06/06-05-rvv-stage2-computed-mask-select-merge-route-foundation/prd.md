# Stage2 RVV computed-mask select/merge selected-body route foundation

## Goal

Establish one bounded Stage 2 route-supported computed-mask vector
select/merge proof route. A selected `tcrv.exec` RVV variant must carry either
an explicit typed `tcrv_rvv` body or a pre-realized selected body where a
vector compare produces an in-body predicate mask, that mask selects between
explicit true-value and false-value vector operands, and the selected result is
stored to the output buffer:

```text
active lanes:   out[i] = true_value[i]
inactive lanes: out[i] = false_value[i]
```

The route must flow through RVV plugin-owned selected-body realization and
route planning, provider-built `TCRVEmitCLowerableRoute`, Common EmitC neutral
materialization, and target artifact validation. Executable `ssh rvv` evidence
is in scope only if the existing generated-bundle path is reachable within the
bounded task.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-mask select/merge selected-body route foundation`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `190209c6 rvv: close computed-mask macc route foundation`.
- No `.trellis/.current-task` existed at the start of the round, so this task
  was created before source edits.
- The preceding computed-mask MAcc task proved provider-owned computed-mask
  MAcc facts and real `ssh rvv` correctness for explicit and pre-realized
  selected bodies.
- Current source inspection shows existing production surfaces for
  `computed_mask_select`: `typed_computed_mask_select_pre_realized_body`,
  generic `tcrv_rvv.select`, compare/select statement-plan owners, provider
  route-family validation, target artifact mirror validation, explicit and
  pre-realized target fixtures, and generated-bundle script support.
- Existing pre-realized target fixture
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-select.mlir`
  already checks positive route/header facts and some stale mirror rejection
  for mask producer and mask/tail policy fields.
- Existing explicit fixture
  `test/Target/RVV/explicit-selected-body-artifact-computed-mask-select-sle.mlir`
  already proves a selected explicit typed `tcrv_rvv.select` route with
  predicate `sle`, operand binding, provider mirror, and target header output.
- Focused route/header FileCheck checks and `tianchenrv-target-artifact-export-test`
  pass before code edits, confirming the production route is already real and
  this round should close focused evidence gaps rather than invent new route
  semantics.

## Requirements

- Keep scope to one vector-compare computed-mask select/merge route family:
  `computed_mask_select`.
- Preserve the authority chain:
  selected typed `tcrv_rvv` body/config/runtime facts -> RVV selected-body
  realization/validation -> RVV route-family provider facts -> lowerable route
  -> Common EmitC/export -> target artifact mirrors -> optional generated
  bundle runtime evidence.
- The route/provider/target path must structurally carry or derive:
  - ABI roles and order `cmp_lhs,cmp_rhs,true_value,false_value,out,n`;
  - compare predicate and compare-produced mask role/source/form;
  - true/false operand binding and result/output layout;
  - select/merge result layout
    `select-true-value-when-mask-else-false-value`;
  - element type, SEW, LMUL, tail policy, mask policy, config contract,
    runtime AVL/VL boundary, and route-control plan;
  - route operand-binding plan/summary with exported header/prototype
    participation markers;
  - required headers, C type mapping, compare/select/store leaves, and
    `provider_supported_mirror`.
- Common EmitC/export may only carry provider-built route payloads and metadata
  mirrors. It must not infer select semantics, mask semantics, dtype, SEW/LMUL,
  runtime ABI roles, intrinsic spelling, or true/false lane behavior.
- Stale or missing facts must fail closed before target artifact acceptance:
  predicate, mask role/source/form, true/false operand binding, select layout,
  runtime ABI/order, route operand binding, SEW/LMUL/policy, header/type
  mapping, compare/select/store leaves, target leaf profile, provider support
  mirror, and cross-family stale route-family mirrors.
- If production code already supports the route, finish by proving focused
  positive and negative evidence instead of adding helper-only changes.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` describe this bounded
      computed-mask select/merge route foundation and cite the relevant specs.
- [x] Focused code inspection identifies whether current production
      provider/route/target/generated-bundle support for `computed_mask_select`
      is complete or where it fails.
- [x] Positive route/artifact/header evidence proves provider-derived
      computed-mask select facts for predicate, mask role/source/form,
      true/false operand binding, select layout, result/output layout, runtime
      AVL/VL, runtime ABI order, operand binding, typed config,
      header/type/intrinsic mapping, target leaf profile, and
      `provider_supported_mirror`.
- [x] Negative evidence proves stale or missing predicate, mask facts,
      true/false operand binding, select layout, runtime ABI/order, operand
      binding, SEW/LMUL/policy, header/type/intrinsic mapping, provider support
      mirror, and cross-family residue fail closed before artifact acceptance.
- [x] Common EmitC/export remains neutral and does not choose computed-mask
      select/merge semantics.
- [x] Focused `tcrv-opt` / `tcrv-translate` / RVV provider / target artifact
      checks for changed behavior pass.
- [x] If executable behavior is claimed, generated artifact plus real
      `ssh rvv` correctness passes over counts including `0`, `1`, a VL
      boundary, a tail case, and a multi-chunk case, with mixed true/false lane
      selection, scalar oracle coverage, source preservation, and tail sentinel
      preservation.
- [x] If executable closure is too large, finish route-supported plus
      target-validation closure and record generated-bundle `ssh rvv` as the
      exact next continuation point. Not applicable: executable closure passed.
- [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, dtype-prefixed
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor, source-front-door,
      source-artifact, route-string/artifact-name/ABI-string/test-name,
      exact-intrinsic-spelling, or Common EmitC semantic authority.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation,
      final clean worktree, archive, and one coherent commit complete the round
      if the module behavior is complete.

## Evidence Plan

- Run focused positive route/header checks for explicit and pre-realized
  `computed_mask_select`.
- Inspect provider and target validation for stale predicate, mask facts,
  select layout, operand binding, provider mirror, and cross-family route-family
  mirror rejection.
- Add or repair production provider/target validation only where live evidence
  shows missing route-supported behavior or stale fact rejection.
- Prefer target artifact and provider negative tests for fail-closed mirror and
  route payload validation.
- Extend generated-bundle dry-run and `ssh rvv` evidence only if the route
  reaches generated artifact form without broad harness work.

## Definition Of Done

- Current HEAD either has route-supported computed-mask select/merge
  selected-body provider and target artifact validation closure, or records the
  precise production blocker and next continuation point without false
  executable claims.
- Any implementation changes are production-path changes or focused evidence
  for already-existing production support; helper/test/spec-only work is not a
  completion unless production support was already real.
- Specs are updated only if this round discovers a durable rule not already
  captured in `.trellis/spec/`.
- The task is finished/archived and one coherent commit is created when the
  module behavior is complete.

## Out Of Scope

- High-level Linalg, Vector, StableHLO, or source-front-door lowering.
- Broad dtype/LMUL/predicate/select matrix expansion.
- Runtime-scalar compare/select, dual compare/select, f32 clamp/select,
  reductions, MAcc, widening dot, Gearbox, TensorExt, IME, Offload, performance
  benchmarking, autotuning, or dashboards.
- One-intrinsic wrapper routes.
- Descriptor-driven computation or Common EmitC semantic inference.
- Treating route ids, artifact names, ABI strings, test names, intrinsic
  spellings, status fields, manifests, or metadata mirrors as route authority.

## Technical Notes

Specs read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Relevant archived tasks read:

- `.trellis/tasks/archive/2026-06/06-05-06-05-rvv-stage2-computed-mask-macc-selected-body-route-foundation/prd.md`
- `.trellis/tasks/archive/2026-06/06-05-rvv-stage2-computed-mask-standalone-reduce-add-route/prd.md`

Primary inspection surfaces from the direction brief:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/`
- `lib/Plugin/RVV/Construction/`
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/*computed-mask-select*`
- `test/Target/TargetArtifactExportTest.cpp`

## Completion Evidence

Completed as executable computed-mask select/merge selected-body route
foundation. Production provider, route planning, selected-body realization,
target artifact validation, generated bundle support, and harness behavior
already supported the route. This round closed focused evidence gaps by adding
target fixture stale-mirror checks and by running real `ssh rvv` correctness
for both pre-realized and explicit selected-body inputs.

Focused fixture changes:

- Added target artifact fail-closed checks for stale
  `tcrv_rvv.compare_predicate_kind`, `tcrv_rvv.runtime_abi_order`,
  `tcrv_rvv.route_operand_binding_plan`,
  `tcrv_rvv.route_operand_binding_operands`,
  `tcrv_rvv.provider_supported_mirror`, `tcrv_rvv.target_leaf_profile`,
  `tcrv_rvv.mask_role`, `tcrv_rvv.mask_source`,
  `tcrv_rvv.mask_memory_form`, and `tcrv_rvv.select_layout`.
- Retained existing stale checks for computed-mask producer source,
  tail/mask policy, mask-tail policy plan, and mask-tail policy owner.

Provider and target route facts proved:

- Operation and typed compute: `computed_mask_select`, `tcrv_rvv.select`.
- Runtime ABI order: `cmp_lhs,cmp_rhs,true_value,false_value,out,n`.
- Binding summary:
  `rvv-route-operand-binding:computed_mask_select.v1` with `abi` and `hdr`
  participation for all exported operands.
- Mask/predicate facts: predicates `slt` and `sle`, mask role
  `predicate-mask-produced-by-compare`, mask source
  `compare-produced-mask-same-vl-scope`, mask memory form
  `compare-produced-mask`.
- Select/merge layout:
  `select-true-value-when-mask-else-false-value`.
- Target leaf/profile facts:
  `rvv-v1-typed-computed-mask-select-leaf-profile.v1`,
  `provider_supported_mirror:rvv-computed-mask-select-plan-validated`,
  headers `stddef.h,stdint.h,riscv_vector.h`, and C type mapping
  `vl:size_t,compare:true_false:typed-vector,mask:typed-mask,result:typed-vector`.

Generated artifact evidence:

- Pre-realized dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/computed-mask-select-foundation-pre-realized-dryrun`
- Explicit dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/computed-mask-select-foundation-explicit-sle-dryrun`
- Pre-realized `ssh rvv`:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/computed-mask-select-foundation-pre-realized-ssh`
- Explicit `ssh rvv`:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/computed-mask-select-foundation-explicit-sle-ssh`

Both real RVV runs passed:

```text
PASS op=computed_mask_select counts=0,1,16,17,257 compare_data_patterns=2
PASS op=computed_mask_select_sle counts=0,1,16,17,257 compare_data_patterns=2
```

Self-repair:

- Direct explicit generated-bundle invocation with `--op-kind computed_mask_select`
  failed because the script supports that key only for pre-realized selected
  bodies; the explicit supported key is `computed_mask_select_sle`. Re-ran
  explicit dry-run and `ssh rvv` with `computed_mask_select_sle`, which passed.

Old-authority scan:

- Code/test added-line scan over
  `test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-select.mlir`
  produced no matches for legacy `RVVI32M1`, `rvv-i32m1`, dtype-prefixed
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-artifact,
  descriptor, exact intrinsic spelling, or route-string/artifact-name/ABI-string
  authority terms.
- Full staged added-line scan matched only Trellis task/check guardrail text
  that rejects those authority sources, not code/test positive route authority.

Final checks:

- `git diff --check` passed.
- `git diff --cached --check` passed.
- Trellis task context validation passed before and after archive.
