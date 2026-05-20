# Stage2 RVV masked strided-input widening dot-reduction executable slice

## Goal

Implement one bounded Stage2 RVV computed-mask runtime-strided-input signed
widening dot-product reduction executable slice:

```text
out_i32[0] = acc_seed_i32
           + sum_i ((cmp_lhs_i32[i] < cmp_rhs_i32[i])
                    ? sign_extend(lhs_i16[i * lhs_stride])
                      * sign_extend(rhs_i16[i * rhs_stride])
                    : 0)
```

The selected RVV path must carry compare-produced mask provenance, typed signed
i16 source roles for `lhs` and `rhs`, runtime element-stride ABI roles for each
source, an i32 scalar accumulator seed role, an i32 scalar result role, source
vector config `SEW16 LMUL mf2`, compare/mask config, reduction/result config
`SEW32 LMUL m1`, signed masked widening dot-product reduction operation kind,
scalar accumulator/result boundary, strided-source memory forms, element stride
units, mask/tail policy, runtime `n` / AVL, route planning, generated artifact
emission, and real `ssh rvv` correctness evidence for representative counts.

This is one bounded low-level RVV masked contraction-supporting scalar
reduction slice. It is not masked memory movement, matmul lowering, a memory or
reduction matrix, a broad contraction framework, source-front-door work,
descriptor/direct-C/source-export work, or a performance task.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV masked strided-input widening
  dot-reduction executable slice`.
- Module owner: RVV plugin-owned computed-mask plus runtime-strided source-load
  widening dot-product reduction path for one bounded signed i16 `SEW16 LMUL
  mf2` lhs/rhs pair, compare-produced mask, runtime lhs/rhs element strides,
  and signed i32 scalar seed/result.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `40a9cad8 rvv: add strided input widening dot slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- Current HEAD supports computed-mask signed i16 widening dot-product reduction
  and runtime-strided-input signed i16 widening dot-product reduction as
  separate bounded executable slices through typed body/config/runtime facts
  and RVV plugin-owned route planning/provider output.
- The production gap is the composition point: compare-produced mask SSA must
  gate horizontal signed widening dot products whose lhs/rhs source values are
  loaded from runtime-strided source windows with explicit element-stride ABI
  roles.
- `tcrv.exec` declares ABI/runtime roles only. It must not infer compare,
  mask, stride, dtype/config, dot-product semantics, reduction identity, scalar
  result role, intrinsic spelling, or route support from ABI names or metadata.
- RVV selected-body realization may materialize legal generic typed compare,
  mask, strided-load, masked widening product/reduction, scalar seed, and
  scalar store structure. It must not alter computation semantics, dtype
  semantics, parameter roles, selected variant origin, dispatch/fallback
  behavior, or runtime `n` / AVL values.
- RVV route planning/provider own compare and mask types/provenance, source and
  result C types, byte-stride derivation from element stride and source element
  width, strided load leaves, masked widening product/reduction leaves, scalar
  seed/result leaves, ABI order, header/artifact mirror metadata, and
  fail-closed diagnostics. Common EmitC/export remains neutral and consumes
  provider-built payloads.
- Stage1/Stage2 guardrails remain active: no positive `RVVI32M1*`,
  `rvv-i32m1-*`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-seed, descriptor/direct-C/source-export,
  route-id authority, artifact-name authority, exact intrinsic spelling as
  route authority, or common/export RVV semantic authority.

## Requirements

1. Keep scope to one signed `i16mf2 x i16mf2 -> i32 scalar`
   computed-mask runtime-strided-input widening dot-product reduction route
   with `cmp_lhs`, `cmp_rhs`, `lhs`, `rhs`, `acc_seed`, `out`, `n/AVL`,
   `lhs_stride`, and `rhs_stride` ABI/runtime roles.
2. The selected/pre-realized body or explicit typed body must structurally
   carry compare lhs/rhs source facts, compare predicate, produced mask SSA
   value, mask provenance/config, lhs/rhs i16 source windows, lhs/rhs runtime
   element-stride roles, scalar seed/result boundary roles, source element
   type/config, reduction/result element type/config, signed masked widening
   dot-product reduction kind, accumulator seed/layout, scalar result layout,
   strided source memory forms, explicit element stride units, mask/tail
   policy, runtime `n/AVL`, and ABI role order.
3. Runtime stride roles and mask provenance must be consumed as body facts.
   They must not be inferred from parameter names, route ids, ABI strings,
   artifact names, test names, descriptors, exact intrinsic spelling,
   generated C strings, harness predicates, or common EmitC/export code.
4. `RVVSelectedBodyRealization`, if needed for a pre-realized fixture path,
   must realize only legal generic typed structure: `setvl`, compare lhs/rhs
   loads, generic compare producing a mask, i16mf2 strided-load lhs, i16mf2
   strided-load rhs, explicit scalar accumulator seed, mask-consuming signed
   widening product plus horizontal i32 reduction or equivalent typed RVV
   dot-reduction op, and scalar i32 output store boundary.
5. RVVEmitCRoutePlanning must derive ABI order, compare/source/seed/result C
   types, mask C type, vector C types, byte-stride conversion from element
   stride and source element width, compare leaves, strided load intrinsic
   leaves, masked widening product/reduction leaves, scalar boundary metadata,
   header/artifact mirrors, and targeted diagnostics from typed
   body/config/runtime facts.
6. Provider/materializer must consume the provider-built plan. Common
   materialization must not infer predicate, mask semantics, dtype, stride
   semantics, widening, dot-product kind, reduction layout, scalar result role,
   memory form, policy, intrinsic choice, or ABI meaning.
7. Unsupported masked strided dot-reduction kind, missing compare producer,
   missing mask provenance, mask/config mismatch, missing lhs/rhs stride role,
   invalid stride unit, invalid source/result SEW/LMUL relation, missing or
   mismatched source/result dtype, missing accumulator seed role, missing scalar
   result role, missing `n/AVL`, stale route-id authority, and incomplete typed
   body structure must fail closed with targeted diagnostics.
8. Generated-bundle evidence must use positive and negative i16 operands,
   mixed active/inactive mask lanes, non-unit lhs/rhs strides, nonzero i32 seed,
   skipped-source sentinels, runtime `n`, and scalar output sentinel checking
   so expected output proves skipped source elements are ignored, inactive mask
   lanes do not contribute, active signed products and seed addition are
   correct, runtime `n` is honored, and only scalar `out[0]` is written.
9. Real `ssh rvv` evidence is required for any executable correctness claim.

## Acceptance Criteria

- [ ] PRD, implement/check context, and task metadata describe this bounded
      computed-mask runtime-strided-input signed i16 widening dot-product
      reduction slice.
- [ ] A selected/pre-realized or explicit typed RVV body structurally carries
      compare-produced mask facts, lhs/rhs i16mf2 strided source facts,
      lhs/rhs runtime element-stride roles, i32 scalar accumulator seed/result
      facts, signed masked widening dot-product reduction kind, scalar
      boundary, policy, runtime `n/AVL`, and ABI roles.
- [ ] `RVVSelectedBodyRealization` materializes the bounded slice into legal
      generic typed compare-mask plus strided-load plus masked widening
      dot-reduction scalar structure, or the explicit typed body path already
      carries equivalent structure.
- [ ] RVVEmitCRoutePlanning derives compare/mask/source/seed/result ABI types,
      vector types, element-to-byte stride conversion, compare leaves, strided
      source load leaves, masked widening dot-reduction leaves, scalar boundary
      metadata, header/artifact mirrors, and diagnostics from typed facts.
- [ ] Positive route/materialization tests prove compare-produced mask facts,
      typed i16mf2 strided lhs/rhs inputs, explicit lhs/rhs stride roles, and
      i32 scalar seed/result facts reach `TCRVEmitCLowerableRoute` and
      provider-owned route metadata.
- [ ] Negative fail-closed tests cover missing compare producer, missing mask
      provenance, mask/config mismatch, missing lhs/rhs stride roles, invalid
      stride unit, missing `n/AVL`, invalid SEW/LMUL pair, missing seed,
      missing scalar result role, mismatched source/result dtype, stale
      route-id authority, and incomplete typed body structure.
- [ ] Generated-bundle dry-run passes for the masked strided-input widening
      dot-reduction slice at counts `7,16,23` with non-unit lhs/rhs strides and
      mixed active/inactive mask lanes.
- [ ] Real `ssh rvv` correctness passes for counts `7,16,23` if executable
      correctness is claimed, with non-vacuous expected outputs proving
      runtime-strided masked signed horizontal dot product, seed addition,
      runtime `n`, scalar output correctness, skipped-source sentinel
      preservation, inactive-lane non-contribution, and scalar output sentinel
      preservation.
- [ ] Focused build, lit, C++ unit, script, and artifact checks for touched RVV
      dialect/config, runtime ABI if touched, construction protocol,
      selected-body realization, route planning/provider, materializer/export,
      target metadata, and generated-bundle paths pass.
- [ ] Active-authority scan confirms no positive legacy/source/descriptor or
      common/export RVV semantic authority is reintroduced.
- [ ] `git diff --check`, Trellis task validation, task status update/archive,
      and one coherent production-code commit are completed if the task is
      finished.

## Non-Goals

- No matmul lowering, high-level Linalg/Vector/StableHLO lowering, broad
  contraction framework, unmasked clone expansion, gather/indexed loads,
  segmented reductions, unsigned variants, dtype/LMUL clone batches,
  source-front-door positive route, one-intrinsic wrapper dialect, dashboard,
  report-only inventory, helper-only refactor, or performance claim.
- No Scalar, IME, Offload, TensorExt, future plugin work, global autotuning,
  readiness state machine, source-shape clone batch, Template/Toy examples, or
  Stage2 coverage side quest beyond this bounded masked strided-input
  dot-reduction slice.
- No descriptor-driven computation, direct-C/source-export route restoration,
  route-id authority, artifact-name authority, exact-intrinsic spelling
  authority, or compatibility wrapper preserving old i32 authority.
- No new dtype-prefixed helper op families such as `tcrv_rvv.i32_reduction_*`,
  `tcrv_rvv.i32_accumulator_*`, or `tcrv_rvv.i32_macc`.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, or emission. Python changes, if any,
  are limited to generated-bundle tooling/evidence.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused RVV dialect/verifier lit tests for positive and negative masked
   strided-input widening dot-reduction structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact lit/FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-run for the new masked strided-input widening
   dot-reduction op at counts `7,16,23`.
8. Run real `ssh rvv` correctness for the new masked strided-input widening
   dot-reduction op at counts `7,16,23` if executable correctness is claimed.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run broader `check-tianchenrv` if shared compiler behavior changes.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-masked-widening-dot-reduction/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-strided-input-widening-dot-reduction/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-macc-executable-slice/prd.md`
- `.trellis/workspace/codex/journal-12.md`

Initial implementation surface to inspect:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `include/TianChenRV/Support/RuntimeABI.h`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Support/RuntimeABIContract.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing generated-bundle tests for computed-mask dot reduction and
  strided-input dot reduction.

## Definition Of Done

- One bounded masked runtime-strided-input signed i16 widening dot-product
  reduction route structurally carries compare mask, source, stride,
  accumulator seed, scalar result, operation, policy, runtime, and ABI facts
  through typed RVV body structure into RVV route planning/provider metadata
  and generated artifacts.
- Fresh focused positive, negative, generated-bundle, and runtime evidence is
  current to this task when claimed.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The task is finished/archived only if production code/tests/evidence are
  present in this round; otherwise it remains open with an exact continuation
  point.
