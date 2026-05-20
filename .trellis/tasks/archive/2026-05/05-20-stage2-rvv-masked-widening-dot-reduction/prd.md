# Stage2 RVV computed-mask widening dot-reduction executable slice

## Goal

Implement one bounded Stage2 RVV computed-mask signed widening dot-product
reduction executable slice:

```text
out_i32[0] = acc_seed_i32
           + sum_i (cmp_lhs_i32[i] < cmp_rhs_i32[i]
                    ? sign_extend(lhs_i16[i]) * sign_extend(rhs_i16[i])
                    : 0)
```

The selected RVV path must carry typed compare-mask provenance from bounded
signed i32 `SEW32 LMUL m1` compare lhs/rhs inputs, signed i16 dot-product
lhs/rhs source roles, signed i32 scalar accumulator seed/result roles, source
vector config `SEW16 LMUL mf2`, reduction/result config `SEW32 LMUL m1`,
signed masked widening dot-product reduction operation kind, seed/identity,
accumulator/result scalar boundary, mask/tail policy, runtime `n` / AVL, route
planning, generated artifact emission, and real `ssh rvv` correctness evidence
for representative counts.

This is one bounded low-level RVV masked contraction-supporting scalar
reduction slice. It is not matmul lowering, a reduction matrix, source-front
door work, descriptor/direct-C/source-export work, or a performance task.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV computed-mask widening dot-reduction
  executable slice`.
- Module owner: RVV plugin-owned masked horizontal widening dot-product
  reduction path for one bounded signed i32 `SEW32 LMUL m1` compare-produced
  mask plus signed i16 `SEW16 LMUL mf2` dot-product inputs and signed i32
  scalar seed/result.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `9463f112 rvv: add widening dot reduction executable slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- Current HEAD supports the unmasked signed i16mf2 x i16mf2 to i32 scalar
  widening dot-product reduction route with generated-bundle and real `ssh rvv`
  evidence.
- Current HEAD also has compare-produced mask dataflow, masked memory
  movement, computed-mask strided-store, tail/mask policy, widening
  conversion, and widening macc precedents through typed body/config/runtime
  facts and RVV plugin-owned route planning/provider output.
- The production gap is the composition point: compare-produced mask SSA must
  gate the horizontal widening dot-product reduction so inactive lanes
  contribute zero, while scalar seed addition and scalar result storage remain
  explicit typed boundary facts.
- `tcrv.exec` declares ABI/runtime roles only. It must not own compare
  semantics, mask semantics, dot-product semantics, dtype/config, reduction
  identity, route support, intrinsic spelling, or acceptance state.
- RVV selected-body realization may materialize legal generic typed compare,
  mask, load, masked widening product/reduction, and scalar store structure,
  but it must not alter computation semantics, dtype semantics, parameter
  roles, selected variant origin, dispatch/fallback behavior, or runtime
  `n` / AVL values.
- RVV route planning/provider owns predicate, mask type/provenance, input and
  result C types, widening product/reduction leaves, scalar seed/result
  boundary, ABI order, header/artifact mirrors, and fail-closed diagnostics.
  Common EmitC/export remains neutral and consumes provider-built payloads.
- Stage1/Stage2 guardrails remain active: no positive `RVVI32M1*`,
  `rvv-i32m1-*`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-seed, descriptor/direct-C/source-export,
  route-id authority, artifact-name authority, exact intrinsic spelling as
  route authority, or common/export RVV semantic authority.

## Requirements

1. Keep scope to one signed i32 / `SEW32 LMUL m1` compare-produced mask plus
   signed i16 / `SEW16 LMUL mf2` computed-mask
   `masked_widening_dot_reduce_add` route with compare lhs/rhs, dot lhs/rhs,
   scalar `acc_seed`, scalar `out`, and runtime `n/AVL` ABI/runtime roles.
2. The selected/pre-realized body or typed body must structurally carry:
   compare lhs/rhs memory roles, dot lhs/rhs memory roles, scalar seed/result
   roles, predicate `slt`, produced mask SSA value, mask type/config/provenance,
   source element type/config, reduction/result element type/config, signed
   widening dot-product reduction kind, seed/identity, accumulator/result scalar
   boundary, unit-stride memory form, tail/mask policy, runtime `n/AVL`, and
   ABI role order.
3. A valid route must prove the mask is produced by typed in-body compare
   dataflow and consumed by the masked dot-reduction in the same selected RVV
   body. An externally supplied mask, metadata mirror, route id, artifact name,
   or harness-side predicate is insufficient.
4. `RVVSelectedBodyRealization`, if needed for a pre-realized fixture path,
   must realize only legal generic typed structure: `setvl`, compare lhs/rhs
   loads, generic compare producing a mask, dot lhs/rhs loads, masked signed
   widening product plus horizontal i32 reduction or equivalent typed RVV
   dot-reduction op, scalar seed addition, and scalar i32 output store boundary.
5. RVVEmitCRoutePlanning must derive ABI order, compare/dot/source/result C
   types, mask C type, vector types, compare leaves, masked widening
   product/reduction leaves, scalar seed/result leaves, header/artifact mirror
   metadata, and targeted diagnostics from typed body/config/runtime facts.
6. Provider/materializer must consume the provider-built plan. Common
   materialization must not infer predicate, mask semantics, widening,
   reduction layout, scalar result role, dtype, SEW, LMUL, policy, intrinsic
   choice, or ABI meaning.
7. Unsupported masked reduction kind, missing compare producer, missing mask
   provenance, mismatched mask VL/config, invalid source/result SEW/LMUL pair,
   missing or mismatched source/result dtype, missing seed, missing scalar
   result role, missing `n/AVL`, stale route-id authority, and incomplete typed
   body structure must fail closed with targeted diagnostics.
8. Generated-bundle evidence must use positive and negative i16 operands, mixed
   active/inactive mask lanes, nonzero i32 seed, and scalar output sentinel
   checking so expected output proves active lanes contribute, inactive lanes
   do not contribute, signed products and seed addition are correct, runtime
   `n` is honored, and only the scalar output is written.
9. Real `ssh rvv` evidence is required for any runtime/correctness claim.

## Acceptance Criteria

- [ ] PRD, implement/check context, and task metadata describe this bounded
      computed-mask signed i16 widening dot-product reduction slice.
- [ ] A selected/pre-realized or explicit typed RVV body structurally carries
      compare lhs/rhs, dot lhs/rhs, scalar seed/result, predicate, produced mask
      value, mask provenance/config, source/result dtype/config, signed masked
      widening dot-reduction kind, seed/identity, scalar boundary, policy,
      runtime `n/AVL`, and ABI roles.
- [ ] `RVVSelectedBodyRealization` materializes the bounded pre-realized slice
      into legal generic typed compare-mask plus masked dot-reduction scalar
      structure, or the explicit typed body path already carries equivalent
      structure.
- [ ] RVVEmitCRoutePlanning derives ABI order, source/seed/result C types,
      mask type, vector types, compare leaves, masked widening dot-reduction
      leaves, scalar boundary metadata, header/artifact mirrors, and diagnostics
      from typed facts.
- [ ] Positive route/materialization tests prove compare-produced mask facts,
      typed i16mf2 dot inputs, and i32 scalar seed/result facts reach
      `TCRVEmitCLowerableRoute` and provider-owned route metadata.
- [ ] Negative fail-closed tests cover unsupported masked reduction kind,
      missing compare producer, missing mask provenance, mask VL/config
      mismatch, invalid SEW/LMUL pair, missing seed, missing scalar result role,
      mismatched source/result dtype, missing AVL/runtime roles, stale route-id
      authority, and incomplete typed body structure.
- [ ] Generated-bundle dry-run passes for the masked widening dot-reduction
      slice at counts `7,16,23`.
- [ ] Real `ssh rvv` correctness passes for counts `7,16,23` if executable
      correctness is claimed, with non-vacuous expected outputs proving masked
      signed horizontal dot product, seed addition, runtime `n`, scalar output
      correctness, inactive-lane non-contribution, and sentinel preservation.
- [ ] Focused build, lit, C++ unit, script, and artifact checks for touched RVV
      dialect/config, runtime ABI if touched, construction protocol,
      selected-body realization, route planning/provider, materializer/export,
      and generated-bundle paths pass.
- [ ] Active-authority scan confirms no positive legacy/source/descriptor or
      common/export RVV semantic authority is reintroduced.
- [ ] `git diff --check`, Trellis task validation, task status update/archive,
      and one coherent production-code commit are completed if the task is
      finished.

## Non-Goals

- No matmul lowering, high-level Linalg/Vector/StableHLO lowering, broad
  contraction framework, segmented reductions, unmasked reduction expansion,
  unsigned variants, dtype/LMUL clone batches, source-front-door positive
  route, one-intrinsic wrapper dialect, dashboard, report-only inventory,
  helper-only refactor, or performance claim.
- No Scalar, IME, Offload, TensorExt, future plugin work, global autotuning,
  readiness state machine, source-shape clone batch, Template/Toy examples, or
  Stage2 coverage side quest beyond this bounded masked dot-reduction slice.
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
   widening dot-reduction structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact lit/FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-run for the masked widening dot-reduction op at
   counts `7,16,23`.
8. Run real `ssh rvv` correctness for the masked widening dot-reduction op at
   counts `7,16,23` if executable correctness is claimed.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run broader `check-tianchenrv` if shared compiler behavior changes.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-dot-reduction/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-computed-mask-strided-store/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-masked-memory-movement-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-computed-mask-memory-dataflow/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-cmp-select-executable-abi/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-tail-mask-policy-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-conversion/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-macc-executable-slice/prd.md`

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
- Existing generated-bundle tests for computed-mask memory/strided-store and
  widening dot reduction.

## Definition Of Done

- One coherent computed-mask signed i16 widening dot-product reduction route is
  represented, verified, route-supported, materialized through the production
  RVV provider/common EmitC/export path, and executable on `ssh rvv` for counts
  `7,16,23` if correctness is claimed.
- Existing compare/select, masked memory, computed-mask strided-store,
  widening conversion, widening macc, unmasked widening dot reduction, and
  Stage1 fail-closed behavior remain intact.
- The final report distinguishes route-supported evidence, dry-run generated
  artifact evidence, and executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The task is finished/archived only if production code/tests/evidence are
  present in this round; otherwise it remains open with an exact continuation
  point.
