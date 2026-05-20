# Stage2 RVV computed-mask memory dataflow executable slice

## Goal

Complete one bounded Stage2 RVV computed-mask memory dataflow executable slice
on the corrected typed `tcrv_rvv` surface. The concrete route is signed i32 /
SEW32 / LMUL m1:

```text
dst[i] = cmp_lhs[i] < cmp_rhs[i] ? src[i] : old_dst[i]
```

The selected RVV body must produce the mask inside typed RVV dataflow through a
compare and consume that produced mask in masked unit-stride memory movement.
This task proves that compare operands, compare predicate, produced mask SSA
value, source data role, destination old/result role, inactive-lane policy,
tail/mask policy, runtime `n` / AVL, vector dtype/config, mask config, ABI
roles, route planning, generated artifact emission, and `ssh rvv` correctness
evidence all stay on the same production path.

This is one bounded computed-mask memory slice. It is not a generic fusion
optimizer, broad predicate framework, high-level frontend lowering task, masked
gather/scatter matrix, or descriptor/source-front-door route.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV computed-mask memory dataflow executable
  slice`.
- Module owner: RVV plugin-owned in-body mask SSA dataflow for one bounded i32
  SEW32 LMUL m1 compare-to-masked-unit-store executable slice.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `5856fda0 rvv: add masked memory movement executable slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- The immediately prior task completed `masked_unit_load_store` with typed
  source/mask/destination/n ABI roles, mask load/config, old-destination
  preservation, inactive-lane policy, route planning/provider output,
  generated-bundle dry-run, real `ssh rvv` correctness for counts `7,16,23`,
  and `check-tianchenrv` 211/211.
- Existing compare/select work already established generic typed
  `tcrv_rvv.compare` and compare-produced mask evidence for pre-realized
  `cmp_select`, including real `ssh rvv` executable evidence for counts
  `7,16,23`.
- Existing masked memory movement still treats the mask as an external
  mask-buffer role for its executable slice. This task must make the mask an
  in-body typed SSA value produced by compare, then consumed by masked memory.
- `tcrv.exec` declares ABI/runtime roles only. It must not own compare
  semantics, mask semantics, dtype/config, inactive-lane policy, route support,
  or intrinsic spelling.
- RVV selected-body realization may materialize legal generic typed compare
  plus masked memory structure, but it must not alter computation, dtype,
  parameter roles, selected variant origin, dispatch/fallback behavior, or
  runtime `n` / AVL values.
- RVV route planning/provider owns operation, dtype, SEW, LMUL, predicate,
  mask type, memory form, inactive-lane policy, ABI order, intrinsic leaves,
  and fail-closed diagnostics. Common EmitC/export remains neutral and consumes
  provider-built payloads.
- Stage1/Stage2 guardrails remain active: no positive `RVVI32M1*`,
  `rvv-i32m1-*`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-seed, descriptor/direct-C/source-export,
  route-id authority, artifact-name authority, or common/export RVV semantic
  authority.

## Requirements

1. Keep scope to one signed i32 / SEW32 / LMUL m1 computed-mask unit-stride
   memory route with compare lhs/rhs, source data, destination old/result, and
   runtime `n/AVL` ABI/runtime roles.
2. The selected/pre-realized body or typed body must structurally carry:
   compare lhs/rhs memory roles, source data memory role, destination memory
   role, compare predicate `slt`, produced mask SSA value, mask role/config,
   source and destination unit-stride memory form, vector dtype/config,
   tail/mask policy, inactive-lane preservation policy, runtime `n/AVL`, and
   ABI role order.
3. A valid route must prove the mask is produced by typed in-body compare
   dataflow and consumed by masked memory in the same selected RVV body. An
   externally supplied mask alone is not sufficient for this task.
4. If a pre-realized body is used, `RVVSelectedBodyRealization` must realize
   only legal generic typed structure: `setvl`, compare lhs/rhs loads, generic
   compare producing a mask, source load, old destination load where required,
   masked memory movement or equivalent active-lane merge, and store.
5. RVV route planning must derive ABI order, vector C type, compare mask type,
   compare intrinsic leaf, masked memory intrinsic/merge leaves, inactive-lane
   policy mirrors, header/artifact metadata mirrors, and targeted diagnostics
   from typed body/config/runtime facts.
6. Provider/materializer must consume the provider-built plan. Common
   materialization must not infer RVV predicate, mask semantics, memory form,
   inactive-lane policy, dtype, SEW, LMUL, or intrinsic choices.
7. Missing compare producer, mismatched mask consumer, unsupported predicate,
   invalid mask config, unsupported inactive-lane policy, missing
   old-destination preservation, missing `n/AVL`, mismatched dtype/config,
   stale route-id authority, unsupported memory form, and incomplete typed
   body structure must fail closed with targeted diagnostics.
8. Generated bundle evidence must use non-vacuous compare inputs and
   sentinel-filled destination buffers so correctness proves compare-driven
   active updates and inactive plus tail sentinel preservation.
9. Real `ssh rvv` evidence is required for any executable correctness claim.

## Acceptance Criteria

- [ ] PRD, implement/check context, and task metadata match this bounded
      computed-mask memory dataflow task.
- [ ] A selected/pre-realized or explicit typed RVV body structurally carries
      compare lhs/rhs, source, destination, predicate, produced mask value,
      memory forms, mask role/config, vector dtype/config, inactive-lane
      policy, runtime `n/AVL`, and ABI roles.
- [ ] `RVVSelectedBodyRealization` materializes the bounded pre-realized slice
      into legal generic typed compare plus masked memory structure, or the
      explicit typed body path already carries equivalent structure.
- [ ] RVVEmitCRoutePlanning derives ABI order, vector C types, compare mask
      type/intrinsic leaves, masked memory leaves, header/artifact metadata,
      and diagnostics from typed facts.
- [ ] Positive route/materialization tests prove compare-produced mask facts
      reach masked memory through `TCRVEmitCLowerableRoute` and provider-owned
      route metadata.
- [ ] Negative fail-closed tests cover missing compare producer, mismatched
      mask type/config/consumer, unsupported predicate, missing/invalid masked
      memory consumer, unsupported inactive-lane policy, missing
      old-destination preservation, mismatched dtype/config, missing
      AVL/runtime roles, stale route-id authority, and incomplete typed body
      structure.
- [ ] Generated-bundle dry-run passes for the computed-mask memory slice at
      counts `7,16,23`.
- [ ] Real `ssh rvv` correctness passes for counts `7,16,23` if executable
      correctness is claimed, with non-vacuous expected outputs proving
      compare-driven active updates plus inactive/tail sentinel preservation.
- [ ] Focused build, lit, C++ unit, script, and artifact checks for touched RVV
      dialect/config, selected-body realization, route planning/provider,
      materializer/export, and generated-bundle paths pass.
- [ ] Active-authority scan confirms no positive legacy/source/descriptor or
      common/export RVV semantic authority is reintroduced.
- [ ] `git diff --check`, Trellis task validation, task status update/archive,
      and one coherent production-code commit are completed if the task is
      finished.

## Non-Goals

- No generic fusion optimizer, arbitrary multi-op scheduler, high-level
  Linalg/Vector/StableHLO frontend lowering, broad compare predicate matrix,
  masked gather/scatter matrix, segmented memory, atomics, source-front-door
  positive route, one-intrinsic wrapper dialect, dtype/LMUL clone batch,
  dashboard, global tuning database, report-only inventory, or performance
  claim.
- No new reduction, macc, conversion, broadcast, compare-select expansion, or
  memory-form side quest beyond the bounded compare-produced mask plus masked
  unit-stride memory slice.
- No descriptor-driven computation, direct-C/source-export route restoration,
  or compatibility wrapper preserving old i32 route authority.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, or emission. Python changes, if any,
  are limited to generated-bundle tooling/evidence.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused RVV dialect/verifier lit tests for positive and negative
   computed-mask memory structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact lit/FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-run for the computed-mask memory op at counts
   `7,16,23`.
8. Run real `ssh rvv` correctness for the computed-mask memory op at counts
   `7,16,23` if executable correctness is claimed.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run broader `check-tianchenrv` if shared compiler behavior changes.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-compare-select-route/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-cmp-select-executable-abi/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-masked-add-route-semantics/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-masked-memory-movement-executable-slice/prd.md`
- `.trellis/workspace/codex/journal-12.md`

Initial implementation surface to inspect:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing compare/select, masked add, and masked unit-load/store generated
  bundle tests.

## Definition Of Done

- One coherent computed-mask memory dataflow route is represented, verified,
  route-supported, materialized through the production RVV provider/common
  EmitC/export path, and executable on `ssh rvv` for counts `7,16,23`.
- Existing compare/select, masked add, masked memory movement, strided/indexed
  memory, reduction, and macc Stage2 routes remain intact.
- The final report distinguishes route-supported evidence, dry-run generated
  artifact evidence, and executable `ssh rvv` evidence.
