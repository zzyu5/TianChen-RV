# Stage2 RVV computed-mask strided-store executable slice

## Goal

Complete one bounded Stage2 RVV computed-mask destination-strided-store
executable slice on the corrected typed `tcrv_rvv` surface. The concrete route
is signed i32 / SEW32 / LMUL m1:

```text
if cmp_lhs[i] < cmp_rhs[i]:
  dst[i * dst_stride] = src[i]
else:
  dst[i * dst_stride] remains old_dst
```

The compare lhs/rhs and payload source are unit-stride inputs. The destination
uses an explicit runtime destination stride with explicit stride unit. The
selected RVV body must produce the mask through typed compare dataflow and
consume that mask in a masked strided store, so active masked lanes update
strided destination offsets while inactive masked lanes, skipped destination
slots, and tail storage remain sentinel-preserved.

This is one bounded computed-mask strided-store slice. It is not a generic
masked/strided memory matrix, high-level frontend lowering task, source-front
door route, descriptor route, or performance task.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV computed-mask strided-store executable
  slice`.
- Module owner: RVV plugin-owned masked strided destination-store path for one
  bounded i32 SEW32 LMUL m1 compare-produced-mask plus unit-load to
  runtime-strided-store executable slice.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `0e5c7fdc rvv: add unit-load strided-store executable slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- The immediately prior task completed `unit_load_strided_store` with typed
  source/destination/runtime destination stride, element stride unit,
  destination strided-store memory form, route planning/provider output,
  generated-bundle evidence, explicit and pre-realized `ssh rvv` correctness
  for counts `7,16,23` with `dst_stride=3`, and `check-tianchenrv` 227/227.
- The earlier computed-mask memory task completed
  `computed_masked_unit_load_store`, proving compare-produced mask SSA dataflow
  into masked unit-stride memory movement with runtime correctness for counts
  `7,16,23`.
- The earlier masked memory movement task established inactive-lane
  destination preservation and mask/tail policy precedent for unit-stride
  masked stores.
- This task composes those production patterns only for the bounded case:
  compare-produced mask plus unit payload load plus runtime element-strided
  destination store.
- `tcrv.exec` declares ABI/runtime roles only. It must not own compare
  semantics, mask semantics, destination stride semantics, memory form,
  dtype/config, route support, intrinsic spelling, or acceptance state.
- RVV selected-body realization may materialize legal generic typed compare
  plus masked strided-store structure, but it must not alter computation,
  dtype semantics, parameter roles, selected variant origin, required
  capabilities, dispatch/fallback behavior, or runtime `n` / AVL values.
- RVV route planning/provider owns predicate, mask type, vector type,
  destination stride scaling, memory leaves, ABI order, header/artifact mirror
  metadata, and fail-closed diagnostics. Common EmitC/export remains neutral
  and consumes provider-built payloads.
- Stage1/Stage2 guardrails remain active: no positive `RVVI32M1*`,
  `rvv-i32m1-*`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-seed, descriptor/direct-C/source-export,
  route-id authority, artifact-name authority, or common/export RVV semantic
  authority.

## Requirements

1. Keep scope to one signed i32 / SEW32 / LMUL m1
   `computed_masked_strided_store` route with `cmp_lhs`, `cmp_rhs`, `src`,
   `dst`, `n/AVL`, and runtime `dst_stride` ABI/runtime roles.
2. The selected/pre-realized body or typed body must structurally carry:
   compare lhs/rhs memory roles, payload source memory role, destination memory
   role, runtime destination stride binding, predicate `slt`, produced mask SSA
   value, mask role/source/form, stride unit, source unit-load memory form,
   destination masked strided-store memory form, vector dtype/config,
   tail/mask policy, runtime `n/AVL`, and ABI role order.
3. A valid route must prove the mask is produced by typed in-body compare
   dataflow and consumed by masked strided memory in the same selected RVV
   body. An externally supplied mask or metadata mirror alone is insufficient.
4. If a pre-realized body is used, `RVVSelectedBodyRealization` must realize
   only legal generic typed structure: `setvl`, compare lhs/rhs loads, generic
   compare producing a mask, payload source load, and masked destination
   `strided_store` or equivalent active-lane-preserving structure.
5. RVV route planning must derive ABI order, vector C type, mask C type,
   compare intrinsic leaf, masked strided-store leaf, byte-stride scaling when
   the typed unit is element-based, header/artifact metadata mirrors, and
   targeted diagnostics from typed body/config/runtime facts.
6. Provider/materializer must consume the provider-built plan. Common
   materialization must not infer predicate, mask semantics, destination memory
   form, stride unit, inactive-lane semantics, dtype, SEW, LMUL, policy,
   intrinsic choices, or ABI meaning.
7. Missing destination stride binding, wrong stride role/order, unsupported
   stride unit, unsupported predicate, missing or non-produced mask, invalid
   destination role/form, unsupported inactive-lane policy, missing `n/AVL`,
   mismatched dtype/config, stale route-id authority, and incomplete typed body
   structure must fail closed with targeted diagnostics.
8. Generated bundle evidence must use `dst_stride > 1`, non-vacuous compare
   inputs, non-default payload source values, and sentinel-filled destination
   storage so expected output proves active masked writes at strided offsets,
   inactive masked-lane preservation, skipped destination-slot preservation,
   and tail preservation.
9. Real `ssh rvv` evidence is required for any executable correctness claim.

## Acceptance Criteria

- [ ] PRD, implement/check context, and task metadata describe this bounded
      computed-mask strided-store slice.
- [ ] A selected/pre-realized or explicit typed RVV body structurally carries
      compare lhs/rhs, payload source, destination, runtime `dst_stride`,
      stride unit, predicate, produced mask value, mask role/source/form,
      source unit-load form, destination masked strided-store form, vector
      dtype/config, tail/mask policy, runtime `n/AVL`, and ABI roles.
- [ ] `RVVSelectedBodyRealization` materializes the bounded pre-realized slice
      into legal generic typed compare plus masked strided-store structure if
      this slice uses a pre-realized fixture path.
- [ ] RVVEmitCRoutePlanning derives ABI order, vector C type, mask C type,
      byte-stride scaling, compare leaves, masked strided-store leaves,
      header/artifact metadata, and diagnostics from typed facts.
- [ ] Positive route/materialization tests prove typed compare-produced mask,
      payload source, destination, runtime destination stride, stride unit, and
      masked strided-store memory-form facts reach `TCRVEmitCLowerableRoute`
      and provider-owned route metadata.
- [ ] Negative fail-closed tests cover missing destination stride binding,
      wrong stride role/order, unsupported stride unit, unsupported predicate,
      missing or stale mask, invalid destination role/form, mismatched
      dtype/config, missing AVL/runtime roles, stale route-id authority, and
      incomplete typed body structure.
- [ ] Generated-bundle dry-run passes for the computed-mask strided-store
      slice at counts `7,16,23`.
- [ ] Real `ssh rvv` correctness passes for counts `7,16,23` if executable
      correctness is claimed, with non-vacuous expected outputs proving
      compare-driven masked writes at `dst[i * dst_stride]`, inactive masked
      lane preservation, skipped destination-slot preservation, and tail
      preservation.
- [ ] Focused build, lit, C++ unit, script, and artifact checks for touched RVV
      dialect/config, runtime ABI, construction protocol, selected-body
      realization, route planning/provider, materializer/export, and
      generated-bundle paths pass.
- [ ] Active-authority scan confirms no positive legacy/source/descriptor or
      common/export RVV semantic authority is reintroduced.
- [ ] `git diff --check`, Trellis task validation, task status update/archive,
      and one coherent production-code commit are completed if the task is
      finished.

## Non-Goals

- No broad masked/strided/indexed/segmented memory matrix, masked segment
  stores, gather-scatter fusion, dtype/LMUL clone batch, high-level
  Linalg/Vector/StableHLO lowering, source-front-door positive route,
  one-intrinsic wrapper dialect, dashboard, report-only inventory, helper-only
  refactor, or performance claim.
- No reduction, macc, conversion, broadcast, compare/select expansion,
  segment2, indexed memory, or scalar-front-door side quest.
- No descriptor-driven computation, direct-C/source-export route restoration,
  route-id authority, artifact-name authority, or compatibility wrapper
  preserving old i32 route authority.
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
   computed-mask strided-store structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact lit/FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-run for the computed-mask strided-store op at
   counts `7,16,23`.
8. Run real `ssh rvv` correctness for the computed-mask strided-store op at
   counts `7,16,23` if executable correctness is claimed.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run broader `check-tianchenrv` if shared compiler behavior changes.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`
- `.trellis/spec/testing/index.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-computed-mask-memory-dataflow/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-masked-memory-movement-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-strided-store-memory-executable-slice/prd.md`
- `.trellis/workspace/codex/journal-12.md`

Initial implementation surface to inspect:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `include/TianChenRV/Support/RuntimeABI.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Support/RuntimeABIContract.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing computed-mask memory and unit-load-strided-store generated-bundle
  tests.

## Definition Of Done

- One coherent computed-mask unit-load to runtime destination-strided-store
  route is represented, verified, route-supported, materialized through the
  production RVV provider/common EmitC/export path, and executable on `ssh rvv`
  for counts `7,16,23` if correctness is claimed.
- Existing masked memory, computed-mask memory, strided-store, indexed,
  segment2, reduction, macc, broadcast, conversion, tail/mask, and dtype/LMUL
  Stage2 routes remain intact.
- The final report distinguishes route-supported evidence, dry-run generated
  artifact evidence, and executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
