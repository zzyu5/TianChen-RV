# Stage2 RVV segment2 interleave memory executable slice

## Goal

Complete one bounded Stage2 RVV segmented-memory executable slice on the
corrected typed `tcrv_rvv` surface. The concrete route is signed i32 / SEW32 /
LMUL m1:

```text
dst[2 * i] = src0[i]
dst[2 * i + 1] = src1[i]
```

The two sources are unit-stride field buffers. The destination is an
interleaved unit-stride segment2 memory destination. Source roles, destination
role, segment count, field order, vector dtype/config, memory forms,
tail/mask policy, runtime `n` / AVL, route planning, generated artifact
emission, and `ssh rvv` correctness evidence must all stay on the same
production path.

This is one bounded segment2 interleave memory movement slice. It is not a
general segmented-memory framework, segment3/segment4 matrix, masked segmented
task, indexed segmented task, high-level frontend task, or one-intrinsic
wrapper route.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV segment2 interleave memory executable slice`.
- Module owner: RVV plugin-owned segmented-store memory-form and multi-source
  ABI path for one bounded i32 SEW32 LMUL m1 two-unit-stride-loads to segment2
  store executable slice.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `cdb2d38d rvv: add segment2 deinterleave executable slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- The immediately prior task completed `segment2_deinterleave_unit_store` with
  typed segment2 load, field0/field1 extraction order, dual destination ABI,
  selected-body realization, route planning/provider output, generated-bundle
  dry-run, real `ssh rvv` correctness for counts `7,16,23`, full
  `check-tianchenrv` 217/217, and a clean worktree.
- Existing Stage2 memory movement covers strided source load, indexed gather,
  indexed scatter, masked unit-stride memory, computed mask memory, and
  segment2 deinterleave. The current memory gap is segmented destination
  store with multi-source field packing and one interleaved destination role.
- `tcrv.exec` declares ABI/runtime roles only. It must not own segmented
  memory semantics, segment count, field order, dtype/config, route support,
  tuple packing, or intrinsic spelling.
- RVV selected-body realization may materialize legal generic typed dual
  unit-stride loads plus segment2 store, but it must not alter computation,
  dtype, parameter roles, selected variant origin, dispatch/fallback behavior,
  field order, or runtime `n` / AVL values.
- RVV route planning/provider owns segment count validation, field/source role
  mapping, vector and tuple C types, load/store leaves, ABI order, headers,
  artifact metadata mirrors, and fail-closed diagnostics. Common EmitC/export
  remains neutral and consumes provider-built payloads.
- Stage1/Stage2 guardrails remain active: no positive `RVVI32M1*`,
  `rvv-i32m1-*`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-seed, descriptor/direct-C/source-export, route-id
  authority, artifact-name authority, or common/export RVV semantic authority.

## Requirements

1. Keep scope to one signed i32 / SEW32 / LMUL m1 segment2 interleave route
   with `src0`, `src1`, `dst`, and runtime `n/AVL` ABI/runtime roles.
2. The selected/pre-realized body or explicit typed RVV body must structurally
   carry source0 and source1 `mem_window` roles, interleaved destination
   `mem_window` role, `segment_count = 2`, field0/field1 source order,
   source unit-stride memory forms, destination segmented-store memory form,
   vector dtype/config, tail/mask policy, runtime `n/AVL`, and ABI role order.
3. If a pre-realized body is used, `RVVSelectedBodyRealization` must realize
   only legal generic typed structure: `setvl`, two unit-stride loads
   producing field vector values, and one segment2 store in the declared field
   order.
4. RVVEmitCRoutePlanning must derive ABI order, vector C type, segment tuple C
   type, unit-load leaves, segment-store leaf, header/artifact metadata
   mirrors, and targeted diagnostics from typed body/config/runtime facts.
5. Provider/materializer must consume the provider-built plan. Common
   materialization must not infer segment count, field order, memory form,
   dtype, SEW, LMUL, tuple spelling, or intrinsic choices.
6. Unsupported segment count, missing/duplicated field source role, swapped or
   mismatched field order/config, invalid destination memory form, missing
   `n/AVL`, stale route-id authority, and incomplete typed segmented-store
   body structure must fail closed with targeted diagnostics.
7. Generated bundle evidence must use distinct source0/source1 values and a
   sentinel-filled interleaved destination/tail storage so correctness proves
   field0/field1 source lanes interleave into even/odd destination positions
   and tail preservation.
8. Real `ssh rvv` evidence is required for any executable correctness claim.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata match this bounded
      segment2 interleave memory task.
- [x] A selected/pre-realized or explicit typed RVV body structurally carries
      `src0`, `src1`, `dst`, `segment_count = 2`, field order, memory forms,
      vector dtype/config, tail/mask policy, runtime `n/AVL`, and ABI roles.
- [x] `RVVSelectedBodyRealization` materializes the bounded pre-realized slice
      into legal generic typed dual unit-stride loads plus segment2 store
      structure, or the explicit typed body path already carries equivalent
      structure.
- [x] RVVEmitCRoutePlanning derives ABI order, vector and segment tuple C
      types, unit-load leaves, segment-store leaves, header/artifact metadata,
      and diagnostics from typed facts.
- [x] Positive route/materialization tests prove typed source-field and
      interleaved destination facts reach `TCRVEmitCLowerableRoute` and
      provider-owned route metadata.
- [x] Negative fail-closed tests cover unsupported segment count, missing
      field source role, duplicated field source role, swapped or mismatched
      field configs, missing destination role, invalid memory form, missing
      AVL/runtime roles, stale route-id authority, and incomplete typed
      segmented-store body structure.
- [x] Generated-bundle dry-run passes for the segment2 interleave slice at
      counts `7,16,23`.
- [x] Real `ssh rvv` correctness passes for counts `7,16,23` if executable
      correctness is claimed, with non-vacuous expected outputs proving
      source0/source1 fields interleave into even/odd destination positions
      and tail sentinels are preserved.
- [x] Focused build, lit, C++ unit, script, and artifact checks for touched RVV
      dialect/config, selected-body realization, route planning/provider,
      materializer/export, and generated-bundle paths pass.
- [x] Active-authority scan confirms no positive legacy/source/descriptor or
      common/export RVV semantic authority is reintroduced.
- [x] `git diff --check`, Trellis task validation, task status update/archive,
      and one coherent production-code commit are completed if the task is
      finished.

## Non-Goals

- No broad segmented-memory framework, segment3/segment4 matrix, masked
  segmented memory, indexed segmented memory, atomics, high-level
  Vector/Linalg/StableHLO frontend lowering, generic fusion/scheduling,
  source-front-door positive route, one-intrinsic wrapper dialect,
  dtype/LMUL clone batch, dashboard, global tuning database, report-only
  inventory, or performance claim.
- No new reduction, macc, conversion, broadcast, compare/select, masked
  memory, gather/scatter, or strided-memory side quest beyond the bounded
  segment2 interleave slice.
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
   segment2 interleave memory structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact lit/FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-run for the segment2 interleave op at counts
   `7,16,23`.
8. Run real `ssh rvv` correctness for the segment2 interleave op at counts
   `7,16,23` if executable correctness is claimed.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run broader `check-tianchenrv` if shared compiler behavior changes.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-segment2-deinterleave-memory-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-strided-memory-slice/prd.md`

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
- Existing segment2 deinterleave generated-bundle evidence tests.

## Definition Of Done

- One coherent segment2 interleave memory route is represented, verified,
  route-supported, materialized through the production RVV provider/common
  EmitC/export path, and executable on `ssh rvv` for counts `7,16,23`.
- Existing strided, indexed gather/scatter, masked memory, computed-mask
  memory, reduction, macc, broadcast, conversion, tail/mask, and segment2
  deinterleave Stage2 routes remain intact.
- The final report distinguishes route-supported evidence, dry-run generated
  artifact evidence, and executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
