# Stage2 RVV indexed scatter memory executable slice

## Goal

Complete one bounded Stage2 RVV indexed scatter memory-movement executable
slice on the corrected typed `tcrv_rvv` surface. The concrete route is signed
i32 / SEW32 / LMUL m1:

```text
dst[index[i]] = src[i]
```

The source data buffer, index buffer, indexed destination buffer, runtime
`n` / AVL, index element / offset EEW, offset unit, data vector dtype/config,
source unit-stride memory form, destination indexed-store memory form, tail/mask
policy, and ABI roles must be explicit typed facts. The route must start from a
selected `tcrv.exec` RVV boundary and typed or pre-realized `tcrv_rvv` body,
flow through RVV selected-body realization if needed, RVV plugin-owned route
planning/provider output, neutral common EmitC/export, and real `ssh rvv`
correctness evidence if executable correctness is claimed.

This is one bounded unit-stride-load to indexed-store slice. It is not a broad
gather/scatter framework, masked scatter task, segmented memory task,
high-level frontend task, or one-intrinsic wrapper route.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV indexed scatter memory executable slice`.
- Module owner: RVV plugin-owned indexed scatter memory-form and index-vector
  ABI path for one bounded i32 SEW32 LMUL m1 unit-stride-load to indexed-store
  executable slice.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `743403ee rvv: add indexed gather memory executable slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- The immediately prior task completed the indexed gather slice with typed
  index vector ABI, `tcrv_rvv.index_load`, `tcrv_rvv.indexed_load`,
  selected-body realization, route planning/provider output,
  generated-bundle dry-run, real `ssh rvv` correctness for counts `7,16,23`,
  full `check-tianchenrv`, and a clean worktree.
- Existing Stage2 memory movement also covers strided source load to
  unit-stride destination store. Indexed scatter is the opposite indirect
  memory direction: unit-stride source load to indexed destination store.
- `tcrv.exec` declares ABI/runtime roles only. It must not own indexed-store
  semantics, offset unit, index EEW, dtype/config, route support, duplicate
  index policy, or intrinsic spelling.
- Common EmitC/export must stay neutral and only materialize provider-built
  route payloads.
- Stage1/Stage2 guardrails remain active: no positive `RVVI32M1*`,
  `rvv-i32m1-*`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-seed, descriptor/direct-C/source-export, route-id
  authority, artifact-name authority, or common/export RVV semantic authority.

## Requirements

1. Keep scope to one signed i32 / SEW32 / LMUL m1 indexed scatter route with
   `src`, `index`, `dst`, and `n/AVL` ABI/runtime roles.
2. The selected boundary or typed body must structurally carry: source
   `mem_window`, index `mem_window`, destination `mem_window`, source
   unit-stride memory form, destination indexed-store memory form, index
   element type or offset EEW, byte-vs-element offset unit, vector dtype/config,
   tail/mask policy, runtime `n/AVL`, and ABI roles.
3. If a pre-realized body is used, RVV selected-body realization may materialize
   only legal generic typed indexed-store memory structure. It must not change
   computation, dtype semantics, parameter roles, selected variant origin,
   required capabilities, dispatch/fallback behavior, duplicate-index policy,
   or runtime `n` / AVL values.
4. RVV route planning must derive ABI order, vector C types, unit-load and
   indexed-store intrinsic leaves, header/artifact metadata mirrors, index
   layout mirrors, duplicate-index diagnostics, and fail-closed diagnostics
   from typed source/index/destination/config/runtime facts.
5. Provider/materializer must consume the provider-built plan. Common
   materialization must not infer RVV indexed-store memory form, offset unit,
   index EEW, dtype, SEW, LMUL, duplicate-index semantics, or intrinsic choices.
6. Unsupported index EEW, missing index role, invalid offset unit, missing
   `n/AVL`, mismatched data/index config, stale route-id authority, unsupported
   memory form, duplicate-index ambiguity when not explicitly supported, and
   incomplete typed indexed-store body structure must fail closed with targeted
   diagnostics.
7. Generated bundle evidence must use unique, non-contiguous, non-monotonic
   index values with sentinel-filled destination buffers so correctness proves
   index-driven stores and untouched sentinel lanes.
8. Real `ssh rvv` evidence is required for any runtime/correctness claim.

## Acceptance Criteria

- [x] `tcrv_rvv` accepts a valid typed unit-stride-load to indexed-store body
      and rejects missing/wrong index operands, missing index role, unsupported
      index EEW, invalid byte-vs-element offset unit, wrong memory form,
      missing AVL/runtime roles, type/config mismatches, stale route-id
      authority, duplicate-index ambiguity if represented, and incomplete
      indexed-store body structure.
- [x] RVV selected-body realization materializes the pre-realized indexed
      scatter body into legal generic typed `setvl`, unit-stride source load,
      `index_load`, indexed destination store or equivalent scatter store
      structure if this slice uses a pre-realized fixture path.
- [x] RVVEmitCRoutePlanning derives route metadata, runtime ABI order, vector C
      type, unit-load intrinsic leaf, indexed-store intrinsic leaf,
      offset-unit/index layout mirrors, destination memory-form mirrors, and
      target header/artifact facts from typed body/config/runtime facts.
- [x] Positive route/materialization tests prove typed indexed-store memory
      facts reach provider-derived metadata and generated output through
      `TCRVEmitCLowerableRoute`.
- [x] Negative fail-closed tests cover unsupported index EEW, missing index
      role, invalid offset unit, mismatched dtype/config, missing AVL/runtime
      roles, stale route-id authority, unsupported memory form, duplicate-index
      ambiguity when applicable, and incomplete typed body structure.
- [x] Generated-bundle dry-run passes for the indexed scatter slice at counts
      `7,16,23`.
- [x] Real `ssh rvv` correctness passes for counts `7,16,23` if executable
      correctness is claimed, with non-vacuous expected outputs proving
      non-contiguous and non-monotonic indexed destination stores and sentinel
      preservation.
- [x] Focused build, lit, C++ unit, script, and artifact checks for touched RVV
      dialect/config, selected-body realization, route planning/provider,
      materializer/export, and generated-bundle paths pass.
- [x] Active-authority scan confirms no positive legacy/source/descriptor or
      common/export RVV semantic authority is reintroduced.
- [x] `git diff --check`, Trellis task validation, task status update/archive,
      and one coherent production-code commit are completed if the task is
      finished.

## Non-Goals

- No broad gather/scatter framework, masked scatter, segmented memory, atomics,
  duplicate-index resolution beyond fail-closed or unique-test-data evidence,
  broad index/layout matrix, dtype/LMUL clone batch,
  high-level Linalg/Vector/StableHLO frontend lowering, one-intrinsic wrapper
  dialect, dashboard, global tuning database, report-only inventory,
  source-front-door positive route, or performance claim.
- No macc/reduction/compare/select/broadcast/conversion side quest.
- No descriptor-driven computation, direct-C/source-export route restoration,
  or compatibility wrapper preserving old i32 route authority.
- No Python implementation of compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused RVV dialect/verifier lit tests for positive and negative
   indexed scatter memory movement structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact lit/FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-run for the indexed scatter op at counts
   `7,16,23`.
8. Run real `ssh rvv` correctness for the indexed scatter op at counts
   `7,16,23` if executable correctness is claimed.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run broader `check-tianchenrv` if shared compiler behavior changes.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/variant-pipeline/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-indexed-gather-memory-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-strided-memory-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-tail-mask-policy-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-conversion/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-reduce-add-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-macc-executable-slice/prd.md`

Initial implementation surface to inspect:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `include/TianChenRV/Support/RuntimeABI.h`
- `lib/Support/RuntimeABIContract.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing generated-bundle `ssh` evidence tests and target artifact fixtures,
  especially the `indexed_gather_unit_store` slice.

## Definition Of Done

- One bounded unit-stride source load to indexed destination store route is
  represented, verified, route-supported, materialized through the production
  RVV provider path, dry-run validated, and runtime-validated on `ssh rvv` if
  executable correctness is claimed.
- Existing Stage2 slices remain intact.
- The task report distinguishes route-supported, generated-bundle dry-run, and
  executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.

## Completion Evidence

- Added the generic typed indexed scatter route surface:
  `tcrv_rvv.typed_indexed_scatter_memory_pre_realized_body` and
  `tcrv_rvv.indexed_store`, with explicit `src,index,dst,n` ABI roles,
  source unit-load form, destination indexed-store form, `index_eew=32`,
  `offset_unit="element"`, `index_uniqueness="unique"`, SEW32, LMUL m1, and
  agnostic policy validation.
- Extended RVV selected-body realization so pre-realized indexed scatter lowers
  to `setvl`, `with_vl`, unit-stride `load`, `index_load`, `move`, and
  `indexed_store`.
- Extended RVV route planning/provider to derive `src,index,dst,n` ABI order,
  metadata mirrors, vector/index C types, unit-load leaf, element-to-byte index
  scale leaf, indexed-store leaf, and generated artifact payload from typed
  route facts.
- Generated-bundle explicit and pre-realized dry-runs passed for counts
  `7,16,23`.
- Real `ssh rvv` explicit and pre-realized generated-bundle correctness passed
  for counts `7,16,23`, printing
  `unique_non_monotonic_indexed_scatter` for each case and preserving sentinel
  lanes outside runtime `n`.
- `cmake --build build --target check-tianchenrv -j2` passed `205/205`.
- `git diff --check` passed.
- Active-authority scan found no new positive `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m`, source-front-door/source-seed,
  descriptor/direct-C/source-export, or common/export RVV semantic authority.
  The scan reported one negative description string and the provider-derived
  `__riscv_vsoxei32_v_i32m1` target leaf; the latter is derived from typed
  indexed scatter facts and is not route authority.
