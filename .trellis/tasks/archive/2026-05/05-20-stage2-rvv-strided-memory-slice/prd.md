# Stage2 RVV strided memory movement executable slice

## Goal

Complete one bounded Stage2 RVV memory-movement executable slice on the
corrected typed `tcrv_rvv` surface. The concrete route is signed i32 / SEW32 /
LMUL m1:

```text
out[i] = src[i * runtime_stride]
```

The source side is an explicit strided memory form with an ABI-visible runtime
stride value. The destination side is a unit-stride store. The route must start
from a selected `tcrv.exec` RVV boundary and typed or pre-realized
`tcrv_rvv` body, flow through RVV selected-body realization if needed, RVV
plugin-owned route planning/provider output, neutral common EmitC/export, and
real `ssh rvv` correctness evidence if executable status is claimed.

This is one bounded memory-movement slice, not a broad memory-layout framework
and not a new arithmetic, reduction, macc, gather/scatter, source-front-door, or
high-level frontend task.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV strided memory movement executable slice`.
- Module owner: RVV plugin-owned strided memory-form and runtime stride ABI
  path for one bounded i32 SEW32 LMUL m1 strided-load to unit-stride-store
  executable slice.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `dc851ba7 rvv: make macc layout typed body authority`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- Stage2 has recent executable slices for elementwise/broadcast,
  compare/select, non-i32 dtype/SEW, LMUL, tail/mask policy, widening
  conversion, reduce-add, and macc.
- The current branch already contains generic typed RVV dataflow and provider
  surfaces for `setvl`, loads/stores, binary, select, reduction, macc, and some
  memory forms. Existing strided arithmetic work may be reused only if it
  already derives route facts from typed body/config/runtime facts.
- The current task is not `strided_add`. It must prove memory movement itself:
  strided source load, unit-stride destination store, explicit source stride
  runtime ABI role, non-contiguous expected-output checks, and fail-closed
  diagnostics for stale or incomplete memory facts.
- `tcrv.exec` declares ABI/runtime roles only. It must not own memory form,
  stride semantics, dtype/config, route support, or intrinsic spelling.
- Common EmitC/export must stay neutral and only materialize provider-built
  route payloads.
- Stage1/Stage2 guardrails remain active: no positive `RVVI32M1*`,
  `rvv-i32m1-*`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-seed, descriptor/direct-C/source-export, route-id
  authority, artifact-name authority, or common/export RVV semantic authority.

## Requirements

1. Keep scope to one signed i32 / SEW32 / LMUL m1 `strided_load_to_unit_store`
   route with `src`, `out`, `n/AVL`, and source stride ABI/runtime roles.
2. The selected boundary or typed body must structurally carry:
   source and destination `mem_window` roles, source memory form, destination
   memory form, stride kind/unit/value source, vector dtype/config, tail/mask
   policy, runtime `n/AVL`, and ABI roles.
3. If a pre-realized body is used, RVV selected-body realization may materialize
   only legal generic typed memory structure. It must not change computation,
   dtype semantics, parameter roles, selected variant origin, required
   capabilities, dispatch/fallback behavior, or runtime `n` / AVL values.
4. RVV route planning must derive ABI order, vector C types, memory intrinsic
   leaves, header/artifact metadata mirrors, and diagnostics from typed facts.
5. Provider/materializer must consume the provider-built plan. Common
   materialization must not infer RVV memory form, stride unit, dtype, SEW,
   LMUL, or intrinsic choices.
6. Unsupported memory form, missing stride role, invalid stride unit, missing
   `n/AVL`, mismatched dtype/config, stale route-id authority, and incomplete
   typed memory body structure must fail closed with targeted diagnostics.
7. Generated bundle evidence must use non-contiguous input and expected output
   that proves runtime stride is actually consumed.
8. Real `ssh rvv` evidence is required for any runtime/correctness claim.

## Acceptance Criteria

- [x] `tcrv_rvv` accepts a valid typed strided-load to unit-stride-store body
      and rejects missing/wrong stride operands, wrong stride roles, invalid
      stride unit, wrong memory form, missing AVL/runtime roles, type/config
      mismatches, stale route-id authority, and incomplete memory body
      structure.
- [x] RVV selected-body realization materializes the pre-realized memory
      movement body into legal generic typed `setvl`, `strided_load` or
      equivalent strided load, and unit-stride `store` structure if this slice
      uses a pre-realized fixture path.
- [x] RVVEmitCRoutePlanning derives route metadata, runtime ABI order, vector C
      type, strided load intrinsic leaf, unit-stride store leaf, stride layout
      mirrors, and target header/artifact facts from typed body/config/runtime
      facts.
- [x] Positive route/materialization tests prove typed strided memory facts
      reach provider-derived metadata and generated output through
      `TCRVEmitCLowerableRoute`.
- [x] Negative fail-closed tests cover unsupported memory form, missing or
      invalid stride role/unit, mismatched dtype/config, missing AVL/runtime
      roles, stale route-id authority, and incomplete typed body structure.
- [x] Generated-bundle dry-run passes for the strided memory movement slice at
      counts `7,16,23`.
- [x] Real `ssh rvv` correctness passes for counts `7,16,23` if executable
      correctness is claimed, with non-vacuous expected outputs proving
      non-contiguous source input was used.
- [x] Focused build, lit, C++ unit, script, and artifact checks for touched RVV
      dialect/config, selected-body realization, route planning/provider,
      materializer/export, and generated-bundle paths pass.
- [x] Active-authority scan confirms no positive legacy/source/descriptor or
      common/export RVV semantic authority is reintroduced.
- [x] `git diff --check`, Trellis task validation, task status update/archive,
      and one coherent production-code commit are completed if the task is
      finished.

## Non-Goals

- No gather/scatter, segmented memory, mask memory, broad stride/layout matrix,
  dtype/LMUL clone batch, high-level Linalg/Vector/StableHLO frontend lowering,
  one-intrinsic wrapper dialect, dashboard, global tuning database, report-only
  inventory, source-front-door positive route, or performance claim.
- No macc/reduction/compare/select side quest.
- No descriptor-driven computation, direct-C/source-export route restoration,
  or compatibility wrapper preserving old i32 route authority.
- No Python implementation of compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission.

## Validation Plan

1. Validate Trellis task context and start the task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused RVV dialect/verifier lit tests for positive and negative
   strided memory movement structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact lit/FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-run for the strided memory movement op at counts
   `7,16,23`.
8. Run real `ssh rvv` correctness for the strided memory movement op at counts
   `7,16,23` if executable correctness is claimed.
9. Run active-authority scans over active RVV include/lib/test/script paths.
10. Run `git diff --check`.
11. Run broader `check-tianchenrv` if shared compiler behavior changes.

## Completion Evidence

- Added `tcrv_rvv.typed_strided_memory_pre_realized_body` and `tcrv_rvv.move`
  as the bounded structural memory-movement slice. The verified ABI/runtime
  roles are `src`, `out`, `n`, and `src_stride`; the source form is strided
  load, the destination form is unit-stride store, and the stride unit is
  explicit element stride.
- Extended RVV selected-body realization to lower the pre-realized memory body
  into `setvl`, `with_vl`, `strided_load`, `move`, and `store` while preserving
  selected-boundary ABI/runtime roles.
- Extended RVV route planning, construction protocol facts, provider metadata,
  and generated-bundle evidence so `strided_load_unit_store` is planned from
  typed body/config/runtime facts and materialized by the existing provider
  path.
- Added explicit and pre-realized target artifact tests plus dialect and EmitC
  negative tests for unsupported or incomplete typed memory facts.
- Generated-bundle dry-runs passed for explicit and pre-realized bodies at
  counts `7,16,23`.
- Real `ssh rvv` executable correctness passed for explicit and pre-realized
  bodies at counts `7,16,23`. The harness uses `src_stride=3`,
  non-contiguous sentinel-filled input, checks `out[i] == src[i * src_stride]`,
  and verifies the unit-store tail sentinel remains untouched.
- Focused build, focused C++ tests, focused lit, full `check-tianchenrv`,
  Python script self-test, `git diff --check`, and active-authority scans
  passed.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-macc-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-typed-strided-memory-route/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-rhs-broadcast-memory-form-executable-route/prd.md`

Initial implementation surface to inspect:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing generated-bundle `ssh` evidence tests under `test/Scripts` and
  target artifact fixtures under `test/Target/RVV`.

## Definition Of Done

- One bounded strided source load to unit-stride destination store route is
  represented, verified, route-supported, materialized through the production
  RVV provider path, dry-run validated, and runtime-validated on `ssh rvv` if
  executable correctness is claimed.
- Existing Stage2 slices remain intact.
- The task report distinguishes route-supported, generated-bundle dry-run, and
  executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
