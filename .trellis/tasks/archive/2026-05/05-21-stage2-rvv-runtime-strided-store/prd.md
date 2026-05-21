# Stage2 RVV runtime byte-strided store executable slice

## Goal

Implement one bounded Stage2 RVV executable memory-movement slice where a
contiguous signed-i32 source load stores each active lane to a runtime
byte-strided destination:

```text
store input_i32[i] to dst at byte offset i * dst_stride_bytes, for i < n
destination bytes/elements not selected by the strided store remain sentinel
tail beyond n remains sentinel
```

This task is the symmetric destination-store counterpart to the archived
runtime byte-strided source-load slice. It must reuse the existing bounded
`unit_load_strided_store` route surface by migrating its active authority from
destination element stride to destination byte stride.

## Direction Source

- Direction title: `Stage2 RVV runtime byte-strided store executable slice`.
- Module owner: RVV plugin-owned runtime byte-strided store memory movement
  route for one bounded typed e32m1 path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `0a78a9c4 rvv: add runtime byte-strided load slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.

## Current Repository Facts

- The archived runtime byte-strided load task at
  `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-runtime-strided-load/`
  completed `strided_load_unit_store` with `src,out,n,stride_bytes`,
  `source-byte-stride`, `stride_unit = "byte"`, generated bundle dry-run, and
  `ssh rvv` evidence over counts `7,16,23` and byte strides `4,8,12`.
- Current code already has a bounded `unit_load_strided_store` route, explicit
  and pre-realized target tests, route planning/provider support, construction
  protocol support, and generated-bundle script support.
- That existing store route still uses `dst_stride`, runtime ABI role
  `output-stride`, memory layout
  `unit-stride-source-element-strided-destination-runtime-abi`, and
  `stride_unit = "element"` as positive executable authority.
- The generated-bundle script currently tests `unit_load_strided_store` with a
  fixed logical element stride of `3`; it does not cover byte strides
  `4,8,12` or byte-addressed sentinel preservation for this route.

## Requirements

1. Keep scope to one bounded signed i32 / SEW32 / LMUL m1 contiguous source
   load plus runtime-byte-strided destination store slice.
2. The selected/pre-realized body must structurally carry:
   - source input mem_window / runtime ABI value;
   - destination mem_window / runtime ABI value;
   - runtime `n` / AVL value use;
   - runtime destination byte-stride ABI value, preferably named
     `dst_stride_bytes`;
   - a destination byte-stride role distinct from generic element
     `output-stride`;
   - e32m1 vector config;
   - source memory form `unit-stride-load`;
   - destination memory form `strided-store`;
   - `stride_unit = "byte"` for this store route;
   - tail/mask agnostic policy for this bounded unmasked slice;
   - runtime ABI order.
3. RVV selected-body realization must materialize legal generic typed
   `setvl` / `with_vl`, contiguous vector `load`, `move`, and runtime
   byte-strided `strided_store`.
4. RVV route planning/provider must derive the destination stride C type,
   source/result vector C types, target leaves, required headers, runtime ABI
   order, provider mirrors, artifact mirrors, and diagnostics from typed
   body/config/runtime facts.
5. Provider-generated C/C++ must pass the runtime byte stride directly to the
   RVV strided-store leaf. It must not reinterpret the store stride as an
   element stride or multiply it by `sizeof(int32_t)` inside common code.
6. Common EmitC/export must remain neutral. It may materialize provider-built
   pointer/address payloads, but it must not infer destination stride unit,
   dtype, policy, ABI order, intrinsic choice, memory form, or RVV semantics.
7. Missing destination byte-stride role, wrong stride type, wrong stride ABI
   order/name, old element `output-stride` authority for this store route,
   zero/unsupported byte stride if rejected by the bounded contract, missing
   runtime `n` / AVL, invalid tail/mask policy, stale fixed-VL or route-id
   authority, incomplete typed body, descriptor/direct-C/source-front-door
   authority, and common/export semantic inference must fail closed with
   targeted diagnostics.
8. Generated-bundle dry-run and `ssh rvv` evidence must cover runtime counts
   `7,16,23` and destination byte strides `4,8,12`, with sentinel-padded
   destinations proving runtime stride use, selected destination writes,
   untouched sentinel preservation, runtime `n`, and tail preservation.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe this bounded
      runtime byte-strided destination-store task.
- [x] The selected/pre-realized `unit_load_strided_store` body accepts and
      requires a runtime destination byte-stride fact and no longer treats
      `output-stride` + `stride_unit = "element"` as positive executable
      authority for this route.
- [x] RVV selected-body realization materializes `setvl`, `with_vl`, unit
      `load`, `move`, and `strided_store` from typed body facts.
- [x] RVV route planning/provider mirrors byte-strided destination layout,
      source/destination memory forms, destination byte-stride source, ABI
      order, C type mapping, required headers, target leaves, and
      provider-supported status only after typed body validation.
- [x] Generated header/artifact evidence exists for explicit and pre-realized
      `unit_load_strided_store` and shows a byte-stride ABI parameter.
- [x] Negative fail-closed coverage exists for missing/wrong destination
      byte-stride role, wrong stride C type or ABI order, unsupported
      element-stride positive authority for this route, missing n/AVL, invalid
      policy/config, stale fixed-VL or route-id authority, incomplete typed
      body, and common/export semantic inference where meaningful.
- [x] Generated-bundle dry-run passes for counts `7,16,23` and destination
      byte strides `4,8,12` with sentinel-padded destination checks.
- [x] Real `ssh rvv` generated-bundle runs pass for the same bounded coverage.
- [x] Active-authority scan confirms no positive `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export, source-front-door, public exact
      intrinsic route authority, or common/export RVV semantic authority is
      introduced.
- [x] Focused build/lit/C++/script checks, `check-tianchenrv`,
      `git diff --check`, task validation, finish/archive, clean git status,
      and one coherent commit are completed if the task finishes.

## Completion Evidence

- Production owners changed: RVV runtime ABI role contract, RVV dialect
  verifier/docs, RVV config contract, selected-body realization, route
  planning/provider, construction protocol, generated-bundle evidence script,
  and targeted MLIR/script tests.
- Explicit and pre-realized generated-bundle dry-runs passed for
  `unit_load_strided_store` with runtime counts `7,16,23` and
  `dst_stride_bytes` `4,8,12`.
- Real `ssh rvv` generated-bundle runs passed for explicit and pre-realized
  selected bodies over the same nine count/stride combinations. Each case
  reported `byte_strided_store selected_destination_writes
  sentinel_preserved tail_preserved`.
- Self-repair performed:
  - fixed harness sentinel logic so `dst_stride_bytes=4` is not treated as a
    skipped-slot case;
  - restored computed-mask strided-store element-stride metadata mirror after
    full lit exposed accidental byte-stride mirror reuse;
  - removed destination-byte-stride from strided-load binding candidates and
    synchronized the verifier expectation.
- Final checks passed:
  - `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`;
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`;
  - focused lit for 6 byte-strided store tests;
  - focused regression lit for computed-mask strided-store and generic dataflow;
  - RVV dialect/extension/construction/target C++ smoke tests;
  - generated-bundle explicit/pre-realized dry-runs;
  - generated-bundle explicit/pre-realized `ssh rvv` runs;
  - active-authority scan;
  - `git diff --check`;
  - `cmake --build build --target check-tianchenrv -j2` with 259/259 passed.

## Non-Goals

- No broad memory framework or memory movement matrix.
- No gather/scatter, masked strided memory, strided load expansion, dtype/LMUL
  clone batch, contraction/matmul/Linalg/frontend lowering, dashboards, or
  source-front-door positive route.
- No descriptor-driven computation, direct-C/source-export route restoration,
  compatibility wrapper preserving old i32 route authority, or helper-only
  cleanup as the main achievement.
- No new dtype-prefixed `tcrv_rvv.i32_*` helpers or one-intrinsic wrapper
  dialect.
- No RVV strided-store semantics in common EmitC/export, target metadata,
  artifact names, route ids, descriptors, ABI strings alone, tests, or exact
  intrinsic spelling.
- Do not change the separate computed-mask strided-store route unless a
  narrowly required shared helper would otherwise break it; this task's
  executable evidence is the unmasked `unit_load_strided_store` slice.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused RVV dialect/verifier lit tests for positive and negative
   destination byte-strided store selected-body surface.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact FileCheck tests for `unit_load_strided_store`.
5. Run touched C++ tests for route/provider/export/construction helpers.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes.
7. Run generated-bundle dry-runs for explicit and pre-realized
   `unit_load_strided_store` at counts `7,16,23` and destination byte strides
   `4,8,12`.
8. Run real `ssh rvv` correctness for explicit and pre-realized
   `unit_load_strided_store` at counts `7,16,23` and destination byte strides
   `4,8,12`.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run `cmake --build build --target check-tianchenrv -j2`.

## Definition Of Done

- One bounded runtime-byte-strided destination-store route is represented,
  verified, route-supported, materialized through the production RVV provider
  path, dry-run validated, and runtime-validated on `ssh rvv`.
- The old positive element-stride destination-store authority is not preserved
  for this route.
- Existing Stage2 slices remain intact.
- The report distinguishes route-supported, generated-bundle dry-run, and
  executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is introduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; otherwise it remains open with an exact continuation point.
