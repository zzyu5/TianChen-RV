# Stage2 RVV runtime-strided load executable slice

## Goal

Implement one bounded Stage2 RVV executable memory-movement slice where a
runtime byte-stride source load feeds a contiguous output store:

```text
out_i32[i] = load_i32_at_byte_stride(src, i, stride_bytes), for i < n
out_i32[i] remains tail sentinel, for i >= n
```

The slice must carry source buffer, output buffer, runtime `n` / AVL, runtime
`stride_bytes`, e32m1 config, strided-load source memory form, contiguous-store
destination memory form, tail/mask policy, route-provider mirrors, generated
artifact, and real `ssh rvv` evidence through the RVV-owned compiler path.

## Direction Source

- Direction title: `Stage2 RVV runtime-strided load memory movement executable slice`.
- Module owner: RVV plugin-owned runtime-strided load plus contiguous store
  route for one bounded typed e32m1 path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `1ccbb0e8 rvv: add masked store policy executable slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.

## Current Repository Facts

- The previous masked-store policy task archived at
  `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-masked-store-policy/`
  completed explicit mask/tail policy plus generated-bundle and `ssh rvv`
  evidence.
- The previous runtime AVL/VL task archived at
  `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-runtime-avl-vl-control-boundary/`
  introduced the shared runtime-control contract for selected-body consumers.
- Current code already contains an older bounded `strided_load_unit_store`
  path, but that path uses `stride_unit = "element"`, ABI name `src_stride`,
  role `lhs-input-stride`, and provider code that multiplies the stride by
  4 before calling the strided-load intrinsic.
- This task must migrate or tighten that bounded source-strided load slice to
  byte-stride authority. It must not add a parallel helper-only route beside
  the old element-stride route.

## Requirements

1. Keep scope to one bounded signed i32 / SEW32 / LMUL m1 source
   runtime-byte-strided load plus contiguous output store slice.
2. The selected/pre-realized RVV body must structurally carry:
   - source buffer ABI value;
   - output buffer ABI value;
   - runtime `n` / AVL value use;
   - runtime byte-stride ABI value, preferably `stride_bytes` or an explicit
     source byte-stride role;
   - e32m1 vector config;
   - source memory form `strided-load`;
   - destination memory form `unit-stride-store`;
   - tail/mask agnostic policy for this bounded unmasked slice;
   - runtime ABI order.
3. RVV selected-body realization must materialize legal generic typed
   `setvl` / `with_vl`, runtime-byte-strided `tcrv_rvv.strided_load`,
   structural `tcrv_rvv.move`, and contiguous `tcrv_rvv.store`.
4. RVV route planning/provider must derive the stride C type, source/result
   vector C types, target leaves, headers, runtime ABI order, provider mirrors,
   artifact mirrors, and diagnostics from typed body/config/runtime facts.
5. Provider-generated C/C++ must pass byte stride directly to the RVV
   strided-load leaf. It must not reinterpret a byte stride as an element
   stride or multiply it by `sizeof(int32_t)` inside the provider.
6. Common EmitC/export must remain neutral. It may materialize provider-built
   payloads, but it must not infer stride unit, dtype, policy, ABI order,
   intrinsic choice, memory form, or RVV semantics.
7. Missing stride role, wrong stride type, wrong stride ABI order, zero or
   unsupported stride if rejected by the bounded contract, missing runtime
   `n` / AVL, invalid tail/mask policy, stale fixed-VL or route-id authority,
   incomplete typed body, descriptor/direct-C/source-front-door authority, and
   common/export semantic inference must fail closed with targeted diagnostics.
8. Generated-bundle dry-run and `ssh rvv` evidence must cover runtime counts
   `7,16,23` and byte strides `4,8,12`, with sentinel-padded source/output
   buffers proving runtime stride use, runtime `n`, multi-VL behavior where
   applicable, contiguous output correctness, and tail sentinel preservation.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe this bounded
      byte-strided source-load task.
- [x] The selected/pre-realized body accepts and requires a runtime byte-stride
      source load fact and no longer treats the old element-stride source
      `strided_load_unit_store` path as positive executable authority.
- [x] RVV selected-body realization materializes `setvl`, `with_vl`,
      `strided_load`, `move`, and contiguous `store` from the typed
      byte-stride body.
- [x] RVV route planning/provider mirrors byte-stride layout/source, ABI order,
      C type mapping, required headers, target leaves, source/destination
      memory forms, and provider-supported status only after typed body
      validation.
- [x] Generated header/artifact evidence exists for the production
      `strided_load_unit_store` route and shows a byte-stride ABI parameter.
- [x] Negative fail-closed coverage exists for missing/wrong byte-stride role,
      wrong stride C type or ABI order, unsupported element-stride positive
      authority for this source-strided load route, missing n/AVL, invalid
      policy/config, stale fixed-VL or route-id authority, incomplete typed
      body, and common/export semantic inference where meaningful.
- [x] Generated-bundle dry-run passes for counts `7,16,23` and byte strides
      `4,8,12` with sentinel-padded source/output checks.
- [x] Real `ssh rvv` generated-bundle runs pass for the same bounded coverage.
- [x] Active-authority scan confirms no positive `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export, source-front-door, public exact
      intrinsic route authority, or common/export RVV semantic authority is
      introduced.
- [x] Focused build/lit/C++/script checks, `check-tianchenrv`,
      `git diff --check`, task validation, finish/archive, clean git status,
      and one coherent commit are completed if the task finishes.

## Non-Goals

- No broad memory movement matrix.
- No gather/scatter, masked strided memory, strided store expansion, dtype/LMUL
  clone batch, contraction/matmul/Linalg/frontend lowering, or source-front-door
  positive route.
- No descriptor-driven computation, direct-C/source-export route restoration,
  compatibility wrapper preserving old i32 route authority, or helper-only
  cleanup as the main achievement.
- No new dtype-prefixed `tcrv_rvv.i32_*` helpers or one-intrinsic wrapper
  dialect.
- No RVV strided-load semantics in common EmitC/export, target metadata,
  artifact names, route ids, descriptors, ABI strings alone, tests, or exact
  intrinsic spelling.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused RVV dialect/verifier lit tests for the positive and negative
   byte-strided load selected-body surface.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact FileCheck tests.
5. Run touched C++ tests for route/provider/export helpers if the build
   surface changes them.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes.
7. Run generated-bundle dry-runs for `strided_load_unit_store` at counts
   `7,16,23` and byte strides `4,8,12`.
8. Run real `ssh rvv` correctness for `strided_load_unit_store` at counts
   `7,16,23` and byte strides `4,8,12`.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run `cmake --build build --target check-tianchenrv -j2`.

## Completion Evidence

- Production route authority migrated from `src_stride` / `lhs-input-stride` /
  `stride_unit = "element"` to `stride_bytes` / `source-byte-stride` /
  `stride_unit = "byte"` for the bounded `strided_load_unit_store` slice.
- RVV dialect/config/selected-body realization now require `source-input-buffer`
  plus `source-byte-stride`; legacy `lhs-input-buffer` source authority and
  element stride are rejected for this route.
- RVV route planning/provider now derives runtime ABI order
  `src,out,n,stride_bytes`, byte-strided source layout, source/destination
  memory forms, and direct byte-stride emission from typed body/config/runtime
  facts.
- Common EmitC materialization gained neutral support for provider-supplied
  C-style casted scaled pointer expressions by lowering them through EmitC
  casts/mul/add; it does not choose RVV memory semantics.
- Generated-bundle dry-run artifacts:
  - `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-strided-load-byte-explicit-dry`
  - `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-strided-load-byte-pre-dry`
- Real `ssh rvv` artifacts:
  - `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-strided-load-byte-explicit-ssh`
  - `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-strided-load-byte-pre-ssh`
- Both `ssh rvv` runs passed all nine runtime cases:
  `n = 7,16,23` crossed with `stride_bytes = 4,8,12`, reporting
  `byte_strided_load contiguous_output tail_preserved` and
  `PASS op=strided_load_unit_store counts=7,16,23 stride_bytes=4,8,12`.
- Focused lit passed for dialect negative, EmitC negative, target artifact, and
  generated-bundle dry-run tests.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- Touched C++ smoke/unit tests passed:
  `tianchenrv-rvv-dialect-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`, and
  `tianchenrv-target-artifact-export-test`.
- Active-authority scan found no added positive `RVVI32M1`, `rvv-i32m1`,
  finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  `__riscv_*_i32m1`, source-front-door/source-export, descriptor/direct-C, or
  old `src_stride` route authority.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed 259/259 lit
  tests.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-masked-store-policy/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-runtime-avl-vl-control-boundary/prd.md`

Initial implementation surface inspected:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `include/TianChenRV/Support/RuntimeABI.h`
- `include/TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing strided load/store, masked-store, runtime AVL/VL, and
  generated-bundle tests.

## Definition Of Done

- One bounded runtime-byte-strided source-load plus contiguous output-store
  route is represented, verified, route-supported, materialized through the
  production RVV provider path, dry-run validated, and runtime-validated on
  `ssh rvv`.
- The old positive element-stride source-load authority is not preserved for
  this route.
- Existing Stage2 slices remain intact.
- The report distinguishes route-supported, generated-bundle dry-run, and
  executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is introduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; otherwise it remains open with an exact continuation point.
