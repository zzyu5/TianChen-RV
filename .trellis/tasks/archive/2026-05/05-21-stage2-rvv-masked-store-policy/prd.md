# Stage2 RVV masked-store mask/tail policy executable slice

## Goal

Implement one bounded Stage2 RVV executable slice for explicit mask/tail policy
plus masked unit-stride store on the corrected typed `tcrv_rvv` surface:

```text
out_i32[i] = mask[i] ? payload_i32[i] : old_out_i32[i], for i < n
out_i32[i] remains tail sentinel, for i >= n
```

The slice must carry a typed mask value or mask import, payload vector, output
`mem_window`, runtime `n` / AVL, e32m1 config, masked-store memory form,
explicit mask policy, explicit tail policy, inactive-lane preservation
semantics, and ABI order from a selected/pre-realized `tcrv.exec` RVV boundary
through RVV plugin-owned selected-body realization, route planning/provider
emission, generated artifact, and real `ssh rvv` evidence.

## Direction Source

- Direction title: `Stage2 RVV masked-store mask/tail policy executable slice`.
- Module owner: RVV plugin-owned explicit mask/tail policy plus masked memory
  store route for one bounded typed e32m1 path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `77868b1b rvv: add runtime avl vl control boundary`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.

## What I Already Know

- The previous completed runtime AVL/VL task archived at
  `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-runtime-avl-vl-control-boundary/`
  introduced a shared RVV-owned runtime control plan consumed by existing
  executable routes and validated with generated-bundle dry-runs plus real
  `ssh rvv` evidence.
- The computed-mask vector select task archived at
  `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-computed-mask-vector-select-executable-slice/`
  established a bounded compare-produced mask executable precedent.
- Specs require the authority chain to remain:
  `tcrv.exec` envelope -> typed low-level `tcrv_rvv` body -> RVV plugin
  selected-body realization / legality / route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact ->
  `ssh rvv` evidence for correctness/runtime claims.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI/runtime
  roles only. They do not define masked-store semantics, dtype/config, policy,
  memory form, route support, or intrinsic spelling.
- Common EmitC/export may only materialize provider-built payloads. It must
  not infer RVV masked-store semantics, inactive-lane behavior, mask type,
  tail policy, dtype, SEW/LMUL, ABI order, or intrinsic choices.
- Current Stage1/Stage2 guardrails remain active: no positive legacy
  `RVVI32M1*`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export, source-front-door,
  public exact intrinsic route authority, or common/export RVV semantic
  authority.

## Requirements

1. Keep scope to one bounded signed i32 / SEW32 / LMUL m1 masked unit-stride
   store executable slice.
2. The selected/pre-realized RVV body must structurally carry:
   - mask producer or explicit typed mask import;
   - payload vector source;
   - output `mem_window`;
   - runtime `n` / AVL value use;
   - e32m1 vector config;
   - masked-store unit-stride memory form;
   - explicit mask policy;
   - explicit tail policy;
   - inactive false-lane preservation semantics;
   - runtime ABI order.
3. RVV selected-body realization must materialize legal generic typed
   `setvl` / `with_vl`, mask value, payload load or compute, and masked store.
   It must not change computation semantics, dtype semantics, ABI roles,
   selected variant origin, required capabilities, dispatch/fallback behavior,
   runtime `n` / AVL, or inactive/tail preservation semantics.
4. RVV route planning/provider must derive mask type, data vector C type,
   result/void route shape, target leaves, required headers, ABI order,
   provider mirrors, artifact mirrors, and fail-closed diagnostics from typed
   body/config/runtime facts.
5. Missing mask, invalid mask policy, invalid tail policy, mask/vector shape
   mismatch, missing output `mem_window`, missing runtime `n` / AVL, stale
   fixed-VL or route-id authority, incomplete typed body, unsupported
   masked-store memory form, and common/export semantic inference must fail
   closed with targeted diagnostics.
6. Generated bundle evidence must use non-trivial mixed true/false masks and
   sentinel-filled output buffers so correctness proves true-lane stores,
   false-lane preservation, runtime `n`, multi-VL behavior where applicable,
   and tail sentinel preservation.
7. Any runtime/correctness claim requires real `ssh rvv` evidence.

## Acceptance Criteria

- [x] Positive dialect/pass/FileCheck or equivalent coverage shows a selected
      or pre-realized typed masked-store body carrying mask, payload, output
      mem_window, runtime n/AVL, e32m1 config, explicit mask/tail policy,
      masked-store memory form, and ABI order.
- [x] RVV selected-body realization materializes or validates legal generic
      typed setvl/with_vl, mask value, payload vector, and masked store without
      inventing semantics in common EmitC/export.
- [x] RVV route planning/provider derives mask type, vector/result C types,
      target leaves, headers, mirrors, and diagnostics from typed
      body/config/runtime facts.
- [x] Positive generated header/artifact evidence exists for the masked-store
      slice through the production RVV provider route.
- [x] Negative fail-closed tests cover missing mask, mask/vector shape
      mismatch, missing output mem_window, invalid mask policy, invalid tail
      policy, missing n/AVL, stale fixed-VL or route-id authority, incomplete
      typed body, and common/export semantic inference.
- [x] Generated-bundle dry-run passes at counts `7,16,23` with mixed
      true/false masks and sentinel output.
- [x] Real `ssh rvv` PASS evidence proves true-lane stores, false-lane
      preservation, runtime `n`, multi-VL behavior where applicable, and tail
      sentinel preservation.
- [x] Active-authority scan confirms no positive `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export, source-front-door, public exact
      intrinsic route authority, or common/export RVV semantic authority is
      introduced.
- [x] Focused build/lit/C++/script checks, `check-tianchenrv`,
      `git diff --check`, task validation, finish/archive, clean git status,
      and one coherent commit are completed if the task finishes.

## Non-Goals

- No broad mask framework or mask-policy matrix.
- No dtype/LMUL clone batch.
- No gather/scatter, strided memory, contraction, matmul, Linalg, Vector, or
  StableHLO frontend lowering.
- No source-front-door positive route.
- No dashboard/report-only/helper-only work as the main result.
- No descriptor-driven computation, direct-C/source-export route restoration,
  or compatibility wrapper preserving old i32 route authority.
- No new dtype-prefixed `tcrv_rvv.i32_*` helpers or one-intrinsic wrapper
  dialect.
- No RVV masked-store semantics in common EmitC/export, target metadata,
  artifact names, route ids, descriptors, ABI strings, tests, or exact
  intrinsic spelling.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused RVV dialect/verifier lit tests for positive and negative
   masked-store policy structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact lit/FileCheck tests.
5. Run touched C++ tests for route/provider/export helpers.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-run for masked store at counts `7,16,23`.
8. Run real `ssh rvv` correctness for masked store at counts `7,16,23`.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run broader `check-tianchenrv`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-runtime-avl-vl-control-boundary/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-computed-mask-vector-select-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-masked-memory-movement-executable-slice/prd.md`

Initial implementation surface to inspect:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `include/TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing masked add, computed-mask select, runtime AVL/VL, and masked
  memory movement tests.

## Definition Of Done

- One bounded explicit-policy masked-store route is represented, verified,
  route-supported, materialized through the production RVV provider path,
  dry-run validated, and runtime-validated on `ssh rvv`.
- Existing Stage2 slices remain intact.
- The report distinguishes route-supported, generated-bundle dry-run, and
  executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; otherwise it remains open with an exact continuation point.

## Completion Evidence

- Realization: `tcrv_rvv.typed_masked_memory_pre_realized_body` with
  `op_kind = "masked_unit_store"`, `memory_form = "masked-unit-store"`, and
  tail/mask undisturbed policy materializes `setvl`, `with_vl`,
  `tcrv_rvv.load`, `tcrv_rvv.mask_load`, and `tcrv_rvv.masked_store`.
- Route/provider: emission metadata mirrors
  `rvv-generic-masked-unit-store-emitc-route`,
  `rvv-generic-masked-unit-store-callable-c-abi.v1`, ABI order
  `src,mask,dst,n`, `masked-store-false-lanes-preserve-output-buffer`,
  and destination memory form `masked-unit-store`.
- Dry-run evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind masked_unit_store --runtime-count 7 --runtime-count 16 --runtime-count 23 ...`
  passed.
- Real hardware evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind masked_unit_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --ssh-target rvv ...`
  passed with active/inactive lane counts `3/4`, `7/9`, `10/13` and
  `tail_preserved` for counts `7,16,23`.
- Full project check: `cmake --build build --target check-tianchenrv -j2`
  passed `257/257`.
- Formatting and residue checks: `git diff --check` passed; generated
  evidence/header/harness residue scan had no `RVVI32M1`, `rvv-i32m1`,
  positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  descriptor/direct-C/source-export/source-front-door, or public exact
  intrinsic residue.
