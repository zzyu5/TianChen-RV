# Stage2 RVV computed-mask byte-strided store executable slice

## Goal

Implement one bounded Stage2 RVV executable slice where a typed comparison
computes a mask and active lanes store signed-i32 payload values to a
destination byte stride chosen at runtime:

```text
if cmp_lhs[i] < cmp_rhs[i]:
  store payload_i32[i] to dst at byte offset i * dst_stride_bytes
else:
  preserve the previous destination value
preserve tail sentinels for i >= n
```

This task composes the already landed computed-mask strided-store, masked
store policy, runtime byte-strided load, and runtime byte-strided store work
only for one e32m1 path. It must make the combined computed-mask plus runtime
destination byte-strided store route explicit in the RVV-owned compiler path,
not add a broad masked-memory framework or rely on element-stride residue.

## Direction Source

- Direction title: `Stage2 RVV computed-mask byte-strided store executable slice`.
- Module owner: RVV plugin-owned computed-mask plus runtime destination
  byte-strided store route for one bounded typed e32m1 path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `eee6a525 rvv: add runtime byte-strided store slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief before source edits.

## Current Repository Facts

- The archived runtime byte-strided store task completed
  `unit_load_strided_store` with `destination-byte-stride`,
  `dst_stride_bytes : size_t`, byte-addressed provider mechanics, generated
  bundle dry-run, and real `ssh rvv` evidence for counts `7,16,23` crossed
  with destination byte strides `4,8,12`.
- The archived masked-store policy task completed an explicit masked
  unit-store slice that proves true-lane writes, false-lane preservation,
  runtime `n`, and tail sentinel preservation.
- The archived computed-mask strided-store task completed a compare-produced
  mask plus payload load plus destination strided store slice using the older
  element-stride contract with `dst_stride=3`.
- The current bottleneck is not adding a new mask or stride family. It is
  replacing the combined computed-mask strided-store byte-stride gap with a
  typed-body and provider-derived byte-strided destination contract.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI/runtime
  roles only. Compare semantics, mask shape, payload dtype, destination stride
  unit, memory form, policy, ABI order, intrinsic choice, route support, and
  executable acceptance must come from the typed `tcrv_rvv` body plus RVV
  plugin legality/realization/provider facts.
- Common EmitC/export may materialize provider-supplied neutral address
  mechanics. It must not infer RVV semantics, predicate, inactive-lane
  behavior, dtype, SEW/LMUL, stride unit, memory form, policy, ABI order, or
  intrinsic choices.
- Stage1/Stage2 guardrails remain active: no positive `RVVI32M1*`,
  `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  descriptor/direct-C/source-export, source-front-door, public exact
  intrinsic route authority, route-id authority, artifact-name authority, or
  common/export RVV semantic authority.

## Requirements

1. Keep scope to one bounded signed i32 / SEW32 / LMUL m1 computed-mask
   runtime byte-strided destination-store slice.
2. The selected/pre-realized RVV body must structurally carry:
   - compare lhs/rhs inputs or an existing typed compare result;
   - simple comparison predicate for mask production, initially `slt` unless
     current code proves another bounded predicate is the established fixture;
   - source payload vector input;
   - destination `mem_window`;
   - runtime `n` / AVL value use;
   - runtime destination byte-stride role, preferably `dst_stride_bytes`;
   - e32m1 vector config;
   - source/payload unit-load memory form;
   - destination masked byte-strided-store memory form;
   - explicit mask/tail policy preserving false lanes and tail sentinels;
   - runtime ABI order.
3. RVV selected-body realization must materialize legal generic typed
   `setvl` / `with_vl`, compare mask dataflow, payload load or compute, and
   runtime byte-strided masked store. It must not change computation
   semantics, dtype semantics, ABI roles, selected variant origin, required
   capabilities, dispatch/fallback behavior, or runtime `n` / AVL values.
4. RVV route planning/provider must derive mask type, destination stride C
   type, vector/result C types, target leaves, headers, ABI order,
   byte-addressed destination mechanics, provider mirrors, artifact mirrors,
   and diagnostics from typed body/config/runtime facts.
5. The byte-strided store route must pass `dst_stride_bytes` as a byte stride.
   It must not preserve the old element-stride `dst_stride` as positive
   authority for this byte-stride route, and must not multiply an already-byte
   stride by `sizeof(int32_t)` in common code.
6. Missing compare/mask producer, non-produced/stale mask, mask/vector/VL
   mismatch, missing payload, missing destination mem_window, missing
   destination byte-stride role, wrong stride C type or ABI order, old
   element-stride authority for this byte-stride route, missing `n` / AVL,
   invalid mask or tail policy/config, stale fixed-VL or route-id authority,
   incomplete typed body, descriptor/direct-C/source-front-door authority, and
   common/export semantic inference must fail closed with targeted diagnostics.
7. Generated-bundle dry-run and `ssh rvv` evidence must cover counts
   `7,16,23` and destination byte strides `4,8,12`, with mixed true/false
   masks proving computed mask use, runtime stride use, selected strided
   destination writes, false-lane preservation, untouched destination sentinel
   preservation, and tail sentinel preservation.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe this bounded
      computed-mask runtime byte-strided destination-store task.
- [x] A selected/pre-realized or explicit typed RVV body structurally carries
      compare/mask facts, payload input, destination mem_window, runtime
      `n` / AVL, destination byte-stride role, e32m1 config, mask/tail policy,
      byte-strided masked-store memory form, and ABI order.
- [x] `RVVSelectedBodyRealization` materializes the bounded pre-realized slice
      into legal generic typed compare, mask, payload, and runtime
      byte-strided masked destination store structure if this slice uses a
      pre-realized fixture path.
- [x] RVV route planning/provider derives ABI order, mask type, vector C type,
      stride C type, destination byte-stride mirrors, target leaves, required
      headers, artifact mirrors, and diagnostics from typed facts.
- [x] Positive route/materialization/generated-artifact tests prove the
      compare-produced mask, payload source, destination, runtime byte-stride,
      byte-strided masked-store memory form, and provider-owned metadata reach
      the production `TCRVEmitCLowerableRoute` path.
- [x] Negative fail-closed coverage exists for missing compare/mask producer,
      non-produced/stale mask, mask/vector mismatch, missing payload, missing
      destination, missing/wrong destination byte-stride role, wrong stride C
      type or ABI order, unsupported old element-stride authority for this
      byte-stride route, missing `n` / AVL, invalid mask/tail policy/config,
      stale fixed-VL or route-id authority, incomplete typed body, and
      common/export semantic inference where meaningful.
- [x] Generated-bundle dry-run passes for the pre-realized selected-body
      fixture at counts `7,16,23` and `dst_stride_bytes` `4,8,12`.
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

- No broad mask matrix, masked-memory framework, gather/scatter framework, all
  strided-memory combinations, dtype/LMUL clone batch, high-level
  Linalg/Vector/StableHLO/frontend lowering, source-front-door positive route,
  dashboard, report-only work, helper-only cleanup, or performance claim.
- No reduction, macc, conversion, broadcast, compare/select expansion,
  indexed/segmented memory expansion, scalar fallback, IME, Offload,
  TensorExt, Template/Toy, or future plugin side quest.
- No descriptor-driven computation, direct-C/source-export route restoration,
  route-id authority, artifact-name authority, or compatibility wrapper
  preserving old i32 route authority.
- No new dtype-prefixed `tcrv_rvv.i32_*` helpers or one-intrinsic wrapper
  dialect.
- Do not migrate unrelated element-stride paths unless current evidence shows
  they are direct active authority for this byte-stride contract. A retained
  element-stride computed-mask route must remain clearly separate or fail
  closed for this byte-stride route.
- No RVV computed-mask byte-strided-store semantics in common EmitC/export,
  target metadata, artifact names, route ids, descriptors, ABI strings alone,
  tests, or exact intrinsic spelling.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused RVV dialect/verifier lit tests for positive and negative
   computed-mask byte-strided-store structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact lit/FileCheck tests.
5. Run touched C++ tests for route/provider/export/construction helpers.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-runs for the computed-mask byte-strided-store op
   at counts `7,16,23` and destination byte strides `4,8,12`.
8. Run real `ssh rvv` correctness for the same counts and byte strides.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run `cmake --build build --target check-tianchenrv -j2`.

## Completion Evidence

- Built focused tools: `cmake --build build --target tcrv-opt tcrv-translate -j2`.
- Script checks passed:
  - `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
  - `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Focused lit passed, 4/4:
  - `pre-realized-computed-mask-strided-store-memory-negative.mlir`
  - `generic-stage2-dataflow.mlir`
  - `pre-realized-selected-body-artifact-computed-masked-strided-store.mlir`
  - `rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-strided-store-dry-run.test`
- Generated-bundle dry-run passed:
  - artifact root: `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-computed-mask-byte-strided-store-pre-dry`
  - mode: `pre-realized-selected-body`
  - op kind: `computed_masked_strided_store`
  - counts: `7,16,23`
  - `dst_stride_bytes`: `4,8,12`
- Real `ssh rvv` generated-bundle run passed:
  - artifact root: `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-computed-mask-byte-strided-store-pre-ssh`
  - counts and byte strides: all nine combinations of `7,16,23` x `4,8,12`
  - evidence output reported `computed_mask`, `byte_strided_store`,
    `selected_destination_writes`, `false_lanes_preserved`,
    `sentinel_preserved`, and `tail_preserved` for every case.
- Active-authority scan on the generated computed-mask byte-strided-store
  artifact had no matches for `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
  `!tcrv_rvv.i32m`, `source-front-door`, `source-export`, `direct-C`, or
  `descriptor`.
- Changed-file scan matches were limited to existing negative tests, forbidden
  FileCheck strings, inherited dialect documentation, and existing RVV provider
  intrinsic mapping text; no new positive authority was introduced for this
  byte-strided computed-mask route.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed, 259/259.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-runtime-strided-store/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-masked-store-policy/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-runtime-strided-load/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-computed-mask-strided-store/prd.md`
- `.trellis/workspace/codex/journal-12.md`

Initial implementation surface to inspect:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Support/RuntimeABI.h`
- `lib/Support/RuntimeABIContract.cpp`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `include/TianChenRV/Plugin/RVV/RVVRuntimeAVLVLControl.h`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing computed-mask strided-store, masked-store, runtime byte-strided
  store, and generated-bundle tests.

## Definition Of Done

- One bounded computed-mask runtime byte-strided destination-store route is
  represented, verified, route-supported, materialized through the production
  RVV provider/common EmitC/export path, dry-run validated, and runtime
  validated on `ssh rvv`.
- The relation to any retained element-stride computed-mask path is explicit:
  it is separate legacy coverage or fail-closed for the byte-stride contract,
  not authority for this route.
- Existing Stage2 slices remain intact.
- The report distinguishes route-supported, generated-bundle dry-run, and
  executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is introduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; otherwise it remains open with an exact continuation point.
