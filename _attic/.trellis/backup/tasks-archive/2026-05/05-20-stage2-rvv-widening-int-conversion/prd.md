# Stage2 RVV widening integer conversion executable slice

## Goal

Implement one bounded Stage2 RVV signed widening conversion executable slice:

```text
out_i32[i] = sign_extend_i16_to_i32(in_i16[i])
```

The selected RVV path must carry an i16 source memory role, i32 destination
memory role, source vector config `SEW16 LMUL mf2`, result vector config
`SEW32 LMUL m1`, conversion kind `sign_extend_widen_vf2`, tail/mask policy,
runtime `n` / AVL, ABI roles, route planning, generated artifact emission,
and real `ssh rvv` correctness evidence for representative counts.

This is one bounded non-i32-input widening conversion slice. It is not a
conversion matrix, dtype/LMUL clone batch, frontend lowering task, source-front
door route, descriptor route, or performance task.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV widening integer conversion executable slice`.
- Module owner: RVV plugin-owned typed SEW/LMUL conversion path for one signed
  i16 SEW16 LMUL mf2 to signed i32 SEW32 LMUL m1 sign-extension executable
  slice.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `1038f995 rvv: add computed-mask strided-store executable slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- HEAD already contains an earlier Stage2 widening conversion route for
  `i32 m1 -> i64 m2`, including `tcrv_rvv.widening_convert`,
  selected-body realization, route planning/provider, construction protocol,
  generated-bundle support, and tests.
- The existing conversion path is still bounded to `widen_i32_to_i64` /
  `signed-i32m1-to-i64m2` in RVV dialect verification, selected-body
  realization, route planning, construction metadata, script expectations,
  and tests.
- Recent Stage2 memory slices established typed route planning and generated
  bundle patterns for explicit runtime ABI roles, tail preservation, and
  `ssh rvv` evidence at counts `7,16,23`.
- This task extends the corrected typed `tcrv_rvv` conversion surface with one
  non-i32 source case. Dtype, SEW, LMUL, conversion semantics, memory form,
  policy, and intrinsic mapping must be validated or derived from typed
  body/config/runtime facts, not from route ids, ABI strings, artifact names,
  test names, descriptors, or exact intrinsic spelling.
- `tcrv.exec` declares ABI/runtime roles only. RVV conversion compute/config,
  selected-body realization, legality, intrinsic mapping, and fail-closed
  diagnostics remain RVV plugin-owned. Common EmitC/export must stay neutral.

## Requirements

1. Keep scope to one signed `i16mf2 -> i32m1` unit-stride conversion route with
   `lhs`, `out`, and runtime `n/AVL` ABI roles.
2. The selected/pre-realized body or explicit typed body must structurally
   carry source buffer role, destination buffer role, source element type,
   source SEW/LMUL, result element type, result SEW/LMUL, conversion kind
   `sign_extend_widen_vf2`, memory form `unit-stride-conversion`, tail/mask
   policy, runtime `n/AVL`, and ABI role order.
3. `RVVSelectedBodyRealization` must materialize only legal generic typed
   structure for this slice: destination-config `setvl/with_vl`, source
   i16mf2 unit load, generic `tcrv_rvv.widening_convert`, and i32m1 unit store.
4. RVV route planning must derive ABI order, source/destination C pointer
   types, source/result vector C types, setvl/load/store/conversion intrinsic
   leaves, route metadata mirrors, and targeted diagnostics from typed
   body/config/runtime facts.
5. Provider/materializer must consume the provider-built plan. Common
   materialization must not infer dtype, SEW, LMUL, conversion kind, memory
   form, policy, intrinsic choice, or ABI meaning.
6. Unsupported conversion kind, invalid SEW/LMUL pairing, missing source or
   result config, source/result dtype mismatch, missing `n/AVL`, wrong ABI
   role/order, stale route-id authority, and incomplete typed body structure
   must fail closed with targeted diagnostics.
7. Generated-bundle evidence must use positive and negative i16 lanes and
   sentinel-filled output storage so expected output proves sign extension for
   negative lanes, preservation of positive values, and tail preservation.
8. Real `ssh rvv` evidence is required for any executable correctness claim.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe this bounded
      `i16mf2 -> i32m1` signed widening conversion slice.
- [x] Positive selected/pre-realized or explicit typed RVV body structurally
      carries source dtype/config, result dtype/config, signed widening
      conversion kind, unit-stride source/destination memory form, policy,
      runtime `n/AVL`, and ABI roles.
- [x] `RVVSelectedBodyRealization` materializes the bounded conversion into
      explicit typed `setvl/with_vl/load/widening_convert/store` structure.
- [x] RVVEmitCRoutePlanning derives source/destination ABI types, vector
      types, setvl/load/store/conversion intrinsics, route metadata, and
      runtime ABI order from typed body/config facts.
- [x] Positive route/materialization tests prove typed i16mf2 source facts and
      i32m1 result facts reach `TCRVEmitCLowerableRoute` and provider-owned
      route metadata.
- [x] Negative fail-closed tests cover unsupported conversion kind, invalid
      SEW/LMUL relation, missing/mismatched source or result config, missing
      AVL/runtime role, wrong memory form, stale route-id authority, and
      incomplete typed body structure.
- [x] Generated-bundle dry-run passes for the widening conversion slice at
      counts `7,16,23`.
- [x] Real `ssh rvv` correctness passes for counts `7,16,23` if executable
      correctness is claimed, with expected output proving negative i16 inputs
      sign-extend to negative i32 outputs, positive inputs remain positive,
      and tail sentinels are preserved.
- [x] Focused build, lit, C++ unit, script, and artifact checks for touched RVV
      dialect/config, runtime ABI, construction protocol, selected-body
      realization, route planning/provider, materializer/export, and
      generated-bundle paths pass.
- [x] Active-authority scan confirms no positive legacy/source/descriptor or
      common/export RVV semantic authority is reintroduced.
- [x] `git diff --check`, Trellis task validation, task status update/archive,
      and one coherent production-code commit are completed if the task is
      finished.

## Non-Goals

- No broad dtype/LMUL matrix, unsigned/zero-extension matrix, narrowing
  conversions, floating-point support, high-level Linalg/Vector/StableHLO
  lowering, source-front-door positive route, one-intrinsic wrapper dialect,
  dashboard, report-only inventory, helper-only refactor, or performance
  claim.
- No Scalar, IME, Offload, TensorExt, future plugin work, global autotuning,
  readiness state machine, or source-shape clone batch.
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
   widening conversion structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact lit/FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-run for the new widening conversion op at counts
   `7,16,23`.
8. Run real `ssh rvv` correctness for the new widening conversion op at counts
   `7,16,23` if executable correctness is claimed.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run broader `check-tianchenrv` if shared compiler behavior changes.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-computed-mask-strided-store/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-widening-conversion/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-runtime-scalar-broadcast/prd.md`
- `.trellis/workspace/codex/journal-12.md`

Initial implementation surface inspected:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `include/TianChenRV/Support/RuntimeABI.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Support/RuntimeABIContract.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing conversion and generated-bundle tests for `widen_i32_to_i64`.

## Completion Evidence

- Focused C++ build succeeded for `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`, and
  `tianchenrv-target-artifact-export-test`.
- Focused verifier/materialization checks passed for
  `generic-widening-conversion-dataflow.mlir`,
  `rvv-pre-realized-widening-conversion-negative.mlir`, and
  `pre-realized-selected-body-artifact-widen-i16-to-i32.mlir`.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- `check-tianchenrv` passed `232/232`.
- Generated-bundle dry-run passed for `widen_i16_to_i32` at counts `7,16,23`
  under `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-widen-i16-to-i32-dry`.
- Real `ssh rvv` evidence passed for `widen_i16_to_i32` at counts `7,16,23`
  under `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-widen-i16-to-i32-ssh`;
  remote output reported `sign_extension_checked tail_preserved` for each
  count and `PASS op=widen_i16_to_i32 counts=7,16,23`.
- Active-authority scan of the new SSH evidence files returned no forbidden
  legacy/source/descriptor/direct-C/source-export tokens. Broader repo scan
  hits were existing guardrail text, deprecated parser inventory, fail-closed
  source-front-door paths, or negative tests.
