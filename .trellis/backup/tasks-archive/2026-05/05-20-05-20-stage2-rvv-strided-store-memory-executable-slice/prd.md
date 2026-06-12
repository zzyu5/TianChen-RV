# Stage2 RVV strided-store memory executable slice

## Goal

Complete one bounded Stage2 RVV memory-movement executable slice on the
corrected typed `tcrv_rvv` surface. The concrete route is signed i32 / SEW32 /
LMUL m1:

```text
dst[i * dst_stride] = src[i]
```

The source side is a unit-stride load. The destination side is an explicit
runtime-strided store with an ABI-visible destination stride value and explicit
stride unit. The route must start from a selected `tcrv.exec` RVV boundary and
typed or pre-realized `tcrv_rvv` body, flow through RVV selected-body
realization if needed, RVV plugin-owned route planning/provider output,
neutral common EmitC/export, and real `ssh rvv` correctness evidence if
executable correctness is claimed.

This is one bounded destination-strided memory-movement slice, not a broad
load/store matrix, not indexed or masked memory, and not a high-level frontend
task.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV strided-store memory executable slice`.
- Module owner: RVV plugin-owned strided destination-store memory form for one
  bounded i32 SEW32 LMUL m1 unit-stride-load to runtime-strided-store
  executable slice.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `aafa8210 trellis: record scalar broadcast current-head validation`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- Recent completed Stage2 executable slices cover strided source load to
  unit-stride store, indexed gather/scatter, masked memory, computed-mask
  memory, segment2 deinterleave/interleave, and current-head validation of
  runtime scalar broadcast.
- The existing strided memory movement route is the opposite direction:
  strided source load to unit-stride destination store. This task must prove
  destination strided-store semantics: contiguous source lanes land at
  `dst[i * dst_stride]`, skipped destination lanes remain sentinel-preserved,
  and destination tail storage remains untouched.
- `tcrv.exec` declares ABI/runtime roles only. It must not own destination
  stride semantics, memory form, dtype/config, route support, intrinsic
  spelling, or acceptance state.
- Common EmitC/export must stay neutral and only materialize provider-built
  route payloads.
- Stage1/Stage2 guardrails remain active: no positive `RVVI32M1*`,
  `rvv-i32m1-*`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-seed, descriptor/direct-C/source-export, route-id
  authority, artifact-name authority, or common/export RVV semantic authority.

## Requirements

1. Keep scope to one signed i32 / SEW32 / LMUL m1
   `unit_load_to_strided_store` route with `src`, `dst`, `dst_stride`, and
   `n/AVL` ABI/runtime roles.
2. The selected boundary or typed body must structurally carry source
   `mem_window`, destination `mem_window`, destination stride
   `runtime_param`, stride unit, source unit-stride memory form, destination
   strided-store memory form, vector dtype/config, tail/mask policy, runtime
   `n/AVL`, and ABI roles.
3. If a pre-realized body is used, RVV selected-body realization may
   materialize only legal generic typed memory structure: `setvl`, unit-stride
   source load, optional movement op, and destination `strided_store` or
   equivalent typed store structure. It must not change computation, dtype
   semantics, parameter roles, selected variant origin, required capabilities,
   dispatch/fallback behavior, or runtime `n` / AVL values.
4. RVV route planning must derive ABI order, vector C types, byte-stride
   scaling when the typed unit is element-based, unit-load and strided-store
   intrinsic leaves, header/artifact metadata mirrors, and diagnostics from
   typed body/config/runtime facts.
5. Provider/materializer must consume the provider-built plan. Common
   materialization must not infer RVV destination memory form, stride unit,
   dtype, SEW, LMUL, policy, intrinsic choices, or ABI meaning.
6. Missing destination stride binding, wrong stride role/order, unsupported
   stride unit, invalid destination role, invalid destination memory form,
   missing `n/AVL`, mismatched dtype/config, stale route-id authority, and
   incomplete typed strided-store body structure must fail closed with targeted
   diagnostics.
7. Generated bundle evidence must use `dst_stride > 1`, contiguous
   non-default source values, and sentinel-filled destination/tail storage so
   expected output proves active strided writes, skipped-lane preservation, and
   tail preservation.
8. Real `ssh rvv` evidence is required for any runtime/correctness claim.

## Acceptance Criteria

- [ ] PRD, implement/check context, and task metadata describe this bounded
      strided destination-store memory slice.
- [ ] A selected/pre-realized or explicit typed RVV body structurally carries
      `src`, `dst`, `dst_stride`, stride unit, source unit-load memory form,
      destination strided-store memory form, vector dtype/config, tail/mask
      policy, runtime `n/AVL`, and ABI roles.
- [ ] `RVVSelectedBodyRealization` materializes the bounded pre-realized slice
      into legal generic typed unit-load plus strided-store structure if this
      slice uses a pre-realized fixture path.
- [ ] RVVEmitCRoutePlanning derives ABI order, vector C types, destination
      byte-stride scaling for element-based stride, unit-load leaves,
      strided-store leaves, header/artifact metadata, and diagnostics from
      typed facts.
- [ ] Positive route/materialization tests prove typed source, destination,
      runtime destination stride, stride unit, and destination memory-form
      facts reach `TCRVEmitCLowerableRoute` and provider-owned route metadata.
- [ ] Negative fail-closed tests cover missing destination stride binding,
      wrong stride role/order, unsupported stride unit, invalid destination
      role/form, mismatched dtype/config, missing AVL/runtime roles, stale
      route-id authority, and incomplete typed body structure.
- [ ] Generated-bundle dry-run passes for the strided-store slice at counts
      `7,16,23`.
- [ ] Real `ssh rvv` correctness passes for counts `7,16,23` if executable
      correctness is claimed, with non-vacuous expected outputs proving
      `src[i]` lands at `dst[i * dst_stride]`, skipped destination lanes remain
      sentinels, and tail sentinels are preserved.
- [ ] Focused build, lit, C++ unit, script, and artifact checks for touched RVV
      dialect/config, selected-body realization, route planning/provider,
      materializer/export, and generated-bundle paths pass.
- [ ] Active-authority scan confirms no positive legacy/source/descriptor or
      common/export RVV semantic authority is reintroduced.
- [ ] `git diff --check`, Trellis task validation, task status update/archive,
      and one coherent production-code commit are completed if the task is
      finished.

## Non-Goals

- No repeat of scalar broadcast validation, Trellis-only success, broad
  strided load/store matrix, indexed+strided combinations, masked strided
  store, dtype/LMUL clone batches, high-level Linalg/Vector/StableHLO
  lowering, source-front-door positive route, one-intrinsic wrapper dialect,
  dashboard, report-only inventory, helper-only refactor, or performance
  claim.
- No new reduction, macc, conversion, broadcast, compare/select, masked memory,
  gather/scatter, segment memory, or source-front-door side quest.
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
   destination strided-store structure.
4. Run focused selected-body realization, route planning/provider, EmitC, and
   target artifact lit/FileCheck tests.
5. Run touched C++ tests where route/provider/export helpers are covered.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or generated-bundle evidence is claimed.
7. Run generated-bundle dry-run for the strided-store op at counts `7,16,23`.
8. Run real `ssh rvv` correctness for the strided-store op at counts `7,16,23`
   if executable correctness is claimed.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run broader `check-tianchenrv` if shared compiler behavior changes.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-strided-memory-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-indexed-scatter-memory-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-segment2-interleave-memory-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-runtime-scalar-broadcast/prd.md`
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
- Existing strided-load, indexed-scatter, and segment2 generated-bundle tests.

## Definition Of Done

- One coherent unit-load to runtime destination-strided-store route is
  represented, verified, route-supported, materialized through the production
  RVV provider/common EmitC/export path, and executable on `ssh rvv` for counts
  `7,16,23` if correctness is claimed.
- Existing strided-load, indexed gather/scatter, masked memory, computed-mask
  memory, reduction, macc, broadcast, conversion, tail/mask, and segment2
  Stage2 routes remain intact.
- The final report distinguishes route-supported evidence, dry-run generated
  artifact evidence, and executable `ssh rvv` evidence.
- No legacy i32/source/descriptor/common-export authority is reintroduced.

## Completion Evidence

Implemented in this Trellis round:

- Added `tcrv_rvv.typed_strided_store_memory_pre_realized_body` for the bounded
  unit-load to runtime destination-strided-store slice. The verifier requires
  `src`, `dst`, runtime `n`, runtime `dst_stride`, `stride_unit = "element"`,
  `memory_form = "unit-load-strided-store"`, SEW32, LMUL m1, and agnostic
  tail/mask policy.
- Added selected-body realization from the pre-realized op to
  `setvl -> with_vl -> load -> move -> strided_store`.
- Added RVV config/runtime ABI contract `src,dst,n,dst_stride` with
  `dst_stride` role `output-stride`.
- Added RVV construction/protocol, route planning, route provider, metadata,
  header/export, and generated-bundle script support for
  `unit_load_strided_store`.
- Added positive explicit and pre-realized selected-body lit tests, negative
  fail-closed tests for verifier/provider paths, and generated-bundle dry-run
  lit tests.
- Updated C++ construction/target artifact tests for the new bounded route.

Validation completed:

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- `build/bin/tianchenrv-rvv-dialect-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-construction-protocol-common-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- Focused lit filter for the six new RVV/EmitC/script tests: 6/6 passed.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/stage2_unit_load_strided_store_evidence --run-id explicit-unit-load-strided-store --overwrite --op-kind unit_load_strided_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_unit_load_strided_store_evidence --run-id pre-realized-unit-load-strided-store --overwrite --op-kind unit_load_strided_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv`
- `git diff --check`
- `ninja -C build check-tianchenrv` passed 227/227.

Executable evidence:

- Explicit selected-body generated bundle passed on `ssh rvv` for counts
  `7,16,23` with `dst_stride=3`.
- Pre-realized selected-body generated bundle passed on `ssh rvv` for counts
  `7,16,23` with `dst_stride=3`.
- Harness checks prove `src[index]` lands at `dst[index * dst_stride]`, all
  skipped destination lanes keep the sentinel, and destination tail storage
  keeps the sentinel.

Active-authority scan:

- No new positive production/default `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed,
  descriptor/direct-C/source-export, or common/export RVV semantic authority
  was introduced.
- New forbidden-string hits are limited to negative/guardrail text: the new op
  documentation says descriptor/source-front-door/direct-C are not authority,
  generated-bundle tests use `implicit-check-not`, and the stale `rvv-i32m1`
  string appears only in a negative verifier test that must reject `route_id`
  authority metadata.
