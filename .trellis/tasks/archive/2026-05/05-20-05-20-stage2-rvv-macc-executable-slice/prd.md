# Stage2 RVV multiply-accumulate executable slice

## Goal

Complete one bounded Stage2 RVV multiply-accumulate executable slice on the
corrected generic typed `tcrv_rvv` surface. The concrete operation is signed
i32 / SEW32 / LMUL m1 / unit-stride vector macc:

```text
out[i] = out[i] + lhs[i] * rhs[i]
```

The current repository already contains a generic `tcrv_rvv.macc` op,
`tcrv_rvv.typed_macc_pre_realized_body`, selected-body realization, route
planning/provider support, target artifact fixtures, and generated-bundle
dry-run tests. The production gap is that `typed_macc_pre_realized_body`
carries `accumulator_layout` and `result_layout`, but the realized
`tcrv_rvv.macc` op consumed by route planning only carries `kind = "add"`;
route planning then fills the macc layout mirrors from constants instead of
deriving them from realized typed body facts.

This task makes accumulator/result layout structural on the realized typed
`tcrv_rvv.macc` body, makes selected-body realization consume the pre-realized
layout facts into that op, and makes route planning/provider derive macc
metadata from those typed body facts before generated artifact and `ssh rvv`
correctness evidence is claimed.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV multiply-accumulate executable slice`.
- Module owner: RVV plugin-owned contraction-supporting multiply-accumulate
  route for one bounded i32 unit-stride vector macc path.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `41c2c96b rvv: make reduce add layout typed body authority`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- Stage2 reduce-add just made layout facts structural on realized
  `tcrv_rvv.reduce` and proved the route with real `ssh rvv` evidence.
- Current macc code has a parallel shape but has not made layout facts
  structural on realized `tcrv_rvv.macc`.
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` defines
  `TypedMAccPreRealizedBodyOp` with `accumulator_layout` and `result_layout`,
  but `MAccOp` only accepts `kind`.
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp` realizes
  `typed_macc_pre_realized_body` to `tcrv_rvv.macc {kind = "add"}` without
  transferring layout facts.
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` sets
  `maccAccumulatorLayout` and `maccResultLayout` from bounded constants when
  the operation profile is multiply-accumulate.
- Existing dry-run fixtures already assert macc metadata mirrors, but they do
  not prove the realized typed macc body structurally carries those layout
  facts.
- Prior memory/guardrails note a previous i32 macc drift; this task must not
  introduce `tcrv_rvv.i32_macc`, `RVVI32M1*`, `rvv-i32m1-*`, source-front-door,
  descriptor, direct-C, or common/export semantic authority.

## Requirements

1. Keep the scope to one signed i32 / SEW32 / LMUL m1 / unit-stride
   `macc_add` route with `lhs`, `rhs`, `out/accumulator`, and runtime `n/AVL`
   ABI roles.
2. `tcrv.exec` remains only the execution envelope and ABI/runtime boundary;
   it must not own macc compute, dtype/config, accumulator layout, result
   layout, route selection, or intrinsic spelling.
3. `tcrv_rvv.typed_macc_pre_realized_body` must carry macc kind, memory form,
   accumulator role/layout, result layout, dtype/config/policy, and runtime
   ABI value uses.
4. The realized typed `tcrv_rvv.macc` op must structurally carry:
   `kind = "add"`, `accumulator_layout =
   "output-buffer-vector-accumulator-input"`, and `result_layout =
   "store-multiply-accumulate-result-to-output-buffer"`.
5. RVV selected-body realization must copy the pre-realized macc layout facts
   into realized `tcrv_rvv.macc`; it must not infer them from op kind, route
   id, ABI strings, artifact names, or C strings.
6. `MAccOp::verify()` and route planning must fail closed with targeted
   diagnostics for missing or unsupported macc layout facts.
7. RVVEmitCRoutePlanning/provider must derive macc metadata mirrors from the
   realized `tcrv_rvv.macc` attrs rather than from unconditional constants.
8. Generated artifact paths must continue through provider-built
   `TCRVEmitCLowerableRoute` and neutral common EmitC/export.
9. Existing explicit and pre-realized macc dry-run/artifact paths must remain
   valid after the structural layout attrs are required.
10. Preserve Stage1/Stage2 guardrails: no positive legacy `RVVI32M1`,
    `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
    source-front-door/source-seed, descriptor/direct-C/source-export,
    route-id authority, artifact-name authority, or common/export RVV semantic
    authority.

## Acceptance Criteria

- [ ] `tcrv_rvv.macc` admits explicit accumulator/result layout attrs for the
      bounded macc route and rejects missing/unsupported layout facts.
- [ ] RVV selected-body realization materializes pre-realized macc into
      `setvl`, `with_vl`, lhs load, rhs load, output-buffer accumulator load,
      `tcrv_rvv.macc {kind = "add", accumulator_layout = ..., result_layout =
      ...}`, and store.
- [ ] RVV route planning reads macc layout facts from `tcrv_rvv.macc` and uses
      them to populate provider-derived macc metadata mirrors.
- [ ] Positive explicit/pre-realized macc tests prove the structural layout
      attrs reach materialized route metadata and target artifact output.
- [ ] Negative tests cover unsupported kind, missing accumulator layout,
      missing result layout, unsupported accumulator layout, unsupported result
      layout, missing/wrong lhs/rhs/out/n roles, wrong memory/result form,
      stale route-id authority, and incomplete typed body structure.
- [ ] Generated-bundle dry-run passes for `macc_add` counts `7,16,23` for both
      explicit and pre-realized paths as relevant.
- [ ] Real `ssh rvv` correctness evidence passes for counts `7,16,23` if this
      round claims executable correctness.
- [ ] Focused build/lit/unit/script checks for touched RVV dialect,
      selected-body realization, route planning/provider, target artifact, and
      generated-bundle paths pass.
- [ ] Active-authority scan confirms no positive legacy/source/descriptor or
      common/export RVV semantic authority is reintroduced.
- [ ] `git diff --check`, Trellis task validation, final task status update,
      archive, and one coherent production-code commit are completed if the
      task is finished.

## Non-Goals

- No high-level matmul/convolution/linalg lowering.
- No broad macc or contraction framework.
- No floating-point, widening, extra dtype, or LMUL expansion.
- No source-front-door positive route, source-seed route, one-intrinsic wrapper
  dialect, compatibility wrapper, dashboard, global tuning database, report-only
  inventory, or performance claim.
- No Python implementation of compiler core, dialects, passes, plugin registry,
  capability model, lowering, or emission.
- No metadata-only closeout.

## Validation Plan

1. Validate Trellis task context.
2. Build focused targets:
   `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-dialect-test`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused lit/FileCheck tests for generic RVV dataflow macc verifier,
   macc selected-body realization, explicit/pre-realized target artifacts, macc
   route/provider negatives, and generated-bundle dry-run.
4. Run touched C++ tests where route/provider/export helpers are covered.
5. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if script
   behavior changes or generated-bundle evidence is claimed.
6. Run generated-bundle dry-run for `macc_add` counts `7,16,23`.
7. Run real `ssh rvv` correctness for `macc_add` counts `7,16,23` if
   executable correctness is claimed.
8. Run active-authority scans over active RVV include/lib/test/script paths.
9. Run `git diff --check`.
10. Run `check-tianchenrv` if focused validation or shared source changes
    justify the broader gate.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-reduce-add-production-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-compare-select-route/prd.md`

Initial code/test surfaces inspected:

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
- Existing macc tests under `test/Dialect/RVV`,
  `test/Transforms/LoweringBoundary`, `test/Target/RVV`, and `test/Scripts`.

## Definition Of Done

- One bounded macc route structurally carries accumulator/result layout facts
  through realized typed `tcrv_rvv.macc` into RVV route planning/provider
  metadata and generated artifacts.
- Fresh focused positive/negative/generated-bundle/runtime evidence is current
  to this task when claimed.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The task is finished/archived only if production code/tests/evidence are
  present in this round; otherwise it remains open with an exact continuation
  point.

## Implementation Results

- Added optional structural `accumulator_layout` and `result_layout` attrs to
  generic `tcrv_rvv.macc`.
- Tightened `MAccOp::verify()` so the bounded Stage2 macc route fails closed
  when either layout fact is missing or unsupported.
- Updated RVV selected-body realization so
  `tcrv_rvv.typed_macc_pre_realized_body` consumes its explicit
  accumulator/result layout facts into the realized `tcrv_rvv.macc` op before
  provider route construction.
- Updated RVVEmitCRoutePlanning so provider-derived macc metadata mirrors are
  read from the realized `tcrv_rvv.macc` layout attrs instead of filled only
  from bounded constants.
- Updated the generated-bundle evidence script to require materialized
  selected-body MLIR for `macc_add` to contain the structural macc layout
  attrs.
- Updated positive explicit/pre-realized macc fixtures, target export C++
  fixture generation, and EmitC macc materialization fixtures to use the new
  typed body layout facts.
- Added dialect negative coverage for missing accumulator layout, missing
  result layout, unsupported accumulator layout, and unsupported result layout.

## Validation Results

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-20-05-20-stage2-rvv-macc-executable-slice`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] Focused lit from `build/test`: 8/8 passed for generic macc dataflow,
  EmitC macc materialization/negative coverage, explicit/pre-realized target
  artifacts, pre-realized macc negatives, and generated-bundle macc dry-run
  fixtures.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Local generated-bundle dry-run for explicit `macc_add` counts `7,16,23`.
- [OK] Local generated-bundle dry-run for pre-realized `macc_add` counts
  `7,16,23`.
- [OK] Real `ssh rvv` explicit selected-body correctness:
  `PASS op=macc_add counts=7,16,23`.
- [OK] Real `ssh rvv` pre-realized selected-body correctness:
  `PASS op=macc_add counts=7,16,23`.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 193/193 passed.
- [OK] `git diff --check`
- [OK] Added-authority scan over diff-added RVV include/lib/script/test hunks
  found no `RVVI32M1`, `rvv-i32m1`, positive `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-seed, descriptor/direct-C/
  source-export, or route-id authority reintroduction. Broader repository
  matches remain pre-existing deprecated inventory, fail-closed tests, or
  harness guardrails.

## Status

[OK] Completed and ready to archive. This round claims executable correctness
only for the bounded signed i32 / SEW32 / LMUL m1 / unit-stride `macc_add`
slice at counts `7,16,23`; no performance claim is made.

## Next Continuation Point

No continuation is required for this bounded slice. Future Stage2 work, if
requested, should open a separate PRD for the next low-level RVV capability
class without expanding this task into matmul, broad contraction, dtype/LMUL
matrix, source-front-door, or one-intrinsic wrapper work.
