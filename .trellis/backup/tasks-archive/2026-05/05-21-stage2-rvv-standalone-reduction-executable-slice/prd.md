# Stage2 RVV standalone reduction executable slice

## Goal

Implement one bounded RVV plugin-owned standalone horizontal reduction slice:

```text
out_scalar_i32[0] = seed_i32 + sum_{i = 0 .. n-1} input_i32[i]
```

The slice must use the corrected typed low-level `tcrv_rvv` body surface and
the RVV-owned selected-body realization, route planning, provider emission, and
target artifact path. This is a new scalar-output standalone reduction
boundary; it is not the existing chunk-wise `reduce_add` route where each
dynamic-VL chunk stores a lane-0 result at the chunk base.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV standalone reduction executable slice`.
- Module owner: RVV plugin-owned standalone horizontal reduction route family
  for one bounded typed e32m1 add-reduction path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `1763d813 rvv: consolidate scalar broadcast route family plan`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.

## Current Repository Facts

- The previous scalar-broadcast family task consolidated the existing
  `scalar_broadcast_add` path behind a plugin-local route-family plan consumed
  by route planning, provider emission, target mirrors, generated-bundle
  tooling, and `ssh rvv` evidence.
- Archived reduction tasks already provide a generic `tcrv_rvv.reduce` route
  surface and a bounded chunk-wise `reduce_add` executable path with structural
  accumulator/result layout attrs on `tcrv_rvv.reduce`.
- The existing `reduce_add` route is not the target behavior for this task:
  it reduces each dynamic-VL chunk and stores lane 0 at the chunk base. This
  task needs a standalone vector-to-scalar reduction accumulating across the
  full runtime `n` into `out_scalar[0]`.
- Archived contraction selected-body realization work provides reusable
  patterns for plugin-local family realization helpers and plan objects, but
  contraction route ids, dot-product layouts, and special cases must not
  become standalone reduction authority.
- Specs require the current RVV authority chain:
  `tcrv.exec` envelope -> selected RVV variant -> typed low-level `tcrv_rvv`
  body -> RVV plugin legality / selected-body realization / route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> `ssh rvv`
  evidence for runtime/correctness claims.

## Requirements

1. Add one bounded standalone add-reduction selected-body route:
   signed i32 input, scalar i32 seed, scalar i32 output, SEW32, LMUL m1,
   runtime `n`/AVL, and the existing legal policy shape unless current code
   exposes a stricter policy convention.
2. The selected or pre-realized body must structurally carry input mem_window,
   seed runtime value or initial accumulator role, scalar output binding,
   runtime `n`/AVL, e32m1 typed vector config, reduction kind, accumulator
   layout, scalar result layout, memory form, and tail policy.
3. RVV selected-body realization must materialize legal typed RVV structure
   before route construction. The intended structure is:
   setvl/with_vl over runtime `n`, vector input loads, a vector reduction
   update for each dynamic-VL chunk, explicit accumulation across chunks, final
   scalar extraction/store to `out_scalar[0]`, and no semantic changes to
   `tcrv.exec` ABI/runtime roles.
4. RVV route planning must derive scalar/vector/result C types, ABI order,
   standalone reduction layout mirrors, header facts, target leaves, intrinsic
   leaves, and diagnostics from typed body/config/runtime facts.
5. Provider emission must select RVV intrinsic/backend leaves only after
   validation. Exact `__riscv_*` spellings may appear only as provider-owned
   leaves derived from the typed route, never as public route authority.
6. Common EmitC/export and target metadata remain neutral. They may carry
   provider-built payloads and mirrors; they must not infer reduction kind,
   accumulator layout, dtype, policy, ABI order, intrinsic choice, or route
   support.
7. Unsupported/malformed cases fail closed with targeted diagnostics for
   missing seed/output role, scalar/vector dtype mismatch, invalid accumulator
   layout, invalid scalar result layout, missing `n`/AVL, invalid policy/config,
   stale route-id authority, incomplete typed body structure, and attempts to
   use existing chunk-wise reduce layout as standalone scalar-output authority.
8. Generated-bundle tooling may drive, inspect, compile, and run evidence only.
   It must not implement compiler core, route planning, dialect behavior, or
   emission semantics.

## Acceptance Criteria

- [ ] PRD, implement/check context, and task metadata describe a bounded
      standalone scalar-output add-reduction task and distinguish it from the
      existing chunk-wise `reduce_add`.
- [ ] A selected/pre-realized body structurally carries input buffer, scalar
      seed, scalar output, runtime `n`/AVL, e32m1 config, reduction add kind,
      standalone accumulator layout, scalar result layout, and tail policy.
- [ ] RVV selected-body realization materializes the standalone reduction into
      typed `tcrv_rvv` structure before route planning.
- [ ] RVV route planning/provider derive standalone reduction route facts,
      C types, ABI order, header facts, intrinsic leaves, and metadata mirrors
      from typed body/config/runtime facts.
- [ ] Positive FileCheck or equivalent evidence covers selected-body
      realization, route-plan/provider mirrors, generated header/artifact, and
      scalar-output materialization.
- [ ] Negative fail-closed tests cover missing seed/output role, scalar/vector
      dtype mismatch, invalid accumulator layout, invalid scalar result layout,
      missing runtime `n`/AVL, invalid policy/config, stale route-id authority,
      incomplete typed body structure, and stale chunk-wise reduce authority.
- [ ] Generated-bundle dry-run passes for the standalone reduction at counts
      `7,16,23` with at least two seed values and mixed positive/negative
      input data.
- [ ] Real `ssh rvv` generated-bundle run passes for the same counts/seeds,
      proving runtime `n`, seed use, multi-VL behavior where applicable,
      scalar output correctness, and output/tail sentinel preservation.
- [ ] Active-authority scan confirms no positive `RVVI32M1`, `rvv-i32m1`,
      finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      descriptor/direct-C/source-export, source-front-door, public
      exact-intrinsic route authority, or common/export RVV semantic authority
      is introduced.
- [ ] Focused build/lit/C++/script checks, `check-tianchenrv`,
      `git diff --check`, task validation, archive/finish, clean git status,
      and one coherent commit are completed if the task finishes.

## Non-Goals

- No broad reduction framework or matrix.
- No dtype/LMUL clone batch, min/max/and/or/floating-point reductions,
  segmented reductions, contraction/matmul work, high-level frontend lowering,
  Linalg/Vector/StableHLO source path, global autotuning, reports/dashboards,
  or performance claim.
- No use of contraction route ids, dot-product special cases, or chunk-wise
  reduce layout as standalone reduction authority.
- No revival of legacy i32 route authority. The e32m1 instance is allowed only
  as an ordinary bounded instance of the generic typed RVV surface.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, route planning, or emission.

## Validation Plan

1. Validate task context:
   `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-21-stage2-rvv-standalone-reduction-executable-slice`
2. Inspect current RVV dialect/config/runtime ABI, selected-body realization,
   route planning/provider, construction, target support, script harness, and
   focused tests.
3. Build focused targets expected to include:
   `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-dialect-test`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
4. Run focused C++ tests for touched RVV dialect/plugin/construction/export
   behavior.
5. Run focused lit/FileCheck tests for standalone-reduction selected-body
   realization, route plan/provider mirrors, target artifacts, EmitC
   materialization, and negative fail-closed cases.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
7. Run generated-bundle dry-run for standalone reduction counts `7,16,23` and
   at least two seed values.
8. Run real `ssh rvv` generated-bundle correctness for the same counts/seeds.
9. Run active-authority scans over active RVV include/lib/script/test paths.
10. Run `git diff --check`.
11. Run `cmake --build build --target check-tianchenrv -j2`.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-scalar-broadcast-elementwise-family-consolidation/prd.md`
- `.trellis/tasks/archive/2026-05/05-21-stage2-rvv-contraction-selected-body-family/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-selected-body-realization-extraction/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-reduce-add-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-reduce-add-production-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-05-19-stage2-generic-rvv-reduction-accumulation-route/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-generic-rvv-reduction-executable-closure/prd.md`

Relevant journal entries read:

- `.trellis/workspace/codex/journal-12.md` scalar-broadcast family closeout.
- `.trellis/workspace/codex/journal-12.md` contraction selected-body family closeout.
- `.trellis/workspace/codex/journal-11.md` reduce-add production/executable closeouts.

Initial code surface to inspect:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `include/TianChenRV/Support/RuntimeABI.h`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `lib/Support/RuntimeABIContract.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`

## Definition Of Done

- One bounded standalone scalar-output add-reduction route is implemented on
  the typed RVV selected-body/provider path.
- Route-supported evidence and executable `ssh rvv` evidence are current to
  this task.
- No legacy/source/descriptor/common-export authority is reintroduced.
- The Trellis task is truthful, finished, archived, and committed when
  complete; otherwise it remains open with an exact continuation point.
