# Bounded MLIR-to-RVV selected-boundary lowering seed

## Goal

Add one bounded RVV-owned C++/MLIR materialization seed that starts from an
explicitly marked minimal MLIR vector/arithmetic i32 add input and produces the
already proven selected RVV i32m1 execution boundary:

```text
bounded MLIR func/vector/scf/arith i32 add seed
  -> tcrv.exec.kernel selected variant shell
  -> explicit target-export runtime ABI operands lhs/rhs/out/n
  -> tcrv_rvv.setvl + tcrv_rvv.with_vl
  -> tcrv_rvv.i32_load / tcrv_rvv.i32_add / tcrv_rvv.i32_store
  -> existing RVV construction/EmitC/target artifact route
```

This task seeds one coherent source-to-selected-boundary path. It must not
claim generic RVV lowering or restore the deleted core linalg/vector
source-to-exec pass family.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting worktree was clean at HEAD `8822cf0`.
- No `.trellis/.current-task` existed at session start; this task was created
  from the supplied Direction Brief.
- The previous task proved the current explicit RVV i32m1 add/sub/mul selected
  route reaches generated object/header artifacts and passes `ssh rvv`.
- `.trellis/spec/index.md` requires compiler implementation in
  C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck, keeps computation out of
  `tcrv.exec`, and treats descriptor/direct-C paths as invalid architecture.
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md` says the old
  core RVV source frontend slice is deleted. Any rebuild must be
  plugin/interface-owned and must not inspect source arithmetic from common
  core transforms.
- The bounded RVV executable route already consumes explicit
  `tcrv_rvv.runtime_abi_value`, `tcrv_rvv.setvl`, selected `tcrv_rvv.with_vl`,
  and RVV dataflow ops through `RVVConstructionProtocol` and
  `RVVEmitCRouteProvider`.

## Requirements

- Add a plugin-owned RVV materialization pass, not a common/core RVV semantic
  branch.
- The accepted source shape is intentionally narrow:
  - one `func.func` marked with an RVV seed attribute;
  - arguments are `lhs`, `rhs`, `out`, and `n` in the ABI order modeled as
    `memref<?xi32>`, `memref<?xi32>`, `memref<?xi32>`, and `index`;
  - the body contains one bounded `scf.for` from zero to `n` with a fixed
    vector i32 chunk, two `vector.load` operations, one `arith.addi`, one
    `vector.store`, and no extra compute.
- The pass must materialize a `tcrv.exec.kernel` with one selected
  `origin = "rvv-plugin"` variant requiring `@rvv`.
- The materialized variant must contain explicit RVV runtime ABI bindings for
  `lhs`, `rhs`, `out`, and `n`, then the selected i32m1 `setvl` / `with_vl`
  boundary and RVV add dataflow body.
- The selected path marker must target the materialized RVV variant so the
  existing selected-boundary, emission-plan, EmitC materialization, and target
  artifact path can consume it.
- Unsupported dtype, rank, vector shape, source operation shape, missing
  runtime ABI operand, stale pre-existing `tcrv.exec`/`tcrv_rvv` selected
  boundary metadata, missing `with_vl`, and unselected variant residue must
  fail closed.
- Existing explicit RVV construction-backed add/sub/mul materialization and
  target artifact tests must continue to pass.

## Acceptance Criteria

- [x] Focused build succeeds for the new RVV seed pass surface plus
      `TianChenRVRVVPlugin`, `TianChenRVRVVConstructionProtocol`,
      `TianChenRVRVVEmitCRouteProvider`, `TianChenRVTransforms`, `tcrv-opt`,
      and `tcrv-translate` as touched.
- [x] Positive lit/FileCheck input lowers from the bounded MLIR vector i32 add
      seed to a selected `tcrv.exec.variant` containing
      `tcrv_rvv.with_vl`, explicit `lhs`/`rhs`/`out`/`n` runtime ABI bindings,
      and `tcrv_rvv.i32_add`.
- [x] A pipeline test proves the materialized selected boundary is accepted by
      existing RVV selected-boundary and emission/EmitC route consumption.
- [x] Negative lit/FileCheck coverage proves fail-closed behavior for
      unsupported dtype/rank/shape, missing runtime ABI operands, stale
      selected-boundary metadata, missing source `with_vl` equivalent, and
      unselected pre-existing variant residue.
- [x] Focused existing RVV construction-backed add/sub/mul EmitC and target
      artifact tests still pass.
- [x] `git diff --check` passes.
- [x] No descriptor/direct-C/Python compiler-core path, compatibility wrapper,
      generic RVV lowering claim, or common/core RVV semantic branch is added.
- [x] If the new source seed reaches generated artifacts during the round, run
      the existing RVV artifact route and `ssh rvv` correctness harness for the
      seed. If not, state explicitly that the verified boundary stops at
      selected-boundary/EmitC-route consumption.

## Validation Results

- Focused build:
  `cmake --build build --target TianChenRVRVVConstructionProtocol
  TianChenRVRVVEmitCRouteProvider TianChenRVRVVPlugin TianChenRVTransforms
  tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test
  tianchenrv-construction-protocol-common-test
  tianchenrv-rvv-dialect-test -j2`.
- Focused C++ tests:
  `./build/bin/tianchenrv-rvv-extension-plugin-test`,
  `./build/bin/tianchenrv-target-artifact-export-test`,
  `./build/bin/tianchenrv-construction-protocol-common-test`,
  and `./build/bin/tianchenrv-rvv-dialect-test`.
- Focused lit:
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'rvv-i32m1-selected-boundary-seed|rvv-first-slice-materialization|Target/RVV/i32m1|rvv-with-vl-selected-boundary'`
  from `build/test`, 19/19 passed.
- Full practical check:
  `cmake --build build --target check-tianchenrv -j2`, 104/104 passed.
- `git diff --check` passed.
- Seed artifact/evidence directory:
  `artifacts/tmp/rvv_selected_boundary_seed/20260516T132508Z`.
- Generated seed object/header:
  `seed.o` and `seed.h` from `tcrv-opt
  --tcrv-rvv-materialize-i32m1-selected-boundary-seed
  --tcrv-materialize-emission-plans` followed by
  `tcrv-translate --tcrv-export-target-artifact` and
  `--tcrv-export-target-header-artifact`. `file` reported a RISC-V ELF
  relocatable object.
- `ssh rvv` link/run evidence:
  `artifacts/tmp/rvv_selected_boundary_seed/20260516T132508Z/ssh_rvv_link_run.log`.
  Remote run printed
  `tcrv_rvv_i32m1_selected_boundary_seed status=PASS n=4 add=[12,6,16,12]`
  and exited with status 0.
- Changed-surface check: tracked common transform and core exec dialect diffs
  are empty. The implementation lives in RVV plugin-owned code plus
  `tcrv-opt` registration, tests, and the durable variant-pipeline spec.

## Definition Of Done

- Task context files are curated and truthful.
- The implementation is bounded to the RVV-owned seed path and focused tests.
- The task is finished/archived when complete.
- One coherent commit is created when complete. If the task cannot finish,
  leave it open and name the exact next continuation point.

## Out Of Scope

- Generic MLIR vector lowering, scalable-vector lowering, linalg/stablehlo/tosa
  frontend ownership, new RVV dtype/SEW/LMUL/op families, sub/mul source
  expansion unless it is mechanically forced by the same owner, TensorExt/IME/
  offload work, new target artifact routes, performance tuning, descriptor or
  binary-family registries, direct C semantic exporters, source-export
  skeletons, Python compiler-core logic, GCC-default routes, compatibility
  wrappers, state-machine/checkpoint ledgers, docs-only work, and broad smoke
  matrices.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  and shared guides under `.trellis/spec/guides/`.
- Prior task read:
  `.trellis/tasks/archive/2026-05/05-16-rvv-construction-backed-hardware-artifact-proof/prd.md`.
- Relevant journal read:
  `.trellis/workspace/codex/journal-8.md` session 93.
- Primary code surfaces inspected:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `include/TianChenRV/Transforms/`,
  `lib/Transforms/`,
  `include/TianChenRV/Conversion/`,
  `lib/Conversion/`,
  `tools/tcrv-opt/tcrv-opt.cpp`.
