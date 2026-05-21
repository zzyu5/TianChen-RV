# Stage2 RVV runtime scalar broadcast add executable slice

## Goal

Validate the current HEAD for the bounded Stage2 RVV executable slice:

```text
out_i32[i] = in_i32[i] + scalar_i32
```

The original 2026-05-21 brief requested implementation from selected
`tcrv.exec` RVV boundary through typed `tcrv_rvv` body, RVV route
planning/provider emission, generated artifact, and real `ssh rvv` evidence.
Current repository inspection showed that the requested production behavior had
already landed earlier in commit `a4cac384 rvv: add scalar broadcast executable
path`, then was revalidated in commit `aafa8210 trellis: record scalar broadcast
current-head validation`. This task therefore became a stale-brief
current-head validation and Trellis closeout round, not a second implementation
path.

## Current Repository Facts

- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Current branch: `main`.
- Current HEAD at validation time: `46a7d16f rvv: add computed-mask select executable slice`.
- Existing completed tasks:
  - `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-vector-scalar-broadcast-executable-path`
  - `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-runtime-scalar-broadcast`
- Existing production scalar-broadcast path includes:
  - explicit runtime ABI role `rhs-scalar-value`;
  - generic typed `tcrv_rvv.splat`;
  - pre-realized `memory_form = "rhs-scalar-broadcast"`;
  - selected-body realization to `setvl/with_vl/load/splat/binary/store`;
  - RVV route planning/provider operation `scalar_broadcast_add`;
  - generated target header/artifact fixture;
  - generated-bundle dry-run and real `ssh rvv` support for counts `7,16,23`.
- No compiler source change was needed in this round. Adding a duplicate route
  would have violated the bounded Stage2 shape and risked treating stale task
  selection as architecture authority.

## Requirements

1. Keep the validation bounded to signed i32 / SEW32 / LMUL m1 runtime scalar
   broadcast add with ABI order `lhs,rhs_scalar,out,n`.
2. Confirm that the selected/pre-realized path structurally carries input
   mem-window ABI role, scalar runtime parameter ABI role, output mem-window ABI
   role, runtime `n`/AVL, typed vector config, scalar element type, add
   operation kind, and policy.
3. Confirm that RVV selected-body realization materializes explicit
   `setvl/with_vl/load/splat/binary/store` structure before route construction.
4. Confirm that RVV route planning/provider derive the scalar-broadcast route
   from typed body/config/runtime facts and fail closed for unsupported scalar
   shapes.
5. Confirm that common EmitC/export remains neutral and consumes provider-built
   route payloads.
6. Refresh generated-bundle dry-run and real `ssh rvv` correctness evidence for
   counts `7`, `16`, and `23`.
7. Record the stale-brief finding truthfully and archive the task without
   modifying production compiler code.

## Acceptance Criteria

- [x] Positive selected-body/FileCheck coverage shows realization for the
      runtime scalar broadcast add path.
- [x] Positive route-planning/provider coverage shows support derived from
      typed body/config/runtime facts.
- [x] Positive generated header/artifact evidence exists for the runtime scalar
      broadcast add path.
- [x] Negative fail-closed coverage exists for unsupported scalar operation
      shape and mixed RHS buffer broadcast/scalar splat authority; adjacent
      selected-body negative coverage continues to validate missing/incomplete
      runtime/body structure.
- [x] Generated-bundle dry-run passes for counts `7`, `16`, and `23`.
- [x] Real `ssh rvv` execution passes for counts `7`, `16`, and `23`, proving
      runtime scalar use and tail sentinel preservation.
- [x] Active-authority scan over this task's diff introduces no positive
      `RVVI32M1`, `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, source-front-door/source-seed,
      descriptor/direct-C/source-export, route-id authority, artifact-name
      authority, or common/export RVV semantic authority.
- [x] Focused build/tests, `check-tianchenrv`, `git diff --check`, task
      validation, and final status checks pass.
- [x] No production code was changed because current HEAD already satisfied the
      module behavior.

## Non-Goals

- No broad broadcast framework.
- No dtype or LMUL clone batch.
- No masked broadcast, strided broadcast, compare/select expansion,
  contraction, matmul, Linalg, or frontend lowering.
- No source-front-door positive route.
- No one-intrinsic wrapper dialect.
- No dashboard/report-only/helper-only work as the claimed compiler
  achievement.
- No revival of legacy i32 route authority. The existing i32 scalar add is an
  ordinary bounded instance of the generic typed RVV surface.
- No RVV scalar broadcast/add semantics in common EmitC/export, target
  metadata, artifact names, route ids, descriptors, ABI strings, tests, or
  exact intrinsic spellings as authority.

## Validation Performed

- `pwd`
- `git status --short`
- `git log --oneline -8`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-21-stage2-rvv-runtime-scalar-broadcast-add-executable-slice`
- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- `build/bin/tianchenrv-rvv-dialect-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-construction-protocol-common-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir --tcrv-materialize-selected-lowering-boundaries | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir --check-prefix=REALIZED`
- `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir --check-prefix=PLAN`
- `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-header-artifact | /usr/lib/llvm-20/bin/FileCheck test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir --check-prefix=HEADER`
- `build/bin/tcrv-opt test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-add-materialization.mlir --tcrv-materialize-emitc-lowerable-routes | /usr/lib/llvm-20/bin/FileCheck test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-add-materialization.mlir`
- `build/bin/tcrv-opt test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-negative.mlir --split-input-file --tcrv-materialize-emitc-lowerable-routes` with the file's negative FileCheck expectation.
- `build/bin/tcrv-opt test/Dialect/RVV/generic-stage2-dataflow.mlir --split-input-file --verify-diagnostics | /usr/lib/llvm-20/bin/FileCheck test/Dialect/RVV/generic-stage2-dataflow.mlir`
- `build/bin/tcrv-opt test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir --split-input-file --verify-diagnostics --tcrv-materialize-selected-lowering-boundaries`
- Generated-bundle dry-run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id current-head-scalar-broadcast-20260521-dryrun --overwrite --op-kind scalar_broadcast_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- Real RVV evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id current-head-scalar-broadcast-20260521-ssh --overwrite --op-kind scalar_broadcast_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --connect-timeout 8 --timeout 120`
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2`

Fresh `ssh rvv` output:

```text
rvv_generated_bundle_abi_e2e: success
artifact_dir: artifacts/tmp/rvv_generated_bundle_abi_e2e/current-head-scalar-broadcast-20260521-ssh
[scalar_broadcast_add] scalar_broadcast_add case n=7 ok rhs_scalar=-37
scalar_broadcast_add case n=16 ok rhs_scalar=-37
scalar_broadcast_add case n=23 ok rhs_scalar=-37
tcrv_rvv_generated_bundle_abi_scalar_broadcast_add_ok counts=7,16,23
PASS op=scalar_broadcast_add counts=7,16,23
```

Evidence artifacts:

- Dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/current-head-scalar-broadcast-20260521-dryrun`
- Real RVV:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/current-head-scalar-broadcast-20260521-ssh`

Self-repair performed:

- Initial manual conversion/materialization checks used the wrong pass entry
  (`--tcrv-materialize-emission-plans`) for conversion fixtures. They failed on
  selected-boundary validation before exercising the intended route. The checks
  were rerun with the fixtures' actual RUN-style pass,
  `--tcrv-materialize-emitc-lowerable-routes`, and passed.
- Initial manual negative run used `--verify-diagnostics` for a `not ... |
  FileCheck` fixture. It was rerun using the fixture's negative FileCheck style
  and passed.

## Final Status

Completed. The current task was a duplicate stale implementation request for a
module already implemented and validated in prior commits. Current HEAD
retains the requested executable chain, and this round refreshed focused tests,
generated-bundle dry-run, real `ssh rvv` correctness, full `check-tianchenrv`,
and Trellis closeout evidence.

## Spec Update Judgment

No `.trellis/spec/**` update was needed in this round. The round did not add or
change a compiler contract, command/API signature, cross-layer request/response
contract, validation rule, or implementation convention. It only confirmed that
the existing RVV-first scalar-broadcast contract remains valid on current HEAD
and that the new task brief was stale.
