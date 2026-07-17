# Stage2 RVV runtime scalar broadcast executable slice

## Goal

Validate and, only if needed, repair one bounded Stage2 RVV runtime
scalar-to-vector broadcast executable slice on the current HEAD. The concrete
path is signed i32 / SEW32 / LMUL m1:

```text
out[i] = in[i] + bias
```

`bias` must be an explicit scalar `tcrv.exec` / `tcrv_rvv.runtime_abi_value`
runtime parameter imported into the typed RVV body and consumed by generic
`tcrv_rvv.splat` plus `tcrv_rvv.binary {kind = "add"}` before storing to
`out`.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV runtime scalar broadcast executable slice`.
- Module owner: RVV plugin-owned runtime scalar-to-vector dataflow for one
  bounded i32 SEW32 LMUL m1 scalar-broadcast plus vector add executable slice.
- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `77f9f098 rvv: add segment2 interleave executable slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.

## Current Repository Facts

- Current HEAD already contains commit `a4cac384 rvv: add scalar broadcast
  executable path`, which added `rhs-scalar-value`, generic `tcrv_rvv.splat`,
  `memory_form = "rhs-scalar-broadcast"`, route planning/provider support,
  target fixture coverage, generated-bundle dry-run tooling, and `ssh rvv`
  evidence for `scalar_broadcast_add`.
- The archived task
  `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-vector-scalar-broadcast-executable-path`
  records the same executable module class as completed, with `ssh rvv`
  correctness for counts `7,16,23`.
- The current Hermes brief is therefore stale as an implementation selector
  unless current HEAD validation reveals a regression or missing evidence after
  the later memory movement commits.
- Stage2 work after `a4cac384` added strided, indexed, masked, computed-mask,
  segment2 deinterleave, and segment2 interleave routes. This task must check
  that those later changes did not regress the scalar runtime parameter path.
- Stage1/Stage2 guardrails remain active: no positive `RVVI32M1*`,
  `rvv-i32m1-*`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-seed, descriptor/direct-C/source-export, route-id
  authority, artifact-name authority, or common/export RVV semantic authority.

## Requirements

1. Keep the task bounded to one signed i32 / SEW32 / LMUL m1 runtime scalar
   broadcast add path with `lhs`, `rhs_scalar`, `out`, and runtime `n/AVL`
   ABI/runtime roles.
2. The selected/pre-realized body or explicit typed RVV body must structurally
   carry input mem_window/runtime ABI role, scalar runtime parameter role,
   output mem_window/runtime ABI role, operation kind, scalar/vector dtype and
   config, tail/mask policy, runtime `n/AVL`, and ABI order.
3. RVV selected-body realization must materialize or preserve only legal
   generic typed structure: `setvl`, `with_vl`, unit-stride load,
   `tcrv_rvv.splat`, `tcrv_rvv.binary {kind = "add"}`, and store.
4. RVVEmitCRoutePlanning must derive ABI order `lhs,rhs_scalar,out,n`, scalar C
   parameter type, vector C types, splat/binary/store leaves, artifact/header
   mirrors, and targeted diagnostics from typed body/config/runtime facts.
5. Provider/materializer must consume the provider-built plan. Common
   materialization must not infer scalar broadcast semantics, dtype, SEW, LMUL,
   policy, intrinsic choice, or ABI meaning.
6. Missing scalar runtime binding, wrong scalar role/order, mismatched
   scalar/vector dtype or config, unsupported operation kind, missing
   `n/AVL`, stale route-id authority, and incomplete typed body structure must
   fail closed with targeted diagnostics.
7. Generated-bundle evidence must use a non-zero/non-default scalar value so
   expected output proves active lanes are scalar-driven. If executable
   correctness is claimed in this round, it must use real `ssh rvv` evidence.

## Acceptance Criteria

- [x] PRD, implement/check context, and task metadata describe this bounded
      runtime scalar broadcast slice and record the current-HEAD stale-brief
      finding truthfully.
- [x] Current HEAD still has the generic typed `tcrv_rvv.splat` surface whose
      scalar operand comes from explicit runtime ABI facts, not parameter
      names, route ids, artifact names, helper names, C strings, descriptors,
      or scripts.
- [x] Current HEAD still realizes the scalar pre-realized body into explicit
      `setvl/with_vl/load/splat/binary/store` structure.
- [x] Current HEAD route planning/provider still recognizes
      `scalar_broadcast_add` and emits provider-derived runtime ABI parameters,
      intrinsic leaves, route metadata, and header prototype.
- [x] Focused positive tests pass for dialect/dataflow, selected-body
      realization, route materialization, target artifact generation, and
      generated-bundle dry-run for `scalar_broadcast_add`.
- [x] Focused negative tests still fail closed for unsupported scalar
      operation shape and mixed RHS broadcast/scalar splat authority.
- [x] Generated-bundle dry-run passes for counts `7,16,23`.
- [x] If fresh executable correctness is claimed, real `ssh rvv` passes for
      counts `7,16,23` with non-default `rhs_scalar` and expected-output checks
      proving scalar-driven active-lane updates.
- [x] Active-authority scan confirms no positive legacy/source/descriptor or
      common/export RVV semantic authority was introduced by this task.
- [x] `git diff --check`, Trellis task validation, task status update/archive,
      and final worktree/commit state are truthful.

## Non-Goals

- No generic scalar ABI framework, all-op arithmetic matrix, dtype/LMUL clone
  batch, floating-point support, high-level Linalg/Vector/StableHLO frontend
  lowering, source-front-door positive route, one-intrinsic wrapper dialect,
  broad performance claim, dashboard, report-only inventory, or helper-only
  refactor.
- No new positive `RVVI32M1*`, `rvv-i32m1-*`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-seed, descriptor/direct-C, or
  common/export RVV semantic authority.
- No production code change if current HEAD validation proves the module
  behavior already complete and still passing.

## Validation Plan

1. Validate and start the Trellis task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused lit/FileCheck fixtures for scalar broadcast positive target,
   conversion/materialization, dialect/dataflow, script dry-run, and negative
   fail-closed behavior.
4. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
5. Run generated-bundle dry-run for `scalar_broadcast_add` at counts
   `7,16,23`.
6. Run real `ssh rvv` correctness for `scalar_broadcast_add` counts `7,16,23`
   if this round refreshes executable evidence.
7. Run active-authority scans over active RVV include/lib/script/test paths.
8. Run `git diff --check`.
9. Run broader `check-tianchenrv` only if validation or edits touch shared
   compiler behavior enough to justify it.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-vector-scalar-broadcast-executable-path/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-vector-scalar-broadcast-executable-path/implement.jsonl`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-vector-scalar-broadcast-executable-path/check.jsonl`
- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-segment2-interleave-memory-executable-slice/prd.md`
- `.trellis/workspace/codex/journal-11.md`

Initial current-code surface inspected:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Support/RuntimeABI.h`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir`
- `test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-add-materialization.mlir`
- `test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-negative.mlir`
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-scalar-broadcast-add-dry-run.test`

## Current-Head Validation Results

Fresh validation on HEAD `77f9f098` showed no production-code repair was
needed. The current Hermes brief was stale as an implementation selector
because the bounded runtime scalar broadcast executable path already exists in
commit `a4cac384`, and it remains valid after later Stage2 memory movement
commits.

Checks run in this round:

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-20-stage2-rvv-runtime-scalar-broadcast`
- `python3 ./.trellis/scripts/task.py start .trellis/tasks/05-20-stage2-rvv-runtime-scalar-broadcast`
- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- `build/bin/tianchenrv-rvv-dialect-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-construction-protocol-common-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- Focused FileCheck pipelines for
  `test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir`
  with `REALIZED`, `PLAN`, and `HEADER` prefixes.
- Focused FileCheck pipeline for
  `test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-add-materialization.mlir`.
- Focused negative pipeline for
  `test/Conversion/EmitC/rvv-generic-stage2-scalar-broadcast-negative.mlir`.
- `build/bin/tcrv-opt test/Dialect/RVV/generic-stage2-dataflow.mlir --split-input-file --verify-diagnostics`
  with FileCheck.
- `build/bin/tcrv-opt test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir --split-input-file --verify-diagnostics --tcrv-materialize-selected-lowering-boundaries`
- Generated-bundle dry-run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id current-head-scalar-broadcast-dryrun --overwrite --op-kind scalar_broadcast_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- Real RVV evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id current-head-scalar-broadcast-ssh --overwrite --op-kind scalar_broadcast_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --connect-timeout 8 --timeout 120`
- `git diff --check`
- Diff authority scan excluding this task directory.

Fresh `ssh rvv` output:

```text
rvv_generated_bundle_abi_e2e: success
[scalar_broadcast_add] scalar_broadcast_add case n=7 ok rhs_scalar=-37
scalar_broadcast_add case n=16 ok rhs_scalar=-37
scalar_broadcast_add case n=23 ok rhs_scalar=-37
tcrv_rvv_generated_bundle_abi_scalar_broadcast_add_ok counts=7,16,23
PASS op=scalar_broadcast_add counts=7,16,23
```

Evidence artifacts:

- Dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/current-head-scalar-broadcast-dryrun`
- Real RVV:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/current-head-scalar-broadcast-ssh`

Self-repair performed:

- Initial manual FileCheck commands used nonexistent `build/bin/FileCheck`;
  reran with `/usr/lib/llvm-20/bin/FileCheck`.
- Initial dry-run used `llvm-readobj` without a PATH-resolved executable;
  reran with `/usr/lib/llvm-20/bin/llvm-readobj`.

Authority scan result:

- No production/source code was changed in this task, so no positive legacy or
  source/descriptor authority was introduced by the diff.
- Broad active scan still finds pre-existing legacy i32 parser/fail-closed
  fixtures and guardrail text. Those were not introduced by this task and were
  not used as route, runtime, artifact, or correctness evidence.

## Definition Of Done

- Current HEAD is proven to retain the bounded runtime scalar broadcast
  selected-body -> typed `tcrv_rvv.splat` -> RVV route planning/provider ->
  generated artifact -> optional fresh `ssh rvv` evidence chain, or production
  code is repaired until that is true.
- The final report distinguishes previous evidence from fresh evidence in this
  round.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
