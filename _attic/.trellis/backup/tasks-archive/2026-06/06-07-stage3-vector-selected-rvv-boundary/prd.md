# Stage3 bounded MLIR Vector to selected RVV body materialization boundary

## Goal

Introduce one bounded production source-boundary workflow that accepts a small
MLIR Vector-like i32 add pattern and materializes it into a selected
`tcrv.exec` RVV variant containing a typed low-level `tcrv_rvv` body. The
materialized selected body must then flow through the existing RVV
plugin/provider route, common EmitC materialization, and target artifact export
without making source names, route ids, artifact names, or Common EmitC code
responsible for RVV semantics.

## What I Already Know

* Current HEAD is `1f90ad25` and the worktree was clean before the task was
  created.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-runtime-scalar-cmp-composite-executable-abi-boundary/`
  closed executable ABI evidence for explicit and pre-realized selected
  composite RVV bodies. This task starts at the upstream selected-boundary
  source edge instead of reworking that composite evidence.
* `.trellis/spec/index.md` keeps the RVV-first authority chain as
  `tcrv.exec envelope -> selected RVV variant -> typed low-level tcrv_rvv body
  -> RVV plugin-owned route -> common EmitC -> target artifact`.
* `.trellis/spec/extension-plugins/rvv-plugin.md` says dtype/config/operation
  authority must be structural in the typed `tcrv_rvv` body/config and
  provider-owned route facts, not in route ids, ABI strings, test names, exact
  intrinsic spellings, or Common EmitC.
* `.trellis/spec/lowering-runtime/emitc-route.md` says Common EmitC consumes a
  provider-built `TCRVEmitCLowerableRoute` and must not infer RVV dtype, SEW,
  LMUL, policy, operation kind, schedule, or ABI role semantics.
* Existing
  `test/Target/RVV/generic-selected-body-artifact-arithmetic.mlir` proves a
  hand-authored selected generic i32 add body reaches emission plans and header
  export.
* Existing `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp` only fail-closes the
  legacy RVV source-front-door pass. No positive bounded Vector-like source
  materializer is present at task start.

## Requirements

* Add one RVV plugin-owned source-front-door materializer for a bounded source
  pattern:
  `func.func` with source marker, three memory operands for
  `lhs/rhs/out`, one `%n : index`, two `vector.transfer_read` ops, one
  `arith.addi`, and one `vector.transfer_write`.
* The source marker may select the bounded materializer, but it must not carry
  compute semantics. Acceptance must be derived from the function signature and
  source operations: memory roles, vector element type, vector shape, add
  operation kind, output store, and runtime `%n`.
* Materialize the accepted source pattern into one `tcrv.exec.kernel` with an
  RVV capability, scalar fallback envelope, selected RVV dispatch case, and a
  typed generic `tcrv_rvv` body using `runtime_abi_value`, `setvl`,
  `with_vl`, `load`, `binary {kind = "add"}`, and `store`.
* Carry structural RVV facts in the selected body: source dtype `i32`, vector
  element type, SEW 32, LMUL m1, agnostic tail/mask policy, memory roles
  `lhs/rhs/out`, runtime `n` as AVL/VL source, selected variant symbol, source
  kernel name, and construction protocol.
* Register the positive materializer as an RVV plugin source-front-door pass
  while retaining the existing fail-closed legacy pass.
* Add positive tests proving the Vector-like source pattern materializes into
  selected `tcrv.exec` plus typed `tcrv_rvv` structure and reaches the existing
  provider/export/artifact evidence path.
* Add fail-closed tests for stale or unsupported source-boundary facts, such
  as missing runtime `%n`, unsupported dtype/vector type, stale pre-existing
  `tcrv.exec`/`tcrv_rvv` residue, or missing accepted source marker.

## Acceptance Criteria

* [x] A positive lit test shows the bounded Vector-like input materializes into
      selected `tcrv.exec` plus typed `tcrv_rvv` i32/m1 add body with runtime
      `n` consumed by `tcrv_rvv.setvl`.
* [x] The same or a companion lit test shows the existing RVV route/provider
      path consumes the materialized body and emits plan/header facts including
      `rvv_selected_body_operation = add`,
      `tcrv_rvv.binary`, runtime ABI order `lhs,rhs,out,n`, and
      `rvv-generic-binary-add-callable-c-abi.v1`.
* [x] A target artifact export or dry-run bundle/header test proves the
      selected route reaches existing artifact export without Common EmitC
      inventing RVV semantics.
* [x] A focused negative test fails closed for at least one stale/missing
      selected-boundary fact from the source side.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant lit tests for the new RVV source-front-door path pass.
* [x] `git diff --check` and `git diff --cached --check` pass.
* [x] A bounded old-authority scan over touched files and added diff lines
      shows no new positive legacy `RVVI32M1`, `rvv-i32m1`,
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor-driven compute,
      source-front-door route-id authority, or Common EmitC semantic branch.

## Out Of Scope

* No Linalg, StableHLO, tensor/tile IR, broad MLIR Vector lowering, or
  per-Linalg route authority.
* No dtype/LMUL clone batch; this task accepts only the bounded i32/m1 add
  source pattern.
* No new one-op-per-intrinsic wrapper dialect and no route-id/test-name/artifact
  semantic authority.
* No relocation of Stage 2 selected-body realization into a generic Stage 3
  mechanism.
* No runtime correctness claim unless a generated bundle is actually run on
  `ssh rvv`.

## Completion Notes

* Implemented
  `tcrv-rvv-materialize-vector-add-source-front-door` as an RVV plugin-owned
  bounded source-front-door materializer. The source marker is
  `tcrv_rvv.source_front_door = "bounded_vector_source"` so marker text does
  not encode dtype, LMUL, or operation authority; the pass derives acceptance
  from the source function signature and Vector/arith structure.
* The generated selected variant is `@rvv_vector_add` with typed
  `tcrv_rvv.runtime_abi_value`, `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`,
  `tcrv_rvv.load`, `tcrv_rvv.binary {kind = "add"}`, and `tcrv_rvv.store`.
  The RVV provider then emits existing `rvv-generic-binary-add` plan/header
  facts; Common EmitC/export remains a consumer of the provider route.
* Negative coverage fails closed for unsupported source dtype, missing runtime
  `%n`, and stale pre-existing selected-boundary residue.
* No `ssh rvv` runtime correctness claim is made in this task; evidence is
  bounded to materialization, provider/export plan/header, and target artifact
  smoke coverage.

## Technical Notes

* Direction source: Hermes/Codex worker direction brief supplied on
  2026-06-07.
* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
* Previous task context read:
  `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-runtime-scalar-cmp-composite-executable-abi-boundary/prd.md`.
* Primary implementation candidates:
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/InitTianChenRVDialects.cpp`,
  `lib/CMakeLists.txt`,
  `lib/Plugin/RVV/CMakeLists.txt`,
  and focused lit/plugin tests under `test/Transforms/SourceFrontDoor`,
  `test/Target/RVV`, and `test/Plugin/RVVExtensionPluginTest.cpp`.
