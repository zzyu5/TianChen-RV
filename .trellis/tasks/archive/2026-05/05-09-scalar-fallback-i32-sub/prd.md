# scalar fallback i32 sub plugin support

## Goal

Extend the existing plugin-local scalar fallback i32 add slice to include the
bounded i32 subtract family member already exposed by the linalg add/sub
frontend and RVV plugin path.

## What I Already Know

- The repo root is `/home/kingdom/phdworks/TianchenRV`.
- HEAD at task start is `1449917 feat(rvv): lower linalg i32 add sub frontend artifacts`.
- The worktree was clean before this task was created.
- Existing linalg frontend lowering preserves `tcrv_frontend_lowering =
  "i32-vadd"` or `"i32-vsub"` on the generated `tcrv.exec.kernel`.
- RVV already maps `"i32-vsub"` to `i32-vsub-microkernel.v1`, materializes
  `tcrv_rvv.i32_vsub_microkernel`, and exports RVV subtract intrinsic source.
- Scalar fallback currently proposes and exports only
  `i32-vadd-microkernel.v1` and explicitly declined frontend vsub.

## Requirements

- Preserve the existing scalar vadd descriptor, route ids, ABI identity,
  operation label, and emitted add source.
- Add a scalar plugin-local i32-vsub descriptor for frontend-lowered subtract.
- Materialize a scalar plugin-local subtract microkernel attachment for the
  selected scalar vsub fallback path.
- Emit scalar fallback subtract C source through a new vsub identity, not stale
  vadd route, ABI name, runtime glue role, operation label, or arithmetic.
- Keep `tcrv.exec` compute-free and avoid scalar/RVV semantic branches in
  generic selection, dispatch, emission, or target artifact routing.
- Keep scope bounded to i32 add/sub scalar fallback.
- Do not run `ssh rvv` or make RVV runtime/correctness/performance claims.

## Acceptance Criteria

- Scalar plugin proposal/legality coverage proves `"i32-vadd"` remains vadd and
  `"i32-vsub"` receives a scalar fallback vsub descriptor.
- A bounded linalg i32-vsub input can flow through frontend lowering and
  execution planning to a scalar fallback selected path with vsub descriptor,
  scalar vsub microkernel, supported emission plan, and target source export.
- Scalar vsub source export emits subtract semantics (`lhs - rhs`) and no RVV
  intrinsics or stale vadd arithmetic.
- Existing vadd tests still pass.
- `git diff --check`, CMake configure, and `check-tianchenrv` pass locally.

## Out Of Scope

- Generic scalar lowering.
- New `tcrv.exec` compute ops.
- i64/e64, masks, dynamic shape expansion, vectorized scalar loops, performance
  tuning, or new RVV arithmetic families.
- RVV runtime execution or hardware evidence collection.

## Technical Notes

- Relevant long-term specs: plugin integration, scalar fallback plugin, RVV
  plugin, emission/runtime contract, and MLIR testing contract.
- Expected implementation surface: scalar dialect op, scalar plugin descriptor
  proposal/materialization/emission plan, scalar target source exporter, and
  focused C++/lit/FileCheck tests.
