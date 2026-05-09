# frontend linalg i32 add/sub to RVV artifact path

## Goal

Make the bounded hand-written/test `linalg.generic` i32 elementwise frontend
slice cover both add and subtract, so an i32 subtract input can lower into
TianChen-RV `tcrv.exec`/RVV selected-boundary surfaces and then export through
the existing generic target source artifact route to RVV C intrinsic source.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* HEAD at takeover was `e8a039a feat(rvv): make vsub artifact metadata family-aware`.
* Existing `--tcrv-lower-linalg-i32-vadd-to-exec` creates a bounded
  `tcrv.exec.kernel` plus `mem_window`/`runtime_param` callable ABI boundary
  from explicitly marked i32 vadd `linalg.generic` wrappers.
* Existing RVV plugin/export paths already support the finite add/sub
  microkernel family descriptors and target-owned C intrinsic export.
* Existing scalar fallback only supports the finite i32 vadd microkernel.
* This round must not implement Python compiler internals, new `tcrv.exec`
  compute ops, broad frontend lowering, StableHLO/TOSA lowering, new arithmetic
  families, runtime correctness, or performance evidence.

## Requirements

* Recognize only the existing bounded linalg wrapper shape and i32 scalar body
  form, adding only `arith.subi` / `linalg.yield` as the subtract sibling of the
  existing `arith.addi` form.
* Preserve the public vadd pass route for compatibility while making its
  description and implementation cover the bounded add/sub family.
* Map `tcrv_frontend_lowering = "i32-vadd"` to the existing RVV
  `i32-vadd-microkernel.v1` descriptor and map `"i32-vsub"` to the existing RVV
  `i32-vsub-microkernel.v1` descriptor.
* Keep callable ABI layering unchanged: lhs/rhs/out are direct
  `tcrv.exec.mem_window` roles and runtime `n` is a direct
  `tcrv.exec.runtime_param` role.
* Do not let scalar fallback's existing vadd-only support silently become a
  subtract fallback. For vsub frontend kernels, scalar should decline rather
  than materialize stale vadd metadata.
* Reuse existing plugin proposal, legality, selected-boundary,
  emission-plan, and generic target artifact export routes.

## Acceptance Criteria

* A marked i32 vsub `linalg.generic` input lowers through the local planning
  pipeline to a selected RVV path with `i32-vsub-microkernel.v1`,
  `tcrv_rvv.i32_vsub_microkernel`, and `tcrv_rvv.i32_sub`.
* The same vsub pipeline reaches
  `tcrv-translate --tcrv-export-target-source-artifact` and emits
  `__riscv_vsub_vv_i32m1`, not `__riscv_vadd_vv_i32m1`.
* Existing vadd lowering still reaches the same pipeline and generic source
  artifact route and emits `__riscv_vadd_vv_i32m1`.
* Local checks pass: `git diff --check`, CMake configure, and
  `check-tianchenrv`.

## Out Of Scope

* No full generic linalg frontend, dynamic shape frontend, StableHLO/TOSA
  frontend, i64/e64, masks, new RVV policy families, new arithmetic families,
  runtime integration, or performance work.
* No fresh `ssh rvv` run unless runtime correctness is intentionally claimed.
  This task is expected to claim local compiler lowering/export behavior only.

## Technical Notes

* Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/extension-plugin-integration.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Implementation is expected to touch the bounded frontend lowering pass,
  RVV/scalar plugin proposal behavior, focused lit tests, and minimal
  README/spec wording for the durable add/sub frontend contract.
