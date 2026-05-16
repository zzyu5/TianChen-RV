# Finite RVV i64 Family Erasure

## Goal

Delete the unowned finite RVV i64m1 dataflow/profile slice from active
TianChen-RV source, specs, and tests. This is a Wrong Logic Deletion Campaign
round: remove protected i64m1 authority without adding a replacement dtype
family, generalized SEW/LMUL model, descriptor path, direct-C exporter, EmitC
route, or compatibility layer.

## What I Already Know

- The repository root is `/home/kingdom/phdworks/TianchenRV`.
- The starting worktree is clean at commit `79e4ac4`.
- `.trellis/.current-task` was missing, so this task was created from the
  Hermes/user direction brief.
- Current RVV first-slice ownership is i32 m1/m2 plus `vl`, policy, `setvl`,
  and `with_vl` control-plane surfaces.
- Active residue includes `!tcrv_rvv.i64m1`,
  `tcrv_rvv.i64_load/add/sub/mul/store`, i64 verifier helpers,
  `test/Dialect/RVV/dataflow.mlir` i64 coverage, `setvl` SEW64 acceptance, and
  spec/profile wording around `rvv.i64_m1.*`.

## Requirements

- Remove the RVV i64m1 type from active TableGen source.
- Remove the RVV i64 load/add/sub/mul/store ops from active TableGen source.
- Remove C++ verifier/helper code that only supports the finite i64m1 slice.
- Narrow `setvl` / `with_vl` accepted first-slice config to SEW32 with LMUL
  m1 or m2.
- Delete active lit coverage that protects i64m1 dataflow as a current
  compiler surface.
- Rewrite affected diagnostics/tests so they preserve the owned i32 m1/m2
  control/dataflow surface.
- Remove active `.trellis/spec` wording that protects `rvv.i64_m1.*`,
  SEW64/i64m1 first-slice legality, or i64 dataflow as current architecture.

## Acceptance Criteria

- [ ] Active `include`, `lib`, `test`, and `.trellis/spec` surfaces contain no
      `I64M1`, `i64m1`, `tcrv_rvv.i64_`, `rvv.i64_m1`, `sew64`, `SEW64`, or
      i64 dataflow/profile authority outside this task record.
- [ ] No new dtype family, wrapper, descriptor, direct-C exporter, EmitC route,
      generalized RVV lowering/modeling, or compatibility path is added.
- [ ] Focused RVV dialect/parser/verifier lit tests pass.
- [ ] `check-tianchenrv` is run if the build remains coherent.
- [ ] `git diff --check` passes.
- [ ] If deletion exposes a future generalized RVV config/lowering gap, it is
      reported as rebuild-phase work rather than patched by restoring i64.

## Out Of Scope

- General RVV SEW/LMUL modeling.
- New MLIR vector lowering.
- New EmitC materialization.
- New target artifact routes.
- New `ssh rvv` evidence.
- New i64 replacement tests or compatibility aliases.
- Deleting owned i32 m1/m2 dataflow or runtime AVL/VL control surfaces unless a
  line only exists to preserve i64 residue.

## Technical Notes

- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/capability-model/capability-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Relevant code/test entry points:
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
  - `lib/Dialect/RVV/IR/RVVDialect.cpp`
  - `test/Dialect/RVV/dataflow.mlir`
  - `test/Dialect/RVV/setvl.mlir`
- Initial residue scan:
  - `rg -n "I64M1|i64m1|tcrv_rvv\\.i64_|rvv\\.i64_m1|sew64|SEW64|i64 dataflow|i64 family|bounded i64" include lib test .trellis/spec`
  - `rg -n "sew\\s*=\\s*64|SEW 64|sew_bits\\s*=\\s*64|i64_m1|i64m1|I64M1|tcrv_rvv\\.i64_|SEW64|sew64" include lib test .trellis/spec`
