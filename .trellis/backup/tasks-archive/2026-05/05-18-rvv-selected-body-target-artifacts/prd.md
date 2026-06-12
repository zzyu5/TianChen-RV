# RVV selected-body authority for target artifacts

## Goal

Replace the existing RVV materialized EmitC target artifact candidate authority
from bounded i32m1 route metadata to the selected typed `tcrv_rvv` body and the
RVV-owned EmitC route builder. The existing RVV add/sub/mul object/header/bundle
path should remain bounded and positive, but target artifact export must no
longer infer RVV compute from `rvv_emitc_lowerable_route`,
`rvv_arithmetic_op`, artifact names, route ids, or stale source/front-door
metadata alone.

## What I Already Know

- Current HEAD is `b4e3fc2` on `main`; worktree was clean at task start.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief.
- `.trellis/spec/index.md` keeps the primary compiler stack in
  C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck; Python is tooling only.
- `.trellis/spec/extension-plugins/rvv-plugin.md` says RVV Stage 1 replaces
  bounded i32m1/source/artifact/route metadata authority with selected
  vector-level typed `tcrv_rvv` body plus RVV legality/realization/route
  construction.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` allows
  target/export code to validate route metadata and ABI mirrors, but forbids it
  from synthesizing RVV compute from route ids, artifact names, descriptor
  residue, or old i32m1 helper names.
- `lib/Target/RVV/RVVTargetSupportBundle.cpp` currently derives the arithmetic
  op from candidate metadata (`rvv_emitc_lowerable_route` and
  `rvv_arithmetic_op`) before rebuilding the route, which keeps metadata as
  executable authority.
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` already has the RVV-owned
  `buildRVVI32M1ArithmeticEmitCLowerableRoute` path that collects and validates
  the selected typed `tcrv_rvv` body, then derives the route from the body.
- The brief-listed fixture
  `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir` is not present in
  current HEAD; current RVV source/materialized artifact coverage lives under
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door*.mlir` and
  `test/Target/RVV/vector-materialized-target-artifact-exporters.mlir`.

## Requirements

- RVV target artifact candidate validation must first resolve the selected
  variant, require a typed RVV body, and build the route through the RVV-owned
  route builder.
- Candidate route metadata may remain as provenance and consistency-check
  material only after the selected typed body has produced the authoritative
  route.
- Runtime ABI identity for add/sub/mul must be derived from the body-built
  route, not from candidate metadata.
- The existing positive RVV materialized EmitC object/header/bundle path must
  continue to export a RISC-V relocatable object and declaration-only header.
- Negative coverage must reject:
  - forged or missing route metadata without typed selected-body authority;
  - route/arithmetic metadata that disagrees with the selected body;
  - fallback-only RVV target candidates;
  - multiple selected candidates;
  - stale selected source/front-door or boundary metadata without the typed
    body required by the route builder.
- Common target/export code must stay target-neutral; RVV body recognition and
  route construction remain RVV plugin/target-owned.

## Acceptance Criteria

- [ ] `RVVTargetSupportBundle.cpp` no longer selects arithmetic operation from
  `rvv_emitc_lowerable_route` or `rvv_arithmetic_op` before body route
  construction.
- [ ] RVV construction metadata verification can check metadata against the
  body-derived lowerable route id.
- [ ] Focused C++ target/plugin tests cover body-derived acceptance and
  metadata/body mismatch rejection.
- [ ] Focused lit target artifact coverage proves stale body/metadata mismatch
  fails before object bytes are accepted.
- [ ] Targeted scans over changed RVV target/plugin/export/test surfaces show
  no remaining target artifact acceptance path that treats RVV route metadata as
  standalone authority.
- [ ] Focused build/tests pass; `check-tianchenrv` is run if practical.

## Out Of Scope

- Stage 2 RVV coverage expansion, dtype/LMUL clone batches, new high-level
  frontend lowering, scalar/IME/Offload/TensorExt work, descriptor/direct-C
  restoration, source-export route restoration, runtime correctness claims, and
  performance claims.
- New runtime/hardware evidence unless emitted object semantics change. This
  task changes preflight authority and should not claim fresh RVV runtime
  correctness.

## Technical Notes

- Main files: `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `test/Target/RVV/vector-materialized-target-artifact-exporters.mlir`.
- The body-owned builder to use is
  `plugin::rvv::buildRVVI32M1ArithmeticEmitCLowerableRoute`, not
  `buildRVVI32M1ArithmeticEmitCLowerableRouteForOperation` selected from
  metadata.
