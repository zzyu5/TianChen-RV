# Bounded source-vector-to-RVV selected-boundary lowering

## Goal

Implement one bounded RVV plugin-owned source-shape recognizer for the existing
i32m1 add source-seed route. The marker may select this bounded route, but the
RVV selected boundary must be authorized by the MLIR source function signature
and body shape before RVV extension-family ops are materialized.

This round should make the current proof path more real:

```text
source MLIR func/vector/scf/arith add shape
  -> RVV selected boundary
  -> existing EmitC/artifact route
  -> ssh rvv correctness proof
```

## Current Direction Brief

- The previous runtime-VL contract task made SEW=32, LMUL=m1, tail/mask
  policy, runtime AVL source, VL def/scope/uses, ABI order, artifact metadata,
  target export, and ssh rvv evidence explicit.
- The next bottleneck is upstream of the selected boundary: the current source
  seed path must not treat the marker as sufficient semantic authority.
- The first coherent slice is the existing i32m1 add path:
  `memref<?xi32>` lhs/rhs/out plus runtime index `n`, with a bounded
  `scf`/`vector`/`arith` add body matching the current proof input.

## Requirements

- Add or tighten RVV-owned C++/MLIR source-shape recognition before selected
  boundary creation.
- Validate the source function signature:
  - exactly four arguments in the accepted first slice;
  - three one-dimensional dynamic `memref<?xi32>` buffer arguments;
  - one runtime `index` argument for `n`;
  - no function results;
  - ABI roles and order derived or verified as `lhs`, `rhs`, `out`, `n`.
- Validate the accepted source body shape enough that marker-only,
  empty-body, unrelated-body, stale-body, and unsupported-body inputs fail
  before RVV boundary materialization.
- Keep the route plugin-owned. Common/core passes may invoke plugin/common
  interfaces but must not gain RVV semantic branches.
- Reuse the existing RVV i32m1 add selected extension ops, config/runtime-VL
  contract, EmitC route, target object/header/bundle export, and ssh rvv
  harness.
- Keep the implementation in the C++/MLIR/LLVM/TableGen/CMake stack.
  Python is only allowed for tooling or artifact orchestration.

## Acceptance Criteria

- Positive source-seed materialization still reaches selected RVV boundary IR
  with explicit runtime-VL/config metadata and source-derived ABI mapping.
- The existing source-seed artifact front door still produces emission-plan
  metadata, materialized EmitC, RVV object/header/bundle artifacts, and a real
  `ssh rvv` PASS for the i32m1 add source seed.
- Negative lit coverage proves fail-closed rejection for:
  - wrong arithmetic op;
  - wrong vector element type or vector shape;
  - missing runtime `n`;
  - extra runtime `n` or extra function argument;
  - wrong memref argument order/count;
  - unsupported loop bounds or step;
  - stale `tcrv.exec` / `tcrv_rvv` residue;
  - marker with empty or unrelated body.
- Focused C++/lit checks pass for the touched RVV dialect/plugin/construction,
  EmitC route, target artifact, and source-seed front-door behavior.
- A changed-surface scan shows no descriptor-driven compute route, no direct C
  semantic exporter, no Python compiler-core logic, and no RVV semantic branch
  added to common/core orchestration.

## Non-Goals

- Generic MLIR vector lowering.
- Scalable vector lowering.
- Full multi-iteration `n` loop execution.
- New RVV op, dtype, SEW, LMUL, or artifact families.
- i32m1 sub/mul source lowering.
- i32m2 artifact routes.
- High-level tensor lowering.
- TensorExt, IME, offload, or new extension-family work.
- New artifact routes or performance tuning.
- Descriptor/binary-family registries, direct C semantic exporters, Python
  compiler-core logic, core RVV semantic branches, compatibility wrappers, or
  broad smoke matrices.
- Generic RVV backend maturity claims.

## Minimal Evidence Plan

- Inspect and update the existing RVV selected-boundary source seed path:
  `lib/Plugin/RVV/RVVSelectedBoundarySeed.cpp` and nearby RVV construction /
  config contract surfaces as needed.
- Build focused touched targets, including RVV dialect/plugin/construction,
  RVV EmitC/target support, `tcrv-opt`, and `tcrv-translate`.
- Run focused lit for RVV source-seed selected-boundary, source-seed artifact
  front door, RVV i32m1 add target artifact/header/bundle, and the new negative
  source-shape failures.
- Run relevant RVV/plugin C++ tests.
- Rerun the i32m1 add source-seed artifact path with real `ssh rvv` evidence.
- Run `git diff --check` and changed-surface scans.
- Run full `check-tianchenrv` if practical after focused checks pass.

## Definition of Done

- Source body validation influences whether selected RVV boundary materializes.
- Existing positive route still reaches artifact export and real RVV hardware
  evidence.
- Unsupported source shapes fail closed with actionable diagnostics.
- Trellis task status, journal, archive state, and commit are truthful.

## Implementation Summary

- `lib/Plugin/RVV/RVVSelectedBoundarySeed.cpp` now models the accepted source
  seed as a recognized `BoundedI32AddSourceSeed`.
- The recognizer validates the source function signature, source-only module
  rule, exact `scf.for` bounds/step, no loop-carried `iter_args`, body op
  order, load/store source argument use, `arith.addi` dataflow, and
  `vector<4xi32>` shape before materialization.
- Runtime ABI bindings are now consumed from the accepted source ABI mapping
  rather than created as disconnected literals. The materialized
  `tcrv_rvv.runtime_abi_value` ops carry source argument provenance:
  `source-arg-0:lhs`, `source-arg-1:rhs`, `source-arg-2:out`, and
  `source-arg-3:n`.
- The existing RVV i32m1 add extension-family boundary, runtime-VL/config
  contract, emission-plan metadata, EmitC route, object/header/bundle target
  export, and callable C ABI remain the production path.

## Accepted / Rejected Cases Covered

- Accepted: one marked source function with
  `memref<?xi32>, memref<?xi32>, memref<?xi32>, index`, one loop
  `0 -> n step 4`, two vector loads, one `arith.addi`, and one vector store.
- Rejected: missing runtime `n`, extra runtime argument, unsupported seed
  marker, wrong buffer dtype, wrong vector lane shape, wrong arithmetic op,
  wrong output buffer use, unsupported lower bound, unsupported upper bound,
  unsupported step, loop-carried values, unrelated body, empty body, stale
  `tcrv_rvv` residue, and stale `tcrv.exec` residue.

## Validation Results

- Focused build passed:
  `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin
  TianChenRVTarget tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-construction-protocol-common-test tianchenrv-rvv-dialect-test
  tianchenrv-target-artifact-export-test -j2`.
- Focused C++ tests passed:
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`,
  `tianchenrv-rvv-dialect-test`, and
  `tianchenrv-target-artifact-export-test`.
- Focused lit passed from `build/test`:
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'rvv-i32m1-selected-boundary-seed|source-seed-artifact-front-door|i32m1-add-object-artifact'`
  with 6/6 selected tests passed.
- Full practical check passed:
  `cmake --build build --target check-tianchenrv -j2`, 110/110 tests passed.
- `git diff --check` passed.
- Changed-surface scans showed no common/core/tooling files changed under
  `include/TianChenRV/Transforms`, `lib/Transforms`,
  `include/TianChenRV/Dialect/Exec`, `lib/Dialect/Exec`,
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/ExtensionPlugin.cpp`, `tools/tcrv-opt`, or
  `tools/tcrv-translate`.
- Changed RVV seed/test diff scan found no descriptor, direct-C,
  source-export, or Python compiler-core route terms.

## RVV Hardware Evidence

- Artifact directory:
  `artifacts/tmp/bounded_source_vector_to_rvv_selected_boundary/20260516T153540Z`.
- Generated artifacts include `seed.planned.mlir`, `seed.o`, `seed.h`,
  `bundle/tianchenrv-target-artifact-bundle.index`, bundle object/header, and
  `seed_harness.c`.
- `file seed.o` reported:
  `ELF 64-bit LSB relocatable, UCB RISC-V, RVC, double-float ABI`.
- Real `ssh rvv` link/run passed:
  `tcrv_rvv_i32m1_selected_boundary_seed status=PASS n=4 add=[12,6,16,12]`.
- Evidence log:
  `artifacts/tmp/bounded_source_vector_to_rvv_selected_boundary/20260516T153540Z/ssh_rvv_link_run.log`.

## Self-Repair

- Fixed a C++ constness error in the erase loop after focused build failed on
  `seed.func.erase()` through a const seed reference.
- Fixed the new loop-carried negative test so it reaches the intended
  `scf.for iter_args` gate instead of failing earlier on an unrelated i32
  constant in the source function body.

## Spec Update Judgment

The task produced one durable executable contract update. The existing
`.trellis/spec/variant-pipeline/generation-selection-tuning.md` RVV-owned
bounded vector i32 add seed scenario now records source-argument ABI
provenance purpose strings and loop-carried `scf.for` rejection requirements.
