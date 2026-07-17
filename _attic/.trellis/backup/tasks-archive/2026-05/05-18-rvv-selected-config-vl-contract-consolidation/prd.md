# RVV selected config/VL contract consolidation

## Goal

Consolidate the bounded RVV i32m1 SEW32/LMUL m1/tail-agnostic/mask-agnostic
runtime AVL/VL contract into one plugin-owned C++ contract surface and make the
existing source front door, selected-boundary construction, EmitC route
provider, and RVV target artifact validation consume it. This is a modeling
and validation refactor for the existing add/sub/mul route; it must not add a
new descriptor, finite route registry, source-export path, or family coverage.

## What I Already Know

- Repository state before edits: `/home/kingdom/phdworks/TianchenRV`, clean
  worktree, HEAD `f25e3c6 rvv: require tail-safe vector source front door`.
- No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
- The previous task made the source front door tail-safe by requiring
  `remaining = n - iv`, `vector.create_mask`, masked
  `vector.transfer_read`, and masked `vector.transfer_write` before
  materializing the bounded selected RVV i32m1 route.
- The previous target/export task already introduced
  `RVVConfigContract.{h,cpp}` with config/VL artifact metadata and
  `validateRVVI32M1ArithmeticConfigVLContract(setvl, with_vl)`, but several
  consumers still encode adjacent assumptions separately.
- Current consumers include:
  - `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`, which constructs `setvl`
    and `with_vl` attributes separately from helper calls.
  - `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`, which owns
    route, runtime ABI, role graph, and construction artifact metadata.
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, which validates selected
    `setvl`/`with_vl`, runtime ABI values, dataflow shape, and EmitC loop
    payload.
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`, which validates selected
    artifact candidates and duplicates RVV config/VL metadata evidence in the
    header adapter configuration.
- Specs require RVV-specific facts to stay plugin/dialect local. Common/core
  orchestration must not add RVV family semantic branches, and descriptors or
  target metadata must never become compute authority.

## Requirements

1. The bounded RVV i32m1 config/VL contract must be represented by one
   reusable plugin-owned/dialect-owned C++ surface, not copied string/attribute
   assumptions spread across source materialization, construction metadata,
   EmitC route construction, and target validation.
2. Source-front-door materialization must construct `tcrv_rvv.setvl` and
   `tcrv_rvv.with_vl` from that shared contract: SEW32, LMUL m1,
   `#tcrv_rvv.policy<tail = agnostic, mask = agnostic>`, runtime AVL from ABI
   `n`, and visible VL consumed by `with_vl`.
3. Hand-authored selected-boundary fixtures must still be validated against the
   same contract. Mismatched SEW, LMUL, tail/mask policy, missing
   `with_vl` metadata, or wrong `with_vl` dataflow must fail closed before
   route construction or artifact export.
4. EmitC route construction must derive loop and route payload assumptions
   from the shared contract where applicable: runtime AVL parameter `n`,
   full-chunk initial `setvl`, per-iteration `n - offset` AVL, `vl` use by
   loads/compute/store, and stable ABI order `lhs,rhs,out,n`.
5. Target artifact validation and header/bundle metadata evidence must consume
   the same RVV config/VL artifact metadata contract rather than keeping a
   second handwritten list.
6. Route/op mismatches must still fail closed: selected candidate route id,
   `rvv_arithmetic_op`, typed body op, runtime ABI name, and rebuilt route must
   agree.
7. The existing add/sub/mul tail-safe vector-source object/header/bundle export
   must continue to work.
8. Targeted scans over changed RVV plugin/target/test surfaces must show that
   descriptor route authority, direct-C semantic exporter, source-export route,
   and the old fixed-width source-shape compatibility path were not restored.

## Acceptance Criteria

- [ ] A single RVV i32m1 config/AVL/VL contract API is used by the source front
      door, selected-boundary validation/route construction, and target
      artifact metadata validation.
- [ ] Source-front-door positive add/sub/mul fixtures still materialize the
      selected `runtime_abi_value(lhs,rhs,out,n) -> setvl(n) -> with_vl ->
      i32_load/i32_load -> i32_add|i32_sub|i32_mul -> i32_store` boundary.
- [ ] Negative coverage rejects stale lowering-seed metadata, wrong ABI `n`,
      mismatched SEW, mismatched LMUL, non-agnostic policy, wrong
      `setvl`/`with_vl` VL dataflow, route/op mismatch, and stale target
      artifact metadata.
- [ ] Target object/header/bundle export for the existing route still works and
      exposes the same config/VL evidence derived from the contract.
- [ ] C++ tests cover the shared contract API directly and prove the expected
      metadata/runtime ABI payload remains stable.
- [ ] `git diff --check` passes.
- [ ] `check-tianchenrv` runs if practical; if not, the environment reason is
      recorded.

## Non-Goals

- No new SEW/LMUL/dtype family, generic RVV lowering, new source shape, or new
  arithmetic op coverage.
- No descriptor adapter, source-export route, direct C semantic exporter,
  compatibility wrapper, legacy fixed-width source acceptance, or Python
  compiler-core behavior.
- No common/core RVV semantic branch.
- No broad smoke matrix unless a focused check shows the changed behavior
  requires it.
- No new artifact ledger or standalone evidence packaging.

## Expected Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-rvv-selected-config-vl-contract-consolidation`
- Build focused targets as needed, at minimum `tcrv-opt`, `tcrv-translate`, and
  affected C++ test binaries.
- Focused C++ tests for RVV dialect/config contract and target artifact export
  validation.
- Focused lit for:
  - `test/Dialect/RVV/setvl.mlir`
  - `test/Dialect/RVV/with-vl.mlir`
  - `test/Dialect/RVV/dataflow.mlir`
  - `test/Transforms/RVV/rvv-i32m1-vector-source-front-door*.mlir`
  - `test/Conversion/EmitC/rvv-first-slice-*-contract-negative.mlir`
  - `test/Target/RVV/vector-source-target-artifact-exporters.mlir`
  - `test/Target/RVV/vector-materialized-target-artifact-exporters.mlir`
  - `test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir`
- Targeted `rg` scans over touched RVV plugin/target/tests for descriptor,
  direct-C, source-export, direct microkernel, and unsafe `vector.load` /
  `vector.store` positive-source residues.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` if practical.

## Definition Of Done

- The production/default path consumes one shared bounded RVV i32m1
  config/AVL/VL contract across source recognition/materialization, selected
  route construction, and target artifact validation.
- Existing add/sub/mul source-to-selected-boundary-to-object/header/bundle
  behavior remains intact.
- The task status, journal, archive state, and final commit truthfully reflect
  the completed scope and any remaining rebuild gaps.

## Technical Notes

- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`.
- Previous archived PRD read:
  `.trellis/tasks/archive/2026-05/05-18-05-18-rvv-vector-source-tail-safe-avl-boundary/prd.md`.
- Initial code inspected:
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`, RVV dialect/source/target lit
  fixtures, and current workspace journal entries.

## Completion Notes

- Extended the exact contract owner
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h` and
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp` into a shared bounded
  `RVVI32M1ArithmeticConfigVLContract` surface.
- The shared surface now owns SEW32, LMUL m1, tail/mask agnostic policy,
  runtime AVL ABI parameter `n`, runtime ABI order `lhs,rhs,out,n`, selected
  `setvl -> with_vl` VL names, EmitC loop naming, config/VL artifact
  metadata, and runtime ABI parameter verification.
- Wired the source front door to consume the shared contract for runtime ABI
  value construction and for `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` config
  attributes.
- Wired RVV construction protocol runtime ABI validation to the shared
  contract instead of a private duplicate list.
- Wired EmitC route payload construction to the shared contract for runtime
  ABI verification and loop/VL naming while preserving the same generated
  `n - offset`, `full_chunk_vl`, and `vl` route payload.
- Wired RVV target artifact preflight/header metadata evidence to the shared
  contract metadata vector rather than a second handwritten `tcrv_rvv.*`
  evidence list.
- Added C++ coverage in `tianchenrv-rvv-dialect-test` for the shared contract
  API, runtime ABI verifier, artifact metadata verifier, stale ABI rejection,
  stale metadata rejection, and remaining-AVL expression.
- Updated target/header lit checks to assert the expanded contract evidence
  exported through source and materialized target artifact routes.
- No descriptor route authority, direct-C/source-export compute path, Python
  compiler-core behavior, new dtype/LMUL family, new source shape, or common
  RVV semantic branch was added.
- Real `ssh rvv` was not rerun because emitted object/header/bundle behavior
  was not semantically changed; focused object/header/bundle lit and
  `check-tianchenrv` revalidated the materialized artifact path.

## Checks Run

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-rvv-selected-config-vl-contract-consolidation`
- `cmake --build build --target tianchenrv-rvv-dialect-test tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-rvv-dialect-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `cd build/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Dialect/RVV/setvl.mlir ../test/Dialect/RVV/with-vl.mlir ../test/Dialect/RVV/dataflow.mlir ../test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir ../test/Transforms/RVV/rvv-i32m1-vector-source-front-door-sub.mlir ../test/Transforms/RVV/rvv-i32m1-vector-source-front-door-mul.mlir ../test/Transforms/RVV/rvv-i32m1-vector-source-front-door-negative.mlir ../test/Conversion/EmitC/rvv-first-slice-config-vl-contract-negative.mlir ../test/Conversion/EmitC/rvv-first-slice-vl-contract-negative.mlir ../test/Target/RVV/vector-source-target-artifact-exporters.mlir ../test/Target/RVV/vector-materialized-target-artifact-exporters.mlir ../test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir`
- `cmake --build build --target check-tianchenrv -j2`
- `git diff --check`
- Targeted residue scan over touched RVV dialect/plugin/target/tests for
  `descriptor`, `direct-C`, `direct_c`, `source-export`, `source_export`,
  `rvv-direct-microkernel`, `vector.load`, and `vector.store`.

## Self-Repair Notes

- First focused build exposed that the new C++ contract test had been inserted
  into the raw MLIR string of the existing dataflow test. Moved the function
  after the dataflow test body and rebuilt successfully.
- First `check-tianchenrv` exposed an observable diagnostic text drift in the
  construction protocol test: the shared verifier said `ordered callable ABI
  parameters` while the existing test expected `ordered runtime ABI
  parameters`. Restored the existing diagnostic phrase in the shared verifier
  and reran `check-tianchenrv` successfully.
