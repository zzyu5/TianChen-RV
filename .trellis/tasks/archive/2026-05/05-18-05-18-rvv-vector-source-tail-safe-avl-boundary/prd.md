# RVV vector-source tail-safe AVL selected-boundary materialization

## Goal

Repair the RVV vector-source front door so accepted source MLIR proves the same
runtime `n -> remaining AVL -> setvl -> VL -> with_vl` semantics that the RVV
materialized EmitC artifact route validates. The accepted source form must make
tail behavior explicit for runtime counts that are not multiples of four, or
the production source-artifact path must fail closed before claiming arbitrary
`n` correctness.

## What I Already Know

- Repository state before edits: `/home/kingdom/phdworks/TianchenRV`, branch
  `main`, clean worktree, HEAD `177b02d rvv: validate runtime avl vl selected boundary`.
- No `.trellis/.current-task` existed; this task was created from the Hermes
  Direction Brief.
- The previous archived task made target artifact export validate selected
  route id, selected body op, unique `tcrv_rvv.with_vl` provenance, runtime ABI
  order `lhs,rhs,out,n`, stale-boundary failures, and real `ssh rvv` evidence
  for add/sub/mul counts 7, 16, and 23.
- Current `RVVVectorSourceFrontDoor.cpp` still accepts a fixed
  `scf.for` step-4 `vector.load` / `vector.store` source shape. That source
  shape is not source-level tail-safe for arbitrary runtime `n`.
- The local MLIR toolchain accepts a tail-explicit source pattern using:
  `arith.subi %n, %i : index`, `vector.create_mask`, masked
  `vector.transfer_read`, arithmetic on `vector<4xi32>`, and masked
  `vector.transfer_write`.
- Specs require RVV source recognition and typed-route materialization to
  remain RVV plugin-owned. Common front-door/target code may compose registered
  passes and target exporters, but must not infer RVV semantics from route ids,
  descriptors, artifact names, or generic target branches.

## Requirements

1. The positive RVV vector-source matcher must require explicit source-level
   tail semantics: compute `remaining = n - iv`, create `vector<4xi1>` mask
   from that remaining count, use that same mask for both input transfers and
   the output transfer, and keep loop upper bound tied to the runtime ABI `n`.
2. The accepted source form must still map function ABI operands to
   `lhs,rhs,out,n` in that order and materialize `n` as the
   `runtime-element-count` ABI value feeding `tcrv_rvv.setvl`.
3. The materialized RVV selected boundary must remain the current bounded
   i32m1 add/sub/mul route:
   `runtime_abi_value -> setvl -> with_vl -> i32_load/i32_load ->
   i32_add|i32_sub|i32_mul -> i32_store`.
4. The old fixed-width `vector.load` / `vector.store` source shape must no
   longer be accepted as a valid arbitrary-count source input when it lacks an
   explicit tail or multiple-of-width contract.
5. Existing fail-closed checks for stale source metadata, stale
   `tcrv.exec`/`tcrv_rvv` residue, mismatched loop bounds, unsupported dtype or
   op family, wrong ABI order, and descriptor/direct-C/source-export residue
   must remain fail-closed.
6. The production source-artifact front door must still reach selected
   dispatch, emission plan, materialized EmitC route, object, header, and bundle
   through existing target artifact exporters for accepted tail-safe source
   fixtures.

## Acceptance Criteria

- [x] Positive add/sub/mul source fixtures use the explicit tail-safe source
      form with `remaining = n - iv`, `vector.create_mask`, masked
      `vector.transfer_read`, and masked `vector.transfer_write`.
- [x] `tcrv-opt --tcrv-source-artifact-front-door-pipeline` materializes
      `tcrv_rvv.runtime_abi_value`, `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`,
      load/compute/store, selected dispatch, and supported RVV emission-plan
      metadata from the accepted tail-safe source.
- [x] Source-artifact target object/header/bundle exports still work through
      the materialized EmitC route and preserve runtime ABI order
      `lhs,rhs,out,n`.
- [x] Negative coverage rejects the old unsafe fixed `vector.load` /
      `vector.store` source shape when it lacks an explicit tail mask or
      multiple-of-width contract.
- [x] Negative coverage rejects mismatched loop bounds, stale source metadata,
      pre-existing `tcrv.exec`/`tcrv_rvv` residue, unsupported arithmetic or
      dtype, stale source claims, and source ABI order not equal to
      `lhs,rhs,out,n`.
- [x] Real `ssh rvv` evidence is refreshed for the accepted source route with
      at least one non-multiple-of-4 count and one multiple-of-4 count if this
      round keeps runtime correctness claims active.
- [x] Targeted scans over changed RVV source/target/tests show no descriptor
      authority, no direct-C/source-export compute route, and no test
      preserving the unsafe source shape as valid arbitrary-count input.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` runs if practical; if not, the environment reason is
      recorded.

## Non-Goals

- No new RVV dtype, SEW/LMUL family, arithmetic op, generic RVV backend, or
  scalar fallback compute body.
- No new target artifact route, descriptor route authority, direct C semantic
  exporter, source-export compatibility path, legacy mode, or compatibility
  wrapper.
- No Python compiler-core behavior. Python may only remain evidence/tooling.
- No RVV semantic branch in common/core orchestration; source recognition stays
  plugin-owned.
- No docs-only, report-only, helper-only, or broad smoke-matrix result.

## Expected Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-05-18-rvv-vector-source-tail-safe-avl-boundary`
- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- Focused lit for:
  - `test/Transforms/RVV/rvv-i32m1-vector-source-front-door*.mlir`
  - `test/Transforms/SourceFrontDoor/source-artifact-front-door-pipeline-*.mlir`
  - `test/Target/RVV/vector-source-target-artifact-exporters.mlir`
  - `test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py` for accepted source route
  with at least one non-multiple-of-4 and one multiple-of-4 count, using
  `--ssh-target rvv` if runtime claims remain.
- Targeted `rg` scans over touched RVV source/target/tests for descriptor,
  direct-C, source-export, and unsafe `vector.load` / `vector.store` positive
  source residues.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2` if practical.

## Definition Of Done

- The production/default RVV source-artifact route accepts only an explicit
  tail-safe source form for arbitrary runtime `n`, or truthfully leaves the
  task open after fail-closing the unsafe shape.
- Positive materialized EmitC object/header/bundle behavior remains intact for
  the accepted source form.
- The Trellis task status, context, journal, archive state, and final commit
  accurately reflect what was completed.

## Technical Notes

- Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous archived PRD read:
  `.trellis/tasks/archive/2026-05/05-18-rvv-runtime-avl-vl-selected-boundary-contract/prd.md`.
- Relevant code/tests inspected before implementation:
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  RVV source/target lit fixtures, and
  `scripts/rvv_generated_bundle_abi_e2e.py`.

## Completion Notes

- Implemented the source-front-door semantic repair in
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`.
- The accepted source form is now:
  `scf.for %i = 0 to %n step 4 -> remaining = n - i ->
  vector.create_mask remaining -> masked vector.transfer_read(lhs/rhs) ->
  vector add/sub/mul -> masked vector.transfer_write(out)`.
- The materialized selected boundary remains the existing RVV typed route:
  `runtime_abi_value(lhs,rhs,out,n) -> setvl(n) -> with_vl ->
  i32_load/i32_load -> i32_add|i32_sub|i32_mul -> i32_store`.
- The old unsafe fixed `vector.load` / `vector.store` shape is now rejected by
  the production source matcher when used without explicit tail semantics.
- Positive add/sub/mul source fixtures, source-artifact target exporter
  fixtures, disabled/stale source-front-door fixtures, and the RVV bundle ABI
  script inputs now start from the tail-safe source form.
- Added durable source-front-door contract text to
  `.trellis/spec/extension-plugins/rvv-plugin.md`.
- No descriptor route authority, direct-C/source-export compute path, scalar
  fallback compute body, new dtype/LMUL family, compatibility wrapper, or common
  RVV semantic branch was added.

## Checks Run

- `cmake --build build --target tcrv-opt tcrv-translate -j2`
- `cd build/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir ../test/Transforms/RVV/rvv-i32m1-vector-source-front-door-sub.mlir ../test/Transforms/RVV/rvv-i32m1-vector-source-front-door-mul.mlir ../test/Transforms/RVV/rvv-i32m1-vector-source-front-door-negative.mlir ../test/Transforms/SourceFrontDoor/source-artifact-front-door-pipeline-disabled.mlir ../test/Transforms/SourceFrontDoor/source-artifact-front-door-pipeline-negative.mlir ../test/Target/RVV/vector-source-target-artifact-exporters.mlir ../test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir`
- `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260518T-rvv-vector-source-tail-safe-avl-boundary-dry-run --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260518T-rvv-vector-source-tail-safe-avl-boundary --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- Targeted residue scan over changed RVV source/target/tests/script for
  descriptor, direct-C, source-export, `rvv-direct-microkernel`, and unsafe
  `vector.load` / `vector.store`. Hits were limited to existing script
  forbidden-token checks, FileCheck `implicit-check-not` guards, and the new
  old-shape negative fixture.
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-05-18-rvv-vector-source-tail-safe-avl-boundary`
- `cmake --build build --target check-tianchenrv -j2`

Real `ssh rvv` run artifact path:

```text
artifacts/tmp/rvv_generated_bundle_abi_e2e/20260518T-rvv-vector-source-tail-safe-avl-boundary
```

Remote PASS markers:

```text
tcrv_rvv_generated_bundle_abi_add_ok counts=7,16,23
PASS op=add counts=7,16,23
tcrv_rvv_generated_bundle_abi_sub_ok counts=7,16,23
PASS op=sub counts=7,16,23
tcrv_rvv_generated_bundle_abi_mul_ok counts=7,16,23
PASS op=mul counts=7,16,23
```

## Spec Update Review

`.trellis/spec/extension-plugins/rvv-plugin.md` was updated because the
accepted RVV vector-source front-door form changed from an unsafe fixed-width
source pattern to an explicit tail-safe source contract.
