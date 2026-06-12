# RVV unseeded vector-loop selected-boundary materializer

## Goal

Move the bounded RVV i32m1 add source front door from seed-attribute authority
to structural source recognition. The existing source pattern
`func(lhs, rhs, out, n) -> scf.for 0 to n step 4 -> vector<4xi32> load/load/add/store`
must materialize the same selected `tcrv.exec` dispatch boundary, explicit
`tcrv_rvv` runtime ABI / `setvl` / `with_vl` / load / add / store body, and
the already-real multi-VL EmitC object/header/bundle path without requiring
`tcrv_rvv.lowering_seed`.

## What I Already Know

- Current HEAD at task creation is `e813854 rvv: materialize runtime avl multivl emitc loop`; worktree was clean and `.trellis/.current-task` did not exist.
- The previous RVV task made the downstream selected route real: selected
  `tcrv_rvv` i32m1 arithmetic ops carry runtime `n` into materialized
  multi-VL `emitc.for`, MLIR C/C++ emission, object/header/bundle packaging,
  and real `ssh rvv` correctness for `n > VL`.
- `RVVSelectedBoundarySeed.cpp` already structurally validates the bounded
  source body after finding `tcrv_rvv.lowering_seed = "i32m1_add"`, then emits
  the selected RVV dispatch case plus conservative scalar fallback envelope.
- The production weakness is the discovery/authorization point: the pass only
  scans functions with `tcrv_rvv.lowering_seed`, and the positive transform and
  target artifact fixtures still carry that attribute.
- The active route must remain extension family ops -> common materialized
  EmitC route -> MLIR C/C++ emitter -> intrinsic/runtime C/C++ -> native
  compiler. Target/export code may validate and package the selected route,
  but must not infer RVV semantics or synthesize compute bodies from metadata.
- Existing public source-seed pass/front-door plumbing is plugin-owned and may
  remain as pass registration. This task changes the RVV pass authority, not
  the common registry contract.

## Requirements

- Recognize exactly the bounded unseeded RVV i32 add source shape:
  - one source function with positional ABI
    `(%lhs: memref<?xi32>, %rhs: memref<?xi32>, %out: memref<?xi32>, %n: index)`;
  - no function results;
  - one block;
  - one `scf.for` from constant index `0` to the runtime `%n`;
  - fixed step matching the bounded `vector<4xi32>` chunk;
  - exactly two source vector loads, one `arith.addi`, one vector store;
  - loads read `%lhs[%i]` and `%rhs[%i]`, store writes `%out[%i]`, and store
    value is the add result.
- Materialize the same selected RVV extension-family route already consumed by
  the downstream pipeline:
  - `tcrv.exec.kernel`;
  - `@rvv` capability and selected `origin = "rvv-plugin"` variant;
  - explicit `tcrv_rvv.runtime_abi_value` values for `lhs`, `rhs`, `out`, and
    runtime `n`;
  - `tcrv_rvv.setvl`, selected `tcrv_rvv.with_vl`, `i32_load`, `i32_add`, and
    `i32_store`;
  - existing selected dispatch case and conservative scalar fallback envelope.
- Keep route authority inside RVV plugin-owned materialization. Common/core
  passes must continue to see only plugin-registered source-seed passes,
  materialized selected surfaces, and generic emission/target handoff metadata.
- Treat `tcrv_rvv.lowering_seed` as stale metadata, not positive authority.
  Seed metadata alone must not create a selected route.
- Unsupported RVV source shapes must fail closed before emission planning and
  before target artifact export.
- Preserve plugin-disabled behavior: if built-in plugins are disabled, the RVV
  source materialization pass/front-door remains unavailable and source input
  must not silently become a selected route.

## Acceptance Criteria

- [x] Main positive transform fixture removes `tcrv_rvv.lowering_seed` and still
      shows selected `tcrv.exec` dispatch, explicit RVV runtime ABI
      provenance, `setvl` / `with_vl`, `i32_load`, `i32_add`, `i32_store`, and
      supported emission-plan metadata.
- [x] Positive source artifact object/header/bundle fixtures consume the
      unseeded source path and still produce the selected materialized multi-VL
      EmitC object/header/bundle metadata.
- [x] Negative lit coverage includes absent RVV plugin registration, wrong
      function ABI/order, non-runtime-`n` loop upper bound, nonzero lower
      bound, wrong step or vector shape, non-add arithmetic, extra loop ops,
      missing store, stale `tcrv_rvv.lowering_seed`-only metadata, and stale
      pre-existing `tcrv.exec` / `tcrv_rvv` residue.
- [x] Existing explicit RVV add/sub/mul extension-route tests continue to pass;
      this task does not expand source-frontdoor families beyond i32 add.
- [x] Targeted scans over touched RVV materializer, target/export surfaces, and
      tests show no descriptor route authority, no direct-C/source-export route,
      and no active positive test requiring `tcrv_rvv.lowering_seed`.
- [x] Focused lit/C++ checks for the RVV source materializer and source artifact
      route pass; `check-tianchenrv` is run if practical.
- [x] Real `ssh rvv` evidence is refreshed from the unseeded source path, or the
      final report explicitly states why existing runtime AVL artifact evidence
      was reused.

## Implementation Summary

- Reworked `RVVSelectedBoundarySeed.cpp` so the positive route discovers one
  bounded RVV source function candidate by structure instead of scanning for
  `tcrv_rvv.lowering_seed`.
- Treats RVV lowering-seed metadata as stale route residue; it is rejected with
  a fail-closed diagnostic and cannot authorize a selected RVV route.
- Preserves the public plugin source materialization pass/front-door name while
  changing its description and generated policy metadata to source-pattern
  wording.
- Keeps the materialized downstream IR unchanged in substance: the source
  pattern still creates the selected `tcrv.exec` dispatch envelope, RVV
  `origin = "rvv-plugin"` variant, runtime ABI values for `lhs/rhs/out/n`,
  `setvl`, `with_vl`, `i32_load`, `i32_add`, and `i32_store`.
- Updated RVV transform and SourceSeed front-door tests so positive inputs are
  unseeded, while negative tests cover stale seed metadata and unsupported
  unseeded source shapes.
- Updated RVV and variant-pipeline specs to make the bounded source body, not
  seed metadata, the durable positive authority.

## Validation

- Focused build:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- Focused C++:
  `./build/bin/tianchenrv-rvv-extension-plugin-test` and
  `./build/bin/tianchenrv-target-artifact-export-test`.
- Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-i32m1-selected-boundary-seed|source-seed-artifact-front-door-pipeline|source-seed-target-artifact-(header|object)'`
  from `build/test`; 7 selected tests passed.
- Focused explicit route lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-first-slice-materialization|i32m1-(add|sub|mul)|emitc-to-cpp-handoff|emitc-to-cpp-non-materialized'`
  from `build/test`; 8 selected tests passed.
- Full check:
  `cmake --build build --target check-tianchenrv -j2`; 106 lit tests passed.
- `git diff --check` passed.
- Generated unseeded source-path artifact evidence under
  `artifacts/tmp/rvv_unseeded_vector_loop_selected_boundary_materializer/20260516T202408Z`:
  selected plan, materialized EmitC, generated C++, relocatable RVV object,
  declaration-only header, bundle index, readobj output, harness, and ssh log.
- `/usr/lib/llvm-20/bin/llvm-readobj -h rvv_target_artifact.o` reports
  `Format: elf64-littleriscv`, `Arch: riscv64`, and `Type: Relocatable`.
- Real `ssh rvv` evidence from the unseeded source path reported
  `vlmax_e32m1=4` and passed `n=4`, `n=5`, and `n=11`, ending with
  `unseeded_multi_vl_tail_status=PASS`.
- Targeted scans confirmed no positive RVV transform/target fixture still
  requires `tcrv_rvv.lowering_seed`. Remaining RVV seed attr hits are only the
  stale-metadata rejection code/spec and negative tests. Descriptor/direct-C/
  source-export hits in touched target tests are `implicit-check-not` guards,
  not route authority.

## Self-Repair

- The first artifact evidence command used `llvm-readobj` from PATH, which was
  unavailable in this shell. Re-ran with `/usr/lib/llvm-20/bin/llvm-readobj`
  and completed the header/bundle evidence.
- Kept source-seed API names where they are public plugin pass/front-door
  plumbing, but changed RVV production diagnostics, policy metadata, tests, and
  specs so seed metadata is not positive authority.

## Non-Goals

- No generic RVV/vector lowering.
- No scalable-vector lowering.
- No new SEW/LMUL/dtype/op families.
- No sub/mul source-frontdoor parity.
- No descriptor tables/adapters or descriptor-driven computation.
- No direct C semantic exporters or source-export routes.
- No scalar fallback compute body.
- No compatibility wrapper that keeps `tcrv_rvv.lowering_seed` as a legal
  production authority.
- No Python compiler-core behavior.
- No extension-specific semantic branch in common/core passes.
- No broad smoke matrix or report-only achievement.

## Technical Notes

- Specs read for this task:
  `.trellis/spec/index.md`,
  `.trellis/spec/architecture/design-boundaries.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous completed task PRD read:
  `.trellis/tasks/archive/2026-05/05-17-05-17-rvv-runtime-avl-multivl-emitc-loop/prd.md`.
- Primary implementation surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVSelectedBoundarySeed.h`,
  `lib/Plugin/RVV/RVVSelectedBoundarySeed.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Transforms/ExecutionPlanningPipeline.cpp`,
  `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir`,
  `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed-negative.mlir`,
  `test/Target/RVV/source-seed-target-artifact-object.mlir`, and
  `test/Target/RVV/source-seed-target-artifact-header.mlir`.

## Definition Of Done

- The production positive RVV source path is structurally recognized without
  seed metadata and feeds the existing selected multi-VL EmitC artifact route.
- Focused checks, targeted scans, and ssh-rvv evidence status are recorded.
- Trellis task status, context files, and workspace journal are truthful.
- One coherent commit records the completed round, or the task remains open
  with an exact next continuation point.
