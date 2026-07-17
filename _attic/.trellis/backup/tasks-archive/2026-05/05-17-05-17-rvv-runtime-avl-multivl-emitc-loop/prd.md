# RVV runtime AVL multi-VL EmitC loop

## Goal

Replace the current `multi_vl = unsupported` one-VL execution limit for the
bounded RVV i32m1 arithmetic target artifact route with a real runtime
multi-VL loop. The loop must be driven by the selected runtime `n`/AVL ABI,
materialized through the existing RVV extension-family ops and common EmitC
route, emitted by the MLIR C/C++ emitter, and validated through the existing
object/header/bundle target artifact path plus real `ssh rvv` correctness
evidence.

## What I Already Know

- Current HEAD at task creation is `40382b8 rvv: close runtime avl vl artifact
  abi`; worktree was clean and no `.trellis/.current-task` existed.
- The previous completed RVV task closed the selected one-VL i32m1 arithmetic
  object/header/bundle ABI boundary and published artifact metadata including
  `bounded_slice = one-vl-i32m1-arithmetic` and `multi_vl = unsupported`.
- The active architectural route is explicit extension-family ops -> common
  materialized EmitC route -> MLIR C/C++ emitter -> intrinsic/runtime C/C++ ->
  native compiler. Target artifact export may validate/package the route but
  must not synthesize RVV compute bodies, loops, source text, or intrinsic
  semantics from metadata.
- Runtime `n`, AVL, VL, pointer/index advancement, and tail handling are
  runtime control/ABI behavior. They must not be encoded as descriptor-local
  element counts or target-owned selected-shape metadata.
- The supported scope is the existing RVV i32m1 add/sub/mul route family only.

## Requirements

- Materialize a bounded runtime loop for selected RVV i32m1 arithmetic that:
  - initializes remaining AVL from runtime ABI parameter `n`;
  - repeatedly computes VL from the remaining AVL through the selected
    `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` ownership contract;
  - advances lhs/rhs/out pointers or indices after each chunk;
  - handles a tail chunk when `n` is not a multiple of the hardware VL.
- Keep RVV extension-family ops and the selected runtime ABI/VL metadata as
  the only semantic authority for the route.
- Represent the loop in the materialized MLIR/EmitC path consumed by the MLIR
  C/C++ emitter. Do not add raw source-string loop printers or target-side loop
  synthesis.
- Update object/header/bundle artifact metadata truthfully for the supported
  multi-VL route, replacing the one-VL-only claim only where the materialized
  loop is present and validated.
- Validate target artifact candidates fail closed when multi-VL metadata is
  missing, stale, malformed, or claims support without a materialized loop.
- Preserve declaration-only header behavior: headers may expose bounded
  metadata comments but must not contain compute bodies, intrinsic calls,
  self-check code, runtime probing, or evidence text.

## Acceptance Criteria

- [x] Positive lit/C++ evidence shows selected RVV i32m1 arithmetic with
      runtime `n` materializes a loop through EmitC and the MLIR C/C++ emitter,
      including remaining-AVL update, repeated VL computation, pointer/index
      advancement, and tail handling.
- [x] Object/header/bundle routes package the materialized function with
      metadata truthfully stating multi-VL support for the bounded i32m1
      arithmetic route.
- [x] Positive real `ssh rvv` evidence runs at least one `n` greater than a
      single hardware VL and at least one tail case through the generated
      bundled header/object path.
- [x] Negative coverage rejects missing runtime AVL ABI, mismatched
      `setvl`/`with_vl` ownership, unsupported configs, malformed loop
      metadata, fallback-only selection, multiple selected supported
      candidates, descriptor/source-export/direct-C route residue, and
      headers/bundles that claim multi-VL support without the materialized
      loop.
- [x] Focused C++ tests and lit tests cover RVV EmitC route materialization,
      config/VL contract, target artifact preflight, header, object, and bundle
      packaging.
- [x] Targeted scans over touched RVV plugin/target/translate/test surfaces
      show no restored descriptor route authority, no direct C semantic
      exporter, no source-export route, and no tests protecting old paths.

## Implementation Summary

- Extended the common `TCRVEmitCLowerableRoute` contract with structured loop
  payloads and taught the materializer to emit `emitc.for`, remaining-AVL
  subtraction, and pointer/index advancement through EmitC values.
- Rewired the RVV i32m1 add/sub/mul EmitC route so runtime `n` first computes a
  full-chunk VL, then loops over `offset`, recomputes VL from `n - offset` for
  each chunk, loads from `lhs + offset` and `rhs + offset`, computes the selected
  arithmetic intrinsic, and stores to `out + offset`.
- Updated RVV config/artifact metadata from the one-VL unsupported contract to
  `bounded_slice = multi-vl-i32m1-arithmetic` and `multi_vl = supported` only for
  the materialized loop route.
- Hardened target artifact preflight so multi-VL metadata requires a selected
  materialized EmitC module containing an actual `emitc.for` with route/body
  provenance.
- Updated declaration-only headers, object/header bundle metadata, focused lit
  tests, and C++ negative coverage for stale or missing multi-VL loop metadata.
- Promoted the durable multi-VL loop boundary into the RVV plugin, EmitC route,
  and runtime emission Trellis specs.

## Validation

- Built focused targets:
  `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tianchenrv-emitc-lowerable-interface-test tcrv-opt tcrv-translate -j2`.
- Ran focused C++ tests:
  `./build/bin/tianchenrv-target-artifact-export-test`,
  `./build/bin/tianchenrv-rvv-extension-plugin-test`, and
  `./build/bin/tianchenrv-emitc-lowerable-interface-test`.
- Ran focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-first-slice-materialization|rvv-i32m1-selected-boundary-seed|emitc-to-cpp-handoff|source-seed-target-artifact-header|source-seed-target-artifact-object'`
  from `build/test`; 11 selected tests passed.
- Ran project check:
  `cmake --build build --target check-tianchenrv -j2`; 106 lit tests passed.
- Generated artifact evidence under
  `artifacts/tmp/rvv_runtime_avl_multivl_emitc_loop/20260516T200010Z`:
  materialized EmitC, generated C++, relocatable RVV object, declaration-only
  header, bundle index, readobj output, harness, and ssh log.
- Real `ssh rvv` evidence from that artifact reported `vlmax_e32m1=4` and
  passed `n=4`, `n=5`, and `n=11`; `n=5` and `n=11` cross a single hardware VL
  and include tail chunks.
- Targeted scans confirmed no stale one-VL unsupported markers. The only
  remaining implementation hit for `descriptor` is the target preflight code
  that rejects descriptor-local or hardcoded element-count residue.
- `git diff --check` passed.

## Non-Goals

- No new SEW/LMUL/dtype families, generic vector lowering, high-level tensor
  frontend, scalar fallback compute, descriptor tables/adapters, compatibility
  routes, direct C semantic exporters, Python compiler-core behavior,
  target-owned RVV intrinsic branches, raw source-string loop printers,
  performance matrix work, or metadata-only multi-VL claims.
- No new artifact ledger, state machine, checkpoint protocol, broad smoke
  matrix, or report/prompt-only achievement.
- If the existing route cannot express the loop cleanly, complete one coherent
  IR/control-boundary submodule and keep unsupported cases fail-closed instead
  of restoring deleted direct-source behavior.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Prior task PRD read:
  `.trellis/tasks/archive/2026-05/05-17-rvv-runtime-avl-vl-abi-artifact-closure/prd.md`.
- Primary implementation surfaces from the brief:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir`,
  `test/Target/RVV/`, and `test/Target/TargetArtifactExportTest.cpp`.

## Definition Of Done

- Production/default RVV i32m1 arithmetic object/header/bundle route uses the
  materialized multi-VL EmitC loop for supported selected candidates.
- Focused build/tests/lit, `git diff --check`, targeted scans, and real
  `ssh rvv` evidence are recorded.
- Trellis task status, context, and workspace journal are truthful.
- One coherent commit records the completed round, or the task remains open
  with the exact next continuation point.
