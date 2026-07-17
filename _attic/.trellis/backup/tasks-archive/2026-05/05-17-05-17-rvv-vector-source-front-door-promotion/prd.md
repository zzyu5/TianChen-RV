# Bounded RVV Vector-Source Front Door Promotion

## Goal

Promote the bounded RVV i32 add source materialization path from
evidence/demo-oriented `source-seed` naming into a production-scoped compiler
front door:

```text
ordinary MLIR func/vector i32 add source
  -> plugin-owned RVV selected boundary and dispatch
  -> selected emission plan
  -> materialized EmitC
  -> generated object/header/bundle
```

The path remains bounded to the existing i32 add vector-load/add/store source
pattern, selected RVV i32m1 variant, plugin-owned runtime ABI metadata,
materialized EmitC route, and already-proven generated bundle ABI.

## Current Repository Facts

- HEAD `8213e2a` proved the generated RVV object/header bundle is externally
  C-ABI consumable and runs `PASS` on real `ssh rvv` hardware.
- `RVVSelectedBoundarySeed.cpp` already recognizes an ordinary
  `func.func` source pattern containing `scf.for`, `vector.load`,
  `arith.addi`, and `vector.store`, then materializes a selected RVV dispatch
  case with explicit `tcrv_rvv` ops and scalar fallback.
- The public pass, C++ factory, pipeline, and target tests still expose the
  path as `source-seed` / `SelectedBoundarySeed`, which makes the working
  compiler front door look like a seed-only artifact harness.
- Existing RVV target artifact object/header/bundle routes already consume the
  selected emission-plan path and produce coherent object/header/index output.
- Relevant specs require the RVV plugin to own source-pattern recognition and
  selected-boundary materialization. Common code may run registered plugin
  passes and verify generic selected/emission-plan interfaces, but must not
  infer RVV semantics through descriptor metadata, direct C source export, or
  family-specific core branches.

## Requirements

- Add a production-named or production-scoped front door for the bounded RVV
  vector i32 add source path.
- Keep the source input as ordinary MLIR `func.func` plus MLIR
  `vector`/`arith`/`memref`/`scf` operations. Positive tests must not start
  from prebuilt `tcrv.exec` or `tcrv_rvv` IR.
- The RVV plugin must remain the owner of source-pattern recognition,
  selected RVV boundary materialization, runtime ABI values, selected dispatch
  case, and conservative fallback envelope.
- The resulting IR must contain:
  - selected RVV dispatch case;
  - explicit `tcrv_rvv` ops;
  - ordered runtime ABI values `lhs`, `rhs`, `out`, `n`;
  - supported RVV emission plan;
  - unsupported scalar fallback emission diagnostic.
- The target artifact object/header/bundle export must consume the promoted
  selected path and still emit the coherent RVV object/header/index with the
  unmangled C ABI.
- If old public `source-seed` naming remains, it must be demoted to a
  compatibility/internal alias and not be the primary positive target artifact
  front door.
- Add or refresh negative coverage for unsupported vector shapes/dtypes/ops,
  stale seed metadata, source modules that already contain selected
  `tcrv.exec`/`tcrv_rvv` residue, missing selected RVV case, ambiguous
  selected candidates, origin mismatch, fallback-only paths, and
  descriptor/source-export/direct-C route residue.

## Acceptance Criteria

- [ ] A production-scoped RVV vector-source pass or pipeline exists and is the
      primary tested front door for the bounded i32 add source path.
- [ ] Positive RVV transform coverage starts from ordinary MLIR source and
      checks the selected RVV dispatch, explicit `tcrv_rvv` ops, ordered ABI
      values, supported emission plan, and unsupported fallback diagnostic.
- [ ] RVV target artifact object/header/bundle tests consume the promoted
      front door and still check object ELF shape, unmangled C ABI symbol,
      header declaration metadata, and bundle index coherence.
- [ ] Old `source-seed` public naming, if preserved, is documented and tested
      as a compatibility/internal alias rather than the main artifact path.
- [ ] Negative lit coverage rejects unsupported source shapes/dtypes/ops,
      stale seed metadata, pre-existing selected IR residue, missing RVV case,
      ambiguous selected candidates, origin mismatch, fallback-only paths, and
      descriptor/source-export/direct-C residue.
- [ ] Targeted scans over touched RVV plugin/target/translate/tests show no
      descriptor route authority, no direct-C/source-export semantic route, no
      legacy RVV route aliases, and no new core/common RVV semantic branch.

## Out Of Scope

- No sub/mul source expansion, broader SEW/LMUL/dtype families, generic RVV
  lowering, scalar fallback compute, generic runtime dispatcher, or new
  performance claim.
- No descriptor adapters, compatibility wrappers around old route IDs,
  source-export/direct-C semantic exporters, Python compiler-core behavior, or
  handwritten C compute replacement.
- No new artifact ledger, checkpoint protocol, or state machine. Reuse the
  existing generated object/header/bundle ABI path.
- Do not replace the already-proven generated bundle ABI product; only update
  the bounded source front door that feeds it.

## Minimal Evidence

- `git diff --check`.
- Focused build for `tcrv-opt`, `tcrv-translate`, and touched RVV/plugin/target
  test binaries.
- Focused lit for RVV vector-source front door, RVV selected
  boundary/emission-plan checks, source front-door negatives, and generated
  RVV target artifact object/header/bundle checks.
- Focused C++ tests for plugin registration/pipeline integration if touched.
- Reuse existing `ssh rvv` C-ABI proof from commit `8213e2a` unless pass name,
  symbol, ABI, emitted object/header contents, or bundle layout changes require
  refreshing it.
- Targeted scans over touched RVV plugin/target/translate/tests for descriptor,
  direct-C, source-export, and legacy RVV route residue.
- `check-tianchenrv` if practical after focused checks pass.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous PRD read:
  `.trellis/tasks/archive/2026-05/05-17-rvv-generated-artifact-bundle-link-run-abi-proof/prd.md`.
- Initial code and test files inspected:
  `lib/Plugin/RVV/RVVSelectedBoundarySeed.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVSelectedBoundarySeed.h`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Transforms/ExecutionPlanningPipeline.cpp`,
  `test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`,
  `test/Target/RVV/vector-source-target-artifact-object.mlir`, and
  `test/Target/RVV/vector-source-target-artifact-header.mlir`.
