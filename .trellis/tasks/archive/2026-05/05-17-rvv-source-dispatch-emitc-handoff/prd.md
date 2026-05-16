# Common materialized EmitC handoff for RVV source dispatch

## Goal

Make one bounded production handoff proof real for the existing RVV i32m1 add
source-seed selected-dispatch path:

```text
source MLIR
  -> source-derived selected dispatch
  -> RVV extension-family ops
  -> common materialized MLIR EmitC module
  -> MLIR EmitC C/C++ emitter
  -> RVV object/header/bundle artifact packaging
  -> ssh rvv evidence when generated artifact behavior changes
```

The immediate bottleneck is not another source-seed shape or arithmetic family.
It is proving that the selected RVV compute body reaches an explicit,
materialized MLIR EmitC module boundary before C/C++ emission and target
artifact packaging, rather than letting target export synthesize compute from
selected metadata, descriptors, route strings, or direct source printers.

## What I Already Know

- The previous task committed source-derived RVV i32m1 add through
  `tcrv.exec.dispatch`, `tcrv.exec.case`, scalar fallback envelope, emission
  plan, object/header/bundle packaging, and real `ssh rvv` evidence.
- `lib/Plugin/RVV/RVVSelectedBoundarySeed.cpp` owns the bounded source seed and
  the currently accepted i32m1 add source shape.
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` owns the RVV route mapping
  from typed RVV extension-family ops to bounded EmitC route payloads.
- `lib/Transforms/EmitCLowerableMaterialization.cpp` is the common
  materialized EmitC boundary. It may verify and materialize MLIR EmitC IR, but
  common code must not emit runtime C/C++ source bodies directly.
- `lib/Transforms/EmissionReadiness.cpp` and target artifact export already
  reason about selected dispatch surfaces, emission-plan diagnostics, runtime
  ABI metadata, and artifact candidates.
- `lib/Target/TargetArtifactExport.cpp` and
  `lib/Target/RVV/RVVTargetSupportBundle.cpp` are the production consumers that
  must either consume the materialized EmitC handoff or verify the same common
  materialization helper before invoking the MLIR EmitC C/C++ emitter and
  artifact packaging.
- There is no `.trellis/spec/target-artifacts/index.md` in this repository.
  The closest live specs are `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.

## Requirements

- Keep the task bounded to the already accepted RVV i32m1 add source shape,
  existing RVV extension ops, existing runtime-VL/config contract, and existing
  target artifact kinds.
- Add or repair one production handoff where the accepted source-seed selected
  dispatch reaches a verified materialized MLIR EmitC module with:
  - source-derived ABI provenance;
  - selected dispatch-case role;
  - scalar fallback excluded from compute export;
  - RVV typed-op provenance;
  - required route/lowering provenance;
  - MLIR EmitC C/C++ output before object/header/bundle export.
- Target artifact export must consume that materialized EmitC handoff, or share
  and verify the same common materialization helper before invoking the MLIR
  EmitC C/C++ emitter.
- Target artifact export must not synthesize compute C directly from metadata,
  descriptors, route strings, selected plan text, or file-name conventions.
- Common code must remain extension-neutral. Any RVV-specific intrinsic names,
  ABI details, headers, or runtime-VL facts must stay in RVV-owned plugin or
  target code and flow through existing interfaces/registries.
- Preserve scalar fallback as unsupported envelope fallback only. Do not add
  scalar fallback compute semantics.
- Keep implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python is allowed only for evidence tooling or artifact parsing.

## Acceptance Criteria

- Positive compiler coverage proves the front-door/source-seed input reaches
  materialized EmitC with:
  - `lhs`, `rhs`, `out`, and `n` source-derived ABI provenance;
  - selected RVV dispatch-case role;
  - fallback excluded from compute export;
  - RVV op provenance;
  - runtime-VL/config metadata preserved;
  - MLIR EmitC C/C++ output before RVV object/header/bundle export.
- Positive target coverage proves object/header/bundle export consumes or
  verifies the common materialized EmitC handoff before packaging the bounded
  RVV artifact.
- Negative tests fail closed for:
  - fallback-only selection;
  - ambiguous multiple non-fallback selected plans;
  - stale or missing selected dispatch targets;
  - missing source/runtime ABI provenance;
  - materialized EmitC missing required route provenance;
  - unrelated source body plus dispatch marker;
  - manually pre-existing `tcrv.exec` or `tcrv_rvv` residue.
- Focused checks cover source-seed selected-dispatch to materialized EmitC,
  EmitC-to-C++ handoff, target object/header/bundle export, and the fail-closed
  cases above.
- Relevant C++ tests for EmitC materialization, emission readiness, target
  artifact export, and RVV plugin/target support pass.
- If object/source generation or artifact packaging behavior changes,
  refreshed real `ssh rvv` evidence is recorded and reported.
- A changed-surface scan shows no descriptor-driven compute, no direct C
  semantic exporter, no source-export route, no Python compiler-core logic, and
  no RVV-specific semantic branch in common/core.

## Non-Goals

- New RVV sub/mul source seeds.
- New arithmetic families.
- Generic runtime dispatch selection.
- Scalar fallback compute semantics.
- New target artifact kinds.
- New state-machine, artifact-ledger, or checkpoint protocols.
- Descriptor/source-export/direct-C compatibility paths.
- Python compiler-core implementation.
- Extension-specific semantic branches in common/core passes.
- High-level frontend lowering beyond the already accepted source-seed shape.
- Performance tuning or broad benchmark claims.

## Technical Notes

- Specs read for this PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous task PRD read:
  `.trellis/tasks/archive/2026-05/05-16-rvv-source-seed-selected-dispatch-artifact/prd.md`.
- Recent workspace journal context read:
  `.trellis/workspace/codex/journal-8.md`.
- Current source files to inspect before implementation:
  `lib/Transforms/EmitCLowerableMaterialization.cpp`,
  `lib/Transforms/EmissionReadiness.cpp`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `lib/Plugin/RVV/RVVSelectedBoundarySeed.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, and
  `tools/tcrv-translate/tcrv-translate.cpp`.
- Current tests to inspect before implementation:
  `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir`,
  `test/Target/RVV/emitc-to-cpp-handoff.mlir`, and
  `test/Target/RVV/i32m1-add-object-artifact.mlir`.

## Definition Of Done

- The accepted source-derived RVV dispatch case has an explicit, verified
  materialized MLIR EmitC handoff before C/C++ emission and target artifact
  packaging.
- Obsolete metadata/direct-C/descriptor authority is not restored or expanded.
- Focused lit and relevant C++ checks pass, with self-repair documented.
- Trellis task state and workspace journal are updated truthfully.
- The task is finished and archived if complete.
- One coherent commit records the completed round, or the task remains open
  with an exact continuation point.

## Implementation Summary

- `lib/Target/TargetArtifactExport.cpp` now verifies selected EmitC artifact
  handoff materialization before C/C++ emission or object packaging. The common
  helper requires an EmitC function boundary, route source-op provenance, and
  call source-op provenance in the materialized EmitC module.
- `materializeSelectedEmitCArtifactModule` is now the shared target-side
  handoff gate for selected EmitC artifact export. RVV object export already
  calls `emitSelectedEmitCArtifactCppSource`, so the production object route now
  materializes and verifies the common EmitC handoff before invoking the MLIR
  EmitC C/C++ emitter and clang RISC-V object packaging.
- `test/Target/TargetArtifactExportTest.cpp` now proves the common selected
  EmitC front door preserves route provenance in emitted C/C++ and fails closed
  when the route builder omits route source-op provenance.
- `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir` now checks the
  source-seed selected-dispatch path through materialized EmitC into the MLIR
  EmitC C/C++ emitter output before target object/header/bundle checks.
- `.trellis/spec/lowering-runtime/emitc-route.md` records the executable
  contract: selected artifact routes that emit C/C++ or package objects from
  EmitC must validate materialized route/call provenance before emitter or
  packaging steps.

## Validation

- Focused build:
  `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`.
- Focused C++ tests:
  `tianchenrv-emission-readiness-test`,
  `tianchenrv-emitc-lowerable-interface-test`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-rvv-dialect-test`, and
  `tianchenrv-target-artifact-export-test`.
- Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-i32m1-selected-boundary-seed|emitc-to-cpp-handoff|i32m1-add-object-artifact|i32m1-selected-dispatch-artifact|i32m1-object-stale-route-op|i32m1-object-missing-contract-metadata|i32m1-artifact-ambiguous-selected|source-seed-artifact-front-door'`
  from `build/test`, 11/11 selected tests passed.
- Full check:
  `cmake --build build --target check-tianchenrv -j2`, 110/110 lit tests
  passed.
- `git diff --check` passed.
- Changed-surface scan:
  - no Python files changed;
  - no descriptor-driven compute, direct-C semantic exporter, or source-export
    route was added;
  - common target code adds only extension-neutral EmitC handoff validation;
  - RVV-specific strings appear only in the existing RVV fixture assertions for
    the expected RVV EmitC route evidence.
- Refreshed artifacts under
  `artifacts/tmp/source_seed_emitc_handoff/20260516T163726Z`.
- Real RVV run passed:
  `artifacts/tmp/source_seed_emitc_handoff/20260516T163726Z/ssh_rvv_link_run.log`
  records
  `tcrv_rvv_i32m1_source_emitc_handoff status=PASS n=4 add=[12,6,16,12]`
  with `ssh_status=0`.

## Self-Repair

- `clang-format` was not installed in this environment, so the one affected
  C++ line-wrap was manually normalized after the first build.
- The broad full-check command was first started with an unnecessary build-tool
  suffix; it still completed successfully, and the final reported full check is
  the clean `cmake --build build --target check-tianchenrv -j2` run.

## Status

- Complete. The task can be archived after Trellis finish bookkeeping and one
  coherent commit.
