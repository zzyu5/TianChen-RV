# RVV materialized EmitC target artifact bridge

## Goal

Rebuild one bounded RVV target artifact bridge after descriptor-route erasure.
The bridge must accept only a verified, already materialized RVV EmitC module,
run it through the MLIR EmitC C/C++ emitter path, and package a real target
artifact path without restoring RVV descriptor/table route authority.

## What I Already Know

- Current HEAD is `8acbfff rvv: erase descriptor target routes`; the worktree
  was clean before this task started.
- The previous deletion task removed the RVV Target/RVV i32m1 add/sub/mul
  descriptor table and old object/header/bundle route fixtures.
- The previous materialized EmitC handoff task proved a selected RVV i32m1 add
  path can produce a verified materialized MLIR EmitC module before invoking
  MLIR EmitC C/C++ emission.
- `.trellis/spec/index.md` defines the current route as extension-family ops to
  EmitC ops to C/C++ emitter to intrinsic/runtime/native compiler, and rejects
  descriptor-driven computation.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires selected artifact
  routes that emit or package from EmitC to materialize the selected route
  through the common EmitC materialization helper and verify the materialized
  module before emitter/toolchain steps.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` says direct
  metadata-driven C exporters are deleted or fail-closed debt until a real
  materialized MLIR EmitC module route exists.
- `.trellis/spec/extension-plugins/rvv-plugin.md` permits the bounded explicit
  RVV i32m1 add/sub/mul executable slice only through selected emission-plan
  target artifact handoff, explicit typed RVV body, validated lowering
  boundary, RVV-owned EmitC lowerable route, MLIR EmitC C/C++ emitter, and
  clang RISC-V artifact packaging.

## Requirements

- Rewire the RVV target artifact/export production path so a selected RVV
  i32m1 source/typed-op path can reach a generated target artifact only after
  materialized EmitC verification and MLIR EmitC C/C++ emission.
- Keep Target/RVV descriptor route tables, descriptor adapters, legacy route
  ids, old object/header/bundle route fixtures, and compatibility wrappers
  deleted.
- The bridge may validate selected RVV materialized EmitC provenance, invoke
  the common EmitC-to-C/C++ emitter route, and package or compile the emitted
  source for bounded artifact evidence.
- The bridge must not synthesize compute bodies from selected metadata,
  descriptors, route ids, family registries, status records, or per-op tables.
- Unsupported, stale, ambiguous, non-selected, or non-materialized inputs must
  fail closed before C/C++ emission or artifact packaging.
- Keep implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python is allowed only for existing test runners or artifact
  parsing, not compiler core behavior.

## Acceptance Criteria

- [x] A selected RVV i32m1 source/extension-op path reaches a generated target
  artifact through verified materialized EmitC and MLIR C/C++ emission.
- [x] The artifact bridge is not owned by per-op descriptor/table route
  authority and does not restore deleted descriptor route names or old
  fixtures.
- [x] Non-materialized, unsupported, stale, or ambiguous inputs fail closed
  before source emission or artifact packaging.
- [x] Focused C++ and lit checks prove the production/default path changed,
  rather than only adding helper-only tests.
- [x] Targeted scans show no restored
  `RVVI32M1ArithmeticTargetRouteDescriptor`,
  `kRVVI32M1ArithmeticTargetRoutes`, old RVV object/header/bundle route ids,
  descriptor adapters, direct C semantic exporters, or Python compiler-core
  logic in the touched surfaces.
- [x] If RVV runtime correctness or object usability is claimed, focused
  `ssh rvv` compile/run evidence is recorded.

## Non-Goals

- Do not restore `RVVI32M1ArithmeticTargetRouteDescriptor`,
  `kRVVI32M1ArithmeticTargetRoutes`, `getRVVI32M1Arithmetic*RouteID`, old
  object/header/bundle route ids, descriptor adapters, legacy mode, direct C
  semantic exporters, or source strings generated from metadata.
- Do not add generic RVV family expansion, new SEW/LMUL coverage, new
  extension work outside the bounded i32m1 path, independent-backend wording,
  or a new artifact ledger/checkpoint protocol.
- Do not treat prompt edits, reports, helper-only changes, broad smoke tests,
  or status metadata as the main result.

## Technical Notes

- Read first:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/tasks/archive/2026-05/05-17-rvv-target-descriptor-route-erasure/prd.md`.
- Primary code surfaces:
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`, and
  `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir`.
- Relevant recent evidence is in `.trellis/workspace/codex/journal-8.md`
  sessions for source-seed selected-dispatch artifact, materialized EmitC
  handoff, and RVV target descriptor-route erasure.

## Definition Of Done

- Production/default RVV artifact export uses the materialized EmitC bridge
  and no descriptor/table route authority.
- Focused build, C++ tests, lit tests, `git diff --check`, and targeted scans
  are recorded.
- `ssh rvv` compile/run evidence is recorded if object/runtime usability is
  claimed; otherwise the exact blocker is stated.
- Trellis task status and workspace journal are updated truthfully.
- One coherent commit records the completed bridge, or the task remains open
  with the exact continuation point.

## Implementation Summary

- Changed RVV selected emission readiness/planning from deleted-route
  unsupported diagnostics to a supported family-level route:
  `rvv-i32m1-arithmetic-emitc-route-family`.
- Kept the per-op RVV EmitC lowerable routes inside the RVV provider as
  provider-owned intrinsic mapping; Target/RVV registers one family-level
  target artifact exporter and does not restore old target route tables or
  descriptor accessors.
- Added Target/RVV candidate preflight for selected origin, emission kind,
  artifact kind, lowering boundary, runtime ABI identity, ordered ABI
  parameters, and `rvv_emitc_lowerable_route` provenance.
- Reused the common selected EmitC artifact bridge:
  selected emission-plan candidate -> provider-built
  `TCRVEmitCLowerableRoute` -> verified materialized MLIR EmitC module ->
  MLIR EmitC C/C++ emitter.
- Added clang-based RISC-V object packaging for the MLIR-emitted RVV C/C++
  source. The exporter searches `PATH` and standard LLVM/system tool paths for
  `clang`.
- Updated durable specs under `.trellis/spec/extension-plugins/rvv-plugin.md`
  and `.trellis/spec/lowering-runtime/emission-runtime-contract.md` to reflect
  the rebuilt materialized EmitC object route while keeping direct/descriptor
  RVV routes deleted.

## Validation

- Focused build:
  `cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tcrv-opt tcrv-translate -j2`.
- Focused C++ tests:
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-construction-protocol-common-test`.
- Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-i32m1-selected-boundary-seed|source-seed-target-artifact-object|emitc-to-cpp-handoff|emitc-to-cpp-non-materialized|source-seed-artifact-front-door|toy-template-target-artifact'`
  from `build/test`, 14/14 selected tests passed.
- Full check:
  `cmake --build build --target check-tianchenrv -j2`, 101/101 lit tests
  passed.
- `git diff --check` passed.
- Targeted scans found no restored
  `RVVI32M1ArithmeticTargetRouteDescriptor`,
  `kRVVI32M1ArithmeticTargetRoutes`,
  `getRVVI32M1Arithmetic*Target*RouteID`,
  old `tcrv-rvv-i32m1-{add,sub,mul}` target route strings, deleted-route
  diagnostic text, descriptor adapters, legacy mode, source-authority APIs, or
  direct C semantic exporter residue in the touched RVV target/plugin/test
  surfaces.
- Artifact evidence:
  `artifacts/tmp/rvv_materialized_emitc_target_artifact_bridge/20260516T174758Z-ok`
  contains `source_seed_selected_plan.mlir`, `materialized_rvv_emitc.cpp`,
  `rvv_target_artifact.o`, `rvv_target_artifact.readobj.txt`,
  `rvv_harness.cpp`, and `ssh_rvv_compile_run.txt`.
- Local object evidence:
  `llvm-readobj -h rvv_target_artifact.o` reports
  `Format: elf64-littleriscv`, `Arch: riscv64`, and `Type: Relocatable`.
- Real `ssh rvv` compile/run evidence:
  `tcrv_rvv_materialized_emitc_target_artifact status=PASS n=4 add=[12,6,16,12]`.

## Self-Repair

- The first source sidecar generation tried to run the bounded
  `--tcrv-materialize-emitc-lowerable-routes` pass on a dispatch+fallback
  source-seed module. That correctly failed because the pass still requires
  one direct variant. The artifact evidence now uses the source-seed pipeline
  for the selected target object and the retained materialized EmitC handoff
  fixture for source provenance.
- Direct shell artifact generation initially failed because `clang` was not on
  the normal shell `PATH` even though lit injects the LLVM tools directory.
  The exporter now searches `PATH`, `/usr/lib/llvm-20/bin`, `/usr/local/bin`,
  and `/usr/bin`.
- A newly added target route accessor was renamed from the forbidden
  `getRVVI32M1Arithmetic*RouteID` shape to
  `getRVVMaterializedEmitCTargetArtifactRouteID`.

## Spec Update Judgment

Spec update was required because this round changes the cross-layer RVV
emission/runtime contract from deleted-route unsupported diagnostics to a
bounded supported object route for explicit typed RVV i32m1 arithmetic. The
updated specs keep historical descriptor/direct-C routes deleted and document
the materialized EmitC artifact bridge as the only active rebuild authority.

## Status

Complete. The task can be archived after Trellis finish bookkeeping and one
coherent commit.
