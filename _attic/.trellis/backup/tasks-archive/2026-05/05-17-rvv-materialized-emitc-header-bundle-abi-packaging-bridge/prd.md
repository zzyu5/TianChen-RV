# RVV materialized EmitC header and bundle ABI packaging bridge

## Goal

Rebuild the declaration-only header artifact and coherent artifact bundle for
the existing bounded RVV materialized EmitC object route. The rebuilt path must
use the same selected emission-plan candidate, plugin-owned materialized EmitC
route, runtime ABI contract, and object handoff identity as the already
supported RVV i32m1 arithmetic object bridge.

## What I Already Know

- Current HEAD is `51e76f9 rvv: rebuild materialized emitc artifact bridge`;
  the worktree was clean before this task started.
- The previous task rebuilt the RVV object path:
  selected emission-plan candidate -> RVV-owned EmitC lowerable route ->
  verified materialized MLIR EmitC module -> MLIR EmitC C/C++ emitter ->
  clang RISC-V relocatable object.
- The previous task kept historical RVV descriptor target routes and old
  `tcrv-rvv-i32m1-{add,sub,mul}` object/header/bundle route ids deleted.
- `.trellis/spec/extension-plugins/rvv-plugin.md` still treats historical
  header and bundle route ids as deleted until rebuilt through the same
  materialized EmitC authority.
- `.trellis/spec/lowering-runtime/emitc-route.md` allows target artifact
  exporters to materialize object, header, bundle, or metadata artifacts from
  selected emission-plan candidates, but requires selected candidate authority,
  runtime ABI checks, and verified materialized EmitC handoff before emitting
  C/C++ or packaging object-like outputs.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` rejects
  descriptor-driven or metadata-driven C body/header/object/bundle exporters.
- Current code has an RVV standalone object exporter registered from
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`; generic target artifact bundle
  code already supports header/object component grouping and runtime ABI
  signature matching.

## Requirements

- Add a declaration-only RVV header artifact route derived from the selected
  RVV materialized EmitC candidate, runtime ABI parameter contract, materialized
  EmitC function boundary, and object route identity.
- Keep the header non-semantic: it may declare the callable function and include
  minimal ABI support headers, but must not embed RVV intrinsic compute bodies,
  loops, self-check harnesses, hardware probing, logs, artifact paths,
  correctness claims, or performance claims.
- Package the existing RVV object route and the new header route into one
  coherent selected-variant artifact bundle group with the same selected
  variant, origin plugin, materialized EmitC route, runtime ABI kind/name,
  ordered runtime ABI parameters, and object handoff identity.
- Reject stale or mixed candidate surfaces: fallback-only selection, missing
  selected materialized EmitC provenance, missing or mismatched runtime ABI
  parameters, mismatched route identity, multiple selected supported candidates,
  historical deleted route ids, and descriptor/source-export/direct-C residue.
- Keep implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only be used for existing tooling or evidence capture,
  not compiler core behavior.

## Acceptance Criteria

- [x] Source-seed selected RVV add path produces a RISC-V relocatable object
  through the existing materialized EmitC object route.
- [x] The same selected path produces a callable header whose prototype uses
  the selected materialized EmitC function name and ordered runtime ABI
  parameter types/names.
- [x] Header output contains declarations/ABI includes only and no RVV
  intrinsic body, `main`, self-check harness, descriptor text, source-export
  text, runtime probing, or evidence logs.
- [x] Target artifact bundle export writes an index and artifact files tying
  object and header to the same selected variant, origin plugin, materialized
  EmitC route, runtime ABI name/kind, runtime ABI parameter order, component
  group, and object handoff.
- [x] Focused C++ tests cover RVV target exporter registration, header
  candidate validation, bundle composition, and fail-closed mismatches.
- [x] Focused lit tests prove source-seed selected dispatch exports object,
  header, and bundle through the materialized EmitC route.
- [x] Negative coverage fails closed for fallback-only selection, missing
  materialized EmitC provenance, runtime ABI parameter mismatch, candidate
  route mismatch, multiple selected supported candidates, deleted historical
  header/bundle route ids, descriptor/source-export/direct-C residue, and
  header content that embeds RVV intrinsic compute bodies.
- [x] Targeted scans show no restored descriptor route authority, no direct C
  semantic exporter, no source-export route, and no tests protecting historical
  header/bundle route ids.

## Non-Goals

- Do not restore historical `tcrv-rvv-i32m1-{add,sub,mul}` header/object/bundle
  route ids.
- Do not add descriptor tables, descriptor adapters, compatibility wrappers,
  legacy mode, direct C semantic exporters, source-export routes, scalar
  fallback compute, new sub/mul source seeds, broader SEW/LMUL families,
  generic RVV lowering, Python compiler-core behavior, or common/core RVV
  semantic branches.
- Do not introduce a new artifact ledger, checkpoint protocol, state machine,
  high-level frontend lowering, or executable plugin template.
- Do not treat reports, helper-only code, prompt edits, broad smoke tests, or
  status metadata as the main achievement.

## Technical Notes

- Read first:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/tasks/archive/2026-05/05-17-05-17-rvv-materialized-emitc-target-artifact-bridge/prd.md`.
- Primary code surfaces:
  `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `lib/Target/TargetArtifactExport.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `test/Target/RVV/source-seed-target-artifact-object.mlir`,
  `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir`, and
  `test/Target/TargetArtifactExportTest.cpp`.

## Definition Of Done

- Production/default RVV target support registers object/header/bundle packaging
  around the same selected materialized EmitC route.
- Focused build, C++ tests, lit tests, `git diff --check`, and targeted scans
  are recorded.
- Fresh `ssh rvv` evidence is recorded if object or bundle packaging changes
  affect runtime usability; otherwise the reason for reusing existing object
  proof is stated.
- Trellis task status and workspace journal are updated truthfully.
- One coherent commit records the completed bridge, or the task remains open
  with the exact continuation point.

## Implementation Summary

- Added the RVV materialized EmitC header route
  `rvv-i32m1-arithmetic-emitc-route-family.header` as a composite target
  artifact route matched from the existing selected RVV object candidate.
- Kept the existing object route id
  `rvv-i32m1-arithmetic-emitc-route-family` as the single selected
  materialized EmitC object authority.
- Added a shared bundle component group
  `rvv-i32m1-arithmetic-materialized-emitc-bundle.v1` so object and header
  records must preserve the same selected variant, runtime ABI kind/name,
  ordered runtime ABI parameters, and object handoff identity.
- Header emission materializes the selected EmitC module, verifies the single
  materialized `emitc.func` boundary, checks arity against ordered runtime ABI
  parameters, and prints only a declaration with `<stddef.h>` / `<stdint.h>`.
- Extended composite bundle metadata with `handoffKind` so the bundle index can
  tie the header component to the same materialized object handoff identity.
- Updated durable specs to state the active bounded object/header/bundle route
  and keep historical descriptor/direct-C/header/bundle ids deleted.

## Validation

- Focused build:
  `cmake --build build --target tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`.
- Focused C++:
  `./build/bin/tianchenrv-target-artifact-export-test`.
- Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'source-seed-target-artifact-header|source-seed-target-artifact-object|rvv-i32m1-selected-boundary-seed|emitc-to-cpp-handoff|emitc-to-cpp-non-materialized'`
  from `build/test`, 6/6 selected tests passed.
- Full check:
  `cmake --build build --target check-tianchenrv -j2`, 102/102 lit tests
  passed.
- `git diff --check` passed.
- Artifact evidence:
  `artifacts/tmp/rvv_materialized_emitc_header_bundle_bridge/20260516T181607Z`
  contains the selected plan, declaration-only header, RISC-V relocatable
  object, bundle index, bundle object/header files, readobj output, and the
  remote compile/run log.
- Local object evidence:
  `/usr/lib/llvm-20/bin/llvm-readobj -h rvv_target_artifact.o` reports
  `Format: elf64-littleriscv`, `Arch: riscv64`, and `Type: Relocatable`.
- Real `ssh rvv` header+object evidence:
  `tcrv_rvv_materialized_emitc_header_bundle status=PASS n=4 add=[12,6,16,12]`.
- Targeted scans found no restored
  `RVVI32M1ArithmeticTargetRouteDescriptor`,
  `kRVVI32M1ArithmeticTargetRoutes`, old
  `getRVVI32M1Arithmetic{Object,Header,Bundle,Target}*RouteID` route accessors,
  historical `tcrv-rvv-i32m1-{add,sub,mul}` header/object/bundle route strings,
  descriptor-driven/direct-C/source-export/source-authority residue in RVV
  target/plugin/test surfaces, or RVV intrinsic/body text in generated header
  artifacts.

## Self-Repair

- The first local evidence command used `llvm-readobj` from the normal shell
  `PATH`; this shell does not expose it. The readobj evidence was regenerated
  with `/usr/lib/llvm-20/bin/llvm-readobj`.

## Status

Complete. The task can be archived after Trellis finish bookkeeping and one
coherent commit.
