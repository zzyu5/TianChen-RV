# RVV source-bundle ssh runtime ABI proof

## Goal

Prove that the production RVV source-front-door artifact bundle is an
externally usable runtime ABI artifact. The existing bounded source-vector
i32 add path must emit a bundle containing a RISC-V relocatable object,
a callable C header, and a bundle index; an external C/C++ harness must consume
the generated header and object from that bundle, link on `ssh rvv`, run with
a nontrivial runtime `n` covering at least one full vector chunk plus tail, and
print a clear PASS result.

## What I already know

- Repository state before edits: `/home/kingdom/phdworks/TianchenRV`, clean
  worktree, HEAD `f4c5670 rvv: consolidate selected config vl contract`.
- No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
- The previous archived task consolidated the bounded RVV i32m1 SEW32/LMUL m1
  tail-agnostic/mask-agnostic runtime AVL/VL contract across the RVV dialect,
  source front door, construction protocol, EmitC route payload, and target
  artifact metadata.
- The previous task explicitly did not rerun `ssh rvv`; this task must prove
  that the source MLIR path carries the shared runtime ABI into generated
  object/header/bundle artifacts that an external harness can link and run on
  real RVV hardware.
- There is no separate `.trellis/spec/target-artifacts/` layer in the current
  tree. Target artifact and bundle rules for this task are in
  `.trellis/spec/lowering-runtime/emitc-route.md`.

## Requirements

1. Use the existing production source artifact pipeline:
   `func/scf/vector source MLIR -> RVV source front door -> selected RVV
   boundary -> plugin-owned materialized EmitC route -> object + callable
   header + bundle`.
2. Keep the module scope to the existing bounded RVV i32m1 add path. Do not add
   new dtypes, LMULs, arithmetic families, generic RVV lowering, scalar fallback
   compute, or new source op families.
3. If the generated header, object, bundle index, symbol name, ordered ABI
   params, config/VL metadata, or object handoff metadata are insufficient for
   an external harness to compile, link, and run, repair the generated
   target/export ABI boundary. Do not hide broken generated output behind an
   out-of-band wrapper.
4. The generated bundle/header metadata must tie together the same selected
   variant, route, runtime ABI name, ordered ABI params, RVV config/VL contract,
   object handoff, and component group.
5. Fail-closed behavior must remain for missing or mismatched materialized
   EmitC provenance, runtime ABI params, config/VL metadata, stale descriptor
   residue, direct-C/source-export residue, and stale source-front-door residue.
6. Any runtime/correctness claim must be backed by real `ssh rvv` compile/link/run
   evidence, not local-only object/header/index shape checks.
7. Python may be used only as tooling to generate/inspect/copy/run artifacts or
   parse results. It must not implement compiler core, lowering, dialect,
   target export, or runtime ABI behavior.

## Acceptance Criteria

- [x] A source MLIR fixture using the production source artifact bundle front
      door emits a bundle directory containing a RISC-V relocatable object,
      callable C header, and bundle index.
- [x] A small external harness consumes the generated header and object from
      that bundle, compiles and links on `ssh rvv`, runs with a runtime count
      covering at least one full vector chunk plus tail, validates i32 add
      results, and prints a clear PASS line.
- [x] Bundle/header metadata expose one coherent selected variant, route id,
      runtime ABI name, ordered ABI parameter signature, config/VL contract,
      object handoff kind, and component group.
- [x] Focused lit or C++ tests protect the external runtime ABI boundary where
      repository behavior changed.
- [x] Negative/fail-closed coverage remains for missing or mismatched EmitC
      provenance, runtime ABI params, config/VL metadata, stale descriptor or
      source-export residue, and incompatible artifact metadata.
- [x] Targeted residue scans over touched RVV dialect/plugin/target/tests show
      no replacement descriptor-driven compute path, direct C semantic
      exporter, source-export route, compatibility wrapper, Python
      compiler-core logic, or common/core RVV semantic branch.

## Non-Goals

- No new RVV dtype/LMUL family, new source op family, sub/mul expansion as the
  main result, generic RVV lowering, descriptor tables, descriptor adapters,
  direct C semantic exporters, source-export routes, compatibility wrappers,
  legacy modes, Python compiler-core behavior, scalar fallback compute, common
  or core RVV semantic branches, broad smoke matrices, or new artifact ledgers.
- No rebuild work outside the existing materialized EmitC object/header/bundle
  route unless the generated route is currently insufficient for the external
  ABI proof.

## Expected Checks

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-rvv-source-bundle-ssh-runtime-abi-proof`
- Focused `tcrv-opt` / `tcrv-translate` commands for source bundle generation.
- Object/header/bundle inspection proving the harness consumes generated
  artifacts directly from the emitted bundle.
- `ssh rvv` compile/link/run of the external harness with captured PASS output.
- Focused lit/C++ tests for changed target/runtime ABI boundary behavior.
- Relevant RVV source/front-door target artifact tests and bundle export tests.
- Targeted residue scans over touched RVV dialect/plugin/target/tests for
  descriptor route authority, direct-C semantic exporter, source-export route,
  compatibility wrappers, direct microkernel residue, and unsafe old source
  shape.
- `git diff --check`.
- `cmake --build build --target check-tianchenrv -j2` if practical.

## Definition of Done

- The production RVV source bundle artifacts are proven externally usable on
  real RVV hardware through `ssh rvv`.
- The generated artifact ABI remains coherent, selected-candidate driven, and
  materialized EmitC backed.
- The task status, journal, archive state, and final commit truthfully reflect
  the completed scope and any remaining rebuild gaps.

## Technical Notes

- Specs read for PRD: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/implementation-stack/index.md`.
- Previous archived PRD read:
  `.trellis/tasks/archive/2026-05/05-18-rvv-selected-config-vl-contract-consolidation/prd.md`.
- Initial code targets from the brief:
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `test/Target/RVV/vector-source-target-artifact-exporters.mlir`, and
  `test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir`.

## Completion Notes

- No compiler source change was required. The current production source
  artifact bundle front door already emits an externally consumable RVV i32m1
  add object/header/bundle; this round refreshed the missing real `ssh rvv`
  evidence after commit `f4c5670`.
- Generated bundle evidence path:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260518T-rvv-source-bundle-ssh-runtime-abi-proof-add`.
- The harness consumed these generated bundle artifacts directly:
  - object:
    `add/generated_bundle/artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o`
  - header:
    `add/generated_bundle/artifact-1-runtime-callable-c-header-rvv-i32m1-arithmetic-emitc-route-family.header.h`
  - index:
    `add/generated_bundle/tianchenrv-target-artifact-bundle.index`
  - external harness:
    `add/rvv_generated_bundle_abi_add_harness.c`
- Source bundle generation command recorded in evidence:
  `build/bin/tcrv-translate --tcrv-source-artifact-bundle-front-door --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_generated_bundle_abi_e2e/20260518T-rvv-source-bundle-ssh-runtime-abi-proof-add/add/generated_bundle test/Transforms/RVV/rvv-i32m1-vector-source-front-door.mlir`.
- Remote compile/link environment recorded:
  `remote_arch=riscv64`, `clang_path=/usr/bin/clang`,
  `clang_version=Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Remote run output:

```text
add case n=7 ok
add case n=16 ok
add case n=23 ok
tcrv_rvv_generated_bundle_abi_add_ok counts=7,16,23
PASS op=add counts=7,16,23
```

- The bundle index tied both object and header records to
  `selected_variant = @vector_source_rvv_i32_add`,
  `route = rvv-i32m1-arithmetic-emitc-route-family`,
  `runtime_abi_name = rvv-i32m1-add-callable-c-abi.v1`,
  ordered ABI params `lhs,rhs,out,n`,
  `component_group = rvv-i32m1-arithmetic-materialized-emitc-bundle.v1`,
  `handoff_kind = materialized-emitc-cpp-rvv-intrinsic-object`, and the shared
  RVV config/VL metadata contract.
- No descriptor adapter, direct C semantic exporter, source-export route,
  compatibility wrapper, scalar fallback compute, Python compiler-core logic,
  new RVV family, or common/core RVV semantic branch was added.

## Checks Run

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-18-rvv-source-bundle-ssh-runtime-abi-proof`
- `cmake --build build --target tcrv-translate tcrv-opt tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-target-artifact-export-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260518T-rvv-source-bundle-ssh-runtime-abi-proof-dry-add --overwrite --op-kind add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260518T-rvv-source-bundle-ssh-runtime-abi-proof-add --overwrite --op-kind add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- `cd build/test && python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-rvv.mlir ../test/Target/RVV/vector-source-target-artifact-exporters.mlir ../test/Scripts/rvv-generated-bundle-abi-e2e-self-test.test ../test/Scripts/rvv-generated-bundle-abi-e2e-source-family-dry-run.test ../test/Target/TargetArtifactBundleExport/source-artifact-bundle-front-door-fail-closed.mlir`
- `git diff --check`
- Targeted residue scan over
  `scripts/rvv_generated_bundle_abi_e2e.py`, `lib/Target/RVV`,
  `test/Target/RVV`, `test/Target/TargetArtifactBundleExport`,
  `test/Scripts`, and `test/Transforms/RVV` for descriptor/direct-C/
  source-export/compatibility/manual-pipe residue. Matches were limited to
  evidence-tool negative guards, fail-closed target checks, FileCheck
  `implicit-check-not` guards, and retained debug two-step fixtures.
- `cmake --build build --target check-tianchenrv -j2` passed 125/125.

## Self-Repair Notes

- None. No source or test behavior had to be repaired; current production
  artifacts satisfied the external runtime ABI proof.
