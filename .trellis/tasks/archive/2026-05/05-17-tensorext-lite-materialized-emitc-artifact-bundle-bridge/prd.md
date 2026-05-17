# TensorExtLite Materialized EmitC Artifact Bundle Bridge

## Goal

Package the existing TensorExtLite first-slice materialized EmitC object route
and its declaration-only ABI header route into one coherent target artifact
bundle. The bundle must be derived from the same selected supported
TensorExtLite materialized EmitC object candidate:

```text
source-front-door
  -> selected TensorExtLite role ops
  -> plugin-owned EmitC route
  -> materialized MLIR EmitC module
  -> MLIR EmitC C/C++ emitter
  -> clang RISC-V relocatable object
  -> object + declaration-only header + target artifact bundle
```

This task proves bundle packaging and candidate identity only. It does not
claim TensorExtLite runtime execution, hardware correctness, performance,
vendor intrinsic behavior, or broader TensorExt lowering.

## Current Repository Facts

- Current HEAD before this task is `e7015b4 tensorext: add materialized emitc
  object artifact`, with a clean worktree.
- No `.trellis/.current-task` existed before this round; this task was created
  from the Hermes Direction Brief as
  `05-17-tensorext-lite-materialized-emitc-artifact-bundle-bridge`.
- The previous TensorExtLite object task made the selected materialized EmitC
  object route real and kept the header declaration-only through an
  object-backed composite header route.
- The generic target artifact bundle layer already writes deterministic bundle
  files and an index from registry-derived object/header records.
- TensorExtLite currently registers one object route and one object-backed
  header composite route, but the TensorExtLite header/object records do not
  yet publish a shared component group or bundle metadata that forces downstream
  consumers to treat the object and declaration as one ABI package.
- RVV already uses the expected pattern: the object exporter and header
  composite publish a shared component group and the header composite supplies
  bundle metadata preserving the object handoff identity.
- TensorExtLite has an empty runtime ABI parameter signature and therefore a
  `void function(void)` declaration. The bundle contract must preserve this
  as a valid zero-parameter ABI signature rather than inventing fake runtime
  parameters.

## Requirements

- Add a TensorExtLite-owned materialized EmitC bundle component identity shared
  by the object exporter and declaration-only header composite.
- Preserve the selected object route as the executable artifact authority. The
  header must remain an object-backed declaration-only composite and must not
  become a separate source, compute, descriptor, or object authority.
- Add TensorExtLite route-local bundle metadata so the bundle index preserves
  the TensorExtLite component group, external ABI identity, object handoff kind,
  owner plugin, selected variant, selected role, runtime ABI kind/name, and
  existing candidate artifact metadata.
- Keep object/header candidate identity consistency fail-closed: exactly one
  selected supported TensorExtLite materialized EmitC object candidate must be
  matched before the header or bundle record is accepted.
- Support the valid TensorExtLite zero-argument ABI signature in the generic
  component contract without weakening typed parameter validation for non-empty
  signatures.
- Keep bundle output deterministic and metadata-derived: no absolute paths,
  timestamps, credentials, raw logs, runtime success, correctness, or
  performance fields.
- Do not add new source-output routes, descriptor compatibility, direct C
  semantic exporters, Python compiler-core behavior, or generic/core branches
  on TensorExtLite semantics.

## Acceptance Criteria

- [x] A TensorExtLite source-front-door fixture exports a target artifact bundle
      through `--tcrv-export-target-artifact-bundle` after the materialized
      EmitC path is selected.
- [x] The bundle contains a RISC-V ELF relocatable object and a
      declaration-only header generated from the same selected object
      candidate.
- [x] The bundle index records `artifact_count: 2`, deterministic object/header
      file names, shared TensorExtLite component group, external ABI name,
      owner plugin, selected variant, selected role, route identities, runtime
      ABI kind/name, object handoff kind, evidence roles, and TensorExtLite
      construction/source-op metadata.
- [x] The header remains declaration-only and contains no `__riscv_`, hidden
      `main`, descriptor, source-export, direct-C, source-seed, RVV, or Toy
      residue.
- [x] Focused C++ tests prove TensorExtLite exporter registration shape,
      object/header identity matching, bundle metadata, zero-argument component
      contract acceptance, and fail-closed diagnostics for missing provenance,
      wrong artifact kind, direct-C/descriptor metadata, ambiguous/multiple
      candidates, and mismatched runtime ABI metadata.
- [x] Negative behavior stays fail-closed for fallback-only or no selected
      candidate, stale source-front-door metadata, missing lowering boundary,
      unsupported artifact kind, missing materialized EmitC handoff, and
      source/direct-C/descriptor route residue.
- [x] Focused checks pass for TensorExtLite plugin/target artifact tests and
      focused lit under `test/Target/TensorExtLite`; `check-tianchenrv` runs if
      practical after focused validation.
- [x] Targeted scans over touched TensorExtLite target/plugin/tests and common
      `TargetArtifactExport` / EmitC files show no descriptor route authority,
      direct C semantic exporter, source-export route resurrection, runtime
      correctness claim, or performance claim.

## Completion Evidence

- Focused build passed:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-tensorext-lite-extension-plugin-test -j2`.
- C++ tests passed:
  `./build/bin/tianchenrv-target-artifact-export-test` and
  `./build/bin/tianchenrv-tensorext-lite-extension-plugin-test`.
- Focused lit passed:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'TensorExtLite|tensorext-lite|TargetArtifactBundleExport|vector-source-target-artifact-object'`
  from `build/test`: 15/15 selected tests passed.
- Follow-up focused lit for the new/changed bundle tests passed:
  `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'tensorext-lite-source-front-door-target-artifact-bundle|tensorext-lite-target-artifact-header'`
  from `build/test`: 2/2 selected tests passed.
- Full `check-tianchenrv` passed twice after implementation/test edits:
  112/112 lit tests passed.
- `git diff --check` passed.
- Manual bundle evidence:
  `tcrv-opt test/Transforms/TensorExtLite/tensorext-lite-fragment-mma-source-front-door.mlir --tcrv-source-artifact-front-door-pipeline | tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=<tmp>`
  produced `artifact_count: 2`, shared component group
  `tensorext-lite-fragment-mma-materialized-emitc-bundle.v1`, both object and
  header records with `runtime_abi_parameter_count: 0`, and
  `handoff_kind: "materialized-emitc-cpp-tensorext-lite-fragment-object"`.
- Local object evidence:
  `llvm-readobj-20 -h` reported `Format: elf64-littleriscv`, `Arch: riscv64`,
  and `Type: Relocatable`; `llvm-readobj-20 --symbols` reported the unmangled
  symbol
  `tcrv_emitc_tensorext_lite_header_export_tensorext_lite_tile_mma_first_slice`.
- Targeted scan over touched TensorExtLite target/tests and common
  `TargetArtifactExport` files found descriptor/direct-C/source-export strings
  only in fail-closed validation, `CHECK-NOT` assertions, or explicit
  no-runtime/no-correctness/no-performance boundary text.
- `clang-format` was not installed in this environment; formatting was kept
  manually consistent and validated by build plus `git diff --check`.

## Out Of Scope

- No TensorExtLite runtime execution, link/run harness, correctness claim,
  performance claim, vendor intrinsic semantics, or `ssh rvv` evidence.
- No broader TensorExt families, IME/offload/scalar behavior, CUDA/vendor
  lowering, high-level frontend semantics, linalg/tensor/tile IR work, or new
  plugin templates.
- No replacement architecture beyond this bounded bundle packaging bridge.
- No descriptor-driven computation, descriptor compatibility, direct C semantic
  exporter, source-output route, legacy route id, bundle ledger protocol,
  checkpoint protocol, or common/core semantic branch on TensorExtLite.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous PRD read:
  `.trellis/tasks/archive/2026-05/05-17-tensorext-lite-materialized-emitc-object-artifact-bridge/prd.md`.
- Current code/test surfaces inspected:
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `include/TianChenRV/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.h`,
  `lib/Target/TensorExtLite/TensorExtLiteTargetSupportBundle.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `test/Target/TensorExtLite/`,
  `test/Target/RVV/vector-source-target-artifact-object.mlir`, and
  `tools/tcrv-translate/tcrv-translate.cpp`.
