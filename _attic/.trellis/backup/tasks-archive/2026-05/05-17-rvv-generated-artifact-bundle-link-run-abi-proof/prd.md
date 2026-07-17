# RVV Generated Artifact Bundle Link-And-Run ABI Proof

## Goal

Prove that the bounded source-seed selected RVV i32m1 add path produces a real
external runtime ABI product: the compiler-generated target artifact bundle
must contain a coherent declaration header and relocatable object that an
external harness can include, link, and run on the real `ssh rvv` RVV 1.0
machine with a checked `PASS`.

The route under proof is:

```text
source MLIR
  -> selected RVV i32m1 add boundary
  -> RVV extension-family ops
  -> materialized EmitC module
  -> generated object + common declaration header bundle
  -> external harness compile/link
  -> ssh rvv PASS
```

## Current Repository Facts

- The previous completed task adopted the common materialized EmitC header
  helper for the RVV selected i32m1 route while preserving the RVV object
  exporter and object/header bundle composition.
- `lib/Target/RVV/RVVTargetSupportBundle.cpp` registers the RVV object route
  `rvv-i32m1-arithmetic-emitc-route-family` and the composite header route
  `rvv-i32m1-arithmetic-emitc-route-family.header`.
- The object route compiles MLIR EmitC C/C++ emitter output into a RISC-V ELF
  relocatable object through local clang with `-target riscv64 -march=rv64gcv
  -mabi=lp64d`.
- The header route delegates to the common declaration-only
  `exportMaterializedEmitCHeaderArtifact` helper and carries the ordered ABI
  parameters `lhs`, `rhs`, `out`, and `n`.
- Existing lit coverage checks local object bytes, header declaration evidence,
  and bundle index coherence, but it does not yet prove external compile/link
  and execution of the generated bundle on `ssh rvv`.
- RVV runtime/correctness claims require real `ssh rvv` evidence. Local object
  generation, header checks, or dry-run scripts alone are not runtime evidence.

## Requirements

- Use the existing source-seed pipeline and selected materialized RVV add
  candidate; do not introduce a new source route or new RVV arithmetic family.
- Export the generated target artifact bundle through the existing
  `tcrv-translate --tcrv-export-target-artifact-bundle` front door.
- Consume exactly the generated bundle object and generated bundle header for
  the selected add variant. The harness may be handwritten, but the compute
  function body must come only from the generated object.
- The harness must include the generated header and call the generated
  function declaration using the emitted ABI:
  `const int32_t *lhs`, `const int32_t *rhs`, `int32_t *out`, `size_t n`.
- The external compile/link/run proof must execute on `ssh rvv` and report a
  checked `PASS` for i32 add over at least one runtime `n` large enough to
  exercise the generated multi-VL loop.
- Evidence must record the generated object/header/index names, compile/link
  commands, `llvm-readobj` or equivalent object inspection, and remote runtime
  output in an artifact directory under `artifacts/tmp`.
- Any local evidence tooling added in this round is runner/probe/artifact
  tooling only. It must not implement compiler IR, dialects, passes, plugin
  registry behavior, capability modeling, lowering, emission, runtime ABI
  semantics, or descriptor adapters in Python.
- Keep stale descriptor/source-export/direct-C/header-route compatibility
  rejected. Do not restore old RVV direct microkernel route ids, source
  exporters, direct C semantic printers, compatibility wrappers, or descriptor
  authority.

## Acceptance Criteria

- [ ] Bundle export produces exactly one object component and one header
      component for the selected RVV i32m1 add route.
- [ ] The consumed object file is the generated
      `artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o`.
- [ ] The consumed header file is the generated
      `artifact-1-runtime-callable-c-header-rvv-i32m1-arithmetic-emitc-route-family.header.h`.
- [ ] The bundle index records the coherent component group
      `rvv-i32m1-arithmetic-materialized-emitc-bundle.v1`, external ABI name
      `rvv-i32m1-add-callable-c-abi.v1`, runtime ABI kind/name, and ordered
      `lhs`, `rhs`, `out`, `n` parameters for the bundle.
- [ ] An external harness including the generated header and linking the
      generated object compiles on `ssh rvv` with clang/LLVM RVV flags.
- [ ] The remote executable runs on `ssh rvv` and prints a checked `PASS` for
      add.
- [ ] Focused lit or C++ coverage still guards local source-seed bundle export,
      header/object coherence, and absence of descriptor/direct-C/source-export
      residue.
- [ ] Targeted scans show no descriptor route authority, direct C semantic
      exporter, source-export route, legacy route alias, or core/common RVV
      semantic branch was introduced.

## Out Of Scope

- No new RVV arithmetic families, sub/mul source-seed expansion, generic RVV
  lowering, generic runtime integration, scalar fallback compute, dispatch
  expansion, performance measurement, or broad correctness claim.
- No new bundle index protocol, artifact ledger, checkpoint protocol,
  compatibility wrapper, descriptor adapter, direct C semantic exporter,
  source-export route, or Python compiler-core behavior.
- No handwritten C compute body. Only the external caller/harness around the
  generated header/object may be handwritten for evidence.
- No common/core branch that infers RVV semantics from family names, route ids,
  metadata keys, file names, or artifact paths.

## Minimal Evidence

- `git diff --check`.
- Focused build for `tcrv-opt`, `tcrv-translate`, RVV target support/plugin
  tests, and target artifact export tests.
- Focused lit for RVV source-seed target artifact object/header/bundle checks.
- Generated artifact directory containing:
  - emitted header;
  - emitted object;
  - bundle index;
  - external harness source;
  - compile/link command record;
  - readobj output;
  - `ssh rvv` run log.
- Focused scans over target/plugin/translate/test paths for descriptor,
  direct-C, source-export, legacy route alias, and core/common RVV semantic
  branch residue.
- `check-tianchenrv` if practical after focused checks pass.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/validation/index.md`.
- Previous task PRD read:
  `.trellis/tasks/archive/2026-05/05-17-rvv-common-materialized-emitc-header-adoption/prd.md`.
- Main files to inspect or touch:
  `lib/Target/TargetArtifactExport.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `test/Target/RVV/source-seed-target-artifact-header.mlir`,
  `test/Target/RVV/source-seed-target-artifact-object.mlir`, and
  `test/Transforms/RVV/rvv-i32m1-selected-boundary-seed.mlir`.
