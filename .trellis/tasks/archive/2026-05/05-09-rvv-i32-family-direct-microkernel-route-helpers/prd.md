# RVV i32 family direct microkernel route helpers

## Goal

Make the bounded RVV i32 add/sub/mul direct microkernel artifact route ids
callable through matching `tcrv-translate` direct helpers. The compiler already
has target-owned route ids for the finite family in the i32 binary registry; this
task closes the command-facing gap so source/header/object route ids and public
translation registrations stay aligned, with stale-family inputs failing before
artifact output.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current HEAD is `f0a50d0 feat(rvv): add i32 microkernel family evidence bridge`.
- `.trellis/.current-task` was empty before this task was created.
- Latest complete Hermes audit is
  `artifacts/tmp/hermes_codex_supervisor/runs/20260509T091344Z-r0013-20260509T140824Z/repo_audit.md`.
- Recent compiler work generalized the bounded i32 family through descriptor
  registry, linalg lowering, RVV/scalar dispatch routes, direct RVV microkernel
  artifact export, and evidence runners.
- `include/TianChenRV/Target/I32BinaryFamilyRegistry.h` defines RVV direct
  route ids for add/sub/mul source/header/object artifacts.
- `tools/tcrv-translate/tcrv-translate.cpp` currently exposes generic direct
  RVV microkernel helpers, but not family-specific route-id helpers for
  `i32-vsub` and `i32-vmul`.
- The relevant spec requires compiler/tool registrations to use the same
  finite family route registry/manifest surface and to fail closed on stale
  family mismatches.

## Requirements

- Keep implementation in C++ / MLIR / LLVM / CMake / lit / FileCheck.
- Do not add Python compiler semantics, helper-only achievements, smoke-only
  work, or broad runtime claims.
- Preserve `tcrv.exec` as compute-free; RVV behavior must remain plugin-local
  and target-owned emission must remain in target/export code.
- Register command-facing RVV microkernel source/header/object direct helpers
  from the finite i32 add/sub/mul registry, not by unrelated ad hoc lists.
- Keep the existing generic direct helpers working for the selected family.
- Add route-specific family validation so a family-specific helper rejects a
  selected record from a different family before printing C/header/object bytes.
- Keep object routes switching stdout to binary only for object output.
- Preserve the bounded RVV C-intrinsic route:
  `tcrv_rvv` structured IR/dataflow -> target-owned C source/header/object
  exporter -> `riscv_vector.h` RVV C intrinsics -> optional `ssh rvv` evidence.

## Acceptance Criteria

- `tcrv-translate` exposes route-id helpers for RVV i32 add/sub/mul
  source/header/object exports.
- `--tcrv-export-rvv-i32-vsub-microkernel-c` emits vsub source from a selected
  vsub input and does not emit stale vadd/vmul identity.
- `--tcrv-export-rvv-i32-vmul-microkernel-c` emits vmul source from a selected
  vmul input and does not emit stale vadd/vsub identity.
- At least one stale-family direct helper negative proves route-specific
  validation fails before artifact output.
- C++ registry coverage proves standalone RVV family source/header/object routes
  are registered consistently with runtime ABI/component metadata.
- Focused lit/FileCheck coverage passes for the changed route helpers.
- `git diff --check` passes.
- The task is finished and archived only after validation.

## Out of Scope

- Generic RVV lowering beyond the finite i32 add/sub/mul microkernel family.
- MLIR vector dialect or LLVM/RISC-V intrinsic-level lowering.
- Runtime integration, dynamic probing, benchmarking, or performance claims.
- New broad evidence runner work unless a route-specific validation failure
  directly blocks this compiler path.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/implementation-stack/index.md`
  - `.trellis/spec/implementation-stack/compiler-stack-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/index.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Likely implementation surfaces:
  - RVV target exporter API and route-local validation.
  - `tcrv-translate` translation registration.
  - RVV microkernel route lit tests.
  - i32 binary family registry C++ test coverage.
