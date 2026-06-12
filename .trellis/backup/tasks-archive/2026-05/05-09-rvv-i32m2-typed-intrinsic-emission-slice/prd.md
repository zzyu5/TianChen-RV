# RVV i32m2 typed intrinsic emission slice

## Goal

Make one finite additional RVV i32 LMUL shape real across the existing
tcrv_rvv dialect, RVV plugin boundary, target-owned C/header/object exporter,
and focused artifact tests. The slice adds bounded SEW=32, LMUL=m2 support for
the existing i32 add/sub/mul microkernel family while preserving current
!tcrv_rvv.i32m1 behavior and avoiding any broad generic RVV lowering claim.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- The worktree was clean before this task was created.
- `.trellis/.current-task` was absent, so this task was created explicitly as
  `.trellis/tasks/05-09-rvv-i32m2-typed-intrinsic-emission-slice/`.
- The previous task
  `.trellis/tasks/archive/2026-05/05-09-rvv-i32-lmul-policy-intrinsic-emission-contract/`
  is finished and archived and must not be reopened.
- The previous task added `RVVIntrinsicConfig` and body-driven e32/m1 intrinsic
  derivation, but its completion evidence explicitly left m2 blocked by ODS
  type support, verifier checks, plugin materialization, capability/proposal
  legality, suffix generation, and source/header/object tests.
- Current `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` exposes only
  `!tcrv_rvv.i32m1` as the finite i32 vector dataflow type.
- Current `lib/Dialect/RVV/IR/RVVDialect.cpp` verifies `setvl` /
  `with_vl` LMUL as `"m1"` and dataflow operands/results as
  `!tcrv_rvv.i32m1`.
- Current `lib/Plugin/RVV/RVVExtensionPlugin.cpp` proposal, legality, and
  materialization are tied to the m1 config capability ids.
- Current `lib/Target/RVV/RVVMicrokernel.cpp` can derive m1 intrinsic spellings
  from validated typed body metadata, but rejects any non-m1 LMUL before
  emission.

## Requirements

- Keep implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only remain tooling/orchestration.
- Keep `tcrv.exec` compute-free. RVV-specific typing, legality,
  materialization, and emission behavior must stay in `tcrv_rvv`, RVV plugin,
  or RVV target/export code.
- Add the smallest finite representation needed for
  `!tcrv_rvv.i32m2` dataflow values. Do not add new dtypes, m4/m8/fractional
  LMULs, or a generic RVV vector type system.
- Allow SEW=32 LMUL=m1 and LMUL=m2 in `tcrv_rvv.setvl` and
  `tcrv_rvv.with_vl`; reject any other LMUL.
- Verify that i32 load/arithmetic/store operands/results agree with the
  enclosing `tcrv_rvv.with_vl` LMUL. M1/M2 mismatches must fail before target
  artifact emission.
- Extend RVV plugin capability/proposal/materialization only as needed so m2 is
  capability-gated and plugin-local. Existing m1 proposal/materialization must
  remain stable.
- Extend target-owned `RVVIntrinsicConfig` or equivalent derivation so m2 emits
  `vint32m2_t`, `e32m2`, `i32m2`, and the corresponding load/store/arithmetic
  RVV C intrinsic suffixes from typed RVV IR metadata.
- Ensure selected variant capability metadata and typed RVV body LMUL agree
  before source/header/object output. Stale selected m1 versus body m2, or m2
  versus body m1, must fail closed.
- Keep implementation family-generic for the existing bounded i32 add/sub/mul
  family. At minimum prove a non-add family, preferably vsub or vmul, through
  source/header/object tests.

## Acceptance Criteria

- `!tcrv_rvv.i32m2` parses/prints/verifies in focused dialect tests.
- A valid SEW=32 LMUL=m2 `tcrv_rvv.setvl` + `tcrv_rvv.with_vl` body with
  i32 load/sub/store typed as `!tcrv_rvv.i32m2` passes.
- A with_vl/dataflow LMUL mismatch fails in the RVV dialect verifier with a
  bounded diagnostic.
- RVV plugin proposal/legality can select m2 when the target capability facts
  satisfy the finite m2 config ids, while existing m1 tests remain valid.
- Plugin materialization emits m2 `setvl`, `with_vl`, and typed i32m2
  load/arithmetic/store body when the selected variant requires the m2 config.
- Target source export for at least one non-add m2 family emits `vint32m2_t`,
  `__riscv_vsetvl_e32m2`, `__riscv_vle32_v_i32m2`,
  `__riscv_vsub_vv_i32m2` or `__riscv_vmul_vv_i32m2`, and
  `__riscv_vse32_v_i32m2`.
- Target header/object export tests cover the same m2 route sufficiently to
  prove the target artifact path consumes the typed m2 body.
- Existing i32m1 add/sub/mul source/header/object behavior is preserved.
- No new RVV runtime/correctness/performance claim is made unless real
  `ssh rvv` evidence is collected and saved under `artifacts/tmp/...`.

## Out of Scope

- New dtypes beyond i32.
- LMUL m4/m8/fractional support.
- Generic vector dialect lowering or LLVM/RISC-V lowering.
- New high-level tensor/tile compute semantics.
- RVV-specific branches in core orchestration passes.
- Performance claims, broad correctness claims, or evidence-wrapper-only work.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/plugin-protocol/extension-plugin-integration.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/guides/index.md`
  - `.trellis/spec/guides/capability-first-design-guide.md`
  - `.trellis/spec/guides/plugin-locality-review-guide.md`
  - `.trellis/spec/guides/compute-boundary-review-guide.md`
- Previous task PRD/check read:
  - `.trellis/tasks/archive/2026-05/05-09-rvv-i32-lmul-policy-intrinsic-emission-contract/prd.md`
  - `.trellis/tasks/archive/2026-05/05-09-rvv-i32-lmul-policy-intrinsic-emission-contract/check.jsonl`
- Primary implementation surfaces:
  - `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
  - `lib/Dialect/RVV/IR/RVVDialect.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVCapabilityProfile.h`
  - `lib/Plugin/RVV/RVVCapabilityProfile.cpp`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - focused RVV dialect, plugin, and target tests.

## Validation Plan

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-i32-binary-family-registry-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- Focused lit for `test/Dialect/RVV`, `test/Plugin/rvv-extension-plugin.test`,
  and `test/Target/RVVMicrokernel` covering m1 compatibility, m2 positive
  source/header/object emission, and mismatch failures.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  before finish/archive if local build state supports it.

## Completion Notes

- Added finite `!tcrv_rvv.i32m2` typing and verifier support for the bounded
  i32 RVV dataflow surface while preserving `!tcrv_rvv.i32m1`.
- Extended `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` verification to admit only
  LMUL `m1` or `m2`, and added verifier checks that load/add/sub/mul/store
  dataflow values agree with the enclosing LMUL metadata.
- Added RVV plugin-local i32m2 capability ids, proposal selection,
  materialization, legality, and explicit emission readiness checks without
  adding RVV-specific behavior to core orchestration.
- Extended the target-owned RVV intrinsic config/exporter to derive m1/m2 C
  vector type and intrinsic suffixes from selected capability metadata and typed
  RVV body metadata, with fail-closed selected-config/body mismatch checks.
- Proved the non-add `i32-vsub` m2 route through source/header/object-focused
  lit tests and plugin materialization tests. Existing i32m1 add/sub/mul
  behavior remains covered by the existing tests.
- Updated the RVV plugin spec to document the finite i32m2 slice and the
  exactly-one finite config-shape rule.

## Checks Run

- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-i32-binary-family-registry-test -j2`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-emission-readiness-test tianchenrv-rvv-extension-plugin-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emission-readiness-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='(EmissionReadiness/emission-readiness|PluginVariantLegality/plugin-variant-legality-pass-invalid|Dialect/RVV|Plugin/rvv-extension-plugin|Target/RVVMicrokernel)'`
  from `artifacts/tmp/tianchenrv-build/test`
- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`

No new `ssh rvv` evidence was collected, so this task makes no new RVV
runtime, correctness, or performance claim.
