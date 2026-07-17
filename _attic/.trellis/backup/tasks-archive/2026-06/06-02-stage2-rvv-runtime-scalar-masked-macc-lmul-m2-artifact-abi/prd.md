# Stage2 RVV runtime-scalar masked MAcc LMUL m2 config artifact ABI boundary

## Goal

Close one bounded Stage 2 RVV end-to-end artifact/runtime ABI boundary for the
existing pre-realized `runtime_scalar_cmp_masked_macc_add` selected body with
`lmul=m2`. The task proves that LMUL is derived from typed `tcrv_rvv`
body/config/capability/runtime facts and consistently controls realized vector
and mask types, route facts, C type and intrinsic-family mirrors, generated
header/prototype facts, target artifact validation, generated-bundle ABI
harness behavior, and real `ssh rvv` execution.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV runtime-scalar masked MAcc LMUL m2 config artifact ABI boundary`

## What I Already Know

* The previous archived task
  `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-runtime-scalar-cmp-masked-macc-add-artifact-abi-boundary/`
  completed the m1 `runtime_scalar_cmp_masked_macc_add` ABI path with
  generated-bundle dry-run tests, target artifact fail-closed validation, and
  real `ssh rvv` correctness.
* The current repository already contains
  `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add-lmul-m2.mlir`.
  That fixture realizes `setvl`, `with_vl`, vector loads, scalar splat,
  compare mask, `masked_macc`, merge/store path facts, vector types, and mask
  types with `lmul = "m2"`.
* `scripts/rvv_generated_bundle_abi_e2e.py` already has an
  `OpExpectation` entry for
  `runtime_scalar_cmp_masked_macc_add_lmul_m2`, and a local dry-run probe can
  generate a bundle for that op when given the versioned LLVM tool path.
* No current `test/Scripts` generated-bundle tests were found for
  `runtime_scalar_cmp_masked_macc_add_lmul_m2` or
  `computed_masked_macc_add_lmul_m2`.
* The target artifact MAcc validator already validates provider-built route
  type mappings, ABI mappings, headers, computed-mask statement shape, RHS
  scalar splat, compare/MAcc/merge/store calls, and exact runtime ABI binding
  summaries. This round should add only directly needed m2 stale-config/type
  validation coverage if the active validator already consumes the facts.

## Requirements

* Scope exactly one production-positive op:
  `runtime_scalar_cmp_masked_macc_add_lmul_m2`.
* The selected body must remain pre-realized input consumed by the RVV
  selected-body realization boundary before provider route construction.
* `cmp_lhs`, `rhs_scalar`, `lhs`, `rhs`, `acc`, `out`, and `n` ABI roles must
  keep the same runtime boundary and order as m1 while vector/mask config
  changes to LMUL m2.
* LMUL m2 must be visible as typed/config-derived facts, not inferred from the
  op kind, fixture filename, artifact name, route id, exact intrinsic spelling,
  descriptor residue, C string, or common EmitC code.
* Generated route evidence must show m2 consistently in config contract,
  bounded slice, target capability mirror, `tcrv_rvv.lmul`, vector type,
  vector C type, mask C type, setvl, load, scalar splat, compare, MAcc, merge,
  store intrinsics, and generated C/C++ statement plan.
* Generated header/prototype and target artifact validation must preserve the
  ABI order and exact provider-owned operand binding summary, especially
  `rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr`.
* The generated harness must execute counts `0`, `1`, a VL-boundary count, a
  tail count, and a larger count, with at least two `rhs_scalar` values and
  two payload/accumulator patterns. It must prove scalar compare behavior,
  vector-vector MAcc payload, accumulator contribution, inactive-lane
  accumulator preservation, output tail sentinel preservation, source
  preservation, and runtime `n` behavior under m2.
* Any stale or missing LMUL/config/type/provider facts directly touched by
  this round must fail closed before artifact acceptance.

## Acceptance Criteria

* [x] Focused lit/FileCheck coverage proves the m2 fixture materializes
      realized `tcrv_rvv` vector and mask types, `setvl`, `with_vl`, loads,
      scalar splat, compare, masked MAcc, merge/store facts, and route/header
      metadata as provider-derived `m2` facts.
* [x] A generated-bundle dry-run test for
      `runtime_scalar_cmp_masked_macc_add_lmul_m2` checks representative
      evidence JSON, generated C/C++, and harness facts for `m2` config,
      `vint32m2_t`, `vbool16_t`, `__riscv_vsetvl_e32m2`,
      `__riscv_vle32_v_i32m2`, `__riscv_vmv_v_x_i32m2`,
      `__riscv_vmsle_vv_i32m2_b16`, `__riscv_vmacc_vv_i32m2`,
      `__riscv_vmerge_vvm_i32m2`, and `__riscv_vse32_v_i32m2`.
* [x] Direct pre-realized route-entry fail-closed coverage includes the m2
      runtime-scalar masked MAcc op, proving the selected-boundary producer
      remains required.
* [x] Target artifact validation or focused C++ tests reject stale or missing
      LMUL/config/type/provider mirror facts for this route family when those
      facts are part of the changed production boundary.
* [x] `ssh rvv` generated-bundle correctness passes for counts `0`, `1`,
      `16`, `17`, and `257`, rhs scalar values `-37` and `91`, and patterns
      `0` and `1`.
* [x] A bounded old-authority scan over touched files finds no new positive
      reliance on legacy `i32m1` route authority, descriptors,
      source-front-door positive route authority, direct-C/source-export
      compute, exact-intrinsic authority, or common EmitC RVV semantics.
* [x] Focused build/test/script commands pass, the task is finished/archived,
      one coherent commit is created, and `git status --short` is clean.

## Definition Of Done

* PRD, implement/check context, and journal entries are truthful.
* Focused implementation and tests cover only the bounded m2 runtime-scalar
  masked MAcc path plus directly required stale/fail-closed checks.
* No Stage 2 clone batch is introduced.
* Runtime correctness is claimed only with real `ssh rvv` evidence.

## Technical Approach

1. Strengthen the existing m2 target fixture checks if they do not explicitly
   assert provider-derived `m2` config/header facts.
2. Add focused `test/Scripts` dry-run and direct pre-realized fail-closed
   tests for `runtime_scalar_cmp_masked_macc_add_lmul_m2`, following the m1
   runtime-scalar MAcc tests and existing m2 generated-bundle tests.
3. Add the narrowest target artifact stale-config/type regression needed after
   inspecting the current target validator behavior.
4. Run the generated-bundle dry-run, focused lit/FileCheck commands, target
   artifact test if changed, `ssh rvv` runtime evidence, authority scan,
   Trellis validation, and `git diff --check`.

## Decision (ADR-lite)

**Context**: The m2 fixture and script expectation already exist, but no
matching generated-bundle tests or runtime evidence prove that LMUL m2 is the
typed route/config authority for runtime-scalar masked MAcc.

**Decision**: Close this as a bounded evidence and validation round for exactly
`runtime_scalar_cmp_masked_macc_add_lmul_m2`, using the existing generic typed
route infrastructure and adding only focused stale/fail-closed validation that
is missing for the active production path.

**Consequences**: The task advances Stage 2 SEW/LMUL propagation without
creating a broad LMUL clone batch. m1 runtime-scalar MAcc remains the behavior
template, but m2 acceptance depends on typed config propagation and real RVV
execution.

## Out Of Scope

* Broad LMUL, dtype, arithmetic-kind, widening, reduction, matmul, frontend, or
  tuning expansion.
* Adding `m4`, `m8`, non-i32 SEW changes, or new high-level operation families.
* Treating route ids, artifact names, test names, C strings, descriptors,
  exact intrinsic spellings, mirror metadata, or common EmitC as LMUL/config
  authority.
* Moving RVV semantics into common EmitC/export.
* Reworking `computed_masked_macc_add_lmul_m2` except as a comparison point.

## Technical Notes

Relevant specs and prior task:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-runtime-scalar-cmp-masked-macc-add-artifact-abi-boundary/`

Likely implementation/test files:

* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add-lmul-m2.mlir`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-macc-add-lmul-m2-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-macc-add-lmul-m2-fail-closed.test`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`

## Completion Evidence

Implemented behavior:

* `validateRVVMAccTargetArtifactCandidateMirrors` now requires candidate
  metadata mirrors for `tcrv_rvv.config_contract`, `tcrv_rvv.element_type`,
  `tcrv_rvv.sew`, `tcrv_rvv.lmul`, and `tcrv_rvv.bounded_slice` against the
  provider-derived MAcc route description before target artifact acceptance.
* `TargetArtifactExportTest.cpp` covers the positive
  `runtime_scalar_cmp_masked_macc_add` m2 fixture and rejects stale m2
  candidate mirrors for LMUL/config contract drift.
* The pre-realized m2 MLIR fixture now checks plan/header m2 config facts in
  addition to realized vector/mask operations.
* Generated-bundle script tests cover dry-run m2 evidence and direct
  pre-realized route-entry fail-closed behavior for exactly
  `runtime_scalar_cmp_masked_macc_add_lmul_m2`.

Focused checks run:

* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-02-stage2-rvv-runtime-scalar-masked-macc-lmul-m2-artifact-abi`
* `cmake --build build --target tianchenrv-target-artifact-export-test`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `tcrv-opt ... --tcrv-materialize-selected-lowering-boundaries | FileCheck --check-prefix=REALIZED`
* `tcrv-opt ... --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | FileCheck --check-prefix=PLAN`
* `tcrv-opt ... --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | tcrv-translate --tcrv-export-target-header-artifact | FileCheck --check-prefix=HEADER`
* `scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind runtime_scalar_cmp_masked_macc_add_lmul_m2 ...`
* `FileCheck` ROOT/MACC/CPP/HARNESS checks over the generated dry-run bundle.
* Direct pre-realized route-entry fail-closed probe for
  `runtime_scalar_cmp_masked_macc_add_lmul_m2`.
* `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* `ssh rvv` generated-bundle compile/run evidence for counts `0,1,16,17,257`,
  rhs scalar values `-37,91`, and patterns `0,1`.
* `git diff --check`
* Bounded authority scan over touched files. Newly added legacy-authority hits
  are only negative `implicit-check-not` guards; tracked-file added lines have
  no positive `i32m1`, descriptor, source-front-door, direct-C/source-export,
  exact-intrinsic authority, or common EmitC RVV semantic dependency.

Runtime evidence:

* Artifact:
  `artifacts/tmp/06-02-stage2-rt-scalar-macc-m2-ssh-rvv/pre-realized-runtime-scalar-cmp-masked-macc-add-lmul-m2/runtime_scalar_cmp_masked_macc_add_lmul_m2/evidence.json`
* Status: `success`
* `ssh_evidence`: `true`
* Remote compile/run: `remote_compile_succeeded=true`,
  `remote_run_succeeded=true`
* PASS marker:
  `PASS op=runtime_scalar_cmp_masked_macc_add_lmul_m2 counts=0,1,16,17,257 rhs_scalars=-37,91 patterns=0,1`

Spec update review:

* No `.trellis/spec/` update was required. This round implemented and tested
  existing RVV Stage 2 typed-body/config authority rules; it did not introduce
  a new reusable convention beyond the current RVV plugin and testing specs.

## Current Phase

Finish/archive.
