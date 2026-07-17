# Stage2 RVV compare/select executable artifact ABI boundary

## Goal

Make the existing pre-realized compare/select selected-body RVV route family
truthfully executable as generated target artifacts, or fail closed at the
precise missing artifact/ABI boundary. The owned boundary is typed
`tcrv_rvv` compare predicate facts, mask value provenance, select true/false
and result roles, dtype/config/policy, runtime AVL/VL, per-operand ABI/header
bindings, RVV plugin-owned route validation, common EmitC materialization,
target artifact export, generated bundle ABI, and `ssh rvv` correctness
evidence.

## Why Now

Commit `80a48e25` closed the previous computed-masked MAcc executable evidence
task. Compare/select is the adjacent Stage2 mask-producing and mask-consuming
dataflow class. It must now be audited and, if needed, repaired as a production
artifact/ABI path instead of remaining dry-run-only, metadata-authorized, or
under-validated.

## What I Already Know

- There is no previous active Trellis task; this task was created from the
  Hermes direction brief.
- This is Stage2 RVV work on corrected typed low-level `tcrv_rvv` selected
  bodies, not Stage1 legacy `i32m1` route growth.
- The authority chain is `tcrv.exec` selected RVV variant -> typed or realized
  `tcrv_rvv` body -> RVV plugin legality/realization/route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> generated
  bundle -> `ssh rvv` evidence when runtime correctness is claimed.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI/runtime
  roles only; compare predicate, mask provenance, select layout, dtype/config,
  intrinsic spelling, and route support must be structural provider facts.
- Common EmitC/export may carry provider-built payloads and mirrors, but must
  not infer compare/select semantics from route ids, ABI strings, artifact
  metadata, test names, helper names, or exact intrinsic spellings.
- Relevant specs define three compare/select boundaries: selected-body
  realization, route-provider facts preflight, and compare/select statement
  plan consumption. Target artifact validation must consume
  `RVVCompareSelectRouteValidationContract` with embedded runtime AVL/VL
  selected-boundary facts before accepting mirrors.
- Existing generated-bundle tests named in the brief are dry-run or
  fail-closed coverage; the current audit must determine whether a non-dry-run
  `ssh rvv` evidence gap or a production validation gap remains.
- The previous computed-masked MAcc task is a reference for evidence shape only;
  it is not proof that compare/select is complete.

## Requirements

- Preserve RVV plugin ownership for compare/select selected-body realization,
  family/provider facts, operand binding, statement planning, predicate/mask
  provenance, select true/false/result roles, runtime AVL/VL, and target
  validation contracts.
- Prove or repair the pre-realized compare/select path from selected boundary
  materialization through provider route facts, common EmitC materialization,
  target artifact export, generated bundle compile, and `ssh rvv` correctness
  when executable behavior is claimed.
- If the compare/select family is too broad for one round, complete one
  coherent production submodule starting with pre-realized plain `cmp_select`,
  and leave the exact next continuation point for computed-mask or
  runtime-scalar compare/select.
- Harden the production seam if audit shows dry-run-only support, stale route
  validation, missing provider preflight, missing runtime AVL/VL target
  contract consumption, wrong ABI/header binding, swapped true/false role, or
  metadata/route-id/common-EmitC semantic authority.
- Add or retain focused fail-closed evidence for at least one stale or missing
  executable-boundary fact such as predicate kind, mask provenance, select
  operand role, result binding, header/type summary, ABI order, runtime AVL/VL,
  or statement-plan leaf.
- Keep support levels separate: parseable/verifier-legal is not
  route-supported; route-supported is not executable without complete
  ABI/runtime/export support and real `ssh rvv` evidence for runtime claims.

## Acceptance Criteria

- [x] PRD and Trellis context identify the compare/select executable artifact
  ABI boundary, non-goals, and relevant specs.
- [x] Repository audit records whether current pre-realized compare/select is
  dry-run-only, stale, under-validated, already production-valid, or too broad
  for a single round.
- [x] Production code is changed when the executable boundary is incomplete or
  under-validated; otherwise the PRD records a precise no-source-change
  justification backed by focused evidence.
- [x] Positive evidence covers the selected scope through selected boundary
  materialization, emission plan, target artifact export, generated bundle
  compile, and `ssh rvv` correctness if runtime behavior is claimed.
- [x] Focused fail-closed evidence rejects at least one stale/missing
  compare/select executable-boundary fact before artifact acceptance or
  executable-route claim.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test` pass.
- [x] Relevant generated-bundle dry-run and direct pre-realized fail-closed
  tests pass for the selected compare/select scope.
- [x] Bounded old-authority scan over touched files and added diff lines shows
  no new positive `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*` route
  authority, source-front-door route, descriptor compute path, or exact
  intrinsic spelling as route authority.
- [x] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are clean after commit.

## Out of Scope

- No broad compare/select matrix, dtype/LMUL clone batch, high-level
  Linalg/Vector/StableHLO frontend, per-Linalg route authority,
  source-front-door positive route, performance tuning database, dashboard, or
  report-only closeout.
- No MAcc, reduction, memory, segment2, conversion, product/dequant,
  contraction, or unrelated mask route rework except as bounded reference.
- No common EmitC invention of RVV predicate, mask, select layout, dtype,
  runtime VL, ABI, or route support semantics.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, emission, or route semantics. Python
  may support generated-bundle tooling and evidence collection only.

## Technical Approach

1. Audit the current compare/select realization, route-provider preflight,
   statement-plan, target validator, generated-bundle script, and named tests.
2. Narrow the executable scope if needed, starting with pre-realized plain
   `cmp_select`.
3. Patch production C++/MLIR/test/tooling only where the executable
   artifact/ABI seam is missing or stale.
4. Run focused C++ tests, lit/generated-bundle dry-run/fail-closed tests, and
   non-dry-run `ssh rvv` generated-bundle evidence for any claimed executable
   behavior.
5. Finish/archive the Trellis task and commit one coherent change.

## Technical Notes

- Read specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Reference archive:
  - `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-computed-masked-macc-artifact-abi/`
- Bounded source/test targets from the direction brief:
  - `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-cmp-select-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-mask-select-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-select-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-dual-cmp-mask-and-select-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-cmp-select-fail-closed.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-select-fail-closed.test`
  - `test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-select.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-f32-clamp-select.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir`
  - `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-dual-cmp-mask-and-select.mlir`

## Audit Conclusion

The production C++ compare/select seam is already provider-owned and
fail-closed at the route and target artifact boundary:

- `RVVEmitCRouteProvider.cpp` verifies selected-body route family provider
  plans, materialization facts, operand binding facts, compare/select
  statement plans, and provider facts before constructing a
  `TCRVEmitCLowerableRoute`.
- `RVVEmitCCompareSelectStatementPlanOwners.cpp` carries compare/select
  statement-plan ownership for plain compare-select, computed-mask select,
  runtime-scalar compare-select, dual compare-mask-and-select, f32 clamp
  select, and dequant clamp epilogue families.
- `RVVEmitCRoutePlanning.cpp` populates
  `RVVCompareSelectRouteValidationContract`, including predicate, select
  layout, operand/result roles, runtime ABI parameters, runtime AVL/VL
  selected-boundary facts, typed config, mask/tail policy, header summaries,
  type mappings, and stale mirror rejection.
- `RVVTargetArtifactRouteFamilyValidation.cpp` consumes the
  compare/select validation contract before artifact acceptance and validates
  route id, runtime AVL/VL mirrors, ABI/header/type mappings, statement-plan
  leaves, predicate facts, mask provenance, select layout, and stale unrelated
  metadata.

The actual executable evidence gap was in the generated-bundle harness for
the plain compare-select route. The explicit/pre-realized `cmp_select`
harness checked active-lane correctness but did not allocate or validate a
tail sentinel beyond runtime `n`. That made a runtime AVL/VL boundary
regression harder to catch for the plain compare/select artifact even though
computed-mask and runtime-scalar compare/select harnesses already checked tail
preservation.

## Implementation Summary

- Added output tail-sentinel allocation and validation to the plain
  `cmp_select` generated-bundle harness in
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- The harness now initializes lanes after active runtime `n`, fails with a
  targeted `touched tail sentinel` diagnostic if an artifact writes past `n`,
  and prints `tail_preserved` only after checking the guard region.
- Updated explicit and pre-realized compare-select dry-run lit tests to check
  the generated harness includes the call, expected true/false result
  expression, tail-sentinel failure diagnostic, and `tail_preserved` success
  marker.
- No compiler-core C++ change was needed because the audited provider,
  planning, statement-plan, and target validation seam already carries and
  validates the compare/select artifact ABI facts structurally.

## Evidence

Positive executable `ssh rvv` evidence:

- Pre-realized plain compare-select:
  `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body
  --op-kind cmp_select --op-kind cmp_select_sle --runtime-count
  0,1,7,16,23,257` completed with `ssh_evidence: true` and `status:
  success`.
- `cmp_select` and `cmp_select_sle` both reported active-lane true/false
  correctness, `tail_preserved`, and final pass markers:
  `tcrv_rvv_generated_bundle_abi_cmp_select_ok` and
  `tcrv_rvv_generated_bundle_abi_cmp_select_sle_ok`.
- The generated bundle executed on remote `riscv64` with `/usr/bin/clang`
  reporting Ubuntu clang 18.1.3.

Positive executable family evidence:

- Pre-realized computed-mask/runtime-scalar compare-select family:
  `--op-kind computed_mask_select --op-kind computed_mask_select_sle
  --op-kind runtime_scalar_cmp_select --op-kind
  runtime_scalar_dual_cmp_mask_and_select --runtime-count
  0,1,7,16,23,257 --rhs-scalar -37 --rhs-scalar 91` completed with
  `ssh_evidence: true` and `status: success`.
- The family run reported compare data pattern coverage, scalar true/false
  lane coverage, dual-mask composed/single-mask-only coverage,
  `tail_preserved`, and final pass markers for all four selected route
  families.
- The generated bundle executed on remote `riscv64` with `/usr/bin/clang`
  reporting Ubuntu clang 18.1.3.

Focused dry-run and build evidence:

- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- Explicit `cmp_select` generated-bundle dry-run passed after adding the tail
  harness check.
- Pre-realized `cmp_select`/`cmp_select_sle` generated-bundle dry-run passed
  after adding the tail harness check.
- `lit -sv . --filter cmp-select` from `build/test` passed 19 tests.
- `ninja -C build tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate` passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `build/bin/tianchenrv-target-artifact-export-test` passed.

Focused fail-closed evidence:

- Existing direct/pre-realized compare-select fail-closed tests remain in the
  focused lit set and reject stale provider mirror, binding plan, runtime ABI
  order, header, type mapping, predicate, and select layout facts.
- Existing computed-mask compare-select artifact tests remain in the focused
  lit set and reject stale mask producer/provenance, true-value operand role,
  provider mirror, target leaf, mask role/source/form, select layout,
  tail/mask policy, and mask-tail plan owner facts.

Self-repair:

- The first generated-bundle dry-run invocation failed because `llvm-readobj`
  was not on `PATH`. The checks were rerun with
  `/usr/lib/llvm-20/bin/llvm-readobj` and passed.

## Spec Update Decision

No `.trellis/spec/` update is needed. The testing contract already requires
generated-bundle memory-writing evidence to preserve runtime `n`/tail guard
behavior; this round brought the plain compare-select harness into compliance
with that existing rule.

## Current Phase

Finish / archive.
