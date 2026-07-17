# RVV i64m1 front-door dispatch runtime ABI boundary

## Goal

Carry the existing bounded linalg `i64-vadd` / `i64m1` RVV front-door route
through the dispatch-runtime ABI boundary. Starting from
`test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir`, the normal
plan/export path must produce a dispatch-capable artifact bundle whose external
caller exercises a dispatch wrapper rather than only the direct RVV microkernel
symbol. The selected RVV variant, scalar fallback, dispatch wrapper, bundle
index, generated header/object/source records, and evidence runner must agree
on one int64 pointer/count/mem-window ABI contract.

## Background

The archived `rvv-i64m1-frontdoor-linalg-ssh-evidence` task proved the direct
RVV i64-vadd front-door path with real `ssh rvv` compile/link/run evidence. The
archived `rvv-finite-binary-descriptor-planning-contract` task then moved finite
RVV binary family planning into the RVV-owned descriptor contract. Planning is
no longer the active blocker.

The next bounded module is the dispatch handoff: the compiler must preserve the
same front-door linalg family and IR-backed runtime ABI through selected RVV
dispatch case, scalar fallback, host dispatch wrapper, and generated external
caller evidence.

## Module Boundary

Owned by this task:

- The i64-vadd / i64m1 RVV+scalar host dispatch path for the existing bounded
  linalg front-door input.
- Runtime ABI agreement across RVV callable candidate, scalar fallback callable
  candidate, dispatch availability guard, dispatch wrapper source/header/object
  records, and bundle index metadata.
- Focused C++/MLIR/lit/script coverage that proves the route selects
  `frontend_i64_vadd`, preserves `rvv_first_slice`, `i64m1`, and
  `__riscv_vadd_vv_i64m1`, includes a scalar fallback callable, and exposes a
  dispatch wrapper external ABI.
- Real `ssh rvv` dispatch-wrapper compile/link/run evidence if runtime
  correctness is claimed.
- Trellis task state, validation, archive, and one coherent commit if complete.

Out of scope:

- Generic RVV backend, broad family matrix, benchmark sweep, or performance
  claim.
- New arithmetic families beyond the bounded existing add/sub/mul finite
  compatibility surface.
- Hand-written runtime source as truth when it bypasses compiler/export output.
- RVV or scalar semantic branches in generic core passes.
- Compute semantics in `tcrv.exec`.
- Compiler internals implemented in Python.
- Docs-only, helper-only, smoke-only, report-only, or evidence-only closeout.

## Requirements

- Use the existing `tcrv.exec.mem_window`, `tcrv.exec.runtime_param`,
  `RuntimeABIContract`, `RuntimeABICallablePlan`, emission-plan, and
  target-artifact route abstractions. Do not invent a parallel wrapper ABI.
- The dispatch route must consume the centralized RVV finite descriptor planning
  contract from the previous task. Dispatch code must not reintroduce
  family-specific proposal inference.
- Preserve selected metadata for `frontend_i64_vadd`, `rvv_first_slice`,
  `rvv_first_slice` selected-path role, `i64m1`, selected SEW/LMUL/policy
  fields, and `__riscv_vadd_vv_i64m1`.
- Ensure RVV and scalar callable plans agree on:
  `const int64_t *lhs`, `const int64_t *rhs`, `int64_t *out`, `size_t n`,
  mem-window roles, runtime element-count semantics, and target/export-owned
  ABI parameter ownership.
- Ensure the dispatch wrapper adds only the selected dispatch availability guard
  parameter resolved from typed `tcrv.exec.case runtime_guard_required` plus
  same-kernel `runtime_guard` linkage. Printable condition/guard/policy strings
  are not sufficient.
- Ensure generated external caller validation exercises the dispatch wrapper
  path and validates deterministic `lhs + rhs` output, not only the direct RVV
  microkernel symbol.
- If i64 scalar fallback, i64 dispatch route registration, bundle grouping,
  or runner support is missing, implement the smallest active
  C++/MLIR/target-owned or runner-orchestration repair needed for this bounded
  route.

## Acceptance Criteria

- Focused compiler/lit coverage proves the linalg input can plan/export a
  dispatch-capable bundle selecting `frontend_i64_vadd`, `rvv_first_slice`, and
  `i64m1`, preserving `__riscv_vadd_vv_i64m1`.
- The bundle contains source/header/object records for one dispatch external ABI
  component group, with `component_role = source/header/object`,
  `external_abi_name`, ordered `runtime_abi_parameter[index]` fields, and
  matching int64 parameter names/types/ownership.
- The selected scalar fallback callable is present and family-matched to
  `i64-vadd`; stale scalar-only, direct RVV-only, i32, i64-sub, or i64-mul
  routes fail closed before a complete dispatch bundle is accepted.
- Generated dispatch source/header exposes a dispatch wrapper entry point with
  int64 pointer ABI, `size_t n`, and one dispatch availability guard parameter;
  direct callable-only export is not accepted as dispatch evidence.
- Runner dry-run coverage consumes compiler-emitted bundle/index metadata rather
  than hardcoded file names or detached ABI guesses.
- If runtime correctness is claimed, real `ssh rvv` evidence records artifact
  path, selected kernel, selected variant, wrapper symbol, RVV intrinsic, scalar
  fallback symbol, remote architecture, remote clang, runtime counts, and output
  comparison success for the dispatch wrapper path.
- Focused build/test checks pass, `git diff --check` passes, and
  `check-tianchenrv` runs if practical after focused checks.
- Trellis task validates and archives only if the dispatch-runtime ABI module is
  complete and any runtime claim has real `ssh rvv` evidence.

## Validation Plan

- Inspect and update the focused runtime ABI, dispatch synthesis/guard,
  planning, RVV/scalar target export, and runner paths named in the task brief.
- Build focused targets that cover changed C++ plus `tcrv-opt` /
  `tcrv-translate`.
- Run focused C++ tests for runtime ABI / target artifact export / RVV plugin or
  dispatch manifest surfaces touched by the implementation.
- Run focused lit tests for RVV scalar dispatch, scalar dispatch bundle,
  microkernel bundle/front-door linalg, and any new negative route fixtures.
- Run `python3 scripts/rvv_scalar_dispatch_e2e.py` or a minimal orchestration
  extension for:
  `--arithmetic-family=i64-vadd --input test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir --expect-selected-kernel=frontend_i64_vadd`.
- Run real `ssh rvv` dispatch-wrapper evidence only when the compiler-generated
  dispatch wrapper path is ready to support a bounded runtime correctness
  claim.
- Run `git diff --check`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` if practical after focused checks pass.

## Initial Technical Notes

- `RuntimeABIContract` and `RuntimeABICallablePlan` are the expected typed ABI
  boundary owners for mem-window/runtime-param role validation.
- Existing RVV direct i64-vadd bundle evidence already proves direct callable
  export; this task must not reuse that as dispatch evidence.
- `RVVBinaryPlanning` now owns finite family/shape proposal planning and should
  remain the source of RVV family selection.
- The scalar fallback spec already includes finite i64 add/sub/mul callable
  source/header/object routes; dispatch should consume those target-owned
  routes, not duplicate scalar semantics in generic code.
- The dispatch route manifest is expected to cover i32 and i64 finite families
  with source/header/object/self-check variants. If i64 registration is partial,
  repair the target-owned manifest/registration boundary.
- Python runner changes, if needed, are orchestration and evidence parsing only.
  They must consume compiler-emitted component metadata and cannot define the
  compiler ABI contract.

## Definition Of Done

The task is done when the bounded linalg i64-vadd/i64m1 path exports a coherent
dispatch runtime ABI bundle through existing compiler/target abstractions,
focused local tests pass, real `ssh rvv` dispatch-wrapper evidence exists for
any runtime correctness claim, Trellis validates/archives the task, and one
coherent commit records the completed module. If the work cannot be completed
in one round, leave the task open and record the exact blocker, failing command,
artifact path, and source file/function for continuation.

## Completion Notes

Completed in this round:

- `test/Transforms/LinalgToExec/linalg-i64-vadd-to-rvv-artifact.mlir` now
  carries the existing front-door `frontend_i64_vadd` route through the
  RVV-plus-scalar dispatch artifact boundary by adding the scalar fallback
  capability/provider wiring and no-RVV conflict policy expected by the
  existing dispatch synthesis path.
- The generated bundle preserves `frontend_i64_vadd`, `rvv_first_slice`,
  `i64m1`, and `__riscv_vadd_vv_i64m1`, and emits scalar fallback plus
  dispatch wrapper source/header/object records under
  `rvv-scalar-i64-vadd-dispatch-external-abi.v1`.
- `scripts/rvv_scalar_dispatch_e2e.py` now validates the compiler-emitted
  `selected_kernel` comment against `--expect-selected-kernel` and records the
  expected and actual selected kernel in evidence. This is runner orchestration
  only; the compiler ABI contract remains owned by MLIR/C++/target export.
- Focused lit coverage was updated so the linalg i64-vadd fixture is validated
  as dispatch evidence, while direct microkernel bundle coverage remains on
  direct RVV fixtures.

Runtime evidence:

- Evidence path:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-i64-vadd-dispatch-frontdoor-ssh/evidence.json`.
- Result: `status=success`, `pass_fail_result=pass`,
  `ssh_evidence_verified=true`.
- Remote target: `ssh_target=rvv`, architecture `riscv64`, clang
  `/usr/bin/clang`, clang version `Ubuntu clang version 18.1.3 (1ubuntu1)`.
- Dispatch wrapper:
  `tcrv_dispatch_i64_vadd_frontend_i64_vadd(const int64_t *lhs, const int64_t *rhs, int64_t *out, size_t n, int rvv_available)`.
- Selected RVV callable:
  `tcrv_rvv_i64_vadd_microkernel_frontend_i64_vadd_rvv_first_slice`.
- Scalar fallback callable:
  `tcrv_scalar_i64_vadd_microkernel_frontend_i64_vadd_scalar_fallback_first_slice`.
- RVV intrinsic set includes `__riscv_vadd_vv_i64m1`; compile flags were
  `-O2 -march=rv64gcv -mabi=lp64d`.
- The generated external caller checked runtime counts `7` and `16`, exercised
  `rvv_available=0` and `rvv_available=1`, validated `lhs + rhs`, and checked
  output overrun preservation. Both source-built and bundle-object linked
  executables printed
  `tcrv_rvv_scalar_i64_vadd_bundle_external_abi_ok runtime_counts=7,16 branches=scalar_and_rvv`.
