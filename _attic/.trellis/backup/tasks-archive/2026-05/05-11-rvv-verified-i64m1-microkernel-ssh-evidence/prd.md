# RVV verified i64m1 microkernel artifact-to-ssh evidence

## Goal

Carry one verifier-checked i64m1 RVV direct microkernel from checked-in MLIR
through the existing TianChen-RV selected-boundary, RVV body verifier, target
artifact source/header/object or bundle export, and real `ssh rvv`
compile/link/run evidence. Prefer the checked-in `i64-vadd` fixture from the
previous verifier round; use `i64-vmul` only if a concrete `i64-vadd` blocker is
found and recorded here.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Session start worktree was clean at
  `a179f07 feat(rvv): verify selected microkernel bodies before export`.
- `.trellis/.current-task` did not exist at session start, so this task was
  created from the Hermes/current task brief and set current.
- The previous task
  `.trellis/tasks/archive/2026-05/05-11-rvv-microkernel-body-selected-config-verifier/`
  is complete and archived; it must remain archived.
- That previous verifier task added `RVVBinaryMicrokernelBodyVerifier` and wired
  RVV artifact export validation through it for finite i32/i32m2 and i64m1
  binary routes.
- A prior scalar-dispatch evidence task has real `ssh rvv` evidence for an
  i64m1 `i64-vmul` dispatch bundle route, but this task is about the direct RVV
  microkernel artifact route and must not reuse dispatch evidence as proof.

## Requirements

- Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only orchestrate scripts, probes, artifact parsing, and
  evidence collection.
- Keep RVV-specific semantics local to RVV plugin/target/export code. Do not add
  RVV branches to `tcrv.exec`, generic core passes, or generic target artifact
  routing.
- Start from checked-in MLIR with selected metadata and a structured
  `tcrv_rvv` body accepted by `validateRVVBinaryMicrokernelBody`.
- Carry the selected route through the existing target artifact path to
  generated source/header/object or source/header plus remote object build
  inputs.
- Compile and run on the real RVV host via `ssh rvv`. Evidence must record
  remote architecture, compiler path/version, compile/link/run commands or
  script invocation, and deterministic numeric output comparison.
- Evidence must show the verified RVV microkernel body path was consumed. A
  hand-written standalone C file unrelated to compiler/export output is not
  acceptable as the truth source.
- If the direct route already works without compiler code changes, do not pad
  the commit with unrelated smoke wrappers. Record the minimal durable test,
  runner, PRD, task notes, and evidence path.
- If the route fails, fix the active compiler/export/runtime ABI path at the
  owning layer: C++/MLIR/target code for compiler/export issues, Python only for
  runner/probe/artifact orchestration issues.

## Acceptance Criteria

- [x] Trellis task exists, is current, and validates before finish/archive.
- [x] The selected route is one finite i64m1 RVV direct microkernel, with
      `i64-vadd` preferred unless a concrete blocker is recorded.
- [x] The route starts from checked-in MLIR and reaches compiler-generated
      source/header/object or source/header plus remote object build inputs
      through existing target artifact export.
- [x] Local checks prove the chosen fixture still uses the structured RVV body
      verifier and artifact exporter, not a detached evidence helper.
- [x] Real `ssh rvv` compile/link/run evidence succeeds for the chosen finite
      route and verifies deterministic numeric output for that family.
- [x] Evidence artifacts are written under `artifacts/tmp/...` and are not
      committed.
- [x] Durable source changes, if any, are limited to fixtures, script runner
      improvements, focused tests, C++/MLIR/target fixes needed by this route,
      and Trellis records.
- [x] No runtime, correctness, throughput, latency, or performance claim exceeds
      the exact `ssh rvv` evidence collected.
- [x] `git diff --check` passes.
- [x] Focused build and tests for `tcrv-translate`, RVV target/export code, and
      touched RVV microkernel/script tests pass.
- [x] Trellis validation passes before and after archive if the task completes.

## Minimal Validation Plan

- `git diff --check`
- Build focused targets as needed:
  `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`
- Run focused C++ tests as needed:
  `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
  and `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- Run focused lit/FileCheck tests for the chosen direct RVV i64 microkernel
  fixture, artifact exporter, target artifact bundle route, and script dry-run
  behavior.
- Run `python3 scripts/rvv_microkernel_e2e.py --self-test`.
- Run local dry-run for the chosen family before ssh.
- Run real `ssh rvv` evidence for the chosen family, preferably:
  `python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vadd --ssh-target rvv --run-id <run-id> --overwrite --timeout 120`
- Run
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-11-rvv-verified-i64m1-microkernel-ssh-evidence`
  before finish/archive and again after archive if completed.

## Out of Scope

- No generic RVV backend.
- No broad family matrix or multi-kernel sweep.
- No performance, throughput, latency, or speedup claim.
- No fake ssh evidence and no local-only substitution for runtime evidence.
- No hand-written runtime source as the proof if it bypasses compiler/export
  output.
- No compute semantics in `tcrv.exec`.
- No RVV semantic branches in generic core passes or generic target artifact
  routing.
- No compiler internals in Python.
- No docs-only, metadata-only, helper-only, or report-only closeout.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/validation/experiment-reference.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Previous task context read:
  `.trellis/tasks/archive/2026-05/05-11-rvv-microkernel-body-selected-config-verifier/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-11-rvv-selected-vector-capability-metadata-contract/prd.md`.
- Relevant prior journal context: Session 30 recorded real `ssh rvv` evidence
  for `i64-vmul` RVV+scalar dispatch bundle, but that is a dispatch route and
  only fallback evidence if `i64-vadd` direct microkernel is blocked.

## Completion Notes

- Selected the preferred checked-in direct fixture:
  `test/Target/RVVMicrokernel/rvv-microkernel-i64-vadd.mlir`.
- No C++ compiler/export fix was needed. Current HEAD already carries the
  direct i64m1 `i64-vadd` path through selected metadata,
  `validateRVVBinaryMicrokernelBody`, target artifact source/header/object
  bundle export, and generated external caller construction.
- Added focused lit coverage in
  `test/Scripts/rvv-microkernel-bundle-e2e.test` for the exact direct
  `i64-vadd` target-artifact-bundle dry-run. It checks selected kernel
  `export_i64_vadd`, direct variant `rvv_i64_slice`, i64 ABI parameter types,
  route ids, i64m1 intrinsic evidence, and external caller arithmetic
  `lhs + rhs`.
- Real ssh command:
  `python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --arithmetic-family=i64-vadd --expect-selected-kernel=export_i64_vadd --ssh-target rvv --run-id codex-i64-vadd-direct-bundle-ssh --overwrite --timeout 120`
- Evidence JSON:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-i64-vadd-direct-bundle-ssh/evidence.json`.
- Evidence result: `status=success`, `mode=ssh`, `ssh_evidence.success=true`,
  `remote_compile_succeeded=true`, `remote_link_succeeded=true`,
  `remote_run_succeeded=true`, and `output_validation_succeeded=true`.
- Remote facts: architecture `riscv64`, clang path `/usr/bin/clang`, clang
  `Ubuntu clang version 18.1.3 (1ubuntu1)`, compile flags
  `-O2 -march=rv64gcv -mabi=lp64d`.
- Runtime/evidence scope: generated caller checked `lhs + rhs` for runtime
  counts `7` and `16`, linked and ran both the source-built object and the
  compiler-generated bundle object, and observed
  `tcrv_rvv_i64_vadd_microkernel_external_abi_ok`.
- A plan-and-export bundle attempt against the hand-selected direct fixture was
  intentionally not used as the final route: that front door re-runs proposal
  planning and declined the already selected i64 fixture because its proposal
  gate still expects finite i32 config capability ids. The completed route uses
  the two-step selected-fixture path required by this task brief: materialize
  selected lowering boundaries/emission plans, then export the target artifact
  bundle.
- No performance, throughput, latency, generic RVV lowering, broad correctness,
  or runtime integration claim is made.

## Validation Results

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-11-rvv-verified-i64m1-microkernel-ssh-evidence`: passed before source/test validation.
- `python3 scripts/rvv_microkernel_e2e.py --self-test`: passed.
- `python3 scripts/rvv_microkernel_e2e.py --dry-run --use-target-artifact-bundle --arithmetic-family=i64-vadd --expect-selected-kernel=export_i64_vadd --run-id codex-i64-vadd-direct-bundle-postpatch-dry --overwrite --timeout 120`: passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`: passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`: passed.
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`: passed.
- Focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-microkernel-bundle-e2e'`
  from `artifacts/tmp/tianchenrv-build/test`: 1/1 selected test passed.
- `git diff --check`: passed.
- `python3 -m py_compile scripts/rvv_microkernel_e2e.py`: passed.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`:
  199/199 lit tests passed.

## Continuation Rule If Unfinished

Keep the task open and record the exact blocker category: export generation,
remote compile, remote link, remote execution, numeric comparison, or test
wiring. Name the exact source file/function or script step to continue from.
Do not archive this task without real `ssh rvv` evidence if runtime correctness
is claimed.
