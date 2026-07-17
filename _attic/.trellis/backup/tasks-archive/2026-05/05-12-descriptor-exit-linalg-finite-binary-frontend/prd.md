# Descriptor exit from the bounded linalg finite-binary frontend path

## Goal

Move the bounded `LowerLinalgRVVBinaryToExec` frontend boundary away from
descriptor-shaped compute authority. The marked `linalg.generic` source body,
operand/result memref element types, and scalar region argument/result types
must determine the finite binary family. The `tcrv_frontend_lowering` marker is
only a bounded route request and cross-check, not the source of arithmetic or
dtype semantics.

This is a structural migration task. It is not a coverage-only task, an
evidence-packaging task, a helper-only task, or a report-only task.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Branch is `main`; the worktree was clean before task creation.
* Current HEAD before this task is
  `f37846e feat: remove descriptor authority from i64 sub mul defaults`.
* No `.trellis/.current-task` existed at session start; this task was created
  as `.trellis/tasks/05-12-descriptor-exit-linalg-finite-binary-frontend/`.
* The previous descriptor-exit tasks for i32 add, i32 sub/mul, i64 add, and
  i64 sub/mul are archived and are reference patterns only.
* The downstream RVV/scalar default selected/export paths now use typed
  extension-family ops plus common EmitC-backed route provenance for the
  supported i32/i64 add/sub/mul families.
* Live frontend source still uses
  `FiniteBinaryFrontendLoweringDescriptor` to resolve marker, arithmetic op,
  element bit width, ABI pointer spellings, and family identity before checking
  the source body. That means the default frontend path still begins from
  descriptor-shaped semantics.
* This round should move source-of-truth one stage earlier: source MLIR facts
  authority the finite family; marker/descriptor metadata only checks that the
  requested route matches the already inferred source identity.

## Requirements

* In `LowerLinalgRVVBinaryToExec.cpp`, infer finite binary identity from the
  marked `linalg.generic` before materializing `tcrv.exec.kernel`:
  dtype from memref operand element types and scalar region/result types,
  arithmetic from the actual `arith.addi`, `arith.subi`, or `arith.muli`
  feeding `linalg.yield`, and family id from those inferred facts.
* Treat `tcrv_frontend_lowering` as a bounded route marker/cross-check only.
  If the marker says `i64-vmul` but the body is `arith.subi`, or if the marker
  says i64 while source types are i32, the pass must fail before creating an
  exec kernel.
* Split, rename, or quarantine `FiniteBinaryFrontendLoweringDescriptor` so the
  default frontend lowering no longer consumes computation fields such as
  `arithmetic`, `elementBitWidth`, `loweringDescriptor`, or `cOperator` as
  authority. Any remaining frontend support table must be a source-derived
  marker/ABI contract, not a descriptor compute table.
* Materialize exec kernel attributes and ABI metadata from inferred source
  facts plus existing runtime ABI helpers. Operand buffer ABI facts come from
  source operands; runtime element count remains runtime/control ABI; selected
  vector config stays in RVV/scalar plugin planning; hardware facts remain
  capability facts.
* Preserve downstream behavior for the already supported finite families:
  `i32-vadd`, `i32-vsub`, `i32-vmul`, `i64-vadd`, `i64-vsub`, and `i64-vmul`.
  The default pipeline must still reach typed extension-family ops and common
  EmitC-backed export evidence.
* Do not reintroduce `selected_lowering_descriptor` or descriptor strings as
  the arithmetic selector in the frontend default path.
* Add fail-closed diagnostics for marker/body mismatch, marker/dtype mismatch,
  unsupported arithmetic body, missing yield from arithmetic result, nonuniform
  source dtype, unsupported dtype, and stale legacy descriptor metadata if such
  metadata can still enter the frontend path.
* Update specs only where they describe the frontend boundary, making clear
  that source `linalg.generic` body and typed source facts are the compute
  authority while marker/descriptor metadata are route hints or legacy
  cross-checks.

## Acceptance Criteria

* [x] Default `i32-vadd` linalg frontend lowering succeeds because source
      memrefs/region args are i32 and the body yields `arith.addi`.
* [x] Default `i32-vsub` linalg frontend lowering succeeds because source
      memrefs/region args are i32 and the body yields `arith.subi`.
* [x] Default `i32-vmul` linalg frontend lowering succeeds because source
      memrefs/region args are i32 and the body yields `arith.muli`.
* [x] Default `i64-vadd` linalg frontend lowering succeeds because source
      memrefs/region args are i64 and the body yields `arith.addi`.
* [x] Default `i64-vsub` linalg frontend lowering succeeds because source
      memrefs/region args are i64 and the body yields `arith.subi`.
* [x] Default `i64-vmul` linalg frontend lowering succeeds because source
      memrefs/region args are i64 and the body yields `arith.muli`.
* [x] A marker/body mismatch fails before creating `tcrv.exec.kernel`.
* [x] A marker/dtype mismatch fails before creating `tcrv.exec.kernel`.
* [x] Unsupported arithmetic, missing arithmetic result yield, nonuniform
      source dtype, or unsupported dtype fails before creating
      `tcrv.exec.kernel`.
* [x] Frontend support APIs no longer present a default-path
      `FiniteBinaryFrontendLoweringDescriptor` carrying descriptor compute
      authority. Any remaining registry is explicitly marker/ABI/cross-check
      scoped.
* [x] Focused lit/FileCheck proves the six supported default linalg paths still
      feed the existing execution-planning/export pipeline.
* [x] Focused C++ tests affected by frontend marker/family contracts,
      RVV/scalar planning, and target export metadata continue to pass.
* [x] `tcrv.exec` remains compute-free and no compiler core, dialect, pass,
      plugin registry, capability model, lowering, or emission logic is
      implemented as Python data structures.

## Out Of Scope

* Adding new finite families beyond i32/i64 add/sub/mul.
* Implementing new RVV, scalar, IME, offload, vendor, MLIR vector, LLVM
  scalable-vector, inline-assembly, or descriptor-to-C routes.
* Moving compute semantics into `tcrv.exec`.
* Replacing the default frontend with a parallel helper path that is not wired
  into production/default lowering in the same round.
* Preserving obsolete descriptor-driven tests as default-path requirements.
  Tests may remain only if they are explicit legacy mismatch/cross-check
  coverage.
* Making RVV runtime, correctness, throughput, latency, or performance claims.
  This task is compile/export/FileCheck focused and should not require
  `ssh rvv`.

## Minimal Validation

* Run `git diff --check`.
* Use `artifacts/tmp/tianchenrv-build` if configured; otherwise run the
  standard CMake configure first.
* Build focused targets:
  `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-scalar-extension-plugin-test`,
  `tianchenrv-rvv-binary-planning-test`, and
  `tianchenrv-target-artifact-export-test`.
* Run focused lit/FileCheck under `test/Transforms/LinalgToExec/` for all six
  i32/i64 add/sub/mul default frontend paths and new mismatch-negative cases.
* Run focused C++ tests affected by frontend marker/family contract,
  RVV/scalar planning, and target export metadata.
* Run `cmake --build artifacts/tmp/tianchenrv-build --target
  check-tianchenrv -j2` unless a local toolchain issue blocks it; if blocked,
  record the exact blocker and still run focused checks.
* Run `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/05-12-descriptor-exit-linalg-finite-binary-frontend` before
  finishing, and validate the archived task path after archive if the task is
  archived.

## Technical Notes

* `.trellis/spec/index.md` states descriptor-driven computation is bounded
  implementation debt and the current main route is extension family ops to
  EmitC to intrinsic/runtime C/C++.
* `.trellis/spec/architecture/unified-riscv-mlir.md` forbids descriptors from
  defining computation semantics or standing in for extension family ops.
* `.trellis/spec/variant-pipeline/generation-selection-tuning.md` already
  defines a bounded linalg frontend slice; this task updates that boundary so
  source body/types authority compute identity and marker values cross-check
  the inferred identity.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines the downstream
  correct route: typed extension op, generated lowerable interface, target or
  plugin mapping, and common EmitC route provenance.
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md` requires ABI
  role/name/type/ownership metadata to mirror IR-backed `mem_window` and
  `runtime_param` boundaries.
* `.trellis/spec/plugin-protocol/interfaces-and-registry.md` requires common
  orchestration through interfaces/registry without family-specific semantic
  branches in core paths.
* `.trellis/spec/testing/mlir-testing-contract.md` requires lit/FileCheck and
  C++ compiler tests for frontend lowering, planning/export behavior, and
  fail-closed mismatch diagnostics; Python tooling is not compiler evidence.

## Round Result

Completed on 2026-05-12.

* `FiniteBinaryFrontendLoweringDescriptor` was removed from the frontend
  support surface and replaced with `FiniteBinaryFrontendContract`, a bounded
  marker/ABI contract that no longer carries `arithmetic`,
  `loweringDescriptor`, or `cOperator` fields.
* `LowerLinalgRVVBinaryToExec` now derives finite family identity from real
  source IR before any marker cross-check: source memref element widths, scalar
  region argument/result types, and the actual `arith.addi`, `arith.subi`, or
  `arith.muli` feeding `linalg.yield`.
* `tcrv_frontend_lowering` is now checked only after source inference. Dtype or
  body/family mismatches fail before creating `tcrv.exec.kernel`, and legacy
  descriptor metadata on the frontend wrapper/marked op is rejected.
* Existing downstream RVV/scalar planning and target export behavior for
  i32/i64 add/sub/mul remains on typed extension-family ops plus common
  EmitC-backed route provenance. No `selected_lowering_descriptor` arithmetic
  selector was reintroduced.
* The frontend specs and pass description now state that source linalg
  body/types are the compute authority, while marker/descriptor metadata are
  route hints or legacy cross-checks.
* No RVV runtime, correctness, throughput, latency, or performance claim was
  made; no `ssh rvv` evidence was required or collected.

Validation completed:

* `git diff --check`
* `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-rvv-extension-plugin-test
  tianchenrv-scalar-extension-plugin-test tianchenrv-rvv-binary-planning-test
  tianchenrv-target-artifact-export-test -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-scalar-extension-plugin-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* Focused lit: `Transforms/LinalgToExec`, 14/14 passed.
* Direct negative-file check:
  `artifacts/tmp/tianchenrv-build/bin/tcrv-opt
  test/Transforms/LinalgToExec/linalg-finite-binary-frontend-invalid.mlir
  --split-input-file --verify-diagnostics
  --tcrv-lower-linalg-rvv-binary-to-exec`
* Focused lit filter for `ExecutionPlanning`, `RVVScalarDispatch`,
  `TargetArtifactBundleExport`, `RVVMicrokernel`, and `Scalar`: 85/85 passed.
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed 206/206.
