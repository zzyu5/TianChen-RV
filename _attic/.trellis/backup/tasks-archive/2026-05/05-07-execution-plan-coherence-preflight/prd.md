# execution plan coherence preflight

## Goal

Add a bounded C++/MLIR pre-export coherence verifier for TianChen-RV execution
planning metadata. The verifier must run after selected-path planning,
lowering-boundary materialization, runtime-ABI emission-plan materialization,
and before target artifact export, so stale or contradictory compiler-visible
handoff metadata fails closed.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Starting HEAD is `bdcd5d5 feat: add builtin target artifact exporter registry`.
* Initial worktree inspection was clean and `predoc/` was absent.
* Current stack is C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck.
* Existing code has plugin proposal materialization, legality, selection,
  dispatch synthesis, selected lowering-boundary materialization,
  emission-plan diagnostics, generic artifact export, and built-in target
  exporter registration.
* This task must not add a new exporter route, microkernel, runtime execution,
  target feature, or RVV runtime/correctness/performance claim.

## Requirements

* Implement active C++/MLIR verifier or pass logic for execution-plan/export
  preflight coherence.
* Keep the check target-neutral and selected-path based.
* Validate agreement across selected path, dispatch/fallback references,
  lowering-boundary metadata, runtime-ABI metadata, emission-plan diagnostics,
  artifact route metadata, exporter registration, and origin plugin
  registration.
* Delegate origin/plugin checks through the existing ExtensionPluginRegistry
  where appropriate.
* Do not branch on RVV, scalar, offload, IME, Sophgo, AME, vendor,
  runtime-family, dtype, shape, toolchain, or microarchitecture in core
  transform code.
* Reuse/factor existing target artifact export candidate validation where it is
  useful, instead of maintaining a second divergent route checker.
* Preserve existing execution-planning behavior unless the new check is
  deliberately appended in a fail-closed position.

## Acceptance Criteria

* Legal RVV explicit microkernel planned path passes the coherence check.
* Legal scalar fallback microkernel planned path passes.
* Legal offload descriptor planned path passes.
* Existing `tcrv-execution-planning-pipeline` can be followed by the new check
  on at least one built-in fixture.
* Negative lit coverage rejects:
  * stale selected-path variant symbol;
  * selected-path origin mismatch;
  * unregistered selected-path origin plugin;
  * lowering-boundary kernel/variant/origin mismatch;
  * emission-plan route/origin mismatch;
  * offload-style descriptor path missing required runtime ABI metadata;
  * source/descriptor artifact route cross-spoofing;
  * multiple ambiguous supported artifact candidates.

## Out Of Scope

* No artifact export implementation changes beyond shared validation plumbing.
* No LLVM/RISC-V lowering, object generation, linking, runtime execution,
  correctness evidence, benchmark, or performance claim.
* No generic compute op or computation semantics in `tcrv.exec`.
* No Python compiler internals.
* No `ssh rvv` evidence unless a new RVV runtime/correctness/performance claim
  is intentionally made, which this task does not require.

## Definition Of Done

* C++/MLIR/TableGen/CMake/lit changes are implemented and covered.
* `git diff --check` passes.
* CMake configure with LLVM/MLIR 20 succeeds.
* `check-tianchenrv` passes.
* Trellis task is archived and validated.
* One coherent commit is created and the repo is clean.

## Technical Notes

* Relevant specs: core dialect, plugin protocol, variant pipeline,
  lowering-runtime, RVV plugin, scalar fallback plugin, offload runtime plugin,
  and MLIR testing contract.
* Existing export candidate checks live in `lib/Target/TargetArtifactExport.cpp`.
* Existing selected lowering-boundary checks live in
  `lib/Transforms/EmissionReadiness.cpp`.
* Public pass entry should live under `include/TianChenRV/Transforms/` and
  `lib/Transforms/`, with TableGen registration if exposed through `tcrv-opt`.
