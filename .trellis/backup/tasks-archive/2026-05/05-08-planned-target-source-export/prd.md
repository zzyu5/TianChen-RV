# Planned target-source export for synthesized dispatch

## Goal

Carry the synthesized conflict-aware RVV+scalar i32-vadd dispatch plan through the generic target artifact export path so that a module produced by `--tcrv-execution-planning-pipeline` can be translated with `--tcrv-export-target-source-artifact` into the bounded RVV+scalar dispatch C source.

## What I already know

* The repository root is `/home/kingdom/phdworks/TianchenRV`.
* HEAD is expected to include `47051ba feat: synthesize conflict-aware dispatch plans`; the required precondition check confirmed it is current HEAD.
* The supervisor-policy files were clean at task start, so this task proceeds with compiler ownership and must not touch supervisor policy.
* The previous run added active C++/MLIR transform behavior for conflict-aware dispatch synthesis and selection.
* A required capability that is available but conflicts with another available capability must not be silently static-selected or used as fallback.
* A guarded conflict-aware case may survive only as an explicit guarded `tcrv.exec.case` with generic policy metadata, while fallback remains conflict-free.
* The remaining gap is artifact export: the newly synthesized guarded dispatch plan must become a real planned target artifact, not remain only a transform-level structure or hand-authored dispatch fixture.

## Requirements

* Keep implementation in C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck.
* Do not implement core IR, dialects, passes, target artifact selection, lowering, emission, runtime glue, or compiler decisions as Python data structures.
* Keep scope bounded to the existing RVV+scalar i32-vadd dispatch source artifact and generic target artifact selection/coherence path.
* Ensure `tcrv-opt %s --tcrv-execution-planning-pipeline | tcrv-translate --tcrv-export-target-source-artifact` can export the bounded RVV+scalar dispatch source from synthesized planned IR.
* If existing RVV+scalar dispatch export already supports planned IR, exercise it through the generic artifact export route instead of only direct hand-authored RVVScalarDispatch translation.
* If the exporter assumes hand-authored metadata, refactor only the smallest target-owned code necessary to read synthesized dispatch, case/fallback, selected lowering boundaries, emission plans, and role-bound runtime ABI metadata structurally.
* Preserve the existing direct RVV+scalar dispatch route unless a real coherence bug requires narrowing it.
* Treat `policy = capability_dispatch_guard` as planning metadata unless the target-owned exporter maps it to an existing concrete availability guard ABI role.
* Keep target-specific RVV/scalar emission decisions inside target/plugin/exporter code.
* Core transforms and generic target artifact registry logic must not branch on RVV, scalar, march strings, vendor names, IME, Sophgo, AME, or offload.
* Do not add a generic runtime predicate language in `tcrv.exec`.
* Preserve parameter layering: hardware facts / target capability, compile-time variant config, runtime SSA/control/ABI values, and descriptor-local target metadata stay distinct.

## Acceptance Criteria

* A focused lit/FileCheck test starts from input requiring the built-in planning pipeline, pipes `tcrv-opt` into `tcrv-translate --tcrv-export-target-source-artifact`, and verifies bounded RVV+scalar dispatch C source is emitted from the synthesized plan.
* The test proves a synthesized guarded conflict-aware `tcrv.exec.case` reaches target artifact export.
* The emitted source uses the RVV callable, scalar fallback callable, and one target-owned availability guard ABI role structurally.
* The generated source is dispatch source, not a smoke probe, manifest-only output, self-check harness, or direct hand-authored fixture shortcut.
* The generated source does not contain `runtime_success`, throughput, latency, password, token, `artifacts/tmp`, or raw runner paths.
* Add one focused negative only if a concrete new coherence check is needed.
* Existing dispatch synthesis, variant selection, RVV scalar dispatch, and target artifact export tests continue to pass.

## Required Checks

* `git diff --check`
* `cmake --build build --target tcrv-opt tcrv-translate -j2`
* Targeted lit for the new or changed target artifact/RVV scalar dispatch test.
* Targeted lit for affected execution-planning or dispatch-synthesis tests if modified.
* `cmake --build build --target check-tianchenrv -j2`
* If Trellis task state is archived, validate it with `python3 ./.trellis/scripts/task.py validate <archived-task-path>`.

## Out of Scope

* Full conflict solver.
* Provider ranking or cost/tuning model.
* New RVV dialect semantics.
* Broad runtime ABI framework.
* Hardware probes, performance measurement, or ssh rvv evidence unless making runtime/correctness/performance claims.
* Generic target backend.
* Python compiler/lowering/target-export/runtime-ABI implementation.
* Smoke-only probes, helper wrappers, broad negative matrices, or evidence packaging.

## Technical Notes

* Required repo reading is listed in the user request and includes the main spec spine, transform passes, target artifact export code, RVV scalar dispatch exporter, runtime ABI support, and focused existing tests.
* This task intentionally avoids subagents and parallel agent workflows per user instruction.
