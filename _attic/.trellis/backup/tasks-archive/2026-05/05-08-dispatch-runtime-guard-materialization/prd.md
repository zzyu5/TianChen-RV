# compiler-owned dispatch runtime guard materialization

## Goal

Make capability-guarded dispatch self-contained in compiler IR: after variant selection or dispatch synthesis, selected dispatch cases that need runtime capability/control guarding must have a direct same-kernel `tcrv.exec.runtime_param` dispatch-availability guard and a `tcrv.exec.case runtime_guard` symbol link before lowering-boundary, emission-plan, coherence, or bundle export paths consume the plan.

## What I Already Know

- `tcrv.exec.runtime_param` and `tcrv.exec.case runtime_guard` already exist in ODS.
- The core verifier already checks that `runtime_guard` resolves to a direct same-kernel `tcrv.exec.runtime_param` with ABI role `dispatch-availability-guard`.
- Current RVV lowering-boundary materialization creates dispatch runtime params and links selected RVV cases, which makes the runtime guard path plugin-owned rather than compiler-owned.
- The execution planning pipeline currently runs materialize variants, verify legality, select variants, check capability requires, materialize selected lowering boundaries, materialize emission plans, and check execution-plan coherence.
- Target RVV+scalar dispatch exporters already require an explicit host-provided dispatch guard parameter and reject missing or stale links.

## Requirements

- Add a generic C++/MLIR transform/support path that materializes a bounded dispatch-availability runtime guard parameter under the owning kernel and links selected `tcrv.exec.case` ops through `runtime_guard`.
- Keep the pass target-neutral: no RVV, scalar, IME, Sophgo, AME, offload, vendor, dtype, shape, or microkernel branches in generic runtime-guard materialization.
- Reuse the existing runtime guard parameter concept and support-layer ABI constants instead of introducing compute IR or Python compiler logic.
- Integrate the pass after selected dispatch surfaces are produced and before selected lowering-boundary, emission-plan, coherence, and bundle export consumption.
- Preserve fallback semantics: `tcrv.exec.fallback` must not receive dispatch-case runtime guard metadata.
- Strengthen tests so pipeline-generated guarded dispatch contains the runtime guard param and symbol link, and malformed/missing/stale links fail before target bundle success.
- Preserve runtime ABI signature and selected-plan metadata roles in target artifact bundle/export output.

## Acceptance Criteria

- [x] A focused lit or C++ transform test proves a module with capability-guarded dispatch gains a materialized `dispatch-availability-guard` runtime param and `runtime_guard` case link.
- [x] The normal planning pipeline produces valid guarded dispatch accepted by capability requires, emission readiness, and execution-plan coherence.
- [x] Plan-and-export target artifact bundle still publishes ordered runtime ABI signature including the dispatch guard and does not confuse it with selected-plan metadata.
- [x] Negative coverage proves missing, wrong-kind, duplicate, stale, or out-of-kernel guard links fail before bundle success or target artifact output.
- [x] Fallback ops do not receive runtime guard metadata.
- [x] No new RVV runtime/correctness/performance claim is made without `ssh rvv` evidence.

## Out Of Scope

- No Python implementation of compiler IR, registry, capability model, lowering, emission, runtime ABI, or compiler decisions.
- No new compute operation in `tcrv.exec`.
- No automatic hardware probing in generated C.
- No IME, AME, Sophgo, vendor, or offload expansion.
- No performance, runtime, or correctness claim.

## Technical Notes

- Relevant specs: `.trellis/spec/core-dialect/tcrv-exec-contract.md`, `.trellis/spec/capability-model/capability-contract.md`, `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, `.trellis/spec/testing/mlir-testing-contract.md`.
- Expected implementation files include `include/TianChenRV/Transforms/Passes.td`, `include/TianChenRV/Transforms/Passes.h`, `lib/Transforms/ExecutionPlanningPipeline.cpp`, and a new or existing transform implementation file.
- Existing plugin-specific link helpers in `lib/Plugin/RVV/RVVExtensionPlugin.cpp` should no longer be the primary owner of selected dispatch `runtime_guard` linkage.
