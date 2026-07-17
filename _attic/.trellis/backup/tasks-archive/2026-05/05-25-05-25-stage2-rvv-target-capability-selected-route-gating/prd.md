# Stage2 RVV target-capability selected-route gating closure

## Goal

Prove one selected `tcrv.exec` RVV variant is gated by explicit target/capability facts before the existing `scalar_broadcast_macc_add` RVV route is considered route-supported or executable. The closure must keep typed `tcrv_rvv` body/config/runtime facts as compute authority, keep RVV legality and route construction inside the RVV plugin, and keep common EmitC/artifact mechanics neutral.

## What I Already Know

* Current HEAD is `78452035 rvv: close selected exec abi envelope`.
* The previous archived task closed selected exec-envelope ABI for `scalar_broadcast_macc_add`: runtime ABI values link to same-kernel `tcrv.exec.mem_window` / `runtime_param` bindings, provider opt-in fails closed on missing exec ABI bindings, artifact headers mirror verified bindings, and explicit plus direct pre-realized paths passed `ssh rvv` correctness.
* The next bottleneck is capability-driven selected-route gating: RVV route support must depend on explicit selected target/capability facts, not on route names, target names, artifact names, tests, ABI strings, manifests, descriptors, or common EmitC behavior.
* `tcrv.exec` owns execution envelope facts such as target/capability/variant/ABI/runtime roles. It must not own compute, dtype, SEW/LMUL, memory form, scalar/broadcast role, accumulator layout, policy, intrinsic spelling, or selected route authority.
* `tcrv_rvv` typed body/config owns low-level RVV operation/config/runtime facts. The RVV plugin owns legality, selected-body realization, route support, intrinsic/C type/ABI mapping, fail-closed diagnostics, and route construction.
* Artifact metadata and emission-plan fields are mirrors only and must use explicit mirror labels when they reflect provider decisions.

## Requirements

* Add one bounded capability-gated selected route for the already route-supported `scalar_broadcast_macc_add` RVV kernel.
* Make explicit selected `tcrv.exec` target/capability facts structural inputs to RVV plugin legality before provider route construction.
* Fail closed before route/export/artifact authority when selected target facts are absent, RVV capability facts are missing, target/capability ownership is ambiguous, or capability/config/body/runtime facts are incompatible.
* Preserve typed `tcrv_rvv` body/config/runtime authority for operation kind, dtype, SEW/LMUL, policy, memory form, scalar/broadcast role, accumulator role, and VL/runtime count use.
* Keep common EmitC lowering and target artifact plumbing neutral; they may materialize a provider-built route but must not infer RVV capability semantics.
* Use mirror-only metadata labels for provider support and verified target/capability facts in emitted/artifact outputs.
* Add focused verifier/provider/FileCheck coverage for positive gating and fail-closed negative cases.
* If executable behavior is claimed, run one generated-bundle ABI/e2e `ssh rvv` compile/run for deterministic correctness across at least three runtime counts.

## Acceptance Criteria

* [x] Positive test shows selected `tcrv.exec` RVV target/capability facts reach RVV plugin legality and allow the bounded `scalar_broadcast_macc_add` route.
* [x] Negative tests fail closed for missing RVV capability, incompatible capability/config facts, absent selected variant facts, and ambiguous target ownership before provider route or artifact export.
* [x] FileCheck or equivalent evidence shows provider route metadata, emitted signature/header, and artifact mirrors carry explicit mirror labels and do not become authority.
* [x] Bounded scan over touched exec/RVV/target/script/fixture files finds no name-, metadata-, descriptor-, ABI-string-, common-EmitC-, artifact-, harness-, or legacy i32-derived capability authority.
* [x] `git diff --check` passes.
* [x] `check-tianchenrv` runs successfully, or an exact blocker is documented.
* [x] `ssh rvv` correctness evidence is collected for the generated bounded route if executable behavior is claimed; otherwise the final report states the blocker and makes no executable claim.

## Definition of Done

* Task status, PRD, context files, and workspace journal reflect the final state truthfully.
* Relevant tests/checks are run and any focused failures caused by this work are repaired.
* The task is finished/archived when complete.
* One coherent commit is created if the task reaches completion.

## Out of Scope

* New RVV operation families, dtype/LMUL matrices, broad Stage2 coverage expansion, high-level frontend lowering, Linalg/Vector/StableHLO generalization, source-front-door positive routes, runtime feature databases, autotuning, dashboards, performance claims, one-op-per-intrinsic wrappers, IME/Offload/TensorExt work, scalar fallback expansion, and compatibility wrappers that preserve legacy i32 authority.
* Inferring RVV support, dtype, policy, memory form, scalar role, accumulator layout, runtime count semantics, or intrinsic spelling from target names, route ids, helper names, artifact names, test names, ABI strings, manifests, descriptors, common EmitC, or harness constants.

## Technical Notes

* Created from the Hermes Direction Brief in the Codex worker prompt on 2026-05-25.
* Read first: `.trellis/spec/index.md`; relevant `tcrv.exec` target/capability/variant/dispatch/fallback specs; `.trellis/spec/extension-plugins/rvv-plugin.md`; `.trellis/spec/lowering-runtime/emitc-route.md`; archived PRD `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-selected-exec-envelope-abi-closure/prd.md`; current RVV provider/planning/config/fixture/bundle code.
