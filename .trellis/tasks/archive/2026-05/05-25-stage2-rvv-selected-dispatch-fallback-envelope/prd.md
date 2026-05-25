# Stage2 RVV selected dispatch/fallback envelope closure

## Goal

Close one bounded selected execution envelope around the already route-supported
RVV `scalar_broadcast_macc_add` path: an explicit selected `tcrv.exec.dispatch`
case must carry selected-variant, target/capability, runtime guard, and
fallback-envelope facts into the RVV plugin route handoff, while unsupported,
ambiguous, or unselected paths fail closed before RVV route or artifact
authority is produced.

## What I Already Know

* Current HEAD is `9a43fd2e rvv: gate selected route on target capabilities`.
* The previous archived task
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-target-capability-selected-route-gating/prd.md`
  closed target/capability gating for `scalar_broadcast_macc_add`: selected
  target/capability facts gate route support before EmitC, SEW/LMUL/policy
  mismatches fail closed, and artifact metadata mirrors explicit capability
  facts.
* `tcrv.exec` owns execution-envelope structure only: kernel, target,
  capability, variant, dispatch, case, fallback, diagnostics, ABI and runtime
  provenance. It does not own RVV compute, dtype, policy, memory form, selected
  route authority, intrinsic spelling, or fallback compute semantics.
* The selected typed or realized `tcrv_rvv` body/config/runtime structure
  remains authority for operation kind, dtype, SEW/LMUL, policy, memory form,
  scalar role, accumulator role, and VL/AVL use.
* The RVV plugin owns capability legality, selected-body realization, route
  provider construction, intrinsic/type/header/ABI mapping, and fail-closed
  diagnostics.
* Common EmitC and target artifact export may consume a provider-built
  `TCRVEmitCLowerableRoute` and mirror provider facts after route construction;
  they must not infer RVV semantics from dispatch, names, metadata, route ids,
  ABI strings, artifacts, manifests, or harness constants.
* The core dispatch contract requires a `tcrv.exec.dispatch` directly under a
  kernel, at least one `tcrv.exec.case`, exactly one `tcrv.exec.fallback`, and
  same-kernel `runtime_guard = @...` linkage to a
  `dispatch-availability-guard` runtime parameter when
  `runtime_guard_required = true`.
* A fallback branch must point to a same-kernel `tcrv.exec.variant` that is
  explicitly fallback-eligible, such as `fallback_role = "conservative"`.
  Core or RVV code must not invent fallback semantics from target family names
  or detached callable/artifact candidates.

## Requirements

1. Add one coherent module-level closure for selected dispatch/fallback around
   the existing `scalar_broadcast_macc_add` RVV selected-body route.
2. Ensure a capability-legal selected RVV dispatch case reaches RVV plugin
   legality/realization/route construction through structural `tcrv.exec`
   variant/case/fallback/runtime-guard facts.
3. Require selected dispatch coherence before RVV route or target artifact
   export can claim selected-dispatch authority: selected RVV case, same-kernel
   variant, explicit runtime guard when required, and exactly one explicit
   fallback target must be structurally valid.
4. Fail closed before provider route or artifact export when RVV capability
   facts are missing, incompatible, ambiguous, unselected, or only present as
   target names, route ids, metadata, ABI strings, artifact names, manifests,
   descriptors, source-front-door markers, or harness constants.
5. Preserve the existing fallback boundary as an envelope/source-of-truth only.
   This task may verify or mirror fallback linkage, but must not implement new
   scalar compute fallback behavior.
6. Keep artifact and emission-plan metadata mirror-explicit. Any provider or
   export metadata reflecting selected dispatch/fallback facts must use
   mirror-labeled fields and remain downstream of RVV provider route
   construction.
7. Keep common EmitC and target artifact plumbing neutral. Any new semantic
   dispatch/fallback/RVV legality choice must live in core generic envelope
   checks or the RVV plugin, not common EmitC.
8. Add focused verifier/transform/provider/FileCheck coverage for the positive
   selected dispatch handoff and negative unsupported, ambiguous, or
   fallback-incoherent paths.

## Acceptance Criteria

* [x] Positive FileCheck evidence shows a selected `tcrv.exec.dispatch` RVV
      case for `scalar_broadcast_macc_add` with explicit target/capability
      facts reaches RVV plugin legality and provider route metadata.
* [x] Positive generated artifact evidence mirrors selected dispatch case,
      selected variant, provider-supported status, target/capability facts,
      and fallback-envelope linkage with explicit mirror labels, without using
      those mirrors as authority.
* [x] Negative tests fail closed before provider route or artifact export for
      at least: missing required runtime guard on selected dispatch case,
      unresolved/wrong-role runtime guard linkage, missing or unavailable RVV
      capability, ambiguous target/capability ownership, and missing or
      ineligible fallback target.
* [x] Unsupported or fallback-only paths produce no RVV route/artifact
      authority unless an existing explicit fallback contract is structurally
      present; no new fallback compute implementation is introduced.
* [x] Bounded scan over touched exec/RVV/target/script/fixture files finds no
      dispatch, route, fallback, compute, dtype, policy, executable, or artifact
      authority derived from target names, route ids, metadata, ABI strings,
      artifacts, manifests, descriptors, common EmitC, source-front-door
      markers, harness constants, or legacy i32 route authority.
* [x] Focused tests for the changed behavior pass.
* [x] `git diff --check` passes.
* [x] `check-tianchenrv` passes, or an exact blocker is recorded.
* [x] If runtime/correctness behavior is claimed, a generated-bundle dry-run and
      one `ssh rvv` compile/run for the selected legal path across at least
      three runtime counts are recorded.

## Definition Of Done

* Task status, PRD, context files, and workspace journal reflect the final state
  truthfully.
* Relevant tests/checks are run and focused failures caused by this work are
  repaired.
* The task is finished and archived when complete.
* One coherent commit is created if the task reaches completion.

## Out Of Scope

* New RVV operation families, scalar compute fallback implementations,
  high-level Linalg/Vector/StableHLO frontend lowering, source-front-door
  positive routes, runtime feature-detection databases, broad dispatch
  matrices, autotuning, performance claims, dtype/LMUL expansion, legacy i32
  authority, one-op-per-intrinsic wrappers, dashboards, broad smoke matrices,
  IME/Offload/TensorExt work, or future plugin work.
* Inferring dispatch, fallback, route, compute, dtype, policy, executable
  behavior, or artifact authority from target names, route ids, artifact names,
  test names, ABI strings, manifests, descriptor residue, common EmitC, or
  harness-only constants.

## Technical Notes

* Created from the Hermes Direction Brief in the Codex worker prompt on
  2026-05-25.
* Specs read before PRD: `.trellis/spec/index.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/capability-model/capability-contract.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and the three design
  guides under `.trellis/spec/guides/`.
* Prior task context read:
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-target-capability-selected-route-gating/prd.md`.
* Implemented surfaces: `RVVEmitCRoutePlanning.cpp` selected dispatch/fallback
  envelope collection and fail-closed diagnostics,
  `RVVEmitCRouteProvider.h` selected dispatch mirror fields,
  `RVVTargetSupportBundle.cpp` mirror validation/header evidence, and focused
  `scalar_broadcast_macc_add` positive/negative Target RVV tests.
* Focused lit passed from `build/test`:
  `selected-dispatch-fallback-envelope-scalar-broadcast-macc-negative`,
  `explicit-selected-body-artifact-scalar-broadcast-macc-add`,
  `pre-realized-selected-body-artifact-scalar-broadcast-macc-add`, and
  existing `rvv-target-capability-selected-route-gating` coverage.
* Generated-bundle dry-run evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/selected-dispatch-fallback-envelope-dry`,
  status `dry_run_success`; artifact/header/index evidence contains
  `tcrv_rvv.selected_dispatch_case_mirror` and
  `tcrv_rvv.selected_dispatch_fallback_mirror`.
* Real `ssh rvv` evidence:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/selected-dispatch-fallback-envelope-ssh`,
  status `success`, `ssh_evidence=true`; the generated bundle passed
  `scalar_broadcast_macc_add` for runtime counts `7,16,23` and RHS scalars
  `-37,91`.
* Final checks: `git diff --check` passed; `check-tianchenrv` passed
  379/379; changed-line authority scan over touched C++/MLIR files found no
  new descriptor/source-front-door/common-EmitC/harness/name-derived or legacy
  i32 route-authority additions.
