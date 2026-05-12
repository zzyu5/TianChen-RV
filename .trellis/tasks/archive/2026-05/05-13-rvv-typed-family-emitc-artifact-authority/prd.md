# RVV typed-family EmitC artifact authority

## Goal

Ensure the production/default RVV binary microkernel artifact path is authorized
by typed `tcrv_rvv` family/body operations, generated
`TCRVEmitCLowerableOpInterface` provenance, and the common
`TCRVEmitCLowerableRoute` / lower-to-EmitC source-authority boundary. Legacy
`tcrv_rvv.lowering_descriptor` metadata may remain only as optional mirror data
after typed authority exists, and descriptor-only production inputs must fail
closed before source/header/object export.

## What I Already Know

* Clean starting HEAD is `5ba4385 chore(supervisor): bound Hermes review inspection`.
* The predecessor archived task
  `.trellis/tasks/archive/2026-05/05-12-05-12-direct-artifact-route-plan-authority/`
  moved marked direct artifact routes through planning, coherence, and exact
  target artifact export before target-local callbacks.
* Current source already has typed RVV finite binary ops implementing
  `TCRVEmitCLowerableOpInterface` in `RVVOps.td`.
* Current RVV body verification queries the generated op interface before
  constructing the EmitC route and records interface-backed provenance in the
  generated source.
* Current direct descriptor-only i32 RVV production input fails during selected
  lowering-boundary materialization before exact target artifact export. This
  round should pin that contract in focused lit coverage instead of adding fake
  source churn.

## Requirements

* Production RVV binary source export must consume a selected typed
  `tcrv_rvv.*_microkernel` body and its structured
  `setvl -> with_vl -> load/load/arithmetic/store` dataflow.
* RVV arithmetic dataflow ops must provide generated
  `TCRVEmitCLowerableOpInterface` provenance before common EmitC route
  construction.
* Generated source must record common EmitC source-authority evidence,
  interface-backed compute provenance, route source ops, and call_opaque
  mapping.
* Descriptor-only RVV production inputs must fail closed before source, header,
  object, or exact artifact output.
* Legacy `tcrv_rvv.lowering_descriptor`, if present, is mirror metadata only:
  it must agree with the typed body and must not select the family, route,
  runtime ABI, or emitted body by itself.
* Keep `tcrv-translate` and common routing family-neutral; RVV-specific behavior
  stays in RVV plugin/target materialization, planning, validation, and export.

## Acceptance Criteria

* [x] Auto-planned direct RVV runtime-callable C source from descriptor-free
      typed RVV binary IR emits common EmitC source-authority evidence.
* [x] Generated source includes interface-backed EmitC route provenance rather
      than descriptor authority.
* [x] Descriptor-only RVV production input fails closed before source output
      through the artifact-backed direct route path.
* [x] Retained descriptor mirror mismatch fails closed after typed body
      authority is known.
* [x] Marked artifact-backed routes continue to use common planning/coherence
      and exact target artifact export, not target-local production fallback.
* [x] Focused build, C++ target artifact test, focused lit, `git diff --check`,
      `git diff --cached --check`, and Trellis validation pass.

## Definition Of Done

* Focused C++/MLIR changes are scoped to RVV typed-family artifact authority or
  its regression coverage.
* No compiler core, dialect, pass, plugin registry, lowering, or emission logic
  is implemented in Python.
* No new arithmetic family, dtype family, backend detour, runtime correctness
  claim, performance claim, or standalone `ssh rvv` claim is introduced.
* Trellis task status and journal accurately describe whether this round changed
  behavior or pinned already-present behavior.

## Technical Approach

Use the existing RVV plugin/target-owned authority path as the production
surface: selected path planning materializes typed RVV bodies, body validation
queries `TCRVEmitCLowerableOpInterface`, RVV target code maps the verified
family ops to a bounded `TCRVEmitCLowerableRoute`, and the common conversion
boundary emits MLIR EmitC / C++ source. This round adds focused negative lit
coverage for descriptor-only i32 production input through the marked direct
artifact route, because live inspection and a local probe show the source code
already implements the fail-closed behavior.

## Decision (ADR-lite)

**Context**: The Hermes brief names a migration from descriptor authority to
typed-family EmitC authority. Live repository inspection shows the migration is
already present for the bounded RVV source route, but the descriptor-only i32
production rejection is not directly pinned by a focused lit test.

**Decision**: Do not invent source changes when current code already enforces
the desired boundary. Add a focused lit regression for descriptor-only RVV i32
production input and validate the existing typed family/body source route.

**Consequences**: The commit is intentionally evidence-heavy but still tied to
the production artifact behavior named by the task. Follow-up source/header/
object changes should be made only if focused checks expose a real gap.

## Out Of Scope

* New arithmetic or dtype family behavior.
* MLIR vector, LLVM RVV intrinsic IR, inline assembly, backend patches, or
  backend detours.
* Computation semantics in `tcrv.exec`.
* RVV-specific branches in shared/core routing or common passes.
* Python compiler internals.
* Runtime correctness, hardware execution, or performance claims without exact
  `ssh rvv` evidence.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior task read:
  `.trellis/tasks/archive/2026-05/05-12-05-12-direct-artifact-route-plan-authority/prd.md`.
* Relevant implementation surfaces inspected:
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Conversion/EmitC/`,
  `lib/Conversion/EmitC/`,
  `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`,
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`,
  `lib/Target/RVV/RVVBinaryMicrokernelBodyVerifier.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `test/Target/RVVMicrokernel/`,
  `test/Target/RVVScalarDispatch/`.

## Completion Notes

* Added focused lit regression
  `test/Target/RVVMicrokernel/rvv-microkernel-descriptor-only-production-rejects.mlir`.
* The regression proves `tcrv-translate --tcrv-export-rvv-i32-vmul-microkernel-c`
  rejects descriptor-only RVV production input during artifact-backed direct
  route planning before exact target artifact export and before any RVV source
  is emitted.
* No `.trellis/spec/` update was needed: the existing EmitC route, emission
  runtime, RVV plugin, plugin protocol, and testing specs already describe the
  typed-family authority and descriptor-only fail-closed contract.
