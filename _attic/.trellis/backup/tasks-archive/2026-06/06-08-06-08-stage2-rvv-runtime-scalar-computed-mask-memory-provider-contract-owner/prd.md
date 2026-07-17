# Stage2 RVV runtime-scalar computed-mask memory provider contract owner

## Goal

Move repeated runtime-scalar computed-mask memory facts into a provider-owned
contract boundary that is consumed before `TCRVEmitCLowerableRoute`
construction and mirrored by target validation. The immediate production seam
is runtime-scalar computed-mask memory, especially the segment2 load/store
routes that currently carry `rhs_scalar` ABI/splat, compare-produced mask,
active/inactive lane policy, memory form, operand roles, ABI order,
header/type, runtime AVL/VL, and provider-supported mirror facts through
several local checks.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* Repository inspection started from clean `main` at commit `6920a911`.
* Recent archived load and store tasks closed specific runtime-scalar
  segment2 seams with classification fixes, focused C++ coverage, and `ssh rvv`
  evidence, but the store task explicitly did not change production source.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires segment2 memory
  routes to use RVV-owned family plans, materialization facts, memory
  operand-binding facts, route-control facts, and statement-plan owners before
  provider route construction.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires provider-owned
  route-family fact surfaces and embedded runtime AVL/VL selected-boundary
  contracts for route families consumed by target validation.
* Source inspection found `verifyRVVSelectedBodyRuntimeScalarComputedMaskMemoryRouteProviderFacts`
  covers non-segment runtime-scalar memory/indexed/composite routes and
  explicitly rejects segment2 facts.
* Segment2 runtime-scalar load/store routes use
  `verifyRVVSelectedBodySegment2MemoryRouteProviderFacts`, which checks many
  computed-mask memory and segment2 facts independently.
* Target segment2 validation already checks runtime-scalar segment2 load/store
  statements for `rhs_scalar` splat before compare, but provider verification
  only requires `rhsScalarBroadcastLeaf` and the splat statement leaf for
  runtime-scalar segment2 store, not runtime-scalar segment2 load.

## Requirements

* Keep the task bounded to source-level runtime-scalar computed-mask memory
  provider/statement/target-validation contracts.
* Do not add a new route family, feature expansion, dtype/LMUL clone batch,
  source-front-door route, descriptor-driven compute path, or Common EmitC RVV
  semantic inference.
* Preserve route authority in selected typed or realized `tcrv_rvv` body facts,
  RVV family plans, route materialization facts, route-control facts,
  operand-binding facts, statement plans, and provider-built route
  descriptions.
* Make the provider fail closed when runtime-scalar computed-mask memory facts
  are stale or incomplete before route construction, including
  `rhs_scalar` ABI/splat, compare mask source, runtime ABI order, provider
  mirror, memory form, active/inactive lane contract, statement leaves, and
  runtime AVL/VL facts.
* The focused production repair should cover the currently observed stale
  consumer: runtime-scalar computed-mask segment2 load must require the same
  runtime-scalar splat materialization and statement-plan leaf as store.
* Target validation should continue to consume rebuilt provider contracts and
  metadata mirrors only; no target-local route truth should be introduced.
* Additional tests are evidence for the production contract change only.
* Real `ssh rvv` evidence is not required unless emitted runtime behavior,
  ABI order, runtime counts, pointer advancement, or correctness behavior is
  changed.

## Acceptance Criteria

* [x] A production source diff in the RVV provider/statement/target-validation
      ownership path makes runtime-scalar computed-mask memory facts a shared
      provider-consumed contract or repairs an existing stale production
      consumer with exact source justification.
* [x] Runtime-scalar computed-mask segment2 load and store provider preflight
      both require `rhs_scalar` ABI/splat materialization and statement-plan
      leaves before `TCRVEmitCLowerableRoute` construction.
* [x] Focused C++ coverage proves the positive runtime-scalar segment2
      provider path still consumes computed-mask family facts, segment2
      provider plan, statement-plan owner, runtime scalar ABI/splat facts,
      provider-supported mirror, and route consumption.
* [x] Focused fail-closed C++ coverage mutates a stale or missing
      runtime-scalar splat fact for segment2 load and fails before provider
      route construction.
* [x] Representative lit/generated-bundle checks for the touched
      runtime-scalar computed-mask segment2 seam remain passing; dry-run
      evidence is enough unless executable behavior is claimed.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Bounded old-authority scan over touched files and added diff lines finds
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C, source-front-door, or
      mirror-only route authority.
* [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are clean or explicitly reported.

## Definition Of Done

* The runtime-scalar computed-mask memory provider contract fails closed before
  route construction for the observed stale segment2 load splat consumer.
* Provider and target validation remain aligned around provider-owned facts and
  mirrors, with Common EmitC kept neutral.
* Trellis task context, journal, archive status, and one coherent commit are
  completed if all checks pass.

## Out Of Scope

* No broad smoke matrix.
* No dashboard/report/index work.
* No source-front-door positive route.
* No high-level frontend.
* No Stage3 plugin generality.
* No dtype/LMUL clone batch.
* No unrelated MAcc, reduction, gearbox, segment2 feature expansion, or
  common EmitC RVV semantic inference.
* No new executable runtime claim without `ssh rvv` evidence.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
* Archived tasks read:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-load-artifact-abi-boundary/`
  and
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-store-artifact-abi-boundary/`.
* Source inspected before implementation:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Results

* Production source changed:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` now exposes
  `RVVRuntimeScalarComputedMaskMemorySplatProviderContract` plus
  `verifyRVVRuntimeScalarComputedMaskMemorySplatProviderContract(...)`.
* The shared provider contract checks runtime ABI order, provider-supported
  mirror, `rhs_scalar` ABI binding and role, runtime-scalar computed-mask
  operand binding, provider-owned splat intrinsic, materialization splat leaf,
  and the RVV-owned statement-plan splat leaf before route construction.
* Non-segment runtime-scalar computed-mask memory routes now consume this
  shared contract instead of locally checking the splat materialization and
  statement leaf.
* Segment2 runtime-scalar computed-mask load/store provider preflight now
  consumes the same contract. This repairs the stale load consumer: load now
  requires `rhsScalarBroadcastLeaf` and the statement-plan splat leaf before
  `TCRVEmitCLowerableRoute` construction, matching store and target artifact
  validation.
* Focused C++ coverage in `test/Plugin/RVVExtensionPluginTest.cpp` now runs a
  producer-realized runtime-scalar segment2 load through provider preflight and
  mutates `rhsScalarBroadcastLeaf` to prove fail-closed rejection at the new
  splat provider contract.
* Existing target validation already checks runtime-scalar segment2 load/store
  statement shape, including the `rhs_scalar` splat before compare; no target
  source rewrite was required.
* Checks run:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `cmake --build build --target tianchenrv-target-artifact-export-test`,
  `build/bin/tianchenrv-target-artifact-export-test`,
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'runtime-scalar-cmp-masked-segment2'`
  from `build/test`, `git diff --check`, and a bounded added-line
  old-authority scan. `git diff --cached --check` and final clean status were
  run after staging/commit.
* Runtime evidence judgment:
  no new `ssh rvv` run was required because this round changes provider
  fail-closed validation only and does not change emitted runtime ABI order,
  generated statement semantics, runtime counts, pointer advancement, or
  correctness behavior.
* Spec update:
  `.trellis/spec/lowering-runtime/emitc-route.md` now records the executable
  `RVVRuntimeScalarComputedMaskMemorySplatProviderContract` API shape,
  contracts, error matrix, tests, and wrong-vs-correct flow so future provider
  owners do not leave runtime-scalar splat checking to target validation.
