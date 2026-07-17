# RVV source-front-door positive-route demotion

## Goal

Demote the RVV vector source-front-door production surface so
`tcrv_rvv.source_front_door` markers cannot act as default route, artifact, or
executable authority. The only positive RVV artifact chain in this task remains:

```text
explicit selected tcrv.exec RVV variant
  -> typed tcrv_rvv body facts
  -> RVV provider legality / route construction
  -> TCRVEmitCLowerableRoute
  -> target artifact export
```

Explicit source-front-door materializer passes may remain as non-default
fixture/import scaffolding only when they immediately create a typed selected
body and the later provider still owns all route facts. They must not be
eligible for the default source-artifact front-door pipeline, and the generated
bundle evidence script must not accept `--vector-source-front-door` as a
positive artifact mode.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* Repository inspection started from clean `main` at
  `a8e3080b rvv: record runtime scalar segment2 ssh evidence`.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-executable-artifact-abi/`
  completed runtime-scalar masked segment2 load/store evidence and left no
  production source changes.
* `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md` state that
  source-front-door metadata is not route/artifact authority and that
  `DefaultArtifactFrontDoorPolicy::Eligible` is unsafe for current RVV
  Stage1/Stage2 work.
* `lib/Transforms/ExecutionPlanningPipeline.cpp` only inserts source
  materializer passes into `--tcrv-source-artifact-front-door-pipeline` when
  `SourceFrontDoorPassRegistration::isDefaultArtifactFrontDoorEligible()` is
  true.
* `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp` currently registers the
  bounded vector binary, compare/select, and runtime-scalar compare/select
  source-front-door families with `DefaultArtifactFrontDoorPolicy::Eligible`.
* `test/Plugin/RVVExtensionPluginTest.cpp` currently asserts those three RVV
  source-front-door family registrations are eligible for the default artifact
  front-door pipeline.
* The explicit materializer passes currently create selected
  `tcrv.exec.variant` bodies containing generic typed `tcrv_rvv` ops
  (`load`, `binary`, `compare`, `splat`, `select`, `store`) before provider
  route construction.
* `scripts/rvv_generated_bundle_abi_e2e.py --vector-source-front-door`
  currently performs positive selected-body materialization, EmitC export,
  target artifact bundle export, harness generation, and dry-run evidence.
* The directly related script tests
  `test/Scripts/rvv-generated-bundle-abi-e2e-vector-source-front-door-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-vector-compare-select-source-front-door-dry-run.test`,
  and
  `test/Scripts/rvv-generated-bundle-abi-e2e-vector-runtime-scalar-cmp-select-source-front-door-dry-run.test`
  currently expect positive generated-bundle output from source-front-door
  markers.

## Requirements

* Keep the owner bounded to RVV vector source-front-door registration,
  materialization policy, generated-bundle script behavior, directly related
  tests, and spec text that directly contradicts the demotion.
* Make all RVV bounded vector source-front-door registrations explicit-only or
  otherwise non-eligible for the default source-artifact front-door pipeline.
* Preserve the existing explicit materializer passes only as non-default
  typed-body scaffolding. They may derive bounded source pattern facts into a
  typed `tcrv_rvv` body, but they must not by themselves claim route support,
  artifact support, or executable status.
* Ensure `--tcrv-source-artifact-front-door-pipeline` no longer produces
  positive RVV source-front-door emission plans from source-only RVV markers.
* Ensure `scripts/rvv_generated_bundle_abi_e2e.py --vector-source-front-door`
  fails closed before selected-body materialization or target artifact bundle
  export.
* Remove or repair positive generated-bundle script tests that assert RVV
  source-front-door marker-to-bundle success.
* Keep explicit selected-body and pre-realized selected-body generated-bundle
  modes unaffected.
* Keep Toy and TensorExtLite source-front-door behavior out of scope unless a
  shared registry invariant must be preserved.

## Acceptance Criteria

* [x] RVV bounded vector source-front-door family registrations are not default
  artifact-front-door eligible.
* [x] C++ plugin tests assert all RVV source-front-door registrations,
  including bounded vector families, are explicit-only / non-default for
  artifact-front-door use.
* [x] `--tcrv-source-artifact-front-door-pipeline` on RVV bounded vector
  source-front-door inputs fails before positive RVV emission-plan or artifact
  facts are produced.
* [x] The generated-bundle ABI script rejects `--vector-source-front-door`
  before materialization/export, with a diagnostic naming the retired mode and
  directing users to explicit typed selected-body modes.
* [x] Directly related script tests no longer assert positive RVV
  source-front-door generated bundle success; they instead prove fail-closed
  behavior.
* [x] Explicit materializer lit tests still prove that manual opt-in
  materialization creates typed selected-body facts and that provider route
  planning remains downstream of those facts, not marker metadata.
* [x] No new RVV source-front-door positive route, object/header/bundle,
  generated-bundle success, artifact metadata authority, descriptor route, or
  legacy `i32m1` authority is introduced.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Directly affected lit tests for RVV source-front-door, source-artifact
  front-door pipeline, and script fail-closed behavior pass.
* [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passes.
* [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passes if
  the script self-test contract is touched or remains relevant after the mode
  is retired.
* [x] Focused before/after scans for `SourceFrontDoor`, `source_front_door`,
  and `source-front-door` over `include/TianChenRV/Plugin/RVV`,
  `lib/Plugin/RVV`, directly related tests, and `.trellis/spec` show no
  remaining positive RVV source-front-door artifact authority.
* [x] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are clean or explicitly reported.
* [x] Trellis task context, workspace journal, finish/archive status, and one
  coherent commit are completed if the task finishes.

## Definition of Done

RVV source-front-door markers can no longer enter the default source-artifact
or generated-bundle artifact path. Any retained explicit materializer is only a
manual typed-body import helper, and all route/artifact support still comes
from the typed `tcrv_rvv` body plus RVV provider and target validators.

## Out Of Scope

* No new RVV source-front-door family or source pattern.
* No new positive source-front-door object/header/bundle test.
* No new dtype, LMUL, operation, reduction, MAcc, memory, segment2, or indexed
  route coverage.
* No high-level frontend, Linalg/Vector/StableHLO lowering expansion, or
  source-shape generalization.
* No common EmitC inference of RVV semantics.
* No Toy/TensorExtLite policy change unless required by shared registry
  invariants.
* No `ssh rvv` claim is required because this task removes an artifact
  authority path rather than claiming runtime correctness.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
* Previous task read:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-runtime-scalar-cmp-masked-segment2-executable-artifact-abi/`.
* Primary production files:
  `include/TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h`,
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVExtensionPlugin.h`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `include/TianChenRV/Plugin/ExtensionPlugin.h`,
  `lib/Plugin/ExtensionPlugin.cpp`, and
  `lib/Transforms/ExecutionPlanningPipeline.cpp`.
* Primary script/test files:
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Transforms/RVV/rvv-vector-binary-source-front-door.mlir`,
  `test/Transforms/RVV/rvv-vector-compare-select-source-front-door.mlir`,
  `test/Transforms/RVV/rvv-vector-runtime-scalar-cmp-select-source-front-door.mlir`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-vector-source-front-door-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-vector-compare-select-source-front-door-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-vector-runtime-scalar-cmp-select-source-front-door-dry-run.test`, and
  `test/Scripts/rvv-generated-bundle-abi-e2e-vector-source-front-door-fail-closed.test`.
* Spec update completed:
  `.trellis/spec/extension-plugins/rvv-plugin.md` now describes RVV bounded
  vector source-front-door families as explicit-only materializer scaffolding,
  not default source-artifact or generated-bundle artifact authority.

## Results

* `RVVVectorSourceFrontDoor` now registers bounded vector binary,
  compare/select, and runtime-scalar compare/select source-front-door families
  with `DefaultArtifactFrontDoorPolicy::ExplicitOnly`.
* `test/Plugin/RVVExtensionPluginTest.cpp` now asserts that all RVV
  source-front-door families, including bounded vector families, are
  non-default for artifact-front-door use.
* The default `--tcrv-source-artifact-front-door-pipeline` no longer inserts
  the RVV bounded vector source-front-door materializer passes, so source-only
  RVV markers fail before emission-plan or target artifact facts are produced.
* `scripts/rvv_generated_bundle_abi_e2e.py --vector-source-front-door` is
  retired and fails before selected-body materialization or target artifact
  bundle export. Explicit selected-body and pre-realized selected-body modes
  remain the positive generated-bundle evidence paths.
* Directly related generated-bundle script tests were converted from positive
  source-front-door dry-run success to negative fail-closed coverage.
* Explicit materializer lit coverage remains, but only under direct manual pass
  invocation; that path creates typed selected-body facts and leaves route
  support downstream of provider checks.
* No `ssh rvv` evidence was required or claimed because this task removes an
  alternate artifact-authority path rather than asserting runtime correctness.
