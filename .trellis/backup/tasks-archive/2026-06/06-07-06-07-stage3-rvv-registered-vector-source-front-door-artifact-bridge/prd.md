# Stage3 RVV registered vector source-front-door artifact bridge

## Goal

Prove and harden one production workflow submodule: the artifact-capable bridge
from the two active registered RVV vector source-front-door families to the
existing selected typed RVV route path. The bridge must start from bounded
source-only Vector-like fixtures, run the matching plugin-owned registered
family materializer, produce a selected `tcrv.exec` RVV variant with typed
`tcrv_rvv` body facts, let the RVV provider build the
`TCRVEmitCLowerableRoute`, export the target artifact bundle, and record
evidence that treats all source marker and artifact fields as mirrors only.

## What I Already Know

* No current Trellis task existed at session start; this task was created from
  the Hermes direction brief.
* Commit `0b51bc00` and archived task
  `.trellis/tasks/archive/2026-06/06-07-stage3-rvv-vector-source-front-door-family-registry-boundary/`
  already introduced the plugin-local active family descriptor registry and
  rewired `RVVExtensionPlugin` source-front-door pass registration through it.
* The active registered families are exactly
  `bounded-vector-binary-source-front-door` and
  `bounded-vector-compare-select-source-front-door`.
* `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp` already materializes source
  fixtures into selected `tcrv.exec` RVV variants with typed `tcrv_rvv`
  `setvl`, `load`, `binary` or `compare/select`, and `store` bodies.
* `RVVExtensionPlugin` validates selected RVV variants, unique `setvl/with_vl`
  boundaries, construction protocol metadata, target capability facts, and
  provider route construction before emission support is reported.
* `scripts/rvv_generated_bundle_abi_e2e.py` already has a
  `--vector-source-front-door` mode that runs the matching materializer, emits
  C++ via the RVV EmitC route, exports a target artifact bundle, writes
  evidence JSON, and generates a harness.
* Existing dry-run script tests prove binary `add/sub/mul` and compare/select
  `eq` representative artifact bundle shape. They do not by themselves claim
  RVV runtime correctness.

## Assumptions

* The right module owner for this round is the generated-bundle/artifact
  evidence bridge around the registered source-front-door families, not a new
  RVV route family and not a new source parser.
* If current production code is already wired correctly, source changes may be
  limited to hardening the production evidence/tooling bridge and its tests.
* A representative compare/select artifact is sufficient for executable
  compare/select family evidence in this round; materializer lit coverage
  remains responsible for `eq/slt/sle` source-shape variants.
* Runtime correctness will be claimed only if a non-dry-run `ssh rvv` script
  execution succeeds in this round.

## Requirements

* Preserve the registered family production path:
  source-only Vector fixture -> matching registered family materializer ->
  selected typed `tcrv_rvv` body -> RVV provider route ->
  `TCRVEmitCLowerableRoute` -> Common EmitC -> target artifact bundle.
* Strengthen the generated-bundle/artifact bridge so evidence records explicit
  source-front-door family descriptor facts:
  family name, marker, materializer pass, selected variant prefix, runtime ABI
  purpose prefix, dispatch policy mirror, and default artifact-front-door
  eligibility.
* Validate those descriptor facts against each generated artifact expectation
  before artifact evidence is accepted.
* Validate that generated object/header bundle metadata agrees with the
  expected selected variant, function prototype, runtime ABI name/order,
  provider-supported mirror, route operand binding summary, selected-dispatch
  mirrors, typed compute op, and source-front-door family mirrors.
* Add focused fail-closed self-tests or lit coverage for descriptor mismatch
  cases that could otherwise make marker strings, selected variant names,
  runtime purpose prefixes, dispatch policy mirrors, or ABI roles look like
  authority.
* Keep family-specific source-shape parsing and typed body materialization
  owned by the two existing families.
* Keep Common EmitC and target export neutral; they may consume provider facts
  and mirror validated metadata but must not infer RVV semantics from source
  markers or artifact names.

## Acceptance Criteria

* [x] Binary source-front-door dry-run evidence proves registered family
  descriptor facts, selected typed body facts, provider route facts, target
  artifact export, header/object metadata agreement, harness call shape, source
  preservation, and tail sentinel checks for `add/sub/mul`.
* [x] Compare/select source-front-door dry-run evidence proves the same bridge
  for the active compare/select family representative.
* [x] Focused fail-closed coverage rejects descriptor/family mismatch before
  artifact evidence acceptance, including at least wrong selected variant
  prefix or wrong runtime purpose/dispatch policy facts.
* [x] Existing fail-closed coverage still rejects unsupported source-front-door
  op-kind, mixed source-front-door plus selected-body mode, unknown marker,
  stale lowering seed, stale TCRV residue, malformed source shape, unsupported
  arithmetic, unsupported predicate/layout, and legacy i32m1 source front door.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Focused lit/script filters for the touched source-front-door families
  pass.
* [x] If runtime correctness is claimed, non-dry-run generated-bundle evidence
  runs on real `ssh rvv` and records PASS output.
* [x] Bounded old-authority scan over touched files and added diff lines shows
  no new route authority from legacy i32m1, descriptor/direct-C/source-export,
  source marker, artifact name, route id, or Common EmitC semantics.
* [x] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are clean after commit.

## Non-Goals

* No new source-front-door family.
* No high-level Linalg/Vector/StableHLO frontend generalization beyond the
  bounded existing fixtures.
* No source-front-door route authority or source-artifact shortcut.
* No Common EmitC invention of RVV semantics.
* No dtype, LMUL, MAcc, reduction, segment, memory-form, or performance
  expansion.
* No resurrection of legacy `RVVI32M1*`, `rvv-i32m1`, `tcrv_rvv.i32_*`, or
  `!tcrv_rvv.i32m*` route authority.
* No dashboard/report-only completion.

## Technical Notes

Read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* archived registry-boundary task PRD/context under
  `.trellis/tasks/archive/2026-06/06-07-stage3-rvv-vector-source-front-door-family-registry-boundary/`
* `include/TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h`
* `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`
* `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Transforms/RVV/rvv-vector-binary-source-front-door.mlir`
* `test/Transforms/RVV/rvv-vector-compare-select-source-front-door.mlir`
* `test/Transforms/RVV/rvv-vector-source-front-door-family-registry-negative.mlir`
* source-front-door generated-bundle script tests under `test/Scripts/`.

## Completion Evidence

### Production Bridge

* Added an evidence-tooling contract for the two active registered source
  families:
  `bounded-vector-binary-source-front-door` and
  `bounded-vector-compare-select-source-front-door`.
* The script now validates source-front-door family name, marker, materializer
  pass, selected variant prefix, runtime purpose prefix, dispatch policy
  mirror, default artifact-front-door eligibility, expected generated function,
  selected dispatch mirrors, and materialized selected-body runtime purpose
  uses before accepting source-front-door artifact evidence.
* `source_front_door_artifact_boundary`, `local_bundle_generation`, and
  `materialized_selected_body_checks` now expose registered family contract
  evidence while preserving marker/artifact metadata as mirror-only.
* Added self-test fail-closed coverage for selected variant prefix mismatch,
  selected dispatch policy mismatch, materialized runtime purpose prefix
  mismatch, and materialized dispatch policy mismatch.
* Updated binary and compare/select source-front-door dry-run FileCheck tests to
  assert the registered family contract fields and materialized agreement
  flags.

### Evidence Commands

* `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter "rvv-generated-bundle-abi-e2e-(vector-source-front-door-dry-run|vector-compare-select-source-front-door-dry-run|self-test)"` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter "rvv-vector-(binary-source-front-door|compare-select-source-front-door|source-front-door-family-registry-negative)|rvv-i32m1-vector-source-front-door|rvv-generated-bundle-abi-e2e-vector-source-front-door-fail-closed"` from `build/test`
* `build/bin/tianchenrv-rvv-extension-plugin-test`
* `build/bin/tianchenrv-target-artifact-export-test`
* Non-dry-run binary family `ssh rvv` evidence:
  `add/sub/mul` passed for counts `0,1,17,257`; harness output included
  `source_preserved tail_preserved`.
* Non-dry-run compare/select family `ssh rvv` evidence:
  `cmp_select` passed for counts `0,1,17,257`; output included true/false
  predicate-lane coverage and `tail_preserved`.
* Added-diff-line old-authority scan over touched script/test files found no
  new legacy i32m1, descriptor/direct-C/source-export, route-id, artifact-name,
  or Common EmitC authority terms.
* `git diff --check` passed.
