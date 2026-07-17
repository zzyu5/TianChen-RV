# Stage2 RVV f32 Clamp/Select Executable ABI Closure

## Goal

Close executable support for the existing Stage2 RVV f32 clamp/select route
added in `b9a315dc`.

The bounded path under test is:

```text
selected tcrv.exec RVV variant
  -> typed f32 pre-realized clamp/select tcrv_rvv body
  -> RVV plugin-local selected-body realization
  -> setvl/load/splat-lower/splat-upper/compare/select/store
  -> provider-derived TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> generated header/object bundle
  -> external C ABI harness
  -> ssh rvv correctness evidence
```

This task improves the executable support level for one already route-supported
typed RVV route. It must not add new RVV route coverage or move semantics into
the harness, route ids, artifact names, ABI strings, or metadata mirrors.

## What I Already Know

* The previous task `stage2-rvv-f32-clamp-select-route-foundation` is archived
  as completed and added the production route foundation.
* The positive fixture
  `test/Target/RVV/pre-realized-selected-body-artifact-f32-clamp-select.mlir`
  already starts from a typed pre-realized f32 clamp/select body and checks
  selected-boundary realization, emission-plan mirrors, target header export,
  and stale mirror rejection.
* `test/Dialect/RVV/pre-realized-f32-clamp-select-negative.mlir` already
  covers missing/stale roles, input dtype, config, policy, bound order, and
  route-authority metadata rejection.
* `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md` require typed `tcrv_rvv`
  body/config/runtime facts to be route authority, with common EmitC remaining
  neutral.
* `.trellis/spec/testing/mlir-testing-contract.md` requires real `ssh rvv`
  output before claiming runtime/correctness evidence and tail/sentinel
  preservation for memory-writing generated-bundle evidence.
* `scripts/rvv_generated_bundle_abi_e2e.py` is evidence tooling only and is the
  right place for a minimal external ABI harness if it starts from the existing
  MLIR fixture and verifies provider/export facts rather than choosing route
  semantics.

## Requirements

* Use the existing typed f32 clamp/select MLIR fixture as source authority.
* Prove selected-body realization happens before emission; the evidence must
  reject or fail if the pre-realized body remains in the emitted route path.
* Bind exported ABI roles in this exact order:
  `input,lower_bound,upper_bound,out,n`.
* The generated C ABI harness must call the generated function with:
  `const float *input`, runtime `float lower_bound`, runtime `float upper_bound`,
  `float *out`, and `size_t n`.
* Runtime correctness evidence must compare against a host/reference clamp:
  `min(max(input[i], lower_bound), upper_bound)`.
* Runtime cases must cover:
  * multiple runtime counts, including at least one count larger than a single
    vector chunk;
  * input patterns containing values below, inside, and above the bounds;
  * at least two nontrivial ordered bound pairs;
  * source preservation after the generated call;
  * output tail sentinel preservation beyond `n`;
  * explicit f32 tolerance.
* Harness support must remain neutral evidence tooling. It may generate test
  input/reference values for the external ABI consumer, but it must not choose
  compiler route support from route ids, artifact names, ABI strings, metadata
  mirrors, or op-kind strings.

## Acceptance Criteria

* [ ] A focused generated-bundle dry run succeeds for
      `--pre-realized-selected-body --op-kind f32_clamp_select`.
* [ ] The dry-run artifact/evidence proves the input fixture is the f32
      clamp/select pre-realized selected-body fixture and that selected-body
      realization ran before emission.
* [ ] The dry-run generated harness checks bound-pair loops, runtime counts,
      below/inside/above-bound patterns, f32 tolerance, source preservation,
      and tail sentinel preservation.
* [ ] Real `ssh rvv` compile/run succeeds before claiming executable
      correctness.
* [ ] Runtime output or evidence JSON records the tested counts, bound pairs,
      pattern coverage, tolerance, source preservation, and tail sentinel
      preservation.
* [ ] If only harness/evidence tooling changes, production compiler code is not
      modified.
* [ ] If production compiler code must change due to a verifier/provider/target
      bug, run the focused lit check plus
      `tianchenrv-rvv-extension-plugin-test` and
      `tianchenrv-target-artifact-export-test`.
* [ ] Run `py_compile` or an equivalent syntax check for script changes.
* [ ] Run the focused lit check for the f32 clamp/select fixture if the fixture
      or production path is touched.
* [ ] Run a bounded old-authority and q-name-authority scan over touched files.
* [ ] `git diff --check` and `git diff --cached --check` pass before commit.
* [ ] Trellis task status and journal are truthful, task is finished/archived
      when complete, one coherent commit is created, and the final worktree is
      clean.

## Definition Of Done

* The existing f32 clamp/select typed route has generated-bundle dry-run
  evidence and real `ssh rvv` correctness evidence through an external ABI
  harness.
* The evidence demonstrates the compiler path, not harness-derived route
  authority.
* Any script addition is limited to neutral evidence generation for this
  route.

## Out Of Scope

* New route coverage or route-family expansion.
* Unrelated low-precision benchmark routes.
* Zero-point or clamp fusion beyond this f32 clamp/select executable closeout.
* High-level activation, Linalg, Vector, StableHLO, or frontend lowering.
* dtype or LMUL clone batches.
* A second realization family.
* Compatibility wrappers preserving legacy i32m1 authority.
* Broad smoke matrices, dashboards, or report-only work.
* Repeating contraction-dequant evidence as the main achievement.
* Common EmitC/export inference of RVV compute, dtype, bound semantics,
  schedule, policy, or realization semantics.

## Technical Notes

Specs and prior task context read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/index.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-f32-clamp-select-route-foundation/task.json`
* `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-f32-clamp-select-route-foundation/prd.md`
* `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-f32-clamp-select-route-foundation/implement.jsonl`
* `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-f32-clamp-select-route-foundation/check.jsonl`

Initial files to inspect or change:

* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-f32-clamp-select.mlir`
* `test/Dialect/RVV/pre-realized-f32-clamp-select-negative.mlir`
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `include/TianChenRV/Support/RuntimeABI.h`
* `lib/Support/RuntimeABIContract.cpp`
* `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
