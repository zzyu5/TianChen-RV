# Stage2 RVV dequant-clamp epilogue executable ABI closure

## Goal

Close executable ABI evidence for the bounded Stage 2 RVV dequant-clamp
epilogue route added in `701e4070`:

```text
selected tcrv.exec RVV variant
  -> typed pre-realized dequant-clamp tcrv_rvv body
  -> RVV plugin-local realization
  -> i32 load
  -> runtime f32 scale dequantization
  -> runtime f32 lower/upper clamp-select
  -> f32 store
  -> provider-derived route and common EmitC materialization
  -> generated header/object bundle
  -> external ABI harness
  -> ssh rvv correctness evidence
```

This is an executable-closure task for one already route-supported production
path. It must not become a new RVV route-coverage task or a repeated
harness-only loop.

## What I Already Know

* The session started in `/home/kingdom/phdworks/TianchenRV` on `main` with a
  clean worktree at `701e4070 rvv: add dequant clamp epilogue route foundation`.
* No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief and started as the current Trellis task.
* `.trellis/spec/index.md` keeps the executable authority chain at selected
  `tcrv.exec` envelope -> typed low-level `tcrv_rvv` body -> RVV plugin-owned
  realization/provider -> `TCRVEmitCLowerableRoute` -> common EmitC -> target
  artifact -> `ssh rvv` evidence for runtime/correctness claims.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires selected-body
  realization to consume pre-realized bodies before provider route facts are
  collected, and forbids route ids, artifact names, ABI strings, diagnostics,
  scripts, descriptors, or legacy i32 helper names as realization or route
  authority.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires Common EmitC/export
  to remain neutral and carry provider-built route payloads without choosing
  RVV compute, dtype, scale, bound, schedule, policy, intrinsic, or ABI
  semantics.
* `.trellis/spec/testing/mlir-testing-contract.md` requires real `ssh rvv`
  output for RVV runtime/correctness claims and requires active-lane reference
  checks plus tail/sentinel preservation for generated-bundle evidence.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-dequant-clamp-epilogue-composition-foundation/`
  completed the route-supported production foundation and explicitly did not
  claim executable correctness.
* The relevant existing fixture is
  `test/Target/RVV/pre-realized-selected-body-artifact-dequant-clamp-f32-epilogue.mlir`.
* The likely evidence harness entrypoint is
  `scripts/rvv_generated_bundle_abi_e2e.py`, which already has executable
  closure patterns for nearby RVV generated-bundle routes.

## Requirements

* Use the existing typed selected/pre-realized dequant-clamp route as the only
  compiler authority.
* Select or generate the pre-realized dequant-clamp fixture and prove
  realization occurs before emission.
* Bind the external ABI roles for:
  * i32 input or accumulator source;
  * runtime f32 scale;
  * runtime f32 lower bound;
  * runtime f32 upper bound;
  * f32 output;
  * runtime `n` / AVL.
* Run a generated-bundle dry run before any remote execution.
* If executable correctness is claimed, compile and run the generated artifact
  on `ssh rvv`.
* Compare remote output against a host/reference calculation over:
  * multiple runtime counts;
  * signed i32 source patterns covering below-bound, in-bound, and above-bound
    scaled values;
  * at least two nonzero scale values;
  * at least two ordered lower/upper bound pairs;
  * explicit f32 tolerance;
  * source preservation;
  * output tail sentinel preservation.
* If the existing harness lacks this op, add only minimal neutral harness
  support. Harness support may assemble inputs and reference checks for this
  generated function, but it must not authorize compiler semantics from route
  ids, artifact names, ABI strings, metadata mirrors, op-kind strings, q-names,
  or legacy helper names.
* If runtime evidence exposes a real production verifier/provider/target bug,
  repair the production path and run the focused compiler checks required by
  that change. Otherwise keep code changes to neutral harness/evidence support.

## Acceptance Criteria

* [ ] The task PRD and context files truthfully describe executable closure for
      one dequant-clamp epilogue route and its non-goals.
* [ ] Dry-run generated-bundle evidence proves the source fixture is the
      selected pre-realized dequant-clamp path and that it is realized before
      emission.
* [ ] The generated external ABI exposes the i32 source, runtime f32 scale,
      runtime f32 lower/upper bounds, f32 output, and runtime count/AVL roles.
* [ ] The harness compiles the generated header/object bundle and calls the
      generated function through that external ABI.
* [ ] The runtime harness checks multiple counts, signed i32 patterns,
      two nonzero scale values, two ordered bound pairs, explicit f32 tolerance,
      source preservation, and output tail sentinel preservation.
* [ ] `ssh rvv` compile/run output is captured if executable correctness is
      claimed; otherwise the report states exactly why executable evidence is
      blocked.
* [ ] If only harness/evidence code changes, run `python3 -m py_compile` and
      the focused dry-run/self-test path for `scripts/rvv_generated_bundle_abi_e2e.py`.
* [ ] If production C++/MLIR/TableGen/target code changes, run the touched lit
      check plus `tianchenrv-rvv-extension-plugin-test` and
      `tianchenrv-target-artifact-export-test`.
* [ ] Run the lit check for the dequant-clamp fixture if touched.
* [ ] Run a bounded old-authority and q-name-authority scan over touched files.
* [ ] `git diff --check` and `git diff --cached --check` pass.
* [ ] Trellis metadata is truthful, the task is finished/archived when
      complete, one coherent commit is created, and final worktree status is
      clean.

## Definition Of Done

The generated dequant-clamp epilogue artifact is exercised end to end through
the external ABI on real RVV hardware, with host/reference correctness,
source-preservation, tail-preservation, and tolerance evidence. Any code change
made to reach that evidence remains within the existing typed-body ->
realization -> provider route -> common EmitC -> artifact authority chain.

## Out Of Scope

* New RVV route coverage.
* q8/q4/llama benchmark routes.
* Zero-point or contraction fusion.
* Standalone dequant evidence repeat as the primary deliverable.
* Standalone clamp/select evidence repeat as the primary deliverable.
* High-level activation, Linalg, Vector, StableHLO, or frontend work.
* dtype/LMUL clone batches.
* A second realization family.
* Compatibility wrappers preserving legacy i32m1 authority.
* Broad smoke matrices, dashboards, report-only work, or readiness state
  machines.
* Common EmitC/export choosing RVV compute, dtype, scale, bound semantics,
  schedule, policy, or realization semantics.

## Technical Notes

Specs and prior context read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/variant-pipeline/index.md`
* `.trellis/spec/guides/capability-first-design-guide.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/spec/guides/compute-boundary-review-guide.md`
* `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-dequant-clamp-epilogue-composition-foundation/prd.md`

Initial source and evidence surfaces to inspect:

* `test/Target/RVV/pre-realized-selected-body-artifact-dequant-clamp-f32-epilogue.mlir`
* `test/Dialect/RVV/pre-realized-dequant-clamp-f32-epilogue-negative.mlir`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Dialect/RVV/IR/RVVDialect.cpp`
* `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
