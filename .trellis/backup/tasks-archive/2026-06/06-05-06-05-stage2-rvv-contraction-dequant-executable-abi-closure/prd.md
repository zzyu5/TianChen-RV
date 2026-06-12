# Stage2 RVV contraction-dequant executable ABI closure

## Goal

Close the executable support level for the already route-supported composed RVV
chain that starts from a selected `tcrv.exec` RVV variant and typed low-level
`tcrv_rvv` body, loads signed i8 sources, forms i16 widening products, reduces
through a signed i32 boundary, dequantizes with a runtime f32 scale, and stores
f32 output through the generated bundle ABI.

The task exists because commit `b153046e` made this composed chain
route-supported and target-artifact validated, but did not claim executable
correctness on RVV hardware.

## What I Already Know

* The current repository has no active Trellis task for this direction.
* The active branch is `main`; the worktree was clean before task creation.
* The latest relevant commit is `b153046e rvv: compose product reduction
  dequant route`.
* The source of authority is the existing typed composed selected-body route,
  not a route id, q-name, artifact name, ABI string, descriptor, source-front
  door, exact intrinsic spelling, or old i32m1 surface.
* Runtime, correctness, or performance claims require real `ssh rvv` evidence.

## Requirements

* Use the existing composed low-precision contraction-to-f32 selected-body
  artifact as the operation authority.
* Bind external ABI roles for signed i8 `lhs`, signed i8 `rhs`, runtime f32
  `scale`, f32 `out`, and runtime count/AVL.
* Generate or select the composed-chain artifact and run a focused dry-run
  generated-bundle path before remote execution.
* If executable correctness is claimed, compile and run the generated bundle on
  `ssh rvv`.
* Compare device output against a host/reference calculation over multiple
  element counts, signed i8 patterns that exercise nontrivial i16 products and
  i32 accumulation, and at least two nonzero f32 scale values.
* Validate source preservation, output tail sentinel preservation, and explicit
  f32 tolerance.
* If the existing harness lacks this operation, add only minimal neutral
  consumer support. The harness must consume generated bundle ABI/artifacts and
  must not choose RVV semantics from names, route strings, metadata mirrors, or
  test names.
* If evidence exposes a production verifier/provider/target bug, repair the
  production path and rerun focused checks for that surface.

## Acceptance Criteria

* [x] Focused generated-bundle dry run succeeds for the composed chain.
* [x] `ssh rvv` compile/run succeeds before any executable correctness claim.
* [x] Host/reference comparison covers multiple counts, at least two nonzero
  scales, signed i8 source patterns, f32 tolerance, source preservation, and
  output tail sentinel preservation.
* [x] The composed-chain lit fixture passes if touched.
* [x] Harness changes pass Python compile/self-test.
* [x] If provider or target code changes, run the relevant plugin/target tests.
* [x] A bounded scan over touched files finds no new old-authority route
  dependency, q-name authority, descriptor-driven compute, or source-front-door
  authority.
* [x] `git diff --check` and `git diff --cached --check` pass.
* [x] Trellis task status and journal notes are truthful.

## Out Of Scope

* New route coverage.
* Zero-point, clamp, q4/q8/llama benchmark-specific routes, or standalone
  product-reduction/dequant evidence repeats.
* High-level Linalg/frontend work.
* Handwritten C demos as the main deliverable.
* Compatibility wrappers preserving legacy i32m1 authority.
* Broad smoke matrices, dashboards, or report-only closure.
* Inferring dtype, compute, schedule, scale semantics, or support from route ids,
  artifact names, ABI strings, operation-name strings, metadata mirrors, common
  EmitC, or test names.

## Technical Notes

Read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/validation/index.md`
* `.trellis/tasks/archive/2026-06/06-05-stage2-rvv-low-precision-contraction-dequant-chain/`

Likely code/test/evidence surfaces to inspect:

* `test/Target/RVV/explicit-selected-body-artifact-widening-product-reduce-dequantize-f32.mlir`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`

## Definition Of Done

* The module-level executable ABI behavior is implemented or confirmed through
  focused dry-run and RVV hardware evidence.
* Checks listed in acceptance criteria have been run, repaired as needed, and
  recorded.
* The task is finished/archived only if executable closure is truly complete.
* One coherent commit records the work if the task reaches completion.
