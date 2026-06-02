# Stage2 RVV scalar-broadcast macc-add artifact ABI boundary

## Goal

Build and prove one bounded Stage 2 RVV end-to-end artifact/runtime ABI path
for the existing `scalar_broadcast_macc_add` selected body. The path must run
from a selected `tcrv.exec` RVV variant through RVV plugin-owned
realization/validation, scalar-broadcast macc route-family provider facts,
common EmitC materialization, generated RVV target artifact bundle, and focused
`ssh rvv` correctness evidence.

## What I Already Know

* The previous completed computed-mask segment2 update task proved the expected
  pattern for provider-derived ABI/header operand facts, target artifact
  validation, generated-bundle dry-run checks, and real `ssh rvv` evidence.
* This task is a different Stage 2 class: lhs vector load, rhs scalar runtime
  ABI value, scalar splat/broadcast, accumulator load, macc add, output store,
  runtime `n`/AVL, provider route facts, artifact ABI, and runtime correctness.
* The production authority must stay structural: selected typed/realized
  `tcrv_rvv` body/config/capability/runtime facts and RVV plugin-owned
  owners/providers, not route ids, artifact names, manifests, test names,
  C strings, descriptors, exact intrinsic spellings, or mirror metadata.
* Existing fixtures and script hooks for `scalar_broadcast_macc_add` are
  expected to exist; this round should prove what already works and add only
  missing production guards, provider facts, target validation, generated
  bundle harness support, focused tests, or runtime evidence.

## Assumptions

* `scalar_broadcast_macc_add` is already present as a bounded route family or
  fixture path, and this task should not invent broader macc coverage.
* A positive retained i32 example is allowed only as an ordinary instance of the
  corrected typed RVV surface; old `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor, source-front-door, or exact intrinsic
  spelling authority must remain absent or fail-closed.
* Any runtime/correctness claim must be backed by real `ssh rvv` evidence.

## Requirements

* RVV plugin-local selected-body owners must realize or validate the
  `scalar_broadcast_macc_add` selected body before route construction.
* The route provider must carry structural ABI/header binding facts for
  `lhs`, `rhs_scalar`, `acc`, `out`, and `n` through route operand facts,
  generated header/prototype facts, diagnostics, and target artifact
  validation.
* Explicit `tcrv.exec.mem_window` / `tcrv.exec.runtime_param` bindings and
  `tcrv_rvv.runtime_abi_value` exec-binding mirrors must agree for all exported
  ABI roles.
* Lhs load, rhs scalar splat, accumulator load, macc add, output store,
  runtime `n`/AVL, SEW/LMUL/policy, accumulator/result layout, target
  capability mirrors, dispatch/fallback mirrors, `provider_supported_mirror`,
  C type mapping, and required `riscv_vector.h` header facts must survive
  structurally into the generated bundle.
* Fail closed for direct pre-realized shortcuts or stale/missing
  route/provider plan, ABI order, exec binding mirror, type mapping, header
  fact, accumulator layout, capability mirror, dispatch/fallback mirror, or
  provider support mirror relevant to this exact boundary.
* Keep common EmitC/export neutral; RVV scalar-broadcast, macc, dtype/config,
  accumulator, memory form, policy, and ABI semantics must stay in RVV
  plugin-local owners/providers.

## Acceptance Criteria

* [x] Focused evidence shows `scalar_broadcast_macc_add` is realized or
  validated before route construction by RVV plugin-local owners.
* [x] Focused EmitC/export and target-artifact tests prove ABI order and
  `abi`/`hdr` markers for `lhs`, `rhs_scalar`, `acc`, `out`, and `n`.
* [x] Generated bundle evidence proves exec ABI binding mirrors, route operand
  facts, header/prototype facts, scalar splat, macc add, accumulator/result
  layout, runtime `n`/AVL, SEW/LMUL/policy, capability/dispatch/fallback
  mirrors, `provider_supported_mirror`, C type mapping, and required headers
  survive structurally.
* [x] Real `ssh rvv` compile/run correctness covers representative counts
  including `0`, `1`, one VL-boundary count, one tail count, and one larger
  count, with at least two scalar/accumulator/source patterns proving rhs
  scalar broadcast, multiply-accumulate arithmetic, accumulator contribution,
  source preservation, output tail sentinel preservation, and runtime `n`
  behavior.
* [x] Negative/fail-closed evidence covers stale or missing route/provider
  plan, ABI/header facts, ABI order, exec binding mirror, type mapping, header
  fact, accumulator layout, capability/dispatch/fallback mirror, and provider
  support mirror as appropriate for this path.
* [x] Bounded old-authority scan over touched plugin/provider/materializer/
  target/script/test/spec files classifies any `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_`, `i32m` type residue, exact `i32m1` intrinsic authority,
  `source-front-door`, `source-artifact`, `emission_plan`, `descriptor`, or
  selected-route residue.
* [x] Focused build/test/script commands pass, git status is clean after the
  final task commit, and one coherent commit records the task.

## Completion Evidence

* RVV MAcc route-family owner now emits compact scalar-broadcast
  `route_operand_binding_operands` with `abi` and `hdr` for all exported
  operands: `lhs`, `rhs_scalar`, `acc`, `out`, and `n`.
* Math operand-binding facts now require scalar-broadcast MAcc `lhs-load`,
  `splat`, `acc-load`, `macc-lhs`, `macc-rhs`, `macc-acc`, `macc-pass`,
  `store`, runtime `setvl-avl`, `loop`, and all `hdr` markers before route
  construction can proceed.
* Target artifact validation now rejects stale scalar-broadcast MAcc provider
  route operand-binding summaries before accepting the rebuilt provider route.
* Explicit and pre-realized generated-bundle evidence passed on real
  `ssh rvv` for counts `0,1,16,17,257` and RHS scalar values `-37,91`.
  Output included `scalar_broadcast_macc`, `explicit_accumulator`,
  `signed_products`, and `tail_preserved` for every runtime case.
* Bounded old-authority scan over added diff lines had no new hits for
  `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  exact `i32m1` authority, `source-front-door`, `source-artifact`,
  `emission_plan`, `descriptor`, or selected-route residue. Broader touched-file
  hits are existing fail-closed fixtures, historical negative tests, or
  provider-derived RVV intrinsic leaf evidence, not new route authority.

## Definition Of Done

* PRD and task context truthfully describe the implemented module boundary.
* Focused lit/script/C++ checks for the changed path pass after self-repair.
* Runtime evidence is collected on `ssh rvv` when correctness is claimed.
* Trellis task status is finished/archived per project convention.
* One coherent commit is created, or the exact blocker and next continuation
  point are recorded.

## Out Of Scope

* Adding more macc arithmetic kinds, scalar-broadcast add/sub/mul coverage,
  dtype or LMUL clone batches, reductions, matmul, high-level frontend
  lowering, source-front-door positive routes, or a generic performance/tuning
  system.
* Revisiting computed-masked segment2 paths except to reuse metadata-bound and
  artifact-validation lessons.
* Moving RVV semantics into common EmitC/export.
* Treating route ids, artifact names, manifests, test names, C strings,
  descriptors, exact intrinsic spellings, or metadata mirrors as semantic
  authority.

## Technical Notes

* Required specs read before source changes: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Required prior-task context: archived computed-masked segment2 update task
  under `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-segment2-update-unit-load-abi/`.
* Likely module owner files from the brief: `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  RVV selected-body realization and route planning/provider owners,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `scripts/rvv_remote_probe.py`, and focused tests.
