# Stage2 RVV plain macc-add vector-vector artifact ABI boundary

## Goal

Build and prove one bounded Stage 2 RVV end-to-end artifact/runtime ABI path
for the existing plain vector-vector `macc_add` selected body. The path must
run from a selected `tcrv.exec` RVV variant through RVV plugin-owned
realization/validation, plain MAcc route-family provider facts, common EmitC
materialization, generated RVV target artifact bundle, and focused `ssh rvv`
correctness evidence.

## What I Already Know

* The previous scalar-broadcast MAcc task closed the adjacent
  `scalar_broadcast_macc_add` boundary with provider-owned `abi`/`hdr`
  operand-binding facts, target validation, generated-bundle dry-runs, and real
  `ssh rvv` evidence.
* The existing repo already has a plain `macc_add` selected-body path, fixtures,
  generated-bundle script support, and direct pre-realized route-entry
  fail-closed coverage.
* Current plain `macc_add` operand-binding summaries still use older
  `runtime-abi-mirror` / `materialized-*` / `*-call` tokens, unlike the newer
  scalar-broadcast MAcc provider facts that explicitly carry `abi`, `hdr`,
  `lhs-load`, `rhs-load`, `acc-load`, `macc-lhs`, `macc-rhs`, `macc-acc`,
  `macc-pass`, `store`, `setvl-avl`, and `loop`.
* The production authority must stay structural: selected typed/realized
  `tcrv_rvv` body/config/capability/runtime facts and RVV plugin-owned
  owners/providers, not route ids, artifact names, manifests, test names,
  C strings, descriptors, exact intrinsic spellings, or mirror metadata.

## Assumptions

* This task should improve exactly one route family: plain vector-vector
  `macc_add`.
* `scalar_broadcast_macc_add`, computed-masked MAcc, widening MAcc, reductions,
  frontend lowering, dtype/LMUL clone batches, and performance/tuning systems
  are out of scope except where shared validators or scripts need to remain
  consistent.
* Any runtime/correctness claim must be backed by real `ssh rvv` evidence.

## Requirements

* RVV plugin-local selected-body owners must realize or validate the
  pre-realized `macc_add` selected body before route construction.
* The plain MAcc route provider must carry structural ABI/header binding facts
  for `lhs`, `rhs`, `acc`, `out`, and `n` through route operand facts,
  generated header/prototype facts, diagnostics, and target artifact
  validation.
* Explicit vector-vector source structure must survive: lhs vector load, rhs
  vector load, accumulator load, MAcc add, output store, runtime `n`/AVL,
  SEW/LMUL/policy, accumulator/result layout, `provider_supported_mirror`,
  plain MAcc route-family plan, C type mapping, required
  `riscv_vector.h` header facts, and selected dispatch/capability mirrors
  where applicable.
* Target artifact validation must reject stale or missing plain MAcc route
  provider facts, especially old/stale operand-binding summaries, ABI order,
  type mapping, header facts, provider support mirror, route-family plan, and
  accumulator/result layout.
* Generated-bundle evidence must include both explicit selected-body and
  pre-realized selected-body modes and must prove vector-vector multiply,
  accumulator contribution, output tail sentinel preservation, source
  preservation, and runtime `n` behavior.
* Common EmitC/export remains neutral; RVV MAcc semantics, dtype/config,
  accumulator, memory form, policy, ABI roles, and intrinsic/backend mapping
  stay in RVV plugin-local owners/providers.

## Acceptance Criteria

* [x] Focused evidence shows `macc_add` is realized or validated before route
  construction by RVV plugin-local owners.
* [x] Focused EmitC/export and target-artifact tests prove ABI order and
  `abi`/`hdr` markers for `lhs`, `rhs`, `acc`, `out`, and `n`.
* [x] Generated bundle evidence proves route operand facts, header/prototype
  facts, lhs/rhs/acc vector loads, MAcc add, output store,
  accumulator/result layout, runtime `n`/AVL, SEW/LMUL/policy,
  `provider_supported_mirror`, C type mapping, and required headers survive
  structurally.
* [x] Real `ssh rvv` compile/run correctness covers representative counts
  including `0`, `1`, one VL-boundary count, one tail count, and one larger
  count, with at least two lhs/rhs/accumulator patterns proving vector-vector
  multiply, accumulator contribution, source preservation, output tail
  sentinel preservation, and runtime `n` behavior.
* [x] Negative/fail-closed evidence covers direct pre-realized shortcut and
  stale or missing plain MAcc route/provider plan, ABI/header facts, ABI order,
  type mapping, header fact, accumulator layout, and provider support mirror as
  appropriate for this path.
* [x] Bounded old-authority scan over touched plugin/provider/materializer/
  target/script/test/spec files classifies any `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_`, `i32m` type residue, exact `i32m1` intrinsic authority,
  `source-front-door`, `source-artifact`, `emission_plan`, `descriptor`, or
  selected-route residue.
* [x] Focused build/test/script commands pass, git status is clean after the
  final task commit, and one coherent commit records the task.

## Completion Evidence

* Plain vector-vector `macc_add` now uses structural provider route operand
  facts with explicit `abi`, `hdr`, vector-load, MAcc operand, accumulator-pass,
  store, setvl-AVL, and loop markers for `lhs`, `rhs`, `acc`, `out`, and `n`.
* The route planner and target artifact validator reject stale plain MAcc
  summaries before accepting an artifact route. Focused C++ coverage also
  exercises a missing `rhs` header mirror in the route operand-binding facts.
* Generated-bundle summaries now expose `multiply_accumulate_boundary` provider
  route facts for runtime ABI order, operand binding, runtime control, target
  leaf profile, provider support mirror, required headers, C type mapping,
  accumulator layout, and result layout.
* The generated C harness runs two lhs/rhs/accumulator patterns for counts
  `0`, `1`, `16`, `17`, and `257`, verifying vector-vector multiply,
  accumulator contribution, source preservation, output tail sentinel
  preservation, and runtime `n` behavior.
* Direct pre-realized route-entry shortcut remains fail-closed for this path.
* Added-line old-authority scan over touched files found no new `RVVI32M1`,
  `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
  exact `__riscv_*i32m1` authority, `source-front-door`,
  `source-artifact`, `emission_plan`, `descriptor`, or selected-route residue.
  Broader touched-file hits are existing test/script mirrors or negative
  guards, not new authority for this task.
* No `.trellis/spec` update was needed: the provider-owned operand-binding
  and generated-bundle multi-pattern evidence rules already exist in the
  relevant RVV plugin, EmitC route, and testing specs.
* Focused checks passed:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test`;
  `./build/bin/tianchenrv-rvv-extension-plugin-test`; direct `tcrv-opt` /
  `tcrv-translate` / `FileCheck-20` checks for explicit and pre-realized MAcc
  target fixtures; dry-run generated-bundle FileCheck checks for explicit and
  pre-realized modes; manual direct-pre-realized fail-closed script
  reproduction; `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`;
  explicit and pre-realized real `ssh rvv` compile/run correctness for counts
  `0`, `1`, `16`, `17`, and `257` with patterns `0,1`; and
  `git diff --check`.

## Definition Of Done

* PRD and task context truthfully describe the implemented module boundary.
* Focused lit/script/C++ checks for the changed path pass after self-repair.
* Runtime evidence is collected on `ssh rvv` when correctness is claimed.
* Trellis task status is finished/archived per project convention.
* One coherent commit is created, or the exact blocker and next continuation
  point are recorded.

## Out Of Scope

* Computed-masked MAcc, runtime-scalar computed-mask MAcc, widening MAcc, LMUL
  m2 expansion, more arithmetic kinds, reductions, matmul, high-level frontend
  lowering, source-front-door positive routes, dtype or LMUL clone batches, and
  generic performance/tuning systems.
* Revisiting scalar-broadcast MAcc except to reuse its validation pattern.
* Moving RVV semantics into common EmitC/export.
* Treating route ids, artifact names, manifests, test names, C strings,
  descriptors, exact intrinsic spellings, or metadata mirrors as semantic
  authority.

## Technical Notes

* Required specs read before source changes: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior-task context read: archived scalar-broadcast MAcc task under
  `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-scalar-broadcast-macc-add-artifact-abi-boundary/`.
* Likely touched owner files: `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, focused Target/Scripts tests,
  and directly required C++ plugin tests.
