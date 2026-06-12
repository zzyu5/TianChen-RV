# Stage2 RVV computed-masked macc-add artifact ABI boundary

## Goal

Build and prove one bounded Stage 2 RVV end-to-end artifact/runtime ABI path
for the existing vector-compare `computed_masked_macc_add` selected body. The
path must run from a selected `tcrv.exec` RVV variant through RVV
plugin-owned realization/validation, computed-mask MAcc provider facts, common
EmitC materialization, generated RVV target artifact bundle, and focused real
`ssh rvv` correctness evidence.

## What I Already Know

* The previous plain vector-vector `macc_add` task closed the adjacent MAcc ABI
  boundary with compact `abi`/`hdr` operand-binding facts, provider route
  facts, target validation, generated-bundle dry-runs, and `ssh rvv` evidence.
* A prior computed-mask MAcc realization-boundary task already proved
  plugin-local selected-body realization and owner/provider entrypoints for
  vector and runtime-scalar computed-mask MAcc.
* Current explicit and pre-realized computed-mask MAcc fixtures already carry
  selected-body structure for compare lhs/rhs, payload lhs/rhs, accumulator,
  output, runtime `n`, compare-produced mask, `masked_macc`, inactive-lane
  accumulator preservation, provider support mirror, header/type facts, and
  target artifact mirror checks.
* The remaining risk is the artifact ABI boundary: target artifact provider
  validation currently checks the computed-mask MAcc operand-binding plan id,
  but not the complete provider-derived operand-binding summary for
  `cmp_lhs`, `cmp_rhs`, `lhs`, `rhs`, `acc`, `out`, and `n`.
* Existing generated-bundle dry-run tests for this route use counts
  `0,7,16,23`; the current brief requires representative counts including
  `0`, `1`, a VL-boundary count, a tail count, and a larger count, plus at
  least two mask/payload/accumulator patterns.

## Requirements

* RVV plugin-local selected-body owners must realize or validate the
  pre-realized `computed_masked_macc_add` selected body before route
  construction.
* Provider-derived ABI/header binding facts for `cmp_lhs`, `cmp_rhs`, `lhs`,
  `rhs`, `acc`, `out`, and `n` must survive through route diagnostics,
  generated header/prototype, generated bundle evidence, and target artifact
  validation.
* Target artifact provider validation must fail closed when the
  `computed_masked_macc_add` operand-binding summary is missing, stale, or
  lacks required `abi`/`hdr` participation tokens, even if the plan id and
  candidate metadata mirror are present.
* Generated-bundle evidence must prove compare-produced mask source, predicate
  kind, payload loads, accumulator load/pass-through, active MAcc add,
  inactive-lane accumulator preservation, output store, runtime `n`/AVL,
  SEW/LMUL/policy, accumulator/result layout, accumulation route-family plan,
  provider support mirror, C type mapping, and required headers.
* Runtime evidence must run the explicit and pre-realized selected-body
  generated bundles on real `ssh rvv` for counts `0`, `1`, `16`, `17`, and
  `257`, with two data/mask patterns proving active masked MAcc behavior,
  inactive accumulator preservation, output tail sentinel preservation, source
  preservation, and runtime `n` behavior.
* Common EmitC/export remains neutral; RVV semantics, dtype/config, mask
  source, accumulator contract, memory form, policy, ABI roles, and intrinsic
  mapping stay in RVV plugin-local owners/providers.

## Acceptance Criteria

* [x] Focused evidence shows `computed_masked_macc_add` is realized or
  validated before route construction by RVV plugin-local owners.
* [x] Focused target artifact validation rejects a stale computed-mask MAcc
  provider operand-binding summary before artifact acceptance.
* [x] Explicit and pre-realized Target/RVV fixtures prove ABI order,
  `abi`/`hdr` operand facts, header/prototype facts, mask/passthrough facts,
  accumulation plan facts, type mapping, provider support mirror, and stale
  mirror fail-closed coverage.
* [x] Generated-bundle dry-run tests prove the same structural facts and
  harness source for counts `0`, `1`, `16`, `17`, and `257` with two
  computed-mask MAcc data patterns.
* [x] Real `ssh rvv` compile/run correctness passes for explicit and
  pre-realized selected-body generated bundles over counts `0`, `1`, `16`,
  `17`, and `257` with both patterns.
* [x] Direct pre-realized route-entry shortcut remains fail-closed for this
  path.
* [x] Bounded old-authority scan over touched plugin/provider/materializer/
  target/script/test/spec files classifies any `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_`, `i32m` type residue, exact `i32m1` intrinsic authority,
  `source-front-door`, `source-artifact`, `emission_plan`, `descriptor`, or
  selected-route residue.
* [x] Focused build/test/script commands pass, git status is clean after the
  final task commit, and one coherent commit records the task.

## Completion Evidence

* Target artifact provider validation now exact-matches the complete
  computed-mask MAcc route operand-binding summary for
  `cmp_lhs`, `cmp_rhs`, `lhs`, `rhs`, `acc`, `out`, and `n`, including
  required `abi` and `hdr` participation tokens. The same guard is present for
  the existing runtime-scalar computed-mask MAcc route-family member without
  expanding that route's scope.
* `TargetArtifactExportTest` now rejects a stale computed-mask MAcc binding
  summary that omits the `cmp_rhs` header/prototype marker before target
  artifact acceptance.
* Generated-bundle harness generation now runs two computed-mask MAcc data
  patterns for each runtime count, checks source preservation, inactive
  accumulator preservation, active `acc + lhs * rhs`, add-only/mul-only
  distinguishing cases, output tail sentinel preservation, and prints
  `patterns=0,1`.
* Explicit and pre-realized generated-bundle dry-runs now require counts
  `0`, `1`, `16`, `17`, and `257`, exact route operand-binding facts,
  runtime ABI order, header/type facts, provider support mirror, and
  two-pattern harness source.
* Direct pre-realized route-entry shortcut remains fail-closed for
  `computed_masked_macc_add`.
* Real `ssh rvv` evidence passed for explicit and pre-realized selected-body
  generated bundles over counts `0`, `1`, `16`, `17`, and `257`, with patterns
  `0,1`. Both runs reported active/inactive mask lanes, inactive accumulator
  preservation, add-only/mul-only distinguishing, source preservation, and tail
  preservation.
* Added-line authority scan over touched target/script/test/task files found
  no new `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`, exact
  `__riscv_*i32m1` authority, `source-front-door`, `source-artifact`,
  `emission_plan`, `descriptor`, or selected-route residue. Broader touched
  file hits are existing negative guards, other route evidence constants, or
  PRD checklist text.
* No `.trellis/spec` update was needed: the computed-mask MAcc evidence
  boundary and route operand-binding contracts already exist in
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
* Focused checks passed:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test`;
  `./build/bin/tianchenrv-target-artifact-export-test`;
  `./build/bin/tianchenrv-rvv-extension-plugin-test`;
  direct `tcrv-opt` / `tcrv-translate` / `FileCheck-20` checks for explicit
  and pre-realized computed-mask MAcc target fixtures;
  manual stale route/provider/binding/ABI/header/type/accumulation/layout
  fail-closed FileCheck reproduction;
  explicit and pre-realized generated-bundle dry-run FileCheck reproduction;
  direct pre-realized fail-closed script reproduction;
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`;
  explicit and pre-realized real `ssh rvv` compile/run correctness; and
  `git diff --check`.

## Definition Of Done

* PRD and task context truthfully describe the implemented module boundary.
* Focused C++/lit/script checks for the changed path pass after self-repair.
* Runtime evidence is collected on `ssh rvv` before any correctness claim.
* Trellis task status is completed/archived per project convention.
* One coherent commit is created, or the exact blocker and next continuation
  point are recorded.

## Out Of Scope

* Runtime-scalar computed-mask MAcc expansion, widening MAcc, LMUL m2
  expansion, additional arithmetic kinds, reductions, matmul, high-level
  frontend lowering, source-front-door positive routes, dtype/LMUL clone
  batches, and generic performance/tuning systems.
* Revisiting plain or scalar-broadcast MAcc except to reuse their validation
  and evidence patterns.
* Moving RVV semantics into common EmitC/export.
* Treating route ids, artifact names, manifests, test names, C strings,
  descriptors, exact intrinsic spellings, or metadata mirrors as semantic
  authority.

## Technical Approach

Tighten the target artifact consumer first: add exact expected binding summary
validation for computed-mask MAcc provider payloads, matching the
provider-owned `cmp_lhs/cmp_rhs/lhs/rhs/acc/out/n` summary already emitted by
the route. Then add a focused C++ stale-summary negative test, update
generated-bundle dry-run tests and self-test expectations to require
`0,1,16,17,257` plus two pattern loops, and run explicit plus pre-realized real
`ssh rvv` evidence.

## Technical Notes

* Required specs read before source changes: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior-task context read: archived plain MAcc task under
  `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-plain-macc-add-vector-vector-artifact-abi-boundary/`.
* Relevant journal context read: Session 382 plain MAcc ABI boundary and
  Session 368 computed-mask MAcc realization boundary in
  `.trellis/workspace/codex/journal-20.md`.
* Likely touched files: `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, focused Target/RVV and Scripts
  tests, task context files, and workspace journal.
