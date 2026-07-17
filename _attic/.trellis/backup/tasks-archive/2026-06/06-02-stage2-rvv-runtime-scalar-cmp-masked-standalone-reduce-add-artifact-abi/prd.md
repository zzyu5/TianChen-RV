# Stage2 RVV runtime-scalar masked standalone reduce-add artifact ABI boundary

## Goal

Close one bounded Stage 2 RVV end-to-end artifact/runtime ABI boundary for
exactly `runtime_scalar_cmp_masked_standalone_reduce_add`. The selected
`tcrv.exec` RVV variant must keep `rhs_scalar` as a typed runtime scalar ABI
value, realize it through `tcrv_rvv.splat` as the compare RHS, carry the
compare-produced mask into `masked_standalone_reduce add`, preserve scalar
seed/result ABI behavior, pass target artifact validation, generate a runnable
bundle harness, and produce real `ssh rvv` correctness evidence.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV runtime-scalar masked standalone reduce-add artifact ABI boundary`

## What I Already Know

* The repository started clean from commit
  `d82f8626 rvv: prove computed mask standalone reduce add abi`.
* There was no active `.trellis/.current-task`, so this task was created from
  the Hermes direction brief before source edits.
* The archived computed-mask standalone reduce-add task closed the vector RHS
  computed-mask boundary, including selected-body realization, provider binding
  facts, target artifact validation, generated-bundle evidence, and real
  `ssh rvv` correctness.
* Current runtime-scalar standalone reduce-add code already has ODS surface,
  explicit and pre-realized fixtures, provider route-family plans, target
  artifact validation hooks, generated-bundle dry-run scaffolding, direct
  pre-realized fail-closed inventory, and a runtime harness skeleton.
* The current route operand binding summary for
  `runtime_scalar_cmp_masked_standalone_reduce_add` exports `acc` in the C
  header/prototype but does not mark the `acc` route operand binding with
  `hdr`. This violates the provider operand binding summary contract for
  exported runtime ABI parameters.
* The current runtime-scalar standalone reduction harness covers counts,
  `rhs_scalar` values, seed values, all-inactive masks, seed preservation, and
  scalar output/tail preservation, but it uses only one compare/source pattern
  and its pass marker does not report `patterns=0,1`.

## Requirements

* Scope exactly one production-positive behavior:
  `runtime_scalar_cmp_masked_standalone_reduce_add` selected body through
  generated target artifact and runtime ABI evidence.
* The pre-realized selected body must be consumed by the RVV plugin-local
  selected-body realization owner before provider route construction. Direct
  pre-realized route-entry support must remain fail-closed.
* Provider-derived ABI roles `cmp_lhs`, `rhs_scalar`, `src`, `acc`, `out`, and
  `n` must survive in route diagnostics, target artifact metadata, generated
  header/prototype, and validation. Every exported header/prototype parameter,
  including scalar seed `acc`, must carry `abi` and `hdr` participation in the
  binding summary.
* `rhs_scalar` must remain a typed runtime scalar ABI value with role
  `rhs-scalar-value`; it must be realized by `tcrv_rvv.splat`, consumed as the
  compare RHS in the same VL scope, and mirrored as provider-derived fact only
  after route construction.
* The route and target artifact validator must preserve and validate compare
  predicate kind, mask role/source/form, runtime-scalar mask producer facts,
  inactive-lane zeroing before reduction, scalar seed, masked standalone
  reduce-add, runtime VL chunk carry, lane-0 scalar output, runtime `n`/AVL,
  SEW/LMUL/policy, `provider_supported_mirror`, C type mapping, required
  headers, route-family facts, and scalar-result runtime boundary.
* Generated-bundle harness behavior must prove active masked lanes contribute,
  all-inactive masks preserve the seed, multi-VL masked accumulation carries
  through `out[0]`, lane-0 scalar output is correct, source/seed inputs are
  preserved, non-scalar output lanes/tail remain sentinel-preserved, runtime
  `n` controls execution, and at least two compare/source patterns execute.
* Real RVV correctness evidence must run counts `0`, `1`, one VL-boundary
  count, one tail count, and one larger count with at least two `rhs_scalar`
  values, two seed values, and two compare/source patterns.

## Acceptance Criteria

* [x] The runtime-scalar computed-mask standalone reduce-add provider binding
      summary marks `cmp_lhs`, `rhs_scalar`, `src`, `acc`, `out`, and `n` with
      exported `abi|...|hdr` participation where the generated prototype
      exports them.
* [x] Provider-side route construction and target artifact validation reject
      stale or missing runtime-scalar computed-mask standalone reduce-add
      binding/header facts, including a stale `acc` binding summary without
      `hdr`.
* [x] Explicit and pre-realized target fixtures check selected-body
      realization/validation, route-family/provider facts, `rhs_scalar` typed
      scalar ABI role, `tcrv_rvv.splat` compare RHS, compare mask producer,
      inactive-lane zeroing, scalar seed/result layout, runtime VL carry,
      required headers, C type mapping, `provider_supported_mirror`, and the
      exact route operand binding summary.
* [x] Generated-bundle dry-run tests check evidence JSON and harness source for
      counts `0,1,<VL-boundary>,<tail>,<larger>`, two `rhs_scalar` values, two
      seed values, two compare/source patterns, all-inactive seed preservation,
      active/inactive lane accounting, source and seed preservation, scalar
      `out[0]`, output sentinel preservation, runtime scalar splat compare RHS,
      and direct pre-realized route-entry fail-closed status.
* [x] Real `ssh rvv` generated-bundle correctness passes for explicit and
      pre-realized `runtime_scalar_cmp_masked_standalone_reduce_add` with
      counts `0,1,<VL-boundary>,<tail>,<larger>`, `rhs_scalar` values
      `-37,91`, seeds `-11,17`, and patterns `0,1`.
* [x] A bounded old-authority scan over touched files finds no new positive
      reliance on legacy `i32m1`, descriptors, source-front-door positive
      routes, direct-C/source-export compute, exact-intrinsic authority, or
      common EmitC RVV semantic inference.
* [x] Focused build/test/script commands pass, the task is finished/archived,
      one coherent commit is created, and `git status --short` is clean.

## Definition Of Done

* PRD, implement/check context, and journal entries are truthful.
* Production/default provider and target validation paths are updated where
  required; the task is not closed by helper-only evidence.
* Runtime correctness is claimed only with real `ssh rvv` evidence.
* No LMUL m2 ownership expansion, i64, min/max, widening-dot reduction, strided
  input, broad reduction framework, frontend lowering, source-front-door
  positive route, tuning database, or common EmitC RVV semantic inference is
  introduced.

## Technical Approach

1. Add exported header participation to the runtime-scalar computed-mask
   standalone reduction accumulator binding summary and keep provider/target
   expected summaries exact.
2. Strengthen focused target validation with a stale
   `runtime_scalar_cmp_masked_standalone_reduce_add` binding/header negative
   case.
3. Update explicit/pre-realized fixtures and generated-bundle dry-run checks to
   match the strengthened provider summary and required runtime scalar
   splat/compare/reduction facts.
4. Strengthen the generated-bundle harness for this route to run two
   compare/source patterns and print `patterns=0,1` in the pass marker.
5. Run focused MLIR/FileCheck, generated-bundle dry-runs, script self-test,
   target artifact C++ test, real `ssh rvv` evidence, authority scan, Trellis
   validation, and `git diff --check`.

## Out Of Scope

* LMUL m2 as the owner, i64, min/max, widening-dot reduction, strided input,
  broad reduction frameworks, frontend lowering, source-front-door positive
  routes, and tuning/performance databases. Existing m2/min/max shared code may
  remain consistent when touched, but it is not this task's completion target.
* Moving RVV semantics into common EmitC/export.
* Treating route ids, artifact names, manifests, test names, C strings,
  descriptors, exact intrinsic spellings, mirror metadata, or runtime harness
  constants as `rhs_scalar`, mask, reduction, scalar seed/output, dtype/config,
  memory form, policy, or route authority.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Prior task read:

* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-mask-standalone-reduce-add-artifact-abi/prd.md`

Likely implementation/test files inspected:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-add.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-add.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-standalone-reduce-add-fail-closed.test`
* `test/Target/TargetArtifactExportTest.cpp`

## Completion Evidence

Focused checks run:

* `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* Direct `tcrv-opt`/`tcrv-translate`/`FileCheck-20` checks for explicit and
  pre-realized add fixtures, plus touched runtime-scalar min/max shared
  binding-summary fixtures.
* Generated-bundle dry-run and `FileCheck-20` evidence/harness checks for
  pre-realized add+m2 and add+i64 consistency.
* Direct pre-realized route-entry fail-closed generated-bundle check for add,
  i64, and m2.
* `ssh rvv` generated-bundle correctness for explicit
  `runtime_scalar_cmp_masked_standalone_reduce_add` at counts `0,1,16,23,257`,
  rhs scalars `-37,91`, seeds `-11,17`, and patterns `0,1`.
* `ssh rvv` generated-bundle correctness for pre-realized
  `runtime_scalar_cmp_masked_standalone_reduce_add` at counts `0,1,16,23,257`,
  rhs scalars `-37,91`, seeds `-11,17`, and patterns `0,1`.
* Bounded old-authority scan over touched code/test added lines.
* `git diff --check`
* `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-02-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-artifact-abi`
* `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/archive/2026-06/06-02-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-artifact-abi`

Runtime evidence paths:

* `artifacts/tmp/06-02-runtime-scalar-cmp-masked-standalone-reduce-add-ssh-rvv/explicit/runtime_scalar_cmp_masked_standalone_reduce_add/evidence.json`
* `artifacts/tmp/06-02-runtime-scalar-cmp-masked-standalone-reduce-add-ssh-rvv/pre-realized/runtime_scalar_cmp_masked_standalone_reduce_add/evidence.json`

## Current Phase

Finish.
