# Stage2 RVV runtime-scalar masked standalone reduce-min artifact ABI boundary

## Goal

Close one bounded Stage 2 RVV end-to-end artifact/runtime ABI boundary for
exactly `runtime_scalar_cmp_masked_standalone_reduce_min`. The selected
`tcrv.exec` RVV variant must keep `rhs_scalar` as a typed runtime scalar ABI
value, realize it through `tcrv_rvv.splat` as the compare RHS, carry the
compare-produced mask into `masked_standalone_reduce min`, preserve scalar
seed/result ABI behavior, use min-specific provider facts rather than add-path
inheritance, pass target artifact validation, generate a runnable bundle
harness, and produce real `ssh rvv` correctness evidence.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV runtime-scalar masked standalone reduce-min op-kind artifact ABI boundary`

## What I Already Know

* The repository started clean from commit
  `bb50a38f rvv: prove runtime scalar masked standalone reduce add abi`.
* There was no active `.trellis/.current-task`, so this task was created from
  the Hermes direction brief before source edits.
* The archived runtime-scalar computed-mask standalone reduce-add task closed
  the analogous add boundary with selected-body realization, provider binding
  facts, target artifact validation, generated-bundle evidence, and real
  `ssh rvv` correctness.
* Current min fixtures already expose a pre-realized
  `typed_runtime_scalar_computed_mask_standalone_reduce_pre_realized_body`
  selected body and check materialization into `setvl/with_vl/load/splat/
  compare/load/masked_standalone_reduce/store`.
* The current script contains expectation entries for
  `runtime_scalar_cmp_masked_standalone_reduce_min`, but the task brief says
  there is no focused generated-bundle/ssh owner proving this non-add op-kind.
* Specs require operation kind, dtype/config, runtime scalar RHS, mask
  producer, inactive-lane handling, scalar seed/result, ABI order, headers, and
  intrinsic/type facts to be derived from typed `tcrv_rvv` body/config/runtime
  facts by RVV plugin owners. Route ids, artifact names, manifests, C strings,
  mirror metadata, descriptors, and exact intrinsic spellings are not
  authority.

## Requirements

* Scope exactly one production-positive behavior:
  `runtime_scalar_cmp_masked_standalone_reduce_min` selected body through
  generated target artifact and runtime ABI evidence.
* The pre-realized selected body must be consumed by the RVV plugin-local
  selected-body realization owner or validated by RVV plugin-local owners
  before provider route construction. Direct pre-realized route-entry support
  must remain fail-closed.
* Provider-derived ABI roles `cmp_lhs`, `rhs_scalar`, `src`, `acc`, `out`, and
  `n` must survive in route diagnostics, target artifact metadata, generated
  header/prototype, and validation. Every exported header/prototype parameter
  must carry `abi` and `hdr` participation in the binding summary.
* `rhs_scalar` must remain a typed runtime scalar ABI value with role
  `rhs-scalar-value`; it must be realized by `tcrv_rvv.splat`, consumed as the
  compare RHS in the same VL scope, and mirrored as provider-derived fact only
  after route construction.
* The route and target artifact validator must preserve and validate compare
  predicate kind, mask role/source/form, runtime-scalar mask producer facts,
  neutral inactive-lane handling for min, scalar seed, masked standalone
  reduce-min op-kind, runtime VL chunk carry, lane-0 scalar output, runtime
  `n`/AVL, SEW/LMUL/policy, `provider_supported_mirror`, C type mapping,
  required headers, route-family facts, and scalar-result runtime boundary.
* Generated-bundle harness behavior must prove active masked lanes contribute
  to the min, all-inactive masks preserve the seed, multi-VL masked min carries
  through `out[0]`, lane-0 scalar output is correct, source/seed inputs are
  preserved, non-scalar output lanes/tail remain sentinel-preserved, runtime
  `n` controls execution, and at least two compare/source patterns execute.
* Real RVV correctness evidence must run counts `0`, `1`, one VL-boundary
  count, one tail count, and one larger count with at least two `rhs_scalar`
  values, two seed values, and two compare/source patterns.

## Acceptance Criteria

* [x] The runtime-scalar computed-mask standalone reduce-min provider binding
      summary marks `cmp_lhs`, `rhs_scalar`, `src`, `acc`, `out`, and `n` with
      exported `abi|...|hdr` participation where the generated prototype
      exports them.
* [x] Provider-side route construction and target artifact validation reject
      stale or missing runtime-scalar computed-mask standalone reduce-min
      route-family facts, including stale add op-kind, stale binding plan,
      stale binding operands, stale inactive-lane contract, stale C type/header
      facts, or stale provider mirror.
* [x] Pre-realized target fixture checks selected-body realization/validation,
      route-family/provider facts, `rhs_scalar` typed scalar ABI role,
      `tcrv_rvv.splat` compare RHS, compare mask producer, neutral inactive
      lanes for min, scalar seed/result layout, runtime VL carry, required
      headers, C type mapping, `provider_supported_mirror`, and the exact route
      operand binding summary.
* [x] Generated-bundle dry-run tests check evidence JSON and harness source for
      counts `0,1,<VL-boundary>,<tail>,<larger>`, two `rhs_scalar` values, two
      seed values, two compare/source patterns, all-inactive seed preservation,
      active/inactive lane accounting, source and seed preservation, scalar
      `out[0]`, output sentinel preservation, runtime scalar splat compare RHS,
      min-specific intrinsic/type/header facts, and direct pre-realized
      route-entry fail-closed status.
* [x] Real `ssh rvv` generated-bundle correctness passes for pre-realized
      `runtime_scalar_cmp_masked_standalone_reduce_min` with counts
      `0,1,<VL-boundary>,<tail>,<larger>`, at least two `rhs_scalar` values,
      two seeds, and two compare/source patterns.
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
* No max acceptance owner, min/max batch, LMUL m2 owner, i64, new dtype, broad
  reduction framework, frontend lowering, source-front-door positive route,
  tuning database, or common EmitC RVV semantic inference is introduced.

## Technical Approach

1. Run the current pre-realized min generated-bundle dry-run to identify the
   live failure surface instead of assuming the add path already generalized.
2. Strengthen provider/target validation so min-specific route facts are
   required before bundle acceptance and stale add/missing min facts fail
   closed.
3. Update focused pre-realized min fixture checks and generated-bundle dry-run
   checks for exact min route facts, neutral inactive-lane handling, ABI order,
   rhs scalar splat/compare RHS, scalar seed/result, and provider mirrors.
4. If shared min/max helpers are touched, keep max changes mechanical and do
   not make max an acceptance owner.
5. Run focused MLIR/FileCheck, generated-bundle dry-run, script self-test,
   target artifact C++ test, real `ssh rvv` evidence, bounded old-authority
   scan, Trellis validation, and `git diff --check`.

## Out Of Scope

* Max as a second acceptance owner, min/max batch ownership, LMUL m2 ownership,
  i64, new dtypes, broad reduction frameworks, frontend lowering,
  source-front-door positive routes, tuning/performance databases, and another
  add cleanup. Existing shared min/max helpers may stay consistent when
  touched, but max is not this task's completion target.
* Moving RVV semantics into common EmitC/export.
* Treating route ids, artifact names, manifests, test names, C strings,
  descriptors, exact intrinsic spellings, mirror metadata, or runtime harness
  constants as `rhs_scalar`, mask, reduction, scalar seed/output,
  dtype/config, memory form, policy, or route authority.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Prior task read:

* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-add-artifact-abi/prd.md`

Likely implementation/test files inspected:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-standalone-reduce-min.mlir`
* `test/Target/TargetArtifactExportTest.cpp`
* `test/Scripts/*runtime-scalar-cmp-masked-standalone-reduce*`

## Completion Evidence

Focused checks run:

* `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* `./build/bin/tianchenrv-target-artifact-export-test`
* Direct `tcrv-opt`/`tcrv-translate`/`FileCheck-20` checks for the
  pre-realized runtime-scalar computed-mask standalone reduce-min fixture:
  `REALIZED`, `PLAN`, and `HEADER`.
* Generated-bundle dry-run for
  `runtime_scalar_cmp_masked_standalone_reduce_min` with counts
  `0,1,16,23,257`, rhs scalars `-37,91`, seeds `-11,17`, and patterns `0,1`.
* Manual `FileCheck-20` equivalents for the new dry-run test's `STDOUT`,
  `ROOT`, `MIN`, and `HARNESS` prefixes.
* Direct pre-realized route-entry fail-closed check for
  `runtime_scalar_cmp_masked_standalone_reduce_min`.
* Real `ssh rvv` generated-bundle correctness for pre-realized
  `runtime_scalar_cmp_masked_standalone_reduce_min` at counts
  `0,1,16,23,257`, rhs scalars `-37,91`, seeds `-11,17`, and patterns `0,1`.
* Bounded old-authority scan over touched files. The only exact intrinsic
  matches in new tests are provider-derived generated output assertions such
  as `__riscv_vredmin_vs_i32m1_i32m1`; they are not route inputs or common
  EmitC authority.
* `git diff --check`
* `python3 ./.trellis/scripts/task.py validate
  .trellis/tasks/06-02-06-02-stage2-rvv-runtime-scalar-cmp-masked-standalone-reduce-min-artifact-abi`

Runtime evidence path:

* `artifacts/tmp/06-02-runtime-scalar-cmp-masked-standalone-reduce-min-ssh-rvv/pre-realized/runtime_scalar_cmp_masked_standalone_reduce_min/evidence.json`

## Current Phase

Finish.
