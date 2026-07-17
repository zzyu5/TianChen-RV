# Stage2 RVV computed-mask standalone reduce-add artifact ABI boundary

## Goal

Close one bounded Stage 2 RVV end-to-end artifact/runtime ABI boundary for
exactly `computed_mask_standalone_reduce_add`. The selected `tcrv.exec` RVV
variant must flow through RVV plugin-local realization/validation, provider
route facts, common EmitC materialization, target artifact validation, generated
bundle harness behavior, and real `ssh rvv` correctness evidence.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-mask standalone reduce-add artifact ABI boundary`

## What I Already Know

* The repository started clean from commit
  `abfaffa0 rvv: prove standalone reduce add scalar abi`.
* There was no active `.trellis/.current-task`, so this task was created from
  the Hermes direction brief before source edits.
* The previous archived standalone reduce-add task closed the unmasked scalar
  seed/result ABI boundary for `lhs,acc,out,n`, including compact `abi|hdr`
  binding facts, target validation, generated bundle harness evidence, and
  real `ssh rvv` correctness.
* Current computed-mask standalone reduce-add fixtures already have explicit
  and pre-realized selected-body skeletons, provider/target metadata mirrors,
  direct pre-realized fail-closed script coverage, and generated-bundle dry-run
  scaffolding.
* Current computed-mask standalone reduction binding summaries export the
  `acc` parameter in the generated prototype but do not mark the `acc` route
  operand binding with `hdr`. That weakens the provider binding/header
  contract this task is meant to prove.
* Current generated-bundle dry-run coverage uses counts `0,7,16,23`, which is
  weaker than the required runtime boundary set `0,1,16,17,257`.

## Requirements

* Scope exactly one production-positive behavior:
  `computed_mask_standalone_reduce_add` selected body through generated target
  artifact and runtime ABI evidence.
* The selected pre-realized body must be consumed by the RVV selected-body
  realization owner before route construction. Direct pre-realized route-entry
  support must remain fail-closed.
* Provider-derived ABI roles `cmp_lhs`, `cmp_rhs`, `src`, `acc`, `out`, and
  `n` must survive in route diagnostics, target artifact metadata, generated
  header/prototype, and validation. Every exported header/prototype parameter
  must carry `abi` and `hdr` participation in the binding summary.
* The route and target artifact validator must preserve and validate:
  compare-produced mask facts, predicate kind, mask role/source/form,
  inactive-lane zeroing before add reduction, scalar seed, masked
  `standalone_reduce add`, runtime VL chunk carry, lane-0 scalar result,
  runtime `n`/AVL, SEW/LMUL/policy, provider-supported mirror, C type mapping,
  required `riscv_vector.h` header, route-family facts, and scalar-result
  runtime boundary.
* Generated-bundle harness behavior must prove active masked lanes contribute,
  all-inactive masks preserve the seed, multi-VL masked accumulation carries
  through `out[0]`, lane-0 scalar output is correct, source/seed inputs are
  preserved, non-scalar output lanes/tail remain sentinel-preserved, and
  runtime `n` controls execution.
* Real RVV correctness evidence must run counts `0`, `1`, `16`, `17`, and
  `257` with at least two seed values and two compare/source patterns.

## Acceptance Criteria

* [x] The computed-mask standalone reduce-add provider binding summary marks
      `cmp_lhs`, `cmp_rhs`, `src`, `acc`, `out`, and `n` with exported
      `abi|...|hdr` participation where the generated prototype exports them.
* [x] Provider-side route construction and target artifact validation reject
      stale or missing computed-mask standalone reduce-add binding/header
      facts, including a stale `acc` binding summary.
* [x] Explicit and pre-realized target fixtures check selected-body
      realization/validation, route-family/provider facts, compare mask
      producer/source, inactive-lane zeroing, scalar seed/result layout,
      runtime VL carry, required headers, C type mapping,
      `provider_supported_mirror`, and the exact route operand binding summary.
* [x] Generated-bundle dry-run tests check evidence JSON and harness source for
      counts `0,1,16,17,257`, two seed values, two compare/source patterns,
      all-inactive seed preservation, active/inactive lane accounting, source
      and seed preservation, scalar `out[0]`, output sentinel preservation, and
      direct pre-realized route-entry fail-closed status.
* [x] Real `ssh rvv` generated-bundle correctness passes for explicit and
      pre-realized `computed_mask_standalone_reduce_add` with counts
      `0,1,16,17,257`, seeds `-11,17`, and two compare/source patterns.
* [x] A bounded old-authority scan over touched files finds no new positive
      reliance on legacy `i32m1`, descriptors, source-front-door positive
      routes, direct-C/source-export compute, exact-intrinsic authority, or
      common EmitC RVV semantic inference.
* [x] Focused build/test/script commands pass, the task is finished/archived,
      one coherent commit is created, and `git status --short` is clean.

## Definition Of Done

* PRD, implement/check context, and journal entries are truthful.
* Production/default provider and validator paths are updated where required;
  the task is not closed by helper-only evidence.
* Runtime correctness is claimed only with real `ssh rvv` evidence.
* No runtime-scalar masked reduction, min/max expansion as a goal, widening dot
  reduction, strided input, LMUL m2 proof expansion, new dtype, frontend
  lowering, source-front-door positive route, broad reduction framework, or
  tuning database work is introduced.

## Technical Approach

1. Add missing exported header participation to the computed-mask standalone
   reduction accumulator binding summary and keep provider/target expected
   summaries exact.
2. Strengthen focused target validation with a stale `computed_mask_standalone_reduce_add`
   binding/header negative case.
3. Update explicit/pre-realized fixtures and generated-bundle dry-run checks to
   match the strengthened provider summary and required runtime count/pattern
   surface.
4. Strengthen the generated-bundle harness for this route to run two
   compare/source patterns across counts `0,1,16,17,257` and to report pattern
   coverage in the pass marker.
5. Run focused MLIR/FileCheck, generated-bundle dry-runs, script self-test,
   target artifact C++ test, real `ssh rvv` evidence, authority scan,
   Trellis validation, and `git diff --check`.

## Out Of Scope

* Runtime-scalar computed-mask standalone reduction, min/max as a new proof
  target, widening-dot reduction, strided input, LMUL m2 proof expansion, new
  dtypes, broad reduction frameworks, frontend lowering, source-front-door
  positive routes, and tuning/performance databases.
* Moving RVV semantics into common EmitC/export.
* Treating route ids, artifact names, manifests, test names, C strings,
  descriptors, exact intrinsic spellings, mirror metadata, or runtime harness
  constants as mask, reduction, scalar seed/output, dtype/config, memory form,
  policy, or route authority.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Prior task read:

* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-standalone-reduce-add-scalar-artifact-abi/prd.md`

Likely implementation/test files inspected:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/RVV/explicit-selected-body-artifact-computed-mask-standalone-reduce-add.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-mask-standalone-reduce-add.mlir`
* `test/Scripts/rvv-generated-bundle-abi-e2e-computed-mask-standalone-reduce-add-dry-run.test`
* `test/Target/TargetArtifactExportTest.cpp`

## Completion Evidence

Implemented behavior:

* Vector computed-mask standalone reduction route operand binding now marks the
  exported `acc` scalar seed parameter with `hdr`, matching the generated
  `cmp_lhs,cmp_rhs,src,acc,out,n` prototype/header boundary.
* Provider-side math operand binding facts now reject a vector computed-mask
  standalone reduction route that omits the `acc` header binding before route
  construction.
* Target artifact validation expects the strengthened exact provider summary,
  and `TargetArtifactExportTest` rejects a stale
  `computed_mask_standalone_reduce_add` binding summary that omits the `acc`
  header marker.
* Generated-bundle harness generation for vector computed-mask standalone
  reduction now runs two compare/source patterns, checks source preservation,
  keeps the two scalar seed values, preserves the all-inactive seed oracle, and
  prints `patterns=0,1` in the pass marker.
* Explicit/pre-realized add fixtures and shared vector computed-mask
  standalone min/max fixtures were updated to the new exact binding summary.

Focused checks run:

* `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
* `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `rtk ./build/bin/tianchenrv-target-artifact-export-test`
* Direct `tcrv-opt` / `tcrv-translate` / `FileCheck-20` checks for explicit
  and pre-realized `computed_mask_standalone_reduce_add`.
* Direct `tcrv-opt` / `tcrv-translate` / `FileCheck-20` checks for changed
  vector computed-mask standalone min/max fixtures.
* Generated-bundle dry-run and FileCheck for explicit/pre-realized
  `computed_mask_standalone_reduce_add`, counts `0,1,16,17,257`, seeds
  `-11,17`, patterns `0,1`.
* Generated-bundle dry-run and FileCheck for shared vector computed-mask
  standalone min/max expectations after the binding summary update.
* Direct pre-realized route-entry fail-closed reproduction for
  `computed_mask_standalone_reduce_add`.
* Real `ssh rvv` generated-bundle compile/run for explicit and pre-realized
  `computed_mask_standalone_reduce_add`, counts `0,1,16,17,257`, seeds
  `-11,17`, patterns `0,1`.
* `rtk cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
* `rtk ./build/bin/tianchenrv-rvv-extension-plugin-test`
* Added-line old-authority scan over touched files.
* `rtk git diff --check`
* `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-02-stage2-rvv-computed-mask-standalone-reduce-add-artifact-abi`

Runtime evidence:

* Explicit artifact:
  `artifacts/tmp/06-02-computed-mask-standalone-reduce-add-ssh-rvv/explicit/computed_mask_standalone_reduce_add/evidence.json`
* Pre-realized artifact:
  `artifacts/tmp/06-02-computed-mask-standalone-reduce-add-ssh-rvv/pre-realized/computed_mask_standalone_reduce_add/evidence.json`
* PASS marker:
  `PASS op=computed_mask_standalone_reduce_add counts=0,1,16,17,257 seeds=-11,17 patterns=0,1`
* Remote compile/run succeeded on `rvv`; evidence records `remote_arch=riscv64`
  and `Ubuntu clang version 18.1.3`.

Spec update review:

* No `.trellis/spec/` update was required. Existing RVV plugin, EmitC route,
  and testing specs already require provider-owned route operand-binding facts,
  `hdr` participation for exported header/prototype parameters, fail-closed
  target artifact validation, and real `ssh rvv` evidence for runtime
  correctness claims.

## Current Phase

Finish/archive.
