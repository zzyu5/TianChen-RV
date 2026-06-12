# Stage2 RVV computed-masked segment2 store unit-load artifact ABI boundary

## Goal

Prove and tighten one bounded Stage 2 executable path for the existing
pre-realized `computed_masked_segment2_store_unit_load` selected RVV body:

```text
selected tcrv.exec RVV variant
  -> RVV plugin-owned realized/validated tcrv_rvv compare plus masked_segment2_store body
  -> computed-mask plus segment2 route-family provider facts
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> RVV target artifact bundle
  -> ssh rvv correctness evidence
```

The module boundary is where compare-produced mask ownership meets two unit-load
source fields and an interleaved segment2 destination store. Compare lhs/rhs,
unit-load `src0`/`src1`, destination `dst`, runtime `n`/AVL, mask facts,
segment tuple/config, dtype/config, memory form, inactive-lane destination
preservation, VL/tail policy, provider route operand binding, generated
header/prototype facts, and target artifact validation must come from typed
`tcrv_rvv` structure and RVV plugin/provider validation. They must not be
inferred from route ids, artifact names, manifests, test names, C strings,
descriptors, exact intrinsic spellings, source-front-door residue, common
EmitC/export logic, or mirror metadata.

## What I Already Know

* The repository started clean on `main`.
* No `.trellis/.current-task` existed; this task was created from the current
  Hermes Direction Brief.
* The relevant specs require the current RVV path to flow through typed
  `tcrv_rvv` selected bodies, RVV plugin legality/realization/provider facts,
  common EmitC materialization, and target artifact validation.
* The immediately previous `computed_masked_segment2_load_unit_store` task
  closed the reverse direction with provider `abi` plus header `hdr` binding
  facts, target artifact fail-closed validation, focused generated-bundle
  coverage, direct pre-realized route-entry rejection, and real `ssh rvv`
  evidence over counts `0,1,16,17,257` and patterns `0,1`.
* This round must inspect the live store-unit-load path and either prove the
  required contract already exists or add the smallest missing production
  guard, provider fact, target validation, focused test, or runtime evidence.

## Requirements

* Validate or realize the pre-realized selected body in RVV plugin-local owners
  before provider route construction. The combined path must consume the
  computed-mask owner facts and the segment2 planning/statement-plan owner
  facts; direct route-entry shortcuts must remain fail-closed.
* Preserve provider-derived facts for compare-produced mask source, predicate
  kind, mask role/source/memory form, two unit-load field sources, masked
  segment2 interleaved destination store, segment count, tuple/config, runtime
  `n`/AVL, SEW/LMUL/policy, tail/mask policy, inactive-lane destination
  preservation, and provider route operand binding.
* Ensure exported runtime ABI parameters `cmp_lhs`, `cmp_rhs`, `src0`, `src1`,
  `dst`, and `n` carry provider ABI marker `abi` and header/prototype marker
  `hdr` in route operand-binding facts, generated header/prototype metadata,
  and target artifact validation.
* Add or prove focused target artifact fail-closed validation for missing,
  stale, or mirror-only computed-mask segment2 store binding summaries,
  including stale `cmp_lhs`, `cmp_rhs`, `src0`, `src1`, `dst`, or `n` facts.
* Keep common EmitC/export neutral. Common materialization may carry
  provider-owned payloads unchanged, but it must not infer RVV mask semantics,
  segment semantics, field roles, dtype/config, memory form, ABI order, or
  policy.
* Strengthen focused generated-bundle dry-run evidence so the route binding
  summary, compare mask facts, unit-load field sources, interleaved segment2
  destination store, runtime ABI order, typed memory forms, tuple/config,
  runtime `n`/AVL, SEW/LMUL/policy, inactive-lane policy, and tail policy are
  explicitly visible.
* Record real `ssh rvv` correctness evidence for counts `0`, `1`, a
  VL-boundary count, a tail count, and a larger count. Use at least two
  mask/source/destination patterns that distinguish active masked segment2
  interleaved store order, masked-off destination preservation, source
  preservation, and destination tail sentinel preservation.

## Acceptance Criteria

* [x] Focused evidence shows the pre-realized body is consumed by RVV
      plugin-local computed-mask and segment2 selected-body/planning owners
      before provider route construction.
* [x] Focused EmitC/export or target-artifact coverage shows `cmp_lhs`,
      `cmp_rhs`, `src0`, `src1`, `dst`, and `n` ABI order, provider `abi`
      markers, header/prototype `hdr` markers, compare mask source, predicate
      kind, unit-load field sources, interleaved segment2 destination store,
      tuple type/config, runtime `n`/AVL, SEW/LMUL/policy, inactive-lane
      policy, and tail policy survive into the generated bundle.
* [x] Target artifact validation fails closed when the computed-mask segment2
      store binding summary is stale, missing, from a stale candidate mirror,
      or lacks required `abi`/`hdr` markers for exported runtime ABI
      parameters.
* [x] Focused script dry-run coverage checks the exact provider binding
      summary and computed-mask segment2 store boundary fields.
* [x] Direct pre-realized route-entry or artifact export without the public
      selected lowering-boundary producer is rejected with focused evidence.
* [x] Real `ssh rvv` compile/run evidence passes counts `0`, `1`, `16`, `17`,
      and `257` with at least two non-vacuous mask/source/destination patterns
      and verifies active masked segment2 interleaved store order, masked-off
      destination preservation, source preservation, and destination tail
      sentinel preservation.
* [x] Bounded old-authority scan over touched plugin/provider/materializer/
      target/script/test/spec files classifies remaining hits for:
      `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `!tcrv_rvv.i32m`,
      `__riscv_.*_i32m1`, `source-front-door`, `source-artifact`,
      `emission_plan`, `descriptor`, and `selected route`.
* [x] Smallest relevant build/test/script commands pass, with self-repair for
      failures.
* [x] The task is finished/archived and one coherent commit is created if the
      module behavior is complete.

## Technical Approach

Use the reverse computed-mask segment2 load ABI task and the plain segment2
interleave/deinterleave ABI tasks as production patterns, but apply them only to
`computed_masked_segment2_store_unit_load`:

* inspect the dialect op, computed-mask selected-body owner, segment2 selected
  body/planning/statement-plan owners, route planning, route provider, common
  EmitC materializer, target artifact validation, target support bundle,
  generated-bundle script, remote probe, and focused tests;
* if the production path already carries most facts, add only the missing
  fail-closed guard, provider binding fact, target validation,
  generated-bundle expectation, or runtime evidence that protects this exact
  ABI boundary;
* align the combined provider binding summary with the `abi` plus `hdr`
  contract for `cmp_lhs,cmp_rhs,src0,src1,dst,n`;
* update explicit/pre-realized target artifact fixtures and generated-bundle
  dry-run checks for the combined computed-mask segment2 store boundary;
* strengthen the generated-bundle runtime harness only as needed to prove
  active masked segment2 interleaved store order, masked-off destination
  preservation, source preservation, and tail preservation over the required
  counts and patterns.

## Out Of Scope

* Computed-masked segment2 load or update follow-up except direct reuse of its
  validated contract.
* Indexed or strided memory rework.
* Reductions, contractions, dtype or LMUL clone batches.
* High-level frontend authority or per-Linalg lowering.
* Source-front-door positive routes.
* Common EmitC semantic inference.
* Dashboards, broad smoke matrices, or performance claims.

## Technical Notes

Relevant specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/index.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/testing/mlir-testing-contract.md`
* `.trellis/spec/guides/index.md`
* `.trellis/spec/guides/capability-first-design-guide.md`
* `.trellis/spec/guides/plugin-locality-review-guide.md`
* `.trellis/spec/guides/compute-boundary-review-guide.md`

Closest archived pattern:

* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-segment2-load-unit-store-artifact-abi/prd.md`

Initial implementation focus:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.h`
* `include/TianChenRV/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `scripts/rvv_remote_probe.py`
* focused computed-mask segment2 store target/script/C++ tests

## Completion Status - 2026-06-02

Status: local implementation, focused verification, and real `ssh rvv`
runtime evidence are complete. The task is ready to archive and commit.

Completed local work:

* Added `hdr` participation to the provider operand-binding summary for
  `cmp_lhs`, `cmp_rhs`, `src0`, and `src1`, completing the
  `cmp_lhs,cmp_rhs,src0,src1,dst,n` `abi|hdr` binding summary for
  `computed_masked_segment2_store_unit_load`.
* Added route-construction fail-closed checks requiring those store-unit-load
  `hdr` binding facts before provider route construction; update-unit-load is
  intentionally left outside this task.
* Updated target artifact validator expectations and added target C++ negative
  coverage for stale provider summaries and stale candidate mirrors that omit
  exported header markers.
* Updated explicit/pre-realized target artifact fixtures and script dry-run
  checks to assert the exact provider binding summary.
* Fixed the generated-bundle store harness to run counts across patterns
  `0,1` and print `patterns=0,1`; added script self-test checks so a
  `run_case(counts[index])` single-pattern regression cannot pass local
  dry-run while failing real `ssh rvv` compilation.
* Updated `.trellis/spec/testing/mlir-testing-contract.md` with the
  multi-pattern generated-bundle harness contract.

Checks passed:

```bash
rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2
rtk build/bin/tianchenrv-target-artifact-export-test
rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test
rtk /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter computed-masked-segment2-store .
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/computed-mask-segment2-store-dry-run --run-id computed-mask-segment2-store --overwrite --op-kind computed_masked_segment2_store_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/computed-mask-segment2-store-pre-realized-dry-run --run-id pre-realized-computed-mask-segment2-store --overwrite --op-kind computed_masked_segment2_store_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk git diff --check
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-02-stage2-rvv-computed-masked-segment2-store-unit-load-artifact-abi
```

Runtime evidence completed:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/computed-mask-segment2-store-rvv-evidence --run-id pre-realized-computed-mask-segment2-store --overwrite --op-kind computed_masked_segment2_store_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 20
```

The run passed on the real RVV target:

```text
rvv_generated_bundle_abi_e2e: success
tcrv_rvv_generated_bundle_abi_computed_masked_segment2_store_unit_load_ok counts=0,1,16,17,257 patterns=0,1
PASS op=computed_masked_segment2_store_unit_load counts=0,1,16,17,257 patterns=0,1
```

Self-repair:

* The first `ssh rvv` run failed at remote clang compile with
  `too few arguments to function call, expected 2, have 1` because the store
  harness generator emitted `run_case(size_t n, int pattern)` but main still
  called `run_case(counts[index])`. The script generator, self-test, FileCheck
  harness expectations, and runtime evidence were fixed and rerun.

Bounded old-authority scan classification:

* `__riscv_*_i32m1x2` hits in touched store target/script tests are
  provider-derived segment2 intrinsic mirror/evidence strings, not route
  authority.
* `source-front-door`, `source-export`, `descriptor`, and `tcrv_rvv.i32_`
  hits in touched script tests are negative `implicit-check-not` guards.
* `selected route` hits in touched provider code are diagnostic text about
  same-analysis provenance, not selected-route authority.
* Large pre-existing `tcrv_rvv.i32_`, exact intrinsic, `descriptor`, and
  `emission_plan` hits in `test/Target/TargetArtifactExportTest.cpp` are
  unrelated legacy/negative target-artifact inventory outside this task's new
  computed-mask segment2 store section.
