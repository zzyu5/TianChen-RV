# Stage2 RVV computed-masked segment2 load unit-store artifact ABI boundary

## Goal

Prove and tighten one bounded Stage 2 executable path for the existing
pre-realized `computed_masked_segment2_load_unit_store` selected RVV body:

```text
selected tcrv.exec RVV variant
  -> RVV plugin-owned realized/validated tcrv_rvv compare plus masked_segment2_load body
  -> computed-mask plus segment2 route-family provider facts
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> RVV target artifact bundle
  -> ssh rvv correctness evidence
```

The module boundary is where compare-produced mask ownership meets segment2
interleaved source load and two unit field stores. Compare lhs/rhs, interleaved
source, old field passthroughs, field0/field1 output stores, runtime `n`/AVL,
mask facts, segment tuple/config, dtype/config, memory form, inactive-lane
policy, VL/tail policy, provider route operand binding, generated
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
* The immediately previous segment2 deinterleave and interleave tasks closed
  plain segment2 ABI boundaries with provider `abi` plus header `hdr`
  operand-binding facts, target artifact fail-closed validation, focused
  generated-bundle coverage, direct pre-realized route-entry rejection, and
  real `ssh rvv` evidence over counts `0,1,16,17,257`.
* The computed-mask indexed gather/scatter tasks provide the closest pattern
  for compare-produced mask facts, inactive-lane preservation, provider
  binding summaries, target artifact fail-closed checks, generated harness
  pattern variation, and runtime evidence.
* Initial repository search shows existing bounded support for
  `computed_masked_segment2_load_unit_store`, including the dialect
  pre-realized body, `tcrv_rvv.masked_segment2_load`, target artifact fixtures,
  route planning code, target artifact validation code, and generated-bundle
  script support. This round must inspect the live path and either prove the
  required contract already exists or add the smallest missing production
  guard, provider fact, target validation, focused test, or runtime evidence.

## Requirements

* Validate or realize the pre-realized selected body in RVV plugin-local owners
  before provider route construction. The combined path must consume the
  computed-mask owner facts and the segment2 planning/statement-plan owner
  facts; direct route-entry shortcuts must remain fail-closed.
* Preserve provider-derived facts for compare-produced mask source, predicate
  kind, mask role/source/memory form, inactive-lane passthrough policy, old
  field0/field1 passthrough loads, masked segment2 interleaved source load,
  segment count, tuple/config, field0/field1 unit stores, runtime `n`/AVL,
  SEW/LMUL/policy, tail/mask policy, and provider route operand binding.
* Ensure exported runtime ABI parameters `cmp_lhs`, `cmp_rhs`, `src`, `out0`,
  `out1`, and `n` carry provider ABI marker `abi` and header/prototype marker
  `hdr` in route operand-binding facts, generated header/prototype metadata,
  and target artifact validation.
* Add or prove focused target artifact fail-closed validation for missing,
  stale, or mirror-only computed-mask segment2 load binding summaries,
  including stale `cmp_lhs`, `cmp_rhs`, `src`, `out0`, `out1`, or `n` facts.
* Keep common EmitC/export neutral. Common materialization may carry
  provider-owned payloads unchanged, but it must not infer RVV mask semantics,
  segment semantics, field roles, dtype/config, memory form, ABI order, or
  policy.
* Strengthen focused generated-bundle dry-run evidence so the route binding
  summary, compare mask facts, old passthrough fields, interleaved segment2
  source role, field0/field1 stores, runtime ABI order, typed memory forms,
  tuple/config, runtime `n`/AVL, SEW/LMUL/policy, inactive-lane policy, and
  tail policy are explicitly visible.
* Record real `ssh rvv` correctness evidence for counts `0`, `1`, a
  VL-boundary count, a tail count, and a larger count. Use at least two
  mask/source/passthrough patterns that distinguish active masked segment2
  field order, masked-off old field preservation, source preservation where
  applicable, and output tail sentinel preservation.

## Acceptance Criteria

* [x] Focused evidence shows the pre-realized body is consumed by RVV
      plugin-local computed-mask and segment2 selected-body/planning owners
      before provider route construction.
* [x] Focused EmitC/export or target-artifact coverage shows `cmp_lhs`,
      `cmp_rhs`, `src`, `out0`, `out1`, and `n` ABI order, provider `abi`
      markers, header/prototype `hdr` markers, compare mask source, predicate
      kind, old field passthroughs, interleaved segment2 source, field0/field1
      stores, tuple type/config, runtime `n`/AVL, SEW/LMUL/policy,
      inactive-lane policy, and tail policy survive into the generated bundle.
* [x] Target artifact validation fails closed when the computed-mask segment2
      load binding summary is stale, missing, from a stale candidate mirror, or
      lacks required `abi`/`hdr` markers for exported runtime ABI parameters.
* [x] Focused script dry-run coverage checks the exact provider binding
      summary and computed-mask segment2 load boundary fields.
* [x] Direct pre-realized route-entry or artifact export without the public
      selected lowering-boundary producer is rejected with focused evidence.
* [x] Real `ssh rvv` compile/run evidence passes counts `0`, `1`, `16`, `17`,
      and `257` with at least two non-vacuous mask/source/passthrough patterns
      and verifies active masked segment2 field order, inactive old field
      preservation, and output tail sentinel preservation.
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

Use the plain segment2 deinterleave/interleave ABI tasks and the
computed-mask indexed gather/scatter ABI tasks as production patterns, but
apply them only to `computed_masked_segment2_load_unit_store`:

* inspect the dialect op, computed-mask selected-body owner, segment2 selected
  body/planning/statement-plan owners, route planning, route provider, common
  EmitC materializer, target artifact validation, target support bundle,
  generated-bundle script, remote probe, and focused tests;
* if the production path already carries most facts, add only the missing
  fail-closed guard, provider binding fact, target validation, generated-bundle
  expectation, or runtime evidence that protects this exact ABI boundary;
* align the combined provider binding summary with the `abi` plus `hdr`
  contract for `cmp_lhs,cmp_rhs,src,out0,out1,n`;
* update explicit/pre-realized target artifact fixtures and generated-bundle
  dry-run checks for the combined computed-mask segment2 load boundary;
* strengthen the generated-bundle runtime harness only as needed to prove
  active masked segment2 field order, inactive old field preservation, and
  tail preservation over the required counts and patterns.

## Out Of Scope

* Computed-masked segment2 store or update follow-up.
* More plain segment2 deinterleave/interleave rework except direct reuse of
  their validated contracts.
* Indexed or strided memory rework.
* Reductions, contractions, dtype or LMUL clone batches.
* High-level frontend authority or per-Linalg lowering.
* Source-front-door positive routes.
* Common EmitC semantic inference.
* Dashboards, broad smoke matrices, or performance claims.

## Technical Notes

Relevant specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Closest archived patterns:

* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-segment2-deinterleave-unit-store-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-segment2-interleave-unit-load-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-indexed-gather-load-unit-store-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-indexed-scatter-store-unit-load-artifact-abi/prd.md`

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
* focused computed-mask segment2 load target/script/C++ tests

## Continuation Status - 2026-06-02

Status: local implementation, focused local verification, and real `ssh rvv`
runtime evidence are complete. The task is ready to archive and commit.

Completed local work:

* Added `hdr` markers to the provider operand-binding summary for
  `cmp_lhs`, `cmp_rhs`, and `src`, so the computed-mask segment2 load route
  now exports `cmp_lhs,cmp_rhs,src,out0,out1,n` with provider `abi` and
  header/prototype `hdr` participation.
* Added target artifact fail-closed validation for stale or missing
  computed-mask segment2 load binding summaries, including stale candidate
  mirror metadata.
* Updated explicit and pre-realized target artifact fixtures, generated-bundle
  dry-run expectations, direct pre-realized route-entry negative evidence, and
  the computed-mask segment2 load harness count/pattern evidence.
* Strengthened the harness to run two mask/source/passthrough patterns for
  counts `0,1,16,17,257`, checking active field order, inactive old-field
  preservation, source preservation, and output tail sentinels.

Checks passed in this continuation:

```bash
rtk git diff --check
rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test
rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2
rtk build/bin/tianchenrv-target-artifact-export-test
rtk /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter computed-masked-segment2-load .
rtk /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter direct-pre-realized-computed-masked-segment2-load .
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/computed-mask-segment2-load-dry-run --run-id computed-mask-segment2-load --overwrite --op-kind computed_masked_segment2_load_unit_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/computed-mask-segment2-load-pre-realized-dry-run --run-id pre-realized-computed-mask-segment2-load --overwrite --op-kind computed_masked_segment2_load_unit_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Runtime evidence completed:

```text
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/computed-mask-segment2-load-rvv-evidence --run-id pre-realized-computed-mask-segment2-load --overwrite --op-kind computed_masked_segment2_load_unit_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 20
```

The run passed on the real RVV target:

```text
rvv_generated_bundle_abi_e2e: success
tcrv_rvv_generated_bundle_abi_computed_masked_segment2_load_unit_store_ok counts=0,1,16,17,257 patterns=0,1
PASS op=computed_masked_segment2_load_unit_store counts=0,1,16,17,257 patterns=0,1
```

The per-case evidence covered counts `0,1,16,17,257` and patterns `0,1`,
including active masked segment2 field order, inactive old-field preservation,
source preservation, and output tail sentinel preservation.

Spec update judgment:

* No `.trellis/spec/` edit is needed in this final continuation. The
  implementation follows existing RVV plugin, EmitC operand-binding, target
  artifact validation, and generated-bundle `ssh rvv` evidence contracts; this
  continuation added missing real hardware evidence but did not introduce a new
  reusable contract or convention.

Bounded old-authority scan classification:

* `__riscv_*_i32m1x2` / `__riscv_*_i32m1` hits in touched tests and
  `scripts/rvv_generated_bundle_abi_e2e.py` are provider-derived intrinsic
  mirror and generated-bundle evidence strings, not route authority.
* `descriptor`, `source-export`, and `tcrv_rvv.i32_` hits in touched script
  and tests are negative checks, self-test residue rejection, or pre-existing
  fail-closed inventory.
* No new `RVVI32M1`, `rvv-i32m1`, `source-front-door`, `source-artifact`, or
  descriptor-driven route authority was added to the touched production path.

Final checks:

```bash
rtk git diff --check
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-02-06-02-stage2-rvv-computed-masked-segment2-load-unit-store-artifact-abi
```

Both final checks passed. The next action is archive plus one coherent commit.
