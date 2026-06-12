# Stage2 RVV segment2 interleave unit-load artifact ABI boundary

## Goal

Prove and tighten one bounded Stage 2 executable path for the existing
pre-realized `segment2_interleave_unit_load` selected RVV body:

```text
selected tcrv.exec RVV variant
  -> RVV plugin-owned realized/validated tcrv_rvv segment2 interleave body
  -> segment2 memory route-family provider facts
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> RVV target artifact bundle
  -> ssh rvv correctness evidence
```

The module boundary is segment2 interleave memory movement ownership:
`field0` and `field1` source ABI roles, ordinary unit-load source memory forms,
segment tuple construction/type/config, interleaved segment2 destination role,
runtime `n`/AVL, SEW/LMUL/policy, VL/tail policy, provider route operand
binding facts, and generated header/prototype facts must come from typed
`tcrv_rvv` structure and RVV plugin/provider validation. They must not be
inferred from route ids, artifact names, manifests, test names, C strings,
descriptors, exact intrinsic spellings, source-front-door residue, or common
EmitC/export logic.

## What I Already Know

* The repository started clean on `main`.
* No `.trellis/.current-task` existed; this task was created from the current
  Hermes Direction Brief.
* The relevant specs require the current RVV path to flow through typed
  `tcrv_rvv` selected bodies, RVV plugin legality/realization/provider facts,
  common EmitC materialization, and target artifact validation.
* The previous archived segment2 deinterleave task tightened provider
  `abi`/`hdr` operand-binding facts for `src`, `out0`, `out1`, and `n`,
  added target artifact fail-closed validation, strengthened generated-bundle
  dry-run/runtime harness evidence, and recorded `ssh rvv` correctness.
* This round must apply the reverse direction boundary to exactly the existing
  `segment2_interleave_unit_load` path: two field inputs are loaded, a segment
  tuple is constructed, and the tuple is unit-stored to the interleaved
  destination.

## Requirements

* Keep the RVV segment2 memory owner as the gate that validates or realizes the
  pre-realized selected body before provider route construction.
* Preserve provider-derived facts for `field0`, `field1`, `dst`, and `n` ABI
  order; ordinary field source memory forms; interleaved segment2 destination
  role; tuple type/config; runtime `n`/AVL; SEW/LMUL/policy; memory forms;
  VL/tail policy; and provider route operand binding.
* Ensure the `segment2_interleave_unit_load` route operand binding summary
  marks every exported runtime ABI parameter with provider ABI marker `abi` and
  header/prototype marker `hdr`.
* Add focused target artifact fail-closed validation if live code does not
  already reject stale, missing, or mirror-only `field0`, `field1`, `dst`, or
  `n` binding summaries.
* Keep common EmitC/export neutral; no RVV semantic inference may move into
  common materialization or target packaging.
* Strengthen focused generated-bundle dry-run and target artifact expectations
  so runtime ABI order, source/destination memory forms, tuple config, segment
  direction, and provider/header binding summary are visible.
* Record real `ssh rvv` correctness evidence for counts `0`, `1`, a
  VL-boundary count, a tail count, and a larger count, with at least two field
  input patterns that distinguish interleave from contiguous copy or
  single-field store and prove interleaved destination plus tail sentinel
  preservation.

## Acceptance Criteria

* [x] Focused evidence shows the pre-realized body is consumed by the RVV
      segment2 memory selected-body owner before route construction.
* [x] Focused EmitC/export or target-artifact coverage shows `field0`,
      `field1`, `dst`, `n` ABI order, dual field source memory forms,
      interleaved segment2 destination role, tuple type/config, runtime
      `n`/AVL, SEW/LMUL/policy, and tail policy survive into the generated
      bundle/header/prototype.
* [x] Target artifact validation fails closed when the segment2 interleave
      binding summary is stale, missing, or lacks required `abi`/`hdr` markers
      for exported runtime ABI parameters.
* [x] Focused script dry-run coverage checks the exact provider binding summary
      and segment2 interleave boundary fields.
* [x] Direct pre-realized route-entry or artifact export without realization is
      rejected or already covered with focused evidence.
* [x] Real `ssh rvv` compile/run evidence passes counts `0`, `1`, a
      VL-boundary count, a tail count, and a larger count with at least two
      non-vacuous field input patterns and verifies interleaved destination
      plus tail sentinel preservation.
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

Use the just-archived segment2 deinterleave ABI boundary as the closest
production pattern:

* inspect the existing `segment2_interleave_unit_load` selected-body owner,
  route-family planning owner, memory statement-plan owner, route provider,
  common materializer, target artifact validator, generated-bundle script, and
  focused tests;
* if the production path already carries most facts, add only the smallest
  missing fail-closed guard or route-fact validation that protects the
  interleave ABI boundary;
* align `segment2_interleave_unit_load` provider binding summaries and target
  validation with the provider `abi` plus header `hdr` marker contract;
* update explicit/pre-realized target artifact fixtures and generated-bundle
  dry-run checks for `field0`, `field1`, `dst`, and `n`;
* strengthen the generated-bundle runtime harness only as needed to prove
  interleave behavior and tail preservation over the required counts and
  patterns.

## Out Of Scope

* Computed-masked segment2 load/store/update.
* More segment2 deinterleave rework.
* Indexed or strided memory rework.
* Contraction follow-up.
* Dtype or LMUL clone batches.
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

Closest archived pattern:

* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-segment2-deinterleave-unit-store-artifact-abi/prd.md`

Initial implementation focus:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `scripts/rvv_remote_probe.py`
* focused segment2 interleave lit/script/C++ tests

## Completed Behavior

* Tightened the RVV-owned
  `rvv-route-operand-binding:segment2_interleave_unit_load.v1` provider
  summary so exported `src0`, `src1`, `dst`, and `n` entries now carry
  provider ABI marker `abi` and header/prototype marker `hdr`.
* Updated segment2 interleave memory operand-binding validation so route
  construction requires the new `hdr` uses before segment2 provider plan and
  statement-plan handoff.
* Added target-owned
  `validatePlainSegment2InterleaveHeaderBindingSummary`. Target artifact
  validation now fails closed if the interleave route summary lacks the
  expected provider plan, required runtime ABI order, or `abi`/`hdr` markers
  for `src0`, `src1`, `dst`, or `n`.
* Updated explicit and pre-realized target artifact fixtures and generated
  bundle dry-run expectations to assert the new provider/header binding
  summary.
* Strengthened the generated bundle harness for segment2 interleave to run two
  field input patterns for each runtime count and check interleaved destination
  field order plus tail sentinel preservation.
* Updated the lowering-runtime EmitC route spec with the segment2 interleave
  `src0,src1,dst,n` `abi`/`hdr` contract.

## Validation Evidence

Commands run:

```bash
rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test
rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2
rtk build/bin/tianchenrv-target-artifact-export-test
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'segment2-interleave-unit-load'   # workdir: build/test
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/segment2-interleave-dry-run --run-id pre-realized-segment2-interleave-unit-load --overwrite --op-kind segment2_interleave_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry --artifact-root artifacts/tmp/segment2-interleave-direct-negative --run-id direct-pre-realized-segment2-interleave-unit-load --overwrite --op-kind segment2_interleave_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/segment2-interleave-rvv-evidence --run-id pre-realized-segment2-interleave-unit-load --overwrite --op-kind segment2_interleave_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 20
rtk git diff --check
```

Direct pre-realized route-entry fail-closed evidence:

```text
rvv_generated_bundle_abi_e2e: failed
--direct-pre-realized-route-entry is unsupported for selected pre-realized op kind(s): segment2_interleave_unit_load; the direct route-entry shortcut is retired and these fixtures must use the public selected lowering-boundary producer before target bundle export
```

`ssh rvv` evidence:

```text
rvv_generated_bundle_abi_e2e: success
segment2_interleave_unit_load case n=0 pattern=0 ok field_order_distinguishing_lanes=0 tail_preserved
segment2_interleave_unit_load case n=0 pattern=1 ok field_order_distinguishing_lanes=0 tail_preserved
segment2_interleave_unit_load case n=1 pattern=0 ok field_order_distinguishing_lanes=1 tail_preserved
segment2_interleave_unit_load case n=1 pattern=1 ok field_order_distinguishing_lanes=1 tail_preserved
segment2_interleave_unit_load case n=16 pattern=0 ok field_order_distinguishing_lanes=16 tail_preserved
segment2_interleave_unit_load case n=16 pattern=1 ok field_order_distinguishing_lanes=16 tail_preserved
segment2_interleave_unit_load case n=17 pattern=0 ok field_order_distinguishing_lanes=17 tail_preserved
segment2_interleave_unit_load case n=17 pattern=1 ok field_order_distinguishing_lanes=17 tail_preserved
segment2_interleave_unit_load case n=257 pattern=0 ok field_order_distinguishing_lanes=257 tail_preserved
segment2_interleave_unit_load case n=257 pattern=1 ok field_order_distinguishing_lanes=256 tail_preserved
PASS op=segment2_interleave_unit_load counts=0,1,16,17,257
```

Self-repair performed:

* The first target artifact export test run exposed that the wrong-summary
  interleave negative asserted a stale string that the earlier provider-plan
  prefix diagnostic does not echo; tightened the expected diagnostic fragments
  to the fail-closed condition actually enforced.
* The first focused lit rerun exposed a HARNESS FileCheck order issue after the
  two-pattern loop was added; reordered the checks to match generated C.
* The first `ssh rvv` run exposed a harness template bug where `pattern` was
  used before being passed to `run_case`; fixed the interleave harness
  signature and reran `py_compile`, `--self-test`, dry-run, focused lit, and
  `ssh rvv`.

## Old Authority Scan

Bounded scan over touched plugin/provider/target/script/test/spec/task files
found no new positive legacy route authority. Remaining hits are classified:

* `__riscv_vsseg2e32_v_i32m1x2`, `__riscv_vcreate_v_i32m1x2`, and related
  exact intrinsic spellings are provider-derived RVV segment2 leaf mirrors or
  pre-existing generated-bundle validation checks for this typed e32m1 body,
  not route authority.
* `emission_plan` appears as existing diagnostic mirror text in FileCheck
  fixtures and spec guardrails.
* `descriptor`, `source-front-door`, `source-artifact`, `tcrv_rvv.i32_`,
  `!tcrv_rvv.i32m`, `rvv-i32m1`, and `RVVI32M1` hits in the touched spec,
  broad target artifact test, and script are negative guardrails, fail-closed
  residue checks, or historical invalid-route inventory.
* `selected route` appears in same-analysis provider diagnostic text and does
  not select RVV semantics.
