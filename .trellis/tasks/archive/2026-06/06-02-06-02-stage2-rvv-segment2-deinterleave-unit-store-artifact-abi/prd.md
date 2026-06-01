# Stage2 RVV segment2 deinterleave unit-store artifact ABI boundary

## Goal

Prove and tighten one bounded Stage 2 executable path for the existing
pre-realized `segment2_deinterleave_unit_store` selected RVV body:

```text
selected tcrv.exec RVV variant
  -> RVV plugin-owned realized/validated tcrv_rvv segment2 deinterleave body
  -> segment2 memory route-family provider facts
  -> TCRVEmitCLowerableRoute
  -> common EmitC materializer
  -> RVV target artifact bundle
  -> ssh rvv correctness evidence
```

The module boundary is segment2 deinterleave memory movement ownership:
interleaved segment2 source ABI, tuple load/config, field0 and field1 output
roles, two unit-store destinations, runtime `n`/AVL, SEW/LMUL/policy,
source/destination memory forms, provider route operand-binding facts, and
generated header/prototype facts must come from typed `tcrv_rvv` structure and
RVV plugin/provider validation. They must not be inferred from route ids,
artifact names, manifests, test names, C strings, descriptors, exact intrinsic
spellings, source-front-door residue, or common EmitC/export logic.

## What I Already Know

* The repository started clean on `main`.
* No `.trellis/.current-task` existed; this task was created from the current
  Hermes Direction Brief.
* The relevant specs require the current RVV path to flow through typed
  `tcrv_rvv` selected bodies, RVV plugin legality/realization/provider facts,
  common EmitC materialization, and target artifact validation.
* Current code already has:
  * `tcrv_rvv.typed_segment2_deinterleave_memory_pre_realized_body`;
  * plugin-local pre-realized validation and realization into
    `setvl/with_vl/segment2_load/move/move/store/store`;
  * segment2 route-family provider planning;
  * segment2 memory statement-plan ownership;
  * explicit and pre-realized target artifact fixtures;
  * target artifact candidate/provider validation for segment2 tuple type,
    field roles, memory forms, route-family mirrors, and statement plan;
  * C++ negative tests for stale segment2 metadata and wrong binding summary.
* Live inspection shows the plain segment2 deinterleave route binding summary
  still exposes exported ABI/header participation as
  `runtime-abi-mirror|...|header`, unlike newer provider ABI boundaries that
  require `abi` and `hdr`.
* Live inspection shows target artifact validation compares the segment2
  deinterleave binding summary exactly, but does not independently validate
  that `src`, `out0`, `out1`, and `n` all carry provider ABI marker `abi` plus
  header/prototype marker `hdr`.

## Requirements

* Keep the RVV segment2 memory owner as the gate that validates or realizes the
  pre-realized selected body before provider route construction.
* Preserve provider-derived facts for interleaved segment2 source role, segment
  tuple type/config, field0 and field1 output roles, two unit-store
  destinations, runtime `n`/AVL, SEW/LMUL/policy, memory forms, VL/tail policy,
  and provider route operand binding.
* Tighten `segment2_deinterleave_unit_store` route operand binding summary so
  all exported runtime ABI parameters carry provider ABI marker `abi` and
  header/prototype marker `hdr`.
* Add focused target artifact fail-closed validation for missing or stale
  `src`, `out0`, `out1`, or `n` binding markers.
* Keep common EmitC/export neutral; no RVV semantic inference may move into
  common materialization or target packaging.
* Strengthen focused generated-bundle dry-run and target artifact expectations
  so runtime ABI order, source memory form, field0/field1 output roles, tuple
  config, and the provider/header binding summary are visible.
* Record real `ssh rvv` correctness evidence for counts `0`, `1`, a
  VL-boundary count, a tail count, and a larger count, with at least two
  interleaved source patterns that distinguish deinterleave from contiguous copy
  or single-field load and prove both field outputs plus tail sentinels.

## Acceptance Criteria

* [x] Focused evidence shows the pre-realized body is consumed by the RVV
      segment2 memory selected-body owner before route construction.
* [x] Focused EmitC/export or target-artifact coverage shows `src`, `out0`,
      `out1`, `n` ABI order, segment2 source memory form, field0/field1 output
      roles, tuple type/config, runtime `n`/AVL, SEW/LMUL/policy, and tail
      policy survive into the generated bundle/header/prototype.
* [x] Target artifact validation fails closed when the segment2 deinterleave
      binding summary is stale, missing, or lacks required `abi`/`hdr` markers
      for exported runtime ABI parameters.
* [x] Focused script dry-run coverage checks the exact provider binding summary
      and segment2 deinterleave boundary fields.
* [x] Direct pre-realized route-entry or artifact export without realization is
      rejected or already covered with focused evidence.
* [x] Real `ssh rvv` compile/run evidence passes counts `0`, `1`, a
      VL-boundary count, a tail count, and a larger count with at least two
      non-vacuous interleaved source patterns and verifies output tail sentinel
      preservation.
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

Use the current segment2 deinterleave implementation and the just-completed
computed-masked indexed scatter ABI boundary as the closest production
patterns:

* update the provider-owned `segment2_deinterleave_unit_store` binding plan
  uses from mirror-only wording to `abi` plus `hdr`;
* add target-owned segment2 deinterleave header-binding summary validation for
  `src`, `out0`, `out1`, and `n`;
* update explicit/pre-realized target artifact fixture expectations and
  generated-bundle dry-run expectations;
* add C++ target artifact negative coverage for stale provider summary and
  stale candidate mirror metadata that omit `abi`/`hdr`;
* strengthen the generated-bundle runtime harness only as needed to prove
  deinterleave behavior and tail preservation over the required counts and
  patterns.

## Out Of Scope

* Segment2 interleave.
* Computed-masked segment2 load/store/update.
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

* `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-computed-masked-indexed-scatter-store-unit-load-artifact-abi/prd.md`

Initial implementation focus:

* `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* `test/Target/RVV/pre-realized-selected-body-artifact-segment2-deinterleave-unit-store.mlir`
* `test/Target/RVV/explicit-selected-body-artifact-segment2-deinterleave-unit-store.mlir`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* focused `test/Scripts` segment2 deinterleave tests

## Completed Behavior

* Tightened the RVV-owned
  `rvv-route-operand-binding:segment2_deinterleave_unit_store.v1` provider
  summary so exported `src`, `out0`, `out1`, and `n` entries now carry
  provider ABI marker `abi` and header/prototype marker `hdr`.
* Updated segment2 deinterleave memory operand-binding validation so route
  construction requires the new `hdr` uses before segment2 provider plan and
  statement-plan handoff.
* Added target-owned
  `validatePlainSegment2DeinterleaveHeaderBindingSummary`. Target artifact
  validation now fails closed if the deinterleave route summary lacks the
  expected provider plan, required runtime ABI order, or `abi`/`hdr` markers for
  `src`, `out0`, `out1`, or `n`.
* Updated explicit and pre-realized target artifact fixtures to assert the new
  provider/header binding summary.
* Added C++ negative coverage for stale deinterleave binding summaries that
  preserve old `runtime-abi-mirror` / `header` wording.
* Strengthened the generated bundle harness for deinterleave to run two
  interleaved source patterns for each runtime count and check both field
  outputs plus tail sentinel preservation.
* Updated the lowering-runtime EmitC route spec with the segment2 deinterleave
  `src,out0,out1,n` `abi`/`hdr` contract.

## Validation Evidence

Commands run:

```bash
rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test
rtk git diff --check
rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2
rtk build/bin/tianchenrv-target-artifact-export-test
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'segment2-deinterleave-unit-store'   # workdir: build/test
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/segment2-deinterleave-dry-run --run-id pre-realized-segment2-deinterleave-unit-store --overwrite --op-kind segment2_deinterleave_unit_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/segment2-deinterleave-rvv-evidence --run-id pre-realized-segment2-deinterleave-unit-store --overwrite --op-kind segment2_deinterleave_unit_store --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 20
```

`ssh rvv` evidence:

```text
rvv_generated_bundle_abi_e2e: success
segment2_deinterleave_unit_store case n=0 pattern=0 ok field_order_distinguishing_lanes=0 tail_preserved
segment2_deinterleave_unit_store case n=0 pattern=1 ok field_order_distinguishing_lanes=0 tail_preserved
segment2_deinterleave_unit_store case n=1 pattern=0 ok field_order_distinguishing_lanes=1 tail_preserved
segment2_deinterleave_unit_store case n=1 pattern=1 ok field_order_distinguishing_lanes=1 tail_preserved
segment2_deinterleave_unit_store case n=16 pattern=0 ok field_order_distinguishing_lanes=16 tail_preserved
segment2_deinterleave_unit_store case n=16 pattern=1 ok field_order_distinguishing_lanes=16 tail_preserved
segment2_deinterleave_unit_store case n=17 pattern=0 ok field_order_distinguishing_lanes=17 tail_preserved
segment2_deinterleave_unit_store case n=17 pattern=1 ok field_order_distinguishing_lanes=17 tail_preserved
segment2_deinterleave_unit_store case n=257 pattern=0 ok field_order_distinguishing_lanes=257 tail_preserved
segment2_deinterleave_unit_store case n=257 pattern=1 ok field_order_distinguishing_lanes=257 tail_preserved
PASS op=segment2_deinterleave_unit_store counts=0,1,16,17,257
```

Self-repair performed:

* The first lit attempt used the source `test/` config directly and lacked
  build-site variables; reran lit from `build/test`.
* The first dry-run attempt used bare `llvm-readobj`; reran with
  `/usr/lib/llvm-20/bin/llvm-readobj`.
* The first FileCheck pass exposed a harness check-order issue; reordered the
  checks.
* A script patch initially changed the computed-mask segment2 load harness
  signature instead of the plain deinterleave harness signature; restored that
  signature and reran `py_compile`, `--self-test`, and focused lit.

## Old Authority Scan

Bounded scan over touched plugin/provider/target/script/test/spec/task files
found no new positive legacy route authority. Remaining hits are classified:

* `__riscv_vlseg2e32_v_i32m1x2` and
  `__riscv_vget_v_i32m1x2_i32m1`: provider-derived RVV segment2 leaf mirrors
  for this typed e32m1 body, not route authority.
* `emission_plan`: existing diagnostic reason in FileCheck fixtures.
* `descriptor`, `source-front-door`, `tcrv_rvv.i32_`: negative FileCheck
  assertions or existing broad target artifact test guardrails.
* `selected route`: diagnostic text requiring facts from the same selected
  route analysis.
* Broad `TargetArtifactExportTest.cpp` and script hits for old intrinsic
  spellings are pre-existing coverage/negative cases outside this module's
  production path.
