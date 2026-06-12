# Stage2 RVV computed-masked segment2 update unit-load artifact ABI boundary

## Goal

Build and prove one bounded Stage 2 RVV end-to-end artifact/runtime ABI path for the existing `computed_masked_segment2_update_unit_load` selected body. The path must run from a selected `tcrv.exec` RVV variant through RVV plugin-owned realization/validation, route-family provider facts, common EmitC materialization, generated RVV target artifact bundle, and focused `ssh rvv` correctness evidence.

## What I Already Know

* The previous completed task proved `computed_masked_segment2_store_unit_load` with provider ABI/header facts for `cmp_lhs`, `cmp_rhs`, `src0`, `src1`, `dst`, and `n`, focused dry-run/lit checks, and `ssh rvv` evidence.
* This task is the remaining computed-mask segment2 update boundary: compare-produced mask, two unit-load payload sources, `add(field0, field1)`, tuple construction, masked segment2 interleaved store, destination preservation, runtime `n`/AVL, provider route facts, artifact ABI, and runtime correctness.
* The production authority must stay structural: typed `tcrv_rvv` body/config/capability/runtime facts and RVV plugin-owned owners, not route ids, artifact names, manifests, test names, C strings, descriptors, exact intrinsic spellings, or mirror metadata.
* The work is bounded to one selected-body-to-generated-artifact e2e path, not a broad Stage 2 coverage expansion.

## Assumptions

* Existing load/store computed-mask segment2 tasks under `.trellis/tasks/archive/2026-06/` provide reusable contracts and harness patterns.
* Existing fixtures for `computed_masked_segment2_update_unit_load` may already contain part of the path; this round should prove what exists and add only missing production guards, provider facts, target validation, script support, or tests.
* Any runtime/correctness claim must be backed by real `ssh rvv` evidence.

## Requirements

* RVV computed-mask and segment2 selected-body owners must realize or validate the update selected body before route construction.
* The route-provider path must carry structural ABI/header binding facts for `cmp_lhs`, `cmp_rhs`, `src0`, `src1`, `dst`, and `n` through operand-binding facts, generated header/prototype, and target artifact validation.
* Compare mask source, predicate kind, unit-load field payloads, binary add update, tuple construction, masked segment2 interleaved destination store, inactive-lane destination preservation, runtime `n`/AVL, SEW/LMUL/policy, arithmetic kind, and tail/mask policy must survive structurally.
* Fail closed for direct pre-realized route-entry shortcuts and stale/missing ABI binding/header facts, provider mirror, route, ABI order, mask role, segment count, and arithmetic residue.
* Keep common EmitC/export neutral; RVV semantics must stay in RVV plugin-local owners/providers.

## Acceptance Criteria

* [x] Focused evidence shows `computed_masked_segment2_update_unit_load` is realized or validated before route construction by RVV plugin-local owners.
* [x] Focused EmitC/export and target-artifact tests prove ABI order and `abi`/`hdr` markers for `cmp_lhs`, `cmp_rhs`, `src0`, `src1`, `dst`, and `n`.
* [x] Tests prove compare mask source, unit-load field payloads, add arithmetic kind, tuple construction, interleaved masked segment2 destination store, runtime `n`/AVL, SEW/LMUL/policy, inactive-lane destination preservation, and tail policy survive into the generated bundle.
* [x] Real `ssh rvv` compile/run correctness covers representative counts including `0`, `1`, one VL-boundary count, one tail count, and one larger count, with at least two mask/source/destination patterns.
* [x] Negative/fail-closed evidence covers stale or missing route/ABI/header/mask/segment/arithmetic facts relevant to this exact boundary.
* [x] Bounded old-authority scan over touched files classifies any `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, `i32m`, exact `i32m1` intrinsic authority, `source-front-door`, `source-artifact`, `emission_plan`, `descriptor`, or selected-route residues.
* [x] Focused build/test/script commands pass, git status is clean after the final task commit, and one coherent commit records the task.

## Definition Of Done

* PRD and task context truthfully describe the implemented module boundary.
* Focused lit/script checks for the changed path pass after self-repair.
* Runtime evidence is collected on `ssh rvv` when correctness is claimed.
* Trellis task status is finished/archived per project convention.
* One coherent commit is created, or the exact blocker and next continuation point are recorded.

## Out Of Scope

* Adding arithmetic kinds beyond the existing bounded add update.
* Adding reductions, contractions, dtype or LMUL clone batches, indexed or strided memory, high-level frontend authority, per-Linalg lowering, source-front-door positive routes, or Stage 2 broad coverage expansion.
* Moving RVV semantics into common EmitC/export.
* Treating route ids, artifact names, manifests, test names, C strings, descriptors, exact intrinsic spellings, or metadata mirrors as semantic authority.

## Technical Notes

* Required specs to read before source changes: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, `.trellis/spec/lowering-runtime/emitc-route.md`, and `.trellis/spec/testing/mlir-testing-contract.md`.
* Required prior-task context: archived computed-masked segment2 load/store tasks under `.trellis/tasks/archive/2026-06/`.
* Likely module owner files are listed in the Hermes Direction Brief and should be inspected before implementation.

## Continuation Status - 2026-06-02

Status: local implementation, focused local verification, and real `ssh rvv`
runtime evidence are complete. The task is ready to archive and commit.

Completed local work:

* Added exported header/prototype `hdr` participation to the computed-mask
  segment2 update route operand binding for `cmp_lhs`, `cmp_rhs`, `src0`, and
  `src1`, completing the `cmp_lhs,cmp_rhs,src0,src1,dst,n` `abi|hdr`
  provider summary.
* Kept the full operand fact set under target metadata bounds by shortening
  the provider plan label to `rvv-route-operand-binding:cmseg2_update_unit_load.v1`
  instead of dropping structural use tokens.
* Updated target artifact validation, RVV plugin operand-binding tests,
  explicit/pre-realized target artifact fixtures, and generated-bundle dry-run
  checks for the exact provider summary.
* Added C++ fail-closed checks for stale update provider summaries and stale
  candidate mirrors that omit exported header markers.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the bounded
  metadata rule for provider operand-binding summaries.

Checks passed in this continuation:

```bash
rtk git diff --check
rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test
rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2
rtk build/bin/tianchenrv-target-artifact-export-test
rtk build/bin/tianchenrv-rvv-extension-plugin-test
rtk /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter computed-masked-segment2-update .
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --artifact-root artifacts/tmp/computed-mask-segment2-update-dry-run --run-id computed-mask-segment2-update --overwrite --op-kind computed_masked_segment2_update_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/computed-mask-segment2-update-pre-realized-dry-run --run-id pre-realized-computed-mask-segment2-update --overwrite --op-kind computed_masked_segment2_update_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk ssh -o BatchMode=yes -o ConnectTimeout=5 rvv 'echo rvv-probe-ok'
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --artifact-root artifacts/tmp/computed-mask-segment2-update-rvv --run-id computed-mask-segment2-update-rvv --overwrite --op-kind computed_masked_segment2_update_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/computed-mask-segment2-update-pre-realized-rvv --run-id pre-realized-computed-mask-segment2-update-rvv --overwrite --op-kind computed_masked_segment2_update_unit_load --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```
