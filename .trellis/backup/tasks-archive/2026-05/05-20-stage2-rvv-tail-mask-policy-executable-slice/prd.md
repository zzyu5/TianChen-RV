# Stage2 RVV tail/mask policy executable slice

## Goal

Implement one bounded Stage2 RVV policy-authority slice: a masked i32
SEW32 LMUL m1 unit-stride add path must carry explicit tail policy, mask
policy, mask role, passthrough/inactive-lane source, memory form, operation
kind, dtype/config, runtime `n`/AVL, and ABI roles from the selected
`tcrv.exec` RVV boundary through typed `tcrv_rvv` body structure, RVV
selected-body realization or already-realized validation, RVVEmitCRoutePlanning,
provider/materializer handoff, generated artifact packaging, and real `ssh rvv`
correctness evidence.

This task is about policy authority for one masked add route. It is not a
mask/tail policy matrix, not broad Stage2 coverage, and not a legacy i32 route
compatibility path.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV tail/mask policy executable slice`.
- Module owner: RVV plugin-owned tail/mask policy derivation for one bounded
  masked unit-stride add path.
- Bounded positive proof: one i32 / SEW32 / LMUL m1 / unit-stride masked add
  path with explicit policy and inactive-lane contract.
- Runtime/correctness evidence: real `ssh rvv` PASS for counts such as 7, 16,
  and 23 if executable correctness is claimed.

## Initial Repository Facts

- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `c53e73a4 rvv: add lmul m2 pre-realized route slice`.
- No `.trellis/.current-task` existed at session start, so this task was
  created from the Hermes brief before source edits.
- Recent completed tasks prove the corrected generic typed RVV route surface
  for selected-body realization, provider route planning extraction, explicit
  masked add, compare/select executable ABI, scalar broadcast executable path,
  i64 SEW64 route policy, and LMUL m2 route policy.
- The archived selected-body masked add task already introduced a bounded
  `typed_masked_binary_pre_realized_body` path and real `ssh rvv` evidence for
  `masked_add` under agnostic policy. This round must not merely repeat that
  evidence; it must make tail/mask policy and inactive-lane semantics explicit
  route facts and fail closed when the policy boundary is missing or
  unsupported.
- Stage1 guardrails remain active: do not reintroduce positive `RVVI32M1`,
  `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-seed, descriptor/direct-C/source-export, status,
  route-id, artifact-name, exact intrinsic spelling, or common/export RVV
  semantic authority.

## Requirements

1. Keep the positive compiler behavior bounded to one masked i32 SEW32 LMUL m1
   unit-stride add slice.
2. The selected/pre-realized body or already-typed body/config facts must
   explicitly carry:
   - mask role and mask source;
   - tail policy and mask policy;
   - passthrough/inactive-lane source;
   - memory form and operation kind;
   - element type, SEW, LMUL, and policy config;
   - runtime `n`/AVL binding;
   - lhs/rhs/out/mask/passthrough ABI roles as needed by the chosen structure.
3. The inactive-lane contract for this task is preservation of masked-off
   lanes from an explicit passthrough/output source. The generated runtime
   evidence must make true and false mask lanes non-vacuous.
4. Tail behavior must be tested with counts that are not all exact VL
   multiples. Counts 7, 16, and 23 are the default evidence set.
5. `RVVSelectedBodyRealization`, if changed, may only materialize legal typed
   generic `tcrv_rvv` structure from explicit selected facts. It must not alter
   computation semantics, dtype semantics, parameter roles, dispatch/fallback,
   or runtime `n`/AVL values.
6. `RVVEmitCRoutePlanning` must derive route type, header, intrinsic leaves,
   policy choice, mask/passthrough metadata, and runtime ABI metadata from typed
   body/config/capability/runtime facts.
7. RVV provider/materializer/target artifact paths must consume the provider
   plan. Common EmitC/export may carry provider payloads and mirrors only; it
   must not infer mask/tail policy, inactive-lane behavior, dtype, SEW, LMUL,
   operation kind, schedule, or intrinsic spelling.
8. Unsupported or inconsistent policy combinations must fail closed with
   targeted diagnostics. Missing mask role, missing passthrough source, missing
   runtime AVL, wrong memory form, unsupported policy, mismatched config, and
   incomplete typed body structure must not reach artifact emission.
9. Generated-bundle tooling may be extended only as an external validation
   harness. It must not become compiler route authority.
10. Preserve existing explicit masked add, pre-realized masked add, compare/
    select, scalar broadcast, i64 add, LMUL m2 add, reduction, macc, and Stage1
    fail-closed behavior unless a focused fix is required by this task.

## Acceptance Criteria

- [x] PRD, implement/check context, task metadata, and journal notes describe
      this bounded tail/mask policy slice and do not drift into a policy
      matrix or old i32 compatibility route.
- [x] Positive selected-body or typed-body fixture carries explicit mask role,
      mask source, tail/mask policy, passthrough/inactive-lane source, memory
      form, op kind, dtype/config, runtime `n`/AVL, and ABI roles.
- [x] RVV selected-body realization and/or typed-body verifier consumes those
      explicit policy facts into legal generic `tcrv_rvv` structure before
      route construction.
- [x] RVVEmitCRoutePlanning derives policy/mask/passthrough route metadata,
      intrinsic leaves, runtime ABI, and header artifacts from typed route
      facts. Exact intrinsic spelling appears only as provider-derived output.
- [x] Provider/materializer/target artifact paths consume provider-built route
      payloads and remain neutral outside RVV plugin ownership.
- [x] Generated artifact correctness evidence includes both true and false mask
      lanes, verifies masked-off lane preservation from passthrough/output
      source, and covers tail-sensitive counts such as 7, 16, and 23.
- [x] Negative fail-closed tests cover missing mask role, missing passthrough
      source, unsupported policy, mismatched policy/config, missing AVL/runtime
      roles, wrong memory form, and incomplete typed body structure.
- [x] Real `ssh rvv` generated-bundle correctness evidence passes counts 7,
      16, and 23 if executable correctness is claimed. If executable closure is
      not supportable in this round, the route-supported/fail-closed policy
      boundary is complete and the exact continuation point is recorded.
- [x] Active-authority scan confirms no reintroduced positive `RVVI32M1`,
      `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      source-front-door/source-seed, descriptor/direct-C/source-export, or
      common/export RVV semantic authority.
- [x] Focused build/lit/C++/script checks, `git diff --check`, task
      validation, and final worktree cleanliness checks pass.

## Non-Goals

- No broad tail/mask policy matrix.
- No all-LMUL or all-dtype expansion.
- No floating point, conversion, widening, reduction, matmul, gather/scatter,
  segmented memory, compare/select, broadcast, or contraction side quest.
- No high-level Linalg/Vector/StableHLO frontend lowering.
- No source-front-door positive RVV routes.
- No one-intrinsic wrapper dialect and no dtype-prefixed helper op family.
- No descriptor-driven computation, descriptor-driven C/source export,
  direct-C exporter revival, or compatibility wrapper preserving legacy i32
  route authority.
- No dashboards, report-only inventory, broad smoke matrix, performance claim,
  or global autotuning.
- No Scalar, IME, Offload, TensorExt, Template/Toy, or future-plugin work.

## Validation Plan

1. Validate Trellis context:
   `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-20-stage2-rvv-tail-mask-policy-executable-slice`
2. Build focused compiler/test targets as needed:
   `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
3. Run focused C++ tests for touched RVV config, dialect, plugin, provider,
   materializer, and target artifact behavior.
4. Run focused lit/FileCheck tests for policy-bearing selected-body/typed-body
   realization, route planning/materialization, target artifact export,
   generated-bundle dry-run, and negative fail-closed diagnostics.
5. Run script checks if the generated-bundle harness changes:
   `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
6. Run a local generated-bundle dry-run for the policy slice with counts
   `7,16,23`.
7. Run real `ssh rvv` generated-bundle correctness evidence for counts
   `7,16,23` if executable correctness is claimed.
8. Run active-authority scans over touched active RVV include/lib/script/test
   paths.
9. Run `git diff --check`.
10. Run `check-tianchenrv` if shared compiler behavior changes enough to
    justify the broader gate.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/index.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/index.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/guides/index.md`
- `.trellis/spec/guides/capability-first-design-guide.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Prior context read:

- `.trellis/tasks/archive/2026-05/05-19-05-19-stage2-rvv-selected-body-masked-add-policy/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-cmp-select-executable-abi/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-vector-scalar-broadcast-executable-path/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-non-i32-dtype-sew-route-policy/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-lmul-route-policy-executable-slice/prd.md`

Initial code/test surfaces:

- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing masked add, compare/select, scalar broadcast, i64 add, LMUL m2 add,
  and generated-bundle evidence tests.

## Definition Of Done

- The production/default RVV selected-body or typed-body path carries policy
  authority through plugin-local realization/provider route planning.
- Focused checks pass and any failures are repaired.
- Real `ssh rvv` evidence is recorded if executable correctness is claimed.
- Task status, PRD, and journal notes stay truthful.
- Task is finished/archived if complete.
- One coherent commit records the implementation, validation, task closeout,
  and evidence, or the task remains open with an exact continuation point.

## Completion Evidence

Implemented and validated:

- `RVVSelectedBodyEmitCRouteDescription` now carries provider-derived
  `maskRole` and `inactiveLaneContract` for bounded `masked_add`; non-masked
  routes verify those fields are empty.
- `RVVEmitCRoutePlanning` derives and verifies:
  `predicate-mask-produced-by-compare`,
  `compare-produced-mask-same-vl-scope`,
  `masked-off-lanes-preserve-passthrough-vector`, and
  `passthrough-vector-preserves-inactive-lanes`.
- Target artifact metadata mirrors the explicit policy/mask facts:
  `tcrv_rvv.tail_policy`, `tcrv_rvv.mask_policy`,
  `tcrv_rvv.mask_role`, `tcrv_rvv.mask_source`,
  `tcrv_rvv.inactive_lane_contract`, and
  `tcrv_rvv.masked_passthrough_layout`.
- Generated-bundle harness now checks non-vacuous true/false mask lanes,
  verifies masked-off lane passthrough preservation, records the inactive-lane
  and mask coverage contracts, and uses counts `7,16,23`.
- Provider-level C++ test verifies the masked-add route description, the
  compare/add/merge lowerable route steps, and fail-closed stale
  `maskRole`/`inactiveLaneContract` route descriptions.
- Existing pre-realized negative lit coverage verifies missing/unsupported
  `mask_source`, missing/unsupported passthrough source, unsupported policy,
  wrong op kind, missing runtime `n`, stale route metadata, and incomplete or
  mixed typed body structure.

Checks run:

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-20-stage2-rvv-tail-mask-policy-executable-slice`
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] focused lit filter:
  `rvv-generated-bundle-abi-e2e-(pre-realized-)?masked-add-dry-run|pre-realized-selected-body-artifact-masked-add|explicit-selected-body-artifact-masked-add|rvv-pre-realized-selected-body-negative`
- [OK] generated-bundle dry-run:
  `--pre-realized-selected-body --op-kind masked_add --runtime-count 7 --runtime-count 16 --runtime-count 23`
- [OK] real `ssh rvv` generated-bundle evidence:
  `PASS op=masked_add counts=7,16,23`
- [OK] remote stdout showed non-vacuous lane coverage:
  `n=7 true=2 false=5 passthrough=5`,
  `n=16 true=4 false=12 passthrough=12`,
  `n=23 true=6 false=17 passthrough=17`.
- [OK] `cmake --build build --target check-tianchenrv -j2`
  passed `189/189`.
- [OK] `git diff --check`
- [OK] diff-added active-authority scan found no positive reintroduction of
  `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-seed, descriptor/direct-C/
  source-export, or common/export RVV semantic authority.

No self-repair was needed after the final implementation pass. `clang-format`
was not available in this environment (`clang-format` and versioned
`clang-format-*` were absent); formatting was checked with compile and
`git diff --check`.
