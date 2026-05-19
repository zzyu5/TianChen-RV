# Stage2 RVV LMUL route-policy executable slice

## Goal

Make one bounded Stage2 LMUL route-policy slice executable on the corrected
generic typed RVV surface: an i32 element type, SEW32, LMUL m2, unit-stride
binary selected-body path must carry explicit typed body/config/capability and
runtime facts through RVV selected-body realization, RVVEmitCRoutePlanning,
RVVEmitCRouteProvider, common EmitC materialization, generated artifact
packaging, and real `ssh rvv` correctness evidence for representative counts.

The purpose is to prove that LMUL is plugin-derived route policy from typed
`tcrv_rvv` facts, not a hardcoded m1 assumption and not residue from helper
names, old `!tcrv_rvv.i32m2` types, route ids, artifact names, ABI strings,
descriptors, source-front-door metadata, exact intrinsic spelling, or common
export logic.

## Direction Source

- Hermes Direction Brief: `Stage2 RVV LMUL route-policy executable slice`.
- Module owner: RVV plugin-owned LMUL route-policy derivation for realized typed
  `tcrv_rvv` bodies.
- Bounded positive proof: one i32 / SEW32 / LMUL m2 / unit-stride binary route.
- Runtime/correctness evidence: real `ssh rvv` PASS for counts such as 7, 16,
  and 23 if executable correctness is claimed.

## Initial Repository Facts

- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `69468183 rvv: add i64 sew64 executable route slice`.
- No `.trellis/.current-task` existed at session start, so this task was
  created from the Hermes brief before source edits.
- The previous completed task proves non-i32 dtype/SEW route policy with an
  i64 / SEW64 / LMUL m1 executable selected-body route. This task must extend
  the same generic typed route policy to LMUL m2 without building a dtype/LMUL
  matrix.
- An older archived 05-19 LMUL m2 executable task used now-legacy
  `!tcrv_rvv.i32m2` and `tcrv_rvv.i32_*` selected-body surfaces. That task is
  historical evidence only. This round must not revive those surfaces as
  positive executable route authority.
- Stage1 policy remains active: do not reintroduce positive `RVVI32M1*`,
  `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
  source-front-door/source-seed, descriptor/direct-C/source-export, status,
  route-id, artifact-name, or common/export RVV semantic authority.

## Requirements

1. Represent LMUL m2 as an explicit typed route fact carried by selected
   `tcrv.exec` RVV boundary, typed `tcrv_rvv` vector/config structure,
   capability/config contract, runtime `n`/AVL binding, and selected-body
   dataflow.
2. Keep the positive path bounded to one i32 SEW32 LMUL m2 unit-stride binary
   instance, preferably add unless repository evidence shows another existing
   unit-stride binary op is the only viable path.
3. RVV selected-body realization, if touched, must only materialize typed
   generic `setvl/with_vl/load/binary/store` structure. It must not alter
   computation semantics, dtype semantics, parameter roles, dispatch/fallback,
   or runtime AVL values.
4. RVVEmitCRoutePlanning must derive vector type, C vector type, header,
   vsetvl/load/store/binary intrinsic leaves, route mirrors, and runtime ABI
   metadata from typed body/config/capability/runtime facts.
5. RVVEmitCRouteProvider and common EmitC/materializer must consume the plan.
   Common EmitC/export may carry provider payloads and mirrors only; it must
   not infer LMUL, SEW, dtype, operation kind, schedule, or intrinsic spelling.
6. Unsupported or inconsistent LMUL/capability facts must fail closed with
   targeted diagnostics. Required negative cases include unsupported LMUL or
   missing capability, mismatched vector/config facts, wrong dtype/SEW coupling,
   wrong policy, missing AVL/runtime roles, and incomplete typed body
   structure.
7. Generated-bundle tooling may be extended as an external validation harness
   only. It must not become compiler route authority.
8. Keep this task out of broad dtype/LMUL expansion, m4/m8/fractional LMUL,
   floating point, conversion/widening, high-level frontend lowering,
   source-front-door positives, one-intrinsic wrapper dialects, reductions,
   matmul, compare/select, broadcast side quests, dashboards, report-only
   inventories, and performance claims.

## Acceptance Criteria

- [x] PRD, implement/check context, task metadata, and journal notes describe
      this bounded LMUL m2 route-policy slice and do not drift into a dtype/LMUL
      matrix or old m2 compatibility route.
- [x] Explicit typed body/config/capability/runtime facts carry element type
      i32, SEW32, LMUL m2, agnostic policy, unit-stride memory form, binary op
      kind, `n`/AVL, and ordered ABI roles.
- [x] Positive selected-body realization and/or explicit selected-body fixtures
      use generic typed `tcrv_rvv` structure, not finite positive
      `tcrv_rvv.i32_*` ops or `!tcrv_rvv.i32m*` types.
- [x] RVVEmitCRoutePlanning derives m2 route type/header/intrinsic leaves and
      route mirrors from typed facts. Exact intrinsic spelling appears only as a
      provider-derived leaf.
- [x] Provider/materializer/target artifact paths consume provider-built route
      payloads and remain neutral outside RVV plugin ownership.
- [x] Positive lit/C++/script evidence proves i32 SEW32 LMUL m2 route planning,
      materialization, target artifact export, and generated-bundle dry-run.
- [x] Negative fail-closed tests cover unsupported/missing LMUL capability,
      mismatched LMUL vector/config facts, wrong dtype/SEW coupling, wrong
      policy, missing AVL/runtime roles, and incomplete typed body structure.
- [x] Real `ssh rvv` generated-bundle correctness evidence passes counts 7, 16,
      and 23 if executable correctness is claimed.
- [x] Active-authority scan confirms no reintroduced positive `RVVI32M1`,
      `rvv-i32m1`, finite `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      source-front-door/source-seed, descriptor/direct-C/source-export, or
      common/export RVV semantic authority.
- [x] Focused build/lit/C++/script checks, `git diff --check`, task validation,
      and final worktree cleanliness checks pass.

## Validation Plan

1. Validate Trellis task context:
   `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-20-stage2-rvv-lmul-route-policy-executable-slice`
2. Build focused targets as needed:
   `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
3. Run focused C++ tests for touched RVV dialect/config/plugin/provider/export
   behavior.
4. Run focused lit/FileCheck tests for selected-body realization,
   route-planning/materialization, target artifact export, generated-bundle
   dry-run, and new negative diagnostics.
5. Run script checks:
   `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
6. Run local generated-bundle dry-run for the LMUL m2 path with counts 7, 16,
   and 23.
7. Run real `ssh rvv` generated-bundle correctness evidence for counts 7, 16,
   and 23 if the executable path is reached.
8. Run an active-authority scan over touched active RVV include/lib/script/test
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
- `.trellis/spec/core-dialect/index.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/capability-model/index.md`
- `.trellis/spec/capability-model/capability-contract.md`
- `.trellis/spec/capability-model/profiles.md`
- `.trellis/spec/variant-pipeline/index.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior context read:

- `.trellis/tasks/archive/2026-05/05-20-stage1-legacy-rvv-i32-residue-cleanup/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-selected-body-realization-extraction/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-route-provider-planning-extraction/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-vector-scalar-broadcast-executable-path/prd.md`
- `.trellis/tasks/archive/2026-05/05-20-stage2-rvv-non-i32-dtype-sew-route-policy/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-lmul-m2-arithmetic-selected-body-executable-route/prd.md`

Likely code/test surfaces to inspect before implementation:

- `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
- `lib/Dialect/RVV/IR/RVVConfigContract.cpp`
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing i32 add, i64 add, scalar broadcast add, compare/select, and
  generated-bundle ssh-evidence tests.

## Completion Evidence

Implemented the bounded positive path as a pre-realized selected-body
`lmul_m2_add` fixture:

- `tcrv_rvv.typed_binary_pre_realized_body` now accepts only the bounded
  SEW32 / LMUL m2 / `op_kind = "add"` / `memory_form = "vector-rhs-load"` case
  outside the existing m1 and i64/m1 slices.
- RVV selected-body realization validates the same bounded tuple and realizes
  it into generic typed `setvl`, `with_vl`, `load`, `binary`, and `store`
  structure using `!tcrv_rvv.vector<i32, "m2">`.
- Existing RVVEmitCRoutePlanning and the provider consume the realized typed
  facts and derive the m2 route plan, config contract, bounded slice, header
  prototype, route mirrors, and intrinsic leaves. No common EmitC/export owner
  was changed to infer RVV semantics.
- `scripts/rvv_generated_bundle_abi_e2e.py` gained a harness-only
  `lmul_m2_add` operation selector for pre-realized selected-body evidence.
  It maps back to provider operation `add` and verifies the realized MLIR
  vector type is `!tcrv_rvv.vector<i32, "m2">`.

Positive evidence:

- `test/Target/RVV/pre-realized-selected-body-artifact-lmul-m2-add.mlir`
  checks pre-realized body consumption, realized generic typed m2 structure,
  emission-plan metadata, route mirrors, and target header export.
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-lmul-m2-add-dry-run.test`
  checks generated-bundle dry-run evidence for counts `7,16,23` and guards
  against descriptor/direct-C/source-export/finite `tcrv_rvv.i32_` authority.
- Real ssh evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260520T-pre-realized-lmul-m2-add-ssh --overwrite --op-kind lmul_m2_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
  passed with `PASS op=lmul_m2_add counts=7,16,23`.

Negative and guardrail evidence:

- The pre-realized negative suite keeps missing operation, unsupported
  operation, unsupported memory form, missing stride roles, wrong ABI role,
  missing runtime n/AVL, stale metadata, incomplete mixed body, and masked
  pre-realized failures covered.
- This round rewrote the old m2/add negative into an m2/sub rejection and added
  a direct SEW32/LMUL m2/add wrong-policy rejection.
- Existing conversion/provider negative suites continue to cover selected-body
  required capability, route profile/policy, missing ABI, incomplete body, and
  vector/config consistency failures.
- Diff-added active-authority scan over active `include`, `lib`, `scripts`,
  and `test` paths found no positive `RVVI32M1`, `rvv-i32m1`, finite
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed,
  descriptor/direct-C/source-export, or common/export RVV semantic authority.
  Broader repository matches remain pre-existing deprecated parse-only
  inventory, fail-closed tests, script guardrails, or provider-derived intrinsic
  leaves.

Checks run:

- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `build/bin/tianchenrv-rvv-dialect-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- Focused lit for the new LMUL m2 positive fixture, dry-run script test,
  pre-realized negative suite, and existing i64/m2 dry-run controls.
- Manual generated-bundle dry-run for `lmul_m2_add` counts `7,16,23`.
- Real `ssh rvv` generated-bundle correctness for `lmul_m2_add` counts
  `7,16,23`.
- `cmake --build build --target check-tianchenrv -j2`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-20-stage2-rvv-lmul-route-policy-executable-slice`
- `git diff --check`

Self-repair:

- The first focused lit rerun after adding the direct m2 wrong-policy negative
  failed because the expected diagnostic text included an extra `binary` word.
  The expected diagnostic was corrected to the verifier's actual message and
  the focused lit rerun passed.

## Definition Of Done

- Production/default RVV selected-body route policy consumes LMUL m2 from typed
  body/config/runtime facts and reaches executable evidence.
- Focused checks pass and any failures are repaired.
- Task status, PRD, and journal notes stay truthful.
- Task is finished/archived if complete.
- One coherent commit records the PRD, implementation, validation, archive, and
  evidence. If unfinished, the task remains open with an exact continuation
  point.
