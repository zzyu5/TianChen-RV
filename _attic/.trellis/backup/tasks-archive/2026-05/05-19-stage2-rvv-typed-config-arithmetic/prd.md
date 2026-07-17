# Stage2 RVV typed SEW/LMUL config derivation for generic arithmetic

## Goal

Close a bounded Stage2 task for typed SEW/LMUL configuration derivation on
existing generic RVV arithmetic routes. The RVV provider must consume typed
`tcrv_rvv` vector/config/runtime facts to derive route/header/intrinsic
payloads for the existing i32/m1 arithmetic instance and at least one
additional coherent typed configuration, with i32/m2 as the current candidate,
without reviving legacy i32 route authority or cloning route tables.

## Current Facts

- Repo root is `/home/kingdom/phdworks/TianchenRV`; initial `git status --short`
  was clean before task creation.
- Current HEAD at task creation is `ffa6702 rvv: add generic masked add route
  semantics`.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief as
  `.trellis/tasks/05-19-stage2-rvv-typed-config-arithmetic`.
- Specs require the RVV authority chain:
  `tcrv.exec` selected RVV variant -> explicit typed low-level `tcrv_rvv` body
  -> RVV plugin legality/provider -> `TCRVEmitCLowerableRoute` -> neutral
  common EmitC -> target artifact -> `ssh rvv` evidence for runtime claims.
- Stage1 guardrails remain active: no positive `rvv-i32m1`, `RVVI32M1`,
  finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door,
  descriptor, or artifact-name authority may be reintroduced.
- Current code already contains a provider-local selected-body route profile
  surface, explicit generic vector types `!tcrv_rvv.vector<i32, "m1">` and
  `!tcrv_rvv.vector<i32, "m2">`, m2 target fixtures, m2 generated-bundle
  evidence tooling, and archived m2 executable evidence.
- The current task therefore starts as a truth-reconciliation and focused
  validation/repair task, not a request to duplicate old m2 support or add new
  route families.

## Requirements

1. Keep the task scoped to generic arithmetic add/sub/mul selected-body routes.
2. Confirm or repair provider-local typed config derivation so operation kind,
   vector element type, SEW, LMUL, policy, memory form, runtime ABI binding,
   and selected RVV variant facts drive C vector type, mask type where relevant,
   setvl/load/store/header/intrinsic payload selection.
3. Existing i32/m1 arithmetic must remain route-supported and executable only
   as one validated instance of the generic typed route surface.
4. i32/m2 arithmetic must remain or become route-supported through the same
   generic typed selected-body path, with route/header/artifact metadata
   emitted only as mirrors after validation.
5. Unsupported config combinations must fail closed before target artifact
   construction with diagnostics that distinguish parseable/verifier-legal
   typed IR from provider route-supported configs.
6. Common EmitC/export must stay neutral: no RVV dtype, SEW/LMUL, operation,
   schedule, or intrinsic inference belongs in common code.
7. Do not extend reductions, masks, compare/select, broadcast, conversions,
   contraction/FMA, high-level frontends, source-front-door positives,
   dashboards, tuning DBs, or broad matrix coverage in this task.

## Acceptance Criteria

- [x] Current PRD and Trellis context reflect the Hermes brief and current
      repository facts.
- [x] Provider/config implementation either already derives or is repaired to
      derive m1 and m2 arithmetic route/header/intrinsic payloads from typed
      selected-body/config/runtime facts rather than route ids, ABI labels,
      artifact names, or legacy helper names.
- [x] Positive generic arithmetic materialization/artifact coverage exists for
      the additional typed config, expected as m2 add/sub/mul selected bodies.
- [x] Unsupported or stale config/metadata cases fail closed before artifact
      construction.
- [x] Focused local build/tests for touched or validated RVV provider,
      config/dialect, construction, target artifact, and script behavior pass.
- [x] Local generated-bundle dry-run evidence passes for m2 add/sub/mul.
- [x] If executable status is claimed for m2 add/sub/mul in this round, real
      `ssh rvv` correctness evidence passes and the output is recorded.
- [x] Active-authority scan confirms this round introduces no active
      `rvv-i32m1`, `RVVI32M1`, `i32_binary_pre_realized_body`, finite positive
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door, descriptor, or
      common/export RVV semantic authority.
- [x] Task status is truthful; if complete, the task is finished/archived and
      the worktree is clean after one coherent commit, or the exact reason for
      no commit is recorded.

## Technical Approach

Use the current provider-local selected-body route profile as the bounded
resolver surface. First inspect whether the existing m2 path is a real generic
typed route or stale artifact residue. If the current provider already consumes
typed config/body facts correctly, this task will close by validating the
current behavior and recording the evidence. If a gap is found, repair it
inside RVV dialect/config/provider/construction/script code only, then rerun
focused checks.

## Out Of Scope

- No dtype/LMUL clone batch beyond the bounded m2 arithmetic proof.
- No broad Stage2 route family extension.
- No high-level Linalg/Vector/StableHLO frontend lowering.
- No reduction, mask, compare/select, broadcast, conversion, contraction, or
  runtime-boundary expansion unless required to keep existing behavior passing.
- No source-front-door positive route restoration.
- No descriptor-driven computation or Python compiler-core implementation.
- No common EmitC/export RVV semantic branch.
- No performance claim.

## Validation Plan

1. Validate Trellis task context and start the task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused C++ plugin/target tests that cover m1/m2 route profile and
   provider-derived intrinsic selection.
4. Run focused lit/FileCheck for m2 target artifact fixtures and the m2
   generated-bundle dry-run test.
5. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   the script self-test if the script is validated or changed.
6. Run generated-bundle dry-run for `--lmul-m2-selected-body --op-kind add
   --op-kind sub --op-kind mul`.
7. Run real `ssh rvv` m2 add/sub/mul correctness evidence only if executable
   status is claimed in this round.
8. Run `git diff --check` and an active-authority scan over active RVV
   include/lib/test/script paths.
9. Run broader `check-tianchenrv` only if shared provider/export behavior
   changes enough to justify it.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Prior context read:
  `.trellis/tasks/archive/2026-05/05-19-stage1-gate-a-rvv-route-identity-cleanup/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-05-19-stage2-rvv-executable-artifact-closure/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-rvv-lmul-m2-arithmetic-selected-body-executable-route/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-stage2-generic-rvv-reduction-executable-closure/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-masked-add-route-semantics/prd.md`.
- Initial implementation surface:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/RVV/explicit-selected-body-artifact-m2-*.mlir`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-lmul-m2-dry-run.test`.

## Implementation Results

- Confirmed the existing m2 arithmetic support is a real generic typed
  selected-body route: the current positive fixtures use
  `!tcrv_rvv.vector<i32, "m2">` plus SEW32/LMUL m2 `setvl` and `with_vl`
  config, and the route provider emits m2 type/intrinsic/header facts as
  provider-derived mirrors.
- Repaired the provider-local typed config resolver so it cross-checks the
  generic vector and mask element width/LMUL against the selected compile-time
  config before deriving route/profile/intrinsic payloads. This closes a
  bypass where in-memory stale config metadata could otherwise rely on dialect
  verification having already run.
- Added C++ provider coverage that first proves the m2 add route derives
  `__riscv_vsetvl_e32m2`, `__riscv_vle32_v_i32m2`,
  `__riscv_vadd_vv_i32m2`, and `__riscv_vse32_v_i32m2`, then mutates the same
  typed m2 body to stale LMUL m1 config metadata and verifies the provider
  fails closed with a typed-config resolver diagnostic.
- Kept common EmitC/export neutral. The change is limited to RVV provider
  validation and C++ test coverage; no route family, dtype, mask/reduction,
  frontend, source-front-door, or common/export semantic branch was added.

## Validation Results

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-stage2-rvv-typed-config-arithmetic`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] Focused lit/FileCheck:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Target/RVV/explicit-selected-body-artifact-m2-add.mlir ../test/Target/RVV/explicit-selected-body-artifact-m2-sub.mlir ../test/Target/RVV/explicit-selected-body-artifact-m2-mul.mlir ../test/Scripts/rvv-generated-bundle-abi-e2e-lmul-m2-dry-run.test`
  from `build/test`: 4/4 passed.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Local m2 generated-bundle dry-run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --lmul-m2-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-stage2-typed-config-m2-dry --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [OK] Real `ssh rvv` m2 correctness evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --lmul-m2-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260519T-stage2-typed-config-m2-ssh --overwrite --op-kind add --op-kind sub --op-kind mul --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- [OK] `ssh rvv` output:
  `PASS op=add counts=7,16,23`,
  `PASS op=sub counts=7,16,23`,
  `PASS op=mul counts=7,16,23`.
- [OK] Evidence roots:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-stage2-typed-config-m2-dry`
  and
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260519T-stage2-typed-config-m2-ssh`.
- [OK] `git diff --check`
- [OK] Diff active-authority scan over touched provider/test sources found no
  introduced `rvv-i32m1`, `RVVI32M1`, `i32_binary_pre_realized_body`, finite
  positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door,
  descriptor, or common/export RVV semantic authority.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 157/157 passed.

## Spec Update Decision

No `.trellis/spec/` update was needed. Existing specs already require RVV
provider-owned typed body/config authority, fail-closed unsupported route
profiles, neutral common EmitC/export, and real `ssh rvv` evidence for runtime
claims. This task instantiates that contract for one bounded generic arithmetic
typed-config resolver repair.

## Definition Of Done

- [x] Provider-local typed config validation consumes generic vector/mask
      element width and LMUL before route/profile/intrinsic selection.
- [x] m1 arithmetic behavior remains covered by existing selected-body routes.
- [x] m2 add/sub/mul stays route-supported, reaches materialized artifacts, and
      passes real `ssh rvv` correctness evidence for counts `7,16,23`.
- [x] Stale config metadata for an m2 typed body fails closed inside the RVV
      provider.
- [x] No Stage1 legacy route authority or common EmitC/export RVV semantic
      branch was introduced.
