# Stage1 residue hygiene and source-front-door default hardening

## Goal

Harden the remaining RVV Stage1 residue surfaces that could let future work
treat legacy finite `tcrv_rvv.i32_*` / `!tcrv_rvv.i32m*`,
source-seed evidence modes, or source-front-door defaults as RVV route,
artifact, or evidence authority. This round must change real code, tests, and
script behavior while preserving the corrected generic typed RVV Stage2 routes.

## What I Already Know

- Repo root is `/home/kingdom/phdworks/TianchenRV`.
- Initial repository check was clean at `9d0fac0 rvv: add generic macc route skeleton`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief as
  `.trellis/tasks/05-19-stage1-residue-source-front-door-hardening`.
- Specs require RVV positive route authority to flow through:
  selected `tcrv.exec` RVV variant -> explicit typed low-level `tcrv_rvv`
  body -> RVV plugin legality/provider -> provider-built
  `TCRVEmitCLowerableRoute` -> neutral common EmitC/export.
- Archived Stage1 work moved active add/sub/mul authority onto the generic
  typed surface and made the legacy RVV source-front-door pass fail closed.
- Archived Stage2 macc work added a generic `tcrv_rvv.macc` route with dry-run
  and `ssh rvv` evidence, without reintroducing legacy i32 route authority.
- Current residue scan still finds:
  - `SourceFrontDoorPassRegistration` defaults to
    `DefaultArtifactFrontDoorPolicy::Eligible` in
    `include/TianChenRV/Plugin/ExtensionPlugin.h`.
  - `scripts/rvv_generated_bundle_abi_e2e.py` still exposes `--source-seed`
    and `SOURCE_SEED_OP_EXPECTATIONS` as a selectable RVV evidence input mode.
  - legacy `tcrv_rvv.i32_*` / `!tcrv_rvv.i32m*` fixtures remain in dialect,
    conversion, target, and plugin tests. Some are negative/fail-closed
    inventory; any positive parser/verifier fixtures must be demoted or
    classified so they cannot be mistaken for route support.

## Requirements

1. Make source-front-door default artifact eligibility explicit-only or
   disabled by default. A plugin must pass `Eligible` deliberately if it owns a
   future/non-RVV source-front-door path.
2. Keep RVV source-front-door registration explicit-only and fail-closed.
3. Preserve non-RVV later-stage source-front-door tests only through explicit
   plugin opt-in; do not let the common default make them eligible.
4. Remove or hard-fail `--source-seed` / `SOURCE_SEED_OP_EXPECTATIONS` as a
   positive RVV generated-bundle evidence mode. The script may mention legacy
   source-seed only as an unsupported/deprecated diagnostic.
5. Demote remaining positive legacy `tcrv_rvv.i32_*` / `!tcrv_rvv.i32m*`
   parser/verifier fixtures where feasible. If retained, tests must classify
   them as deprecated parseable inventory or fail-closed coverage, not route
   or artifact support.
6. Keep generic typed RVV Stage2 routes working, including explicit selected
   body arithmetic, RHS broadcast, compare/select, reduce, masked add, macc,
   LMUL m2 typed-config, and pre-realized selected-body flows.
7. Do not add Stage2 operation coverage, dtype/LMUL clone batches,
   source-front-door positive RVV routes, frontend generalization,
   performance tuning, dashboards, or spec-only cleanup.

## Acceptance Criteria

- [x] Default `SourceFrontDoorPassRegistration` construction is
      explicit-only/disabled, and tests prove default registrations are not
      artifact-front-door eligible.
- [x] Plugins that intentionally keep non-RVV later-stage source-front-door
      coverage pass `Eligible` explicitly at registration sites.
- [x] RVV registration remains explicit-only and fail-closed.
- [x] `scripts/rvv_generated_bundle_abi_e2e.py --source-seed` is unusable as
      positive RVV evidence and exits with a targeted diagnostic before local
      bundle generation.
- [x] `SOURCE_SEED_OP_EXPECTATIONS` is removed or converted to unsupported
      inventory so no script path selects legacy RVV source seed fixtures for
      artifact proof.
- [x] Legacy RVV dialect/test fixtures using `tcrv_rvv.i32_*` /
      `!tcrv_rvv.i32m*` are either removed, renamed to deprecated inventory,
      or asserted as negative/fail-closed. No positive route/artifact/script
      path depends on them.
- [x] Focused plugin/provider/export/script tests pass for changed behavior.
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passes
      if the script is touched.
- [x] `git diff --check` passes.
- [x] A focused residue scan over `include lib scripts test` classifies every
      remaining high-risk match as negative, fail-closed, deprecated inventory,
      explicit non-RVV opt-in, or intentional provider-derived intrinsic leaf.
- [x] Run `check-tianchenrv` if source-front-door defaults, dialect definitions,
      provider/export behavior, or shared plugin registry behavior changed
      broadly enough to justify the full suite.

## Out Of Scope

- No new RVV Stage2 operation coverage.
- No positive RVV source-front-door/source-artifact routes.
- No high-level Linalg/Vector/StableHLO frontend work.
- No Scalar, IME, Offload, TensorExt, Template/Toy feature expansion beyond
  explicit opt-in needed to preserve existing tests after default hardening.
- No descriptor-driven computation or direct C/source export restoration.
- No runtime/correctness/performance claim unless fresh `ssh rvv` evidence is
  collected for a changed executable path.
- No Python compiler-core implementation.

## Validation Plan

1. Validate Trellis context.
2. Build focused targets for plugin registry, RVV plugin, target export,
   `tcrv-opt`, and `tcrv-translate` as needed.
3. Run focused C++ plugin registry / RVV plugin / Toy or TensorExtLite tests
   affected by source-front-door registration policy.
4. Run focused lit tests for RVV source-front-door fail-closed behavior and
   source-artifact bundle front-door policy behavior.
5. Run focused script dry-run/self-test coverage for
   `scripts/rvv_generated_bundle_abi_e2e.py`, including the unsupported
   `--source-seed` path.
6. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
7. Run `git diff --check`.
8. Run the required focused residue scan and classify remaining matches.
9. Run `cmake --build build --target check-tianchenrv -j2` if shared defaults
   or broad plugin/export behavior changed.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-19-stage1-gate-a-rvv-route-identity-cleanup/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-05-19-stage1-generic-typed-rvv-body-surface/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-multiply-add-route-skeleton/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-source-front-door-route-authority-demotion/prd.md`

Initial implementation surface:

- `include/TianChenRV/Plugin/ExtensionPlugin.h`
- `lib/Plugin/ExtensionPlugin.cpp`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`
- `lib/Plugin/Toy/ToyExtensionPlugin.cpp`
- `lib/Plugin/TensorExtLite/TensorExtLiteExtensionPlugin.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- focused tests under `test/Plugin`, `test/Transforms/RVV`,
  `test/Transforms/SourceFrontDoor`, `test/Target/TargetArtifactBundleExport`,
  `test/Scripts`, and legacy RVV dialect/conversion fixtures as needed.

## Implementation Results

- Changed `SourceFrontDoorPassRegistration` default artifact-front-door policy
  from `Eligible` to `ExplicitOnly`.
- Updated the registry smoke test so default source-front-door registrations
  are not artifact-front-door eligible.
- Kept RVV source-front-door registration explicit-only and fail-closed.
- Made Toy and TensorExtLite source-front-door registrations pass `Eligible`
  explicitly, preserving their later-stage positive source-front-door tests
  without relying on a permissive common default.
- Removed `SOURCE_SEED_OP_EXPECTATIONS` and all source-seed bundle-generation
  branches from `scripts/rvv_generated_bundle_abi_e2e.py`.
- Kept `--source-seed` only as a targeted unsupported CLI diagnostic that exits
  before local bundle generation.
- Added lit coverage for unsupported `--source-seed`.
- Marked finite `tcrv_rvv.i32_*` / `!tcrv_rvv.i32m*` ODS descriptions and
  dialect parser/printer tests as deprecated Stage1 parse-only inventory, not
  route, target artifact, source-front-door, runtime, correctness, or
  performance evidence.

## Validation Results

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-stage1-residue-source-front-door-hardening`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-plugin-registry-test tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-toy-extension-plugin-test tianchenrv-tensorext-lite-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-plugin-registry-test`
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-toy-extension-plugin-test`
- [OK] `build/bin/tianchenrv-tensorext-lite-extension-plugin-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] Focused lit filter covering RVV dataflow inventory,
  source-seed unsupported behavior, RVV source-front-door fail-closed tests,
  source-artifact bundle front-door RVV/Toy/TensorExtLite behavior, and RVV
  selected-body/macc script dry-runs: 13/13 passed.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Unsupported source-seed smoke:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --source-seed ...`
  failed before bundle generation with the targeted unsupported diagnostic.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2`: 162/162 passed.

## Residue Scan Summary

Focused scan commands covered `include lib scripts test` for:

```text
RVVI32M1
rvv-i32m1
i32_binary_pre_realized_body
tcrv_rvv.i32_
!tcrv_rvv.i32m
--source-seed
legacy-rvv-source-seed
DefaultArtifactFrontDoorPolicy::Eligible
source-front-door
__riscv_*_i32m1 / vint32m1_t
```

Classification:

- `DefaultArtifactFrontDoorPolicy::Eligible`: no implicit default remains.
  Remaining uses are the enum/check API plus explicit Toy/TensorExtLite opt-in.
- `--source-seed`: only the unsupported CLI diagnostic and its negative test
  remain. `SOURCE_SEED_OP_EXPECTATIONS` and source-seed bundle-generation
  paths are gone.
- `legacy-rvv-source-seed`: only negative FileCheck `implicit-check-not`
  guards remain in pre-realized script tests.
- `RVVI32M1`, `rvv-i32m1`, and `i32_binary_pre_realized_body`: no active
  include/lib route/provider match; remaining matches are supervisor/prompt
  guardrail text or RVV source-front-door negative fixture paths.
- `tcrv_rvv.i32_*` / `!tcrv_rvv.i32m*`: remaining production code matches are
  RVV dialect parser/verifier diagnostics and provider fail-closed rejection
  of stale legacy selected-body ops. Remaining tests are deprecated parse-only
  inventory, negative fail-closed EmitC/target/plugin tests, script
  `implicit-check-not` guards, or legacy RVV source-front-door negative
  fixtures.
- `source-front-door`: RVV occurrences are fail-closed/explicit-only; Toy and
  TensorExtLite occurrences are explicit non-RVV opt-in examples. Common
  pipeline still filters by `isDefaultArtifactFrontDoorEligible()`.
- `__riscv_*_i32m1` / `vint32m1_t`: remaining active RVV provider/test matches
  are provider-derived intrinsic/type leaves after typed body/config validation,
  or machine-probe/toolchain evidence snippets. They are not route authority.

## Definition Of Done

- The task context and PRD reflect the actual bounded Stage1 hygiene module.
- Code/script/test changes close the identified residue paths.
- Checks listed above pass or any skipped full-suite check is justified.
- The task is finished/archived if complete.
- One coherent commit records the completed round, or the task remains open
  with a precise continuation point.
