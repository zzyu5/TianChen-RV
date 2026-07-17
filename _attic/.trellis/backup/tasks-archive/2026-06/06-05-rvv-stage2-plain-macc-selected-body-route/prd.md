# Stage2 RVV plain MAcc selected-body route foundation

## Goal

Close one bounded Stage 2 plain multiply-accumulate route-supported proof:
a selected `tcrv.exec` RVV variant with an explicit typed `tcrv_rvv.macc`
body must carry `lhs`, `rhs`, `acc`, `out`, and runtime `n` roles through RVV
plugin-owned selected-body realization or validation, provider legality and
route planning, `TCRVEmitCLowerableRoute`, Common EmitC/export, target artifact
validation, and generated-bundle evidence. Real `ssh rvv` is required only if
this round changes route emission, generated C/C++ runtime ABI behavior, or
claims new executable correctness.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV plain MAcc selected-body route foundation`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `94841611 rvv: close computed-mask standalone reduce add`.
- No `.trellis/.current-task` existed, so this task was created from the
  Direction Brief before source edits.
- The immediately previous archived task closed computed-mask standalone
  reduce-add evidence. The next owner should not be another reduction-only
  round.
- Prior archived MAcc tasks already built the selected-body realization,
  explicit and pre-realized plain `macc_add` generated artifact ABI path, and
  provider-owned unit-stride MAcc target validation contract.

## Current Repository Evidence

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` defines generic
  `tcrv_rvv.macc`, not a dtype-prefixed helper op. It carries structural
  `kind`, accumulator layout, result layout, typed vector operands, and VL.
- `lib/Plugin/RVV/RVVMAccSelectedBodyRealizationOwner.cpp` consumes
  `typed_macc_pre_realized_body` into realized `setvl` / `with_vl` /
  lhs load / rhs load or splat / accumulator load / `tcrv_rvv.macc` / store.
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` already carries
  `maccArithmeticKind`, accumulator/result layout, SEW, LMUL, policy,
  runtime-control, headers, C type mapping, and ABI parameters in the plain
  MAcc route-family plan.
- `test/Target/TargetArtifactExportTest.cpp` already consumes the
  provider-owned MAcc validation contract and has stale runtime AVL/VL,
  ABI/order, provider mirror, binding, header/type, accumulator role, and
  stale cross-family negative checks.
- `tcrv-opt` output for both explicit and pre-realized plain MAcc fixtures
  already emits `{key = "tcrv_rvv.macc_arithmetic_kind", value = "add"}`.
  The fixture FileCheck assertions did not pin that emission-plan mirror yet.
- Generated bundle index/header metadata already carries MAcc-local layout and
  arithmetic mirrors, but the script's plain MAcc provider-route summary did
  not extract `tcrv_rvv.macc_arithmetic_kind`.
- `scripts/rvv_generated_bundle_abi_e2e.py` already emits a
  `multiply_accumulate_boundary`, but its `provider_route_facts` summary omits
  the explicit `tcrv_rvv.macc_arithmetic_kind` mirror even though route metadata
  contains it.
- `.trellis/spec/testing/mlir-testing-contract.md` has computed-mask MAcc and
  widening-MAcc generated-bundle evidence sections, but no plain MAcc section.

## Requirements

- Keep scope to one plain vector-vector `macc_add` route.
- Preserve the authority chain:
  selected typed `tcrv_rvv.macc` body/config/runtime facts -> RVV provider
  route facts -> `TCRVEmitCLowerableRoute` -> Common EmitC/export -> target
  artifact mirrors -> generated-bundle evidence.
- The route/evidence contract must explicitly include:
  - operand roles and ABI order `lhs,rhs,acc,out,n`;
  - arithmetic kind `add`;
  - accumulator layout and result layout;
  - source/result dtype, SEW, LMUL, tail policy, and mask policy;
  - runtime AVL/VL plan and runtime `n` binding;
  - route operand binding plan/summary with `abi` and `hdr` markers;
  - required headers, C type mapping, target leaf profile, and
    `provider_supported_mirror`.
- Target artifacts and generated-bundle evidence may mirror provider facts
  only after route construction. They must not infer MAcc semantics from route
  IDs, artifact names, ABI strings, test names, exact intrinsic spelling,
  descriptors, common EmitC, or legacy i32 helpers.
- Unsupported or stale arithmetic kind, layout, runtime ABI/order, dtype/config
  mirrors, header/type mirrors, provider support mirrors, or cross-family MAcc
  facts must remain fail-closed through existing provider/target validation.

## Acceptance Criteria

- [x] Explicit and pre-realized plain MAcc target fixtures pin
      `tcrv_rvv.macc_arithmetic_kind = add` in emission-plan output.
- [x] Generated-bundle plain MAcc evidence exposes
      `macc_arithmetic_kind = add` under provider route facts for both explicit
      and pre-realized selected-body modes, backed by bundle metadata.
- [x] Testing spec defines plain MAcc generated-bundle evidence requirements,
      including typed body authority, ABI order, arithmetic kind,
      accumulator/result layout, runtime AVL/VL, route operand binding,
      headers/types, provider-supported mirror, harness oracle, and
      fail-closed evidence expectations.
- [x] Focused tests for explicit and pre-realized plain MAcc target fixtures
      and generated-bundle dry-runs pass.
- [x] `test/Target/TargetArtifactExportTest.cpp` remains green, proving the
      existing provider-owned MAcc validation contract still rejects stale
      facts before artifact acceptance.
- [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive legacy `i32m1`, descriptor, source-front-door,
      route-string/artifact-name/ABI-string/test-name, intrinsic-spelling, or
      Common EmitC RVV semantic authority.
- [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are recorded.
- [x] The Trellis task is finished/archived and one coherent commit records the
      completed round.

## Out Of Scope

- High-level matmul/Linalg/Vector/StableHLO frontend work.
- Scalar-broadcast MAcc, computed-mask MAcc, runtime-scalar computed-mask MAcc,
  widening MAcc, widening dot reduction, broad dtype/LMUL matrix expansion, or
  new contraction batches.
- One-intrinsic wrapper routes or dtype-prefixed op families.
- Common EmitC choice of MAcc semantics.
- Performance benchmarks, autotuning, Gearbox expansion, source-front-door
  positive routes, or metadata-only route authority.

## Evidence Plan

- Run focused `tcrv-opt` / `tcrv-translate` FileCheck coverage for:
  - `test/Target/RVV/explicit-selected-body-artifact-macc-add.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-macc-add.mlir`
- Run generated-bundle dry-run tests for:
  - `test/Scripts/rvv-generated-bundle-abi-e2e-macc-add-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-macc-add-dry-run.test`
- Build and run `tianchenrv-target-artifact-export-test`.
- Run `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the
  script evidence shape changes.
- Run bounded authority scans and diff whitespace checks.
- Do not run `ssh rvv` unless this round changes route emission/runtime ABI or
  makes a fresh executable correctness claim. Existing archived plain MAcc
  evidence remains the runtime evidence for unchanged executable behavior.

## Completion Evidence

Completed as a route-contract evidence closure for the existing production
plain `macc_add` selected-body route. No C++ provider or target route semantics
changed.

Changes:

- Added explicit emission-plan FileCheck coverage for
  `tcrv_rvv.macc_arithmetic_kind = add` in both explicit and pre-realized
  plain MAcc target fixtures.
- Added `tcrv_rvv.macc_arithmetic_kind` to the plain MAcc generated-bundle
  metadata extraction path and exposed it under `provider_route_facts` as
  `macc_arithmetic_kind = add`.
- Added dry-run FileCheck coverage for explicit and pre-realized generated
  bundles to require the provider-route arithmetic kind.
- Added a `Plain MAcc Generated-Bundle Evidence` testing-spec section covering
  typed body authority, ABI order, arithmetic kind, accumulator/result layout,
  runtime AVL/VL, route operand binding, headers/types, provider-supported
  mirror, harness oracle, and fail-closed evidence expectations.

Self-repair:

- Initial lit run from the source root failed because the source-side
  `lit.cfg.py` lacks build-only `tianchenrv_obj_root`. Re-ran from
  `build/test` through `lit.site.cfg.py`.
- Initial direct header FileCheck assertions expected
  `tcrv_rvv.macc_arithmetic_kind` in `--tcrv-export-target-header-artifact`
  output. Live evidence showed the generated bundle header/index carries the
  MAcc-local arithmetic/layout mirrors, while the direct header fixture remains
  a narrower generic header check. The task was corrected to pin arithmetic
  kind in emission-plan and generated-bundle provider facts.
- Initial script extraction produced `macc_arithmetic_kind: null` because
  `expected_metadata_for(macc_add)` did not include the field. Added it for
  plain and scalar-broadcast MAcc metadata expectations so plain MAcc evidence
  reads the existing bundle metadata.

Checks run:

- `rtk cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- `rtk python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `rtk python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'Target/RVV/(explicit-selected-body-artifact-macc-add|pre-realized-selected-body-artifact-macc-add)|Scripts/rvv-generated-bundle-abi-e2e-(macc-add|pre-realized-macc-add)-dry-run'` from `build/test`
- `rtk cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- `rtk build/bin/tianchenrv-target-artifact-export-test`
- `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-05-rvv-stage2-plain-macc-selected-body-route`
- `rtk git diff --check`
- `rtk git diff --cached --check`
- Bounded added-line old-authority scan over touched files. Matches were
  limited to negative guardrail text in the new testing-spec section, not
  positive route authority.

Runtime evidence:

- `ssh rvv` was not run. This round did not change route emission, generated
  C/C++ runtime ABI behavior, accumulator runtime behavior, correctness, or
  performance claims. Existing archived plain MAcc executable evidence remains
  applicable for unchanged runtime behavior.

## Technical Notes

Specs and guides read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/index.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/guides/index.md`
- `.trellis/spec/guides/capability-first-design-guide.md`
- `.trellis/spec/guides/plugin-locality-review-guide.md`
- `.trellis/spec/guides/compute-boundary-review-guide.md`

Relevant archive context read:

- `.trellis/tasks/archive/2026-06/06-05-rvv-stage2-computed-mask-standalone-reduce-add-route/`
- `.trellis/tasks/archive/2026-06/06-02-06-02-stage2-rvv-plain-macc-add-vector-vector-artifact-abi-boundary/prd.md`
- `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-unit-stride-macc-production-validation-boundary/prd.md`
