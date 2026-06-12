# Stage2 RVV reduce-add production executable slice

## Goal

Redirect the reduce-add work away from the metadata-only closeout at
`d13843f8` and make one production RVV reduce-add slice more structurally real.
The bounded module is one signed i32 / SEW32 / LMUL m1 / unit-stride
`reduce_add` selected-body path whose typed `tcrv_rvv` reduction body carries
the accumulator layout and result layout facts consumed by the RVV route
provider.

Current HEAD already contains a pre-realized reduce-add body, selected-body
realization, route planning, generated-bundle support, and prior archived
`ssh rvv` evidence. This task must not re-close by paperwork. The production
gap found in current code is that `tcrv_rvv.typed_reduce_pre_realized_body`
carries `accumulator_layout` and `result_layout`, but the realized
`tcrv_rvv.reduce` op only carries `kind = "add"`; provider code then fills
bounded reduction layout constants from the operation kind. This round makes
those layout facts structural in the typed body consumed by route planning.

## Direction Source

- Hermes Direction Brief: `Redirect: Stage2 RVV reduce-add executable implementation`.
- Module owner: RVV plugin-owned reduction and accumulator-layout path for one
  bounded i32 unit-stride reduce-add.
- Reason for redirect: latest commit `d13843f8` changed only Trellis archive
  files and the Codex journal; it did not change RVV dialect, plugin, route
  planning, materializer, harness, or target artifact code/tests.

## Current Repository Facts

- Repo root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `d13843f8 rvv: close reduce add executable slice`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
- `git show --stat d13843f8` confirms that the previous closeout was
  metadata-only: archived Trellis task files plus
  `.trellis/workspace/codex/journal-11.md`.
- Direct current code inspection confirms active reduce-add surfaces exist:
  `tcrv_rvv.typed_reduce_pre_realized_body`, `tcrv_rvv.reduce`,
  `RVVSelectedBodyRealization`, `RVVEmitCRoutePlanning`,
  `RVVEmitCRouteProvider`, target artifact tests, and
  `scripts/rvv_generated_bundle_abi_e2e.py --op-kind reduce_add`.
- Direct code inspection also confirms the production gap: layout facts are
  explicit on the pre-realized op, but are not structural on the realized
  `tcrv_rvv.reduce` op that route planning consumes.

## Requirements

1. Keep the scope to one signed i32 / SEW32 / LMUL m1 / unit-stride
   `reduce_add` selected-body route.
2. `tcrv.exec` remains an execution envelope and ABI/runtime boundary only; it
   must not own reduction compute, dtype/config, accumulator layout, result
   layout, or intrinsic spelling.
3. The realized typed `tcrv_rvv.reduce` body must structurally carry:
   reduction kind, accumulator layout, and result layout. Dtype/config/policy
   remain structural in typed vector values and enclosing `setvl/with_vl`.
4. RVV selected-body realization must consume the pre-realized reduce
   accumulator/result layout facts into the realized `tcrv_rvv.reduce` op
   before route construction.
5. RVV dialect verification and RVV route planning must fail closed with
   targeted diagnostics for missing or unsupported reduce accumulator/result
   layout facts.
6. RVVEmitCRoutePlanning/provider must derive reduction metadata mirrors and
   intrinsic/header route facts from the typed body/config/runtime facts,
   including the new structural reduce layout attrs.
7. Existing explicit and pre-realized reduce-add target artifact paths and
   generated-bundle dry-run/runtime harness behavior must remain valid.
8. Preserve Stage 1/2 guardrails: do not reintroduce positive `RVVI32M1`,
   `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
   source-front-door/source-seed, descriptor/direct-C/source-export, route-id
   authority, artifact-name authority, or common/export RVV semantic authority.

## Acceptance Criteria

- [ ] `tcrv_rvv.reduce` admits explicit accumulator/result layout attrs for
      the bounded reduce-add route and rejects missing/unsupported layout facts.
- [ ] RVV selected-body realization materializes pre-realized reduce-add into
      `setvl`, `with_vl`, input load, accumulator load,
      `tcrv_rvv.reduce {kind = "add", accumulator_layout = ..., result_layout = ...}`,
      and store.
- [ ] RVV route planning reads the layout facts from `tcrv_rvv.reduce` and
      uses them to populate provider-derived reduction metadata mirrors.
- [ ] Positive explicit/pre-realized reduce-add tests prove the structural
      layout attrs reach materialized route metadata and target artifact output.
- [ ] Negative tests cover missing accumulator layout, missing result layout,
      unsupported accumulator layout, unsupported result layout, and the
      existing unsupported kind / wrong memory form / missing AVL / wrong ABI
      role failures.
- [ ] Generated-bundle dry-run for
      `--pre-realized-selected-body --op-kind reduce_add` passes for counts
      `7,16,23`.
- [ ] Real `ssh rvv` correctness evidence passes for counts `7,16,23` if this
      round claims executable correctness.
- [ ] Focused build/lit/unit/script checks for touched RVV dialect,
      selected-body realization, route planning/provider, target artifact, and
      generated-bundle paths pass.
- [ ] Active-authority scan confirms no positive legacy/source/descriptor or
      common/export RVV semantic authority is reintroduced.
- [ ] `git diff --check`, Trellis task validation, final status update, and a
      coherent production-code commit are completed if the task is finished.

## Non-Goals

- No broad reduction matrix, floating-point reductions, min/max/and/or batches,
  segmented reductions, contraction/matmul implementation, high-level frontend
  lowering, source-front-door positive route, one-intrinsic wrapper dialect,
  dtype/LMUL clone batch, dashboard, global tuning database, or performance
  claim.
- No Python implementation of compiler core, dialects, passes, plugin
  registry, capability model, lowering, or emission.
- No metadata-only closeout.

## Validation Plan

1. Validate Trellis task context.
2. Build focused targets:
   `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-dialect-test`,
   `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused lit/FileCheck tests for generic RVV dataflow, reduce-add route
   materialization, explicit/pre-realized target artifact output, reduce-add
   negative diagnostics, and generated-bundle dry-run.
4. Run focused C++ tests for touched route/provider/export helpers where lit
   is insufficient.
5. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` if the script
   changes or if generated-bundle evidence is claimed.
6. Run generated-bundle dry-run for pre-realized reduce-add counts `7,16,23`.
7. Run real `ssh rvv` correctness for pre-realized reduce-add counts
   `7,16,23` if executable correctness is claimed.
8. Run active-authority scans over active RVV include/lib/test/script paths.
9. Run `git diff --check`.
10. Run `check-tianchenrv` if focused validation or shared source changes
    justify the broader gate.

## Technical Notes

Relevant specs read:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Prior task context read:

- `.trellis/tasks/archive/2026-05/05-20-05-20-stage2-rvv-reduce-add-executable-slice/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-05-19-stage2-generic-rvv-reduction-accumulation-route/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-generic-rvv-reduction-executable-closure/prd.md`
- `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-selected-reduce-add-realization/prd.md`
- Archived Stage2 sibling task PRDs for widening conversion, tail/mask policy,
  LMUL, non-i32 dtype/SEW, scalar broadcast, and compare/select were checked
  for validation style and guardrails.

Initial code/test surfaces inspected:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- Existing reduce-add tests under `test/Dialect/RVV`, `test/Conversion/EmitC`,
  `test/Transforms/LoweringBoundary`, `test/Target/RVV`, and `test/Scripts`.

## Definition Of Done

- One bounded reduce-add production path now carries accumulator/result layout
  facts structurally through realized typed `tcrv_rvv.reduce` into RVV route
  planning/provider metadata.
- Focused positive, negative, generated-bundle, and hardware evidence are
  current to this task when claimed.
- No legacy i32/source/descriptor/common-export authority is reintroduced.
- The task is finished/archived only if production code/tests/evidence are
  present in this round; otherwise it remains open with an exact continuation
  point.

## Implementation Results

- Added optional structural `accumulator_layout` and `result_layout` attrs to
  generic `tcrv_rvv.reduce`.
- Tightened `ReduceOp::verify()` so the bounded Stage2 reduce-add route fails
  closed when either layout fact is missing or unsupported.
- Updated RVV selected-body realization so
  `tcrv_rvv.typed_reduce_pre_realized_body` consumes its explicit
  accumulator/result layout facts into the realized `tcrv_rvv.reduce` op before
  provider route construction.
- Updated RVVEmitCRoutePlanning so the provider validates the realized
  `tcrv_rvv.reduce` layout attrs and derives reduction metadata mirrors from
  those typed body facts, rather than filling them only from the operation
  kind.
- Updated the generated-bundle evidence script to require the materialized
  selected-body MLIR for `reduce_add` to contain the structural reduce layout
  attrs.
- Updated positive explicit/pre-realized reduce-add fixtures and target export
  C++ fixture generation to include the new typed body layout facts.
- Added dialect negative coverage for missing accumulator layout, missing
  result layout, unsupported accumulator layout, and unsupported result layout.

## Validation Results

- [OK] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-20-stage2-rvv-reduce-add-production-executable-slice`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test tianchenrv-construction-protocol-common-test tianchenrv-target-artifact-export-test -j2`
- [OK] `build/bin/tianchenrv-rvv-dialect-test`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] `build/bin/tianchenrv-construction-protocol-common-test`
- [OK] `build/bin/tianchenrv-target-artifact-export-test`
- [OK] Focused lit from `build/test`: 6/6 passed for
  `generic-stage2-dataflow`, reduce-add materialization, reduce-add negative
  route coverage, explicit/pre-realized reduce-add target artifacts, and
  pre-realized generated-bundle dry-run.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] Generated-bundle dry-run:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260520T-typed-reduce-layout-dry --overwrite --op-kind reduce_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj`
- [OK] Dry-run evidence root:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260520T-typed-reduce-layout-dry`
- [OK] Real `ssh rvv` generated-bundle correctness:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e --run-id 20260520T-typed-reduce-layout-ssh --overwrite --op-kind reduce_add --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --timeout 180 --connect-timeout 10`
- [OK] `ssh rvv` output:
  `PASS op=reduce_add counts=7,16,23`
- [OK] Runtime evidence root:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260520T-typed-reduce-layout-ssh`
- [OK] `git diff --check`
- [OK] Diff-only active-authority scan over active RVV include/lib/script/test
  paths: no newly added positive `RVVI32M1`, `rvv-i32m1`, finite positive
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door/source-seed,
  descriptor/direct-C/source-export, route-id, artifact-name, exact intrinsic
  spelling authority, or common/export RVV semantic authority.
- [OK] Full active-authority scan classified remaining matches as pre-existing
  deprecated parse-only inventory, negative fail-closed fixtures, source
  front-door rejection code, script guardrails, supervisor prompt guardrails,
  capability descriptor tests, emission-plan mirrors, or provider-derived
  intrinsic leaves after typed-body validation.
- [OK] `cmake --build build --target check-tianchenrv -j2`: 193/193 passed.

## Spec Update Decision

No `.trellis/spec/` update was needed. The existing specs already require
Stage2 RVV reduction/accumulator/result facts to be structural in typed or
realized `tcrv_rvv` body structure, require selected-body realization to
consume code-affecting facts before route construction, and require common
EmitC/export neutrality. This round implements that existing rule in the
concrete `tcrv_rvv.reduce` route surface rather than adding a new durable
architecture rule.

## Finish Status

- Production owner code changed in RVV dialect ODS/verifier, selected-body
  realization, RVV EmitC route planning/provider consumption, generated-bundle
  evidence tooling, and focused tests.
- Route-supported evidence and executable `ssh rvv` evidence are both current
  to this task.
- Task may be finished and archived after journal/status update and commit.
