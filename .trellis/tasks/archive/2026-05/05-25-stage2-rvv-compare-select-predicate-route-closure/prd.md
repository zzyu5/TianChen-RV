# Stage2 RVV compare/select predicate route closure

## Goal

Close one bounded Stage 2 compare/select predicate route boundary for the
existing pre-realized plain `cmp_select` RVV selected-body path. The round must
prove that compare predicate kind, predicate source/result, selected true/false
value operands, dtype/config, policy, and runtime VL are carried by the typed
`tcrv_rvv` body/config, validated and planned by RVV plugin-owned route facts
and statement plans, emitted as the expected RVV compare/select intrinsic
forms, mirrored by generated artifacts as mirrors only, and backed by one
focused generated-bundle evidence path. Missing, stale, or metadata-derived
predicate/value authority must fail closed before route or artifact authority.

## Direction Source

- Direction title: `Stage2 RVV compare/select predicate route closure`.
- Module owner: RVV plugin-owned compare/select predicate and value-selection
  route boundary from typed `tcrv_rvv` body/config into route facts, provider
  emission, generated artifact mirrors, and one focused `ssh rvv` correctness
  path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `f1f876e9 rvv: prove mask tail policy route boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- `.trellis/spec/index.md` requires the RVV authority chain to run through a
  selected `tcrv.exec` RVV variant, typed/realized `tcrv_rvv` body, RVV
  plugin-owned legality / selected-body realization / route provider, common
  EmitC, target artifact, and `ssh rvv` evidence when runtime correctness is
  claimed.
- `.trellis/spec/extension-plugins/rvv-plugin.md` defines compare/select as
  RVV-owned typed body structure. The selected-body realization boundary must
  consume pre-realized compare/select bodies into `setvl`, `with_vl`, `load`,
  `compare`, `select`, and `store` structure before provider route
  construction.
- The same RVV spec defines the elementwise/select operand-binding facts
  boundary and compare/select statement-plan boundary. Provider code must
  consume those RVV-owned facts/plans and must not reconstruct compare/select
  semantics from operation names, ABI strings, route ids, artifact metadata, or
  fixture names.
- `.trellis/spec/lowering-runtime/emitc-route.md` keeps common EmitC neutral:
  it must materialize provider-built `TCRVEmitCLowerableRoute` payloads without
  choosing RVV intrinsics, dtype, predicate, policy, schedule, or ABI role
  semantics.
- The previous archived mask/tail task shows the expected evidence shape:
  generated-bundle evidence should include an explicit boundary summary tying
  typed body/config facts, provider/export mirrors, emitted RVV C++ intrinsic
  operands, runtime counts, and harness checks.
- Current code already has route planning surfaces for plain compare/select:
  `RVVSelectedBodyPlainCompareSelectRouteFamilyPlan`,
  `RVVSelectedBodyElementwiseSelectRouteOperandBindingFacts`, and
  `RVVSelectedBodyCompareSelectRouteStatementPlan`.
- The existing `pre-realized-selected-body-artifact-cmp-select.mlir` fixture
  carries `predicate_kind = "eq"`, `mask_source =
  "compare-produced-mask-same-vl-scope"`, `select_layout =
  "select-lhs-when-mask-else-rhs"`, `sew = 32`, `lmul = "m1"`, and
  `tail/mask = agnostic`.
- A dry-run probe showed the generated C++ already emits
  `__riscv_vmseq_vv_i32m1_b32` and `__riscv_vmerge_vvm_i32m1`, but
  `scripts/rvv_generated_bundle_abi_e2e.py` currently leaves
  `emitted_rvv_cpp_checks.intrinsics`, `vector_c_type`, and emitted
  `runtime_avl_vl_boundary` empty for `cmp_select`. Evidence only records a
  harness predicate coverage contract, not an explicit compare/select route
  boundary.

## Requirements

1. Keep the executable/evidence closure bounded to the existing pre-realized
   plain `cmp_select` path unless repository evidence proves that path unsafe
   or stale.
2. The selected input must visibly carry compare predicate kind, predicate
   mask source, select layout, true/false value operand roles, element dtype,
   SEW, LMUL, policy, and runtime count/VL facts in typed `tcrv_rvv`
   structure.
3. RVV selected-body realization must consume those typed facts into realized
   `setvl`, `with_vl`, `load`, `compare`, `select`, and `store` structure.
4. RVV route-family facts, elementwise/select operand-binding facts, and the
   compare/select statement plan must carry the predicate kind/source,
   selected true/false operands, output, dtype/config, policy, and runtime VL
   before provider route construction.
5. Generated RVV C/C++ evidence must prove emitted compare and select
   intrinsic forms and operands derive from the typed body/config/runtime
   facts: `lhs` and `rhs` loads feed the compare mask, the compare mask feeds
   `vmerge`, select true/false operands match `select-lhs-when-mask-else-rhs`,
   and every relevant intrinsic uses the loop VL derived from runtime `n`.
6. Generated artifact metadata may mirror compare predicate kind, mask source,
   select layout, provider support, operand binding, dtype/config, and runtime
   VL facts only after provider route construction. Mirror fields must not be
   used as authority.
7. Generated-bundle evidence must include an explicit
   `compare_select_predicate_boundary` summary or equivalent tying typed
   materialized body facts, emitted RVV C++ operands, route metadata mirrors,
   artifact ABI, runtime counts, and harness predicate true/false coverage.
8. Unsupported compare kinds, missing compare/select operands, stale
   operand-binding or statement-plan dependencies, dtype/config mismatches, or
   metadata/name-derived authority must fail closed before common EmitC or
   target artifact authority.
9. Do not add broad operation-family coverage, reductions, contractions,
   conversion batches, high-level frontend lowering, source-front-door positive
   routes, legacy i32 authority, one-op-per-intrinsic wrapper growth,
   dtype/LMUL clone batches, dashboards, or broad smoke matrices.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` reference the relevant RVV
      plugin, EmitC route, testing policy, and previous archived task.
- [x] Focused lit/FileCheck coverage for pre-realized `cmp_select` proves the
      realized body consumes the pre-realized predicate/select facts into
      `tcrv_rvv.compare` and `tcrv_rvv.select`, and target artifact mirrors
      expose compare predicate, mask source/role/form, select layout,
      operand-binding, provider-supported mirror, and runtime VL facts.
- [x] Focused provider/C++ coverage proves plain compare/select route-family
      facts, typed config facts, elementwise/select operand-binding facts, and
      compare/select statement-plan facts carry predicate kind, true/false
      operands, output, policy/config, and runtime VL before route
      construction.
- [x] `scripts/rvv_generated_bundle_abi_e2e.py` extracts emitted RVV C++ facts
      for pre-realized `cmp_select`, including vector/mask C types, setvl,
      load, compare, select, store intrinsics, runtime loop VL, predicate
      variable, compare operands, select true/false operands, and store result.
- [x] Focused generated-bundle evidence for pre-realized `cmp_select` records
      a `compare_select_predicate_boundary` or equivalent summary tying typed
      body/config, provider/export mirrors, emitted compare/select intrinsic
      operands, runtime counts, and predicate true/false harness coverage.
- [x] One focused dry-run generated-bundle script test checks the new evidence
      summary and representative compare/select mirror metadata.
- [x] One real `ssh rvv` generated-bundle ABI/e2e run is attempted for
      pre-realized `cmp_select` across runtime counts `7,16,23`; if blocked,
      record the exact blocker and do not claim runtime correctness.
- [x] Existing fail-closed tests still reject unsupported/stale
      compare/select predicate, missing operands, bad operand bindings, and
      stale statement-plan dependencies before route/artifact authority.
- [x] Bounded scan over touched RVV dialect/realization/planning/provider/
      target/script/fixture files finds no new metadata/name-derived
      compare/select authority and no positive legacy i32 route authority.
- [x] `git diff --check` passes.
- [x] Focused checks and `check-tianchenrv` pass, or an exact blocker is
      documented.
- [x] Trellis task status, journal, archive, and commit are truthful.

## Non-Goals

- No new compare/select sub-family coverage beyond the chosen plain
  pre-realized `cmp_select` closure.
- No Stage 2 reduction, contraction, conversion, Linalg/Vector/StableHLO
  frontend, descriptor, direct-C, source-export, or source-front-door positive
  route work.
- No legacy `RVVI32M1*`, `rvv-i32m1-*`, `tcrv_rvv.i32_*`, or
  `!tcrv_rvv.i32m*` executable authority.
- No performance claim.
- No multi-agent or sub-agent workflow in this round.

## Technical Approach

1. Start the Trellis task and validate context.
2. Tighten focused `cmp_select` lit/FileCheck coverage so header/artifact
   mirrors explicitly include predicate, mask source/role/form, select layout,
   provider-supported mirror, operand binding, dtype/config, and runtime VL
   facts.
3. Extend generated-bundle evidence extraction for `cmp_select` so emitted RVV
   C++ compare/select intrinsics and operands are parsed into a
   compare/select-specific boundary summary.
4. Add or tighten focused script dry-run FileCheck coverage for the new
   evidence summary.
5. Reuse the existing generated-bundle command shape:

   ```bash
   python3 scripts/rvv_generated_bundle_abi_e2e.py \
     --pre-realized-selected-body \
     --artifact-root artifacts/tmp/stage2_cmp_select_predicate_route_closure \
     --run-id pre-realized-cmp-select-ssh-rvv \
     --overwrite \
     --op-kind cmp_select \
     --runtime-count 7 \
     --runtime-count 16 \
     --runtime-count 23 \
     --tcrv-opt build/bin/tcrv-opt \
     --tcrv-translate build/bin/tcrv-translate \
     --ssh-target rvv
   ```

6. Keep validation focused on changed behavior, then run full
   `check-tianchenrv` if feasible.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-compare-select-predicate-route-closure`
2. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
3. `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
4. Focused dry-run for pre-realized `cmp_select` with counts `7,16,23`.
5. Focused lit/FileCheck tests covering the changed RVV target and script
   behavior.
6. Focused C++ plugin test executable if C++ provider tests are changed.
7. Real `ssh rvv` generated-bundle ABI/e2e run for pre-realized `cmp_select`
   with counts `7,16,23`, unless blocked.
8. Bounded authority scan over touched and relevant compare/select files.
9. `git diff --check`
10. `cmake --build build --target check-tianchenrv -j2`

## Definition Of Done

The pre-realized plain `cmp_select` path has a reviewable compare/select
predicate authority chain from typed selected body/config through RVV
realization, route-family facts, operand-binding facts, statement-plan facts,
emitted RVV compare/select intrinsic operands, generated artifact mirrors, and
focused generated-bundle evidence. Unsupported or inconsistent
predicate/value-selection dependencies fail closed before route/artifact
authority. The task is finished/archived and committed, or an exact blocker
and continuation point is recorded.

## Implementation Result

Completed for the bounded pre-realized plain `cmp_select` path.

- `scripts/rvv_generated_bundle_abi_e2e.py` now records and validates a
  `compare_select_predicate_boundary` for `cmp_select`, including typed
  materialized body facts, emitted RVV C++ compare/select operands,
  mirror-only artifact metadata, runtime counts, and an explicit statement
  that runtime counts are execution cases, not predicate authority.
- The emitted RVV C++ verifier now extracts the `cmp_select` vector and mask C
  types, `__riscv_vsetvl_e32m1`, `__riscv_vle32_v_i32m1`,
  `__riscv_vmseq_vv_i32m1_b32`, `__riscv_vmerge_vvm_i32m1`, and
  `__riscv_vse32_v_i32m1` operands. It proves the `lhs`/`rhs` loads feed the
  compare predicate, the predicate feeds the select/merge, the selected result
  feeds the store, and all intrinsic calls use the runtime-derived loop VL.
- `test/Target/RVV/pre-realized-selected-body-artifact-cmp-select.mlir` now
  checks direct target-header mirrors for compare predicate, mask
  role/source/form, select layout, provider-supported mirror, operand binding,
  and the plain compare/select route-family plan.
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-cmp-select-dry-run.test`
  now checks the generated-bundle compare/select predicate boundary,
  representative emitted intrinsic evidence, and predicate true/false harness
  coverage.
- `.trellis/spec/testing/mlir-testing-contract.md` now records the durable
  compare/select predicate generated-bundle evidence contract, including
  fields, validation failures, required tests, and wrong-vs-correct authority
  examples.

## Validation Result

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-compare-select-predicate-route-closure`
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Focused dry-run generated-bundle evidence for pre-realized
      `cmp_select`, counts `7,16,23`.
- [x] Focused lit from `build/test` for
      `pre-realized-selected-body-artifact-cmp-select` and
      `rvv-generated-bundle-abi-e2e-pre-realized-cmp-select-dry-run`, 3/3
      passed including the existing `cmp-select-sle` filtered fixture.
- [x] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] Real `ssh rvv` generated-bundle compile/run:
      `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_cmp_select_predicate_route_closure --run-id pre-realized-cmp-select-ssh-rvv --overwrite --op-kind cmp_select --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv`
- [x] Remote output ended with `PASS op=cmp_select counts=7,16,23`. The harness
      reported predicate coverage for all three counts:
      `n=7 true=2 false=5`, `n=16 true=4 false=12`, and
      `n=23 true=6 false=17`.
- [x] Bounded authority scan over touched and relevant RVV planning/provider/
      target/script/fixture/test files found only mirror checks, fail-closed
      guardrails, negative legacy/source-front-door tests, and existing
      forbidden-token evidence checks; no new metadata/name-derived
      compare/select authority or positive legacy i32 authority was introduced.
- [x] `git diff --check`
- [x] `cmake --build build --target check-tianchenrv -j2`, 365/365 passed.

## Status

Completed; ready to finish/archive and commit.
