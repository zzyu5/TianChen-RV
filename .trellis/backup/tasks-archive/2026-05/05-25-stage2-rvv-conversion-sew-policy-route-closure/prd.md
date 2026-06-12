# Stage2 RVV conversion SEW policy route closure

## Goal

Close one bounded Stage 2 RVV conversion/SEW policy route boundary for the
existing pre-realized signed `widen_i16_to_i32` selected-body path. The round
must prove that conversion kind, source/result element types, source/result
SEW and LMUL, policy, runtime VL, and ABI operands are carried by typed
`tcrv_rvv` body/config structure, validated by RVV plugin-owned route facts,
consumed through an RVV-owned statement-plan boundary before provider route
construction, emitted as the expected RVV widening conversion intrinsic form,
mirrored by generated artifacts as mirrors only, and backed by focused
generated-bundle evidence plus real `ssh rvv` correctness when executable.

## Direction Source

- Direction title: `Stage2 RVV conversion/SEW policy route closure`.
- Module owner: RVV plugin-owned conversion and SEW policy route boundary for
  one bounded typed `tcrv_rvv` conversion path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `8c583c8a rvv: prove compare select predicate boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- `.trellis/spec/index.md` keeps the current RVV-first authority chain as
  selected `tcrv.exec` RVV variant -> typed/realized `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/route provider -> common EmitC -> target
  artifact -> `ssh rvv` evidence for runtime/correctness claims.
- `.trellis/spec/extension-plugins/rvv-plugin.md` assigns dtype, SEW, LMUL,
  policy, conversion, runtime VL, legality, intrinsic mapping, C vector type
  mapping, selected-body realization, and fail-closed diagnostics to the RVV
  plugin and typed `tcrv_rvv` body. Common EmitC/export must stay neutral.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires common EmitC to
  consume provider-built `TCRVEmitCLowerableRoute` payloads and not infer RVV
  dtype, SEW/LMUL, conversion semantics, intrinsic spelling, ABI role, or route
  support from route ids, artifact metadata, ABI strings, or names.
- `.trellis/spec/testing/mlir-testing-contract.md` already defines
  generated-bundle evidence contracts for mask/tail policy and compare/select
  predicate boundaries; conversion/SEW evidence should follow the same shape:
  typed body/config/runtime facts are authority, artifacts are mirrors only,
  emitted C++ and harness behavior are evidence.
- The archived compare/select PRD shows the expected bounded closure pattern:
  tighten focused lit/FileCheck, extract emitted RVV C++ facts into a
  family-specific evidence summary, run dry-run script checks, attempt real
  `ssh rvv`, scan for authority drift, then finish/archive/commit.
- Existing code already has `tcrv_rvv.typed_widening_conversion_pre_realized_body`
  and `tcrv_rvv.widening_convert`, route-family facts for `widen_i32_to_i64`
  and `widen_i16_to_i32`, math operand-binding facts, generated artifacts, dry
  run tests, and harnesses.
- The remaining gap for this round is not a new conversion op surface. Current
  provider construction still falls through to older generic provider-local
  statement assembly for widening conversion after route-family and operand
  facts are available. The selected conversion path should consume an RVV-owned
  widening-conversion statement plan before generic provider-local fallback.
- `widen_i16_to_i32` is the chosen bounded path because it explicitly exercises
  source SEW16/LMUL `mf2`, result SEW32/LMUL `m1`, signed extension, unit-stride
  conversion memory form, runtime VL, source/result C vector type mapping, and
  real sign-extension harness checks.

## Requirements

1. Keep the executable/evidence closure bounded to the existing pre-realized
   `widen_i16_to_i32` path unless repository evidence proves it unsafe or
   stale.
2. The selected input must visibly carry conversion kind, conversion relation,
   source element/vector type, result element/vector type, source/result SEW,
   source/result LMUL, memory form, policy, runtime count/VL, and ABI roles in
   typed `tcrv_rvv` structure.
3. RVV selected-body realization must consume the pre-realized conversion body
   into realized `setvl`, `with_vl`, `load`, `widening_convert`, and `store`
   structure.
4. RVV route-family facts and math operand-binding facts must carry conversion
   kind, source/result type policy, source/result SEW/LMUL, conversion
   relation, runtime VL, source/output/runtime-count ABI operands, and provider
   leaf choices before route statement construction.
5. Add a focused RVV-owned widening-conversion statement-plan boundary for the
   selected conversion route and route provider consumption through the
   migrated statement-plan path. Provider code must not locally rebuild the
   `widen_i16_to_i32` setvl/load/convert/store sequence from operation names,
   ABI strings, route ids, intrinsic mirrors, artifact metadata, or fixture
   names after this boundary is available.
6. Unsupported, stale, or inconsistent conversion/type-policy cases must fail
   closed before common EmitC or target artifact authority, including missing
   route-family plan, missing math operand-binding facts, missing source/result
   materialized ops, missing conversion intrinsic, wrong source/result
   SEW/LMUL, wrong conversion relation, and missing runtime count/VL binding.
7. Generated RVV C/C++ evidence must prove emitted `widen_i16_to_i32` intrinsic
   forms and operands derive from typed body/config/runtime facts:
   `vsetvl_e32m1` uses runtime `n`/remaining AVL, `vle16_v_i16mf2` loads from
   `lhs`, `vwcvt_x_x_v_i32m1` consumes the loaded source vector and loop VL,
   `vse32_v_i32m1` stores the converted result to `out`, and all relevant
   intrinsic calls use runtime-derived loop VL.
8. Generated artifact metadata may mirror conversion kind, conversion relation,
   source/result SEW/LMUL, C type mapping, provider support, operand binding,
   and runtime VL only after provider route construction. Mirror fields must
   not be used as authority.
9. Generated-bundle evidence must include an explicit
   `conversion_sew_policy_boundary` summary or equivalent tying typed
   materialized body facts, RVV-owned route/statement-plan facts, emitted RVV
   C++ operands, route metadata mirrors, artifact ABI, runtime counts, and
   sign-extension/tail harness checks.
10. Do not add broad conversion matrices, dtype/LMUL clone batches, reductions,
    contractions, high-level frontend lowering, source-front-door positive
    routes, legacy i32 authority, one-op-per-intrinsic wrapper growth,
    dashboards, or broad smoke matrices.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` reference the relevant RVV
      plugin, EmitC route, testing policy, and previous compare/select task.
- [x] Focused lit/FileCheck coverage for pre-realized `widen_i16_to_i32` proves
      the realized body consumes conversion facts into `tcrv_rvv.widening_convert`
      and target artifact mirrors expose source/result SEW/LMUL, conversion
      relation, provider-supported mirror, operand binding, and runtime VL facts.
- [x] RVV planning/provider code exposes a widening-conversion statement-plan
      boundary and the migrated provider boundary consumes it for
      `widen_i16_to_i32` before generic provider-local statement assembly.
- [x] Focused C++/lit coverage proves the statement-plan boundary fails closed
      when required route-family facts, math operand-binding facts, source/load,
      conversion, store, leaf intrinsic, or runtime VL dependencies are stale
      or missing.
- [x] `scripts/rvv_generated_bundle_abi_e2e.py` extracts emitted RVV C++ facts
      for pre-realized `widen_i16_to_i32`, including source/result vector C
      types, setvl, load, conversion, store intrinsics, runtime loop VL,
      conversion input, conversion output, store result, and source/result
      SEW/LMUL policy facts.
- [x] Focused generated-bundle evidence for pre-realized `widen_i16_to_i32`
      records a `conversion_sew_policy_boundary` or equivalent summary tying
      typed body/config, provider/export mirrors, RVV-owned statement-plan
      consumption, emitted conversion intrinsic operands, runtime counts, and
      sign-extension/tail harness coverage.
- [x] One focused dry-run generated-bundle script test checks the new evidence
      summary and representative conversion/SEW mirror metadata.
- [x] One real `ssh rvv` generated-bundle ABI/e2e run is attempted for
      pre-realized `widen_i16_to_i32` across runtime counts `7,16,23`; if
      blocked, record the exact blocker and do not claim runtime correctness.
- [x] Bounded scan over touched RVV dialect/realization/planning/provider/
      target/script/fixture files finds no new metadata/name-derived
      conversion/dtype/SEW/LMUL authority and no positive legacy i32 route
      authority.
- [x] `git diff --check` passes.
- [x] Focused checks and `check-tianchenrv` pass, or an exact blocker is
      documented.
- [x] Trellis task status, journal, archive, and commit are truthful.

## Non-Goals

- No new conversion route beyond the chosen pre-realized `widen_i16_to_i32`
  closure.
- No full dtype/SEW/LMUL conversion matrix.
- No broad Stage 2 reduction, contraction, memory, compare/select, Linalg,
  Vector, StableHLO frontend, descriptor, direct-C, source-export, or
  source-front-door positive route work.
- No legacy `RVVI32M1*`, `rvv-i32m1-*`, `tcrv_rvv.i32_*`, or
  `!tcrv_rvv.i32m*` executable authority.
- No performance claim.
- No multi-agent or sub-agent workflow in this round.

## Technical Approach

1. Start the Trellis task and validate context.
2. Add a bounded widening-conversion statement-plan struct/API in RVV route
   planning for `widen_i16_to_i32`, consuming route-family and math
   operand-binding facts and producing provider-ready full-chunk setvl plus
   loop setvl/load/convert/store steps.
3. Wire that statement plan into the migrated statement-plan aggregate so the
   provider attaches it and returns before the older generic provider-local
   assembly path for the selected conversion route.
4. Tighten focused `widen_i16_to_i32` artifact checks for statement-plan and
   mirror-only route facts as needed.
5. Extend generated-bundle evidence extraction for emitted RVV C++ conversion
   intrinsics and operands and add a `conversion_sew_policy_boundary` summary.
6. Add or tighten focused dry-run FileCheck coverage for the new evidence
   summary.
7. Reuse the existing generated-bundle command shape:

   ```bash
   python3 scripts/rvv_generated_bundle_abi_e2e.py \
     --pre-realized-selected-body \
     --artifact-root artifacts/tmp/stage2_conversion_sew_policy_route_closure \
     --run-id pre-realized-widen-i16-to-i32-ssh-rvv \
     --overwrite \
     --op-kind widen_i16_to_i32 \
     --runtime-count 7 \
     --runtime-count 16 \
     --runtime-count 23 \
     --tcrv-opt build/bin/tcrv-opt \
     --tcrv-translate build/bin/tcrv-translate \
     --ssh-target rvv
   ```

8. Keep validation focused on changed behavior, then run full
   `check-tianchenrv` if feasible.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-conversion-sew-policy-route-closure`
2. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
3. `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
4. Focused dry-run for pre-realized `widen_i16_to_i32` with counts `7,16,23`.
5. Focused lit/FileCheck tests covering the changed RVV target and script
   behavior.
6. Focused C++ plugin test executable if C++ planning/provider APIs are changed.
7. Real `ssh rvv` generated-bundle ABI/e2e run for pre-realized
   `widen_i16_to_i32` with counts `7,16,23`, unless blocked.
8. Bounded authority scan over touched and relevant conversion files.
9. `git diff --check`
10. `cmake --build build --target check-tianchenrv -j2`

## Implementation Result

- Added a provider-visible
  `RVVSelectedBodyWideningConversionRouteStatementPlan` for the bounded
  `widen_i16_to_i32` conversion path.
- Wired the widening conversion statement plan into the migrated RVV
  statement-plan aggregate, so provider route construction consumes the
  RVV-owned setvl/load/convert/store plan before the older generic
  provider-local assembly path for this selected conversion route.
- Added focused plugin coverage for positive statement-plan construction,
  provider route attachment, migrated-family selection, unrelated-family empty
  result, and fail-closed stale dependencies for missing route-family plan,
  missing math operand-binding facts, and missing conversion callee.
- Extended generated-bundle evidence with a bounded
  `conversion_sew_policy_boundary` for `widen_i16_to_i32` only. The summary
  records typed source/result type policy, conversion relation/kind, emitted
  RVV C++ setvl/load/convert/store intrinsics and operands, mirror-only route
  metadata, artifact ABI, and runtime-count non-authority.
- Tightened the focused dry-run FileCheck fixture to require the new
  conversion/SEW evidence and sign-extension/tail harness checks.
- Added the durable testing contract for conversion/SEW generated-bundle
  evidence under `.trellis/spec/testing/mlir-testing-contract.md`.

## Validation Result

- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`: passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`:
  passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test`: passed.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`: passed.
- Focused dry-run generated-bundle run for pre-realized `widen_i16_to_i32`
  with counts `7,16,23`: passed with `dry_run_success`.
- Manual FileCheck over the focused target fixture `REALIZED`, `PLAN`, and
  `HEADER` prefixes: passed.
- Manual FileCheck over focused script evidence `ROOT`, `WIDEN`, and `HARNESS`
  prefixes: passed after correcting ROOT check ordering to match the JSON
  evidence structure.
- Real `ssh rvv` generated-bundle ABI/e2e run for pre-realized
  `widen_i16_to_i32` with counts `7,16,23`: passed. Output included
  `sign_extension_checked tail_preserved` for all three counts and
  `PASS op=widen_i16_to_i32 counts=7,16,23`.
- Bounded authority scan over the touched implementation/script/test/spec
  diff found only negative testing-spec language for route-id/artifact/ABI
  non-authority; no new positive descriptor/source-export/legacy
  `tcrv_rvv.i32_*` conversion authority was introduced.
- `git diff --check`: passed.
- `cmake --build build --target check-tianchenrv -j2`: passed,
  `365/365` tests.

## Definition Of Done

The pre-realized signed `widen_i16_to_i32` path has a reviewable conversion/SEW
authority chain from typed selected body/config through RVV realization,
route-family facts, math operand-binding facts, RVV-owned statement-plan facts,
provider-built route, emitted RVV conversion intrinsic operands, generated
artifact mirrors, and focused generated-bundle evidence. Unsupported or
inconsistent conversion/type-policy dependencies fail closed before
route/artifact authority. The task is finished/archived and committed, or an
exact blocker and continuation point is recorded.
