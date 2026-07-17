# Stage2 RVV reduction accumulation route closure

## Goal

Close one bounded Stage 2 RVV reduction/accumulation route boundary for the
plain typed `standalone_reduce_add` path. The round must prove that reduction
kind, source vector type, scalar accumulator seed, accumulator/result layout,
runtime AVL/VL, policy, and final scalar result binding are carried by typed
`tcrv_rvv` body/config/runtime facts, validated by RVV plugin-owned family and
operand-binding facts, consumed through an RVV-owned statement-plan boundary
before provider route construction, emitted as the expected RVV horizontal
reduction form, mirrored by generated artifacts as mirrors only, and backed by
focused generated-bundle plus real `ssh rvv` correctness evidence.

## Direction Source

- Direction title: `Stage2 RVV reduction/accumulation route closure`.
- Module owner: RVV plugin-owned reduction/accumulation route boundary for one
  bounded typed `tcrv_rvv` reduction path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `88cf6b64 rvv: close conversion sew policy route boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- `.trellis/spec/index.md` keeps the current RVV-first authority chain as
  selected `tcrv.exec` RVV variant -> typed/realized `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/route provider -> common EmitC -> target
  artifact -> `ssh rvv` evidence for runtime/correctness claims.
- `.trellis/spec/extension-plugins/rvv-plugin.md` assigns reduction primitive,
  accumulator layout, result layout, dtype/config, runtime VL, intrinsic
  mapping, selected-body realization, and fail-closed diagnostics to the RVV
  plugin and typed `tcrv_rvv` body. Common EmitC/export must stay neutral.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires common EmitC to
  consume provider-built `TCRVEmitCLowerableRoute` payloads and not infer RVV
  reduction semantics, dtype, SEW/LMUL, accumulator layout, intrinsic spelling,
  ABI role, or route support from route ids, artifact metadata, ABI strings, or
  names.
- `.trellis/spec/testing/mlir-testing-contract.md` requires positive RVV
  generated artifact tests to use corrected typed `tcrv_rvv` body authority and
  real `ssh rvv` output for RVV runtime/correctness claims.
- The archived conversion/SEW task shows the expected bounded closure pattern:
  add an RVV-owned statement plan, wire provider consumption before generic
  provider-local assembly, extend generated-bundle evidence, run focused dry
  run and real `ssh rvv`, scan for authority drift, then finish/archive/commit.
- Existing code already has `tcrv_rvv.typed_standalone_reduce_pre_realized_body`,
  `tcrv_rvv.standalone_reduce`, standalone reduction route-family facts, math
  operand-binding facts, explicit/pre-realized target fixtures, dry-run script
  tests, and runtime harness generation for `standalone_reduce_add`.
- The remaining gap for this round is not a new reduction op surface. Current
  provider construction still falls through to older generic provider-local
  statement assembly for plain standalone reduction after route-family and
  operand-binding facts are available. The selected path should consume an
  RVV-owned standalone-reduction statement plan before that fallback.

## Requirements

1. Keep the executable/evidence closure bounded to plain
   `standalone_reduce_add` unless repository evidence proves it unsafe or
   stale.
2. The selected input must visibly carry reduction kind `add`, input vector
   type, scalar accumulator seed role, scalar output role, accumulator layout,
   result layout, store VL, dtype/config, policy, runtime count/VL, and ABI
   roles in typed `tcrv_rvv` structure.
3. RVV selected-body realization must consume the pre-realized standalone
   reduction body into realized `setvl`, `with_vl`, `load`, scalar seed splat,
   `standalone_reduce`, and `store` structure.
4. RVV route-family facts and math operand-binding facts must carry reduction
   kind, source vector type, scalar accumulator seed, output/result binding,
   accumulator/result layout, runtime VL, source/accumulator/output/count ABI
   operands, and provider leaf choices before route statement construction.
5. Add a focused RVV-owned standalone-reduction statement-plan boundary for
   `standalone_reduce_add`, and route provider consumption through the migrated
   statement-plan path. Provider code must not locally rebuild the selected
   standalone reduction initialization, setvl/load/reduce/store sequence from
   operation names, ABI strings, route ids, intrinsic mirrors, artifact
   metadata, or fixture names after this boundary is available.
6. Unsupported, stale, or inconsistent standalone reduction/accumulator cases
   must fail closed before common EmitC or target artifact authority, including
   missing standalone route-family plan, missing math operand-binding facts,
   missing accumulator seed binding, missing source load, missing scalar seed
   splat, missing reduction intrinsic, missing store, wrong accumulator/result
   layout, and missing runtime count/VL binding.
7. Generated RVV C/C++ evidence must prove emitted `standalone_reduce_add`
   intrinsic forms and operands derive from typed body/config/runtime facts:
   `vsetvl_e32m1` uses runtime `n`/remaining AVL, `vle32_v_i32m1` loads source
   chunks from `lhs`, scalar seed splat initializes the accumulator from the
   seed/output scalar boundary, the horizontal reduction consumes the loaded
   source vector and accumulator vector with runtime loop VL, and `vse32_v_i32m1`
   stores lane 0 to the scalar output boundary with store VL `1`.
8. Generated artifact metadata may mirror reduction kind, accumulator/result
   layout, store VL, C type mapping, provider support, operand binding, and
   runtime VL only after provider route construction. Mirror fields must not be
   used as authority.
9. Generated-bundle evidence must include a
   `reduction_accumulation_boundary` summary or equivalent tying typed
   materialized body facts, RVV-owned route/statement-plan facts, emitted RVV
   C++ operands, route metadata mirrors, artifact ABI, runtime counts, scalar
   seed behavior, empty-count behavior, and tail harness checks.
10. Do not add broad reduction coverage, min/max matrices, masked standalone
    reductions, dot reductions, MAcc/contraction work, high-level frontend
    lowering, source-front-door positive routes, legacy i32 authority,
    one-op-per-intrinsic wrapper growth, dashboards, or broad smoke matrices.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` reference the relevant RVV
      plugin, EmitC route, testing policy, and previous conversion task.
- [x] Focused lit/FileCheck coverage for pre-realized
      `standalone_reduce_add` proves the realized body consumes reduction facts
      into `tcrv_rvv.standalone_reduce` and target artifact mirrors expose
      reduction kind, accumulator/result layout, store VL, provider-supported
      mirror, operand binding, and runtime VL facts.
- [x] RVV planning/provider code exposes a standalone-reduction statement-plan
      boundary and the migrated provider boundary consumes it for
      `standalone_reduce_add` before generic provider-local statement assembly.
- [x] Focused C++/lit coverage proves the statement-plan boundary fails closed
      when required route-family facts, math operand-binding facts, source load,
      accumulator seed binding/splat, reduction intrinsic, store, or runtime VL
      dependencies are stale or missing.
- [x] `scripts/rvv_generated_bundle_abi_e2e.py` extracts emitted RVV C++ facts
      for pre-realized `standalone_reduce_add`, including vector type, setvl,
      load, scalar seed splat, reduction, store intrinsic, runtime loop VL,
      accumulator/result layout, store VL, seed source, and final scalar result
      binding.
- [x] Focused generated-bundle evidence for pre-realized
      `standalone_reduce_add` records a
      `reduction_accumulation_boundary` or equivalent summary tying typed
      body/config, provider/export mirrors, RVV-owned statement-plan
      consumption, emitted reduction intrinsic operands, runtime counts, scalar
      seed/empty-count behavior, and tail harness coverage.
- [x] One focused dry-run generated-bundle script test checks the new evidence
      summary and representative reduction/accumulation mirror metadata.
- [x] One real `ssh rvv` generated-bundle ABI/e2e run is attempted for
      pre-realized `standalone_reduce_add` across runtime counts `0,7,16,23`;
      if blocked, record the exact blocker and do not claim runtime
      correctness.
- [x] Bounded scan over touched RVV dialect/realization/planning/provider/
      target/script/fixture files finds no new metadata/name-derived
      reduction/accumulator authority and no positive legacy i32 route
      authority.
- [x] `git diff --check` passes.
- [x] Focused checks and `check-tianchenrv` pass, or an exact blocker is
      documented.
- [x] Trellis task status, journal, archive, and commit are truthful.

## Non-Goals

- No standalone reduction route beyond the chosen `standalone_reduce_add`
  closure.
- No min/max, computed-mask standalone reduction, runtime-scalar computed-mask
  standalone reduction, MAcc, widening dot, or contraction expansion.
- No full dtype/SEW/LMUL reduction matrix.
- No Linalg, Vector, StableHLO frontend, descriptor, direct-C, source-export,
  or source-front-door positive route work.
- No legacy `RVVI32M1*`, `rvv-i32m1-*`, `tcrv_rvv.i32_*`, or
  `!tcrv_rvv.i32m*` executable authority.
- No performance claim.
- No multi-agent or sub-agent workflow in this round.

## Technical Approach

1. Start the Trellis task and validate context.
2. Add a bounded standalone-reduction statement-plan struct/API in RVV route
   planning for `standalone_reduce_add`, consuming standalone route-family and
   math operand-binding facts and producing provider-ready initial seed store,
   full-chunk setvl, loop setvl/source-load/accumulator-splat/reduce/store
   steps.
3. Wire that statement plan into the migrated statement-plan aggregate so the
   provider attaches it and returns before older generic provider-local
   assembly for the selected standalone reduction route.
4. Tighten focused `standalone_reduce_add` artifact checks for statement-plan
   and mirror-only route facts as needed.
5. Extend generated-bundle evidence extraction for emitted RVV C++ standalone
   reduction intrinsics and operands and add a
   `reduction_accumulation_boundary` summary.
6. Add or tighten focused dry-run FileCheck coverage for the new evidence
   summary.
7. Reuse the generated-bundle command shape:

   ```bash
   python3 scripts/rvv_generated_bundle_abi_e2e.py \
     --pre-realized-selected-body \
     --artifact-root artifacts/tmp/stage2_reduction_accumulation_route_closure \
     --run-id pre-realized-standalone-reduce-add-ssh-rvv \
     --overwrite \
     --op-kind standalone_reduce_add \
     --runtime-count 0 \
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

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-reduction-accumulation-route-closure`
2. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
3. `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
4. Focused dry-run for pre-realized `standalone_reduce_add` with counts
   `0,7,16,23`.
5. Focused lit/FileCheck tests covering the changed RVV target and script
   behavior.
6. Focused C++ plugin test executable if C++ planning/provider APIs are
   changed.
7. Real `ssh rvv` generated-bundle ABI/e2e run for pre-realized
   `standalone_reduce_add` with counts `0,7,16,23`, unless blocked.
8. Bounded authority scan over touched and relevant reduction files.
9. `git diff --check`
10. `cmake --build build --target check-tianchenrv -j2`

## Completion Evidence

- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-reduction-accumulation-route-closure`
  passed.
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passed.
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
- Focused dry-run generated-bundle evidence for pre-realized
  `standalone_reduce_add` with runtime counts `0,7,16,23` passed and recorded
  `reduction_accumulation_boundary`.
- Focused lit test
  `Scripts/rvv-generated-bundle-abi-e2e-pre-realized-standalone-reduce-add-dry-run.test`
  passed.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2 &&
  build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- Real `ssh rvv` generated-bundle ABI/e2e passed for pre-realized
  `standalone_reduce_add` with counts `0,7,16,23` and seeds `-11,17`, including
  `n=0` seed-only behavior and non-empty horizontal accumulation.
- Bounded diff scan over touched RVV planning/provider/script/fixture files
  found no newly added positive `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, descriptor/source-front-door/source-artifact, route-id,
  artifact-name, or harness-only authority.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed, 365/365 tests.

## Definition Of Done

The pre-realized `standalone_reduce_add` path has a reviewable
reduction/accumulation authority chain from typed selected body/config through
RVV realization, standalone route-family facts, math operand-binding facts,
RVV-owned statement-plan facts, provider-built route, emitted RVV horizontal
reduction operands, generated artifact mirrors, and focused generated-bundle
evidence. Unsupported or inconsistent reduction/accumulator dependencies fail
closed before route/artifact authority. The task is finished/archived and
committed, or an exact blocker and continuation point is recorded.
