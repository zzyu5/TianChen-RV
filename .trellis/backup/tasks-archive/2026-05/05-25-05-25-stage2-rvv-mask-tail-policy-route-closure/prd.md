# Stage2 RVV mask/tail policy route closure

## Goal

Close one bounded Stage 2 mask/tail policy authority path for the existing
pre-realized `masked_unit_store` RVV selected-body artifact. The round must
prove that explicit mask input, mask role, tail policy, mask policy, and
inactive-lane behavior are carried by the typed `tcrv_rvv` body/config,
validated by RVV plugin realization and route planning, consumed by provider
route facts and statement plans, mirrored by generated artifacts as mirrors
only, and backed by one focused generated-bundle evidence path. Unsupported or
inconsistent mask/policy combinations must fail closed before route/artifact
authority.

## Direction Source

- Direction title: `Stage2 RVV mask/tail policy route closure`.
- Module owner: RVV plugin-owned mask/tail policy authority from typed
  `tcrv_rvv` body/config into route facts, provider emission, generated
  artifact mirrors, and focused evidence for an existing masked
  route-supported family.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean before task creation.
- Initial HEAD: `cd0ac79b rvv: prove runtime avl vl boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  Hermes Direction Brief before source edits.

## What I Already Know

- `.trellis/spec/index.md` requires the current RVV authority chain to run
  through selected `tcrv.exec` variant, typed/realized `tcrv_rvv` body, RVV
  plugin legality / realization / route provider, common EmitC, target
  artifact, and real `ssh rvv` evidence when runtime correctness is claimed.
- `.trellis/spec/extension-plugins/rvv-plugin.md` states that mask/tail
  behavior belongs in typed `tcrv_rvv` body/config and RVV plugin legality /
  provider facts. Common EmitC/export must consume provider-built routes
  neutrally and must not infer policy or mask semantics from route ids,
  metadata, ABI names, artifact names, or tests.
- `.trellis/spec/lowering-runtime/emitc-route.md` explicitly forbids common
  EmitC from inferring mask/tail policy or RVV intrinsics.
- `.trellis/spec/testing/mlir-testing-contract.md` requires lit/FileCheck,
  focused C++ API tests when textual MLIR is insufficient, and real `ssh rvv`
  evidence for runtime/correctness claims.
- The archived runtime AVL/VL closure task proved runtime `n` flows into
  `tcrv_rvv.setvl`, provider mirrors, generated RVV C++, generated artifact
  ABI, and real `ssh rvv` correctness for pre-realized `i64_add`.
- Current ODS and implementation already expose the narrow target path:
  `tcrv_rvv.typed_masked_memory_pre_realized_body` for
  `masked_unit_store`, `tcrv_rvv.mask_load`, `tcrv_rvv.masked_store`,
  `#tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>`,
  `inactive_lane_policy = "preserve-output-on-false-lanes"`, base memory
  route-family facts, memory operand-binding facts, a base memory
  statement-plan boundary, generated bundle script support, and negative
  verifier coverage for bad policy/inactive-lane/runtime-role cases.

## Requirements

1. Keep the executable/evidence closure bounded to the already
   route-supported pre-realized `masked_unit_store` path unless live repository
   evidence proves that path unsafe or stale.
2. The selected input must visibly carry an explicit mask input ABI value,
   `mask_role = "predicate-mask-input-buffer"`,
   `mask_memory_form = "unit-stride-mask-load"`,
   `policy = #tcrv_rvv.policy<tail = undisturbed, mask = undisturbed>`, and
   `inactive_lane_policy = "preserve-output-on-false-lanes"` in typed
   `tcrv_rvv` structure.
3. RVV selected-body realization must consume those typed facts into realized
   `setvl`, `with_vl`, `mask_load`, `load`, and `masked_store` structure. The
   provider must consume the realized structure through verified route-family
   facts, memory operand-binding facts, and the base memory statement-plan
   boundary.
4. Provider route facts and statement plans must expose that mask operand,
   mask role, undisturbed tail/mask policy, and inactive-lane contract are
   derived from typed body/config/runtime facts. Route metadata and generated
   artifact fields are mirrors only, not authority.
5. Generated RVV C/C++ evidence must show the emitted route uses a mask load /
   compare-derived RVV mask and the masked store intrinsic form with the mask
   operand, destination, payload, and runtime VL.
6. Generated bundle evidence must include an explicit mask/tail policy boundary
   summary tying typed body/config facts, provider/export metadata mirrors,
   emitted RVV C++ intrinsic forms, and harness inactive/tail lane checks.
7. Missing mask operands, wrong mask role, inconsistent policy, unsupported
   inactive-lane behavior, stale route-family facts, or policy/body mismatches
   must fail closed before common EmitC or target artifact authority.
8. Do not add broad operation-family coverage, reductions, contractions,
   high-level frontend lowering, source-front-door positive routes, legacy i32
   authority, one-op-per-intrinsic wrapper growth, dtype/LMUL clone batches,
   dashboards, or broad smoke matrices.
9. Do not infer mask/tail policy from route ids, helper names, artifact names,
   test names, ABI strings, manifests, descriptor residue, intrinsic spellings,
   or harness-only constants.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` reference the relevant RVV
      plugin, EmitC route, testing policy, and previous archived task.
- [x] Focused lit/FileCheck coverage for pre-realized `masked_unit_store`
      proves the realized body consumes the explicit `mask` ABI through
      `tcrv_rvv.mask_load`, emits `tcrv_rvv.masked_store`, and preserves
      `tail = undisturbed`, `mask = undisturbed`, and
      `inactive_lane_policy = "preserve-output-on-false-lanes"`.
- [x] Focused provider/C++ coverage proves the base memory route-family plan,
      typed config facts, memory operand-binding facts, and base memory
      statement-plan boundary carry the `masked_unit_store` mask/policy facts
      before route construction.
- [x] Focused generated artifact checks prove mirror-only metadata includes
      `tcrv_rvv.tail_policy`, `tcrv_rvv.mask_policy`,
      `tcrv_rvv.mask_role`, `tcrv_rvv.mask_memory_form`,
      `tcrv_rvv.inactive_lane_contract`, route operand bindings, and provider
      supported mirror for `masked_unit_store`.
- [x] Focused generated-bundle evidence for pre-realized `masked_unit_store`
      records a `mask_tail_policy_boundary` or equivalent summary tying typed
      body/config, provider/export mirrors, emitted masked intrinsic operands,
      runtime counts, and inactive/tail lane harness contracts.
- [x] One focused dry-run generated-bundle script test checks the new evidence
      summary and representative masked/policy metadata.
- [x] One real `ssh rvv` generated-bundle ABI/e2e run is attempted for
      pre-realized `masked_unit_store` across runtime counts `7,16,23` if the
      route is executable; if blocked, record the exact blocker and do not
      claim runtime correctness.
- [x] Existing fail-closed tests still reject bad mask/tail policy,
      inactive-lane policy, wrong memory form, bad runtime roles, and stale
      authority metadata before route/artifact authority.
- [x] Bounded scan over touched RVV dialect/realization/planning/provider/
      target/fixture/script files finds no new metadata-derived mask/tail
      authority or positive legacy i32 route authority.
- [x] `git diff --check` passes.
- [x] Focused checks and `check-tianchenrv` pass, or an exact blocker is
      documented.
- [x] Trellis task status, journal, archive, and commit are truthful.

## Non-Goals

- No new RVV operation-family coverage class.
- No Stage 2 reduction, contraction, compare/select, computed-mask expansion,
  Linalg/Vector/StableHLO frontend, descriptor, direct-C, source-export, or
  source-front-door positive route work.
- No legacy `RVVI32M1*`, `rvv-i32m1-*`, `tcrv_rvv.i32_*`, or
  `!tcrv_rvv.i32m*` executable authority.
- No performance claim.
- No multi-agent or sub-agent workflow in this round.

## Technical Approach

1. Start the Trellis task and validate context.
2. Tighten focused `masked_unit_store` lit/FileCheck coverage so policy, mask
   operand, mask role, inactive-lane contract, provider mirrors, and header
   mirrors are explicit.
3. Add or tighten C++ provider coverage around `masked_unit_store` so route
   analysis/materialization/operand binding/statement plan facts carry the
   mask/tail policy boundary.
4. Harden the generated-bundle ABI/e2e evidence script to record and verify a
   `mask_tail_policy_boundary` for masked routes, initially focused on
   `masked_unit_store`.
5. Reuse the existing generated-bundle command shape:

   ```bash
   python3 scripts/rvv_generated_bundle_abi_e2e.py \
     --pre-realized-selected-body \
     --artifact-root artifacts/tmp/stage2_mask_tail_policy_route_closure \
     --run-id pre-realized-masked-unit-store-ssh-rvv \
     --overwrite \
     --op-kind masked_unit_store \
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

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-05-25-stage2-rvv-mask-tail-policy-route-closure`
2. `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
3. `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
4. Focused dry-run for pre-realized `masked_unit_store` with counts `7,16,23`.
5. Focused lit/FileCheck tests covering the changed RVV target and script
   behavior.
6. Focused C++ plugin test executable if C++ provider tests are changed.
7. Real `ssh rvv` generated-bundle ABI/e2e run for pre-realized
   `masked_unit_store` with counts `7,16,23`, unless blocked.
8. Bounded authority scan over touched and relevant mask/tail policy files.
9. `git diff --check`
10. `cmake --build build --target check-tianchenrv -j2`

## Definition Of Done

The pre-realized `masked_unit_store` path has a reviewable mask/tail policy
authority chain from typed selected body/config through RVV realization,
route-family facts, operand-binding facts, statement-plan facts, emitted
masked RVV intrinsic operands, generated artifact mirrors, and focused
generated-bundle evidence. Unsupported or inconsistent mask/policy combinations
fail closed before route/artifact authority. The task is finished/archived and
committed, or an exact blocker and continuation point is recorded.

## Implementation Result

Completed for the bounded pre-realized `masked_unit_store` path.

- `scripts/rvv_generated_bundle_abi_e2e.py` now records and validates a
  `mask_tail_policy_boundary` for `masked_unit_store`, including typed
  materialized body facts, emitted RVV C++ mask/load/compare/store operands,
  mirror-only artifact metadata, runtime counts, and an explicit statement that
  runtime counts are execution cases, not policy authority.
- `test/Plugin/RVVExtensionPluginTest.cpp` now asserts that the RVV provider's
  base memory route-family plan, typed config facts, route materialization
  facts, and memory operand-binding facts carry `tail = undisturbed`,
  `mask = undisturbed`, `mask` ABI binding, `predicate-mask-input-buffer`,
  `unit-stride-mask-load`, target leaf profile, provider mirror, compare leaf,
  and masked-store intrinsic before statement planning.
- `test/Target/RVV/pre-realized-selected-body-artifact-masked-unit-store.mlir`
  now checks header mirrors for config/policy, mask role/source/form,
  inactive-lane contract, masked memory layout, destination memory form,
  route operand binding, provider supported mirror, and base memory plan.
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-masked-unit-store-dry-run.test`
  now checks the new generated-bundle evidence summary and representative
  mirror metadata.
- `.trellis/spec/testing/mlir-testing-contract.md` now records the durable
  mask/tail policy generated-bundle evidence contract, including fields,
  validation failures, and required tests.

## Validation Result

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-05-25-stage2-rvv-mask-tail-policy-route-closure`
- [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Focused dry-run generated-bundle evidence for pre-realized
      `masked_unit_store`, counts `7,16,23`.
- [x] Focused lit from `build/test` for
      `pre-realized-selected-body-artifact-masked-unit-store`, 1/1 passed.
- [x] Focused lit from `build/test` for
      `rvv-generated-bundle-abi-e2e-pre-realized-masked-unit-store-dry-run`,
      1/1 passed.
- [x] Focused fail-closed lit from `build/test` for
      `pre-realized-masked-unit-store-negative`, 1/1 passed.
- [x] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [x] Real `ssh rvv` generated-bundle compile/run:
      `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/stage2_mask_tail_policy_route_closure --run-id pre-realized-masked-unit-store-ssh-rvv --overwrite --op-kind masked_unit_store --runtime-count 7 --runtime-count 16 --runtime-count 23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --ssh-target rvv`
- [x] Remote output ended with `PASS op=masked_unit_store counts=7,16,23`.
      The harness checked active lanes, inactive-lane preservation, and tail
      sentinel preservation.
- [x] Bounded authority scan over touched and relevant RVV dialect,
      realization, planning, provider, target, fixture, and script files found
      only typed body/config facts, mirror metadata, fail-closed negative
      cases, and existing legacy fail-closed tests; no new metadata/name-derived
      mask/tail authority or positive legacy i32 authority was introduced.
- [x] `git diff --check`
- [x] `cmake --build build --target check-tianchenrv -j2`, 365/365 passed.

## Status

Completed; ready to finish/archive and commit.
