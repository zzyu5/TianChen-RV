# Stage2 RVV typed strided memory-form route semantics

## Goal

Add one bounded Stage 2 generic typed RVV strided memory-form route for an
explicit selected-body add path. The selected `tcrv.exec` RVV variant must bind
runtime stride operands explicitly, the typed `tcrv_rvv` body must structurally
consume those stride values in load/store memory-form ops, and the RVV provider
must derive the route/header/intrinsic/artifact facts only after typed
body/config/runtime validation.

This task closes one structured-kernel memory movement gap: strided load/store
for a generic add body. It is not a broad memory framework.

## Current Facts

- Repo root is `/home/kingdom/phdworks/TianchenRV`; initial `pwd`,
  `git status --short`, and `git log --oneline -8` showed a clean worktree at
  `779b635 rvv: harden stage1 residue front doors`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief as
  `.trellis/tasks/05-19-stage2-rvv-typed-strided-memory-route`.
- Relevant specs require the RVV authority chain:
  selected `tcrv.exec` RVV variant -> explicit typed low-level `tcrv_rvv` body
  -> RVV plugin legality/provider -> provider-built
  `TCRVEmitCLowerableRoute` -> neutral common EmitC/export -> target artifact.
- Stage 1 guardrails remain active: no positive `rvv-i32m1`, `RVVI32M1`,
  `i32_binary_pre_realized_body`, finite positive `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, source-front-door/source-seed, descriptor, artifact-name,
  or exact intrinsic-spelling route authority may be reintroduced.
- Current Stage 2 generic typed surface already has unit-stride
  `tcrv_rvv.load` / `tcrv_rvv.store`, RHS broadcast, `binary`, compare/select,
  masked add, reduce add, macc, and typed config validation.
- Current provider recognizes two memory forms: vector RHS load and RHS
  broadcast load. There is no generic typed strided load/store memory form yet.

## Requirements

1. Keep scope to one coherent strided add route, expected as generic typed
   `tcrv_rvv.strided_load` / `tcrv_rvv.strided_store` or an equivalent equally
   structural low-level memory-form surface.
2. The typed body must explicitly consume runtime stride operands imported
   through `tcrv_rvv.runtime_abi_value`, not artifact names, route ids,
   descriptors, ABI strings, or Python script state.
3. The bounded positive route must structurally carry:
   - lhs/rhs/out base pointer roles;
   - lhs/rhs/out runtime stride values;
   - runtime `n` / AVL source;
   - active VL token;
   - typed vector element/config facts;
   - operation kind `add`;
   - strided load/store memory form.
4. The verifier must reject missing or wrongly typed stride operands,
   non-runtime-ABI stride values, wrong stride roles, wrong VL tokens, vector
   type/config mismatches, unsupported memory forms, and unsupported op
   combinations.
5. The construction protocol must recognize structural memory roles and the
   strided add operation without treating manifests, fixture names, route ids,
   artifact names, status fields, or source-front-door metadata as authority.
6. The RVV provider must derive strided route/profile/header/intrinsic payloads
   only after typed body/config/runtime validation, or fail closed with targeted
   diagnostics.
7. Common EmitC/export must stay neutral. Any support needed for carrying
   provider-built stride expressions must remain generic expression mechanics,
   not RVV semantic inference.
8. Add positive materialization, target artifact dry-run, and generated-bundle
   evidence for strided add. Collect real `ssh rvv` correctness evidence if and
   only if executable status is claimed.
9. Preserve existing arithmetic, RHS broadcast, compare/select, masked add,
   reduction, macc, typed-config, and Stage1 residue hardening behavior.

## Acceptance Criteria

- [x] `tcrv_rvv` verifier accepts a valid typed strided add body and rejects
      missing/wrong stride operands, wrong stride roles, type/config/VL
      mismatches, and unsupported memory-form combinations.
- [x] RVV construction protocol recognizes one provider-owned strided add route
      from typed memory-role structure.
- [x] RVV provider derives route metadata, strided load/store intrinsic leaves,
      stride layout mirrors, header/artifact facts, and materialized payload
      only from typed body/config/runtime facts.
- [x] Positive target artifact/lit coverage proves the strided add path reaches
      provider-derived route metadata and generated header/object bundle through
      generic typed `tcrv_rvv` body structure.
- [x] Negative fail-closed coverage proves unsupported strided combinations and
      stale legacy/source-front-door authority fail before target artifact
      construction.
- [x] `scripts/rvv_generated_bundle_abi_e2e.py --op-kind strided_add` supports
      local dry-run verification and real `ssh rvv` correctness evidence if
      executable status is claimed.
- [x] Focused build/tests for touched RVV dialect/provider/construction/script
      and target paths pass.
- [x] Active-authority scan confirms no active `rvv-i32m1`, `RVVI32M1`,
      `i32_binary_pre_realized_body`, finite positive `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, source-seed/source-front-door positive RVV authority,
      descriptor, or common/export RVV semantic authority is reintroduced.

## Implementation Results

- Added generic typed `tcrv_rvv.strided_load` and `tcrv_rvv.strided_store`
  verifier coverage for base pointer roles, stride runtime ABI roles, active VL,
  vector type/config agreement, and selected-body placement.
- Added explicit runtime ABI stride roles:
  `lhs-input-stride`, `rhs-input-stride`, and `output-stride`, with a bounded
  seven-parameter selected-body ABI contract for strided add.
- Added RVV construction protocol recognition for `strided_add` from typed
  memory-role structure and operation kind, not fixture names or artifact
  metadata.
- Extended the RVV provider with a bounded `StridedAdd` / `StridedLoadStore`
  route path that derives stride layout mirrors, route id, runtime ABI,
  intrinsic leaves, header facts, and EmitC payload after typed body/config/
  runtime validation.
- Kept common EmitC/export neutral by adding generic expression materialization
  for provider-built product and scaled pointer expressions; RVV semantics stay
  in the RVV provider.
- Extended target artifact validation so dynamic runtime ABI identity is checked
  against the provider-derived route signature instead of a fixed four-parameter
  assumption.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` with
  `--op-kind strided_add`, dry-run checks, and a real RVV correctness harness
  using explicit lhs/rhs/out runtime stride values.

## Validation Results

- [x] `cmake --build build --target tcrv-opt tcrv-translate
      tianchenrv-rvv-dialect-test tianchenrv-rvv-extension-plugin-test
      tianchenrv-construction-protocol-common-test
      tianchenrv-target-artifact-export-test -j2`
- [x] Focused lit filter from `build/test`:
      `generic-stage2-dataflow|rvv-generic-stage2-strided-add|
      explicit-selected-body-artifact-strided-add|
      rvv-generated-bundle-abi-e2e-strided-add` passed 5/5.
- [x] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [x] Focused C++ tests:
      `build/bin/tianchenrv-rvv-dialect-test`,
      `build/bin/tianchenrv-rvv-extension-plugin-test`,
      `build/bin/tianchenrv-construction-protocol-common-test`,
      `build/bin/tianchenrv-target-artifact-export-test`.
- [x] Local strided add generated-bundle dry-run passed at
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/local-strided-add-dry-run`.
- [x] Real `ssh rvv` correctness evidence passed at
      `artifacts/tmp/rvv_generated_bundle_abi_e2e/ssh-strided-add` for counts
      `7,16,23`, with `PASS op=strided_add counts=7,16,23`.
- [x] `cmake --build build --target check-tianchenrv -j2` passed 166/166
      after repairing stale FileCheck diagnostics.
- [x] `git diff --check`
- [x] Active-authority scan over RVV include/lib/script/test paths found only
      existing legacy parse-only, fail-closed, negative-test, or prompt mentions;
      no new positive legacy RVV route authority was introduced.

## Self-Repair

- Repaired strided runtime ABI verifier handling because stride
  `runtime_abi_value` results are `index`, not `!tcrv_rvv.runtime_abi_value`.
- Replaced undeclared C-name `emitc.literal` pointer payloads with neutral
  EmitC SSA expression materialization for `offset * stride` and
  `base + (offset * stride)`.
- Repaired target artifact export preflight so provider-derived dynamic runtime
  ABI params can validate the seven-parameter strided route.
- Updated stale fail-closed FileCheck expectations after the supported generic
  selected-body op list gained `tcrv_rvv.strided_load/store`.

## Non-Goals

- No gather/scatter, segmented memory, mask memory, conversion, broad memory
  framework, dtype/LMUL clone batch, high-level Linalg/Vector/StableHLO
  frontend lowering, global tuning, dashboards, or performance claim.
- No source-front-door positive RVV route and no source-seed evidence mode.
- No descriptor-driven computation.
- No one-intrinsic wrapper surface detached from typed memory-form roles.
- No Scalar, IME, Offload, TensorExt, Template/Toy, or future-plugin work.

## Validation Plan

1. Validate Trellis task context and start the task.
2. Build focused targets expected to include `tcrv-opt`, `tcrv-translate`,
   `tianchenrv-rvv-dialect-test`, `tianchenrv-rvv-extension-plugin-test`,
   `tianchenrv-construction-protocol-common-test`, and
   `tianchenrv-target-artifact-export-test`.
3. Run focused RVV dialect lit tests for strided verifier positives/negatives.
4. Run focused EmitC/target artifact lit/FileCheck tests for strided
   materialization and fail-closed unsupported cases.
5. Run `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
   script self-test if the script changes.
6. Run generated-bundle dry-run for `--op-kind strided_add`.
7. Run real `ssh rvv` correctness evidence for `--op-kind strided_add` only if
   the generated bundle compiles/runs through the evidence harness.
8. Run `git diff --check`.
9. Run an active-authority scan over active RVV include/lib/test/script paths.
10. Run broader `check-tianchenrv` if shared route/provider/script behavior
    changed enough to justify it.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/index.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-19-stage1-residue-source-front-door-hardening/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-rvv-arithmetic-pre-realized-realization-closure/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-typed-config-arithmetic/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-masked-add-route-semantics/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-stage2-generic-rvv-reduction-executable-closure/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-stage2-rvv-multiply-add-route-skeleton/prd.md`,
  `.trellis/tasks/archive/2026-05/05-19-rvv-rhs-broadcast-memory-form-executable-route/prd.md`.
- Initial implementation surface:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  focused tests under `test/Dialect/RVV`, `test/Conversion/EmitC`,
  `test/Target/RVV`, and `test/Scripts`.

## Definition Of Done

- [x] One coherent Stage 2 strided add memory-form submodule is represented,
  verified, route-supported, materialized through the production RVV provider
  path, and validated with focused evidence.
- [x] Existing Stage 2 selected-body routes remain intact.
- [x] The task report distinguishes route-supported, dry-run artifact evidence,
  and executable `ssh rvv` evidence.
