# Stage2 RVV dequant-clamp f32 epilogue selected-body artifact foundation

## Goal

Establish a truthful route-supported foundation for one RVV
`dequant_clamp_f32_epilogue` boundary from selected `tcrv.exec` RVV variant and
typed/pre-realized `tcrv_rvv` body through RVV plugin-owned selected-body
realization, route-family planning, `TCRVEmitCLowerableRoute`, common EmitC
materialization, and target artifact validation. If generated-bundle execution
is already reachable without broadening scope, include `ssh rvv` evidence;
otherwise finish at route-supported plus target-validation closure and record
executable ABI as the exact continuation.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV dequant-clamp f32 epilogue selected-body artifact foundation`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `e9021470 rvv: close f32 clamp select executable abi evidence`.
- No `.trellis/.current-task` existed at the start of the round, so this task
  was created and started before source edits.
- The previous f32 clamp/select artifact task closed provider/header mirror
  validation for lower/upper clamp bounds, route operand binding,
  route-family plan, required headers, C type mapping, and provider mirror.
- The previous f32 clamp/select executable ABI task proved generated-bundle
  execution on real `ssh rvv` for counts `0,1,16,17,257`, two bound pairs,
  source preservation, scalar oracle, and tail sentinel preservation.
- The construction-protocol repair task aligned the common route proof with
  current RVV route mnemonics including `dequant_clamp_f32_epilogue`, but that
  proof is not route support or artifact authority.
- Existing specs require authority to flow from typed selected `tcrv_rvv`
  body/config/runtime facts through RVV plugin realization and provider-built
  routes. Common EmitC and target export may only consume and mirror those
  provider facts.
- Existing specs have a pure `dequantize_i32_to_f32` conversion dtype-policy
  contract and a f32 clamp/select generated-bundle evidence contract. This
  task must verify whether `dequant_clamp_f32_epilogue` already has an
  equivalent provider/target contract or needs one.

## Requirements

- Establish, or prove already existing, production support for
  `dequant_clamp_f32_epilogue` from typed selected body through target
  artifact.
- Validate dequant operation facts, source/result dtype and C type mapping,
  scale/zero-point or equivalent dequant parameters, lower/upper clamp bound
  roles, runtime ABI order, operand binding, VL placement, policy,
  result layout, route-family plan, header/intrinsic mirrors, and explicit
  `provider_supported_mirror` from RVV provider-owned facts.
- Fail closed for stale or missing dequant parameters, lower/upper bound roles,
  operation kind, source/result dtype/config, runtime ABI/order, operand
  binding, result layout, selected route family, provider mirror, and
  intrinsic/header mirrors.
- Keep selected-body realization and route-family planning in the RVV plugin;
  common EmitC/export must remain neutral and must not infer dequant or clamp
  semantics from route strings, artifact names, ABI names, test names, scripts,
  descriptors, manifests, or construction-protocol metadata.
- Preserve existing f32 clamp/select and pure dequantization behavior if shared
  provider, target, or script code is touched.
- Keep the task bounded to `dequant_clamp_f32_epilogue`; do not add fused
  `widening_product_reduce_dequant_clamp_f32` route work in this round.

## Acceptance Criteria

- [x] Positive route evidence shows `dequant_clamp_f32_epilogue` support from
      typed selected/pre-realized RVV body through `TCRVEmitCLowerableRoute`.
- [x] Positive target/header evidence shows artifact export mirrors
      provider-derived dequant, clamp, dtype/config, runtime ABI, operand
      binding, route-family, header/type/intrinsic, result-layout, and
      `provider_supported_mirror` facts.
- [x] Negative checks cover stale or missing dequant scale/parameter facts,
      lower/upper clamp roles and C types, bound order/clamp relation,
      operation kind, source/result dtype/config, runtime ABI order, route
      operand binding, result layout, selected route-family plan, header/type
      mapping, intrinsic mirrors, and provider mirror.
- [x] Non-consumer or cross-family routes carrying stale
      `dequant_clamp_f32_epilogue` mirrors fail closed.
- [x] Common EmitC/export remains neutral and no new common RVV semantic
      branch is introduced.
- [x] Focused `tcrv-opt`/`tcrv-translate`, RVV plugin, target artifact/export,
      generated-bundle dry-run or executable checks run as applicable.
- [x] f32 clamp/select regression checks run if shared compare/select or
      target validation code is touched.
- [x] `git diff --check`, `git diff --cached --check`, task validation, and a
      bounded old-authority scan over touched files and added diff lines pass.
- [x] If `ssh rvv` correctness is claimed, evidence includes counts `0`, `1`,
      a VL-boundary size, a tail size, and a multi-chunk size with scalar
      oracle, source preservation, and output tail preservation.

## Out Of Scope

- No `widening_product_reduce_dequant_clamp_f32` fused route expansion.
- No broad dequant, dtype, SEW, LMUL, or policy matrix.
- No high-level Linalg, Vector, StableHLO, or source frontend work.
- No per-Linalg route authority or one-intrinsic wrapper dialect.
- No performance, autotuning, tuning database, dashboard, or readiness state.
- No common EmitC invention of dequant or clamp semantics.
- No route-string, artifact-name, ABI-string, test-name, descriptor, manifest,
  script, or construction-protocol metadata authority.

## Evidence Plan

- Inspect existing RVV dialect op definitions, selected-body realization,
  route-family planning, provider preflight, target artifact validation,
  generated-bundle script support, and nearby fixtures for
  `dequant_clamp_f32_epilogue`.
- If production support is incomplete, implement the missing provider-owned
  fact surface or target validation closure in the smallest module boundary
  that satisfies the requirements.
- Run focused build targets for changed C++ libraries/tests.
- Run focused `tcrv-opt` and `tcrv-translate` commands for the
  `pre-realized-selected-body-artifact-dequant-clamp-f32-epilogue` fixture.
- Run direct C++ plugin/target artifact tests for provider/target contracts.
- Run generated-bundle dry-run for `--op-kind dequant_clamp_f32_epilogue`; run
  real `ssh rvv` only if the script path is already production-ready or the
  required repair is minimal and within this bounded owner.
- Run f32 clamp/select regression if shared route/target validation code is
  touched.

## Technical Notes

- Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/guides/index.md` plus the capability, plugin locality, and
  compute-boundary guides.
- Relevant prior tasks read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-f32-clamp-select-artifact-closure/prd.md`,
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-f32-clamp-select-executable-abi-closure/prd.md`, and
  `.trellis/tasks/archive/2026-06/06-06-extension-plugin-construction-protocol-common-repair/prd.md`.
- Primary code owners to inspect:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/`,
  `lib/Plugin/RVV/Construction/`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and nearby target/dialect
  fixtures.

## Completion Evidence

- Production inspection found the route already implemented in the RVV
  provider/realization/target stack:
  `tcrv_rvv.typed_dequant_clamp_f32_epilogue_pre_realized_body` verifies
  op kind, dequant relation, scale role, bounds, ABI roles, and SEW/LMUL/policy;
  RVV selected-body realization lowers it to setvl/load/dequantize/splats,
  two compare/select clamp stages, and store; RVV provider preflight builds a
  computed-mask select route-family plan carrying source load, convert, scale,
  lower/upper bound, result layout, ABI, header, C type, and provider mirror
  facts; target artifact validation mirrors those facts and rejects stale
  consumer metadata.
- The implementation change is focused test/evidence closure in
  `test/Target/RVV/pre-realized-selected-body-artifact-dequant-clamp-f32-epilogue.mlir`.
  It adds positive route/header assertions for route operand binding,
  computed-mask select route family, required headers, C type mapping,
  lower/upper bound C types, source vector facts, dequant convert/scale
  intrinsics, dequant scale role/type/name, and provider mirror.
- The same fixture now adds fail-closed mutations for stale provider mirror,
  ABI order, operand binding, selected route family, required headers,
  C type mapping, lower bound role, upper bound C type, source load intrinsic,
  dequant convert intrinsic, dequant scale intrinsic, dequant scale role,
  dequant scale C type, and dequant scale name.
- No production C++/TableGen/Python code was changed, so common EmitC/export
  neutrality and existing f32 clamp/select behavior are preserved by
  construction. The f32 clamp/select regression gate is not applicable to this
  round because shared compare/select or target validation code was not touched.
- `ssh rvv` correctness is not claimed for this task. The generated-bundle ABI
  path was validated with `--dry-run`; real remote execution remains the exact
  next continuation if executable behavior is requested.
