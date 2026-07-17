# Stage2 RVV widening product reduce dequant-clamp f32 artifact foundation

## Goal

Establish a truthful route-supported artifact foundation for the
`widening_product_reduce_dequant_clamp_f32` RVV boundary:

```text
selected tcrv.exec RVV variant
  -> explicit or pre-realized typed tcrv_rvv body carrying widening product,
     reduction/accumulator, dequant, clamp, dtype/config, runtime ABI, VL, and
     result-layout facts
  -> RVV plugin-owned selected-body realization and route planning
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact validation and generated-bundle evidence
```

If generated-bundle execution is already reachable without broadening scope,
include real `ssh rvv` evidence. Otherwise finish route-supported plus target
validation/generated-bundle dry-run closure and record executable ABI as the
exact continuation.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV widening product reduce dequant-clamp f32 artifact foundation`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `0050a9d6 rvv: close dequant clamp epilogue executable abi evidence`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created before source edits.
- The previous `dequant_clamp_f32_epilogue` artifact task closed provider,
  target/header mirror, stale-mirror, and generated-bundle dry-run evidence for
  a bounded dequant+clamp epilogue route.
- The previous `dequant_clamp_f32_epilogue` executable ABI task proved real
  `ssh rvv` execution with ABI order `lhs, scale, lower_bound, upper_bound,
  out, n`, signed lane coverage, scalar oracle, source preservation, output
  tail preservation, and counts `0,1,16,17,257`.
- The construction protocol repair task aligned route construction proof with
  current RVV route mnemonics including
  `widening_product_reduce_dequant_clamp_f32`, but construction-protocol
  metadata is not route support or artifact authority.
- Specs require RVV route facts to be derived or validated by the RVV plugin
  from typed `tcrv_rvv` body/config/runtime facts. Common EmitC/export and
  target artifacts may only consume and mirror provider-built facts.

## Requirements

- Establish, or prove already existing, production support for
  `widening_product_reduce_dequant_clamp_f32` from typed selected/pre-realized
  RVV body through `TCRVEmitCLowerableRoute` and target artifact validation.
- Validate widening product operand roles, product/reduction/accumulator
  semantics, reduction layout, dequant scale/zero or equivalent parameters,
  lower/upper clamp bound roles, operation kind, source/result dtype and C type
  mapping, SEW/LMUL/policy, runtime ABI order, operand binding, VL placement,
  result layout, route-family plan, required header/intrinsic mirrors, and
  explicit `provider_supported_mirror` from provider-owned facts.
- Fail closed for stale or missing widening product facts, reduction or
  accumulator layout, dequant parameters, clamp roles, operation kind,
  source/result dtype/config, runtime ABI/order, operand binding, result
  layout, selected route family, header/type/intrinsic mirrors, and provider
  mirror.
- Keep selected-body realization and route planning in the RVV plugin. Common
  EmitC/export must remain neutral and must not infer product, reduction,
  dequant, or clamp semantics from route strings, artifact names, ABI names,
  test names, descriptors, manifests, construction-protocol metadata, or
  scripts.
- Preserve existing `dequant_clamp_f32_epilogue` and `f32_clamp_select`
  behavior if shared provider, target, or script code is touched.

## Acceptance Criteria

- [x] Positive route evidence shows
      `widening_product_reduce_dequant_clamp_f32` support from typed selected
      or pre-realized RVV body through `TCRVEmitCLowerableRoute`.
- [x] Positive target/header evidence shows artifact export mirrors
      provider-derived widening product, reduction/accumulator, dequant, clamp,
      dtype/config, runtime ABI, operand binding, route-family, header/type,
      intrinsic, result-layout, and `provider_supported_mirror` facts.
- [x] Negative checks cover stale or missing widening product roles,
      reduction/accumulator layout, dequant parameters, lower/upper clamp
      roles, operation kind, source/result dtype/config, runtime ABI order,
      route operand binding, result layout, selected route-family plan,
      header/type mapping, intrinsic mirrors, and provider mirror.
- [x] Non-consumer or cross-family routes carrying stale
      `widening_product_reduce_dequant_clamp_f32` mirrors fail closed.
- [x] Common EmitC/export remains neutral and no new common RVV semantic branch
      is introduced.
- [x] Focused `tcrv-opt`/`tcrv-translate`, RVV plugin, target artifact/export,
      and generated-bundle dry-run checks run as applicable.
- [x] `dequant_clamp_f32_epilogue` and `f32_clamp_select` regressions run if
      shared code is touched.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation, and
      a bounded old-authority scan over touched files and added diff lines
      pass.
- [x] If `ssh rvv` correctness is claimed, evidence includes counts `0`, `1`,
      a VL-boundary size, a tail size, and a multi-chunk size with scalar
      oracle, source preservation, and output tail preservation.

## Completion Evidence

- Existing production RVV plugin/target code already supported the bounded
  fused route. This round tightened route-supported and target artifact
  evidence in
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32.mlir`
  and added generated-bundle evidence in
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequant-clamp-f32-dry-run.test`.
- Positive route facts now check selected body operation,
  typed compute chain, ABI order `lhs,rhs,acc,scale,lower_bound,upper_bound,out,n`,
  route operand binding, contraction route family plan, target leaf profile,
  provider support mirror, header declarations, C type mapping, source/product/
  accumulator/result dtype and LMUL facts, reduction layouts, product/reduction
  intrinsics, dequant relation and scale facts, clamp roles, predicate/select
  layout, and clamp/select intrinsics.
- Negative target-export checks now stale-mutates route operand binding,
  required headers, C type mapping, target leaf profile, accumulator layout,
  result layout, product C type, widening product intrinsic, widening reduction
  intrinsic, dequant conversion intrinsic, dequant scale intrinsic, lower-bound
  C type, bound splat intrinsic, ABI order, provider mirror, dtype chain,
  policy, scale role, lower/upper roles, reduction relation, and old route-id
  authority. Each stale mirror fails closed through target validation or
  selected-body realization.
- Generated-bundle dry-run evidence checks the pre-realized selected-body input,
  prototype, runtime ABI, route operand binding, route family, provider mirror,
  required headers/type mapping, widening product/reduction facts, dequant facts,
  clamp facts, `provider_route_facts`, product-reduction target validator
  consumption, and harness oracle/tail-preservation coverage.
- Direct `ssh rvv` execution succeeded:
  `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --op-kind widening_product_reduce_dequant_clamp_f32 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257`
  returned `rvv_generated_bundle_abi_e2e: success` with artifact directory
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/20260605T191743Z`.
- Real RVV output covered counts `0,1,16,17,257`, two signed-lane patterns,
  runtime scales `-0.125` and `0.375`, lower/upper clamp pairs
  `-1.5:2.25` and `-8:-0.75`, scalar oracle comparison, below/inside/above
  clamp cases, source preservation, accumulator preservation, and output tail
  preservation.
- A neighboring `f32_clamp_select` target artifact fixture had the same
  order-sensitive `PLAN-SAME`/header fact issue. It was self-repaired by
  matching provider metadata in emitted order and converting header fact checks
  to order-independent existence checks. No production C++ or Python logic was
  changed in this round.

## Out Of Scope

- No broad reduction, matmul, dtype, SEW, LMUL, or policy matrix.
- No high-level Linalg, Vector, StableHLO, or source frontend work.
- No per-Linalg route authority or one-intrinsic wrapper dialect.
- No performance, autotuning, tuning database, dashboard, or readiness state.
- No common EmitC invention of widening product, reduction, dequant, or clamp
  semantics.
- No route-string, artifact-name, ABI-string, test-name, descriptor, manifest,
  script, or construction-protocol metadata authority.
- No compatibility wrappers preserving old legacy i32 route authority.

## Evidence Plan

- Inspect existing RVV dialect op definitions, selected-body realization,
  route planning, provider preflight, target artifact validation,
  generated-bundle script support, and nearby fixtures for
  `widening_product_reduce_dequant_clamp_f32`.
- If production support is incomplete, implement the missing provider-owned
  typed-body, realization, route planning, target validation, or
  generated-bundle closure in the smallest bounded module owner that satisfies
  the requirements.
- Run focused build targets for changed C++ libraries/tests.
- Run focused `tcrv-opt` and `tcrv-translate` commands for the
  `widening_product_reduce_dequant_clamp_f32` fixture or the closest fixture
  created in this task.
- Run direct C++ plugin/target artifact tests for provider/target contracts.
- Run generated-bundle dry-run for
  `--op-kind widening_product_reduce_dequant_clamp_f32`; run real `ssh rvv`
  only if the generated-bundle path is production-ready or the required repair
  is minimal and within this bounded owner.
- Run `dequant_clamp_f32_epilogue` and `f32_clamp_select` regressions if shared
  provider, target, compare/select, or script code is touched.

## Technical Notes

- Specs read before implementation:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Relevant prior tasks read:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-dequant-clamp-f32-epilogue-artifact-foundation/prd.md`,
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-dequant-clamp-f32-epilogue-executable-abi/prd.md`,
  and
  `.trellis/tasks/archive/2026-06/06-06-extension-plugin-construction-protocol-common-repair/prd.md`.
- Primary code owners to inspect:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/`,
  `lib/Plugin/RVV/Construction/`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and nearby widening product
  reduce dequant-clamp fixtures.
