# Stage2 RVV resource-selected widening product-reduce dequant-clamp f32 executable artifact ABI boundary

## Goal

Make the existing resource-selected
`widening_product_reduce_dequant_clamp_f32` selected-body route executable as
a generated RVV artifact with truthful ABI/runtime evidence, or fail closed at
the exact executable artifact boundary if any selected-body product/reduction
chain fact, dequant scale, clamp bound, compare/select policy, dtype/config
fact, ABI/header mapping, runtime AVL/VL fact, resource-selection fact, or
provider mirror is missing or stale.

This round is the clamp follow-up to the archived
`widening_product_reduce_dequantize_f32` executable evidence task. The new risk
surface is lower/upper scalar bound ordering, clamp compare/select semantics,
provider-owned operand binding, generated header/prototype ordering, and
runtime correctness evidence for explicit and pre-realized selected bodies.

## What I Already Know

- The repo started this round on `main` with a clean worktree and HEAD
  `6877677e rvv: prove product dequant executable abi evidence`.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief before source edits.
- The global spec requires the RVV-first authority chain:
  `tcrv.exec` envelope -> selected typed `tcrv_rvv` body -> RVV provider ->
  `TCRVEmitCLowerableRoute` -> Common EmitC -> target artifact -> `ssh rvv`
  evidence for runtime/correctness claims.
- Common EmitC must carry provider payloads neutrally. It must not infer RVV
  dtype, SEW, LMUL, policy, operation kind, ABI, or intrinsic spelling from
  route ids, artifact names, parameter names, descriptors, or metadata.
- Operand-binding summaries must be provider-derived, cover every exported
  runtime ABI parameter, and mark header/prototype participation with `hdr`.
- Runtime AVL/VL facts must be represented as provider-owned selected-boundary
  facts before target artifact acceptance.
- Low-precision contraction resource-selection facts must come from typed
  selected-body/config/runtime and target capability facts, not route tokens,
  artifact names, ABI strings, or test names.
- The previous archived dequantize-f32 task concluded that production code
  already had provider-owned/resource-selected dequantize-f32 behavior, and it
  closed the exact blocker with explicit and pre-realized generated-bundle
  `ssh rvv` evidence.

## Requirements

- The production path for `widening_product_reduce_dequant_clamp_f32` must flow
  from selected/pre-realized typed `tcrv_rvv` facts through RVV-owned resource
  selection, contraction route facts, `TCRVEmitCLowerableRoute`, Common EmitC,
  target artifact export, generated bundle ABI, and focused correctness
  evidence.
- Clamp lower and upper bound ABI entries must be provider-derived, exported in
  the generated prototype/header in the intended order, and consumed by the
  generated body as lower/upper clamp limits rather than mirror-only metadata.
- Product/reduction, dequant scale, compare/select clamp relation, source dtype,
  result dtype, accumulator layout, output, runtime count/AVL/VL, headers,
  type mappings, and provider support mirrors must be validated as provider
  facts before artifact acceptance.
- If inspection shows the production seam is still dry-run-only, stale, or
  under-validated, repair the smallest production owner boundary rather than
  adding report-only evidence.
- If production behavior is already complete, name the exact evidence blocker
  and close it with focused non-dry-run explicit and pre-realized `ssh rvv`
  generated-bundle evidence for clamp behavior.
- Keep fail-closed evidence for at least one stale or missing executable
  boundary fact relevant to the clamp seam, preferably swapped/stale clamp
  bounds, stale binding summary, wrong ABI mapping, or stale provider mirror.
- Do not add broad dtype/LMUL clone batches, unrelated MAcc/mask work, high
  level frontend routes, source-front-door positive routes, descriptor-driven
  compute, common EmitC RVV semantics, or performance/tuning database work.

## Acceptance Criteria

- [x] PRD and task context files capture the bounded module owner and relevant
      specs before implementation.
- [x] Repository inspection identifies whether the dequant-clamp executable
      path needs production code changes or only focused executable evidence,
      with a concrete no-source-change justification if applicable.
- [x] Explicit selected-body `widening_product_reduce_dequant_clamp_f32`
      reaches materialized selected boundary, emission plan, target artifact
      export, generated bundle compile, and `ssh rvv` correctness evidence when
      executable behavior is claimed.
- [x] Pre-realized selected-body `widening_product_reduce_dequant_clamp_f32`
      reaches the same evidence chain.
- [x] Positive correctness evidence covers counts, input patterns, dequant
      scales, lower/upper bound pairs, source preservation, accumulator
      preservation, output correctness, and tail preservation.
- [x] Focused fail-closed evidence covers at least one stale or missing clamp
      artifact/ABI/provider fact.
- [x] Focused C++ targets and relevant generated-bundle dry-run tests pass:
      `build/bin/tianchenrv-rvv-extension-plugin-test`,
      `build/bin/tianchenrv-target-artifact-export-test`, and the explicit and
      pre-realized dequant-clamp generated-bundle dry-run tests.
- [x] Bounded old-authority scan over touched files and added diff lines shows
      no new descriptor/source-front-door/direct-C/common-EmitC semantic
      authority and no legacy `i32m1` route-authority growth.
- [x] `git diff --check`, `git diff --cached --check`, and final `git status`
      are clean before commit.

## Definition Of Done

- Production source is changed only if the executable artifact/ABI seam is
  actually incomplete or under-validated.
- Task state and journal record the evidence and any no-source-change
  justification.
- The task is finished/archived according to repo Trellis convention.
- One coherent commit records the PRD/context, code/test/evidence updates, and
  journal/task bookkeeping.

## Out Of Scope

- Broad product/reduction matrix expansion.
- Dtype, LMUL, EMUL, or source-shape clone batches.
- New source-front-door positive routes.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Performance tuning databases, dashboards, or report-only work.
- Mass rewrites of memory, segment2, reduction, compare/select, conversion, or
  mask routes outside this dequant-clamp artifact seam.

## Technical Notes

- Relevant specs read first:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Previous archive read first:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-resource-selected-dequantize-f32-executable-abi/`.
- Primary code and test files to inspect:
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/RVVElementwiseSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-widening-product-reduce-dequant-clamp-f32-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequant-clamp-f32-dry-run.test`,
  `test/Target/RVV/explicit-selected-body-realization-widening-product-reduce-dequant-clamp-f32.mlir`, and
  `test/Target/RVV/pre-realized-selected-body-artifact-widening-product-reduce-dequant-clamp-f32.mlir`.

## Completion Notes

- No production source change was required in this round. Current HEAD already
  derives the product/reduction/dequant-clamp route from typed selected-body
  facts and RVV provider-owned route-family plans:
  - `RVVEmitCContractionRouteFamilyPlanOwners.cpp` validates the explicit and
    pre-realized `widening_product_reduce_dequant_clamp_f32` body signature,
    source/product/accumulator/result config, dequant scale role, lower/upper
    bound roles, bound order, clamp select layout, policy, and ABI C types.
  - `RVVEmitCRoutePlanning.cpp` requires the product-reduction-dequant-clamp
    contraction plan before math operand binding, binds scale/lower/upper/out/n
    through provider-derived operand facts, rejects missing bound `hdr` uses,
    and requires a selected legal low-precision resource candidate before
    route construction.
  - Target artifact validation mirrors provider facts for runtime ABI order,
    route operand bindings, header/type mapping, provider support, product and
    dequant facts, clamp lower/upper role/type/order/relation, and resource
    selection before accepting the bundle.
  - `scripts/rvv_generated_bundle_abi_e2e.py` extracts the generated C ABI,
    rejects loop-local clamp/dequant placement, verifies the post-loop
    dequant-clamp chain, builds an external harness from the generated header
    and object only, and checks source, accumulator, and output tail
    preservation.
- Focused fail-closed evidence already exists in the explicit and pre-realized
  target fixtures for missing scale, missing/swapped lower/upper roles,
  unsupported policy, stale route authority, stale provider mirror, stale ABI
  order, stale binding summary, stale header/type mapping, stale leaf profile,
  stale product/reduction/dequant/clamp facts, and stale bound splat/type
  mirrors.
- Stable evidence artifacts produced in this round:
  - Explicit dry-run:
    `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-dequant-clamp-explicit-dry/evidence.json`
  - Pre-realized dry-run:
    `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-dequant-clamp-prerealized-dry/evidence.json`
  - Explicit `ssh rvv` correctness:
    `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-dequant-clamp-explicit-ssh/evidence.json`
  - Pre-realized `ssh rvv` correctness:
    `artifacts/tmp/rvv_generated_bundle_abi_e2e/stage2-product-dequant-clamp-prerealized-ssh/evidence.json`
- Both non-dry-run evidence paths report `status=success`,
  `ssh_evidence=true`, runtime counts `0,1,16,17,257`, input patterns `0,1`,
  scale values `-0.125,0.375`, ordered bound pairs `-1.5:2.25,-8:-0.75`,
  f32 tolerance `1e-05`, and `source_preserved accumulator_preserved
  tail_preserved`.
- Spec update judgment: no `.trellis/spec/` change is needed. Existing RVV
  plugin, EmitC route, testing, and variant-pipeline specs already capture the
  provider-owned resource-selection, operand-binding, clamp ABI, runtime
  AVL/VL, fail-closed mirror, and `ssh rvv` evidence contracts exercised here.

## Checks Run

- `ninja -C build tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `lit -sv . --filter widening-product-reduce-dequant-clamp-f32` from
  `build/test`: 4/4 filtered tests passed.
- Explicit and pre-realized generated-bundle dry-runs for
  `widening_product_reduce_dequant_clamp_f32`.
- Explicit and pre-realized generated-bundle `ssh rvv` correctness runs for
  `widening_product_reduce_dequant_clamp_f32`.
