# Stage2 RVV runtime scalar splat-store route repair/closure

## Goal

Close the bounded Stage 2 RVV runtime scalar splat-store route boundary. A
selected `tcrv.exec` RVV variant must carry either an explicit typed
`tcrv_rvv` body or a pre-realized selected body where one runtime scalar is
splatted into a typed vector and stored to the output buffer under runtime
`n`/AVL control:

```text
runtime_scalar_splat_store:
  for i in [0, n):
    out[i] = rhs_scalar
```

The accepted route must flow through RVV plugin-owned selected-body
realization and route planning, provider-built `TCRVEmitCLowerableRoute`,
Common EmitC neutral materialization, target artifact validation, and focused
evidence. Unsupported or stale scalar role, output/n ABI, splat semantics,
typed config, VL/store layout, route-family facts, or provider mirrors must
fail closed before route construction or target artifact acceptance.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV runtime scalar splat-store route repair/closure`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `829e122b rvv: close strided memory route foundation`.
- No `.trellis/.current-task` existed at the start of the round, so this task
  was created before source edits.
- Specs require this route to derive scalar runtime binding, output/n ABI
  order, splat semantics, element dtype, SEW/LMUL/policy, VL placement, store
  layout, route-family facts, operand binding, and `provider_supported_mirror`
  from selected typed `tcrv_rvv` body/config/runtime facts and RVV provider
  legality. Common EmitC/export may only carry provider-built route payloads
  and mirrors.
- Focused reproduction of the named failing test from `build/test` fails in
  FileCheck before source edits:

  ```text
  TianChenRV :: Conversion/EmitC/rvv-generic-stage2-runtime-scalar-splat-store-negative.mlir
  ```

  The first `CHECK` expects the stale verifier text without `const float *`,
  while the current verifier diagnostic includes `const float *` as an
  accepted `lhs-input-buffer` C type. This classifies the named failure as
  stale expectation, not immediate evidence of production route acceptance.
- Current source inspection shows production splat-store surfaces:
  `typed_runtime_scalar_splat_store_pre_realized_body`, explicit
  `tcrv_rvv.splat` + `tcrv_rvv.store`, runtime scalar splat-store route-family
  plan validation, typed config mirror checks, materialization leaf checks,
  residual operand binding checks, route-control provider plan checks,
  statement plan checks, target artifact header mirrors, and generated-bundle
  dry-run coverage.
- Existing positive target fixtures already assert runtime ABI order
  `rhs_scalar,out,n`, memory form `runtime-scalar-splat-store`, route operand
  binding, route-family plan, target leaf profile, required headers/type
  mapping, and `provider_supported_mirror` for explicit and pre-realized
  splat-store paths.
- Existing generated-bundle dry-run coverage for the pre-realized path checks
  pre-realized body consumption, provider route facts, runtime scalar RHS
  values, count set `0,1,16,23,257`, scalar oracle use, and output tail
  sentinel preservation.

## Requirements

- Keep scope to the runtime scalar splat-store route family only:
  scalar runtime RHS -> vector splat -> unit-stride output store.
- Preserve the authority chain:
  selected typed/pre-realized `tcrv_rvv` body/config/runtime facts -> RVV
  selected-body realization/validation -> RVV route-family provider facts ->
  lowerable route -> Common EmitC/export -> target artifact mirrors ->
  optional generated-bundle runtime evidence.
- The provider/target route path must structurally carry or derive:
  - scalar runtime ABI role `rhs-scalar-value`;
  - output buffer ABI role `output-buffer`;
  - runtime count ABI role `runtime-element-count`;
  - runtime ABI order `rhs_scalar,out,n`;
  - typed scalar/vector element type, SEW, LMUL, policy, vector C type, scalar
    splat intrinsic, setvl intrinsic, and store intrinsic;
  - VL loop control and runtime AVL/VL plan;
  - memory form `runtime-scalar-splat-store`;
  - route operand-binding plan/summary with exported header/prototype mirror
    participation;
  - route-family plan, target leaf profile, required headers/type mapping,
    and `provider_supported_mirror`.
- Common EmitC/export may only carry provider-built route payloads and metadata
  mirrors. It must not infer scalar splat semantics, dtype/config, ABI roles,
  intrinsic spelling, route support, or store layout.
- Stale or missing facts must fail closed before provider route construction or
  target artifact acceptance: scalar role, output role, runtime count role,
  ABI order, splat/store dataflow, typed config, VL/store layout, operand
  binding, route-family plan, target header/type mirrors, and
  `provider_supported_mirror`.
- Resolve the known negative-test drift by aligning fixture evidence to the
  current provider/verifier truth. Do not count the stale expectation tweak as
  complete unless focused negative evidence covers the missing route boundary
  cases named above.
- Do not merge unrelated construction-protocol-common or template extension
  plugin failures into this owner.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` describe this bounded
      runtime scalar splat-store route repair/closure and cite the relevant
      specs.
- [x] The named failing negative test is reproduced before repair and the root
      cause is recorded as stale expectation or production legality drift.
- [x] If production support is already real, the negative fixture is repaired
      and extended with focused fail-closed coverage for stale or missing
      scalar role, output/n ABI, ABI order, splat semantics, dtype/config,
      VL/store layout, route-family facts, route operand binding, and
      `provider_supported_mirror` as applicable to the Conversion/EmitC layer.
- [x] Positive route/materialization evidence proves provider-derived scalar
      splat, store, runtime ABI order, memory form, typed config,
      route-family plan, operand binding, target leaf profile, and
      provider mirror.
- [x] Common EmitC/export remains neutral and does not choose scalar splat
      semantics.
- [x] Focused `tcrv-opt` / `tcrv-translate` / RVV provider / target artifact /
      generated-bundle checks for changed behavior pass as applicable.
- [x] If executable behavior is claimed, generated artifact plus real
      `ssh rvv` correctness passes over counts including `0`, `1`, a VL
      boundary, a tail case, and a multi-chunk case, with scalar oracle and
      tail/sentinel preservation across at least two runtime scalar values.
      Not applicable: this round did not add a new executable correctness
      claim beyond existing generated-bundle dry-run evidence.
- [x] If executable closure is too large, finish route-supported plus
      target-validation closure and record generated-bundle `ssh rvv` as the
      exact next continuation point. Not applicable: no new ssh-rvv executable
      claim was made.
- [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, dtype-prefixed
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor, source-front-door,
      source-artifact, route-string/artifact-name/ABI-string/test-name,
      exact-intrinsic-spelling, or Common EmitC semantic authority.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation,
      final clean worktree, archive, and one coherent commit complete the
      round if the module behavior is complete.

## Evidence Plan

- Run focused lit reproduction for
  `Conversion/EmitC/rvv-generic-stage2-runtime-scalar-splat-store-negative.mlir`
  from `build/test`.
- Inspect splat-store provider/route/target/generated-bundle surfaces before
  deciding whether production code or fixture evidence needs repair.
- Repair stale expectation and add focused negative slices if current
  production support is already provider-owned.
- Run focused positive materialization and target header fixtures for explicit
  and pre-realized splat-store paths.
- Run generated-bundle dry-run for pre-realized splat-store; run real `ssh rvv`
  only if the round claims executable correctness beyond existing dry-run
  evidence.

## Definition Of Done

- Current HEAD either has route-supported typed runtime scalar splat-store
  provider and target artifact validation closure, or records the precise
  production blocker and next continuation point without false executable
  claims.
- Any implementation changes are production-path changes or focused evidence
  for already-existing production support; helper/test/spec-only work is not a
  completion unless production support was already real.
- Specs are updated only if this round discovers a durable rule not already
  captured in `.trellis/spec/`.
- The task is finished/archived and one coherent commit is created when the
  module behavior is complete.

## Out Of Scope

- Broad suite cleanup or unrelated failing tests.
- Construction protocol common, template extension plugin, Scalar dialect
  expansion, IME, Offload, high-level Linalg/Vector/StableHLO frontends,
  performance/autotuning, and broad dtype/LMUL matrices.
- One-intrinsic wrapper routes.
- Descriptor-driven computation or Common EmitC semantic inference.
- Treating route ids, artifact names, ABI strings, test names, intrinsic
  spellings, status fields, manifests, or metadata mirrors as route authority.

## Technical Notes

Specs read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/index.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/index.md`
- `.trellis/spec/testing/mlir-testing-contract.md`

Relevant archived task read:

- `.trellis/tasks/archive/2026-06/06-06-rvv-stage2-strided-memory-selected-body-route-foundation/prd.md`
- `.trellis/tasks/archive/2026-06/06-06-rvv-stage2-strided-memory-selected-body-route-foundation/check.jsonl`

Primary inspection surfaces from the direction brief:

- `test/Conversion/EmitC/rvv-generic-stage2-runtime-scalar-splat-store-negative.mlir`
- `test/Conversion/EmitC/rvv-generic-stage2-runtime-scalar-splat-store-materialization.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-splat-store.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-splat-store.mlir`
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-splat-store-dry-run.test`
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`

## Completion Evidence

Completed as a bounded Stage 2 runtime scalar splat-store route repair and
evidence closure. The named suite failure was reproduced before repair and
classified as stale fixture expectation: the current verifier diagnostic for a
bad `lhs-input-buffer` C type now includes `const float *`, while the fixture
still expected only integer pointer forms. Production route support was already
provider-owned in the inspected path: explicit and pre-realized selected bodies
materialize through `tcrv_rvv.setvl`, `tcrv_rvv.splat`, and `tcrv_rvv.store`;
RVV route planning validates the runtime scalar splat-store family plan,
typed config, materialization leaf facts, route-control provider plan,
residual operand binding, runtime ABI roles/order, statement plan, stale
non-splat facts, and route description mirrors before constructing
`TCRVEmitCLowerableRoute`.

Focused fixture changes:

- `test/Conversion/EmitC/rvv-generic-stage2-runtime-scalar-splat-store-negative.mlir`
  now matches the current verifier diagnostic and adds fail-closed negative
  slices for wrong runtime ABI operation order, SEW/config mismatch, and
  store/VL ownership mismatch, in addition to the existing scalar role,
  output role, runtime count role, and binary fallback rejection checks.
- `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-splat-store.mlir`
  now asserts pre-realized target header and emission-plan mirrors for runtime
  ABI parameters/order, runtime control plan, provider support mirror,
  required headers, and route type mapping.
- `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-splat-store-dry-run.test`
  now checks generated-bundle evidence JSON for target leaf profile,
  provider support mirror, required headers, and C type mapping.

Provider and target route facts proved:

- Operation: `runtime_scalar_splat_store`.
- Typed compute op: `tcrv_rvv.splat`.
- Runtime ABI order: `rhs_scalar,out,n`.
- Memory form: `runtime-scalar-splat-store`.
- Route family plan:
  `rvv-runtime-scalar-splat-store-route-family-plan.v1`.
- Route operand binding plan:
  `rvv-route-operand-binding:runtime_scalar_splat_store.v1`.
- Target leaf profile:
  `rvv-v1-typed-runtime-scalar-splat-store-leaf-profile.v1`.
- Provider mirror:
  `provider_supported_mirror:rvv-runtime-scalar-splat-store-plan-validated`.
- Required header/type facts:
  `stddef.h,stdint.h,riscv_vector.h` and
  `vl:size_t,rhs_scalar:typed-scalar,result:typed-vector`.

Checks run:

- [OK] Focused lit reproduction before repair for
  `Conversion/EmitC/rvv-generic-stage2-runtime-scalar-splat-store-negative.mlir`
  failed with stale `const float *` diagnostic expectation.
- [OK] Focused lit for repaired
  `Conversion/EmitC/rvv-generic-stage2-runtime-scalar-splat-store-negative.mlir`.
- [OK] Focused lit for
  `Conversion/EmitC/rvv-generic-stage2-runtime-scalar-splat-store-materialization.mlir`.
- [OK] Focused lit for explicit and pre-realized target fixtures:
  `Target/RVV/explicit-selected-body-artifact-runtime-scalar-splat-store.mlir`
  and
  `Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-splat-store.mlir`.
- [OK] Focused lit for generated-bundle dry-run:
  `Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-splat-store-dry-run.test`.
- [OK] `build/bin/tianchenrv-target-artifact-export-test`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [EXPECTED UNRELATED FAIL] `cmake --build build --target check-tianchenrv -j2`
  reported `492/494` passed. The original runtime scalar splat-store negative
  failure is resolved; remaining failures are the brief-excluded
  `Plugin/construction-protocol-common.test` and
  `Plugin/template-extension-plugin.test`.
- [OK] `git diff --check`.
- [OK] `git diff --cached --check`.
- [OK] Trellis task context validation.
- [OK] Bounded added-line old-authority scan over touched test files had no
  matches.

No `.trellis/spec/` update was needed: the existing RVV plugin, EmitC route,
variant-pipeline, and MLIR testing contracts already captured the durable
rule that scalar splat-store support is provider-owned and mirror-only after
route construction.
