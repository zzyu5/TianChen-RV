# RVV production-kernel capability campaign

## Goal

Promote the current RVV work from repeated generated-bundle evidence closeouts
to a macro production-kernel capability campaign. The campaign covers
Gearbox/resource-aware selected-body realization, low-precision contraction
primitive surface, generated artifact/runtime evidence only when new
production behavior is claimed, and measured same-target comparison for
production-like RVV kernels.

This task remains open across rounds until the campaign gates below are met.
This round continues Gate 2 with low-precision widening-reduction primitive
facts, making the product-reduction / `vwredsum`-style path consume typed RVV
source/product/accumulator facts, provider-owned intrinsics/layouts, exact
target mirrors, and neutral EmitC materialization.

## Direction Brief Source

Hermes first selected:

```text
RVV production-kernel capability campaign: low-precision primitive surface first slice
```

The brief explicitly redirects away from adjacent standalone reduction evidence
tasks and no-production-source generated-bundle closeouts. The current slice
must change production compiler surface in the RVV dialect/plugin/provider or
target validation path unless live repository evidence proves a narrower
production blocker.

Hermes then continued the same macro task with:

```text
RVV production-kernel capability campaign: Gate 2 unsigned low-precision primitive and vwredsum validation slice
```

The second slice must keep this macro task active, make the unsigned u8
widening-product boundary explicit in the production provider/target contract,
and deepen product-reduction/`vwredsum` accumulator/result validation facts
without introducing q8/q4 route authority or metadata-derived semantics.

Hermes now continues the same macro task with:

```text
RVV production-kernel capability campaign: Gate 2 accepted unsigned u8
widening-product provider/target facts
```

This continuation must keep the macro task active and implement the bounded
unsigned route-supported path if production facts are complete: unsigned vector
C type/header mapping, unsigned u8/u16 load/store leaves, `vwmulu.vv` /
`__riscv_vwmulu_vv_u16mf2` intrinsic mapping, provider route-description
facts, target mirror validation, and focused fail-closed tests for stale or
missing mirrors.

Hermes now continues the same macro task with:

```text
RVV production-kernel capability campaign: Gate 2 low-precision
widening-reduction primitive facts
```

This continuation keeps the macro task active after the accepted unsigned u8
widening-product slice. The current round must advance product-reduction /
`vwredsum`-style primitive facts that consume typed signed i8 source, i16
product, i32 accumulator/result, and the existing widening-product facts. The
slice should keep the facts provider-owned, expose a bounded target validation
contract, and reject stale dtype/sign/SEW/LMUL/accumulator/reduction mirrors
before artifact acceptance.

## What I Already Know

- The repository had no active Trellis task at session start, so this macro task
  owns the Hermes direction rather than creating a neighboring evidence task.
- Recent journal entries show several completed RVV tasks closed existing
  generated-bundle/`ssh rvv` ABI seams without production source changes.
- `.trellis/spec/index.md`, `extension-plugins/rvv-plugin.md`,
  `lowering-runtime/emitc-route.md`,
  `variant-pipeline/generation-selection-tuning.md`, and
  `implementation-stack/supervision-loop.md` all require the production path to
  be typed `tcrv_rvv` body -> RVV plugin/provider -> common EmitC route ->
  target artifact, with metadata only as mirrors.
- Current code already has a generic typed RVV vector surface and some
  low-precision pieces:
  `!tcrv_rvv.vector<i8, "mf4">`, `!tcrv_rvv.vector<i16, "mf2">`,
  `tcrv_rvv.widening_product`, product-reduction/dequantize routes,
  contraction route-family plan owners, and target validation mirrors.
- The existing widening product surface is visibly bounded around signed i8:
  tests and provider strings reference `signed-i8mf4xi8mf4-to-i16mf2`,
  `widening_product_i8_i16`, and low-precision resource candidates using
  `i8mf4-i16mf2-i32m1`.
- The Hermes brief asks for typed `i8/u8` vector/config legality, `i8/u8`
  memory roles, widening product semantics, widened accumulation/reduction
  facts, and fail-closed provider/target validation derived from typed body and
  config facts.

## Campaign Gates

- [ ] Gate 1 Gearbox/resource-aware selected-body realization:
      build/prune/select/realize phases, resource facts, and
      provider-consumed plan or realized `tcrv_rvv` structure.
- [ ] Gate 2 Low-precision contraction primitive surface:
      typed i8/u8 vector/config, i8/u8 loads, i8*i8 or u8*u8 widening
      product, widening reduction or `vwredsum`-style provider route facts,
      and fail-closed validation.
- [ ] Gate 3 Generated artifact/runtime correctness evidence:
      only when executable behavior is newly claimed by production compiler
      path or is the named blocker.
- [ ] Gate 4 Measured same-target comparison path:
      production-like RVV kernels compared on the same target; llama.cpp-style
      q8/q4 kernels are pressure tests, not route authority.

## Current Round Milestone

Continue Gate 2 by adding the smallest production compiler submodule that makes
low-precision product-reduction / `vwredsum` facts explicit and consumable by
RVV provider and target validation. The slice starts after signed product facts
and accepted unsigned u8 widening-product facts are already present.

The preferred continuation slice is:

- Add or harden a provider-owned widening accumulation/reduction primitive fact
  surface for the bounded signed product-reduction chains: i8 source, i16
  product, i32 accumulator/result, signed widening product,
  `__riscv_vwredsum_vs_i16mf2_i32m1`, scalar seed splat, reduction store VL,
  accumulator layout, result layout, and product-reduction chain relation.
- Validate those facts before `TCRVEmitCLowerableRoute` construction by
  comparing the selected route description to canonical provider-owned facts,
  not by trusting route IDs, target artifact names, metadata, or intrinsic
  strings as authority.
- Thread the same primitive-reduction facts into target validation only as exact
  mirrors. Stale source/product/accumulator/result dtype, product SEW/LMUL,
  reduction intrinsic, scalar seed, or store-VL mirrors must fail closed.
- Keep Common EmitC neutral; it may carry provider payloads and metadata mirrors
  but must not derive `vwredsum`, accumulator dtype, result dtype, sign, SEW,
  LMUL, or resource facts.
- Do not claim runtime correctness/performance in this slice unless emitted
  runtime behavior changes and `ssh rvv` evidence is collected.

## Requirements

- Preserve the current architecture chain:
  `tcrv.exec` envelope -> selected typed `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/provider -> `TCRVEmitCLowerableRoute` ->
  common EmitC -> target artifact.
- Do not add q8/q4 route IDs, q8/q4 artifact names, llama.cpp wrappers,
  source-front-door authority, descriptor-driven computation, or new
  dtype-prefixed op namespaces.
- Keep low-precision primitive facts provider-owned and derived from typed
  body/config/capability/runtime facts.
- Keep Common EmitC neutral; it may carry provider payloads but must not infer
  RVV dtype, sign, SEW, LMUL, memory form, operation kind, intrinsic spelling,
  accumulator layout, or resource facts.
- Update focused dialect/provider/target tests for changed behavior.
- Keep this macro task active after this slice unless all campaign gates are
  complete.

## Acceptance Criteria For This Round

- [x] A production source diff lands in the RVV dialect/plugin/provider/target
      validation path, not only task/report/test evidence.
- [x] Product-reduction / `vwredsum` primitive facts are provider-owned and
      explicitly cover source/product/accumulator/result dtype, product
      SEW/LMUL, signed widening product relation, product-reduction chain
      relation, reduction intrinsic, scalar seed splat, accumulator/result
      layout, and store-VL facts.
- [x] Provider validation rejects stale or unsupported product-reduction facts
      before route construction.
- [x] Target artifact validation consumes the same primitive-reduction facts as
      exact mirrors and rejects stale source/product/accumulator/result dtype,
      product SEW/LMUL, reduction intrinsic, scalar seed, or store-VL mirrors.
- [x] Focused lit/C++ tests prove accepted product-reduction facts remain
      typed-fact gated and stale low-precision reduction facts fail closed.
- [x] Existing signed i8 product and unsigned u8 widening-product provider /
      target facts remain accepted.
- [x] A bounded old-authority scan over changed/added lines shows no new
      positive `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor, source-front-door, route-id, or artifact
      authority.
- [x] `git diff --check` and `git diff --cached --check` pass for this coherent
      slice.
- [x] The slice is committed coherently and final worktree status is verified
      clean in the round report.

## Current Round Result

Completed the Gate 2 low-precision widening-reduction primitive facts slice
while keeping the macro campaign open:

- Added provider-owned `RVVLowPrecisionWideningReductionPrimitiveFacts` for the
  bounded signed product-reduction chains. The facts cover source `i8/mf4`,
  product `i16/mf2`, accumulator/reduction result `i32/m1`, final result dtype,
  signed product relation, product-reduction chain relation, `vwmul`,
  `vwredsum`, scalar seed splat, accumulator/result layouts, and reduction
  store VL.
- Threaded the primitive facts through widening dot-reduce route facts and the
  provider validation contract so stale dtype, SEW/LMUL, intrinsic, relation,
  layout, or store-VL facts fail before `TCRVEmitCLowerableRoute`
  construction.
- Hardened target artifact validation so existing metadata is consumed only as
  exact mirrors of the provider-owned primitive facts. Stale
  `tcrv_rvv.low_precision_primitive.accumulator_dtype` and stale
  `tcrv_rvv.widening_reduction_intrinsic` mirrors now fail before artifact
  acceptance.
- Preserved the accepted signed i8 product-reduction and unsigned u8
  widening-product paths through the focused widening/product-reduction lit
  matrix and C++ target artifact checks.
- Updated the RVV plugin and EmitC route specs with the executable
  product-reduction primitive-fact contract.

This round does not claim runtime correctness/performance, so no `ssh rvv`
evidence was required. Gate 2 has advanced with provider/target-consumable
primitive-reduction facts, but the macro campaign is not complete: Gearbox/
resource-aware selected-body realization, runtime evidence for newly claimed
executable behavior, and measured same-target comparison remain open.

## Previous Round Result

Completed the Gate 2 accepted unsigned u8 widening-product provider/target
facts slice while keeping the macro campaign open:

- Removed the previous provider fail-closed boundary for the bounded unsigned
  u8 widening-product body. The accepted path is still typed-fact gated:
  `!tcrv_rvv.vector<ui8, "mf4">` sources, `!tcrv_rvv.vector<ui16, "mf2">`
  result, relation `unsigned-u8mf4xu8mf4-to-u16mf2`, and runtime ABI roles
  `const uint8_t *` / `uint16_t *`.
- Added provider-owned unsigned route facts for the standalone
  widening-product primitive: `u8` source dtype, `u16` product/result dtype,
  `vuint8mf4_t` / `vuint16mf2_t`, `__riscv_vle8_v_u8mf4`,
  `__riscv_vwmulu_vv_u16mf2`, `__riscv_vse16_v_u16mf2`,
  `rvv-route-operand-binding:widening_product_u8_u16.v1`, unsigned C type
  mapping, and target leaf profile
  `rvv-v1-u8mf4-u16mf2-contraction-leaf-profile.v1`.
- Threaded the unsigned facts through route planning, contraction
  route-family plan validation, runtime ABI/config construction contracts,
  `TCRVEmitCLowerableRoute`, common EmitC materialization, emission-plan
  metadata, and target artifact mirror validation.
- Added accepted conversion coverage proving the common EmitC materializer
  emits unsigned RVV load/product/store intrinsics from provider facts, and
  target artifact coverage proving stale unsigned intrinsic, primitive dtype,
  and C type mapping mirrors fail before header acceptance.
- Preserved signed i8 widening-product and product-reduction coverage through
  the existing typed low-precision primitive facts and the focused
  widening/product-reduction lit matrix.

This round does not claim runtime correctness/performance, so no `ssh rvv`
evidence was required. Gate 2 has advanced to accepted unsigned product facts,
but the macro campaign is not complete: Gearbox/resource-aware selected-body
realization, executable/runtime evidence for newly claimed behavior, and
measured same-target comparison remain open.

## Previous Round Result

Completed the Gate 2 second slice while keeping the macro campaign open:

- Promoted the unsigned u8 widening-product boundary from dialect-only
  rejection to a verifier-legal typed surface: `unsigned_widening_product` with
  `!tcrv_rvv.vector<ui8, "mf4"> -> !tcrv_rvv.vector<ui16, "mf2">` and
  `unsigned-u8mf4xu8mf4-to-u16mf2` relation now parses/verifies as a typed RVV
  body fact.
- Added the minimal ABI/type surface needed for that u8 body to reach the RVV
  provider: `lhs`/`rhs` runtime ABI roles accept `const uint8_t *`, output
  accepts `uint16_t *`, and u8 source loads may participate in the standalone
  widening-product SEW16/LMUL mf2 body shape.
- Kept executable u8 support fail-closed in the RVV provider before
  `TCRVEmitCLowerableRoute` construction. The provider diagnostic names the
  missing unsigned widening product intrinsic fact
  `__riscv_vwmulu_vv_u16mf2` and the matching target type/header mirror
  validation.
- Preserved signed i8 widening-product and product-reduction support through
  the provider-owned low-precision primitive contract.
- Added target artifact stale-mirror coverage for product-reduction
  `vwredsum` intrinsic provenance and low-precision primitive accumulator dtype
  provenance, proving accumulator/result facts are not metadata authority.

This round does not claim new executable/runtime behavior, so no `ssh rvv`
evidence was required. The campaign is not complete. Gate 2 still needs
accepted unsigned u8 provider/target facts if executable u8 support is desired,
and Gates 1, 3, and 4 remain open.

## Previous Round Result

Completed the Gate 2 first slice while keeping the macro campaign open:

- Added provider-owned `low_precision_primitive.*` facts for signed
  `i8mf4 x i8mf4 -> i16mf2` widening product and
  `i8mf4 -> i16mf2 -> i32m1/f32m1` product-reduction chains.
- Threaded those facts through the contraction route-family plan, selected-body
  route description, emission-plan metadata, target support bundle export, and
  widening-product target artifact mirror validation.
- Kept u8 typed widening-product support fail-closed with an explicit dialect
  diagnostic until unsigned provider/target intrinsic facts are implemented.
- Added focused product-only target artifact coverage with a stale primitive
  source-dtype mirror rejection, and widened existing product-reduction checks
  to prove the reduction primitive fact contract.

The campaign is not complete. Gate 2 still needs accepted u8 widening-product
facts or a broader unsigned fail-closed/provider contract, plus deeper
`vwredsum`/accumulator route-family coverage. Gates 1, 3, and 4 remain open.

## Out Of Scope

- No standalone generated-bundle or `ssh rvv` evidence seam by default.
- No adjacent reduce-min/max, provider-test-only, fixture-only, metadata-only,
  or report/index closeout.
- No high-level Linalg/Vector/StableHLO frontend work.
- No per-Linalg route authority.
- No Common EmitC invention of RVV semantics.
- No Gearbox performance claim before primitive/resource facts are
  production-owned.
- No broad dashboard/report/index work.

## Technical Notes

- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/implementation-stack/supervision-loop.md`.
- Current likely implementation owners:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Focused existing tests to inspect/update:
  `test/Dialect/RVV/generic-widening-product-dataflow.mlir`,
  `test/Conversion/EmitC/*widening-product*`,
  `test/Target/RVV/*widening-product*`,
  `test/Target/RVV/*widening-dot*`,
  and `test/Plugin/RVVExtensionPluginTest.cpp`.

## Continuation Point

Continue the macro campaign from Gate 1/Gate 2 integration: make the
low-precision product/reduction primitive and resource facts consumable by
Gearbox/resource-aware selected-body realization, then collect Gate 3 runtime
evidence only when a newly executable behavior is claimed. Gate 4 same-target
comparison remains after the production-kernel path has typed primitive facts,
realization/resource facts, and runtime correctness evidence.
