# RVV production-kernel capability campaign

## Goal

Promote the current RVV work from repeated generated-bundle evidence closeouts
to a macro production-kernel capability campaign. The campaign covers
Gearbox/resource-aware selected-body realization, low-precision contraction
primitive surface, generated artifact/runtime evidence only when new
production behavior is claimed, and measured same-target comparison for
production-like RVV kernels.

This task remains open across rounds until the campaign gates below are met.
This round continues Gate 2 with accepted unsigned u8 widening-product
provider/target facts, moving the already verifier-legal unsigned body from the
provider fail-closed boundary to route-supported only through typed RVV
body/config facts, provider-owned intrinsic/type facts, exact target mirrors,
and neutral EmitC materialization.

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

Continue Gate 2 by adding the smallest production compiler submodule that lets
RVV route-support the unsigned u8 low-precision widening-product primitive from
typed `tcrv_rvv` body/config facts, while keeping signed i8 product and
product-reduction facts provider-owned.

The preferred continuation slice is:

- Move verifier-legal unsigned u8 widening-product bodies past the previous
  provider fail-closed boundary only when the RVV provider derives unsigned
  primitive facts from the typed body/config/runtime facts.
- Accept unsigned u8 only through provider-owned unsigned source/product/result
  facts: `ui8`, `ui16`, `vuint8mf4_t`, `vuint16mf2_t`, unsigned load/store
  leaves, `__riscv_vwmulu_vv_u16mf2`, and exact target type/header mirrors.
- Keep target artifact metadata as mirror-only output from the provider-owned
  route description; stale unsigned intrinsic/type/header/primitive mirrors
  must fail before artifact acceptance.
- Keep signed i8 widening-product and product-reduction facts accepted only
  through the low-precision primitive contract derived from typed
  source/product/accumulator/result facts.
- Deepen widening-reduction/`vwredsum` validation so accumulator/result dtype,
  reduction intrinsic, and store-VL facts are checked as typed primitive facts
  rather than dispersed strings. If this is too large, finish the u8 boundary
  and leave the exact continuation point.
- Mirror provider-owned facts into target validation only as exact mirrors,
  with stale/missing mirror rejection where the changed target validation owner
  is touched.

If live implementation inspection shows the IR/type surface blocks accepted u8
support, this milestone may close that prerequisite in the same slice and
record the exact next continuation point.

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
- [x] The unsigned u8 widening-product boundary is accepted through the same
      provider-owned low-precision primitive contract, with unsigned typed
      source/product/result facts, unsigned RVV vector C types, unsigned
      load/store leaves, `__riscv_vwmulu_vv_u16mf2`, and exact target mirrors.
- [x] Signed i8 widening-product and product-reduction facts remain accepted
      only when typed source/product/accumulator/result facts agree.
- [x] Widening-reduction/`vwredsum` accumulator/result facts remain validated as
      typed primitive facts where this slice touches the route family; otherwise
      the exact next continuation point is recorded.
- [x] Focused lit/C++ tests prove accepted signed i8 remains typed-fact gated
      and unsigned u8 takes the accepted provider/target mirror path.
- [x] Target artifact validation coverage rejects stale/missing unsigned
      primitive source/product/result, intrinsic, type, or header mirrors where
      touched.
- [x] A bounded old-authority scan over changed/added lines shows no new
      positive `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor, source-front-door, route-id, or artifact
      authority.
- [x] `git diff --check` and `git diff --cached --check` pass for this coherent
      slice.
- [x] The slice is committed coherently and final worktree status is verified
      clean in the round report.

## Current Round Result

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

Continue Gate 2 by implementing accepted unsigned u8 widening-product provider
and target facts if the materializer is ready: unsigned vector C type mapping,
u8/u16 load-store leaves, `vwmulu.vv` intrinsic mapping, target mirror support,
and focused artifact tests. Then continue broadening the
widening-reduction/`vwredsum` primitive contract for product, accumulator,
result, and resource facts that Gearbox can consume without relying on metadata
or route names.
