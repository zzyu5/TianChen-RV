# Stage2 RVV Low-Precision Contraction Primitive Surface Campaign

## Goal

Create one active macro owner for RVV low-precision contraction primitive
surface maturity. The campaign moves q8/q4-style pressure into typed
low-level `tcrv_rvv` primitive facts and RVV plugin-owned legality/planning,
not q8/q4 route ids, artifact names, source-front-door markers, or
generated-bundle-only evidence.

Gate 1 is complete. The current round implements Gate 2: provider-owned
low-precision widening-product facts for signed i8 and unsigned u8
multiplicands, including explicit multiplicand-role and extension-policy facts
that route-family planning, route validation, target metadata, and target
artifact validation consume before any later reduction/accumulation claim.

## What I Already Know

- Commit `3ebe7adc rvv: close gearbox realization campaign` completed and
  archived the Gearbox resource-aware selected-body realization campaign.
- No `.trellis/.current-task` existed at session start, so this task is created
  from the Hermes Direction Brief.
- Specs require the authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality / selected-body realization / route provider ->
  provider-built `TCRVEmitCLowerableRoute` -> neutral Common EmitC -> target
  artifact -> `ssh rvv` evidence only when runtime/correctness/performance is
  claimed.
- `tcrv.exec` binds ABI/runtime roles; it does not own low-precision compute,
  dtype, SEW/LMUL, sign-extension, or intrinsic spelling.
- Common EmitC/export must not infer signedness, byte-load form, extension
  behavior, widening, schedule, mask, contraction semantics, or target
  support. It may only carry provider-built payloads and mirrors.
- Historical q8/q4 and llama.cpp examples are pressure tests that expose
  broader Stage 2 low-precision/quantized contraction gaps. They are not route
  authority.

## Requirements

- Keep this Trellis task active across rounds until all macro campaign gates
  are complete or human steering redirects the campaign.
- Complete one coherent unfinished gate slice per round and update this PRD
  with completed and remaining gates plus a precise continuation point.
- Gate 1 must establish or harden a production surface where typed i8/u8
  vector/config, byte load, and sign-or-zero-extension facts are modeled and
  consumed by RVV plugin-owned admission/planning before later widening
  product or reduction routes depend on them.
- Gate 1 admission must derive from typed `tcrv_rvv` body/config, target
  capability, and runtime ABI facts. It must fail closed when required facts
  are missing, stale, or metadata-derived.
- Gate 2 must structurally represent signed i8 and unsigned u8 widening-product
  multiplicand roles, extension/sign policy, product dtype/SEW/LMUL, typed
  config/policy, runtime ABI operand facts, route-family validation facts, and
  target artifact mirrors.
- Gate 2 facts must be provider-owned and derived from typed body/config,
  runtime ABI, and target capability facts. Artifact metadata, route ids,
  helper names, ABI strings, or test names must not invent support.
- Targeted diagnostics must distinguish missing typed vector/config facts,
  unsupported signedness/extension form, missing byte-load facts, and
  metadata/route-id/artifact-name authority attempts where the code surface can
  observe them.
- Retain Common EmitC/export neutrality.
- Do not add new dtype-prefixed op families, q8/q4-named routes, one-intrinsic
  wrapper dialects, high-level Linalg/Vector/StableHLO frontends, or
  descriptor-driven C/source export.

## Macro Campaign Gates

- [x] Gate 1: typed i8/u8 RVV primitive surface and byte-load
  sign-or-zero-extension admission are modeled and consumed by RVV plugin
  legality/planning with fail-closed diagnostics.
- [x] Gate 2: low-precision widening product facts are structurally
  represented and provider-consumed without route-id or artifact-name
  authority.
- [ ] Gate 3: widening reduction/accumulation facts are represented and
  consumed for contraction-style kernels.
- [ ] Gate 4: Gearbox/resource-aware selected-body realization and measured
  same-target comparison consume these primitive facts with source-backed
  evidence.

## Current Round Slice: Gate 2

Implementation first inspected the existing typed RVV value/config/op surface,
RVV contraction route-family facts, target artifact validation, and signed/u8
widening-product fixtures. The existing production path already derived signed
i8 and unsigned u8 product relation, intrinsic, source load/extension, product
dtype, runtime ABI, and C type facts. The missing Gate 2 hardening was an
explicit provider-owned product-fact surface for lhs/rhs multiplicand roles and
extension policy that survives route-family planning, route description,
target metadata, and target validation without being hidden inside route
binding strings.

Acceptance criteria:

- [x] Production C++ models signed i8 and unsigned u8 widening-product
  multiplicand roles, extension policy, source/load/product facts, typed
  config/policy, runtime ABI order, and target mirrors as provider-owned RVV
  route facts.
- [x] RVV route-family plan validation and route description validation consume
  those facts before a widening-product route can claim support.
- [x] Target artifact validation compares provider-owned multiplicand-role,
  extension-policy, source/load/extension/product, relation, intrinsic, ABI,
  and C type mirrors before candidate acceptance.
- [x] Focused tests prove accepted signed i8 sign-extension and unsigned u8
  zero-extension product facts from typed body/config/runtime ABI facts.
- [x] Focused tests prove stale/missing/metadata-derived product facts fail
  closed with targeted diagnostics.
- [x] Bounded scan confirms touched code and added diff do not introduce legacy
  RVV route-authority markers as positive support.
- [x] Relevant focused build/test targets pass.
- [x] Task remains active with Gates 1-2 marked complete and Gates 3-4 remaining,
  unless repository evidence proves all gates are already complete.

Completed Gate 1 slice:

- Added provider-owned low-precision primitive source-load and
  source-extension facts:
  `lowPrecisionPrimitiveSourceLoadKind` and
  `lowPrecisionPrimitiveSourceExtensionKind`.
- Signed i8 primitive facts derive `unit-stride-byte-load` and
  `sign-extend-i8-to-i16-product`; unsigned u8 widening-product facts derive
  `unit-stride-byte-load` and `zero-extend-u8-to-u16-product`.
- RVV contraction route-family plan validation, route description mirror
  verification, widening-product route facts, widening-reduction primitive
  facts, target validation contracts, and target artifact metadata mirror
  checks now consume these fields.
- Target artifact support bundles export the new mirror keys:
  `tcrv_rvv.low_precision_primitive.source_load` and
  `tcrv_rvv.low_precision_primitive.source_extension`.

Completed Gate 2 slice:

- Added provider-owned standalone widening-product fact fields:
  `wideningProductMultiplicandRoleSummary` and
  `wideningProductExtensionPolicy`.
- Signed i8 widening-product facts now explicitly mirror:
  `lhs=lhs-input-buffer:wprod-lhs:src-i8mf4`,
  `rhs=rhs-input-buffer:wprod-rhs:src-i8mf4`, and
  `source=signed;extension=sign-extend-i8-to-i16-product;product=i16mf2`.
- Unsigned u8 widening-product facts now explicitly mirror:
  `lhs=lhs-input-buffer:wprod-lhs:src-u8mf4`,
  `rhs=rhs-input-buffer:wprod-rhs:src-u8mf4`, and
  `source=unsigned;extension=zero-extend-u8-to-u16-product;product=u16mf2`.
- RVV contraction route-family plan validation, route description validation,
  target provider-facts validation, target support bundle metadata, and target
  artifact mirror validation all consume the new fields.
- Signed and unsigned widening-product lit fixtures now prove accepted mirrors
  and stale metadata rejection for multiplicand roles and extension policy.

## Non-Goals

- No generated-bundle-only or `ssh rvv`-only closeout unless it validates
  production code changed in this slice.
- No q8/q4-named route ids, artifact names, wrappers, or llama.cpp-specific
  authority.
- No high-level frontend, per-Linalg route authority, dtype/LMUL clone batch,
  one-intrinsic wrapper dialect, broad MAcc evidence matrix, or source-front
  door positive route.
- No Common EmitC invention of dtype, widening, schedule, mask, extension, or
  contraction semantics.
- No campaign archive after only Gate 1.

## Technical Notes

- Read `.trellis/spec/index.md`.
- Read `.trellis/spec/extension-plugins/rvv-plugin.md`.
- Read `.trellis/spec/lowering-runtime/emitc-route.md`.
- Read `.trellis/spec/variant-pipeline/index.md`.
- Read `.trellis/spec/testing/mlir-testing-contract.md`.
- Read archived Gearbox campaign PRD:
  `.trellis/tasks/archive/2026-06/06-10-stage2-rvv-gearbox-realization-campaign/prd.md`.
- Relevant source areas to inspect:
  `include/TianChenRV/Dialect/RVV/`,
  `lib/Dialect/RVV/`,
  `include/TianChenRV/Plugin/RVV/`,
  `lib/Plugin/RVV/`,
  `lib/Plugin/RVV/EmitC/`,
  `lib/Target/RVV/`,
  and focused tests under `test/Plugin/` and `test/Target/RVV/`.

## Verification Plan

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- `build/bin/tianchenrv-target-artifact-export-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  Target/RVV/explicit-selected-body-artifact-widening-product.mlir
  Target/RVV/explicit-selected-body-artifact-widening-product-unsigned-u8.mlir
  Target/RVV/explicit-selected-body-artifact-widening-product-reduce-add.mlir`
- Additional focused dialect/target tests if the implementation changes
  dialect syntax, verifier behavior, or target artifact validation.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-10-stage2-rvv-low-precision-contraction-primitive-surface-gate1`
- `git diff --check`
- `git diff --cached --check`
- Bounded scan over touched files and added diff lines for legacy RVV
  route-authority markers.

## Verification Results

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test` passed.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `cmake --build build --target tianchenrv-target-artifact-export-test`
  passed with pre-existing switch-coverage warnings in
  `TargetArtifactExportTest.cpp`.
- `build/bin/tianchenrv-target-artifact-export-test` passed.
- `cmake --build build --target tcrv-opt tcrv-translate` passed.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  Target/RVV/explicit-selected-body-artifact-widening-product.mlir
  Target/RVV/explicit-selected-body-artifact-widening-product-unsigned-u8.mlir
  Target/RVV/explicit-selected-body-artifact-widening-product-reduce-add.mlir`
  passed from `build/test`.
- `git diff --check` passed.
- Bounded added-line scan for `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
  `!tcrv_rvv.i32m`, source-front-door, q8/q4 route naming, and exact
  `__riscv_*_i32m1` old-authority introduction found no matches.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate` passed with
  pre-existing switch/unused-function warnings.
- `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
- `build/bin/tianchenrv-target-artifact-export-test` passed.
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv
  Target/RVV/explicit-selected-body-artifact-widening-product.mlir
  Target/RVV/explicit-selected-body-artifact-widening-product-unsigned-u8.mlir
  Target/RVV/explicit-selected-body-artifact-widening-product-reduce-add.mlir`
  passed from `build/test`.

## Spec Update Decision

- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` to record the
  provider-owned standalone widening-product primitive-fact contract, including
  multiplicand-role and extension-policy mirrors plus target rejection
  requirements.

## Continuation Point

Keep this macro task active. The next unfinished milestone is Gate 3:
represent and consume widening reduction/accumulation facts for
contraction-style kernels, building on the Gate 2 standalone
widening-product facts rather than introducing q8/q4 wrappers, artifact-name
authority, or Common EmitC semantic inference. Gate 4 Gearbox/resource-aware
selected-body realization plus measured same-target comparison remains after
Gate 3.
