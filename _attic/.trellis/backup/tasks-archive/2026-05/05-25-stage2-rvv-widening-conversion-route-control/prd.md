# Stage2 RVV widening conversion route-control provider-plan integration

## Goal

Make the existing production-active RVV widening conversion statement/provider
path consume the shared RVV route-control provider plan before route statement
construction. Already-supported widening conversion routes must validate
runtime AVL/VL, typed config, selected capability, SEW/LMUL, source/result type
policy, tail policy, mask policy, runtime ABI order, materialization,
conversion-form facts, and math operand-binding ownership through the RVV-owned
route-control boundary.

## Direction Source

- Direction title: `Stage2 RVV widening conversion route-control provider-plan
  integration`.
- Module owner: RVV plugin-local route-control provider-plan consumption by
  the existing widening conversion route-family/provider/statement path.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `0f12989c rvv: consume route control plan in segment2 memory`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.

## Current Repository Facts

- The long-term RVV authority chain is selected `tcrv.exec` envelope -> typed
  or realized `tcrv_rvv` body -> RVV plugin-owned legality / route-family
  planning / selected-body realization / route provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact mirrors.
- Specs require route-control consumers to validate runtime AVL/VL,
  SEW/LMUL, tail policy, mask policy, runtime ABI order, selected capability,
  and typed config through `RVVSelectedBodyRouteControlProviderPlan` before
  statement construction.
- `RVVSelectedBodyWideningConversionRouteFamilyPlan` already owns the active
  conversion family facts for existing widening conversion routes.
- Existing math operand-binding facts already cover widening conversion
  source/output/runtime operands and materialized use closure.
- The previous completed segment2 task added `controlsSegment2Memory` to the
  shared route-control provider plan and required segment2 statement planning
  to consume that plan.
- The current gap is that the widening conversion statement path still has
  family and statement validation but no route-control consumer flag/boundary
  requirement.

## Scope

1. Add a widening-conversion route-control consumer marker or equivalent
   structural flag to `RVVSelectedBodyRouteControlProviderPlan`.
2. Extend route-control consumer detection to include only the existing active
   widening conversion route-family paths; do not add operation coverage.
3. In `getRVVSelectedBodyRouteControlProviderPlan(...)`, validate same-analysis
   widening conversion family/materialization facts, conversion relation/form,
   source/result type policy, runtime AVL/VL control, typed config, selected
   capability, and runtime ABI mirror consistency before returning the plan.
4. In `getRVVSelectedBodyWideningConversionRouteStatementPlan(...)`, require
   the widening conversion route-control provider plan before building
   setvl/load/convert/store statement steps.
5. Update focused C++ tests for positive route-control consumption and
   targeted fail-closed diagnostics.
6. Update `.trellis/spec/extension-plugins/rvv-plugin.md` only if the code
   change makes widening conversion a durable route-control consumer.

## Requirements

1. Widening conversion statement planning must require a route-control plan
   whose runtime-control pointer is the same runtime-control plan owned by the
   verified widening conversion family plan.
2. Route-control construction must fail closed when the widening conversion
   family plan or route materialization facts are missing, stale, from another
   selected route analysis, or classified with the wrong conversion kind,
   conversion relation, source/result type policy, route shape, or operation.
3. Route-control construction must fail closed when typed config, selected
   target capability, runtime AVL role/source, SEW/LMUL, tail policy, mask
   policy, runtime ABI order, runtime VL contract, setvl/with_vl names, loop
   facts, or mirror fields disagree with validated body/config/runtime facts.
4. Widening conversion statement planning must still fail closed for
   missing/stale math operand-binding facts, missing runtime `n`, missing
   source/output ABI roles, missing source load/convert/store materialized
   uses, stale conversion/type markers, missing materialization leaves, or
   wrong source operation provenance before common EmitC materialization.
5. Common EmitC, target export, scripts, route ids, artifact names, ABI
   strings, metadata, manifests, descriptors, and tests may only mirror
   provider-built facts after route construction; they must not become AVL/VL,
   source/result type, conversion-form, policy, dtype, or compute authority.
6. Do not add new conversion operation kinds, dtype/LMUL clone batches,
   accumulation, contraction, reduction, high-level frontend lowering,
   source-front-door positive routes, dashboards, broad smoke matrices, or
   evidence-only fixture copying.

## Acceptance Criteria

- [x] `RVVSelectedBodyRouteControlProviderPlan` exposes a widening conversion
      consumer flag or equivalent structural marker.
- [x] Route-control consumer detection includes only existing active widening
      conversion routes and does not expand conversion coverage.
- [x] `getRVVSelectedBodyRouteControlProviderPlan(...)` validates
      same-analysis widening conversion family/materialization facts, runtime
      AVL/VL control, typed config, selected capability, tail/mask policy,
      runtime ABI order, source/result type policy, and conversion-form facts.
- [x] `getRVVSelectedBodyWideningConversionRouteStatementPlan(...)` consumes
      the route-control provider plan before constructing widening conversion
      setvl/load/convert/store statements.
- [x] Positive C++ tests prove route-control consumption and provider/statement
      attachment for representative existing widening conversion routes.
- [x] Negative C++ tests fail closed for representative stale or missing
      dependencies: missing widening conversion family/materialization plan,
      missing route-control plan, stale same-analysis ownership, wrong runtime
      AVL role, policy mismatch, unsupported selected capability/config,
      runtime ABI mirror mismatch, stale conversion/type marker, stale
      conversion-form facts, and stale operand binding.
- [x] Existing generated artifact or lit coverage continues to show explicit
      mirror-only labels for conversion/SEW facts if mirrors change. If emitted
      code, ABI, and target mirrors stay unchanged, final report explains why
      historical `ssh rvv` evidence remains sufficient.
- [x] Bounded scans over touched RVV planning/provider/test/spec/target/script
      files find no new name-, route-id-, metadata-, descriptor-, ABI-string-,
      script-, artifact-, common-EmitC-, source-front-door-, or legacy-i32-
      derived AVL/VL, source/result type, conversion-form, policy, dtype, or
      compute authority.
- [x] `git diff --check` passes.
- [x] Focused RVV plugin tests and `check-tianchenrv` pass, or an exact blocker
      is recorded.
- [x] Task status, journal/archive, and one coherent commit complete if this
      task finishes.

## Out Of Scope

- New widening conversion operation coverage or new dtype/LMUL route cases.
- Moving conversion semantics into `tcrv.exec`, common EmitC, target export,
  scripts, metadata, descriptors, route ids, or ABI strings.
- Source-front-door positive routes, direct-C/source-export paths, legacy i32
  route authority, scalar/IME/offload/future plugin work, broad dashboards,
  performance claims, or unrelated cleanup.
- Runtime/correctness/performance claims unless emitted target code or ABI
  changes and real `ssh rvv` evidence is collected.

## Technical Approach

1. Start and validate the Trellis task.
2. Inspect current route-control, widening conversion family, materialization,
   math operand-binding, statement-plan, provider, realization, and tests.
3. Add the widening conversion route-control consumer marker and consumer
   detection.
4. Extend route-control provider-plan construction with a widening conversion
   branch that validates same-analysis family/materialization facts and then
   reuses the existing typed config/runtime control/selected capability
   validation tail.
5. Require the widening conversion statement-plan boundary to acquire the
   route-control plan and confirm the widening conversion flag and
   runtime-control pointer before emitting any statement steps.
6. Update focused C++ provider tests rather than adding broad fixture copies.
7. Update `rvv-plugin.md` route-control and widening conversion/statement-plan
   sections if production code and tests establish the durable consumer.
8. Run focused build/tests, bounded authority scans, `git diff --check`, and
   `check-tianchenrv`.

## Validation Plan

1. `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-widening-conversion-route-control`
2. `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
3. `./build/bin/tianchenrv-rvv-extension-plugin-test`
4. Focused lit/FileCheck or generated-bundle dry-run only if target mirrors,
   emitted artifacts, ABI order, or conversion evidence output changes.
5. Bounded authority scan over touched RVV planning/provider/test/spec files.
6. `git diff --check`
7. `cmake --build build --target check-tianchenrv -j2`

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
- Previous task context read:
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-runtime-vl-policy-provider-plan/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-segment2-route-control/prd.md`,
  `.trellis/tasks/archive/2026-05/05-23-stage2-rvv-widening-conversion-route-family-ownership/prd.md`,
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-conversion-sew-policy-route-closure/prd.md`, and
  `.trellis/tasks/archive/2026-05/05-21-05-21-stage2-rvv-conversion-operand-binding/prd.md`.
- Initial code surfaces to inspect:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Implementation Result

- Added `controlsWideningConversion` to
  `RVVSelectedBodyRouteControlProviderPlan`.
- Made the two existing active widening conversion routes,
  `widen_i32_to_i64` and `widen_i16_to_i32`, explicit route-control consumers
  when they use the existing unit-stride conversion memory form.
- Extended `getRVVSelectedBodyRouteControlProviderPlan(...)` so widening
  conversion requires the verified widening conversion family plan and
  materialization facts from the same selected route analysis.
- The route-control owner now validates widening conversion source/result type
  policy, source/result SEW/LMUL, source/result vector/C types, setvl/source
  load/conversion/store intrinsic leaves, conversion relation, typed config,
  selected target capability, runtime AVL/VL, tail/mask policy, runtime ABI
  order, and materialization fact mirrors before statement construction.
- Extended the widening conversion statement-plan boundary to cover both
  already-supported active conversion routes. The provider now gets both
  `widen_i32_to_i64` and `widen_i16_to_i32` through the migrated
  provider-neutral statement-plan boundary before generic provider-local
  assembly.
- Added focused C++ positive/fail-closed coverage for route-control
  consumption, stale/missing widening conversion materialization facts, stale
  source/type materialization markers, stale policy facts, unsupported selected
  target capability facts, and provider statement-plan attachment.
- Updated `.trellis/spec/extension-plugins/rvv-plugin.md` so widening
  conversion is listed as a durable route-control provider-plan consumer.
- No target ABI, runtime behavior, target mirror schema, generated-bundle script
  behavior, or conversion semantics changed. No new `ssh rvv` claim was made;
  historical runtime evidence for the unchanged executable conversion output
  remains sufficient.

## Validation Result

- [x] `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-25-stage2-rvv-widening-conversion-route-control`
- [x] `git diff --check`
- [x] `cmake --build build --target tianchenrv-rvv-extension-plugin-test -j2`
- [x] `./build/bin/tianchenrv-rvv-extension-plugin-test`: RVV extension plugin
      smoke test passed.
- [x] Focused lit/FileCheck and generated-bundle dry-run coverage from
      `build/test`:
      `/usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='widen-i(16|32)|widening-conversion|conversion'`
      ran 60 selected tests and passed.
- [x] Bounded added-line authority scan over touched planning/provider/test/spec
      files found no new source-front-door, descriptor, direct-C/source-export,
      common-EmitC, artifact/script, route-id, ABI-string, or legacy-i32
      authority.
- [x] `cmake --build build --target check-tianchenrv -j2`: 379/379 tests
      passed.
- [x] `clang-format` was not available in PATH and no versioned
      `clang-format-*` binary was present; formatting was checked with
      `git diff --check` and focused build/test compilation.
