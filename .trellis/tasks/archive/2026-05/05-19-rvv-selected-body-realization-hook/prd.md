# RVV plugin-local selected-body realization hook

## Goal

Add the first bounded RVV plugin-owned selected-body realization hook on the
corrected current RVV surface. The hook must accept a pre-realized typed
`tcrv_rvv` selected body for the existing i32m1 add specialization, verify its
operation, dtype/config, memory, policy, and runtime AVL facts inside the RVV
plugin, materialize the already-supported realized
`setvl -> with_vl -> load -> compute -> store` body, and then let the existing
provider route/common EmitC/target artifact path consume that realized body.

## Current Facts

- Current HEAD at task creation is `b229385`; the worktree was clean.
- No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief before source edits.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-19-rvv-selected-body-front-door-demotion`
  made explicit selected `tcrv.exec`/typed `tcrv_rvv` fixtures the default
  artifact/evidence input and kept the legacy source front door explicit-only.
- Current RVV provider route construction still requires a fully realized body:
  one `tcrv_rvv.setvl`, one `tcrv_rvv.with_vl`, two loads or one load plus one
  broadcast load, one supported compute op, and one store.
- Current `RVVExtensionPlugin::materializeSelectedLoweringBoundary` validates
  that realized body but does not yet provide a plugin-local rewrite from a
  pre-realized typed selected body.
- Specs require RVV realization and route semantics to stay in the RVV plugin;
  common EmitC/materialization/target export must remain neutral.

## Requirements

1. Add a bounded RVV dialect surface for one pre-realized typed selected body
   that structurally carries the add operation, i32/m1 config, tail/mask policy,
   lhs/rhs/out runtime ABI values, and runtime `n`/AVL value. This surface is
   RVV-owned input to realization, not route metadata, descriptor authority, a
   source-front-door default, or common EmitC/export semantics.
2. Add RVV plugin-local realization logic that runs from the selected-body
   boundary path and rewrites a valid pre-realized add body into the existing
   realized body consumed by `describeRVVSelectedBodyEmitCRoute` and
   `buildRVVSelectedBodyEmitCLowerableRoute`.
3. Preserve computation semantics, dtype semantics, parameter roles, selected
   variant origin, required capabilities, dispatch/fallback behavior, and the
   exact runtime `n`/AVL SSA value.
4. Fail closed before provider route construction when the pre-realized body is
   missing, malformed, unsupported, mismatched, or stale relative to the
   selected variant/kernel/path role.
5. Existing fully realized explicit add/sub/mul selected-body fixtures must keep
   working.
6. Add at least one positive add fixture showing pre-realized selected body ->
   RVV plugin-local realization -> existing provider route/artifact path.
7. Add focused negative coverage for missing or mismatched operation,
   dtype/config, memory/ABI roles, policy, runtime AVL/n, and stale
   source/metadata authority where these are in this module's control.
8. Do not add new RVV coverage classes, new dtype/LMUL support, source frontend
   generalization, high-level Linalg/Vector/StableHLO lowering, descriptor-
   driven computation, direct-C/source-export semantics, or common-code RVV
   semantic branches.

## Acceptance Criteria

- [ ] RVV plugin exposes and owns a bounded pre-realized selected-body surface
      for i32m1 add.
- [ ] Running the selected-lowering-boundary materialization path on a selected
      variant containing that pre-realized add surface materializes exactly the
      realized `setvl/with_vl/load/add/store` body used by the existing provider.
- [ ] The positive pre-realized add fixture reaches supported emission-plan
      metadata and the existing target artifact/header or object route.
- [ ] Fully realized explicit selected-body add/sub/mul fixtures still pass.
- [ ] Negative tests fail before route/artifact construction for malformed
      pre-realized facts.
- [ ] Bounded scans over touched files show no source-front-door default
      authority, descriptor/direct-C/source-export restoration, route-id or
      artifact-name authority, or common-code RVV semantic branching.
- [ ] Focused build/lit/C++ checks for touched RVV plugin/provider/materializer
      surfaces pass. `ssh rvv` is not required unless executable runtime or
      correctness status is explicitly claimed.

## Technical Approach

Use the existing selected lowering-boundary materialization pass as the first
compiler-visible realization hook. The RVV plugin will recognize a selected
variant that already has the pre-realized RVV op, validate its facts, create the
realized `runtime_abi_value -> setvl -> with_vl -> i32_load/i32_add/i32_store`
structure in the same selected variant, delete the pre-realized op after
successful rewrite, and then reuse the existing provider route validation.

The existing provider and common EmitC/materializer should not learn new RVV
semantics for pre-realized input. They should continue to consume only the
realized selected body. Unsupported or incomplete pre-realized bodies therefore
fail in RVV plugin realization before provider route construction.

## Out of Scope

- No sub/mul realization unless the add submodule finishes early and the
  operation-family mechanics are already in place.
- No reduction, conversion, compare/select expansion, new dtype, new LMUL, new
  intrinsic families, or broad Stage 2 coverage expansion.
- No high-level frontend rebuild or source-front-door default restoration.
- No descriptor/direct-C/source-export route restoration.
- No runtime/correctness/performance claim without new real `ssh rvv` evidence.

## Validation Plan

- Build focused targets: `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test` as needed.
- Run focused lit for the new pre-realized selected-body positive/negative
  cases and existing explicit selected-body add/sub/mul fixtures.
- Run the relevant RVV plugin C++ test binary if changed.
- Generate local artifact/header/object evidence for the pre-realized add path
  if the positive route reaches target export.
- Run bounded residue scans over touched RVV/common files for
  `source-front-door`, `lowering_seed`, `descriptor`, `direct-C`,
  `source-export`, `route id`, `artifact name`, and common-code RVV branching.
- Run `python3 ./.trellis/scripts/task.py validate` before finishing.
- Run `git diff --check`.

## Relevant Specs And Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-selected-body-front-door-demotion/prd.md`

## Initial Code Surface

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `include/TianChenRV/Plugin/RVV/RVVExtensionPlugin.h`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`
- `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp`
- `test/Target/RVV/explicit-selected-body-artifact-add.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-sub.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-mul.mlir`

## Implementation Notes

- Added RVV dialect op `tcrv_rvv.i32_binary_pre_realized_body` as the bounded
  pre-realized selected-body input for the current add specialization. It
  directly carries explicit `lhs`, `rhs`, `out`, and runtime `n` operands plus
  `op_kind`, `memory_form`, `sew`, `lmul`, and `policy`.
- The op verifier rejects stale selected-boundary/route/artifact/capability
  authority metadata and validates the supported add/vector-rhs-load/SEW32
  LMUL m1/tail-mask-agnostic surface.
- Updated `RVVExtensionPlugin::materializeSelectedLoweringBoundary` so the RVV
  plugin realizes a selected pre-realized body into the existing
  `setvl -> with_vl -> i32_load -> i32_add -> i32_store` body before provider
  route construction. Existing fully realized bodies continue through the old
  validation path.
- Common EmitC/materializer and target export were not taught RVV
  pre-realized semantics. They still consume the provider-built route from the
  realized selected body.
- Added long-term code-spec coverage in
  `.trellis/spec/extension-plugins/rvv-plugin.md` for the pre-realized
  selected-body realization contract.

## Evidence

- Positive fixture:
  `test/Target/RVV/pre-realized-selected-body-artifact-add.mlir`.
- Negative fixture:
  `test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir`.
- Local materialized artifact bundle:
  `artifacts/tmp/rvv_selected_body_realization_hook_add/`.
- Bundle output:
  `artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o`,
  `artifact-1-runtime-callable-c-header-rvv-i32m1-arithmetic-emitc-route-family.header.h`,
  and `tianchenrv-target-artifact-bundle.index`.
- The bundle index records `selected_variant:
  @pre_realized_body_rvv_i32_add`, route
  `rvv-i32m1-arithmetic-emitc-route-family`, runtime ABI
  `rvv-i32m1-add-callable-c-abi.v1`, and ordered `lhs,rhs,out,n` ABI
  parameters.
- No runtime/correctness/performance claim was made; no `ssh rvv` run was
  collected for this task.

## Checks Run

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- `cmake --build build --target tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Target/RVV/pre-realized-selected-body-artifact-add.mlir ../test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir ../test/Target/RVV/explicit-selected-body-artifact-add.mlir ../test/Target/RVV/explicit-selected-body-artifact-sub.mlir ../test/Target/RVV/explicit-selected-body-artifact-mul.mlir`
  passed 5/5.
- Local bundle generation:
  `build/bin/tcrv-opt test/Target/RVV/pre-realized-selected-body-artifact-add.mlir --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans | build/bin/tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_selected_body_realization_hook_add`
- Bounded residue scans over touched RVV/common files:
  common `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp` has no
  RVV/intrinsic/`tcrv_rvv` semantic branch matches; RVV touched-file matches
  are the new RVV-owned op/negative tests, existing explicit-only source-front
  door registration, and explanatory forbidden-pattern text.
- `git diff --check`

## Status

Completed locally. Ready for Trellis finish/archive and one coherent commit.
