# RVV arithmetic-family pre-realized realization closure

## Goal

Extend the RVV plugin-owned pre-realized selected-body realization hook from
the completed add-only proof to the already-supported i32m1 arithmetic family
specializations: add, sub, and mul. The realized output must remain the
existing explicit `setvl -> with_vl -> load -> compute -> store` selected RVV
body consumed by the RVV provider, common EmitC materializer, and target
artifact exporter.

## Current Facts

- Current HEAD at task creation is `f52b457`; the worktree was clean.
- No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief before source edits.
- The archived task
  `.trellis/tasks/archive/2026-05/05-19-rvv-selected-body-realization-hook`
  added `tcrv_rvv.i32_binary_pre_realized_body` and completed the add-only
  selected-body realization hook.
- Current `I32BinaryPreRealizedBodyOp::verify` rejects every `op_kind` except
  `"add"`.
- Current `RVVExtensionPlugin::validatePreRealizedRVVSelectedBody` and
  `realizePreRealizedRVVSelectedBody` are add-only.
- The existing realized selected-body provider route already consumes
  `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, and `tcrv_rvv.i32_mul` bodies and
  has retained operation-specific route/ABI labels after typed body
  validation.
- Common EmitC/materializer and target export must remain neutral consumers of
  the provider-built route; they must not learn pre-realized RVV semantics.

## Requirements

1. Keep `tcrv_rvv.i32_binary_pre_realized_body` as the bounded RVV-owned typed
   pre-realized selected-body surface, not a route id, descriptor,
   source-front-door marker, direct-C exporter, artifact authority, or common
   EmitC semantic hook.
2. Extend verifier support from add-only to exactly `op_kind = "add"`,
   `"sub"`, and `"mul"` for the existing i32/m1, vector-rhs-load,
   tail/mask-agnostic specialization.
3. Extend RVV plugin-local selected-body realization so each supported
   `op_kind` produces the matching realized compute op:
   `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, or `tcrv_rvv.i32_mul`.
4. Preserve the exact runtime ABI value operands and runtime `n`/AVL SSA value
   from the pre-realized op into the realized body.
5. Preserve selected variant origin, required capabilities, dispatch/fallback
   structure, selected path role, dtype/config, memory form, and policy facts.
6. Fail closed before route construction for unsupported op kinds, unsupported
   config/memory/policy, wrong runtime ABI roles, missing runtime `n`/AVL, stale
   authority metadata, or mixed pre-realized plus already-realized bodies.
7. Existing pre-realized add behavior and existing fully realized add/sub/mul
   fixture behavior must remain unchanged.
8. Add positive pre-realized sub and mul target fixtures proving selected-body
   realization reaches emission-plan metadata and local target header/artifact
   routing.

## Acceptance Criteria

- [x] `tcrv_rvv.i32_binary_pre_realized_body` verifier accepts only add/sub/mul
      on the existing bounded i32/m1 vector-rhs-load policy/config surface.
- [x] RVV plugin realization materializes add/sub/mul pre-realized bodies into
      matching realized `i32_add`/`i32_sub`/`i32_mul` body shapes.
- [x] The RVV provider derives operation, route id, runtime ABI name, typed
      compute op, and artifact metadata from the realized selected body, not
      from route IDs or artifact names.
- [x] Positive pre-realized add/sub/mul fixtures reach supported emission-plan
      metadata and target header or bundle artifact generation.
- [x] Existing fully realized add/sub/mul selected-body artifact fixtures still
      pass.
- [x] Negative coverage proves unsupported op kind, unsupported config or
      memory form, wrong runtime ABI role, missing runtime `n`/AVL, stale
      route/artifact/source authority metadata, and mixed realized/pre-realized
      bodies fail before route construction.
- [x] Bounded residue scans over touched files show no source-front-door
      default authority, descriptor/direct-C/source-export restoration,
      route-id or artifact-name authority, or common-code RVV semantic branch.
- [x] No runtime, correctness, or performance claim is made unless real
      `ssh rvv` evidence is collected for the generated artifact.

## Technical Approach

Reuse the existing selected lowering-boundary materialization entry point.
Replace the add-only validation and builder helpers with a small RVV
plugin-local operation-kind mapping for the existing selected arithmetic
specializations. The mapping may choose among RVV-owned realized compute op
builders after the pre-realized op has already carried the typed
operation/config/memory/runtime facts and after verifier/plugin checks have
accepted those facts.

The provider and common EmitC/target export path should remain unchanged unless
current evidence shows a stale add-only assumption after realization. The
provider should continue to see only the realized body and use its existing
typed body analysis.

## Out of Scope

- No reductions, compare/select extension, conversion, broadcast expansion,
  new dtype, new LMUL, new source shape, new intrinsic family, or high-level
  Linalg/Vector/StableHLO lowering.
- No expansion of legacy route tables as architecture. Add/sub/mul retained
  labels are ordinary specializations after typed body validation.
- No source-front-door default restoration.
- No descriptor-driven computation, direct-C/source-export route restoration,
  common EmitC RVV semantic branching, dashboard/status-only work, or broad
  smoke matrix.
- No RVV runtime/correctness/performance claim without real `ssh rvv` evidence.

## Validation Plan

- Build focused targets: `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-rvv-extension-plugin-test`, and
  `tianchenrv-target-artifact-export-test` if impacted.
- Run focused lit for:
  - `test/Target/RVV/pre-realized-selected-body-artifact-add.mlir`
  - new pre-realized sub and mul target fixtures
  - `test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir`
  - existing explicit selected-body add/sub/mul target fixtures
- Generate local target artifact bundles for pre-realized add/sub/mul through
  the public selected-body realization, emission-plan, and bundle export path.
- Run bounded residue scans over touched RVV/common files for
  source-front-door default authority, descriptors, direct-C/source-export,
  route-id/artifact-name authority, and common EmitC RVV semantic branches.
- Run `python3 ./.trellis/scripts/task.py validate`.
- Run `git diff --check`.

## Relevant Specs And Context

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/tasks/archive/2026-05/05-19-rvv-selected-body-realization-hook/prd.md`

## Initial Code Surface

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `lib/Dialect/RVV/IR/RVVDialect.cpp`
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
- `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
- `test/Target/RVV/pre-realized-selected-body-artifact-add.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-add.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-sub.mlir`
- `test/Target/RVV/explicit-selected-body-artifact-mul.mlir`
- `test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir`

## Implementation Notes

- Extended the RVV dialect verifier for
  `tcrv_rvv.i32_binary_pre_realized_body` from add-only to exactly add, sub,
  and mul on the existing SEW32/LMUL m1/vector-rhs-load/tail-mask-agnostic
  surface.
- Replaced the add-only RVV plugin realization helper with a plugin-local
  operation-kind mapping from the verified pre-realized typed body to the
  matching realized compute op name. The realized body still uses explicit
  `runtime_abi_value`, `setvl`, `with_vl`, two vector loads, one compute op,
  and one store.
- Added a fail-closed mixed-body guard in
  `RVVExtensionPlugin::materializeSelectedLoweringBoundary`: an input that
  contains both a pre-realized body and an already-realized `setvl/with_vl`
  body is rejected before route construction.
- Updated the long-term RVV plugin spec scenario so the pre-realized
  selected-body contract is add/sub/mul arithmetic-family scoped instead of
  add-only.
- Common EmitC/materializer and target export code were not changed.

## Evidence

- Positive fixtures:
  - `test/Target/RVV/pre-realized-selected-body-artifact-add.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-sub.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-mul.mlir`
- Existing fully realized fixtures preserved:
  - `test/Target/RVV/explicit-selected-body-artifact-add.mlir`
  - `test/Target/RVV/explicit-selected-body-artifact-sub.mlir`
  - `test/Target/RVV/explicit-selected-body-artifact-mul.mlir`
- Negative fixture:
  `test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir`.
- Local materialized artifact bundles:
  - `artifacts/tmp/rvv_arithmetic_pre_realized_realization_closure/add/`
  - `artifacts/tmp/rvv_arithmetic_pre_realized_realization_closure/sub/`
  - `artifacts/tmp/rvv_arithmetic_pre_realized_realization_closure/mul/`
- Each bundle contains:
  - `artifact-0-riscv-elf-relocatable-object-rvv-i32m1-arithmetic-emitc-route-family.o`
  - `artifact-1-runtime-callable-c-header-rvv-i32m1-arithmetic-emitc-route-family.header.h`
  - `tianchenrv-target-artifact-bundle.index`
- Bundle indexes record selected variants
  `@pre_realized_body_rvv_i32_add`,
  `@pre_realized_body_rvv_i32_sub`, and
  `@pre_realized_body_rvv_i32_mul`, runtime ABI names
  `rvv-i32m1-add-callable-c-abi.v1`,
  `rvv-i32m1-sub-callable-c-abi.v1`, and
  `rvv-i32m1-mul-callable-c-abi.v1`, and matching typed compute metadata
  `tcrv_rvv.i32_add`, `tcrv_rvv.i32_sub`, and `tcrv_rvv.i32_mul`.
- No runtime/correctness/performance claim was made; no `ssh rvv` run was
  collected for this task.

## Checks Run

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
- `./build/bin/tianchenrv-rvv-extension-plugin-test`
- `./build/bin/tianchenrv-target-artifact-export-test`
- From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv ../test/Target/RVV/pre-realized-selected-body-artifact-add.mlir ../test/Target/RVV/pre-realized-selected-body-artifact-sub.mlir ../test/Target/RVV/pre-realized-selected-body-artifact-mul.mlir ../test/Transforms/LoweringBoundary/rvv-pre-realized-selected-body-negative.mlir ../test/Target/RVV/explicit-selected-body-artifact-add.mlir ../test/Target/RVV/explicit-selected-body-artifact-sub.mlir ../test/Target/RVV/explicit-selected-body-artifact-mul.mlir`
  passed 7/7.
- Local bundle generation for pre-realized add/sub/mul through
  `tcrv-opt --tcrv-materialize-selected-lowering-boundaries --tcrv-materialize-emission-plans`
  piped into
  `tcrv-translate --tcrv-export-target-artifact-bundle`.
- Bounded residue scans over touched RVV/common surfaces:
  - Common `lib/Conversion/EmitC/TCRVEmitCLowerableMaterializer.cpp` has no
    RVV/`tcrv_rvv`/`__riscv`/`i32_` semantic branch matches.
  - RVV touched-file matches are the existing explicit-only source front door,
    negative-test stale metadata, and explanatory forbidden-pattern text.
- `git diff --check`
- Initial bare `python3 ./.trellis/scripts/task.py validate` failed because
  this repository script requires an explicit task dir argument.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-19-rvv-arithmetic-pre-realized-realization-closure`

## Status

Completed locally. Ready for Trellis finish/archive and one coherent commit.
