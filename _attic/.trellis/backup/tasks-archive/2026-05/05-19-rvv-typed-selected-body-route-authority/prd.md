# RVV typed selected-body route authority

## Goal

Replace the RVV EmitC route provider's active route-authority surface with a
typed selected-body route description derived from the selected `tcrv_rvv`
body/config/memory/runtime ABI facts. The retained bounded i32m1 arithmetic
route remains only as an ordinary specialization of that validated description.

## What I already know

* Current repo root is `/home/kingdom/phdworks/TianchenRV`.
* Current HEAD is `03b9c49 docs: align RVV stage steering with grill 518`.
* Worktree was clean before task creation.
* No `.trellis/.current-task` existed; this task was created from the Hermes
  direction brief.
* The RVV plugin spec requires the chain:
  selected `tcrv.exec` RVV variant -> explicit typed `tcrv_rvv` body ->
  RVV-owned legality/route construction -> `TCRVEmitCLowerableRoute` ->
  common EmitC/export mechanics.
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` already walks and validates
  selected `tcrv_rvv` body structure, but its public and internal route surface
  is still centered on `RVVI32M1ArithmeticOp`,
  `RVVI32M1ArithmeticRouteSpec`, route-id-to-operation lookup, and exact
  i32m1 intrinsic spellings.
* `lib/Plugin/RVV/RVVExtensionPlugin.cpp` currently derives emission-plan
  runtime ABI by symbolizing the built route id back to an arithmetic op.
* `lib/Target/RVV/RVVTargetSupportBundle.cpp` currently validates target
  artifact candidates by rebuilding the route, then symbolizing the route id
  back to the arithmetic op.

## Requirements

* Build a typed selected-body route description in RVV plugin code after
  validating:
  * selected variant ownership and explicit `tcrv_rvv` body presence;
  * operation kind from typed body ops, not route id or artifact metadata;
  * SEW/LMUL/policy and VL relationship from `setvl`/`with_vl`;
  * memory form from explicit load vs RHS broadcast body structure;
  * runtime AVL/VL/value bindings from explicit `tcrv_rvv.runtime_abi_value`
    SSA definitions and roles;
  * intrinsic mapping from the validated body-derived operation/config/memory
    description.
* Make route ids and runtime ABI names labels produced after validation, not
  inputs that authorize route semantics.
* Rewire RVV emission planning and RVV target artifact validation to consume
  the body-derived route description instead of route-id-to-operation
  symbolization.
* Keep common EmitC/export code semantic-neutral; no RVV dtype/policy/op
  selection belongs in common code.
* Keep scope bounded to the existing provider route boundary and focused tests.

## Acceptance Criteria

* [ ] `RVVEmitCRouteProvider` exposes a selected-body route-description entry
      point and uses it to build `TCRVEmitCLowerableRoute`.
* [ ] The retained i32m1 add/sub/mul/compare-select cases are selected only
      after typed body/config/runtime ABI validation succeeds.
* [ ] Route id and artifact/source metadata mismatches fail closed after the
      selected body has been rebuilt and validated.
* [ ] Emission planning and RVV target artifact candidate validation no longer
      derive arithmetic semantics by parsing route ids.
* [ ] Focused positive coverage proves one selected typed `tcrv_rvv` body
      route succeeds.
* [ ] Focused negative coverage proves missing/inconsistent typed authority or
      legacy/metadata-authority mismatch fails closed.
* [ ] Focused build/lit checks for the changed provider boundary pass.

## Out of Scope

* No new broadcast, compare/select, reduction, conversion, dtype, LMUL,
  source-shape, or intrinsic coverage beyond preserving already-retained
  bounded cases.
* No scalar, IME, Offload, TensorExt, global autotuning, dashboards, reports,
  or broad smoke matrices.
* No high-level Linalg/Vector/StableHLO frontend generalization.
* No descriptor-driven computation, direct-C semantic export, or compatibility
  wrappers preserving old i32 route authority.
* No runtime/correctness/performance claim without real `ssh rvv` evidence.

## Technical Notes

* Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
* Relevant code read:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  focused tests under `test/Conversion/EmitC`, `test/Transforms/RVV`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.
