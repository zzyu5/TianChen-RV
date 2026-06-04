# Stage2 RVV i32-to-f32 dequantization route foundation

## Goal

Implement one production route-supported RVV dequantization/conversion
foundation: typed i32 source or accumulator values in a selected `tcrv_rvv`
body become typed f32 output values through RVV plugin-owned i32-to-f32
conversion and runtime scale application, with the output store/runtime
boundary represented structurally in `tcrv_rvv`.

This task closes the next typed output-dtype boundary after the completed
product-reduction chain. It is generic low-level RVV conversion/dequant support,
not a q8/q4/llama benchmark route and not a new high-level frontend.

## What I Already Know

- The session began on `main` with a clean worktree at
  `0bcd8361 rvv: close product reduction executable abi`.
- No active Trellis task existed, so this task was created from the Hermes
  Direction Brief.
- The prior archived task
  `.trellis/tasks/archive/2026-06/06-05-06-05-stage2-rvv-product-reduction-executable-abi-closure/`
  closed the i32 product-reduction executable ABI boundary and left no
  unfinished continuation point.
- `.trellis/spec/index.md` defines the current RVV-first chain as selected
  `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/route provider -> `TCRVEmitCLowerableRoute` -> common
  EmitC -> target artifact.
- `.trellis/spec/extension-plugins/rvv-plugin.md` and
  `.trellis/spec/lowering-runtime/emitc-route.md` already contain a widening
  conversion dtype-policy owner, provider facts preflight, and target
  validation contract for integer widening conversions.
- Current source search found existing widening conversion surfaces around
  `tcrv_rvv.typed_widening_conversion_pre_realized_body`,
  `tcrv_rvv.widening_convert`, `RVVWideningConversionRouteFacts`, conversion
  dtype-policy validation, statement-plan owners, and target route-family
  validation. No obvious dequant/runtime-scale support was present.

## Scope

- Add the smallest coherent production route-supported dequantization slice
  for i32 input/accumulator values, f32 output values, and a runtime f32 scale
  parameter.
- The selected body must structurally carry source/result dtype facts,
  SEW/LMUL/policy, AVL/VL placement, runtime `scale`, input/output ABI roles,
  operation kind, conversion, scale application, and output store.
- RVV plugin route planning must derive conversion, scale, vector/C type,
  header/intrinsic, route-family plan, route operand binding, statement-plan,
  and mirror facts from typed body/config/runtime facts.
- Common EmitC/export must remain neutral materialization of provider-built
  route payloads.
- Target artifact validation must consume provider-owned contracts and reject
  stale mirrors, missing scale, dtype mismatch, unsupported policy/config, and
  any route-string/artifact-name authority.
- Add focused positive and negative tests that prove route-supported behavior
  and fail-closed behavior for the new slice.

## Requirements

- Production code changes are required before task closure.
- Dtype/config/scale authority must come from typed `tcrv_rvv` body/config and
  imported runtime ABI facts, not route ids, artifact names, ABI names, test
  names, op-kind strings, metadata mirrors, descriptors, or common EmitC.
- The runtime scale must be a structurally imported/consumed runtime value; a
  constant scale or harness-side convention is not enough for route support.
- The route may be bounded to one initial supported form, such as unit-stride
  i32 source -> f32 output with runtime f32 scale, as long as unsupported
  zero-point, clamp, alternate dtype, LMUL, policy, or memory forms fail closed
  with targeted diagnostics.
- A retained i32 case is allowed only as an ordinary source dtype instance of
  the corrected typed RVV surface; do not add `tcrv_rvv.i32_*`,
  `!tcrv_rvv.i32m*`, `RVVI32M1*`, `rvv-i32m1`, or q-name authority.
- If executable correctness is claimed, real `ssh rvv` evidence is required.
  This task may stop at route-supported and generated-artifact validation
  without claiming runtime correctness.

## Acceptance Criteria

- [ ] A typed RVV body/config surface accepts the chosen i32-to-f32
  dequantization form only when source dtype, result dtype, runtime scale,
  SEW/LMUL/policy, AVL/VL, operation kind, and output boundary are structurally
  present.
- [ ] RVV plugin route planning builds a provider-owned route/fact surface for
  i32-to-f32 conversion plus runtime-scale application.
- [ ] Statement-plan ownership emits the route steps from provider facts,
  including setvl/source load/conversion/scale/store or an equivalent
  structurally validated sequence.
- [ ] Target route-family validation rejects stale mirrors, missing runtime
  scale, source/result dtype mismatch, unsupported policy/config, stale
  non-dequant facts, and route-string/artifact-name authority.
- [ ] Focused positive lit and/or C++ tests prove route-supported
  i32-to-f32 dequantization behavior through the provider path.
- [ ] Focused negative tests cover at least missing scale, dtype mismatch, and
  stale metadata/mirror or unsupported config.
- [ ] `tianchenrv-rvv-extension-plugin-test` and
  `tianchenrv-target-artifact-export-test` pass if provider/target code is
  changed.
- [ ] A generated artifact check is run if target export behavior changes.
- [ ] Bounded old-authority and q-name scan over touched files passes.
- [ ] `git diff --check` and `git diff --cached --check` pass.
- [ ] The task is finished/archived and one coherent commit is created if all
  acceptance criteria are satisfied; otherwise the task remains open with the
  exact continuation point.

## Definition Of Done

- One bounded i32-to-f32 runtime-scale dequantization route is supported by the
  production RVV typed-body/provider/statement-plan/target-validation path.
- Unsupported or stale forms fail closed before common EmitC/export can
  authorize the route.
- The final report states whether the capability is route-supported only or
  executable with `ssh rvv` evidence.
- The worktree is clean after the final commit if the task is complete.

## Out Of Scope

- q8/q4/llama benchmark-specific route authority.
- Zero-point, clamp, full quantized-kernel closure, full contraction-to-output
  pipeline, or broad dtype/LMUL clone batches.
- High-level Linalg/frontend lowering.
- Descriptor-driven computation, source-front-door/source-artifact positive
  routes, handwritten C demos as the main deliverable, dashboards, or broad
  smoke matrices.
- Compatibility wrappers preserving legacy i32m1 authority.
- Repeating product-reduction `ssh rvv` evidence unless the new dequant route
  claims executable correctness.

## Technical Notes

- Specs read before PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/index.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-06/06-05-06-05-stage2-rvv-product-reduction-executable-abi-closure/prd.md`.
- First implementation targets to inspect:
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`,
  `lib/Dialect/RVV/IR/RVVDialect.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCResidualStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseArithmeticStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  and focused RVV dialect/target tests.
