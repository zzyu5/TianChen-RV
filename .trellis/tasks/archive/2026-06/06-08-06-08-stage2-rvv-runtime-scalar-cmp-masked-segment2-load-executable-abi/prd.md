# Stage2 RVV runtime-scalar-cmp masked segment2 load executable artifact ABI boundary

## Goal

Make the bounded RVV runtime-scalar compare + computed mask + masked
segment2-load route truthful at the executable artifact ABI boundary. The
production path must either prove that selected/pre-realized segment2-load
typed `tcrv_rvv` bodies reach RVV provider-owned segment2 route facts, common
EmitC materialization, target artifact export, generated bundle ABI, and
`ssh rvv` correctness, or fail closed before executable support is claimed
when runtime scalar binding, mask producer, tuple/field roles, field output ABI
order, inactive-lane policy, dtype/config, header/prototype binding, or
runtime AVL/VL facts are missing or stale.

## What I Already Know

- Hermes names this as the next RVV-first Stage2 owner after commit `96504ee5`
  completed focused composite runtime-scalar indexed gather-MAcc-scatter
  operand-binding artifact hardening.
- Current authority chain is selected `tcrv.exec` RVV variant -> typed
  low-level `tcrv_rvv` body -> RVV plugin-local realization/provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> generated
  bundle -> `ssh rvv` evidence when runtime correctness is claimed.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires RVV route support to
  start from typed/realized `tcrv_rvv` body facts, not route ids, artifact
  names, helper names, ABI strings, source-front-door markers, descriptors, or
  common EmitC semantic inference.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires provider-built
  operand-binding summaries for every exported ABI/header parameter and target
  artifact validation to compare candidate mirrors exactly against provider
  facts.
- `.trellis/spec/testing/mlir-testing-contract.md` requires generated-bundle
  dry-run evidence for source/header/harness facts and real `ssh rvv` evidence
  before claiming RVV runtime correctness.
- The bounded reference files include explicit and pre-realized
  runtime-scalar-cmp masked segment2-load fixtures, generated-bundle dry-run
  tests, RVV segment2 selected-body realization owners, route-family plan
  owners, route provider construction, target artifact validation, and
  generated-bundle harness generation.

## Requirements

- Inspect the current runtime-scalar-cmp masked segment2-load explicit and
  pre-realized production path across selected-body realization, RVV provider
  route facts, common EmitC payload, target artifact validation, generated
  bundle ABI, and existing dry-run/runtime evidence.
- Keep the module owner bounded to the executable artifact ABI seam for this
  route family:
  - typed segment2-load body facts;
  - runtime scalar comparison operands;
  - computed predicate/mask facts;
  - tuple field0/field1 roles;
  - source and output-field ABI roles;
  - masked active/inactive lane policy;
  - dtype/SEW/LMUL/config/policy;
  - runtime AVL/VL;
  - per-operand ABI/header bindings;
  - route validation, EmitC materialization, target artifact export, generated
    bundle ABI, and runtime evidence.
- If the production path is incomplete, implement one coherent submodule that
  wires the selected/pre-realized segment2-load path into provider-owned route
  facts, common EmitC materialization, target export, generated bundle ABI, and
  focused executable evidence.
- If the production path is already implemented but under-validated, harden the
  artifact/ABI seam with focused fail-closed coverage for stale or missing
  executable-boundary facts such as runtime scalar binding, mask producer,
  tuple field role, field output ABI order, inactive-lane policy, provider
  mirror, header/prototype binding, C type mapping, ABI value mapping, or
  unsupported executable route claim.
- Do not rely on route id, metadata, helper name, test name, descriptor,
  source-front-door marker, or common EmitC code as semantic authority.

## Acceptance Criteria

- [x] PRD truthfully identifies the bounded module owner and non-goals.
- [x] Focused code inspection records whether current HEAD has a production
  positive executable path or names the exact missing executable/fail-closed
  blocker.
- [x] Any source changes are production-path changes or focused tests tied to
  the runtime-scalar-cmp masked segment2-load executable artifact/ABI seam.
- [x] Positive executable claims include explicit and pre-realized evidence
  through materialized selected boundary, emission plan, target artifact export,
  generated bundle compile, and `ssh rvv` correctness.
- [x] Fail-closed claims include focused evidence for at least one stale or
  missing executable-boundary fact, such as stale runtime scalar binding, stale
  mask producer, stale tuple field role, stale field output ABI order, stale
  inactive-lane policy, stale provider mirror, stale header/prototype binding,
  wrong generated C type, wrong ABI value mapping, or unsupported executable
  route claim.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test` pass, or any inability to
  run them is reported with the exact blocker.
- [x] Relevant generated-bundle dry-run tests pass.
- [x] Bounded old-authority/source-front-door scan over touched files and added
  diff lines finds no new positive legacy route authority.
- [x] `git diff --check`, `git diff --cached --check`, and final `git status`
  are clean after commit unless the task is intentionally left open.

## Completion Notes

- Inspection found the explicit and pre-realized runtime-scalar-cmp masked
  segment2-load production paths are already wired through RVV selected-body
  realization/provider facts, common EmitC materialization, target artifact
  validation, generated bundle export, and executable runtime evidence.
- This round hardened the artifact ABI regression surface in focused fixtures:
  both explicit and pre-realized target artifact export now prove stale
  `tcrv_rvv.route_operand_binding_operands` mirrors for the `out1`
  field-output ABI role fail closed before target artifact acceptance.
- The pre-realized fixture now also checks the full provider-derived
  route-operand binding summary in the emission plan and generated header
  mirror, closing the missing positive assertion for the pre-realized owner
  path.
- No production runtime code changed. The existing provider/target validators
  already reject the stale field-output ABI summary; this round records that
  executable-boundary proof and refreshed `ssh rvv` correctness evidence.
- Spec update judgment: no `.trellis/spec/` edit was needed because
  `.trellis/spec/lowering-runtime/emitc-route.md` already contains the
  provider-owned computed-mask segment2 memory route facts contract, including
  exact route operand binding summary and target mirror validation.

## Definition Of Done

- Trellis task context, PRD, status, and journal are truthful.
- Relevant implementation/check context points to specs, not arbitrary source
  files.
- Focused checks pass after any self-repair.
- Task is finished/archived if complete.
- One coherent commit records the result, or the exact unfinished continuation
  point is documented.

## Out Of Scope

- Broad segment2 matrix expansion.
- Dtype/LMUL clone batches.
- Segment2 store/update/interleave/deinterleave expansion except as bounded
  reference.
- Further gather-MAcc-scatter composite mirror hardening.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Performance tuning databases, dashboards, indexes, or report-only work.
- Source-front-door positive routes.
- Common EmitC invention of RVV semantics.
- Mass rewrite of MAcc, indexed gather/scatter, reduction, compare/select,
  conversion, scalar fallback, or unrelated memory routes.

## Technical Notes

- Read first:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Previous bounded reference:
  `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-runtime-scalar-indexed-gather-macc-scatter-executable-abi/`.
- Likely code owners to inspect:
  `include/TianChenRV/Plugin/RVV/RVVEmitCSegment2RouteFamilyPlanOwners.h`,
  `include/TianChenRV/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVSegment2MemorySelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCSegment2RouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  focused dry-run tests and fixtures matching
  `runtime-scalar-cmp-masked-segment2-load`.
