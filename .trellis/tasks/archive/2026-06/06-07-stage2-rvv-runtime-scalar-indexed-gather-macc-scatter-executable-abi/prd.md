# Stage2 RVV runtime-scalar indexed gather-MAcc-scatter executable artifact ABI boundary

## Goal

Make the bounded RVV runtime-scalar compare + computed mask + masked indexed
gather + masked MAcc + masked indexed scatter composite path truthful at the
executable artifact ABI boundary. The production path must either prove the
selected/pre-realized composite body reaches provider-owned route facts,
common EmitC materialization, generated bundle export, and ssh rvv correctness,
or fail closed before executability is claimed when a required runtime,
mask/index, gather, MAcc, scatter, dtype/config, ABI/header, or runtime AVL/VL
fact is missing or stale.

## What I Already Know

- Hermes direction names this as the next RVV-first executable compiler path
  owner after vector source-front-door family-owner cleanup.
- Current authority chain is selected `tcrv.exec` RVV variant -> typed
  low-level `tcrv_rvv` body -> RVV plugin-local realization/provider ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact -> generated
  bundle -> `ssh rvv` evidence when runtime correctness is claimed.
- `.trellis/spec/extension-plugins/rvv-plugin.md` already defines the
  runtime-scalar indexed gather-MAcc-scatter contract, including ABI order
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`, shared compare mask
  and index, gather passthrough, MAcc operand roles, scatter storing the MAcc
  result, composite plan facts, pre-realized owner requirements, and
  fail-closed diagnostics for stale or incomplete clusters.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires RVV provider-built
  route validation contracts and forbids Common EmitC or target export from
  inferring RVV semantics from metadata, route ids, descriptors, artifact names,
  source-front-door markers, or C spelling.
- `.trellis/spec/testing/mlir-testing-contract.md` requires generated-bundle
  dry-run evidence for composite resource mirrors and `ssh rvv` compile/run
  evidence before claiming runtime correctness.

## Requirements

- Inspect the current composite selected-body realization owner, RVV provider,
  target artifact validation, generated-bundle script, and focused fixtures to
  determine whether the executable composite boundary is implemented,
  dry-run-only, under-validated, or stale.
- Keep the module owner bounded to the runtime-scalar indexed
  gather-MAcc-scatter executable artifact ABI boundary.
- If a production gap exists, implement one coherent submodule that either:
  - wires the selected/pre-realized composite path into provider-owned route
    facts, common EmitC materialization, target export, generated bundle ABI,
    and focused executable evidence; or
  - hardens the composite owner/provider/target/export boundary so incomplete
    or stale gather/MAcc/scatter clusters fail closed with targeted diagnostics
    before executable artifact support is claimed.
- Do not add source-front-door positive routes, common EmitC semantic branches,
  descriptor/direct-C computation, dtype/LMUL clone batches, unrelated route
  reworks, dashboards, broad smoke matrices, or Stage3 registry polishing.
- Do not use route ids, artifact metadata, helper names, ABI strings, test
  names, descriptors, source-front-door markers, or Common EmitC code as route
  authority.

## Acceptance Criteria

- [x] PRD truthfully identifies the bounded module owner and non-goals.
- [x] Focused code inspection records whether current HEAD already has a
  production positive path or which fail-closed boundary is missing.
- [x] Any source changes are production-path changes or focused tests tied to
  the composite executable artifact/ABI seam.
- [x] Positive executable claims include generated bundle compile/run evidence
  on `ssh rvv` for the runtime-scalar indexed gather-MAcc-scatter route.
- [x] Fail-closed claims include focused evidence for at least one stale or
  missing executable-boundary fact, such as stale mask/index, gather not feeding
  MAcc, scatter not storing MAcc result, inactive-lane passthrough drift, stale
  ABI/header binding, or unsupported executable route claim.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test` pass, or any inability to
  run them is reported with the exact blocker.
- [x] Relevant generated-bundle dry-run or lit coverage passes.
- [x] Bounded old-authority/source-front-door scan over touched files and added
  diff lines finds no new positive legacy route authority.
- [x] `git diff --check`, `git diff --cached --check`, and final `git status`
  are clean after commit unless the task is intentionally left open.

## Completion Notes

- Inspection found the production explicit and pre-realized composite paths are
  already wired through plugin-local selected-body realization, provider-owned
  composite route facts, target artifact validation, generated-bundle export,
  and prior ssh rvv evidence.
- This round hardened the executable artifact ABI boundary with focused lit
  fail-closed coverage for stale `tcrv_rvv.route_operand_binding_operands`
  mirrors in both explicit and pre-realized composite fixtures.
- No runtime code changed, so no new ssh rvv runtime correctness claim is made
  in this round. Existing generated-bundle dry-run evidence was rerun for both
  explicit and pre-realized composite paths.

## Definition Of Done

- Trellis task context, PRD, status, and journal are truthful.
- Relevant implementation/check context points to specs, not arbitrary source
  files.
- Focused checks pass after any self-repair.
- Task is finished/archived if complete.
- One coherent commit records the result, or the exact unfinished continuation
  point is documented.

## Out Of Scope

- Further vector source-front-door cleanup or Stage3 plugin-general registry
  polishing.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Performance tuning database or dashboard/index/report-only work.
- Source-front-door positive route construction.
- Common EmitC invention of RVV semantics.
- Mass rewrite of unrelated memory, segment2, reduction, compare/select,
  conversion, scalar fallback, or non-composite routes.

## Technical Notes

- Read first:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Likely code owners to inspect:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h`,
  `lib/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCIndexedGatherScatterRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  focused tests/fixtures matching
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`.
