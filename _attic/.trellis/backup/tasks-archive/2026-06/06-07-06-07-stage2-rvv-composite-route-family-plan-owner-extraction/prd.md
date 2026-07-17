# Stage2 RVV composite route-family plan owner extraction

## Goal

Extract or harden one production RVV plugin-owned plan contract for the
`runtime_scalar_cmp_masked_indexed_gather_macc_scatter` route family. The
selected-body runtime scalar compare, computed mask, indexed gather, MAcc,
scatter, accumulator/result preservation, dtype/SEW/LMUL/policy,
runtime AVL/VL, resource/capability, ABI/header, and provider/target
validation facts must flow through an explicit composite plan owner consumed by
provider construction and target artifact validation. The facts must not be
accepted from route ids, artifact names, mirror strings, or Common EmitC
semantic inference.

## What I Already Know

* Current HEAD is `beeb3621`, and the worktree was clean at session start.
* The preceding archived task
  `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-runtime-scalar-cmp-indexed-gather-macc-scatter-artifact-abi/`
  proved explicit and pre-realized generated bundles compile/run on `ssh rvv`
  for counts `0,1,16,17,257`, RHS scalars `-37,91`, and patterns `0,1`.
* `.trellis/spec/extension-plugins/rvv-plugin.md` defines this composite as a
  Stage 2 plugin-owned route contract with ABI order
  `cmp_lhs,rhs_scalar,gather_src,payload,acc,index,dst,n`.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires provider-built
  `TCRVEmitCLowerableRoute` payloads and forbids Common EmitC from inferring
  RVV dtype, operation, mask, ABI, intrinsic, or schedule semantics.
* Current code has explicit resource selection facts in
  `RVVCompositeGatherMAccScatterResourceSelection`, but the wider composite
  route-family facts are still distributed across route slice, route
  description, statement planning, metadata mirror, and target validation
  fields.

## Requirements

* Add or harden an explicit RVV plugin-owned composite gather/MAcc/scatter
  route-family plan contract that carries the route's runtime scalar compare,
  computed mask, indexed gather, MAcc, scatter, dtype/config, runtime AVL/VL,
  resource selection, ABI/header binding, and provider support facts.
* Route construction/provider facts for
  `RuntimeScalarComputedMaskIndexedGatherMAccScatter` must consume that
  composite plan contract rather than re-accepting the same facts only through
  incidental route description or mirror strings.
* Target artifact validation must consume the explicit composite plan contract
  or a validation contract derived from it, and must fail closed for at least
  one stale or missing composite fact.
* Preserve the existing positive explicit and pre-realized composite generated
  bundle behavior. If executable behavior is claimed after the refactor, run at
  least one non-dry-run `ssh rvv` preservation check.
* Keep Common EmitC/export neutral: no new RVV semantic inference, no
  descriptor/source-front-door positive route, and no new route coverage.

## Acceptance Criteria

* [x] A focused source diff in the RVV plugin/target composite planning path
      introduces or hardens an explicit composite route-family plan owner
      consumed by production provider or target validation code.
* [x] The plan contract includes runtime scalar compare, computed mask,
      indexed gather, masked MAcc, masked indexed scatter,
      accumulator/result preservation, dtype/SEW/LMUL/policy, runtime AVL/VL,
      composite resource/capability, ABI/header, and provider support facts.
* [x] At least one targeted fail-closed diagnostic proves stale or missing
      composite facts are rejected before target artifact acceptance or Common
      EmitC semantic inference.
* [x] Positive coverage for the focused explicit/pre-realized composite
      fixtures still passes.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Relevant lit/FileCheck coverage for the composite fixtures passes.
* [x] `scripts/rvv_generated_bundle_abi_e2e.py` generated-bundle dry-run
      coverage for the route passes, and one `ssh rvv` non-dry-run
      preservation check passes if runtime correctness is claimed.
* [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      `!tcrv_rvv.i32m*`, descriptor-driven compute, source-front-door, or
      Common EmitC semantic authority.
* [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are clean or explicitly explained.

## Completion Notes

* Added `RVVCompositeGatherMAccScatterRouteFamilyPlan` and a production
  computed-mask family owner path that builds and verifies the explicit
  composite plan before provider route construction.
* Added composite route-family plan id and typed compute chain facts to the
  route description, indexed validation contract, metadata mirror contract, and
  target artifact validation contract.
* Tightened target validation so composite resource operation, memory form,
  dtype/config/policy, ABI order, and legal resource selection must match the
  provider-owned composite contract.
* Added header metadata evidence for `tcrv_rvv.composite_route_family_plan` and
  `tcrv_rvv.composite_typed_compute_chain`.
* Added focused FileCheck coverage for positive explicit/pre-realized plan
  metadata and a stale composite plan mirror rejection.
* No `.trellis/spec/` update was needed: the existing RVV plugin and EmitC
  route specs already describe this ownership boundary.

## Validation Evidence

* `cmake --build build --target tcrv-opt tcrv-translate -j 8` passed after
  relinking lit tools.
* `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j 8`
  passed after target support changes.
* `build/bin/tianchenrv-rvv-extension-plugin-test` passed.
* `build/bin/tianchenrv-target-artifact-export-test` passed.
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='runtime-scalar-cmp-masked-indexed-gather-macc-scatter'`
  from `build/test` passed: 3 tests selected, 3 passed.
* `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed.
* Explicit and pre-realized generated-bundle dry-runs for
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter` passed.
* Non-dry-run `ssh rvv` explicit preservation passed for runtime counts
  `0,1,16,17,257`, RHS scalars `-37,91`, and patterns `0,1`.
* `git diff --check` passed before staging.
* Bounded old-authority scan over touched files found only pre-existing
  descriptor guards, old i32m1 intrinsic constants, and old-op rejection checks;
  added diff lines had no legacy authority matches.

## Definition Of Done

* Composite route-family plan ownership is more explicit in production source,
  not only in tests, scripts, metadata mirrors, or reports.
* Unsupported or stale composite runtime scalar, mask, index/gather, MAcc,
  scatter, resource, capability, ABI/header, dtype/config, or policy facts
  fail closed with targeted RVV-owned diagnostics.
* Trellis task context and workspace journal are updated truthfully.
* Specs are reviewed; update `.trellis/spec/` only if this round discovers a
  reusable contract not already captured.
* One coherent commit is created when the task is complete.

## Technical Approach

Use the smallest production owner extraction that moves real authority:

* Introduce or harden a composite gather/MAcc/scatter route-family plan object
  near `RVVEmitCRoutePlanning` / computed-mask memory plan owners.
* Build the plan from selected typed body/config/runtime/capability facts,
  including the existing composite resource selection.
* Consume the plan when deriving route validation metadata and target artifact
  validation expectations.
* Add focused fail-closed test coverage for a stale/missing plan fact and rerun
  the existing positive composite checks.

## Decision (ADR-lite)

**Context**: The route is already executable with explicit and pre-realized
`ssh rvv` evidence, so another evidence-only closure is too small. The next
bottleneck is durable production ownership of the composite plan facts.

**Decision**: Complete one coherent submodule around the composite plan
contract owner. Keep the change inside RVV plugin/target validation boundaries
and preserve Common EmitC neutrality.

**Consequences**: This may not finish every possible split of the large
monolithic route slice, but it must move at least one production validation
path to consume the explicit composite contract and leave an exact continuation
point for any remaining decomposition.

## Out Of Scope

* New route family coverage, dtype/LMUL clone batches, broad fixture matrices,
  dashboards, or report-only work.
* High-level Linalg/Vector/StableHLO frontend work or per-Linalg route
  authority.
* Scalar fallback executable routes, IME/offload/TensorExt work, performance
  databases, or global tuning/profile systems.
* Descriptor-driven C/source export, source-front-door positive routes, or
  Common EmitC RVV semantic invention.
* Mass rewrites of unrelated elementwise, segment2, reduction, conversion, or
  low-precision routes.

## Technical Notes

* Direction source: Hermes/Codex worker direction brief supplied on
  2026-06-07.
* Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`.
* Primary implementation files:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
* Selected-body realization files to preserve:
  `include/TianChenRV/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.h`,
  `lib/Plugin/RVV/RVVCompositeGatherMAccScatterSelectedBodyRealizationOwner.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`.
* Evidence files/scripts:
  `scripts/rvv_generated_bundle_abi_e2e.py` and the explicit/pre-realized
  Target/RVV fixtures for
  `runtime_scalar_cmp_masked_indexed_gather_macc_scatter`.
