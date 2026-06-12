# Stage2 RVV computed-masked indexed scatter-store executable artifact ABI boundary

## Goal

Complete one bounded Stage 2 RVV workflow submodule: selected or pre-realized
computed-masked indexed scatter-store bodies must carry compare-produced mask
facts, indexed destination/index/value roles, active-lane indexed store
semantics, inactive-lane no-write preservation policy, dtype/SEW/LMUL/config/
policy, runtime AVL/VL, RVV plugin-owned route validation, EmitC
materialization, target artifact export, generated bundle ABI, and truthful
runtime evidence through the provider-owned route boundary.

## What I Already Know

* Session-start repository state was clean on `main`; recent HEAD is
  `19abf4c4 rvv: add runtime scalar cmp indexed gather ABI`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes Direction Brief before source edits.
* The previous archived task completed runtime-scalar-cmp masked indexed
  gather-load through selected/pre-realized body realization, route validation,
  target artifact export, generated bundle ABI, and `ssh rvv` evidence.
* The relevant spec already defines a provider-owned computed-mask indexed
  memory fact surface and validation contract for both gather and scatter.
* Scatter differs from gather at the executable boundary: it must carry
  `tcrv_rvv.masked_indexed_store`, memory form
  `computed-mask-unit-load-indexed-scatter-store`, indexed destination memory
  form `masked-indexed-store`, index uniqueness `unique`, and a masked indexed
  store leaf. It must not carry a masked indexed load leaf or ordinary
  unit-store residue as store authority.
* Specs require RVV semantics to come from typed or realized `tcrv_rvv`
  body/config/capability/runtime facts. Route ids, metadata, helper names,
  test names, ABI strings, descriptors, artifact names, and common EmitC
  behavior are mirrors or neutral carriers only.

## Requirements

* Scope is one route family only:
  `computed_masked_indexed_scatter_store_unit_load`.
* Add or repair the production path:
  selected/pre-realized RVV body -> RVV selected-body realization owner ->
  realized typed `tcrv_rvv` setvl/compare-source-load/index-load/value-load/
  compare-mask/masked-indexed-store body -> RVV route-family/provider
  validation -> `TCRVEmitCLowerableRoute` -> common EmitC materialization ->
  target artifact export -> generated bundle ABI -> `ssh rvv` correctness
  when executable behavior is claimed.
* The selected/pre-realized body must structurally carry compare predicate,
  compare-produced mask facts, source/value load role, destination scatter
  role, index role, ABI order, runtime AVL/VL, dtype/SEW/LMUL/config, index
  EEW/offset unit/source/uniqueness, and mask/tail/inactive-lane no-write
  policy.
* Scatter route facts must use runtime ABI order
  `cmp_lhs,cmp_rhs,src,index,dst,n` and exported header/prototype binding for
  every ABI parameter.
* The RVV plugin owner must derive or validate route/type/header/intrinsic
  facts from typed body/config/capability/runtime facts, and fail closed with
  targeted diagnostics if any executable-boundary fact is missing or stale.
* Target artifact validation must consume the provider-owned
  `RVVComputedMaskIndexedMemoryRouteValidationContract` before accepting route
  payloads, statements, ABI mappings, headers/types, or metadata mirrors.
* Common EmitC/export must remain neutral: they may consume provider-built
  routes and explicit mirror fields, but must not invent RVV semantics,
  intrinsic spelling, dtype, memory form, index roles, inactive-lane policy, or
  output ordering.
* Preserve existing computed-mask indexed gather, runtime-scalar indexed
  gather, strided memory, segment2 memory, and unrelated mask-route behavior
  unless live code proves a directly shared validation repair is required.

## Acceptance Criteria

* [ ] Positive explicit and pre-realized computed-masked indexed scatter-store
      evidence reaches materialized selected boundary, emission plan, target
      artifact export, generated bundle compile, and `ssh rvv` correctness
      when executable behavior is claimed.
* [ ] Positive generated evidence exposes typed compute op, memory form,
      compare predicate construction, compare-produced mask facts, mask and
      inactive-lane policy, indexed destination role, index source/EEW/offset
      unit/uniqueness, source value load role, runtime AVL/VL, dtype/config,
      header/prototype binding, ABI order, and explicit provider-supported
      mirrors.
* [ ] Scatter facts carry a masked indexed store leaf and empty masked indexed
      load plus ordinary unit-store residue; stale gather/unit-store residue
      fails closed before target artifact acceptance.
* [ ] Focused fail-closed evidence rejects at least one stale or missing
      executable-boundary fact such as stale mask producer, stale indexed
      destination role, stale index role, stale stored value role, stale
      inactive-lane policy, stale header/prototype binding, stale route-family
      validation contract, wrong generated C type, wrong ABI value mapping, or
      unsupported executable route claim.
* [ ] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass.
* [ ] Relevant focused lit/generated-bundle dry-run tests pass.
* [ ] `scripts/rvv_generated_bundle_abi_e2e.py` self-test or focused dry-run
      path passes if the script changes.
* [ ] Bounded old-authority scan over touched files and added diff lines finds
      no new positive `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
      `!tcrv_rvv.i32m`, descriptor, direct-C, source-export, or
      source-front-door route authority.
* [ ] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are checked.

## Definition of Done

* The computed-masked indexed scatter-store executable artifact/ABI seam is
  implemented end to end, or the task remains open with a precise production
  blocker and exact next continuation point.
* Trellis task status/context and the codex workspace journal record the
  outcome truthfully.
* One coherent commit is created only after the bounded module is complete and
  verified.

## Out of Scope

* No runtime-scalar-cmp scatter expansion in this round.
* No broad indexed-memory matrix.
* No dtype/LMUL clone batch.
* No indexed gather rework except as reference.
* No gather-unit-store, indexed scatter-unit-load beyond this base scatter
  seam, strided-memory, segment2-memory, reduction, MAcc, compare/select,
  conversion, or unrelated mask-route expansion.
* No high-level Linalg, Vector, StableHLO, per-Linalg route authority, or
  source-front-door positive route.
* No dashboard, index, report-only, helper-only, or broad smoke-test work as
  the main achievement.
* No common EmitC invention of RVV semantics.
* No descriptor-driven, direct-C, source-export, or legacy i32 route authority.

## Technical Approach

Inspect the existing computed-mask indexed scatter-store selected-body,
route-family facts, statement-plan owner, target artifact validation, generated
bundle script, and dry-run fixtures. If the route is dry-run-only or
under-validated, repair the production seam so the scatter-store path consumes
RVV-owned typed body/config/runtime facts and provider-owned validation
contracts before common EmitC or target artifact export can accept it. Add
focused positive and fail-closed evidence for the changed seam only.

## Decision (ADR-lite)

**Context**: The read-side indexed gather boundary was completed in the
previous round, and the write-side indexed scatter boundary is now the narrow
blocking route family.

**Decision**: Implement or harden only the base computed-masked indexed
scatter-store executable artifact/ABI boundary, using the existing
computed-mask indexed memory provider fact surface and target validation
contract.

**Consequences**: This preserves Stage 2's typed RVV route authority while
avoiding a broad indexed-memory matrix. Runtime-scalar scatter and other
indexed/strided/segment2 expansions remain explicit continuation work.

## Technical Notes

Read before implementation:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/testing/index.md`
* `.trellis/spec/validation/index.md`
* `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-runtime-scalar-cmp-indexed-gather-load-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-runtime-scalar-cmp-indexed-gather-load-abi/implement.jsonl`
* `.trellis/tasks/archive/2026-06/06-07-stage2-rvv-runtime-scalar-cmp-indexed-gather-load-abi/check.jsonl`

Likely implementation and evidence files from the direction brief and repo
inventory:

* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
* `include/TianChenRV/Dialect/RVV/IR/RVVConfigContract.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/RVVComputedMaskMemorySelectedBodyRealizationOwner.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCComputedMaskMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCMemoryStatementPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCControlPolicyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
* `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-indexed-scatter-store-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-indexed-scatter-store-dry-run.test`
* `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-indexed-scatter-store-fail-closed.test`
* `test/Target/RVV/explicit-selected-body-artifact-computed-masked-indexed-scatter-store.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-indexed-scatter-store.mlir`

## Outcome

* Production seam inspection found the existing computed-mask indexed scatter
  route already flows through provider-owned indexed memory validation and the
  shared compare/select mask statement-plan validator.
* Added focused target artifact fail-closed evidence for stale scatter gather
  residue, stale ordinary unit-store residue, unit-stride destination pointer
  residue, and unscaled index-vector route payload residue.
* `ssh rvv` correctness passed for explicit and pre-realized selected-body
  computed-masked indexed scatter-store artifacts with runtime counts
  `0,1,16,17,257` and patterns `0,1`.
* Local generated-bundle dry-run passed for explicit and pre-realized
  selected-body scatter when `--llvm-readobj ''` skipped the missing local
  LLVM readobj tool. The host still lacks `llvm-readobj`, so local ELF
  header/symbol readobj checks were not covered in this session.
* No `.trellis/spec/` update was needed because the existing specs already
  describe the scatter indexed-store leaf, empty gather/unit-store residue,
  metadata mirror ordering, and required tests.
