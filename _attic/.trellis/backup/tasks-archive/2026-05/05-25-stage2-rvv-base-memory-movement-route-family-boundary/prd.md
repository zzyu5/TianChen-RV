# Stage2 RVV base memory-movement route-family boundary

## Goal

Close the RVV plugin-local route-family boundary for the already route-supported
base memory-movement forms. The base memory module must own legality,
operand-role validation, materialization facts, ordered statement planning, and
provider/artifact mirrors for strided, indexed, and static masked load/store
movement routes. The selected-body provider may only consume the validated
family-owned plan to build `TCRVEmitCLowerableRoute`; common EmitC, target
artifact export, scripts, tests, route ids, ABI strings, and metadata remain
neutral evidence or mirrors.

## Direction Source

- Direction title: `Stage2 RVV base memory-movement route-family boundary`.
- Module owner: RVV plugin-local base memory-movement route-family module and
  registry ownership for already route-supported low-level memory movement.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `0ac30a56 rvv: reuse standalone reduction route family boundary`.
- `.trellis/.current-task` was absent, so this task was created from the
  provided Direction Brief before source edits.

## Current Repository Facts

- The required authority chain is selected `tcrv.exec` envelope -> typed or
  realized `tcrv_rvv` body -> RVV plugin-owned legality/realization/family
  planning/provider -> `TCRVEmitCLowerableRoute` -> common EmitC -> target
  artifact -> `ssh rvv` evidence when executable behavior is claimed.
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h` already defines
  `RVVSelectedBodyBaseMemoryMovementRouteFamilyPlan`,
  `RVVSelectedBodyBaseMemoryMovementRouteStatementPlan`, the memory
  route-family owner registry hooks, and the migrated statement-plan aggregate
  family tag `BaseMemoryMovement`.
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` already derives a base
  memory family plan for route-supported base memory operations, validates
  route/type/runtime/mirror facts, builds a base memory statement plan, and
  routes it through `getRVVSelectedBodyMigratedRouteStatementPlan(...)`.
- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` already calls
  `verifyRVVSelectedBodyRouteFamilyProviderPlans(...)`, materialization facts,
  memory operand-binding facts, and the migrated statement-plan aggregate before
  falling back to older central statement assembly.
- `test/Plugin/RVVExtensionPluginTest.cpp` already has substantial C++ coverage
  for base memory owner classification, missing/stale family plan diagnostics,
  strided/indexed/static-masked plan derivation, operand binding, materialization
  facts, and provider statement-plan consumption.
- Existing target FileCheck fixtures assert base memory route-family mirrors for
  explicit selected-body memory movement, while some pre-realized memory
  fixtures and generated-bundle dry-run tests do not yet assert the same
  family/mirror/statement-plan evidence as strongly as the latest standalone
  reduction and scalar-broadcast MAcc closures.
- `scripts/rvv_generated_bundle_abi_e2e.py` already knows expected base memory
  plan IDs, target leaf profiles, provider-supported mirrors, and C type mapping
  metadata for base memory forms, but generated evidence does not yet expose a
  dedicated base-memory route-family boundary summary with ordered statement
  plan facts.

## Scope

1. Treat this as a closure task over the existing base memory module, not as a
   new parallel memory planning path.
2. Keep base memory ownership limited to already route-supported forms:
   `strided_load_unit_store`, `unit_load_strided_store`,
   `indexed_gather_unit_store`, `indexed_scatter_unit_load`,
   `masked_unit_load_store`, and `masked_unit_store`.
3. If current provider code already consumes the family-owned statement plan,
   preserve that path and strengthen evidence, tests, or diagnostics around it
   rather than reimplementing the same route.
4. Generated-bundle evidence should explicitly expose the base memory family
   boundary, source/destination/index/mask/stride roles, route metadata mirrors,
   ordered pre-loop and loop callees, and the fact that runtime counts and
   harness constants are execution cases only.
5. At least one executable migrated base memory fixture must have real `ssh rvv`
   correctness evidence across multiple runtime counts when runtime correctness
   is claimed.

## Requirements

1. Base memory family ownership must classify only the six in-scope operations.
   Computed-mask memory and segment2 memory remain under their own families.
2. The family plan must own memory form, pointer/value/index/mask/stride roles,
   runtime AVL/VL plan, SEW/LMUL/policy, type/header/intrinsic mapping,
   source/destination memory-form facts, layout facts, target leaf profile,
   provider-supported mirror, and route operand-binding summary.
3. Provider construction must require a validated base memory family plan,
   materialization facts, and memory operand-binding facts before attaching the
   ordered statement plan to `TCRVEmitCLowerableRoute`.
4. Missing family plans, stale family plans on non-consumers, stale
   name/metadata-derived mirrors, invalid pointer/value/index/mask/stride roles,
   missing runtime `n`, invalid SEW/LMUL/policy/type facts, and missing
   statement-plan dependencies must fail closed before common EmitC or target
   artifact authority.
5. Common EmitC and target artifact code may consume provider route payloads and
   mirror explicit fields, but must not infer memory semantics, dtype, ABI
   order, intrinsic spelling, or route support from metadata, route ids,
   artifact names, ABI strings, descriptors, source-front-door markers, or
   harness constants.
6. Pre-realized and explicit selected-body base memory evidence must remain
   consistent with the same base memory family plan and mirror labels.
7. No positive legacy `RVVI32M1*`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`,
   `!tcrv_rvv.i32m*`, descriptor, direct-C/source-export, or source-front-door
   authority may be introduced.

## Acceptance Criteria

- [x] Task context files reference the RVV plugin, EmitC route, testing
      contract, previous scalar-broadcast MAcc boundary, previous standalone
      reduction boundary reuse, and relevant current code surfaces.
- [x] Current base memory provider/family code is inventoried against HEAD
      before implementation, including what is already closed and what evidence
      remains missing.
- [x] Focused C++ provider tests prove base memory owner isolation, family-plan
      validation, materialization facts, memory operand-binding facts, ordered
      statement-plan construction, migrated aggregate consumption, and provider
      route attachment for representative strided, indexed, and static masked
      forms.
- [x] Focused fail-closed tests cover missing base memory plan, stale base plan
      on non-consumer, stale route-description mirror, stale route operand
      binding mirror, invalid source/destination/index/mask/stride roles, stale
      typed config or intrinsic facts, and missing statement-plan dependency.
- [x] FileCheck target fixtures for explicit and pre-realized base memory forms
      assert `tcrv_rvv.base_memory_movement_route_family_plan`, explicit
      `provider_supported_mirror`, route operand-binding mirrors, memory-form
      facts, and generated header mirrors where the provider route supports the
      form.
- [x] Generated-bundle dry-run evidence for at least one explicit and one
      pre-realized base memory fixture records a dedicated base memory boundary
      summary, including ordered pre-loop/loop callees and mirror-only route
      metadata.
- [x] Real `ssh rvv` correctness succeeds for at least one migrated executable
      base memory fixture across counts `7,16,23`; this round's executable
      fixture is `strided_load_unit_store` with stride bytes `4,8,12`.
- [x] Bounded authority scans over touched RVV planning/provider/target/script/
      fixture/test files find no new central ad hoc memory authority,
      metadata/name-derived authority, descriptor/source-front-door authority,
      common-EmitC semantic ownership, harness-derived authority, or legacy i32
      route authority.
- [x] Focused checks, `git diff --check`, and `check-tianchenrv` pass, or an
      exact blocker is recorded.
- [x] Task status, journal, archive, clean final git status, and one coherent
      commit complete if this task finishes.

## Completion Evidence

This round confirmed that the production RVV provider/family path already owns
the base memory family-plan and migrated statement-plan boundary for the six
already route-supported forms at the C++ route-planning/provider level. The
bounded implementation closed the missing evidence surface for the explicit and
pre-realized `strided_load_unit_store` executable fixture:

- `scripts/rvv_generated_bundle_abi_e2e.py` now emits a
  `base_memory_movement_boundary` evidence object for base memory operations,
  derived from structural bundle metadata and the expected provider-owned
  statement plan rather than route names, artifacts, ABI strings, descriptors,
  or harness constants.
- The evidence object records typed-body authority, mirror-only route metadata,
  ordered pre-loop/loop callees, selected source ABI, pointer roles, emitted
  artifact paths, runtime counts, and an explicit note that runtime counts are
  execution cases, not memory-route authority.
- Explicit and pre-realized generated-bundle dry-run tests now check the
  dedicated boundary summary, family label, ordered callees, and
  `provider_supported_mirror`.
- The pre-realized strided target artifact FileCheck fixture now checks the base
  memory family-plan mirror, target leaf profile, provider-supported mirror,
  required header declarations, and C type mapping mirrors.
- `.trellis/spec/extension-plugins/rvv-plugin.md` now captures the durable
  generated-bundle and `ssh rvv` evidence requirements for executable base
  memory movement routes.

Executed checks:

- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- Explicit `strided_load_unit_store` generated-bundle dry-run with counts
  `7,16,23` and stride bytes `4,8,12`.
- Pre-realized `strided_load_unit_store` generated-bundle dry-run with counts
  `7,16,23` and stride bytes `4,8,12`.
- Focused generated-bundle lit tests for explicit and pre-realized strided
  base memory dry-runs.
- Focused target FileCheck/lit tests for explicit and pre-realized strided
  selected-body artifacts.
- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2`
- `ssh -o BatchMode=yes -o ConnectTimeout=8 rvv true`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- Real `ssh rvv` generated-bundle correctness for `strided_load_unit_store`
  across counts `7,16,23` and stride bytes `4,8,12`.
- `git diff --check`
- `cmake --build build --target check-tianchenrv -j2`

No performance claim is made. Optional future extension, if wanted, is to
replicate the same generated-bundle boundary-summary assertions across the
indexed and static-masked generated-bundle fixtures; this is not required for
the bounded executable subfamily closed in this task because the C++ provider
tests already cover those base-memory ownership and fail-closed paths.

## Non-Goals

- No new RVV operation families, new memory semantics, dtype/LMUL batches,
  high-level Linalg/Vector/StableHLO frontend lowering, source-front-door
  positive routes, global autotuning, dashboards, broad smoke matrices,
  compatibility wrappers for legacy i32 authority, or one-op-per-intrinsic
  wrapper growth.
- No migration of memory semantics into common EmitC, target artifact plumbing,
  `tcrv.exec`, route ids, metadata fields, manifests, artifact names, ABI
  strings, descriptors, scripts, tests, or harness constants.
- No performance claim.
- No subagents or parallel-agent workflow.

## Validation Plan

1. Validate and start the Trellis task.
2. Run focused C++ coverage for RVV extension plugin route-family/provider
   behavior after changes.
3. Run focused FileCheck/lit tests for selected explicit and pre-realized base
   memory target artifacts and generated-bundle dry-runs.
4. Run generated-bundle dry-run for the selected base memory fixture(s), with
   multiple runtime counts and stride/index/mask cases as applicable.
5. Run real `ssh rvv` generated-bundle evidence for at least one migrated base
   memory fixture if claiming runtime correctness.
6. Run bounded authority scans over touched source/test/script/evidence paths.
7. Run `git diff --check`.
8. Run `cmake --build build --target check-tianchenrv -j2`, or record the exact
   blocker.

## Technical Notes

- Specs read: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Shared guide read: `.trellis/spec/guides/index.md`.
- Prior task context read:
  `.trellis/tasks/archive/2026-05/05-25-stage2-rvv-route-family-boundary/prd.md`
  and
  `.trellis/tasks/archive/2026-05/05-25-05-25-stage2-rvv-standalone-reduction-route-family-boundary-reuse/prd.md`.
- Initial code/test surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVSelectedBodyRealization.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/RVV/*memory*`,
  `test/Target/RVV/*strided*`,
  `test/Target/RVV/*indexed*`,
  `test/Conversion/EmitC/*memory*`,
  and `test/Scripts/*memory*` / `test/Scripts/*strided*` /
  `test/Scripts/*indexed*`.
- Initial gap hypothesis: production provider/family ownership is largely
  present at HEAD; this round should close mirror/evidence/test visibility for
  base memory movement to the same standard as the latest route-family boundary
  tasks and repair any production-path gaps discovered by focused checks.
