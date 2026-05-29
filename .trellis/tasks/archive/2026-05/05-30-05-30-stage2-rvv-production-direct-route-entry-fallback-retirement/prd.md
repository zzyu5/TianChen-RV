# Stage2 RVV production direct route-entry fallback retirement

## Goal

Retire the remaining production RVV direct pre-realized route-entry selected-body
fallback from route construction. Executable RVV artifacts must require the
public selected lowering-boundary materialization path before provider route
facts are collected. A pre-realized selected RVV body that reaches emission-plan
or EmitC route construction without an explicit `setvl/with_vl` selected
boundary must fail closed with a targeted diagnostic.

## Direction Source

- Direction title: `Redirect: Stage2 RVV production direct route-entry fallback
  retirement`.
- Module owner: RVV plugin production route boundary and selected-body
  realization owner registry.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `154cab04 rvv: close runtime scalar cmp masked load-store artifact`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint from the brief: one Codex worker; no subagents,
  spawned agents, parallel agents, or multi-agent workflows.

## Current Repository Facts

- The previous archived task
  `.trellis/tasks/archive/2026-05/05-30-05-30-stage2-rvv-runtime-scalar-cmp-masked-load-store-executable-artifact`
  closed focused selected-boundary generated-bundle evidence for
  `runtime_scalar_cmp_masked_load_store`, direct CLI fail-closed coverage,
  `ssh rvv` correctness, and `check-tianchenrv` 464/464.
- `scripts/rvv_generated_bundle_abi_e2e.py` now reports
  `supports_direct_pre_realized_route_entry` as `False` for all selected
  pre-realized expectations and rejects `--direct-pre-realized-route-entry`
  before bundle generation.
- Production RVV plugin route construction still contains a fallback:
  `getOrRealizeRVVSelectedBodyRouteBoundary()` calls
  `realizePreRealizedRVVRouteEntrySelectedBody()` when no explicit
  `setvl/with_vl` boundary exists.
- `RVVSelectedBodyRealization.cpp` still carries active route-entry predicates
  and owner registry entries for elementwise/compare-select, contraction, and
  base memory movement.
- There is a positive direct route-entry lit test for pre-realized elementwise
  route materialization. That test encodes the shortcut that this task retires
  and must be rewritten as fail-closed or moved to the selected-boundary route.
- The long-term authority chain remains:

```text
selected tcrv.exec RVV variant
  -> explicit selected lowering-boundary materialization
  -> realized typed tcrv_rvv body
  -> RVV provider route facts
  -> TCRVEmitCLowerableRoute
  -> neutral common EmitC
  -> target artifact
  -> ssh rvv evidence when runtime/correctness/performance is claimed
```

## Requirements

1. Production emission-plan and EmitC route construction must not auto-realize a
   pre-realized selected RVV body through a direct route-entry fallback.
2. If a pre-realized selected RVV body reaches production route construction
   without the public selected lowering-boundary materialization pass, the RVV
   plugin must fail closed before provider facts are collected.
3. The fail-closed diagnostic must mention that pre-realized RVV selected bodies
   must use public selected lowering-boundary materialization before provider
   route construction. It must not present route-entry predicates, route ids,
   op names, descriptors, metadata, script options, artifact names, exact
   intrinsic spellings, source-front-door markers, or legacy i32 residue as
   route authority.
4. The explicit selected-boundary materialization path must continue to work for
   representative families, including:
   - `runtime_scalar_cmp_masked_load_store`;
   - at least one arithmetic family;
   - at least one memory family;
   - at least one contraction family.
5. Existing script-level direct pre-realized route-entry CLI behavior must remain
   fail-closed for representative families.
6. Route-entry predicates, registries, helper declarations, and tests must be
   removed, demoted, or made unambiguously diagnostic/negative-test inventory
   only. They must not imply executable route support.
7. Common EmitC/export must remain neutral and must not become a semantic
   realization fallback.
8. No new RVV op families, dtype/LMUL clone batches, frontend/Linalg work,
   one-intrinsic wrappers, broad dashboards, or artifact-evidence-only closure
   are in scope.

## Acceptance Criteria

- [x] PRD, implement context, and check context accurately describe the bounded
      production owner change, specs, non-goals, and validation plan.
- [x] `RVVExtensionPlugin.cpp` no longer calls the direct route-entry selected
      body realization path from production emission-plan or EmitC route
      construction.
- [x] A pre-realized RVV selected body without an explicit selected boundary
      fails closed before provider route facts are collected.
- [x] Active route-entry owner predicates/registries are cleaned up or proven
      diagnostic-only; no realization owner remains route-entry capable for
      production route construction.
- [x] The former positive direct pre-realized route-entry lit coverage is
      rewritten as a focused fail-closed diagnostic test.
- [x] Selected-boundary dry-run remains green for
      `runtime_scalar_cmp_masked_load_store` and representative arithmetic,
      memory, and contraction families.
- [x] At least one `ssh rvv` generated-bundle correctness run still passes
      through selected-boundary materialization.
- [x] Focused C++ or lit coverage proves direct pre-realized bodies reaching
      route construction without selected-boundary materialization fail closed.
- [x] Bounded authority scan confirms no route or executable claim depends on
      route-entry-only, pre-realized-fixture-only, name-derived,
      metadata-derived, descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      exact-intrinsic-derived, or legacy-i32-derived authority.
- [x] `git diff --check` passes.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded.
- [x] If complete, the task is finished, archived, journaled, and committed
      coherently.

## Technical Approach

Replace the production route-boundary helper with a boundary-only requirement:
return an existing unique realized `tcrv_rvv.with_vl` boundary, reject mixed
realized/pre-realized bodies, and fail closed when the variant still contains a
pre-realized selected body. Keep `materializeSelectedLoweringBoundary()` as the
only path that consumes pre-realized RVV selected-body fixtures into realized
typed `tcrv_rvv` structure.

Demote route-entry realization inventory by removing active owner predicates
from the realization owner registry and making the direct route-entry helper
diagnostic-only. Update tests so direct route-entry is negative coverage while
selected-boundary tests continue to prove executable paths.

## Decision (ADR-lite)

Context: The script-level generated-bundle interface already rejects direct
pre-realized route entry, but production RVV plugin code still had an
auto-realization fallback before provider route construction. That mismatch let
route construction depend on a shortcut even when artifact evidence was
selected-boundary-only.

Decision: Retire production direct route-entry fallback and make selected
lowering-boundary materialization the required precondition before provider route
construction for pre-realized RVV selected bodies.

Consequences: Positive direct route-entry tests are removed or inverted. Existing
pre-realized executable families remain supported only through the selected
lowering-boundary producer. Future direct route-entry support would require a new
explicit owner task, specs, facts, diagnostics, generated-bundle evidence, and
real RVV evidence.

## Out of Scope

- Do not add new Stage2 RVV coverage families.
- Do not expand dtype/LMUL matrices.
- Do not start high-level frontend/Linalg lowering.
- Do not move RVV semantics into common EmitC/export.
- Do not re-open direct pre-realized route-entry support for any completed
  family.
- Do not treat artifact metadata, route ids, descriptors, ABI strings, scripts,
  exact intrinsic spellings, source-front-door markers, or legacy i32 names as
  route authority.

## Validation Plan

1. Focused C++ test for selected-body owner registry and direct route-entry
   fail-closed behavior.
2. Focused lit test for `--tcrv-materialize-emitc-lowerable-routes` rejecting a
   pre-realized elementwise body that did not pass through selected-boundary
   materialization.
3. Focused selected-boundary generated-bundle dry-run for:
   - `runtime_scalar_cmp_masked_load_store`;
   - representative arithmetic;
   - representative memory;
   - representative contraction.
4. Focused direct pre-realized CLI fail-closed generated-bundle tests for
   representative families.
5. One selected-boundary `ssh rvv` generated-bundle correctness run.
6. Bounded authority scan over RVV plugin route-boundary/realization code,
   scripts, tests, common EmitC/export, source-front-door, descriptor, route-id,
   exact-intrinsic, and legacy-i32 authority surfaces.
7. `git diff --check`.
8. `cmake --build build --target check-tianchenrv -j2`, or exact blocker.

## Spec Update Judgment

Required and completed. The RVV plugin spec now states that direct route-entry
selected-body realization is retired for active production. Remaining
route-entry inventory is diagnostic/negative-test-only unless a future explicit
owner task reintroduces it with full facts, specs, generated-bundle evidence,
and `ssh rvv` evidence.

## Completion Evidence

- Production route-boundary change:
  `RVVExtensionPlugin.cpp` now requires an existing selected lowering boundary
  before emission-plan or EmitC route construction and emits
  `pre-realized RVV selected body must use public selected lowering-boundary
  materialization before provider route construction` for pre-realized bodies
  that skipped the public materialization pass.
- Route-entry realization demotion:
  `RVVSelectedBodyRealization.cpp` owner registry entries no longer install
  `isRouteEntryConsumer` predicates, and
  `realizePreRealizedRVVRouteEntrySelectedBody()` is diagnostic-only with
  `direct pre-realized RVV route-entry realization is retired`.
- Script alignment:
  `scripts/rvv_generated_bundle_abi_e2e.py` keeps
  `supports_direct_pre_realized_route_entry == False`, fail-closes
  `--direct-pre-realized-route-entry`, and records pre-realized executable
  evidence through `tcrv-materialize-selected-lowering-boundaries` with
  `route_entry_realization: False`.
- Focused checks passed:
  `tianchenrv-rvv-extension-plugin-test`, focused lit direct fail-closed and
  selected-boundary tests, 30 direct pre-realized generated-bundle fail-closed
  lit tests, script `py_compile` and `--self-test`, representative selected
  boundary dry-run for `runtime_scalar_cmp_masked_load_store`,
  `scalar_broadcast_add`, `strided_load_unit_store`, and
  `widening_macc_add`.
- Real RVV evidence:
  `ssh rvv` generated-bundle correctness passed for
  `runtime_scalar_cmp_masked_load_store` through selected-boundary
  materialization with counts `0,1,16,23,257` and RHS scalars `-37,91`.
- Full quality gate:
  `cmake --build build --target check-tianchenrv -j2` passed 464/464 tests, and
  `git diff --check` passed.
