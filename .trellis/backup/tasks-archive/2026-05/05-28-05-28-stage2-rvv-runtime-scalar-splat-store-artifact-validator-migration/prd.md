# Stage2 RVV runtime-scalar splat-store artifact validator migration

## Goal

Migrate runtime-scalar splat-store target artifact validation from
`RVVTargetSupportBundle.cpp` into the target-owned RVV route-family artifact
validator registry. The registry entry must own runtime-scalar splat-store
provider-fact checks and candidate metadata mirror checks, while the central
support bundle remains a neutral selected-route rebuild, bridge, and registry
dispatch layer.

## Direction Source

- Direction title: `Switch: Stage2 RVV runtime-scalar splat-store artifact
  validator migration`.
- Module owner: target-owned RVV route-family artifact validator registry entry
  for runtime-scalar splat-store.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `696d2e79 rvv: migrate standalone reduction artifact
  validation`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes direction brief before source edits.
- Serial worker constraint: no subagents, spawned agents, parallel agents, or
  multi-agent workflows.

## What I Already Know

- Commit `696d2e79` completed standalone reduction/accumulation artifact
  validator migration and left a clean worktree.
- `RVVTargetArtifactRouteFamilyValidation.cpp` currently owns migrated
  route-family validators for compare/select-mask, conversion dtype policy,
  segment2 memory, standalone reduction/accumulation, elementwise arithmetic,
  MAcc, and widening-dot reduction.
- `RVVTargetSupportBundle.cpp` still contains runtime-scalar splat-store
  semantic artifact validation helpers for provider headers, type mappings,
  ABI mappings, statement plan, provider payload facts, and candidate metadata
  dispatch.
- The support bundle already builds a
  `RVVTargetArtifactRouteFamilyValidationContext` and calls
  `validateRVVTargetArtifactRouteFamilyProviderFacts` /
  `validateRVVTargetArtifactRouteFamilyCandidateMirrors` for migrated families.
- Specs require target artifact export to rebuild the selected RVV provider
  route and dispatch route-family validators from the rebuilt provider
  description. Artifact metadata can only mirror provider facts after route
  construction.
- Runtime-scalar splat-store is an active Stage2 residual RVV route family. Its
  executable semantics must stay in typed `tcrv_rvv` body facts, RVV planning,
  RVV provider facts, `TCRVEmitCLowerableRoute`, and target-owned route-family
  validation, not common EmitC or artifact metadata.

## Requirements

1. Register a production runtime-scalar splat-store route-family validator in
   `RVVTargetArtifactRouteFamilyValidation.cpp`.
2. Dispatch active runtime-scalar splat-store target artifact candidates through
   the route-family validator registry using an explicit family id.
3. Move runtime-scalar splat-store provider-fact validation ownership out of
   `RVVTargetSupportBundle.cpp`.
4. Move runtime-scalar splat-store candidate mirror validation ownership out of
   `RVVTargetSupportBundle.cpp`.
5. Keep `RVVTargetSupportBundle.cpp` limited to neutral responsibilities:
   selected-body route rebuild, generic candidate shape checks, selected
   boundary/runtime checks, forbidden descriptor/direct-C/source-export residue
   rejection, neutral materialized route verification, artifact mechanics,
   metadata evidence listing, and registry dispatch.
6. The new validator must validate provider-derived facts for:
   - `RuntimeI32SplatStore` operation classification;
   - `RuntimeScalarSplatStore` memory form;
   - rebuilt route id agreement with the provider description;
   - provider support mirror and route-family plan facts;
   - route operand binding plan and binding summary;
   - required headers;
   - VL/vector C type mappings;
   - runtime ABI order and ABI parameter mapping for `rhs_scalar,out,n`;
   - runtime AVL/VL control plan and loop facts;
   - runtime scalar splat intrinsic and statement layout;
   - store intrinsic and store statement layout;
   - result name, SEW, LMUL, tail/mask policy, target leaf profile, and type
     summary facts;
   - selected typed RVV source provenance on pre-loop and loop statements;
   - stale non-splat-store route-family facts.
7. The new validator must validate candidate metadata only as mirrors of
   rebuilt provider facts, including:
   - `tcrv_rvv.provider_supported_mirror`;
   - `tcrv_rvv.runtime_scalar_splat_store_route_family_plan`;
   - `tcrv_rvv.route_operand_binding_plan`;
   - `tcrv_rvv.route_operand_binding_operands`;
   - `tcrv_rvv.memory_form`;
   - `tcrv_rvv.target_leaf_profile`;
   - `tcrv_rvv.runtime_control_plan`;
   - `tcrv_rvv.runtime_abi_order`;
   - `tcrv_rvv.required_header_declarations`;
   - `tcrv_rvv.c_type_mapping`.
8. The validator must fail closed on missing provider support mirror, stale
   route id, bad memory form, missing route-family plan, missing binding facts,
   bad ABI order, bad header/type mapping, missing AVL/VL loop facts, missing
   runtime scalar splat facts, missing store facts, stale candidate mirrors,
   stale elementwise/conversion/reduction/MAcc/segment2/compare/select memory
   facts, descriptor/source/script/artifact-name residue, exact-intrinsic-as-
   authority, and common/central semantic choice.
9. Preserve existing runtime-scalar splat-store generated C/runtime semantics
   and existing explicit/pre-realized artifact fixtures.
10. Preserve existing migrated validators and unrelated route-family behavior as
    non-regressions.
11. Do not introduce central ad hoc, name-derived, metadata-derived,
    descriptor-derived, ABI-string-derived, script-derived, artifact-name-
    derived, common-EmitC-derived, source-front-door-derived, route-id-derived,
    exact-intrinsic-derived, direct-route-entry-only,
    pre-realized-fixture-only, or legacy-i32-derived route authority.

## Acceptance Criteria

- [x] `RVVTargetArtifactRouteFamilyValidation.cpp` registers and dispatches a
      unique `runtime-scalar-splat-store` validator for active
      `RuntimeI32SplatStore` route descriptions.
- [x] `RVVTargetSupportBundle.cpp` no longer contains runtime-scalar
      splat-store semantic validator helpers or direct payload validation calls,
      except neutral registry dispatch/context references if unavoidable.
- [x] The new validator checks provider-derived route id agreement, provider
      support, runtime-scalar splat-store family plan, route operand binding
      facts, memory form, target leaf profile, headers, type mappings, ABI
      order/mapping, runtime AVL/VL plan, runtime scalar splat statement,
      store statement, result facts, policy/config facts, and source
      provenance.
- [x] The new validator checks candidate mirrors for provider support, route
      family plan, operand binding, memory form, target leaf profile, runtime
      control, runtime ABI order, headers, and C type mapping.
- [x] Negative or mutation coverage proves stale route id, missing provider
      support mirror, bad ABI order, bad header/type mapping, bad AVL/VL loop
      facts, missing runtime scalar splat facts, missing store facts, stale
      candidate mirrors, and stale non-splat-store facts fail closed.
- [x] Existing runtime-scalar splat-store artifact fixtures still pass without
      changing generated C/runtime semantics.
- [x] Existing migrated route-family validators remain covered and
      non-regressed.
- [x] `git diff --check` passes.
- [x] Focused target artifact export and RVV plugin tests pass.
- [x] Focused build targets for `TianChenRVRVVTarget`, `tcrv-opt`,
      `tcrv-translate`, `tianchenrv-target-artifact-export-test`, and
      `tianchenrv-rvv-extension-plugin-test` pass, or exact blocker is
      recorded.
- [x] `check-tianchenrv` passes, or exact blocker is recorded.
- [x] Bounded authority scan over `lib/Target/RVV` confirms no
      runtime-scalar splat-store semantic artifact validation owner remains in
      `RVVTargetSupportBundle.cpp` and no artifact claim depends on forbidden
      authority sources.
- [x] Trellis task status, journal, archive state, and commit state are
      truthful at the end of the round.

## Out Of Scope

- Adding new RVV operations or route coverage.
- Changing selected-body realization, provider planning, route construction,
  common EmitC materialization, generated-bundle behavior, or runtime behavior.
- New `ssh rvv` runtime/correctness/performance claims.
- Standalone reduction, segment2, compare/select, conversion, widening dot,
  MAcc, source-front-door, Scalar, IME, Offload, TensorExt, or frontend
  generalization work.
- Broad report-only work, broad audits, broad smoke matrices, dashboards, or
  helper-only changes.
- Compatibility wrappers that preserve support-bundle semantic authority.

## Technical Approach

1. Keep this Trellis task current and bounded to the supplied Hermes direction.
2. Inspect the runtime-scalar splat-store helpers in
   `RVVTargetSupportBundle.cpp` and migrated validator patterns in
   `RVVTargetArtifactRouteFamilyValidation.cpp`.
3. Move the runtime-scalar splat-store family classifier, provider-fact
   validators, statement-plan validator, and candidate mirror validator into
   `RVVTargetArtifactRouteFamilyValidation.cpp`.
4. Register a `runtime-scalar-splat-store` validator entry before generic or
   unrelated validators can accept it.
5. Remove support-bundle runtime-scalar splat-store semantic helper calls and
   rely on registry provider-fact and candidate-mirror validation.
6. Add or strengthen focused C++ mutation coverage in target artifact export
   tests for stale/missing runtime-scalar splat-store mirrors/facts.
7. Run focused build/lit/C++ checks, `git diff --check`, bounded authority
   scan, and `check-tianchenrv` if feasible.
8. Finish/archive the task and create one coherent commit if acceptance is met.

## Validation Plan

1. Validate/start this Trellis task after PRD and context files are written.
2. Build focused targets:
   `cmake --build build --target TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
3. Run focused C++ tests:
   `build/bin/tianchenrv-target-artifact-export-test` and
   `build/bin/tianchenrv-rvv-extension-plugin-test`.
4. Run focused lit tests from the configured build lit root:
   `cd build/test && /usr/bin/python3.10 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVV/explicit-selected-body-artifact-runtime-i32-splat-store.mlir Target/RVV/pre-realized-selected-body-artifact-runtime-i32-splat-store.mlir`.
5. Run `git diff --check`.
6. Run a bounded authority scan over `lib/Target/RVV`.
7. Run `cmake --build build --target check-tianchenrv -j2` when focused checks
   pass.

## Technical Notes

Required specs and prior context:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/core-dialect/tcrv-exec-contract.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- archived task
  `.trellis/tasks/archive/2026-05/05-28-05-28-stage2-rvv-standalone-reduction-accumulation-artifact-validator-migration/prd.md`
- archived runtime-scalar splat-store provider/runtime closure tasks under
  `.trellis/tasks/archive/2026-05/`

Primary production files:

- `include/TianChenRV/Target/RVV/RVVTargetArtifactRouteFamilyValidation.h`
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`

Focused tests:

- `test/Target/TargetArtifactExportTest.cpp`
- `test/Plugin/RVVExtensionPluginTest.cpp`
- `test/Target/RVV/explicit-selected-body-artifact-runtime-i32-splat-store.mlir`
- `test/Target/RVV/pre-realized-selected-body-artifact-runtime-i32-splat-store.mlir`

## Completion Evidence

- Production owner: `RVVTargetArtifactRouteFamilyValidation.cpp` now owns the
  `runtime-scalar-splat-store` registry entry, provider-fact validation, and
  candidate mirror validation.
- Support-bundle demotion: bounded scan found no runtime-scalar splat-store
  semantic validator helper or direct payload validation call in
  `RVVTargetSupportBundle.cpp`; it keeps neutral registry context/dispatch.
- Focused build passed:
  `cmake --build build --target TianChenRVRVVTarget tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j2`.
- Focused C++ tests passed:
  `build/bin/tianchenrv-target-artifact-export-test` and
  `build/bin/tianchenrv-rvv-extension-plugin-test`.
- Focused lit passed from `build/test`: 2/2 runtime-scalar splat-store artifact
  fixtures.
- `git diff --check` passed.
- `cmake --build build --target check-tianchenrv -j2` passed 456/456.
