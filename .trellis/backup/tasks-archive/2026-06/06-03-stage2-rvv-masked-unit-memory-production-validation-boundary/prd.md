# Stage2 RVV masked unit-stride memory production validation boundary

## Goal

Make the production provider-to-target validation boundary for existing RVV
unit-stride masked memory routes fail closed on provider-owned memory, mask,
passthrough, runtime-scalar, ABI, header/type, intrinsic, target-profile, and
mirror facts instead of accepting stale plain-memory/base-memory,
computed-mask-only, route-id, artifact-name, descriptor, C string, or
metadata-only evidence.

The bounded route set is:

* `masked_unit_load_store`
* `masked_unit_store`
* `computed_masked_unit_load_store`
* `runtime_scalar_cmp_masked_store`
* `runtime_scalar_cmp_masked_load_store`

## Direction Source

Hermes Direction Brief:

`Stage2 RVV masked unit-stride memory production validation boundary`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` returned no dirty file list through RTK.
* Initial `git log --oneline -8` started at
  `2cd40f98 rvv: validate elementwise broadcast binding facts`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md` keeps the Stage2 authority chain as selected typed
  `tcrv_rvv` body -> RVV plugin-owned provider facts ->
  `TCRVEmitCLowerableRoute` -> common EmitC -> target artifact validation.
* `.trellis/spec/extension-plugins/rvv-plugin.md` puts mask/tail behavior,
  memory form, runtime ABI use, C/RVV type mapping, intrinsic mapping, and
  fail-closed diagnostics under the RVV plugin/provider.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines provider-owned route
  fact surfaces and requires target artifact validation to consume provider
  facts rather than duplicate route-family truth.
* The previous completed task archived under
  `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-elementwise-broadcast-arithmetic-production-validation/`
  completed the adjacent elementwise/broadcast provider-to-target validation
  boundary and did not cover masked unit-stride memory.
* Live inspection found `masked_unit_load_store` and `masked_unit_store`
  currently participate in `RVVBaseMemoryMovementRouteFacts` and the base
  memory target-artifact positive path, even though they carry mask,
  passthrough, inactive-lane, masked load/store, and mask type facts.
* Live inspection found `computed_masked_unit_load_store`,
  `runtime_scalar_cmp_masked_store`, and
  `runtime_scalar_cmp_masked_load_store` already participate in the
  computed-mask memory operand-binding path, but this task needs the target
  artifact validation owner to consume operation-specific unit-stride masked
  memory facts and reject stale strided/indexed/segment2/compare-only residue.

## Requirements

* Keep common EmitC/export neutral. Common and target code may consume
  provider-built route facts and mirrors, but must not infer RVV semantics from
  route ids, artifact names, descriptors, C strings, test names, or candidate
  metadata.
* Provide or complete a provider-owned canonical facts surface for the in-scope
  unit-stride masked memory routes, including operation kind, memory form,
  load/store/update leaf roles, source/destination/passthrough roles, mask
  source, mask-tail policy owner, inactive-lane behavior, runtime ABI order and
  parameter roles, SEW/LMUL/tail/mask policy, vector/mask/scalar C types,
  setvl/load/store/mask/compare intrinsic leaves, operand binding summary with
  header participation, required headers, target leaf profile,
  `provider_supported_mirror`, and route-family mirror labels.
* Runtime-scalar variants must keep threshold/RHS-scalar role and runtime-scalar
  mask producer facts separate from vector compare mask facts.
* Static-mask `masked_unit_load_store` and `masked_unit_store` must not pass as
  stale plain/base memory when mask, passthrough, masked intrinsic, mask type,
  inactive-lane, ABI, target-profile, provider mirror, or route-family mirror
  facts are missing or wrong.
* Computed-mask unit-stride load/store and runtime-scalar masked store/load-store
  validation must reject strided, indexed, segment2, elementwise, compare/select
  only, runtime-scalar-vs-vector-mask, store-vs-load-store, passthrough,
  ABI-order, intrinsic, header/type, target-profile, provider mirror, and
  route-family mirror residue.
* Do not redo indexed, strided, segment2, elementwise/broadcast arithmetic,
  MAcc, widening MAcc, reductions, conversions, or compare/select owners.

## Acceptance Criteria

* [x] RVV provider/planning code exposes or completes operation-specific
      provider-owned facts for all five in-scope unit-stride masked memory
      routes.
* [x] Provider plan validation fails closed when the in-scope routes carry
      stale base/plain memory facts, wrong mask producer source, wrong
      passthrough/inactive-lane contract, wrong runtime-scalar threshold role,
      wrong ABI order/roles, stale header/type/intrinsic facts, stale target
      leaf profile, stale provider mirror, or wrong route-family mirror.
* [x] Target artifact validation consumes the provider-owned masked unit-stride
      memory facts and rejects stale or operation-mismatched provider facts and
      candidate mirrors before bundle acceptance.
* [x] Focused C++ target-artifact tests directly mutate representative provider
      facts and candidate mirrors for static-mask, computed-mask, and
      runtime-scalar in-scope routes.
* [x] Focused lit/script filters for existing explicit and pre-realized
      `masked_unit_load_store`, `masked_unit_store`,
      `computed_masked_unit_load_store`, `runtime_scalar_cmp_masked_store`, and
      `runtime_scalar_cmp_masked_load_store` forms still pass.
* [x] Generated-bundle dry-run and direct fail-closed script checks pass for
      script-supported in-scope forms.
* [x] No route emission, generated runtime behavior, runtime ABI order,
      mask/passthrough behavior, load/store semantics, tail behavior, or
      destination preservation semantics are changed. If any of those behaviors
      change, run `ssh rvv` correctness for counts `0,1,16,17,257` with data
      patterns distinguishing store-only vs load-store, vector mask vs
      runtime-scalar mask, inactive lanes, tail preservation, and destination
      preservation.
* [x] Bounded old-authority scan over touched files finds no new positive
      dependency on legacy `i32m1`, descriptor, source-front-door, route-id,
      artifact-name, exact intrinsic spelling, or mirror-only authority.

## Out Of Scope

* Indexed, strided, segment2, elementwise/broadcast arithmetic, MAcc, widening
  MAcc, widening dot-reduce, standalone reductions, conversions, compare/select
  expansion, source-front-door routes, and high-level frontend lowering.
* New dtype/LMUL clone batches, global tuning, dashboards, broad smoke
  matrices, or evidence-only packaging.
* Moving RVV semantics into common EmitC/export or target artifact metadata.

## Technical Approach

* Follow the provider-owned facts pattern from the existing computed-mask
  strided/indexed memory and previous elementwise/broadcast validation tasks.
* Split or specialize static unit-stride masked memory facts out of the base
  memory validation surface so `masked_unit_*` routes are accepted for
  mask-specific facts rather than plain/base-memory facts.
* Reuse the existing computed-mask memory family surface for
  `computed_masked_unit_load_store`, `runtime_scalar_cmp_masked_store`, and
  `runtime_scalar_cmp_masked_load_store` where it is already the correct owner,
  but make target artifact validation enforce the unit-stride masked-memory
  fact subset explicitly.
* Keep all operation-specific constants in RVV provider/planning code and make
  target validation consume accessors rather than define duplicate target-local
  truth.
* Add focused C++ mutations around stale provider descriptions and candidate
  mirrors; update lit/script expectations only where provider-owned fact labels
  or mirrors become more precise.

## Definition of Done

* [x] Focused implementation and tests are ready for one coherent commit.
* [x] `tianchenrv-target-artifact-export-test` builds and passes.
* [x] `tcrv-opt` and `tcrv-translate` build if route fixtures/scripts are
      touched.
* [x] Focused lit/script filters for in-scope route fixtures pass.
* [x] Generated-bundle dry-runs and direct fail-closed script checks pass for
      script-supported in-scope forms.
* [x] `rtk git diff --check` passes.
* [x] Task status and journal record the completed module behavior and
      evidence.

## Completion Evidence

* Added `RVVUnitStrideMaskedMemoryRouteFacts` plus provider accessors for the
  five in-scope route kinds and wired route operand binding summaries with
  header participation for static-mask, computed-mask, and runtime-scalar
  unit-stride masked memory.
* Target artifact validation now consumes the provider-owned masked memory
  facts and rejects stale provider/candidate residue for memory form,
  operation kind, mask source, passthrough/inactive-lane policy,
  runtime-scalar RHS/splat facts, ABI order and parameter roles, header/type
  summaries, intrinsic leaves, target profile, provider mirror, and
  route-family mirrors.
* Focused C++ fail-closed tests mutate representative static-mask,
  computed-mask, and runtime-scalar provider descriptions and candidate mirrors.
* Focused lit/script filter passed from `build/test` for the existing
  explicit/pre-realized masked unit-stride memory forms, including
  generated-bundle dry-runs and direct fail-closed script checks.
* No `ssh rvv` rerun was required: this round tightened validation metadata,
  mirrors, and header-participation checks without changing route emission,
  generated C runtime behavior, runtime ABI order, mask/passthrough behavior,
  load/store semantics, tail behavior, or destination preservation semantics.
* Diff-only old-authority scan found only fail-closed expected-fragment strings
  for legacy intrinsic spellings in C++ negative tests, not positive route
  authority.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant archived task read:

* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-elementwise-broadcast-arithmetic-production-validation/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-elementwise-broadcast-arithmetic-production-validation/implement.jsonl`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-elementwise-broadcast-arithmetic-production-validation/check.jsonl`

Relevant live files inspected before PRD:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Target/TargetArtifactExportTest.cpp`
* `test/Target/RVV/*masked-unit*.mlir`
* `test/Target/RVV/*runtime-scalar-cmp-masked*load-store*.mlir`
* `test/Target/RVV/*runtime-scalar-cmp-masked*store*.mlir`
* `test/Scripts/*masked-unit*.test`
* `test/Scripts/*runtime-scalar-cmp-masked*memory*.test`
* `test/Scripts/*runtime-scalar-cmp-masked*load-store*.test`
