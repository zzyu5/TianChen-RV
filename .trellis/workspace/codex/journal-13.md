# Journal - codex (Part 13)

> Continuation from `journal-12.md` (archived at ~2000 lines)
> Started: 2026-05-22

---



## Session 155: Stage2 RVV closure-gated strided masked load movement

**Date**: 2026-05-22
**Task**: Stage2 RVV closure-gated strided masked load movement
**Branch**: `main`

### Summary

Added closure-gated computed-mask byte-strided masked-load to unit-store RVV route support with typed tcrv_rvv.masked_strided_load body facts, RouteOperandBindingPlan materialization, explicit/pre-realized fixtures, generated-bundle dry-runs, and real ssh rvv evidence.

### Main Changes

- Added `tcrv_rvv.masked_strided_load` and the bounded
  `tcrv_rvv.typed_computed_mask_strided_load_pre_realized_body` surface for
  computed-mask byte-strided loads to unit-stride destination stores.
- Rewired RVV selected-body realization, route planning, route provider, route
  operand binding closure, construction metadata, target artifact fixtures, and
  generated-bundle ABI/runtime harness support for
  `computed_masked_strided_load_unit_store`.
- Added explicit and pre-realized route fixtures plus fail-closed negative
  coverage for missing/wrong typed body facts, stale or wrong binding plans,
  mirror-only authority, materialized-use mismatch, old masked_move fallback,
  route-id/helper fallback, source/front-door/descriptor/direct-C authority,
  and common/export semantic inference.

### Git Commits

Final round commit is created after task archive in the same Codex turn.

### Testing

- [OK] Focused RVV build targets for `tcrv-opt`, `tcrv-translate`, RVV plugin,
  construction protocol, and target artifact export tests.
- [OK] RVV dialect, RVV plugin, construction protocol, and target artifact
  export binaries.
- [OK] Positive explicit and pre-realized computed masked strided-load target
  lit fixtures plus bounded negative fixture.
- [OK] `rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Generated-bundle dry-runs for explicit and pre-realized
  `computed_masked_strided_load_unit_store`, counts `7,16,23`, source byte
  strides `4,8,12`.
- [OK] Real `ssh rvv` generated-bundle runs for explicit and pre-realized
  `computed_masked_strided_load_unit_store`, counts `7,16,23`, source byte
  strides `4,8,12`, proving active strided loads, inactive passthrough,
  untouched source gaps, runtime n/AVL behavior, and tail/sentinel
  preservation.
- [OK] Focused conversion lit self-repair after diagnostic surface growth.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 300/300 passed.
- [OK] `git diff --check`.
- [OK] Diff-level authority scan found no new positive legacy/source/descriptor
  route authority; the exact intrinsic diff hit is the provider-owned target
  leaf selected after typed closure, not route authority.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 168: Stage2 RVV contraction explicit strided/masked selected-body realization

**Date**: 2026-05-23
**Task**: Stage2 RVV contraction explicit strided masked selected-body realization
**Branch**: `main`

### Summary

Enabled the remaining explicit selected-body contraction routes for strided
input and computed-mask widening dot-reduce, proved they reach the validated
RVV contraction family plan and generated bundle boundary, and validated them
with real `ssh rvv` execution.

### Main Changes

- Added explicit selected-body `tcrv.exec` RVV fixtures for
  `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`.
- Extended `scripts/rvv_generated_bundle_abi_e2e.py` expectations so the three
  explicit routes use the new fixtures and verify generated-bundle evidence.
- Added focused generated-bundle dry-run lit tests for the three explicit
  routes, including plan metadata, binding closure, stride/mask facts, harness
  behavior, and correctness markers.

### Routes Covered

- `strided_input_widening_dot_reduce_add`
- `computed_masked_widening_dot_reduce_add`
- `computed_masked_strided_input_widening_dot_reduce_add`

### Testing

- [OK] Focused PLAN/HEADER FileCheck for the three new target fixtures.
- [OK] Focused lit filter for the three target fixtures plus three script
  fixtures: 6/6 selected tests passed.
- [OK] Explicit generated-bundle dry-run for the three new routes with counts
  `7,16,23`.
- [OK] Base explicit preservation dry-run for `widening_macc_add` and
  `widening_dot_reduce_add` with counts `7,16,23`.
- [OK] Pre-realized preservation dry-run for all five active contraction routes
  with counts `7,16,23`.
- [OK] Real `ssh rvv` explicit run for all three new routes with counts
  `7,16,23`, including stride `2,3`, inactive-lane, accumulator, scalar output,
  and tail-preservation checks.
- [OK] Focused build:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Active-authority scan over touched paths, `git diff --check`, and
  `cmake --build build --target check-tianchenrv -j2` passed 361/361.

### Self-Repair

- Shortened explicit fixture kernel/variant names after bundle export rejected
  overlong C/EmitC function identifiers.
- Reordered two FileCheck assertions to match the actual diagnostic metadata
  order while keeping the stride/mask facts checked.

### Status

[OK] **Completed and archived**

### Next Steps

- None - task complete


## Session 168: Stage2 RVV contraction explicit selected-body realization

**Date**: 2026-05-23
**Task**: Stage2 RVV contraction explicit selected-body realization
**Branch**: `main`

### Summary

Enabled explicit selected-body generated-bundle and real `ssh rvv` evidence for
the base RVV contraction routes `widening_macc_add` and
`widening_dot_reduce_add`, reusing the existing plugin-owned generic typed body
planning/provider path and validated contraction family plan.

### Main Changes

- Added explicit generic `tcrv_rvv` body target fixtures for
  `widening_macc_add` and `widening_dot_reduce_add`.
- Added explicit generated-bundle expectations for both routes in
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Required contraction generated-bundle metadata verification to include
  `tcrv_rvv.contraction_route_family_plan =
  rvv-contraction-route-family-plan.v1`.
- Added focused script dry-run tests for both explicit contraction routes.
- Left strided-input and computed-mask contraction explicit selected bodies as
  the exact continuation point; pre-realized preservation for all five active
  contraction routes was verified.

### Testing

- [OK] Focused PLAN/HEADER FileCheck for both new explicit target fixtures.
- [OK] Explicit generated-bundle dry-runs for `widening_macc_add` and
  `widening_dot_reduce_add`, counts `7,16,23`.
- [OK] Pre-realized generated-bundle preservation dry-run for all five active
  contraction op kinds, counts `7,16,23`.
- [OK] Real `ssh rvv` explicit generated-bundle run for `widening_macc_add`
  and `widening_dot_reduce_add`, counts `7,16,23`.
- [OK] Focused build:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Active-authority scan and `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2` passed 355/355.

### Status

[OK] **Completed**

### Next Steps

- Continue within the same family by enabling explicit selected-body fixtures
  and evidence for strided-input and computed-mask widening dot-reduce variants.


## Session 168: Stage2 RVV contraction and widening dot-reduce route-family ownership

**Date**: 2026-05-23
**Task**: Stage2 RVV contraction and widening dot-reduce route-family ownership
**Branch**: `main`

### Summary

Moved active widening MAcc and widening dot-reduce routes behind a validated RVV
plugin-owned contraction family plan and proved target/header mirrors,
generated-bundle evidence, ssh rvv runtime evidence, authority scan, and
check-tianchenrv.

### Main Changes

- Added `rvv-contraction-route-family-plan.v1` as the validated family plan
  mirror for contraction routes.
- Added planning-owned contraction route-family consumer predicates and
  provider-plan verification.
- Made provider materialization require the validated contraction plan before
  building `TCRVEmitCLowerableRoute`.
- Strengthened plan validation for operation/memory-form classification,
  runtime ABI order, target leaf/header/type facts, strided-input facts,
  computed-mask facts, accumulator/result contracts, and route operand binding
  closure.
- Added target/header/generated-bundle checks for
  `tcrv_rvv.contraction_route_family_plan`.

### Routes Covered

- `widening_macc_add`
- `widening_dot_reduce_add`
- `strided_input_widening_dot_reduce_add`
- `computed_masked_widening_dot_reduce_add`
- `computed_masked_strided_input_widening_dot_reduce_add`

### Testing

- [OK] Focused build: `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Manual REALIZED/PLAN/HEADER FileCheck for all five pre-realized target fixtures.
- [OK] Pre-realized generated-bundle dry-runs for all five active op kinds, counts `7,16,23`.
- [OK] Real `ssh rvv` for `widening_macc_add`, `widening_dot_reduce_add`, and `computed_masked_strided_input_widening_dot_reduce_add`, counts `7,16,23`.
- [OK] Added-line active-authority scan and full touched-file scan review.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2` passed 351/351.

### Self-Repair

- Trellis task slug was created with a duplicated date prefix by the task tool;
  left the directory stable and kept task metadata truthful.
- Explicit-selected-body dry-run was tried and rejected by the script as
  unsupported for contraction op kinds; recorded the current active surface as
  pre-realized selected-body.

### Spec Update Judgment

- No `.trellis/spec/**` update needed. This round applied existing RVV plugin
  ownership and EmitC-neutrality rules rather than introducing a new rule.

### Status

[OK] **Completed; pending archive and commit**

### Next Steps

- Archive the Trellis task and create one coherent commit.


## Session 168: Stage2 RVV computed-mask MAcc accumulation route-family ownership

**Date**: 2026-05-23
**Task**: Stage2 RVV computed-mask MAcc accumulation route-family ownership
**Branch**: `main`

### Summary

Moved active computed-mask MAcc provider materialization behind a
planning-owned computed-mask accumulation family verifier and proved focused
lit, generated-bundle dry-run, real `ssh rvv`, authority scan, and
`check-tianchenrv` evidence.

### Main Changes

- Added planning-owned computed-mask MAcc accumulation consumer predicates and
  provider-plan verification.
- Made RVV EmitC provider require the validated computed-mask accumulation
  family plan before materializing `computed_masked_macc_add` and
  `runtime_scalar_cmp_masked_macc_add`.
- Replaced the provider-local computed-mask MAcc consumer predicate in the
  materialization path with the planning-owned MAcc accumulation consumer
  predicate.
- Preserved plain `macc_add` as a separate route outside computed-mask
  accumulation family metadata.
- Tightened focused FileCheck/generated-bundle evidence for vector compare and
  runtime-scalar mask producer distinction, accumulator/result contracts,
  inactive-lane passthrough, route operand binding closure, provider mirrors,
  and header metadata.

### Routes Covered

- `computed_masked_macc_add`
- `runtime_scalar_cmp_masked_macc_add`

### Testing

- [OK] Focused build:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused lit filter for computed-mask MAcc and adjacent MAcc fixtures:
  15/15 selected tests passed.
- [OK] Generated-bundle dry-runs for explicit and pre-realized
  `computed_masked_macc_add` and `runtime_scalar_cmp_masked_macc_add`, counts
  `7,16,23`, with runtime scalar thresholds `-37,91`.
- [OK] Real `ssh rvv` explicit run for both active routes, counts `7,16,23`,
  runtime scalar thresholds `-37,91`.
- [OK] Real `ssh rvv` pre-realized run for both active routes, counts
  `7,16,23`, runtime scalar thresholds `-37,91`.
- [OK] Added-line active-authority scan; full touched-file scan only found
  negative FileCheck guards and existing rejection code.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2` passed 351/351.

### Self-Repair

- Corrected the new runtime-scalar generated-bundle harness FileCheck ordering
  by switching independent harness checks to `HARNESS-DAG`.

### Status

[OK] **Completed and archived**

### Next Steps

- Commit this coherent round.


## Session 163: Stage2 RVV computed-mask strided-load route family

**Date**: 2026-05-23
**Task**: Stage2 RVV computed-mask strided-load route-family interface
**Branch**: `main`

### Summary

Migrated the active `computed_masked_strided_load_unit_store` route onto the
shared RVV plugin-owned computed-mask memory route-family interface. The route
now carries source byte-stride, masked strided-load leaf, strided layout,
provider mirror/header/type facts, and binding closure through the family plan
instead of relying on strided-load-specific post-analysis fallback fields.

### Main Changes

- Added `ComputedMaskStridedLoadUnitStore` to
  `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan` membership.
- Added source-stride and strided-load metadata to the family plan, including
  `runtime_abi:src_stride_bytes`, computed-mask strided-load memory layout,
  target leaf profile, provider-supported mirror, headers, and C type mapping.
- Made family validation require `tcrv_rvv.masked_strided_load`, old output
  passthrough, final unit store, runtime `n/AVL`, and source byte-stride role.
- Made the provider require the shared computed-mask memory family before
  materializing the strided-load route.
- Updated explicit/pre-realized target/header tests and generated-bundle
  metadata expectations for family-derived strided-load facts.

### Testing

- [OK] Focused build:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- [OK] Focused explicit/pre-realized FileCheck runs for computed-mask
  strided-load target/header artifacts and selected-body realization.
- [OK] Negative FileCheck for bad source stride binding.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test`.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
  script `--self-test`.
- [OK] Generated-bundle dry-runs for explicit and pre-realized
  `computed_masked_strided_load_unit_store`, counts `7,16,23`, stride bytes
  `8,12`.
- [OK] Real `ssh rvv` evidence for explicit and pre-realized
  `computed_masked_strided_load_unit_store`, counts `7,16,23`, stride bytes
  `8,12`, proving active lanes, inactive passthrough, skipped source slots,
  tail preservation, and runtime `n` variation.
- [OK] Active-authority scan, added-line forbidden-authority scan,
  `git diff --check`, and `cmake --build build --target check-tianchenrv -j2`
  passed 349/349.

### Spec Update

- No `.trellis/spec/**` update needed. This task applied existing RVV
  plugin-owned route-family and EmitC neutrality rules; it did not add a new
  durable rule.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 163: Stage2 RVV computed-mask memory producer-source family

**Date**: 2026-05-23
**Task**: Stage2 RVV computed-mask memory producer-source family migration
**Branch**: `main`

### Summary

Migrated the active vector/non-runtime computed-mask memory consumers
`computed_masked_unit_load_store` and `computed_masked_strided_store` onto the
shared RVV plugin-local computed-mask memory route-family plan while preserving
the already migrated `runtime_scalar_cmp_masked_store` and
`runtime_scalar_cmp_masked_load_store` routes.

### Main Changes

- Generalized the runtime-scalar-only computed-mask memory plan into
  `RVVSelectedBodyComputedMaskMemoryRouteFamilyPlan` with explicit
  `maskProducerSource`, runtime-scalar/vector producer facets, store-only versus
  load-merge/store facets, target leaf/header/type metadata, and binding closure.
- Made planning derive the shared plan for four scoped routes:
  `runtime_scalar_cmp_masked_store`,
  `runtime_scalar_cmp_masked_load_store`,
  `computed_masked_unit_load_store`, and
  `computed_masked_strided_store`.
- Made provider materialization require the shared plan for those four routes
  and consume plan-owned setvl/load/splat/compare/store/header facts while
  keeping common EmitC/export neutral.
- Added computed-mask memory family mirror metadata to target bundle evidence,
  generated-bundle expectations, and focused artifact FileCheck assertions.
- Left strided-load, indexed, segmented, select, accumulation, reduction,
  frontend, and source-front-door paths out of scope.

### Testing

- [OK] Focused build:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] Focused lit filter for migrated computed-mask memory and preserved
  runtime-scalar memory target artifacts: 5/5 selected tests passed.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Generated-bundle dry-run for pre-realized vector
  `computed_masked_strided_store` and `computed_masked_unit_load_store` with
  counts `7,16,23`.
- [OK] Generated-bundle dry-run for explicit runtime-scalar
  `runtime_scalar_cmp_masked_store` and
  `runtime_scalar_cmp_masked_load_store` with counts `7,16,23` and thresholds
  `-37,91`.
- [OK] Real `ssh rvv` generated-bundle run for pre-realized vector store and
  load-store with counts `7,16,23` and destination strides `4,8,12`.
- [OK] Real `ssh rvv` generated-bundle run for explicit runtime-scalar store
  and load-store with counts `7,16,23` and thresholds `-37,91`.
- [OK] Active-authority scan, `git diff --check`, and
  `cmake --build build --target check-tianchenrv -j2` passed 349/349.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 163: Stage2 RVV computed-mask accumulation producer-source family

**Date**: 2026-05-23
**Task**: Stage2 RVV computed-mask accumulation producer-source family migration
**Branch**: `main`

### Summary

Generalized the shared computed-mask accumulation route-family plan from
runtime-scalar-only to producer-source-aware support and migrated the existing
vector/vector `computed_masked_macc_add` and
`computed_mask_standalone_reduce_add` production routes to consume it.

### Main Changes

- Replaced the runtime-scalar-specific accumulation plan surface with
  `RVVSelectedBodyComputedMaskAccumulationRouteFamilyPlan`, carrying explicit
  vector-compare versus runtime-scalar mask producer facts.
- Kept runtime-scalar macc and standalone reduction on the same shared family
  through the runtime-scalar producer source while preserving their runtime ABI
  order and materialized operand closure.
- Migrated vector/vector computed-mask macc and standalone reduction to derive
  target leaf/header/mirror fields, suffix contracts, and provider compare
  producer binding from the shared family.
- Added `accumulation_mask_producer_source` to target/header metadata and
  generated-bundle expectations; common EmitC/export remains a neutral mirror
  path.
- Added generated-bundle self-test negatives for stale producer-source and
  wrong accumulation suffix metadata.

### Testing

- [OK] Focused build:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused lit filter for computed-mask macc/standalone and runtime-scalar
  accumulation routes: 17/17 selected tests passed.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Generated-bundle dry-runs for explicit and pre-realized
  `computed_masked_macc_add`, `computed_mask_standalone_reduce_add`,
  `runtime_scalar_cmp_masked_macc_add`, and
  `runtime_scalar_cmp_masked_standalone_reduce_add`, counts `7,16,23` and
  thresholds `-37,91` for runtime-scalar cases.
- [OK] Real `ssh rvv` generated-bundle runs for explicit and pre-realized
  `computed_masked_macc_add` and `computed_mask_standalone_reduce_add`, counts
  `7,16,23`, proving active/inactive mask behavior, accumulator passthrough,
  scalar carry, and tail preservation.
- [OK] Diff-level active-authority scan found no new positive
  `RVVI32M1`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, descriptor,
  source-front-door, helper-string, artifact-name, or mirror-only route
  authority; the only exact intrinsic diff hit is the existing typed provider
  runtime-scalar splat leaf validation.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2` passed 349/349.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 161: Stage2 RVV computed-mask select route-family interface

**Date**: 2026-05-23
**Task**: Stage2 RVV computed-mask select route-family interface
**Branch**: `main`

### Summary

Introduced a shared RVV plugin-local runtime-scalar computed-mask select
route-family plan and migrated the active `runtime_scalar_cmp_select` and
`runtime_scalar_dual_cmp_mask_and_select` production planning/provider paths to
consume it.

### Main Changes

- Replaced the two independent runtime-scalar select route-family plan optionals
  with `RVVSelectedBodyRuntimeScalarComputedMaskSelectRouteFamilyPlan`.
- Consolidated route-family derive/validate/apply logic for runtime ABI order,
  target leaf/header fields, c type mapping, mask role/source/form, select
  layout, and optional dual compare/mask-and facts.
- Consolidated RouteOperandBindingPlan construction and provider closure checks
  for the shared true/false/out/n roles while preserving dual-specific
  `cmp_lhs_b`, `rhs_scalar_b`, mask-and provenance, and output header mirror
  requirements.
- Left common EmitC/export neutral and did not migrate unrelated vector-select,
  memory, macc, reduction, or future route families.

### Git Commits

| Hash | Message |
|------|---------|
| `pending` | `rvv: share computed-mask select route family` |

### Testing

- [OK] Focused build targets:
  `tcrv-opt`, `tcrv-translate`, `tianchenrv-rvv-extension-plugin-test`,
  `tianchenrv-construction-protocol-common-test`,
  `tianchenrv-target-artifact-export-test`.
- [OK] Focused lit filter from `build/test`: 6/6 selected tests passed.
- [OK] Generated-bundle dry-runs for explicit/pre-realized
  `runtime_scalar_cmp_select` and
  `runtime_scalar_dual_cmp_mask_and_select`.
- [OK] Real `ssh rvv` evidence for explicit dual, pre-realized dual, and one
  explicit single-mask regression route.
- [OK] Continuation retry revalidated live dry-runs
  `20260523-cmsel-single-explicit`, `20260523-cmsel-single-pre`,
  `20260523-cmsel-dual-explicit`, and `20260523-cmsel-dual-pre`.
- [OK] Continuation retry revalidated live `ssh rvv`
  `20260523-cmsel-dual-explicit-ssh`,
  `20260523-cmsel-dual-pre-ssh`, and
  `20260523-cmsel-single-explicit-ssh`.
- [OK] Active-authority scan, `git diff --check`, and `check-tianchenrv`
  349/349 passed.
- [OK] Spec-update review found no `.trellis/spec/**` change was needed; the
  existing RVV plugin, unified EmitC route, and MLIR testing contracts already
  cover this boundary.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 159: Stage2 RVV runtime scalar computed-mask masked store

**Date**: 2026-05-22
**Task**: Stage2 RVV closure-gated runtime scalar computed-mask store boundary
**Branch**: `main`

### Summary

Added a closure-gated `runtime_scalar_cmp_masked_store` RVV route that carries lhs vector payload, runtime RHS scalar threshold, compare-produced mask, source payload load, output masked store side effect, runtime n/AVL, and typed SEW/LMUL/policy facts through the RVV dialect, selected-body realization, route planning/provider, target artifact export, generated bundle harness, and real ssh rvv evidence.

### Main Changes

- Added `tcrv_rvv.typed_runtime_scalar_computed_mask_store_pre_realized_body` and verifier rules for `op_kind = "runtime_scalar_cmp_masked_store"`, predicate `sle`, SEW32/LMUL m1, undisturbed tail/mask policy, compare-produced mask facts, source payload role, output role, and runtime element-count role.
- Realized the pre-realized body into `setvl`, `with_vl`, lhs load, RHS scalar `splat`, source payload load, `compare`, and `masked_store` with false-lane output preservation.
- Added RVV planning/provider support for `RVVSelectedBodyOperationKind::RuntimeScalarComputedMaskStore`, `RuntimeScalarComputedMaskStore` memory form, runtime ABI order `lhs,rhs_scalar,src,dst,n`, compact exported `RouteOperandBindingPlan` mirror, target leaf/profile mirrors, header/C type mirrors, and fail-closed route validation.
- Materialized the provider-owned EmitC/RVV route as lhs load, scalar threshold splat, source load, compare, and masked store; common EmitC/export remains mechanical and does not infer semantics.
- Added generated-bundle explicit and pre-realized runtime harness support with value-distinguishing lhs/source/destination patterns, positive and negative RHS thresholds, runtime count variation, active lane writes, false-lane preservation, source preservation, and tail/sentinel preservation.
- Added explicit/pre-realized target fixtures and dialect verifier negative coverage.
- Self-repair performed: compacted exported binding metadata under the 512-byte artifact limit; shortened the explicit fixture function name under EmitC identifier bounds; fixed runtime-scalar masked-store analysis so the provider bridge sees the real masked-store route; restored computed-mask strided/unit-load-store target leaf behavior after the new masked-memory branch made compare/leaf selection too broad; updated construction-protocol common test expectations for the new route.
- Checks passed: manual dialect FileCheck; explicit PLAN/HEADER; pre-realized REALIZED/PLAN/HEADER; `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`; generated-bundle dry-runs for explicit and pre-realized `runtime_scalar_cmp_masked_store`; real `ssh rvv` explicit and pre-realized PASS for counts `7,16,23` and `rhs_scalar=-37,91`; focused recovery checks for prior computed-mask strided/unit-load-store dry-runs; `git diff --check`; `ninja -C build check-tianchenrv` with `337/337` tests.
- Authority scan found no new legacy RVVI32M1/rvv-i32m1/finite `tcrv_rvv.i32_*` positive route authority; hits are PRD negative-boundary wording and typed-provider target leaf intrinsic validation after route closure.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | `rvv: add runtime scalar computed mask store` |

### Testing

- [OK] Manual FileCheck, generated-bundle dry-runs, real ssh rvv evidence, active-authority scan, `git diff --check`, and `check-tianchenrv 337/337`.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 159: Stage2 RVV closure-gated runtime scalar-vector add boundary

**Date**: 2026-05-22
**Task**: Stage2 RVV closure-gated runtime scalar-vector add boundary
**Branch**: `main`

### Summary

Completed the bounded scalar-vector add/store boundary by adding the missing
explicit selected-body `scalar_broadcast_add` artifact/runtime consumer and
tightening RouteOperandBindingPlan header-mirror closure for scalar-broadcast
operands.

### Main Changes

- Added explicit selected-body target/header fixture for `scalar_broadcast_add`
  carrying `lhs`, runtime `rhs_scalar`, `out`, runtime `n`/AVL, SEW32, LMUL m1,
  agnostic policy, `splat`, `binary {kind = "add"}`, and `store`.
- Repaired scalar-broadcast elementwise binding summaries so `lhs`,
  `rhs_scalar`, `out`, and `n` all carry header mirrors, and provider
  construction requires the scalar-broadcast header mirror uses before route
  construction.
- Added C++ plugin checks for the scalar-broadcast binding summary and
  fail-closed RHS scalar header mirror lookup.
- Added explicit generated-bundle ABI expectation and dry-run test for
  `scalar_broadcast_add`; updated pre-realized scalar-broadcast add evidence
  checks for the closure-gated binding summary.
- Shared scalar-broadcast sub/mul FileCheck summaries were updated only to
  match the common route-family binding summary; no new sub/mul route coverage
  was added.

### Git Commits

Final round commit is created after task archive in the same Codex turn.

### Testing

- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- [OK] `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`
- [OK] Focused explicit/pre-realized scalar-broadcast add/sub/mul target and header FileCheck coverage.
- [OK] Focused scalar-broadcast conversion and negative checks.
- [OK] Explicit and pre-realized generated-bundle dry-runs for counts `7,16,23` and `rhs_scalar=-37,91`.
- [OK] Explicit and pre-realized real `ssh rvv` PASS for counts `7,16,23` and `rhs_scalar=-37,91`.
- [OK] Added-line authority scan found no new positive legacy/source/descriptor/common-export route authority.
- [OK] `git diff --check`
- [OK] `cmake --build build --target check-tianchenrv -j2` with `331/331` tests passing.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 158: Stage2 RVV closure-gated typed widening conversion boundary

**Date**: 2026-05-22
**Task**: Stage2 RVV closure-gated typed widening conversion boundary
**Branch**: `main`

### Summary

Closed the explicit selected-body evidence boundary for the existing
closure-gated `widen_i32_to_i64` signed widening conversion route. Current HEAD
already had plugin-owned conversion planning/provider closure, so this round
did not duplicate route logic.

### Main Changes

- Created the Trellis task and PRD from the Hermes brief because no current
  task existed.
- Added `test/Target/RVV/explicit-selected-body-artifact-widen-i32-to-i64.mlir`
  to prove explicit typed `tcrv_rvv` route authority:
  `runtime_abi_value -> setvl -> load<i32,m1> -> widening_convert ->
  store<i64,m2>`.
- Added explicit generated-bundle support for `widen_i32_to_i64` in
  `scripts/rvv_generated_bundle_abi_e2e.py`.
- Added
  `test/Scripts/rvv-generated-bundle-abi-e2e-widen-i32-to-i64-dry-run.test`
  to verify explicit evidence JSON and harness output include route binding
  plan, binding summary, source/destination config, signed relation, and
  explicit selected-body front-door mirrors.
- Retained `widen_i16_to_i32` as existing active conversion support and did not
  add a conversion matrix, unsigned/narrowing variants, source-front-door
  route, or dtype/LMUL clone batch.

### Checks

- [OK] Explicit PLAN and HEADER FileCheck for
  `explicit-selected-body-artifact-widen-i32-to-i64.mlir`.
- [OK] Existing dialect/verifier conversion FileCheck.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused lit: explicit target/script tests 2/2 passed.
- [OK] Focused lit: pre-realized target/script regression tests 2/2 passed.
- [OK] Explicit and pre-realized generated-bundle dry-runs for counts
  `7,16,23`.
- [OK] Real `ssh rvv` explicit and pre-realized `widen_i32_to_i64` runs PASS
  for counts `7,16,23`.
- [OK] Diff-level active-authority scan found no new positive
  legacy/source/descriptor/common-export route authority.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 324/324 passed.

### Spec Update Review

No `.trellis/spec/**` change was needed. Existing RVV plugin, EmitC route, and
MLIR testing specs already encode the long-term boundary exercised here.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 159: Stage2 RVV closure-gated masked horizontal reduce-sum accumulation

**Date**: 2026-05-22
**Task**: Stage2 RVV closure-gated masked horizontal reduce-sum accumulation
**Branch**: `main`

### Summary

Resolved a stale Hermes direction brief against current HEAD: the requested
masked horizontal reduce-sum accumulation production route already exists as
`computed_mask_standalone_reduce_add` from `d57f5c6c rvv: add closure-gated
computed-mask reduction`. This round did not duplicate the production route.
It added post-macc fail-closed dialect regressions proving elementwise masked
add and masked macc cannot claim the standalone reduction route, then refreshed
explicit/pre-realized generated-bundle and real `ssh rvv` evidence.

### Main Changes

- Created the Trellis task and PRD documenting current-head inventory, stale
  brief resolution, non-goals, and completion evidence.
- Added
  `computed_mask_standalone_reduce_rejects_elementwise_add_fallback_claim` to
  reject `tcrv_rvv.masked_binary` route-id authority for the standalone
  reduction route.
- Added `computed_mask_standalone_reduce_rejects_masked_macc_fallback_claim`
  to reject `tcrv_rvv.masked_macc` carrying the standalone reduction result
  layout.
- Left production RVV dialect/config/construction/realization/planning/provider
  code unchanged because the requested production add route is already live and
  focused validation passed.

### Testing

- [OK] `build/bin/tcrv-opt test/Dialect/RVV/computed-mask-standalone-reduction-dataflow.mlir --split-input-file --verify-diagnostics`.
- [OK] Explicit selected-body PLAN and HEADER FileCheck for
  `computed_mask_standalone_reduce_add`.
- [OK] Pre-realized selected-body REALIZED, PLAN, and HEADER FileCheck for
  `computed_mask_standalone_reduce_add`.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Generated-bundle dry-runs for explicit and pre-realized
  `computed_mask_standalone_reduce_add`, counts `7,16,23`.
- [OK] Real `ssh rvv` explicit and pre-realized generated-bundle runs, counts
  `7,16,23`, seeds `-11,17`, active-lane counts `5,10,14`, inactive-lane
  counts `2,6,9`, scalar outputs `1,-33,-26,29,-5,2`, tail preserved.
- [OK] Active-authority scan found only the intended negative `route_id`,
  `masked_binary`, and `masked_macc` fail-closed cases in the touched RVV test
  diff.
- [OK] `git diff --check`.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 322/322 passed.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 158: Stage2 RVV closure-gated two-field segmented masked store movement

**Date**: 2026-05-22
**Task**: Stage2 RVV closure-gated two-field segmented masked store movement
**Branch**: `main`

### Summary

Added one bounded closure-gated `computed_masked_segment2_store_unit_load`
route family on the corrected typed RVV surface. The new path carries
compare-produced mask facts, field0/field1 payload sources, interleaved
destination memory, segment count 2, inactive-lane no-write policy, runtime
n/AVL, SEW/LMUL/policy, materialized operands, header mirrors, and provider
owned target leaf through RVV dialect/config, selected-body realization,
construction protocol, route planning/provider, target artifact fixtures, and
generated-bundle runtime evidence.

### Main Changes

- Added `tcrv_rvv.masked_segment2_store` and
  `tcrv_rvv.typed_computed_mask_segment2_store_pre_realized_body`.
- Added RVV config/runtime ABI support for order
  `cmp_lhs,cmp_rhs,src0,src1,dst,n`.
- Added selected-body realization from pre-realized facts into explicit
  `setvl`, typed loads, `tcrv_rvv.compare`, and
  `tcrv_rvv.masked_segment2_store`.
- Added closure-gated route planning/provider materialization with
  `rvv-route-operand-binding:computed_masked_segment2_store_unit_load.v1` and
  masked segmented store target leaf
  `__riscv_vsseg2e32_v_i32m1x2_m`.
- Added explicit/pre-realized target fixtures and generated-bundle ABI/runtime
  support for value-distinguishing interleaved masked stores.
- Routes intentionally not converted: segment3/4, broad segmented matrices,
  indexed/strided/contiguous masked movement, segment2 masked load redo,
  reductions, compare/select expansion, dtype/LMUL clone batches, frontend
  lowering, source-front-door routes, and future plugin work.

### Testing

- [OK] Focused dialect and target FileCheck coverage for explicit and
  pre-realized computed-mask segment2 masked store.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`.
- [OK] `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Explicit and pre-realized generated-bundle dry-runs for
  `computed_masked_segment2_store_unit_load`, counts `7,16,23`.
- [OK] Real `ssh rvv` explicit and pre-realized generated-bundle runs, counts
  `7,16,23`, proving active segmented masked stores, false-lane no-write
  preservation, field-distinguishing payload order, unrelated source
  preservation, runtime n/AVL variation, and tail/sentinel preservation.
- [OK] Regression dry-runs for explicit and pre-realized
  `computed_masked_segment2_load_unit_store`, counts `7,16,23`.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 316/316 passed.
- [OK] `git diff --check`.
- [OK] Active-authority scan found no new positive legacy i32/source-front-door
  / descriptor / direct-C / common-export route authority. New exact intrinsic
  hits are provider-owned target leaf mirrors after typed closure.

### Self-Repair

- Repaired target header metadata ordering for the new route.
- Preserved computed segment2 masked-load source/destination metadata after
  de-duplicating generic masked-memory metadata.
- Added the new store runtime ABI branch and role step count to construction
  protocol tests.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 158: Stage2 RVV closure-gated two-field segmented masked load movement

**Date**: 2026-05-22
**Task**: Stage2 RVV closure-gated two-field segmented masked load movement
**Branch**: `main`

### Summary

Added closure-gated two-field computed-mask segmented load RVV route support with explicit/pre-realized fixtures, generated-bundle dry-runs, and real ssh rvv evidence.

### Main Changes

- Added the bounded `computed_masked_segment2_load_unit_store` route family on typed RVV body facts, including `tcrv_rvv.masked_segment2_load` and `tcrv_rvv.typed_computed_mask_segment2_load_pre_realized_body`.
- Rewired RVV config ABI ordering, dialect verification, selected-body realization, construction protocol metadata, route planning, route provider materialization, closure-gated operand binding, target artifact mirrors, and generated-bundle ABI/runtime harness support for two-field segmented masked load movement.
- The route carries compare lhs/rhs, interleaved source memory, per-field old passthrough buffers, per-field outputs, compare-produced mask, runtime n/AVL, SEW/LMUL/policy, segment count 2, inactive passthrough policy, and tail/source preservation through the RVV dialect/plugin-owned path.
- Added explicit and pre-realized target fixtures plus dialect dataflow/negative coverage and generated-bundle dry-run tests; updated focused conversion/plugin allowlists for the new generic RVV op surface.
- Self-repair performed: changed field extract materialization steps to load role, updated construction protocol common-test expectations for typed op/role sequence/runtime ABI, repaired stale Stage2 generic-op diagnostic allowlists, and added target artifact helper enum handling.
- Checks passed: manual FileCheck for dialect verifier, explicit PLAN/HEADER, pre-realized REALIZED/PLAN/HEADER, and script dry-run fixtures; `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`; generated-bundle self-test; explicit and pre-realized dry-runs for counts `7,16,23`; real `ssh rvv` explicit and pre-realized PASS for counts `7,16,23`; focused lit filter 21/21; `cmake --build build --target check-tianchenrv -j2` with `313/313` tests; `git diff --check`.
- Runtime evidence proved active segmented field loads, field-distinguishing interleaved source data, mixed true/false mask behavior, per-field inactive-lane passthrough preservation, source preservation, tail/sentinel preservation, and runtime n/AVL variation.
- Authority scan found no new positive legacy/source/descriptor/common-export route authority; descriptor/source-front-door/direct-C diff hits are negative-boundary wording in `RVVOps.td`, and target intrinsic spelling remains provider-owned after typed closure.
- Spec update review: no `.trellis/spec/**` change was needed because existing RVV plugin, EmitC route, and MLIR testing contracts already define the long-term boundary; this round added bounded route coverage inside that boundary.
- Segment counts beyond two fields, segmented stores, indexed/strided fallback, dtype/LMUL clone batches, source-front-door routes, and high-level frontend work were intentionally not converted in this bounded owner.

Final round commit is created after task archive in the same Codex turn.


### Git Commits

(Commit pending in this session)

### Testing

- [OK] Focused RVV dialect/target/script fixtures for explicit and pre-realized
  computed-mask segment2 load routes.
- [OK] Generated-bundle dry-runs and real `ssh rvv` runs for explicit and
  pre-realized `computed_masked_segment2_load_unit_store`, counts `7,16,23`.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 313/313 passed.
- [OK] `git diff --check`.
- [OK] Diff-level authority scan found no new positive legacy/source/descriptor
  or common-export route authority.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 156: Stage2 RVV closure-gated indexed masked gather-load movement

**Date**: 2026-05-22
**Task**: Stage2 RVV closure-gated indexed masked gather-load movement
**Branch**: `main`

### Summary

Added typed closure-gated computed-mask indexed gather-load RVV route support with explicit/pre-realized fixtures, generated-bundle dry-runs, and real ssh rvv evidence.

### Main Changes

- Added the typed `tcrv_rvv.masked_indexed_load` and bounded pre-realized computed-mask indexed gather body surface for signed i32 / SEW32 / LMUL m1 gather-load to unit-store behavior.
- Rewired selected-body realization, config ABI ordering, construction protocol metadata, RVV route planning, route provider materialization, binding-plan closure, target artifact mirrors, and generated-bundle ABI/runtime harness support for `computed_masked_indexed_gather_load_unit_store`.
- Added explicit and pre-realized route fixtures, dialect dataflow coverage, generated-bundle dry-run tests, and regression updates for generic RVV allowlist and fail-closed diagnostic surfaces.
- Checks passed: focused dialect/target lit fixtures, script py_compile, generated-bundle self-test, generated-bundle dry-runs for counts `7,16,23`, real `ssh rvv` explicit and pre-realized runs for counts `7,16,23`, focused self-repair reruns, `git diff --check`, and `cmake --build build --target check-tianchenrv -j2` with `305/305` tests.
- Spec update review: no `.trellis/spec/**` change was needed because the existing RVV plugin, EmitC route, and MLIR testing contracts already define the long-term boundary; this round added bounded route coverage inside that boundary.
- Authority scan: no new positive legacy i32/source-front-door/descriptor/direct-C/common-export route authority was introduced. The exact intrinsic occurrence is provider-owned target leaf spelling after typed closure.


### Git Commits

Final round commit is created after task archive in the same Codex turn.

### Testing

- [OK] Focused RVV dialect/target lit fixtures for explicit and pre-realized
  computed-mask indexed gather-load routes.
- [OK] Script validation: `python3 -m py_compile
  scripts/rvv_generated_bundle_abi_e2e.py` and
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Generated-bundle dry-runs for explicit and pre-realized
  `computed_masked_indexed_gather_load_unit_store`, counts `7,16,23`.
- [OK] Real `ssh rvv` generated-bundle runs for explicit and pre-realized
  `computed_masked_indexed_gather_load_unit_store`, counts `7,16,23`,
  proving active indexed loads, inactive passthrough, noncontiguous/permuted
  index behavior, runtime n/AVL handling, source preservation, and
  tail/sentinel preservation.
- [OK] Focused self-repair reruns for FileCheck metadata ordering and route
  allowlist diagnostic growth.
- [OK] `cmake --build build --target check-tianchenrv -j2` - 305/305 passed.
- [OK] `git diff --check`.
- [OK] Diff-level authority scan found no new positive legacy/source/descriptor
  or common-export route authority; the exact intrinsic diff hit is the
  provider-owned target leaf selected after typed closure.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 157: Stage2 RVV closure-gated indexed masked scatter-store movement

**Date**: 2026-05-22
**Task**: Stage2 RVV closure-gated indexed masked scatter-store movement
**Branch**: `main`

### Summary

Added closure-gated computed-mask indexed scatter-store RVV route support with explicit/pre-realized fixtures, generated-bundle dry-runs, and real ssh rvv evidence.

### Main Changes

- Added the bounded `computed_masked_indexed_scatter_store_unit_load` route family on typed RVV body facts, including `tcrv_rvv.masked_indexed_store` and `tcrv_rvv.typed_computed_mask_indexed_scatter_pre_realized_body`.
- Rewired RVV config ABI ordering, selected-body realization, construction protocol metadata, route planning, route provider materialization, closure-gated operand binding, target artifact mirrors, and generated-bundle ABI/runtime harness support for indexed masked scatter-store movement.
- The route carries compare lhs/rhs, source payload, index vector, destination memory, runtime n/AVL, SEW/LMUL/policy, unique index policy, inactive-lane no-write policy, and tail preservation through the RVV dialect/plugin-owned path.
- Added explicit and pre-realized target fixtures plus dialect dataflow/negative coverage; updated focused conversion negative allowlists for the new generic RVV op surface.
- Self-repair performed: excluded indexed scatter from the computed-mask masked_load provider branch, added compare predicate metadata for target artifact validation, and repaired conversion FileCheck allowlists after the new op surface changed diagnostics.
- Checks passed: manual FileCheck for explicit/pre-realized PLAN/REALIZED/HEADER fixtures and dialect negative verifier coverage; `cmake --build build -j2`; construction/RVV extension/target artifact C++ tests; `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`; generated-bundle self-test; explicit and pre-realized dry-runs for counts `7,16,23`; real `ssh rvv` explicit and pre-realized PASS for counts `7,16,23`; `cmake --build build --target check-tianchenrv -j2` with `308/308` tests; `git diff --check`.
- Runtime evidence proved active indexed stores, mixed true/false mask behavior, inactive-lane no-write preservation, unrelated/tail sentinel preservation, runtime n/AVL variation, and noncontiguous/permuted unique index behavior.
- Authority scan found no new positive legacy/source/descriptor/common-export route authority; staged hits are negative-boundary wording, one fail-closed stale route-id input, and the provider-owned target leaf selected after typed closure.
- Spec update review: no `.trellis/spec/**` change was needed because existing RVV plugin, EmitC route, and MLIR testing contracts already define the long-term boundary; this round added bounded route coverage inside that boundary.
- Segmented movement, broad gather/scatter matrices, additional dtype/LMUL clones, source-front-door routes, and Stage2 class expansion were intentionally not converted in this bounded owner.

Final round commit is created after task archive in the same Codex turn.


### Git Commits

(No commits - planning session)

### Testing

- [OK] Focused build:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`.
- [OK] Focused explicit/pre-realized target/header pipelines for indexed
  gather/scatter selected-body fixtures.
- [OK] Existing indexed-memory verifier diagnostics:
  `masked-indexed-load-dataflow.mlir`,
  `computed-mask-indexed-scatter-store-dataflow.mlir`,
  `indexed-gather-memory-dataflow.mlir`, and
  `indexed-scatter-memory-dataflow.mlir`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test`.
- [OK] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` and
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`.
- [OK] Generated-bundle dry-runs for explicit and pre-realized
  `computed_masked_indexed_gather_load_unit_store` and
  `computed_masked_indexed_scatter_store_unit_load` with counts `7,16,23`.
- [OK] Real `ssh rvv` generated-bundle runs for explicit and pre-realized
  indexed gather/scatter with counts `7,16,23`; both routes reported active
  lanes, inactive preservation, non-contiguous index lanes, source
  preservation, tail preservation, and `PASS`.
- [OK] Active-authority scan, `git diff --check`, and
  `cmake --build build --target check-tianchenrv -j2` passed 349/349.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 158: Stage2 RVV runtime scalar splat store

**Date**: 2026-05-22
**Task**: Stage2 RVV runtime scalar splat store
**Branch**: `main`

### Summary

Added closure-gated runtime_i32_splat_store route with explicit/pre-realized selected-body support, RouteOperandBindingPlan closure, generated-bundle dry-run, and real ssh rvv evidence.

### Main Changes

- Added planning-owned memory family predicates for computed-mask memory,
  plain segment2 memory, and the combined memory-family consumer set.
- Added `verifyRVVSelectedBodyMemoryRouteFamilyProviderPlans` and made the
  RVV EmitC provider call it before materializing active memory-family routes.
- Removed duplicated provider-local computed-mask memory and plain segment2
  family predicate tables.
- Factored computed-mask segment2 and plain segment2 metadata mirror emission
  into `addRVVSelectedBodySegment2MemoryRouteFamilyMetadataMirrors`.
- Preserved route ids, selected-body semantics, ABI order, binding closure,
  generated artifacts, and common EmitC/export neutrality.

### Git Commits

| Hash | Message |
|------|---------|
| `same-squashed-commit` | (see git log) |

### Testing

- [OK] Focused build:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused lit filter for representative memory-family routes:
  29/29 selected tests passed.
- [OK] Generated-bundle dry-run for representative computed-mask memory,
  computed-mask strided/indexed/segment2, and plain segment2 routes with
  counts `7,16,23`.
- [OK] Real `ssh rvv` smoke for `runtime_scalar_cmp_masked_store` with counts
  `7,16,23`.
- [OK] Real `ssh rvv` smoke for `segment2_deinterleave_unit_store` with counts
  `7,16,23`.
- [OK] Active-authority scan, `git diff --check`, and
  `cmake --build build --target check-tianchenrv -j2` passed 349/349.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 160: Stage2 RVV computed-mask composition boundary

**Date**: 2026-05-23
**Task**: Stage2 RVV computed-mask composition boundary
**Branch**: `main`

### Summary

Implemented runtime_scalar_dual_cmp_mask_and_select across RVV typed body, selected-body realization, route planning/provider, generated bundle harness, ssh rvv evidence, and task archive. Final commit is created after this journal entry.

### Main Changes

- Added rhs-secondary-scalar-value ABI role, typed tcrv_rvv.mask_and, and pre-realized dual runtime scalar compare mask-and select/store body.
- Rewired RVV construction/realization/planning/provider/export so dual runtime scalar thresholds produce two compare masks, compose them with typed mask_and, and feed select/store through RouteOperandBindingPlan closure.
- Added explicit and pre-realized generated-bundle harness evidence with counts 7,16,23 and rhs threshold pairs -37,91; real ssh rvv runs passed for both modes.
- Final validation before archive: check-tianchenrv 349/349, git diff --check, and active-authority scan.


### Git Commits

(No commits - planning session)

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 161: Stage2 RVV computed-mask memory route family

**Date**: 2026-05-23
**Task**: Stage2 RVV computed-mask memory route family
**Branch**: `main`

### Summary

Migrated runtime_scalar_cmp_masked_store and runtime_scalar_cmp_masked_load_store to a shared RVV plugin-owned computed-mask memory route-family plan; focused lit, generated-bundle dry-runs, explicit/pre-realized ssh rvv, authority scan, and check-tianchenrv passed.

### Main Changes

- Replaced the store-named runtime scalar computed-mask route-family plan with a shared memory route-family plan consumed by both `runtime_scalar_cmp_masked_store` and `runtime_scalar_cmp_masked_load_store`.
- Added `usesLoadMerge` to keep store-only and load-merge/store facts explicit inside one plugin-owned family abstraction.
- Made RVV EmitC route provider require the shared memory family plan before materializing runtime-scalar computed-mask store/load-store routes and consume plan-owned setvl/load/splat/compare/store/header facts.
- Preserved RouteOperandBindingPlan closure for lhs, rhs_scalar, src, dst, and n; common EmitC/export remains neutral.

### Git Commits

this commit

### Testing

- [OK] Focused build: `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] Focused lit filter for runtime-scalar computed-mask store/load-store dialect and target artifacts.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Generated-bundle dry-runs for explicit and pre-realized store/load-store with counts `7,16,23` and thresholds `-37,91`.
- [OK] Real `ssh rvv` generated-bundle runs for explicit and pre-realized store/load-store with counts `7,16,23` and thresholds `-37,91`.
- [OK] Active-authority scan, stale store-family symbol scan, `git diff --check`, and `cmake --build build --target check-tianchenrv -j2`.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 162: Stage2 RVV computed-mask accumulation route family

**Date**: 2026-05-23
**Task**: Stage2 RVV computed-mask accumulation route-family interface
**Branch**: `main`

### Summary

Migrated the existing `runtime_scalar_cmp_masked_macc_add` and
`runtime_scalar_cmp_masked_standalone_reduce_add` production routes to consume
one shared RVV plugin-local runtime-scalar computed-mask accumulation
route-family plan while preserving distinct vector macc and scalar horizontal
reduction suffix semantics.

### Main Changes

- Added `RVVSelectedBodyRuntimeScalarComputedMaskAccumulationRouteFamilyPlan`
  for common runtime scalar threshold compare, produced-mask, runtime AVL/VL,
  header/type, accumulator/result contract, and binding-closure facts.
- Demoted the old macc one-off target/profile/runtime-control description
  population behind the shared accumulation plan.
- Made the provider require the shared accumulation plan before materializing
  either route and share common `cmp_lhs`, `rhs_scalar`, and `n` operand
  binding checks.
- Added accumulation-family mirror fields to target/header evidence while
  keeping common EmitC/export neutral.
- Left select, memory, non-runtime computed-mask macc/reduction, widening
  contraction, source-front-door, and generated-bundle script behavior
  unchanged.

### Testing

- [OK] Focused build:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] Focused lit filter for runtime-scalar computed-mask macc and standalone
  reduction dialect/target artifacts: 6/6 selected tests passed.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Generated-bundle dry-runs for explicit and pre-realized macc/reduction
  with counts `7,16,23` and thresholds `-37,91`.
- [OK] Real `ssh rvv` generated-bundle runs for explicit and pre-realized
  macc/reduction with counts `7,16,23` and thresholds `-37,91`.
- [OK] Active-authority scan, `git diff --check`, and
  `cmake --build build --target check-tianchenrv -j2` passed 349/349.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 164: Stage2 RVV computed-mask indexed-memory route family

**Date**: 2026-05-23
**Task**: Stage2 RVV computed-mask indexed-memory route family
**Branch**: `main`

### Summary

Migrated computed-mask indexed gather/scatter routes onto the shared RVV plugin-owned computed-mask memory family and proved explicit/pre-realized generated bundles on ssh rvv.

### Main Changes

- Extended the shared computed-mask memory route-family plan with indexed gather/scatter facets for index vector type, index load/scale leaves, indexed store leaf, index EEW, offset unit, index source, scatter uniqueness, and indexed data/destination forms.
- Made `computed_masked_indexed_gather_load_unit_store` and `computed_masked_indexed_scatter_store_unit_load` consume family derivation, validation, provider-required planning, target leaf/profile mirrors, headers, C type mapping, and RouteOperandBindingPlan closure.
- Updated explicit and pre-realized target/header checks plus generated-bundle metadata expectations for family plan id, mask producer source, provider mirror, indexed memory layout, index source/EEW/offset, and indexed source/destination forms.
- Kept segmented memory and unrelated memory forms outside scope.


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 165: Stage2 RVV plain segment2 memory route family

**Date**: 2026-05-23
**Task**: Stage2 RVV plain segment2 memory route family
**Branch**: `main`

### Summary

Migrated Segment2DeinterleaveUnitStore and Segment2InterleaveUnitLoad onto a plugin-owned plain segment2 memory route-family plan; focused positive and negative lit, dry-runs, ssh rvv, authority scan, and check-tianchenrv passed.

### Main Changes

- Added `RVVSelectedBodySegment2MemoryRouteFamilyPlan` and made both
  `Segment2DeinterleaveUnitStore` and `Segment2InterleaveUnitLoad` derive
  route support through that plugin-local family plan.
- Moved segment count/layout, field roles, source/destination memory forms,
  runtime AVL/VL order, target leaf/profile mirrors, headers, C type mapping,
  and operand binding evidence into plan-owned/provider-validated facts.
- Removed the residual plain segment2 one-off description authority and made
  the provider fail closed unless the family plan is present.
- Updated target-header and generated-bundle script evidence while keeping
  common EmitC/export neutral.

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] Focused build:
  `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] Focused positive lit for the two migrated plain segment2 target/script
  fixtures: 4/4 selected tests passed.
- [OK] Focused negative/dataflow lit for segment2 dataflow, operand binding,
  and incomplete typed-body fail-closed surfaces: 5/5 selected tests passed.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Generated-bundle dry-runs for pre-realized deinterleave/interleave
  routes with counts `7,16,23`.
- [OK] Real `ssh rvv` generated-bundle runs for both migrated routes with
  counts `7,16,23`, including field-order and tail preservation checks.
- [OK] Active-authority scan, `git diff --check`, and
  `cmake --build build --target check-tianchenrv -j2` passed 349/349.

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 166: Stage2 RVV memory route-family ownership extraction

**Date**: 2026-05-23
**Task**: Stage2 RVV memory route-family ownership extraction
**Branch**: `main`

### Summary

Extracted planning-owned memory family provider predicates and segment2 family metadata mirror production while preserving active RVV memory route semantics; focused build/lit/dry-run/ssh rvv/check-tianchenrv passed.

### Main Changes

(Add details)

### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete


## Session 167: Stage2 RVV standalone reduction route-family ownership

**Date**: 2026-05-23
**Task**: Stage2 RVV standalone reduction route-family ownership
**Branch**: `main`

### Summary

Moved active standalone reduction routes behind a validated RVV plugin-owned family plan and proved focused lit, bundle, ssh rvv, authority scan, and check-tianchenrv evidence.

### Main Changes

### Main Changes

- Added `rvv-standalone-reduction-route-family-plan.v1` as an explicit standalone reduction family plan mirror carried through route description, target metadata, and generated-bundle expectations.
- Added planning-owned standalone reduction family consumer predicates and provider-plan verification.
- Made provider materialization require the validated standalone reduction family plan for plain, computed-mask, and runtime-scalar computed-mask standalone reduction routes.
- Replaced provider-local standalone reduction route predicates with plan-derived materialization booleans while preserving shared computed-mask accumulation producer mechanics for add/runtime-scalar add.
- Updated focused explicit/pre-realized target/header FileCheck coverage for standalone family plan presence.

### Routes Covered

- `standalone_reduce_add/min/max`
- `computed_mask_standalone_reduce_add/min/max`
- `runtime_scalar_cmp_masked_standalone_reduce_add`

### Testing

- [OK] Focused build: `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test -j2`.
- [OK] `build/bin/tianchenrv-rvv-extension-plugin-test`.
- [OK] Focused lit filter for standalone reduction target/dialect/script fixtures: 21/21 selected tests passed.
- [OK] Generated-bundle dry-runs for explicit and pre-realized standalone reduction routes, all seven active op kinds, counts `7,16,23`, thresholds `-37,91` where applicable.
- [OK] Real `ssh rvv` explicit representative run for plain, computed-mask, and runtime-scalar masked standalone reduce add.
- [OK] Real `ssh rvv` pre-realized representative run for plain and runtime-scalar masked standalone reduce add.
- [OK] Added-line active-authority scan, full touched-file scan review, `git diff --check`, and `cmake --build build --target check-tianchenrv -j2` passed 349/349.

### Self-Repair

- Moved provider-facing standalone family verifier out of the anonymous implementation namespace after compile ambiguity.
- Corrected FileCheck ordering for diagnostic metadata versus target header metadata.

### Status

[OK] **Completed and archived**

### Next Steps

- None - task complete


### Git Commits

| Hash | Message |
|------|---------|
| `this commit` | (see git log) |

### Testing

- [OK] (Add test results)

### Status

[OK] **Completed**

### Next Steps

- None - task complete
