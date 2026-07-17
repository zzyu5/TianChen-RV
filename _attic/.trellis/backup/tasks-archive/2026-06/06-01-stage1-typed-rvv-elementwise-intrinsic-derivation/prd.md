# Stage1 typed RVV elementwise intrinsic derivation

## Goal

Replace the remaining bounded `i32m1`/`i32m2` exact-spelling authority in the
RVV elementwise EmitC route-family plan owner with typed selected-body/config
derivation. Existing route-supported elementwise add/sub/mul and directly
connected scalar-broadcast, strided, mask/result type, and memory/splat leaves
may still produce today's `i32m1` or `i32m2` RVV intrinsic names, but only as
validated outputs derived from operation kind, element type, SEW, LMUL, policy,
memory form, runtime AVL/VL, and selected typed `tcrv_rvv` body facts.

## What I Already Know

* The Hermes brief is the source of this task.
* The repository starts from `main` with a clean worktree at commit
  `1f8bd22b rvv: derive base memory leaves from typed facts`.
* There is no active `.trellis/.current-task`; this task was created for this
  direction brief as
  `.trellis/tasks/06-01-stage1-typed-rvv-elementwise-intrinsic-derivation`.
* The prior archived task
  `.trellis/tasks/archive/2026-06/06-01-06-01-stage1-typed-rvv-base-memory-elementwise-intrinsic-derivation/`
  completed base-memory typed derivation and one scalar-broadcast validation
  proof point.
* `RVVSelectedBodyTypedConfigFacts` already carries element type, SEW, LMUL,
  policy, vector/mask/index C types, setvl, load/store, and base-memory memory
  leaves.
* `RVVEmitCElementwiseRouteFamilyPlanOwners.cpp` still contains exact
  elementwise output mappings for arithmetic, compare/select, strided memory,
  scalar splat, vector/mask/result C types, and scalar-broadcast profile/summary
  strings.
* Specs require selected `tcrv.exec` RVV variants to consume typed low-level
  `tcrv_rvv` bodies through the RVV plugin route provider. Common EmitC/export
  must not choose RVV dtype, schedule, C types, intrinsics, or operation
  semantics.

## Requirements

* Keep the production route as selected `tcrv.exec` RVV variant -> typed
  low-level `tcrv_rvv` body/config -> RVV plugin route-family planning ->
  provider-built `TCRVEmitCLowerableRoute` -> common EmitC materialization.
* For existing route-supported elementwise add/sub/mul, derive arithmetic
  intrinsic names from operation kind plus typed element/SEW/LMUL facts.
* For connected scalar-broadcast elementwise add/sub/mul, derive scalar element
  C type, scalar splat intrinsic, arithmetic intrinsic, vector/result C types,
  setvl/load/store leaves, and profile/summary mirrors from the same typed
  config snapshot.
* For connected strided elementwise add and masked elementwise add/sub/mul,
  derive strided memory leaves, mask/result type names, mask/result C types, and
  arithmetic/merge leaves from typed config, memory form, and mask policy facts.
* Treat current `i32m1` and `i32m2` intrinsic spelling as derived output only;
  do not use old route names, artifact names, helper op names, ABI strings, or
  owner-local legacy constants as selected-route authority.
* Unsupported or incomplete element type, SEW, LMUL, policy, memory-form, scalar
  type, mask type, or typed config facts must fail closed with targeted
  diagnostics naming the elementwise route-family boundary and missing/stale
  field.
* Keep changes bounded to elementwise route-family planning/provider surfaces,
  focused tests, and Trellis task evidence.

## Acceptance Criteria

* [x] `RVVEmitCElementwiseRouteFamilyPlanOwners.cpp` no longer treats exact
  `__riscv_*_i32m1`, `__riscv_*_i32m2`, `vint32m1_t`, `vint32m2_t`,
  `vbool32_t`, or scalar `int32_t` strings as owner-local route authority for
  supported elementwise paths.
* [x] Existing add/sub/mul and scalar-broadcast add/sub/mul produce intrinsic
  leaves only through typed element/SEW/LMUL plus operation-kind derivation, and
  fail closed for unsupported typed combinations.
* [x] Existing strided and masked elementwise leaves consume typed config,
  memory-form, mask/result, and policy facts rather than hardcoded i32m route
  constants.
* [x] Positive focused tests prove typed config-derived elementwise planning for
  already supported cases, including at least plain add/sub/mul and one
  connected scalar-broadcast or strided/masked case.
* [x] Negative focused tests mutate missing or unsupported typed config facts and
  assert fail-closed diagnostics before provider route construction.
* [x] Bounded scans over touched elementwise/base-memory planning files, tests,
  and relevant specs classify remaining legacy strings as derived output,
  parse-only/deprecated, negative/spec text, or unrelated remaining Stage 1
  debt rather than active authority in this owner.
* [x] Focused build/test commands for the touched RVV plugin paths pass.

## Definition of Done

* Source changes are limited to RVV plugin-owned planning/provider tests and
  directly required Trellis task files unless inspection proves a narrow header
  or shared planning helper change is necessary.
* `git diff --check` passes.
* Focused RVV plugin tests pass.
* No runtime correctness or performance claim is made without `ssh rvv`
  evidence.
* Trellis task status, evidence, and journal truthfully record what changed and
  the exact continuation point if unfinished.
* One coherent commit is created when the task is complete.

## Out of Scope

* New RVV operation coverage, reductions, contractions, conversion/dtype
  expansion, compare/select expansion beyond validating existing leaves, or
  Stage 2 coverage growth.
* Source-front-door/source-artifact positive routes.
* Descriptor-driven C/source export.
* Common EmitC semantic branches.
* Compatibility wrappers preserving old i32 route authority.
* Broad test matrices or report-only/metadata-only work.

## Technical Approach

Use the current base-memory typed derivation pattern as the local model. Add or
reuse owner-local typed derivation helpers that take
`RVVSelectedBodyTypedConfigFacts`, operation kind, and route memory/mask/scalar
sub-family facts, then return provider leaf outputs. The helpers should reject
missing or unsupported typed facts before a route-family plan is accepted. Plan
validation should compare plan fields against typed config-derived expected
values, not against route-name or hardcoded `i32m1` templates.

If the whole elementwise owner is too large for one round, complete one coherent
plain add/sub/mul plus scalar-broadcast/strided/masked leaf submodule and leave
the exact remaining owner-local derivation continuation point.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/index.md` and
  `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/testing/index.md`, and `.trellis/spec/variant-pipeline/index.md`.
* Read prior task PRD/evidence:
  `.trellis/tasks/archive/2026-06/06-01-06-01-stage1-typed-rvv-base-memory-elementwise-intrinsic-derivation/`.
* Read current implementation surfaces:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp`, and
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`.
