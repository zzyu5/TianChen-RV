# Expand: Stage2 RVV standalone-reduction route-family config authority closure

## Goal

Close the standalone-reduction route-family authority boundary so operation
kind, source/work vector LMUL, scalar accumulator/result channel, seed and
lane-0 output binding, runtime `n`/AVL/VL, policy, runtime ABI order, provider
facts, and artifact mirrors are validated as facts derived from the typed
`tcrv_rvv` body/config/runtime path. Fixture names, op-kind suffixes, artifact
names, script selectors, route ids, and metadata may remain only as selectors
or mirrors after provider route construction.

## Direction Source

- Direction title: `Expand: Stage2 RVV standalone-reduction route-family config authority closure`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `9f0c5582 rvv: add standalone reduce add m2 route`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint: one Codex worker; no spawned agents or multi-agent
  workflow.

## What I Already Know

- The previous archived task
  `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-standalone-reduce-add-lmul-m2`
  proved the last single plain `standalone_reduce_add_lmul_m2` member through
  selected-boundary realization, provider route construction, generated bundle
  evidence, ssh RVV correctness, and `check-tianchenrv` 456/456.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires standalone
  reductions to expose distinct source/work vector facts and scalar
  accumulator/result facts when source LMUL differs from scalar-result LMUL.
- `.trellis/spec/core-dialect/tcrv-exec-contract.md` keeps `mem_window` and
  `runtime_param` as ABI/runtime role declarations only; selected typed RVV
  body and provider facts must carry RVV compute/config authority.
- `.trellis/spec/lowering-runtime/emitc-route.md` keeps common EmitC neutral:
  it materializes provider-built `TCRVEmitCLowerableRoute` payloads and must
  not infer operation kind, dtype, SEW, LMUL, policy, or runtime ABI from
  names or metadata.
- Bounded inspection shows the standalone-reduction family plan validates its
  own source/scalar-result types, runtime ABI order, scalar seed/output roles,
  and operation-specific leaves, but provider mirror verification did not
  explicitly compare every standalone source/scalar-result mirror field in the
  route description against the validated family plan.
- Bounded inspection also shows target artifact payload validation checks
  generic vector type mapping for the family but should explicitly require both
  standalone-reduction source-vector and scalar-result-vector type mappings to
  be present in the rebuilt provider route.

## Requirements

1. Keep compiler implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python may only be used for generated-bundle evidence tooling.
2. Do not add another one-off LMUL clone or new standalone-reduction operation
   family. Expand authority checks for the existing route family.
3. Provider verification must fail closed when a standalone-reduction route
   description carries stale or missing source vector type/C type mirrors,
   scalar-result vector type/C type mirrors, operation kind, runtime ABI order,
   scalar seed/output binding, route operand binding, runtime `n`/AVL/VL, or
   policy facts.
4. Target artifact validation must require the rebuilt provider route to carry
   both source/work vector type mapping and scalar-result vector type mapping
   for standalone reductions, including m1 equality cases and m2 -> m1 split
   cases.
5. Common EmitC/export must remain neutral and must not choose RVV semantics;
   any target-side validation must compare rebuilt provider route facts, not
   derive semantics from metadata.
6. Script changes are allowed only if the script is missing consumer/mirror
   evidence for existing representative add/min/max m1/m2 fixtures. The script
   must not become route authority.
7. Direct pre-realized route-entry for standalone reductions must remain
   fail-closed; selected-boundary producer remains the positive path.

## Acceptance Criteria

- [x] Provider-level verification rejects stale standalone-reduction source
      vector type/C type mirrors before provider materialization.
- [x] Provider-level verification rejects stale standalone-reduction scalar
      result vector type/C type mirrors before provider materialization.
- [x] Target artifact payload validation requires rebuilt provider route type
      mappings for both standalone source/work and scalar-result channels.
- [x] Focused C++ coverage proves the existing plain m2 add route consumes
      source m2 and scalar-result m1 facts from the family plan and that stale
      mirrors fail closed.
- [x] Focused C++ or lit coverage proves target/export validation remains based
      on rebuilt provider route facts rather than metadata-only support.
- [x] Generated-bundle dry-run evidence covers representative plain standalone
      reductions for add and at least one min/max across existing m1/m2
      fixtures without adding clone families.
- [x] Direct pre-realized route-entry dry-run still rejects standalone
      reductions before provider/common route construction.
- [x] Bounded authority scan over touched RVV planning/provider/target/script
      surfaces finds no new route authority from op-kind suffixes, artifact
      names, script selectors, route ids, exact intrinsic spellings, ABI
      strings, descriptor residue, source-front-door paths, common EmitC, or
      legacy i32 helpers.
- [x] `git diff --check` passes.
- [x] Focused standalone-reduction tests pass.
- [x] `check-tianchenrv` passes, or an exact blocker is recorded with the task
      left open.
- [x] If runtime/correctness behavior is claimed beyond unchanged existing
      paths, run representative `ssh rvv` evidence; otherwise record why the
      task is provider/target contract-only and reuses prior runtime evidence.

## Implementation Summary

- Added provider fail-closed verification that standalone-reduction route
  descriptions must carry source-vector type/C type and scalar-result-vector
  type/C type mirrors matching the validated family plan before route
  materialization.
- Added target artifact consumer validation that rebuilt provider routes carry
  explicit source-vector and scalar-result-vector type mappings before artifact
  export.
- Extended focused RVV plugin coverage so the existing plain
  `standalone_reduce_add_lmul_m2` route proves source m2, scalar-result m1, and
  ABI facts are consumed from provider facts, and stale source/scalar mirrors
  fail closed.
- No common EmitC semantic branch, descriptor route, source-front-door route,
  exact intrinsic spelling authority, new operation family, or LMUL clone batch
  was added.

## Validation Evidence

- `git diff --check`: passed.
- Focused provider test:
  `cmake --build build --target tianchenrv-rvv-extension-plugin-test` followed
  by `build/bin/tianchenrv-rvv-extension-plugin-test`: passed with
  `RVV extension plugin smoke test passed`.
- Focused target artifact test:
  `cmake --build build --target tianchenrv-target-artifact-export-test` followed
  by `build/bin/tianchenrv-target-artifact-export-test`: passed.
- Generated-bundle representative dry-run:
  `scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body`
  for `standalone_reduce_add`, `standalone_reduce_add_lmul_m2`,
  `standalone_reduce_min`, `standalone_reduce_min_lmul_m2`, and
  `standalone_reduce_max_lmul_m2`: passed with artifact root
  `artifacts/tmp/standalone-family-authority-dry-run/plain-standalone-family`.
- Direct pre-realized route-entry dry-run:
  `scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --direct-pre-realized-route-entry`
  for `standalone_reduce_add_lmul_m2`: failed closed as expected with the
  selected-boundary-only diagnostic.
- `check-tianchenrv`: passed after cleaning lit output directories and rerunning
  as one serial check; final result was 456/456 passed.
- Representative `ssh rvv` evidence:
  `scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body` for
  `standalone_reduce_add_lmul_m2` with counts `0,1,16,23,257` and seeds
  `-11,17`: passed, including
  `tcrv_rvv_generated_bundle_abi_standalone_reduce_add_lmul_m2_ok`.
- Authority scan over the touched production/test surfaces found only the new
  provider-derived source/scalar mirror checks and fixture stale-value tests; no
  new op-kind suffix, artifact-name, route-id, exact-intrinsic, descriptor,
  source-front-door, common-EmitC, or legacy i32 route authority was introduced.

## Spec Update Judgment

No durable spec update was required in this round. The existing RVV plugin,
`tcrv.exec`, and EmitC route specs already state the authority boundary being
enforced here: typed RVV body/config/runtime facts and provider-built routes
own standalone-reduction semantics, while target artifact metadata is mirror
data and common EmitC stays neutral.

## Non-goals

- Do not add m4/m8 expansion, a new dtype/LMUL clone batch, high-level Linalg
  reduction lowering, source-front-door positive RVV routes, descriptor-driven
  computation, one-intrinsic wrapper dialects, dashboards, broad smoke matrices,
  or unrelated proof-only tests.
- Do not change common EmitC/export to choose standalone-reduction semantics.
- Do not treat generated artifacts, route ids, status fields, script options,
  fixture names, or artifact names as acceptance or route authority.

## Technical Notes

- Primary production surfaces:
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
  - `test/Plugin/RVVExtensionPluginTest.cpp`
  - focused standalone-reduction lit/script evidence as needed
- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
