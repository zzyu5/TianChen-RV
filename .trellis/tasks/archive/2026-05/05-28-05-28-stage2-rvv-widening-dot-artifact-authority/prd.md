# Stage2 RVV widening dot target artifact authority closure

## Goal

Close the target-artifact authority boundary for the existing widening
dot-reduce contraction family. Generated artifact acceptance must depend on a
rebuilt provider `TCRVEmitCLowerableRoute` and provider-derived route facts, not
on optional `widening_dot_*`, `contraction_route_family_plan`, artifact-name,
script, route-id, exact-intrinsic, or common EmitC mirrors.

The bounded family is:

- `widening_dot_reduce_add`
- `strided_input_widening_dot_reduce_add`
- `computed_masked_widening_dot_reduce_add`
- `computed_masked_strided_input_widening_dot_reduce_add`

## Direction Source

- Direction title: `Switch: Stage2 RVV widening dot-reduce contraction target-artifact authority closure`.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `c754ca78 rvv: close standalone reduction route authority`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Hermes Direction Brief before source edits.
- Serial worker constraint: one Codex worker; no subagents, spawned agents,
  parallel agents, or multi-agent workflow.

## What I Already Know

- The previous archived task
  `.trellis/tasks/archive/2026-05/05-28-stage2-rvv-standalone-reduction-route-family-authority`
  closed standalone-reduction provider/target route authority by making
  provider and target checks consume rebuilt source/scalar type facts rather
  than stale mirrors.
- `.trellis/spec/index.md` requires the RVV-first chain:
  selected `tcrv.exec` envelope -> typed `tcrv_rvv` body -> RVV legality /
  selected-body realization / provider -> `TCRVEmitCLowerableRoute` -> common
  EmitC -> target artifact -> `ssh rvv` evidence when runtime/correctness is
  claimed.
- `.trellis/spec/extension-plugins/rvv-plugin.md` already defines the active
  direct-provider contraction routes and requires a direct contraction provider
  plan before route construction. That plan carries dot-reduction,
  computed-mask, strided-input, ABI, VL, type, mask, product, accumulator, and
  store facts.
- `.trellis/spec/extension-plugins/rvv-plugin.md` also gives a target-export
  consumer pattern for segment2: target export rebuilds the provider route and
  compares mirrors against provider description before accepting artifact
  claims.
- `.trellis/spec/lowering-runtime/emitc-route.md` keeps common EmitC neutral.
  It must not infer operation kind, dtype, SEW, LMUL, mask/tail policy, runtime
  ABI, schedule, or RVV intrinsic choices from names or metadata.
- Bounded inspection target files are expected to be:
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `test/Plugin/RVVExtensionPluginTest.cpp`, and target fixtures under
  `test/Target/RVV`.

## Requirements

1. Keep implementation in C++ / MLIR / LLVM / TableGen / CMake / lit /
   FileCheck. Python may be used only for generated-bundle evidence tooling.
2. Do not add new widening dot variants, new MAcc variants, dtype/LMUL clone
   batches, high-level Linalg/frontend lowering, one-intrinsic wrappers, direct
   pre-realized route-entry resurrection, broad smoke matrices, dashboards, or
   evidence-only changes.
3. The provider route description must expose the facts the target artifact
   consumer needs for widening dot-reduce contraction routes:
   operation kind, widening dot relation, i16 lhs/rhs source vector types and C
   types, i32 accumulator/result scalar or vector channel facts,
   accumulator/result layouts, reduction store VL, source/result SEW and LMUL
   relation, computed-mask facts when active, strided-input runtime ABI facts
   when active, required headers, route operand binding plan, runtime ABI
   order, and statement/intrinsic plan.
4. Target artifact validation must rebuild the provider route from the selected
   typed RVV body and require matching provider-derived type mappings, ABI
   mappings, headers, route operand bindings, layout/relation facts, mask/stride
   facts, and statement facts before accepting generated artifacts.
5. Target artifact validation must fail closed on stale or missing
   `widening_dot_*` mirrors, stale or missing
   `contraction_route_family_plan` mirrors, metadata-only support,
   artifact-name/script-derived claims, route-id-only claims,
   exact-intrinsic-as-authority, descriptor/source-front-door residue, and
   common EmitC semantic invention.
6. Common EmitC/export must remain neutral. Any target-side validation may
   compare rebuilt provider route facts but must not choose RVV semantics from
   metadata.
7. Direct pre-realized route-entry support must stay globally fail-closed in
   generated-bundle evidence; selected-boundary realization remains the positive
   path.

## Acceptance Criteria

- [ ] Production diff exists in RVV route planning and/or RVV target support;
      tests alone are not sufficient.
- [ ] Provider-side facts for the widening dot family are available in route
      description or provider description at the target-consumer boundary.
- [ ] Target artifact validation requires rebuilt provider route type mappings
      for i16 dot sources and i32 accumulator/result channels.
- [ ] Target artifact validation requires rebuilt provider ABI mappings,
      runtime ABI order, route operand binding, required headers, and statement
      facts for the widening dot family.
- [ ] Target artifact validation distinguishes plain, strided, computed-mask,
      and computed-mask-strided widening dot members by provider facts rather
      than artifact names, route ids, script selectors, or optional mirrors.
- [ ] Focused C++ tests prove stale or missing widening dot provider facts fail
      before artifact acceptance.
- [ ] Target fixture checks prove provider-derived route type mappings, ABI
      mappings, layouts, relation, mask/stride facts, and statement facts are
      consumed.
- [ ] Generated-bundle selected-boundary dry-run passes for representative
      plain, strided, computed-mask, and computed-mask-strided widening dot
      variants with `route_entry_realization: false`.
- [ ] At least one `ssh rvv` run passes for the highest-risk
      computed-mask-strided widening dot variant.
- [ ] Focused non-regression for an existing plain widening dot artifact
      passes.
- [ ] Direct pre-realized route-entry remains fail-closed for the generated
      bundle harness.
- [ ] Bounded touched-file authority scan finds no new central ad hoc,
      name-derived, metadata-derived, descriptor-derived, ABI-string-derived,
      script-derived, artifact-name-derived, common-EmitC-derived,
      source-front-door-derived, route-id-derived, exact-intrinsic-derived,
      direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
      executable authority.
- [ ] `git diff --check` passes.
- [ ] Focused provider/target tests pass.
- [ ] `check-tianchenrv` passes, or an exact blocker is recorded with the task
      left open.

## Non-goals

- Do not add new widening dot variants, MAcc variants, dtype/LMUL clone batches,
  high-level Linalg/frontend lowering, one-intrinsic wrappers, broad smoke
  matrices, dashboards, or report-only work.
- Do not reopen standalone reduction, compare/select, segment2, widening
  conversion, elementwise, runtime-scalar memory, or MAcc coverage unless a
  tiny shared helper is the single blocker for this authority closure.
- Do not make optional mirrors, emitted metadata, exact intrinsic spellings,
  artifact names, route ids, ABI strings, or generated script selectors route
  authority.

## Technical Notes

- Primary production surfaces:
  - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
  - `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- Primary focused test surfaces:
  - `test/Plugin/RVVExtensionPluginTest.cpp`
  - `test/Target/RVV`
  - `scripts/rvv_generated_bundle_abi_e2e.py` only as evidence consumer
- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`

## Round Result

- Production owner changed:
  `lib/Target/RVV/RVVTargetSupportBundle.cpp` now has a dedicated
  widening-dot target artifact consumer for the four existing dot-reduction
  operations. The consumer rebuilds the provider route, then requires matching
  route id, provider support label, operand binding plan, runtime ABI order and
  ABI mappings, required headers, VL/result/source/mask type mappings,
  i16mf2 -> i32m1 widening relation facts, accumulator/result layouts,
  reduction store VL, source/result SEW/LMUL relation, computed-mask facts,
  strided-input facts, and provider-built pre-loop/loop statement facts.
- Metadata/mirror behavior changed:
  widening-dot `contraction_route_family_plan`, `widening_dot_*`, mask,
  stride, header, ABI, and C type metadata are accepted only as mirrors of the
  rebuilt provider route description. Stale provider support, binding, ABI,
  header, C type, contraction plan, relation, store-VL, stride intrinsic, or
  masked product mirrors fail before target artifact export.
- Test owner changed:
  `test/Target/RVV/explicit-selected-body-artifact-computed-masked-strided-input-widening-dot-reduce-add.mlir`
  now has focused stale-mirror RUN lines for the highest-risk
  computed-mask-strided widening-dot artifact.
- Evidence:
  `cmake --build build --target TianChenRVTarget tcrv-translate tcrv-opt`
  passed; `tianchenrv-rvv-extension-plugin-test` passed;
  `tianchenrv-target-artifact-export-test` passed; `check-tianchenrv` passed
  456/456; generated-bundle dry-run passed for all four widening-dot variants
  with `route_entry_realization=false`; deprecated direct pre-realized
  route-entry failed closed for the computed-mask-strided widening-dot variant;
  `ssh rvv` passed for
  `computed_masked_strided_input_widening_dot_reduce_add` and plain
  `widening_dot_reduce_add`; `git diff --check` passed.
- Authority scan result:
  touched production code does not introduce descriptor/source-front-door,
  artifact-name, route-id, `rvv-i32m1`, or legacy `tcrv_rvv.i32_*` route
  authority. Exact RVV intrinsic spellings remain provider-derived statement
  facts that the target consumer validates against the rebuilt route, not route
  selection authority.
