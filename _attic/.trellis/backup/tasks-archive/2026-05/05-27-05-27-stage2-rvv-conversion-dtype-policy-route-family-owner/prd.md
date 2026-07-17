# Stage2 RVV Conversion Dtype-Policy Route-Family Owner

## Goal

Introduce or extract a plugin-local RVV conversion dtype-policy route-family
owner across the active selected-body, planning/provider, statement-plan, and
target artifact ABI consumer path. The primary migrated consumer is the existing
widening conversion route. One adjacent existing scalar-broadcast elementwise
consumer must prove that source/result dtype, SEW, LMUL, policy, type mapping,
intrinsic selection, runtime/VL facts, and operand binding are shared typed
family facts rather than route names, exact intrinsic spellings, ABI strings,
artifact metadata, scripts, or per-route clones.

The completed route must stay on the current RVV-first chain:

```text
selected tcrv.exec RVV variant
  -> typed low-level tcrv_rvv conversion or scalar-broadcast body
  -> RVV plugin-owned conversion dtype-policy route-family ownership
  -> provider facts derived from typed source/result dtype/config/runtime facts
  -> provider-built TCRVEmitCLowerableRoute
  -> neutral common EmitC materialization
  -> target artifact ABI consumer
  -> generated RVV C artifact/harness
  -> ssh rvv evidence for executable claims
```

## Direction Source

- Direction title: `Stage2 RVV conversion dtype-policy route-family owner`.
- Module owner: RVV plugin-local conversion and dtype/SEW/LMUL policy
  route-family boundary, carried from typed selected-body facts through route
  planning/provider construction and target artifact ABI validation.
- Repository root: `/home/kingdom/phdworks/TianchenRV`.
- Initial `pwd`: `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short`: clean.
- Initial HEAD: `24672fb7 rvv: add compare select mask route owner`.
- `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
- Serial worker constraint from the brief: do not use subagents, spawned
  agents, parallel agents, or multi-agent workflows.

## What I Already Know

- The predecessor archived task
  `.trellis/tasks/archive/2026-05/05-27-stage2-rvv-compare-select-mask-route-family-owner`
  completed compare/select mask route-family ownership, target artifact ABI
  validation, generated-bundle evidence, `ssh rvv` evidence for `cmp_select`,
  and `check-tianchenrv` 390/390.
- Specs require dtype/config/operation authority to be structural in typed or
  realized `tcrv_rvv` body/config facts. `tcrv.exec.mem_window` and
  `tcrv.exec.runtime_param` bind ABI/runtime roles only; they do not define
  RVV dtype, conversion semantics, SEW/LMUL, policy, intrinsic spelling, or
  route support.
- Current RVV planning code already exposes widening conversion route-family
  plan mirrors, source/result SEW/LMUL fields, conversion relation fields,
  scalar-broadcast elementwise route-family plans, materialization facts,
  route-control provider plans, and migrated statement-plan owner plumbing.
  This task should extract or tighten a real conversion dtype-policy owner over
  those production surfaces instead of adding a wrapper beside them.
- Existing code has diagnostic strings for widening conversion provider and
  statement-plan requirements, scalar-broadcast elementwise statement-plan
  requirements, source/result SEW/LMUL validation, route-control validation,
  metadata mirror emission, and target artifact consumers. The implementation
  should reuse or demote one-off checks behind a shared owner boundary.
- Common EmitC/export must remain neutral. Target artifact export may validate
  provider-built mirrors after route construction, but it must not infer
  conversion support, dtype policy, type/header mapping, intrinsic selection,
  ABI order, or executable claims from scripts, artifact names, route ids,
  metadata, exact intrinsic spelling, descriptors, comments, or legacy i32
  helper names.

## Requirements

1. Add or extract one explicit RVV plugin-local conversion dtype-policy
   route-family owner surface. It must cover the existing widening conversion
   route and one adjacent existing scalar-broadcast elementwise route.
2. The owner must derive support, conversion kind/relation, source/result
   element types, source/result SEW and LMUL, tail/mask policy where applicable,
   VL/AVL use, operand bindings, ABI order, required headers, C type/vector
   type names, intrinsic family facts, statement-plan eligibility,
   provider-supported mirror labels, and artifact ABI claims from typed
   `tcrv_rvv` body/config/runtime/conversion/dtype facts and selected target
   capability facts.
3. The active provider route construction path must consume this owner before
   building or accepting `TCRVEmitCLowerableRoute` payload facts for the
   migrated widening conversion and scalar-broadcast elementwise consumers.
   Existing one-off widening conversion and scalar-broadcast dtype/config
   checks should be reused behind the owner or demoted, not duplicated as
   parallel authority.
4. Target artifact/runtime ABI validation must consume provider-built
   conversion dtype-policy family facts for the migrated consumers before
   accepting generated artifact/header claims. Artifact metadata may mirror the
   family owner only after provider route construction.
5. Unsupported or inconsistent family, stale route id, stale mirror metadata,
   wrong conversion kind/relation, source/result dtype mismatch, source/result
   SEW/LMUL mismatch, policy mismatch, wrong ABI order, missing operand
   binding, wrong type/header mapping, exact-intrinsic-as-authority, and
   artifact-name or script-derived authority attempts must fail closed with
   targeted diagnostics before common EmitC or artifact export.
6. Existing completed segment2, MAcc, standalone reduction/accumulation,
   compare/select, computed-mask memory, runtime-scalar, contraction,
   scalar-broadcast MAcc, and base memory selected-body/provider/artifact paths
   must not be weakened or reclassified as conversion dtype-policy evidence.
7. The task must not add broad dtype/LMUL clone batches, broad route coverage
   expansion, new high-level Linalg/frontend routes, source-front-door positive
   routes, descriptor/direct-C/source-export paths, dashboards, report-only
   work, one-intrinsic wrapper dialects, compare/select follow-on polish,
   segment2 follow-on polish, MAcc follow-on polish, or reduction/accumulation
   follow-on polish.

## Acceptance Criteria

- [ ] A production conversion dtype-policy route-family owner exists in
      RVV plugin-local C++ and is consumed by the active route/provider path.
- [ ] The existing widening conversion route and one adjacent existing
      scalar-broadcast elementwise route are migrated behind that owner as
      active consumers.
- [ ] The provider-built route for the migrated consumers proves owner-derived
      typed body/config/runtime/conversion/dtype/SEW/LMUL/policy facts before
      materialization, including route-control, operand binding,
      materialization, statement-plan, header/type/intrinsic, ABI order, and
      provider-supported mirror agreement.
- [ ] Target artifact ABI validation consumes provider-built conversion
      dtype-policy family facts for the migrated consumers before accepting
      object/header artifact claims.
- [ ] Focused C++ or lit tests prove positive owner membership/dispatch and
      fail-closed diagnostics for stale route id, stale mirror metadata, wrong
      family, wrong conversion kind/relation, source/result dtype mismatch,
      source/result SEW/LMUL mismatch, policy mismatch, wrong ABI order,
      missing operand binding, wrong type/header mapping, exact-intrinsic
      authority attempts, and artifact-name or script-derived authority
      attempts where the current test surface exposes them.
- [ ] Generated-bundle dry-run covers both migrated consumers.
- [ ] Real `ssh rvv` generated-bundle execution covers a conversion
      representative over counts including `0`, small, exact, tail, and stress
      cases.
- [ ] Focused non-regression covers completed segment2 update/store/load,
      segment2 deinterleave/interleave, computed-mask MAcc,
      scalar-broadcast MAcc, runtime-scalar MAcc, standalone
      reduction/accumulation, compare/select, computed-mask memory,
      scalar-broadcast, runtime-scalar, contraction, and base memory paths.
- [ ] Bounded touched-file authority scan finds no new executable or route
      support depending on central ad hoc, name-derived, metadata-derived,
      descriptor-derived, ABI-string-derived, script-derived,
      artifact-name-derived, common-EmitC-derived, source-front-door-derived,
      route-id-derived, exact-intrinsic-derived, or legacy-i32-derived
      authority.
- [ ] `git diff --check` passes.
- [ ] Focused build/tests pass, and `check-tianchenrv` passes or an exact
      blocker is recorded with the task left open.
- [ ] Trellis task status/context/journal/archive are truthful, and one
      coherent commit is created if all acceptance criteria are satisfied.

## Validation Plan

1. Start and validate this Trellis task after PRD/context setup.
2. Inspect current widening conversion, scalar-broadcast elementwise,
   route-family planning, route-control provider, materialization facts,
   migrated statement-plan owner, target artifact ABI validation,
   generated-bundle, and focused test surfaces.
3. Implement the conversion dtype-policy owner in the smallest production RVV
   plugin surface that currently owns duplicated or ad hoc conversion/dtype
   policy checks.
4. Wire provider construction and target artifact validation to consume the
   owner-derived facts for the migrated widening conversion and scalar-broadcast
   elementwise consumers.
5. Add focused C++ and/or lit tests for owner membership, provider/target
   consumption, and fail-closed stale or missing family facts.
6. Run focused build/tests for touched RVV plugin and target surfaces.
7. Run generated-bundle dry-runs for both migrated consumers.
8. Run real `ssh rvv` evidence for one conversion representative over counts
   `0`, small, exact, tail, and stress cases.
9. Run the required focused non-regression dry-runs/lit coverage.
10. Run bounded authority scan, `git diff --check`, task validation, and
    `check-tianchenrv`.

## Out Of Scope

- Compare/select follow-on polish or using compare/select as the active
  achievement.
- Segment2, MAcc, standalone reduction/accumulation, contraction, gather, or
  scatter follow-on polish.
- New conversion kinds beyond the existing active widening conversion
  representative, unless needed to make the owner truthful for current active
  consumers.
- Dtype/LMUL clone batches, broad Stage2 coverage expansion, high-level
  frontend/Linalg/Vector/StableHLO routes, global tuning/profile databases,
  dashboards, reports-only changes, or smoke-only work.
- Any new `tcrv_rvv.i32_*` helper op, legacy `RVVI32M1`/`rvv-i32m1` route, or
  exact intrinsic spelling as route authority.
- Common EmitC or target export inferring conversion support, dtype, SEW/LMUL,
  policy, ABI order, type/header mapping, intrinsic, or executable authority
  from route ids, metadata, artifact names, scripts, descriptors, ABI strings,
  or source-front-door fixtures.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Predecessor archived context read:
  `.trellis/tasks/archive/2026-05/05-27-stage2-rvv-compare-select-mask-route-family-owner/prd.md`,
  `implement.jsonl`, `check.jsonl`, and the matching journal entry in
  `.trellis/workspace/codex/journal-16.md`.
- Initial code surfaces inspected:
  `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and
  `test/Plugin/RVVExtensionPluginTest.cpp`.

## Open Questions

None blocking. The Direction Brief and current specs point to a bounded
implementation: extract or add a conversion dtype-policy owner over the
existing widening conversion route and an adjacent scalar-broadcast elementwise
consumer, then make provider/target consumers depend on that owner for typed
source/result dtype, SEW/LMUL, policy, conversion relation, ABI, type/header,
intrinsic, and artifact mirror validation.

## Definition Of Done

The conversion dtype-policy family owner is the active production boundary for
the migrated consumers, provider-built routes and target artifacts consume
owner-derived facts, focused positive and fail-closed tests pass,
generated-bundle and `ssh rvv` evidence are recorded, authority scans are
clean, Trellis state is truthful, and one coherent commit records the work.
