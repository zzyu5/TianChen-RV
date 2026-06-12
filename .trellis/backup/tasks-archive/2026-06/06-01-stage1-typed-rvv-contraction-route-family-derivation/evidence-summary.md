# Evidence Summary

## Task

Stage1 typed RVV contraction route-family derivation.

## Repository Baseline

- Started on `main` with a clean worktree.
- Baseline commit: `d9f0a81d rvv: derive macc route leaves from typed facts`.
- No active task existed at session start; this task was created and started
  from the Hermes direction brief.

## Implementation Summary

- Added typed result/config and typed source snapshots to
  `RVVSelectedBodyContractionRouteFamilyPlan`.
- Replaced contraction owner-local complete i32m1 intrinsic/type/profile
  constants with helpers that derive:
  - source element type, source SEW/LMUL, source vector type/C type, unit and
    strided source loads from selected typed source operands;
  - result vector type/C type, mask type/C type, setvl, store, policy, and
    config mirrors from selected typed config facts;
  - widening MAcc, widening product, masked product, reduction, compare,
    masked merge, scalar seed splat, target profile, C type mapping summary,
    and relation strings from validated source/result typed facts.
- Made contraction owner validation fail closed for unsupported source/result
  SEW/LMUL and stale relation/leaf mirrors before provider route construction.
- Removed an unreachable common route-description branch that still encoded
  fixed contraction source facts in `RVVEmitCRoutePlanning.cpp`; common
  verification now only requires contraction source facts to be provider
  derived, while owner mirror verification checks RVV-specific leaves.
- Updated focused C++ tests to assert typed source/result plan snapshots and
  the new provider-derived unsupported source/result relation diagnostic.

## Self-Repair Notes

- First C++ test run failed only because a negative test still expected the old
  later config/relation diagnostic. The test was updated to assert the new
  provider-derived `dot_product_relation` source/result typed-facts diagnostic.
- Review found an unreachable legacy fixed-source branch in the common route
  description verifier. It was deleted and focused checks were rerun.
- Review also moved unsupported contraction source/result SEW/LMUL diagnosis
  before derived leaf mirror checks in the contraction owner verifier.

## Checks

- `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-stage1-typed-rvv-contraction-route-family-derivation`
  - Passed: `implement.jsonl` has 7 entries; `check.jsonl` has 6 entries.
- `rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2`
  - Passed.
- `rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
  - Passed: `RVV extension plugin smoke test passed`.
- From `artifacts/tmp/tianchenrv-build/test`:
  `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter='widening-dot-reduction|strided-input-widening-dot-reduction|computed-mask-widening-dot-reduction|computed-mask-strided-input-widening-dot-reduction|widening-dot-reduce-add|strided-input-widening-dot-reduce-add|computed-masked-widening-dot-reduce-add|computed-masked-strided-input-widening-dot-reduce-add|widening-macc-add' .`
  - Passed: 28 selected tests passed, 436 excluded.
- `rtk git diff --check`
  - Passed.

## Legacy Pattern Scan

Pattern:

```text
__riscv_.*_i32m1|__riscv_.*_i32m2|vint32m[12]_t|vbool32_t|RVVI32M1|rvv-i32m1|tcrv_rvv\.i32_|!tcrv_rvv\.i32m
```

Commands and classification:

- `rtk rg -n <pattern> lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h || true`
  - No hits. The contraction owner, neighboring MAcc owner, and route-planning
    header no longer carry these exact legacy spellings as owner-local
    authority.
- `rtk git diff -U0 -- include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp test/Plugin/RVVExtensionPluginTest.cpp | rtk rg -n '^\+.*(<pattern>)' || true`
  - No added-line hits. Legacy exact spellings in this diff are removed lines
    or pre-existing test/spec text, not newly introduced authority.
- `rtk rg -c <pattern> lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp test/Plugin/RVVExtensionPluginTest.cpp .trellis/spec/extension-plugins/rvv-plugin.md .trellis/spec/lowering-runtime/emitc-route.md .trellis/spec/testing/mlir-testing-contract.md || true`
  - Remaining counts:
    - `.trellis/spec/lowering-runtime/emitc-route.md`: 7, all spec text.
    - `.trellis/spec/extension-plugins/rvv-plugin.md`: 23, all spec text
      documenting forbidden legacy authority.
    - `.trellis/spec/testing/mlir-testing-contract.md`: 6, all spec text.
    - `test/Plugin/RVVExtensionPluginTest.cpp`: 251, derived output checks,
      stale negative tests, and one legacy parseable/fail-closed test.
    - `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`: 88, pre-existing
      non-contraction route-family Stage 1 debt outside this contraction
      owner round.

## Spec Update Judgment

No `.trellis/spec/` update was made. This round implements existing spec
contracts: contraction route authority must be plugin-owned and derived from
selected typed RVV body/config facts, while common EmitC route verification
must stay neutral. No new cross-layer contract or durable convention was
introduced beyond the already documented RVV Stage 1 rules.

## Completion State

The bounded contraction owner task is complete. Remaining exact i32m1/i32m2
hits in shared route planning belong to unrelated non-contraction Stage 1
owners and are not a continuation point for this task.
