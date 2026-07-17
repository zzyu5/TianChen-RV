# Evidence Summary

Date: 2026-06-01

## Completed Behavior

* Replaced shared selected-body leaf spelling tables in
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` with typed derivation
  helpers for element names, vector/mask/index type names, C types, pointer
  types, element byte size, setvl, load/store, strided/indexed memory leaves,
  scalar splat, compare/select/mask-and, masked memory, segment tuple/load/store,
  reduction, and widening conversion leaves.
* Rewired `deriveRVVSelectedBodyConfigProfile(...)` so supported config
  profiles keep the same current SEW/LMUL/policy support scope but materialize
  all concrete RVV C/intrinsic spellings from typed facts.
* Rewired masked/computed-mask memory, segment2 memory, computed-mask select,
  standalone reduction, and widening conversion validation/description paths so
  concrete i32 m1/m2 spellings are checked as derived mirrors rather than used
  as owner-local route authority.
* Preserved fail-closed behavior for unsupported SEW, LMUL, policy, indexed
  support, segment tuple form, runtime-scalar ABI form, stale leaves, and
  legacy `tcrv_rvv.i32_*` selected-body ops.

## Checks Run

```bash
rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2
rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv --filter='computed-mask|segment2|runtime-scalar|masked' .
rtk git diff --check
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-01-stage1-typed-rvv-shared-route-profile-leaf-derivation
```

Results:

* Build passed.
* RVV extension plugin smoke test passed.
* Focused lit filter passed: 178 passed, 286 excluded, 464 discovered.
* `git diff --check` passed.
* Trellis task context validation passed.

## Bounded Legacy-String Scan

Command:

```bash
rtk rg -n "__riscv_.*_i32m1|__riscv_.*_i32m2|vint32m[12]_t|vbool32_t|RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m" \
  lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp \
  include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h \
  lib/Plugin/RVV/EmitC/*RouteFamilyPlanOwners.cpp \
  .trellis/spec/extension-plugins/rvv-plugin.md \
  .trellis/tasks/06-01-stage1-typed-rvv-shared-route-profile-leaf-derivation/prd.md
```

Classification:

* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp:12657` remains only as a
  legacy selected-body `tcrv_rvv.i32_*` fail-closed detector.
* `.trellis/spec/extension-plugins/rvv-plugin.md` hits are Stage 1 guardrail
  spec text describing forbidden legacy authority.
* Task PRD hits are acceptance/specification text.
* Relevant C++ test hits are expected derived output assertions, stale-leaf
  negative test inputs, or parseable/deprecated legacy negative fixtures; they
  are not production route authority.

## Continuation Point

No exact continuation point remains for this bounded shared route-profile leaf
derivation task. Later work should continue Stage 1 only if a fresh scan finds
new active production/default RVV route authority outside typed derivation.
