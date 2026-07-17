# Evidence Summary

Task: Stage1 typed RVV elementwise intrinsic derivation

## Implementation Evidence

Changed files:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp`
* `test/Plugin/RVVExtensionPluginTest.cpp`
* `test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-{add,sub,mul}.mlir`
* `test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-{add,sub,mul}.mlir`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/tasks/06-01-stage1-typed-rvv-elementwise-intrinsic-derivation/`

Behavior completed:

* `RVVSelectedBodyTypedConfigFacts` now carries a provider-derived
  `scalarSplatIntrinsic` leaf from the selected typed RVV config profile.
* Scalar-broadcast elementwise planning consumes the typed config scalar-splat
  leaf instead of selecting a local `i32m1` splat spelling.
* Strided elementwise planning consumes typed config strided load/store leaves.
* Elementwise arithmetic, compare, and masked-merge leaves are composed from
  operation kind plus typed SEW/LMUL mirrors and interned as derived leaf output,
  so the elementwise owner no longer contains complete `__riscv_*_i32m1/i32m2`
  route-authority strings.
* Provider verification now rejects stale scalar-broadcast and strided
  elementwise memory/broadcast leaves when they disagree with
  `RVVSelectedBodyTypedConfigFacts`.
* Scalar-broadcast elementwise target profile and C type mirror strings now use
  typed wording instead of `e32m1` wording.
* RVV plugin spec records the typed scalar-splat/strided leaf contract for
  elementwise route-family planning.

Self-repair performed:

* Initial C++ smoke run exposed that elementwise owner must still populate the
  compute intrinsic plan leaf. Fixed by interning the composed typed leaf string
  instead of reading an initially empty description field.
* Initial `check-tianchenrv` run exposed a stale scalar-broadcast lit mutation
  still targeting `rhs_scalar:i32`. Updated it to mutate the new
  `rhs_scalar:typed-scalar` mirror.

## Checks

Passed:

```bash
git diff --check
cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2
artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test
cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2
```

Final lit result:

```text
Total Discovered Tests: 464
Passed: 464 (100.00%)
```

No `ssh rvv` run was performed because this task makes no runtime correctness
or performance claim.

## Bounded Ref Scan

Scan command:

```bash
rg -n "__riscv_.*_i32m1|__riscv_.*_i32m2|RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m" \
  include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h \
  lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp \
  lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp \
  lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp \
  test/Plugin/RVVExtensionPluginTest.cpp \
  test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-add.mlir \
  test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-sub.mlir \
  test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-mul.mlir \
  test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-add.mlir \
  test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-sub.mlir \
  test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-mul.mlir \
  .trellis/spec/extension-plugins/rvv-plugin.md .trellis/spec/index.md
```

Classification:

* `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp`: no
  remaining hits for complete `__riscv_*_i32m1/i32m2`, `RVVI32M1`,
  `rvv-i32m1`, `tcrv_rvv.i32_`, or `!tcrv_rvv.i32m` patterns. Elementwise
  arithmetic/compare/select/splat leaf names are composed from typed facts.
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`: no old authority
  spelling hits in the touched planning header.
* `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`: no old
  authority spelling hits; included as the prior typed derivation pattern.
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`: remaining hits are central
  route-profile/config mapping outputs and unrelated route families such as
  runtime-scalar computed-mask memory, segment2, widening, reductions, MAcc,
  and explicit legacy fail-closed detection. The elementwise owner no longer
  consumes those exact spellings as local authority.
* `test/Plugin/RVVExtensionPluginTest.cpp`: remaining hits are expected derived
  output checks, stale mirror mutations, fail-closed legacy tests, or tests for
  unrelated RVV route families.
* `test/Target/RVV/*scalar-broadcast-{add,sub,mul}.mlir`: remaining exact RVV
  intrinsic strings are expected generated output from the provider-built route,
  not route authority. Scalar-broadcast profile/type mirrors use typed wording.
* `.trellis/spec/`: hits are Stage 1 guardrails and negative examples.

Continuation point:

* No continuation is required for this bounded elementwise owner task.
* Broader Stage 1 exact-spelling debt remains in central route-profile mapping
  and other RVV route families; that is outside this task's elementwise owner
  scope.
