# Evidence Summary

Task: Stage1 typed RVV base-memory and elementwise intrinsic derivation

## Implementation Evidence

Changed files:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp`
* `test/Plugin/RVVExtensionPluginTest.cpp`
* `.trellis/tasks/06-01-06-01-stage1-typed-rvv-base-memory-elementwise-intrinsic-derivation/`

Behavior completed:

* `RVVSelectedBodyTypedConfigFacts` now carries base-memory memory leaf facts for index load/scale, indexed load/store, strided load/store, masked load, and masked store.
* Base-memory movement route-family derivation copies a typed config snapshot into the family plan and derives route-specific memory leaves from that snapshot plus operation/memory-form facts.
* Base-memory movement validation no longer compares active vector/index/mask C types or memory leaf intrinsics to owner-local exact `i32m1` constants.
* Base-memory provider verification now checks the family plan's typed config snapshot against the selected `RVVSelectedBodyTypedConfigFacts` before route materialization.
* Scalar-broadcast elementwise validation no longer uses m1-only expected helper functions for vector type/C type, setvl, vector load, or store; provider verification ties those fields to `RVVSelectedBodyTypedConfigFacts`.
* Existing add/sub/mul arithmetic and scalar-splat leaves remain derived from operation kind plus typed SEW/LMUL facts.

## Focused Tests

Passed:

```bash
rtk git diff --check
rtk cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test -j2
rtk artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test
```

Final smoke test output:

```text
RVV extension plugin smoke test passed
```

Self-repair performed:

* Added a focused unsupported typed-config negative check by mutating the typed base-memory facts before plan derivation. This covers the base-memory fail-closed diagnostic without relying on route-profile rejection earlier in analysis.
* Re-ran build and the RVV extension plugin smoke test after the negative-test adjustment.

No `ssh rvv` run was performed because this task makes no runtime correctness or performance claim.

## Bounded Ref Scan

Scan command:

```bash
rtk rg -n "__riscv_.*_i32m1|RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m" include lib test .trellis/spec
```

Touched-owner scan:

```bash
rtk rg -n "__riscv_.*_i32m1|RVVI32M1|rvv-i32m1|tcrv_rvv\\.i32_|!tcrv_rvv\\.i32m" include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp test/Plugin/RVVExtensionPluginTest.cpp
```

Classification:

* `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`: no remaining `__riscv_.*_i32m1`, `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, or `!tcrv_rvv.i32m` hits. This touched owner no longer carries exact `i32m1` route authority constants.
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`: no old authority spelling hits in the touched planning header.
* `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp`: remaining exact spellings are operation/config-to-intrinsic mapping outputs for arithmetic, compare/select, strided, and scalar-splat leaves. The scalar-broadcast vector type/C type, setvl, vector-load, and store verification no longer uses m1-only expected helpers.
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`: remaining exact spellings are centralized route-profile/config mapping outputs and unrelated families such as computed-mask memory, segment2, widening, reduction, and macc. Base-memory owner derivation consumes the typed facts exported from this layer rather than owner-local constants.
* `test/Plugin/RVVExtensionPluginTest.cpp`: remaining exact spellings are expected derived emitted callees, stale-mirror negative mutations, fail-closed legacy checks, or tests for unrelated route families.
* `.trellis/spec/`: hits are architecture guardrails and negative examples describing forbidden legacy authority.
* `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`, `lib/Dialect/RVV/IR/RVVDialect.cpp`, `test/Dialect/RVV/dataflow.mlir`, and `test/Dialect/RVV/RVVDialectTest.cpp`: hits are deprecated parse/verifier-only legacy `tcrv_rvv.i32_*` / `!tcrv_rvv.i32m*` surfaces and associated negative/parser coverage, not the touched production route-family owner path.
* Several `test/Target/RVV` and `test/Scripts` hits are expected derived output checks, stale negative mutations, fail-closed legacy front-door tests, or unrelated Stage2-family evidence tests.

Continuation point:

* No continuation is required for this bounded owner task.
* Broader Stage1 cleanup still has unrelated remaining exact-spelling debt in common route-profile mapping and other RVV families; that is outside this task's bounded base-memory plus scalar-broadcast proof-point scope.
