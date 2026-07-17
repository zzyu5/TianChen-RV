# Stage1 generic typed RVV runtime-scalar splat-store replacement

## Goal

Replace the finite i32-shaped runtime scalar splat-store route authority with a bounded Stage 1 generic typed RVV route surface. The RVV provider must derive runtime scalar splat-store route facts from selected `tcrv_rvv` body/config/capability/runtime facts, or fail closed with targeted diagnostics.

## What I already know

* The current task source is the Hermes Direction Brief for "Stage1 generic typed RVV runtime-scalar splat-store replacement".
* The previous commit closed the runtime splat-store provider ABI boundary, but the route remains shaped around `RuntimeI32SplatStore` / `runtime_i32_splat_store` authority.
* TianChen-RV Stage 1 requires replacing or fail-closing bounded i32m1-as-route-authority before Stage 2 coverage expansion.
* The target path is selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body/config -> RVV plugin route provider -> common `TCRVEmitCLowerableRoute` -> neutral EmitC materializer.

## Assumptions

* This task should stay bounded to runtime scalar splat-store route planning and its directly coupled tests/spec notes.
* Existing generic typed RVV surface pieces may already exist for some route classes; this task should reuse them if present instead of inventing a parallel model.
* i32 add/store behavior may remain only as an ordinary typed instance, not as route id, op namespace, type namespace, or artifact-authority compatibility.

## Requirements

* Remove or fail-close `RuntimeI32SplatStore` / `runtime_i32_splat_store` as active production/default route authority.
* Introduce or reuse a generic typed runtime scalar splat-store route fact surface that carries element type, SEW, LMUL, policy, memory form, runtime AVL/VL use, operation kind, and scalar source/runtime binding.
* Make the RVV provider derive route/type/header/intrinsic choices from typed body/config/capability/runtime facts.
* Keep RVV semantic decisions out of common EmitC/export code.
* Provide targeted diagnostics for missing typed facts, legacy i32 authority, and unsupported typed combinations.
* Update tests to cover the generic typed route-supported path and fail-closed missing/legacy-fact cases.
* Run a bounded reference scan for legacy authority terms listed in the task brief and remove or explain any active authority residue for this path.

## Acceptance Criteria

* [x] No active runtime scalar splat-store production/default path uses `RuntimeI32SplatStore`, `runtime_i32_splat_store`, `RVVI32M1*`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, exact `__riscv_*_i32m1` spelling, source-front-door patterns, artifact names, descriptor residue, or common/export RVV branches as route authority.
* [x] A selected typed RVV body/config can express runtime scalar splat-store facts and the RVV provider can derive a `TCRVEmitCLowerableRoute` from those facts.
* [x] Missing or legacy route facts fail closed with diagnostics that identify the missing/legacy authority boundary.
* [x] Focused tests cover the positive typed runtime scalar splat-store path and fail-closed cases.
* [x] Relevant RVV plugin build/test commands pass.
* [x] Trellis task status, notes, and journal truthfully describe the implementation and remaining continuation point, if any.

## Completion Notes

* The bounded Stage 1 replacement is complete for runtime scalar splat-store.
* Current executable support remains the typed SEW32/LMUL m1 instance, but it is now routed through `runtime_scalar_splat_store` typed body/config facts rather than an i32-shaped operation mnemonic.
* Evidence is recorded in `evidence-summary.md`.

## Out of Scope

* Stage 2 coverage expansion such as broadcast, compare/select, reduction, matmul, conversion, dtype/LMUL clone batches, or new one-op-per-intrinsic wrappers.
* Source-front-door routes or descriptor-driven C/source export.
* Compatibility wrappers preserving old i32 authority.
* Docs-only, status-only, dashboard, or broad smoke-test work as the main achievement.
* Common EmitC/export RVV semantic branching.

## Technical Notes

* Read first per brief: `.trellis/spec/index.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`, archived prior task `06-01-06-01-stage2-rvv-runtime-scalar-splat-store-provider-abi-preflight`, `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`, `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`, directly relevant RVV dialect/config files, and `test/Plugin/RVVExtensionPluginTest.cpp`.
* Repository state at task creation: `main`, clean worktree, recent head `93a01d16 rvv: close runtime splat-store provider ABI boundary`.
