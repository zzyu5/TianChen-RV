# Stage1 typed RVV base-memory and elementwise intrinsic derivation

## Goal

Replace the remaining bounded `i32m1`-as-route-authority in the RVV base-memory movement route-family plan owner, and close one connected elementwise proof point, so currently supported load/store and add/sub/mul-style paths derive concrete RVV C types and intrinsic spellings from selected typed `tcrv_rvv` body/config facts rather than from owner-local exact `i32m1` constants.

## What I Already Know

* The Hermes brief is the source of this task.
* The prior completed task genericized runtime scalar splat-store from `RuntimeI32SplatStore` / `runtime_i32_splat_store` into `runtime_scalar_splat_store` typed route-family facts.
* Stage 1 is still open while active route owners treat exact `__riscv_*_i32m1` spelling, `RVVI32M1*`, `rvv-i32m1`, finite `tcrv_rvv.i32_*`, or `!tcrv_rvv.i32m*` as production route authority.
* `tcrv_rvv` already has generic typed vector/config/load/store/binary surfaces, and `RVVSelectedBodyTypedConfigFacts` carries element type, SEW, LMUL, policy, vector/mask/index C types, setvl, load, and store leaves.
* `RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp` still validates base-memory vector/mask/index types and memory intrinsics against owner-local exact `e32m1` constants.
* `RVVEmitCElementwiseRouteFamilyPlanOwners.cpp` already derives ordinary arithmetic leaves from typed facts, but scalar-broadcast validation still uses m1-only expected helper functions rather than the typed config snapshot as the verification anchor.

## Requirements

* Keep the production path as selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body/config -> RVV plugin family plan -> provider-built `TCRVEmitCLowerableRoute` -> common EmitC.
* For base-memory route-family plans, derive vector type/C type, index type/C type, mask type/C type, setvl, unit load/store, strided load/store, indexed load/store, index load/scale, and static masked load/store leaves from selected typed config/body facts and operation-specific memory-form facts.
* Keep current supported base-memory instances bounded, but accept `i32m1` outputs only as typed config-derived results, not owner constants or route names.
* For unsupported element type, SEW, LMUL, policy, or missing typed config facts, fail closed with diagnostics naming the base-memory or elementwise route-family boundary.
* Convert one connected elementwise proof point so validation compares scalar-broadcast elementwise vector/setvl/load/store facts to the typed config snapshot rather than m1-only helper expectations.
* Do not add new RVV operation coverage, reductions, contractions, compare/select expansion, source-front-door routes, dtype/LMUL clone batches, compatibility wrappers, common EmitC semantic branches, broad smoke matrices, or docs-only/status-only work.

## Acceptance Criteria

* [x] Base-memory plan validation no longer treats owner-local exact `__riscv_*_i32m1`, `vint32m1_t`, `vuint32m1_t`, `vbool32_t`, or `!tcrv_rvv.vector<i32, "m1">` constants as the route authority for active supported paths.
* [x] Base-memory plan derivation requires typed config facts and derives any current `i32m1` load/store/index/mask intrinsic spellings as validated outputs from those facts plus memory-form facts.
* [x] Unsupported typed config combinations fail closed before provider route construction with targeted base-memory diagnostics.
* [x] Scalar-broadcast elementwise validation is tied to `RVVSelectedBodyTypedConfigFacts` for vector/setvl/load/store facts, while arithmetic/splat leaves remain derived from operation kind and typed config.
* [x] Focused C++ or lit tests cover positive typed config-derived base-memory and elementwise planning for existing supported cases plus at least one missing/unsupported typed-config failure.
* [x] A bounded scan over `include`, `lib`, `test`, and `.trellis/spec` classifies remaining `__riscv_.*_i32m1`, `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`, and `!tcrv_rvv.i32m` hits as derived output, parse-only/deprecated, negative/spec text, or unrelated remaining Stage 1 debt rather than active authority in the touched owner path.
* [x] Focused build/test commands for touched RVV plugin paths pass.

## Definition of Done

* Source changes are limited to RVV plugin-owned planning/provider tests and directly required Trellis task files.
* `git diff --check` passes.
* Focused RVV plugin test target passes.
* Trellis task status, notes, evidence, and journal truthfully record what changed and the exact continuation point if unfinished.
* One coherent commit is created when the task is complete.

## Technical Approach

Introduce a small base-memory owner-local typed leaf derivation layer that consumes `RVVSelectedBodyTypedConfigFacts` and selected body operation/memory-form facts. The owner may still reject unsupported shapes, but it should reject them as unsupported typed facts instead of comparing against a hardcoded `i32m1` route template. Then adjust scalar-broadcast elementwise validation to require plan fields to mirror `typedConfigFacts` for vector type/C type, setvl, vector load, and store.

## Decision (ADR-lite)

**Context**: The brief allows finishing the base-memory derivation submodule plus one connected elementwise proof point if the full owner is too large.

**Decision**: This round will complete base-memory derivation and use scalar-broadcast elementwise as the proof point. Ordinary elementwise add/sub/mul already derives most relevant facts from typed config, so the scalar-broadcast path is the smallest useful closure.

**Consequences**: Stage 1 moves the active base-memory owner away from exact `i32m1` constants without pretending that broad dtype/LMUL coverage is complete. Any broader typed memory forms or elementwise expansion remain future Stage 1/Stage 2 work.

## Out of Scope

* New operation families or coverage expansion beyond already supported base-memory and elementwise routes.
* Source-front-door/source-artifact positive RVV routes.
* Descriptor-driven C/source export.
* Common EmitC/export semantic branching.
* Compatibility wrappers preserving old i32 route authority.
* Runtime correctness or performance claims requiring `ssh rvv`.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/rvv-plugin.md`, especially Stage 1 route-authority reset, route materialization facts, elementwise statement-plan, and base-memory statement-plan boundaries.
* Read prior task: `.trellis/tasks/archive/2026-06/06-01-stage1-generic-typed-rvv-runtime-scalar-splat-store-replacement/prd.md` and `evidence-summary.md`.
* Read code: `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`, `lib/Dialect/RVV/IR/RVVConfigContract.cpp`, `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`, `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, `lib/Plugin/RVV/EmitC/RVVEmitCBaseMemoryRouteFamilyPlanOwners.cpp`, and `lib/Plugin/RVV/EmitC/RVVEmitCElementwiseRouteFamilyPlanOwners.cpp`.
* Repository state at task creation: `main`, clean worktree, head `efb1a52d rvv: genericize runtime scalar splat-store route`.
