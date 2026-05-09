# Plugin i32 binary family descriptor consumption

## Goal

Migrate the RVV and scalar extension plugin proposal/materialization consumers
for the bounded i32 add/sub microkernel families to consume the C++
`I32BinaryFamilyRegistry` for shared stable family facts. This makes the
descriptor registry useful across the plugin boundary while preserving
plugin-local wording, legality, capability, policy, fallback, and
materialization semantics.

## Requirements

- Consume `include/TianChenRV/Target/I32BinaryFamilyRegistry.h` from both:
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
- Replace duplicated plugin-local add/sub constants or local family table fields
  when the field is already represented by the registry, including frontend
  lowering identity, finite lowering descriptor value, microkernel op identity,
  emission kind, route id, runtime ABI id/kind/name, and runtime glue role.
- Keep plugin-local:
  - supported-message/explanation text;
  - RVV capability-property validation and required-march construction;
  - RVV typed policy metadata and capacity-derived element-count decision;
  - scalar conservative fallback role and fallback policy;
  - plugin-local legality diagnostics and materialization decisions;
  - smoke-probe handling and metadata-only fallback wording.
- Preserve externally visible behavior for existing add/sub plugin proposal,
  selected-boundary materialization, emission readiness, emission plans, route
  ids, runtime ABI metadata, generated IR, required capabilities, RVV policy
  metadata, and scalar fallback metadata.
- Preserve stale family mismatch rejection. A selected vsub descriptor must not
  accept vadd microkernel, route, ABI, or lowering metadata, and vice versa.
- Keep the registry bounded to exactly `i32-vadd` and `i32-vsub`; this task must
  not add a new arithmetic family.

## Acceptance Criteria

- [x] RVV plugin proposal still emits bounded `i32-vadd` by default and
      `i32-vsub` when `tcrv_frontend_lowering = "i32-vsub"` is present.
- [x] RVV plugin materialization/readiness/plan logic consumes registry-backed
      family facts for add/sub descriptor identity, op identity, route, emission
      kind, and runtime ABI metadata.
- [x] Scalar plugin proposal still emits bounded `i32-vadd` by default and
      `i32-vsub` when `tcrv_frontend_lowering = "i32-vsub"` is present.
- [x] Scalar plugin materialization/readiness/plan logic consumes
      registry-backed family facts for add/sub descriptor identity, op identity,
      route, emission kind, and runtime ABI metadata.
- [x] Plugin tests prove both families remain available and distinct and stale
      descriptor/microkernel mismatches fail closed.
- [x] Registry tests still prove the registry covers exactly `i32-vadd` and
      `i32-vsub`.
- [x] Focused dispatch/export tests touched by this migration remain
      descriptor-equivalent.
- [x] `git diff --check` passes.
- [x] CMake configure with repository LLVM/MLIR paths passes.
- [x] Focused plugin, registry, dispatch/export, and e2e checks pass.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
      passes before the task is archived.
- [x] Trellis task validates, finishes/archives, journal is updated, and one
      coherent commit is created if the module is complete.

## Definition of Done

- Both RVV and scalar plugin consumers use the descriptor registry for shared
  add/sub family facts.
- Plugin-specific behavior remains in the owning plugin code and is not moved
  into the target registry as generic semantics.
- No generated IR, route id, runtime ABI name, arithmetic semantics, or
  externally visible proposal name changes intentionally.
- No new `ssh rvv` runtime correctness claim is made for this refactor. Local
  compiler/export checks are sufficient unless the implementation changes
  runtime behavior.

## Technical Approach

Use the existing bounded registry as the source of stable add/sub family facts,
and keep small plugin-local specs as adapters that point at registry
descriptors plus plugin-owned explanation text. The registry provides identity
and route/ABI facts; RVV and scalar plugins continue to own validation,
diagnostics, selected-boundary materialization, proposal policy, capability
requirements, and fallback semantics.

## Decision (ADR-lite)

Context: The previous registry module migrated target/export consumers but left
plugin proposal and materialization family structs local because they also held
plugin-specific wording and semantics.

Decision: Migrate only the stable family facts represented by the registry into
plugin consumption. Preserve plugin-local text and legality/materialization
control as plugin code.

Consequences: The add/sub family identity is now shared across target/export and
plugin consumers without turning the registry into a semantic interpreter for
RVV or scalar fallback behavior.

## Out of Scope

- No arithmetic family beyond existing finite `i32-vadd` and `i32-vsub`.
- No i64/e64, masks, new RVV policy family, dynamic-shape frontend expansion,
  StableHLO/TOSA lowering, generic RVV lowering, or performance tuning.
- No new `tcrv.exec` compute ops.
- No extension-specific semantic branch in generic core passes.
- No Python implementation of compiler registry, lowering, emission, route
  selection, runtime ABI decisions, or source generation.
- No docs-only, helper-only, smoke-only, report-only, or metadata-only closeout.
- No remote RVV execution unless this refactor unexpectedly changes runtime
  correctness behavior.

## Technical Notes

- Relevant specs:
  - `.trellis/spec/plugin-protocol/extension-plugin-integration.md`
  - `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  - `.trellis/spec/plugin-protocol/locality-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Prior module:
  - `.trellis/tasks/archive/2026-05/05-09-rvv-i32-binary-family-descriptor-registry/prd.md`
- Primary source surfaces:
  - `include/TianChenRV/Target/I32BinaryFamilyRegistry.h`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `lib/Plugin/LoweringBoundary.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVExtensionPlugin.h`
  - `include/TianChenRV/Plugin/Scalar/ScalarExtensionPlugin.h`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `test/Plugin/`
  - `test/Target/I32BinaryFamilyRegistry/`
  - `test/Target/RVVScalarDispatch/`

## Continuation Rule If Unfinished

Keep this task open. Record which plugin consumer was migrated, which duplicated
family fields remain, which fields intentionally stayed local, which tests
passed, and the next plugin consumer or field boundary to migrate. Do not
archive or claim the plugin path is descriptor-backed until both RVV and scalar
plugin consumers use the registry for shared family facts.

## Completion Notes

- Migrated `RVVExtensionPlugin.cpp` to consume
  `I32BinaryFamilyRegistry` for RVV i32 add/sub frontend lowering lookup,
  lowering descriptor identity, microkernel op identity, arithmetic op identity,
  emission kind, route id, runtime ABI id/kind/name, and runtime glue role.
- Migrated `ScalarExtensionPlugin.cpp` to consume
  `I32BinaryFamilyRegistry` for scalar i32 add/sub frontend lowering lookup,
  lowering descriptor identity, microkernel op identity, emission kind, route
  id, runtime ABI id/kind/name, and runtime glue role.
- Kept plugin-local wording and semantics local: RVV/scalar supported messages,
  descriptor diagnostic nouns, RVV capability/property validation, RVV policy
  and capacity-derived element-count decision, scalar conservative fallback
  metadata, selected-boundary decisions, and smoke-probe/metadata-only wording.
- Generated IR, proposal names, route ids, runtime ABI names, required
  capability metadata, RVV policy metadata, scalar fallback role, and add/sub
  arithmetic semantics were not intentionally changed.
- No new runtime correctness claim was made; no `ssh rvv` evidence was run
  because this is a local plugin/registry refactor.

## Validation

- `git diff --check`
- `cmake -S . -B artifacts/tmp/tianchenrv-build -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir -DLLVM_EXTERNAL_LIT=/usr/lib/llvm-20/build/utils/lit/lit.py -DTIANCHENRV_LLVM_LIT=/usr/lib/llvm-20/build/utils/lit/lit.py`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-i32-binary-family-registry-test tianchenrv-rvv-extension-plugin-test tianchenrv-scalar-extension-plugin-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-scalar-extension-plugin-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-extension-plugin|scalar-extension-plugin|I32BinaryFamilyRegistry|rvv-scalar-i32-vadd-dispatch-generic-route|rvv-scalar-i32-vsub-dispatch-generic-route|plan-linalg-i32-vadd-and-export-target-artifact-bundle|plan-linalg-i32-vsub-and-export-target-artifact-bundle'`
  from `artifacts/tmp/tianchenrv-build/test`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
