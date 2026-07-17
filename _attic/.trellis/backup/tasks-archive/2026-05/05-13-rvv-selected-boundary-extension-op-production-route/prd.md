# RVV selected-boundary extension-op production route

## Goal

Migrate one bounded RVV microkernel production slice, centered on the existing
i32 vector-add default path, so the selected production artifact route is
explicitly owned by `tcrv_rvv` extension-family ops and plugin-owned
selected-boundary state rather than descriptor-owned computation. The route
must remain bounded: selected RVV capability/config metadata, materialized
`tcrv_rvv.lowering_boundary`, a typed `tcrv_rvv.i32_vadd_microkernel` body,
RVV plugin emission planning, and RVV target source/header/object preflight for
that one family/config slice.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Initial state for this round: worktree clean, HEAD
  `f673da0 feat(plugin): share construction protocol artifact route`.
* No `.trellis/.current-task` existed, so this task was created from the Hermes
  Direction Brief.
* The previous completed task moved construction-protocol validation and
  generated artifact support into a shared C++ route consumed by Template, Toy,
  and TensorExtLite; that route is reusable evidence of the plugin-local,
  common-interface pattern only, not permission to force RVV into Template/Toy
  semantics.
* Current RVV code already has a descriptorless default i32-vadd proposal path,
  RVV selected lowering-boundary materialization, automatic typed
  `tcrv_rvv.i32_vadd_microkernel` materialization, target source/header/object
  exporters, and descriptor-only quarantine tests.
* Session notes show prior RVV op-owned EmitC artifact work rewired intrinsic
  callee selection to typed RVV source-op provenance and collected bounded
  `ssh rvv` evidence for dynamic i32-vsub. This round should not redo that
  work; it should tighten the selected-boundary production route.

## Module Goal

For the bounded default i32-vadd RVV production route, make the selected
boundary and target preflight explicitly carry and validate plugin-owned binary
source identity derived from typed `tcrv_rvv` family ops:

* selected source kind;
* selected binary dtype/family/operator;
* executable microkernel op name;
* EmitC source op and generated interface identity.

The generated artifact path must continue to consume the typed microkernel body
and IR-backed callable ABI, while descriptor metadata remains only an optional
legacy mirror after typed body authority has already been established.

## Boundaries

* Scope is one coherent RVV family/config slice: default i32-vadd, finite
  i32m1/i32m2 selected config as already supported by the capability/profile
  path, and the existing runtime-callable C source/header/object artifact
  routes.
* Implementation stays in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck.
* Python is allowed only for Trellis validation, probes, runners, and artifact
  tooling; it must not implement compiler decisions.
* `tcrv.exec` and generic transforms must not gain RVV semantic branches.
* RVV-specific validation and selected-route identity belong in RVV plugin,
  RVV dialect, or RVV target/export code.
* Descriptor-only RVV computation must remain fail-closed or explicitly
  quarantined from the default production route.
* Do not expand the main result to new RVV families, dtype coverage, LMUL
  coverage, performance work, or Template/Toy/TensorExtLite changes.

## Requirements

* Preserve the current default descriptorless RVV proposal behavior for i32
  vector add.
* Ensure selected-boundary materialization emits or validates concrete
  `tcrv_rvv` selected source identity for the bounded typed microkernel path.
* Ensure RVV plugin planning carries selected vector config plus typed binary
  source identity through supported emission-plan metadata.
* Ensure RVV target source/header/object artifact preflight validates that the
  selected plan identity matches the actual typed microkernel body before
  artifact bytes are emitted.
* Keep generated source on the clang/LLVM-compatible
  `riscv_vector.h` intrinsic route via the existing EmitC lowerable route.
* Keep the runtime-callable ABI backed by `tcrv.exec.mem_window` and
  `tcrv.exec.runtime_param`.
* Preserve fail-closed behavior for descriptor-only, stale descriptor, missing
  selected boundary, stale boundary, wrong op, wrong source identity, and wrong
  policy/body cases.
* Keep descriptor mirrors optional and non-authoritative when a matching typed
  body already exists.
* Add or update focused lit/C++ coverage for the selected-boundary-to-RVV-op-
  to-artifact route and the new fail-closed source-identity mismatch case.

## Acceptance Criteria

* [x] The default i32-vadd selected RVV path materializes a
      `tcrv_rvv.lowering_boundary` and a matching typed
      `tcrv_rvv.i32_vadd_microkernel` body without emitting
      `tcrv_rvv.lowering_descriptor`.
* [x] The selected boundary or selected-route metadata records typed binary
      source identity from the RVV plugin, not from descriptor-owned
      computation.
* [x] RVV target artifact export validates selected source identity against
      the actual typed microkernel body before source/header/object output.
* [x] Descriptor-only RVV variants fail closed before production artifact
      output for the migrated slice.
* [x] Stale descriptor mirrors remain rejected when they disagree with the
      typed microkernel body.
* [x] The generated source still includes `riscv_vector.h`, the selected RVV
      intrinsic, and the existing runtime-callable ABI with no hidden `main` in
      the default library artifact.
* [x] Core `tcrv.exec`, generic transforms, and common construction code do
      not gain RVV semantic branches.
* [x] Focused build/test/lit checks for touched RVV dialect/plugin/target
      paths pass.
* [x] `git diff --check` passed; `git diff --cached --check` is run after
      staging.
* [x] Trellis task status, archive state, and git commit are truthful.

## Definition Of Done

* Focused C++/TableGen build targets pass for touched RVV dialect, plugin,
  target, generated headers, `tcrv-opt`, `tcrv-translate`, and affected C++
  tests.
* Focused lit/FileCheck coverage passes for RVV selected-boundary
  materialization, microkernel source export, generic artifact export, and the
  fail-closed source-identity cases touched in this round.
* A bounded reference scan shows descriptor-only compute export is not the
  default production route for the migrated i32-vadd slice.
* `ssh rvv` evidence is run only if generated RVV source/header/object
  semantics change in a way that affects runtime artifacts; if not run, the
  final report states why no RVV runtime/correctness claim is made.
* Work is finished, archived, and committed as one coherent commit if the task
  completes.

## Out Of Scope

* New RVV family, dtype, LMUL, or performance expansion as the main result.
* Replacing the existing RVV op-owned EmitC route already present in the repo.
* Descriptor-to-C production exporters.
* Moving computation semantics into `tcrv.exec`.
* Adding RVV semantic branches in generic transforms or common construction
  code.
* GCC/vendor compiler default routes.
* RVV runtime, correctness, or performance claims without real `ssh rvv`
  evidence for changed emitted artifacts.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/architecture/design-boundaries.md`
* `.trellis/spec/core-dialect/tcrv-exec-contract.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/plugin-protocol/index.md`
* `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/testing/mlir-testing-contract.md`

Prior context read:

* `.trellis/tasks/archive/2026-05/05-13-common-construction-protocol-artifact-route/prd.md`
* `.trellis/workspace/codex/journal-5.md` recent RVV and construction-protocol
  entries.

Initial code evidence:

* `lib/Plugin/RVV/RVVBinaryPlanning.cpp` already rejects descriptor-only
  direct RVV binary planning metadata before typed body authority.
* `lib/Plugin/RVV/RVVBinarySelectedLoweringBoundary.cpp` already auto-
  materializes descriptorless default typed i32-vadd microkernels during
  selected-boundary materialization.
* `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp` already builds
  supported emission plans only when a matching typed RVV microkernel
  attachment is present.
* `lib/Target/RVV/RVVMicrokernel.cpp` already exports through verified RVV
  family ops and the common EmitC lowerable route, with descriptor mirrors
  treated as non-authoritative compatibility metadata.

## Completion Summary

This round kept the scope to the bounded RVV binary selected-boundary route and
made the production path carry explicit typed RVV source identity:

* `tcrv_rvv.lowering_boundary` now has verifier-backed optional selected binary
  source identity fields: source kind, dtype, family, operator, microkernel op,
  EmitC source op, and generated EmitC lowerable interface.
* RVV selected-boundary materialization copies the plugin-selected binary
  source identity from the selected variant/plan into the concrete boundary.
* RVV target artifact preflight validates those boundary fields against the
  actual typed microkernel body before source/header/object emission.
* Candidate source-authority validation is now kernel-local for the selected
  variant/role instead of requiring the enclosing module to contain only one
  RVV microkernel record; direct standalone source/header/object exports keep
  the singleton module constraint.
* RVV/Scalar target CMake dependencies now declare their direct dependency on
  the common `TianChenRVTarget` artifact/export framework.
* Focused lit tests were updated so FileCheck and negative mutators anchor on
  the actual microkernel op/body rather than the new boundary identity attrs.

No `ssh rvv` evidence was run: this round changes selected-boundary metadata,
target preflight, and local compile/lit coverage only. It does not change the
generated RVV intrinsic body/header/object semantics and makes no RVV runtime,
correctness, or performance claim.

## Checks Run

* `cmake --build build --target TianChenRVRVVDialect TianChenRVRVVPlugin TianChenRVRVVTarget TianChenRVScalarTarget tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-selected-lowering-boundary-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate -j2`
* `cmake --build build --target tianchenrv-rvv-binary-variant-legality-test -j2`
* `./build/bin/tianchenrv-rvv-extension-plugin-test`
* `./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
* `./build/bin/tianchenrv-target-artifact-export-test`
* `./build/bin/tianchenrv-rvv-binary-planning-test`
* `./build/bin/tianchenrv-rvv-binary-variant-legality-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv Target/RVVMicrokernel/rvv-microkernel-auto-materialization.mlir Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-runtime-abi-role-binding.mlir Transforms/VariantMaterialization/plugin-variant-materialization-rvv-selected-shape.mlir` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter='rvv-microkernel-auto-materialization|RVVScalarDispatch|RVVSmokeProbe|LoweringBoundary|rvv-'` from `build/test`
* `git diff --check`
* Core neutrality scan: `git diff -- lib/Transforms include/TianChenRV/Dialect/Exec lib/Dialect/Exec`
