# RVV+scalar dispatch selected-plan route authority

## Goal

Make the RVV+scalar dispatch production/export path establish its default
bundle route, composite ABI identity, component roles, generated source
metadata, and self-check arithmetic from typed selected RVV/scalar component
plans plus IR-backed runtime ABI before consulting finite descriptor-derived
registration records. Descriptor-shaped family records may remain only as
route registration, compatibility lookup, legacy mirror validation, or
negative-test fixtures after selected-plan authority is established.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Branch is `main`; the worktree was clean at task creation.
- Task start HEAD is `835ad7c feat(rvv): require typed selected variant authority`.
- `.trellis/.current-task` was absent at session start, so this task was
  created as
  `.trellis/tasks/05-12-rvv-scalar-dispatch-selected-plan-route-authority/`.
- The predecessor task
  `.trellis/tasks/archive/2026-05/05-12-rvv-selected-variant-descriptor-attr-exit/`
  completed and must remain archived.
- Specs require the current main route to stay extension family ops -> common
  EmitC route -> intrinsic/runtime C/C++ -> native compiler. Descriptor-driven
  computation is transition debt, not architecture.
- The dispatch composite exporter already has selected-component metadata
  surfaces and route-local bundle metadata callbacks, but live inventory shows
  candidate family discovery in `lib/Target/Builtin/RVVScalarDispatch.cpp`
  still starts from RVV/scalar route registration records before it derives the
  selected component family identity.

## Initial Inventory Classification

- Production/default migration target:
  - `lib/Target/Builtin/RVVScalarDispatch.cpp` uses
    `getRVVCallableCandidateFamily` and `getScalarCallableCandidateFamily` to
    classify selected callable candidates from route/registration metadata
    before `deriveDispatchCompositeIdentityFromSelectedComponents` validates
    typed selected-plan family metadata. This is the main dependency to rewire.
  - `validateRVVCallableCandidateShapeForFamily` and
    `validateScalarCallableCandidateShapeForFamily` compare candidate route id,
    emission kind, runtime ABI, runtime ABI name, runtime glue role, origin,
    role, and artifact kind against route records. These should remain
    fail-closed compatibility checks, but only after selected component family
    authority is known from typed selected plans.
- Route registration / compatibility manifest:
  - `include/TianChenRV/Target/RVVScalarBinaryFamily.h` defines
    `RVVScalarBinaryFamilyDescriptor`, `DispatchBinaryFamilyDescriptor`, and
    scalar descriptor-shaped records. These finite records derive route ids and
    compatibility ABI strings from the RVV family table; comments already state
    they are not selected-family authority.
  - `getRVVScalarDispatchRouteManifest`,
    `lookupRVVScalarDispatchRoute`, and
    `registerRVVScalarDispatchTargetExporters` in
    `lib/Target/Builtin/RVVScalarDispatch.cpp` are route registration surfaces.
    They may remain, but exported bundle metadata must be derived from the
    selected dispatch pair rather than static route fallback metadata.
  - `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` registers enabled
    plugin-owned route bundles. This is registration composition, not compute
    authority.
- Legacy mirror / quarantine:
  - `selected_lowering_descriptor` and `lowering_descriptor` uses in RVV and
    scalar plugin/target code are legacy mirror checks or negative fixtures
    when they appear beside typed source metadata.
  - `validateDispatchLegacyDescriptorMirrorMetadata` validates legacy RVV
    mirror metadata after the selected config contract is available.
- Negative-test / fixture surfaces:
  - `test/Target/TargetArtifactExportTest.cpp`,
    `test/Target/I32BinaryFamilyRegistryTest.cpp`, and
    `test/Plugin/ScalarExtensionPluginTest.cpp` include descriptor vocabulary
    to prove registration shape, stale mirror rejection, or legacy quarantine.
    Tests that imply descriptor authority must be rewritten or renamed.
  - `test/Scripts/rvv-scalar-dispatch-e2e.test` and
    `scripts/rvv_scalar_dispatch_e2e.py` are tooling/evidence surfaces only;
    update them only if C++ manifest/source semantics change.

## Requirements

- Resolve the selected RVV dispatch-case component family from
  `tcrv_rvv.selected_binary_family` selected-plan metadata and the selected
  scalar fallback component family from
  `tcrv_scalar.selected_binary_family` selected-plan metadata before route
  registration records are consulted as compatibility data.
- Reject missing, duplicate, malformed, or disagreeing RVV/scalar selected
  family metadata before dispatch source/header/object/self-check export.
- Reject stale route/registration data that disagrees with the selected
  component family. This includes route id, emission kind, runtime ABI,
  runtime ABI kind/name, runtime glue role, origin, role, and artifact kind.
- Keep dispatch composite ABI identity, component group, external ABI name,
  dispatcher function stem, header guard stem, self-check marker, and generated
  source comments derived from selected component plans plus IR-backed ABI.
- Keep callable ABI and dispatch guard parameters derived from
  `tcrv.exec.mem_window`, `tcrv.exec.runtime_param`, and selected
  `tcrv.exec.dispatch` runtime_guard IR. Detached dispatch ABI metadata must
  remain fail-closed.
- Keep RVV/scalar behavior plugin/target-owned. Do not move family-specific
  semantics into common core passes.
- Keep `tcrv.exec` compute-free and preserve parameter layering:
  hardware facts are target capability data, vector shape/config is
  compile-time variant metadata, runtime guard/count are SSA/control ABI
  values, and descriptor-local element_count remains bounded fixture metadata.
- Python must remain tooling-only.

## Acceptance Criteria

- [x] Focused inventory over the dispatch/export path classifies remaining
      `RVVScalarBinaryFamilyDescriptor`, `DispatchBinaryFamilyDescriptor`,
      `loweringDescriptor`, `lowering_descriptor`, and
      `selected_lowering_descriptor` uses as production migration target,
      compatibility route registration, legacy mirror, or test fixture.
- [x] The default RVV+scalar dispatch pair collection path resolves component
      family authority from typed RVV/scalar selected-plan metadata before
      validating route/registration records.
- [x] Stale route/registration data cannot silently choose selected family,
      runtime ABI identity, component group, source operation, self-check
      arithmetic, or generated dispatch source comments.
- [x] Descriptor fields that remain are documented in code/test expectations as
      route registration or legacy mirrors only.
- [x] Focused C++ tests prove selected component metadata drives dispatch
      route/source/ABI authority and stale registration/metadata mismatch
      fails closed.
- [x] Focused lit/FileCheck or dry-run dispatch validation proves generated
      source/header/object route comments still reflect selected-plan authority.
- [x] `git diff --check` and Trellis task validation pass.
- [x] Focused build/test targets pass; `check-tianchenrv -j2` is run if
      practical after focused checks pass.
- [x] One coherent commit records the completed module, or the task remains
      open with an exact continuation point.

## Non-Goals

- No new RVV/scalar family, dtype, LMUL, operation matrix, benchmark, broad
  evidence matrix, generic RVV backend claim, or MLIR vector/scalable-vector
  route.
- No IME, AME, Sophgo/offload, Template, Toy, or unrelated extension work.
- No descriptor-to-C exporter and no descriptor-driven computation semantics.
- No compute semantics in `tcrv.exec`.
- No Python compiler internals. Python may run existing dry-run/evidence
  tooling only if C++ output semantics changed.
- No helper-only, report-only, metadata-only, smoke-only, wrapper-only, or
  broad-test-only closeout.
- No RVV runtime correctness/performance claim unless a fresh `ssh rvv` run is
  performed through the changed path.

## Minimal Validation

- Build focused changed owners, as available:
  `TianChenRVBuiltinTargetArtifactExporters`, `TianChenRVRVVTarget`,
  `TianChenRVScalarTarget`, `tcrv-translate`, and affected C++ test binaries.
- Run focused C++ tests, especially:
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-i32-binary-family-registry-test`, and
  `tianchenrv-scalar-extension-plugin-test` if touched.
- Run focused lit/FileCheck filters for RVV+scalar dispatch bundle/source
  output and stale mismatch diagnostics.
- Run
  `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run` on an existing bounded
  dispatch fixture, preferably `i32-vmul` or `i32-vadd`, if C++ output changes.
- Run `git diff --check`.
- Run
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-rvv-scalar-dispatch-selected-plan-route-authority`.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if focused checks pass and the build tree is usable.

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/unified-riscv-mlir.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`
- Prior context read:
  - `.trellis/tasks/archive/2026-05/05-12-rvv-selected-variant-descriptor-attr-exit/prd.md`
  - `.trellis/workspace/codex/journal-4.md`
- Initial source focus:
  - `include/TianChenRV/Target/RVVScalarBinaryFamily.h`
  - `include/TianChenRV/Target/RVVScalarDispatch.h`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`
  - `lib/Target/RVV/RVVMicrokernel.cpp`
  - `lib/Target/Scalar/ScalarMicrokernel.cpp`
  - `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `test/Target/I32BinaryFamilyRegistryTest.cpp`
  - `test/Plugin/ScalarExtensionPluginTest.cpp`
  - `test/Scripts/rvv-scalar-dispatch-e2e.test`
  - `scripts/rvv_scalar_dispatch_e2e.py`

## Completion Notes

- Rewired RVV+scalar dispatch candidate family discovery so recognized
  dispatch component routes first read typed RVV/scalar
  `selected_binary_family` selected-plan metadata, then use finite
  registration records only for route id, emission kind, runtime ABI, runtime
  ABI kind/name, runtime glue role, origin, role, and artifact-kind
  consistency checks.
- Kept non-RVV/scalar composite route matching deterministic by using route ids
  only as a relevance filter before the selected-plan authority check in the
  composite matcher.
- Added C++ target artifact export coverage that rejects missing RVV selected
  family metadata before route-registration lookup and rejects stale scalar
  route data after selected-plan family authority is established.
- Updated the RVV+scalar i32-vsub i32m2 lit negative expectation to the new
  earlier stale route diagnostic produced by selected-plan-first validation.
- Python remained tooling-only. `tcrv.exec` stayed compute-free. Extension
  details stayed in target/plugin-owned RVV/scalar dispatch code. No `ssh rvv`
  runtime, correctness, or performance claim was made.

## Checks Run

- `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVBuiltinTargetArtifactExporters tianchenrv-target-artifact-export-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
- `cmake --build artifacts/tmp/tianchenrv-build --target TianChenRVRVVTarget TianChenRVScalarTarget tcrv-translate tianchenrv-i32-binary-family-registry-test tianchenrv-scalar-extension-plugin-test -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-i32-binary-family-registry-test`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-scalar-extension-plugin-test`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'RVVScalarDispatch|rvv-scalar-dispatch-e2e|rvv-scalar-dispatch-bundle-e2e'`
- `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'target-artifact-bundle-guards'`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --arithmetic-family=i32-vmul --lower-linalg-frontend --run-id codex-selected-plan-route-authority-vmul --overwrite --input test/Target/RVVScalarDispatch/rvv-scalar-i32-vmul-dispatch-generic-route.mlir`
- `git diff --check`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-rvv-scalar-dispatch-selected-plan-route-authority`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`

## Definition Of Done

This task is complete only when the default RVV+scalar dispatch source/header/
object path establishes typed selected RVV/scalar component-plan authority
before descriptor-derived route mirrors, stale descriptor/register mismatches
fail closed, focused tests prove the ordering and diagnostics, Trellis state is
truthful, and one coherent commit records the module. If unfinished, leave the
task open and record the exact continuation point.
