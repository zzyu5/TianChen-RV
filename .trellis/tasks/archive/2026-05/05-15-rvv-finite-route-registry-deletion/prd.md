# Descriptor Erasure Owner: target-side RVV finite route registry deletion

## Goal

Delete one coherent Wrong Logic Deletion Campaign slice: remove or fail-close
the target-side RVV finite family/route registry surfaces that still make
descriptor-era records the authority for RVV route IDs, runtime-callable C
artifact kinds, ABI names, intrinsic names, selected-config emission views, and
RVV+scalar dispatch route coupling.

Future RVV rebuild work must use extension-family IR plus common EmitC/artifact
interfaces. This round must not add a replacement registry, compatibility
table, descriptor wrapper, direct C exporter, route quarantine, or new RVV
lowering feature.

## What I Already Know

- HEAD `d609bc0` archived the previous RVV selected microkernel descriptor
  materialization deletion task.
- The worktree was clean before this task started.
- Previous task removed plugin-side automatic selected microkernel/body/ABI
  materialization from `selectedPlan.descriptor` and made selected emission
  planning/body verification fail closed.
- Remaining active residue is target-side:
  - `include/TianChenRV/Target/RVV/RVVBinaryFamily.h` still stores finite
    family records with legacy lowering tokens, route IDs, runtime ABI strings,
    artifact route IDs, and RVV intrinsic prefixes.
  - `include/TianChenRV/Target/RVV/RVVBinaryRoute.h` still wraps these records
    as intrinsic/runtime route descriptors.
  - `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h` still exposes
    selected-config helper APIs that can derive intrinsic names and runtime ABI
    parameters from finite family records.
  - `lib/Target/RVV/RVVMicrokernel.cpp` still owns direct RVV
    source/header/object route manifests and exporter registrations.
  - `lib/Target/RVV/RVVScalarDispatch.cpp` and
    `include/TianChenRV/Target/RVVScalarBinaryFamily.h` still couple RVV and
    scalar candidate validation through finite route descriptors.
  - Target tests such as `test/Target/I32BinaryFamilyRegistryTest.cpp` and
    `test/Target/TargetArtifactExportTest.cpp` still protect old finite route
    identity in places.

## Requirements

- Delete or fail-close target-side RVV finite route/artifact/export authority
  that is derived from family records or intrinsic-route descriptors.
- Remove deleted direct source/object/header/self-check route IDs instead of
  preserving them as compatibility metadata.
- Remove `RVVBinaryIntrinsicRoute` as a public target route/intrinsic/ABI
  descriptor surface.
- Strip `RVVBinaryFamilyRecord` usage down to non-compute source/family facts
  only if a complete removal would make this slice too large; any remaining use
  must not expose route ID, artifact kind, ABI identity, or intrinsic spelling
  authority.
- Selected-config helpers may remain only for bounded compile-time vector
  config/profile facts and typed-source metadata. They must not derive direct
  C routes, callable ABI parameters, or intrinsic-source emission authority
  from descriptor records.
- Dispatch code must stop validating selected callable artifacts against finite
  RVV/scalar route descriptor tables and must fail closed for deleted
  direct-dispatch artifact routes until a real EmitC route exists.
- Tests that protect old finite route registry ownership must be deleted or
  rewritten to absence/fail-closed behavior.
- If deletion exposes a missing new architecture, record the gap and do not
  restore the old route table or wrapper to make tests pass.

## Acceptance Criteria

- [x] Target/include code no longer exposes `RVVBinaryIntrinsicRoute` as route,
      artifact, ABI, or intrinsic authority.
- [x] `RVVBinaryFamilyRecord` no longer carries or publishes direct RVV
      route IDs, header/object route IDs, runtime-callable ABI strings,
      runtime glue role strings, or arithmetic intrinsic prefixes as production
      authority.
- [x] Deleted direct RVV source/header/object/self-check route IDs are absent
      from built-in target artifact exporter registration.
- [x] RVV+scalar dispatch route registration and candidate validation no longer
      depend on finite RVV/scalar route descriptor tables for production truth;
      deleted dispatch routes fail closed or are absent.
- [x] Selected-config helpers that remain are pure selected vector config,
      capability/profile, typed source identity, and runtime-control facts;
      they do not build callable ABI parameter lists or direct intrinsic
      emission views from descriptor-owned routes.
- [x] Target tests are deleted or rewritten so they assert route absence,
      unsupported deleted-route behavior, or pure non-compute config behavior.
- [x] No Common Lower-To-EmitC rebuild, replacement route registry,
      compatibility table, descriptor wrapper, direct C exporter restoration,
      selected microkernel materializer restoration, finite family expansion,
      scalar fallback feature work, ssh RVV campaign, or Python compiler
      semantics is added.
- [x] Bounded ref-scans classify residuals for the requested terms as active
      authority, fail-closed/deleted-route wording, pure config/source identity,
      or next deletion owner.
- [x] Focused affected builds/tests run first, then full
      `cmake --build build --target check-tianchenrv -j2` if coherent.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation
      before finish and after archive, final clean `git status --short`, and
      one coherent commit are produced if the task is complete.

## Non-Goals

- No Common Lower-To-EmitC rebuild.
- No new RVV extension-family lowering.
- No replacement finite route registry, route quarantine, compatibility table,
  descriptor wrapper, or legacy alias.
- No direct C exporter or selected microkernel materializer restoration.
- No finite RVV family expansion.
- No scalar fallback feature work.
- No ssh RVV evidence campaign.
- No Python compiler semantics.
- No broad repo audit beyond bounded owner/ref-scan evidence.

## Minimal Evidence

- Inventory of target-side RVV finite family/route registry definitions, active
  call sites, and protecting tests.
- Bounded scans over include/lib/test/spec for:
  `RVVBinaryFamilyRecord`, `RVVBinaryIntrinsicRoute`,
  `RVVSelectedConfigContract`, `RVVBinaryMicrokernelBodyVerifierRequest`,
  `getRVVBinaryFamilyRegistrationRecords`,
  `lookupRVVBinaryFamilyRegistration`, `getRVVBinaryIntrinsicRoute`,
  `legacyLoweringToken`, `i32-vadd-microkernel.v1`,
  `descriptor-element-count`, `arithmeticIntrinsicPrefix`, `__riscv_`,
  `runtime-callable-c-source`, `tcrv-export-rvv`,
  `rvv-runtime-callable-c-abi`, and RVV/scalar dispatch route IDs.
- Focused C++ build/test targets for affected RVV target/plugin/export tests.
- Affected lit subsets under `test/Target/RVVMicrokernel`,
  `test/Target/RVVScalarDispatch`, `test/Target/ArtifactExport`,
  `test/Target/EmissionManifest`, and RVV plugin tests when touched.

## Technical Notes

- `.trellis/spec/index.md` rejects descriptor-driven computation and requires
  C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck for compiler behavior.
- `.trellis/spec/architecture/design-boundaries.md` lists
  descriptor-driven microkernel/exporter frameworks as non-goals.
- `.trellis/spec/lowering-runtime/emitc-route.md` defines the accepted future
  path as extension family ops -> EmitC -> intrinsic/vendor/runtime C/C++.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` requires
  direct descriptor-to-C source/exporter routes to be deleted or fail-closed
  until a materialized MLIR EmitC module route exists.
- `.trellis/spec/extension-plugins/rvv-plugin.md` describes RVV as an
  extension family inside one TCRV system, not an independent backend dialect.
- `.trellis/spec/testing/mlir-testing-contract.md` requires deleting or
  rewriting tests that only protect old direct-C or descriptor-positive
  behavior.
- Previous archived task:
  `.trellis/tasks/archive/2026-05/05-15-05-15-rvv-selected-microkernel-descriptor-materialization-deletion/prd.md`.

## Completion Evidence

- Deleted public route/registry headers:
  `include/TianChenRV/Target/RVV/RVVBinaryRoute.h` and
  `include/TianChenRV/Target/I32BinaryFamilyRegistry.h`.
- Deleted the protecting registry test:
  `test/Target/I32BinaryFamilyRegistryTest.cpp`, and removed its CMake target.
- Refactored `include/TianChenRV/Target/RVV/RVVBinaryFamily.h` so
  `RVVBinaryFamilyRecord` keeps only finite source/family/config facts. It no
  longer carries route IDs, artifact kinds, runtime ABI identity strings,
  runtime glue roles, component groups, legacy lowering tokens, or RVV
  intrinsic prefixes.
- Refactored `include/TianChenRV/Target/RVV/RVVSelectedConfigContract.h` so
  selected-config helpers are vector shape, selected policy, profile, typed
  source, and runtime-control facts only. Intrinsic derivation, callable ABI
  parameter construction, and arithmetic intrinsic metadata emission were
  removed.
- Removed `RVVBinaryIntrinsicRoute` from active include/lib/test consumers and
  removed the body-verifier request's descriptor-route field.
- Deleted public RVV direct route manifest/query/export authority from
  `RVVMicrokernel.h/.cpp`. The remaining export helpers fail closed with the
  deleted direct-C route diagnostic, and target exporter/translate registration
  is a no-op.
- Deleted public RVV+scalar dispatch route manifest/query/export authority from
  `RVVScalarDispatch.h/.cpp`. The remaining target exporter and translate
  registration hooks are no-ops.
- Refactored `RVVScalarBinaryFamily.h` so dispatch records no longer carry
  route IDs, artifact route IDs, runtime ABI identity strings, or component
  groups. Scalar-family facts that feed the scalar target owner remain outside
  this RVV dispatch-route deletion slice.
- Updated RVV plugin planning so selected emission identity is metadata-only
  unsupported after route deletion; it no longer synthesizes direct
  `emissionPath`, route IDs, artifact kinds, runtime ABI identities, or
  direct-route diagnostics from family records.
- Rewrote target/plugin tests to assert route absence, no-op registration,
  empty selected emission identity, and pure selected-config profile behavior.

## Bounded Ref-Scan Classification

- No active include/lib/test hits remain for
  `RVVBinaryIntrinsicRoute`, `getRVVBinaryIntrinsicRoute`, `RVVBinaryRoute`,
  `legacyLoweringToken`, `arithmeticIntrinsicPrefix`,
  `i32-vadd-microkernel.v1`, `RVVMicrokernelDirectRoute`, or
  `RVVScalarDispatchRoute`.
- `RVVBinaryFamilyRecord`, `getRVVBinaryFamilyRegistrationRecords`, and
  `lookupRVVBinaryFamilyRegistration*` remain in RVV plugin planning,
  legality, selected-config, and focused tests as finite source/family/config
  identity. This is active source identity, not route/artifact/ABI/intrinsic
  authority.
- `runtime-callable-c-source`, `tcrv-export-rvv*`, and
  `rvv-runtime-callable-c-abi` still appear in generic target artifact export,
  emission-manifest, support runtime-ABI, scalar, smoke-probe, conversion, and
  negative fixture tests. In this slice they are either non-RVV-finite-route
  infrastructure, explicit deleted-route/fail-closed expectations, smoke-probe
  route behavior, or next-owner support/scalar residuals.
- `__riscv_` still appears in the RVV smoke probe and common EmitC interface
  tests. Those are not the target-side finite RVV family/route registry deleted
  in this owner.
- `descriptor-element-count` remains only in target artifact export fixtures
  and negative/preflight test material; it is not emitted from the deleted RVV
  family/route registry path.

## Checks Run

- `rtk cmake --build build --target TianChenRVRVVTarget tianchenrv-target-artifact-export-test tianchenrv-runtime-abi-callable-plan-test tianchenrv-rvv-binary-planning-test tianchenrv-rvv-selected-lowering-boundary-test -j2`
- `rtk cmake --build build --target tianchenrv-scalar-extension-plugin-test -j2`
- `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-binary-planning-test tianchenrv-rvv-extension-plugin-test -j2`
- `rtk ./build/bin/tianchenrv-target-artifact-export-test`
- `rtk ./build/bin/tianchenrv-runtime-abi-callable-plan-test`
- `rtk ./build/bin/tianchenrv-rvv-binary-planning-test`
- `rtk ./build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
- `rtk ./build/bin/tianchenrv-scalar-extension-plugin-test`
- `rtk ./build/bin/tianchenrv-rvv-extension-plugin-test`
- `rtk cmake --build build --target check-tianchenrv -j2`
  passed 125/125.
- `rtk git diff --check`
- `rtk git diff --cached --check`
- `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-15-rvv-finite-route-registry-deletion`
- `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-15-rvv-finite-route-registry-deletion`
