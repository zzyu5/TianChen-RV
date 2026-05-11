# rvv-scalar-dispatch-descriptor-exit

## Goal

Remove descriptor authority from the RVV+scalar dispatch composite artifact
export path for the bounded finite binary families, starting with the existing
default i32 add route and preserving the already-supported add/sub/mul i32/i64
family slice where the current implementation naturally shares the code path.
The default dispatch source/header/object route must derive component identity,
callable ABI, selected family, emitted calls, external ABI signature, and bundle
records from the selected RVV callable component plan, selected scalar fallback
component plan, typed extension-family op/common EmitC metadata, and
exec-IR-backed runtime ABI boundary.

Static RVV/scalar family descriptors may remain as finite registry and legacy
route compatibility metadata after the selected component candidates are known.
They must not select the dispatch compute family, callable ABI parameters,
runtime guard, emitted RVV/scalar call targets, component route identity, source
rendering, artifact kind, or bundle external ABI fields for the migrated
default route.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Task start HEAD is `f68d8ee feat(scalar): remove descriptor selected emission authority`.
- Worktree was clean and `.trellis/.current-task` was absent at task start.
- The previous scalar selected-emission descriptor-exit task is archived at
  `.trellis/tasks/archive/2026-05/05-12-scalar-selected-emission-descriptor-exit/`
  and must stay archived.
- The previous RVV selected-emission descriptor-exit task is archived at
  `.trellis/tasks/archive/2026-05/05-12-rvv-selected-emission-descriptor-exit/`.
- RVV and scalar callable candidates already carry typed-source
  `selected_plan_metadata`, common EmitC route provenance, and exec-IR-backed
  `runtime_abi_parameters`.
- `lib/Target/Builtin/RVVScalarDispatch.cpp` still identifies dispatch pairs
  by matching `TargetArtifactCandidate` fields against
  `DispatchBinaryFamilyDescriptor` fields and route manifests. It also derives
  dispatcher function names, component call names, route metadata, external ABI
  group/name, selected config validation, and bundle runtime ABI parameters
  through descriptor-backed `pair.family`.
- `include/TianChenRV/Target/RVVScalarBinaryFamily.h` still exposes
  `RVVScalarBinaryFamilyDescriptor`, `DispatchBinaryFamilyDescriptor`, and
  helper functions that can reconstruct scalar/dispatch route ids, runtime ABI
  names, runtime glue roles, and descriptor metadata from the RVV finite binary
  family descriptor.
- `lib/Target/TargetArtifactExport.cpp` already supports composite exporter
  runtime ABI parameter callbacks and bundle component-group validation. The
  gap is that dispatch composite records still receive route/external ABI
  identity from the dispatch manifest rather than the selected component plans.
- `lib/Plugin/RVV/RVVExtensionPlugin.cpp` and
  `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp` already build supported
  emission plans from typed RVV/scalar bodies and IR-backed callable ABI
  parameter plans after the previous two descriptor-exit tasks.

## Requirements

- Create one coherent production/default dispatch migration for the existing
  bounded RVV+scalar binary route. Do not stop at helper code or test-only
  coverage.
- The default RVV+scalar dispatch source/header/object export path must collect
  a composite dispatch plan from the selected RVV dispatch-case callable
  candidate and selected scalar dispatch-fallback callable candidate.
- The composite dispatch plan must derive callable ABI parameters from the
  exec-IR-backed callable plan over `tcrv.exec.mem_window` and
  `tcrv.exec.runtime_param`, then append exactly one dispatch availability
  guard resolved from typed `runtime_guard` linkage on the selected
  `tcrv.exec.case`.
- The selected RVV/scalar component candidates must be validated against their
  registered direct source route contracts and typed selected-plan metadata
  before dispatch source/header/object emission.
- Component call targets must be derived from selected component candidate
  identity and typed component authority checks, not from descriptor-only route
  reconstruction.
- Dispatch route/component-group/external ABI/bundle records must be derived
  from the selected component candidates and dispatcher ABI plan, with the
  dispatch manifest kept only as finite route registration metadata after
  component selection.
- Descriptor-only dispatch export must fail closed: a module with descriptor
  metadata but without supported typed RVV/scalar component emission plans must
  not produce a supported source/header/object dispatch artifact.
- Stale descriptor or descriptor-mirrored route metadata beside otherwise
  valid component plans must be rejected as a mirror mismatch or ignored as
  non-authoritative mirror data. It must not alter selected calls, ABI,
  route, artifact kind, component group, external ABI name, manifest fields, or
  generated source.
- Rewrite or quarantine tests that still assert descriptor-owned dispatch
  compute/ABI/source authority. Legacy descriptor tests may remain only when
  explicitly metadata-only and disconnected from the default path.
- Keep implementation in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck. Python is
  allowed only for existing tooling/dry-run validation.

## Non-Goals

- No new RVV/scalar family matrix, dtype expansion, benchmark, performance
  claim, runtime scheduler, generic RVV backend, or generic scalar backend.
- No compiler core, dialect, pass, plugin registry, capability model, lowering,
  or emission implementation in Python.
- No compute semantics in `tcrv.exec`.
- No descriptor-to-C export and no new route around extension family ops to
  common EmitC to generated C/C++ artifacts.
- No IME, AME, Sophgo/offload, Template, Toy, or unrelated extension work.
- No helper-only, metadata-only, fixture-only, evidence-only, or broad
  smoke-test closeout.
- No RVV runtime, correctness, throughput, latency, or performance claim unless
  a fresh `ssh rvv` run is actually performed.

## Acceptance Criteria

- [ ] Default RVV+scalar dispatch source export succeeds from selected
  RVV/scalar component emission plans whose bodies are typed family ops lowered
  through the common EmitC route.
- [ ] Dispatch wrapper ABI parameters are derived from the exec-IR callable
  ABI/component plans plus typed runtime guard linkage, not reconstructed from
  descriptor tables or detached dispatch ABI metadata.
- [ ] Dispatch component call targets and selected component roles come from
  selected component candidates/artifacts/plans.
- [ ] Bundle index records for dispatch source/header/object carry
  component group, external ABI name, selected component paths, and ordered
  runtime ABI parameters derived from the selected composite plan.
- [ ] Descriptor-only dispatch export fails closed before source/header/object
  artifact output.
- [ ] Stale or mismatched descriptor metadata beside valid selected component
  plans is rejected or ignored as non-authoritative mirror data; it cannot
  change the selected calls, ABI, route, artifact kind, manifest fields, or
  bundle metadata.
- [ ] Focused C++ target/export tests cover component-plan-driven default
  dispatch success and descriptor-only or stale-descriptor fail-closed cases.
- [ ] Affected dispatch/registry tests no longer preserve descriptor-owned
  dispatch compute semantics unless explicitly marked as legacy metadata-only.
- [ ] Focused lit for RVV+scalar dispatch target export and bundle/e2e dry-run
  routes passes.
- [ ] `git diff --check` and `git diff --cached --check` pass before commit.
- [ ] Trellis task validation passes on the active or archived task path.
- [ ] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  runs if the existing build tree remains usable.

## Technical Notes

- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/unified-riscv-mlir.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Relevant archived PRDs read:
  - `.trellis/tasks/archive/2026-05/05-12-rvv-selected-emission-descriptor-exit/prd.md`
  - `.trellis/tasks/archive/2026-05/05-12-scalar-selected-emission-descriptor-exit/prd.md`
- Main implementation entry points:
  - `include/TianChenRV/Target/RVVScalarDispatch.h`
  - `include/TianChenRV/Target/RVVScalarBinaryFamily.h`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`
  - `test/Target/I32BinaryFamilyRegistryTest.cpp`
  - `test/Target/TargetArtifactExportTest.cpp`
  - `test/Target/RVVScalarDispatch/`
  - `test/Scripts/rvv-scalar-dispatch-e2e.test`
  - `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`

## Task Status Notes

- Planning started from clean worktree at `f68d8ee`.
- Created this task and PRD before source edits.
- Implemented a route-local composite bundle metadata callback in
  `TargetArtifactCompositeExporter` and wired RVV+scalar dispatch bundle
  records through the selected component candidate group before static route
  fallback metadata can fill fields.
- Rewired dispatch composite identity for the default bounded family path so
  dispatch function stem, header guard stem, self-check marker,
  runtime ABI name, component group, and external ABI name are derived from
  the selected RVV and scalar component `selected_binary_family` metadata,
  with finite descriptor metadata retained only as route registration and
  mismatch validation.
- Added C++ coverage proving vmul dispatch bundle metadata comes from selected
  component plans and stale selected component metadata fails closed before it
  can alter bundle identity.
- Focused and full local validation passed. No fresh `ssh rvv` run was
  performed, so this task makes no RVV runtime, correctness, or performance
  claim.
