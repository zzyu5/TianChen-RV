# Target-owned artifact route translation registration

## Goal

Move `tcrv-translate` artifact route-family helper command registration behind a target-support-owned C++ translation route contribution hook. RVV direct binary microkernel source/header/object routes and RVV+scalar dispatch source/header/object/self-check routes must be contributed by target modules through the same generic route registry, without target-family-specific manifest loops or RVV direct/dispatch includes in `tools/tcrv-translate/tcrv-translate.cpp`.

## What I already know

* The repository is `/home/kingdom/phdworks/TianchenRV`, HEAD is `4340aad`, and the worktree was clean at task start.
* No `.trellis/.current-task` existed, so this task was created as `.trellis/tasks/05-10-target-owned-artifact-route-translation-registration`.
* `TargetArtifactExport` already has a target-owned built-in exporter registration boundary via `registerBuiltinTargetArtifactExporters`.
* `tools/tcrv-translate/tcrv-translate.cpp` still owns two RVV-specific route-family translation loops:
  * `registerRVVMicrokernelDirectRouteTranslations()` over `getRVVMicrokernelDirectRouteManifest()`.
  * `registerRVVScalarDispatchManifestTranslations()` over `getRVVScalarDispatchRouteManifest()`.
* Legacy/standalone translate helpers currently outside this module:
  * `tcrv-export-emission-manifest`.
  * `tcrv-export-rvv-smoke-probe-c`.
  * `tcrv-export-rvv-microkernel-self-check-c`.
  * Generic target artifact/source/header/bundle and plan-and-export bundle front doors.
* RVV direct route manifest covers six finite families (`i32/i64` x add/sub/mul) and three direct route kinds per family: source, header, object.
* RVV+scalar dispatch route manifest covers the same six finite families and five route kinds per family: source, header, object, self-check source, self-check object.

## Requirements

* Add the smallest target-neutral C++ abstraction for target modules to contribute translate route registrations.
* The abstraction must carry:
  * route id;
  * description;
  * target-owned export callback;
  * binary stdout requirement;
  * enough shape for `tcrv-translate` to attach its existing dialect-registration hook.
* Add generic duplicate/malformed route diagnostics for empty route ids, empty descriptions, null callbacks, and duplicate route ids.
* Add a built-in target translation route contribution helper that delegates to target-owned RVV direct and RVV+scalar dispatch registration functions.
* Migrate RVV direct binary microkernel source/header/object route-family helper commands to that hook, preserving existing route ids, descriptions, binary stdout behavior, selected-family generic aliases, and family mismatch fail-closed behavior.
* Migrate RVV+scalar dispatch source/header/object/self-check source/self-check object helper commands to the same hook, preserving route ids, descriptions, binary stdout behavior, and family mismatch fail-closed behavior.
* Keep `TargetArtifactExport` behavior stable. This task does not change exporter semantics, runtime ABI contracts, generated source/header/object content, or dispatch compute behavior.
* Keep non-route standalone helpers direct in `tcrv-translate`; do not broaden this task into smoke-probe, emission-manifest, planning, bundle, or runtime evidence work.
* Add focused C++ coverage for the translate route registry and built-in target route contributions, including duplicate/malformed route rejection and representative RVV direct plus RVV+scalar dispatch routes.
* Reuse existing lit/FileCheck output tests for representative route behavior, and run focused suites for RVV microkernel, RVV+scalar dispatch, target artifact export, and e2e script wrappers.

## Acceptance Criteria

* [ ] `tools/tcrv-translate/tcrv-translate.cpp` calls a generic built-in target translate route registration function and no longer manually loops over RVV direct or RVV+scalar dispatch route manifests.
* [ ] `tcrv-translate` no longer needs `RVVScalarDispatch.h` and uses `RVVMicrokernel.h` only for the legacy standalone direct self-check helper that remains out of this module.
* [ ] RVV direct microkernel route ids remain visible through `tcrv-translate --help`, including i32 generic aliases and i64 family-specific routes.
* [ ] RVV+scalar dispatch route ids remain visible through `tcrv-translate --help`, including source/header/object/self-check commands.
* [ ] Object route commands preserve binary stdout switching through the generic adapter.
* [ ] Duplicate target translate route contributions fail clearly instead of silently shadowing an existing route.
* [ ] Focused changed targets build and focused tests pass.
* [ ] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2` passes before archive.
* [ ] No new RVV runtime/correctness/performance claim is made; no `ssh rvv` evidence is required unless generated runtime-callable behavior changes.

## Out of Scope

* New RVV compute families, shapes, masks, VL policy, performance tuning, ABI semantics, or runtime behavior.
* Python implementation of compiler behavior.
* IME, AME, Sophgo/offload route expansion, provider work, status-only/report-only work, or broad evidence matrices.
* Moving `tcrv.exec` into high-level compute semantics.
* Refactoring target artifact exporter semantics beyond the route translation registration owner boundary.

## Technical Notes

* Relevant specs read:
  * `.trellis/spec/index.md`
  * `.trellis/spec/plugin-protocol/extension-plugin-integration.md`
  * `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
  * `.trellis/spec/plugin-protocol/locality-contract.md`
  * `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  * `.trellis/spec/extension-plugins/rvv-plugin.md`
  * `.trellis/spec/testing/mlir-testing-contract.md`
* Relevant code inspected:
  * `tools/tcrv-translate/tcrv-translate.cpp`
  * `include/TianChenRV/Target/BuiltinTargetArtifactExporters.h`
  * `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`
  * `include/TianChenRV/Target/TargetArtifactExport.h`
  * `lib/Target/TargetArtifactExport.cpp`
  * `include/TianChenRV/Target/RVV/RVVMicrokernel.h`
  * `lib/Target/RVV/RVVMicrokernel.cpp`
  * `include/TianChenRV/Target/RVVScalarDispatch.h`
  * `lib/Target/Builtin/RVVScalarDispatch.cpp`
  * `test/Target/TargetArtifactExportTest.cpp`
  * `test/Target/I32BinaryFamilyRegistryTest.cpp`
  * representative `test/Target/RVVMicrokernel/`, `test/Target/RVVScalarDispatch/`, and `test/Scripts/` files.
