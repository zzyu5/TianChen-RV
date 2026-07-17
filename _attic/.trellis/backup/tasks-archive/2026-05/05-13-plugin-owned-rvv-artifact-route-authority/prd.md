# Plugin-owned RVV artifact route authority

## Goal

Make the bounded RVV binary microkernel source/header/object artifact route
authority live in one production C++ RVV target/plugin-owned route description.
The generic target artifact bundle/front-door path and the remaining direct
`tcrv-translate --tcrv-export-rvv-microkernel-*` compatibility commands must
consume that same route authority, so route id, artifact kind, owner, runtime
ABI identity, component grouping, and preflight metadata cannot drift.

## What I Already Know

* Initial repository state for this task: `pwd` was
  `/home/kingdom/phdworks/TianchenRV`, worktree was clean, and HEAD was
  `3710182 test(rvv): route source-fronted proof through bundle front door`.
* No `.trellis/.current-task` existed at session start. This task was created
  from the Hermes Direction Brief before source edits.
* The previous archived task
  `.trellis/tasks/archive/2026-05/05-13-source-fronted-rvv-target-artifact-front-door/prd.md`
  proved the source-fronted `i32-vadd` path through
  `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` and real
  `ssh rvv` runtime evidence for `n = 7, 16, 23`.
* The next bottleneck is production C++ ownership, not another evidence-runner
  proof. The runner should consume compiler-emitted route metadata only.
* Current C++ already has finite RVV binary family descriptors and an RVV
  direct-route manifest, but source/header/object registration still derives
  several route facts through separate switch/local registration logic instead
  of one route-authority object.

## Requirements

* Define one RVV target/plugin-owned route authority for the finite direct RVV
  binary microkernel source/header/object family.
* Preserve the bounded `i32-vadd` compatibility route ids
  `tcrv-export-rvv-microkernel-c`,
  `tcrv-export-rvv-microkernel-header`, and
  `tcrv-export-rvv-microkernel-object`, but make them wrappers over the same
  route authority as generic target artifact registration and bundle export.
* Route authority must expose route id, artifact kind, route owner, emission
  kind, runtime ABI kind/name, runtime ABI contract parameters, external ABI
  component group/name, direct-helper compatibility status, binary stdout
  behavior, export callback, candidate preflight, and route metadata in C++.
* Generic target artifact registration and target translate route registration
  must iterate the same route authority; no separate hand-maintained direct
  helper route source may remain for the RVV microkernel family.
* Missing/stale selected RVV body, selected-config metadata, runtime
  element-count ABI, route owner, or selected-config route metadata must fail
  before source/header/object/bundle output through existing preflight
  boundaries.
* The evidence runner may be adjusted only as a consumer if compiler-emitted
  metadata changes; it must not become the route metadata source of truth.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits.
* [x] RVV source/header/object route metadata for the bounded binary
      microkernel family is described once by a C++ route-authority API.
* [x] `registerRVVMicrokernelTargetExporters` consumes that route authority for
      both standalone source exporters and header/object composite exporters.
* [x] `registerRVVMicrokernelTargetTranslateRoutes` consumes the same route
      authority for direct compatibility commands, including binary stdout for
      object routes.
* [x] C++ target/export tests prove generic registry metadata and direct
      compatibility metadata agree for representative source/header/object
      routes, including owner, artifact kind, runtime ABI kind/name, component
      group/name, direct-helper status, export callback, runtime ABI parameter
      source, and preflight callbacks.
* [x] Focused lit coverage still proves bundle/front-door source/header/object
      records preserve route, owner, runtime ABI, component role, selected
      surface, and preflight-fail behavior.
* [x] Exact `tcrv-translate` command demonstrates the source-fronted bundle
      route still emits source/header/object records from the unified route
      authority.
* [x] Focused build and tests pass for touched target/export and RVV surfaces.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.

## Definition Of Done

* Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit.
* Python remains a runner/consumer only.
* No computation semantics move into descriptors, Python, or `tcrv.exec`.
* Shared generic artifact/export code remains target-neutral and does not gain
  RVV-specific semantic branches.
* This task does not claim new runtime correctness, RVV performance, broad
  dtype/family coverage, or a generic RVV backend.

## Out Of Scope

* New arithmetic families, dtype matrices, performance claims, broad smoke
  matrices, generic RVV backend claims, or standalone evidence repackaging.
* Replacing the existing EmitC/C route or adding descriptor-driven computation.
* Runner-only, docs-only, helper-only, or broad-test-only changes as the main
  result.
* New `ssh rvv` evidence unless route output semantics materially change.

## Technical Notes

* Specs read for this task:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior task context read:
  `.trellis/tasks/archive/2026-05/05-13-source-fronted-rvv-target-artifact-front-door/prd.md`.
* Initial code inspection focused on:
  `include/TianChenRV/Target/RVV/RVVMicrokernel.h`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`,
  `test/Target/TargetArtifactBundleExport/`,
  `test/Target/EmissionManifest/`,
  `test/Target/RVVMicrokernel/`,
  `test/Transforms/LinalgToExec/`,
  `test/Scripts/rvv-microkernel-e2e.test`, and
  `scripts/rvv_microkernel_e2e.py`.

## Completion Notes

* `RVVMicrokernelDirectRouteManifestEntry` now exposes the production RVV
  artifact route authority fields directly: owner, emission kind, artifact
  kind, runtime ABI kind/name, runtime glue role, external ABI component
  group/name, component role, binary stdout behavior, and direct-helper
  compatibility status. `RVVMicrokernelArtifactRouteDescriptor` is the public
  route-authority alias, and `getRVVMicrokernelArtifactRouteAuthority()` is the
  authoritative source consumed by both generic target artifact registration
  and direct translate-route compatibility registration.
* `registerRVVMicrokernelTargetExporters` now builds RVV source exporters and
  header/object composite exporters from the route authority instead of keeping
  separate local switch metadata. Header/object routes also publish the same
  conservative route-claim metadata as source routes.
* `registerRVVMicrokernelTargetTranslateRoutes` now iterates the same route
  authority, so direct `--tcrv-export-rvv-microkernel-*` compatibility commands
  share route id, binary stdout, target-artifact route id, and export callback
  identity with the generic artifact registry.
* The RVV extension bundle now requires route metadata for the full finite
  source/header/object RVV artifact route set, not only source routes. Missing
  header/object composite metadata is therefore caught at the plugin-owned
  exporter bundle boundary.
* `test/Target/TargetArtifactExportTest.cpp` now checks the route-authority API,
  generic source/composite exporter registration, direct translate-route
  compatibility registration, route owner/runtime ABI/component metadata
  agreement, route-local runtime ABI callbacks, and route-local preflight
  callbacks.
* `test/Target/TargetArtifactBundleExport/target-artifact-bundle-positive.mlir`
  now checks source/header/object bundle records preserve compiler-artifact-only
  route claims and runtime-element-count ABI metadata for the direct RVV
  compatibility route family.
* Exact source-fronted bundle command:

```bash
build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/plugin_owned_rvv_route_authority/bundle test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir
```

* The generated bundle index at
  `artifacts/tmp/plugin_owned_rvv_route_authority/bundle/tianchenrv-target-artifact-bundle.index`
  emitted three complete direct RVV records:
  `tcrv-export-rvv-microkernel-c`,
  `tcrv-export-rvv-microkernel-header`, and
  `tcrv-export-rvv-microkernel-object`. Each record preserved
  `owner = "rvv-plugin"`,
  `runtime_abi_kind = "rvv-runtime-callable-c-abi"`,
  `runtime_abi_name = "rvv-i32-vadd-runtime-callable-c-function.v1"`,
  component roles `source` / `header` / `object`, ordered runtime ABI
  parameters including `runtime-element-count`, selected-plan metadata, and
  conservative route claims.
* Generated runtime semantics did not materially change, so this round did not
  make a new `ssh rvv` runtime correctness claim. The previous bounded
  source-fronted runtime proof remains applicable; this round proves route
  authority equivalence through production C++ registry, bundle, and FileCheck
  evidence.

## Validation

* `cmake --build build --target TianChenRVTarget TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
* `./build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'TargetArtifactBundleExport|EmissionManifest|RVVMicrokernel|LinalgToExec|rvv-microkernel-e2e'`
  with 73/73 selected tests passed.
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-plugin-owned-rvv-artifact-route-authority`
* `git diff --check`
* `git diff --cached --check`
