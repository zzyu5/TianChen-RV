# RVV construction-backed hardware artifact proof

## Goal

Refresh and harden the current RVV i32m1 add/sub/mul production path so the
default and exact target artifact routes are proven, at current HEAD, to flow
through selected RVV extension-family ops, RVV construction-protocol-checked
EmitC route construction, common EmitC materialization, target artifact export,
generated object/header artifacts, and real `ssh rvv` correctness evidence.

The bounded route for this task is:

```text
tcrv.exec selected path
  -> explicit tcrv_rvv i32m1 add/sub/mul body
  -> tcrv_rvv.with_vl selected boundary
  -> RVV construction protocol + typed role/interface realization
  -> RVV-owned EmitC route provider
  -> common EmitC materializer + MLIR C/C++ emitter
  -> RVV target object/header export and translate routes
  -> ssh rvv link/run correctness evidence
```

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting worktree was clean at HEAD `4d8c923`.
- No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes direction brief.
- Recent commits already built the required route pieces:
  `c70febc` generalized the RVV i32m1 arithmetic EmitC route family,
  `ba7fe66` centralized the i32m1 config/VL contract,
  `577b3d6` made `tcrv_rvv.with_vl` the selected boundary,
  `dc3eecb` wired RVV construction-protocol consumption, and `4d8c923`
  finished the Toy target/export template after RVV.
- Journal evidence records prior `ssh rvv` correctness for add/sub/mul under
  `artifacts/tmp/rvv_i32m1_arithmetic_route_family/20260516T101424Z`, but this
  task must refresh current-HEAD evidence after the latest target/export work.
- `RVVConstructionProtocol` owns add/sub/mul construction route mappings for
  EmitC route ids, object/header artifact route ids, translate route ids,
  callable component groups, runtime ABI names, and runtime ABI parameters.
- `RVVEmitCRouteProvider` validates explicit runtime ABI SSA bindings,
  `setvl`/`with_vl`, load/compute/store provenance, construction mapping, and
  typed `TCRVEmitCLowerableOpInterface` roles before route construction.
- `RVVExtensionPlugin` currently builds supported emission plans from the
  selected `with_vl` boundary and verifies construction mapping before
  reporting the RVV object route.
- `RVVTargetSupportBundle` registers add/sub/mul object/header/export/translate
  routes and invokes the common selected EmitC artifact front door before
  compiling generated C/C++ to a RISC-V relocatable object.

## Requirements

- Validate that current HEAD still exposes add/sub/mul selected paths through
  `tcrv-translate` default target artifact export, exact object/header
  translate routes, and target artifact bundle export.
- Prove that generated C/C++ comes from materialized EmitC via the common
  selected EmitC artifact front door, not descriptor/direct-C semantic export.
- Prove that object/header or bundle metadata carries the expected callable C
  runtime ABI boundary: `lhs`, `rhs`, `out`, and `n`, stable types, stable
  roles, target-export ownership, route ids, and callable component group.
- Confirm stale route IDs, stale route/op mismatches, missing selected path,
  unselected multi-variant input, missing selected boundary, unsupported shape,
  and missing runtime ABI/materialized EmitC evidence fail closed.
- If any default/export path can reach target artifacts without construction
  mapping, selected RVV boundary, or materialized EmitC, remove or fail-close
  that bypass in this task.
- Generate current add/sub/mul object/header or bundle artifacts and run them
  on `ssh rvv` through an external correctness harness.
- Keep common/core orchestration extension-neutral. RVV intrinsic names,
  headers, typed-body shape rules, and runtime ABI semantics must remain in
  RVV plugin/target code.

## Acceptance Criteria

- [x] Focused build succeeds for RVV construction protocol, RVV EmitC route
      provider, RVV plugin, RVV target support, target artifact export,
      `tcrv-opt`, and `tcrv-translate`.
- [x] Focused C++ tests pass for RVV plugin/unit coverage and target artifact
      export coverage.
- [x] Focused lit/FileCheck coverage passes for RVV EmitC materialization and
      RVV target artifact routes, including add/sub/mul positive paths and
      fail-closed negative paths.
- [x] Current `tcrv-translate` outputs for add/sub/mul selected routes produce
      object/header or bundle artifacts whose route/runtime ABI metadata match
      the RVV construction mapping.
- [x] Current generated add/sub/mul artifacts link and run on `ssh rvv` with
      correctness checks passing through the declared callable C ABI.
- [x] `git diff --check` passes.
- [x] A changed-surface scan shows descriptor/direct-C/source-export/Python
      compiler-core legacy paths were not restored and common/core code does
      not gain RVV intrinsic/header names or RVV semantic branches.
- [x] If implementation changes are required, focused checks are rerun after
      self-repair; if no source change is required, record why current code
      already satisfies the production path and commit only truthful Trellis
      task/context/evidence updates if appropriate.

## Validation Results

- Focused build:
  `cmake --build build --target TianChenRVRVVConstructionProtocol
  TianChenRVRVVEmitCRouteProvider TianChenRVRVVPlugin TianChenRVRVVTarget
  TianChenRVTarget tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test -j2`.
- Focused C++ tests:
  `./build/bin/tianchenrv-rvv-extension-plugin-test`,
  `./build/bin/tianchenrv-target-artifact-export-test`,
  `./build/bin/tianchenrv-construction-protocol-common-test`,
  `./build/bin/tianchenrv-rvv-dialect-test`.
- Focused lit:
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter
  'rvv-first-slice-materialization|Target/RVV/i32m1|rvv-with-vl-selected-boundary'`
  from `build/test`, 17/17 passed.
- Full practical check:
  `cmake --build build --target check-tianchenrv -j2`, 102/102 passed.
- Artifact/evidence directory:
  `artifacts/tmp/rvv_construction_backed_hardware_artifact_proof/20260516T125646Z`.
- Generated default and exact object/header artifacts for add/sub/mul selected
  dispatch routes. `file` reported RISC-V ELF relocatable objects.
- Generated target artifact bundles for add/sub/mul. Bundle indexes recorded
  object/header routes, callable component groups, `plugin-owned-runtime-abi`,
  runtime ABI names, and ordered `lhs`, `rhs`, `out`, `n` parameters with
  target-export ownership.
- `ssh rvv` link/run evidence:
  `artifacts/tmp/rvv_construction_backed_hardware_artifact_proof/20260516T125646Z/ssh_rvv_link_run.log`.
  Remote run printed
  `tcrv_rvv_i32m1_arithmetic_current_head status=PASS n=4
  add=[12,6,16,12] sub=[2,-12,24,0] mul=[35,-27,-80,36]`
  and exited with status 0.
- `git diff --check` passed.
- Changed-surface scan:
  common/core RVV intrinsic/header search found only plugin registration/CMake
  references outside RVV-owned code; direct-C/source-export scan found only
  existing dialect boundary comments, not restored production routes.
- No source code change was required. Current code already routes add/sub/mul
  through RVV construction mapping, selected `with_vl`, plugin-owned EmitC
  route construction, common EmitC materialization, target object/header export,
  and `ssh rvv` correctness evidence.

## Definition Of Done

- Trellis task state and context files are truthful.
- The PRD remains scoped to the existing RVV i32m1 add/sub/mul family.
- Current-HEAD local and `ssh rvv` evidence is recorded under `artifacts/tmp/`
  or an equivalent task-local artifact path.
- The task is finished/archived when complete.
- One coherent commit is created when complete, unless there are no repository
  changes worth committing; in that case the final report must state why.

## Out Of Scope

- New RVV SEW/LMUL/dtype/op families, i32m2 execution, generic RVV lowering,
  MLIR vector lowering, high-level frontend lowering, TensorExt/IME/offload
  work, scalar fallback expansion, new descriptor or binary-family registries,
  direct C semantic exporters, source skeletons, Python compiler-core logic,
  broad target framework rewrites, performance claims, GCC-default routes,
  compatibility wrappers, artifact ledgers, state machines, or common/core
  RVV semantic branches.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/plugin-protocol/extension-family-plugin-template.md`.
- Prior task PRDs read:
  `.trellis/tasks/archive/2026-05/05-16-rvv-executable-construction-protocol/prd.md`,
  `.trellis/tasks/archive/2026-05/05-16-toy-target-artifact-export-template/prd.md`.
- Relevant journal entries read:
  `.trellis/workspace/codex/journal-7.md` sessions 88, 89, 91, 92, 93, and
  Toy target artifact export notes.
- Primary implementation surface inspected:
  `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`,
  `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `include/TianChenRV/Target/RVV/RVVTargetSupportBundle.h`,
  `lib/Target/RVV/RVVTargetSupportBundle.cpp`,
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Target/TargetTranslateRegistration.cpp`.
- Primary test surface inspected:
  `test/Conversion/EmitC/rvv-first-slice-materialization*.mlir`,
  `test/Target/RVV/`,
  `test/Plugin/RVVExtensionPluginTest.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`.
