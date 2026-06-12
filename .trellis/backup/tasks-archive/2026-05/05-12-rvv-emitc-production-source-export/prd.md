# Make RVV Source Export Use Common EmitC Route Authority

## Goal

Migrate the bounded default RVV runtime-callable C source artifact path so the
production emitted function body is rendered from the typed-body-derived
`TCRVEmitCLowerableRoute` call sequence, not from a parallel RVV-only
handwritten dataflow printer. The RVV target still owns intrinsic spelling,
vector suffixes, ABI names, and selected vector config, but the exported call
sequence must use the common EmitC lowerable route as source authority.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Branch is `main`; worktree was clean at session start.
* Current HEAD before this task is
  `b5b193d feat(rvv): quarantine descriptor-only selected export`.
* No `.trellis/.current-task` existed at session start; this task was created
  as `.trellis/tasks/05-12-rvv-emitc-production-source-export/`.
* The archived predecessor
  `.trellis/tasks/archive/2026-05/05-12-rvv-descriptor-only-production-quarantine/`
  completed descriptor-only selected export quarantine and must not be
  reopened.
* `lib/Target/RVV/RVVMicrokernel.cpp` already builds and verifies a
  typed-body-derived `TCRVEmitCLowerableRoute`, but the production function
  body is still printed by `printMicrokernelFunction` using RVV dataflow step
  switches as the body authority.
* `verifyTCRVEmitCLowerableRouteMaterializesToEmitC` proves the route can
  become MLIR EmitC, but before this task it is only a side-channel guard for
  C source emission.

## Requirements

* The migrated RVV source body must be generated from the
  `TCRVEmitCLowerableRoute` ABI mappings and ordered `emitc.call_opaque`
  steps.
* The common rendering layer must fail closed before source export when the
  route is malformed, lacks the expected ABI boundary, references unknown
  values, emits duplicate results, has missing compute results, or cannot
  express the expected bounded loop shape.
* The production RVV source exporter must consume that common route-to-C body
  rendering path for the bounded finite typed RVV binary families, including
  frontend/default `i32-vadd` and the existing finite `i32`/`i64`
  add/sub/mul source routes when naturally covered by the abstraction.
* RVV-specific code may continue to build the route from the verified typed RVV
  body and selected intrinsic config, and may continue to emit RVV-specific
  comments, headers, self-check harnesses, header artifacts, and object
  compilation helpers.
* Descriptor metadata remains compatibility metadata only. It must not choose
  computation semantics or recover a missing route/body.
* Comments and FileCheck expectations that describe "bounded legacy C source
  output" must be updated to describe production route authority.
* Parameter layers must remain distinct: capability/config metadata, runtime
  `n`/AVL/VL/control values, runtime ABI parameter mappings, and
  descriptor-local compatibility metadata.

## Acceptance Criteria

* [x] `printMicrokernelSource` no longer emits the migrated function body from
      RVV-only `dataflowPlan` switches.
* [x] A shared route-to-C source body renderer consumes
      `TCRVEmitCLowerableRoute` and is used by RVV production source export.
* [x] Focused EmitC C++ coverage proves route-to-C rendering succeeds for the
      bounded RVV route and fails closed for malformed route provenance or call
      sequence cases.
* [x] Focused RVV source/export lit coverage shows exported C source records
      common EmitC route authority and still emits the expected RVV intrinsic
      sequence for at least frontend/default `i32-vadd`.
* [x] Existing finite typed RVV `i32`/`i64` add/sub/mul source routes remain
      supported when their typed bodies and common EmitC routes are valid.
* [x] Negative coverage proves stale/malformed route provenance fails before
      source output and descriptor-only metadata cannot recover computation.

## Out Of Scope

* New RVV arithmetic families, dtypes, LMULs, tuning, performance work, broad
  smoke matrices, runtime correctness claims, or fresh `ssh rvv` evidence.
* Descriptor-to-C exporters or descriptor-driven source computation.
* Moving compiler core, dialects, passes, plugin registry, capability model,
  lowering, or emission behavior into Python.
* `tcrv.exec` compute semantics.
* IME, AME, Sophgo/offload, TensorExt, Template, Toy, scalar-only, or
  unrelated plugin work.

## Minimal Validation

* `git diff --check`
* Focused build targets: `tcrv-opt`, `tcrv-translate`,
  `tianchenrv-emitc-lowerable-interface-test`,
  `tianchenrv-target-artifact-export-test`,
  `tianchenrv-rvv-binary-planning-test`, and relevant RVV target/export
  targets.
* Focused C++ tests for EmitC lowerable materialization/rendering, RVV target
  artifact export, RVV binary planning, and selected lowering boundary.
* Focused lit/FileCheck tests for migrated RVV source artifact paths,
  including at least one frontend/default linalg-to-RVV artifact path.
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if focused checks pass and local tooling/time allow.
* Trellis task validation before finish/archive and archived task validation if
  completed.

## Technical Notes

* `.trellis/spec/lowering-runtime/emitc-route.md` defines the current route as
  typed extension family ops -> common EmitC route -> intrinsic/runtime C/C++.
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md` requires
  target artifact export to preserve compiler-owned ABI contracts and fail
  closed before target output when selected route metadata is incoherent.
* `.trellis/spec/extension-plugins/rvv-plugin.md` keeps RVV emission
  plugin/target-owned while preserving common route authority.
* `.trellis/spec/architecture/unified-riscv-mlir.md` marks
  descriptor-driven computation and handwritten C string emission as
  implementation debt.
* `.trellis/spec/testing/mlir-testing-contract.md` requires focused MLIR,
  FileCheck, C++ test, and build evidence while keeping structural compiler
  checks separate from RVV hardware runtime/correctness/performance claims.

## Round Result

Completed in this round:

* Added `renderTCRVEmitCLowerableRouteAsCFunction` in the common EmitC
  conversion layer. The renderer consumes route ABI mappings and ordered
  `TCRVEmitCCallOpaqueStep` entries, derives the bounded runtime `n` loop from
  the route's `runtime-element-count` ABI mapping, emits call statements from
  route operands/results/callees, and fails closed for malformed loop shape,
  unknown values, duplicate values, missing compute results, or missing
  generated op-interface provenance.
* Buffered route-to-C rendering so malformed routes return an error without
  emitting partial C source to the final artifact output stream.
* Rewired RVV production source emission so `printMicrokernelFunction` delegates
  to the common renderer. The former RVV-only body printer switch over
  load/arithmetic/store dataflow step kinds no longer authors the emitted C
  function body.
* Kept RVV target ownership of route construction, intrinsic spelling, vector
  suffixes, selected ABI identity, comments, headers, object helpers, and
  self-check harness generation.
* Updated RVV source comments and FileCheck expectations from bounded legacy
  output to route-authored production C source authority.
* Added C++ coverage for route-to-C rendering success for add/sub/mul-style
  routes and fail-closed rendering diagnostics for missing
  `TCRVEmitCLowerableOpInterface` provenance, wrong first call order, and
  unknown operand values.

Self-repair performed:

* Fixed a C++ test syntax issue found by the focused build after adding the new
  renderer assertions.
* Tightened the renderer after review so fail-closed route diagnostics do not
  leak partial source text before the error is returned.

Validation completed:

* `git diff --check`
* `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate tianchenrv-emitc-lowerable-interface-test
  tianchenrv-target-artifact-export-test
  tianchenrv-rvv-binary-planning-test
  tianchenrv-rvv-selected-lowering-boundary-test -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emitc-lowerable-interface-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-binary-planning-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-selected-lowering-boundary-test`
* `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-rvv-extension-plugin-test -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
* Focused lit/FileCheck with `-j1` for RVV microkernel production source,
  auto-materialization, frontend/default `i32-vadd`, and frontend `i32-vmul`
  artifact paths: 5/5 passed.
* Focused lit/FileCheck with `-j1` for descriptor-only quarantine and
  descriptor/body mismatch negative paths: 2/2 passed.
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed with 206/206 lit tests.

No `ssh rvv` command was run or claimed. This round is a structural compiler
emission migration and makes no fresh RVV hardware runtime, correctness,
throughput, latency, or performance claim.
