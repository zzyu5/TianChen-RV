# Make Scalar Fallback Source Export Use Common EmitC Route Authority

## Goal

Migrate bounded scalar fallback runtime-callable C source export so the
production emitted function body is rendered from the typed
`tcrv_scalar.*_microkernel` body and its `TCRVEmitCLowerableRoute`, not from a
scalar-only handwritten C body printer or descriptor/family metadata.

## What I Already Know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Branch is `main`; the worktree was clean at session start.
* Current HEAD before this task is
  `e1ffccb feat(rvv): render source from emitc route authority`.
* No `.trellis/.current-task` existed at session start; this task was created
  as `.trellis/tasks/05-12-scalar-emitc-production-source-export/`.
* The archived predecessor
  `.trellis/tasks/archive/2026-05/05-12-rvv-emitc-production-source-export/`
  completed the analogous RVV migration and must not be reopened.
* The current scalar target code builds and verifies a
  `TCRVEmitCLowerableRoute`, but production scalar source export still has the
  old migration smell: scalar-specific handwritten body emission can choose the
  arithmetic from family metadata.

## Requirements

* Scalar source export must build a `TCRVEmitCLowerableRoute` from the typed
  `tcrv_scalar.i32_*_microkernel` / `tcrv_scalar.i64_*_microkernel` op and
  render the production C body through common EmitC route source rendering.
* The production/default scalar source path must no longer use
  `family.cOperator` or a scalar-only body printer as compute authority.
* Common route source rendering may be extended for the generic scalar bounded
  runtime-element-count loop shape, but it must not branch on scalar/RVV family
  names to recover semantics.
* Scalar target/plugin code may continue to own route construction, source op
  provenance, callee/runtime helper names, inline helper declarations, operand
  expressions, result names, ABI names, and C types.
* Generated scalar source must remain deterministic and callable with
  `int32_t`/`int64_t` pointers plus runtime `size_t n`.
* Generated scalar source must record common EmitC route authority and generated
  op-interface provenance, analogous to the migrated RVV evidence comments.
* Malformed scalar routes must fail closed before source output for missing
  generated op-interface provenance, unknown operand value, duplicate
  value/result, missing compute result, missing runtime-element-count ABI
  mapping, or stale descriptor/body mismatch.
* Descriptor metadata remains compatibility/cross-check metadata only. It must
  not choose arithmetic, dtype, C operator, or source body when a typed scalar
  body and route disagree.
* Header/object/bundle helpers must continue to use the same scalar callable
  candidate and must not recover computation from descriptor-only metadata.

## Acceptance Criteria

* [x] Scalar runtime-callable C source export uses common
      `renderTCRVEmitCLowerableRouteAsCFunction`-style route authority for the
      bounded `i32`/`i64` add/sub/mul fallback routes.
* [x] The scalar-specific production body printer that directly emits
      `out[index] = lhs[index] <op> rhs[index]` from family metadata is deleted,
      bypassed, or reduced to non-compute scaffolding.
* [x] Common route rendering supports the required scalar loop shape through
      ABI mappings and ordered `call_opaque` steps, without family-name
      semantic branches.
* [x] Focused C++ or lit coverage proves scalar source export records common
      EmitC route authority and no longer uses descriptor/handwritten body
      authority.
* [x] Negative coverage proves malformed provenance and descriptor/body
      mismatch fail before source output.
* [x] Existing scalar header/object/bundle helper behavior remains wired to the
      same callable candidate without descriptor-driven compute recovery.

## Out Of Scope

* New scalar arithmetic families, dtypes, broad route matrices, performance
  work, runtime correctness claims, or fresh RVV hardware evidence.
* Descriptor-to-C computation exporters or descriptor-driven source semantics.
* Compiler core, dialects, passes, plugin registry, capability model, lowering,
  or emission behavior implemented as Python data structures.
* New compute semantics in `tcrv.exec`.
* Family-name semantic branching in common EmitC route rendering.
* Broad e2e evidence matrices beyond the focused scalar source export checks.

## Minimal Validation

* `git diff --check`
* Focused build targets if present:
  `tianchenrv-emitc-lowerable-interface-test`,
  `tianchenrv-target-artifact-export-test`, and
  `tianchenrv-scalar-extension-plugin-test`.
* Focused C++ tests for common EmitC route rendering, scalar plugin/source
  export, target artifact export, and any scalar object/header helper affected.
* Focused lit/FileCheck tests proving scalar source export records common EmitC
  route authority and no longer uses descriptor/handwritten body authority.
* Negative coverage for malformed route provenance and descriptor/body mismatch
  failing before source output.
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  after focused checks pass if the build tree is usable.
* Trellis task validation before archive and archived task validation if
  completed.

## Technical Notes

* `.trellis/spec/lowering-runtime/emitc-route.md` defines the route as typed
  extension family ops -> common EmitC route -> intrinsic/runtime C/C++.
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md` requires target
  artifact export to preserve compiler-owned ABI contracts and fail closed
  before target output when selected route metadata is incoherent.
* `.trellis/spec/extension-plugins/scalar-fallback-plugin.md` keeps scalar
  fallback behavior plugin/target-owned while using the shared compiler route.
* `.trellis/spec/architecture/unified-riscv-mlir.md` marks
  descriptor-driven computation and handwritten C string emission as migration
  debt.
* `.trellis/spec/testing/mlir-testing-contract.md` requires focused MLIR,
  FileCheck, C++ test, and build evidence while keeping structural compiler
  checks separate from hardware runtime/performance claims.

## Round Result

Completed in this round:

* Extended the common EmitC C source renderer so it supports two generic loop
  shapes from `TCRVEmitCLowerableRoute`: the existing runtime-AVL-to-VL loop
  and a runtime-element-count scalar loop. The scalar loop consumes ABI
  mappings plus ordered `call_opaque` compute/store steps and rejects malformed
  non-compute result-producing scalar-loop routes.
* Rewired scalar fallback source export so `printMicrokernelSource` builds and
  verifies the typed scalar `TCRVEmitCLowerableRoute`, records route authority
  comments, and renders the production callable body through
  `renderTCRVEmitCLowerableRouteAsCFunction`.
* Removed the old scalar body printer that recovered arithmetic from
  `family.cOperator`. Scalar target code now emits bounded inline helper
  definitions, while the callable body itself invokes the route-authored
  `tcrv_scalar_<dtype>_<op>` compute callee and scalar store callee.
* Updated scalar source/export, dispatch, linalg, bundle, e2e-runner, and
  common EmitC tests to assert common route body authority and route
  compute/store calls instead of descriptor/handwritten body authority.
* Updated scalar fallback and testing specs so the long-term contract matches
  the route-authored scalar production body.

Self-repair performed:

* Updated stale FileCheck expectations that still required direct
  `out[index] = lhs[index] <op> rhs[index]` source text.
* Updated the Python e2e runner's generated-source structural snippets so it
  validates route-authored scalar compute/store helper calls. Python remains a
  runner/validator only, not compiler implementation.

Validation completed:

* `cmake --build artifacts/tmp/tianchenrv-build --target
  tianchenrv-emitc-lowerable-interface-test
  tianchenrv-target-artifact-export-test
  tianchenrv-scalar-extension-plugin-test -j2`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-emitc-lowerable-interface-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-target-artifact-export-test`
* `artifacts/tmp/tianchenrv-build/bin/tianchenrv-scalar-extension-plugin-test`
* `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate -j2`
* Focused lit/FileCheck for scalar source/header/object routes: 3/3 passed.
* Focused lit/FileCheck for scalar dispatch/linalg/coherence routes: 14/14
  passed.
* Focused lit/FileCheck for target artifact bundle and rvv-scalar e2e runner
  routes: 10/10 passed.
* `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
* `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
* `git diff --check`
* `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  passed with 206/206 lit tests.

No `ssh rvv` command was run or claimed. This round is a structural compiler
source-export migration and makes no fresh RVV/scalar hardware runtime,
correctness, throughput, latency, or performance claim.
