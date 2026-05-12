# RVV Scalar Dispatch Common EmitC Source Boundary

## Goal

Rewire the RVV+scalar host dispatch source body so the target-owned dispatcher
control route is emitted through the common `lowerTCRVEmitCLowerableToEmitCSource`
boundary instead of calling the lower-level EmitC source-authority materializer
directly from `RVVScalarDispatch.cpp`.

This is a structural production-path migration. It must change the real
dispatch source exporter used by direct routes, generic target source export,
bundle source/object helpers, and self-check source/object helpers. It must
not stop at comments, helper-only scaffolding, or broad smoke coverage.

## What I Already Know

* Current HEAD is `d2b11f5 feat(rvv): use common emitc source boundary`.
* The previous rounds added `lowerTCRVEmitCLowerableToEmitCSource`, rewired
  scalar callable source emission through it, then rewired RVV callable source
  emission through it.
* `lib/Target/Builtin/RVVScalarDispatch.cpp` still imports
  `TCRVEmitCSourceAuthorityOptions` and
  `emitTCRVEmitCLowerableRouteAsCppSource` and uses them in
  `emitDispatchFunctionFromEmitC`.
* The dispatch target already builds a bounded `TCRVEmitCLowerableRoute` for
  `tcrv.exec.dispatch` control over the selected RVV callable and scalar
  fallback callable artifacts.
* Dispatch route construction and validation must remain target/export-local:
  selected RVV/scalar callable candidates, family matching, ABI parameter
  binding, runtime guard linkage, fallback target linkage, dispatcher symbol
  naming, and self-check harness ownership stay in the RVV+scalar dispatch
  target code.
* The common boundary must own route-to-EmitC-to-C++ source emission and must
  continue to require the MLIR Cpp emitter authority marker.

## Requirements

* Replace the direct dispatch source-authority emission call in
  `lib/Target/Builtin/RVVScalarDispatch.cpp` with
  `lowerTCRVEmitCLowerableToEmitCSource`.
* Introduce a target-local dispatch-control lowerable adapter, or equivalent
  target-local interface-backed path, so the common boundary can consume a
  `TCRVEmitCLowerableInterface` while dispatch-specific route construction
  stays outside common conversion code.
* Preserve dispatch-specific source-authority options, especially dispatcher
  function name, dispatch guard C value name, and
  `requireInterfaceBackedCompute = false` for `tcrv.exec.case` /
  `tcrv.exec.fallback` control calls.
* Keep embedded RVV callable and scalar callable source generation unchanged;
  those component bodies already use the common boundary through their
  target-local exporters.
* Add generated-source evidence that the host dispatcher body crossed the
  common lower-to-EmitC source-authority boundary, while preserving
  `tcrv_emitc.source_authority=mlir_emitc_cpp_emitter`,
  `tcrv_emitc.dispatch_control_source=tcrv.exec.dispatch`, and route/call
  metadata.
* Remove obsolete direct dispatch target imports/usings of
  `emitTCRVEmitCLowerableRouteAsCppSource` and
  `TCRVEmitCSourceAuthorityOptions` once no production dispatch code calls
  them.
* Update focused FileCheck and evidence-helper expectations for dispatch
  source output to require the dispatch common-boundary marker.
* Do not move RVV/scalar/dispatch semantics into common conversion code, add
  descriptor-driven computation, add new families, change public dispatch ABI,
  add automatic hardware probing, claim performance, or add compute semantics
  to `tcrv.exec`.

## Acceptance Criteria

* [x] `RVVScalarDispatch.cpp` no longer imports or calls
  `emitTCRVEmitCLowerableRouteAsCppSource` or uses
  `TCRVEmitCSourceAuthorityOptions` directly for production dispatch source
  emission.
* [x] Dispatch source emission calls `lowerTCRVEmitCLowerableToEmitCSource`.
* [x] The target-local dispatch route still records two `call_opaque` steps:
  one `tcrv.exec.case` dispatch-case call and one `tcrv.exec.fallback`
  dispatch-fallback call.
* [x] Generated dispatch source records a dispatch-specific common
  lower-to-EmitC boundary marker and still records
  `tcrv_emitc.source_authority=mlir_emitc_cpp_emitter`.
* [x] Direct route, generic target-source route, bundle source, and dry-run
  evidence checks still prove the dispatcher body calls the selected RVV
  callable when the runtime guard is true and scalar fallback when false.
* [x] Focused build succeeds for the touched target/export code.
* [x] Focused lit coverage passes for RVV+scalar dispatch source routes and
  bundle/export routes affected by the marker change.
* [x] Focused dry-run evidence for `scripts/rvv_scalar_dispatch_e2e.py` passes
  without making runtime/correctness/performance claims.
* [x] `git diff --check` passes.
* [x] `cmake --build build --target check-tianchenrv -j2` passes unless a
  precise environment/tool blocker is recorded.
* [x] The Trellis task validates, finishes, archives, and the work is committed.

## Completion Notes

* Replaced dispatch target direct use of `emitTCRVEmitCLowerableRouteAsCppSource`
  / `TCRVEmitCSourceAuthorityOptions` with a target-local
  `DispatchControlEmitCLowerable` adapter consumed by
  `lowerTCRVEmitCLowerableToEmitCSource`.
* Kept dispatch route construction in `RVVScalarDispatch.cpp`: selected RVV and
  scalar callable symbols, ABI parameter bindings, runtime guard C name,
  fallback link validation, route id, and the two `call_opaque` steps remain
  target/export-owned.
* Preserved `requireInterfaceBackedCompute = false` for dispatch-control
  `tcrv.exec.case` / `tcrv.exec.fallback` calls while still requiring the MLIR
  Cpp emitter source-authority marker through the common boundary.
* Generated dispatch source now records
  `dispatch_emitc_common_lower_to_emitc_boundary:
  TCRVLowerToEmitCSourceAuthority` before the dispatch function body.
* Updated RVV+scalar dispatch FileCheck expectations and the dry-run evidence
  helper validator/sample text to require the dispatch common-boundary marker.
* No runtime/correctness/performance claim was made; dry-run evidence is local
  compiler artifact validation only.

## Checks Run

* `cmake --build build --target TianChenRVConversionEmitC TianChenRVBuiltinTargetArtifactExporters tcrv-translate tcrv-opt -j2`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-scalar-i32-vadd-dispatch-c|rvv-scalar-i32-vadd-dispatch-generic-route|rvv-scalar-i32-vsub-dispatch-generic-route|rvv-scalar-i32-vmul-dispatch-generic-route|rvv-scalar-i64-v(add|sub|mul)-dispatch-generic-route|rvv-scalar-dispatch-e2e|rvv-scalar-dispatch-bundle-e2e|plan-linalg-i32-vadd-and-export-target-artifact-bundle|plan-linalg-i32m2-vmul-and-export-target-artifact-bundle|plan-linalg-i64-vmul-and-export-target-artifact-bundle'`
  with 12/12 selected tests passing after one FileCheck ordering repair.
* `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --arithmetic-family=i32-vadd --run-id codex-dispatch-common-emitc-boundary-dry --overwrite`
* `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vmul --run-id codex-dispatch-common-emitc-boundary-bundle-dry --overwrite`
* `git diff --check`
* `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-12-05-12-rvv-scalar-dispatch-common-emitc-source-boundary`
* `cmake --build build --target check-tianchenrv -j2`, 210/210 tests passed.
* `./build/bin/tianchenrv-target-artifact-export-test`

## Out of Scope

* No new dispatch families, dtype expansion, ABI shape change, object format
  change, or public command rename.
* No generic RVV/scalar/dispatch backend maturity claim.
* No automatic RVV hardware probing or runtime integration change.
* No descriptor-to-C exporter, descriptor-selected computation, or direct
  handwritten dispatcher C rendering path.
* No Python implementation of compiler core, dialects, passes, lowering, or
  emission. Python changes are limited to evidence helper expectations if
  needed.
* No `tcrv.exec` compute semantics.

## Technical Notes

* Latest supervisor artifacts read:
  `artifacts/tmp/hermes_codex_supervisor/runs/20260512T-retry-guard-resume-r0003-20260512T103210Z/repo_audit.md`
  and matching `review_input.md`.
* Relevant specs read: `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Previous task context read:
  `.trellis/tasks/archive/2026-05/05-12-rvv-common-emitc-source-boundary-production-owner/prd.md`
  and workspace journal session 43.
* Primary source target:
  `lib/Target/Builtin/RVVScalarDispatch.cpp`.
* Expected implementation shape: wrap the existing dispatch route construction
  in a target-local `TCRVEmitCLowerableInterface`, call
  `lowerTCRVEmitCLowerableToEmitCSource` with dispatch source options, use the
  returned source and route for output metadata, and update tests/evidence
  validators to require the dispatch common-boundary marker.
