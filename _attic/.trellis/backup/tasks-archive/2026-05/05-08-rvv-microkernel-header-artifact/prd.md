# RVV microkernel external ABI header artifact

## Goal

Make the bounded RVV i32-vadd microkernel object externally consumable through a target-owned C ABI handoff: export a matching runtime-callable C header derived from the same selected microkernel path and runtime ABI plan as the existing source/object routes, route it through the generic target header artifact front door, and, when `ssh rvv` is available, prove the generated header plus generated object with a bounded external caller correctness check.

## What I already know

* Repository root is `/home/kingdom/phdworks/TianchenRV`.
* Current expected HEAD is `2068763 feat: export RVV microkernel object artifacts`.
* The previous round added a target-owned RVV i32-vadd relocatable object route and kept RVV details in the target/export layer.
* `tcrv.exec` must remain compute-free and focused on execution/capability/variant/dispatch/fallback.
* Python is allowed only for evidence orchestration, runner scripts, remote probes, artifact parsing, fixtures, and small support utilities.
* The generic target artifact registry must remain artifact-kind based and must not gain RVV/vendor branches in core selection logic.

## Requirements

* Add a C++ RVV target/export header emitter for the bounded RVV i32-vadd microkernel.
* Derive the generated prototype from the same selected path, microkernel op, runtime ABI, mem window, runtime parameter, and capability metadata used by runtime-callable C source/object routes.
* Emit a library header only: include guard, required portable includes, and exactly the runtime-callable function declaration.
* Register the header as a built-in target artifact exporter route with artifact kind `runtime-callable-c-header`.
* Ensure `tcrv-translate --tcrv-export-target-header-artifact` selects the RVV microkernel header route for a direct selected RVV microkernel path.
* Keep `--tcrv-export-target-source-artifact`, `--tcrv-export-target-artifact`, and existing dispatch header routes separable and unambiguous.
* Add focused lit/FileCheck and C++ registry coverage for route kind separation and generated header content.
* Add or extend bounded `ssh rvv` evidence only if it directly consumes generated header and object from an external caller and verifies finite i32-vadd correctness.
* Update specs/docs only for durable behavior of the target-owned header artifact and bounded evidence contract.

## Acceptance Criteria

* [ ] Generic header front door exports an RVV microkernel C header for the selected bounded i32-vadd path.
* [ ] Header contains include guard, required includes, and the ABI-derived prototype, with no body, main, self-check harness, RVV intrinsics, paths, secrets, benchmark text, or evidence logs.
* [ ] Source/header/object target artifact routes remain registry-routed and separable.
* [ ] Capability and runtime ABI objects participate in exporter decisions.
* [ ] Local build and `check-tianchenrv` pass.
* [ ] If remote RVV evidence is available, it is bounded to generated header plus generated object external caller correctness and excludes performance/generic lowering claims.
* [ ] Trellis task is validated and archived before final commit.

## Out of Scope

* Generic RVV lowering or arbitrary kernel emission.
* IME/AME/Sophgo/offload target work.
* Runtime integration beyond the bounded generated header plus generated object external caller.
* Performance benchmarking or performance claims.
* Python implementation of compiler internals, exporter decisions, ABI modeling, or target artifact selection.

## Technical Notes

* Required pre-edit inspection includes the files listed in the supervisor request, including `AGENTS.md`, core specs, RVV target/export headers and implementations, `tcrv-translate`, existing RVV microkernel tests, target artifact tests, and the RVV e2e script/test.
* Required minimum checks: `git diff --check`, `cmake --build build --target tcrv-opt tcrv-translate -j2`, and `cmake --build build --target check-tianchenrv -j2`.
