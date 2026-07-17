# Direct C semantic source frontdoor deletion

## Goal

Delete the active RVV smoke-probe direct source-artifact frontdoor slice in the
Wrong Logic Deletion Campaign, and record the still-larger scalar/generic
source-route residuals as the next deletion owner.

This is deletion/refactor-only work. Future executable output must come from a
rebuilt extension-family IR to materialized MLIR EmitC module route, not from
standalone smoke-probe C generators, descriptor-era route IDs, or scalar
compatibility source fixtures. After focused inventory, scalar compatibility
route metadata is deliberately left as residual for the next Hermes review
because it is entangled with dispatch/component preflight fixtures and is a
separate deletion slice from the RVV smoke-probe frontdoor.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial repo state for this round was clean at HEAD `f0a92c9`.
- Previous archived task `05-15-rvv-finite-route-registry-deletion` removed
  RVV finite route registry authority and passed full `check-tianchenrv`.
- Remaining direct source artifact residues include:
  - `tools/tcrv-translate/tcrv-translate.cpp` registers
    `tcrv-export-rvv-smoke-probe-c` directly.
  - `lib/Target/RVV/RVVSmokeProbe.cpp` emits standalone C containing
    `#include <riscv_vector.h>` and `__riscv_` RVV intrinsic compute.
  - `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp` registers the RVV
    smoke-probe exporter as a non-plugin target artifact route.
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp` still turns
    `tcrv_rvv.smoke_probe_descriptor` into a supported standalone C artifact
    emission plan.
  - `include/TianChenRV/Target/RVVScalarBinaryFamily.h` still stores scalar
    direct route IDs, header/object route IDs, runtime ABI identity strings,
    runtime glue role strings, and descriptor-era naming in
    `ScalarBinaryMicrokernelRecord`.
  - Scalar/generic target artifact fixtures still carry old direct source route
    strings as selected-plan inputs.
- `.trellis/spec/index.md`, architecture boundaries, EmitC route, emission
  runtime, and MLIR testing specs define descriptor/direct-C source export as
  implementation debt to delete or fail-close until a real EmitC route exists.
- Some spec sections still describe RVV smoke-probe as an allowed standalone
  route; this task updates those sections because this round intentionally
  deletes that exception.

## Requirements

- Remove or fail-close `tcrv-export-rvv-smoke-probe-c` so invoking it cannot
  emit C source.
- Remove RVV smoke-probe target artifact exporter registration so generic
  target source artifact export cannot select `standalone-c-source` smoke-probe
  output.
- Remove RVV plugin supported smoke-probe emission-plan behavior. A variant
  carrying old smoke-probe descriptor metadata must not become a supported
  artifact plan.
- Delete or rewrite tests that protect positive smoke-probe C source output,
  `riscv_vector.h`, `__riscv_` intrinsics, or smoke-probe route registration.
- Classify scalar compatibility route IDs, header/object route IDs, runtime ABI
  identity strings, runtime glue roles, descriptor naming, and generic direct
  source fixture strings as residuals for the next deletion owner unless a
  small, isolated deletion is proven safe during this round.
- Confirm scalar direct source/object exporters remain absent or fail-closed in
  this round; do not add scalar source/header/object route registration.
- Generic target artifact infrastructure may keep route-agnostic packaging,
  registry, artifact-kind filtering, and selected-plan validation behavior, but
  it must not synthesize compute semantics or preserve smoke-probe/scalar
  direct source fixtures as legal output.
- If deleting a path breaks build/tests because rebuild architecture is missing,
  record the failure as a deletion gap; do not restore a direct C exporter,
  descriptor wrapper, or compatibility route to make checks pass.

## Acceptance Criteria

- [x] `tcrv-export-rvv-smoke-probe-c` is absent from `tcrv-translate` command
      registration or fails closed before any C source bytes are emitted.
- [x] Active target/export code no longer emits RVV smoke-probe source with
      `#include <riscv_vector.h>` or `__riscv_` intrinsic compute.
- [x] Built-in target artifact exporter registration no longer publishes
      `tcrv-export-rvv-smoke-probe-c` / `standalone-c-source` as a supported
      route.
- [x] RVV plugin emission readiness/planning no longer reports a supported
      standalone smoke-probe C artifact route from
      `tcrv_rvv.smoke_probe_descriptor`.
- [x] Tests that previously expected positive RVV smoke-probe C source output
      are deleted or rewritten to route absence / fail-closed behavior.
- [x] Scalar compatibility route IDs/runtime ABI fields and generic direct
      source fixture strings are truthfully classified as residuals or, if
      touched, deleted without adding replacement compatibility behavior.
- [x] Scalar source/object direct export surfaces remain absent or fail-closed;
      this round does not add scalar source/header/object route registration.
- [x] No Common Lower-To-EmitC rebuild, new RVV/scalar lowering, replacement
      smoke probe, descriptor wrapper, compatibility layer, legacy direct route
      quarantine, new artifact route, ssh RVV campaign, or Python compiler
      semantics is added.
- [x] Bounded ref-scans classify residuals for the requested terms as deleted,
      fail-closed/negative fixture, route-agnostic infrastructure, or next
      deletion owner.
- [x] Focused affected builds/tests run first; full
      `cmake --build build --target check-tianchenrv -j2` runs when coherent.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation
      before finish and after archive, final clean `git status --short`, and
      one coherent commit are produced if complete.

## Non-goals

- No Common Lower-To-EmitC rebuild.
- No new RVV, scalar, IME, offload, or dispatch lowering.
- No replacement smoke probe, descriptor wrapper, compatibility mode, direct
  route quarantine, new artifact route, or legacy alias.
- No `ssh rvv` evidence campaign.
- No Python compiler semantics.
- No broad repo audit beyond bounded owner/ref-scan evidence.
- No restoring old direct C paths to satisfy stale tests.

## Minimal Evidence

- Focused inventory/ref-scan over touched files/directories for:
  `tcrv-export-rvv-smoke-probe-c`, `RVVSmokeProbe`, `__riscv_`,
  `standalone-c-source`, `runtime-callable-c-source`, `tcrv-export-scalar`,
  `ScalarBinaryMicrokernelRecord`, `routeID`, `headerRouteID`,
  `objectRouteID`, `runtime-callable-c-abi`, `descriptor`, and
  `descriptor-element-count`.
- Focused build targets for `tcrv-translate`, RVV target, scalar target,
  built-in target artifact exporters, and affected C++ tests.
- Affected lit subsets under `test/Target/RVVSmokeProbe`,
  `test/Target/ArtifactExport`, `test/Target/TargetArtifactBundleExport`,
  `test/Target/EmissionManifest`, and scalar/RVV scalar dispatch tests when
  touched.
- Full `check-tianchenrv` when the deletion slice is coherent.

## Technical Notes

- `.trellis/spec/architecture/design-boundaries.md` lists descriptor-driven
  microkernel/exporter frameworks as non-goals.
- `.trellis/spec/lowering-runtime/emitc-route.md` defines the accepted future
  path as extension family ops -> EmitC -> intrinsic/vendor/runtime C/C++.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` requires direct
  descriptor-to-C source/exporter routes to be deleted or fail-closed until a
  materialized MLIR EmitC module route exists.
- `.trellis/spec/testing/mlir-testing-contract.md` requires deleting or
  rewriting tests whose only assertion is successful old direct source/header/
  object/bundle output.
- Previous task:
  `.trellis/tasks/archive/2026-05/05-15-rvv-finite-route-registry-deletion/prd.md`.

## Completion Evidence

- Deleted the RVV smoke-probe target source generator:
  `include/TianChenRV/Target/RVV/RVVSmokeProbe.h` and
  `lib/Target/RVV/RVVSmokeProbe.cpp`.
- Removed `RVVSmokeProbe.cpp` from `TianChenRVRVVTarget`.
- Removed the direct `tcrv-translate --tcrv-export-rvv-smoke-probe-c`
  registration and the `tools/tcrv-translate` callback.
- Removed non-plugin built-in target artifact exporter registration for the
  smoke-probe route.
- Removed RVV plugin supported smoke-probe readiness/plan construction:
  `tcrv_rvv.smoke_probe_descriptor` no longer produces a supported standalone
  C artifact route.
- Reworked RVV variant legality so historical smoke-probe descriptor metadata
  is rejected as a deleted direct source artifact frontdoor.
- Deleted positive and path-specific RVV smoke-probe lit tests and replaced
  them with `test/Target/RVVSmokeProbe/rvv-smoke-probe-route-deleted.mlir`,
  which proves the public command is now unknown and emits no C source.
- Updated target artifact C++ tests to expect smoke-probe route absence in
  built-in target artifact exporter registration.
- Updated Trellis specs in `extension-plugins/rvv-plugin.md`,
  `lowering-runtime/emission-runtime-contract.md`, and
  `testing/mlir-testing-contract.md` so smoke-probe is documented as deleted,
  not as a supported exception.

## Bounded Ref-Scan Classification

- Active source scan over `tools include lib` for
  `tcrv-export-rvv-smoke-probe-c`, `RVVSmokeProbe`,
  `exportRVVSmokeProbeC`, `registerRVVSmokeProbe`,
  `rvv-smoke-probe-standalone`, and
  `standalone-c-toolchain-smoke-probe` has no active command/exporter hits.
  Remaining active hits are only `tcrv_rvv.smoke_probe_descriptor` constants in
  RVV legality and selected-boundary validation, where the descriptor is
  rejected as deleted-route input or triggers legality checking.
- Active target/tool scan for `#include <riscv_vector.h>` and `__riscv_` under
  `lib/Target`, `include/TianChenRV/Target`, and `tools` has no hits.
- `test/Target/RVVSmokeProbe/rvv-smoke-probe-route-deleted.mlir` intentionally
  mentions `tcrv-export-rvv-smoke-probe-c` only to prove command-line absence.
- `standalone-c-source` remains in generic target artifact infrastructure and
  registry-shape tests as route-agnostic artifact-kind support, not as a
  built-in RVV smoke-probe route.
- `ScalarBinaryMicrokernelRecord`, `tcrv-export-scalar*`,
  `runtime-callable-c-source`, scalar `runtime-callable-c-abi`,
  `headerRouteID`, `objectRouteID`, and `descriptor-element-count` remain as
  the next deletion owner. They are entangled with scalar compatibility records,
  dispatch/component candidate fixtures, and generic negative/spoofing tests.
  This round did not add any scalar registration or compatibility layer; scalar
  source/object export remains fail-closed or unregistered.

## Checks Run

- `rtk cmake --build build --target tcrv-translate TianChenRVRVVTarget TianChenRVRVVPlugin tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test tianchenrv-rvv-binary-variant-legality-test -j2`
- `rtk ./build/bin/tianchenrv-target-artifact-export-test`
- `rtk ./build/bin/tianchenrv-rvv-extension-plugin-test`
- `rtk ./build/bin/tianchenrv-rvv-binary-variant-legality-test`
- `rtk cmake --build build --target check-tianchenrv -j2`
  passed 114/114 lit tests on the final tree.
- `rtk git diff --check`
- `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-15-direct-c-source-frontdoor-deletion`
- `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/archive/2026-05/05-15-direct-c-source-frontdoor-deletion`
- `rtk git diff --cached --check`
