# Delete RVV direct route fixture/API residue

## Goal

Delete the remaining RVV direct runtime-callable C source/header/object route
surface from the public target/artifact boundary. This is a Wrong Logic
Deletion Campaign round: RVV executable artifacts must not be justified by
`tcrv-export-rvv-microkernel-c`, RVV source/header/object route strings,
selected-config authority stubs, runtime-callable ABI fixture metadata, or
direct C exporter compatibility APIs.

Future RVV executable output must come from extension-family operations through
a materialized MLIR EmitC route. This task does not rebuild that route.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial repo state for this round was clean at HEAD `b4738f3`.
- `.trellis/.current-task` did not exist at session start; this task was
  created from the Hermes Direction Brief.
- The previous archived scalar deletion task removed scalar direct
  runtime-callable source/header/object route authority and passed
  `check-tianchenrv`.
- `include/TianChenRV/Target/RVV/RVVMicrokernel.h` still exposes public
  direct C/header/object export, selected-config authority, and validation
  declarations.
- `lib/Target/RVV/RVVMicrokernel.cpp` still keeps fail-closed implementations
  and no-op route registration functions for deleted direct RVV routes.
- Production RVV plugin target-support already goes through
  `RVVTargetSupportBundle`, which contributes no target artifact exporters and
  no target translate routes.
- `test/Target/TargetArtifactExportTest.cpp` still contains RVV direct route
  helper strings, no-op registry tests for the old RVV microkernel API, and
  unused direct source candidate validation/exporter authority tests.
- `test/Target/ArtifactExport/Inputs` still contains positive/spoof fixture
  metadata using `runtime-callable-c-source`,
  `rvv-explicit-i32-vadd-microkernel-c-source`,
  `tcrv-export-rvv-microkernel-c`,
  `rvv-runtime-callable-c-abi`, and
  `rvv-i32-vadd-runtime-callable-c-function.v1`.
- Relevant specs define direct descriptor/source export as bounded deletion
  debt and define the accepted future route as extension family ops -> EmitC
  -> C/C++ emitter.

## Requirements

- Remove public RVV direct source/header/object exporter APIs and selected
  config authority resolver APIs from production headers.
- Remove production implementation stubs for RVV direct source/header/object,
  self-check, selected-config authority, and direct-route no-op registration
  when no remaining production call site requires them.
- Remove the obsolete RVV direct microkernel target source file from the build
  if all production call sites are gone.
- Rewrite C++ tests so they check route absence through current built-in
  target-support boundaries rather than calling deleted RVV microkernel
  registration APIs.
- Delete unused target artifact fixture files whose only purpose is preserving
  old RVV direct source route or runtime-callable ABI identity.
- Keep at most necessary deleted-route negative assertions for the removed
  command-line option and built-in registry absence.
- Preserve generic route-agnostic artifact infrastructure, generic artifact
  kinds, bundle filename sanitization, Toy/Template/TensorExt metadata routes,
  scalar deletion behavior, and RVV metadata-only planning.
- Do not add a replacement direct C route, descriptor wrapper, compatibility
  layer, new selected-config resolver, new RVV lowering, Common EmitC rebuild,
  ssh evidence campaign, or Python compiler semantics.

## Acceptance Criteria

- [x] No production header exposes `exportRVVMicrokernel*`,
      `validateRVVMicrokernelSourceAuthority`, or
      `resolveRVVMicrokernelSelectedConfigContractAuthority`.
- [x] No production target implementation keeps deleted RVV direct-route
      compatibility stubs or selected-config authority resolvers.
- [x] No production registry path publishes RVV source/header/object direct
      route IDs; current RVV target-support registration remains route-free.
- [x] Positive/spoof target artifact fixtures no longer treat
      `tcrv-export-rvv-microkernel-c`, `rvv-explicit-*`,
      `rvv-*-runtime-callable-c-abi.v1`, or
      `rvv-*-runtime-callable-c-function.v1` as supported source artifact
      authority.
- [x] Remaining references to old RVV direct route strings are classified as
      deleted-route negative coverage, route-agnostic generic infrastructure,
      historical spec text, or residual rebuild debt.
- [x] No new direct-C, descriptor, wrapper, compatibility, or replacement route
      path is added.
- [x] Focused CMake/test targets covering RVV target support, target artifact
      export, runtime ABI tests touched by cleanup, and lit artifact export
      routes pass or fail only with truthfully recorded deletion gaps.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation
      before finish and after archive, final clean `git status --short`, and
      one coherent commit are produced if complete.

## Non-goals

- No new RVV lowering.
- No Common Lower-To-EmitC implementation.
- No replacement direct route registry.
- No descriptor wrapper or compatibility/legacy mode.
- No new selected-config resolver.
- No scalar rebuild.
- No ssh RVV runtime/correctness/performance evidence campaign.
- No Python compiler-core semantics.
- No broad repo audit beyond bounded RVV/artifact scans.
- No restoring old RVV direct C/header/object routes to satisfy stale tests.

## Minimal Evidence

- Before/after scans over touched RVV/artifact files for:
  `tcrv-export-rvv-microkernel-c`,
  `tcrv-export-rvv-microkernel-header`,
  `tcrv-export-rvv-microkernel-object`, `rvv-explicit`,
  `runtime-callable-c-source`, `rvv-runtime-callable-c-abi`,
  `runtime-callable-c-function`, `exportRVVMicrokernel`,
  `validateRVVMicrokernelSourceAuthority`,
  `resolveRVVMicrokernelSelectedConfigContractAuthority`,
  `RVVSelectedConfigContract`, `selected-config`, and `__riscv_`.
- Focused build/test coverage for target artifact export C++ tests, RVV
  target library build, runtime ABI callable-plan tests if touched, and lit
  target artifact/RVV microkernel deleted-route tests.
- Full `cmake --build build --target check-tianchenrv -j2` when coherent.
- `git diff --check`, `git diff --cached --check`, Trellis validation before
  finish and after archive.

## Technical Notes

- `.trellis/spec/architecture/design-boundaries.md` rejects descriptor-driven
  microkernel/exporter frameworks.
- `.trellis/spec/lowering-runtime/emitc-route.md` defines the current/future
  accepted compiler route as extension family ops -> EmitC -> C/C++ emitter.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` requires
  deleted direct RVV/scalar/dispatch C routes to be absent or fail closed until
  a materialized MLIR EmitC module route exists.
- `.trellis/spec/testing/mlir-testing-contract.md` requires deleting or
  rewriting tests whose only assertion is successful old direct source/header/
  object/bundle output.

## Completion Evidence

- Deleted production public header:
  `include/TianChenRV/Target/RVV/RVVMicrokernel.h`.
- Deleted production implementation:
  `lib/Target/RVV/RVVMicrokernel.cpp`.
- Removed `RVVMicrokernel.cpp` from `TianChenRVRVVTarget` CMake ownership.
- Removed RVV direct callable source identity from
  `FiniteBinaryRuntimeABIContractSpec`: `getRVVCallableIdentity()` and
  `rvv-*-runtime-callable-c-abi.v1` / `rvv-*-runtime-callable-c-function.v1`
  no longer exist in support production code.
- Rewrote `test/Target/TargetArtifactExportTest.cpp` so it no longer includes
  `RVVMicrokernel.h`, calls deleted RVV direct route registration/export APIs,
  or keeps direct RVV source/header/object helper candidate validation.
- Rewrote `test/Support/RuntimeABICallablePlanTest.cpp` so invocation-contract
  tests use deleted-route/unsupported markers rather than `RVVMicrokernel.cpp`
  or RVV direct runtime-callable ABI names.
- Deleted stale artifact fixtures:
  `missing-boundary.txt`, `missing-microkernel.txt`,
  `missing-runtime-abi-parameter-role.txt`, `multiple-supported-artifacts.txt`,
  `offload-spoof-rvv-source-route.txt`, `scalar-spoof-route.txt`, and
  `unsupported-artifact-kind.txt`.
- Rewrote `unknown-route-id.txt` to keep only a route-agnostic unknown-route
  test; it no longer carries RVV direct route or direct runtime ABI identity.
- Bounded final ref-scan found no production hits for `RVVMicrokernel.h`,
  `RVVMicrokernel.cpp`, `exportRVVMicrokernel`,
  `validateRVVMicrokernelSourceAuthority`,
  `resolveRVVMicrokernelSelectedConfigContractAuthority`,
  `rvv-runtime-callable-c-abi`, or standalone
  `rvv-i32-v*-runtime-callable-c-function.v1`. Remaining
  `tcrv-export-rvv-microkernel-c` hits are deleted-option negative coverage
  and one built-in registry absence assertion.
- Focused evidence passed:
  `cmake --build build --target TianChenRVRVVTarget TianChenRVSupport
  tianchenrv-target-artifact-export-test
  tianchenrv-runtime-abi-callable-plan-test -j2`,
  `./build/bin/tianchenrv-target-artifact-export-test`,
  `./build/bin/tianchenrv-runtime-abi-callable-plan-test`, and
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv .
  --filter='(ArtifactExport|RVVMicrokernel)'` from `build/test`.
- Full evidence passed:
  `cmake --build build --target check-tianchenrv -j2` passed 114/114 tests.
- `git diff --check` passed before finish.
