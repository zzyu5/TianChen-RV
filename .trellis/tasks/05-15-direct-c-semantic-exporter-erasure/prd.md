# delete direct C semantic exporters

## Goal

Delete or fail-close the direct C semantic exporter authority for
runtime-callable RVV, scalar, and RVV+scalar dispatch source routes. This is a
Wrong Logic Deletion Campaign round: remove production/default paths that
synthesize executable C compute bodies from selected records, family metadata,
or route records before a real materialized MLIR EmitC module becomes the
source authority.

## Why Now

Commit `a7a6643` completed the Offload descriptor-only route deletion and left
the worktree clean. The next active deletion surface is direct C semantic
export: `RVVMicrokernel.cpp`, `ScalarMicrokernel.cpp`, and
`RVVScalarDispatch.cpp` still publish `runtime-callable-c-source` artifact
routes and target translate helpers that build C bodies from target-owned
records. `RVVSmokeProbe.cpp` also prints a standalone RVV intrinsic C probe;
this task must keep it only if it remains classified as harness/toolchain probe
support rather than kernel semantic route authority.

## Scope

- Remove or fail-close production RVV selected binary runtime-callable source
  routes, including direct target artifact exporters, direct translate routes,
  and source/object routes that derive executable C from
  `RVVMicrokernelRecord` / selected metadata.
- Remove or fail-close production scalar fallback runtime-callable source
  routes, including scalar plugin emission-plan support and scalar target
  artifact exporters that derive executable C from scalar family records.
- Remove or fail-close RVV+scalar dispatch runtime-callable source/object and
  self-check source/object routes that compose generated RVV/scalar C bodies
  and then synthesize a dispatcher body.
- Rewrite target-support bundle route metadata registration so deleted direct
  C source/object routes are not advertised as active built-in target artifact
  routes.
- Delete or rewrite tests and fixtures that protect raw emitted RVV/scalar/
  dispatch C loops, RVV intrinsic strings, scalar arithmetic bodies,
  embedded callable sources, or direct source/object bundle success as legal
  compiler output.
- Update directly stale spec/comment wording so it describes the source/object
  route deletion or fail-closed gap rather than active production authority.
- Keep non-semantic metadata, route ids as historical diagnostics where needed,
  header-only ABI surfaces, bundle mechanics, artifact readers, and smoke-probe
  harness code only when they do not generate or claim compute semantics.

## Acceptance Criteria

- [ ] No production route synthesizes RVV, scalar, or RVV+scalar dispatch
      compute C bodies directly from selected metadata, family records, route
      records, or descriptor-like selected records.
- [ ] RVV and scalar selected emission planning no longer advertises active
      runtime-callable C source support for these deleted direct C semantic
      routes; unsupported paths fail closed with bounded diagnostics.
- [ ] Target artifact exporter registration no longer publishes deleted RVV,
      scalar, or dispatch runtime-callable source/object routes as active
      production routes.
- [ ] Direct target translate helper routes that would print direct C semantic
      source/object outputs are removed or fail closed.
- [ ] Tests that only protected old emitted C bodies, intrinsic strings,
      scalar arithmetic loops, dispatcher branches, embedded callable sources,
      or source/object bundle outputs are deleted or rewritten to expect
      absence/fail-closed behavior.
- [ ] `RVVSmokeProbe.cpp` remains only as standalone harness/toolchain probe
      support, or is deleted if it acts as kernel semantic route authority.
- [ ] No common Lower-To-EmitC rebuild, replacement C generator, wrapper,
      compatibility/quarantine/legacy mode, descriptor restoration, Python
      compiler semantics, or new extension family work is added.
- [ ] If deletion exposes build/test failures, the exact failures are reported
      as missing future common EmitC/source architecture gaps and the old
      direct source generator is not restored.
- [ ] Bounded ref-scan classifies remaining direct-C/descriptor/core-branch
      hits in touched target/plugin/test/spec areas.
- [ ] `git diff --check`, `git diff --cached --check`, Trellis validation, and
      focused build/lit/C++ checks are run when the tree remains buildable.

## Non-Goals

- No Common Lower-To-EmitC implementation.
- No new materialized EmitC route.
- No RVV, scalar, or dispatch rebuild.
- No extension-family expansion or plugin construction template.
- No capability model feature.
- No `ssh rvv` evidence campaign.
- No replacement C generator.
- No wrapper, quarantine, compatibility, or legacy mode around direct C source
  printers.
- No renaming direct C semantics to a new active token.
- No descriptor route restoration.
- No Python compiler-core logic.
- No docs-only milestone unless docs/spec wording is directly tied to route
  deletion.

## Minimal Evidence

- Focused inventory over `printMicrokernelSource`, `printDispatchSource`,
  `runtime-callable-c-source`, `standalone-c-source`, `__riscv_`,
  `emitc_c_source_authority`, `translateToCpp`, route manifests, target
  artifact exporter registration, target translate route registration, and
  plugin selected emission-plan support in RVV/scalar/dispatch areas.
- Bounded ref-scan over touched `lib/Plugin/RVV`, `lib/Plugin/Scalar`,
  `lib/Target/RVV`, `lib/Target/Scalar`, `lib/Target/Builtin`, focused
  `test/Target` directories, and directly edited specs.
- Focused C++/lit checks for touched plugin/target/export route behavior when
  buildable.
- `git diff --check`, `git diff --cached --check`, Trellis validation before
  finish and after archive, final clean worktree, and one coherent commit if
  complete.

## Technical Notes

- `.trellis/spec/index.md` defines descriptor-driven computation as invalid
  long-term architecture and keeps the implementation stack in C++/MLIR/
  LLVM/TableGen/CMake/lit/FileCheck.
- `.trellis/spec/architecture/design-boundaries.md` explicitly rejects a
  descriptor-driven microkernel/exporter framework.
- `.trellis/spec/lowering-runtime/emitc-route.md` says direct handwritten C
  string emission is not the architecture; generated C should come from EmitC
  operations or a bounded legacy helper that is not used as new-extension
  template.
- `.trellis/spec/lowering-runtime/emission-runtime-contract.md` still contains
  active RVV/scalar/dispatch source/header/object route wording and must be
  corrected where touched by this owner.
- `.trellis/spec/testing/mlir-testing-contract.md` still asks tests to protect
  generated direct C outputs for RVV/scalar/dispatch and must be rewritten for
  fail-closed behavior where this deletion removes the route.
- Previous deletion round:
  `.trellis/tasks/archive/2026-05/05-15-offload-descriptor-only-route-deletion/prd.md`.
- Initial source inventory found active direct source exporters in
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Target/Scalar/ScalarMicrokernel.cpp`, and
  `lib/Target/RVV/RVVScalarDispatch.cpp`; plugin selected emission support in
  `lib/Plugin/RVV/RVVBinarySelectedEmissionPlanning.cpp` and
  `lib/Plugin/Scalar/ScalarExtensionPlugin.cpp`; bundle/registration surfaces
  in `lib/Target/RVV/RVVTargetSupportBundle.cpp` and
  `lib/Target/Builtin/BuiltinTargetArtifactExporters.cpp`.
