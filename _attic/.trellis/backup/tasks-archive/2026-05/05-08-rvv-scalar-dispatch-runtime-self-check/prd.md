# RVV scalar dispatch runtime self-check export

## Goal

Add an explicit RVV+scalar i32-vadd dispatch self-check C export that exercises
the existing generated dispatcher through both the RVV and scalar branches, then
collect real `ssh rvv` compile/run evidence for that generated source. This is
the next runtime invocation slice after the existing library-style RVV, scalar,
and RVV+scalar dispatch source exports.

## What I Already Know

* Repo root is `/home/kingdom/phdworks/TianchenRV`.
* Current HEAD is `80a0b9f feat: structure runtime ABI parameters`.
* The worktree was clean before this task.
* There is no active Trellis task at session start.
* Latest `repo_audit.md` / `review_input.md` are under
  `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0025-20260507T165811Z/`.
* The latest completed round added structured runtime ABI parameter metadata
  for bounded RVV, scalar, and RVV+scalar i32-vadd source exports.
* Remaining risk called out by the latest round is object generation, linking,
  runtime invocation, and host/runtime wrapper integration.
* The existing `--tcrv-export-rvv-scalar-i32-vadd-dispatch-c` export emits a
  library-style C dispatcher with no `main`, no self-check harness, and no
  runtime success claim.
* Existing `--tcrv-execution-planning-pipeline` can produce the RVV and scalar
  callable sides for `test/Target/EmissionManifest/emission-manifest-pipeline.mlir`.

## Requirements

* Keep the primary implementation in C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck.
* Keep `tcrv.exec` compute-free; all dispatch invocation code remains
  target-owned RVV+scalar export behavior.
* Preserve the existing default library-style dispatcher export exactly as the
  non-harness artifact: no hidden `main`, no self-check success marker, no
  runtime/correctness claim.
* Add a separate explicit self-check export for the bounded RVV+scalar i32-vadd
  dispatcher.
* The self-check export must reuse the same selected-path, lowering-boundary,
  emission-plan, artifact-route, and structured runtime ABI validation as the
  existing dispatcher export.
* The generated harness must call the dispatcher twice: once with
  `rvv_available = 0` to exercise the scalar fallback branch, and once with
  `rvv_available = 1` to exercise the RVV branch.
* The harness must validate deterministic i32 addition over bounded local
  arrays and return non-zero on failure.
* The generated source may print a bounded success marker, but must not contain
  benchmark, throughput, latency, raw logs, credentials, URLs, absolute paths,
  or artifact-directory claims.
* Update README/specs only for durable behavior: the explicit dispatch harness
  is evidence tooling for the bounded dispatcher, not generic RVV lowering,
  arbitrary runtime integration, object generation pipeline, correctness beyond
  this harness, or performance evidence.
* Add focused lit/FileCheck coverage for the new public export and for
  pipeline-to-self-check generation.
* Run real `ssh rvv` compile/run evidence if the remote host is reachable; any
  claim must be bounded to the generated RVV+scalar i32-vadd dispatch harness.

## Acceptance Criteria

* [ ] `tcrv-translate` exposes a separate
  `--tcrv-export-rvv-scalar-i32-vadd-dispatch-self-check-c` translation.
* [ ] The existing `--tcrv-export-rvv-scalar-i32-vadd-dispatch-c` output remains
  library-style and does not include `main` or self-check success markers.
* [ ] The new self-check export embeds the existing dispatcher and a bounded
  `main` that validates both scalar fallback and RVV branches.
* [ ] Focused FileCheck coverage proves the harness structure and pipeline
  compatibility.
* [ ] `git diff --check` passes.
* [ ] CMake configure/build checks pass with the local MLIR/LLVM toolchain.
* [ ] `check-tianchenrv` passes.
* [ ] Real `ssh rvv` compile/run evidence is collected or an exact blocker is
  reported.
* [ ] One coherent commit is created and the worktree is clean.

## Definition of Done

* Code and tests are scoped to the bounded RVV+scalar i32-vadd dispatch runtime
  invocation slice.
* Durable documentation is updated only where behavior changes.
* No Python compiler-internal implementation is added.
* No broad smoke matrix, dashboard, or evidence packaging is added.
* The Trellis task is validated and archived after implementation.

## Out of Scope

* No generic high-level lowering.
* No object-file artifact route or linker pipeline.
* No dynamic loading, automatic hardware probing, or host runtime library.
* No generic RVV runtime ABI integration beyond this explicit dispatcher
  harness.
* No performance benchmark or throughput/latency claim.
* No broad negative fixture matrix beyond focused export validation.
* No changes to `tcrv.exec` computation semantics.

## Technical Approach

Add a second RVV+scalar dispatch target export mode in the existing target-owned
`lib/Target/Builtin/RVVScalarDispatch.cpp` implementation. The default mode
continues to emit only the library-style dispatcher. The explicit self-check
mode appends a bounded `main` that initializes local inputs, calls the generated
dispatcher through both guard values, checks outputs, and emits one bounded
success marker.

Expose the mode through `include/TianChenRV/Target/RVVScalarDispatch.h` and
`tools/tcrv-translate/tcrv-translate.cpp`. Keep validation centralized in the
existing `collectDispatchPair` / callable-source build path so the harness
cannot bypass selected-path or runtime ABI metadata checks.

## Decision (ADR-lite)

**Context**: The repo already has bounded RVV/scalar callable source artifacts
and a host dispatcher source artifact, but no target-owned runtime invocation
surface for the composed dispatcher.

**Decision**: Add one explicit dispatcher self-check export rather than changing
the default dispatcher artifact or adding a generic object/link pipeline.

**Consequences**: This provides real runtime invocation evidence for the current
bounded dispatcher while preserving the default library artifact contract. It
does not solve object generation, dynamic runtime integration, or arbitrary RVV
lowering.

## Technical Notes

* Latest audit:
  `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0025-20260507T165811Z/repo_audit.md`.
* Latest review input:
  `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0025-20260507T165811Z/review_input.md`.
* Relevant specs:
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Likely implementation files:
  `include/TianChenRV/Target/RVVScalarDispatch.h`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `test/Target/RVVScalarDispatch/rvv-scalar-i32-vadd-dispatch-c.mlir`,
  README and lowering/runtime specs if behavior changes.
