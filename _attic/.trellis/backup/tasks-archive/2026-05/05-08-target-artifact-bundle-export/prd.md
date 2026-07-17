# Target Artifact Bundle Export

## Goal

Materialize the selected target artifact handoff as a build-consumable bundle export. The compiler should write the selected source/header/object target artifacts plus a deterministic bundle index into an explicit output directory, using the same registry-derived candidate and runtime ABI metadata that emission manifests expose.

## Requirements

- Add a C++ Target-layer bundle export API derived from `collectTargetArtifactBundleRecords` and the registered target artifact exporters.
- Add a `tcrv-translate` front door with an explicit output-directory option for multi-file bundle export.
- For the bounded direct RVV i32-vadd microkernel path, export distinct source, header, and relocatable object artifacts when those registered routes are available.
- For the bounded RVV+scalar i32-vadd dispatch path, export distinct composite source, header, and relocatable object artifacts when those registered routes are available.
- Write a deterministic text index containing at least `artifact_kind`, `route`, `owner`, `runtime_abi_kind`, `runtime_abi_name`, and file name for each written artifact.
- Keep file names deterministic, safe, and derived from artifact route/kind metadata; do not embed local absolute paths, credentials, timestamps, random values, or `artifacts/tmp` paths in committed outputs.
- Preserve source/header/object front doors and route separation for existing single-artifact exports.
- Fail closed for unsupported or metadata-only selected paths and invalid output-directory handling; do not produce a fake complete bundle.
- Keep generic bundle traversal target-neutral. It may iterate registry-derived records and call registered exporters, but must not branch on RVV/scalar/IME/Sophgo/offload/vendor family semantics to decide which files exist.
- Keep the bundle index as handoff metadata only. It must not claim link/run success, correctness, runtime success, or performance.

## Acceptance Criteria

- [ ] Lit/FileCheck coverage proves direct RVV microkernel bundle export writes source/header/object files plus an index with stable route, artifact kind, owner, runtime ABI kind/name, evidence role or status, and file names.
- [ ] Lit/FileCheck coverage proves RVV+scalar dispatch bundle export writes source/header/object files plus an index without losing dispatch route identity.
- [ ] Existing source/header/object single-artifact front doors remain covered and route-separated.
- [ ] Guard coverage proves unsupported or metadata-only selected paths fail without a fake complete bundle.
- [ ] Guard coverage proves missing/invalid output-directory handling emits deterministic diagnostics.
- [ ] C++ tests cover new bundle selection/file naming helper behavior if introduced.
- [ ] `git diff --check`, `cmake --build build --target tcrv-opt tcrv-translate -j2`, and `cmake --build build --target check-tianchenrv -j2` pass.

## Definition Of Done

- Active C++/MLIR implementation is committed as one coherent compiler commit.
- Trellis task is validated where applicable and archived before final report.
- Worktree is clean after commit.
- No new RVV runtime/correctness/performance claim is made without `ssh rvv`.

## Technical Approach

Add a target/export bundle materialization API next to existing `TargetArtifactExport` routing. It should collect candidates, collect registry-derived bundle records, select writable records for each kernel, invoke the corresponding single or composite exporter into files under the requested output directory, and only write the final index after all artifacts have been successfully produced. File names should be sanitized from route and artifact kind, with deterministic collision handling. The translate tool will register a normal MLIR translation plus an LLVM command-line output-directory option and run the same execution-plan coherence preflight used by current generic artifact front doors.

## Out Of Scope

- No new generic RVV lowering, arbitrary kernel emission, IME/offload implementation, link/run harness, benchmark, automatic hardware probing, or performance claim.
- No Python implementation of compiler decisions or bundle export logic.
- No changes to `tcrv.exec` compute boundary.

## Technical Notes

- Required repo inspection completed before code edits: `pwd`, `git status --short`, `git log --oneline -12`, `git show --name-status --stat --oneline --decorate HEAD --`, `.trellis/.current-task`, `.trellis/tasks`, and the user-listed specs/source/tests.
- Relevant specs: `.trellis/spec/index.md`, `.trellis/spec/core-dialect/tcrv-exec-contract.md`, `.trellis/spec/lowering-runtime/emission-runtime-contract.md`, `.trellis/spec/testing/mlir-testing-contract.md`, `.trellis/spec/extension-plugins/rvv-plugin.md`.
