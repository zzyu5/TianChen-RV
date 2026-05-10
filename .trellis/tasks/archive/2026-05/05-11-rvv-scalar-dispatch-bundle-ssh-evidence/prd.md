# RVV+scalar dispatch bundle ssh rvv compile/run evidence

## Goal

Prove one bounded RVV+scalar dispatch target artifact bundle, preferably
`i32-vmul`, from the real `tcrv-translate
--tcrv-plan-and-export-target-artifact-bundle` front door through compile,
link, and run on the real `ssh rvv` host. The evidence must consume the
compiler-emitted bundle index to find source, header, and object artifacts,
exercise both explicit dispatch guard branches, and avoid any performance or
generic RVV backend claim.

## Requirements

- Use the existing marked MLIR input
  `test/Target/TargetArtifactBundleExport/plan-linalg-i32-vmul-and-export-target-artifact-bundle.mlir`
  as the primary non-legacy family.
- Run the actual planning/export front door:
  `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle`, not a
  hand-written artifact shortcut.
- Use the generated target artifact bundle index to locate the dispatch C
  source, C header, and RISC-V ELF relocatable object records.
- Validate that source, header, and object records share the same component
  group, external ABI identity, runtime ABI kind/name, selected dispatch
  components, and runtime ABI parameter signature.
- Generate an external C caller from the compiler-emitted header signature and
  exercise both `rvv_available=0` scalar fallback and `rvv_available=1` RVV
  dispatch-case branches with bounded runtime element counts.
- On `ssh rvv`, compile the external caller, compile the generated dispatch
  source into an object, link/run the source-built path, then link/run the
  indexed relocatable object path when present.
- Record evidence JSON with remote architecture/toolchain facts, selected
  family and surface, artifact paths/hashes, branch coverage, compile/link/run
  success fields, and bounded command summaries.
- Reject or fail closed if the generated dispatch artifact, header, object,
  runtime ABI parameter roles, family route metadata, or branch caller shape
  mismatches `i32-vmul`.
- Keep Python changes, if any, limited to runner/evidence tooling. Do not
  implement compiler internals, artifact route semantics, dialects, passes,
  plugin registry, lowering, or emission in Python.

## Acceptance Criteria

- [x] Trellis task exists and records this PRD plus relevant spec context.
- [x] Focused dry-run/self-test coverage for
      `rvv-scalar-dispatch-bundle-e2e` still passes.
- [x] A real `ssh rvv` command is run for `i32-vmul` with
      `--use-target-artifact-bundle` and
      `--use-plan-and-export-bundle-front-door`.
- [x] The produced evidence JSON records
      `remote_compile_succeeded=true`,
      `remote_link_succeeded=true`,
      `remote_run_succeeded=true`, selected family `i32-vmul`, selected
      dispatch surface, both `rvv_available=0` and `rvv_available=1` branches,
      and no throughput/latency/performance fields.
- [x] The evidence includes remote architecture/toolchain facts and artifact
      paths or hashes for the generated source/header/object/caller surfaces.
- [x] `git diff --check` passes.
- [x] If source changed, focused build targets pass:
      `tcrv-opt`, `tcrv-translate`, and any touched C++ test target.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
      passes after focused checks.
- [x] Trellis task validation passes before finish/archive.

## Definition of Done

- The bounded `i32-vmul` dispatch bundle evidence path is either completed on
  real `ssh rvv` and archived, or the task remains open with the exact blocker
  command and first failing diagnostic.
- Any source changes are narrowly scoped to runner/evidence tooling or the
  relevant C++/MLIR target artifact contract; no broad family matrix or
  performance benchmarking is added.
- `tcrv.exec` remains compute-free; generic core/export code avoids RVV/scalar
  semantic branches; parameter layering stays IR-backed or target-export
  ABI-owned as specified.
- One coherent commit records the task, validation, and any implementation
  changes if the task is complete.

## Out of Scope

- Broad dispatch family matrix or generic RVV backend maturity claim.
- Throughput, latency, ratio, benchmark, or performance evidence.
- Arbitrary vector lowering, automatic hardware probing in generated dispatch
  code, or moving dispatch semantics into `tcrv.exec`.
- Committing generated artifacts under `artifacts/tmp` or raw remote logs.
- Faking or substituting local dry-run evidence for real `ssh rvv` evidence.

## Technical Approach

Use the existing `scripts/rvv_scalar_dispatch_e2e.py` bundle mode if current
repo evidence confirms it already enforces the bundle-index contract. The
primary command will run with `--arithmetic-family=i32-vmul`,
`--use-target-artifact-bundle`, and
`--use-plan-and-export-bundle-front-door`. If the script or emitted bundle
fails to meet the PRD, repair only the runner/evidence path or the minimal
target artifact contract needed for this module-sized proof.

## Technical Notes

- Repository root at task creation:
  `/home/kingdom/phdworks/TianchenRV`.
- Initial HEAD at task creation:
  `b8d7463 feat(dispatch): register exporters via plugin bundle`.
- Initial worktree state: clean.
- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/validation/experiment-reference.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`.
- Prior task read:
  `.trellis/tasks/archive/2026-05/05-11-rvv-scalar-dispatch-plugin-owned-exporter-bundle/prd.md`.
- Key existing surfaces inspected:
  `scripts/rvv_scalar_dispatch_e2e.py`,
  `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`,
  `test/Target/TargetArtifactBundleExport/plan-linalg-i32-vmul-and-export-target-artifact-bundle.mlir`,
  `include/TianChenRV/Target/RVVScalarDispatch.h`,
  `lib/Target/Builtin/RVVScalarDispatch.cpp`, and
  `lib/Target/TargetArtifactExport.cpp`.

## Result

- No compiler or runner source change was required. The existing bundle
  evidence runner already consumed the plan-and-export bundle front door and
  enforced the generated bundle-index contract for `i32-vmul`.
- Real ssh evidence command:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vmul --input test/Target/TargetArtifactBundleExport/plan-linalg-i32-vmul-and-export-target-artifact-bundle.mlir --run-id codex-rvv-scalar-i32-vmul-bundle-ssh --overwrite --timeout 120`.
- Evidence JSON:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-rvv-scalar-i32-vmul-bundle-ssh/evidence.json`.
- Evidence outcome: `mode=ssh`, `status=success`,
  `ssh_evidence_verified=true`, remote architecture `riscv64`, remote clang
  `Ubuntu clang version 18.1.3 (1ubuntu1)`,
  `remote_compile_succeeded=true`, `remote_link_succeeded=true`, and
  `remote_run_succeeded=true`.
- The generated external caller exercised both `rvv_available=0` scalar
  fallback and `rvv_available=1` RVV dispatch-case branches at bounded runtime
  element counts `7` and `16`.
- Evidence JSON structural assertion confirmed no throughput, latency, or
  performance metric field names. The preserved compiler metadata may still
  contain negative wording such as "not ... performance evidence", which is not
  a performance claim.
