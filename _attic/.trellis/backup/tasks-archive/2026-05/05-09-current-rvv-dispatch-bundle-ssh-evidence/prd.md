# current rvv dispatch bundle ssh evidence

## Goal

Recover and complete bounded real `ssh rvv` evidence for the current-HEAD
role-bound RVV+scalar i32-vadd dispatch target-artifact bundle external ABI
path.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current HEAD at task start is `8a05665315e7c57bf9085f3344299cdfb6536806`
  (`8a05665 fix: bind direct callable abi by role`).
- The previous attempt was interrupted by Codex stream/model availability, not
  by a repository build/test failure.
- Existing ignored bundle evidence may exist under `artifacts/tmp`, but any
  evidence with `git_sha` older than current HEAD must not be reused.
- The intended evidence path is the existing
  `scripts/rvv_scalar_dispatch_e2e.py` bridge in target-artifact-bundle mode,
  preferably through the plan-and-export bundle front door.

## Requirements

- Rebuild current tools under `artifacts/tmp/tianchenrv-build`.
- Run local `check-tianchenrv` before claiming evidence.
- Run dry-run target-artifact-bundle evidence for current HEAD.
- Run real `ssh rvv` target-artifact-bundle evidence for current HEAD unless an
  external host/credential condition blocks it.
- Evidence JSON must record current full `git_sha`, correct `mode`, `status =
  success`, `bundle_export_mode = plan-and-export-target-artifact-bundle`, and
  sanitized command logs.
- The bundle index must expose source, header, and object records in one
  compiler-emitted external ABI component group.
- Runtime ABI parameters must include the role-typed dispatch signature:
  `lhs-input-buffer`, `rhs-input-buffer`, `output-buffer`,
  `runtime-element-count`, and `dispatch-availability-guard`.
- Real ssh evidence must show both the source-built caller path and the bundle
  object caller path observed
  `tcrv_rvv_scalar_i32_vadd_bundle_external_abi_ok`.
- Keep generated artifacts, command logs, objects, copied source/header files,
  evidence JSON blobs, hashes, and build outputs out of git.
- If no code fix is needed, commit only the Trellis archive summarizing the
  evidence.

## Out Of Scope

- No Python implementation of compiler internals or runtime ABI semantics.
- No broad runtime ABI, target export, dispatch synthesis, plugin registry,
  ODS, or lowering rewrite.
- No generic RVV lowering, arbitrary-kernel correctness, dynamic runtime
  integration, broad RVV support, or performance claim.
- No use of stale pre-`8a05665` evidence as current role-bound evidence.

## Acceptance Criteria

- [ ] `git diff --check` passes.
- [ ] CMake configure succeeds with LLVM/MLIR under `/usr/lib/llvm-20`.
- [ ] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
      passes.
- [ ] Dry-run bundle evidence passes with current HEAD in `evidence.json`.
- [ ] Real `ssh rvv` bundle evidence passes with current HEAD in `evidence.json`,
      or a precise external blocker is reported without faking evidence.
- [ ] Evidence JSON fields and bounded claim scope are validated before final
      report.
- [ ] Trellis task is validated, archived, and not left active at top level.
- [ ] Worktree is clean after one coherent commit.

## Technical Notes

- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Evidence root:
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/<run-id>/`.
- Expected commands use:
  `scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle
  --use-plan-and-export-bundle-front-door`.
