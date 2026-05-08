# RVV Target-Artifact Bundle External ABI Evidence Bridge

## Goal

Move the selected RVV+scalar i32-vadd dispatch target artifact bundle from local build-consumable files to bounded external compile/link/run evidence on the RVV host. The evidence bridge must run the real compiler pipeline, export the registry-derived bundle, parse `tianchenrv-target-artifact-bundle.index`, compile an external C caller against the generated header and generated object on `ssh rvv`, and record sanitized evidence under `artifacts/tmp`.

## Requirements

- Extend `scripts/rvv_scalar_dispatch_e2e.py` with a bundle-export mode, preferably `--use-target-artifact-bundle`.
- The bridge must invoke:
  - `tcrv-opt <input> --tcrv-execution-planning-pipeline`
  - `tcrv-translate --tcrv-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=<fresh-dir>`
- Parse only the stable bundle index file name directly. Discover generated artifact file names from the index.
- Validate index fields needed for external ABI evidence:
  - file name
  - artifact kind
  - route
  - owner
  - runtime ABI kind/name
  - selected dispatch surface and component selected variants/roles
  - evidence role
- Generate a small external C caller that includes the generated dispatch header and links the generated dispatch object.
- The caller must exercise deterministic inputs through both:
  - scalar fallback branch with guard `0`
  - RVV branch with guard `1`
- Copy only the generated header, generated object, and caller to `ssh rvv`.
- Record sanitized evidence under `artifacts/tmp/rvv_bundle_e2e` or an equivalent explicit `artifacts/tmp` path.
- Dry-run/self-test behavior must validate command planning, index parsing, file discovery, caller generation, and secret redaction without claiming runtime success.
- Do not modify core compiler implementation unless an actual C++ bundle-export bug blocks this evidence path.

## Acceptance Criteria

- Local lit coverage under `test/Scripts` covers bundle mode self-test/dry-run behavior.
- Tests cover index parsing and malformed or missing index fields strongly enough to prevent fake runtime claims from incomplete metadata.
- Dry-run output and evidence explicitly do not claim `runtime_success`.
- Existing source/header/object single-artifact and bundle export front doors remain unchanged.
- Real `ssh rvv` mode either records compile/link/run success for the bounded external dispatch ABI or records the exact non-secret blocker.

## Out Of Scope

- New lowering, generic RVV lowering, arbitrary kernels, performance tuning, benchmarks, IME, AME, Sophgo, offload, or vendor implementation.
- Python implementation of compiler decisions, target selection, route discovery beyond parsing the exported index, runtime ABI modeling beyond external caller generation from emitted metadata, lowering, emission, or plugin logic.
- Runtime/correctness claims from dry-run, local build, local lit, object creation, or static bundle inspection.

## Technical Notes

- Primary specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
- Existing implementation surfaces:
  - `scripts/rvv_scalar_dispatch_e2e.py`
  - `include/TianChenRV/Target/TargetArtifactExport.h`
  - `lib/Target/TargetArtifactExport.cpp`
  - `tools/tcrv-translate/tcrv-translate.cpp`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
- Existing bundle index records deterministic `artifact[n]` sections with file name, artifact kind, route, owner, runtime ABI kind/name, evidence role, and component selected variants/roles.
- Existing dispatch object contains embedded RVV callable, scalar callable, and dispatcher symbols and no hidden `main`.
