# Linalg i32m2 vsub RVV scalar dispatch ssh external ABI evidence

## Goal

Produce one bounded real `ssh rvv` executable evidence bridge for the
compiler-produced marked-linalg `i32-vsub` + `i32m2` RVV+scalar dispatch
target artifact bundle. The route must start from the linalg/front-door input,
use the existing C++ planning/export pipeline, generate the source/header/object
bundle and an external C caller, then compile/link/run that caller on the RVV
host.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting inspected state is clean at HEAD `2b50d9f`.
- There was no current Trellis task; this task is new and must not reopen the
  archived `05-10-bounded-linalg-i32-binary-rvv-scalar-artifact-pipeline`
  task.
- The archived front-door task converged local compiler/lit proof for
  `i32-vsub` + `i32m2` at commit `2b50d9f`, but intentionally did not collect
  new `ssh rvv` runtime evidence.
- `scripts/rvv_scalar_dispatch_e2e.py` already has a target-artifact-bundle
  mode, plan-and-export front door mode, `i32-vsub` family selection,
  `i32m2` vector shape selection, bundle index parsing, external caller
  generation, and remote source/object compile-link-run orchestration.
- The default `i32-vsub` + `i32m2` plan-and-export input is
  `test/Target/TargetArtifactBundleExport/plan-linalg-i32m2-vsub-and-export-target-artifact-bundle.mlir`.

## Requirements

- Keep compiler behavior in C++ / MLIR / LLVM / TableGen / CMake / lit /
  FileCheck. Python may only orchestrate tools, remote copy/compile/run,
  artifact parsing, and sanitized evidence.
- Do not implement compiler IR, plugin decisions, capability modeling,
  selection, lowering, emission, or runtime ABI semantics in Python.
- Drive the existing C++ route equivalent to:

  ```text
  tcrv-translate --tcrv-plan-and-export-target-artifact-bundle
    --tcrv-target-artifact-bundle-output-dir=<bundle-dir>
    test/Target/TargetArtifactBundleExport/plan-linalg-i32m2-vsub-and-export-target-artifact-bundle.mlir
  ```

  through the runner options:

  ```text
  --use-target-artifact-bundle
  --use-plan-and-export-bundle-front-door
  --arithmetic-family=i32-vsub
  --vector-shape=i32m2
  ```

- Produce artifacts under
  `artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/<run-id>/`, including the
  generated dispatch source, generated header, generated relocatable object,
  generated external caller, command summary, hashes, logs, and evidence JSON.
- On `ssh rvv`, compile/link/run the generated external caller against both:
  the generated dispatch source path; and the generated bundle object path.
- Evidence must cover both `rvv_available=0` scalar fallback and
  `rvv_available=1` RVV dispatch path, and runtime element counts `n=7` and
  `n=16`.
- Evidence JSON must remain sanitized and include bounded claim scope, git SHA,
  selected family, vector shape, selected RVV config/march/mabi when available,
  bundle component group, route ids, source/header/object/caller paths and
  hashes, runtime ABI signature, branch/count coverage, remote host facts,
  command summaries, and clear success markers for the remote runs.
- The runtime/correctness claim is bounded only to this generated
  `i32-vsub` + `i32m2` RVV+scalar dispatch bundle external ABI handoff.

## Acceptance Criteria

- `git diff --check` passes.
- Tools needed for the evidence path are built, at minimum:

  ```text
  cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2
  ```

- Focused local helper coverage passes when relevant:

  ```text
  python3 scripts/rvv_scalar_dispatch_e2e.py --self-test
  python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --vector-shape=i32m2 --run-id <dry-run-id> --overwrite
  ```

- Real bounded evidence command passes against `ssh rvv` with a unique run id:

  ```text
  python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --vector-shape=i32m2 --run-id <run-id> --overwrite
  ```

- A read-only local evidence JSON check asserts:
  `ssh_evidence` is present, top-level `status` is `success`,
  `pass_fail_result` is `pass`, both remote source and bundle-object markers
  were observed, branches include `rvv_available=0` and `rvv_available=1`,
  runtime counts include `7` and `16`, vector shape is `i32m2`, arithmetic
  family is `i32-vsub`, and obvious secret/token/password text is absent.
- `python3 ./.trellis/scripts/task.py validate <task-or-archive-path>` passes
  before finishing or archiving.
- If source or test files change, run focused lit/tests for the changed surface
  and, if practical, `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`.
- If no source/test code changes are needed, the task may finish with Trellis
  records plus generated untracked evidence artifacts only; generated
  `artifacts/tmp` outputs are not committed.

## Out Of Scope

- No new compiler lowering semantics, RVV dialect semantics, selection policy,
  target artifact semantics, runtime ABI invention, or dynamic runtime
  integration.
- No hand-authored selected-dispatch fixture as the evidence source when the
  linalg/front-door bundle can produce the artifact.
- No performance or throughput measurements.
- No broad smoke matrices or generic RVV/linalg correctness claims.
- No IME, AME, Sophgo/offload, vendor, or future-plugin path work.
- No committed `artifacts/tmp` outputs, raw remote logs, secrets, tokens,
  credential paths, or transient build products.

## Technical Notes

- Specs read for this task:
  - `.trellis/spec/index.md`
  - `.trellis/spec/validation/experiment-reference.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
- Prior task reference read:
  - `.trellis/tasks/archive/2026-05/05-10-bounded-linalg-i32-binary-rvv-scalar-artifact-pipeline/prd.md`
- Primary implementation/test surfaces inspected:
  - `test/Target/TargetArtifactBundleExport/plan-linalg-i32m2-vsub-and-export-target-artifact-bundle.mlir`
  - `test/Scripts/rvv-scalar-dispatch-bundle-e2e.test`
  - `scripts/rvv_scalar_dispatch_e2e.py`
  - `lib/Target/Builtin/RVVScalarDispatch.cpp`
  - `lib/Target/TargetArtifactExport.cpp`
  - `tools/tcrv-translate/tcrv-translate.cpp`

## Current Status

Completed. The existing compiler-produced linalg/front-door bundle route was
able to generate the `i32-vsub` + `i32m2` RVV+scalar dispatch source/header/
object bundle and external caller, then compile/link/run both source-built and
bundle-object handoffs on `ssh rvv`.

Implementation changes:
- Added explicit `runtime_success: true` to real ssh evidence JSON for both
  target-artifact-bundle and legacy self-check modes after the remote
  compile/link/run sequence succeeds. Dry-run evidence still does not claim
  runtime success.

Evidence command:

```text
python3 scripts/rvv_scalar_dispatch_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --vector-shape=i32m2 --run-id codex-i32m2-vsub-ssh-20260510T0001 --overwrite --timeout 120
```

Evidence artifact directory:

```text
artifacts/tmp/tianchenrv-rvv-dispatch-bundle-e2e/codex-i32m2-vsub-ssh-20260510T0001/
```

Evidence result:
- top-level `status = success`, `pass_fail_result = pass`, and
  `runtime_success = true`;
- `ssh_evidence.runtime_success = true`;
- `source_stdout_marker_observed = true`;
- `bundle_object_stdout_marker_observed = true`;
- `source_run_exit_code = 0`;
- `bundle_object_run_exit_code = 0`;
- remote architecture fact is `riscv64`;
- branch coverage records `rvv_available=0` and `rvv_available=1`;
- runtime counts record `7` and `16`;
- selected family is `i32-vsub`;
- selected vector shape is `i32m2`;
- claim scope is bounded to this generated RVV+scalar `i32-vsub` target
  artifact bundle external caller correctness only.

Validation completed:
- `git diff --check`
- `python3 -m py_compile scripts/rvv_scalar_dispatch_e2e.py`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-05-10-linalg-i32m2-vsub-rvv-scalar-ssh-external-abi-evidence`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt tcrv-translate -j2`
- `python3 scripts/rvv_scalar_dispatch_e2e.py --self-test`
- focused dry-run:
  `python3 scripts/rvv_scalar_dispatch_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i32-vsub --vector-shape=i32m2 --run-id codex-i32m2-vsub-dry-run --overwrite`
- focused lit:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-scalar-dispatch-bundle-e2e'`
  from `artifacts/tmp/tianchenrv-build/test` (1/1 selected test passed)
- real `ssh rvv` evidence command above
- read-only evidence JSON assertion for runtime success, branch/count coverage,
  selected family/shape, source/header/object/caller presence, remote arch,
  and absence of obvious secret-like text
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  (172/172 passed)

Spec update judgment:
- No `.trellis/spec/` update is needed. The long-term contract already requires
  bounded real `ssh rvv` evidence and sanitized evidence fields; this task only
  made the runner JSON expose an explicit success boolean for an already
  supported remote evidence path.

Post-commit note:
- The same run id should be rerun once after the final commit with `--overwrite`
  so the untracked evidence JSON `git_sha` records the final committed tree.
