# Front-door RVV binary frontend bundle with ssh rvv execution evidence

## Goal

Complete one coherent evidence path for a representative bounded RVV binary
kernel that starts from marked Linalg frontend input, uses the public
`--tcrv-lower-linalg-rvv-binary-to-exec` / plan-and-export bundle front door,
exports compiler/target-owned source/header/object bundle artifacts, and
compiles, links, runs, and validates the generated runtime-callable artifact on
the real `ssh rvv` target.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial state for this round is clean on `main` at
  `f54a2b3 feat(transforms): expose RVV binary frontend pass`.
- `.trellis/.current-task` was absent. A new task was created for this round:
  `.trellis/tasks/05-10-front-door-rvv-binary-frontend-bundle-ssh-evidence`.
- The previous task is archived at
  `.trellis/tasks/archive/2026-05/05-10-bounded-linalg-rvv-binary-frontend-pass`
  and is context only.
- The previous round exposed `--tcrv-lower-linalg-rvv-binary-to-exec`, kept the
  older i32 aliases as compatibility surfaces, and intentionally made no fresh
  RVV runtime/correctness claim.
- `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` already calls
  `createLowerLinalgRVVBinaryToExecPass()` before execution planning and bundle
  export.
- `test/Target/TargetArtifactBundleExport/plan-linalg-i64-vmul-and-export-target-artifact-bundle.mlir`
  is useful target/export context, but it intentionally selects an RVV+scalar
  dispatch external ABI with a runtime availability guard. For this direct RVV
  microkernel runner round, `test/Transforms/LinalgToExec/linalg-i64-vmul-to-rvv-artifact.mlir`
  is the cleaner representative: it still starts from marked Linalg `i64-vmul`,
  still uses the plan-and-export bundle front door, and produces the direct RVV
  runtime-callable source/header/object bundle consumed by
  `scripts/rvv_microkernel_e2e.py`.
- `scripts/rvv_microkernel_e2e.py` already has bundle mode,
  `--use-plan-and-export-bundle-front-door`, i64-vmul family validation,
  generated external caller construction, and remote `ssh rvv` compile/link/run
  plumbing.

## Requirements

1. Use the direct `i64-vmul` bounded Linalg fixture
   `test/Transforms/LinalgToExec/linalg-i64-vmul-to-rvv-artifact.mlir` as the
   representative because repository inspection showed the similarly named
   target artifact bundle fixture exercises the RVV+scalar dispatch ABI rather
   than the direct RVV microkernel runner path. The route must still start from
   marked Linalg frontend input, not from hand-authored selected-path or
   emission-plan MLIR.
2. The evidence path must use
   `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle` or otherwise
   visibly use the public `--tcrv-lower-linalg-rvv-binary-to-exec` frontend
   name in planned pipeline metadata. New evidence must not rely on deprecated
   i32 pass aliases.
3. Generated source/header/object/bundle artifacts must come from compiler and
   target-owned route registration. Python may parse bundle metadata, generate
   a small caller from the emitted header/signature, run tools, and write
   sanitized evidence only.
4. Live evidence JSON must keep runtime/executability scope narrow:
   correctness/executability for the bounded selected family only, no
   throughput, latency, performance, or generic lowering claims.
5. Live/manual evidence must make `ssh_evidence` non-null and explicitly record
   whether remote compile, link, run, and output validation succeeded. Dry-run
   evidence may remain non-runtime evidence, but it must not look like a
   runtime claim.
6. Evidence and diagnostics must preserve parameter layering: target hardware
   facts, compile-time vector/variant config, runtime ABI/control arguments,
   and descriptor-local facts stay separately named.
7. Focused lit/FileCheck coverage should exercise local dry-run/schema/metadata
   behavior only. It must not invoke `ssh rvv` inside lit.
8. If live `ssh rvv` succeeds, finish/archive the Trellis task and commit one
   coherent change. If live ssh is unavailable or environmentally fails after
   the code path is otherwise correct, leave the task open and record the exact
   command, artifact directory, failure, and next continuation point.

## Acceptance Criteria

- [x] The representative command uses
      `--use-target-artifact-bundle --use-plan-and-export-bundle-front-door`
      with `--arithmetic-family=i64-vmul` and the front-door Linalg bundle
      fixture.
- [x] Evidence records the front-door route, selected kernel/family/shape,
      bundle export mode, selected source/header/object records, runtime ABI
      signature, compile flags, generated external caller, and artifact paths
      under `artifacts/tmp/...`.
- [x] The evidence path validates compiler-emitted source/header/object bundle
      records and generated caller arithmetic for `lhs * rhs`, not stale add or
      subtract semantics.
- [x] Live `ssh rvv` evidence, when run, records non-null structured
      `ssh_evidence` details for remote setup/facts, compile, link, run, and
      output-marker validation.
- [x] No generated evidence file claims performance, broad correctness,
      generic RVV backend support, or generic Linalg lowering.
- [x] Focused lit/FileCheck coverage checks the new evidence schema/metadata
      locally without contacting `ssh rvv`.
- [x] `git diff --check` passes.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
      tcrv-translate -j2` passes.
- [x] `python3 -m py_compile scripts/rvv_microkernel_e2e.py` passes if the
      runner changes.
- [x] The focused lit/FileCheck tests for the changed runner/bundle path pass.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
      -j2` passes before commit unless an earlier focused failure makes it
      meaningless.
- [x] Trellis task validation passes; if live ssh succeeds, the task is
      finished and archived.

## Out Of Scope

- Generic RVV backend implementation or broad RVV lowering.
- New arithmetic families, new vector shapes, or large fixture sweeps.
- Compute semantics in `tcrv.exec`.
- RVV-specific branches in core passes.
- Treating docs, prompts, helper-only cleanup, status files, or broad smoke
  tests as the main result.
- Committing generated `artifacts/tmp` evidence artifacts unless an existing
  tracked fixture convention requires it.
- Throughput, latency, benchmark, or performance claims.

## Minimal Validation Plan

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tcrv-opt
  tcrv-translate -j2`
- Focused lit/FileCheck coverage for the changed bundle runner/schema path.
- `python3 -m py_compile scripts/rvv_microkernel_e2e.py`
- Live command:
  `python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vmul --input test/Transforms/LinalgToExec/linalg-i64-vmul-to-rvv-artifact.mlir --expect-selected-kernel frontend_i64_vmul --ssh-target rvv`
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-front-door-rvv-binary-frontend-bundle-ssh-evidence`

## Technical Notes

- Specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/variant-pipeline/generation-selection-tuning.md`
  - `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
- Previous task context read:
  - `.trellis/tasks/archive/2026-05/05-10-bounded-linalg-rvv-binary-frontend-pass/prd.md`
- Implementation surfaces inspected:
  - `include/TianChenRV/Transforms/Passes.td`
  - `tools/tcrv-translate/tcrv-translate.cpp`
  - `scripts/rvv_microkernel_e2e.py`
  - `test/Scripts/rvv-microkernel-bundle-e2e.test`
  - `test/Scripts/rvv-microkernel-e2e.test`
  - `test/Target/TargetArtifactBundleExport/plan-linalg-i64-vmul-and-export-target-artifact-bundle.mlir`

## Completion Evidence

- Local dry-run proof:
  `python3 scripts/rvv_microkernel_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vmul --input test/Transforms/LinalgToExec/linalg-i64-vmul-to-rvv-artifact.mlir --expect-selected-kernel frontend_i64_vmul --run-id codex-probe-i64-vmul-direct-frontdoor --overwrite`
- Live ssh proof:
  `python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --arithmetic-family=i64-vmul --input test/Transforms/LinalgToExec/linalg-i64-vmul-to-rvv-artifact.mlir --expect-selected-kernel frontend_i64_vmul --ssh-target rvv --run-id codex-frontdoor-i64-vmul-live --overwrite`
- Live evidence artifact:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/codex-frontdoor-i64-vmul-live/evidence.json`
- The live evidence records `ssh_evidence.success = true`,
  `remote_compile_succeeded = true`, `remote_link_succeeded = true`,
  `remote_run_succeeded = true`, and `output_validation_succeeded = true`.
- Scope of the live claim:
  bounded RVV `i64-vmul` target-artifact bundle external caller correctness
  only; no performance, generic lowering, or broad RVV backend claim.
