# Source-fronted RVV target artifact front door

## Goal

Promote the bounded source-fronted `i32-vadd` RVV runtime proof from an
RVV-specific direct helper evidence route into the normal target artifact
front-door workflow. The desired production path starts from the linalg/source
fixture, runs the compiler planning pipeline, selects the RVV extension-family
artifact exporters through generic target artifact or plan-and-export bundle
front doors, and then uses those generated source/header/object artifacts for
bounded `ssh rvv` runtime evidence.

## What I Already Know

* Current HEAD at task start is
  `f3910e9 test(rvv): prove source-fronted runtime artifact`; the worktree was
  clean before task creation.
* No `.trellis/.current-task` existed at session start. This task was created
  from the Hermes Direction Brief before source edits.
* The previous archived task
  `.trellis/tasks/archive/2026-05/05-13-source-fronted-rvv-runtime-artifact-proof/prd.md`
  proved that the source-fronted `i32-vadd` path reaches generated RVV
  source/header/object artifacts and real `ssh rvv` correctness for
  `n = 7, 16, 23`, but its authoritative evidence path still used
  RVV-specific direct helper routes.
* The current owner is the generic target artifact and bundle export front
  door, not another runner-only proof.
* The relevant source fixture is
  `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`.
* The likely affected compiler surfaces are
  `include/TianChenRV/Target/TargetArtifactExport.h`,
  `lib/Target/TargetArtifactExport.cpp`,
  `lib/Target/TargetArtifactBundleExport.cpp`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`,
  `tools/tcrv-translate/tcrv-translate.cpp`,
  `scripts/rvv_microkernel_e2e.py`,
  `test/Scripts/rvv-microkernel-e2e.test`, and
  `test/Target/TargetArtifactBundleExport/plan-linalg-i32-vadd-and-export-target-artifact-bundle.mlir`.

## Requirements

* The evidence path must start from the linalg/source fixture and reach
  planned selected RVV extension-family ops before artifact export.
* Generic target source/header/object front doors or the plan-and-export target
  artifact bundle route must select the RVV artifact exporters through
  registry metadata and the same preflight used by direct RVV export.
* Generated records must preserve selected binary config, runtime AVL
  provenance, callable ABI order, EmitC route provenance, selected variant,
  selected role, and route ownership.
* The evidence runner or lit contract must use generic front-door or bundle
  artifact outputs for the `ssh rvv` runtime proof. RVV-specific direct helper
  routes may remain as compatibility/debug routes, but must not be the only
  production evidence path.
* Missing selected RVV body, stale selected-config metadata, or missing
  runtime-element-count ABI must fail before artifact output.
* If full bundle plus generic single-artifact source/header/object scope is too
  large for one round, finish one coherent generic front-door submodule and
  record the remaining bundle follow-up truthfully.

## Acceptance Criteria

* [x] Trellis PRD, implement context, and check context exist before source
      edits.
* [x] Exact commands from the linalg/source fixture run through lowering,
      execution planning, and a generic target artifact or bundle front door.
* [x] The selected RVV source/header/object artifacts are emitted by
      registry-derived generic front-door or bundle routes, not only by
      `--tcrv-export-rvv-microkernel-*` direct helper commands.
* [x] Source/header/object or bundle index records preserve selected binary
      config, runtime AVL provenance, callable ABI order, EmitC route
      provenance, selected variant, role, route id, owner, and route role.
* [x] Negative coverage fails before artifact output for missing selected RVV
      body, stale selected-config metadata, and missing runtime-element-count
      ABI.
* [x] The bounded runtime evidence runner or lit contract uses the generic
      front-door or bundle-produced artifacts for `ssh rvv` execution for
      representative `n` values below, equal to, and above descriptor-local
      capacity.
* [x] Focused build covers touched targets, including `TianChenRVTarget`,
      `TianChenRVTransforms`, `tcrv-opt`, `tcrv-translate`, and
      `tianchenrv-target-artifact-export-test` when target/export code changes.
* [x] Focused tests cover affected LinalgToExec, RVVMicrokernel,
      EmissionManifest, TargetArtifactExport, TargetArtifactBundleExport, and
      `rvv-microkernel-e2e` surfaces.
* [x] `git diff --check`, `git diff --cached --check`, and Trellis validation
      pass before finish/archive.
* [x] Task status, completion notes, exact artifact paths, runtime evidence,
      and any remaining direct-helper compatibility status are recorded
      truthfully before one coherent commit is created.

## Definition Of Done

* Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit.
* Python is limited to bounded orchestration, artifact parsing, and evidence
  running.
* No computation semantics move into `tcrv.exec`, descriptors, Python, or
  generated harnesses.
* Shared generic front-door code remains target-neutral and does not add
  RVV-specific semantic branches outside target/plugin/export interfaces.
* Runtime evidence is bounded to the source-fronted `i32-vadd` callable ABI and
  is not reported as generic RVV backend support, broad correctness, or
  performance evidence.

## Out Of Scope

* New arithmetic families, dtype matrices, performance claims, broad smoke
  matrices, generic RVV backend claims, or standalone evidence repackaging.
* Moving compute semantics into descriptors or direct descriptor-to-C export.
* A helper-only, metadata-only, test-only, or runner-only change as the main
  result.

## Technical Notes

* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`, and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Prior PRD read:
  `.trellis/tasks/archive/2026-05/05-13-source-fronted-rvv-runtime-artifact-proof/prd.md`.
* Initial implementation search should determine whether the existing bundle
  front door already produces the needed source/header/object records from the
  source fixture. If so, the coherent round should rewire the evidence runner
  and tests to consume those outputs. If not, repair the target artifact
  selection/front-door boundary first.

## Completion Notes

* The production C++ plan-and-export bundle front door already emitted the
  required direct RVV source/header/object records from
  `test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir`.
* The evidence runner now permits
  `--use-target-artifact-bundle --use-plan-and-export-bundle-front-door
  --lower-linalg-frontend`, passes the linalg/source fixture directly to
  `tcrv-translate --tcrv-plan-and-export-target-artifact-bundle`, records
  `input_source = linalg-frontend`, and records the bundle-selected source
  route `tcrv-export-rvv-microkernel-c`.
* The script lit contract now makes the source-fronted `i32-vadd` dry-run use
  the plan-and-export bundle front door, checks the bundle source/header/object
  files, checks the bundle index route/owner/runtime ABI metadata, and verifies
  the command summary contains no direct RVV microkernel source export command
  for this source-fronted proof.
* Direct RVV helper routes remain available as compatibility/debug paths for
  existing direct-route and family-specific evidence modes, but the
  source-fronted `i32-vadd` proof path now consumes bundle-produced artifacts.
* Runtime artifact proof directory:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/20260513T-source-fronted-i32-vadd-plan-bundle-front-door`.
* Generated front-door artifacts:
  `target_artifact_bundle/tianchenrv-target-artifact-bundle.index`,
  `target_artifact_bundle/artifact-0-runtime-callable-c-source-tcrv-export-rvv-microkernel-c.c`,
  `target_artifact_bundle/artifact-1-runtime-callable-c-header-tcrv-export-rvv-microkernel-header.h`,
  `target_artifact_bundle/artifact-2-riscv-elf-relocatable-object-tcrv-export-rvv-microkernel-object.o`,
  `rvv_microkernel_external_caller.c`, `evidence.json`,
  `command_summary.json`, and `hashes.json`.
* Exact front-door command recorded in `command_summary.json`:

```bash
/home/kingdom/phdworks/TianchenRV/build/bin/tcrv-translate --tcrv-plan-and-export-target-artifact-bundle --tcrv-target-artifact-bundle-output-dir=artifacts/tmp/rvv_microkernel_bundle_e2e/20260513T-source-fronted-i32-vadd-plan-bundle-front-door/target_artifact_bundle test/Transforms/LinalgToExec/linalg-i32-vadd-to-exec.mlir
```

* Runtime proof command:

```bash
python3 scripts/rvv_microkernel_e2e.py --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --lower-linalg-frontend --expect-selected-kernel=frontend_i32_vadd --runtime-count=7 --runtime-count=16 --runtime-count=23 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --run-id 20260513T-source-fronted-i32-vadd-plan-bundle-front-door --overwrite --timeout 120
```

* Runtime proof result:
  `status=success`, `ssh_evidence=true`, selected kernel
  `frontend_i32_vadd`, runtime counts `7,16,23`. Remote source-built and
  bundle-object-linked runs both printed
  `tcrv_rvv_microkernel_external_abi_ok counts=7,16,23`.
* Object evidence:
  `file` reports
  `ELF 64-bit LSB relocatable, UCB RISC-V, RVC, double-float ABI, version 1 (SYSV), not stripped`.
  `llvm-readobj-20 --file-headers --symbols` reports
  `Format: elf64-littleriscv`, `Arch: riscv64`,
  `Machine: EM_RISCV`, and function symbol
  `tcrv_rvv_i32_vadd_microkernel_frontend_i32_vadd_rvv_first_slice`.
* Validation run:
  `python3 scripts/rvv_microkernel_e2e.py --self-test`;
  `python3 scripts/rvv_microkernel_e2e.py --dry-run --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --lower-linalg-frontend --expect-selected-kernel=frontend_i32_vadd --runtime-count=7 --runtime-count=16 --runtime-count=23 --run-id codex-plan-bundle-frontend-check --overwrite`;
  `python3 -m py_compile scripts/rvv_microkernel_e2e.py`;
  `cmake --build build --target TianChenRVTarget TianChenRVTransforms tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`;
  `./build/bin/tianchenrv-target-artifact-export-test`;
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-microkernel-e2e'`
  from `build/test`;
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'LinalgToExec|RVVMicrokernel|EmissionManifest|TargetArtifactBundleExport|rvv-microkernel-e2e'`
  from `build/test`;
  `git diff --check`;
  `git diff --cached --check`;
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-13-source-fronted-rvv-target-artifact-front-door`.
