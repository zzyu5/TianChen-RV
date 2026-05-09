# RVV i32 binary microkernel e2e family evidence bridge

## Goal

Generalize `scripts/rvv_microkernel_e2e.py` from a bounded i32-vadd-only
evidence runner into a bounded RVV direct i32 add/sub/mul microkernel evidence
bridge. The script must continue to orchestrate existing C++/MLIR tooling and
optional real `ssh rvv` evidence only; compiler family facts, selected routes,
runtime ABI metadata, lowering, emission, and artifact generation remain owned
by the existing C++/MLIR/exporter stack.

## What I already know

- Current HEAD is `e25cc6e`, the worktree started clean, and no current
  Trellis task existed before this task was created.
- The previous archived frontend-lowering task completed family-named
  linalg-to-exec lowering and documented that `i32-vadd`, `i32-vsub`, and
  `i32-vmul` can flow through the existing planning/export pipeline.
- `include/TianChenRV/Target/I32BinaryFamilyRegistry.h` already records
  family-specific RVV direct microkernel route ids, runtime ABI names,
  runtime glue roles, intrinsics, microkernel op names, component groups, and
  external ABI names for add/sub/mul.
- `lib/Target/RVV/RVVMicrokernel.cpp` and
  `test/Target/RVVMicrokernel/rvv-microkernel-family-sub.mlir` /
  `rvv-microkernel-family-mul.mlir` already show target/export support for
  vsub/vmul direct RVV microkernel source/header/object paths.
- Existing target-artifact-bundle tests include direct RVV vsub bundle
  metadata and family-aware linalg plan-and-export bundle coverage for
  add/sub/mul dispatch routes.
- `scripts/rvv_scalar_dispatch_e2e.py` already has a small explicit
  `ARITHMETIC_FAMILY_SPECS` table and `--arithmetic-family` CLI selector that
  can be used conceptually without copying dispatch-specific semantics.
- `scripts/rvv_microkernel_e2e.py` still hard-codes vadd-only default input,
  required manifest handoff, route ids, component group, external ABI name,
  expected intrinsic, dataflow provenance, generated external caller arithmetic,
  claim scope, and self-test fixtures.

## Requirements

- Add `--arithmetic-family` with supported values `i32-vadd`, `i32-vsub`, and
  `i32-vmul`, defaulting to `i32-vadd`.
- Replace script-level vadd-only constants with a small explicit family
  configuration table covering default input MLIR, source/header/object route
  ids, bundle component group, external ABI name, external ABI success marker,
  self-check success marker, required handoff fields, microkernel op name,
  arithmetic op name, expected intrinsic, result vector name, C operator, and
  claim wording.
- Preserve compatibility for existing vadd dry-run, generic-route, harness,
  bundle, and ssh modes.
- Keep the script as evidence tooling only. It may orchestrate `tcrv-opt`,
  `tcrv-translate`, target bundle export, artifact parsing, C caller
  construction, and `ssh rvv`; it must not implement compiler IR, plugin
  decisions, lowering, emission, runtime ABI ownership, or route selection.
- Make handoff validation, bundle-index validation, generated source/header/
  object discovery, external caller generation, command summary, evidence JSON,
  and claim scope family-aware.
- Fail closed when the selected family sees stale metadata from another family,
  including stale route ids, runtime ABI names, external ABI component group,
  intrinsic, microkernel op, arithmetic op, result vector name, self-check
  marker, or external caller arithmetic.
- Preserve strict evidence boundaries: dry-run output is local compiler and
  artifact evidence only; runtime/correctness claims require real `ssh rvv`
  output. Evidence must record selected arithmetic family and sanitized artifact
  paths/hashes, without secrets, tokens, passwords, raw URLs, throughput,
  latency, or performance claims.
- Update README microkernel wording and examples so the direct microkernel
  bridge is described as bounded i32 add/sub/mul family evidence, while the
  runtime/correctness claim remains only the finite generated external caller
  check for the selected family when backed by real `ssh rvv`.
- Add focused lit/FileCheck coverage for vsub and vmul dry-run family
  selection, bundle or plan-and-export family front doors where route support
  already exists, stale-family metadata rejection, sanitized evidence output,
  and existing vadd behavior.
- If C++ exporter metadata is unexpectedly insufficient for the bridge,
  repair only the minimal target/export metadata path with focused tests.

## Acceptance Criteria

- [x] `scripts/rvv_microkernel_e2e.py --arithmetic-family=i32-vadd` preserves
      the existing default behavior.
- [x] `scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i32-vsub`
      validates vsub handoff/source metadata and rejects stale vadd metadata.
- [x] `scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i32-vmul`
      validates vmul handoff/source metadata and rejects stale add/sub
      metadata.
- [x] Target-artifact-bundle mode consumes compiler-emitted source/header/object
      records for the selected family, including component group,
      component role, external ABI name, route ids, runtime ABI signature, and
      selected component metadata.
- [x] Generated external caller arithmetic is family-correct:
      `lhs + rhs` for add, `lhs - rhs` for sub, and `lhs * rhs` for mul.
- [x] Evidence JSON and printed summaries include `arithmetic_family`,
      sanitized artifact paths/hashes, family-specific success marker, and
      family-specific claim scope.
- [x] Local lit/FileCheck tests cover self-test, existing vadd paths, vsub and
      vmul dry-run selection, bundle/front-door family selection, stale-family
      rejection, and secret redaction.
- [x] README microkernel examples describe the finite add/sub/mul bridge and
      show `--arithmetic-family` usage.
- [x] `git diff --check` passes.
- [x] `python3 scripts/rvv_microkernel_e2e.py --self-test` passes.
- [x] Focused dry-run checks pass for at least
      `--arithmetic-family=i32-vsub` and `--arithmetic-family=i32-vmul`,
      including bundle/front-door paths where implemented.
- [x] Focused lit checks pass for `Scripts/rvv-microkernel-e2e.test` and
      `Target/RVVMicrokernel`; include `Target/TargetArtifactBundleExport` if
      bundle/export behavior changes.
- [x] Local CMake configure under `artifacts/tmp/tianchenrv-build` succeeds if
      build files, C++ code, or lit tests change.
- [x] `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
      passes if feasible; otherwise the exact blocker is recorded.
- [x] At least one bounded real `ssh rvv` run for a non-add family is attempted
      through this bridge and archived as sanitized evidence. If remote access
      or toolchain execution fails, do not fabricate evidence; keep the task
      open unless the blocker is outside this module and is explicitly recorded.

## Non-goals

- Do not add new RVV dialect families, microkernel semantics, descriptor
  families, or generic RVV lowering.
- Do not add compute semantics to `tcrv.exec`.
- Do not add target-family branches to generic core passes.
- Do not rewrite the scalar dispatch bridge except for reference or a narrowly
  necessary tooling-only helper.
- Do not add performance benchmarking, throughput/latency reporting, broad
  smoke matrices, or standalone evidence packaging as the main result.
- Do not claim generic RVV lowering correctness, arbitrary linalg correctness,
  runtime integration, broad correctness coverage, or performance. Runtime
  correctness claims are limited to the bounded generated external caller check
  for the selected i32 binary family and only with real `ssh rvv` output.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/architecture/index.md`,
  `.trellis/spec/architecture/design-boundaries.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/validation/index.md`,
  `.trellis/spec/validation/experiment-reference.md`,
  `.trellis/spec/guides/index.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
- Prior task PRD read:
  `.trellis/tasks/archive/2026-05/05-09-descriptor-backed-i32-binary-linalg-frontend-lowering-contract/prd.md`.
- Primary files inspected:
  `scripts/rvv_microkernel_e2e.py`,
  `scripts/rvv_scalar_dispatch_e2e.py`,
  `README.md`,
  `include/TianChenRV/Target/I32BinaryFamilyRegistry.h`,
  `lib/Target/RVV/RVVMicrokernel.cpp`,
  `test/Scripts/rvv-microkernel-e2e.test`,
  `test/Target/RVVMicrokernel/rvv-microkernel-family-sub.mlir`,
  `test/Target/RVVMicrokernel/rvv-microkernel-family-mul.mlir`, and
  relevant target-artifact bundle tests.

## Completion Notes

- Implemented `--arithmetic-family=i32-vadd|i32-vsub|i32-vmul` in
  `scripts/rvv_microkernel_e2e.py`, defaulting to vadd for compatibility.
- Replaced vadd-only evidence-runner constants with an explicit family table
  covering default input, selected variant, source/header/object route ids,
  runtime ABI fields, component group, external ABI name, microkernel op,
  arithmetic op, RVV intrinsic, result vector, success markers, and caller
  arithmetic.
- Generalized manifest handoff validation, generated source/header validation,
  dataflow provenance checks, bundle record selection, external caller
  construction, evidence JSON, and printed summaries to include and enforce the
  selected family.
- Added stale-family rejection so a non-add run fails on stale vadd handoff,
  route/runtime ABI/component/intrinsic/dataflow/caller metadata instead of
  accepting it as local evidence.
- Updated README direct RVV microkernel sections and examples to describe the
  finite add/sub/mul bridge and `--arithmetic-family` usage.
- Added focused script lit coverage in
  `test/Scripts/rvv-microkernel-e2e.test` and bundle/front-door coverage in
  `test/Scripts/rvv-microkernel-bundle-e2e.test`.
- Updated `.trellis/spec/testing/mlir-testing-contract.md` with a 7-section
  code-spec scenario for the family-aware direct RVV microkernel evidence
  bridge command surface and validation matrix.
- Real non-add RVV evidence was produced with:
  `python3 scripts/rvv_microkernel_e2e.py --arithmetic-family=i32-vsub --use-target-artifact-bundle --use-plan-and-export-bundle-front-door --ssh-target rvv --run-id 20260509T-rvv-microkernel-vsub-bundle-ssh --overwrite`.
- Runtime evidence artifact directory:
  `artifacts/tmp/rvv_microkernel_bundle_e2e/20260509T-rvv-microkernel-vsub-bundle-ssh`.
  The evidence JSON records `arithmetic_family = i32-vsub`, `ssh_evidence =
  true`, `host_facts.architecture = riscv64`, `clang_path = /usr/bin/clang`,
  expected marker `tcrv_rvv_i32_vsub_microkernel_external_abi_ok`, and both
  source-built and bundle-object external caller marker observations as true.
- Validation run:
  `python3 -m py_compile scripts/rvv_microkernel_e2e.py`,
  `python3 scripts/rvv_microkernel_e2e.py --self-test`,
  vadd/vsub/vmul focused dry-runs,
  vsub/vmul bundle dry-runs,
  vsub plan-and-export bundle dry-run,
  stale vadd handoff negative for vsub,
  focused lit from `artifacts/tmp/tianchenrv-build/test` with
  `--filter='rvv-microkernel'` and
  `--filter='rvv-microkernel|TargetArtifactBundleExport'`,
  `git diff --check`,
  CMake configure under `artifacts/tmp/tianchenrv-build`, and
  `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  (163/163 passed).
