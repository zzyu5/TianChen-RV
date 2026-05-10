# Make RVV i64m1 capability replay first-class

## Goal

Make the finite RVV i64m1 add/sub/mul profile and replay path consume
structured capability facts instead of relying only on hand-authored one-off
i64 target fixtures. Sanitized RVV evidence/profile facts should become C++
`TargetCapabilitySet` entries for the i64m1 SEW/LMUL/tail/mask policy slice,
and a replay/profile-derived target should drive an existing i64 RVV compiler
path through planning, selected-boundary materialization, plugin-owned
microkernel materialization, and target source export.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Starting branch is `main`; the working directory was clean at session start.
- No current Trellis task existed, so this task owns the profile/replay parity
  round.
- Recent commits `8c3e3fc` and `361ee6b` already extracted RVV binary planning
  and binary microkernel materialization components.
- Existing i64 add/sub/mul descriptor, microkernel, linalg frontend, dispatch,
  and artifact tests exist, but many positive i64 fixtures still provide
  i64m1 facts by hand.
- `RVVCapabilityProfile` currently constructs first-slice i32m1 capability
  facts from probe facts; it does not expose i64m1 profile capability IDs as a
  first-class profile surface.
- `scripts/rvv_probe_to_mlir.py` is replay/evidence tooling only. If touched,
  it may only preserve sanitized bounded facts into MLIR capability fixtures.

## Boundaries

- Compiler implementation remains C++ / MLIR / LLVM / TableGen / CMake /
  lit / FileCheck.
- Python remains tooling only: probe/replay/artifact parsing and focused test
  fixture generation. It must not decide legality, selection, lowering,
  emission, correctness, or runtime behavior.
- Do not add generic compute semantics to `tcrv.exec`.
- Do not move RVV-specific semantics into core passes.
- Keep parameter layering strict:
  hardware/toolchain facts are capability/profile inputs; selected vector shape
  is compile-time variant metadata; runtime `n`/AVL/VL are SSA/control or ABI
  values; descriptor-local element counts remain descriptor metadata.
- No broad i32/i64/all-family smoke matrix and no report-only closeout.
- No RVV runtime correctness or performance claim unless fresh `ssh rvv`
  evidence is created for that exact claim.

## Requirements

1. Extend the plugin-local C++ RVV capability profile API and construction so
   the finite i64m1 config/policy capability IDs are first-class:
   `rvv.i64_m1.sew64`, `rvv.i64_m1.lmul_m1`,
   `rvv.i64_m1.tail_policy.agnostic`, and
   `rvv.i64_m1.mask_policy.agnostic`.
2. Validate i64m1 profile facts as bounded structured facts: SEW must be 64,
   LMUL must be `m1`, tail policy must be `agnostic`, and mask policy must be
   `agnostic`.
3. Preserve the existing i32m1/i32m2 finite behavior and existing profile
   capacity metadata semantics.
4. Update replay tooling only to preserve sanitized bounded i64m1 profile
   facts into MLIR capability fixtures when those facts are present or implied
   by the bounded probe/replay schema; do not encode compiler decisions there.
5. Add focused positive coverage proving a profile/replay-derived target
   allows an i64 add/sub/mul selected RVV path to pass through planning,
   selected lowering-boundary materialization, plugin microkernel
   materialization, emission planning, and target source export.
6. Add focused negative coverage proving missing or malformed i64m1
   capability/profile facts block RVV selection or emission before any runtime,
   correctness, or performance claim.
7. Update relevant RVV/capability specs only for the implemented durable
   contract.

## Acceptance Criteria

- [x] C++ profile construction exposes deterministic i64m1 capability IDs and
      symbols with bounded properties.
- [x] C++ profile validation rejects malformed i64m1 facts.
- [x] Replay MLIR from sanitized probe/profile facts can include first-class
      i64m1 capability fixture facts without deciding compiler legality.
- [x] At least one i64 compiler path consumes those profile/replay facts and
      emits the existing i64 RVV source artifact route.
- [x] At least one negative test fails closed for missing or malformed i64m1
      profile/capability facts with no artifact/runtime claim.
- [x] Existing i32 profile/replay behavior remains covered.
- [x] Focused build and lit/C++ checks pass.
- [x] `git diff --check` passes.
- [x] Full `check-tianchenrv` runs before archive if feasible.

## Minimal Validation Plan

- `git diff --check`
- `cmake --build artifacts/tmp/tianchenrv-build --target tianchenrv-rvv-extension-plugin-test tcrv-opt tcrv-translate -j2`
- `artifacts/tmp/tianchenrv-build/bin/tianchenrv-rvv-extension-plugin-test`
- Focused lit/FileCheck tests for:
  - `test/Scripts/rvv-probe-to-mlir.test`
  - new or repaired i64 profile/replay positive and negative tests
  - smallest existing i64 selected-boundary/export tests covering add/sub/mul
- `python3 -m py_compile scripts/rvv_probe_to_mlir.py` if touched.
- `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
  if feasible in the existing build environment.
- No new `ssh rvv` evidence is required unless this round makes a fresh
  runtime/correctness/performance claim.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/05-10-rvv-i64-profile-replay-parity`

## Out Of Scope

- No new runtime correctness or performance claim.
- No fresh SSH evidence unless implementation scope changes to claim runtime
  behavior.
- No broad dtype/op matrix.
- No generic i64 lowering beyond the finite RVV i64m1 add/sub/mul slice already
  present in the repo.
- No Python compiler implementation.
- No source artifacts, objects, raw logs, credentials, or `artifacts/tmp`
  outputs committed.

## Technical Notes

- Required specs are listed in `implement.jsonl` and `check.jsonl`.
- Primary source surfaces inspected:
  - `include/TianChenRV/Plugin/RVV/RVVCapabilityProfile.h`
  - `lib/Plugin/RVV/RVVCapabilityProfile.cpp`
  - `include/TianChenRV/Target/RVV/RVVVectorShape.h`
  - `include/TianChenRV/Plugin/RVV/RVVBinaryPlanning.h`
  - `lib/Plugin/RVV/RVVBinaryPlanning.cpp`
  - `include/TianChenRV/Plugin/RVV/RVVBinaryMicrokernelMaterialization.h`
  - `lib/Plugin/RVV/RVVBinaryMicrokernelMaterialization.cpp`
  - `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
  - `scripts/rvv_probe_to_mlir.py`
  - `scripts/rvv_remote_probe.py`
  - existing i64 RVV tests under `test/Transforms/LinalgToExec/`,
    `test/Plugin/`, `test/Scripts/`, and `test/Target/`.
