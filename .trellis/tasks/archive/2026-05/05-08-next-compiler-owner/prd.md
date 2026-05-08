# Capability Identity Fail-Closed

## Goal

Make direct `tcrv.exec.capability` identities unambiguous inside each
`tcrv.exec.kernel` before the C++ capability model drives plugin proposal,
legality, selection, lowering-boundary materialization, or artifact routing.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial worktree was clean.
- Latest committed owner before this task was
  `ab618b2 feat: gate execution planning pipeline coherence`.
- The previous completed round already put execution-plan coherence into the
  canonical planning pipeline, so this task must not redo pipeline, manifest,
  bundle, or evidence packaging work.
- `TargetCapabilitySet::buildFromKernel` indexes capabilities by symbol and id.
  Duplicate symbols are covered by MLIR symbol-table rules, but duplicate
  `id` values are currently not rejected at the `tcrv.exec.kernel` verifier
  boundary.
- Silent duplicate capability ids can make capability-driven compiler
  decisions depend on whichever descriptor was indexed first, while later
  same-id descriptors carry different status, kind, relation, or property
  metadata.

## Requirements

- Add a core verifier invariant that direct `tcrv.exec.capability` ops under
  one `tcrv.exec.kernel` must have unique non-empty `id` values.
- Keep `tcrv.exec` focused on execution organization, capability, variant,
  dispatch, fallback, ABI boundary metadata, and diagnostics; do not add
  compute ops.
- Keep the implementation in C++ / MLIR / TableGen / CMake / lit/FileCheck.
- Do not add Python compiler internals, new evidence scripts, new RVV runtime
  claims, or new artifact packaging.
- Preserve relation-provider semantics: `provides`, `implies`, and `conflicts`
  remain first-class relation lists, but the owning capability id itself must
  still be unique in the kernel.
- Update durable specs where they define capability identity and core dialect
  verification.

## Acceptance Criteria

- `tcrv-opt --verify-diagnostics` rejects duplicate direct capability ids in
  one kernel with a deterministic diagnostic naming the duplicate id.
- Existing relation-provider tests still pass, proving that unique profile ids
  may still provide or imply shared abstract capability ids such as `rvv`.
- `TargetCapabilitySet` C++ smoke coverage includes a positive assertion that
  relation providers remain usable under the unique-id invariant.
- Relevant specs record that capability id uniqueness is a kernel-level
  verifier invariant because C++ compiler decisions use id lookup.
- `git diff --check` passes.
- `cmake --build build --target tcrv-opt tcrv-translate -j2` passes.
- `cmake --build build --target check-tianchenrv -j2` passes.

## Out Of Scope

- Generic RVV lowering, new kernel families, IME, AME, Sophgo runtime
  implementation, remote `ssh rvv` evidence, benchmarks, dashboards, and
  broad negative fixture matrices.
- Reworking `TargetCapabilitySet` into a new target profile abstraction.
- Changing target artifact route selection, bundle export, or runtime ABI
  ownership semantics.

## Technical Notes

- Primary specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/capability-model/capability-contract.md`
  - `.trellis/spec/core-dialect/tcrv-exec-contract.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Relevant source/tests:
  - `lib/Dialect/Exec/IR/ExecOps.cpp`
  - `test/Dialect/Exec/verify.mlir`
  - `test/Support/CapabilityModelTest.cpp`
  - `test/Support/capability-model.test`
