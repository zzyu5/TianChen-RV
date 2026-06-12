# Tighten Dispatch Runtime Guard Semantics

## Goal

Tighten `tcrv.exec.case` runtime guard semantics so
`runtime_guard_required` means only an explicit runtime
dispatch-availability guard required by capability legality/runtime
arbitration. Generic `condition`, `guard`, and `policy` strings remain
printable plugin decision annotations and must not independently create
runtime ABI guard requirements or links.

## Requirements

- Stop `VariantSelection` from setting `runtime_guard_required` merely because
  a selected dispatch case has generic decision metadata.
- Preserve copying of non-empty `condition`, `guard`, and `policy` from
  variants to dispatch cases as annotations.
- Keep capability legality/runtime arbitration as the source of
  `runtime_guard_required`, including unavailable or conflicting required
  capabilities.
- Preserve target-neutral planning: no RVV/IME/Sophgo/AME/offload/scalar/vendor
  branches in core selection, guard materialization, or coherence.
- Do not add Python compiler semantics or core compute ops.
- Update focused tests and specs only where needed to clarify the annotation
  versus typed-guard boundary.

## Acceptance Criteria

- [x] Capability-guarded selected dispatch cases get
  `runtime_guard_required = true` and materialize same-kernel
  `runtime_guard` links to a `dispatch-availability-guard` runtime parameter.
- [x] Cases with only printable `condition` / `guard` / `policy` annotations
  do not get `runtime_guard_required` and do not receive `runtime_guard` links.
- [x] Coherence/export paths do not require a runtime guard link solely because
  printable decision metadata is present.
- [x] Existing stale/missing/wrong runtime guard linkage failures still fail
  before bundle export reports completion.
- [x] `git diff --check`, CMake configure, and `check-tianchenrv` pass locally.

## Definition of Done

- Code changes are in C++ / MLIR / TableGen / CMake / lit or C++ tests only.
- Worktree is clean after one coherent commit.
- Trellis task is archived and validated.
- No new RVV runtime, correctness, or performance claim is made without
  `ssh rvv` evidence.

## Out of Scope

- No RVV runtime execution, correctness, or performance evidence collection.
- No IME, AME, Sophgo, vendor, or offload expansion.
- No new runtime hardware probing in generated dispatch C.
- No Python implementation of compiler IR, capability model, selection,
  dispatch, runtime guard semantics, lowering, emission, coherence, or export.

## Technical Notes

- Main implementation target: `lib/Transforms/VariantSelection.cpp`.
- Runtime guard materialization and coherence already consume the typed marker
  and validate same-kernel `runtime_param` linkage.
- Specs already largely state the desired boundary; update only if local tests
  reveal wording that still treats generic decision strings as semantic runtime
  guard triggers.
