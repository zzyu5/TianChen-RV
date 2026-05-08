# Typed Dispatch Runtime Guard Contract

## Goal

Make dispatch runtime-guard requirements a typed, compiler-owned `tcrv.exec`
IR contract instead of relying on arbitrary non-empty `condition`, `guard`, or
`policy` strings as the semantic trigger for dispatch ABI guard materialization.

## Requirements

- Add the smallest compute-free `tcrv.exec.case` metadata needed to represent
  "this dispatch case requires a runtime dispatch-availability guard".
- Keep generic printable `condition`, `guard`, and `policy` annotations as
  diagnostics or compatibility metadata, but do not treat arbitrary non-empty
  strings as sufficient semantic evidence for runtime ABI guard creation.
- Update selection and dispatch synthesis so runtime-capability-guarded cases
  carry the typed requirement marker.
- Update runtime-guard materialization so it consumes the typed marker and
  capability availability/conflict analysis deterministically.
- Keep fallback ops free of dispatch-case runtime guard metadata.
- Strengthen capability/coherence/export preflight so selected dispatch cases
  that require a guard must have a valid same-kernel `runtime_guard` link to a
  direct `tcrv.exec.runtime_param` with ABI role `dispatch-availability-guard`
  before target artifact bundle/export reports success.
- Preserve plugin locality: no RVV/IME/Sophgo/AME/offload/scalar/vendor
  branches in core passes.
- Preserve runtime ABI boundary: no automatic hardware probe and no runtime,
  correctness, or performance claim in this round.

## Evidence

- lit/FileCheck coverage for typed marker emission from dispatch synthesis or
  selection.
- lit/FileCheck coverage for `tcrv-materialize-dispatch-runtime-guards`
  materializing one same-kernel dispatch availability `runtime_param` and
  attaching `runtime_guard` links by consuming the typed marker.
- Negative coverage showing arbitrary non-empty `condition` / `guard` /
  `policy` annotations alone do not create runtime guards.
- Negative coverage showing typed guard requirements with missing or stale
  runtime_guard linkage fail before target artifact bundle export reports
  completion.
- Local checks:
  - `git diff --check`
  - CMake configure under `artifacts/tmp/tianchenrv-build`
  - `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`

## Completion Evidence

- Recovered the existing failed-run task at
  `.trellis/tasks/05-08-typed-dispatch-runtime-guard-contract/`; no duplicate
  task was created and no corruption was found.
- Added typed `runtime_guard_required = true` metadata on dispatch cases that
  require dispatch-availability guarding.
- Updated dispatch synthesis, variant selection, runtime-guard materialization,
  capability checking, and execution-plan coherence to treat that typed marker
  as the semantic guard requirement. Arbitrary `condition` / `guard` /
  `policy` strings remain printable annotations and are not sufficient to
  create a runtime ABI guard. A dispatch-case `runtime_guard` link is also
  rejected unless the typed requirement marker is present.
- Added/updated lit and C++ tests for positive marker emission/materialization,
  negative arbitrary-string behavior, invalid typed guard links, fallback
  rejection, selected-plan coherence, and target bundle/export preflight.
- Updated durable specs for the typed `tcrv.exec` dispatch guard requirement,
  same-kernel `runtime_guard` ABI link, string-annotation boundary, fallback
  boundary, and local-only evidence scope.
- Local checks passed:
  - `git diff --check`
  - `cmake -S . -B artifacts/tmp/tianchenrv-build -G Ninja -DLLVM_DIR=/usr/lib/llvm-20/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-20/lib/cmake/mlir`
  - `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv -j2`
- No `ssh rvv` run was made and no new RVV runtime, correctness, or
  performance claim was made.
