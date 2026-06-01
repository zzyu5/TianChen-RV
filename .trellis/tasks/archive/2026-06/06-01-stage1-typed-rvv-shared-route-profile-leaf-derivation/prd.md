# Stage1 typed RVV shared route-profile leaf derivation

## Goal

Replace the shared RVV selected-body route/profile leaf spelling surface in
`RVVEmitCRoutePlanning.cpp` with derivation from typed selected-body
configuration, capability, policy, and runtime facts. Existing supported
routes may still emit concrete `__riscv_*_i32m1` / `__riscv_*_i32m2`,
`vint*`, `vuint*`, and `vbool*` leaves, but only as validated derived output
from typed `tcrv_rvv` facts, never as route authority.

## What I Already Know

* The Hermes Direction Brief is the task source.
* The repository started this round on `main` with a clean worktree at
  `64ed6bb6 rvv: derive contraction leaves from typed facts`.
* No active `.trellis/.current-task` existed; this task was created as
  `.trellis/tasks/06-01-stage1-typed-rvv-shared-route-profile-leaf-derivation`.
* Specs require selected RVV routes to flow through typed low-level
  `tcrv_rvv` body/config facts, RVV plugin-owned provider planning, and
  common EmitC materialization. Route ids, artifact names, ABI strings,
  emission metadata, descriptor residue, source-front-door markers, and exact
  intrinsic spelling are mirrors or derived outputs only.
* The immediately preceding contraction task completed the local owner pattern:
  route-family plans snapshot typed facts and validate concrete leaf mirrors
  against derived facts before provider materialization.
* A focused scan shows remaining exact i32m1/i32m2 active source hits in
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`, especially the shared
  config/profile builder, masked/computed-mask memory leaves, segment2 leaves,
  standalone/reduction leaves, and widening-conversion helper leaves.

## Requirements

* Keep the production route shape as selected `tcrv.exec` RVV variant -> typed
  low-level `tcrv_rvv` body/config -> RVV route-family plans -> provider-built
  `TCRVEmitCLowerableRoute` -> common EmitC materialization.
* Derive shared selected-body config/profile leaves from typed
  `RVVSelectedBodyEmitCRouteDescription` facts: element type, SEW, LMUL,
  tail policy, mask policy, VL contract, vector/index/mask type names, C
  vector/index/mask/scalar types, pointer C types, element byte size, setvl,
  unit/strided/indexed load/store, index scale, scalar splat, and ordinary
  store leaves.
* Derive masked and runtime/computed-mask memory leaves from the same typed
  config/profile facts: masked load/store, masked strided load/store, masked
  indexed load/store, runtime-scalar splat, compare, mask type, policy-sensitive
  masked forms, and fail-closed unsupported typed combinations.
* Derive segment2 tuple, load/store, masked segment load/store, field extract,
  and tuple-create leaves from typed SEW/LMUL/segment count facts rather than
  fixed `i32m1x2` constants.
* Preserve existing support scope. This task must not add new RVV coverage,
  new dtype/LMUL clone batches, source-front-door routes, common EmitC semantic
  branches, or compatibility wrappers for legacy i32 authority.
* Unsupported SEW, LMUL, element type, policy, memory form, segmented tuple
  form, relation, or runtime ABI use must fail closed with targeted diagnostics
  before provider route construction.

## Acceptance Criteria

* [x] Shared config/profile construction no longer stores complete
  `__riscv_*_i32m1`, `__riscv_*_i32m2`, `vint32m1_t`, `vint32m2_t`,
  `vuint32m1_t`, `vbool32_t`, or fixed typed vector/mask spellings as a route
  table; it derives them from SEW/LMUL/element/policy facts.
* [x] Masked/computed-mask memory and segment2 profile leaves are derived from
  typed config/profile facts and fail closed when required leaves are absent or
  stale.
* [x] Focused tests prove already-supported i32 m1/m2 outputs still appear as
  derived outputs, not owner-local constants, and negative tests prove stale or
  unsupported typed facts fail closed.
* [x] A bounded scan over touched shared planning files, route-family owners,
  relevant tests, and specs classifies remaining legacy-string hits as derived
  output, parseable-only/deprecated, negative test, spec text, or exact
  continuation debt rather than active authority.
* [x] Focused build/test commands for the touched RVV plugin paths pass and
  `git diff --check` passes.

## Definition of Done

* `implement.jsonl` and `check.jsonl` contain only relevant spec/research
  context entries and validate through the Trellis task script.
* Focused RVV plugin C++ tests pass.
* The smallest relevant lit/export checks are run if route metadata or artifact
  expectations change.
* No runtime, correctness, or performance claim is made without `ssh rvv`
  evidence.
* Trellis task status, evidence, and workspace journal truthfully record the
  change and any exact continuation point.
* One coherent commit is created if the bounded task is complete.

## Out of Scope

* New RVV coverage, reductions, segment operations beyond deriving existing
  segment2 leaves, dtype/LMUL clone batches, source-front-door routes, artifact
  packaging, broad smoke matrices, common EmitC semantic branches, Stage 2
  expansion, or future plugin work.
* Descriptor-driven C/source export.
* Compatibility wrappers that keep legacy `RVVI32M1*`, `rvv-i32m1`, finite
  `tcrv_rvv.i32_*`, or exact intrinsic spelling as route authority.
* Prompt/report/helper-only changes as the main achievement.

## Technical Approach

Reuse the recent elementwise, MAcc, base-memory, and contraction derivation
pattern. Add shared typed leaf derivation helpers in the RVV route-planning
implementation so `deriveRVVSelectedBodyConfigProfile(...)` constructs
profile fields from typed SEW/LMUL/policy facts instead of fixed per-config
string tables. Then route masked/computed-mask memory and segment2 leaf helpers
through that shared profile, adding fail-closed diagnostics for missing or
unsupported derived leaves. If the full shared profile owner is too large,
finish this coherent masked/computed-mask memory plus selected-body
config-profile slice and record standalone reduction / widening conversion
helpers as the exact continuation point.

## Technical Notes

* Read: `.trellis/spec/index.md`.
* Read: `.trellis/spec/extension-plugins/index.md` and
  `.trellis/spec/extension-plugins/rvv-plugin.md`.
* Read: `.trellis/spec/lowering-runtime/index.md` and
  `.trellis/spec/lowering-runtime/emitc-route.md`.
* Read: `.trellis/spec/variant-pipeline/index.md` and
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`.
* Read: `.trellis/spec/testing/index.md` and
  `.trellis/spec/testing/mlir-testing-contract.md`.
* Read shared guides:
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Read archived contraction task:
  `.trellis/tasks/archive/2026-06/06-01-stage1-typed-rvv-contraction-route-family-derivation/`.
