# Stage2 RVV widening conversion executable slice

## Goal

Implement one bounded Stage2 RVV signed widening conversion route: a selected
RVV variant imports an i32 input buffer, an i64 output buffer, and runtime
`n`/AVL, realizes an explicit typed `tcrv_rvv` body for i32 m1 to i64 m2
unit-stride conversion, derives the EmitC/RVV intrinsic route in the RVV
plugin, emits a generated bundle, and proves correctness on real `ssh rvv`
for representative counts.

## Direction Source

Hermes Direction Brief: `Stage2 RVV widening conversion executable slice`.

## What I Already Know

- Current HEAD is `2df62449 rvv: add tail mask policy executable slice`; the
  worktree was clean at task creation.
- Existing Stage2 slices already cover bounded i64 SEW64, LMUL m2 arithmetic,
  scalar broadcast, compare/select, and tail/mask policy through
  selected-body realization, RVVEmitCRoutePlanning, generated bundles, and
  `ssh rvv` evidence.
- The durable RVV authority chain is `tcrv.exec` selected variant plus typed
  `tcrv_rvv` body, then RVV plugin legality/realization/route planning, then
  provider-built `TCRVEmitCLowerableRoute`, then common EmitC as neutral
  materializer.
- The conversion route must not infer dtype, SEW, LMUL, operation kind,
  intrinsic choice, or memory form from helper names, route ids, ABI strings,
  artifact names, tests, descriptors, or C source strings.

## Module Boundary

In scope:

- Add one signed i32-to-i64 widening conversion selected-body surface.
- Carry source vector facts: element i32, SEW32, LMUL m1, unit-stride input
  memory, runtime AVL.
- Carry destination vector facts: element i64, SEW64, LMUL m2, unit-stride
  output memory, tail/mask policy.
- Realize a pre-realized selected body into explicit typed `tcrv_rvv` structure
  before route planning.
- Extend RVV route planning/provider payloads so ABI types, RVV vector C types,
  load/store intrinsics, widening conversion intrinsic leaf, headers, metadata,
  and runtime ABI mirrors are derived from typed source/destination facts.
- Extend generated-bundle dry-run and real `ssh rvv` harness evidence for
  representative counts such as 7, 16, and 23.

Out of scope:

- Conversion matrix expansion, narrowing, saturating, floating-point, unsigned
  conversion, multiple dtype/LMUL combinations, high-level frontend lowering,
  source-front-door positives, one-intrinsic wrapper dialects, reductions,
  matmul, compare/select, broadcast side quests, dashboards, performance
  claims, or descriptor/direct-C/source-export paths.

## Requirements

- `tcrv.exec` continues to bind only ABI/runtime roles and selected variants;
  RVV conversion compute/config must live in typed `tcrv_rvv` body structure.
- `tcrv_rvv` must have an explicit generic typed dataflow op or equivalent
  structure for the widening conversion. It must carry source and destination
  vector/config facts structurally.
- RVV selected-body realization must accept only the bounded signed
  i32 m1 to i64 m2 unit-stride conversion and fail closed for unsupported
  directions/configs/policies/body shapes.
- RVVEmitCRoutePlanning must derive route description fields from typed
  source/destination facts and fail closed for missing or stale mirrors.
- RVVEmitCRouteProvider/common EmitC must consume the provider plan; common
  EmitC/export must not choose RVV conversion semantics itself.
- Runtime/correctness claims require real `ssh rvv` evidence.

## Acceptance Criteria

- [x] Positive selected/pre-realized body structurally carries source dtype,
      destination dtype, SEW32-to-SEW64 relation, LMUL m1-to-m2 relation,
      signed widening conversion kind, unit-stride memory form, policy, and
      runtime `n`/AVL.
- [x] RVVSelectedBodyRealization materializes the bounded conversion into
      explicit typed `tcrv_rvv` setvl/with_vl/load/conversion/store structure.
- [x] RVVEmitCRoutePlanning derives source/destination ABI types, vector types,
      setvl/load/store/conversion intrinsics, route metadata, and runtime ABI
      order from the typed body/config facts.
- [x] Unsupported conversion direction, unsupported SEW/LMUL relation, missing
      source/destination dtype or config, missing AVL/runtime role, wrong
      memory form, wrong policy, and incomplete body fail closed with targeted
      diagnostics.
- [x] Generated-bundle dry-run exists for the conversion slice and contains no
      descriptor/direct-C/source-export/source-front-door authority.
- [x] Real `ssh rvv` correctness evidence passes for counts 7, 16, and 23 if
      executable correctness is claimed.
- [x] Active-authority scan shows no positive reintroduction of `RVVI32M1`,
      `rvv-i32m1`, finite positive `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      source-front-door/source-seed authority, descriptor/direct-C/source-export
      authority, or common/export RVV semantic authority.
- [x] Focused build/lit/C++/script checks pass for touched RVV dialect,
      selected-body realization, route planning/provider, target artifact,
      generated-bundle harness, and relevant fail-closed tests.

## Definition Of Done

- One coherent commit contains the task, implementation, tests, validation
  updates, and journal entry.
- Trellis task status is truthful and archived only if acceptance criteria are
  complete.
- Final report names task id/title, phase, completed module behavior, changed
  files, checks, active-authority scan, task status, commit hash, and any exact
  continuation point.
