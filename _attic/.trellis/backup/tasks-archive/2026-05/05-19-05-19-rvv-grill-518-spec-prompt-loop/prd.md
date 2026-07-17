# brainstorm: reconcile grill 518 into specs prompts and stage1 loop

## Goal

Read and digest `artifacts/grill-rvv-maturity-ladder-20260518.md` end to end,
then audit the current TianChen-RV specs and supervisor prompts for drift. Fix
misleading or stale wording by rewriting or deleting it, not just adding new
paragraphs. The final state should steer the next supervisor loop into RVV
Stage 1 route-authority replacement from a clean, committed worktree.

## What I Already Know

- User explicitly wants this treated as a real task now.
- The grill file is a continuous Q&A, so early statements may be corrected by
  later answers. Stable conclusions must be taken from the converged state, not
  isolated snippets.
- Current code still contains legacy RVV route-table authority:
  `RVVI32M1ArithmeticRouteSpec`, `RVVI32M1ArithmeticSlice`,
  `collectRVVI32M1ArithmeticSlice`, finite `i32_*` cases, broadcast-load,
  compare/select, and exact `__riscv_*_i32m1` intrinsic spellings.
- Current specs already have partial updates, but previous review found one
  concrete bad pattern: adding warnings above old scenario text while leaving
  the old 7-part positive template in place.
- `tcrv.exec` must stay an execution envelope / ABI-runtime binding surface.
  Dtype, operation semantics, SEW/LMUL/policy, memory form, VL/AVL placement,
  and intrinsic mapping must come from typed `tcrv_rvv` body/config plus RVV
  plugin validation/realization, not from route ids, helper names, descriptors,
  artifacts, or common EmitC/export code.

## Requirements

- Read the full grill document and extract stable technical constraints.
- Audit all relevant `.trellis/spec/**` files and the canonical supervisor
  prompt/review prompt against those constraints.
- Delete or rewrite stale sections that still encode old positive architecture.
- Preserve useful current-code facts only when clearly marked as legacy debt,
  migration scaffolding, or fail-closed residue.
- Keep specs as long-term executable contracts, not current progress reports.
- Update supervisor prompt/review prompt so Hermes chooses Stage 1
  route-authority replacement and does not treat legacy route-table expansion
  as progress.
- Commit the spec/prompt corrections before starting the loop.
- Start the supervisor loop from Stage 1 after verification.

## Acceptance Criteria

- [ ] Grill 518 is read end to end, with stable conclusions recorded in this
      task or implementation notes.
- [ ] Specs no longer describe legacy `RVVI32M1*` broadcast/compare/select/
      dtype/LMUL/source-shape additions as Stage 2 coverage or future owner
      templates.
- [ ] Specs clearly separate parseable/verifier-legal, route-supported, and
      executable.
- [ ] Specs clearly separate `tcrv.exec` envelope/runtime role binding from
      typed `tcrv_rvv` body/config authority and RVV plugin realization.
- [ ] Prompt/review prompt contains enough direction to keep Hermes from
      selecting small helper/test/report owners or old route-table expansion.
- [ ] `git diff --check` and Python syntax checks for supervisor script pass.
- [ ] Work is committed cleanly.
- [ ] Supervisor loop is started with a Stage 1 steering brief and status is
      recorded.

## Out Of Scope

- Implementing the RVV route-surface replacement in C++ in this task.
- Adding new RVV dtype/LMUL/broadcast/compare/reduction route cases.
- Running broad unrelated tests.
- Treating grill artifacts as source of truth after their durable rules are
  promoted into specs/prompts.

## Technical Notes

- Relevant specs: extension-plugins/RVV, plugin-protocol, variant-pipeline,
  lowering-runtime, implementation-stack/supervision-loop, testing where
  route-table fixtures are discussed.
- Relevant prompts/scripts: `scripts/codex_serial_supervisor_prompt.md` and
  `scripts/codex_serial_supervisor.py`.
- Relevant source evidence: `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`,
  `lib/Dialect/RVV/IR/RVVConfigContract.cpp`.

## Grill 518 Stable Conclusions

- The final RVV chain is `tcrv.exec` envelope -> selected RVV variant ->
  vector-level typed `tcrv_rvv` body -> RVV plugin legality/realization/route
  provider -> `TCRVEmitCLowerableRoute` -> common MLIR EmitC materialization ->
  C/C++/artifact/runtime evidence.
- Route provider is not a new IR, decorator, dashboard, status machine, or
  direct C printer. It reads the selected typed body, validates legality, maps
  to vector types/intrinsics/ABI/header payload, and builds the route consumed
  by common EmitC.
- `tcrv.exec` binds execution envelope and runtime/ABI roles. It must not infer
  RVV compute, dtype, memory form, policy, schedule, or intrinsic choice from
  parameter names, route ids, artifact labels, descriptors, or C strings.
- The correct replacement surface is typed vector-level `tcrv_rvv`
  value/config/body structure. Adding more `tcrv_rvv.i32_*` helpers or
  route-table cases is not dtype propagation and is not Stage 2 progress unless
  the old case is retained only as an ordinary specialization of the corrected
  surface.
- Stage 1 is route-authority replacement: delete, fail-close, or replace
  legacy i32m1 route authority. Stage 2 begins only after active paths no
  longer use `RVVI32M1*`, finite `i32_*` route cases, route ids, source
  patterns, or exact `__riscv_*_i32m1` spellings as architecture.
- Stage 2 includes both route-supported RVV coverage on the corrected surface
  and one-time RVV plugin-local selected-body realization. It is not high-level
  Linalg/Vector frontend work, one-intrinsic wrappers, dtype/LMUL clone
  batches, global autotuning, or readiness/status machinery.

## Current Code Evidence Checked

- `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` still defines
  `RVVI32M1ArithmeticRouteSpec`, `RVVI32M1ArithmeticSlice`,
  `collectRVVI32M1ArithmeticSlice`, exact `__riscv_*_i32m1` intrinsic names,
  optional `i32_broadcast_load`, and `cmp_select`.
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h` exposes public
  `buildRVVI32M1*EmitCLowerableRoute` helpers and no corrected generic typed
  RVV route surface yet.
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` describes finite
  `!tcrv_rvv.i32m1` / `!tcrv_rvv.i32m2` token types and explicit
  `i32_broadcast_load`, `i32_cmp_eq`, and `i32_select` ops as bounded slice
  surfaces, not final RVV maturity architecture.
