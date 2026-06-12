# rvv i32 vadd dataflow body export consumption

## Goal

Implement the smallest bounded plugin-owned RVV i32 vector-add dataflow body
under the existing `tcrv_rvv.setvl` / `tcrv_rvv.with_vl` structured
control-plane path, and make the RVV exporter validate and consume that body
before emitting the existing finite RVV intrinsic C source route.

## Requirements

- Keep `tcrv.exec` compute-free and focused on execution, capability, variant,
  dispatch, and fallback organization.
- Add only RVV dialect/plugin-local IR needed for the finite i32-vadd
  microkernel dataflow slice.
- Materialize the dataflow body from the selected finite RVV descriptor in the
  RVV plugin.
- Require explicit hand-authored fixtures to carry the same structured
  dataflow body under `tcrv_rvv.with_vl`.
- Make RVV target export fail closed when the dataflow body is missing,
  duplicated, stale, malformed, or mismatched.
- Preserve parameter layering: hardware facts remain capability/profile facts;
  SEW/LMUL/policy remain compile-time variant config; runtime `n`/AVL/VL stays
  runtime/control/ABI; `element_count` remains descriptor-local fixture/export
  metadata.
- Keep generic shared passes target-neutral and free of RVV/scalar/offload/IME
  family-specific compute branching.

## Out of Scope

- No Python implementation of compiler, dialect, plugin, lowering, emission, or
  runtime decisions.
- No generic RVV lowering, generic vector/memory IR, scheduler, full runtime
  ABI layer, performance model, or broad smoke/probe matrix.
- No supervisor-policy edits unless the initial scoped git status shows the
  exact supervisor-policy files are dirty.
- No RVV runtime/correctness/performance claim unless a real `ssh rvv`
  compile/run is performed and recorded.

## Acceptance Criteria

- `tcrv_rvv` contains a bounded i32-vadd dataflow body surface nested under
  `tcrv_rvv.with_vl`.
- RVV plugin materialization emits:
  - one runtime index body argument for target/export-owned `n`/AVL;
  - one `tcrv_rvv.setvl`;
  - one matching `tcrv_rvv.with_vl`;
  - one finite i32-vadd dataflow op under the `with_vl` body.
- RVV exporter validates and consumes the structured dataflow body before
  emitting the existing intrinsic C loop.
- Focused lit/FileCheck or C++ tests cover parse/verify/print, auto
  materialization, direct export, generic target-source export, negative
  missing/duplicate/malformed dataflow cases, and continued RVV+scalar dispatch
  behavior.
- Required local checks pass:
  - `git diff --check`
  - `cmake --build build --target check-tianchenrv -j2`

## Notes

- Initial precondition confirmed:
  - repo root: `/home/kingdom/phdworks/TianchenRV`
  - supervisor-policy files were clean before compiler edits
  - HEAD before work: `c13a02d feat: consume structured RVV i32 vadd control body`
