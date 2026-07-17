# bounded RVV scalar dispatch executable path

## Goal

Advance the existing planned RVV+scalar i32-vadd dispatch self-check from source/object artifact export to one bounded executable/runtime evidence path. The generated dispatch path must consume compiler-produced planned dispatch artifacts and, when making any runtime correctness claim, compile/link/run on the real `ssh rvv` machine with sanitized evidence under `artifacts/tmp`.

## What I already know

* HEAD is expected to include `67841ed feat: route RVV scalar dispatch object artifacts`.
* Existing work added target-owned C++ source/object artifact routing for the planned RVV+scalar i32-vadd dispatch self-check.
* The remaining owner is a real bounded executable path, not generic evidence packaging and not a standalone smoke/probe.
* Generic target artifact routing must remain target-neutral; RVV/scalar-specific matching, ABI checks, selected `march`/`mabi`, and compile/link/run details stay in target-owned code.
* Python may only orchestrate runner/probe/evidence flows and must not encode compiler semantics, IR, legality, target selection, or ABI modeling.

## Requirements

* Use the existing target-owned RVV+scalar dispatch source/object export semantics and add only the minimum runner/evidence bridge needed for a bounded RVV+scalar i32-vadd dispatch self-check executable path.
* Preserve the existing planned dispatch pipeline and target-owned RVV scalar exporter as the source of generated dispatch source/object/executable artifacts.
* Preserve both explicit dispatch guard paths if the current self-check already exercises them; otherwise make the minimum target-owned change to exercise scalar fallback and RVV dispatch cases.
* Keep `tcrv.exec` focused on execution/capability/variant/dispatch/fallback and free of compute ops.
* Keep generic artifact routing target-neutral and based only on generic route metadata.
* Keep RVV/scalar-specific decisions plugin-local or target-owned.
* Preserve parameter layering:
  * VLEN/vlenb are hardware facts or target capability data.
  * SEW/LMUL/policy are compile-time variant/RVV dialect configuration.
  * AVL/vl/runtime `n` and dispatch availability are runtime SSA/control/ABI values.
  * `element_count` is bounded descriptor-local fixture metadata only.
  * `required_march`/`selected_march` stay bounded compile/toolchain metadata for this slice.
* Fail closed if `ssh rvv` compile/link/run is unavailable; do not fabricate runtime evidence.

## Acceptance Criteria

* Existing source and object export tests continue to pass.
* Focused C++/lit tests prove the new bounded executable behavior and route selection.
* If a generic route is added, tests show source-only selects the source route and executable/non-source selection reaches the bounded executable/object route without target-specific generic branches.
* Any runtime correctness claim is backed by real `ssh rvv` evidence, sanitized command logs, hashes, and a self-check success marker under `artifacts/tmp`.
* Local validation includes `git diff --check`, `cmake --build build --target tcrv-opt tcrv-translate -j2`, and `cmake --build build --target check-tianchenrv -j2` unless blocked.
* The task is validated and archived before the implementation commit.

## Out of Scope

* Generic RVV lowering or arbitrary kernel executable generation.
* Broad runtime integration, performance claims, or full correctness claims beyond the bounded self-check.
* Python implementation of compiler decisions, target selection, lowering semantics, IR, ABI models, plugin registry, capability model, variant pipeline, or emission semantics.
* Hard-coded RVV/scalar/vendor branches in generic core passes or generic target routing.
* Hand-authored C dispatch fixtures or prebuilt objects as runtime proof.
* Supervisor-policy changes unless the mandatory precondition discovers dirty supervisor-policy files.

## Technical Notes

* Mandatory precondition completed: repo path is `/home/kingdom/phdworks/TianchenRV`; worktree and supervisor-policy files were clean; HEAD was `67841ed`.
* Required inspection set is provided by the user and will be read before runtime edits.
