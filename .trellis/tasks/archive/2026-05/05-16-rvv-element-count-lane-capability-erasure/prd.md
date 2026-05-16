# RVV element-count and derived lane-capability erasure

## Goal

Delete RVV derived lane-count and descriptor-local element-count authority from
active capability/profile/probe/dialect/test/spec surfaces. RVV target evidence
may keep raw VLENB, hart/toolchain facts, and probe provenance; lane semantics
must come later from explicit RVV setvl/with_vl/config IR and runtime AVL/VL/ABI
control values, not from preserved finite i32/M1 capability facts.

## Context

- Previous HEAD `b04ddfa` removed source/future EmitC artifact placeholder
  authority and left a clean worktree.
- A follow-up scan identified active residue around `i32M1LaneCount`,
  `i32_m1_lane_count`, `rvv.i32_m1_lane_count`, `rvv_i32_m1_lanes`,
  `base_i32_m1_lanes`, `tcrv_rvv.element_count`, and RVV diagnostics that
  describe `element_count` as artifact-local component-capacity metadata.
- This is part of the deletion campaign: remove old descriptor/direct-exporter
  authority before rebuilding any new RVV config, EmitC, source route, or
  executable plugin model.

## Scope

### In Scope

- Remove C++ `i32M1LaneCount` storage, getter, validation coupling, and emitted
  `rvv.i32_m1_lane_count` capability from RVV capability profile code.
- Remove Python RVV probe JSON/schema/self-test/fixture output for
  `i32_m1_lane_count`, while retaining raw VLENB/toolchain/probe evidence.
- Remove or rewrite active tests and lit fixtures that assert
  `rvv_i32_m1_lanes`, `base_i32_m1_lanes`, `tcrv_rvv.element_count`, or RVV
  `element_count` as artifact-local/component-capacity metadata.
- Update directly related `.trellis/spec/` wording so current specs no longer
  preserve derived lane-count or descriptor-local element-count authority.
- Preserve genuine runtime ABI/control element-count surfaces such as
  `abi_runtime_element_count` when they are runtime values rather than RVV
  descriptor-local capacity facts.

### Out of Scope

- No new RVV config model.
- No general RVV lowering or EmitC route.
- No source artifact route, descriptor replacement, compatibility wrapper, or
  performance/evidence matrix.
- No renaming of the old lane count into a new capability key.
- No deletion of typed RVV extension-family ops unless strictly required to
  remove stale derived element-count/lane-count authority.

## Acceptance Criteria

- [ ] Active include/lib/test/scripts/.trellis/spec surfaces outside archived
      task records no longer define, serialize, assert, or document
      `i32M1LaneCount`, `i32_m1_lane_count`, `rvv_i32_m1_lanes`,
      `base_i32_m1_lanes`, `tcrv_rvv.element_count`, or RVV `element_count` as
      artifact-local/component-capacity metadata.
- [ ] RVV probe/profile code still fails closed when raw evidence is invalid or
      missing.
- [ ] RVV probe/profile code still carries raw VLENB, hart/toolchain facts, and
      probe/source/binary digest evidence where those are non-semantic facts.
- [ ] No RVV route or artifact can be selected from profile facts alone.
- [ ] Any remaining breakage requiring future general RVV config/EmitC modeling
      is reported as a missing new-architecture gap, not repaired by restoring
      lane-count or element-count authority.

## Minimal Validation

- Focused scans for `i32M1LaneCount`, `i32_m1_lane_count`,
  `rvv_i32_m1_lanes`, `base_i32_m1_lanes`, `tcrv_rvv.element_count`,
  `element_count as artifact-local`, and `component-capacity metadata` under
  active `include`, `lib`, `test`, `scripts`, and `.trellis/spec`, excluding
  archived Trellis task records and allowing genuine runtime ABI element-count
  names.
- `python3 scripts/rvv_remote_probe.py --self-test`
- Build and run `tianchenrv-rvv-extension-plugin-test`.
- Build and run `tianchenrv-rvv-dialect-test`.
- Run affected RVV lit tests.
- Attempt `check-tianchenrv` if the build remains coherent.
