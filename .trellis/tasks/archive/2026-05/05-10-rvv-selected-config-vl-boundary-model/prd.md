# First-class RVV selected config and runtime VL boundary model

## Goal

Make the existing finite RVV binary selected path carry parameter meaning
through explicit compiler-owned C++/MLIR contracts. Hardware/profile facts,
selected compile-time RVV config, runtime AVL/VL control, and descriptor-local
artifact values must remain separate from proposal planning through selected
lowering-boundary materialization, selected emission planning, microkernel
materialization, and target artifact/export metadata.

This is a bounded compiler-abstraction task for the existing finite RVV binary
path. It is not another evidence-only round and it does not claim a generic RVV
backend.

## Why Now

The latest archived `i64-vsub` evidence task proved the post-extraction
selected path on current code and real `ssh rvv` hardware. Recent source work
already split proposal planning, selected emission planning, selected
lowering-boundary materialization, and variant legality into plugin-local RVV
modules. The next bottleneck is durable parameter layering: selected vector
shape metadata exists, but the selected plan still needs a single first-class
selected-config/runtime-boundary model so hardware capacity facts, selected
SEW/LMUL/policy decisions, runtime AVL/VL ownership, and descriptor-local
`element_count` cannot drift or be read as each other.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial real state for this round is clean on `main` at
  `07d0b5d test(rvv): prove i64-vsub selected artifact path`.
- No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes brief as
  `.trellis/tasks/05-10-rvv-selected-config-vl-boundary-model`.
- The previous archived task
  `.trellis/tasks/archive/2026-05/05-10-rvv-i64-vsub-selected-path-artifact-evidence`
  completed bounded `i64-vsub` selected-path artifact evidence and fresh real
  `ssh rvv` correctness evidence.
- Existing extracted owners are:
  `RVVBinaryPlanning`, `RVVBinaryVariantLegality`,
  `RVVBinarySelectedLoweringBoundary`,
  `RVVBinarySelectedEmissionPlanning`, and
  `RVVBinaryMicrokernelMaterialization`.
- The current finite RVV binary family scope remains exactly `i32-vadd`,
  `i32-vsub`, `i32-vmul`, `i64-vadd`, `i64-vsub`, and `i64-vmul`.
- Hardware/profile facts such as `vlenb` and base i32 M1 lane capacity belong
  to capability/profile/probe objects.
- Selected compile-time config facts such as SEW, LMUL, tail policy, mask
  policy, vector suffix, and setvl suffix belong to plugin/target-owned
  selected variant config metadata.
- Runtime `n`, AVL, VL, dispatch guards, pointers, and callable ABI parameters
  belong to real IR/runtime ABI/control surfaces.
- Descriptor-local bounded values such as `tcrv_rvv.element_count` describe a
  finite emitted descriptor slice only; they are not runtime trip counts,
  problem sizes, AVL, VL, correctness coverage, or performance evidence.

## Requirements

- Inventory where the current finite RVV binary path stores and consumes:
  hardware facts, selected compile-time config, runtime AVL/VL/control, and
  descriptor-local `element_count`; record the inventory in this task context
  as supporting context rather than the main deliverable.
- Introduce or tighten a plugin-local C++ selected-config contract around the
  existing `RVVVectorShapeConfig` / selected-plan metadata so selected
  SEW/LMUL/tail/mask/vector/setvl facts are consumed consistently by proposal
  planning, legality, selected lowering-boundary materialization, selected
  emission planning, and microkernel artifact generation.
- Keep target capability/profile code as the owner of hardware availability and
  shape-availability facts, not the selected per-variant decision. If existing
  names or comments blur this ownership, repair them minimally.
- Make runtime AVL/VL ownership explicit for the current C-intrinsics artifact
  path: identify where runtime AVL enters the callable ABI or generated helper,
  where VL is produced by `vsetvl` / `tcrv_rvv.setvl`, and ensure neither is
  modeled as static target capability or descriptor-local shape.
- Preserve existing finite-family artifact behavior for `i64-vsub` and at
  least one existing i32 family fixture.
- If the implementation scope is too large for one round, complete the
  selected-config contract first, keep the task open, and record the exact
  continuation point for runtime VL/AVL consumption.

## Acceptance Criteria

- [x] PRD and task context explicitly define hardware facts, selected RVV
      config, runtime AVL/VL control, and descriptor-local `element_count` as
      separate layers.
- [x] Inventory names the concrete current C++/MLIR owners and consumers for
      each layer across proposal planning, legality, selected boundary,
      selected emission plan, microkernel materialization, and target export.
- [x] A named plugin-local selected-config contract exists or is tightened so
      selected SEW/LMUL/tail/mask/vector suffix/setvl suffix facts are consumed
      through one structured surface rather than ad hoc duplicated metadata.
- [x] `RVVCapabilityProfile` / target profile code remains limited to hardware
      and capability availability facts; selected per-variant config remains
      plugin/target selected-plan metadata.
- [x] Runtime AVL/VL boundary metadata or validation is visible and checkable
      in the selected C-intrinsics path without pretending to be a full SSA VL
      lowering route.
- [x] Focused coverage proves selected-config metadata and runtime AVL/VL
      boundary metadata survive the selected path for `i64-vsub` and at least
      one existing i32 route without changing generated artifact semantics
      unexpectedly.
- [x] Focused C++ tests for touched RVV plugin/target modules pass.
- [x] Focused lit/FileCheck tests for selected config and runtime AVL/VL
      boundary behavior pass.
- [x] `git diff --check` passes.
- [x] No new RVV runtime correctness or performance claim is made unless backed
      by fresh real `ssh rvv` evidence.

## Non-goals

- No generic RVV backend claim.
- No MLIR vector/scalable-vector lowering route.
- No new dtype, arithmetic family, broad frontend project, or broad smoke
  matrix.
- No Python implementation of compiler internals, dialects, passes, plugin
  registry, capability model, lowering, emission, selected config, runtime ABI,
  or route selection.
- No compute semantics in `tcrv.exec`.
- No RVV semantic branches in generic core passes when plugin-local or
  target-local code owns the behavior.
- No helper-only, report-only, manifest-only, one-extra-test, or evidence-only
  closeout as the main result.
- No runtime correctness or performance claim without fresh real `ssh rvv`
  evidence.

## Minimal Validation Plan

- `git diff --check`
- Build focused touched RVV plugin/target/transform libraries plus `tcrv-opt`
  and `tcrv-translate`.
- Run focused C++ tests for RVV capability profile, binary planning, variant
  legality, selected lowering-boundary, selected emission planning, microkernel
  materialization, and target artifact export as applicable.
- Run focused lit/FileCheck tests showing selected config metadata and runtime
  AVL/VL boundary metadata survive the selected path for `i64-vsub` and one
  existing i32 route.
- Run dry-run artifact evidence for `i64-vsub` after contract changes:
  `python3 scripts/rvv_microkernel_e2e.py --dry-run --arithmetic-family=i64-vsub --lower-linalg-frontend --input test/Transforms/LinalgToExec/linalg-i64-vsub-to-rvv-artifact.mlir --run-id codex-selected-config-vl-boundary-i64-vsub-dry --overwrite`
- Run real `ssh rvv` only if this round changes generated runtime semantics or
  makes a fresh runtime correctness claim.
- Run `cmake --build artifacts/tmp/tianchenrv-build --target check-tianchenrv
  -j2` if practical after focused checks pass.
- Validate this Trellis task path before finish/archive.

## Technical Notes

- Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/capability-model/capability-contract.md`,
  `.trellis/spec/plugin-protocol/interfaces-and-registry.md`,
  `.trellis/spec/plugin-protocol/locality-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md`.
- Archived PRDs read:
  `.trellis/tasks/archive/2026-05/05-10-rvv-binary-proposal-legality-planning/prd.md`,
  `.trellis/tasks/archive/2026-05/05-10-rvv-binary-variant-legality-validation/prd.md`,
  `.trellis/tasks/archive/2026-05/05-10-rvv-binary-selected-lowering-boundary-materialization/prd.md`,
  `.trellis/tasks/archive/2026-05/05-10-rvv-binary-selected-emission-planning/prd.md`,
  `.trellis/tasks/archive/2026-05/05-10-rvv-selected-vector-shape-config-boundary-cleanup/prd.md`,
  `.trellis/tasks/archive/2026-05/05-10-rvv-i64-vsub-selected-path-artifact-evidence/prd.md`.

## Inventory

### Hardware/profile facts

- Owners:
  `RVVProbeCapabilityFacts`,
  `buildRVVTargetCapabilitiesFromProbeFacts`,
  `RVVCapabilityProfile`, and target-side RVV capability descriptors.
- Concrete facts:
  target architecture, hart count, `vlenb_bytes`, base i32 M1 lane capacity
  (`i32_m1_lane_count` / `base_i32_m1_lanes`), first-slice SEW/LMUL,
  i64 M1 shape availability, selected toolchain `march`/`mabi`, and probe
  provenance.
- Consumers:
  `buildRVVBinaryCapabilityPropertyView` reads the capability set and derives a
  planning view; proposal planning and variant legality validate whether a
  selected shape is available; selected emission planning and target export
  validate required capability symbols, required `march`/`mabi`, and capacity
  metadata.
- Boundary rule:
  these facts describe target/profile availability and capacity. They do not
  decide the per-variant selected SEW/LMUL/policy choice and they are not
  runtime AVL/VL values.

### Selected compile-time RVV config

- Owners:
  target-local `RVVVectorShapeConfig` definitions and the plugin-local
  `RVVBinarySelectedConfig` selected-plan wrapper introduced/tightened in this
  task.
- Concrete facts:
  selected shape id, dtype id, SEW, LMUL, tail policy, mask policy, vector type,
  vector suffix, setvl suffix, and required shape capability ids.
- Consumers:
  `RVVBinaryPlanning` selects the config and builds intrinsic descriptors;
  `RVVBinaryVariantLegality` checks selected metadata and capability
  compatibility; `RVVBinarySelectedLoweringBoundary` preserves selected metadata
  while materializing the boundary and explicit microkernel op;
  `RVVBinarySelectedEmissionPlanning` exposes selected-plan metadata and checks
  selected microkernel control; `RVVBinaryMicrokernelMaterialization` materializes
  the typed RVV op/control surface; target export validates the same selected
  config before emitting C intrinsics.
- Boundary rule:
  selected config is a compile-time variant decision. It is not the hardware
  capacity fact, runtime problem size, AVL/VL token, or descriptor-local
  `element_count`.

### Runtime AVL/VL control

- Owners:
  runtime ABI planning, the selected lowering boundary, explicit RVV dialect
  control ops, and target export.
- Concrete facts:
  the selected callable ABI owns runtime parameter `n` with role
  `runtime-element-count`; selected boundary materializes
  `tcrv.exec.runtime_param @abi_runtime_element_count`; the RVV microkernel body
  passes the runtime index block argument as AVL to `tcrv_rvv.setvl`;
  `tcrv_rvv.setvl` produces `!tcrv_rvv.vl`; `tcrv_rvv.with_vl` scopes the
  vector body using that VL token; generated C computes
  `__riscv_vsetvl_*(n - offset)` inside the runtime loop.
- Consumers:
  selected emission planning validates the explicit i32 control plane and now
  emits checkable runtime AVL/VL boundary metadata; i64 selected paths are
  validated by the RVV dialect verifier and target exporter; target export
  validates the body argument, setvl suffix, policy, VL token consumption, and
  callable ABI parameter roles.
- Boundary rule:
  AVL/VL are runtime control values. They are not target/profile capability
  facts, selected vector-shape config, or descriptor-local `element_count`.

### Descriptor-local `element_count`

- Owners:
  the finite RVV binary selected descriptor and its MLIR metadata
  (`tcrv_rvv.element_count`) on selected variants and explicit microkernel ops.
- Concrete facts:
  proposal planning derives it from selected finite descriptor requirements and
  base capacity when needed; lowering-boundary materialization preserves it on
  the selected descriptor and microkernel; emission planning/export use it to
  validate the finite artifact descriptor.
- Consumers:
  proposal metadata, selected lowering-boundary checks, selected emission-plan
  descriptor checks, target export descriptor validation, and harness/materialized
  artifact metadata.
- Boundary rule:
  `element_count` is a bounded descriptor-local finite slice value. It is not
  runtime `n`, not AVL, not VL, not VLEN/vlenb, and not correctness or
  performance evidence.
