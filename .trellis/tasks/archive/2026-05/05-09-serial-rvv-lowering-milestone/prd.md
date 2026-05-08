# serial RVV lowering milestone selection

## Goal

Make `tcrv.exec.hart_parallel` participate in real capability-driven compiler
decisions. A variant-local hart/core parallel organization with
`harts = N` should be checked against structured target capability data, while
extension-specific hart count facts remain plugin-local and are exposed through
a generic capability relation instead of core RVV branches.

## What I Already Know

* The latest supervisor audit is
  `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0079-20260508T191854Z`.
* The previous round only archived bounded RVV+scalar dispatch bundle external
  ABI evidence; it did not change compiler/runtime/exporter code.
* Current HEAD is `dc8a49d`, and the worktree was clean before this task began.
* The repo already has `tcrv.exec.hart_parallel` as a core execution
  organization op, with structural verification only.
* RVV probe-derived capability data already includes `rvv.hart_count` with a
  positive `count` property, but core passes do not consume it for
  `hart_parallel`.
* The project requires capability facts to influence compiler decisions through
  C++/MLIR objects, not comments, Python-only data, or extension-specific core
  branches.

## Requirements

* Add a target-neutral compiler check for `tcrv.exec.hart_parallel`:
  `harts = N` must not exceed the available structured target hart/core count
  capability when such a capability is present.
* Keep RVV-specific facts plugin-local. The RVV hart-count capability may keep
  id `rvv.hart_count`, but should provide a generic capability id consumable by
  the core check.
* Do not add `if RVV` / `hasRVV` style logic to core orchestration.
* Preserve parameter layering: hart/core count is a hardware fact / target
  capability, not a runtime SSA value, descriptor-local element count, AVL, or
  VL.
* Insert the check into the execution-planning pipeline so the decision affects
  compiler planning, not only standalone tests.
* Add only focused lit/FileCheck and, where useful, small script test updates
  for the new compiler behavior.

## Acceptance Criteria

* [ ] A new or extended C++ MLIR pass rejects `tcrv.exec.hart_parallel`
      requests that exceed the structured available target hart count.
* [ ] The pass accepts valid `harts` requests and does not require a hart-count
      capability when no `harts` attribute is present.
* [ ] RVV probe replay emits the plugin-local `rvv.hart_count` capability with
      a generic capability relation for the core check.
* [ ] The RVV C++ capability profile builds the same generic relation from
      probe facts.
* [ ] The execution-planning pipeline runs this capability check before
      selected lowering/emission artifacts are materialized.
* [ ] Relevant specs document the generic relation and the capability-aware
      `hart_parallel` check.
* [ ] `cmake --build ... --target check-tianchenrv` passes locally, or any
      missing toolchain is reported exactly.

## Out Of Scope

* No new RVV executable kernel family.
* No automatic runtime hardware probing or dynamic dispatch integration.
* No performance claims, benchmarks, or broad RVV correctness claims.
* No Python implementation of compiler IR, passes, capability model, lowering,
  or emission.
* No subagents or parallel worker workflows.

## Technical Notes

* Relevant specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/capability-model/capability-contract.md`,
  `.trellis/spec/capability-model/profiles.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`.
* Likely code surfaces:
  `include/TianChenRV/Transforms/Passes.td`,
  `include/TianChenRV/Transforms/Passes.h`,
  `lib/Transforms/`,
  `lib/Plugin/RVV/RVVCapabilityProfile.cpp`,
  `scripts/rvv_probe_to_mlir.py`,
  `tools/tcrv-opt/tcrv-opt.cpp`,
  `test/Transforms/`,
  `test/Scripts/rvv-probe-to-mlir.test`.
