# Stage2 RVV Gearbox true multi-with_vl executable artifact ABI boundary

## Goal

Carry the provider-validated true multi-`with_vl` Gearbox widening-product ->
standalone reduce -> dequant-store route through the executable artifact
boundary. The explicit selected-body and pre-realized selected-body workflows
must line up from typed `tcrv_rvv` body facts through RVV-owned route
collection, `TCRVEmitCLowerableRoute`, common EmitC materialization, target
artifact export, generated bundle ABI, and real `ssh rvv` correctness evidence
when executable behavior is claimed. If the path cannot be made executable in
this round, the missing executable-boundary fact must fail closed in production
with a targeted continuation point.

## What I already know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* Repository started clean at `0f99eeab rvv: collect gearbox multi-with-vl
  routes`.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-gearbox-true-multi-with-vl-route-collection-boundary/`
  completed production route collection, validation, provider/header/target
  support, and fail-closed coverage for the true multi-`with_vl` Gearbox
  boundary.
* That previous task explicitly did not claim `ssh rvv` runtime correctness or
  performance because it stopped at provider/route/header validation.
* The current task must stay on the existing RVV Gearbox route family. It must
  not choose a different RVV route, broaden into dtype/LMUL clones, or make
  Common EmitC invent Gearbox semantics.

## Requirements

* Keep compiler implementation in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
  Python remains tooling, harness, runner, and artifact parsing only.
* Use selected/pre-realized typed `tcrv_rvv` body facts as route authority:
  producer `with_vl`, nested consumer `with_vl`, VL/AVL handoff, resource and
  schedule facts, dequant-store marker, dtype/config/policy, runtime ABI
  imports, and statement facts.
* RVV plugin/provider code must own route collection, validation, route payload,
  C type/header/intrinsic facts, ABI mapping, and fail-closed diagnostics.
* Common EmitC/export may materialize and package provider-built payload only.
  It must not infer Gearbox scope, handoff, dtype, schedule, ABI, intrinsic, or
  route support from route ids, artifact names, helper names, metadata, scripts,
  descriptors, or test names.
* Target artifact/header validation must compare provider-derived mirrors
  before accepting object/header/bundle metadata. Stale or missing mirrors at
  the executable boundary must fail closed before an executable support claim.
* Generated bundle ABI evidence must cover both explicit selected-body and
  pre-realized selected-body Gearbox paths when both are still supported.
* Runtime correctness claims require non-dry-run generated bundle execution on
  the real `ssh rvv` target. Dry-run evidence may validate source, header, and
  metadata shape only.
* If non-dry-run execution is blocked by an actual toolchain, harness, route, or
  artifact gap, the blocker must be made precise and either fail closed in
  production or be reported as the exact unfinished continuation point.

## Acceptance Criteria

* [x] The generated bundle path exposes the true multi-`with_vl` Gearbox route
  as an executable ABI workflow only after provider route facts and target
  object/header mirrors agree.
* [x] Positive explicit selected-body evidence reaches materialized selected
  boundary, emission plan, target artifact export, generated bundle compile,
  and `ssh rvv` correctness if executable behavior is claimed.
* [x] Positive pre-realized selected-body evidence reaches materialized
  selected boundary, emission plan, target artifact export, generated bundle
  compile, and `ssh rvv` correctness if executable behavior is claimed.
* [x] The generated harness verifies the Gearbox dequant-store result against a
  host oracle and protects inactive/tail or sentinel lanes where applicable.
* [x] Evidence records or checked output prove object/header metadata agree on
  provider support mirror, runtime ABI order, operand binding summary,
  multi-`with_vl` handoff facts, header/type mapping, dequant-store facts, and
  statement plan facts.
* [x] At least one focused fail-closed test rejects a stale or missing
  executable-boundary fact such as stale VL/AVL handoff, missing nested consumer
  `with_vl`, stale producer/consumer scope, stale dequant-store marker, stale
  header/prototype binding, stale route-family validation contract, wrong C
  type, wrong ABI value mapping, or unsupported executable route claim.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-target-artifact-export-test`, and relevant construction
  protocol or generated-bundle checks pass.
* [x] Relevant dry-run and non-dry-run
  `scripts/rvv_generated_bundle_abi_e2e.py` commands are run and reported.
* [x] Bounded old-authority scan over touched files and added diff lines shows
  no new positive legacy `i32m1` route authority, descriptor-driven compute,
  source-front-door authority, or artifact/status authority.
* [x] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are reported.

## Validation Completed

* `scripts/rvv_generated_bundle_abi_e2e.py --self-test` passed after adding
  product-dequant Gearbox/resource summary assertions.
* Explicit and pre-realized
  `widening_product_reduce_dequantize_f32` dry-runs passed with runtime counts
  `0, 1, 17, 65` and dequant scales `0.25, -0.5`.
* Explicit and pre-realized
  `widening_product_reduce_dequantize_f32` non-dry-run generated bundle
  executions passed on `ssh rvv` for runtime counts `0, 1, 17, 65`, two data
  patterns, and scales `0.25, -0.5`; output, source preservation, accumulator
  preservation, and tail/sentinel preservation were checked by the harness.
* `build/bin/tianchenrv-rvv-extension-plugin-test`,
  `build/bin/tianchenrv-target-artifact-export-test`, and
  `build/bin/tianchenrv-construction-protocol-common-test` passed.
* `llvm-lit`/`FileCheck` were not available in the current environment, so the
  two MLIR RUN-line fixtures were not replayed through lit in this round.

## Out of Scope

* No broad Gearbox matrix, dtype/LMUL clone batch, or clamp expansion except as
  a bounded harness reference for the existing route family.
* No unrelated MAcc, segment, memory, reduction, conversion, compare/select, or
  source-front-door positive route work.
* No high-level Linalg/Vector/StableHLO frontend, per-Linalg route authority,
  performance tuning database, dashboard, report-only closeout, or descriptor /
  direct-C / source-export route.
* No Common EmitC invention of Gearbox semantics outside consuming provider
  route payload.
* No runtime correctness or performance claim without real `ssh rvv` evidence.

## Technical Notes

* Required specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/validation/index.md`,
  `.trellis/spec/variant-pipeline/index.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`, and
  `.trellis/spec/guides/index.md`.
* Shared guides:
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Previous task context:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-gearbox-true-multi-with-vl-route-collection-boundary/prd.md`,
  `implement.jsonl`, and `check.jsonl`.
* Brief-designated code targets include RVV EmitC route planning/provider code,
  `tcrv_rvv.with_vl` verifier rules, Gearbox schedules and selected-body
  realization owner, RVV extension plugin and construction protocol, target
  construction template artifact adapter, RVV target support bundle,
  `scripts/rvv_generated_bundle_abi_e2e.py`, and the explicit/pre-realized
  widening-product-reduce-dequantize-f32 target artifact fixtures.
