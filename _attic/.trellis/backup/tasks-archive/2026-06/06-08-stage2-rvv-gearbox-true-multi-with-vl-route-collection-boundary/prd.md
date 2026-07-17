# Stage2 RVV Gearbox true multi-with_vl route collection boundary

## Goal

Extend the existing Gearbox widening-product -> reduction -> f32 dequant/store
path from the current bounded two-scope handoff model into a true
multi-`tcrv_rvv.with_vl` route-collection boundary. The RVV dialect/plugin
path must structurally carry producer/consumer region order, explicit VL
handoff facts, runtime AVL/VL, resource/schedule facts, typed config/policy,
and provider-owned route/header mirrors into `TCRVEmitCLowerableRoute`
construction. Unsupported or stale multi-region facts must fail closed before
Common EmitC or target artifact metadata can claim support.

## What I already know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* Repository started clean at `01a7baad rvv: model gearbox two-scope handoff
  boundary`.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-gearbox-multiregion-with-vl-boundary/`
  implemented the bounded Gearbox product-reduction to dequant-store
  two-scope handoff, provider planning, target artifact/header validation, and
  tests.
* The previous PRD explicitly described the remaining limitation: production
  route collection still had a one-`with_vl` shape, and the two Gearbox scopes
  were stripe-ready facts rather than true split-region route collection.
* The current task must stay in the RVV dialect/plugin/provider/target
  validation boundary. Common EmitC and artifact metadata may carry provider
  facts only as mirrors.

## Requirements

* Keep compiler implementation in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
  Python remains tooling only.
* Support the smallest directly related true multi-`with_vl` Gearbox workflow:
  the widening-product -> standalone reduction -> handoff -> f32 dequant/store
  route.
* Region/scope order must be structural: producer product/reduction scope
  precedes consumer dequant/store scope, and both scopes must participate in
  the same selected RVV route.
* VL handoff must be explicit: the consumer scope must derive its active VL
  from the producer handoff and must not silently invent a stale or unrelated
  `setvl`.
* Provider-owned route collection must consume region order, handoff value,
  typed config/policy, runtime AVL/VL, resource/schedule facts, and ABI/header
  mirror facts before declaring route support.
* Common EmitC must remain neutral. It must not infer Gearbox scope, VL
  handoff, schedule, dtype, intrinsic spelling, or route support from names or
  metadata.
* Target artifact/header validation may compare provider-derived mirror facts
  but must not become route authority.
* Missing, stale, duplicated, or inconsistent multi-region facts must fail
  closed with targeted RVV-owned diagnostics.
* Do not claim RVV runtime correctness or performance unless a generated
  artifact is actually compiled and run on `ssh rvv`.

## Acceptance Criteria

* [ ] A true multi-`with_vl` Gearbox route collection path exists in production
  RVV dialect/plugin/provider code for the bounded widening-product
  reduce-dequant route.
* [ ] The route collector distinguishes producer and consumer `with_vl`
  regions and validates their structural order.
* [ ] The Gearbox cross-region handoff carries or derives explicit VL handoff
  facts across those regions; stale or missing handoff facts fail closed.
* [ ] Provider validation consumes resource/schedule, dtype/config/policy,
  runtime AVL/VL, region order, and ABI/header mirror facts before route
  construction.
* [ ] Target artifact/header validation rejects stale or missing provider
  mirrors for the touched Gearbox route facts.
* [ ] Focused positive coverage proves the explicit and/or pre-realized
  widening-product-reduce-dequantize-f32 path can be collected as a provider
  validated multi-`with_vl` route.
* [ ] Focused fail-closed coverage covers stale or missing region order,
  scope pairing, VL handoff, resource facts, runtime AVL/VL, header, or
  provider mirror facts for the touched boundary.
* [ ] Relevant C++ tests and lit/FileCheck tests pass, including RVV extension
  plugin/provider tests, construction protocol tests if touched, and the
  targeted Gearbox artifact fixtures.
* [ ] Bounded old-authority scan over touched diff lines shows no new positive
  legacy `i32m1` route authority, descriptor-driven compute, source-front-door
  authority, or artifact/status authority.
* [ ] `git diff --check`, `git diff --cached --check`, and final
  `git status --short` are reported.

## Out of Scope

* No broad Gearbox architecture rewrite.
* No high-level frontend, Linalg contract, or source-front-door positive route.
* No performance/autotuning database or runtime/performance claim.
* No dtype/LMUL clone batch, unrelated MAcc/mask/reduction/memory expansion,
  or new dtype-prefixed helper op family.
* No Common EmitC invention of Gearbox scope, VL, resource, dtype, schedule, or
  route semantics.
* No artifact metadata, route id, helper name, or test name as route authority.

## Technical Notes

* Required specs read for this PRD:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/variant-pipeline/index.md`, and
  `.trellis/spec/guides/index.md`.
* Shared guides read:
  `.trellis/spec/guides/capability-first-design-guide.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`, and
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Previous task context read:
  `.trellis/tasks/archive/2026-06/06-08-stage2-rvv-gearbox-multiregion-with-vl-boundary/prd.md`.
* Brief-designated code targets include RVV ops/verifier, Gearbox schedule and
  selected-body realization owners, RVV EmitC route planning/provider code,
  target artifact validation/support bundle code, and the focused
  widening-product-reduce-dequantize-f32 fixtures.
