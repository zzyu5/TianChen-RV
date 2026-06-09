# Stage2 RVV low-precision contraction primitive-surface campaign

## Goal

Create a macro Trellis owner for the RVV Stage 2 low-precision contraction
primitive surface. The campaign must make typed low-precision contraction facts
structurally explicit in the RVV plugin/compiler path, fail closed before route
or target acceptance when those facts are missing or stale, and leave later
gates to consume the surface for Gearbox realization, executable artifact
evidence, and measured dispatch/performance policy. The current round owns
Gate 1.

## What I Already Know

- There is no active Trellis task at session start.
- The previous packed-i4 macro campaign is archived at
  `.trellis/tasks/archive/2026-06/06-09-rvv-packed-i4-production-kernel-resource-aware-realization-campaign/`.
- Commit `af53faa3` plus journal commit `4c99df74` completed the packed-i4
  campaign Gates 1-4, including performance-policy consumption of same-target
  no-win/regression evidence.
- The next owner should not be another generated-bundle or `ssh rvv` evidence
  closeout. It should harden the production compiler surface used by selected
  typed `tcrv_rvv` low-precision contraction bodies and RVV provider planning.
- The RVV plugin spec already records bounded unsigned u8 widening-product and
  low-precision widening-reduction primitive-fact contracts. Gate 1 should make
  the production source surface explicit and consumable rather than adding
  q8/q4 route names, artifact metadata authority, or helper-name semantics.
- Durable authority remains:
  selected typed `tcrv_rvv` body -> RVV plugin validation/realization ->
  provider-built `TCRVEmitCLowerableRoute` -> common EmitC materialization ->
  target artifact mirrors -> `ssh rvv` evidence only for runtime/correctness or
  performance claims.

## Requirements

- Keep this as one macro Trellis task until all campaign gates are complete or
  human steering redirects it.
- Implement Gate 1 as one coherent production compiler-surface milestone.
- Make low-precision contraction primitive facts structurally consumable by RVV
  route/provider planning.
- Gate 1 primitive facts must include, for the bounded current slice, explicit
  source input element roles, source signedness, source SEW/LMUL, widened
  product element type/SEW/LMUL, accumulator and reduction/result type facts
  when the route claims accumulation, policy, memory form, and runtime AVL/VL
  participation.
- Provider/target validation must reject missing, stale, or mismatched
  primitive facts before accepting route support or target artifact mirrors.
- The implementation must derive or validate facts from selected typed body,
  config, capability, and runtime/provider-owned route data. It must not accept
  q8/q4 labels, route ids, artifact names, helper names, ABI strings, test
  names, metadata-only mirrors, descriptors, or Common EmitC semantic branches
  as authority.
- If every low-precision primitive cannot be finished in one round, complete a
  coherent production submodule and keep the macro task active with the exact
  continuation point.

## Macro Campaign Gates

- [x] Gate 1: explicit typed low-precision contraction primitive surface and
  fail-closed provider validation for i8/u8 input element roles, signedness,
  widened product type, accumulation/result type, SEW/LMUL/policy, memory form,
  and runtime AVL/VL facts.
- [ ] Gate 2: resource-aware Gearbox selected-body realization consumes that
  surface for packed/low-precision bodies without changing semantics or using
  q8/q4 names as authority.
- [ ] Gate 3: generated artifact and `ssh rvv` evidence for at least one
  production primitive path changed by Gates 1-2.
- [ ] Gate 4: same-target measurement and dispatch/performance policy
  consumption using provider-owned evidence, including conservative fallback
  when no win is measured.

## Current Slice: Gate 1

- [x] Inspect existing widening-product, unsigned-u8, widening-reduction,
  Gearbox, selected-body realization, route planning, and target artifact
  validation code to locate the smallest production owner-local surface.
- [x] Add or harden one production RVV plugin fact model for low-precision
  contraction primitive facts.
- [x] Connect the fact model to provider/route planning or target validation so
  it is consumed by the production path in this round.
- [x] Add positive focused tests for accepted primitive facts.
- [x] Add negative focused tests for stale or missing facts that must fail
  closed before route/provider or target acceptance.
- [x] Run focused affected test binaries and route/export checks.
- [x] Run bounded old-authority scans over touched files and added diff lines.
- [x] Keep the macro task active after the Gate 1 slice because Gates 2-4
  remain.

## Acceptance Criteria

- [x] Production RVV plugin/compiler code exposes low-precision contraction
  primitive facts through an owner-local surface that route/provider planning or
  target validation consumes.
- [x] The current slice covers at least signed/unsigned 8-bit source input
  facts plus widened product and accumulator/result validation for a bounded
  low-precision contraction primitive path, unless source inspection proves a
  narrower owner is the only coherent production seam.
- [x] Missing or stale signedness, source/product/accumulator/result element
  type, SEW/LMUL, policy, memory form, runtime AVL/VL, intrinsic, layout, or
  mirror facts fail closed with targeted RVV/provider/target diagnostics.
- [x] Common EmitC remains neutral: it carries provider payload only and does
  not infer dtype/sign, SEW/LMUL, intrinsic spelling, layout, runtime roles, or
  support state.
- [x] No q8/q4/llama label, route id, artifact name, helper name, ABI string,
  test name, descriptor residue, or metadata-only mirror becomes route,
  dtype/config, or evidence authority.
- [x] Focused C++/lit/script tests exercise accepted facts and stale/missing
  fact rejection for the changed production seam.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes if RVV plugin
  code/tests are touched.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes if target
  artifact validation/export code/tests are touched.
- [x] Relevant `tcrv-opt` or `tcrv-translate` checks pass if IR/export paths
  change.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] One coherent commit records the Gate 1 slice.
- [x] The macro task remains active with Gates 2-4 unchecked and a precise
  continuation point.

## Out Of Scope

- No standalone generated-bundle or `ssh rvv` evidence closeout as the main
  task.
- No one-off q8/q4/llama wrapper or route authority.
- No broad dtype/LMUL clone batch.
- No high-level Linalg/Vector/StableHLO frontend.
- No per-Linalg route authority.
- No Common EmitC semantic invention.
- No source-front-door positive route.
- No IME, Offload, TensorExt, Template, or Toy work.
- No performance tuning database, dashboard, or report-only completion.
- No performance win claim without later same-target `ssh rvv` evidence.

## Technical Notes

- Required specs:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
- Related archived task:
  `.trellis/tasks/archive/2026-06/06-09-rvv-packed-i4-production-kernel-resource-aware-realization-campaign/prd.md`.
- Direction source: Hermes brief for
  `Stage2 RVV low-precision contraction primitive-surface campaign Gate 1`.
- Memory-derived steering: q8/q4/llama.cpp examples are pressure tests for
  broader Stage 2 low-precision contraction maturity, not narrow route
  authority.

## Gate 1 Slice Result

Completed in this round:

- Added explicit `low_precision_primitive.source_signedness` as a
  provider-owned primitive fact for signed i8 and unsigned u8 widening-product
  paths.
- Propagated the signedness fact through RVV route planning descriptions,
  widening-product validation contracts, widening-reduction primitive facts,
  emission-plan metadata, target support-bundle metadata, and target artifact
  mirror validation.
- Added stale source-signedness rejection for provider/target validation and
  focused positive checks for signed widening product, unsigned u8 widening
  product, and signed product-reduction primitive paths.
- Kept Common EmitC neutral: it receives provider-built payloads and does not
  infer dtype/sign, intrinsic, or target support.
- Synchronized `.trellis/spec/extension-plugins/rvv-plugin.md` so future Gate 2
  Gearbox work consumes the same explicit primitive fact instead of re-deriving
  signedness from q8/q4 names, artifact metadata, helper names, or Common EmitC.

## Continuation Point

Gate 1 is complete for the bounded production compiler-surface milestone. The
macro task remains active. Continue with Gate 2: make the resource-aware
Gearbox selected-body realization consume the accepted low-precision primitive
surface, including `source_signedness`, for packed/low-precision bodies without
changing semantics or relying on q8/q4 names as authority.
