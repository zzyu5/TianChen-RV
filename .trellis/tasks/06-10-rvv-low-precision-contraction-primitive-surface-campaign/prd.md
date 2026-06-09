# Stage2 RVV low-precision contraction primitive-surface campaign

## Goal

Create a macro Trellis owner for the RVV Stage 2 low-precision contraction
primitive surface. The campaign must make typed low-precision contraction facts
structurally explicit in the RVV plugin/compiler path, fail closed before route
or target acceptance when those facts are missing or stale, and leave later
gates to consume the surface for Gearbox realization, executable artifact
evidence, and measured dispatch/performance policy. Gate 1 is complete. The
current round owns Gate 2: resource-aware Gearbox selected-body realization
consumes the Gate 1 primitive facts before provider/target mirrors are accepted.

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
- Implement Gate 2 as one coherent production compiler-surface milestone.
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
- If the full Gate 2 surface cannot be finished in one round, complete a
  coherent production submodule and keep the macro task active with the exact
  remaining Gate 2 continuation point.

## Macro Campaign Gates

- [x] Gate 1: explicit typed low-precision contraction primitive surface and
  fail-closed provider validation for i8/u8 input element roles, signedness,
  widened product type, accumulation/result type, SEW/LMUL/policy, memory form,
  and runtime AVL/VL facts.
- [ ] Gate 2: resource-aware Gearbox selected-body realization consumes that
  surface for packed/low-precision bodies without changing semantics or using
  q8/q4 names as authority. The signedness handoff sub-slice is complete;
  broader resource/remediation consumption remains.
- [ ] Gate 3: generated artifact and `ssh rvv` evidence for at least one
  production primitive path changed by Gates 1-2.
- [ ] Gate 4: same-target measurement and dispatch/performance policy
  consumption using provider-owned evidence, including conservative fallback
  when no win is measured.

## Completed Slice: Gate 1

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

## Current Slice: Gate 2 Resource/Remediation Handoff

- [x] Inspect Gearbox schedule derivation, selected-body realization, handoff,
  provider route planning, target artifact validation, and focused tests to
  locate the smallest production owner-local consumption seam.
- [x] Make one Gate 1 low-precision primitive fact, signedness, remediation, or
  resource decision drive Gearbox selected-body realization or its immediate
  provider-owned structural handoff.
- [x] Fail closed when the selected body or realized Gearbox handoff carries
  missing or stale primitive/resource facts.
- [x] Keep target validation as a mirror consumer only: it may accept metadata
  only after provider-owned realization/route validation has consumed the facts.
- [x] Add focused positive tests for the realized Gearbox fact consumption.
- [x] Add focused negative tests for stale or missing realized Gearbox facts.
- [x] Run affected RVV plugin, target export, and textual IR checks.
- [x] Keep the macro task active after this Gate 2 slice unless Gates 2-4 are
  all complete.
- [x] Make selected resource candidate, vector-register budget, peak-live
  resource estimate, operand form/unpack policy, packed-i4 remediation plan,
  and realized product/dequant region shape structural on the realized
  Gearbox handoff.
- [x] Fail closed in RVV-owned verifier/schedule/provider validation when the
  handoff carries stale, missing, contradictory, or metadata-only
  resource/remediation facts.
- [x] Keep Common EmitC and target artifact export as mirror/mechanics
  consumers after provider-owned handoff validation succeeds.

## Gate 1 Acceptance Criteria (completed)

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

## Gate 2 Acceptance Criteria (resource/remediation handoff slice)

- [x] Production RVV plugin/compiler code consumes at least one Gate 1
  low-precision primitive/resource fact inside Gearbox selected-body
  realization or an immediate provider-owned realized-structure handoff.
- [x] The current slice covers signed low-precision primitive source
  signedness or a comparably central primitive/resource fact, and proves that
  the fact reaches realized `tcrv_rvv` structure before provider route support.
- [x] Missing or stale Gearbox handoff primitive/resource facts fail closed in
  dialect verification, Gearbox schedule validation, or provider route
  planning before Common EmitC or target artifact export can accept mirrors.
- [x] Common EmitC remains neutral: it carries provider-built payloads and does
  not infer source signedness, resource decision, intrinsic spelling, or
  Gearbox region structure.
- [x] No q8/q4/llama label, route id, artifact name, helper name, ABI string,
  test name, descriptor residue, or metadata-only mirror becomes route,
  dtype/config, resource, or evidence authority.
- [x] Focused C++/lit checks exercise accepted realized Gearbox facts and stale
  fact rejection for the changed production seam.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes if RVV plugin
  code/tests are touched.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes if target
  artifact validation/export code/tests are touched.
- [x] Relevant `tcrv-opt` or `tcrv-translate` checks pass if IR/export paths
  change.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] One coherent commit records the Gate 2 slice.
- [x] The macro task remains active with unfinished Gate 2 follow-up plus Gates
  3-4 and a precise continuation point unless all macro gates are genuinely
  complete.
- [x] `tcrv_rvv.gearbox_cross_region_handoff` carries selected resource
  candidate, vector register budget, peak-live estimate, operand form,
  packing layout, unpack intent, product/dequant region indexes, and
  packed-i4 remediation plan facts when applicable.
- [x] Selected-body realization derives those handoff facts from the same
  provider-owned low-precision candidate selected after Gate 1 primitive
  signedness/type validation.
- [x] Dialect verification, Gearbox schedule validation, route planning, and
  contraction route-family owners reject stale or missing handoff
  resource/remediation facts before Common EmitC or target artifact mirrors can
  accept the route.
- [x] Focused positive tests prove the new handoff facts survive selected-body
  realization into route/export validation for unpacked-byte and packed-i4
  representatives.
- [x] Focused negative tests prove stale or missing selected candidate,
  budget/peak-live facts, operand/unpack policy, packed-i4 remediation plan,
  and realized region indexes fail closed.

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
  `Stage2 RVV low-precision contraction primitive-surface campaign Gate 2`.
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

## Gate 2 Slice Result

Completed in this round:

- Added required `primitive_source_signedness` to
  `tcrv_rvv.gearbox_cross_region_handoff`, making the Gate 1 primitive
  signedness fact part of the realized Gearbox cross-region structure.
- Added required handoff facts for selected resource candidate, candidate set,
  operand form, packing layout, unpack intent, peak-live vector groups, vector
  register budget, product/dequant region indexes, and packed-i4 remediation
  plan facts when applicable.
- Made selected-body realization validate primitive source signedness against
  the selected low-precision resource candidate and materialize the same
  provider-owned resource/remediation decision facts into the handoff.
- Made dialect verification, Gearbox schedule validation, route planning, and
  contraction route-family owners reject missing or stale handoff signedness,
  resource candidate, budget/peak-live, operand policy, region shape, and
  packed-i4 remediation facts before Common EmitC or target artifact mirrors
  can accept the route.
- Added positive coverage for realized handoff signedness/resource facts across
  product/dequant, dequant-clamp, and packed-i4 selected-body paths, plus
  focused negative stale/missing handoff fact checks.
- Kept target validation and Common EmitC as mirror/mechanics consumers only;
  no q8/q4 label, artifact name, route id, ABI string, helper name, descriptor,
  or metadata-only field became authority.

## Continuation Point

Gate 1 is complete and the Gate 2 Gearbox resource/remediation handoff slice is
complete for the product/dequant and packed-i4 representatives. The macro task
remains active. Continue with Gate 3 only after confirming the changed
production path is the one used for generated artifact evidence: produce the
bounded artifact and `ssh rvv` evidence for at least one path changed by this
campaign. Gate 4 remains later: same-target measurement and dispatch/performance
policy consumption.
