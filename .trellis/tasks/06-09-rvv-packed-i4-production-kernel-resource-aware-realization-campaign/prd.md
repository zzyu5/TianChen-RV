# Stage2 RVV packed-i4 production-kernel resource-aware realization campaign

## Goal

Advance the RVV Stage 2 packed-i4 production-kernel path beyond the accepted
same-target no-win/regression evidence. This macro campaign must turn that
diagnosis into RVV plugin-owned selected-body/resource remediation, provider
and target validation, same-target measurement, and dispatch/performance policy
consumption. Gates 1 and 2 are complete. The current round owns Gate 3 only:
run the repaired generated packed-i4 artifact through the same-target
measurement path on `ssh rvv`, record raw and structured evidence, and leave
dispatch/performance policy consumption for Gate 4.

## What I already know

- The previous macro campaign
  `.trellis/tasks/archive/2026-06/06-09-rvv-low-precision-performance-remediation-campaign/`
  is complete and archived at commit `e66c933c`.
- That campaign made `RVVLowPrecisionPerformancePolicy` consume the accepted
  packed-i4 same-target no-win/regression measurement. Correctness route
  support is preserved, but performance-preferred dispatch and win claims are
  denied until provider/resource remediation and new evidence agree.
- The next bottleneck is not another evidence closeout. The production compiler
  path must make the packed-i4 selected body or primitive/resource plan more
  explicit, cheaper, or fail-closed when remediation facts are stale.
- The relevant architecture chain remains:
  selected typed `tcrv_rvv` body -> RVV plugin selected-body/resource
  realization -> provider-built `TCRVEmitCLowerableRoute` -> common EmitC
  materialization -> target artifact mirrors -> `ssh rvv` evidence for
  runtime/correctness/performance claims.

## Requirements

- Keep this as one macro Trellis task until all campaign gates are complete or
  human steering redirects it.
- Implement the current round as a coherent Gate 3 same-target measurement
  slice for the accepted packed-i4 widening product-reduce-dequantize f32
  representative.
- Reuse the production generated-artifact path: selected/pre-realized typed
  `tcrv_rvv` body, RVV plugin selected-body/resource realization,
  provider-built route, common EmitC materialization, target artifact export,
  generated object/header, external measurement harness, and `ssh rvv`.
- Select the packed scalar C baseline only after generated object/header
  metadata validates the provider-owned `packed-i4-nibbles` resource facts.
- Record evidence that distinguishes correctness execution, real runtime
  measurement, no-win/regression classification, performance-win eligibility,
  and Gate 4 policy handoff input.
- Treat measurement output as evidence input only. It must not rewrite
  provider-owned resource facts, target artifact mirrors, dispatch preference,
  or performance win permission.
- Do not use q4/q8/llama labels, route ids, artifact names, test names,
  Common EmitC branches, descriptor residue, or source-front-door metadata as
  route authority.
- Do not claim a performance improvement unless real `ssh rvv` measurement
  classifies as a win and provider-owned facts make such a claim eligible.

## Macro Campaign Gates

- [x] Gate 1: production selected-body or primitive-surface remediation for one
  packed-i4 contraction/dequant path, with fail-closed validation of stale
  resource/remediation facts.
- [x] Gate 2: provider and target validation mirror only provider-owned
  resource/remediation facts and reject metadata-only win claims.
- [x] Gate 3: same-target measurement compares repaired generated artifacts on
  `ssh rvv` against the prior fallback/no-win path with structured
  accepted/rejected evidence.
- [ ] Gate 4: dispatch/performance policy consumes the new evidence and either
  selects performance-preferred with all required tie-backs or preserves
  correctness fallback with a structured reason.

## Current Round Slice: Gate 3

- [x] Confirm the existing measurement workflow still generates and validates
  the repaired packed-i4 target artifact from the selected typed body without
  new source-front-door, descriptor, q4/q8, or metadata authority.
- [x] Run a focused dry-run/self-test proving the same-target harness structure,
  packed-i4 baseline selection from validated metadata, and structured
  not-measured handoff remain intact.
- [x] Run a real non-dry-run `ssh rvv` same-target measurement for
  `widening_product_reduce_dequantize_f32` using the packed-i4 pre-realized
  selected-body fixture and the campaign input sizes.
- [x] Record the artifact root, raw remote target profile, compile stdout, run
  stdout, parsed measurement records, parsed summary records, classification,
  and performance-win eligibility.
- [x] If the result is no-win or regression, preserve correctness execution but
  keep Gate 4 conservative; do not invent a performance win claim.
- [x] Keep the macro task active after the Gate 3 slice because Gate 4 remains.

Gate 3 slice completed in this round: no production source change was required.
The existing `rvv_generated_bundle_same_target_measure.py` workflow already
generates and validates the repaired packed-i4 artifact, selects the packed
scalar baseline only from validated provider-owned `packed-i4-nibbles` metadata,
runs correctness guards before timing, parses same-target timing, and emits a
policy-consumable packed-i4 maturity evidence input without changing provider
facts or dispatch policy. The real run at
`artifacts/tmp/codex-gate3-packed-i4-real/gate3-packed-i4-same-target-measure-ssh`
completed on `ssh rvv` with 12 correctness records, 60 raw `MEASURE` records,
12 parsed `SUMMARY` records, classification `regression`, outcome family
`no-win`, best speedup range `0.683805..0.705257`,
`provider_performance_selection_eligible = false`,
`provider_dispatch_preference = not-performance-preferred`,
`performance_win_claim_allowed = false`, and
`correctness_execution_allowed = true`.

## Completed Slice: Gate 2

Gate 2 slice completed in this round: the RVV target artifact route-family
validator now rejects packed-i4 metadata-only performance/win/dispatch claim
keys that are not provider-owned selection facts. Existing exact mirror checks
continue to accept valid provider-owned remediation/resource/performance mirrors
and reject stale or missing mirrors. Focused C++ and lit coverage prove that a
packed-i4 artifact cannot add `performance_win_claim_allowed`,
`dispatch_policy_path`, or `win_claim` metadata to override the provider-owned
no-win / not-performance-preferred contract.

## Completed Slice: Gate 1

- [x] Add or repair production RVV plugin code so the selected packed-i4
  low-precision resource plan explicitly carries the remediation/resource
  planning facts needed by the packed-i4 contraction/dequant route.
- [x] Tie those facts to the existing no-win remediation handoff, not to
  artifact names or measurement scripts.
- [x] Add focused positive coverage for the accepted packed-i4 resource plan.
- [x] Add focused negative coverage for stale or missing remediation/resource
  planning facts before route/provider or target acceptance.
- [x] Run focused C++ tests and any directly relevant lit/script dry-runs for
  the changed packed-i4 path.
- [x] Keep the macro task active after the committed Gate 1 slice if any later
  gates remain.

Gate 1 slice completed in this round: the selected packed-i4 resource candidate
now carries provider-owned remediation plan facts
(`remediation_plan_contract`, `remediation_plan`,
`remediation_statement_strategy`, and `remediation_vector_budget`). Gearbox
materializes them, selected-body realization copies them, provider and statement
planning parse/compare/verify them, route metadata mirrors them, and target
artifact validation/support-bundle export fail closed on stale or missing
mirrors. No runtime, correctness, or performance improvement beyond existing
evidence is claimed.

## Acceptance Criteria

- [x] The current repaired packed-i4 generated artifact is measured on real
  `ssh rvv`, or the blocking hardware/plumbing failure is recorded with the
  exact continuation command and no runtime/performance claim is made.
- [x] Per-op evidence contains generated object/header identity and hashes,
  selected input/variant, packed scalar baseline identity, correctness guards
  before timing, raw `MEASURE` records, parsed `SUMMARY` records, target
  profile, compile flags, timing method, and result classification.
- [x] Structured evidence contains provider-owned packed-i4 tie-back fields and
  a policy-consumable handoff distinguishing correctness execution allowed,
  performance preference denied/allowed, performance win claim eligibility, and
  provider contract update requirement.
- [x] Measurement evidence is treated as input only; no dispatch/performance
  policy is changed in Gate 3, and Gate 4 remains responsible for consuming the
  evidence.
- [x] Focused script self-test/dry-run or lit coverage passes for the measured
  workflow. If no source files changed, record the no-source-change justification
  and still run the relevant script checks.
- [x] Bounded old-authority scan over touched files and added diff lines finds
  no new legacy `RVVI32M1`, `rvv-i32m1`, dtype-prefixed helper route authority,
  source-front-door authority, descriptor-driven computation, q4/q8/llama
  route authority, Common EmitC semantic branching, or exact i32m1 intrinsic
  authority.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] One coherent commit records the Gate 3 slice.
- [x] `.trellis/.current-task` remains active and the PRD/journal name Gate 4
  as the next continuation point unless all campaign gates are genuinely
  complete.

## Out of Scope

- No high-level Linalg/Vector/StableHLO frontend work.
- No source-front-door positive route.
- No one-off q4/q8/llama wrapper or artifact-name semantics.
- No broad dtype/LMUL clone batch.
- No measurement-only promotion or generated-bundle-only closeout.
- No Common EmitC invention of RVV semantics.
- No dispatch/performance win claim without new same-target `ssh rvv` evidence.

## Technical Notes

- Required specs: `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`, and focused testing/guides
  for selected-body realization, mirror-only metadata, and fail-closed checks.
- Direction source: Hermes brief for
  `Stage2 RVV packed-i4 production-kernel resource-aware realization campaign`.
- Previous retry failed before implementation with
  `codex_transient:selected model is at capacity`; the only inherited dirty
  state was this Trellis task directory.

## Continuation Point

Gate 4 is the next owner: dispatch/performance policy must consume the Gate 3
same-target regression/no-win evidence and either preserve conservative
correctness fallback or explicitly permit a performance-preferred path only if
all provider-owned measured facts support it. The macro task must remain active
until Gate 4 is complete or human steering redirects it.
