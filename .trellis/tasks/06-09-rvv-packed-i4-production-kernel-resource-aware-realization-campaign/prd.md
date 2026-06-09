# Stage2 RVV packed-i4 production-kernel resource-aware realization campaign

## Goal

Advance the RVV Stage 2 packed-i4 production-kernel path beyond the accepted
same-target no-win/regression evidence. This macro campaign must turn that
diagnosis into RVV plugin-owned selected-body/resource remediation, provider
and target validation, same-target measurement, and dispatch/performance policy
consumption. The current round owns Gate 1 only: change production selected-body
or primitive/resource planning for one packed-i4 contraction/dequant path, with
fail-closed validation of stale remediation/resource facts.

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
- Implement the current round as a coherent Gate 1 production slice in RVV
  plugin selected-body/resource planning, primitive planning, provider
  validation, or target validation.
- Consume the existing packed-i4 no-win/fallback diagnosis at the
  selected-body/resource or primitive-planning boundary.
- Make the packed-i4 contraction/dequant path more realistic or explicit by
  carrying provider-owned remediation/resource facts that distinguish packed
  i4 unpack/resource cost from sibling unpacked-byte paths.
- Fail closed when stale remediation/resource facts claim a packed-i4 plan that
  does not match the provider-owned selected resource plan.
- Do not use q4/q8/llama labels, route ids, artifact names, test names,
  Common EmitC branches, descriptor residue, or source-front-door metadata as
  route authority.
- Do not claim a runtime, correctness, or performance improvement in this
  round unless real `ssh rvv` evidence is collected after the production change.

## Macro Campaign Gates

- [x] Gate 1: production selected-body or primitive-surface remediation for one
  packed-i4 contraction/dequant path, with fail-closed validation of stale
  resource/remediation facts.
- [ ] Gate 2: provider and target validation mirror only provider-owned
  resource/remediation facts and reject metadata-only win claims.
- [ ] Gate 3: same-target measurement compares repaired generated artifacts on
  `ssh rvv` against the prior fallback/no-win path with structured
  accepted/rejected evidence.
- [ ] Gate 4: dispatch/performance policy consumes the new evidence and either
  selects performance-preferred with all required tie-backs or preserves
  correctness fallback with a structured reason.

## Current Round Slice: Gate 1

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

- [x] Production source diff changes the RVV selected-body/resource,
  primitive-planning, provider-validation, target-validation, or policy-consumer
  path for the packed-i4 production kernel representative.
- [x] Tests prove the accepted packed-i4 path carries the new provider-owned
  resource/remediation facts while retaining correctness fallback status.
- [x] Tests prove stale resource/remediation facts fail closed before they can
  authorize route/provider, target artifact, performance-preferred dispatch, or
  win claims.
- [x] Bounded old-authority scan over touched files and added diff lines finds
  no new legacy `RVVI32M1`, `rvv-i32m1`, dtype-prefixed helper route authority,
  source-front-door authority, descriptor-driven computation, q4/q8/llama
  route authority, Common EmitC semantic branching, or exact i32m1 intrinsic
  authority.
- [x] `git diff --check` and `git diff --cached --check` pass.
- [x] One coherent commit records the Gate 1 slice.
- [x] `.trellis/.current-task` remains active and the PRD/journal name Gate 2
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

Gate 2 remains the next owner: provider and target validation should consume
the Gate 1 remediation plan facts as provider-owned mirrors and reject any
metadata-only performance/win claim. The macro task must remain active until
Gate 2, Gate 3 same-target `ssh rvv` measurement, and Gate 4
dispatch/performance policy consumption are complete.
