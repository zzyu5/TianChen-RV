# Stage2 RVV resource-aware composite fallback dispatch ABI boundary

## Goal

Make the fallback side of the resource-aware
`runtime_scalar_cmp_masked_indexed_gather_macc_scatter` dispatch envelope
truthful at the executable ABI boundary. The selected RVV case must keep its
provider-owned route/resource/ABI facts and generated artifact support. When
the selected RVV candidate is unavailable, rejected, over budget, or otherwise
not exported as a supported executable candidate, the current scalar fallback
must either have a real plugin-owned executable route or fail closed with a
targeted diagnostic. This round must not invent scalar or RVV compute in common
EmitC/export code.

## What I Already Know

* The predecessor selected-envelope task proved that the explicit and
  pre-realized composite fixtures contain actual `tcrv.exec.dispatch`,
  selected `tcrv.exec.case`, conservative `tcrv.exec.fallback`, provider
  mirrors, resource candidate mirrors, and fresh selected-case `ssh rvv`
  evidence.
* The RVV plugin spec requires authority to flow from selected
  `tcrv.exec` envelope to typed/realized `tcrv_rvv` body, RVV provider route,
  common `TCRVEmitCLowerableRoute` materialization, target artifact export, and
  hardware evidence when runtime correctness is claimed.
* The scalar fallback plugin spec says the current scalar fallback surface is
  intentionally unsupported for emission: it has no active scalar body, no
  selected lowering boundary, no EmitC route, no runtime ABI, and no executable
  artifact support.
* `EmitCLowerableMaterialization.cpp` ignores unsupported fallback plans when a
  supported non-fallback selected dispatch case exists, but can otherwise be
  asked to materialize a fallback-only selected plan.
* `TargetArtifactExport.cpp` currently builds target artifact candidates only
  from `status = "supported"` emission-plan diagnostics. A selected fallback
  with `status = "unsupported"` is correctly not a candidate, but a kernel with
  no supported candidates needs an explicit fail-closed export diagnostic so an
  executable fallback claim cannot silently degrade into an empty or generic
  artifact result.

## Requirements

* Preserve the selected RVV composite dispatch case behavior, route/provider
  mirrors, resource candidate facts, ABI/header order, common materialization,
  and positive generated artifact evidence.
* Do not make the scalar fallback executable in this task. Current scalar
  fallback is a selection/diagnostic surface only until a later plugin-local
  scalar rebuild adds typed scalar bodies and a real EmitC route.
* Harden target artifact candidate collection so every selected kernel that
  reaches export has at least one supported executable artifact candidate. If
  all selected paths are unsupported, fail closed and list the unsupported
  selected path roles, targets, statuses, origins, emission kinds, and artifact
  kinds.
* Add focused evidence for the composite fallback ABI boundary: if the RVV
  selected case is made unsupported while the scalar fallback remains the
  conservative fallback, target artifact export must reject the executable
  claim with a diagnostic that names both the dispatch case and the dispatch
  fallback as unsupported selected paths.
* Keep common EmitC/export neutral. The new check may reason about selected
  path support status and artifact-candidate presence, but it must not infer
  RVV semantics, scalar semantics, dtype, schedule, intrinsic names, fallback
  computation, or runtime behavior.

## Acceptance Criteria

* [x] `TargetArtifactExport.cpp` rejects a selected kernel whose selected
      dispatch/fallback surface produces no supported executable artifact
      candidate.
* [x] The rejection diagnostic identifies unsupported selected paths, including
      the composite RVV dispatch case and scalar dispatch fallback, instead of
      relying only on a generic "found none" later in export.
* [x] The explicit composite fixture keeps the existing selected-case positive
      header/export checks and gains a fallback-boundary negative check for the
      "RVV case unsupported + scalar fallback unsupported" condition.
* [x] The pre-realized composite selected-case regression remains green.
* [x] No scalar fallback compute route, descriptor-driven fallback route,
      source-front-door route, or common EmitC semantic branch is added.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
* [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
* [x] Focused lit for the touched composite/export behavior passes.
* [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive `i32m1`, descriptor-driven compute, source-front-door, or
      common EmitC RVV/scalar semantic authority.
* [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are clean or explained.

## Definition of Done

* The fallback ABI boundary is truthful: selected RVV artifact export still
  works, and unsupported fallback executable claims fail closed before artifact
  support can be reported.
* Evidence is focused on this seam, not a broad composite matrix.
* Trellis task status/context and workspace journal are updated truthfully.
* Specs are reviewed for durable knowledge; update specs only if this round
  discovers a reusable contract not already captured.
* One coherent commit is created if the task completes.

## Out of Scope

* Implementing an executable scalar fallback body, scalar EmitC route, scalar
  runtime ABI, or scalar generated artifact route.
* New RVV route-family coverage, dtype/LMUL clone batches, high-level frontend
  work, source-front-door positive routes, dashboards, tuning databases, or
  broad smoke matrices.
* Reworking gather, scatter, MAcc, compare/select, memory, segment, reduction,
  or low-precision routes outside this fallback executable ABI seam.

## Technical Notes

* Direction source: Hermes/Codex worker direction brief supplied on
  2026-06-07.
* Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/extension-plugins/scalar-fallback-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/core-dialect/tcrv-exec-contract.md`,
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`, and the shared guides.
* Pre-implementation probe:
  changing the explicit composite selected RVV emission plan to
  `status = "unsupported"` while preserving the scalar dispatch fallback causes
  header export to fail with `requires exactly one supported header artifact
  emission-plan route; found none`. The repair should make this failure
  targeted at the selected dispatch/fallback executable boundary.

## Completed Evidence

* Production repair:
  `lib/Target/TargetArtifactExport.cpp` now fails closed for a selected kernel
  whose selected paths produce no supported executable artifact candidate. The
  diagnostic lists each unsupported selected path with variant symbol, selected
  path role, status, origin, emission kind, and artifact kind.
* Focused fallback-boundary regression:
  `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-indexed-gather-macc-scatter.mlir`
  now mutates the selected RVV dispatch-case emission plan to
  `status = "unsupported"` and verifies target header export rejects the claim
  while also naming the scalar dispatch fallback as unsupported
  `scalar-fallback-unsupported-emission`.
* Durable spec update:
  `.trellis/spec/lowering-runtime/emission-runtime-contract.md` now records
  the selected-path artifact candidate gate, including signatures, contracts,
  validation matrix, good/base/bad cases, required tests, and wrong/correct
  behavior.
* Probe after repair:
  the fallback-boundary command rejects with
  `selected target artifact export requires at least one supported executable
  artifact candidate; selected paths are unsupported: @rvv_explicit_composite
  as dispatch case status 'unsupported' ... @explicit_composite_scalar_fallback
  as dispatch fallback status 'unsupported' ... scalar-fallback-unsupported-emission`.
* Checks run:
  `ninja -C build tcrv-translate tcrv-opt tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test`;
  focused lit filters for explicit composite, pre-realized composite, bounded
  scalar-broadcast fallback negative reference, and generated-bundle dry-run;
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test --op-kind runtime_scalar_cmp_masked_indexed_gather_macc_scatter`;
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  `build/bin/tianchenrv-target-artifact-export-test`;
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-07-stage2-rvv-resource-aware-composite-fallback-dispatch-abi`;
  `git diff --check`.
* Bounded old-authority scan:
  added diff lines had no positive `RVVI32M1`, `rvv-i32m1`,
  `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, source-front-door,
  descriptor-driven, or direct-C legacy authority matches. The only added-diff
  matches were negative journal wording that explicitly says no
  descriptor-driven fallback or source-front-door path was added. Whole
  touched-file scan found only negative PRD wording and existing target-export
  direct-C/descriptor rejection code.
* No `ssh rvv` run was needed in this round because no new runtime
  correctness claim was made; the selected-case generated-bundle dry-run and
  existing selected-envelope evidence remain the selected RVV runtime evidence
  source.
