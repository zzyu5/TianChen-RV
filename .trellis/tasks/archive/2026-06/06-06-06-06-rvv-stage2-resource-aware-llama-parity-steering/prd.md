# Redirect RVV Stage 2 Toward Resource-Aware Llama Parity

## Goal

Record and apply the human steering that the current RVV Stage 2 loop has drifted
toward repeated artifact ABI/evidence closeouts. The next Hermes-owned direction
should preserve the direct-contraction contract work already completed, but then
move toward resource-aware low-precision direct-contraction realization and a
real comparable benchmark path against llama.cpp RVV kernels.

## What I Already Know

- The user confirmed that q8/q4/llama.cpp is an example of a broader weak point,
  not a request to overfit the mainline architecture to a q8-named feature.
- The existing Gearbox pass is a bounded/static schedule materialization MVP for
  selected RVV bodies; it is not yet a full resource-aware autotuning pass.
- The active supervisor loop completed round 149 with
  `rvv: consolidate contraction artifact contract core`, consolidating shared
  MAcc and widening dot-reduce/direct-contraction artifact contract facts.
- The current route has repeatedly closed executable artifact ABI evidence
  seams. That was useful for route support, but it is not enough to claim
  llama.cpp-like performance maturity.

## Assumptions

- Hermes should consume a one-shot steering note in the next official review
  instead of bypassing the supervisor loop.
- Spec updates should be durable but narrow: they should not make the v3 Gearbox
  artifact authoritative, and they should not require immediate full runtime
  autotuning before a bounded static resource model exists.

## Requirements

- Write `artifacts/tmp/hermes_codex_supervisor/manual_steering_once.md` with a
  precise next-owner correction.
- Update Trellis specs so future Stage 2 work distinguishes route-family
  contract/evidence closure, resource-aware selected-body realization, and
  performance comparison evidence.
- Make clear that q8/q4/llama.cpp examples represent low-precision
  direct-contraction maturity gaps, not route-name or benchmark-name authority.
- Make clear that a static one-candidate Gearbox schedule is not a completed
  resource-aware autotuning pass.

## Acceptance Criteria

- [x] Hermes has a one-shot steering file that redirects the next owner after the
  direct-contraction contract task.
- [x] Variant pipeline spec says repeated artifact ABI proof is not the Stage 2
  endpoint after a route family is production-validated.
- [x] RVV plugin spec defines a concrete resource-aware low-precision
  direct-contraction/performance closure contract.
- [x] Testing spec defines what an apples-to-apples llama.cpp/RVV performance
  comparison must prove.
- [x] Spec references keep authority in typed `tcrv_rvv` body, RVV plugin
  realization/provider facts, and real `ssh rvv` evidence.

## Out of Scope

- Do not implement the full Gearbox autotuning pass in this task.
- Do not stop or restart the active supervisor loop unless the status check shows
  it is actually stopped or unable to consume steering.
- Do not claim performance parity with llama.cpp without runtime timing evidence.
- Do not add q8/q4 route-name authority.

## Technical Notes

- Relevant reference artifact:
  `artifacts/tianchenrv_rvv_gearbox_autotuning_pass_v3.md`.
- Relevant specs:
  `.trellis/spec/variant-pipeline/generation-selection-tuning.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`.
- Supervisor one-shot steering file:
  `artifacts/tmp/hermes_codex_supervisor/manual_steering_once.md`.
