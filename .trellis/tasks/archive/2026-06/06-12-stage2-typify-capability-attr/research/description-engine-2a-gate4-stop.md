# MOVE 2A harness — gate4 STOP (parent-owned tune-measurement decision) — 2026-06-14

HEAD fc9aa69f (uncommitted MOVE 2A in tree). The packed-i4 NUMERICAL ORACLE re-sourcing is
DONE + ssh-rvv re-verified (see `ssh-rvv-packed-i4-reverify/`). The gate4 dry-run red is a
SEPARATE, independent MOVE-2A breakage that is parent-owned. This file is the closed-graph.

## gate4 red is NOT the oracle and NOT packed-vs-generic

Proven empirically: gate4's FIRST RUN (non-packed grouped, `--op-kind ...dequantize_f32
--op-kind ...dequant_clamp_f32`, no packed input) crashes:

    widening_product_reduce_dequantize_f32 same-target measurement failed:
    low-precision candidate feedback missing tcrv_rvv.low_precision_resource.candidate_set

i.e. the break fires on the GROUPED path, before any packed detection. It is independent of
the oracle detector (which is now typed-sourced + verified) and independent of packed/generic.

## Root cause: low_precision_candidate_feedback_record reads 27 DELETED channel-2 keys

`scripts/rvv_generated_bundle_same_target_measure.py:2362 low_precision_candidate_feedback_record`
unconditionally builds `fields = {name: resource_field(name) for name in stable_field_names}`
(line 2433) for all 27 `stable_field_names` (2398-2426): candidate_set, selected_candidate,
candidate_count, legal_candidate_count, selected_candidate_index, selection_reason,
planning_contract, route_family_plan, provider_supported_mirror, operand_form, source_signedness,
storage_element_width, effective_element_width, packing_layout, unpack_intent, vsetvl_region_count,
runtime_avl_source, runtime_abi_order, primitive_chain_contract, primitive_chain_kind,
primitive_widening_product_intrinsic, primitive_reduction_intrinsic, realization_decision,
realized_vsetvl_region_count, realized_peak_live_vector_groups, target_capability_provider_mirror,
target_capability_legality_mirror.

`resource_field` reads from target_artifact_metadata / route_metadata / provider_low_precision.
After MOVE 2A, the regenerated bundle carries ONLY the 11 surviving
`selected_dispatch_policy_output_mirror.*` keys (correctness/performance dispatch policy).
None of the 27 candidate-enumeration / primitive / realization / capability-mirror keys survive
in bundle metadata, route_metadata, or provider_low_precision. → first `resource_field` call
(candidate_set/candidate_count) raises EvidenceError.

These 27 keys ARE the description-engine channel-2 I4-mirror keys MOVE 2A deleted (see
`description-engine-2a-boundary.md` §2: appendRVVLowPrecisionStableResourceCompilerFactMetadata
etc.). The Target/RVV HEADER/PLAN fixtures were already re-baselined to drop them. So the
candidate-feedback record is now a dead-mirror reader of deleted keys.

## Why this is STOP, not a harness retirement I should do unilaterally

- **No live channel to re-source from** (unlike the oracle, where the typed nibble-unpack op
  survived). The candidate enumeration / selection-reason / primitive / realization / capability
  mirrors have no genuine survivor in the bundle.
- **`expected` source = tautology** (assert expected==expected); skip-when-absent = vacuity. Both
  forbidden by the task.
- **Retiring the subsystem removes the tune-measurement evidence** (candidate enumeration,
  schedule-decision, performance-admission, remediation) that feeds ~100 derived gate4 `.test`
  assertions via `provider_feedback_tie_back["fields"]` (same_target_measure.py:3327+, 3592+) and
  the candidate-feedback-loop evidence + the negative self-tests (4782-4960). That is the
  N3/tune story — project memory: "tune 是用户拥有的 novelty（别再提砍）". Whether retiring a
  deleted-mirror SNAPSHOT of tune counts as cutting tune is genuinely arguable → parent decision.

## Closed-graph for the gate4 retire-or-restore decision (when authorized)

If RETIRE (the candidate-feedback validates a now-deleted mirror; keep only genuine measurement):
- `low_precision_candidate_feedback_record` (2362) `stable_field_names` reads (2398-2433) — the
  27 deleted-key `resource_field` calls.
- Its derived consumers: `provider_feedback_tie_back["fields"]` reads at 3327-3500 (schedule_decision*,
  resource_cost*, performance_admission*), 3592+; candidate-feedback-loop evidence at 3799-3861.
- The negative self-tests `expect_candidate_feedback_metadata_failure` (4839-4960) that assert the
  record REJECTS stale/missing candidate keys (now-deleted keys → tests no longer meaningful).
- gate4 `.test` CHECK blocks for the packed sub-runs (`%t.packed.*`, `%t.candidate.*`):
  `packed_i4_resource_metadata_selected` (KEEP — now typed-sourced), `packed_i4_reference_oracle` +
  `runtime_n_unit` (KEEP — genuine oracle), tolerance/ABI/sig (KEEP); plus selected_candidate_index /
  candidate_count / legal_candidate_count / ~100 remediation/schedule/admission mirror facts (the
  deleted-mirror surface to retire).
MUST preserve: packed_i4_resource_metadata_selected (typed-sourced), the packed-i4 reference oracle
result, numeric tolerance=1e-05, ABI, function sig. MUST NOT fabricate/re-inject the deleted keys.

If RESTORE: MOVE 2A over-deleted the candidate channel in C++ → fix belongs in the C++ synthesis
(boundary doc already STOPed the C++ deletion for re-authorization on blast-radius grounds), not
in Python papering.

## Hand-off note: gate4's packed-i4 oracle DETECTOR is already typed-sourced (but unverified e2e)

gate4's `uses_packed_i4_resource_from_bundle` (same_target_measure.py:3083) delegates to
`abi.product_dequant_uses_packed_i4_resource_metadata` over the dict from
`abi.widening_product_reduction_metadata_from_bundle`. That dict now carries the genuine
`rvv_selected_body_typed_compute_op` (plumbed in abi_e2e.py this task), so once the
candidate-feedback blocker is resolved, gate4's packed-i4 oracle gating will be typed-sourced too
— NO further detector change needed in same_target_measure.py. But it is UNVERIFIED end-to-end
because gate4 crashes earlier on candidate_count: whoever resolves the candidate-feedback decision
must then confirm the gate4 packed sub-runs emit `packed_i4_resource_metadata_selected: true` +
`packed_i4_reference_oracle` (detector returns True on the packed sub-runs).
