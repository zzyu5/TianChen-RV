# MOVE 2A harness retirement — IMPLEMENT findings + STOP-site (2026-06-14)

HEAD fc9aa69f. C++ MOVE 2A deletion (1122 lines) + Target/RVV .mlir re-baseline already
in tree. Task: turn the 5 NEW dry-run reds green by retiring the harness's
`low_precision_resource.*` schedule-evidence mirror, WITHOUT weakening any genuine
numeric/hardware/ABI assertion.

## What is actually exported now (empirical, regenerated dry-run)

The C++ deletion removed the description-engine `low_precision_resource.*` synthesis from
the bundle metadata/.h. SURVIVING `low_precision_resource.*` keys in the exported .h /
index / evidence = ONLY the `selected_dispatch_policy_output_mirror.*` (11 keys, a
different live channel). Everything else (`selected_candidate`, `operand_form`,
`measurement_disposition_evidence_mirror.*`, `realization_*`, `primitive_*`,
`schedule_decision*`, remediation_*) is GONE from the bundle.

## CLEAN subset (strided reds) — straightforward, can proceed

`explicit-/pre-realized-strided-input-widening-dot-reduce-add-dry-run.test`:
- Only ABSENT (removable) lines are the `low_precision_resource.*` key/value SDOT-DAG
  pairs + the nested `low_precision_resource_selection` resource fields (selected_candidate,
  product_emul). NO genuine numeric/ABI/gearbox term degraded.
- Harness: `provider_route_facts["low_precision_resource_selection"]` (~34563) now emits an
  ALL-NULL dict (every value reads `route_metadata.get(low_precision_resource.*)` → None
  after the already-applied `expected_metadata_for` filter). Hollow shell → retire dict +
  its 2 .test CHECK lines. Genuine `runtime_avl_source: runtime_abi:n` matches elsewhere
  (top-level), stays.

## STOP-SITE (packed-i4 reds + gate4) — genuine NUMERICAL oracle silently broke

`product_dequant_uses_packed_i4_resource_metadata(metadata, expectation)`
(abi_e2e.py:9410) decides whether to emit the GENUINE packed-i4 scalar reference oracle by:

    metadata.get("tcrv_rvv.low_precision_resource.selected_candidate") == <PACKED_I4_CONST>

That bundle-metadata key was deleted by the C++ MOVE 2A. So the detector now ALWAYS
returns False for the packed-i4 fixtures. Consequences observed in the regenerated
packed-i4 dry-run vs the .test expectations:
- `packed_i4_reference_oracle` block → ABSENT (silently dropped).
- `runtime_n_unit: "packed input bytes"` → ABSENT.
- `product_distinguishing_contract` → still present but with the GENERIC value
  ("multi-lane ... i8 products"), NOT the packed-i4 value ("multi-byte packed-i4 ...").
- `uses_packed_i4_harness` (36713) and `widening_product_reduction_metadata_from_bundle`
  packed-i4 agreement check (31370-31377) both now False.

This is NOT a mirror line. It is the **packed-i4 NUMERICAL correctness reference-oracle
selection** — exactly the hardware/numeric assertion the task forbids weakening. Merely
deleting the resource term degrades the oracle to the generic path → the packed-i4
correctness comparison would silently stop running its packed-i4 oracle (vacuity), which
violates "the genuine dequant/numeric assertions MUST still execute, not be skipped into
vacuity."

### The genuine surviving signal to re-source from (if re-sourcing is authorized)

The realized typed-compute-op chain survives in genuine metadata:
`rvv_selected_body_typed_compute_op = "tcrv_rvv.packed_i4_nibble_unpack_product+..."`.
`verify_record_metadata` already keys on `"tcrv_rvv.packed_i4_nibble_unpack_product" in
actual_chain` (abi_e2e.py:12847). So the packed-i4 detector CAN be re-pointed at the
surviving realized-chain signal instead of the deleted resource key. But that is a
DETECTION-LOGIC change (re-source, not delete), a design decision beyond "retire the
mirror." gate4 (same_target_measure.py) inherits the same break via
`low_precision_candidate_feedback_record`.

## gate4 verdict: BLOCKED (same root cause), not separable

`rvv_generated_bundle_same_target_measure.py` has its OWN packed-i4 detector
`uses_packed_i4_resource_from_bundle` (3083) → `abi.product_dequant_uses_packed_i4_resource_metadata`
(3094) — same broken read of the deleted `low_precision_resource.selected_candidate`. The
gate4 `.test` asserts, in its packed sub-runs (`%t.packed.artifacts`, `%t.packed.clamp.artifacts`,
`%t.candidate.artifacts`):
- `packed_i4_resource_metadata_selected: true` (lines 143, 260, 304)
- `packed_i4_reference_oracle: true` (144, 266, 305) + `runtime_n_unit: "packed input bytes"`
  (264) + PACKED-HARNESS `packed_i4_reference_oracle=true runtime_n_unit=packed_bytes` (344) —
  the GENUINE numerical harness/oracle.
- `selected_candidate_index`, `candidate_count`, `legal_candidate_count` from
  `low_precision_candidate_feedback_record` (the candidate-feedback / tune-measurement
  authority — N3-adjacent, parent-owned), plus ~100 provider remediation/schedule/admission
  mirror facts.

Because the detector now always returns False, removing ONLY the authorized resource-feedback
validation does NOT make gate4 green — the packed sub-runs would emit
`packed_i4_resource_metadata_selected: false` and DROP the packed-i4 reference oracle, which
the `.test` requires true. Making gate4 green would force re-baselining the packed sub-runs to
the generic oracle = exactly the numerical-assertion vacuity the task forbids. gate4 therefore
joins the STOP set; it is NOT cleanly separable.

## Recommendation

Proceed with the CLEAN strided subset (2 reds). For the packed-i4 + gate4 subset: the
honest fix is to RE-SOURCE the packed-i4 oracle detector from the surviving realized
compute-op chain so the genuine packed-i4 numerical oracle keeps executing — then retire
only the resource-mirror CHECK lines. Confirm this re-sourcing is in scope before touching
the numeric path.
