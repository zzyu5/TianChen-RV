# Variant Pipeline Specs

This layer defines how high-level MLIR enters TianChen-RV and becomes execution variants.

## Pre-Development Checklist

- [ ] Does the pipeline call plugins directly instead of building a generic `tcrv` compute op?
- [ ] Does each proposed variant include origin, requires, preconditions, extension ops, cost/tuning, and emission metadata?
- [ ] Are core legality and plugin legality both applied?
- [ ] Does selection explain why it chose static variant, dispatch, or fallback?
- [ ] Is tuning represented as capability-aware selection and variant-local metadata?
- [ ] For RVV performance-sensitive work, does tuning include resource-aware
  selected-body realization or a measured runtime comparison path rather than
  another artifact-evidence-only closeout?
- [ ] Does RVV work use typed `tcrv_rvv` body authority rather than legacy i32 route tables or source-front-door metadata? (见 core-invariants I5, I7)
- [ ] Are emission plans, selected-path metadata, route ids, and artifact names mirrors only?

## Guidelines Index

| Spec | Description |
|---|---|
| [Generation Selection Tuning](./generation-selection-tuning.md) | End-to-end variant pipeline contract |

## Quality Check

- Same high-level op should produce different variants under different capability profiles.
- Illegal variants should fail before emission.
- Diagnostics must explain unsupported plugin, unsatisfied capability, legality failure, missing emission path, dispatch retention, and fallback choice.
- RVV route authority is the corrected typed `tcrv_rvv` body; legacy i32m1 and
  source-front-door routes fail closed (见 core-invariants I7). Current RVV work
  expands typed-body coverage, selected-body realization, resource-aware tuning,
  and measured RVV evidence where runtime/performance is claimed. Second-family
  and source-front-door positives are future work, not yet built.
