# Variant Pipeline Specs

This layer defines how high-level MLIR enters TianChen-RV and becomes execution variants.

## Pre-Development Checklist

- [ ] Does the pipeline call plugins directly instead of building a generic `tcrv` compute op?
- [ ] Does each proposed variant include origin, requires, preconditions, extension ops, cost/tuning, and emission metadata?
- [ ] Are core legality and plugin legality both applied?
- [ ] Does selection explain why it chose static variant, dispatch, or fallback?
- [ ] Is tuning represented as capability-aware selection and variant-local metadata?

## Guidelines Index

| Spec | Description |
|---|---|
| [Generation Selection Tuning](./generation-selection-tuning.md) | End-to-end variant pipeline contract |

## Quality Check

- Same high-level op should produce different variants under different capability profiles.
- Illegal variants should fail before emission.
- Diagnostics must explain unsupported plugin, unsatisfied capability, legality failure, missing emission path, dispatch retention, and fallback choice.
