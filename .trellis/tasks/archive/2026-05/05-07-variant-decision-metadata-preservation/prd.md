# Preserve Generic Variant Decision Metadata

## Goal

Preserve plugin-proposed generic decision metadata from proposal generation through materialized `tcrv.exec.variant` operations and synthesized dispatch cases.

## Requirements

* Extend `tcrv.exec.variant` with optional generic string attributes:
  * `condition`
  * `guard`
  * `policy`
* Reject present-but-empty metadata strings after trimming whitespace.
* Copy non-empty `VariantProposal` `condition`, `guard`, and `policy` fields onto materialized variants.
* Synthesize dispatch cases that inherit present metadata from the target variant.
* Preserve existing deterministic fallback behavior: fallback remains the first direct variant whose `requires` are generically available.
* Keep existing generic guard synthesis when an unavailable required capability has no inherited variant metadata.
* Do not attach metadata to `tcrv.exec.fallback` unless the existing ODS contract supports it.

## Non-Goals

* No cost model, tuning engine, full selection pass, lowering, emission, runtime ABI, backend work, concrete RVV/IME/offload legality, or hardware performance work.
* No target-, vendor-, dtype-, shape-, layout-, runtime-, RVV-, IME-, Sophgo-, offload-, scalar-, or family-specific branches.
* No Python implementation of IR, dialects, passes, plugin interfaces, capability decisions, legality verification, selection, lowering, or emission.

## Tests

* lit/FileCheck coverage for `tcrv.exec.variant` parsing, printing, and verification of `condition`, `guard`, and `policy`, including invalid empty strings.
* C++ materialization coverage proving proposal metadata survives into typed `tcrv.exec.variant` ops.
* Dispatch synthesis coverage proving generated cases inherit metadata in deterministic variant order.
* Capability-requires coverage proving inherited metadata guards unavailable requirements, while unguarded paths remain rejected.

## Context

The previous committed HEAD is expected to be `b54605c feat: route plugin variant legality`. That round added plugin-routed legality for materialized variants. This round closes the live-code gap where proposal-level generic decision metadata existed but was lost before dispatch synthesis.
