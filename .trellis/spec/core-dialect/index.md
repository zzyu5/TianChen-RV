# Core Dialect Specs

This layer defines `tcrv.exec`, the stable core dialect for execution organization.

## Pre-Development Checklist

- [ ] Does every new core op organize execution rather than express high-level computation?
- [ ] Are extension-specific types and ops kept out of `tcrv.exec`?
- [ ] Does each variant declare `requires` and `origin`?
- [ ] Is fallback or an explicit external fallback declaration present?
- [ ] Is runtime dispatch represented structurally when multiple conditions are valid?
- [ ] Is the core dialect implemented in C++/MLIR/TableGen rather than Python?

## Guidelines Index

| Spec | Description |
|---|---|
| [tcrv.exec Contract](./tcrv-exec-contract.md) | Core ops/types, verifier rules, relation to high-level MLIR |

## Quality Check

- `tcrv.exec` must be readable as a container for execution variants.
- If a proposed core op includes algorithm-specific semantics, move it to an extension dialect or reject it.
- Verifier behavior must catch missing capabilities, missing fallback, illegal extension ops, and incomplete offload ABI.
- Dialect syntax, parsing, verification, and pass-facing behavior require lit/FileCheck coverage.
