# Target profile capability provider

## Goal

Make a direct kernel-local `tcrv.exec.target` profile anchor participate in
structured target capability decisions when it carries capability identity
metadata. Today `tcrv.exec.target` parses and prints, but the active capability
set used by plugin proposal and legality is populated from
`tcrv.exec.capability` only.

After this round, a target profile with `id`, `kind`, status/provenance
metadata, relation attributes such as `provides`, and structured properties can
serve as a provider in `TargetCapabilitySet`. That makes the compiler path:

```text
tcrv.exec.target profile anchor
  -> TargetCapabilitySet
  -> plugin variant proposal / legality / requires mapping
```

real without adding RVV-specific branches in core orchestration code.

## Non-goals

- Do not add a new high-level tensor/tile IR surface.
- Do not add an RVV runtime, correctness, or performance claim.
- Do not replace MLIR/C++ capability objects with Python data structures.
- Do not redo target artifact ABI preflight, bundle export, or ssh evidence
  work completed in prior rounds.
- Do not make module-global target lookup a prerequisite for this slice.

## Requirements

- Keep `tcrv.exec` focused on execution organization and capability/target
  attachment.
- Treat target profile capability metadata through generic capability model
  machinery, not extension-specific core branches.
- Preserve existing parse-only target anchors that do not yet carry capability
  identity.
- Reject partial capability-provider target profiles that specify only one of
  `id` or `kind`.
- Validate target profile relation attributes consistently with capability
  relation attributes.
- Add focused tests proving that target profiles affect capability lookup and
  plugin proposal.

## Validation

- Build the relevant C++/TableGen targets.
- Run focused capability-model and lit tests.
- Prefer `check-tianchenrv` if local toolchain state allows it within this
  round.
- Do not claim current-head RVV runtime evidence unless a real `ssh rvv` leg is
  run.
