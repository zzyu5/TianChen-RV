# RVV no-body probe/bundle fixture erasure

## Goal

Erase stale no-body RVV fixture authority from active probe replay and
target-artifact-bundle negative tests. The round is deletion/refactor-only:
probe replay and bundle export evidence must remain fail-closed generic
coverage, not a named source-deleted or no-body RVV rebuild-gap contract.

## Context

- The previous completed commit is `20f0b16 chore(rvv): erase lowering-boundary wrapper`.
- The working tree was clean before this task started.
- `test/Scripts/rvv-probe-to-mlir.test` still has an `I64-SOURCE-DELETED`
  negative run and check block for `rvv_probe_i64_replay`.
- `test/Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`
  still checks the specific no-body RVV proposal/rebuild-gap wording.
- The testing spec allows Python RVV probe replay only as sanitized evidence
  tooling and MLIR pipeline input. It must not become RVV lowering, emission,
  runtime ABI, correctness, or performance authority.
- The RVV plugin spec says no-body `tcrv.exec.kernel` input must not select a
  binary family, route metadata, callable ABI, artifact route, or generated
  body. Future executable work must start from explicit extension-family IR
  and a materialized EmitC route.

## Requirements

- Delete or rewrite the `I64-SOURCE-DELETED` active fixture authority so the
  i64 replay test only checks valid replay/profile surface, not the named
  deleted source/no-body route.
- Delete or rewrite the no-viable bundle negative wording so it remains a
  generic planning/bundle failure check without preserving the named no-body
  rebuild-gap phrase as architectural authority.
- Preserve still-valid coverage for:
  - probe-to-MLIR self-test and replay output;
  - parseable replay MLIR;
  - scalar fallback behavior when RVV evidence is unavailable or declined;
  - target-artifact-bundle failure when planning yields no supported bundle.
- Do not add any replacement route, wrapper, alias, compatibility layer,
  runtime ABI, direct C exporter, RVV lowering, or bundle implementation.
- Do not broaden into unrelated RVV legality, plugin emission-plan,
  selected-boundary, or rebuild work unless the same stale fixture text is a
  direct blocker.

## Acceptance Criteria

- [ ] Active tests no longer contain the `I64-SOURCE-DELETED` named check block.
- [ ] Active tests no longer preserve `no-body RVV proposal materialization is
  a rebuild gap` as a required diagnostic.
- [ ] Active tests no longer require `requires an explicit typed RVV
  extension-family body` as a source-deleted/no-body route contract.
- [ ] Generic no-viable planning and bundle failure coverage remains.
- [ ] No source, header, object, runtime ABI, direct exporter, wrapper, alias,
  or RVV rebuild path is added.
- [ ] Focused active-surface reference scan over the requested strings is
  reported, excluding generated/archive/run-artifact paths.
- [ ] Targeted checks for the two affected tests are run, plus `ninja -C build
  tcrv-opt`, `git diff --check`, Trellis validation, and an attempted
  `check-tianchenrv`.
- [ ] Task is finished/archived and committed if the round completes.

## Out Of Scope

- Probe-to-MLIR rewrite.
- Target artifact bundle export implementation.
- New RVV lowering, RVV emission, runtime ABI, direct C export, or source route.
- Compatibility aliases for deleted RVV source/no-body behavior.
- Broad test matrix repair outside the two stale fixture anchors.

## Technical Notes

- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
- Relevant tests:
  - `test/Scripts/rvv-probe-to-mlir.test`
  - `test/Target/TargetArtifactBundleExport/plan-and-export-target-artifact-bundle-no-viable.mlir`
- `scripts/rvv_probe_to_mlir.py` is not expected to change unless the fixture
  edit reveals a directly related stale comment.
