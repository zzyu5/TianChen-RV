# PRD: Stage2 RVV indexed memory route-family ABI and provider ownership

## Direction Source

Hermes/user direction redirects the next owner to the RVV plugin-owned indexed
memory route family, covering:

- `indexed_gather_unit_store`
- `indexed_scatter_unit_load`
- `computed_masked_indexed_gather_load_unit_store`
- `computed_masked_indexed_scatter_store_unit_load`

The previous archived segment2 task closed generated-artifact and `ssh rvv`
evidence, but did not move production code. This task must not complete through
another evidence-only archive.

## Module Goal

Harden the active production boundary where provider-derived indexed
gather/scatter facts are consumed by target artifact validation. The bounded
production submodule for this round is computed-mask indexed scatter artifact
ABI validation: target artifact validation must consume the provider-derived
indexed scatter uniqueness fact and its candidate metadata mirror, and reject
stale or metadata-only mismatches before export.

## Current Repository Facts

- The current mainline is RVV-first selected body:
  `tcrv.exec` selected RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  selected-body realization/provider -> `TCRVEmitCLowerableRoute` -> neutral
  EmitC -> target artifact.
- Existing selected-body owners and route-family plan owners already cover the
  four indexed base/computed-mask forms.
- Base indexed scatter target artifact validation already consumes
  `index_uniqueness`.
- Computed-mask indexed scatter generated metadata already carries
  `tcrv_rvv.index_uniqueness = unique`, but the compare/select-mask target
  artifact validator does not currently require the provider description or
  candidate mirror to match that fact.

## Requirements

1. Move production owner code, not only tests or evidence summaries.
2. Keep `tcrv.exec` as ABI/runtime envelope only; do not infer RVV dtype,
   schedule, operation, index semantics, or route support from ABI strings.
3. Keep common EmitC neutral. RVV-specific indexed scatter facts must be
   consumed in RVV-owned provider/target validation, not invented by common
   materialization.
4. Validate computed-mask indexed scatter `index_uniqueness` as a
   provider-derived route fact:
   - `computed_masked_indexed_scatter_store_unit_load` requires `unique`.
   - non-scatter compare/select-mask routes must reject stale non-empty
     uniqueness facts.
5. Validate `tcrv_rvv.index_uniqueness` as mirror metadata only after provider
   route construction:
   - computed-mask indexed scatter candidate mirrors must match `unique`.
   - non-scatter compare/select-mask candidate mirrors must remain empty.
6. Preserve existing indexed gather, base indexed gather/scatter, strided,
   computed-mask unit, and segment2 behavior.

## Non-Goals

- No segment2 evidence-only closure.
- No direct pre-realized route-entry shortcut.
- No Linalg/frontend lowering, high-level kernel ops, one-intrinsic wrapper
  dialects, dtype/LMUL clone batches, reduction/contraction expansion, global
  autotuning, readiness dashboards, or future plugin work.
- No descriptor-driven computation, source-front-door positive route, route-id
  authority, artifact-name authority, metadata authority, common-EmitC semantic
  invention, or legacy `i32m1` route-table resurrection.

## Acceptance Criteria

- Production diff in the RVV target artifact route-family validation path.
- C++ target artifact coverage proves:
  - provider facts for computed-mask indexed scatter require
    `index_uniqueness = unique`;
  - stale provider-derived uniqueness is rejected;
  - stale candidate metadata mirror is rejected;
  - computed-mask indexed gather remains accepted with no uniqueness authority.
- Focused selected-boundary generated-bundle dry-run covers all four indexed
  base/computed-mask forms.
- Real `ssh rvv` generated-bundle correctness evidence is run for representative
  counts including zero, one, exact-VL, tail, and stress cases, or an exact
  external blocker is recorded.
- Non-regression covers existing target artifact route-family validation and the
  retired direct pre-realized route-entry fail-closed path.
- Authority scan shows no new dependency on central ad hoc, descriptor-derived,
  ABI-string-derived, script-derived, artifact-name-derived, common-EmitC-
  derived, source-front-door-derived, route-id-derived, exact-intrinsic-derived,
  direct-route-entry-only, pre-realized-fixture-only, or legacy-i32-derived
  authority.
- `git diff --check` passes and final git status is clean after commit if the
  task completes.
