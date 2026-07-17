# Stage3 generic Vector compare/select source-front-door materialization boundary

## Goal

Add the bounded Stage3 source-front-door materialization boundary for one
production generic Vector compare/select workflow. Source MLIR may opt into the
boundary with an RVV source-front-door marker, but the marker must only select
materialization. The pass must structurally recognize a plain Vector
compare/select source form, materialize a selected `tcrv.exec` RVV dispatch case
containing explicit realized typed `tcrv_rvv.setvl/load/compare/select/store`
body facts, and then rely on the existing RVV route provider, Common EmitC,
target artifact export, and generated-bundle evidence.

## What I Already Know

* Current HEAD before this task is `c6197d16 rvv: generalize vector source
  binary front door`; worktree was clean before task creation.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes/Codex direction brief.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-07-stage3-generic-vector-binary-source-front-door/`
  generalized the production Vector source-front-door boundary for add/sub/mul
  and registered it as default source-artifact-front-door eligible.
* Current production owner files are
  `include/TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h`,
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`, and
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`.
* Existing compare/select route support already exists for explicit realized
  `tcrv_rvv.setvl/load/compare/select/store` selected bodies, and also for
  pre-realized compare/select bodies after public selected lowering-boundary
  materialization. This task follows the previous binary source-front-door
  shape and directly materializes the explicit realized body.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires compare/select
  source materialization to produce typed selected-body facts first. Provider
  route construction must not infer predicate, mask source, layout, dtype, ABI,
  or route support from source markers, route ids, artifact names, C strings, or
  Common EmitC metadata.
* `.trellis/spec/lowering-runtime/emitc-route.md` keeps Common EmitC/export
  neutral. It may carry provider-built route payloads and mirrors, but it must
  not choose compare/select semantics.
* The source-artifact-front-door pipeline runs all eligible pass registrations
  in order, so RVV sibling source-front-door markers must not cause one RVV pass
  to fail before the matching sibling pass can consume the marker.

## Requirements

* Add a production RVV source-front-door materializer for bounded plain Vector
  compare/select source IR.
* Match only source-only modules explicitly marked for this compare/select
  boundary.
* Recognize one rank-1 i32 Vector source pattern:
  two `vector.transfer_read` ops from structural `lhs` and `rhs` arguments, one
  supported `arith.cmpi` vector compare op, one `arith.select` that consumes
  that compare mask and selects `lhs` when true and `rhs` when false, one
  `vector.transfer_write` to structural `out`, and one runtime `n` index ABI
  input.
* Derive predicate from source IR. Supported predicates for this round are
  `eq`, `slt`, and `sle`, matching the already route-supported plain
  compare/select pre-realized body surface.
* Materialize a selected `tcrv.exec` RVV dispatch case with explicit runtime ABI
  values, `tcrv_rvv.setvl`, `tcrv_rvv.with_vl`, two `tcrv_rvv.load` ops, one
  `tcrv_rvv.compare` carrying the derived predicate, one `tcrv_rvv.select`
  using lhs as true value and rhs as false value, and one `tcrv_rvv.store`.
  The body must carry SEW32, LMUL m1, agnostic policy, selected variant/source
  kernel attributes, and the generic typed body route mapping.
* Preserve the existing RVV provider/Common EmitC/target route authority: the
  source materializer must not build `TCRVEmitCLowerableRoute`, emit intrinsics,
  or add Common EmitC semantic branches.
* Register the compare/select materializer as an RVV source-front-door pass and
  ensure the default source-artifact-front-door pipeline can consume either the
  existing binary marker or the new compare/select marker without sibling-pass
  false failures.
* Fail closed for unsupported predicates, unsupported select layout, missing
  runtime `n`, wrong dtype/config, missing/extra source structure, stale
  selected-body/TCRV residue, stale lowering seed metadata, and invalid source
  kernel names.
* Extend focused generated-bundle evidence tooling so a compare/select
  source-front-door input runs through the new source materializer, public
  selected lowering-boundary materialization, existing RVV provider route,
  target bundle export, and optional `ssh rvv` execution.

## Acceptance Criteria

* [x] Public RVV source-front-door materializer interface and plugin
      registration include a bounded compare/select materializer that is default
      source-artifact-front-door eligible.
* [x] Existing Vector binary source-front-door behavior remains intact.
* [x] Positive lit proves source compare/select materializes to a selected RVV
      variant containing explicit realized `tcrv_rvv.setvl/load/compare/select`
      /`store` body facts with the derived predicate and typed config/mask
      structure.
* [x] Positive lit proves the source-artifact-front-door pipeline reaches
      compare/select emission-plan and header evidence through existing
      selected-boundary realization/provider/target paths.
* [x] Negative lit proves unsupported predicate, unsupported select layout,
      missing runtime role, stale TCRV residue, and stale route/lowering seed
      metadata fail closed before route/export authority.
* [x] `scripts/rvv_generated_bundle_abi_e2e.py` supports a bounded
      compare/select source-front-door mode and records the marker as
      mirror-only opt-in, not route authority.
* [x] Generated-bundle dry-run covers at least one representative
      compare/select source-front-door artifact through the new materializer and
      existing provider/target validators.
* [x] Real `ssh rvv` evidence is collected before claiming runtime correctness
      for the compare/select source-front-door route.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass.
* [x] Focused `tcrv-opt` / `tcrv-translate` / lit/script checks for the changed
      behavior pass.
* [x] `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py` passes.
* [x] Bounded old-authority scan over touched production files and added diff
      lines shows no new positive `RVVI32M1`, `rvv-i32m1`,
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor-driven compute,
      source-front-door route-id authority, or Common EmitC semantic branch.
* [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are clean after commit.

## Out Of Scope

* No high-level Linalg, Vector dialect lowering framework, StableHLO, tensor,
  or frontend generalization.
* No per-Linalg route authority and no new high-level kernel op dialect.
* No Common EmitC invention of predicate, select layout, dtype, ABI, mask
  policy, or route support.
* No descriptor-driven C/source export and no source marker/artifact metadata as
  route authority.
* No dtype/LMUL matrix. This round is bounded to i32, SEW32, LMUL m1.
* No reduction, MAcc, memory/segment expansion, contraction, performance
  database, dashboard, or report-only closeout.
* No rework of the binary add/sub/mul source-front-door except the minimal
  sibling-marker behavior needed for the default source-artifact pipeline.

## Technical Notes

* Direction source: Hermes/Codex worker direction brief supplied on 2026-06-07.
* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`,
  `.trellis/spec/testing/mlir-testing-contract.md`,
  `.trellis/spec/guides/index.md`,
  `.trellis/spec/guides/plugin-locality-review-guide.md`,
  `.trellis/spec/guides/compute-boundary-review-guide.md`.
* Previous task read:
  `.trellis/tasks/archive/2026-06/06-07-stage3-generic-vector-binary-source-front-door/prd.md`,
  `implement.jsonl`, and `check.jsonl`.
* Primary production files:
  `include/TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h`,
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`.
* Primary evidence files:
  `test/Transforms/RVV/rvv-vector-binary-source-front-door.mlir`,
  `test/Transforms/RVV/rvv-vector-binary-source-front-door-negative.mlir`,
  new compare/select source-front-door lit fixtures under `test/Transforms/RVV`
  and `test/Support/RVV`,
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  focused script tests under `test/Scripts`,
  existing compare/select target fixtures under `test/Target/RVV`.

## Completion Evidence

### Production Boundary

* Added public pass/factory
  `createMaterializeRVVVectorCompareSelectSourceFrontDoorPass()` with argument
  `tcrv-rvv-materialize-vector-compare-select-source-front-door`.
* Registered the pass as RVV source-front-door eligible for the default
  source-artifact-front-door pipeline.
* Materializer recognizes the bounded source marker
  `bounded_vector_compare_select_source`, derives only `eq`/`slt`/`sle`
  predicate and lhs-true/rhs-false select layout from Vector source IR, and
  materializes a selected `tcrv.exec` RVV variant with realized typed
  `tcrv_rvv.setvl/load/compare/select/store` facts.
* Existing binary source-front-door pass now no-ops on the compare/select
  sibling marker, and the compare/select pass no-ops on the binary sibling
  marker. Unknown/stale markers still fail closed.
* No Common EmitC semantic branch or provider shortcut was added; route/export
  continues through provider-built `TCRVEmitCLowerableRoute` and existing
  target artifact validation.

### Focused Checks

* `cmake --build build --target tcrv-opt tcrv-translate
  tianchenrv-rvv-extension-plugin-test
  tianchenrv-target-artifact-export-test -j 4`
* `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
* `build/bin/tianchenrv-rvv-extension-plugin-test`
* `build/bin/tianchenrv-target-artifact-export-test`
* From `build/test`:
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv .
  --filter 'rvv-vector-compare-select-source-front-door|rvv-generated-bundle-abi-e2e-vector-compare-select-source-front-door-dry-run|rvv-vector-binary-source-front-door|rvv-generated-bundle-abi-e2e-vector-source-front-door-fail-closed'`
  selected 6 tests and all passed.
* `git diff --check`
* Bounded old-authority scan over touched production C++ files and added
  production diff lines found no new positive old authority. Full script scan
  still contains pre-existing exact intrinsic fixture/evidence strings; added
  diff lines did not add old route authority.

### `ssh rvv` Evidence

Command:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --vector-source-front-door \
  --artifact-root artifacts/tmp/rvv_generated_bundle_abi_e2e \
  --run-id vector-compare-select-source-front-door-ssh \
  --overwrite \
  --op-kind cmp_select \
  --runtime-count 0 \
  --runtime-count 1 \
  --runtime-count 17 \
  --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj \
  --ssh-target rvv
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
artifact_dir: artifacts/tmp/rvv_generated_bundle_abi_e2e/vector-compare-select-source-front-door-ssh
tcrv_rvv_generated_bundle_abi_cmp_select_ok counts=0,1,17,257
PASS op=cmp_select counts=0,1,17,257
```

### Self-Repair

* First focused lit run exposed an over-specific new dry-run FileCheck harness
  assertion that expected `source_preserved`/before-copy text not emitted by
  the compare/select harness. The harness actually checks predicate coverage,
  selected result correctness, and tail sentinel preservation. The test was
  corrected to assert those real guarantees and the focused lit group was
  rerun successfully.

### Spec Update Judgment

* Updated `.trellis/spec/extension-plugins/rvv-plugin.md` with the bounded
  Vector compare/select source-front-door contract, including pass signature,
  source marker, supported source shape, validation matrix, tests required, and
  the sibling-marker pipeline rule.
