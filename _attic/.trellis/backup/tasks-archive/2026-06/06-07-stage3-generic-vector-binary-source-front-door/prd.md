# Stage3 generic Vector binary source-front-door materialization boundary

## Goal

Refactor the RVV Vector source-front-door production materializer from an
add-specific owner into one bounded generic binary-family boundary. The
materializer should structurally recognize a small Vector-like source pattern,
derive the binary kind from source IR, materialize a selected `tcrv.exec` RVV
dispatch case with typed generic `tcrv_rvv.setvl/load/binary/store` body facts,
and then let the existing RVV provider route/export/artifact path validate and
execute it.

## What I Already Know

* Current HEAD before this task is `4a77052b rvv: prove vector source-front-door bundle execution`; worktree was clean before task creation.
* No `.trellis/.current-task` existed, so this task was created directly from the Hermes direction brief.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-07-stage3-vector-source-artifact-boundary/`
  proved the bounded Vector source-front-door add bundle through generated
  artifact export and `ssh rvv` execution.
* The production owner is still add-specific:
  `createMaterializeRVVVectorAddSourceFrontDoorPass`,
  `MaterializeRVVVectorAddSourceFrontDoorPass`,
  `matchBoundedVectorAddSourceFunc`, `VectorAddSourceMatch`,
  `createRVVBinaryAdd`, and add-only registration/diagnostics.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires RVV route authority
  to start from typed `tcrv_rvv` body/config/runtime facts and RVV provider
  validation, not source markers, route ids, artifact names, or Common EmitC.
* `.trellis/spec/lowering-runtime/emitc-route.md` keeps Common EmitC/export
  neutral: it may consume provider-built payloads but must not infer operation
  kind, dtype, SEW/LMUL, ABI, schedule, or intrinsic semantics.
* `scripts/rvv_generated_bundle_abi_e2e.py --vector-source-front-door` is
  currently add-only evidence tooling. It should become evidence for the same
  generic binary source boundary, not a replacement authority.

## Requirements

* Rename/refactor the RVV source-front-door materializer interface and pass so
  the production owner is generic binary-family oriented, not add-only.
* Match only a bounded rank-1 i32 Vector-like source pattern:
  two `vector.transfer_read` ops from structural lhs/rhs arguments, one
  supported `arith` vector binary op, and one `vector.transfer_write` to the
  structural out argument, plus runtime `n` index ABI input.
* Derive binary kind from source IR. For this round the supported set is
  `add`, `sub`, and `mul` from `arith.addi`, `arith.subi`, and `arith.muli`.
* Materialize selected `tcrv.exec` and typed `tcrv_rvv` body facts using the
  derived kind: runtime ABI values, `setvl`, `with_vl`, two loads,
  `tcrv_rvv.binary {kind = ...}`, and store.
* Keep marker/source-front-door metadata as opt-in materialization boundary
  only. It must not decide route support after the body is materialized.
* Keep Common EmitC/export untouched as RVV semantic authority. The existing
  provider must consume the materialized typed body facts for route/export/header
  evidence.
* Fail closed for unsupported source op kinds, stale selected-body/TCRV
  residue, unsupported Vector forms, dtype/config mismatch, ABI role mismatch,
  or invalid source kernel name.
* Preserve legacy `rvv-i32m1` vector-source-front-door fixtures only as
  deprecated/fail-closed inventory; do not use them as templates.

## Acceptance Criteria

* [x] Public RVV source-front-door materializer names and pass registration are
      generic binary-family oriented; add-only names are removed from the
      production owner or kept only as deprecated test inventory if unavoidable.
* [x] `tcrv-opt` FileCheck coverage proves add/sub/mul source inputs
      materialize to selected typed `tcrv_rvv.binary` bodies with the correct
      derived `kind`, runtime ABI bindings, `setvl/load/store`, provider route
      planning, and target header evidence.
* [x] Negative `tcrv-opt` coverage proves unsupported source binary ops and
      stale TCRV/selected-body residue fail closed before emission/bundle
      authority.
* [x] `scripts/rvv_generated_bundle_abi_e2e.py --vector-source-front-door`
      accepts non-add binary op kinds covered by the generic materializer and
      records source marker/artifact metadata as mirror-only.
* [x] Generated-bundle dry-run evidence covers at least `add`, `sub`, and `mul`
      through the generic source-front-door materializer and existing provider
      route/export path.
* [x] At least one non-add `ssh rvv` generated-bundle execution passes before
      any runtime correctness claim for non-add is made.
* [x] `build/bin/tianchenrv-rvv-extension-plugin-test` and
      `build/bin/tianchenrv-target-artifact-export-test` pass.
* [x] Relevant lit/script tests pass.
* [x] A bounded old-authority scan over touched production files and added diff
      lines shows no new positive `RVVI32M1`, `rvv-i32m1`,
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor-driven compute,
      source-front-door route-id authority, or Common EmitC semantic branch.
* [x] `git diff --check`, `git diff --cached --check`, and final
      `git status --short` are clean after commit.

## Out Of Scope

* No broad Vector, Linalg, StableHLO, tensor/tile, or frontend generalization.
* No dtype/LMUL matrix, compare/select, reduction, MAcc, memory-form expansion,
  or per-op clone table beyond the bounded binary family.
* No old `i32m1` compatibility route and no source-artifact/source-seed
  resurrection.
* No Common EmitC RVV semantic selection and no descriptor-to-C route.
* No dashboard/report/index-only closeout.

## Technical Notes

* Direction source: Hermes/Codex worker direction brief supplied on
  2026-06-07.
* Specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/index.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`.
* Previous task read:
  `.trellis/tasks/archive/2026-06/06-07-stage3-vector-source-artifact-boundary/prd.md`.
* Primary production files:
  `include/TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h`,
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`,
  `lib/Plugin/RVV/RVVExtensionPlugin.cpp`.
* Primary evidence files:
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Transforms/RVV/rvv-vector-binary-source-front-door.mlir`,
  `test/Transforms/RVV/rvv-vector-binary-source-front-door-negative.mlir`,
  `test/Support/RVV/rvv-vector-binary-source-front-door-{add,sub,mul}.mlir.inc`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-vector-source-front-door-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-vector-source-front-door-fail-closed.test`.

## Completion Evidence

* Replaced the add-specific production materializer interface with
  `createMaterializeRVVVectorBinarySourceFrontDoorPass` and pass argument
  `tcrv-rvv-materialize-vector-binary-source-front-door`.
* Refactored `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp` around a
  `VectorBinarySourceMatch` that derives `add`, `sub`, or `mul` from
  `arith.addi`, `arith.subi`, or `arith.muli` vector source IR.
* The materializer now builds op-kind-specific selected variants
  `rvv_vector_add`, `rvv_vector_sub`, and `rvv_vector_mul` with typed generic
  `tcrv_rvv.binary {kind = ...}` bodies. Dispatch policy is generic
  `rvv-vector-binary-source-front-door-case`.
* Added generic positive lit coverage for add/sub/mul materialization,
  emission-plan route facts, default artifact front-door pipeline, and target
  header export.
* Added negative lit coverage for stale selected-boundary residue and an
  unsupported vector arithmetic op (`arith.andi`) failing before route/export
  authority.
* Extended `scripts/rvv_generated_bundle_abi_e2e.py --vector-source-front-door`
  from add-only to add/sub/mul and updated evidence strings to
  `bounded-vector-binary-source-front-door`.
* Generated-bundle dry-run lit covers add/sub/mul op evidence, harnesses, and
  bundle metadata through the generic source materializer and existing RVV
  provider path.
* Non-add real RVV execution passed:
  `PASS op=sub counts=0,1,17,257 source_preserved tail_preserved`.
* Self-repair: the first non-add remote run failed closed before bundle use
  because `llvm-readobj` was not on PATH; reran with
  `/usr/lib/llvm-20/bin/llvm-readobj` and passed.
