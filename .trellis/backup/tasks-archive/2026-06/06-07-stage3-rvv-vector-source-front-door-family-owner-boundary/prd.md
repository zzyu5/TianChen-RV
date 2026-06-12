# Stage3 RVV vector source-front-door family-owner modular boundary

## Goal

Tighten the RVV plugin-local vector source-front-door family registry and
materializer-owner boundary for the three active families:
bounded vector binary, bounded vector compare/select, and bounded vector
runtime-scalar compare/select. The workflow must remain source-only MLIR ->
family-owned materializer -> selected `tcrv.exec` case -> typed `tcrv_rvv`
body -> existing RVV provider route -> common EmitC -> target artifact. Source
markers, family names, artifact metadata, script op names, and test names stay
opt-in mirrors only.

## What I Already Know

* No `.trellis/.current-task` existed at session start; this task was created
  from the Hermes Direction Brief.
* Current HEAD is `da5fc456 rvv: materialize runtime scalar cmp select source
  front door`; worktree was clean before task creation.
* The archived binary and compare/select source-front-door work created the
  first family registry boundary and artifact bridge.
* The archived runtime-scalar compare/select source-front-door task added the
  third active family and `ssh rvv` evidence for executable behavior.
* `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp` already has a descriptor table
  for the three active families, but the production pass implementation still
  has three sibling pass classes and per-family `runOnOperation` boilerplate.
* `scripts/rvv_generated_bundle_abi_e2e.py` has evidence-side family contract
  mirrors for the three families; those checks must remain harness evidence,
  not compiler route or dtype authority.

## Requirements

* Keep exactly the three current active RVV vector source-front-door families in
  the active registry.
* Make materializer pass dispatch consume the family descriptor boundary instead
  of maintaining per-family pass class/argument/description/run boilerplate.
* Preserve family-local source matching and typed-body construction semantics:
  binary owns add/sub/mul shape, compare/select owns compare/select shape, and
  runtime-scalar compare/select owns scalar RHS splat plus select shape.
* Keep fail-closed behavior for unknown markers, stale lowering seeds,
  malformed source-only input, legacy i32m1 source-front-door, and non-matching
  sibling markers.
* Keep default artifact-front-door eligibility as a descriptor-owned fact for
  active families and explicit-only behavior for legacy fail-closed pass.
* Do not move RVV semantics into Common EmitC, target artifact export, scripts,
  route ids, artifact names, or descriptor metadata.

## Acceptance Criteria

* [x] Production C++ has one plugin-local family-owned materializer pass class
  or equivalent owner boundary, with argument/description/factory driven from
  the active family descriptor.
* [x] The active family registry still exposes exactly the three active
  families, all owned by the RVV plugin, with correct pass argument, default
  artifact eligibility, and non-null factories.
* [x] Plugin smoke coverage proves registry-created factories materialize pass
  objects whose public argument still matches the registered family argument.
* [x] Focused transform lit evidence passes for binary, compare/select,
  runtime-scalar compare/select, and family-registry negative cases.
* [x] Generated-bundle dry-run evidence passes for the three active
  source-front-door families.
* [x] If executable behavior is claimed after the refactor, runtime-scalar
  compare/select source-front-door evidence is rerun on `ssh rvv`.
* [x] Bounded old-authority scan over touched production files and added diff
  lines shows no new legacy i32m1/source-marker/artifact/Common EmitC route
  authority.
* [x] `git diff --check`, `git diff --cached --check`, final clean worktree,
  Trellis finish/archive, and one coherent commit.

## Definition of Done

* Focused C++/lit/script checks are green or any unavailable check is reported
  with exact reason.
* PRD completion evidence records production boundary improvement, checks,
  self-repair, and any runtime `ssh rvv` result.
* Task status is finished/archived when complete.

## Technical Approach

Use the existing active family descriptor as the owner boundary. Replace the
three sibling materializer pass classes with one descriptor-driven
`MaterializeRVVVectorSourceFrontDoorFamilyPass` that stores a family id and
returns descriptor-owned argument/description. Its `runOnOperation` calls one
small dispatcher that performs the family-specific match/materialize step. Keep
public factory functions for compatibility, but have them instantiate the same
generic family pass. Keep provider, Common EmitC, target export, and Python
evidence semantics unchanged.

## Decision (ADR-lite)

**Context**: After adding the third active family, registry ownership existed
for pass registration but materializer ownership still had per-family pass
boilerplate. That made future family additions require broad hand edits across
pass classes even when descriptor facts were already present.

**Decision**: Consolidate materializer pass ownership around the active family
descriptor while preserving family-local matchers/materializers and existing
public factory names.

**Consequences**: Future active family additions should add one descriptor and
one family-local match/materialize arm rather than a new pass class with
duplicated argument/description/run code. This does not generalize source
matching, add route coverage, or make source-front-door metadata authoritative.

## Out of Scope

* No new source-front-door family.
* No Linalg/StableHLO/frontend generalization.
* No MAcc, reduction, dtype/LMUL clone batch, or route coverage expansion.
* No descriptor-driven compute, source-artifact authority, or Common EmitC RVV
  semantic branch.
* No compatibility path reviving legacy `RVVI32M1*`, `rvv-i32m1`,
  `tcrv_rvv.i32_*`, or `!tcrv_rvv.i32m*` route authority.
* No dashboard/report/index-only completion.

## Technical Notes

Read for this task:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/extension-plugins/index.md`
* `.trellis/spec/lowering-runtime/index.md`
* `.trellis/spec/testing/index.md`
* Archived source-front-door PRDs under `.trellis/tasks/archive/2026-06/`
* `include/TianChenRV/Plugin/RVV/RVVVectorSourceFrontDoor.h`
* `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`
* `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
* `test/Plugin/RVVExtensionPluginTest.cpp`
* `scripts/rvv_generated_bundle_abi_e2e.py`
* `test/Transforms/RVV/rvv-vector-*-source-front-door*.mlir`
* `test/Support/RVV/rvv-vector-*-source-front-door*.mlir.inc`
* `test/Scripts/rvv-generated-bundle-abi-e2e-vector-*-source-front-door*.test`

## Completion Evidence

### Production Boundary Improvement

* Replaced three active-family materializer pass classes with one
  `MaterializeRVVVectorSourceFrontDoorFamilyPass` that stores the active
  family id and reads pass argument/description from the family owner entry.
* Added `materializeRVVVectorSourceFrontDoorFamily(...)` as the single
  materializer-owner dispatch entry. It keeps binary, compare/select, and
  runtime-scalar compare/select parser/materializer semantics family-local, but
  centralizes shared no-op/fail/cleanup pass control.
* Kept the public factory functions
  `createMaterializeRVVVectorBinarySourceFrontDoorPass`,
  `createMaterializeRVVVectorCompareSelectSourceFrontDoorPass`, and
  `createMaterializeRVVVectorRuntimeScalarCompareSelectSourceFrontDoorPass`;
  they now instantiate the same family-owned pass used by the registry.
* Updated `test/Plugin/RVVExtensionPluginTest.cpp` to instantiate each
  registry-created family pass factory and assert that the created pass exposes
  the same family-owned pass argument.
* Updated `.trellis/spec/extension-plugins/rvv-plugin.md` so the durable RVV
  source-front-door registry contract names all three active families and
  requires registry-owned pass factory/materializer ownership.

### Checks Run

* `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test -j2`
* `build/bin/tianchenrv-rvv-extension-plugin-test`
* `build/bin/tianchenrv-target-artifact-export-test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter "rvv-vector-(binary-source-front-door|compare-select-source-front-door|runtime-scalar-cmp-select-source-front-door|source-front-door-family-registry-negative)|rvv-i32m1-vector-source-front-door"` from `build/test`
* `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter "rvv-generated-bundle-abi-e2e-(vector-source-front-door-dry-run|vector-compare-select-source-front-door-dry-run|vector-runtime-scalar-cmp-select-source-front-door-dry-run|vector-source-front-door-fail-closed|self-test)"` from `build/test`
* Rebuilt and reran `build/bin/tianchenrv-rvv-extension-plugin-test` after
  tightening the assertion text.
* `git diff --check`
* `git diff --cached --check`
* Touched-production-file old-authority scan over
  `lib/Plugin/RVV/RVVVectorSourceFrontDoor.cpp`
* Added-line old-authority scan over touched code files

`clang-format`/`clang-format-20`/`clang-format-19` were not installed in this
environment; whitespace checks passed via `git diff --check`.

### Runtime Evidence

Reran runtime-scalar compare/select source-front-door generated-bundle evidence
on `ssh rvv`:

```text
artifact_dir: artifacts/tmp/rvv_generated_bundle_abi_e2e/codex-family-owner-runtime-scalar-cmp-select-ssh
PASS op=runtime_scalar_cmp_select counts=0,1,17,257 rhs_scalars=-37,91 true_lanes=332 false_lanes=216 mixed_cases=4 all_true_cases=0 all_false_cases=0
```

### Self-Repair

* Added `<memory>` to `test/Plugin/RVVExtensionPluginTest.cpp` after introducing
  `std::unique_ptr<mlir::Pass>` in the test.
* Changed the new plugin test assertion wording from `descriptor-owned` to
  `family-owned` so it cannot be confused with forbidden descriptor-driven
  compute authority.
