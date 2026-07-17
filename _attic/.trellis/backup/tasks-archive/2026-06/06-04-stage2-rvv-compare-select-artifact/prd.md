# Stage2 RVV compare/select selected-body realization to artifact evidence

## Goal

Close the bounded artifact-evidence gap for Stage2 RVV compare/select selected-body
realization. A selected `tcrv.exec` RVV variant with an explicit typed
`tcrv_rvv` compare/select body must be visible in generated target-artifact
evidence as:

```text
typed/pre-realized tcrv_rvv compare/select body
  -> RVV plugin-local selected-body realization
  -> realized setvl/with_vl/compare/select/mask structure
  -> provider-owned compare/select route and runtime AVL/VL contract
  -> common EmitC target artifact
```

This round is not a new compare/select route-family expansion. Live repository
inspection shows the production provider, selected-body realization, target
validation, and generated-bundle machinery already exist for the active
compare/select families. The bounded work is to make representative generated
artifact evidence assert the runtime AVL/VL selected-boundary facts alongside
the existing predicate/select evidence, so the artifact evidence cannot be read
as predicate-only or metadata-derived support.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV compare/select selected-body realization to artifact evidence`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk git status --short`: no short status entries were reported.
* Initial `rtk git log --oneline -8` started at
  `b79b6f16 rvv: close scalar splat runtime mirror labels`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* This round is run by one serial Codex worker; no subagents or multi-agent
  workflow are used.

## What I Already Know

* `.trellis/spec/index.md` defines the current RVV-first authority chain:
  selected `tcrv.exec` envelope -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned realization/provider -> `TCRVEmitCLowerableRoute` -> neutral
  common EmitC -> target artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` has explicit
  elementwise/compare-select selected-body realization and compare/select
  provider preflight contracts. Provider/common code must not synthesize
  compare, select, mask, dtype, policy, or runtime facts from route ids,
  artifact names, descriptors, C strings, or metadata mirrors.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires
  `RVVCompareSelectRouteValidationContract` to embed
  `RVVRuntimeAVLVLSelectedBoundaryContract`, and target artifact validation
  must consume that runtime contract before accepting compare/select payloads,
  statement plans, predicate/select facts, ABI/header/type facts, and mirrors.
* `.trellis/spec/testing/mlir-testing-contract.md` allows positive generated
  artifact tests only for corrected typed `tcrv_rvv` routes and requires real
  `ssh rvv` evidence only when runtime/correctness/performance behavior is
  claimed or changed.
* Archived compare/select tasks show the earlier production path has already
  been implemented:
  * selected-body realization for plain, computed-mask, runtime-scalar, and
    runtime-scalar dual compare/select;
  * provider-owned route validation and metadata mirror contracts;
  * target-side runtime AVL/VL contract migration and sole-authority cleanup;
  * candidate metadata runtime mirror closure.
* Live code inspection confirms `RVVCompareSelectRouteValidationContract`
  already carries `runtimeAVLVLContract`, target validation consumes it before
  compare/select route payload validation, and the generated-bundle script
  already emits a `runtime_avl_vl_boundary` evidence object for compare/select
  dry-runs.
* Live generated-bundle tests assert `compare_select_predicate_boundary` for
  representative pre-realized compare/select artifacts, but do not directly
  assert the `runtime_avl_vl_boundary` object in those artifact JSON files.

## Requirements

* Keep the implementation owner bounded to compare/select selected-body
  artifact evidence.
* Preserve the production authority chain and do not move RVV semantics into
  common EmitC/export or target metadata.
* For representative pre-realized compare/select generated-bundle dry-run
  evidence, assert both:
  * compare/select predicate/select boundary facts; and
  * runtime AVL/VL selected-boundary facts for `runtime_param n`, `setvl`,
    materialized loop VL, artifact ABI prototype, and mirror-only route
    metadata role.
* The strengthened evidence must continue to prove that pre-realized selected
  bodies are consumed through `--tcrv-materialize-selected-lowering-boundaries`
  rather than direct route-entry realization.
* Do not add new compare predicates, dtype/LMUL clone batches, new route
  families, source-front-door positive routes, descriptor/direct-C/source-export
  paths, or high-level frontend lowering.
* Do not change emitted C/C++, runtime ABI order, runtime counts, statement
  ordering, compare/select behavior, correctness behavior, or performance
  behavior unless live inspection reveals a production defect that requires it.

## Acceptance Criteria

* [x] At least one representative pre-realized compare/select artifact evidence
      test asserts the generated `runtime_avl_vl_boundary` object, including
      provider-derived authority, mirror-only evidence role, selected runtime
      ABI `n`, materialized setvl/loop facts, emitted setvl/VL facts,
      artifact ABI prototype, runtime counts, and
      `runtime_counts_are_execution_cases_not_vl_authority`.
* [x] Existing compare/select predicate/select evidence assertions remain
      intact for the same artifact path.
* [x] Direct pre-realized route-entry fail-closed tests remain unchanged and
      continue to prove unsupported bypass behavior.
* [x] No generated runtime behavior changes. If final diff is evidence/test
      only, record the no-runtime-change rationale and do not claim new runtime
      correctness.
* [x] Focused compare/select generated-bundle lit dry-run checks pass.
* [x] `tianchenrv-target-artifact-export-test` and
      `tianchenrv-rvv-extension-plugin-test` are run only if provider/target or
      plugin code changes; otherwise record the no-production-code-change
      rationale.
* [x] Bounded old-authority scan over touched files finds no new positive
      dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door,
      route-id/artifact-name authority, common EmitC semantic inference,
      exact `__riscv_*_i32m1`, or mirror-only authority.
* [x] `rtk git diff --check` passes.
* [x] Trellis task state, journal/archive state, and commit state are truthful.

## Technical Approach

1. Confirm the live generated evidence already contains
   `runtime_avl_vl_boundary` for a pre-realized compare/select path.
2. Extend the focused compare/select generated-bundle dry-run FileCheck
   assertions to require that runtime AVL/VL boundary evidence.
3. Prefer the plain pre-realized `cmp_select`/`cmp_select_sle` dry-run because
   it is the narrowest representative selected-body path and already asserts
   predicate/select behavior.
4. Run the focused lit filter for compare/select selected-body artifacts.
5. Run a bounded old-authority scan over the changed test/task files and
   `rtk git diff --check`.

## Evidence Plan

* Focused generated-bundle lit filter from `build/test`:

  ```text
  rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'selected-body-artifact-(cmp-select|computed-mask-select|runtime-scalar-cmp-select|runtime-scalar-dual-cmp-mask-and-select)'
  ```

* If only tests/task files change, do not rebuild or rerun
  `tianchenrv-target-artifact-export-test` /
  `tianchenrv-rvv-extension-plugin-test`; record that provider/target/plugin
  production code was unchanged.
* Added-line old-authority scan over touched files.
* `rtk git diff --check`.
* Trellis validation:

  ```text
  rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-stage2-rvv-compare-select-artifact
  ```

## SSH RVV Rationale

Do not run `ssh rvv` if the final diff only strengthens generated-bundle
artifact evidence assertions. That diff does not change emitted C/C++,
runtime ABI order, runtime counts, statement ordering, compare/select runtime
behavior, correctness behavior, or performance behavior. The evidence remains
a dry-run artifact inspection, not a new runtime correctness claim.

## Out Of Scope

* New RVV compare/select operations, predicates, select layouts, memory forms,
  dtype/LMUL coverage, reduction/contraction/conversion/memory/scalar-splat
  work, source-front-door routes, high-level frontend lowering, global tuning,
  dashboards, or broad smoke matrices.
* Rewriting compare/select production realization/provider/target validation
  unless live evidence shows the current production path is defective.
* Moving compare/select runtime/predicate/selection semantics into common
  EmitC/export, target metadata, scripts, descriptors, route ids, artifact
  names, exact intrinsic spellings, or test names.

## Definition Of Done

Representative pre-realized compare/select generated artifact evidence visibly
ties selected-body realization to both predicate/select facts and runtime
AVL/VL selected-boundary facts; focused lit and scans pass; no runtime behavior
change is claimed; the task is validated, finished/archived, and committed if
complete.

## Implementation Results

* Strengthened
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-cmp-select-dry-run.test`
  so the root evidence must list `runtime_avl_vl_boundary`.
* Added per-artifact FileCheck assertions for both `cmp_select` and
  `cmp_select_sle` proving the generated evidence records:
  * provider-derived typed `tcrv_rvv` body/config/runtime authority;
  * mirror-only evidence role after provider route and materialized EmitC;
  * selected runtime ABI `n` as `runtime-element-count`;
  * materialized `setvl` consuming runtime AVL and `with_vl` consuming `setvl`;
  * emitted `__riscv_vsetvl_e32m1` / loop-VL evidence;
  * route metadata mirrors for runtime AVL source, runtime control plan, and
    runtime VL contract;
  * artifact ABI runtime parameter count;
  * `runtime_counts_are_execution_cases_not_vl_authority`.
* Existing predicate/select boundary assertions and direct route-entry
  fail-closed tests were left unchanged.
* No provider, target validation, plugin realization, common EmitC, generated C
  emission, runtime ABI order, statement ordering, compare/select semantics, or
  runtime behavior changed.

## Evidence Results

### Focused Lit

Single changed file:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-cmp-select-dry-run'
```

Run from `build/test`. Result:

```text
Total Discovered Tests: 477
  Excluded: 476 (99.79%)
  Passed  :   1 (0.21%)
```

Compare/select selected-body artifact filter:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'selected-body-artifact-(cmp-select|computed-mask-select|runtime-scalar-cmp-select|runtime-scalar-dual-cmp-mask-and-select)'
```

Run from `build/test`. Result:

```text
Total Discovered Tests: 477
  Excluded: 457 (95.81%)
  Passed  :  20 (4.19%)
```

### Provider/Target C++ Tests

`tianchenrv-target-artifact-export-test` and
`tianchenrv-rvv-extension-plugin-test` were not rerun. This round did not
change provider/target/plugin production code, route construction, target
validation, or emitted C/C++ behavior; the focused generated-bundle lit checks
cover the changed evidence assertions.

### Old-Authority Scan

Added-line scan over touched task and test files:

```text
rtk proxy bash -lc 'matches=$(git diff -U0 -- .trellis/tasks/06-04-stage2-rvv-compare-select-artifact/prd.md .trellis/tasks/06-04-stage2-rvv-compare-select-artifact/implement.jsonl .trellis/tasks/06-04-stage2-rvv-compare-select-artifact/check.jsonl test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-cmp-select-dry-run.test | rg -n "^\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\.i32_|!tcrv_rvv\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|route-id|artifact-name|common EmitC semantic|mirror-only authority)"); status=$?; if [ $status -eq 0 ]; then printf "%s\n" "$matches"; exit 1; fi; exit 0'
```

Result: passed with no matches.

### Whitespace

```text
rtk git diff --check
```

Result: passed.

### SSH RVV

Not run. This round changed a generated-bundle dry-run test and Trellis task
metadata only. It did not change emitted C/C++, runtime ABI order, runtime
counts, statement ordering, compare/select behavior, correctness behavior, or
performance behavior, so it makes no new runtime correctness claim.

## Spec Update Decision

No `.trellis/spec/` update is needed. The existing RVV plugin and EmitC route
specs already require compare/select selected-body realization, provider-owned
compare/select route contracts, and runtime AVL/VL selected-boundary contract
consumption before artifact acceptance. This round only strengthens an
artifact-evidence assertion for those existing contracts.

## Continuation Point

No continuation is required for this bounded artifact-evidence closeout. Future
compare/select work should start from a new Direction Brief only if it changes
production realization/provider/target behavior or expands route-supported
coverage.
