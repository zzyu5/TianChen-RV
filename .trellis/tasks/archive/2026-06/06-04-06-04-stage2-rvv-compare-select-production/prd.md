# Stage2 RVV compare/select production route and fail-closed legality closure

## Goal

Close a bounded production validation gap in the Stage2 RVV compare/select
target-artifact path. A selected typed `tcrv_rvv` compare/select route must be
accepted only when the provider-built route description and candidate artifact
metadata carry compare/select-owned facts. Stale non-compare/select
route-family provider facts or candidate mirrors must fail closed before target
artifact acceptance.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV compare/select production route and fail-closed legality closure`

## Initial Repository State

* Repository root: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk pwd`: `/home/kingdom/phdworks/TianchenRV`.
* Initial `rtk git status --short`: clean; `rtk` reported `ok`.
* Initial `rtk git log --oneline -8` started at
  `4495b003 rvv: assert compare select runtime avl evidence`.
* `.trellis/.current-task` was absent, so this task was created from the
  supplied Direction Brief before source edits.
* This round is run by one serial Codex worker; no subagents or multi-agent
  workflow are used.

## What I Already Know

* `.trellis/spec/index.md` defines the RVV authority chain as selected
  `tcrv.exec` envelope -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/provider -> `TCRVEmitCLowerableRoute` ->
  common EmitC -> target artifact.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires compare/select
  selected-body realization and provider preflight to consume realized typed
  body/config/runtime facts, not route ids, artifact names, ABI strings,
  descriptors, source-front-door metadata, or common EmitC inference.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires compare/select
  target validation to consume `RVVCompareSelectRouteValidationContract` and
  its embedded `RVVRuntimeAVLVLSelectedBoundaryContract` before accepting
  predicate, select, mask/tail, statement-plan, ABI/type/header, or metadata
  mirrors.
* Live inspection shows the compare/select target validator already consumes
  `RVVCompareSelectRouteValidationContract`, validates runtime AVL/VL facts,
  and checks route-local runtime metadata as mirrors.
* Live inspection also shows a bounded production gap: unlike conversion,
  vector reduction, memory, and other mature validators, compare/select
  provider validation does not explicitly reject stale non-compare/select
  route-family facts such as base-memory, segment2, MAcc, reduction, widening
  conversion, contraction, or runtime-scalar splat route-family facts. Its
  candidate metadata mirror contract also lacks a stale mirror key list for
  those unrelated route-family mirrors.

## Requirements

* Touch active compare/select production code, not only evidence or task files.
* Keep compare/select route support derived from provider-owned typed
  `tcrv_rvv` body/config/runtime facts through
  `RVVCompareSelectRouteValidationContract`.
* Reject stale non-compare/select provider facts in the rebuilt route
  description before target artifact acceptance.
* Reject stale non-compare/select candidate metadata mirrors through the
  compare/select metadata mirror contract.
* Preserve `runtimeControlPlanID` and `runtimeABIOrder` as route-local mirrors
  after the embedded runtime AVL/VL selected-boundary contract.
* Keep common EmitC/export neutral; do not move compare/select semantics into
  common EmitC, target metadata, artifact names, route ids, descriptors, or C
  strings.
* Do not add new compare/select predicate coverage, dtype/LMUL clone batches,
  source-front-door positive routes, high-level frontend lowering, or runtime
  behavior changes.

## Acceptance Criteria

* [ ] `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` rejects stale
      non-compare/select route-family provider facts for compare/select
      producers before route payload/artifact acceptance.
* [ ] `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` adds compare/select
      stale mirror keys for unrelated route-family candidate metadata while
      keeping runtime-control and runtime-ABI metadata labeled as route-local
      mirrors.
* [ ] `test/Target/TargetArtifactExportTest.cpp` includes focused C++ negative
      coverage for stale unrelated provider facts and stale unrelated candidate
      mirrors on compare/select routes.
* [ ] Existing positive compare/select target-artifact fixture validation still
      passes for plain, runtime-scalar, and runtime-scalar dual compare/select.
* [ ] No emitted C/C++, runtime ABI order/counts, statement ordering, runtime
      behavior, correctness, or performance behavior changes are introduced.
* [ ] Run `tianchenrv-target-artifact-export-test` because target validation
      and target C++ tests change.
* [ ] Run `tianchenrv-rvv-extension-plugin-test` only if provider/planning
      changes require it; otherwise record the rationale.
* [ ] Run a focused compare/select generated-bundle lit/export check if the
      route metadata contract changes observable artifact behavior.
* [ ] Bounded old-authority scan over touched production/test files finds no
      new positive dependency on `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_`,
      `!tcrv_rvv.i32m*`, descriptor/direct-C/source-export/source-front-door,
      route-id/artifact-name authority, common EmitC semantic inference,
      exact `__riscv_*_i32m1`, or mirror-only authority.
* [ ] `rtk git diff --check` passes, Trellis task state is truthful, and the
      worktree is clean after commit if the task completes.

## Technical Approach

1. Add a compare/select-specific stale route-family provider-fact rejection in
   target validation, mirroring the existing pattern used by conversion and
   reduction validators.
2. Extend the compare/select metadata mirror contract with stale unrelated
   route-family mirror keys, keeping compare/select-owned keys in the normal
   mirror list.
3. Add C++ negative tests that mutate the manual compare/select fixture:
   * provider description carries a stale base-memory or segment2/MAcc-style
     route-family fact;
   * candidate metadata carries a stale unrelated route-family mirror.
4. Run the focused target artifact C++ test and a compare/select target/lit
   check if metadata output changes.

## Evidence Plan

* `rtk ./build/bin/tianchenrv-target-artifact-export-test`
* If provider/planning changes affect the RVV plugin C++ test surface:
  `rtk ./build/bin/tianchenrv-rvv-extension-plugin-test`
* Focused lit/export check from `build/test` for compare/select selected-body
  artifacts if generated metadata behavior changes.
* Bounded old-authority scan over touched files.
* `rtk git diff --check`
* `rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-06-04-stage2-rvv-compare-select-production`

## SSH RVV Rationale

Do not run `ssh rvv` unless the final diff changes emitted C/C++, runtime ABI
order, runtime counts, statement ordering, runtime behavior, correctness, or
performance behavior. This task is target validation and candidate metadata
contract hardening only; it does not make a new runtime correctness or
performance claim.

## Out Of Scope

* New RVV compare/select operations, new predicates, dtype/LMUL clone batches,
  new route families, source-front-door routes, descriptor/direct-C/source
  export, high-level frontend lowering, dashboards, broad smoke matrices, or
  unrelated reduction/conversion/memory/contraction expansion.
* Moving compare/select predicate, select, mask/tail, runtime-control, ABI,
  header/type, or route-support semantics into common EmitC/export or artifact
  metadata.

## Definition Of Done

The compare/select target-artifact production path fails closed on stale
unrelated provider facts and candidate mirrors, focused C++ validation passes,
the task records no-runtime-change rationale, and a coherent commit is created
if all checks pass.

## Implementation Results

* Hardened `validateRVVCompareSelectRouteValidationContract` so compare/select
  target validation now rejects stale non-compare/select route-family provider
  facts in the rebuilt route description, including elementwise,
  runtime-scalar splat-store, widening conversion, computed-mask memory, MAcc,
  standalone reduction, contraction, base-memory, segment2, and widening
  relation residue.
* Extended the provider-owned compare/select metadata mirror contract with
  explicit stale unrelated route-family mirror keys while preserving
  `tcrv_rvv.runtime_control_plan` and `tcrv_rvv.runtime_abi_order` as
  route-local runtime AVL/VL mirrors.
* Added focused target C++ coverage proving:
  * compare/select mirror contracts carry stale unrelated mirror keys;
  * stale computed-mask memory/base-memory provider facts fail closed on a
    plain compare/select route;
  * stale base-memory and segment2 candidate mirrors fail closed before
    artifact acceptance.
* Existing positive target-artifact validation remains intact for plain,
  runtime-scalar, and runtime-scalar dual compare/select.
* No common EmitC/export code changed. No emitted C/C++, runtime ABI
  order/counts, statement ordering, runtime behavior, correctness behavior, or
  performance behavior changed.

## Evidence Results

### Build

```text
rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 8
```

Result: build completed successfully. Existing warnings were emitted from
pre-existing switch/unused-function sites; no new build failure.

### C++ Tests

```text
rtk ./build/bin/tianchenrv-target-artifact-export-test
```

Result: passed with no output.

```text
rtk ./build/bin/tianchenrv-rvv-extension-plugin-test
```

Result:

```text
RVV extension plugin smoke test passed
```

### Focused Lit

Run from `build/test`:

```text
rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'pre-realized-selected-body-artifact-(cmp-select|computed-mask-select|runtime-scalar-cmp-select|runtime-scalar-dual-cmp-mask-and-select)|explicit-selected-body-artifact-(cmp-select|computed-mask-select|runtime-scalar-cmp-select|runtime-scalar-dual-cmp-mask-and-select)|pre-realized-(typed-)?(cmp-select|computed-mask-select|runtime-scalar-cmp-select|runtime-scalar-dual-cmp-mask-and-select)-dry-run'
```

Result:

```text
Total Discovered Tests: 477
  Excluded: 449 (94.13%)
  Passed  :  28 (5.87%)
```

### Static Checks

```text
rtk git diff --check
```

Result: passed.

Added-line old-authority scan over touched production/test files:

```text
rtk proxy bash -lc 'matches=$(git diff -U0 -- include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp test/Target/TargetArtifactExportTest.cpp | rg -n "^\+.*(RVVI32M1|rvv-i32m1|tcrv_rvv\.i32_|!tcrv_rvv\.i32m|descriptor|direct-C|source-export|source-front-door|__riscv_.*_i32m1|route-id|artifact-name|common EmitC semantic|mirror-only authority)"); status=$?; if [ $status -eq 0 ]; then printf "%s\n" "$matches"; exit 1; fi; exit 0'
```

Result: passed with no matches.

### Trellis Validation

```text
rtk python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-04-06-04-stage2-rvv-compare-select-production
```

Result: passed; both `implement.jsonl` and `check.jsonl` have four valid
entries.

## Spec Update Judgment

No `.trellis/spec/` update is needed. The relevant executable contract already
exists in `.trellis/spec/lowering-runtime/emitc-route.md`: compare/select
target artifact validation must reject stale computed-mask memory,
base-memory, segment2, or unrelated route-family mirrors before artifact
acceptance. This round implements that existing contract for provider facts and
candidate mirrors; it does not introduce a new API, route family, or
architecture rule.
