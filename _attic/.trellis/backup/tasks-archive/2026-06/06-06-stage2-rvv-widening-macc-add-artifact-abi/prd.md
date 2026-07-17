# Stage2 RVV widening MAcc add executable artifact ABI boundary

## Goal

Make the existing RVV widening MAcc add selected-body route executable as a
generated RVV artifact with truthful ABI/runtime evidence, or harden the
production artifact seam if inspection or execution shows stale or missing
narrow operand roles, widened product/accumulator/result facts, header/prototype
bindings, runtime AVL/VL handling, or target artifact validation.

This task is scoped to one direct-provider contraction route family:

```text
selected tcrv.exec RVV variant
  -> explicit or pre-realized typed tcrv_rvv widening_macc body facts
  -> RVV provider-owned contraction/MAcc route facts
  -> direct contraction provider plan
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact export
  -> generated bundle compile/run on ssh rvv
```

## What I already know

- The Direction Brief asks for the widening MAcc add executable artifact ABI
  boundary after commit `09d681e0` proved the plain MAcc add executable seam.
- The stable RVV authority chain requires typed `tcrv_rvv` body/config/runtime
  facts to feed RVV provider-owned route construction. Route ids, artifact
  names, metadata mirrors, script names, helper names, and Common EmitC are not
  authority.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires active direct
  contraction routes such as `widening_macc_add` to go through an RVV-owned
  direct contraction route-provider owner before `TCRVEmitCLowerableRoute`
  statements are attached.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires
  `widening_macc_add` validation to consume provider-derived facts for
  `i16mf2` lhs/rhs sources, `i32m1` accumulator/result vectors,
  `signed-i16mf2xi16mf2-plus-i32m1-to-i32m1`, runtime ABI order
  `lhs,rhs,acc,out,n`, route operand binding, headers, C type mapping, runtime
  AVL/VL contract, statement shape, and metadata mirrors.
- The existing production path contains provider facts through
  `getRVVWideningMAccRouteFacts`, pre-realized selected-body validation through
  `validatePreRealizedRVVSelectedWideningMAccBody`, direct contraction
  provider-plan construction before route construction, and target artifact
  validation through `getRVVMAccRouteValidationContract` and
  `getRVVMAccRouteMetadataMirrorContract`.
- Existing dry-run script tests already exercise explicit and pre-realized
  widening MAcc add generated bundles and a direct pre-realized route-entry
  fail-closed case.

## Requirements

- Prove or repair widening MAcc add executable behavior for both explicit and
  pre-realized selected bodies.
- Preserve role separation:
  `lhs` and `rhs` are narrow signed i16 multiplicand inputs, `acc` is the i32
  accumulator input buffer, `out` is the i32 result buffer, and `n` is runtime
  AVL/count.
- Preserve widened compute facts:
  `i16mf2` source loads, `i32m1` accumulator load, signed widening
  multiply-accumulate, and `i32m1` result store must be provider-derived and
  verified.
- Keep ABI/header authority provider-owned:
  generated prototype/header parameters must match provider runtime ABI order
  and operand-binding summaries.
- Keep Common EmitC neutral; do not add RVV semantic inference there.
- If production inspection or execution finds a stale/missing seam, fix the
  production provider/target/script path and add focused fail-closed evidence.
- If production code is already valid, record the no-source-change
  justification and close the remaining executable evidence gap with non-dry-run
  `ssh rvv` generated-bundle evidence.

## Acceptance Criteria

- [x] Explicit selected-body widening MAcc add generated bundle executes on
  `ssh rvv` and reports correctness for runtime counts including empty, small,
  vector-ish, tail, and multi-chunk cases.
- [x] Pre-realized selected-body widening MAcc add generated bundle executes on
  `ssh rvv` with the same correctness boundary.
- [x] Evidence covers signed widening products, nonzero accumulator
  contribution, source/accumulator preservation, output tail preservation, and
  runtime AVL/VL behavior.
- [x] Focused fail-closed evidence exists or remains valid for stale or missing
  executable-boundary facts such as direct pre-realized route entry, operand
  binding, route metadata mirror, header/prototype binding, ABI mapping, or
  route-family validation contract.
- [x] Focused C++ checks pass:
  `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test`.
- [x] Relevant generated-bundle dry-run lit tests pass for explicit and
  pre-realized widening MAcc add.
- [x] `git diff --check`, `git diff --cached --check`, and bounded
  old-authority scan over touched files/added diff lines are clean.

## Out of Scope

- No broad MAcc matrix.
- No dtype/LMUL clone batch.
- No scalar-broadcast, computed-mask, or runtime-scalar MAcc expansion.
- No product-reduce/dequant/clamp rework except as reference.
- No high-level Linalg/Vector/StableHLO frontend route.
- No source-only positive RVV route.
- No Common EmitC invention of RVV semantics.
- No mass rewrite of memory, segment2, reduction, compare/select, widening
  conversion, or unrelated mask routes.

## Technical Notes

- Task source: Hermes Direction Brief, `Stage2 RVV widening MAcc add executable
  artifact ABI boundary`.
- Prior task reference:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-macc-add-artifact-abi/`.
- Relevant specs:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
- Relevant production files inspected:
  - `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`
  - `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
- Relevant tests and fixtures:
  - `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-widening-macc-add-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-macc-add-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-widening-macc-add-fail-closed.test`
  - `test/Target/RVV/explicit-selected-body-artifact-widening-macc-add.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-widening-macc-add.mlir`

## Execution Plan

1. Verify existing provider, direct contraction owner, target artifact,
   script, and fixture seam for widening MAcc add.
2. Run explicit and pre-realized generated bundle evidence on `ssh rvv`.
3. If any seam fails, fix the smallest production owner and add fail-closed
   evidence.
4. Run focused C++ checks, relevant lit dry-runs, whitespace checks, and
   bounded old-authority scan.
5. Update this PRD with evidence, finish/archive the task, and commit one
   coherent change.

## Completion Notes

Production source change justification:

- No compiler source change was required for the widening MAcc add executable
  artifact ABI seam. The existing production path already starts from explicit
  or pre-realized selected `tcrv_rvv.widening_macc` body facts, validates
  pre-realized narrow source and widened accumulator/result facts, builds
  provider-owned contraction and MAcc route contracts, obtains the direct
  contraction provider plan before route construction, lowers through
  `TCRVEmitCLowerableRoute` and Common EmitC, and validates the target artifact
  against provider-derived header, type, ABI, runtime AVL/VL, statement, and
  metadata mirror contracts.
- The remaining blocker was executable hardware evidence for widening MAcc add,
  not a missing production seam.

Positive executable `ssh rvv` evidence:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --artifact-root /tmp/tcrv-widening-macc-probe \
  --run-id widening-macc-explicit-ssh \
  --overwrite \
  --op-kind widening_macc_add \
  --runtime-count 0 --runtime-count 1 --runtime-count 7 \
  --runtime-count 16 --runtime-count 23 --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=widening_macc_add counts=0,1,7,16,23,257 patterns=0,1
```

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --artifact-root /tmp/tcrv-widening-macc-probe \
  --run-id widening-macc-pr-ssh \
  --overwrite \
  --op-kind widening_macc_add \
  --runtime-count 0 --runtime-count 1 --runtime-count 7 \
  --runtime-count 16 --runtime-count 23 --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=widening_macc_add counts=0,1,7,16,23,257 patterns=0,1
```

Both runs emitted per-case runtime output for counts `0,1,7,16,23,257` and
patterns `0,1`, including:

```text
signed_widening_products accumulation source_preserved accumulator_preserved tail_preserved
tcrv_rvv_generated_bundle_abi_widening_macc_add_ok counts=0,1,7,16,23,257 patterns=0,1
```

Evidence JSON confirmed `ssh_evidence: true`, `dry_run: false`,
`status: success`, runtime ABI order `lhs,rhs,acc,out,n`,
`signed-i16mf2xi16mf2-plus-i32m1-to-i32m1`, and MAcc operand order
`accumulator_vector,lhs_i16_vector,rhs_i16_vector,vl` for the explicit and
pre-realized runs.

Focused checks run:

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(explicit-|pre-realized-)?widening-macc-add-dry-run|rvv-generated-bundle-abi-e2e-direct-pre-realized-widening-macc-add-fail-closed|explicit-selected-body-artifact-widening-macc-add|pre-realized-selected-body-artifact-widening-macc-add'` from `build/test`: 5/5 passed.
- Bounded old-authority scan over production source diff: no production source
  diff in this task.
- `git diff --check`
- `git diff --cached --check`
