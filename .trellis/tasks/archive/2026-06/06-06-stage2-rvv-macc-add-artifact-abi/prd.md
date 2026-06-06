# Stage2 RVV MAcc add executable artifact ABI boundary

## Goal

Make the existing plain RVV MAcc add selected-body route executable as a
generated RVV artifact with truthful ABI/runtime evidence, or harden the
production artifact seam if inspection or execution shows stale or missing
operand role, accumulator/result, dtype/config, ABI/header, or runtime AVL/VL
facts.

This task is scoped to the plain MAcc add route first, covering both explicit
selected bodies and pre-realized selected bodies:

```text
selected tcrv.exec RVV variant
  -> typed tcrv_rvv.macc body facts
  -> RVV provider-owned MAcc route facts
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact export
  -> generated bundle compile/run on ssh rvv
```

## What I already know

- The Direction Brief asks for the MAcc add executable artifact ABI boundary as
  the next Stage 2 bottleneck after product-reduce/dequant-clamp f32 evidence.
- The stable RVV authority chain requires provider-derived typed body/config
  and runtime facts. Route IDs, artifact metadata, helper names, test names, and
  Common EmitC must remain mirrors or neutral carriers only.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires RVV plugin-owned
  selected-body realization, route validation, intrinsic/type/ABI mapping, and
  fail-closed diagnostics.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires Common EmitC to
  consume provider-built `TCRVEmitCLowerableRoute` payloads without inventing RVV
  semantics.
- Existing dry-run script tests for explicit and pre-realized MAcc add already
  check runtime ABI order `lhs,rhs,acc,out,n`, provider operand binding
  `macc_add.v1`, `abi|hdr` participation for exported parameters, accumulator
  layout, result layout, generated harness oracle, source preservation, and tail
  preservation.
- Initial dry-run probes passed for explicit and pre-realized plain MAcc add
  when using `/usr/lib/llvm-20/bin/llvm-readobj`.

## Requirements

- Prove or repair plain MAcc add executable behavior for both explicit and
  pre-realized selected bodies.
- Preserve role separation:
  `lhs` and `rhs` are multiplicand inputs, `acc` is accumulator input,
  `out` is result buffer, and `n` is runtime AVL/count.
- Keep dtype/config authority in typed `tcrv_rvv` body/config facts:
  element type, SEW, LMUL, vector C type, setvl, load, vmacc, store, and policy
  are provider-derived facts.
- Keep ABI/header authority provider-owned:
  generated prototype/header parameters must match provider runtime ABI order
  and operand-binding summaries.
- Keep Common EmitC neutral; do not add RVV semantic inference there.
- If production inspection or execution finds a stale/missing seam, fix the
  production provider/target/script path and add focused fail-closed evidence.
- If production code is already valid, record the no-source-change justification
  and close the remaining executable evidence gap with non-dry-run `ssh rvv`
  generated-bundle evidence.

## Acceptance Criteria

- [x] Explicit selected-body plain MAcc add generated bundle executes on
  `ssh rvv` and reports correctness for runtime counts including empty, small,
  full-vector-ish, tail, and multi-chunk cases.
- [x] Pre-realized selected-body plain MAcc add generated bundle executes on
  `ssh rvv` with the same correctness boundary.
- [x] Evidence covers accumulator contribution, multiply contribution,
  source preservation, output tail preservation, and runtime AVL/VL behavior.
- [x] Focused fail-closed evidence exists or remains valid for a stale or
  missing executable-boundary fact such as direct pre-realized route entry,
  operand binding, route metadata mirror, header/prototype binding, ABI mapping,
  or route-family validation contract.
- [x] Focused C++ checks pass:
  `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test`.
- [x] Relevant generated-bundle dry-run lit tests pass for explicit and
  pre-realized MAcc add.
- [x] `git diff --check`, `git diff --cached --check`, and bounded
  old-authority scan over touched files/added diff lines are clean.

## Out of Scope

- No broad MAcc matrix.
- No dtype/LMUL clone batch.
- No scalar-broadcast, computed-mask, runtime-scalar, or widening MAcc expansion
  unless the same plain MAcc seam forces a shared fix.
- No product-reduce/dequant/clamp rework.
- No high-level Linalg/Vector/StableHLO frontend route.
- No source-only positive RVV route.
- No Common EmitC invention of RVV semantics.
- No mass rewrite of memory, segment2, reduction, compare/select, conversion,
  or unrelated mask routes.

## Technical Notes

- Task source: Hermes Direction Brief, `Stage2 RVV MAcc add executable artifact
  ABI boundary`.
- Relevant production files:
  - `include/TianChenRV/Plugin/RVV/RVVEmitCContractionRouteFamilyPlanOwners.h`
  - `lib/Plugin/RVV/RVVContractionSelectedBodyRealizationOwner.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
- Relevant tests and fixtures:
  - `test/Scripts/rvv-generated-bundle-abi-e2e-macc-add-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-macc-add-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-macc-add-fail-closed.test`
  - `test/Target/RVV/explicit-selected-body-artifact-macc-add.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-macc-add.mlir`

## Execution Plan

1. Verify existing provider, route, target artifact, script, and fixture seam
   for plain MAcc add.
2. Run explicit and pre-realized generated bundle evidence on `ssh rvv`.
3. If any seam fails, fix the smallest production owner and add fail-closed
   evidence.
4. Run focused C++ checks, relevant lit dry-runs, whitespace checks, and bounded
   old-authority scan.
5. Update this PRD with evidence, finish/archive the task, and commit one
   coherent change.

## Completion Notes

Production source change justification:

- No compiler source change was required for the plain MAcc add executable
  artifact ABI seam. The existing production path already starts from explicit
  or pre-realized selected `tcrv_rvv.macc` body facts, realizes/imports runtime
  ABI values for pre-realized bodies, builds provider-owned MAcc route facts,
  lowers through `TCRVEmitCLowerableRoute` and Common EmitC, and validates the
  target artifact against provider-derived header, type, ABI, runtime AVL/VL,
  statement, and metadata mirror contracts.
- The dry-run tests and target validator already cover `lhs,rhs,acc,out,n`,
  route operand binding `rvv-route-operand-binding:macc_add.v1`, `abi|hdr`
  header/prototype participation, accumulator/result layouts, `vmacc` operand
  order `acc,lhs,rhs,vl`, source preservation, and tail preservation.
- The remaining blocker was executable hardware evidence, not a missing
  production seam.

Positive executable `ssh rvv` evidence:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --artifact-root /tmp/tcrv-macc-probe \
  --run-id macc-explicit-ssh \
  --overwrite \
  --op-kind macc_add \
  --runtime-count 0 --runtime-count 1 --runtime-count 16 \
  --runtime-count 17 --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=macc_add counts=0,1,16,17,257 patterns=0,1
```

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --artifact-root /tmp/tcrv-macc-probe \
  --run-id macc-pr-ssh \
  --overwrite \
  --op-kind macc_add \
  --runtime-count 0 --runtime-count 1 --runtime-count 16 \
  --runtime-count 17 --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=macc_add counts=0,1,16,17,257 patterns=0,1
```

Both runs emitted per-case runtime output for counts `0,1,16,17,257` and
patterns `0,1`, including:

```text
explicit_accumulator signed_products source_preserved tail_preserved
tcrv_rvv_generated_bundle_abi_macc_add_ok counts=0,1,16,17,257 patterns=0,1
```

Evidence JSON confirmed `ssh_evidence: true`, `status: success`,
`runtime_abi_order: lhs,rhs,acc,out,n`, and `macc_operand_order:
acc,lhs,rhs,vl` for both explicit and pre-realized runs.

Focused checks run so far:

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(pre-realized-)?macc-add-dry-run|rvv-generated-bundle-abi-e2e-direct-pre-realized-macc-add-fail-closed|explicit-selected-body-artifact-macc-add|pre-realized-selected-body-artifact-macc-add'` from `build/test`: 5/5 passed.
- Bounded old-authority token scan over the archived task directory: no
  matches.
- `git diff --check`
- `git diff --cached --check`
