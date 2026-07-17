# Stage2 RVV computed-masked MAcc add executable artifact ABI boundary

## Goal

Make the existing RVV computed-masked MAcc add selected-body route executable
as a generated RVV artifact with truthful ABI/runtime evidence, or harden the
production artifact seam if inspection or execution shows stale or missing
computed-mask facts, inactive-lane policy, vector operand roles,
addend/accumulator/result roles, dtype/config facts, header/prototype binding,
runtime AVL/VL handling, route-family validation, or target artifact metadata
mirrors.

This task is scoped to one existing Stage 2 route family:

```text
selected tcrv.exec RVV variant
  -> explicit or pre-realized typed tcrv_rvv computed-masked MAcc body facts
  -> RVV provider-owned computed-mask MAcc route facts
  -> MAcc route-family owner and computed-mask accumulation statement-plan owner
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact export
  -> generated bundle compile/run on ssh rvv
```

## What I already know

- The Direction Brief asks for the computed-masked MAcc add executable
  artifact ABI boundary after commit `b73388c2` closed scalar-broadcast MAcc
  add executable evidence.
- The stable RVV authority chain requires typed `tcrv_rvv` body/config/runtime
  facts to feed RVV plugin-owned route construction. Route ids, artifact names,
  metadata mirrors, helper names, test names, exact intrinsic spellings, and
  Common EmitC are not authority.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires production-active
  computed-mask MAcc routes to use the RVV-owned MAcc route-family owner and
  the RVV-owned computed-mask accumulation statement-plan boundary. The central
  provider must attach that plan rather than locally reconstructing
  setvl/compare-load/payload-load/accumulator-load/mask/active-MAcc/merge/store
  from operation names or mirrors.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires computed-mask MAcc
  provider/target validation to consume parameterized provider facts and MAcc
  validation/mirror contracts, including ABI order, predicate/mask source,
  inactive-lane passthrough, accumulator/result layout, header/type summaries,
  operand-binding plan/summary, runtime AVL/VL contract, and statement shape.
- The previous scalar-broadcast MAcc add task found no compiler source change
  was required and closed the remaining blocker with non-dry-run `ssh rvv`
  generated-bundle evidence. This task must re-check computed-mask MAcc
  directly rather than assuming that pattern still holds.
- A prior memory note records evidence-only drift in earlier RVV work. This
  round must verify whether the computed-masked executable path is production
  complete or dry-run-only/under-validated before closing.

## Requirements

- Prove or repair computed-masked MAcc add executable behavior for the base
  explicit and pre-realized selected-body paths.
- Preserve role separation for vector compare operands `cmp_lhs`/`cmp_rhs`,
  payload vector multiplicands/addends `lhs`/`rhs`, accumulator input `acc`,
  output/result `out`, and runtime element count `n`.
- Preserve computed mask semantics: the compare-produced mask, predicate,
  mask type/C type, mask source/form, inactive-lane contract, masked
  passthrough/merge layout, and accumulator preservation must be provider-owned
  facts derived from the typed body/config/runtime structure.
- Preserve typed config facts: dtype, SEW, LMUL, policy, vector type, mask
  type, accumulator/result layout, setvl, runtime AVL/VL, and MAcc operand
  order must be validated or derived by the RVV plugin.
- Keep Common EmitC neutral; do not add RVV semantic inference there.
- If production inspection or execution finds a stale/missing seam, fix the
  production provider/target/script path and add focused fail-closed evidence.
- If production code is already valid, record the no-source-change
  justification and close the remaining executable evidence gap with
  non-dry-run `ssh rvv` generated-bundle evidence.
- Treat LMUL m2 only as a bounded reference. Include it only if it exercises
  the same coherent executable seam in this round; otherwise leave an exact
  continuation point.

## Acceptance Criteria

- [x] Explicit selected-body computed-masked MAcc add generated bundle executes
  on `ssh rvv` and reports correctness for runtime counts including empty,
  small, vector-ish, tail, and multi-chunk cases.
- [x] Pre-realized selected-body computed-masked MAcc add generated bundle
  executes on `ssh rvv` with the same correctness boundary.
- [x] Evidence covers compare-produced mask behavior, active-lane MAcc,
  inactive-lane accumulator/passthrough preservation, output tail
  preservation, vector operand roles, accumulator contribution, and runtime
  AVL/VL behavior.
- [x] Focused fail-closed evidence exists or remains valid for stale or
  missing executable-boundary facts such as direct pre-realized route entry,
  computed-mask family plan mirror, inactive-lane policy, operand binding,
  header/prototype binding, ABI mapping, route metadata mirror, or
  route-family validation contract.
- [x] Focused C++ checks pass:
  `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test`.
- [x] Relevant generated-bundle dry-run lit tests pass for explicit,
  pre-realized, and direct pre-realized fail-closed computed-masked MAcc add.
- [x] Bounded old-authority scan over touched files and added diff lines is
  clean.
- [x] `git diff --check` and `git diff --cached --check` are clean.

## Out of Scope

- No broad MAcc matrix.
- No dtype/LMUL clone batch.
- No runtime-scalar-cmp masked MAcc expansion in this round.
- No additional scalar-broadcast or widening MAcc rework except as reference.
- No product-reduce/dequant/clamp, memory, segment2, reduction, compare/select,
  widening conversion, or unrelated mask route rewrite.
- No high-level Linalg/Vector/StableHLO frontend route.
- No source-front-door positive route.
- No Common EmitC invention of RVV semantics.
- No dashboard/index/report-only closeout.

## Technical Notes

- Task source: Hermes Direction Brief, `Stage2 RVV computed-masked MAcc add
  executable artifact ABI boundary`.
- Prior task reference:
  `.trellis/tasks/archive/2026-06/06-06-06-06-stage2-rvv-scalar-broadcast-macc-add-artifact-abi/`.
- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
- Relevant production files to inspect:
  - `include/TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `lib/Plugin/RVV/RVVMAccSelectedBodyRealizationOwner.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
- Relevant tests and fixtures:
  - `test/Scripts/rvv-generated-bundle-abi-e2e-computed-masked-macc-add-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-computed-masked-macc-add-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-macc-add-fail-closed.test`
  - `test/Target/RVV/explicit-selected-body-artifact-computed-masked-macc-add.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-macc-add.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-macc-add-lmul-m2.mlir`

## Execution Plan

1. Start the Trellis task and curate context files for implementation/check.
2. Inspect the computed-mask MAcc provider, selected-body realization,
   statement-plan, target artifact validation, script, and fixtures against the
   spec contracts.
3. Repair the production seam if any executable ABI fact is stale or missing.
4. Run explicit and pre-realized generated bundle evidence on `ssh rvv`.
5. Run focused C++ checks, relevant lit dry-runs, whitespace checks, and
   bounded old-authority scan.
6. Update this PRD with evidence, finish/archive the task, and commit one
   coherent change.

## Completion Notes

Production source change justification:

- No compiler source change was required for this computed-masked MAcc add
  executable artifact ABI seam. The current production path already starts from
  explicit or pre-realized selected `tcrv_rvv` body facts, realizes
  pre-realized computed-mask MAcc into `setvl -> compare lhs/rhs loads ->
  payload lhs/rhs loads -> accumulator load -> compare mask ->
  tcrv_rvv.masked_macc -> store`, derives provider-owned computed-mask MAcc
  facts from typed config/runtime structure, and lowers through
  `TCRVEmitCLowerableRoute` into neutral Common EmitC materialization.
- The computed-mask MAcc provider facts already include runtime ABI order
  `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n`, `abi|hdr` operand binding summary,
  compare predicate `slt`, mask role/source/form, inactive-lane accumulator
  preservation, masked passthrough layout, accumulator/result layout, required
  headers, C type mapping, runtime AVL/VL facts, target leaf profile, and
  provider-supported mirror.
- The computed-mask accumulation statement-plan owner already consumes the
  verified route-family plan, math operand-binding facts, and route-control
  provider plan before building the provider-ready setvl/load/compare/MAcc/
  merge/store statement sequence.
- Target artifact validation already consumes provider-owned MAcc validation
  and metadata mirror contracts, including route payload, ABI mapping,
  header/type mapping, runtime AVL/VL, compare mask, active MAcc, masked merge,
  output store, and stale mirror rejection.
- The remaining useful work for this round was current-HEAD executable
  evidence for both explicit and pre-realized selected bodies, not a production
  seam change.

Positive executable `ssh rvv` evidence:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --artifact-root /tmp/tcrv-computed-masked-macc-probe \
  --run-id computed-masked-macc-explicit-ssh \
  --overwrite \
  --op-kind computed_masked_macc_add \
  --runtime-count 0 --runtime-count 1 --runtime-count 7 \
  --runtime-count 16 --runtime-count 23 --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=computed_masked_macc_add counts=0,1,7,16,23,257 patterns=0,1
```

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --artifact-root /tmp/tcrv-computed-masked-macc-probe \
  --run-id computed-masked-macc-pr-ssh \
  --overwrite \
  --op-kind computed_masked_macc_add \
  --runtime-count 0 --runtime-count 1 --runtime-count 7 \
  --runtime-count 16 --runtime-count 23 --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=computed_masked_macc_add counts=0,1,7,16,23,257 patterns=0,1
```

Both runs emitted per-case runtime output for counts `0,1,7,16,23,257` and
patterns `0,1`, including active/inactive lane counts, inactive accumulator
preservation, add-only and mul-only distinguishing cases, signed products,
source preservation, and tail preservation. The final success marker was:

```text
tcrv_rvv_generated_bundle_abi_computed_masked_macc_add_ok counts=0,1,7,16,23,257 patterns=0,1
```

Evidence JSON confirmed `dry_run: false`, `ssh_evidence: true`, runtime ABI
order `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n`,
`rvv-route-operand-binding:computed_masked_macc_add.v1`,
`provider_supported_mirror:rvv-computed-mask-macc-add-plan-validated`,
predicate `slt`, inactive-lane contract
`masked-macc-false-lanes-preserve-accumulator`, generated prototype/header
bindings, and runtime AVL/VL loop facts for both explicit and pre-realized
selected bodies.

Focused fail-closed evidence retained:

- `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-macc-add-fail-closed.test`
  rejects `--direct-pre-realized-route-entry` for
  `computed_masked_macc_add`.
- `test/Target/RVV/pre-realized-selected-body-artifact-computed-masked-macc-add.mlir`
  rejects stale MAcc binding, runtime-scalar binding swap, route/provider
  mirrors, ABI order, headers, C type mapping, accumulation plan, predicate,
  mask role/source/form, source memory form, passthrough layout, arithmetic
  kind, accumulator layout, and result layout before target artifact
  acceptance.

Focused checks run:

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- From `build/test`:
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(pre-realized-)?computed-masked-macc-add-dry-run|rvv-generated-bundle-abi-e2e-direct-pre-realized-computed-masked-macc-add-fail-closed|explicit-selected-body-artifact-computed-masked-macc-add|pre-realized-selected-body-artifact-computed-masked-macc-add'`
  passed 6 selected tests.
- Bounded old-authority scan over changed task files found no positive old
  route authority. The only match was the PRD non-goal guardrail
  `No source-front-door positive route`.
- `git diff --check`
- `git diff --cached --check`

## Spec Update Decision

No `.trellis/spec/` update is needed. This round revalidated existing RVV
plugin and EmitC-route contracts: selected typed `tcrv_rvv` body/config/runtime
facts own computed-mask MAcc route support, computed-mask accumulation
statement planning is RVV-owned, Common EmitC remains neutral, target artifact
validation rejects stale mirrors and payload facts, and executable correctness
claims are backed by current `ssh rvv` evidence.

## Current Phase

Finish.
