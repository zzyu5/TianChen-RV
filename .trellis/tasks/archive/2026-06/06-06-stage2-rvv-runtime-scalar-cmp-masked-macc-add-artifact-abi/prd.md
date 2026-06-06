# Stage2 RVV runtime-scalar-cmp masked MAcc add executable artifact ABI boundary

## Goal

Make the existing RVV runtime-scalar-cmp masked MAcc add selected-body route
executable as generated RVV artifacts with truthful ABI/runtime evidence, or
harden the production artifact seam if inspection or execution shows stale or
missing runtime scalar binding, compare operand role, computed mask facts,
inactive-lane policy, vector multiplicand/addend/accumulator roles,
dtype/config facts, header/prototype binding, runtime AVL/VL handling,
route-family validation, or target artifact metadata mirrors.

This task is scoped to one existing Stage 2 route family:

```text
selected tcrv.exec RVV variant
  -> explicit or pre-realized typed tcrv_rvv runtime-scalar-cmp masked MAcc body facts
  -> RVV plugin-owned runtime scalar compare, computed mask, and MAcc route facts
  -> MAcc route-family owner and computed-mask accumulation statement-plan owner
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact export
  -> generated bundle compile/run on ssh rvv
```

## What I already know

- The Direction Brief asks for the runtime-scalar-cmp masked MAcc add executable
  artifact ABI boundary immediately after commit `6e60ef24` proved the base
  vector-compare computed-masked MAcc add executable artifact boundary.
- The stable RVV authority chain requires typed `tcrv_rvv` body/config/runtime
  facts to feed RVV plugin-owned route construction. Route ids, artifact names,
  metadata mirrors, helper names, test names, exact intrinsic spellings, and
  Common EmitC are not authority.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires production-active
  computed-mask MAcc routes, including `runtime_scalar_cmp_masked_macc_add`, to
  use the RVV-owned MAcc route-family owner and the RVV-owned computed-mask
  accumulation statement-plan boundary. The provider must attach that plan
  rather than locally reconstructing setvl/load/splat/compare/MAcc/merge/store
  from operation names or mirrors.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires provider-built
  `TCRVEmitCLowerableRoute` values and provider-owned operand-binding summaries
  with `abi` and `hdr` participation for exported runtime ABI parameters.
- `.trellis/spec/testing/mlir-testing-contract.md` requires runtime-scalar
  computed-mask MAcc evidence to check `rhs_scalar` as the scalar ABI input,
  scalar splat source, and compare RHS for the same-VL mask producer. Runtime
  correctness claims require real `ssh rvv` compile/run output.
- The previous archived computed-mask MAcc task found no compiler source change
  was required for vector-compare computed-mask MAcc and closed the remaining
  blocker with non-dry-run `ssh rvv` generated-bundle evidence. This task must
  re-check the runtime-scalar-cmp path directly because its scalar ABI/header
  boundary is distinct.
- Prior memory records that RVV work should not drift into evidence-only loops
  when the production compiler surface is actually incomplete. This round must
  either repair production runtime-scalar-cmp artifact/ABI behavior or give a
  precise no-source-change justification supported by current evidence.

## Requirements

- Prove or repair runtime-scalar-cmp masked MAcc add executable behavior for
  the base explicit and pre-realized selected-body paths.
- Preserve role separation for runtime scalar compare input `rhs_scalar`,
  payload vector multiplicands/addends `lhs`/`rhs`, accumulator input `acc`,
  output/result `out`, and runtime element count `n`.
- Preserve runtime-scalar computed mask semantics: `rhs_scalar` must be the
  provider-derived runtime ABI value, scalar splat source, and compare RHS for
  the same-VL mask producer. It must not be inferred from a constant, ABI name,
  route id, artifact name, script option, or harness convention.
- Preserve active/inactive lane semantics: active lanes compute
  `out[i] = acc[i] + lhs[i] * rhs[i]`; inactive lanes preserve the
  accumulator/pass-through; output tail sentinels remain untouched.
- Preserve typed config facts: dtype, SEW, LMUL, tail/mask policy, vector type,
  mask type, accumulator/result layout, setvl, runtime AVL/VL, and MAcc operand
  order must be validated or derived by the RVV plugin.
- Keep Common EmitC neutral; do not add RVV semantic inference there.
- If production inspection or execution finds a stale/missing seam, fix the
  production provider/target/script path and add focused fail-closed evidence.
- If production code is already valid, record the no-source-change
  justification and close the remaining executable evidence gap with
  non-dry-run `ssh rvv` generated-bundle evidence.
- Treat LMUL m2 runtime-scalar-cmp files only as bounded references. Include
  LMUL m2 only if the base seam is already complete and the same coherent
  executable boundary fits this round.

## Acceptance Criteria

- [x] Explicit selected-body runtime-scalar-cmp masked MAcc add generated
  bundle executes on `ssh rvv` and reports correctness for runtime counts
  including empty, small, vector-ish, tail, and multi-chunk cases.
- [x] Pre-realized selected-body runtime-scalar-cmp masked MAcc add generated
  bundle executes on `ssh rvv` with the same correctness boundary.
- [x] Evidence covers at least two runtime scalar values and at least two data
  patterns, proving scalar/splat compare behavior, active-lane MAcc,
  inactive-lane accumulator preservation, output tail preservation, vector
  operand roles, accumulator contribution, and runtime AVL/VL behavior.
- [x] Focused fail-closed evidence exists or is added for at least one stale or
  missing executable-boundary fact such as direct pre-realized route entry,
  stale `rhs_scalar` binding, stale computed-mask family plan, stale inactive
  lane policy, stale operand binding, stale header/prototype binding, ABI
  mapping, route metadata mirror, or route-family validation contract.
- [x] Focused C++ checks pass:
  `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test`.
- [x] Relevant generated-bundle dry-run lit tests pass for explicit,
  pre-realized, and direct pre-realized fail-closed runtime-scalar-cmp masked
  MAcc add.
- [x] Bounded old-authority scan over touched files and added diff lines is
  clean.
- [x] `git diff --check` and `git diff --cached --check` are clean.

## Definition of Done

- Production code is changed only if inspection or focused evidence shows a
  runtime-scalar-cmp artifact/ABI seam defect.
- Runtime correctness is claimed only with real `ssh rvv` evidence.
- Trellis task status, PRD evidence notes, and workspace journal are updated.
- The completed work is archived and committed as one coherent change.

## Out of Scope

- No broad MAcc matrix.
- No dtype/LMUL clone batch.
- No LMUL m2 expansion unless it is the same coherent seam after base closure.
- No additional computed-mask-only, scalar-broadcast, widening MAcc,
  product-reduce/dequant/clamp, memory, segment2, reduction, compare/select,
  widening conversion, or unrelated mask route rewrite except as reference.
- No high-level Linalg/Vector/StableHLO frontend route.
- No source-front-door positive route.
- No Common EmitC invention of RVV semantics.
- No dashboard/index/report-only closeout.

## Technical Notes

- Task source: Hermes Direction Brief,
  `Stage2 RVV runtime-scalar-cmp masked MAcc add executable artifact ABI boundary`.
- Prior task reference:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-computed-masked-macc-add-artifact-abi/`.
- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/mlir-testing-contract.md`
- Relevant production files to inspect:
  - `include/TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h`
  - `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
  - `lib/Plugin/RVV/RVVMAccSelectedBodyRealizationOwner.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCCompareSelectStatementPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCStatementPlanOwners.cpp`
  - `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
  - `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
  - `scripts/rvv_generated_bundle_abi_e2e.py`
- Relevant tests and fixtures:
  - `test/Scripts/rvv-generated-bundle-abi-e2e-runtime-scalar-cmp-masked-macc-add-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-runtime-scalar-cmp-masked-macc-add-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-macc-add-fail-closed.test`
  - `test/Target/RVV/explicit-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir`
  - `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir`

## Execution Plan

1. Start the Trellis task and curate context files for implementation/check.
2. Inspect runtime-scalar-cmp MAcc provider, selected-body realization,
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

- No compiler source change was required for this runtime-scalar-cmp masked
  MAcc add executable artifact ABI seam. The current production path already
  starts from explicit or pre-realized selected `tcrv_rvv` body facts, validates
  the runtime scalar compare input as `rhs-scalar-value`, realizes
  pre-realized selected bodies before route construction, derives
  provider-owned runtime-scalar computed-mask MAcc facts, and lowers through a
  provider-built `TCRVEmitCLowerableRoute` into neutral Common EmitC
  materialization.
- The pre-realized validator requires
  `op_kind = runtime_scalar_cmp_masked_macc_add`, `predicate_kind = sle`,
  memory form `runtime-scalar-computed-mask-unit-stride-macc`, mask role/source,
  accumulator/result layouts, SEW32 LMUL m1/m2, TA/MA policy, and ABI roles for
  `cmp_lhs`, `rhs_scalar`, `lhs`, `rhs`, `acc`, `out`, and `n`.
- The provider facts already include runtime ABI order
  `cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n`,
  `rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1`,
  `rhs_scalar=rhs-scalar-value:rhs_scalar:abi|splat|cmp-rhs|hdr`, predicate
  `sle`, runtime-scalar mask producer source
  `runtime-scalar-splat-compare-rhs`, inactive-lane contract
  `masked-macc-false-lanes-preserve-accumulator`, required headers, C type
  mapping, runtime AVL/VL facts, and
  `provider_supported_mirror:rvv-runtime-scalar-cmp-masked-macc-add-plan-validated`.
- The computed-mask accumulation statement-plan owner consumes the verified
  computed-mask accumulation family plan, math operand-binding facts, and
  route-control provider plan before building the setvl/load/splat/compare/
  MAcc/merge/store sequence. Generated C++ confirms `rhs_scalar` is splatted
  with `__riscv_vmv_v_x_i32m1`, compared with `__riscv_vmsle_vv_i32m1_b32`,
  MAcc runs through `__riscv_vmacc_vv_i32m1`, and inactive lanes are preserved
  through `__riscv_vmerge_vvm_i32m1`.
- Target artifact validation already consumes the provider-owned MAcc
  validation and metadata mirror contracts, including ABI mapping order,
  header/type mapping, runtime AVL/VL, RHS scalar splat, compare mask, active
  MAcc, masked merge, output store, stale mirror rejection, and provider
  support mirrors.

Positive executable `ssh rvv` evidence:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --artifact-root /tmp/tcrv-runtime-scalar-cmp-masked-macc-probe \
  --run-id runtime-scalar-cmp-masked-macc-explicit-ssh \
  --overwrite \
  --op-kind runtime_scalar_cmp_masked_macc_add \
  --runtime-count 0 --runtime-count 1 --runtime-count 7 \
  --runtime-count 16 --runtime-count 23 --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=runtime_scalar_cmp_masked_macc_add counts=0,1,7,16,23,257 rhs_scalars=-37,91 patterns=0,1
```

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --artifact-root /tmp/tcrv-runtime-scalar-cmp-masked-macc-probe \
  --run-id runtime-scalar-cmp-masked-macc-pr-ssh \
  --overwrite \
  --op-kind runtime_scalar_cmp_masked_macc_add \
  --runtime-count 0 --runtime-count 1 --runtime-count 7 \
  --runtime-count 16 --runtime-count 23 --runtime-count 257 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=runtime_scalar_cmp_masked_macc_add counts=0,1,7,16,23,257 rhs_scalars=-37,91 patterns=0,1
```

Both runs emitted per-case runtime output for counts `0,1,7,16,23,257`,
runtime scalar thresholds `-37,91`, and patterns `0,1`. Multi-lane cases
reported mixed active/inactive lanes, inactive accumulator preservation,
add-only and mul-only distinguishing lanes, signed products, and tail
preservation. The final remote success marker was:

```text
tcrv_rvv_generated_bundle_abi_runtime_scalar_cmp_masked_macc_add_ok counts=0,1,7,16,23,257 rhs_scalars=-37,91 patterns=0,1
```

Evidence JSON confirmed `dry_run: false`, `ssh_evidence: true`, runtime ABI
order `cmp_lhs,rhs_scalar,lhs,rhs,acc,out,n`, `rhs_scalar` as the
`rhs-scalar-value` runtime ABI parameter and `runtime-scalar-splat-compare-rhs`
compare source, route operand binding plan
`rvv-route-operand-binding:runtime_scalar_cmp_masked_macc_add.v1`, predicate
`sle`, inactive-lane contract
`masked-macc-false-lanes-preserve-accumulator`, provider-supported mirror,
generated prototype/header binding, and runtime AVL/VL loop facts.

Focused fail-closed evidence retained:

- `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-macc-add-fail-closed.test`
  rejects `--direct-pre-realized-route-entry` for
  `runtime_scalar_cmp_masked_macc_add`.
- `test/Target/RVV/pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add.mlir`
  rejects stale provider mirror, stale route operand binding plan, stale
  `rhs_scalar` binding summary, and stale runtime ABI order before target
  artifact acceptance.

Focused checks run:

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- From `build/test`:
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(pre-realized-)?runtime-scalar-cmp-masked-macc-add-dry-run|rvv-generated-bundle-abi-e2e-direct-pre-realized-runtime-scalar-cmp-masked-macc-add-fail-closed|explicit-selected-body-artifact-runtime-scalar-cmp-masked-macc-add|pre-realized-selected-body-artifact-runtime-scalar-cmp-masked-macc-add'`
  passed 6 selected tests.
- Bounded old-authority scan over changed task files found no positive legacy
  route authority. The only source-front-door mention is the PRD out-of-scope
  guardrail.
- `git diff --check`
- `git diff --cached --check`

## Spec Update Decision

No `.trellis/spec/` update is needed. This round revalidated existing RVV
plugin and EmitC-route contracts: selected typed `tcrv_rvv` body/config/runtime
facts own runtime-scalar computed-mask MAcc route support, computed-mask
accumulation statement planning is RVV-owned, Common EmitC remains neutral,
target artifact validation rejects stale mirrors and payload facts, and
executable correctness claims are backed by current `ssh rvv` evidence.

## Current Phase

Finish.
