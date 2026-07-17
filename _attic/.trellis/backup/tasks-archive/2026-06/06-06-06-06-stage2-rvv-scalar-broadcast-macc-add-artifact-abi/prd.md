# Stage2 RVV scalar-broadcast MAcc add executable artifact ABI boundary

## Goal

Make the existing RVV scalar-broadcast MAcc add selected-body route executable
as a generated RVV artifact with truthful ABI/runtime evidence, or harden the
production artifact seam if inspection or execution shows stale or missing
scalar operand role, broadcast/splat facts, vector multiplicand/addend/
accumulator/result facts, dtype/config, header/prototype bindings, runtime
AVL/VL handling, route-family validation, or target artifact metadata mirrors.

This task is scoped to one existing Stage 2 route family:

```text
selected tcrv.exec RVV variant
  -> explicit or pre-realized typed tcrv_rvv scalar-broadcast MAcc body facts
  -> RVV provider-owned MAcc/scalar-broadcast route facts
  -> MAcc route-family owner and statement-plan owner
  -> TCRVEmitCLowerableRoute
  -> common EmitC materialization
  -> target artifact export
  -> generated bundle compile/run on ssh rvv
```

## What I already know

- The Direction Brief asks for the scalar-broadcast MAcc add executable
  artifact ABI boundary after commit `92e0fe7b` closed widening MAcc add
  executable evidence.
- The stable RVV authority chain requires typed `tcrv_rvv` body/config/runtime
  facts to feed RVV plugin-owned route construction. Route ids, artifact names,
  metadata mirrors, helper names, test names, exact intrinsic spellings, and
  Common EmitC are not authority.
- `.trellis/spec/extension-plugins/rvv-plugin.md` requires scalar-broadcast
  MAcc ownership to be selected only for `scalar_broadcast_macc_add`, with a
  validated scalar-broadcast MAcc route-family plan, scalar splat leaf,
  route-control provider plan, math operand-binding facts, and RVV-owned
  statement plan before route construction.
- `.trellis/spec/lowering-runtime/emitc-route.md` requires MAcc route
  validation and metadata mirror contracts to consume provider-derived facts,
  including scalar-broadcast RHS scalar ABI role, splat leaf, accumulator/
  result layout, header/type summaries, operand-binding plan/summary, runtime
  AVL/VL contract, statement shape, and cross-family fail-closed checks.
- The previous widening MAcc add task found no compiler source change was
  required and closed the remaining blocker with non-dry-run `ssh rvv`
  generated-bundle evidence. This task must re-check scalar-broadcast MAcc
  directly rather than assuming that pattern still holds.

## Requirements

- Prove or repair scalar-broadcast MAcc add executable behavior for both
  explicit and pre-realized selected bodies.
- Preserve role separation: vector multiplicand/input role, scalar RHS role,
  accumulator input role, output/result role, and runtime element-count role
  must stay provider-derived and visible through route facts, operand binding,
  header/prototype binding, target metadata, and generated bundle ABI.
- Preserve scalar broadcast semantics: the RHS scalar ABI value must be
  imported as a scalar operand and splatted/broadcast before MAcc; it must not
  be treated as a vector input or inferred from ABI names, route ids, artifact
  names, or test names.
- Preserve typed config facts: dtype, SEW, LMUL, policy, vector type, scalar
  C type, accumulator/result layout, setvl, runtime AVL/VL, and MAcc operand
  order must be validated or derived by the RVV plugin.
- Keep Common EmitC neutral; do not add RVV semantic inference there.
- If production inspection or execution finds a stale/missing seam, fix the
  production provider/target/script path and add focused fail-closed evidence.
- If production code is already valid, record the no-source-change
  justification and close the remaining executable evidence gap with non-dry-run
  `ssh rvv` generated-bundle evidence.

## Acceptance Criteria

- [x] Explicit selected-body scalar-broadcast MAcc add generated bundle
  executes on `ssh rvv` and reports correctness for runtime counts including
  empty, small, vector-ish, tail, and multi-chunk cases.
- [x] Pre-realized selected-body scalar-broadcast MAcc add generated bundle
  executes on `ssh rvv` with the same correctness boundary.
- [x] Evidence covers scalar RHS broadcast/splat, vector multiplicand/source
  preservation, accumulator contribution/preservation, result/tail
  preservation, and runtime AVL/VL behavior.
- [x] Focused fail-closed evidence exists or remains valid for stale or missing
  executable-boundary facts such as direct pre-realized route entry,
  scalar-broadcast family plan mirror, scalar splat fact, operand binding,
  header/prototype binding, ABI mapping, route metadata mirror, or
  route-family validation contract.
- [x] Focused C++ checks pass:
  `build/bin/tianchenrv-rvv-extension-plugin-test` and
  `build/bin/tianchenrv-target-artifact-export-test`.
- [x] Relevant generated-bundle dry-run lit tests pass for explicit,
  pre-realized, and direct pre-realized fail-closed scalar-broadcast MAcc add.
- [x] Bounded old-authority scan over touched files and added diff lines is
  clean.
- [x] `git diff --check` and `git diff --cached --check` are clean.

## Out of Scope

- No broad MAcc matrix.
- No dtype/LMUL clone batch.
- No computed-mask or runtime-scalar computed-mask MAcc expansion.
- No additional widening MAcc, product-reduce/dequant/clamp, memory, segment2,
  reduction, compare/select, widening conversion, or unrelated mask rework
  except as reference.
- No high-level Linalg/Vector/StableHLO frontend route.
- No source-front-door positive route.
- No Common EmitC invention of RVV semantics.
- No dashboard/index/report-only closeout.

## Technical Notes

- Task source: Hermes Direction Brief, `Stage2 RVV scalar-broadcast MAcc add
  executable artifact ABI boundary`.
- Prior task reference:
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-widening-macc-add-artifact-abi/`.
- Relevant specs read:
  - `.trellis/spec/index.md`
  - `.trellis/spec/extension-plugins/index.md`
  - `.trellis/spec/extension-plugins/rvv-plugin.md`
  - `.trellis/spec/lowering-runtime/emitc-route.md`
  - `.trellis/spec/testing/index.md`
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
  - `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-scalar-broadcast-macc-add-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-scalar-broadcast-macc-add-dry-run.test`
  - `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-scalar-broadcast-macc-add-fail-closed.test`
  - `test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-macc-add.mlir`
  - `test/Target/RVV/explicit-selected-body-artifact-scalar-broadcast-macc-add.mlir`
  - `test/Target/RVV/selected-dispatch-fallback-envelope-scalar-broadcast-macc-negative.mlir`

## Execution Plan

1. Start the Trellis task and curate context files for implementation/check.
2. Inspect the scalar-broadcast MAcc provider, selected-body realization,
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

- No compiler source change was required for this scalar-broadcast MAcc add
  executable artifact ABI seam. The current production path already starts from
  explicit or pre-realized selected `tcrv_rvv` body facts, realizes
  pre-realized scalar-broadcast MAcc into `setvl -> load lhs -> splat
  rhs_scalar -> load accumulator -> tcrv_rvv.macc -> store out`, derives
  provider-owned scalar-broadcast MAcc family facts from typed config/runtime
  structure, and lowers through `TCRVEmitCLowerableRoute` into neutral Common
  EmitC materialization.
- The scalar-broadcast MAcc provider facts already include runtime ABI order
  `lhs,rhs_scalar,acc,out,n`, `abi|hdr` operand binding summary, scalar splat
  provenance, accumulator/result layout, MAcc arithmetic kind, required
  headers, C type mapping, runtime AVL/VL facts, target leaf profile, and
  provider-supported mirror.
- Target artifact validation already consumes the provider MAcc validation and
  metadata mirror contracts, including the scalar-broadcast route-family plan,
  RHS scalar splat step, ABI/header/type summaries, runtime AVL/VL contract,
  accumulator load, MAcc operand order, output store, and stale mirror
  rejection.
- The remaining useful work for this round was current-HEAD executable
  evidence for explicit and pre-realized selected bodies, not a production
  seam change.

Positive executable `ssh rvv` evidence:

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --artifact-root /tmp/tcrv-scalar-broadcast-macc-probe \
  --run-id scalar-broadcast-macc-explicit-ssh \
  --overwrite \
  --op-kind scalar_broadcast_macc_add \
  --runtime-count 0 --runtime-count 1 --runtime-count 7 \
  --runtime-count 16 --runtime-count 23 --runtime-count 257 \
  --rhs-scalar -37 --rhs-scalar 91 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=scalar_broadcast_macc_add counts=0,1,7,16,23,257 rhs_scalars=-37,91
```

```bash
python3 scripts/rvv_generated_bundle_abi_e2e.py \
  --pre-realized-selected-body \
  --artifact-root /tmp/tcrv-scalar-broadcast-macc-probe \
  --run-id scalar-broadcast-macc-pr-ssh \
  --overwrite \
  --op-kind scalar_broadcast_macc_add \
  --runtime-count 0 --runtime-count 1 --runtime-count 7 \
  --runtime-count 16 --runtime-count 23 --runtime-count 257 \
  --rhs-scalar -37 --rhs-scalar 91 \
  --tcrv-opt build/bin/tcrv-opt \
  --tcrv-translate build/bin/tcrv-translate \
  --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj
```

Result:

```text
rvv_generated_bundle_abi_e2e: success
PASS op=scalar_broadcast_macc_add counts=0,1,7,16,23,257 rhs_scalars=-37,91
```

Both runs emitted per-case runtime output for counts `0,1,7,16,23,257` and
RHS scalar values `-37,91`, including:

```text
scalar_broadcast_macc explicit_accumulator signed_products tail_preserved
tcrv_rvv_generated_bundle_abi_scalar_broadcast_macc_add_ok counts=0,1,7,16,23,257 rhs_scalars=-37,91
```

Evidence JSON confirmed `dry_run: false`, `ssh_evidence: true`,
`status: success`, runtime ABI order `lhs,rhs_scalar,acc,out,n`,
provider route facts for `rvv-route-operand-binding:scalar_broadcast_macc_add.v1`,
RHS scalar `tcrv_rvv.splat`, MAcc operand order `acc,lhs,rhs,vl`, generated
prototype/header bindings, and runtime AVL/VL loop facts for both explicit and
pre-realized selected bodies.

Focused fail-closed evidence retained:

- `test/Scripts/rvv-generated-bundle-abi-e2e-direct-pre-realized-scalar-broadcast-macc-add-fail-closed.test`
  rejects `--direct-pre-realized-route-entry` for
  `scalar_broadcast_macc_add`.
- `test/Target/RVV/pre-realized-selected-body-artifact-scalar-broadcast-macc-add.mlir`
  rejects stale route id, stale scalar-broadcast MAcc route-family plan, stale
  runtime ABI order, missing `hdr` binding token, and stale C type mapping
  before target artifact acceptance.

Focused checks run:

- `cmake --build build --target tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test tcrv-opt tcrv-translate`
- `build/bin/tianchenrv-rvv-extension-plugin-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- From `build/test`:
  `/usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter 'rvv-generated-bundle-abi-e2e-(pre-realized-)?scalar-broadcast-macc-add-dry-run|rvv-generated-bundle-abi-e2e-direct-pre-realized-scalar-broadcast-macc-add-fail-closed|explicit-selected-body-artifact-scalar-broadcast-macc-add|pre-realized-selected-body-artifact-scalar-broadcast-macc-add|selected-dispatch-fallback-envelope-scalar-broadcast-macc-negative'`
  passed 6 selected tests.
- Bounded old-authority scan over touched files found no production source
  diff; the only match was the PRD non-goal guardrail
  `No source-front-door positive route`.
- `git diff --check`
- `git diff --cached --check`

## Spec Update Decision

No `.trellis/spec/` update is needed. This round revalidated existing RVV
plugin and EmitC-route contracts: selected typed `tcrv_rvv` body/config/runtime
facts own scalar-broadcast MAcc route support, Common EmitC remains neutral,
target artifact validation rejects stale mirrors, and executable correctness
claims are backed by current `ssh rvv` evidence.

## Current Phase

Finish.
