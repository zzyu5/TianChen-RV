# Stage2 RVV computed-mask MAcc selected-body route foundation

## Goal

Establish one bounded Stage 2 route-supported computed-mask `macc_add`
selected-body proof route. A selected `tcrv.exec` RVV variant must carry an
explicit or pre-realized typed `tcrv_rvv` body where a compare-produced mask
guards vector-vector multiply-accumulate:

```text
active lanes:   out[i] = acc[i] + lhs[i] * rhs[i]
inactive lanes: out[i] preserves accumulator/pass-through
```

The route must flow through RVV plugin-owned selected-body realization and
route planning, provider-built `TCRVEmitCLowerableRoute`, Common EmitC neutral
materialization, and target artifact validation. Executable `ssh rvv` evidence
is in scope only if the generated-bundle path is already reachable or can be
closed without turning this task into a broad harness project.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV computed-mask MAcc selected-body route foundation`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `46dc2e26 rvv: close plain macc executable abi`.
- No `.trellis/.current-task` existed at the start of the round, so this task
  was created before any source edits.
- The immediately previous plain MAcc executable ABI task proved explicit and
  pre-realized plain `macc_add` generated bundle correctness on `ssh rvv` with
  counts `0,1,16,17,257`, scalar oracle coverage, source preservation, and tail
  preservation.
- The computed-mask standalone reduce-add task proved a separate masked
  reduction route and executable path, including provider-derived mask
  role/source/form, predicate, inactive-lane behavior, runtime ABI order, and
  stale mirror rejection.
- This round's delta is computed-mask MAcc: mask producer/source/form,
  compare predicate, inactive-lane/pass-through policy, payload lhs/rhs,
  accumulator/result layout, arithmetic kind `add`, runtime ABI/order,
  SEW/LMUL/policy, intrinsic/header/type facts, and provider-supported mirrors
  must come from typed body/config/runtime facts and RVV provider facts.

## Requirements

- Keep scope to one vector-compare computed-mask `macc_add` route:
  `computed_masked_macc_add`.
- Preserve the authority chain:
  selected typed `tcrv_rvv` body/config/runtime facts -> RVV selected-body
  realization/validation -> RVV route-family provider facts -> lowerable route
  -> Common EmitC/export -> target artifact mirrors -> optional generated
  bundle runtime evidence.
- The route/provider/target path must structurally carry or derive:
  - ABI roles and order `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n`;
  - compare predicate, mask role/source, and mask memory form;
  - arithmetic kind `add`;
  - active-lane MAcc contract and inactive accumulator/pass-through contract;
  - accumulator and result layout;
  - selected element type, SEW, LMUL, tail policy, mask policy, config contract,
    runtime AVL/VL boundary, and route-control plan;
  - route operand-binding plan/summary with exported header/prototype
    participation markers;
  - required headers, C type mapping, intrinsic/statement leaves, and
    `provider_supported_mirror`.
- Common EmitC/export may only carry provider-built route payloads and
  metadata mirrors. It must not infer MAcc operation, mask semantics, dtype,
  SEW/LMUL, runtime ABI roles, intrinsic spelling, or inactive-lane behavior.
- Stale or missing facts must fail closed before target artifact acceptance:
  predicate, mask role/source/form, arithmetic kind, accumulator/result layout,
  runtime ABI/order, route operand binding, selected SEW/LMUL/policy,
  header/type mapping, target leaf profile, provider support mirror, stale
  runtime-scalar/plain/standalone-reduction/widening facts, and statement-plan
  shape.
- If production code already supports the route, finish by proving focused
  positive and negative evidence instead of adding helper-only changes.

## Acceptance Criteria

- [x] `prd.md`, `implement.jsonl`, and `check.jsonl` describe this bounded
      computed-mask MAcc route foundation and cite the relevant specs.
- [x] Focused code inspection identifies whether current production
      provider/route/target/generated-bundle support for
      `computed_masked_macc_add` is complete or where it fails.
- [x] Positive route/artifact/header evidence proves provider-derived
      computed-mask MAcc facts for predicate, mask role/source/form, MAcc
      arithmetic kind, active/inactive lane contracts, accumulator/result
      layout, runtime AVL/VL, runtime ABI order, operand binding, typed
      config, header/type mapping, target leaf profile, and
      `provider_supported_mirror`.
- [x] Negative evidence proves stale or missing predicate, mask facts,
      arithmetic kind, accumulator/result layout, runtime ABI/order, operand
      binding, SEW/LMUL/policy, header/type mapping, provider support mirror,
      and cross-family residue fail closed before artifact acceptance.
- [x] Common EmitC/export remains neutral and does not choose computed-mask
      MAcc semantics.
- [x] Focused `tcrv-opt` / `tcrv-translate` / RVV provider / target artifact
      tests for changed behavior pass.
- [x] If executable behavior is claimed, generated artifact plus real
      `ssh rvv` correctness passes over counts including `0`, `1`, a VL
      boundary, a tail case, and a multi-chunk case, with mixed active/inactive
      mask lanes, scalar oracle coverage, inactive accumulator/pass-through
      preservation, source preservation, and tail sentinel preservation.
- [x] If executable closure is too large, finish route-supported plus
      target-validation closure and record generated-bundle `ssh rvv` as the
      exact next continuation point. Not applicable: executable closure passed.
- [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, dtype-prefixed
      `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`, descriptor, source-front-door,
      source-artifact, route-string/artifact-name/ABI-string/test-name,
      exact-intrinsic-spelling, or Common EmitC semantic authority.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation,
      final clean worktree, archive, and one coherent commit complete the
      round if the module behavior is complete.

## Evidence Plan

- Inspect the current computed-mask MAcc operation surface, selected-body
  fixtures, provider planning, statement owners, target validation, generated
  bundle script, and existing tests before source edits.
- Run focused positive route/header/artifact checks for
  `computed_masked_macc_add`.
- Add or repair production provider/target validation only where live evidence
  shows missing route-supported behavior or stale fact rejection.
- Prefer target artifact and provider negative tests for fail-closed mirror and
  route payload validation.
- Extend generated-bundle dry-run and `ssh rvv` evidence only if the route
  reaches generated artifact form with bounded script/harness changes.

## Definition Of Done

- Current HEAD either has route-supported computed-mask `macc_add` selected-body
  provider and target artifact validation closure, or records the precise
  production blocker and next continuation point without false route/executable
  claims.
- Any implementation changes are production-path changes or focused evidence
  for already-existing production support; helper/test/spec-only work is not a
  completion unless production support was already real.
- Specs are updated only if this round discovers a durable rule not already
  captured in `.trellis/spec/`.
- The task is finished/archived and one coherent commit is created when the
  module behavior is complete.

## Out Of Scope

- High-level matmul, Linalg, Vector, StableHLO, or source-front-door lowering.
- Broad dtype/LMUL/op matrix expansion.
- Runtime-scalar computed-mask MAcc unless needed only as a stale-residue
  negative boundary for vector-compare `computed_masked_macc_add`.
- Scalar-broadcast MAcc, widening MAcc, dot-reduction, reductions, Gearbox,
  TensorExt, IME, Offload, performance benchmarking, autotuning, or dashboards.
- One-intrinsic wrapper routes.
- Descriptor-driven computation or Common EmitC semantic inference.
- Treating route ids, artifact names, ABI strings, test names, intrinsic
  spellings, status fields, manifests, or metadata mirrors as route authority.

## Technical Notes

Specs read before implementation:

- `.trellis/spec/index.md`
- `.trellis/spec/extension-plugins/rvv-plugin.md`
- `.trellis/spec/lowering-runtime/emitc-route.md`
- `.trellis/spec/testing/mlir-testing-contract.md`
- `.trellis/spec/variant-pipeline/index.md`

Relevant archived tasks read:

- `.trellis/tasks/archive/2026-06/06-05-rvv-stage2-plain-macc-executable-abi-closure/prd.md`
- `.trellis/tasks/archive/2026-06/06-05-rvv-stage2-computed-mask-standalone-reduce-add-route/prd.md`

Primary inspection surfaces from the direction brief:

- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
- `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
- `lib/Plugin/RVV/EmitC/`
- `lib/Plugin/RVV/Construction/`
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
- `lib/Target/RVV/RVVTargetSupportBundle.cpp`
- `scripts/rvv_generated_bundle_abi_e2e.py`
- `test/Target/RVV/*macc*`

## Completion Evidence

Completed as executable computed-mask MAcc selected-body route foundation.
Production provider, route planning, target artifact validation, generated
bundle support, and harness behavior already supported the route; this round
closed the missing focused evidence by strengthening representative target
fixtures and by running real `ssh rvv` correctness for both explicit and
pre-realized selected-body inputs.

Focused fixture changes:

- Added positive plan/header checks for provider-derived
  `tcrv_rvv.macc_arithmetic_kind = add` on explicit and pre-realized
  `computed_masked_macc_add`.
- Added header checks for computed-mask MAcc mask role/source/form and
  accumulator/result layout on explicit and pre-realized selected-body target
  artifacts.
- Added pre-realized target artifact fail-closed checks for stale
  `tcrv_rvv.mask_role`, `tcrv_rvv.mask_memory_form`, and
  `tcrv_rvv.macc_arithmetic_kind`.

Provider and target route facts proved:

- Operation and typed compute: `computed_masked_macc_add`,
  `tcrv_rvv.masked_macc`.
- Runtime ABI order:
  `cmp_lhs,cmp_rhs,lhs,rhs,acc,out,n`.
- Binding summary:
  `rvv-route-operand-binding:computed_masked_macc_add.v1` with `abi` and
  `hdr` participation for all exported operands.
- Mask/predicate facts: predicate `slt`, mask role
  `predicate-mask-produced-by-compare`, mask source
  `compare-produced-mask-same-vl-scope`, mask memory form
  `compare-produced-mask`.
- MAcc facts: arithmetic kind `add`, accumulator layout
  `separate-i32-vector-accumulator-input`, result layout
  `store-multiply-accumulate-result-to-output-buffer`.
- Inactive-lane/pass-through facts:
  `masked-macc-false-lanes-preserve-accumulator` and
  `accumulator-vector-preserves-inactive-lanes`.
- Target leaf/profile facts:
  `rvv-v1-typed-computed-mask-macc-add-leaf-profile.v1`,
  `provider_supported_mirror:rvv-computed-mask-macc-add-plan-validated`,
  headers `stddef.h,stdint.h,riscv_vector.h`, and C type mapping
  `vl:size_t,cmp_lhs/cmp_rhs/lhs/rhs/acc:typed-vector,mask:typed-mask,result:typed-vector`.

Generated artifact evidence:

- Explicit dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/computed-mask-macc-foundation-explicit-dryrun`
- Pre-realized dry-run:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/computed-mask-macc-foundation-pre-realized-dryrun`
- Explicit `ssh rvv`:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/computed-mask-macc-foundation-explicit-ssh`
- Pre-realized `ssh rvv`:
  `artifacts/tmp/rvv_generated_bundle_abi_e2e/computed-mask-macc-foundation-pre-realized-ssh`

Both real RVV runs passed:

```text
PASS op=computed_masked_macc_add counts=0,1,16,17,257 patterns=0,1
```

Per-case remote output covered `n = 0`, one-lane, full-VL boundary, tail, and
multi-chunk counts. The harness reported mixed active/inactive mask lanes,
inactive accumulator preservation, add-only and mul-only distinguishing lanes,
signed products, source preservation, and output tail preservation.

Focused checks passed:

- `cmake --build build --target tcrv-opt tcrv-translate tianchenrv-target-artifact-export-test -j2`
- `python3 -m py_compile scripts/rvv_generated_bundle_abi_e2e.py`
- `python3 scripts/rvv_generated_bundle_abi_e2e.py --self-test`
- `build/bin/tianchenrv-target-artifact-export-test`
- Direct FileCheck equivalents for explicit/pre-realized computed-mask MAcc
  plan/header/realization paths.
- Direct FileCheck equivalents for stale mask role, mask form, and MAcc
  arithmetic mirrors.
- `build/bin/tcrv-opt test/Dialect/RVV/masked-macc-dataflow.mlir --split-input-file --verify-diagnostics`
- `build/bin/tcrv-opt test/Dialect/RVV/pre-realized-computed-mask-macc-negative.mlir --split-input-file --verify-diagnostics`
- Generated-bundle dry-runs for explicit and pre-realized
  `computed_masked_macc_add`.
- Real `ssh rvv` generated-bundle runs for explicit and pre-realized
  `computed_masked_macc_add`.
- `git diff --check`
- Added-line authority scan over touched target fixtures; no matches.
- `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-05-06-05-rvv-stage2-computed-mask-macc-selected-body-route-foundation`

Spec update decision:

- No `.trellis/spec/` update was needed. The existing RVV plugin,
  EmitC-route, and MLIR testing contracts already require provider-owned
  computed-mask MAcc fact surfaces, MAcc route validation/mirror contracts,
  fail-closed stale metadata rejection, and generated-bundle/`ssh rvv`
  evidence requirements. This round implemented evidence for that existing
  contract rather than creating a new contract.
