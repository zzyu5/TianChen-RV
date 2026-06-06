# Stage2 RVV product-reduction dequant executable artifact ABI boundary

## Goal

Make the existing selected-body
`widening_product_reduce_dequant_clamp_f32` RVV route executable through the
generated artifact ABI with truthful provider-derived operand binding evidence,
or fail closed at the exact stale/missing artifact boundary. This round is
bounded to the product-reduction-dequant-clamp-f32 route that crosses i8
source loads, widening product, widening reduction, scalar i32 carry,
f32 dequantization, lower/upper clamp, f32 output store, generated bundle ABI,
and real `ssh rvv` correctness evidence.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV product-reduction dequant executable artifact ABI boundary`

## What I Already Know

- Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
- Initial `git status --short` was clean.
- Initial `git log --oneline -8` started at
  `ba39671a rvv: verify widening dot artifact abi evidence`.
- No `.trellis/.current-task` existed at the start of this round, so this task
  was created from the Hermes brief before source edits.
- Relevant specs read:
  `.trellis/spec/index.md`,
  `.trellis/spec/extension-plugins/index.md`,
  `.trellis/spec/extension-plugins/rvv-plugin.md`,
  `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/spec/testing/index.md`.
- Previous archived task
  `.trellis/tasks/archive/2026-06/06-06-stage2-rvv-widening-dot-reduce-executable-artifact-abi-boundary/prd.md`
  proved the plain `widening_dot_reduce_add` executable artifact ABI boundary
  with real `ssh rvv` evidence. This round must not repeat that route as the
  main achievement.
- Current dry-run precheck for
  `scripts/rvv_generated_bundle_abi_e2e.py --dry-run --pre-realized-selected-body --op-kind widening_product_reduce_dequant_clamp_f32`
  succeeds locally, but reports `ssh_evidence: false`.
- That precheck evidence exports the correct callable ABI prototype:
  `const int8_t *lhs, const int8_t *rhs, const int32_t *acc, float scale, float lower_bound, float upper_bound, float *out, size_t n`.
- The same evidence still exports a compressed
  `tcrv_rvv.route_operand_binding_operands` mirror:
  `...;abi=lhs,rhs,acc,scale,lower_bound,upper_bound,out,n;chain=...;uses=...`.
  This does not independently bind `scale`, `lower_bound`, `upper_bound`,
  `out`, and `n` with per-operand `abi`/`hdr` facts even though the generated
  header/prototype exports those parameters.
- `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp` already derives a detailed
  `RVVRouteOperandBindingPlan` for this operation with per-operand bindings:
  lhs/rhs/acc/scale/lower_bound/upper_bound/out/n and individual `abi`/`hdr`
  use tokens.
- `stringifyRVVRouteOperandBindingPlan(...)` special-cases this route and
  collapses that detailed plan into the compressed mirror, so the artifact
  boundary loses provider-derived per-operand evidence.
- `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp` also
  records the same compressed summary in the direct-contraction route facts,
  so the target validation contract currently accepts the compressed mirror.
- `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` already validates
  runtime ABI order, ABI parameter count and order, headers, type mappings,
  local scalar carry, loop statements, post-loop dequant/clamp/store
  statements, and candidate mirrors against the provider contract. The weak
  point is the provider contract's compressed operand-binding mirror for this
  route, not Common EmitC semantics.

## Requirements

- Keep the authority chain structural:
  selected/pre-realized `tcrv_rvv` body facts ->
  direct contraction provider facts ->
  `TCRVEmitCLowerableRoute` ->
  Common EmitC materialization ->
  target artifact export ->
  generated bundle ABI ->
  `ssh rvv` correctness evidence.
- Replace the compressed
  `widening_product_reduce_dequant_clamp_f32` route operand binding summary
  with the existing provider-derived per-operand binding structure:
  `lhs`, `rhs`, `acc`, `scale`, `lower_bound`, `upper_bound`, `out`, and `n`
  must each carry its logical role, C ABI name, `abi`, and `hdr` participation.
- Keep route facts, route description, target artifact validation contract,
  generated bundle script expectations, and dry-run FileCheck expectations
  aligned to the same provider-owned binding summary.
- Preserve RVV plugin ownership of product/reduction/dequant/clamp dtype,
  SEW/LMUL, intrinsic, C type, runtime ABI, local carry, and statement facts.
  Common EmitC/export must remain neutral.
- If executable behavior is claimed, run real `ssh rvv` evidence for only the
  focused `widening_product_reduce_dequant_clamp_f32` route.
- Preserve direct pre-realized route-entry fail-closed behavior and do not
  revive source-front-door, descriptor-driven, or direct-C route authority.

## Acceptance Criteria

- [x] A Trellis PRD and context files exist for this bounded task before
      source edits.
- [x] Production code exports a per-operand
      `widening_product_reduce_dequant_clamp_f32` binding summary with
      provider-owned `abi`/`hdr` facts for all generated ABI/header parameters.
- [x] Target artifact validation still rejects stale/mismatched binding mirrors
      by exact comparison with the rebuilt provider contract.
- [x] Generated-bundle dry-run evidence for the pre-realized selected-body route
      checks the expanded operand binding summary, runtime ABI order, type
      mapping, product/reduction/dequant/clamp facts, and generated harness.
- [x] Generated-bundle dry-run evidence for the explicit selected-body
      realization route remains passing if shared script expectations are
      affected.
- [x] `build/bin/tianchenrv-rvv-extension-plugin-test` passes.
- [x] `build/bin/tianchenrv-target-artifact-export-test` passes.
- [x] The relevant generated-bundle dry-run lit test passes.
- [x] Real non-dry-run generated-bundle ABI evidence passes on `ssh rvv` before
      runtime correctness is claimed.
- [x] Bounded old-authority scan over touched files and added diff lines shows
      no new positive legacy `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`,
      descriptor, source-front-door/source-export, direct-C, helper-name, or
      artifact-name route authority drift.
- [x] `git diff --check`, `git diff --cached --check`, Trellis validation, and
      clean final git status pass.
- [x] If the task is complete, finish/archive the Trellis task and create one
      coherent commit.

## Definition Of Done

- Compiler behavior remains in C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck.
- Python remains evidence tooling only; it does not implement compiler core,
  dialects, passes, provider facts, target export, or lowering semantics.
- Product-reduction-dequant-clamp route facts remain provider-owned; artifact
  metadata remains a mirror after route construction.
- Common EmitC/export do not infer RVV semantics, dtype, ABI order, intrinsic
  spelling, scalar carry layout, clamp semantics, or support status.
- No new route family, dtype/LMUL clone batch, high-level frontend, broad
  generated-bundle matrix, dashboard, performance tuning DB, source-front-door
  positive route, or mass rewrite of unrelated route owners is introduced.

## Technical Approach

Use the existing production path and repair only the stale artifact ABI mirror:

```text
pre-realized or explicit selected tcrv_rvv product-reduction-dequant-clamp body
  -> selected lowering-boundary materialization
  -> RVV direct contraction provider facts and binding plan
  -> expanded route_operand_binding_operands mirror
  -> TCRVEmitCLowerableRoute
  -> Common EmitC materialization
  -> target artifact route-family validation
  -> generated object/header bundle
  -> generated external C ABI harness
  -> ssh rvv compile/run correctness
```

Implementation steps:

1. Remove the product-reduction-dequant-clamp special-case compression from
   `stringifyRVVRouteOperandBindingPlan(...)` so the already-derived binding
   plan stringifies into per-operand facts.
2. Change the direct-contraction route facts for
   `WideningProductReduceDequantClampF32` to the same per-operand summary.
3. Update the generated-bundle script constants and focused dry-run FileCheck
   tests to expect the expanded summary.
4. Run focused C++ tests, focused lit/script evidence, a non-dry-run `ssh rvv`
   generated-bundle proof, and bounded authority scans.

## Out Of Scope

- New route families.
- New dtype or LMUL clone batches.
- High-level Linalg/Vector/StableHLO frontend work.
- Per-Linalg route authority.
- Source-front-door or source-artifact positive RVV routes.
- Descriptor-driven or direct-C computation export.
- Common EmitC invention of RVV semantics.
- Direct pre-realized route-entry support.
- Performance tuning, autotuning databases, dashboards, or broad generated
  bundle matrices.
- Mass rewrite of memory, compare/select, conversion/dequantization,
  standalone reduction, MAcc, segment2, elementwise, or unrelated contraction
  owners.

## Technical Notes

- Relevant production files:
  `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`,
  `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`.
- Relevant evidence/tooling files:
  `scripts/rvv_generated_bundle_abi_e2e.py`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-pre-realized-widening-product-reduce-dequant-clamp-f32-dry-run.test`,
  `test/Scripts/rvv-generated-bundle-abi-e2e-explicit-widening-product-reduce-dequant-clamp-f32-dry-run.test`.
- Precheck artifact generated during planning:
  `artifacts/tmp/codex-stage2-wprdc-precheck/pre-realized-widening-product-reduce-dequant-clamp-f32`.
  This is temporary evidence and should not be committed.

## Completion Evidence

- Production change:
  `stringifyRVVRouteOperandBindingPlan(...)` no longer collapses
  `widening_product_reduce_dequant_clamp_f32` into a route-level
  `abi=...;uses=...` summary. It now stringifies the existing provider-derived
  binding plan as per-operand facts while abbreviating route-family-owned use
  tokens to stay within the 512-byte bounded metadata contract.
- Direct contraction route facts now export the same per-operand summary:
  `lhs`, `rhs`, `acc`, `scale`, `lower_bound`, `upper_bound`, `out`, and `n`
  each carry provider-derived logical role, C ABI name, `abi`, and `hdr`
  participation.
- Self-repair performed:
  the first expanded summary exceeded the target artifact bounded metadata
  limit and failed with `artifact metadata value must be bounded single-line
  text`; the repaired summary keeps all operands and required `abi`/`hdr`
  tokens while shortening route-family-owned use tokens such as `src-load` to
  `ld`, `runtime-lower` to `lo`, and `clamped-dequant-result` to `cdeq`.
- Focused fail-closed evidence:
  both pre-realized and explicit Target/RVV fixtures now mutate the
  `lower_bound` operand binding by removing its `hdr` token and require target
  artifact export to reject the stale binding mirror.
- Real executable evidence:
  `python3 scripts/rvv_generated_bundle_abi_e2e.py --pre-realized-selected-body --artifact-root artifacts/tmp/codex-stage2-wprdc-abi-ssh --run-id pre-realized-widening-product-reduce-dequant-clamp-f32 --overwrite --op-kind widening_product_reduce_dequant_clamp_f32 --runtime-count 0 --runtime-count 1 --runtime-count 16 --runtime-count 17 --runtime-count 257 --tcrv-opt build/bin/tcrv-opt --tcrv-translate build/bin/tcrv-translate --llvm-readobj /usr/lib/llvm-20/bin/llvm-readobj --ssh-target rvv --connect-timeout 10 --timeout 120`
  returned `rvv_generated_bundle_abi_e2e: success`, recorded
  `ssh_evidence: true`, and ended with
  `PASS op=widening_product_reduce_dequant_clamp_f32 counts=0,1,16,17,257 patterns=0,1 scales=-0.125,0.375 bound_pairs=-1.5:2.25,-8:-0.75 tolerance=1e-05`.
- Checks run:
  `ninja -C build tcrv-opt tcrv-translate tianchenrv-rvv-extension-plugin-test tianchenrv-target-artifact-export-test`;
  `build/bin/tianchenrv-rvv-extension-plugin-test`;
  `build/bin/tianchenrv-target-artifact-export-test`;
  `python3 /usr/lib/llvm-20/build/utils/lit/lit.py -v . --filter 'widening-product-reduce-dequant-clamp-f32'`
  from `build/test` with `PATH` including `build/bin` and
  `/usr/lib/llvm-20/bin`;
  the non-dry-run `ssh rvv` generated-bundle evidence command above;
  bounded added-diff old-authority scan over touched production/test files;
  `git diff --check`;
  `git diff --cached --check`;
  `python3 ./.trellis/scripts/task.py validate .trellis/tasks/06-06-stage2-rvv-product-reduction-dequant-executable-artifact-abi-boundary`.
- Spec update judgment:
  no `.trellis/spec/` change is needed. `.trellis/spec/lowering-runtime/emitc-route.md`
  already requires per-operand provider binding summaries for generated
  header/prototype parameters and already allows shortening route-family-owned
  labels/tokens to satisfy bounded metadata without dropping logical operands
  or `abi`/`hdr` participation.
