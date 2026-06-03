# Stage2 RVV widening dot-reduce route-family provider contract extraction

## Goal

Extract a provider-owned widening dot-reduce route validation contract from the
existing RVV provider facts and rebuilt route descriptions, then rewire target
artifact validation to consume that contract for the existing widening
dot-reduce route families.

The intended ownership chain is:

```text
selected typed tcrv_rvv widening dot-reduce body
  -> RVV plugin-owned widening dot-reduce route facts
  -> provider-owned widening dot-reduce validation contract
  -> target artifact validation consumes the contract plus rebuilt route
```

Target artifact validation must remain a consumer. It must not reconstruct
widening dot-reduce truth from route names, artifact metadata, C strings,
fixture names, descriptors, exact intrinsic spellings, candidate mirrors, or
test names.

## Direction Source

Hermes Direction Brief:

`Stage2 RVV widening dot-reduce route-family provider contract extraction`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` was clean.
* Initial `git log --oneline -8` started at
  `98677173 rvv: complete macc route validation contract`.
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes brief before source edits.
* `.trellis/spec/index.md` requires the RVV authority chain:
  selected `tcrv.exec` RVV variant -> typed low-level `tcrv_rvv` body -> RVV
  plugin-owned legality/realization/provider -> `TCRVEmitCLowerableRoute` ->
  common EmitC materializer -> target artifact -> `ssh rvv` evidence when
  runtime/correctness/performance is claimed.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires active direct
  contraction routes, including widening dot-reduce variants, to be consumed by
  an RVV plugin-owned direct contraction provider/statement owner after
  route-family facts, materialization facts, math operand-binding facts, and
  route-control facts are validated.
* `.trellis/spec/lowering-runtime/emitc-route.md` defines the existing
  `RVVWideningDotReduceRouteFacts` provider-owned surface for plain,
  strided-input, computed-mask, and computed-mask-strided widening dot-reduce
  variants.
* The archived widening dot-reduce production validation task already unified
  provider facts and rewired provider-side and target-side validation to consume
  those facts.
* The immediately previous MAcc task added
  `RVVMAccRouteValidationContract` and rewired target MAcc payload validation
  to consume provider-owned contract data beyond metadata mirrors.
* Archived ABI tasks already contain real `ssh rvv` correctness evidence for
  widening dot-reduce variants, including computed-mask and strided-input
  cases over counts such as `0,1,16,17,257`.

## Requirements

* Add a provider-owned widening dot-reduce route validation contract API in the
  RVV provider layer for existing active operations:
  `widening_dot_reduce_add`,
  `strided_input_widening_dot_reduce_add`,
  `computed_masked_widening_dot_reduce_add`, and
  `computed_masked_strided_input_widening_dot_reduce_add`.
* Build the contract from existing `RVVWideningDotReduceRouteFacts` plus the
  rebuilt `RVVSelectedBodyEmitCRouteDescription`. Do not add new route
  coverage, dtype/LMUL clone batches, reductions, source-front-door routes, or
  artifact-only evidence.
* Include common route expectations:
  route token, memory form, selected SEW/LMUL, tail/mask policy,
  runtime-control plan, runtime ABI order and parameters, target leaf profile,
  `provider_supported_mirror`, required headers, C type summary,
  route operand binding plan/summary, typed compute op, source/result element
  type facts, and route payload header/type requirements.
* Include widening dot-reduce family expectations:
  narrow source type and SEW/LMUL, widened accumulator/result type and
  SEW/LMUL, source/destination memory form, unit-stride vs strided-input
  layout and stride sources, widening dot product relation, widening product
  intrinsic, masked widening product intrinsic where active, scalar seed splat,
  reduction intrinsic, reduction store VL, compare predicate, mask
  role/source/form, inactive-lane zeroing, and statement-plan shape.
* Rewire target artifact widening dot-reduce provider-fact validation to consume
  this provider contract for rebuilt route payload, header/type mappings,
  runtime ABI mappings, source/accumulator/result layout, memory/stride,
  mask/tail, and statement-plan expectations.
* Preserve separate candidate metadata mirror validation. Candidate mirrors may
  only mirror provider-derived contract/fact data after provider route
  construction.
* Preserve fail-closed behavior for missing, stale, cross-family, cross-route,
  or mismatched contract fields, including stale widening MAcc facts, stale
  standalone reduction facts, stale non-mask facts on masked routes, stale mask
  facts on non-masked routes, stale unit-stride/strided cross-contamination,
  stale route operand binding, stale header/type mapping, stale target profile,
  stale provider mirror, and statement-plan mismatches.
* Keep common EmitC/export neutral. Do not move RVV dot-reduce semantics into
  common EmitC, descriptors, target artifact metadata, route ids, artifact
  names, test names, scripts, or exact intrinsic spelling.
* Do not change route emission, generated C/C++, runtime ABI order, widening
  dot-reduce computation, source/accumulator/result layout, mask/tail policy,
  memory stride behavior, correctness, or performance behavior. If any of
  those change, real `ssh rvv` evidence is required.

## Acceptance Criteria

* [x] Production code exposes a provider-owned widening dot-reduce route
      validation contract API beyond metadata mirrors.
* [x] Target artifact validation consumes that contract for existing plain,
      strided-input, computed-mask, and computed-mask-strided widening
      dot-reduce provider-fact validation.
* [x] Target widening dot-reduce validation no longer owns duplicate expected
      route payload/header/type/runtime ABI/statement-plan truth for fields
      represented in the provider contract.
* [x] Rebuilt route validation still checks route token, headers, type
      mappings, ABI mappings, source provenance, and statement-plan shape
      before artifact acceptance.
* [x] Existing candidate metadata mirror validation still rejects stale or
      mismatched mirrors as a separate mirror consumer.
* [x] Focused target/plugin tests and focused lit/script filters for touched
      widening dot-reduce fixture families pass.
* [x] Bounded old-authority scan over touched files finds no new legacy
      `RVVI32M1`, `rvv-i32m1`, `tcrv_rvv.i32_*`, `!tcrv_rvv.i32m*`,
      source-front-door, source-artifact, descriptor/direct-C/source-export,
      route-id/artifact-name authority, mirror-only authority, or exact
      `i32m1` intrinsic authority additions.
* [x] `rtk git diff --check` passes.
* [x] If complete, finish/archive the Trellis task and create one coherent
      commit.

## Technical Approach

1. Inspect the current MAcc validation contract implementation and mirror its
   shape only where it matches widening dot-reduce needs.
2. Add widening dot-reduce validation contract declarations to the RVV provider
   API.
3. Implement the contract builder in the RVV provider/planning implementation
   that already owns widening dot-reduce facts.
4. Rewire `RVVTargetArtifactRouteFamilyValidation.cpp` to validate widening
   dot-reduce rebuilt route payloads against the provider contract instead of
   target-local expected facts.
5. Add or adjust focused C++ checks for positive consumption and fail-closed
   stale/missing/cross-family contract fields.
6. Run the focused build, C++ tests, lit/script filters, old-authority scan,
   and diff check.

## Out Of Scope

* No new widening dot-reduce coverage, dtype/LMUL clone batch, unsigned route,
  new reduction, MAcc redo, compare/select/conversion/memory expansion, high
  level frontend lowering, source-front-door positive route, dashboards, broad
  smoke matrix, or artifact-only evidence.
* No generated runtime behavior change.
* No movement of RVV semantics into common EmitC/export or target artifact
  metadata.
* No `ssh rvv` run unless this task changes generated code, runtime ABI order,
  emitted intrinsics, computation, mask/tail behavior, memory stride behavior,
  correctness, or performance claims.

## Evidence Plan

* Build `tianchenrv-target-artifact-export-test`.
* Build `tianchenrv-rvv-extension-plugin-test` because provider headers/plugin
  code may change.
* Run `build/bin/tianchenrv-target-artifact-export-test`.
* Run `build/bin/tianchenrv-rvv-extension-plugin-test`.
* Run focused lit filters for widening dot-reduce fixture/script families:
  `widening-dot-reduce-add` and computed-mask/strided variants touched by this
  change.
* Run bounded old-authority scans over touched files and changed lines.
* Run `rtk git diff --check`.
* Do not run `ssh rvv` if the final diff remains a provider/target validation
  ownership extraction with no generated runtime semantic change; state the
  no-runtime-change rationale and point to archived runtime evidence.

## Completion Evidence

Implemented:

* Added `RVVWideningDotReduceRouteValidationContract`,
  `RVVWideningDotReduceRouteTypeMappingContract`,
  `RVVWideningDotReduceRouteValidationKind`, and
  `getRVVWideningDotReduceRouteValidationContract(...)` in the RVV provider
  API.
* Built widening dot-reduce route validation contracts from
  `RVVWideningDotReduceRouteFacts` plus rebuilt provider route-description
  payloads for plain, strided-input, computed-mask, and computed-mask-strided
  widening dot-reduce.
* Rewired target artifact widening dot-reduce provider-fact validation to
  consume the provider contract for route id, description payload, headers,
  type mappings, runtime ABI mappings, source/accumulator/result layout,
  memory/stride facts, mask/tail facts, and statement-plan expectations.
* Preserved candidate metadata mirror validation as a separate consume-only
  mirror check derived from provider contract fields.
* Added focused target test coverage that directly asserts provider contract
  construction and updated fail-closed diagnostics for stale/missing provider
  contract fields.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the executable
  widening dot-reduce route validation contract.

Checks run:

* [OK] `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
* [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* [OK] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter widening-dot-reduce-add` from `build/test`, 20 passed.
* [OK] Diff-only old-authority scan over touched production/test changed lines:
  `RVVI32M1|rvv-i32m1|tcrv_rvv\.i32_|!tcrv_rvv\.i32m|source-front-door|source-artifact|descriptor|direct-C|source-export|__riscv_.*_i32m1`.
* [OK] `rtk git diff --check`

Self-repair:

* Initial build repairs fixed target validator call sites that still expected a
  route description where the new statement-plan validator now consumes a
  provider contract.
* Added provider contract aliases for the target consumer's existing generic
  route-step validators while keeping the underlying widening dot-reduce facts
  explicit.
* Updated focused C++ expected diagnostics after the target validator began
  reporting provider-derived contract field names instead of old target-local
  wording.

No `ssh rvv` rerun: this changed provider/target validation contract ownership
and diagnostics only. It did not change route emission, generated C/C++,
runtime ABI order, emitted intrinsics, widening dot-reduce computation,
source/accumulator/result layout, mask/tail policy, memory stride behavior,
correctness, or performance behavior.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Relevant archived tasks read:

* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-macc-route-contract-completion/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-widening-dot-reduce-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-widening-dot-reduce-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-strided-input-widening-dot-reduce-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-widening-dot-reduce-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-computed-masked-strided-input-widening-dot-reduce-artifact-abi/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-computed-masked-strided-input-widening-dot-reduce-artifact-abi-boundary/prd.md`

Likely live files:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCContractionRouteFamilyPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* `test/Target/RVV/*widening-dot-reduce-add*.mlir`
* `test/Scripts/*widening-dot-reduce-add*.test`
