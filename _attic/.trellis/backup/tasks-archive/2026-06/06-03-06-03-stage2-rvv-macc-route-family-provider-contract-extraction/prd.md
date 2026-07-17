# Stage2 RVV MAcc route-family provider contract extraction

## Goal

Extract a provider-owned normalized validation contract for existing RVV MAcc
route families and rewire target artifact validation to consume that contract.
The target consumer may validate rebuilt route payloads and candidate metadata
mirrors, but it must not independently choose MAcc operation semantics,
accumulator/result layout, runtime ABI order, typed config, mask/tail facts,
intrinsic/header/type facts, or cross-family stale mirror policy.

The intended chain is:

```text
selected typed tcrv_rvv MAcc body
  -> RVV plugin-owned MAcc route facts
  -> provider-owned normalized MAcc validation contract
  -> target artifact validation consumes that contract
```

## Direction Source

Hermes Direction Brief:

`Stage2 RVV MAcc route-family provider contract extraction`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Initial `git status --short` had no dirty file list through RTK.
* Initial `git log --oneline -8` started at
  `81171452 rvv: extract memory mirror contract ownership`.
* No `.trellis/.current-task` existed, so this task was created from the Hermes
  brief before source edits.
* The immediately previous archived memory task added a provider-owned
  normalized metadata mirror contract surface:
  `RVVMemoryRouteMetadataMirrorContract`,
  `RVVMemoryRouteMetadataMirrorContractSet`, and provider accessors consumed
  by `RVVTargetArtifactRouteFamilyValidation.cpp`.
* Current MAcc provider surfaces already exist for unit-stride, computed-mask,
  runtime-scalar computed-mask, and widening MAcc facts.
* Current target validation still contains MAcc-local helper accessors,
  fallback constants, and hand-written candidate mirror checks for MAcc route
  families instead of iterating one provider-owned normalized MAcc contract.
* Archived production validation tasks record completed provider facts and
  fail-closed target checks for unit-stride MAcc, computed-mask MAcc,
  runtime-scalar computed-mask MAcc, and widening MAcc. This round should
  extract and consume the shared contract from those existing facts, not add
  new MAcc coverage.

## Requirements

* Add a small RVV plugin/provider-owned normalized MAcc route validation
  contract surface in C++.
* The contract must package provider-owned mirror entries as key/expected/label
  facts and stale mirror keys as provider-owned rejection lists, matching the
  memory contract pattern where practical.
* Build contract entries from existing provider facts and provider route
  descriptions for the already-supported MAcc families:
  `macc_add`, `scalar_broadcast_macc_add`, `computed_masked_macc_add`,
  `runtime_scalar_cmp_masked_macc_add`, and `widening_macc_add`.
* Rewire target artifact candidate mirror validation for those MAcc families to
  consume the provider-owned MAcc contract instead of owning operation-specific
  mirror tables locally.
* Preserve target validation of rebuilt route payloads, statement plans,
  headers/types, runtime ABI mappings, and existing fail-closed provider fact
  checks. If full payload normalization is too large for one round, complete
  candidate mirror contract extraction first and leave payload normalization as
  the exact continuation point.
* Fail closed when provider contract fields are missing, stale, cross-family,
  cross-route, or mismatched for provider support mirrors, header/type facts,
  runtime ABI order, operand binding, accumulator/passthrough/result fields,
  mask/tail facts, typed operation/config facts, source/destination memory
  facts, and statement-plan expectations.
* Keep common EmitC/export neutral. Do not move RVV MAcc semantics into common
  EmitC, target artifact metadata, route ids, artifact names, descriptors,
  scripts, test names, or exact intrinsic spelling.
* Do not change generated code, runtime ABI order, accumulation behavior,
  mask/tail policy, passthrough/destination preservation, correctness, or
  performance behavior. If any of those change, real `ssh rvv` evidence becomes
  required.

## Acceptance Criteria

* [ ] Production code exposes a provider-owned normalized MAcc metadata mirror
      contract from RVV plugin/provider headers and implementation.
* [ ] The target validator no longer owns MAcc candidate mirror construction
      for plain, scalar-broadcast, computed-mask, runtime-scalar computed-mask,
      or widening MAcc route families.
* [ ] Target artifact validation consumes provider-owned MAcc contract entries
      and stale-key lists after rebuilding the provider route description.
* [ ] Existing provider fact validation still rejects missing/stale runtime ABI
      parameters, SEW/LMUL/policy, typed compute op, arithmetic kind,
      accumulator/result layout, mask/passthrough facts, source/destination
      memory form, header/type facts, target profile, provider mirror, and
      statement-plan mismatches.
* [ ] Candidate metadata mirror validation rejects stale cross-family facts:
      plain vs scalar-broadcast MAcc, computed-mask vs non-mask MAcc,
      runtime-scalar vs vector computed-mask MAcc, widening vs unit-stride
      MAcc, and unrelated reduction/memory/segment/indexed/contraction residue
      where existing coverage expects it.
* [ ] Existing focused C++ target artifact tests and focused MAcc lit/script
      filters continue to pass.
* [ ] No new route coverage, dtype/LMUL clone batch, source-front-door positive
      route, descriptor-driven computation, common EmitC RVV semantics,
      artifact-only authority, exact-intrinsic authority, or legacy `i32m1`
      route authority is introduced.
* [ ] Focused build/tests, bounded old-authority scan, `rtk git diff --check`,
      Trellis finish/archive, clean worktree, and one coherent commit complete
      the round if the module behavior is complete.

## Technical Approach

1. Add provider-owned MAcc mirror contract structs/accessors alongside the
   existing memory mirror contract in `RVVEmitCRouteProvider.h`.
2. Implement contract construction in the RVV MAcc provider/owner source,
   using `RVVUnitStrideMAccRouteFacts`,
   `RVVComputedMaskMAccRouteFacts`,
   `RVVRuntimeScalarComputedMaskMAccRouteFacts`,
   `RVVWideningMAccRouteFacts`, and the rebuilt route description for fields
   already materialized as route-description facts.
3. Rewire `RVVTargetArtifactRouteFamilyValidation.cpp` to iterate provider
   MAcc mirror contract entries and stale-key lists for candidate metadata.
4. Keep existing route payload, ABI mapping, header/type, and statement-plan
   validation intact unless a direct duplicated MAcc fact can be removed safely
   without weakening diagnostics.
5. Add focused C++ test mutations only if the extraction exposes a missing
   fail-closed case not already covered.
6. Run the focused build/test/lit scans requested by the brief.

## Out Of Scope

* No new MAcc, reduction, arithmetic, conversion, memory, offload, frontend, or
  source-front-door route coverage.
* No dtype/LMUL clone batches, high-level Linalg lowering, dashboards, broad
  smoke matrices, or artifact-only evidence work.
* No changes to runtime ABI order, emitted C/C++, MAcc computation,
  accumulator layout, mask/tail behavior, passthrough behavior, or correctness
  claims.
* No movement of RVV semantics into common EmitC/export or target metadata.
* No `ssh rvv` run unless this round changes generated runtime behavior or
  makes a new runtime/correctness/performance claim.

## Evidence Plan

* Build and run `tianchenrv-target-artifact-export-test`.
* Build and run `tianchenrv-rvv-extension-plugin-test`, because provider
  headers or plugin code may change.
* Run focused lit filters for touched MAcc route families:
  `(macc-add|scalar-broadcast-macc-add|computed-masked-macc-add|runtime-scalar-cmp-masked-macc-add|widening-macc-add)`.
* Run a bounded old-authority scan over touched files.
* Run `rtk git diff --check`.
* Do not run `ssh rvv` if the change remains a provider/target validation
  contract extraction with no generated runtime semantic change.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Previous task and MAcc archives read:

* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-provider-owned-memory-route-contract-extraction/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-unit-stride-macc-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-stage2-rvv-computed-mask-macc-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-widening-macc-production-validation-boundary/prd.md`
* `.trellis/tasks/archive/2026-06/06-02-stage2-rvv-runtime-scalar-masked-macc-accumulation-artifact-abi-boundary/prd.md`

Likely live files:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCMAccRouteFamilyPlanOwners.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
* `lib/Plugin/RVV/EmitC/RVVEmitCRoutePlanning.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* focused MAcc fixtures under `test/Target/RVV` and `test/Scripts`

## Completion Evidence

Implemented on 2026-06-03.

Production changes:

* Added provider-owned `RVVMAccRouteMetadataMirrorContract` and
  `RVVMAccRouteMetadataMirrorContractSet` in
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`.
* Added `getRVVMAccRouteMetadataMirrorContract(...)` in the RVV MAcc
  provider/owner implementation. The accessor builds candidate mirror entries
  and stale-key lists from existing provider facts for plain MAcc,
  scalar-broadcast MAcc, computed-mask MAcc, runtime-scalar computed-mask
  MAcc, and widening MAcc.
* Runtime-scalar computed-mask MAcc contract construction uses the actual
  rebuilt route `sew` and `lmul`, so LMUL m2 candidate mirrors are validated
  against LMUL m2 provider facts rather than default m1 facts.
* Rewired non-widening MAcc and widening MAcc target candidate mirror
  validators to consume the provider-owned contract set instead of owning
  MAcc mirror construction locally.
* Preserved existing provider payload, runtime ABI mapping, header/type, and
  statement-plan validation.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the executable
  MAcc metadata mirror contract API, validation matrix, and tests.

Checks run:

* `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* `rtk build/bin/tianchenrv-target-artifact-export-test`
* `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter '(macc-add|scalar-broadcast-macc-add|computed-masked-macc-add|runtime-scalar-cmp-masked-macc-add|widening-macc-add)'` from `build/test`, 28 selected and passed.
* `rtk git diff --check`
* Diff-only bounded old-authority scan over touched source/test files found no
  new legacy i32m1, `RVVI32M1`, `rvv-i32m1`, source-front-door,
  source-artifact, descriptor/direct-C/source-export, route-id/artifact-name,
  bare supported-status, or exact i32m1 intrinsic authority additions.

Self-repair:

* The first focused build failed because the mirror append helper had both
  `StringRef` and `std::string` overloads, making empty string literals
  ambiguous. Removed the `std::string` overload and relied on immediate
  `StringRef` copying into contract entries.
* During static review, common mirror labels were changed from temporary
  `std::string` values to stable string literals because contract labels are
  stored as `StringRef`.

No new `ssh rvv` run was required. This round only moved MAcc candidate
metadata mirror contract ownership into provider APIs and did not change route
emission, generated C/C++, runtime ABI order, MAcc computation, accumulator
layout, mask/tail behavior, passthrough/destination preservation, correctness,
or performance behavior.
