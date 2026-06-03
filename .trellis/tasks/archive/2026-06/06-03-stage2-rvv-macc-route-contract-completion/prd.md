# Stage2 RVV MAcc route-contract completion beyond mirrors

## Goal

Complete the next bounded part of the provider-owned MAcc route validation
contract. The target artifact layer should consume RVV provider-owned MAcc
contract data for rebuilt route payload facts, header/type facts, runtime ABI
parameters, accumulator/passthrough/mask facts, and statement-plan shape for
the existing MAcc route families instead of rebuilding MAcc truth from
target-local constants, route names, artifact metadata, fixture names, or
intrinsic strings.

The intended chain is:

```text
selected typed tcrv_rvv MAcc body
  -> RVV plugin-owned MAcc route facts and route description
  -> provider-owned MAcc validation contract
  -> target artifact validation consumes the contract plus rebuilt route
```

## Direction Source

Hermes Direction Brief:

`Stage2 RVV MAcc route-contract completion beyond mirrors`

## What I Already Know

* Initial `pwd` was `/home/kingdom/phdworks/TianchenRV`.
* Repair-round initial `git status --short` showed dirty in-progress MAcc
  contract changes in `.trellis/spec/lowering-runtime/emitc-route.md`,
  `.trellis/workspace/codex/index.md`,
  `.trellis/workspace/codex/journal-21.md`,
  `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`,
  `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`,
  `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`,
  `test/Target/TargetArtifactExportTest.cpp`, plus the untracked archived task
  directory for this task.
* Initial `git log --oneline -8` started at
  `d808bdbd rvv: extract macc mirror contract ownership`.
* No `.trellis/.current-task` existed in the repair round. The existing
  untracked archived task directory was kept and repaired instead of creating a
  competing task.
* The previous archived task
  `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-macc-route-family-provider-contract-extraction/`
  extracted a provider-owned normalized MAcc metadata mirror contract and
  explicitly preserved existing target validation for route payload,
  header/type, runtime ABI mapping, and statement plans.
* `.trellis/spec/lowering-runtime/emitc-route.md` requires computed-mask MAcc,
  runtime-scalar computed-mask MAcc, and widening MAcc facts to come from
  provider facts, with target validation consuming provider facts rather than
  target-local constants.
* `.trellis/spec/extension-plugins/rvv-plugin.md` requires target family
  validators to dispatch from the rebuilt provider description and rebuilt
  lowerable route, not from metadata, route ids, artifact names, ABI strings,
  tests, scripts, direct route entries, or exact intrinsic spellings.
* Current live target MAcc validation still contains target-local helper
  accessors/constants for expected MAcc runtime ABI order, binding plan,
  target profile, provider support, type summary, computed-mask constants, and
  widening MAcc facts.

## Requirements

* Add a provider-owned MAcc route validation contract API in C++ for the
  existing route families:
  `macc_add`, `scalar_broadcast_macc_add`,
  `computed_masked_macc_add`,
  `runtime_scalar_cmp_masked_macc_add`, and `widening_macc_add`.
* Build the contract from existing provider facts and the rebuilt provider
  route description. Do not add new MAcc routes, dtype/LMUL clone batches, or
  source-front-door paths.
* Include provider-owned expected data for:
  route id, memory form, selected config/policy/runtime-control,
  runtime ABI order and parameters, target leaf profile,
  `provider_supported_mirror`, headers, C type summary, route operand binding
  plan/summary, typed compute op, arithmetic kind, accumulator/result layout,
  scalar-broadcast facts, computed-mask mask/passthrough facts, and widening
  source/result facts.
* Include route-consumption requirements for rebuilt route headers, rebuilt
  type mappings, ABI mappings, source provenance, and statement-plan shape so
  target validation can inspect the rebuilt route without choosing MAcc
  semantics locally.
* Rewire target artifact MAcc provider-fact validation to consume the
  provider-owned contract for non-widening and widening MAcc paths.
* Preserve fail-closed behavior for missing, stale, cross-family,
  cross-route, or mismatched provider fields and stale candidate metadata
  mirrors.
* Keep common EmitC/export neutral. Do not move RVV MAcc semantics into common
  EmitC, target artifact metadata, descriptors, route ids, test names, or exact
  intrinsic spelling.
* Do not change route emission, generated C/C++, runtime ABI order, MAcc
  computation, accumulator layout, mask/tail behavior, passthrough/destination
  preservation, correctness, or performance behavior. If any of those change,
  real `ssh rvv` evidence is required.

## Acceptance Criteria

* [x] Production code exposes a provider-owned MAcc route validation contract
      API beyond metadata mirrors.
* [x] The target validator consumes that contract for existing plain,
      scalar-broadcast, computed-mask, runtime-scalar computed-mask, and
      widening MAcc provider-fact validation.
* [x] Target MAcc validation no longer owns local MAcc expected-fact accessors
      or computed-mask fallback constants for fields already represented in
      provider facts.
* [x] Rebuilt route validation still checks provider headers, type mappings,
      runtime ABI mappings, source provenance, and statement-plan shape before
      artifact acceptance.
* [x] Existing MAcc candidate mirror validation still consumes the provider
      metadata mirror contract from the previous round.
* [x] Focused target/plugin tests and focused lit filters for touched MAcc
      fixture families pass.
* [x] Bounded old-authority scan over touched files finds no new legacy
      `RVVI32M1`, `rvv-i32m1`, source-front-door, descriptor/direct-C,
      artifact-name, bare status/support, or exact `i32m1` intrinsic authority
      additions.
* [x] `rtk git diff --check` passes.
* [x] If the module behavior is complete, finish/archive the Trellis task and
      prepare a coherent commit boundary.

## Technical Approach

1. Add a `RVVMAccRouteValidationContract` and helper contract structs in
   `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`.
2. Implement `getRVVMAccRouteValidationContract(...)` in
   `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`, using
   existing provider facts and the rebuilt route description.
3. Rewire `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp` so
   non-widening and widening MAcc target provider-fact validation receives a
   provider contract and compares the rebuilt route/description against it.
4. Keep the previous metadata mirror contract and target candidate-mirror
   consumer intact.
5. Add or adjust focused C++ target/plugin checks only where the new contract
   changes test-observable diagnostics or where direct contract API coverage is
   needed.

## Out Of Scope

* No new MAcc operation families, reductions, compare/select, conversion,
  memory routes, frontend routes, or source-front-door positive routes.
* No dtype/LMUL clone batch, high-level Linalg lowering, dashboards, broad
  smoke matrices, or artifact-only evidence.
* No generated runtime behavior changes.
* No movement of RVV semantics into common EmitC/export or target artifact
  metadata.
* No `ssh rvv` run unless generated runtime behavior or runtime/correctness/
  performance claims change.

## Evidence Plan

* Build and run `tianchenrv-target-artifact-export-test`.
* Build and run `tianchenrv-rvv-extension-plugin-test`, because provider
  headers/plugin code may change.
* Run focused lit filters for touched MAcc route families:
  `(macc-add|scalar-broadcast-macc-add|computed-masked-macc-add|runtime-scalar-cmp-masked-macc-add|widening-macc-add)`.
* Run a bounded old-authority scan over touched files.
* Run `rtk git diff --check`.
* Do not run `ssh rvv` if the change remains a provider/target validation
  contract extraction with no generated runtime semantic change.

## Completion Evidence

Implemented:

* Added `RVVMAccRouteValidationContract`,
  `RVVMAccRouteTypeMappingContract`, `RVVMAccRouteValidationKind`, and
  `getRVVMAccRouteValidationContract(...)` in the RVV provider API.
* Built MAcc route validation contracts from provider-owned facts and the
  rebuilt provider route description for plain, scalar-broadcast,
  computed-mask, runtime-scalar computed-mask, and widening MAcc.
* Rewired target MAcc provider-fact validation to consume the provider contract
  for route payload, header/type, runtime ABI, accumulator/passthrough,
  mask/tail, widening, and statement-plan expectations.
* Removed target-local MAcc expected-fact accessors/constants for fields now
  covered by the provider contract.
* Preserved the previous MAcc metadata mirror contract as a separate candidate
  mirror consumer.
* Added a target-side fail-closed guard requiring the provider MAcc route
  validation contract to carry the exact runtime ABI parameter count for its
  family before statement-plan validation can index ABI parameters.
* Updated `.trellis/spec/lowering-runtime/emitc-route.md` with the executable
  MAcc route validation contract.

Checks run:

* [OK] `rtk cmake --build build --target tianchenrv-target-artifact-export-test tianchenrv-rvv-extension-plugin-test -j 16`
* [OK] `rtk build/bin/tianchenrv-target-artifact-export-test`
* [OK] `rtk build/bin/tianchenrv-rvv-extension-plugin-test`
* [OK] `rtk python3 /usr/lib/llvm-20/build/utils/lit/lit.py -sv . --filter '(macc-add|scalar-broadcast-macc-add|computed-masked-macc-add|runtime-scalar-cmp-masked-macc-add|widening-macc-add)'` from `build/test`, 28 passed.
* [OK] Bounded diff-only old-authority scan over touched production/test files.
* [OK] `rtk git diff --check`

Self-repair:

* Initial positive plain MAcc target validation showed unit-stride plain/scalar
  MAcc descriptions do not carry `sourceMemoryForm` or
  `destinationMemoryForm`; the unit-stride contract now leaves those fields
  empty while computed-mask and widening MAcc still validate memory forms.
* Widening MAcc initially inherited ordinary MAcc accumulator/result layout
  expectations; the widening contract now clears ordinary MAcc layout fields
  and validates widening-specific layout/relation fields.
* Updated focused C++ diagnostics to assert provider contract field names and
  expected/actual values instead of previous target-local wording.
* Adjusted the MAcc route token expectation to derive from the RVV provider
  route helper rather than treating the rebuilt description field as authority.
* Repair round added an explicit MAcc contract runtime ABI parameter-count
  check so missing provider contract ABI fields fail with a targeted diagnostic
  before route statement-plan validation indexes family-specific parameters.
* Repair-round old-authority scan only found explicit `provider-supported
  mirror` labeling and the `acc-i32m1` diagnostic fragment used to prove
  provider binding facts are checked; neither was introduced as route
  authority.

No `ssh rvv` rerun: this changed provider/target validation contract ownership
and diagnostics only. It did not change route emission, generated C/C++,
runtime ABI order, emitted intrinsics, MAcc computation, accumulator layout,
mask/tail policy, passthrough/destination preservation, correctness, or
performance behavior.

## Technical Notes

Specs read:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`

Previous task read:

* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-macc-route-family-provider-contract-extraction/prd.md`
* `.trellis/tasks/archive/2026-06/06-03-06-03-stage2-rvv-macc-route-family-provider-contract-extraction/task.json`

Likely live files:

* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRoutePlanning.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCMAccRouteFamilyPlanOwners.cpp`
* `lib/Target/RVV/RVVTargetArtifactRouteFamilyValidation.cpp`
* `test/Target/TargetArtifactExportTest.cpp`
* focused MAcc fixtures under `test/Target/RVV` and `test/Scripts`
