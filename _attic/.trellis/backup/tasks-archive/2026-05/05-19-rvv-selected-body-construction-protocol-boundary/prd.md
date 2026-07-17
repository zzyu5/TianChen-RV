# RVV selected-body construction protocol boundary

## Goal

Make the RVV construction protocol and RVV target artifact bridge speak in
selected-body terms. The production path should be:

```text
selected tcrv.exec RVV variant
  -> explicit typed tcrv_rvv body description
  -> RVV selected-body construction protocol
  -> provider-built TCRVEmitCLowerableRoute
  -> target artifact bridge that validates candidate metadata as mirrors
```

The retained SEW32 / LMUL m1 arithmetic route remains a bounded implementation
specialization of that selected-body contract. It must not remain the public
route-construction or target-artifact authority.

## What I Already Know

* Repo root is `/home/kingdom/phdworks/TianchenRV`.
* Initial repo checks for this round:
  * `pwd` -> `/home/kingdom/phdworks/TianchenRV`
  * `git status --short` -> clean
  * `git log --oneline -8` begins with
    `aa14b18 rvv: use selected-body route authority`
* No `.trellis/.current-task` existed, so this task was created from the
  Hermes direction brief.
* The previous archived task
  `.trellis/tasks/archive/2026-05/05-19-rvv-typed-selected-body-route-authority/prd.md`
  moved `RVVEmitCRouteProvider` toward
  `RVVSelectedBodyEmitCRouteDescription`.
* Current specs require RVV authority to flow from selected `tcrv.exec`
  variant and explicit typed `tcrv_rvv` body into RVV-owned route construction,
  then into common EmitC/export mechanics.
* `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h` still exposes
  `RVVI32M1ArithmeticConstructionRoute`,
  `RVVI32M1ArithmeticTargetArtifactMapping`, lookup APIs by old mnemonic /
  operation / route id, and `getRVVI32M1Arithmetic*` public construction APIs.
* `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp` still validates
  the route table around finite `RVVI32M1Arithmetic*` records and artifact
  metadata keyed by old route names.
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp` already analyzes the
  selected typed body but still internally consults old
  `RVVI32M1ArithmeticConstructionRoute` records for route id/runtime ABI
  labels.
* `lib/Target/RVV/RVVTargetSupportBundle.cpp` already rebuilds the provider
  description before artifact validation, but its construction/adapter naming
  and metadata helpers still expose i32m1 arithmetic as the route-family
  contract.

## Requirements

* Introduce or rename RVV construction protocol public data/API so the
  construction route contract is selected-body-owned:
  operation kind, typed compute op, SEW/LMUL bounded config, memory form,
  runtime ABI contract, route labels, artifact handoff labels, and metadata
  mirror fields are selected-body construction facts.
* Keep old i32m1 route names only as bounded implementation labels for the
  current selected-body specialization. They may remain in local constants or
  emitted labels where the selected body has already authorized them.
* Remove public `RVVI32M1Arithmetic*` ownership from construction route and
  target artifact mapping structs/functions where those names present the old
  route family as the authority.
* Rewire RVV provider and RVV target bridge callers to consume selected-body
  construction metadata/API instead of lookup-by-old-route APIs.
* Target artifact validation must treat route id, runtime ABI name,
  arithmetic metadata, source metadata, and artifact names as mirrors of the
  provider-selected body description.
* Stale route id/arithmetic/source/front-door metadata must not authorize
  object/header/bundle export when the selected typed body/provider description
  is absent or inconsistent.
* Keep common EmitC/export code semantic-neutral; do not move RVV operation,
  dtype, config, memory-form, or intrinsic selection into common code.

## Acceptance Criteria

* [ ] Construction protocol public structs/functions no longer make
      `RVVI32M1Arithmetic*` the owner for route construction or target artifact
      mapping.
* [ ] Provider route construction still derives `RVVSelectedBodyEmitCRouteDescription`
      from explicit typed `tcrv_rvv` body/config/runtime facts.
* [ ] RVV target artifact export validates candidate metadata as mirrors of
      the provider-selected body description before object/header/bundle
      output.
* [ ] A metadata-only or stale route-id/arithmetic metadata path fails closed
      without authorizing export.
* [ ] Focused positive coverage proves the retained bounded selected-body
      route still validates.
* [ ] Focused negative coverage proves stale construction/artifact metadata
      authority fails closed.
* [ ] A bounded ref-scan over directly touched RVV construction/target files
      shows remaining `RVVI32M1` names are implementation specializations or
      legacy test assertions, not public construction/target route authority.

## Out Of Scope

* No new RVV coverage: no broadcast, compare/select, reduction, conversion,
  dtype, LMUL, source-shape, or intrinsic expansion beyond preserving already
  retained behavior.
* No high-level Linalg/Vector/StableHLO frontend generalization.
* No scalar, IME, Offload, TensorExt, autotuning, dashboards, broad smoke
  matrices, status machinery, or report-only work.
* No common EmitC/export RVV semantic branches.
* No descriptor-driven computation, direct-C semantic export, or compatibility
  wrapper preserving old i32 route authority.
* No runtime/correctness/performance claims without real `ssh rvv` evidence.

## Definition Of Done

* Code implements the selected-body construction/target API boundary.
* Focused C++/lit checks covering touched construction/provider/target
  behavior pass.
* `git diff --check` passes.
* Task state is truthful; if finished, archive using Trellis task convention.
* One coherent commit records the completed bounded round.

## Technical Notes

Relevant specs and prior task:

* `.trellis/spec/index.md`
* `.trellis/spec/extension-plugins/rvv-plugin.md`
* `.trellis/spec/lowering-runtime/emitc-route.md`
* `.trellis/spec/lowering-runtime/emission-runtime-contract.md`
* `.trellis/spec/plugin-protocol/interfaces-and-registry.md`
* `.trellis/tasks/archive/2026-05/05-19-rvv-typed-selected-body-route-authority/prd.md`

Relevant implementation areas:

* `include/TianChenRV/Plugin/RVV/RVVConstructionProtocol.h`
* `lib/Plugin/RVV/Construction/RVVConstructionProtocol.cpp`
* `include/TianChenRV/Plugin/RVV/RVVEmitCRouteProvider.h`
* `lib/Plugin/RVV/EmitC/RVVEmitCRouteProvider.cpp`
* `lib/Plugin/RVV/RVVExtensionPlugin.cpp`
* `lib/Target/RVV/RVVTargetSupportBundle.cpp`
* Focused tests under `test/Plugin` and `test/Target`.
