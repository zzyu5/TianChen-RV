# offload runtime descriptor artifact export

## Goal

Make the runtime-offload descriptor path more concrete by ensuring selected
offload lowering-boundary and emission-plan metadata reach the generic target
artifact front-door preflight before `tcrv-translate --tcrv-export-target-artifact`
emits a deterministic descriptor.

## What I Already Know

* Current HEAD is expected to be `694197a feat: compose target capability providers`.
* The offload descriptor route already exists as
  `tcrv-export-offload-runtime-descriptor` with artifact kind
  `runtime-offload-handoff-descriptor`.
* The route must stay target/exporter-owned and must not claim vendor runtime,
  DMA, accelerator kernel generation, hardware execution, correctness, or
  performance.
* The current implementation already exports a descriptor, but the generic
  execution-plan coherence candidate path does not carry selected-plan handoff
  metadata into route-local validation.

## Requirements

* Preserve `tcrv.exec` as compute-free execution/capability/variant metadata.
* Keep offload-specific descriptor semantics in offload plugin/dialect/target
  exporter code or behind generic target artifact interfaces.
* Carry and validate required offload selected-plan metadata at the target
  artifact front-door preflight, not only in the final descriptor emitter.
* Preserve existing RVV/scalar/provider-composition behavior.
* Add focused lit coverage tied to the offload descriptor compiler/export path.

## Acceptance Criteria

* A hand-written TianChen-RV MLIR module with target-level offload capability
  composition can flow through `tcrv-opt --tcrv-execution-planning-pipeline` and
  `tcrv-translate --tcrv-export-target-artifact` to descriptor text.
* Missing offload selected-plan handoff metadata fails before descriptor output.
* Existing offload plugin tests still pass.
* Required validation commands pass:
  `git diff --check`, `cmake --build build --target tcrv-opt`,
  `cmake --build build --target tcrv-translate`, and
  `cmake --build build --target check-tianchenrv`.

## Out Of Scope

* No ssh rvv run or RVV runtime/correctness/performance claim.
* No vendor runtime calls, DMA behavior, accelerator kernels, object generation,
  linking, or offload hardware execution.
* No Python implementation of compiler semantics.
