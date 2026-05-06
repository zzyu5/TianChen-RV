# variant proposal materialization

## Goal

Add a bounded generic C++/MLIR materialization API that turns already validated
`VariantProposal` metadata into real `tcrv.exec.variant` operations under the
relevant `tcrv.exec.kernel`.

## What I already know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Current `HEAD` is `72d5e13 feat: validate proposal capability requirements`.
- The current live-code gap is after plugin proposal collection and generic
  capability validation: validated proposal metadata is not yet materialized
  into `tcrv.exec.variant` IR.
- The user requested one serial full-access worker and explicitly disallowed
  subagents, parallel agents, background queues, or multi-agent workflows.
- Primary implementation must remain C++ / MLIR / LLVM / TableGen / ODS /
  CMake / lit / FileCheck or C++ tests.

## Requirements

- Add a reusable C++ transform/materialization-layer API, preferably
  `include/TianChenRV/Transforms/VariantMaterialization.h` and
  `lib/Transforms/VariantMaterialization.cpp`.
- Keep plugin proposal collection in `ExtensionPluginRegistry`; keep IR
  construction in the transform/materialization layer.
- Materialize proposals using MLIR builders and ODS-generated op APIs, not
  string rewriting.
- Preserve generic proposal metadata currently supported by ODS:
  variant symbol/name, origin plugin, and `requires` capability symbol refs.
- Map proposal required capability ids to capability symbol refs using
  `TargetCapabilitySet`; add only the smallest generic support accessor if
  needed.
- Reject malformed inputs before mutating IR when practical: missing kernel,
  duplicate variant symbol, unrepresentable variant symbol, unresolved required
  capability id, unresolved required capability symbol ref, and empty origin.
- Keep diagnostics generic and include plugin/origin, variant name, requirement
  kind, and missing/duplicate symbol or id.
- Preserve existing `CheckCapabilityRequires` behavior.
- Do not implement selection, tuning, lowering, emission, runtime ABI, concrete
  RVV/IME/offload/scalar/vendor plugins, or extension dialect bodies.

## Acceptance Criteria

- [x] C++ API materializes one or more validated proposals as typed
      `tcrv.exec.variant` ops in deterministic proposal order.
- [x] Origin metadata is present on materialized variants.
- [x] Required capability symbol refs are preserved, and required capability ids
      are generically mapped to symbol refs.
- [x] Generated IR can be walked as typed MLIR ops and printed/parsed.
- [x] Generated IR is accepted by the existing check-capability-requires pass
      when requirements are available.
- [x] Negative C++ coverage rejects duplicate variant symbols, unresolved ids,
      unresolved symbols, invalid or empty variant name/origin, and missing
      kernel.
- [x] Existing plugin registry/proposal, capability model, exec dialect, and
      capability-requires tests continue to pass under `check-tianchenrv`.

## Out of Scope

- Public `tcrv-opt` pass plumbing unless it is a real non-empty tested pass.
- Variant selection, cost model, tuning, lowering, emission, runtime ABI, or
  backend work.
- Target-family-specific branches or concrete RVV/IME/offload/scalar/vendor
  plugin implementation.
- Python implementation of IR materialization or compiler model logic.

## Technical Notes

- `tcrv.exec.variant` currently requires `sym_name`, non-empty `origin`, and a
  structured `requires` `ArrayAttr` of `FlatSymbolRefAttr` references resolved
  inside the enclosing `tcrv.exec.kernel`.
- `VariantProposal` already stores variant name, origin plugin, required
  capability ids, required capability symbols, and optional generic
  condition/guard/policy metadata.
- `TargetCapabilitySet` already supports lookup by symbol name and id; a public
  descriptor exposes symbol name, id, kind, status, and availability.
- The first implementation slice will keep generic condition/guard/policy as
  proposal metadata for future dispatch materialization because current
  `tcrv.exec.variant` ODS has no such attributes.
