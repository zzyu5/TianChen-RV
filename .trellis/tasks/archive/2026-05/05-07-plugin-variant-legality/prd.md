# Plugin-local Materialized Variant Legality

## Goal

Add the first C++/MLIR legality-verifier orchestration slice for already
materialized `tcrv.exec.variant` IR. The core plugin registry should route each
variant to its origin plugin through a typed request object carrying the real
MLIR variant, enclosing kernel, and generic `TargetCapabilitySet`.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV` on branch `main`.
- HEAD is `3729aa0 feat: synthesize variant dispatch`.
- Current spine already supports proposal collection, proposal validation,
  variant materialization, dispatch synthesis, and generic capability-requires
  checks.
- The next live-code gap is plugin-owned legality verification for materialized
  variants.
- This round must be serial and must not use subagents or multi-agent queues.

## Requirements

- Extend `ExtensionPlugin` with a C++/MLIR legality hook for materialized
  `tcrv.exec.variant` IR.
- Add a small request object carrying the variant, enclosing kernel, and generic
  target capability set.
- Add registry APIs that verify one variant and all direct variants in a kernel
  by routing to the variant origin plugin.
- Treat disabled origin plugins as an error for materialized variant legality.
- Keep routing generic: no RVV, IME, Sophgo, offload, scalar, vendor, dtype,
  layout, shape, or target-family branches.
- Preserve existing proposal collection, materialization, dispatch synthesis,
  and capability-requires responsibilities.
- Do not add a public `tcrv-opt` pass with an empty or fake global registry.
- Update durable plugin-protocol spec docs for the first-slice legality hook.

## Acceptance Criteria

- [ ] Known origin plugin receives exactly one legality hook call per variant.
- [ ] Hook sees expected variant symbol, enclosing kernel, and capability set.
- [ ] Kernel verification visits direct variants in deterministic IR order.
- [ ] Successful verification leaves module valid and does not mutate variant or
      dispatch structure.
- [ ] Unknown origin, disabled origin, plugin-local failure, malformed request,
      missing kernel, and missing variant paths produce clear generic failures.
- [ ] Non-origin plugins are not called.
- [ ] C++ test uses real parsed MLIR with `tcrv.exec.capability`,
      `tcrv.exec.kernel`, and `tcrv.exec.variant`.
- [ ] lit/FileCheck wrapper runs the C++ test binary.
- [ ] Existing local checks pass, including `check-tianchenrv`.

## Out Of Scope

- New RVV, IME, offload, Sophgo, scalar, vendor, or target-family plugin work.
- Cost model, tuning, selection, lowering, emission, runtime ABI, or hardware
  evidence.
- Python implementation of core compiler IR, passes, plugin protocols, or
  legality decisions.

## Technical Notes

- Stable project boundary: `tcrv.exec` remains execution/capability/variant/
  dispatch/fallback focused and compute-free.
- Plugin locality contract keeps extension-specific legality behind plugin
  interfaces while the core owns generic verifier orchestration.
- Required validation commands for this round are `git diff --check`, CMake
  configure, and `cmake --build ... --target check-tianchenrv -j2`.
