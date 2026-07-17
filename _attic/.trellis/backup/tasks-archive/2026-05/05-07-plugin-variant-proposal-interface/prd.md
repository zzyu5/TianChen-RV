# plugin variant proposal interface

## Goal

Add the first C++ compiler slice for extension-plugin variant proposal: enabled plugins can be queried with a high-level `mlir::Operation *`, a `tcrv.exec.kernel` anchor, and the generic `support::TargetCapabilitySet`, then deterministically contribute compiler-visible variant proposals through the registry.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- HEAD is expected and observed as `b9a9ea2 feat: make capability requires dispatch-aware`.
- The current worktree was clean before this task was created.
- Existing plugin code exposes identity, capability collection, dialect registration, enablement, lookup, and collection in `include/TianChenRV/Plugin/ExtensionPlugin.h` and `lib/Plugin/ExtensionPlugin.cpp`.
- Existing capability query code exposes `support::TargetCapabilitySet::buildFromKernel(tcrv::exec::KernelOp)`.
- This round must run as one serial worker with no subagents or multi-agent workflow.

## Requirements

- Add target-independent VariantBuilder-style hooks to `ExtensionPlugin` with defaults that preserve current mock plugins and tests.
- The hook must accept or access `mlir::Operation *`, `tcrv::exec::KernelOp`, and `support::TargetCapabilitySet`.
- Add small generic C++ proposal/request/result objects that model compiler-visible metadata directly mappable to `tcrv.exec.variant` and `tcrv.exec.case` attributes.
- Add registry orchestration that iterates enabled plugins in deterministic registration order, asks support first, and collects proposals only from supported enabled plugins.
- Skip disabled plugins and enabled unsupported plugins without invoking proposal generation for them.
- Reject malformed proposals with useful `llvm::Error`, at minimum empty variant names and empty origin/plugin ownership.
- Keep orchestration generic: no RVV, IME, Sophgo, AME, offload, scalar, vendor, or target-family branches.
- Do not implement concrete plugins, selection/cost/tuning, lowering, emission, runtime ABI, or IR rewriting in this task.
- Preserve existing capability collection, dialect registration behavior, and tests.

## Acceptance Criteria

- [x] Existing plugin registry, capability model, exec dialect, and capability-requires tests pass.
- [x] New C++ smoke test covers enabled supported proposal contribution.
- [x] New C++ smoke test covers disabled plugin skip.
- [x] New C++ smoke test covers enabled unsupported skip and proves proposal hook is not called.
- [x] New C++ smoke test covers registration-order proposal ordering.
- [x] New C++ smoke test parses/builds real MLIR with `tcrv.exec.kernel` and capabilities, builds `TargetCapabilitySet`, and passes a real `mlir::Operation *`.
- [x] New C++ smoke test proves generic capability-based support without target-family branches.
- [x] New C++ smoke test verifies invalid proposal metadata returns an error with enough plugin/debug context.
- [x] lit wrapper runs the C++ test under `check-tianchenrv`.
- [x] Local configure/build/check and `git diff --check` pass.
- [x] Trellis task validation passes before archive if supported.

## Out Of Scope

- Concrete RVV, IME, offload, AME, scalar, or vendor plugin implementation.
- Variant selection, cost model, tuning, lowering, emission, runtime ABI, performance path, or IR rewrite/materialization.
- Any Python implementation of compiler interfaces, registry orchestration, legality, selection, lowering, or emission.
- Any RVV runtime/correctness/performance claim or `ssh rvv` evidence.

## Technical Notes

- Primary implementation files should remain `include/TianChenRV/Plugin/ExtensionPlugin.h` and `lib/Plugin/ExtensionPlugin.cpp` unless a focused `VariantProposal` file becomes necessary.
- `TianChenRVPlugin` may need to link `TianChenRVSupport`; `Support` must not depend on `Plugin`.
- Relevant specs: plugin protocol interfaces/registry and locality, variant pipeline generation/selection/tuning, capability contract, core dialect contract, MLIR testing contract.
