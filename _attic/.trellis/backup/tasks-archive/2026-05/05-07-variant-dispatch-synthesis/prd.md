# dispatch synthesis for materialized variants

## Goal

Add a generic C++/MLIR compiler path that synthesizes `tcrv.exec.dispatch` from already materialized sibling `tcrv.exec.variant` operations inside each `tcrv.exec.kernel`, so capability-aware dispatch becomes the next spine step after plugin proposal materialization.

## What I already know

* HEAD is expected at `42213df feat: materialize plugin variant proposals`.
* The current gap is reusable dispatch synthesis from materialized variants; proposal collection and variant materialization already exist.
* Implementation must remain C++ / MLIR / LLVM / TableGen / CMake / lit / C++ tests.
* Python is allowed only for Trellis/script/support flow, not core IR/pass/model implementation.
* `tcrv.exec` must stay execution/capability/variant/dispatch/fallback focused and compute-free.
* Extension details must remain plugin-local; dispatch synthesis must be target-neutral.

## Requirements

* Add a bounded public C++/MLIR API for variant dispatch synthesis.
* Add a real `tcrv-opt` pass, preferably `--tcrv-synthesize-variant-dispatch`.
* Scan each `tcrv.exec.kernel` and direct child `tcrv.exec.variant` ops in deterministic IR order.
* Build/use `TargetCapabilitySet` from each kernel.
* Synthesize only structurally valid `tcrv.exec.dispatch` IR.
* Use the first direct variant whose requirements are generically available as fallback.
* Emit cases for remaining direct variants in original kernel order.
* Add a non-empty generic guard/condition/policy for cases whose target has unavailable requirements.
* Do not synthesize if no available fallback exists; emit a clear target-neutral diagnostic and fail the pass.
* Do not create invalid dispatch for fewer than two variants; a single unavailable variant must report the fallback diagnostic path.
* Do not duplicate dispatch if a kernel already has one; leave unchanged and test the behavior.
* Preserve existing `CheckCapabilityRequires` behavior and diagnostics.

## Acceptance Criteria

* [ ] Positive lit tests cover dispatch creation, case/fallback references, first available fallback, deterministic order, guard insertion, piped capability check success, and existing-dispatch no-duplicate behavior.
* [ ] Negative lit tests cover no available fallback, single unavailable variant, and malformed requires not being papered over.
* [ ] If a public helper API is added, C++ tests inspect typed `tcrv.exec.dispatch`, `tcrv.exec.case`, and `tcrv.exec.fallback` ops.
* [ ] Existing plugin registry/proposal, materialization, capability model, exec dialect, and capability-requires tests continue to pass.
* [ ] Trellis task is archived before final commit.

## Out of Scope

* Cost/tuning, extension legality, lowering, emission, runtime ABI, RVV/IME/offload/scalar plugin bodies, hardware probes, and target-family-specific branches.
* RVV runtime/correctness/performance evidence.

## Technical Notes

* User requested required repository inspection before editing.
* User requested one serial full-access non-TUI worker and no subagents/multi-agent workflows.
