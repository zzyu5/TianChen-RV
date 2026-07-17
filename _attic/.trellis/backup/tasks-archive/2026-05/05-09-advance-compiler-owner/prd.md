# advance compiler owner from supervisor review

## Goal

Implement module-level target profile capability providers for TianChen-RV.
After this round, a kernel can reference a module-level `tcrv.exec.target`
profile through `target = @profile`, and the referenced profile participates in
`TargetCapabilitySet`, plugin proposal, variant materialization, legality, and
emission metadata checks. This makes the compiler target decision object less
kernel-local and more reusable without adding extension-specific core logic.

## What I already know

* Repo root is `/home/kingdom/phdworks/TianchenRV`.
* Current branch is `main`; initial worktree state was clean.
* Latest observed HEAD before this task was `c5b7cfc feat: let target profiles provide capabilities`.
* The latest supervisor run directory by mtime is
  `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0085-20260508T213319Z`.
* That latest run added target-profile capability provider behavior and archived
  its Trellis task.
* Current task prompt requires a single full-access non-TUI worker and explicitly
  forbids subagents, spawned agents, parallel agents, and multi-agent workflows.
* Project specs require the primary implementation to stay in
  C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck, with Python limited to runners,
  remote probes, artifact parsing, and small support utilities.
* `tcrv.exec` must remain execution/capability/variant focused and compute-free.
* Extension-specific behavior must stay plugin-local through generic registry
  and interface surfaces.
* New RVV runtime/correctness/performance claims require real `ssh rvv` evidence.

## Assumptions

* The previous target-profile provider milestone is complete enough not to redo.
* Since no current user/Hermes task was appended, this round should choose the
  highest-value next owner from the existing code and specs.
* Trellis task metadata is used for local continuity, but implementation and
  verification will be done serially in this worker because the user explicitly
  disabled subagents.

## Requirements

* Support `tcrv.exec.kernel @k attributes {target = @profile}` when
  `@profile` is a direct module-level `tcrv.exec.target` with capability
  provider identity (`id` and `kind`).
* Preserve existing kernel-local capability and target-provider behavior.
* Do not collect all module-level targets implicitly; only the explicitly
  referenced target profile belongs to the kernel capability set.
* Reject missing, non-target, or parse-only referenced target profiles with
  clear diagnostics.
* Let plugin proposal materialization map required capability IDs through the
  referenced module-level target profile.
* Keep hardware facts / target capability, compile-time variant config, runtime
  SSA/control values, and descriptor-local fixture parameters separated.
* Update durable specs before or together with implementation when behavior
  changes.
* Add only minimal focused tests for the changed compiler behavior.
* Create one coherent commit if the round completes successfully.

## Acceptance Criteria

* [ ] Owner selection is grounded in latest audit/review input and current code.
* [ ] `tcrv.exec.kernel target = @profile` is verified as a module-level target
      profile reference.
* [ ] `TargetCapabilitySet::buildFromKernelChecked` includes the referenced
      module-level target profile and still rejects duplicate capability IDs.
* [ ] Built-in RVV plugin materialization can use the referenced module-level
      profile as the required capability provider for `rvv`.
* [ ] Relevant code/spec changes are implemented in the repo.
* [ ] Focused lit/C++/build checks are run, or missing tools are reported with
      exact diagnostics.
* [ ] Repo ends clean or remaining changes are explicitly reported.
* [ ] Final report follows the requested seven-part format and states invariant
      preservation.

## Definition of Done

* Relevant tests added or updated.
* `check-tianchenrv` or the narrowest sufficient build/test target passes.
* `git diff --check` passes.
* Durable specs updated if behavior changed.
* Commit created for the completed round.

## Out of Scope

* Python-only compiler internals.
* Replacing MLIR/C++/TableGen implementation with Python structures.
* Broad negative fixture matrices or standalone smoke artifacts unless they are
  the single blocker to the chosen compiler step.
* New RVV runtime/correctness/performance claims without `ssh rvv` evidence.
* Subagent, spawned-agent, parallel-agent, or multi-agent workflows.

## Technical Notes

* Latest repo audit reviewed:
  `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0085-20260508T213319Z/repo_audit.md`.
* Latest review input reviewed:
  `artifacts/tmp/hermes_codex_supervisor/runs/20260507T060051Z-r0085-20260508T213319Z/review_input.md`.
* Relevant specs will be selected from `.trellis/spec/` after owner selection.
