# brainstorm: unified TCRV spec and prompt update

## Goal

Update long-term TianChen-RV specs and supervisor/Codex prompts so future work
treats TianChen-RV as a unified RISC-V MLIR extension system, not as one
independent backend dialect per hardware target. This task updates system
definitions only; it must not implement compiler functionality.

## What I Already Know

- The user wants this scoped to spec and prompt changes, not compiler code.
- The desired architecture is a unified TCRV RISC-V MLIR with a core execution /
  capability / ABI / dispatch envelope plus RVV, IME, TensorExt, Offload, and
  future vendor/custom extension families.
- `tcrv.exec` should be an execution envelope, not the hardware IR body.
- The current primary lowering route should be extension family ops -> EmitC ->
  intrinsic/runtime C/C++ -> native compiler.
- Descriptor-driven computation must not be treated as a long-term
  architecture.

## Requirements

- Update durable specs under `.trellis/spec/`.
- Define unified extension family terminology and core/extension boundaries.
- Define common TCRV interfaces and common pass expectations.
- Define an Extension Family Plugin Template and extension manifest shape.
- Define a unified EmitC route and mark direct descriptor-to-C emission as
  implementation debt.
- Update Hermes and Codex prompts with only necessary supervision rules.
- Do not touch compiler implementation, lowering, microkernels, or performance
  experiments.

## Acceptance Criteria

- [ ] Specs contain unified RISC-V MLIR / extension family terminology.
- [ ] Specs define common interfaces and common pass reuse.
- [ ] Specs define extension family ops -> EmitC as the current main route.
- [ ] Specs state descriptor-driven computation is invalid as architecture.
- [ ] Specs include Extension Manifest and Extension Plugin Template guidance.
- [ ] Hermes prompt checks extension-family, interface/common-pass, EmitC, and
      descriptor drift.
- [ ] Codex base prompt includes the concise long-term red lines.
- [ ] Grep validation confirms required keywords.

## Out Of Scope

- No compiler feature implementation.
- No new lowering.
- No new microkernel.
- No performance experiment.
- No code rename from existing concrete namespaces such as `tcrv_rvv`.

## Technical Notes

- Current prompt files:
  - `scripts/codex_serial_supervisor.py`
  - `scripts/codex_serial_supervisor_prompt.md`
- Current spec roots:
  - `.trellis/spec/index.md`
  - `.trellis/spec/architecture/`
  - `.trellis/spec/core-dialect/`
  - `.trellis/spec/plugin-protocol/`
  - `.trellis/spec/lowering-runtime/`
