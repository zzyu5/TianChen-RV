<!-- TRELLIS:START -->
# Trellis Instructions

These instructions are for AI assistants working in this project.

Use the `/trellis:start` command when starting a new session to:
- Initialize your developer identity
- Understand current project context
- Read relevant guidelines

Use `@/.trellis/` to learn:
- Development workflow (`workflow.md`)
- Project structure guidelines (`spec/`)
- Developer workspace (`workspace/`)

If you're using Codex, project-scoped helpers may also live in:
- `.agents/skills/` for reusable Trellis skills
- `.codex/agents/` for optional custom subagents

Keep this managed block so 'trellis update' can refresh the instructions.

<!-- TRELLIS:END -->

## Project Scope

This project is **TianChen-RV MLIR: A Capability-Driven Execution Layer for Extensible RISC-V AI Kernels**.

Before changing design, code, experiments, or task plans, read `.trellis/spec/index.md` and the relevant spec layer under `.trellis/spec/`.

The stable project boundary is:

- TianChen-RV is a capability-driven RISC-V execution layer after high-level MLIR, not a new high-level tensor/tile IR.
- The `tcrv.exec` core dialect only expresses kernel, target, capability, variant, hart_parallel, dispatch, and fallback.
- Computation and hardware execution details belong in extension dialects such as `tcrv.rvv`, `tcrv.ime`, `tcrv.offload`, or future plugin dialects.
- The current real hardware mainline is RVV 1.0 via `ssh rvv`, with a 64-core CPU and sudo access.
- K3/IME is a later IME plugin integration target.
- Sophgo/RISC-V + offload is a runtime-offload capability, not a custom RISC-V ISA extension.

@RTK.md
