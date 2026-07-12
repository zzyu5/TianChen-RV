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

This project is **Weft-RV MLIR: a reference template for a capability-driven, extensible MLIR execution-layer software stack** — with RISC-V quantized LLM inference as its *first high-performance instance*. The headline is the **extensibility** (a reproducible way to organize the stack: capability schema / plugin five-piece protocol / falsifiers / selector skeleton); on-silicon wins over hand-written shipped kernels are the **proof of the template's quality, not the goal itself**. (Positioning updated 2026-07-10; canonical one-pager: `docs/canon/Weft-RV_定位-v2.md`.)

Before changing design, code, experiments, or task plans, read `.trellis/spec/index.md` and the relevant spec layer under `.trellis/spec/`.

The stable project boundary is:

- Weft-RV is a capability-driven RISC-V execution-layer software stack (a reference template) after high-level MLIR, not a new high-level tensor/tile IR. **Template ≠ general-purpose compiler**: "extensible" means the *stack-organization* is reproducible (schema/plugins/falsifiers/selector), not that it admits arbitrary workload domains — the input side stops at a kernel-level interface and the load domain stays locked to ggml-style quantized inference kernels.
- The `weft.exec` core dialect only expresses kernel, target, capability, variant, hart_parallel, dispatch, and fallback.
- Computation and hardware execution details belong in extension dialects such as `weft.rvv`, `weft.ime`, `weft.offload`, or future plugin dialects.
- The current real hardware mainline is RVV 1.0 via `ssh rvv`, with a 64-core CPU and sudo access.
- IME is the keystone second extension family (N2), taken through the same common path as RVV — not a bolt-on backend; specific SpacemiT boards (K1/K3, etc.) are profile facts under `capability-model/profiles`, not scope items.
- Sophgo/RISC-V + offload is a runtime-offload capability, not a custom RISC-V ISA extension.

@RTK.md
