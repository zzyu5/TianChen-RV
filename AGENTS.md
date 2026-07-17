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

## Onboarding: read `.trellis/` only

**Trellis (`.trellis/`) is the single authority** — queue, tasks, spec, issues, and report deliverables all live there.

**Entry point: [`.trellis/spec/index.md`](.trellis/spec/index.md)** (root map: positioning + the six-layer table + the reading order + the N↔C bridge). Follow its **新 agent 上岗顺序** section for the reading order and for when you may start work; it is not restated here (no duplicates).

The six spec layers:

| Layer | Answers |
|---|---|
| [canon](.trellis/spec/canon/index.md) | The **law**: settled rules — may a claim stand, may a number be reported, is a cell a win. Includes [核心不变量](.trellis/spec/canon/核心不变量.md) I1–I9 |
| [measurement](.trellis/spec/measurement/index.md) | **How to measure**: what counts as measurement, which board, which pipeline, against whom, where results land |
| [architecture](.trellis/spec/architecture/index.md) | **Structural law**: what each station of the machine *actually is* in code today, what it should become, what is off-limits |
| [evidence](.trellis/spec/evidence/index.md) | **Evidence map**: which artifact backs which claim, where it lives; how to read a result and which readings are forbidden |
| [governance](.trellis/spec/governance/index.md) | **How to work**: decision-authority card, deferred-ruling rule, queue and briefing, Trellis hygiene, thinking guides |
| [issues](.trellis/spec/issues/index.md) | The **single issue register**: every known gap / pending ruling / debt, as `ISSUE-NNN`, registered here and only here |

Before changing design, code, experiments, or task plans: read the root map, then the layer your task belongs to. Work must be attached to a Trellis task with its scope pre-registered.

## Project Scope

This project is **Weft-RV MLIR: a reference template for a capability-driven, extensible MLIR execution-layer software stack** — with RISC-V quantized LLM inference as its *first high-performance instance*. The headline is the **extensibility** (a reproducible way to organize the stack: capability schema / plugin five-piece protocol / falsifiers / selector skeleton); on-silicon wins over hand-written shipped kernels are the **proof of the template's quality, not the goal itself**.

The authoritative wording of the positioning red line and of the three contributions C1/C2/C3′ lives in the root map and in [canon · 暂定-科研主张](.trellis/spec/canon/暂定-科研主张.md), marked **【暂定 · 随论文侧更新 · 非定论】**. Agents may only downgrade and annotate — **never invent, replace, or "improve" a research claim**.

The stable project boundary is:

- Weft-RV is a capability-driven RISC-V execution-layer software stack (a reference template) after high-level MLIR, not a new high-level tensor/tile IR. **Template ≠ general-purpose compiler**: "extensible" means the *stack-organization* is reproducible (schema/plugins/falsifiers/selector), not that it admits arbitrary workload domains — the input side stops at a kernel-level interface and the load domain stays locked to ggml-style quantized inference kernels.
- The `weft.exec` core dialect only expresses kernel, target, capability, variant, hart_parallel, dispatch, and fallback.
- Computation and hardware execution details belong in extension dialects such as `weft.rvv`, `weft.ime`, `weft.offload`, or future plugin dialects.
- IME is the keystone second extension family (N2), taken through the same common path as RVV — not a bolt-on backend; specific SpacemiT boards (K1/K3, etc.) are profile facts, not scope items.
- Sophgo/RISC-V + offload is a runtime-offload capability, not a custom RISC-V ISA extension.
- The board register and per-board constraints are canonical in [measurement · 板册](.trellis/spec/measurement/板册.md) — not here.

@RTK.md
