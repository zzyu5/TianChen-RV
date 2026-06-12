# compact RVV parameter-flow policy update

## Goal

Update the durable project policy for RVV, lowering, runtime, variant, and supervisor review so future work keeps hardware facts, compile-time variant configuration, runtime SSA/control values, and descriptor-local fixture parameters explicitly separated.

## What I Already Know

- Repository root is `/home/kingdom/phdworks/TianchenRV`.
- Initial live status was clean at `f8a66b6 feat: emit RVV scalar dispatch C artifact`.
- The supervisor-policy files `.trellis/spec/implementation-stack/supervision-loop.md`, `scripts/codex_serial_supervisor.py`, and `scripts/codex_serial_supervisor_prompt.md` were clean at start, so no separate cleanup commit is expected.
- This round is a compact policy/spec/prompt update, not another smoke/probe/guardrail/test/evidence/export-helper task and not another compiler dispatch implementation slice.
- The worker must run serially without subagents or multi-agent workflows.

## Requirements

- Add or tighten a durable parameter layering rule:
  - Hardware facts / target capabilities: VLEN, vlenb-derived capacity, ISA/profile facts, hart/core count, toolchain availability, remote probe evidence, and capability provenance belong in capability/profile/probe objects and constrain legality/selection.
  - Compile-time variant configuration: SEW, LMUL, tail/mask policy, unroll, selected lowering strategy, and other plugin-proposed knobs belong in variant/config/lowering-boundary metadata and must be checked against target capabilities.
  - Runtime SSA/control-plane/ABI values: AVL, vl, pointer arguments, length n, rvv_available/dispatch guard parameters, and other runtime-only values must be modeled as SSA values, region/block arguments, explicitly ABI/control attributes, or generated C ABI parameters.
  - Descriptor-local bounded fixture parameters such as current `element_count` describe finite emitted source slices or test descriptors only, and must not masquerade as tensor shape, global problem size, AVL, or vl.
- Add compact RVV-specific guidance for VLEN/vlenb, SEW/LMUL, AVL/vl, `element_count`, `required_march`, `setvl`, and `with_vl`.
- Add supervisor/Codex prompt review checks for parameter layering confusion and unnecessary growth of `required_march` string-comparison dependence.
- Keep the patch compact and non-duplicative. Do not edit compiler implementation unless live evidence proves the policy update impossible.

## Acceptance Criteria

- [x] Modified specs/prompts define the parameter layering rule and RVV-specific boundaries.
- [x] Supervisor prompt or runner policy tells Hermes/Codex review to audit the same boundaries.
- [x] Readback checks cover the requested terms and concepts.
- [x] `git diff --check` passes.
- [x] Prompt rendering/no-exec validation is run if the supervisor prompt changes.
- [x] No compiler implementation, smoke/probe/guardrail package, build artifact, or unrelated Trellis scratch is included in the parameter-flow commit.
- [x] Task is archived after the work and archive path is validated.

## Out Of Scope

- C++/ODS/MLIR implementation changes unless unavoidable.
- RVV SSH hardware compile/run/correctness/performance validation.
- New tests, probes, dashboards, evidence exports, guardrails, or helper-only scripts.
- Expanding Python into compiler core IR/dialect/pass/registry/capability/variant/lowering decisions.
