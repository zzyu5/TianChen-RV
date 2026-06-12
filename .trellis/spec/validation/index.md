# Validation Specs

This layer defines how experiments should be interpreted. It is validation guidance, not system structure.

## Pre-Development Checklist

- [ ] Does the experiment validate an existing system contract rather than redefining one?
- [ ] Is RVV evidence tied to the `ssh rvv` RVV 1.0 main environment or another named profile?
- [ ] Are offload results described as runtime-offload capability evidence?
- [ ] Are IME results treated as plugin-local extension validation only after hardware/toolchain availability?
- [ ] Does the scenario test the real RVV trunk rather than a not-yet-built frontend/offload/IME/future family? (见 [../guides/trunk-discipline.md](../guides/trunk-discipline.md))
- [ ] Is Linalg/Vector used as comparison or coverage calibration rather than current source-route authority?
- [ ] Are ablations tied to capability model, plugin locality, and variant selection rather than ad hoc claims?
- [ ] Are MLIR behavior claims backed by lit/FileCheck or C++ tests?

## Guidelines Index

| Spec | Description |
|---|---|
| [Experiment Reference](./experiment-reference.md) | Research questions, metrics, and forbidden interpretations |
| [Testing Specs](../testing/index.md) | Required compiler test forms |

## Quality Check

- Experiment docs must not reverse-engineer system structure around a convenient result.
- Results should report capability profile, selected variants, legality/dispatch behavior, fallback behavior, and reproducibility metadata.
- Claims must distinguish architecture evidence, runtime evidence, performance evidence, and plugin-locality evidence.
- RVV runtime/performance/correctness claims must include real `ssh rvv` evidence.
- Validation scenarios validate structure; they do not authorize building frontend, offload, IME, source-front-door, or future-family paths (实验验证结构，不反向定义结构，见 core-invariants I9)。
