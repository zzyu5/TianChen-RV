# Architecture Specs

本层定义 TianChen-RV MLIR 的系统定位和长期边界。进入任何实现或实验解释前，先确认这里的 invariants。

## Pre-Development Checklist

- [ ] 这项工作是否仍然把 TianChen-RV 描述为 high-level MLIR 之后的 RISC-V execution layer？
- [ ] 是否避免新增核心 `tcrv.matmul`、`tcrv.softmax`、`tcrv.reduce`、`tcrv.generic_tile` 这类高层计算 op？
- [ ] 是否把 capability object 放在 pass 决策路径上，而不是作为注释？
- [ ] 是否区分 RVV 当前主线、K3/IME 后续插件、Sophgo runtime offload、future custom ISA 插槽？
- [ ] 是否以当前 RVV-first 链为准：`tcrv.exec` envelope -> selected RVV variant -> typed `tcrv_rvv` body -> RVV plugin route provider -> `TCRVEmitCLowerableRoute` -> common EmitC？
- [ ] RVV dtype/config/operation 是否来自 typed body 或 realized body，而不是 route id、ABI string、artifact name、旧 `i32m1` helper？
- [ ] `mem_window` / `runtime_param` 是否只声明 ABI role，且 selected body 显式 import/consume 参数？
- [ ] 是否没有保留当前正向 legacy `RVVI32M1*` / `rvv-i32m1-*` object/header/bundle route？
- [ ] 是否没有把 emission-plan status、readiness、dashboard、manifest 或 artifact metadata 当作 route/progress authority？
- [ ] 是否没有把实验便利性反向写成系统结构？

## Guidelines Index

| Spec | Description |
|---|---|
| [Unified TCRV RISC-V MLIR](./unified-riscv-mlir.md) | 统一 RISC-V extension IR、extension family、EmitC 主路线和 descriptor 边界 |
| [System Positioning](./system-positioning.md) | 系统定位、研究贡献、模块边界 |
| [Design Boundaries](./design-boundaries.md) | 禁止方向、硬件路线口径、论文表述边界 |

## Quality Check

- 系统叙事必须能用一句话复述为：unified TCRV RISC-V MLIR for capability-scoped extension execution。
- 任何新增核心概念必须说明它属于 capability、variant、plugin、dispatch、fallback 之一，或解释为什么需要扩展核心 interface。
- 文档或代码中出现 high-level compute core op 时，必须视为 architecture violation，除非它明确属于 TCRV extension family 内部执行 op。
- Source-front-door、high-level frontend、IME、Offload、Template/Toy/TensorExtLite 正向流程默认是 Stage3/later；除非明确任务选择，否则不能抢占 Stage1/Stage2 RVV 路线。
