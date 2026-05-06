# TianChen-RV MLIR 文档合集

本合集定义 **TianChen-RV MLIR: A Capability-Driven Execution Layer for Extensible RISC-V AI Kernels**。

TianChen-RV MLIR 的目标不是再设计一个高层 tensor/tile IR，而是在 High-level MLIR 之后，提供一个面向 RISC-V AI 内核的能力驱动执行层。它把 RISC-V 系统中的 ISA 扩展、微架构能力、运行时 offload 能力和工具链能力提升为 MLIR 可查询对象，并把同一个 high-level MLIR 算子实现为一组可验证、可选择、可调优、可 lower 的执行变体。

## 文件目录

| 文件 | 内容 |
|---|---|
| `00_overview.md` | 总纲：科研思想、系统目标、最终形态、模块目录 |
| `01_capability_model.md` | Capability model：目标能力、扩展能力、运行时能力如何进入 MLIR |
| `02_exec_core_dialect.md` | `tcrv.exec` core dialect：kernel、variant、capability、dispatch、hart-level execution |
| `03_extension_plugin_protocol.md` | Extension plugin protocol：如何局部加入 RVV、IME、offload、未来扩展 |
| `04_rvv_plugin.md` | RVV plugin 设计：当前主硬件路径 |
| `05_ime_plugin.md` | IME plugin 设计：后续 K3/IME 到位后的 matrix-extension 插件路径 |
| `06_offload_runtime_plugin.md` | RISC-V + offload 设计：Sophgo/vendor runtime capability，不伪装成 custom ISA |
| `07_variant_generation_and_selection.md` | Variant 生成、合法性检查、选择、dispatch 与能力相关调优 |
| `08_lowering_emission_runtime.md` | Lowering、emission、runtime glue、工具链边界 |
| `09_experiment_reference.md` | 实验参考设计：只作为科研验证参考，不反向决定系统结构 |
| `10_trellis_agent_prompt.md` | 可直接复制给 agent 的 prompt：初始化 Trellis 并基于这些文档生成 specs |

## 核心结论

TianChen-RV MLIR 的核心是：

```text
High-level MLIR
    ↓
TianChen-RV MLIR
    = capability model
    + execution variant IR
    + extension plugin protocol
    + extension-specific op sets
    ↓
RVV / IME / offload runtime / future custom ISA emission
```

其中：

- `tcrv.exec` 是稳定核心 dialect，只表达执行组织、目标能力、变体、dispatch、fallback，不表达通用计算语义。
- `tcrv.rvv`、`tcrv.ime`、`tcrv.offload` 是当前规划中的 extension dialect。
- 未来扩展不被限制在固定类别中；它们通过 plugin protocol 接入。
- Capability model 是系统的第一核心：它把 `-march`、VLEN、core count、运行时 accelerator、toolchain 支持等信息变成 MLIR 中可查询、可验证、可参与 pass 决策的对象。
- Retuning 不作为孤立理论点，而作为 capability-aware variant selection and tuning 的系统能力。

