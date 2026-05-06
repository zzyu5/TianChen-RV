# TianChen-RV MLIR Trellis Specs

这些文件是 TianChen-RV MLIR 的长期项目规范。它们约束后续设计、代码实现、实验解释和 agent 接手方式。

项目定位：

```text
TianChen-RV MLIR:
A Capability-Driven Execution Layer for Extensible RISC-V AI Kernels
```

TianChen-RV MLIR 不是新的高层 tensor/tile IR。它位于 high-level MLIR 之后，把 RISC-V 系统能力建模为 MLIR 可查询、可验证、可参与 pass 决策的对象，并把 high-level MLIR 算子实现组织为多个 execution variants。

## Spec Layers

| Layer | Purpose |
|---|---|
| [implementation-stack](./implementation-stack/index.md) | C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck 技术栈边界 |
| [architecture](./architecture/index.md) | 系统定位、研究边界、禁止误写的方向 |
| [capability-model](./capability-model/index.md) | target capability object、关系、profile、verifier 输入 |
| [core-dialect](./core-dialect/index.md) | `tcrv.exec` core dialect 的长期契约 |
| [plugin-protocol](./plugin-protocol/index.md) | extension plugin registry/interface/locality 规则 |
| [extension-plugins](./extension-plugins/index.md) | RVV、IME、runtime offload、future plugin 的边界 |
| [variant-pipeline](./variant-pipeline/index.md) | variant generation、legality、selection、dispatch、tuning |
| [lowering-runtime](./lowering-runtime/index.md) | plugin-owned lowering/emission/runtime glue 边界 |
| [testing](./testing/index.md) | MLIR dialect/pass/build/runtime evidence testing policy |
| [validation](./validation/index.md) | 实验参考和证据解释口径 |
| [guides](./guides/index.md) | 开发前设计检查清单 |

## Global Invariants

- Primary compiler stack 是 C++ / MLIR / LLVM / TableGen / ODS / CMake / lit / FileCheck。
- Python 只允许用于 runner、supervisor、remote probe、artifact parsing 和小型支持脚本。
- Python 禁止作为 core IR、dialect、passes、plugin registry、capability model、lowering 或 emission pipeline 的实现语言。
- 如果本地 MLIR/LLVM tools 不可用，应添加检测和诊断，不得用 Python data structures 替代真实 MLIR compiler internals。
- Capability model 是系统第一核心，不是字符串 metadata。
- `tcrv.exec` 只表达 kernel、target、capability、variant、requires、region、hart_parallel、mem_window、dispatch、fallback、diagnostics。
- `tcrv.exec` 不表达 matmul、softmax、reduce、generic tile、generic tensor compute。
- 计算语义和硬件执行 op 属于 extension dialect，例如 `tcrv.rvv`、`tcrv.ime`、`tcrv.offload` 或未来插件 dialect。
- 核心 pass 通过 capability registry 和 plugin interfaces 调用插件，不硬编码 RVV、IME、Sophgo 或其他扩展名称。
- 插件化表示新增扩展代码应局部封装，不表示新增硬件零工作量。
- 当前真实主线硬件是 RVV 1.0 环境，通过 `ssh rvv` 访问，64 核 CPU，具备 sudo 权限。
- K3/IME 是后续 IME plugin 接入对象，用于验证新增 extension plugin 的局部化接入。
- Sophgo/RISC-V + offload 必须建模为 runtime-offload capability，不能伪装成 RISC-V custom ISA extension。
- AME、future custom ISA 和其他 vendor extension 只作为未来插件槽位，不能写成当前必须完成的主硬件路径。
- Retuning 是 capability-aware variant selection and tuning 的系统能力，不是单独夸大的理论创新。
- MLIR 行为必须用 lit/FileCheck 覆盖 dialect syntax、parsing、verification 和 passes；必要时补充 C++ tests 和 CMake configure/build checks。
- 实验参考用于验证系统结构，不反向决定系统结构。

## Source Documents

这些 specs 从 `predoc/tianchen_rv_mlir_capability_pack/` 下列设计文档抽取：

- `README.md`
- `00_overview.md`
- `01_capability_model.md`
- `02_exec_core_dialect.md`
- `03_extension_plugin_protocol.md`
- `04_rvv_plugin.md`
- `05_ime_plugin.md`
- `06_offload_runtime_plugin.md`
- `07_variant_generation_and_selection.md`
- `08_lowering_emission_runtime.md`
- `09_experiment_reference.md`

如果这些 source documents 与本 spec 冲突，后续 agent 应先报告冲突，不要自行混合两套口径。
