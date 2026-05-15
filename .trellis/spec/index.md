# TianChen-RV MLIR Trellis Specs

这些文件是 TianChen-RV MLIR 的长期项目规范。它们约束后续设计、代码实现、实验解释和 agent 接手方式。

项目定位：

```text
TianChen-RV MLIR:
Unified TCRV RISC-V MLIR for Capability-Scoped Extension Execution
```

TianChen-RV MLIR 不是新的高层 tensor/tile IR，也不是一个硬件一个互不相关 backend dialect 的集合。它位于 high-level MLIR 之后，把 RISC-V 系统能力建模为 MLIR 可查询、可验证、可参与 pass 决策的对象，并在一个统一 TCRV dialect suite 内组织 capability-scoped extension execution。

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
- TianChen-RV 是 unified RISC-V MLIR；RVV、IME、TensorExt、Offload 和未来 vendor/custom targets 是 TCRV extension families，不是独立 backend dialect。
- `tcrv.exec` 只表达 execution envelope：kernel、target、capability、variant、requires、region、hart_parallel、mem_window、runtime_param、dispatch、fallback、diagnostics。
- `tcrv.exec` 不表达 matmul、softmax、reduce、generic tile、generic tensor compute。
- `tcrv.exec.kernel` 表示 callable execution plan envelope，不是数学 kernel、算子 IR 或硬件 IR 主体。
- 计算语义和硬件执行 op 属于 TCRV extension families，例如 RVV、IME、TensorExt、Offload 或未来插件 family。
- 核心 pass 通过 capability registry 和 plugin interfaces 调用插件，不硬编码 RVV、IME、Sophgo 或其他扩展名称。
- Common passes must operate through TCRV interfaces such as extension/config/resource/memory/compute/EmitC-lowerable interfaces, not family-name branches.
- Current main lowering route is extension family ops -> EmitC ops -> intrinsic/vendor builtin/runtime C/C++ -> native compiler; clang/LLVM is default, GCC is compatible.
- Descriptor-driven computation is invalid architecture. Existing
  descriptor/microkernel/direct-C paths are historical residue, deletion
  targets, or fail-closed implementation debt; they must not be described as a
  transition architecture. Future executable work goes through extension family
  ops and common EmitC lowering.
- 插件化表示新增扩展代码应局部封装，不表示新增硬件零工作量。
- 当前真实主线硬件是 RVV 1.0 环境，通过 `ssh rvv` 访问，64 核 CPU，具备 sudo 权限。
- K3/IME 是后续 IME plugin 接入对象，用于验证新增 extension plugin 的局部化接入。
- Sophgo/RISC-V + offload 必须建模为 runtime-offload capability，不能伪装成 RISC-V custom ISA extension。
- AME、future custom ISA 和其他 vendor extension 只作为未来插件槽位，不能写成当前必须完成的主硬件路径。
- Retuning 是 capability-aware variant selection and tuning 的系统能力，不是单独夸大的理论创新。
- MLIR 行为必须用 lit/FileCheck 覆盖 dialect syntax、parsing、verification 和 passes；必要时补充 C++ tests 和 CMake configure/build checks。
- 实验参考用于验证系统结构，不反向决定系统结构。

## Source Of Truth

`.trellis/spec/` 是 TianChen-RV MLIR 的长期规范源。早期
`predoc/tianchen_rv_mlir_capability_pack/` 设计包已被抽取并归并到本
spec 树中，当前仓库不再把 `predoc/` 作为 live source of truth。

后续 agent 应以本 index 和各层 spec 为准；如果历史任务记录、旧 prompt 或
外部说明提到 `predoc/`，只能把它理解为已迁移的历史输入，不能再要求读取或
维护该目录。
