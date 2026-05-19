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
- `tcrv.exec` 不表达 matmul、softmax、reduce、generic tile、generic tensor compute，也不拥有 selected route、RVV dtype、RVV schedule、intrinsic spelling 或 compute semantics。
- `tcrv.exec.kernel` 表示 callable execution plan envelope，不是数学 kernel、算子 IR 或硬件 IR 主体。
- 计算语义和硬件执行 op 属于 TCRV extension families，例如 RVV、IME、TensorExt、Offload 或未来插件 family。
- 核心 pass 通过 capability registry 和 plugin interfaces 调用插件，不硬编码 RVV、IME、Sophgo 或其他扩展名称。
- Common passes must operate through TCRV interfaces such as extension/config/resource/memory/compute/EmitC-lowerable interfaces, not family-name branches.
- Current RVV-first authority chain is:

  ```text
  tcrv.exec envelope
    -> selected RVV variant
    -> typed low-level tcrv_rvv vector-level body
    -> RVV plugin-owned legality / selected-body realization / route provider
    -> TCRVEmitCLowerableRoute
    -> common EmitC materializer
    -> RVV intrinsic C/C++ or equivalent backend representation
    -> target artifact
    -> ssh rvv evidence when runtime/correctness/performance is claimed
  ```

- Current main lowering route is selected extension-family body -> plugin-built `TCRVEmitCLowerableRoute` -> common EmitC ops -> intrinsic/vendor builtin/runtime C/C++ -> native compiler; clang/LLVM is default, GCC is compatible.
- Descriptor-driven computation is invalid architecture. Existing
  descriptor/microkernel/direct-C paths are historical residue, deletion
  targets, or fail-closed implementation debt; they must not be described as a
  transition architecture. Future executable work goes through extension family
  ops and common EmitC lowering.
- RVV dtype/config/operation authority must be structural in typed `tcrv_rvv`
  body/config or consumed into a realized `tcrv_rvv` body before route
  construction. Do not infer dtype/config/operation from C ABI strings,
  parameter names, route ids, artifact names, test names, exact intrinsic
  spellings, or old dtype-prefixed helper names.
- `tcrv.exec.mem_window` and `tcrv.exec.runtime_param` declare ABI/runtime
  roles. The selected typed extension body must explicitly import and consume
  those values through typed control/dataflow ops. `tcrv.exec` must not infer
  compute from ABI roles.
- Emission-plan diagnostics, readiness/status reports, route ids, artifact
  metadata, manifests, semantic role graphs, source-front-door metadata, and
  dashboards are mirrors or planning aids only. They are never route,
  compute, dtype, progress, or evidence authority.
- 插件化表示新增扩展代码应局部封装，不表示新增硬件零工作量。
- 当前真实主线硬件是 RVV 1.0 环境，通过 `ssh rvv` 访问，64 核 CPU，具备 sudo 权限。
- K3/IME 是后续 IME plugin 接入对象，用于验证新增 extension plugin 的局部化接入。
- Sophgo/RISC-V + offload 必须建模为 runtime-offload capability，不能伪装成 RISC-V custom ISA extension。
- AME、future custom ISA 和其他 vendor extension 只作为未来插件槽位，不能写成当前必须完成的主硬件路径。
- Retuning 是 capability-aware variant selection and tuning 的系统能力，不是单独夸大的理论创新。
- MLIR 行为必须用 lit/FileCheck 覆盖 dialect syntax、parsing、verification 和 passes；必要时补充 C++ tests 和 CMake configure/build checks。
- 实验参考用于验证系统结构，不反向决定系统结构。

## RVV-First Stage Order

### Stage 1: RVV Route-Authority Reset

Replace or fail-close active compiler paths that still treat bounded `i32m1`
arithmetic, `RVVI32M1*` route specs/slices, finite `tcrv_rvv.i32_*` ops,
route ids, artifact names, source-front-door patterns, descriptors, or exact
`__riscv_*_i32m1` spellings as RVV architecture authority. Stage 1 is not done
while any active path preserves a supported legacy `rvv-i32m1-*` object/header
or bundle route.

### Stage 2: Corrected Typed RVV Coverage And Realization

Expand route-supported RVV coverage on the corrected typed low-level
`tcrv_rvv` surface. Coverage is calibrated by Linalg-like structured
computation classes, but this is not a current Linalg frontend phase. Stage 2
also contains RVV plugin-local selected-body realization:

```text
selected pre-realized tcrv_rvv body
  + target capability
  + runtime SSA / ABI values
  + optional hints / policy / profile
    -> realized tcrv_rvv selected body
    -> faithful EmitC / intrinsic lowering
```

Hints/config are not final products. If they affect generated code, they must
be consumed into real `tcrv_rvv` structure.

### Stage 3: Second-Family Integration After RVV Maturity

IME, Offload, TensorExtLite, Template/Toy examples, source-front-door
construction flows, and future plugin examples are Stage3/later or explicitly
selected opt-in work. They must not displace Stage1/Stage2 RVV maturity work.

## Source Of Truth

`.trellis/spec/` 是 TianChen-RV MLIR 的长期规范源。早期
`predoc/tianchen_rv_mlir_capability_pack/` 设计包已被抽取并归并到本
spec 树中，当前仓库不再把 `predoc/` 作为 live source of truth。

后续 agent 应以本 index 和各层 spec 为准；如果历史任务记录、旧 prompt 或
外部说明提到 `predoc/`，只能把它理解为已迁移的历史输入，不能再要求读取或
维护该目录。
