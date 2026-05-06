# 00. 总纲：TianChen-RV MLIR

## 0.1 名称

**TianChen-RV MLIR: A Capability-Driven Execution Layer for Extensible RISC-V AI Kernels**

中文名建议：

**天辰 RV-MLIR：面向可扩展 RISC-V AI 内核的能力驱动执行层**

## 0.2 一句话定义

TianChen-RV MLIR 是 High-level MLIR 之后的 RISC-V 执行层。它不再引入新的高层 tensor/tile 计算抽象，而是把 RISC-V 系统能力显式建模为 MLIR 对象，并把 high-level MLIR 算子实现为一组面向 RVV、IME、offload runtime、future custom ISA 等能力的 execution variants。

## 0.3 为什么不是一个新的高层 IR

TianChen-RV MLIR 不应该包含类似下面的核心计算 op：

```text
tcrv.matmul
tcrv.softmax
tcrv.generic_reduce
tcrv.generic_tile
tcrv.generic_mma
```

这些 op 会把 TianChen-RV 重新变成一个高层或半高层 kernel IR，和上游 high-level MLIR、linalg、tensor IR、TIR 类抽象发生重叠。

TianChen-RV MLIR 应该从 high-level MLIR 接收已经确定的计算语义，然后负责回答：

```text
当前 RISC-V target 有哪些能力？
这个 high-level op 可以由哪些能力实现？
每个实现变体是否合法？
每个变体需要哪些资源、调优参数、运行时条件？
最终选择哪个变体，或者保留哪些 runtime dispatch 分支？
每个变体如何 lower 到 RVV intrinsic、LLVM scalable vector、IME intrinsic、inline asm、vendor C ABI 或 fallback？
```

因此，TianChen-RV 的核心不是“表达计算”，而是“组织 RISC-V 执行”。

## 0.4 最终系统形态

```text
High-level MLIR
(linalg / tensor / stablehlo / tosa / kernel dialect)
        |
        v
TianChen-RV realization generation
        |
        v
TianChen-RV MLIR dialect suite
  |
  |-- tcrv.exec
  |     kernel / target / capability / variant /
  |     hart_parallel / extension_region / dispatch / fallback
  |
  |-- tcrv.rvv
  |     setvl / vector load-store / mask-tail policy /
  |     vector arithmetic / reduction / RVV-specific control
  |
  |-- tcrv.ime
  |     vector-register-backed matrix fragment /
  |     IME config / IME mma / IME load-store
  |
  |-- tcrv.offload
  |     runtime buffer / async call / wait /
  |     vendor C ABI / accelerator runtime
  |
  |-- future extension dialects
        custom ISA / AME / VME / sparse / DMA / cluster / vendor-specific capability
        |
        v
Emission
  |-- MLIR vector / LLVM scalable vector
  |-- LLVM RVV intrinsic
  |-- inline asm / vendor builtin
  |-- vendor runtime C ABI
  |-- scalar or default fallback
```

## 0.5 当前硬件约束下的系统范围

当前系统必须基于真实可用硬件和即将到位硬件来设计：

1. **RVV 真实硬件**  
   主开发环境通过 `ssh rvv` 访问，目标机器是 RVV 1.0 环境，64 核 CPU，具备 sudo 权限。RVV 是系统首先完整打通的真实硬件路径。

2. **后续 IME/K3 环境**  
   IME 是后续优先接入的 matrix-extension 插件。它适合作为“新增 extension plugin 后核心 pass 基本不变”的关键验证对象。

3. **RISC-V + Sophgo/offload 环境**  
   Sophgo 这类能力不应被伪装成 RISC-V custom ISA extension。它应被建模为 runtime-offload capability：RISC-V host 通过 vendor runtime、C ABI、PCIe 或 SoC 内部 runtime 调用 accelerator。

4. **未来 custom ISA / AME / 其他扩展**  
   当前不把 AME 作为主实验目标。系统保留未来 extension plugin 接口，但不以暂时不可控的硬件作为当前主线。

## 0.6 研究思想

RISC-V AI 软件栈面临的问题不是单个 RVV lowering，而是扩展组合的变化：

```text
RVV only
RVV + FP16/BF16
RVV + IME
RVV + vendor runtime offload
future custom ISA
future matrix unit
```

如果每个新能力都要求修改 high-level lowering、核心 pass、cost model、dispatch 和 emission，系统会退化为一组手写 backend。

TianChen-RV MLIR 的研究思想是：

```text
把 RISC-V system capabilities 变成 MLIR 对象；
把每种硬件实现方式表达为 execution variant；
把新扩展能力局部封装为 extension plugin；
核心 pass 只通过 capability registry 和 plugin interface 工作；
同一个 high-level op 根据目标能力生成 RVV / IME / offload / fallback variants。
```

## 0.7 预期科研贡献

### 贡献 1：Capability-driven RISC-V execution model

将 `-march`、VLEN、RVV、IME、vendor runtime、offload accelerator、toolchain 支持等信息统一表示为 MLIR capability object。Capability 不只是字符串注释，而是 pass 查询、合法性验证、variant 生成、variant 选择和 dispatch 的依据。

### 贡献 2：Execution variant IR，而不是 generic compute IR

High-level MLIR 算子进入 TianChen-RV 后，不先变成通用 `tcrv.matmul`，而是由 extension plugins 直接生成 RVV、IME、offload 或 fallback variants。核心 `tcrv.exec` dialect 只组织 kernel、variant、capability、hart-level parallelism、dispatch 和 fallback。

### 贡献 3：Plugin-local extension integration

新增扩展时，扩展方提供 capabilities、ops/types、variant generator、legality verifier、tuning space、cost model、emission/runtime glue。核心 pass 不硬编码扩展名称，而是通过 registry 和 interface 查询插件。

### 贡献 4：Capability-aware variant selection and tuning

系统不是只调 tile size，而是在 capability 条件下选择实现变体以及变体内部资源参数，例如 RVV 的 LMUL/SEW/VL policy，IME 的 fragment shape 和 accumulator policy，offload 的 batching、transfer threshold 和 async overlap。

## 0.8 不做什么

TianChen-RV MLIR 明确不做以下事情：

```text
不设计另一个高层 tensor/tile IR。
不把通用 matmul/reduce/softmax 放入核心 dialect。
不把 Sophgo/offload 伪装成 RISC-V custom ISA。
不把 AME 作为当前必须完成的主实验硬件。
不把普通参数搜索包装成主要理论创新。
不声称任何未来扩展都可以零成本接入。
```

系统应诚实表达：新增扩展需要实现 plugin，但 plugin 应局部化新增代码，避免改动核心 pass。

## 0.9 模块目录

| 模块 | 文档 | 作用 |
|---|---|---|
| Capability model | `01_capability_model.md` | 定义 RISC-V target capability object、query、verification、dispatch condition |
| Core execution dialect | `02_exec_core_dialect.md` | 定义 `tcrv.exec`，只表达 kernel、variant、capability、dispatch、fallback |
| Plugin protocol | `03_extension_plugin_protocol.md` | 定义 extension plugin 如何注册能力、生成 variant、验证、调优、lower |
| RVV plugin | `04_rvv_plugin.md` | 当前主硬件路径，支持 RVV execution variants |
| IME plugin | `05_ime_plugin.md` | 后续 K3/IME 插件路径，验证 extension-local integration |
| Offload plugin | `06_offload_runtime_plugin.md` | Sophgo/vendor runtime offload 能力建模 |
| Variant pipeline | `07_variant_generation_and_selection.md` | 从 high-level op 到 variants、selection、dispatch 的核心流程 |
| Lowering/emission/runtime | `08_lowering_emission_runtime.md` | 各插件如何 lower 到 LLVM、intrinsic、inline asm、C ABI |
| Experiment reference | `09_experiment_reference.md` | 科研验证参考，不反向决定系统形态 |
| Agent prompt | `10_trellis_agent_prompt.md` | 让 agent 初始化 Trellis 并生成 `.trellis/spec/` |

## 0.10 最终判定标准

一个完整的 TianChen-RV MLIR 系统应满足：

```text
同一个 high-level MLIR op 可以在不同 target capability 下生成不同 execution variants。
RVV 真实硬件路径可以完整编译运行。
Offload runtime capability 可以作为 variant 纳入同一个 dispatch/selection 框架。
IME 到位后可以通过新增 IME plugin 接入，并用核心 pass 修改量证明 plugin-local integration。
核心 dialect 不表达高层计算语义。
核心 pass 不硬编码 RVV/IME/offload 的具体名字和内部 lowering 逻辑。
Capability object 参与实际 pass 决策，而不是只作为元数据。
```

