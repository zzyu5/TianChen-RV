# TianChen-RV MLIR Trellis Specs

`.trellis/spec/` 是 TianChen-RV MLIR 的**长期规范（durable spec）**：约束设计、代码、实验解释和 agent 接手方式。

规范只描述**稳定契约**。当前进度、stage 编号、候选算子名（如某个 `*_i32_to_f32`）、measurement 状态、campaign / gate 记录、journal/session 引用——这些都**不是 spec**，属于 `tasks/`、`workspace/` journal、`artifacts/`。看到旧 spec 里夹带这类内容，按状态清除，不要当契约维护。

这些 spec 是给 AI agent（Claude / Codex）读的**判断依据**，不是状态机或门禁。按原则判断、始终对齐主干（见 [trunk-discipline](./guides/trunk-discipline.md)）；不要把条目当成机械打勾的 gate，spec 也不写"做完 X 才能做 Y"这类流程闸门。Spec 给方向和边界，怎么走由 agent 判断。

## 项目定位

TianChen-RV 是 high-level MLIR 之后的**能力驱动统一 RISC-V 执行层**。它把 RISC-V 的目标能力（ISA 扩展、VLEN/uarch、toolchain、runtime/offload）建成 first-class、可查询、可验证、可参与 pass 决策的 MLIR 对象，并用这些能力对象：

1. 驱动 **plugin-local** 的 variant 生成 / 合法性 / 选择 / dispatch —— 驯服 RISC-V 扩展的组合异构性；
2. 参数化一个 **resource-aware 的 tuning / realization 层（Gearbox）** —— 把选中的 extension body 变成调优过的可执行 body。

它**不是**新的高层 tensor/tile IR，也**不是**"一个硬件一个互不相关 backend dialect"的集合。RVV 是第一个完整 family 与硬件证明；IME / offload / 未来 vendor 扩展走**同一条 common 路径**。

规范上的 dataflow spine 见 [system-positioning](./architecture/system-positioning.md)；所有跨文件复用的硬规则集中在 [core-invariants](./architecture/core-invariants.md)，本树其他文件**引用**它而不重述。

## Novelty（论文主张 — 写 spec / 代码前必须对齐）

| # | 主张 | 成立所需证据（缺证据就只是工程，不是贡献） |
|---|---|---|
| N1 | RISC-V 扩展异构性作为 first-class capability IR | 同一 kernel 在多个**真实** profile 上，被 capability 查询导向不同的合法性 / 选择 / dispatch 结果 |
| N2 | 零-core-branch 的 plugin 泛化 | RVV + 至少一个**非-RVV** family（IME）走同一 common pipeline，core/common pass 里不出现 family-name 分支 |
| N3 | capability/resource-aware 的跨 family 调优 | Gearbox 的候选空间由 capability + resource facts 推导，且在若干 kernel 上对**框架自己出厂的同-ISA kernel**（如 ggml 真 RVV `vec_dot`）实测胜出或打平（baseline 纪律见 [validation/experiment-reference](./validation/experiment-reference.md)；scalar/naive 只作内部 sanity，**绝不**作贡献倍数）|

> "execution-variant 容器""plugin 化"本身**不是** novelty —— MLIR 的 dialect + interface 已提供。不要把架构选择当贡献卖点。novelty 只在 N1–N3，且都以证据为准。
>
> **N1 是 substrate，不是独立卖点**：把能力建成可查询对象本身 ≈ LLVM `-mattr`/TTI 已做的工程；N1 的 novelty **只在**它是跨 family 复用的同一事实源——由 N2 的第二 family 证明、由 N3 的 tune 兑现。抽掉跨 family 复用 N1 就塌回纯工程。别把 N1 当 blockbuster 单卖。capability 驱动的 LMUL/形状选择是 enumerate→prune→select→stamp（一个 stamping pass 写 attr），**不是 IR-rewriting transform pass**，别这么描述。

## Spec Layers

| Layer | Purpose |
|---|---|
| [architecture](./architecture/index.md) | 系统定位、核心不变量、研究边界、禁止误写的方向 |
| [capability-model](./capability-model/index.md) | target capability object、关系、profile、verifier 输入 |
| [core-dialect](./core-dialect/index.md) | `tcrv.exec` core dialect 的长期契约 |
| [plugin-protocol](./plugin-protocol/index.md) | extension plugin registry / interface / locality 规则 |
| [extension-plugins](./extension-plugins/index.md) | RVV、IME、runtime offload、scalar fallback、future plugin 边界 |
| [variant-pipeline](./variant-pipeline/index.md) | variant generation、legality、selection、dispatch、tuning（Gearbox） |
| [lowering-runtime](./lowering-runtime/index.md) | plugin-owned lowering / 公共 EmitC route / runtime glue 边界 |
| [implementation-stack](./implementation-stack/index.md) | C++/MLIR/LLVM/TableGen/CMake/lit/FileCheck 技术栈边界 |
| [testing](./testing/index.md) | dialect/pass/build/runtime evidence 测试口径 |
| [validation](./validation/index.md) | 实验参考和证据解释口径 |
| [guides](./guides/index.md) | 开发前设计检查清单 |

## 最高优先不变量（完整列表见 core-invariants）

- Capability 是系统第一对象，是可被 C++ pass 查询的 MLIR 对象，不是字符串 metadata。
- `tcrv.exec` 只承载 execution envelope；compute 语义属于 extension family。
- Common pass 只经 interface / registry 调用插件，绝不按 family 名分支。
- Metadata（diagnostics / manifest / route id / dashboard）永远是 mirror，绝不是 route / dtype / compute / 进度 / 证据的 authority。
- Primary stack 是 C++ / MLIR / LLVM / TableGen / CMake / lit / FileCheck；Python 只做 tooling，绝不实现 core IR / pass / capability model。
- 硬件 / 运行 / 性能主张需要真实证据；RVV 即 `ssh rvv` 证据。本地 build/lit 只是编译器/工具链证据。
