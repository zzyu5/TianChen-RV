# TianChen-RV MLIR Trellis Specs

`.trellis/spec/` 是 TianChen-RV MLIR 的**长期规范（durable spec）**：约束设计、代码、实验解释和 agent 接手方式。

规范只描述**稳定契约**。当前进度、stage 编号、候选算子名（如某个 `*_i32_to_f32`）、measurement 状态、campaign / gate 记录、journal/session 引用——这些都**不是 spec**，属于 `tasks/`、`workspace/` journal、`artifacts/`。看到旧 spec 里夹带这类内容，按状态清除，不要当契约维护。

这些 spec 是给 AI agent（Claude / Codex）读的**判断依据**，不是状态机或门禁。按原则判断、始终对齐主干（见 [trunk-discipline](./guides/trunk-discipline.md)）；不要把条目当成机械打勾的 gate，spec 也不写"做完 X 才能做 Y"这类流程闸门。Spec 给方向和边界，怎么走由 agent 判断。

## 项目定位

TianChen-RV 是**基于 MLIR 的能力驱动（capability-driven）可扩展执行层软件栈之参考模板（reference template）**——为碎片化硬件生态（RISC-V 是极端案例：VLEN 任意、扩展组合爆炸、厂商专有单元各异）给出一个可复制的**栈组织方式**（接入成本可预期 + 正确性机检 + 选择可归因），而不是又一个手写库。**RISC-V 量化 LLM 推理是该模板的首个高性能实例**：真硅上打赢手写出货物是模板质量的证明书，不是定位本身（主角 = 可扩展性；性能 = 证据）。

> **定位沿革（2026-07-10 定位升级 · 用户裁定）**：主角从"高性能 RISC-V 算子编译器"换位为"可扩展执行层软件栈参考模板"。旧句存档："TianChen-RV 是 high-level MLIR 之后的能力驱动统一 RISC-V 执行层。" 三贡献 C1/C2/C3′ 编号与数值不变，仅叙事主次升级；详见 [`docs/canon/TianChen-RV_定位-v2.md`](../../docs/canon/TianChen-RV_定位-v2.md)。

作为软件栈，它是 high-level MLIR 之后的能力驱动统一 RISC-V 执行层：把 RISC-V 的目标能力（ISA 扩展、VLEN/uarch、toolchain、runtime/offload）建成 first-class、可查询、可验证、可参与 pass 决策的 MLIR 对象，并用这些能力对象：

1. 驱动 **plugin-local** 的 variant 生成 / 合法性 / 选择 / dispatch —— 驯服 RISC-V 扩展的组合异构性；
2. 参数化一个 **resource-aware 的 tuning / realization 层（Gearbox）** —— 把选中的 extension body 变成调优过的可执行 body。

它**不是**新的高层 tensor/tile IR，也**不是**"一个硬件一个互不相关 backend dialect"的集合。RVV 是第一个完整 family 与硬件证明；IME / offload / 未来 vendor 扩展走**同一条 common 路径**。

规范上的 dataflow spine 见 [system-positioning](./architecture/system-positioning.md)；所有跨文件复用的硬规则集中在 [core-invariants](./architecture/core-invariants.md)，本树其他文件**引用**它而不重述。

## Novelty（论文主张 — 写 spec / 代码前必须对齐）

论文主张按科研目标总纲 v2 的**三条贡献 C1/C2/C3′ + 成熟编译器**组织；旧的 N1/N2/N3 不作废，而是**下沉为机制轴的构件**（映射见下方 bridge，本树其他文件与 CLAUDE.md 仍按 N1/N2/N3 引用，不推倒）。

**模板叙事（2026-07-10 定位升级）**：三贡献是同一"可扩展软件栈参考模板"的三个面——**C1（头牌）= 模板协议本体**、**C2 = 模板经济学**、**C3′ = 模板产出质量**；性能数字是 C3′ 模板产出质量的**证词**，不另立贡献。编号与全部数值不变。

| # | 主张（终态） | 成立所需证据（缺证据就只是工程，不是贡献） |
|---|---|---|
| **C1**（头牌） | **模板协议本体 · 合取机制的存在性 → 可复制扩展接入协议**：一份带关系的能力 schema + 插件协议（接入五件套）+ falsifier 机检 = 可复制的扩展接入协议，同时驱动编译期变体生成与 fail-closed 运行期（装载期解析形态起步）调度守卫，**跨计算范式（向量 SIMD → 整矩阵 MAC）且跨独立家族（向量缺席的标量家族）**原封复用；接入升级为外部贡献者可循的**协议** | 零家族分支由 falsifier 组 [F-1..F-6]（含独立性判据 [core-invariants](./architecture/core-invariants.md) [F-6]、操作门 [F-2′]）机检并 CI 常绿；schema.def 自第二家族起未被接入触及（[F-2′] 逐 PR 审计）；≥3 家族下证据阶梯成立 |
| **C2** | **模板经济学 · 泛化代价 → 边际成本规律**：零分支/零核心改动不变量下，逐家族接入代价形成**边际递减曲线**，并给出结构解释（成本住模式/谓词/测试哪一处）；第二/第三个异质家族（IME/标量/zvfh）接入多顺 = 模板故事最强证据 | 成本 ledger 脚本自动生成且首点可复算（cloc、测试单列）；≥3 数据点成曲线；术语按 [L-2] integrated/independent-attached；对照锚在位。**诚实标注：曲线现缺（1 点 / 需 ≥3）**——"第三家族小"正是主张本身（边际递减），不是弱点 |
| **C3′** | **模板产出质量 · 能力键控优化模式库 → 带实测与迁移的模板**：模式以能力谓词表达、由机制选出（归因日志）、跨 VLEN/微结构**换键不改条目**地迁移，**正例负例边界齐备、迁移可预测**，对 tuned 框架内核**分相**报告增量 | 注册表是**数据文件**（[PAT-1..3]）；迁移判据双板 diff=0 CI 常绿；对框架自身同-ISA kernel 实测胜出/持平（[core-invariants](./architecture/core-invariants.md) [L-6] vs-framework，过 [PERF-1] 八门）；scalar/naive 只作内部 sanity，**绝不**作贡献倍数。**性能数字在此陈列为模板质量证词，不另立贡献** |
| **成熟编译器** | 覆盖率与正确性门槛达标，成为本负载域内**真正可用的编译器** | 六态阶梯（[core-invariants](./architecture/core-invariants.md) [K-4]）自动读出、四覆盖率指标 + 燃减曲线进 CI；正确性门（字节精确 / ULP 上界 / VLEN 翻转 / objdump golden）全绿。**成熟度进 CI，不进 slides**（见双轴组织原则） |

### 双轴组织原则（引擎轴 × 证据轴）

交付按两轴并列组织，任何一轴单跑都不构成里程碑：

- **引擎轴**——机制覆盖广、主体由机制构造（强义，[L-8]）、可跨家族：N-operand 统一、全 zoo 构造、资源感知选择、真硬件探测、性能击败。让系统成为**成熟编译器**。
- **证据轴**——每条结构性主张机器可检验：schema 冻结哈希（[S-5]）、归因日志、模式注册表作数据、自动成本账、覆盖率进 CI、硬件探测。让成熟度可被机器**证明为论文证据**。
- **裁定**：成熟度指标**进 CI，不进 slides**。引擎轴让它成为成熟编译器；证据轴让成熟度可被证明为论文证据。**两轴并列为交付，任何一轴单跑都不构成里程碑。**（注：引擎/证据两轴 与 学术贡献/工程面 是两条正交的切分——引擎轴本身即含性能击败、全 zoo 构造等 C1/C3′ 的实体，不要把 C1/C2/C3′ 整体等同于证据轴。）

### N1/N2/N3 ↔ C1/C2/C3′ bridge（本文件桥接，其他文件仍按 N 号引用）

- **C1 = N1 ∧ N2 的合取存在性证据**：N1（能力异构性作 first-class IR，是 substrate 而非独立卖点）与 N2（零-core-branch plugin 泛化，用第二 family 证明）在**同一 schema** 上合取复用，即 C1 的"合取机制存在性 → 可复制协议"。
- **C2 = 泛化代价新轴**：在 N2 的零核心改动不变量之上，度量**逐家族接入的边际成本**，把工程不变量升级为经验规律。
- **C3′ ⊇ N3 的升级**：N3（capability/resource-aware 跨 family tune）升级为**能力键控模式库**——不止选变体，而是把优化沉淀为带实测与迁移判据的一等注册表对象。

> "execution-variant 容器""plugin 化"本身**不是** novelty —— MLIR 的 dialect + interface 已提供。不要把架构选择当贡献卖点。novelty 只在 C1/C2/C3′，且都以证据为准；成熟编译器是工程面、进 CI 不作 slide 卖点。
>
> **N1 是 substrate，不是独立卖点**：把能力建成可查询对象本身 ≈ LLVM `-mattr`/TTI 已做的工程；其 novelty **只在**它是跨 family 复用的同一事实源——由 N2 的第二 family 证明（→ C1）、由 N3 的 tune 兑现（→ C3′）。抽掉跨 family 复用就塌回纯工程。capability 驱动的 LMUL/形状选择是 enumerate→prune→select→stamp（一个 stamping pass 写 attr），**不是 IR-rewriting transform pass**，别这么描述。

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
