# Generation, Selection, Tuning

variant pipeline 把 TianChen-RV execution envelope + plugin-owned extension body 变成**选中的、调优过的、可执行的** path。它不创造通用高层 compute IR，也不把 metadata 当 route authority（见 [core-invariants](../architecture/core-invariants.md) I4）。

```text
tcrv.exec envelope
  -> 插件提议的 variants（plugin-local）
  -> capability 驱动的 legality
  -> capability 驱动的 selection / dispatch
  -> Gearbox: resource-aware tuning / realization
  -> plugin route provider -> 公共 EmitC route
```

## Inputs

variant pipeline 可从以下起点工作：手写或生成的 TianChen-RV MLIR；`tcrv.exec` envelope 与 selected variant；typed extension-family body（如 `tcrv_rvv`）；origin 插件能合法消费的 selected boundary；结构化 capability/profile facts；runtime SSA / ABI 声明。

selected-path metadata 只解释"为什么选了这个 variant"，不能当 compute / dtype / route / body / artifact authority（I4、I5）。high-level MLIR frontend lowering（linalg/tosa→tcrv）是显式 opt-in 的未来集成路径。

## Variant required fields

每个可执行 variant 必须有：origin plugin；结构化 `requires`；typed extension-family body 或 selected boundary；plugin legality 结果；ABI 角色声明 + body 内显式 import/消费；可选 cost/tuning hint 作为 realization 输入；emission 支持时的 route-provider 输出；需要时的 fallback/dispatch 关系。

shape/dtype/layout 前置条件与 cost/tuning facts **可以**在 metadata 里镜像，但可执行的 dtype/config/operation 必须结构化在 typed body 或被消费进 realized body（I5）。

## Selection

selection 由 capability 驱动，可产出：单个静态 selected variant；guarded case 间的运行期 dispatch；保守 fallback；无合法可执行 route 时的 unsupported 诊断（fail closed，I7）。

selection **不得**：从 `tcrv.exec` 推断 compute；从 ABI 字符串/参数名推断 dtype；按 artifact 名选 route；把 source-front-door metadata 当可执行 route 排名；把 readiness/status dashboard 当进度。

## 选择器与归因契约（SEL / D）

本节声明选择 / 调度 / 归因的稳定契约。三者均属 wiring（能力调度接线），**不是** body 构造（[core-invariants](../architecture/core-invariants.md) 的 wiring ≠ construction 纪律，对应总纲 [L-6]①）。合法性谓词的定义、implies 闭包与"未知即拒（未知 = 假）"住 I7 与 [capability-contract](../capability-model/capability-contract.md)（legality gate / verifier 职责），此处**按引用**，不重抄。

### [SEL-1] 两段式选择器

selection 是两段式：

1. **合法性过滤**：`feasible = 合法性过滤(candidates, schema-instance)`——按能力谓词（含 implies 闭包）筛出可执行候选；未知能力即排除（I7）。
2. **排序**（在 `feasible` 上定档）：
   - **有实测** → 命中 memoized argmin（该 tuning key 的实测赢家，键控与回填见 Gearbox 权威顺序 + 总纲 [SEL-3]）；
   - **冷启动（无实测记录）** → 走**能力先验排序层**：GEMM 形 ∧ `ime.present` → 矩阵**范式**变体（[L-3]：向量核上的矩阵范式，非"矩阵家族"）；否则寄存器预算内**最宽 LMUL**；否则默认档。

**成本纯度**：先验排序层**独立于成本函数**，是基于能力事实的规则式裁决，**不污染成本纯度**（成本住测量库、按 instance-hash 键控；schema 不内置成本模型）。静态 cost 公式在此仅作(1)候选枚举 / 剪枝的 pruner 与(2)先验层之后的兜底默认，**不是**冷启动的排序 authority。`feasible` 为空 ⇒ fail closed 诊断（I7），不合成 route。

### [SEL-2] 硬时序

能力先验排序层必须**先于或同于**矩阵范式接管 GEMM 形 prefill（总纲 [PAT-2] P7）落地。一旦向量与矩阵两族为同一 GEMM 形内核竞标而先验层缺席，矩阵范式在 GEMM prefill 会**静默落败**——先验层此刻从装饰变裁决者。这是选择器时序的**硬契约**，不是可选优化。

**先验层验证不硬阻塞在自有 IME GEMM 上:** 复现"矩阵静默落败"→"先验层修复"（实验总纲 T4b）只需**一个能与向量变体对同一内核竞标的矩阵候选**;用双层测量键的 `forced` stub 矩阵候选（`selection_mode = forced`，仅供消融科学、不入产品主张）即可复现并验证先验层,**无须等自有机制构造 IME GEMM（那是 gated 的 X2）**。接 G3 pillar 的 agent 勿把先验层验证误锁在 X2 之后。

### [D-1] 编译期门自足 fail-closed

编译期合法性门在 plan 的能力谓词无法在目标 schema 实例（含闭包）下全部满足、或存在任何未知能力时，**自身**拒绝并给 bounded 诊断——**该拒绝必须自足**，不依赖任何后置 pass / 调用方兜底来补救或把被拒 plan 翻译成可执行路径。（合法性谓词与"未知 = 假"语义见 I7 + capability-contract；[D-1] 只加"拒绝自足、不靠后置兜底"这一层契约。）

### [D-2a] 装载期最小解析记录

装载期能力解析的稳定形态：消费 schema 事实实例（**profile 先展开成规范化事实集**）→ 对**展开后的规范化事实集**计算 **declared-instance-hash**（故 profile 写法与语义等价的显式事实列表**哈希相同**）→ 落**每进程一条**解析记录。热路径**零逐次**能力检查；**per-dispatch 强制检查永久禁止**（总纲 [NG-3]）。解析记录是缓存事实 / 镜像，不是 route / dtype / schedule authority（I4）。

### [D-4] 三级归因

每次变体选择可归因，分三级：

1. **编译期选择归因**——JSONL 每条 `{kernel, candidates[], keys_evaluated{}, chosen, reason ∈ {only_feasible, static_order, prior, measured}, declared_instance_hash, ts}`。`reason` 是**所有归因分析的主键**，能力键控与否做在主键上、不做在脚注守卫字段（否则某个查询会漏掉守卫字段，「能力键选中数」静默虚高）：`only_feasible` = 合法性过滤后仅剩一个可执行候选（N/A 能力键）；`static_order` = **能力盲**的冷启动排序（现每插件常量分 + explicit-preference）在 ≥2 候选中裁决——**非**能力派生；`prior` = **严格保留**给能力派生的先验排序层裁决（[SEL-1] 落地后才出现，今天 stage ① 绝不发）；`measured` = 命中 memoized 实测赢家（[SEL-3] 后）。**燃减/诊断不变量：** `static_order` 出现数在 [SEL-1] 落地后应归零；不归零 = 先验层覆盖缺口。此四值使 T4a「是否由能力键选中」列可直接从 reason 推导（static_order→否；prior/measured→是；only_feasible→N/A）。为使 `static_order` 决策可完整重建，参与排序的常量分须进记录（candidates[].score 恒发 / keys_evaluated）。
2. **装载期解析记录**——即 [D-2a] 的每进程一条记录。
3. **运行期归因**——随完整运行期 dispatch 链（hwprobe → 事实 → instance-hash 键控调度）产出。

**归因范围**含选择、**调度**（为何选此 LMUL / 此范式）与**合法性**（为何拒）三个阶段，不止最终 `chosen`。归因日志是 I4 镜像 / 事实，记录"为何选此变体"，**不**反向定义 compute / route / dtype，也不作进度 authority。

## Tuning / Realization — Gearbox

Gearbox 是 **plugin-local 的 MLIR pass pipeline**，把一个 selected pre-realized typed body 变成 realized（调优过的）typed body。它是 N3（capability/resource-aware 跨 family 调优）的承载体。

```text
selected pre-realized typed body
  + target capability / resource facts
  + runtime SSA / ABI values
  + optional hints / policy / profile
    -> Gearbox pass pipeline (build -> prune -> select -> realize)
    -> realized typed body（或 provider 在 route 构造前消费的 owner-local plan）
```

**硬约束**

- Gearbox 是 MLIR pass 内的变换，**不**搬到 common EmitC 或 target-artifact metadata 里。
- 任何影响生成代码的选择（setvl/VL 放置、SEW/LMUL/policy、memory form、mask/tail、unroll/prefetch、accumulator/reduction layout）必须在 route 构造**之前**被 realize 进 typed body 结构，或进 provider 消费的 owner-local plan。tuning facts 不是 route/dtype/schedule/进度 authority（I4、I5）。
- hints/config/profile 不是最终产物；不被 body 结构消费就不成立。

**Resource model（候选空间的来源）**

Gearbox 的候选空间必须由**编译器可见的 capability + body facts** 推导，而非硬编码常量。可推理的事实包括：target VLEN / ELEN / 向量寄存器预算、保留的 mask/v0、SEW/LMUL/EMUL、widening/narrowing 压力、peak live vector groups、load/store 与 mask 活跃区间、accumulator 数量与 reduction layout、vsetvl region 数、memory form/stride、tail/mask policy。resource model 可以**先 static、bounded**，但必须真的 over 这些 facts 推理。

> **怎么判断 tune 是否真的 resource-aware（老实判断，别自欺）**：只有当 Gearbox **枚举并按 resource facts 剪枝候选**时，它才是 resource-aware 的。固定单候选、固定 unroll、固定 LMUL 只是 MVP 占位——能跑，但不是 N3。N3 还要求在若干 kernel 上对**框架自己出厂的同-ISA kernel**（如 ggml 真 RVV `vec_dot`）实测胜出或打平（baseline 纪律见 [experiment-reference](../validation/experiment-reference.md)：scalar/naive 只作内部 sanity、**绝不**作贡献倍数）：没有胜出的 tuning 没有论文故事。这是给 agent 的判断标准，不是流程闸门；当前实现离它多远，写在 task/journal，不写进 spec。

**Autotuning 模式**（分层，按需启用）

- Static/AOT：按确定的 legality/resource/cost model 选档。
- Offline profile：生成候选、编译、可选地查看汇编或在 `ssh rvv` 上跑，缓存某个 tuning key 的赢家。
- JIT/runtime：对某 key 的首次出现调优，命中缓存复用，runtime 调优不可用时退回 static selector。

**权威顺序（实测 > 静态）**：当某 tuning key 存在 offline-profile 的实测记录时，**实测赢家是该 key 的形状 authority**，static cost model 降级为(1)枚举/剪枝候选的 pruner 与(2)无记录时的 fallback（冷启动的排序规则见 [SEL-1]：能力先验排序层，非静态 argmin）。这是 N3 "实测胜出" 的直接含义——静态 cost model 的预测会在 register-pressure / 时序等微架构效应上系统性失真（每加一个 kernel 就要补一维静态 cost 是 curve-fit treadmill），只有真机实测能定档。实测记录必须先过 byte-exact gate（与 `_generic` 逐字节）再排名；记录 fail-closed-revalidate（候选不再合法则退回 static）。**实测记录是缓存事实，不是 route/dtype/schedule authority（I4）；当前覆盖了哪些 key 写 task/journal，不写进 spec。**

tuning key 可含：target identity、VLEN/ELEN、operation signature、dtype/量化方案、memory form、shape bucket（如 `N` 或 `M/N/K`）。数据值不进 key，除非显式建模数据相关属性。

**跨 family**

同一 Gearbox 契约复用到 IME（fragment shape、K blocking、accumulator policy、packing）、Offload（transfer threshold、batch、async overlap、buffer reuse）。这是 N3 "跨 family" 的含义：一套 resource-aware tuning 机制，不是每个 family 各写一个 autotuner。低精度 / 量化 contraction（i8/u8/packed-i4 的 widening product + reduction + dequant）是该机制的代表性压力测试，但只是测试输入，不是 q8/q4-named route authority（I9）。

**复用契约（机制级，非状态）**：这套 tuning 的跨-op/跨-family 复用由**一个通用 schedule/tuning interface** 承载——可调 op **adopt 该 interface** 即被同一个 walk-all 的 schedule-materialization pass 自动发现并调优；**新增一个可调 op = adopt interface，不新写 pass**。候选**空间**由 interface 通用驱动；候选**数据**（某 op 的具体形状档）可以是 per-op 的，但**枚举/剪枝/选择/stamp 的机制是一份**。这是"不是每个 family 各写一个 autotuner"的可执行含义，也是 N2 零-core-branch 在 tuning 轴的体现（core 不按 family 名分支，只经 interface）。具体哪些 op 已 adopt 写 task/journal，不写进 spec。

**Gearbox 调的是"给定 op+layout 怎么 lower"，不是"选/改算法"**（前端 vs 后端判别见 [system-positioning](../architecture/system-positioning.md) 的 N3 边界，此处不重抄）。VL/setvl/SEW/LMUL/policy/memory form/mask-tail/unroll/accumulator layout 都是对一个**固定 op+layout** 的 codegen 调优——属后端 N3。反之，选 repack-vs-block-dot 算法、weight-packing/repack、改交给后端的 layout，改的是 op/algorithm/layout 本身——是前端（库/autotuner/框架）贡献，不是 Gearbox 的 N3 后端 novelty。

**判断依据（别自欺）**：选择在 realize 时由一个 stamping pass 物化（enumerate→prune→select→stamp），**非 IR-rewriting transform**；live 路上 select **先查实测记录（memoization）**、无记录才落**能力先验排序层**（[SEL-1]，非纯静态 argmin——capability-blind 的静态公式是已知成熟度缺口，先验层是其收口方向，静态 cost 退居 pruner + 兜底默认）。**追平框架自己出厂的同-ISA kernel = N3 主张为真（不是失败）；系统性 beat ⟺ Gearbox 综合一个框架没手写的 within-kernel 形状**（更宽 LMUL / VLEN-tuned strip / multi-accumulator），而非匹配其形状。

## Tests required

- selected body 存在性与 legality；
- 各 ABI 角色被 typed body 显式消费（I5）；
- Gearbox 把 code-affecting hint/config realize 进 body 结构，metadata-only 路径 fail closed；
- route provider 输出先于公共 EmitC；
- dispatch/fallback coherence；
- runtime/correctness/performance 主张配真实硬件证据（RVV = `ssh rvv`，I8）。
