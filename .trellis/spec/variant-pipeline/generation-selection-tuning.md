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

> **怎么判断 tune 是否真的 resource-aware（老实判断，别自欺）**：只有当 Gearbox **枚举并按 resource facts 剪枝候选**时，它才是 resource-aware 的。固定单候选、固定 unroll、固定 LMUL 只是 MVP 占位——能跑，但不是 N3。N3 还要求在若干 kernel 上**实测赢** scalar 且赢 naive RVV：没有胜出的 tuning 没有论文故事。这是给 agent 的判断标准，不是流程闸门；当前实现离它多远，写在 task/journal，不写进 spec。

**Autotuning 模式**（分层，按需启用）

- Static/AOT：按确定的 legality/resource/cost model 选档。
- Offline profile：生成候选、编译、可选地查看汇编或在 `ssh rvv` 上跑，缓存某个 tuning key 的赢家。
- JIT/runtime：对某 key 的首次出现调优，命中缓存复用，runtime 调优不可用时退回 static selector。

tuning key 可含：target identity、VLEN/ELEN、operation signature、dtype/量化方案、memory form、shape bucket（如 `N` 或 `M/N/K`）。数据值不进 key，除非显式建模数据相关属性。

**跨 family**

同一 Gearbox 契约复用到 IME（fragment shape、K blocking、accumulator policy、packing）、Offload（transfer threshold、batch、async overlap、buffer reuse）。这是 N3 "跨 family" 的含义：一套 resource-aware tuning 机制，不是每个 family 各写一个 autotuner。低精度 / 量化 contraction（i8/u8/packed-i4 的 widening product + reduction + dequant）是该机制的代表性压力测试，但只是测试输入，不是 q8/q4-named route authority（I9）。

## Tests required

- selected body 存在性与 legality；
- 各 ABI 角色被 typed body 显式消费（I5）；
- Gearbox 把 code-affecting hint/config realize 进 body 结构，metadata-only 路径 fail closed；
- route provider 输出先于公共 EmitC；
- dispatch/fallback coherence；
- runtime/correctness/performance 主张配真实硬件证据（RVV = `ssh rvv`，I8）。
