# 03｜ARS 阶段三：外部证据、Construction 审计与目标重构蓝图

> 状态：Phase 3 Analysis / Synthesis；已由 Phase 4 独立评审。QIGen 正式全文取得后的最终外部差分以 [06-QIGen 全文差分与最终再收敛](../phase5_reaudit/06-QIGen全文差分与最终再收敛.md) 为准。
> 日期：2026-07-21。
> 法源地位：内部方法审计，不是 canon，不改写两柱、六律或论文贡献编号，不建立 Trellis task。
> 核心问题：Weft-RV 需要怎样的最小而真实的 compiler 重构，才能由 typed `g/c/ω`、可复用 mechanisms、analytic knowledge、边界与有限 residual，重建未逐点完整编码的 kernel-family 实例？

<!--
claim_intent_manifest:
  PH3-C1: generic expert-knowledge factorization is not a defensible novelty claim
  PH3-C2: current Weft contains real construction substrate but no demonstrated family-level reconstruction after point-authority erasure
  PH3-C3: target refactor should consolidate causal construction, not rewrite the compiler or the research line
  PH3-C4: q5_1 flat repack is the first authority-erasure implementation milestone, not the primary novelty or prospective unseen witness
  PH3-C5: IME is a real second compute family but currently repeats pointwise construction and emission decisions
  PH3-C6: provider/verifier are implementation techniques, never first principles or novelty
evidence_scope:
  external: phase2 primary-source bibliography and verification report
  internal: current code and tests at the audited working tree
  limitations: QIGen 2026 full text and artifact inspected; no Weft refactor or new hardware campaign has yet been executed
-->

## 0. 综合结论

本轮综合得到五个结论。

第一，**需要降低的是泛化 novelty，不是推翻项目主线**。把专家知识分成规则、机制、计划和经验反馈，并据此生成高性能实现，在 FFTW、SPIRAL、Halide、Exo、ROLLER、Bolt 等系统中都有强先例；量化 LLM kernel generator 也已有 QIGen。因此，Weft-RV 不能再把“知识因式分解”“解析加测量”“typed plugin”或“construction-based generation”单独写成首创。

第二，**当前项目不是只会选择手写代码的薄公式原型**。当前代码中已经有：

- typed format facts 和 typed extension body；
- capability-conditioned planning；
- 真实的 operation-graph construction；
- RVV、IME、Scalar 等 family；
- lowering、EmitC、ABI/runtime、正确性与硬件测量基础；
- 大量对迁移、负结果与生成代码形态的测试。

这些都是需要继承的 compiler 资产。

第三，**在论文最关心的 block-quant family 上，当前成熟度仍是混合态**：

- 部分路径只是 typed full-leaf selection；
- 部分路径已经用 `g/c/ω` 构造承重参数；
- 部分路径在单个逐点 builder 内真实组合 typed bricks；
- 少数 generic source front door 能由 capability 构造完整 operation graph；
- 但尚未有一个严格的、无逐点完整实现泄漏的 family reconstruction 证人。

所以当前不能声称已经达到理想的 Execution Knowledge Factorization；同样，也不能据此否定这个目标。当前缺口正是后续重构要解决的对象。

第四，**最小正确方向不是建立新的全局 Formula IR、provider 框架或 verifier 体系**。目标是让现有 family-local facts、typed bricks、capability model 和 lowering 形成唯一的因果链：

```text
typed g / c / ω
→ analytic candidate construction
→ applicability / resource boundary
→ optional thin residual selection
→ typed selected plan/body
→ family-local mechanical realization/emission
```

物理代码可以继续分布在不同 family 内；共同的是知识职责和因果关系，不是统一 C++ 基类或统一深层 IR。

第五，**q5_1 flat repack 是第一项 authority-erasure 实现里程碑，不是主要科研证人**。现有源码说明 q5_1 是 q5_0 的 high-bit-plane decode 与 q4_1 的 affine scale/offset fold 的组合，但仍保留完整 Q51 GEMV/GEMM builders。删除这些 point implementations 及其等价 authority、再由共享机制与公式重建两个 regime，能够验证本地 reconstructability。QIGen 全文已经覆盖 5-bit 仿射 group、GEVM/GEPM 和 group composition，所以 q5_1 单独成功既不证明 unseen generalization，也不足以证明方法超越 QIGen。

阶段三总裁决为：

> **保留两柱与已确认公式；把待验证差额收缩到完整 per-bit/per-format microkernel leaf 以下的异质 GGML 语义因式分解，并用多 topology 的 point-authority erasure、跨 operation/backend 因果链、rule-level fan-out、稀疏 residual 与真实部署证明。只有主张 unseen/generalization 时才另需冻结后的 prospective transfer。当前代码提供了重构起点，但尚未完成这一证明。**

---

## 1. 术语与编号防混淆

### 1.1 本文的 CM0–CM4 不是论文 C1/C2/C3

阶段一曾用 C0–C4 表示 construction 成熟度。为避免与当前 canon 的论文贡献 C1/C2/C3 冲突，本文统一写为 **CM0–CM4**：

| 审计等级 | 含义 |
|---|---|
| CM0 · metadata only | 信息只影响 attr、trace、reason 或诊断；不改变 executable plan/graph。 |
| CM1 · typed full-leaf selection | raw point identity 最终选择完整 leaf；删除 leaf 后该实例失败。 |
| CM2 · parametric construction | typed facts 改变承重参数，但仍依赖 point-specific skeleton/builder；删除 point builder 后失败。 |
| CM3 · structural construction | raw identity 只投影为独立语义 facts；共享 mechanisms 与公式构造 compute graph，且不存在该 point 的 executable branch。 |
| CM4 · reconstructive evaluation grade | 冻结 CM3 构造栈后，在没有完整 point implementation 的条件下重建实例，并通过预定 correctness、部署与 performance 评价。 |

CM0–CM4 只是本轮代码因果审计工具：

- 不是两柱；
- 不是六律；
- 不是论文贡献编号；
- 不能按“达到几级”简单计算项目完成度；
- 同一 family 的不同 slice 可以处于不同等级。

CM4 是建立在 CM3 上的 **evaluation grade**，不能仅凭静态代码形状判定。团队已知旧实现后再删除 leaf 的 q5_1 属于 retrospective reconstruction；它是本地实现证据而非主要 novelty。只有需要声称 unseen/generalization 时，规则冻结后选择的未参与设计组合才需要称为 prospective holdout。

### 1.2 “construction”采用因果定义

本文只在以下条件下把一项工作计作 construction：

1. 输入是实际承重的 typed `g/c/ω` 或其合法投影；
2. 输出决定 candidate、机制顺序、typed op graph、layout、resource shape 或合法域；
3. downstream 直接消费该输出；
4. 改变或移除输出会按预期改变 IR/code shape/legality；
5. emitter 不会从 format 名或隐藏表中恢复被移除的完整答案。

因此，“创建了 typed op”“写了 provider”“加了 verifier”“stamp 了 reason”都不自动等于 construction。

### 1.3 完整 leaf 与最小 mechanism primitive

未来要求删除的是完整 point implementation，不是删除一切 leaf：

- 可保留真正不可再分的位解码、table/gather、MAC、reduction、scale decode、ABI read 等最小语义 primitive；
- 不应为每个 `format × capability × context × regime` 保存一份完整 kernel body；
- 新格式引入新语义时可以新增最小 primitive；
- 已有机制的新组合应由 facts、formula 与 boundaries 构造，而不是复制完整 pipeline。

---

## 2. 外部文献综合：哪些主张已经被先例覆盖

详细书目信息、原始链接与来源核验见：

- [01-检索策略与注释文献表](../phase2_investigation/01-检索策略与注释文献表.md)
- [02-来源核验与主张压力表](../phase2_investigation/02-来源核验与主张压力表.md)

### 2.1 Literature matrix

符号：`●` 为直接强覆盖，`○` 为部分相关，空白为不是该论文主轴。

| 系统 | 规则/机制生成 | typed/可扩展 | 解析构造 | 经验选择 | 低精度/量化 | 对 Weft 的主要压力 |
|---|---:|---:|---:|---:|---:|---|
| FFTW | ● | ○ | ○ | ● |  | codelets、plans 与 measurement 组合早已有之。 |
| SPIRAL | ● | ○ | ● | ● |  | 规则、公式、ruletree、反馈搜索生成 implementation family，是最强概念近邻。 |
| PetaBricks | ● | ○ | ○ | ● |  | algorithmic choice 与 hybrid construction 不是新命题。 |
| Halide | ○ | ● | ● | ○ |  | compute semantics 与 schedule 分离不是新颖点。 |
| LIFT | ● | ● | ● | ○ |  | typed functional patterns 与 rewrite-based hardware mapping 已存在。 |
| TVM | ● | ● | ○ | ● | ○ | 跨硬件自动 tensor codegen 与 learned search 已成熟。 |
| MLIR |  | ● | ○ |  |  | dialect、typed extension、progressive lowering 不能单独构成 novelty。 |
| BuildIt | ○ | ● | ● |  |  | typed generation-time construction 不是新技术。 |
| Exo | ● | ● | ● | ○ | ○ | 外置硬件知识与可组合 scheduling primitives 已有直接先例。 |
| Exo 2 | ● | ● | ● | ○ | ○ | fine-grained primitives、schedule libraries 与跨 80+ kernels fan-out 已有强证据。 |
| Ansor | ● | ● | ○ | ● | ○ | 自动生成 program space 与经验选择已高度成熟。 |
| ROLLER | ● | ○ | ● | ○ | ○ | “construction-based approach instead of broad search”不是空白。 |
| MetaSchedule | ● | ● | ○ | ● | ○ | 专家 modular rules 与 stochastic choice 的组合已存在。 |
| TensorIR | ● | ● | ● | ● | ○ | typed blocks、hardware constraints 与自动 tensorization 已覆盖一般 tensor 场景。 |
| Bolt | ● | ○ | ● | ● | ○ | expert templates/primitives 加轻量 profiling 与硬件原生性能非常接近。 |
| Ladder | ● | ● | ● | ● | ● | low-precision datatype、layout 与 hardware-aware transformation 不是新颖点。 |
| QIGen 2026 | ● | ○ | ● | ● | ● | 全文确认按 `(M,K,g,b)` 生成/调优 GEVM/GEPM microkernel，并按非均匀 group 序列组合；完整 per-bit unpack/dot body 仍在大分支中。 |

### 2.2 主题综合

#### 主题 A：规则、公式、机制与 family generation 已有长历史

[SPIRAL](https://www.cs.cmu.edu/~mmv/papers/05SpiralIEEE.pdf) 已经把 transform definitions、breakdown rules、applicability、formulas、ruletrees、code generation 与 feedback search 连成完整系统。它直接否定以下宽泛表述：

- “首次把专家知识变成公式”；
- “首次用规则组合生成 implementation family”；
- “首次让新增规则跨实例摊薄”；
- “首次结合解析知识与经验反馈”。

Weft 与 SPIRAL 的可能差额，不在“也有公式”，而在知识对象与 compiler context：不规则 GGML block-quant ABI、RISC-V capability、静态执行 context、typed extension boundary、多 family backend，以及 residual 不能创造 compute 的具体约束。

#### 主题 B：typed extensibility 与 composable primitives 已被 MLIR/Exo 覆盖

[MLIR](https://research.google/pubs/mlir-scaling-compiler-infrastructure-for-domain-specific-computation/) 说明 extensible dialect/type/op/pass infrastructure；[Exo](https://people.csail.mit.edu/yuka/pdf/exo_pldi2022_full.pdf) 与 [Exo 2](https://arxiv.org/abs/2411.07211) 说明 hardware primitives、scheduling operations 与 library growth。由此：

- plugin 不是 novelty；
- typed body 不是 novelty；
- provider 接口不是 novelty；
- verifier 不是 novelty；
- rule fan-out 本身也不是 novelty。

Weft 必须证明的是：typed facts 与 mechanisms 真正自动重建 irregular execution instances，而不是人工 schedule library 或逐点 typed leaf 的另一种包装。

#### 主题 C：解析构造与有限搜索已有直接近邻

[ROLLER](https://www.usenix.org/conference/osdi22/presentation/zhu) 已明确采用 construction-based tensor compilation；[Bolt](https://proceedings.mlsys.org/paper_files/paper/2022/hash/1f8053a67ec8e0b57455713cefdd8218-Abstract.html) 已结合 hardware-native templates、primitives 与轻量 profiling。因此：

- “不用大搜索”不能单独构成 novelty；
- “解析 prior + measurement winner”不能单独构成 novelty；
- “bounded candidate set”也不是独立贡献。

可区分点只能来自所构造对象、边界、residual 密度与实际 family reconstruction 证据。

#### 主题 D：量化 kernel generator 已有危险近邻

[Ladder](https://www.usenix.org/conference/osdi24/presentation/wang-lei) 覆盖 hardware-aware low-precision tensor transformation；[QIGen 正式全文](../QIGen_A_Kernel_Generator_for_Inference_on_Nonuniformly_Quantized_Large_Language_Models.pdf)则直接给出完整生成链：将 quantizer 输出翻译成 W/S/Z，枚举 unique `(M,K,g,b)`，由 cache model 与 grid search 生成/调优 GEVM/GEPM LLVM microkernel，再按每层 group 序列组合并合并相邻同类 group。它已经在 layer/group 粒度完成真实 construction，不是 monolithic whole-matrix leaf lookup。

QIGen 的更深边界是：2/3/4/5/6/8-bit 的完整 unpack、shift、mask、dot 算法仍由显式 `switch(bits)` 编码。准确结论是“它的因式分解粒度停在完整 per-bit microkernel unit”，而不是“它没有 knowledge factorization”。正文还报告 x86→ARM 的手工语义适配，但性能实验只在 Intel；TPOT/TTFT 为 kernel/layer 结果的模拟推算，不是端到端运行。

因此禁止使用：

- “首个量化 LLM CPU kernel generator”；
- “首个由量化格式与 CPU 特征生成 kernel 的 compiler”；
- “首个 low-precision knowledge factorization”。

这已经排除了“cross-combination generation、GEVM/GEPM、analytic+search 或 CPU retargeting 本身是空白”的说法。剩余假说只能位于完整 microkernel leaf 以下，并且不能通过“QIGen 没用 typed mechanism/boundary/residual 这些名称”制造差额。

### 2.3 当前仍可能成立的窄缺口

本轮没有找到单一系统完整覆盖以下组合：

> 在 high-level MLIR 之后的 execution layer 中，将 GGML/llama.cpp 风格的不规则 block-quant execution knowledge 分成 typed format facts、RISC-V capability、static context、可复用 mechanisms、analytic construction、显式边界与稀疏 empirical residual，并在多个 op/family/plugin 中重建未逐点实现的实例。

这只是**可检验的窄缺口**，不是已经成立的 novelty。它必须同时满足：

1. 当前代码经过真实重构，而非术语重命名；
2. 多条 load-bearing path 达到 CM3；
3. 至少一个无泄漏 retrospective CM4 witness；若主张 unseen/generalization，再增加冻结后的 prospective witness；
4. residual 随产品空间扩张仍稀疏；
5. 生成实例正确、可部署且有竞争性能；
6. IME 等第二 family 证明共同知识职责可跨 backend 成立；
7. QIGen 全文直接比较已完成；后续实现必须实际穿过其 per-bit microkernel leaf boundary，而不是只改术语。

---

## 3. 当前 compiler 中的 construction 事实

### 3.1 全局判断：真实 compiler，不是公式玩具

当前系统的实际链路包含：

```text
high-level MLIR / source fixture
→ extension proposal and selection
→ typed family IR / construction
→ capability and static context
→ family lowering / scheduling / body realization
→ conversion legality and EmitC
→ generated bundle / ABI / runtime
→ correctness / measurement / evidence
```

因此，公式只应解释其中“execution knowledge 如何构造候选与实现”的核心因果段，不能取代 dialect、pass、ABI、runtime、fallback 或 measurement control plane。

### 3.2 Dequantize-row：typed 基础真实，family construction 尚未闭合

#### 当前事实

[RVVDequantizeRowConstruction.cpp](../../../lib/Dialect/RVV/IR/RVVDequantizeRowConstruction.cpp) 做了两项重要工作：

1. `lookupDequantizeRowStreamFacts`（约第 24 行）集中投影 `qk`、block stride、scale/quant offsets、entry lanes、nibble carrier、min/qh offsets、codebook scale model 等格式事实；
2. `constructTypedDequantizeRowLoopBody`（约第 152 行）创建 `TypedDequantizeRowLoopBodyOp`，并在 region 内创建 `DequantizeRowDecodeCoreOp`（约第 185 行）。

这不是无价值包装：它已经把原始 format identity 转成了可由下游消费的 typed construction substrate。

但 [RVVToEmitCForwardElementwise.cpp](../../../lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp) 显示不同 mechanism 仍处在不同阶段：

| Slice | 当前真实链路 | 当前等级 | 关键原因 |
|---|---|---:|---|
| flat nibble / q8_0 | descriptor facts → typed core；emission 附近调用 `decideNibbleDecode`；共享 nibble body | CM2 | plan 参数真实影响 strip、bias、min/qh 等，但 decision 发生过晚，carrier 仍选择预存完整 body。 |
| `q4_synth` | 新 descriptor row 复用与 q4_0 完全相同的 tuple 和 nibble body | CM2 witness | 证明 format 名不是唯一 dispatch key；但没有产生新的机制组合，不能算 reconstruction。 |
| codebook | typed facts + capability projection → `decideCodebookGather` → pre-emission selected stamp → emitter 读取 plan | 强 CM2 | `c` 真实决定合法 LMUL；但 emitter 在约第 4403 行仍按 E8M0/fp16/UE4M3/signed6 scale model 展开完整算法分支。 |
| KQuant | `decode_model` → scale-model enum → emission-period provider → full vector-body switch | CM1/弱 CM2 | `kquantScaleMinPlanFromFacts` 主要重包装 facts；`minimumVLEN` 被显式忽略；完整 q2/q3/q45/q6 body 仍逐 leaf 选择。 |
| Grid | `decode_model` → leaf enum → provider → `switch(plan.leaf)` | CM1 | provider 忽略 `minimumVLEN`，plan 保存完整 leaf identity；emitter 选择五个完整 body。 |
| Ternary | `decode_model` → leaf enum → provider → `switch(plan.leaf)` | CM1 | provider 不构造结构且 capability honest-null；完整 tq/iq1 body 仍由 leaf 决定。 |
| q1_0 | typed outer body → format string → monolithic helper | CM1/legacy | 仍是完整 hand-written binary-sign leaf。 |

#### Codebook 为什么是当前最强 dequant CM2

[RVVFormulaDecision.h](../../../include/Weft/Plugin/RVV/RVVFormulaDecision.h) 的 `decideCodebookGather`（约第 259 行）实际消费：

- `g`：entries、strip geometry 等；
- `c`：minimum VLEN、SEW8/32、LMUL support；
- finite anchors：`mf2/m1/m2`。

[RVVCodebookGatherPlanMaterialization.cpp](../../../lib/Conversion/RVV/RVVCodebookGatherPlanMaterialization.cpp) 在 emission 之前投影 selected target capability、运行 decision 并 stamp selected plan。固定 `g` 改变 `c` 会改变合法 anchor，因而是诚实的 capability-driven construction。

但是 plan 只解决了 gather anchor 与若干参数，尚未把 scale decode topology 构造成 typed body。emitter 内仍有：

```text
E8M0 shared exponent
fp16 flat scale
UE4M3 per-sub-block scale
signed-6 super-block scale
```

四个完整控制流分支。这就是“强参数构造但尚未完成结构构造”的准确含义。

#### KQuant/Grid/Ternary provider 为什么不是第一性资产

[RVVToEmitCSupport.cpp](../../../lib/Conversion/RVV/RVVToEmitCSupport.cpp) 中：

- `kquantScaleMinPlanFromFacts` 约第 1086 行；
- `gridLookupPlanFromFacts` 约第 1163 行；
- `ternaryDecodePlanFromFacts` 约第 1212 行。

三者都接收 `minimumVLEN`，但当前明确不使用；它们主要把已经选择的 scale model/leaf 与少量 facts 装进 Plan。这里 provider 作为迁移期局部函数可以保留，但不能据此声称公式或 capability 已经承重，更不能把“每个机制都有 provider”写成目标。

#### Dequant 的目标

目标不是重写全部 emitter，而是把以下结构在 emission 前构造为 typed mechanisms：

```text
block traversal
→ packed carrier load
→ unpack / high-bit merge / sign or table decode
→ scale/min decode
→ convert/fold
→ store
```

emitter 可以继续拥有每个最小 op 的 RVV intrinsic lowering，但不得再按 format/leaf identity 决定完整 pipeline。

### 3.3 Flat repack GEMV/GEMM：已有 typed brick construction，但仍有完整 point builders

#### 当前公式与资源决策

[RVVFormulaDecision.h](../../../include/Weft/Plugin/RVV/RVVFormulaDecision.h) 的 `decideRepackAccumulatorLMUL`（约第 466 行）是当前最接近目标语义的纯 decision：

- `g.weightInterleave` 构造 strip geometry；
- `c.hasFractionalLMUL / halfLanes / vectorRegisterBudget` 限定合法候选；
- finite candidate set 为 `{mf2, m1}`；
- `ω` 的 qualified measurement 只能在合法候选中选 winner；
- miss 时回到 analytic prior 或 only-feasible；
- empty legal set 时拒绝。

这是可信的“analytic construction + bounded residual”局部资产。

[RVVLowerQuantContraction.cpp](../../../lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp) 的 `buildRepackAccumulatorLMULDecision`（约第 1358 行）把真实 geometry、capability 和可选 measurement 投影进去。当前 measurement rows 仍硬编码在同一 C++ 文件约第 1316 行，适合作为迁移起点和历史知识，不应包装成 novelty。

#### 当前 body construction

同一文件的 repack lowering 已经构造：

- `TypedRepackGemvLoopBodyOp` / `TypedRepackGemmLoopBodyOp`；
- per-block core brick；
- per-strip/per-column fold brick；
- typed region args 与 yield；
- capability-driven accumulator LMUL、half lanes 与部分 schedule attrs。

这是真实 construction，不应贬为“只是 emitter”。

然而 flat family 仍通过长链判断进入独立完整 builder：

```text
lowerToRepackGem{v,m}          // q4_0
lowerToRepackGem{v,m}Q41
lowerToRepackGem{v,m}Q50
lowerToRepackGem{v,m}Q51
lowerToRepackGem{v,m}Q80
```

这些函数分别重新创建 loop、region、core、fold、yield 与 ABI。虽然它们使用共享 op 类型，完整 `format × regime` implementation 仍逐点存在。

因此应分两层评价：

| 审计对象 | 当前等级 | 说明 |
|---|---:|---|
| 单个 builder 内的 typed brick graph | 局部 CM3 | 真实创建多个 typed ops 与 SSA edges。 |
| flat family 的跨格式构造 | CM2 | 仍由完整 per-format builder 选择；缺少 facts → mechanism graph 的统一构造。 |
| family-level reconstruction after point-authority erasure | 未达 CM4 | 每个现有组合都已有显式完整 builder。 |

#### q5_1 是当前最清晰的反事实缺口

源码约第 1714–1724 行已经准确描述：

```text
q5_1 = q5_0 的 qh/high-bit-plane decode
     + q4_1 的 affine scale/offset fold（历史名 min fold）
```

但实现仍显式调用：

- `lowerToRepackGemvQ51`（约第 2886 行）；
- `lowerToRepackGemmQ51`（约第 3036 行）。

这两个函数完整重建 region，包含 q5_1 stride/offset、core attrs、fold attrs、yield 与 ABI。也就是说，知识上已经知道它是组合，代码上仍保存了一份 point implementation。这正是目标重构最应该消除的地方。

### 3.4 Generic RVV contraction/body realization：真实 CM3 substrate，但不能代替 block-quant CM4

[RVVContractionSelectedBodyRealizationOwner.cpp](../../../lib/Plugin/RVV/BodyRealization/RVVContractionSelectedBodyRealizationOwner.cpp) 包含真实的 operation-graph construction：

- 约第 1491 行定义 realization plan；
- 约第 1722–1955 行从 typed pre-realized ops 投影 plan；
- 约第 1988、2074 行分别构造 deferred-wide dequant 与 deferred-wide dot-reduce graph；
- 约第 2152–2580 行创建 `setvl/with_vl/load/strided_load/compare/product/reduce/dequant/select/store` 等 typed ops；
- selected resource candidate 会改变 packed-i4 head、grouped product、unroll factor、single-scope/two-scope structure；
- vector-register budget 会改变 LMUL rung，结构最终进入 typed op identity 和 vector types。

这些路径不是 metadata：改变 resource/budget/structure facts 会改变真实 operation graph，属于 CM3 construction substrate。

但必须加两个限制：

1. 很多 `Typed*PreRealizedBodyOp` 当前主要由 `test/Target/RVV/pre-realized-*` fixture 直接书写；源码检索没有发现同等完整的 production producer。不能把 test-authored pre-realized surface 自动写成真实 workload family 的 CM3 证明。
2. production source front doors（例如 `RVVDequantDotSourceFrontDoor.cpp`、`RVVReductionSourceFrontDoor.cpp`、`RVVPackedI4DotSourceFrontDoor.cpp`）确实从 generic vector source 与 capability 构造 typed operation graph，但它们覆盖的是 bounded generic contraction slices，并不等价于从 GGML block-quant facts 重建完整 format family。

正确结论是：

> **项目已经拥有能承担目标重构的 CM3 machinery；论文核心 block-quant family 尚未把这套能力转化成无逐点 leaf 的 CM4 证据。**

### 3.5 IME：真实第二 compute family，局部结构构造强，但仍是 format-point branches

[IMEExtensionPlugin.cpp](../../../lib/Plugin/IME/IMEExtensionPlugin.cpp) 不是空插件。`deriveIMEMatmulCapability` 真实消费：

- march token；
- VLEN/SEW；
- signedness；
- whole-matrix shape；
- weight format；
- slide；
- available harts。

materialization 约第 997–1415 行会为 q4_0、q8_0、q4_K 构造 typed region：

- q4_0：decode core + `VmadotMacLeafOp` + yield；
- q8_0：direct-int8 core + 同一 MAC leaf + yield；
- q4_K：raw-nibble decode + scale/min unpack + MAC + scale-weighted accum + min-bias accum + two-result yield。

q4_K 的六 brick region 是真实结构构造，证明 IME 是第二 compute family，而不是只提供 route metadata。

但当前仍存在明显的 pointwise structure：

- `if (derived->isQ40Weight)`；
- `if (derived->isQ80Weight)`；
- `if (derived->isQ4KWeight)`；
- 三段分别完整创建 tile、attrs、region、ops 与 yield。

[IMEBackendEmissionDriver.cpp](../../../lib/Plugin/IME/IMEBackendEmissionDriver.cpp) 也不完全机械：

- emitter 内选择 batched/unbatched MAC leaf；
- emitter 内运行 wide-vmadot deployment decision；
- q4_0/q8_0/q4_K 分别生成大段 helper body；
- q4_K 的 measured-negative 仍在 emission-period selection 中承重。

因此 IME 当前的准确等级是：

| Slice | 当前等级 |
|---|---:|
| capability → shape/signedness/boundary op | CM2 |
| 单个 q4_0/q8_0/q4_K typed region | 局部 CM3 |
| 跨 weight format 的 family construction | CM1/CM2 |
| emitter 作为 mechanical realization | 未达目标 |
| cross-format reconstruction after point-authority erasure | 未达 CM4 |

IME 对后续最重要的价值不是“强迫它复用 RVV 的 C++ Plan”，而是证明同一**概念合同**可以由不同 family 本地实现：

```text
format/shape facts + target capability + context
→ family-local mechanisms
→ legal typed body
→ family-local emission
```

RVV 的 nibble/gather/reduction mechanism 与 IME 的 matrix fragment/vmadot mechanism 本来就不应共享具体 ABI 或 op 类型。

### 3.6 Scalar：真实后端与参考实现，但不是当前最强 construction witness

[ScalarBackendEmissionDriver.cpp](../../../lib/Plugin/Scalar/ScalarBackendEmissionDriver.cpp) 确实能把：

- `weft_scalar.tq2_0_q8_k_vec_dot`；
- `weft_scalar.dequantize_row_q4_0`；
- tracer-bullet compute skeleton

降低成纯 scalar EmitC。Ternary vec-dot 的 typed attrs 会改变 loop bounds、strides 与 offsets，因而不是完全空洞的 emitter。

但 [ScalarExtensionPlugin.cpp](../../../lib/Plugin/Scalar/ScalarExtensionPlugin.cpp) 当前 `materializeSelectedLoweringBoundary` 仍返回 Unsupported；`test/Target/Scalar/tq2-0-q8-k-ternary-vec-dot.mlir` 也明确说明 typed body 是手写输入，绕过 emission-plan readiness。

因此：

- Scalar 是真实 backend/fallback/reference 资产；
- 它可用于 correctness baseline 和共同 plugin boundary 审计；
- 当前不宜把它作为“第二 family 已实现 knowledge construction”的主证人；
- 它的完整 scalar algorithm 仍主要住在单个 emitter pattern，属于 CM1/有限 CM2。

### 3.7 Measurement residual：局部语义正确，物理位置仍可收敛

当前 repack accumulator decision 已满足重要边界：measurement 只能在 `{mf2,m1}` 合法候选中选择，不能创造 compute。该逻辑应保留。

需要收敛的是：

- qualified rows 当前与 compiler C++ 代码混放；
- key 仍常依赖 scale-model string；
- 部分历史 evidence path 不是最新正式 run lineage；
- IME 和 RVV 各自存在 emission-period measurement choice；
- trace/provenance 字段数量远多于真正影响 code 的字段。

这不是要求先建一个庞大 measurement schema。目标只需做到：

1. analytic layer 先产生有限合法候选；
2. residual row 只保存无法稳定解析判断的 winner 或 rejection；
3. miss/expired/inapplicable 时回到 analytic prior；
4. row 不能携带或重建完整 kernel body；
5. 论文报告 residual density，而不是把表本身当贡献。

---

## 4. 当前 CM0–CM4 总表

| Family/slice | 当前最高可信等级 | 已有资产 | 主要阻塞 | 目标 |
|---|---:|---|---|---:|
| dequant flat nibble | CM2 | typed facts、typed core、共享 vector body、bias/min/qh 参数 | decision 位于 emission 附近；carrier 仍选完整 body | CM3 |
| dequant `q4_synth` | CM2 | descriptor-only fan-out 测试 | 复用完全相同 tuple，不是新组合 | 保留为 CM2 causal test |
| dequant codebook | 强 CM2 | 真实 capability projection、finite legal anchors、pre-emission plan | scale decode topology 仍在 emitter 分支 | CM3 |
| dequant KQuant | CM1/弱 CM2 | typed geometry、Plan 字段、多个正确 vector bodies | provider 重包装、`c` 不承重、full-body switch | CM3 |
| dequant Grid | CM1 | entry geometry、typed leaf enum、正确 grid bodies | leaf identity 是完整答案，`c` 不承重 | CM3 |
| dequant Ternary/q1_0 | CM1 | typed outer body、正确 arithmetic bodies | full leaf/monolith selection | CM3 |
| flat repack GEMV/GEMM | family CM2；builder-local CM3 substrate | typed loop regions、core/fold bricks、real `g/c/ω` LMUL decision | q4_0/q4_1/q5_0/q5_1/q8_0 完整 per-point builders | CM3 + q5_1 local authority-erasure evaluation |
| KQuant/codebook/grid repack | CM2 / 局部 CM3 | family facts structs、typed core/fold bricks、shared builders 部分存在 | decode-model branches 与 emitter-internal leaf semantics 仍多 | CM3 |
| generic RVV source front doors | CM3（bounded） | source matching、capability-driven LMUL、typed op graph | domain 较窄，不能替代 block-quant family reconstruction | 保留并作为 construction reference |
| pre-realized selected-body owner | CM3 substrate | resource-driven graph construction、typed ops、budget structure flip | 许多入口仍 test-authored，protocol/metadata 过重 | 接入真实 producer或缩小主张 |
| IME q4_0/q8_0/q4_K | local CM3；family CM1/CM2 | real second backend、typed tile regions、vmadot primitive | per-format materializer blocks；emitter 重决策 | family CM3 |
| Scalar ternary/dequant | CM1/有限 CM2 | real pure-scalar EmitC、reference semantics | typed body hand-authored，plugin boundary unsupported | 保留为 fallback/reference；非首要 CM3 |
| residual selection | CM2 decision slice | legal-set-bounded winner、analytic fallback | rows 与 source 混放、lineage/keys 不统一 | thin bounded residual |

这张表不表示“项目只有 CM1/CM2”。它表示：

- compiler infrastructure 和部分 generic path 已经能做 CM3；
- 论文主域的 family-level construction 尚未一致达到 CM3；
- CM4 尚未通过反事实实验建立。

---

## 5. 关键矛盾及其统一解释

### 5.1 “已经构造 typed body”与“仍是 full leaf selection”并不矛盾

一个逐点 builder 可以创建非常丰富的 typed region，但如果每个 format 仍有一份完整 builder，family 级别仍未因式分解。评价单位必须明确：

- op graph 是否真实构造；
- 完整 point implementation 是否仍逐点存在；
- 新组合是否需要复制 builder；
- emitter 是否仍按 leaf/model 展开完整算法。

因此 repack q5_1 同时可以是“builder-local CM3”与“family-level CM2”。

### 5.2 formula 是贡献核心，不等于 compiler 只剩 formula

用户确认的核心公式保持：

\[
\theta_{i,v}=f^A_{i,v}(g,c,\omega)
\]

\[
K_{v,\omega}(g,c)
=
Emit_v(c)\circ
\bigoplus_i
\left\{
m_{i,v}(g,\omega)\ \text{with}\ \theta_{i,v}
\right\}
\]

它说明 expert knowledge 如何成为可执行实现，不表示 dialect、pass、runtime 或 ABI 可以消失。compiler 的其他层负责：

- 输入和类型表示；
- source/extension boundary；
- legality 与资源安全；
- selected body 的跨 pass 持久化；
- lowering、ABI 与部署；
- correctness/performance evidence。

公式核心与复杂 compiler 本体是包含关系，不是替代关系。

### 5.3 provider/verifier 有工程价值，但没有第一性地位

统一判断如下：

| 结构 | 应保留 | 应简化/删除 |
|---|---|---|
| provider | 真实 plugin lifecycle owner；消费 typed inputs 并构造承重 result 的 family-local builder | 只包装常数、转发 leaf、为“每个公式都有 provider”制造形式对称的层 |
| verifier | ODS type/region/SSA/ABI/ISA/resource invariant；不可信 pass boundary 的 local fail-closed check | 重新运行 planner维护第二份 authority；验证 reason/provenance 文案；用 source shape 证明研究律 |
| stamp | selected plan/body 跨 pass 必须携带的 code-affecting fields | 大量不影响 code 的 mirror、reason、provider、ceremony attrs |

Codebook selected plan 的跨 pass 表示有真实价值；IME 对 typed region 完整性的 verifier 也有真实价值。问题不在“存在 verifier”，而在它是否保护语义，还是复制决策或证明口号。

### 5.4 capability-driven 允许 honest-null，不允许假消费

并非每个 mechanism 都必须由 `c` 改变：

- codebook gather anchor 中 `c` 当前真实承重；
- repack LMUL/resource decision 中 `c` 当前真实承重；
- Grid/Ternary 当前 narrow pipeline 可以诚实标为 capability-honest-null；
- KQuant 当前接收但忽略 `minimumVLEN`，只能写成“future seam”，不能写成已完成 capability construction。

honest-null 比把 capability 参数穿过 provider 而不产生因果作用更符合第一性原理。

### 5.5 第二 family 不等于共享相同 mechanism ABI

RVV 与 IME 应共享的是：

- `g/c/ω` 的角色；
- analytic construction 先于 residual；
- candidate/legality/selection/emission 的因果边界；
- measurement 不能创造 compute；
- 只增加最小新 mechanism primitive。

它们不应被迫共享：

- 同一个 C++ Plan class；
- 同一种 vector/tile op；
- 同一个 emitter interface 之外的深层 ABI；
- 相同 LMUL/tile resource model；
- 相同 format facts struct。

---

## 6. 目标代码模型：保留原公式，建立唯一因果链

### 6.1 逻辑职责

理想逻辑模型为：

```text
FamilyFacts g
TargetCapability c
StaticContext / bounded residual input ω
        │
        ▼
analytic construction fA
        │  builds θ and candidate mechanism compositions
        ▼
finite CandidatePlan set
        │
        ├─ applicability / semantic / resource boundaries
        ▼
finite LegalPlan set
        │
        ├─ analytic prior
        └─ optional qualified residual winner
        ▼
TypedSelectedPlan / TypedBody
        │
        ▼
family-local realization / mechanical emission
```

这里没有要求增加新的全局层。一个 family 可以用：

- 纯函数 + struct；
- table + typed builder；
- 现有 dialect region builder；
- 小型 local plan；
- 现有 pass 中的局部 helper。

只要因果职责正确，不需要新建统一 provider 框架。

### 6.2 构造公式与薄 selector 的关系

构造仍是主体：

\[
\theta_{i,v}=f^A_{i,v}(g,c,\omega)
\]

\[
K_{v,\omega}(g,c)
=
Emit_v(c)\circ
\bigoplus_i
\left\{
m_{i,v}(g,\omega)\ \text{with}\ \theta_{i,v}
\right\}
\]

外层 selector 只能：

1. 接收 formula 已经构造的 candidates；
2. 在 boundary 给出的合法集合中选择；
3. 使用 analytic prior；
4. 在 qualified residual hit 时做受限修正；
5. miss 时回退 prior，empty 时 fallback/reject。

selector 不能：

- 创建新 mechanism；
- 把非法 candidate 变合法；
- 按 format 名拼一份新 body；
- 改变 candidate 的 compute semantics；
- 让 runtime observer 反向定义 compiler compute。

### 6.3 工程中间表示不能替换公式

实现可以使用 local plan、mechanism graph、typed region 或 body fragments 展开原构造项，但不再引入新的论文主公式。每个承重字段都必须能映射回原式的 mechanism、`θ`、capability 或 boundary；中间表示不能让 selector 生成实现，也不能假设 `\bigoplus` 可交换。本轮代码重构完全不依赖更换记号。

### 6.4 typed selected body 的最小要求

最终承载体不必保存所有 provenance，只要足以机械 lowering：

- mechanism/op identity；
- SSA/dataflow/region topology；
- code-affecting parameters；
- ABI binding；
- 必要 resource/layout facts；
- 已选 candidate identity（若存在多候选）；
- 必要 fallback/legality outcome。

reason、evidence path、provider name、diagnostic mirrors 可以留在 trace 或 side output，不应成为 emitter 重建 compute 的依据。

---

## 7. 从当前实现到目标态的最小重构设计

以下是技术收敛顺序，不是 task 列表，也不表示现在已经完成。

### 7.1 收敛 A：先统一 flat repack 的 family construction

这是最适合先做的 slice，因为：

- 已有正确 typed loop/core/fold ops；
- 已有真实 `g/c/ω` accumulator decision；
- q4_0/q4_1/q5_0/q5_1/q8_0 共享大量结构；
- q5_1 提供天然的首个 authority-erasure implementation pilot；
- 现有 tests 和历史 hardware evidence 较丰富。

建议新增一个 family-local、非全局框架的 facts/plan：

```text
FlatRepackFormatFacts
  qk
  weight/activation layout and strides
  quant offsets
  carrier: nibble | full_i8
  signedness / offset-bias policy
  optional high-bit-plane geometry
  fold: scale_only | scale_plus_min
  scale/min/sum offsets

FlatRepackContext
  regime: decode | prefill
  output columns / interleave

FlatRepackResourceDecision
  integer-core LMUL
  accumulator LMUL
  half lanes / columns per pass
  legal schedule candidate
```

随后用两个共享 builder：

```text
buildFlatRepackGemvBody(facts, resourceDecision)
buildFlatRepackGemmBody(facts, resourceDecision)
```

它们继续创建现有：

- `TypedRepackGemv/GemmLoopBodyOp`；
- `RepackLaneWiseQ4Q8DotOp` / GEMM sibling；
- `RepackDualFp16ScaleFoldOp` / GEMM sibling；
- yield 与 ABI ops。

不需要发明新 dialect；关键是让 optional high-bit plane、bias policy 与 affine scale/offset fold 独立组合，而不是选择 `Q41/Q50/Q51` 完整 builder。

收敛完成后应删除：

- `isQ41/isQ50/isQ51/isQ80` 到完整 builder 的链；
- `lowerToRepackGemvQ41/Q50/Q51/Q80` 的重复 region construction；
- `lowerToRepackGemmQ41/Q50/Q51/Q80` 的重复 region construction；
- emitter 中任何仍按 q4/q5 format 名恢复完整结构的逻辑。

格式 descriptor row 可以保留，因为它只描述 `g`，不是完整实现。

### 7.2 收敛 B：把 dequant scale/decode topology 前移成 typed mechanisms

#### Nibble

保留共享 vector body 的算术 lowering，但把 `decideNibbleDecode` 从 emission 期前移到 construction/materialization 期。typed body 应显式表达：

- bare int8 或 nibble unpack；
- optional qh merge；
- bias policy；
- optional affine scale/offset fold；
- store topology。

#### Codebook

把当前 emitter 中的四种 scale decode 分成最小 typed bricks：

```text
ScaleDecodeFp16
ScaleDecodeE8M0
ScaleDecodeUE4M3SubBlock
ScaleDecodeSigned6SuperBlock
CodebookTableLoad
NibbleIndexSplit
VectorGather
ScaleAndStore
```

`decideCodebookGather` 继续负责 capability-dependent gather anchor；formula 根据 `g` 组合 scale-decode topology。这样 Codebook 可以从强 CM2 升到 CM3，而不丢失当前正确 emitter 算术。

#### KQuant

把当前 q2/q3/q4/q5/q6 full-body switch 分解为共享机制：

```text
PackedLowBitsDecode(bit_width, grouping)
OptionalHighBitPlaneMerge
ScaleMinUnpack(layout)
OptionalMinFold
SingleScaleFold
Store
```

`KQuantScaleModel` 可继续作为 facts taxonomy，但不能再等价于完整 body identity。当前 provider 中忽略 `minimumVLEN` 的路径要么：

- 明确标为 honest-null 并移除伪 `c` 参数；要么
- 真正让 capability 选择 legal strip/LMUL，并做 counterfactual test。

#### Grid/Ternary

保留真正独特的 grid table、sign plane、base-3 decode 等 primitive；移除完整 `plan.leaf` body selection。formula 应从：

- entry width；
- index width；
- sign source；
- scale topology；
- ternary arithmetic model

组合 typed graph。新 leaf 只在引入新语义 primitive 时增加最小 op，而不是复制整条 loop。

### 7.3 收敛 C：让 body realization 接入真实 producer，缩减 protocol 镜像

现有 selected-body realization owner 已证明 compiler 能按 resource candidate 构造 graph。后续应选择两条路之一：

1. 把真实 block-quant/source construction 接到该 machinery；或
2. 将其中成熟的 graph builder 抽回 family-local construction，避免维护 test-only pre-realized protocol。

不应继续增加：

- provider-owned reason strings；
- 同一 plan 的多份 mirror attrs；
- planner recomputation verifier；
- 只为 task gate 存在的 admission/provenance ceremony。

应保留：

- typed op graph；
- resource inequality；
- actual candidate choice；
- malformed body/ABI/SSA/resource rejection；
- deferred-wide 与 packed/grouped 等真实结构差异。

### 7.4 收敛 D：IME 采用同一哲学，不共享 RVV 物理模型

IME family-local refactor可以把三个完整 materializer block改成：

```text
IMEFormatFacts
  carrier/decode topology
  optional scale-min topology
  fragment shape

IMECapabilityFacts
  ime op / signedness / mac shape / whole-matrix shape / harts

IMEBodyPlan
  ordered IME mechanisms
  tile types and carried accumulators
  MAC leaf deployment choice
```

一个 local builder 根据 plan 创建：

- q4_0：decode + MAC；
- q8_0：direct load + MAC；
- q4_K：decode + scale/min + MAC + two folds。

batched/unbatched 与 wide-vmadot candidate 应在 typed body 形成前决定并进入结构，不应由 emitter 再运行 deployment selector。保留唯一必要的 inline-asm instruction leaf，因为 IME 没有标准 intrinsic header；这属于不可共享的最小 primitive，不违反 reconstruction。

### 7.5 收敛 E：Scalar 保持 reference/fallback 定位

Scalar 当前无需为证明“统一 contract”而大修。更合理的是：

- 保持 portable reference kernels；
- 修复 plugin emission readiness 与真实 emitter 的接口不一致；
- 让 typed boundary 能由 plugin 正常 materialize；
- 只有在它能复用 facts/mechanisms 构造多个 scalar formats 时，再把它纳入 CM3 family claim。

这避免为了第二 family 数量而制造形式复用。

### 7.6 收敛 F：verifier/provider 三分法

对现有结构逐个按功能分类：

#### A 类：必须保留的 semantic safety

- typed region op 数量、顺序与 SSA 连接；
- SEW/LMUL/widening chain；
- qk、stride、offset 与 ABI 不变量；
- matrix/tile fragment shape；
- register/resource legality；
- selected body 与 runtime ABI 的一致性。

#### B 类：迁移完成后可删除的 duplicate authority

- verifier 重新运行 planner 并比较 selected attrs；
- emitter 从 attrs 重建一次完整 decision；
- provider、materializer、reader 各保存同一 code-affecting state；
- 同一候选在 producer/consumer scope 重复 materialize。

#### C 类：不应再作为科研纪律的 source-shape ceremony

- 检查某个 provider 名必须出现；
- 检查 reason 字符串必须包含特定短语；
- 用 grep/regex 证明六律；
- 用 provenance 数量代替 knowledge causality；
- 用 census 数字代替 compiler behavior。

---

## 8. 第一项 authority-erasure 实现里程碑：q5_1 flat repack

### 8.1 为什么选 q5_1

q5_1 是真实、已有部署意义的 GGML 格式，其语义恰好是已有机制的交叉组合。它适合让第一轮代码重构证明删除 point authority 后的本地 reconstructability。由于团队已经了解旧实现，且 QIGen 已覆盖 5-bit affine group 与 GEVM/GEPM composition，它既不单独证明 unseen generalization，也不再承担主要科研差额：

| 组成 | 已有来源 |
|---|---|
| unsigned low nibble | q4_1 / q5 family 共享 core |
| optional 5th-bit `qh` plane | q5_0 |
| no `-16` centering bias | q5_1 format fact；由 bias policy 参数表达 |
| dual fp16 scale | flat q4/q5 shared fold |
| affine offset correction `m_x*s_y`（历史实现常称 min fold） | q4_1 |
| decode GEMV / prefill GEMM regimes | 现有 shared loop skeletons |
| LMUL/resource selection | 现有 repack `g/c/ω` decision |

它不要求发明新的核心 primitive，而要求把两个已有机制真正组合，因此比 `q4_synth` 更能区分“descriptor alias”与“真实 authority erasure”。论文级确认仍需在此后跨出 flat affine q4/q5，覆盖至少一种不同的 GGML 语义 topology。

### 8.2 必须 withheld 的对象

确认性实验中必须删除或在构建上完全排除：

- `lowerToRepackGemvQ51`；
- `lowerToRepackGemmQ51`；
- `isQ51 ? ...Q51(...)` dispatch；
- q5_1-specific complete emitter helper/branch（若还有）；
- 任何用 `q5_1` 名称返回一份完整 mechanism sequence 的 registry row。

可以保留：

- q5_1 的格式事实 row：stride、offset、qk、activation layout；
- generic `optional high-bit plane` primitive；
- generic `scale + optional min` fold primitive；
- generic GEMV/GEMM loop skeleton；
- existing resource candidates 与 capability logic；
- q5_1 measurement winner 可以在 analytic-only correctness 已独立通过后用于性能 residual arm；它不能参与 candidate creation、正确性或 prospective generalization 证据。

### 8.3 防止答案泄漏

仅删除两个函数还不够。必须检查：

1. format string switch 是否在别处恢复 q5_1 body；
2. `scale_model` row 是否携带完整 op sequence，而不只是 facts；
3. emitter 是否按 q5_1 名或 Q51 enum 选择 helper；
4. verifier 是否偷偷重建 point plan；
5. test fixture 是否直接书写目标 typed body；
6. generated table 是否保存完整 q5_1 pipeline；
7. measurement row 是否只选择 legal plan，而非编码 compute。
8. 是否用多个布尔 facts 拼成只识别 q5_1 的组合指纹；每个参与机制或 predicate 应在其他实例中有独立用途。

旧 Q51 实现可以留在隔离的 baseline/oracle 构建中，但不得链接回 treatment path。

### 8.4 两个实验臂

#### A. Analytic reconstruction arm

- `ω` 强制 miss/empty；
- 从 q5_1 `g`、真实 `c` 与共享 mechanisms 构造；
- 必须同时生成 decode GEMV 与 prefill GEMM body；
- 证明正确性和合法性不依赖 q5_1 measurement row。

这个实验回答“formula/mechanisms 能否重建 compute”。

#### B. Bounded residual arm

- candidate set 已由 analytic layer 构造；
- qualified measurement 只允许在 legal `{mf2,m1}` 或其他显式候选中选 winner；
- miss、expired、inapplicable 时回到 analytic prior；
- 比较 residual 前后性能，但 compute semantics 保持不变。

这个实验回答“measurement 是否只是残差修正”。

### 8.5 必须通过的证据

#### Construction causality

- q5_1 body 中实际出现 `high-bit-plane merge + no-bias + affine scale/offset fold`；
- 移除 qh fact 后 graph 或 correctness 必须按预期变化；
- 移除 min fact 后 graph 或 correctness必须按预期变化；
- 改变 capability 后 LMUL/half-lanes/legal candidates 按预期变化；
- reason/provenance 删除不改变 emitted code。

#### Correctness

- randomized q5_1 blocks；
- high-bit plane 全 0/全 1/交错；
- nibble 边界 0/15；
- min/scale 正负与极值；
- odd/even block counts；
- decode 与 prefill；
- 与 independent scalar/ggml oracle 对比；
- 对浮点 fold 使用项目既有容差或 byte-exact contract，不事后更换。

#### Compiler and ABI

- typed IR graph test；
- generated C/code-shape test；
- generated bundle/ABI test；
- fail-closed malformed facts test；
- 无 q5_1完整 helper/branch 的静态依赖检查；
- 现有相关回归全部通过。

#### Performance

- 与重构前 q5_1 point implementation 比较，确认重构本身不产生不可解释退化；
- 与 stock/native-vector 或项目既定强对手比较；
- decode GEMV 与 prefill GEMM 分开；
- 至少覆盖一条 VLEN128 与一条 VLEN256 路径；
- kernel 与 e2e 分开报告；
- 历史 1.306x 等数字只能作为可行性线索，最终结论需由新的正式 run lineage 产生。

性能阈值、板卡、工具链、problem sizes 与统计规则应在确认性运行前冻结；本阶段不凭旧结果事后设阈值。

### 8.6 成功和失败分别说明什么

| 结果 | 解释 |
|---|---|
| 无任何 Q51 procedural authority，analytic arm 正确且性能接近/达到既定目标 | 建立首个本地 authority-erasure/reconstruction 证据；支持 flat family 重构，不单独支持 novelty 或 unseen generalization。 |
| correctness 成功但性能显著退化 | 证明 construction 可行，但高性能知识尚未正确因式分解；需收缩性能主张或改机制。 |
| 只有保留 q5_1 measurement row 才能正确 | residual 越权创造 compute，目标失败。 |
| 删除 builder 后仍由 emitter q5_1 branch 工作 | hidden-leaf 泄漏，不计 CM4。 |
| 必须新增完整 q5_1 primitive | 实质仍是 point implementation，不计 family reconstruction。 |
| 仅 descriptor alias 产生与 q4_0 相同代码 | 类似 q4_synth，只是 CM2 fan-out，不是 q5_1 reconstruction。 |

---

## 9. 完整 evaluation program

### 9.1 Construction mutation

对每条核心 path 做最少一项反事实：

| 输入或输出 | 预期影响 |
|---|---|
| format geometry `g` | offsets、loop bounds、mechanism presence 或 layout 改变 |
| capability `c` | legal candidates、LMUL/tile/resource/fallback 改变 |
| context `ω` 中非测量静态事实 | regime、shape、boundary 或 analytic prior 改变 |
| qualified measurement | 只改变 legal candidate winner |
| reason/provenance | 不应改变 code |

### 9.2 Rule fan-out

报告至少三种数量，不用单一“复用率”掩盖问题：

1. 一个 mechanism primitive 被多少格式/操作/family 消费；
2. 一个 analytic rule 构造多少不同 typed graphs；
3. 新增一个 valid combination 需要改多少 code-affecting owner。

同时区分：

- 参数变化；
- topology 变化；
- 新语义 primitive；
- 完整 point implementation。

### 9.3 Factorization ablation

建议比较：

1. 现有 pointwise/full-leaf baseline；
2. mechanisms + fixed defaults；
3. mechanisms + analytic `g/c` construction；
4. mechanisms + analytic construction + bounded residual。

指标包括：

- correctness coverage；
- legal/fallback coverage；
- typed graph/code shape；
- code locality/fan-out；
- compile time；
- kernel performance；
- e2e transmission；
- residual rows 与 hit/miss。

### 9.4 Residual density

至少报告：

\[
D_{coverage}
=
\frac{\#\{\text{具有 qualified residual 的 eligible keys}\}}
{\#\{\text{至少有两个合法候选的 eligible keys}\}}
\]

\[
D_{intervention}
=
\frac{\#\{\text{residual 实际改变 analytic winner 的 keys}\}}
{\#\{\text{eligible keys}\}}
\]

同时报告 analytic-only 到 analytic+residual 的性能增益、miss/expired/inapplicable、以及 residual 是否携带任何 compute 语义。

如果 residual 近似逐点覆盖整个产品空间，则“解析主导、测量修残”没有被证明，应降低主张或重新找 key abstraction。

### 9.5 跨 family 证据

RVV 与 IME 至少各选择一个真实 slice：

- RVV：q5_1 flat repack local authority-erasure reconstruction，随后补一个非 affine-flat topology；
- IME：一个由 format facts + IME capability 构造的 tile family，且 MAC deployment 在 emitter 前确定。

跨 family 的评价不要求共享 C++ 类，而要求相同知识边界在两种硬件范式中都能成立。

---

## 10. Novelty 收敛：可以说什么，不能说什么

### 10.1 当前可以作为目标态研究主张的表述

> Weft-RV investigates how irregular block-quant execution knowledge can be factorized into typed format facts, target capabilities, reusable mechanisms, analytic construction rules, explicit boundaries, and sparse empirical residuals inside a multi-family execution-layer compiler.

中文含义：

> Weft-RV 研究如何在 multi-family execution-layer compiler 中，把不规则 block-quant execution knowledge 分解为 typed 格式事实、目标能力、可复用机制、解析构造规则、显式边界和稀疏经验残差。

这仍是目标态描述。只有完成 CM3/CM4 与性能证据后，才能改为已实现结果。

### 10.2 最可能成立的贡献差额

如果后续证据成立，差额可落在：

1. irregular GGML block-quant execution knowledge 的具体分解对象；
2. high-level MLIR 之后的 typed execution-layer integration；
3. `g/c/ω` 对 mechanism composition、resource boundary 与 selected body 的真实因果作用；
4. analytic construction 主导、measurement 只在合法候选内修残；
5. RVV 与 IME 两种 family-local realization；
6. 多 topology 的 point-authority erasure 与 intra-microkernel mechanism reconstruction；若主张 unseen/generalization，再补冻结后的 prospective transfer；
7. rule fan-out、residual density、部署和强对手联合评价。

### 10.3 不能再使用的宽泛表述

- 首次把专家知识变成可执行公式；
- 首次用机制组合生成 kernel family；
- 首次把 analytic model 与 measurement 结合；
- 首次 construction-based compiler；
- 首次 typed extensible execution compiler；
- 首个量化 LLM CPU kernel generator；
- provider/verifier/Plan 生命周期本身是贡献；
- 当前已经能够从任意 `g/c/ω` 重建新格式；
- `q4_synth` 已经证明 reconstruction；
- 第二 family 已经共享同一 universal formula ABI。

### 10.4 失败时的诚实降级路径

如果 q5_1 或后续 topology witness 不能在无 point branch 条件下保持 correctness/performance，则仍可保留较窄价值：

- 一个 typed、capability-driven、multi-backend execution-layer compiler；
- 对 GGML/RISC-V quant kernels 的工程系统化；
- parameterized construction 与 decision authority consolidation；
- 真实负结果和部署经验。

这会降低 Knowledge Factorization 的 novelty 强度，但不会把整个 compiler 资产归零。

---

## 11. 对 method/spec 的后续影响

在完成实现和 Phase 4 独立审查前，不应大改现有 spec。后续只建议做小范围收敛：

1. 删除已证伪、已完成或明显历史化的状态描述；
2. 明确“当前状态 / 目标态 / 证据要求”；
3. 保留已确认核心公式与 selector 主次；
4. 把 construction 定义为真实 causal body/plan construction，而非 typed wrapper；
5. 把 capability honest-null 写清楚；
6. 把 provider/verifier 降为普通实现技术；
7. 把 measurement 限定为 legal-candidate residual；
8. 增加多 topology retrospective reconstruction、rule-level fan-out 与 residual density 的评价要求；仅在主张 unseen/generalization 时增加冻结后的 prospective transfer；
9. 六律只沿既有意图微调措辞和证据方式，不重新解释；
10. 不创建 task，直到方法与重构边界经最终审查确认。

---

## 12. Devil’s Advocate Checkpoint 2

### Verdict：REVISE before implementation freeze

这不是拒绝方向，而是指出：目标 architecture 已经足够清楚，但在正式冻结重构和论文 novelty 前还有几项必须解决的风险。

### Critical issues

没有发现不可修复的 critical issue。

### Major issue 1：q5_1 只是第一项本地重构，不足以单独承担 novelty

q5_1 的组合关系已写在当前源码注释中。即使删除 Q51 builders，也可能只证明一次优秀重构，而非一般方法。

处理要求：

- q5_1 作为首个 authority-erasure 实现里程碑，不作为 prospective generalization、主要 novelty 或唯一证据；
- 同时报告 rule fan-out 与 residual density；
- 至少再有一条超出 QIGen 仿射 `(g,b,s,z)` 模型的不同 mechanism topology；
- IME 提供跨 family 的概念复用证据。

### Major issue 2：format facts row 仍可能藏入完整答案

如果 `FlatRepackFormatFacts` 最终保存一串完整 op sequence，所谓 factorization 只是把 leaf 从函数搬到数据表。

处理要求：

- facts 只能描述格式、布局与语义轴；
- mechanism sequence 应由通用 analytic rule 从正交 facts 推导；
- 检查新增组合是否只增加 facts，而不增加完整 sequence/branch；
- mutation test 证明每个轴独立承重。

### Major issue 3：当前 strongest CM3 substrate 与论文主 workload 之间有生产链断层

selected-body realization owner 很强，但许多 pre-realized body 是 test-authored；block-quant production front door 又保留逐点 builders。

处理要求：

- 不以 test fixture 证明生产 CM3；
- q5_1 必须从真实 `GgmlQuantContractionOp` 或真实 workload path 进入；
- generated bundle 与板端运行必须覆盖重构后的 production chain。

### Major issue 4：IME emitter 仍持有 decision authority

IME typed region 已经存在，但 batched/unbatched、wide deployment 与 per-format helper 仍在 emitter 决定。若不前移，第二 family 只能证明 typed leaf emission。

处理要求：

- MAC leaf/deployment choice 在 typed body 前确定；
- emitter 读取 op identity/types，不重新调用 selector；
- 保留 unavoidable asm leaf，但不把它扩张成完整 point kernel。

### Major issue 5：QIGen 2026 全文差分已关闭，剩余 claim 必须再次收窄

正式全文确认 QIGen 从 W/S/Z 与 CPU 信息出发，生成 unique `(M,K,g,b)` microkernels，并按非均匀 group sequence 组合 layer implementation；cache model、grid search、GEVM/GEPM 和 x86→ARM port 都已存在。它的完整 per-bit algorithm 仍位于 bitwidth switch 中，因此剩余可检验差额是更深的 intra-microkernel semantic reconstruction，而不是“有无 construction”。

处理要求：

- 不能用“未见相同术语”作为差额；
- q5_1 降为本地 pilot，确认性证据必须跨 topology；
- 现阶段继续把 novelty 标为 provisional。

### Major issue 6：旧硬件数字与新确认性证据不能混用

当前源码注释包含 q5_1 1.306x 等历史结果，但它们不能自动成为重构后论文数字。

处理要求：

- 历史结果只用于选择 witness 和估计可行性；
- 重构后走正式 runner、correctness gate 与 run lineage；
- 预先冻结比较对象和阈值；
- 负结果同样保留。

### Strongest counter-argument

> Weft-RV 已经有许多 typed ops、plans、providers 和 tests，但真正的性能仍由逐格式 builder 与 emitter 大函数保存。把 q5_1 的两个函数合并成一个参数化 builder，只能说明代码去重，不能说明一种新的知识模型。SPIRAL、Exo 2、ROLLER、Ladder 和 QIGen 已经覆盖规则生成、可组合 primitives、construction 与量化 kernel generation；除非 Weft 在真实 production chain 上展示无完整 point implementation 的跨组合重建、稀疏 residual、跨 family 复用和竞争性能，否则它仍是一个工程优秀但 novelty 较弱的专用 compiler。

这个反驳目前成立。QIGen 全文还使它更强：后续不能只靠 q5_1 flat composition，而要用 q5_1 authority-erasure pilot、多 topology reconstruction、第二 operation/backend、ablation、residual density 与真实硬件/端到端结果实质性击败它。只有声称 unseen/generalization 时才另加 prospective witness。

### 第二检查点结论

建议进入 Phase 4 独立审查，但在实现冻结前保持以下状态：

- 两柱：不变；
- 已确认公式：不变；
- 新六律候选：只做语义血缘评审，不直接替换；
- novelty：provisional、已降级；
- q5_1：首个 authority-erasure implementation design，不是当前成果、主要 novelty 或 prospective holdout；
- provider/verifier：普通工程技术；
- spec：暂不大修；
- task：暂不建立。

---

## 13. 阶段三冻结建议

建议将以下内容作为 Phase 4 输入：

1. 泛化 knowledge factorization、typed plugin、analytic+measurement、construction-based generation 均非独立 novelty。
2. Weft 的潜在差额必须限定在 irregular block-quant execution layer、typed `g/c/ω`、mechanism composition、explicit boundaries、sparse residual 与 multi-family realization 的组合。
3. 当前代码拥有真实 CM3 substrate，但 block-quant family 主要处于 CM1/CM2/局部 CM3，尚无 CM4。
4. `q4_synth` 是有价值的 descriptor-only CM2 falsifier，不是 reconstruction。
5. q5_1 flat repack 是第一项未来 authority-erasure 实现里程碑；当前做不到是重构目标，不是方向否决，也不能单独证明 novelty 或 unseen generalization。
6. 首轮代码动作应统一 flat repack facts → mechanisms → typed body，删除 Q51 point builders，而不是大修 compiler。
7. Codebook 是 dequant CM3 的最佳第二 pilot；KQuant/Grid/Ternary 后续按最小 mechanism graph 收敛。
8. IME 是真实第二 compute family，但必须去掉 per-format full materializer 与 emission-period selector 才能支撑跨 family 主张。
9. Scalar 保留为 fallback/reference，不为了形式完整强行升级。
10. 只保留 semantic verifier；删除 duplicate planner、mirror authority 与 source-shape ceremony。
11. 最终证据必须包含多 topology retrospective reconstruction，并覆盖 causal mutation、capability counterfactual、fan-out、ablation、residual density、deployment 和 performance；若主张泛化则再含 prospective holdout。
12. 在 Phase 4 独立审查和实现证据完成前，不改写 canon，不建立 task，不宣称目标态已经实现。
