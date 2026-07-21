# Weft-RV 项目全景与 Spec 重构前方法基线

> **性质**：内部长期维护方法文档，也是下一轮 spec 与代码重构的核心理解基线。它以用户重新确认的项目出发点、[《高级ai思想2》](./高级ai思想2.md)及本文的纠偏综合为主。
>
> **参考关系**：[ARS 完整审计](../执行知识因式分解完整审计与收敛设计-2026-07-21/README.md)继续提供 prior、源码反例、QIGen 差分和证据压力，但不是项目类别或重构架构的核心标准；不能用一个近邻下的最窄差分实验反向定义整个 Weft。
>
> **边界**：不替代同目录的 [原接手文档](./README.md)，不由工程 agent 直接修改两柱、六律或贡献编号。它规定理解和重构方向，不建立新的全局 Formula IR、universal verifier 或日常审批流程。

## 1. 这份文档解决什么

Weft-RV 目前同时有论文侧材料、项目侧统一公式、复杂 compiler 实现、历史 census、实验记录和多轮 agent 施工。接下来重构 spec，首先要恢复这些对象之间的正确关系，而不是重新定义项目。

本文固定六件事：

1. 科研主线由用户和论文 agent 决定，工程 agent 不改律、不重排贡献；
2. 项目侧公式已经完成统一，早期论文公式不是当前完整版本；
3. Weft 是面向生态扩展的 MLIR operator compiler / execution-layer software stack；RISC-V 量化推理是旗舰 reference realization，不是系统类别的上界；
4. 两柱共同回答“变化能否局部接入”与“可扩展架构能否仍产生专家级或有竞争力的专化实现”；
5. 公式是 execution knowledge 构造 kernel candidate 的核心，不是整个 compiler；Knowledge Factorization 是两柱之间“分开变化轴、重新专化”的设计桥梁，不是第三柱或系统身份；
6. 历史文档、census、issue、task 和局部审计都只是辅助材料，不是限制代码演进的手续。

必须始终区分三个层次：

| 层次 | 稳定含义 |
|---|---|
| 系统身份 | 可扩展的 MLIR operator compiler / execution-layer software stack |
| 旗舰实现 | 复杂、碎片化 RISC-V 上的 ggml/llama.cpp 风格量化推理栈 |
| 当前证据 | 各 plugin、operator、公式、硬件与部署路径已经实际完成和验证到哪里 |

当前证据的范围可以限制结果措辞，不能反向缩小系统身份；系统身份也不能把目标态冒充成已经跨任意 operator 和硬件得到验证。

还必须固定一个过去容易混淆的时态：

> **当前代码尚不能普遍在删除完整逐点实现后，仅由 `g/c/ω + mechanisms + formula` 重建该实例；让真实 production compiler 做到这一点，是后续目标，不是本文对现状的虚假描述。**

## 2. 科研主线与工程工作的边界

两柱、六律和贡献组织属于论文设计。工程侧可以核对实现、发现反例、指出版本漂移，但不能自行修改或重新解释。

尤其禁止再次发生以下替换：

- 用当前代码的方便形态替换论文主张；
- 用一轮 task 的验收结构替换六律；
- 用“统一 Plan”“完整 stamp”或“第二 family contract”重新定义项目完成条件；
- 因为某个公式尚未完全落地，就把复杂 compiler 缩成一个直接代码生成器；
- 因为 compiler 很复杂，就反过来弱化公式的科研主体地位。

当前代码和测试决定“现在实现了什么”；论文侧决定“科研主张是什么”；spec 的工作是准确连接两者。

项目出发点已经由用户再次明确：Weft 首先是一个生态友好、可扩展的 MLIR 算子 compiler/software stack。后续 spec 不得再用“当前主要在 RISC-V 上测量”把它改写成 RISC-V 专用量化 compiler，也不得用 QIGen 的局部 prior 把它改写成 microkernel-leaf-below factorization 系统。

## 3. 已确定的公式主次

### 3.1 核心贡献公式：构造式

柱二的主要公式是：

\[
\theta_{i,v}
=
f^A_{i,v}(g,c,\omega)
\]

\[
K_{v,\omega}(g,c)
=
Emit_v(c)\circ
\bigoplus_i
\left\{
m_{i,v}(g,\omega)
\;\text{with}\;
\theta_{i,v}
\right\}
\]

这两式说明专家知识怎样成为可执行实现：

- `g` 表示格式语义、编码、布局等独立 typed facts；
- `c` 表示目标能力；
- `ω` 表示确实承重的有限静态上下文；
- `m` 是可组合的 typed mechanism/component；
- `f^A` 把专家的解析知识变成 typed 参数 `θ`；
- `K_{v,ω}` 是一个真实可实现的 candidate；
- `Emit_v(c)` 将已构造的 candidate 投影到目标实现。

构造公式是贡献主体。它负责创造和组织：

- candidate；
- typed 参数；
- mechanism/body；
- 合法域所需的解析条件、资源要求与边界知识；
- 解析 prior 所需的知识。

“mechanical emitter”表示 emitter 不重新发明 candidate、mechanism 或选择结果，不表示 emitter 不能消费 `c`，也不表示 backend lowering 只有几行代码。

在工程上可以把这条知识链解释为 `M/F/B/R`：

- `M`：可复用 typed mechanisms；
- `F`：从 `g/c/ω` 推导 `θ`、机制组合、candidate structure、资源要求和 analytic prior；
- `B`：显式判断 applicability、legality、resource admissibility、fallback 与 negative case；
- `R`：只在有限合法候选中保存经验 winner。

这四个字母只是解释职责，不是第三柱、四个新贡献或一套必须统一的 C++ 框架。理论上，构造公式及其解析知识仍定义 candidate 和合法域所需条件；工程上的 boundary 只是把适用性与合法性判定显式化，不能成为独立 compute authority。

### 3.2 外层选择：薄包装

定义 measurement key：

\[
\kappa=k(g,c,\omega)
\]

定义合法候选集：

\[
\mathcal V(g,c;\omega)
=
\left\{
v\in\mathcal A(g,c;\omega)
\mid
L_v(g,c,\omega)=1
\right\}
\]

令 `W_M` 只包含已经资格化、未过期、可用于选择的 winner memory：

\[
\sigma_M(g,c;\omega)=
\begin{cases}
W_M(\kappa),
&
W_M(\kappa)\downarrow
\land
W_M(\kappa)\in\mathcal V
\\[4pt]
\pi(\mathcal V;g,c,\omega),
&
W_M(\kappa)\text{ 不可用}
\land
\mathcal V\neq\varnothing
\\[4pt]
\bot,
&
\mathcal V=\varnothing
\end{cases}
\]

最终：

\[
K_M^*(g,c;\omega)=
\begin{cases}
K_{\sigma_M(g,c;\omega),\omega}(g,c),
&
\sigma_M(g,c;\omega)\neq\bot
\\
\mathsf{FallbackOrReject}(g,c,\omega),
&
\sigma_M(g,c;\omega)=\bot
\end{cases}
\]

`qualified`、`fresh` 和 `selection-valid` 的具体定义属于方法或附录，不挤占主公式。

### 3.3 正确理解

主次关系是：

1. 构造公式创造候选、参数、typed body，并给出合法域所需的解析条件，是公式贡献主体；
2. 解析 prior `π` 是公式知识的一部分，不是无知识的默认排序；
3. measurement winner 只是有边界的残差修正；
4. selector 不能创造实现，不能改变计算语义，不能把非法 candidate 变合法；
5. runtime observer 当前不进入主系统。

所以项目不是从“公式贡献”改成“通用 selector 贡献”。准确表述是：

> 原构造公式继续作为核心，外面增加一个符合真实项目状态的薄选择语义。

项目侧当前 [变体流水线](../../.trellis/spec/architecture/变体流水线.md)已经包含这套统一方向。后续只需检查 qualification 是否喧宾夺主，以及 formula、boundary、selector、emitter 是否重复承权；不再发明新公式。

## 4. 公式在完整 operator compiler 中的位置

Weft 是 high-level MLIR 之后承接 operator execution 的可扩展 MLIR compiler/software stack。它不以新增通用 tensor/tile IR 为目标，而是为 operator、format/layout、capability、execution family 和 backend 提供局部 typed 扩展与完整编译链。当前最完整、最具挑战性的实例是 RISC-V 量化推理，但 stack 的 extension contract、typed ownership 和 construction pipeline 不应由 RVV 或 GGML 名称定义。

完整 compiler 层次是：

```text
MLIR operator / kernel-level input
  → source front door 与 extension discovery
  → operator / format / layout typed facts
  → canonical capability 与有限静态 context
  → 解析构造公式、schedule、legality 与薄选择
  → selected typed body / 跨 pass 表示
  → plugin-local body realization 与 conversion
  → RVV / IME / Scalar / Offload / future family backend
  → EmitC、bundle、ABI 与 toolchain
  → runtime integration、correctness 与硬件 evidence
```

这意味着：

- formula/Plan 是知识构造中心，不是 compiler 的全部 IR；
- legality 同时包含公式合法域和真实 IR/ISA/resource invariant；
- typed body 既承载计算语义，也承担跨 pass 的工程接口；
- emitter 应避免重新选择，但仍承载大量 target lowering、vector code shape、ABI 和 runtime 细节；
- measurement 验证并修正有限残差，不能成为 compute authority；
- plugin、dialect、pass、backend、测试和部署都属于两柱落地所需的 compiler 本体。

两柱的共同机制可以概括为：

> **沿独立变化轴分开表示，以获得扩展局部性；再由 capability- and context-conditioned executable knowledge 在编译时重组，以产生专化实现。**

Knowledge Factorization 只描述这座桥：它既不能吞掉整个 software stack，也不能把所有新语义强迫拆成“零 leaf”。完整 point implementation 不应是默认扩展单位；真正不可复用的新语义仍可增加最小 mechanism。

## 5. 当前实现应怎样理解

### 5.1 已有的长期 compiler 资产

当前项目已经拥有：

- `weft.exec` execution envelope；
- RVV 等 typed extension dialect 和 ODS/TableGen op；
- RVV、IME、Scalar、Offload、Template、Toy、Demo、TensorExtLite 等 extension family；
- capability descriptor、provider、TargetCapabilitySet 和部分 typed capability consumption；
- construction、schedule、body realization、conversion 和 EmitC pipeline；
- block-quant、KQuant、grid/codebook、ternary、repack、widening、mask/tail、memory/reduction 等大量量化 kernel 知识；
- generated bundle、ABI、runtime metadata 和部署路径；
- 近千项 lit/C++ 测试以及真板、strong opponent、e2e 和负结果资产；
- official bench runner 与 measurement/evidence 基础设施。

不能把这些资产压缩成“五个 Plan + 一个 emitter”。

### 5.2 当前 construction 的诚实状态

完整源码审计得到的保守结论是：

| 路径 | 当前可信状态 | 不能写成什么 |
|---|---|---|
| RVV dequant nibble/codebook | 参数构造较强；Codebook 中 `c` 真实影响合法 gather/LMUL | 已经从共享机制重建整个 dequant family |
| KQuant/Grid/Ternary/q1_0 | 主要仍是 typed leaf/full-body selection | provider/typed op 存在即真实 construction |
| flat repack GEMV/GEMM | 有真实 typed brick/body construction；family 仍有完整逐格式 builders | 整个 family 已完成无 point-leaf construction |
| selected-body realization | realization 阶段能构造真实 operation graph | test-authored入口即可证明 production block-quant reconstruction |
| IME | 真实第二 compute backend，有局部 typed region construction | 已经是第二个完成 reconstruction 的 family |
| Scalar | 真实 fallback/reference backend | 为了形式对称而算作 construction family |

因此，项目不是“只有 metadata”，也不是“已经完成 knowledge factorization”。更准确的状态是：

> **复杂 compiler 和局部 construction substrate 已经存在；主 block-quant family 的 compute authority 仍散落在逐格式 builder、provider 和 emitter 大分支中。**

### 5.3 后续目标的因果定义

后续要做到的是：

```text
independent format facts + real capability + bounded static context
  → analytic formula and explicit boundary
  → reusable mechanism composition with typed θ
  → typed body
  → mechanical family realization/emission
```

删除某个完整 point implementation 后，如果 downstream 还能按 format 名、leaf enum、完整 plan row、测试 fixture、winner row 或 emitter switch 选回原 body，就不算重建。

允许保留：

- 格式不可约的语义事实；
- 真正不可再分、且可独立复用的 mechanism primitive；
- ABI/ISA/safety boundary；
- 与 treatment 隔离的 reference oracle 和旧 baseline。

目标不是“零 leaf”，而是不再为每个完整 `format × capability × context × regime` 点保存一份程序答案。

### 5.4 后来的 Codex A/B campaign

A1–A4a、B1–B3 campaign 包含 formula authority matrix、集中 decision、Codebook/Repack materialization、measurement control plane、bench coverage 和旧路径退役等工作。

用户已经明确批评其整体方向：A 线逐渐从“把专家知识和 compiler authority 做清楚”偏成了 selected stamp、materializer、reader、planner replay、forged/stale 状态和 task gate 的扩张。

后续不继续这条 campaign，但也不按 commit 整笔判死。正确做法是直接看当前代码：

- 有用的 typed facts、Plan、公式、legality 和测试留下；
- 重复 owner、无意义状态、死 metadata 和只服务旧 task 的结构删除；
- 真实跨 pass invariant、malformed IR 诊断、ABI 与 code-shape 防护不能误删；
- 具体实现该重写就重写，不为保留旧 task 边界维持坏结构。

### 5.5 现有接手文档

[原接手文档](./README.md)正确识别了 A 线被 stamp 生命周期吸走、authority 仍散落、B 线应继续验证真实性能等问题。它是重要的纠偏入口，但主要讨论最近一轮 A/B 偏航，不是整个项目或论文主线的新定义。

## 6. 怎样看待历史 census

项目已经做过多轮 census：格式/能力烘焙、g/c/θ/f、单侧公式、ForwardElementwise、authority matrix、coverage 六态等。这些工作证明项目并不是从零开始，也提供了大量代码入口和历史判断。

但 method 和稳定 spec 不需要建立一套 census 引用仪式。无需规定每次引用数字都必须同时填写 pin、scope、谓词、分类、raw table 等字段，也无需在改代码前先恢复所有历史计数。

更简单的读法是：

- 需要找历史热点或理解某个判断为何出现时，去看对应 census；
- 要判断当前代码时，直接读当前代码和相关测试；
- 数字口径明显不同就不要机械比较；
- census 发现的问题已经被代码变化消解时，不为维护历史曲线阻碍重构；
- census、issue 和 task 都是辅助记忆，不是开发许可或架构 authority。

论文确认性实验仍需足够严谨，但这不应扩张成日常改代码的手续。普通重构依靠当前代码、测试和风险相称的验证；只有准备正式 reconstruction/performance 主张时，才冻结组合、机制/公式、对手和成功条件。

## 7. Novelty 的当前边界

外部 prior 已经否定以下宽泛首创说法：

- 规则/公式驱动的程序生成；
- 可组合 mechanisms 或 rule fan-out；
- analytic construction 代替大搜索；
- typed extensible compiler；
- analytic prior 加 empirical selection；
- hardware-aware low-precision compiler；
- 由量化描述和 CPU 特征生成 LLM kernel。

QIGen 2026 全文进一步确认：它从非均匀量化模型与 CPU 特征出发，枚举 `(M,K,g,b)`，为 unique tuple 生成并调优 GEVM/GEPM LLVM microkernel，再按每层 group 序列组合 implementation。它不是 monolithic whole-matrix leaf lookup。因此，family generation、cross-combination、GEMV/GEMM regime、解析模型加有限搜索和 CPU retargeting 都不是明显空白。

QIGen 的实际生成边界仍是重要事实：其 group/microkernel 组合是真实构造，microkernel 内部则用 2/3/4/5/6/8 位 `switch` 保存完整 unpack/shift/mask/dot body。它限制的是柱二“量化 GEVM/GEPM construction”切片上的宽泛 novelty，而不是 Weft 的系统类别。

Weft 的整体研究命题应恢复为：

> **一个面向生态扩展的 MLIR operator compiler，能否让 operator/format、hardware capability 与 backend family 的变化主要落在各自 typed owner，同时由 `g/c/ω` 条件化的可执行专家知识构造专化、合法且具有专家级质量或竞争性能的 kernel？**

这项系统命题包含两个必须联合验证的方面：

1. **扩展局部性**：增加 operator、格式、能力、mechanism 或 backend 时，不重新形成跨轴笛卡尔积手写；
2. **性能因果性**：语义、能力、资源、场景和有限 residual 如何真实改变 constructed typed body 与最终性能。

QIGen microkernel leaf 以下的 mechanism reconstruction、q5_1 authority erasure、Codebook/KQuant topology、IME authority cleanup 和 residual density 都是辨识这些方面的实验工具，不是 Weft 的定义或唯一生存条件。

`M/F/B/R` 与 Knowledge Factorization 是工程解释语言，不自动构成新理论。最终 novelty 只能来自该 software stack 的具体 architecture、局部扩展行为、可执行 construction 因果链，以及真实实例上的联合系统证据。

### 7.1 Knowledge Factorization 与六律的关系

《高级ai思想2》的核心纠偏应吸收为两柱之间的设计方法：

```text
operator / format / layout facts
+ capability facts
+ reusable mechanisms
+ analytic construction
+ applicability / legality
+ optional empirical residual
→ specialized typed implementation
```

但它不能由工程 agent 直接“定稿新六律”。当前 canon 中受管 measurement、经验点征税、变量充分、fallback 和 falsifier 等约束不能因改名而丢失。后续若论文 agent 修改六律，应把以下解释合入而不是整体换义：

- 知识沿变化轴分开，是变化归位与和积扩展的直观目标；
- formula 必须真实改变 mechanism、typed 参数、resource shape、legal requirements、layout/schedule 或 analytic prior；
- construction、legality、selection、typed body 与 emission 保持单向 authority；
- 不要求全仓零 leaf；新语义允许新增最小 mechanism，已有机制可表达的组合不应继续复制完整 point implementation；
- measurement 只修正解析构造后的合法 residual；
- 证据回流用于修正知识及其适用范围，不建立新的治理框架。

## 8. 资产与结构问题的朴素判断

评估一个 Plan、attr、materializer、reader、verifier、provider 或 registry 时，核心只问：

- 它是否表达真实 compiler 语义？
- 是否有真实 producer 和 consumer？
- 是否和别处重复决定同一件事？
- 删除或改写后，语义、诊断、ABI 和测试由谁继续承担？
- 它是在帮助开发者理解算法，还是只在维护迁移协议？

答案清楚后直接行动：该留就留，该合并就合并，该重写就重写，该删就删。不需要先给每个对象打成熟度标签，也不需要建立一套 universal migration protocol。

代码修改的基本要求只是与风险相称：

- 改公式就验证决定和产物；
- 改 IR/pass 就验证 parse、verifier 和 lowering；
- 改 emitter 就验证 code shape、ABI 和 correctness；
- 做性能主张才走真实硬件和相应 evidence。

## 9. Spec 重构应写清什么

接下来重构 spec，不需要先创建 task、门、表或完整 census。重点是把内容重新放对位置。

### Canon

只承载论文 agent 批准的两柱、六律、贡献和稳定边界。下一轮需要恢复“可扩展 MLIR operator compiler/software stack；RISC-V 是旗舰实例”的系统类别，不能把当前验证域或 QIGen 差分写成架构上界。工程 agent 不直接定稿六律。

### Architecture

讲清完整 operator compiler：source/operator front door、dialect、plugin、typed owner、capability、family-local formula/construction、thin selection、typed body、pass、backend、EmitC、ABI 与 runtime 边界。必须回答新增 operator、format、capability、mechanism、formula、residual 和 backend 分别写在哪里；目标是让开发者理解系统，而不是把某轮迁移 protocol 写成永恒架构。

### Measurement

保留 correctness、真实板、runner 和可报告证据的必要约定，但不扩张成 A 线代码重构的审批系统。

### Evidence

说明哪些论文主张目前有什么代码、测试和硬件支撑。历史材料可引用，但不把整理工作伪装成科研贡献。

### Issues 与 tasks

Issue 记录真实缺口和反例；task 只在后续具体、多阶段工作需要查看时再建。它们都不定义科研主线。

## 10. 第一轮真正重构什么

第一轮不是挑一个特性做纵向样板，也不是再迁移少数公式后把其余留给“后续”。它必须横向建立完整、干净的 formula/construction layer：对当前所有承重解析决策做一次全域盘点，并把每一项迁入新结构、判定为非公式后归回正确 owner，或删除。任务结束时不能存在“新层覆盖五个公式、旧层继续承载其余公式”的双轨状态。

目标结构是：

```text
typed operator / format facts g
  + canonical capability c
  + bounded context ω
  → family-local analytic formula collection
  → candidate / mechanism parameters / resource requirements / prior
  → applicability and legality
  → optional bounded winner selection
  → selected typed body
  → plugin-local realization / emission
```

第一轮 task 必须同时做到：

1. **完整公式 census 后立即收敛**：枚举当前全部 production analytic/formula decisions、真实 caller、`g/c/ω` 输入、typed 输出和下游 consumer；census 是迁移清单，不是长期治理仪式。
2. **一次建立公共最小契约**：所有 formula 采用同一套轻量输入/结果语义和调用阶段；具体实现与知识继续 family-local，不建 global Formula IR、表达式 DSL 或深层 provider hierarchy。
3. **`g/c/ω` 真正解耦**：格式/operator facts、canonical capability 与 bounded context 独立建模；公式只声明和消费真实需要的字段，不用预烘焙 tuple、board 名或 format-specific winner 伪装解耦。
4. **完整公式集合可见**：建立维护者可直接查看的 formula catalog/index，列出每个公式的 owner、输入轴、输出、适用域和 production caller；catalog 由真实注册/代码结构产生或与其同源，不能成为第二份 authority。
5. **所有当前公式一次迁完**：盘点出的每个公式要么进入新层并由 production path 消费，要么明确归为 mechanism、boundary、selector/residual 或 backend lowering 并退出“公式”名义；不得留下兼容旧入口和未迁公式债务。
6. **输出真实承重**：formula 结果必须改变 candidate、mechanism composition、typed body 参数、resource/legality requirement、layout/schedule 或 analytic prior；reason、stamp、provenance 和 optional field 不能冒充公式结果。
7. **错误机制横向退出**：清退所有与公式层重叠的 late decision、planner replay、mirror state、selector/emitter 二次决策、只为 selected-stamp 生命周期服务的 materializer/reader，以及旧 campaign 的兼容双轨；同一决定只保留一个 owner。
8. **覆盖率一次闭合**：对完整公式集合建立行为覆盖，而非只统计文件/函数。每个公式至少覆盖适用、边界/拒绝、关键 `g/c/ω` 变化和 downstream code/IR effect；同时报告 production caller coverage 与未覆盖项，任务完成时未覆盖公式为零。
9. **安全边界不误删**：保留真正的 IR/ABI/ISA verifier、fallback、reference oracle 和 backend lowering；它们保护语义与实现，但不重算公式。
10. **文档与扩展入口同步完成**：architecture/spec 必须让维护者直接知道新增 operator/format、capability、mechanism、formula、residual 和 backend 分别写在哪里，并给出一个完整扩展示例。

flat repack/q5_1、Codebook/KQuant、IME 等不再是分期迁移批次，而是横向公式层完成后用于证明不同知识拓扑、operation 和 backend family 都已经接入同一干净结构的代表性检查点。

## 11. 收束

Weft 的准确形状是：

> **一个面向生态扩展的 MLIR operator compiler / execution-layer software stack。它以 typed owner 局部吸收 operator、format/layout、capability、mechanism 与 backend family 的变化，再由 `g/c/ω` 条件化的可执行专家知识构造专化 kernel。RISC-V 量化推理是当前旗舰实现和主要压力场，不是系统定义的上界。**

当前已经存在的是复杂 compiler stack、通用 plugin/capability/typed-body 基础、统一公式方向、多个 backend family、部署与实验资产；仍需完成的是让代表性 production path 的变化局部性和性能知识因果链真正清楚、模块化且可扩展。

重构 spec 与代码的任务不是发明新的主线，也不是把项目缩成 factorization 或验证框架，而是让这个 operator compiler 的生态接口、formula ownership、`g/c/ω` consumption、typed construction、backend realization 和证据边界与其原始目标一致。
