# Weft-RV 项目全景与 Spec 重构前方法基线

> **性质**：2026-07-21 的内部长期维护方法文档；已吸收 [ARS 完整审计与编辑裁决](../执行知识因式分解完整审计与收敛设计-2026-07-21/phase4_review/05-编辑裁决与收敛修订.md)，以及取得正式论文后的 [QIGen 全文再审](../执行知识因式分解完整审计与收敛设计-2026-07-21/phase5_reaudit/06-QIGen全文差分与最终再收敛.md)。
>
> **边界**：不替代同目录的 [原接手文档](./README.md)，不修改两柱、六律或论文贡献，不是 task，也不规定一套新的开发流程。

## 1. 这份文档解决什么

Weft-RV 目前同时有论文侧材料、项目侧统一公式、复杂 compiler 实现、历史 census、实验记录和多轮 agent 施工。接下来重构 spec，首先要恢复这些对象之间的正确关系，而不是重新定义项目。

本文只固定四件事：

1. 科研主线由用户和论文 agent 决定，工程 agent 不改律、不重排贡献；
2. 项目侧公式已经完成统一，早期论文公式不是当前完整版本；
3. Weft-RV 是完整的 execution-layer compiler，公式是其核心知识构造机制，但不是整个 compiler；
4. 历史文档、census、issue 和 task 都是辅助材料，不是限制代码演进的手续。

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

## 4. 公式在完整 compiler 中的位置

Weft-RV 是 high-level MLIR 之后、面向 ggml/llama.cpp 风格量化推理 kernel 的 execution-layer compiler。它不新增通用 tensor/tile IR，但具备完整的 compiler 层次：

```text
kernel-level input
  → source front door 与 extension discovery
  → typed construction / family dialect body
  → capability 与静态事实
  → 解析构造公式、schedule、legality 与薄选择
  → selected typed body / 跨 pass 表示
  → body realization 与 conversion
  → RVV / IME / Scalar 等 family backend
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

删除某个完整 point implementation 后，如果 downstream 还能按 format 名、leaf enum、完整 plan row、测试 fixture、winner row或 emitter switch 选回原 body，就不算重建。

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

QIGen 的实际生成边界也给出了更精确的剩余问题：其 group/microkernel 组合是真实构造，但 microkernel 内部仍用 2/3/4/5/6/8 位 `switch` 保存完整 unpack/shift/mask/dot body。Weft 不能据此声称 QIGen“没有 factorization”；只能研究能否把因式分解继续推进到完整 per-bit/per-format microkernel leaf 以下。

当前只保留一个待验证的窄假说：

> Weft-RV 能否在 high-level MLIR 之后的 RISC-V execution layer 中，把异质 GGML block-quant 的编码拓扑、scale/bias/layout 语义、能力、静态上下文和可复用 typed mechanisms 放进真实 compiler authority chain，由既定解析公式在显式合法域内构造多个 operation/backend 的 typed implementation；在删除完整 point authority 后仍保持正确、可部署且有竞争力，而 measurement 只修正稀疏的合法性能残差。

这是未来要用代码和实验成立的 novelty hypothesis，不是本文宣称已经完成的贡献。`M/F/B/R` 是解释语言，不自动构成新理论。

### 7.1 新六律候选的 ARS 审查结论

《高级ai思想》提出的“变化归位、知识因式分解、族先于点、公式构成、解析主导测量修残、证据回流”与两柱总体相容，可以作为后续严格评审的候选方向；本轮不直接把它们写入 canon。

定稿前只需做三类微调，不应整体换义：

- “族先于点”约束的是完整逐点 implementation 与逐点 performance winner；格式不可约语义、真正新增的最小 primitive 和安全 negative boundary 仍可逐点存在，不能一律叫 empirical residual。
- “公式构成”继续以原构造式为中心：formula 产生候选、机制组合、`θ`、资源要求和 prior；显式 boundary 对适用性与合法性作判定。二者不能在 provider/verifier/emitter 中重复承权。
- “解析主导、测量修残”要求 analytic-only 已能构造正确候选；measurement 只能改变合法 winner，不能携带 decode、scale、offset、body ID 或完整 plan。

“证据回流”是把失败归因到 mechanism、formula、capability、context、boundary、residual、compiler/HW wall 或 e2e wash，不是建立新的 verification framework。

在 reconstruction 尚未实现前，六律和论文 headline 都必须使用“目标是”“研究能否”之类目标态语气，不能写成系统已经具备 family reconstruction。

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

只承载论文 agent 批准的两柱、六律、贡献和稳定边界。工程 agent 不在这里提出新科研解释。本轮对《高级ai思想》中新六律只提出语义血缘和微调意见，不直接覆盖 canon。

### Architecture

讲清完整 compiler：dialect、plugin、capability、construction、核心公式、thin selection、typed body、pass、backend、EmitC、ABI 与 runtime 边界。目标是让开发者理解系统，而不是把某轮迁移 protocol 写成永恒架构。

### Measurement

保留 correctness、真实板、runner 和可报告证据的必要约定，但不扩张成 A 线代码重构的审批系统。

### Evidence

说明哪些论文主张目前有什么代码、测试和硬件支撑。历史材料可引用，但不把整理工作伪装成科研贡献。

### Issues 与 tasks

Issue 记录真实缺口和反例；task 只在后续具体、多阶段工作需要查看时再建。它们都不定义科研主线。

## 10. 当前真正要做什么

现在还不急着改代码或建立 task。首先用这份理解检查现有 spec：

1. 两柱、六律和贡献有没有被工程 agent 改写；
2. 核心构造公式是否仍然是主体，thin selection 是否喧宾夺主；
3. spec 是否把完整 compiler 缩成 formula/stamp pipeline；
4. 是否把后来 Codex campaign 的临时结构写成了稳定 contract；
5. 是否低估了已有 dialect、plugin、lowering、backend、ABI、测试和性能资产。

查清后只做微创更新：删除错误/历史状态，区分 current、substrate、target 和 evidence，恢复完整 compiler 全景，并补入 reconstruction 目标。遇到科研含义交论文 agent；普通架构和代码问题由实现本身裁决，不再制造额外流程。

第一条后续代码路线仍应集中在 flat repack，而不是大修整个 compiler。它的作用是让项目先学会真正删除完整 point authority，不是用 q5_1 单点承担论文 novelty：

1. 把 q4/q5 的格式事实拆成独立语义轴；
2. 抽出 low-nibble、optional high-bit plane、bias policy、affine scale/offset fold 等真实 mechanisms；
3. 用现有构造公式的工程展开同时生成 GEMV/GEMM typed body；
4. 移除 `lowerToRepackGemvQ51`、`lowerToRepackGemmQ51` 及等价 Q51 authority；
5. 先证明 analytic-only correctness，再评价 bounded residual；
6. 用重构后的正式硬件 campaign 比较旧 point baseline。

q5_1 因为旧实现已经被团队看过，而且 QIGen 已覆盖 5-bit 仿射 group、GEVM/GEPM 与 group composition，应称为 **首个 point-authority-erasure 实现里程碑与本地结构重建证人**。它能证明“删除完整 leaf 后仍可重建”，但不能单独证明方法超越 QIGen、整个异质 domain 已被因式分解或 unseen generalization。

q5_1 完成后，确认性证据必须跨出 flat affine q4/q5：至少覆盖一种不同的真实语义拓扑，例如 KQuant 的量化 scale/min superblock、codebook/grid 或 ternary/q1，并增加语义不同的 operation 或 backend 因果链。只有论文明确主张 unseen/generalization 时，才额外需要机制/公式冻结后的 prospective witness；它不是普通重构的开工手续。

## 11. 收束

Weft-RV 的准确形状是：

> 一个完整的 capability-driven MLIR execution-layer compiler；其目标是以解析构造公式把独立格式/能力/上下文知识与可复用 mechanisms 变成候选、typed 参数、合法实现和解析 prior，外层 measurement 只对有限残差作薄修正。dialect、plugin、pass、backend、ABI、runtime、测试与硬件 evidence 共同让这套知识真正可执行和可验证。

当前已经存在的是复杂 compiler、统一公式和局部 construction substrate；尚未存在的是无完整 point authority 的 family reconstruction 证据。重构 spec 的任务是把这两者准确分开，而不是发明新的主线、扩大 selector 的地位，也不是用 census、task 或迁移纪律把后续代码修改限制死。
