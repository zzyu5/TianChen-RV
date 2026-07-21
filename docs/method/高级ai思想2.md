你这个疑惑是对的，而且它指出了 ARS 这轮审计里最重要的一次**范围倒置**：

> **ARS 为了寻找能与 QIGen 拉开距离的最窄证据，把“一个最危险近邻下的差分实验”逐渐写成了整个项目的身份。**

这不是你原始项目的定位。

ARS 的阶段四其实仍明确保留了“完整 execution-layer compiler”的定位，强调 capability、typed mechanisms、body construction、backend、ABI、runtime 都属于 compiler 本体，不是一个 formula-to-emitter 玩具系统。
但取得 QIGen 全文后，06 又把剩余问题收缩成“必须深入到 QIGen microkernel leaf 以下，分解 GGML 语义拓扑”，甚至说 Weft“只剩一个更窄的系统假说”。

**这一步收得过头了。**

QIGen 可以决定你们不能声称什么，但不应该反过来决定 Weft 是什么。

---

# 一、Weft 到底是什么

Weft 应当被稳定定义为：

> **一个 capability-driven、可扩展的 MLIR execution-layer 算子编译器。它面向碎片化执行硬件，把 operator/format facts、硬件 capability、执行机制和性能知识组织成 typed construction，并生成真实可部署的高性能 kernel。RISC-V 是主要的、最具挑战性的系统实例。**

也就是：

```text
MLIR operator
→ operator / format / layout facts
→ target capability
→ execution-family construction
→ legal typed kernel body
→ backend realization
→ ABI / runtime / deployment
```

你们不是：

* 量化格式知识管理系统；
* q5_1 重建系统；
* Knowledge Factorization 语言；
* point-authority-erasure 验证框架；
* QIGen 的“更细粒度版本”。

ARS 自己最初的 Topic Area 本来也写的是“面向 ggml/llama.cpp 风格 RISC-V 量化推理 kernel 的 capability-driven typed execution-layer compiler”。

所以需要纠正的不是你的原始定位，而是后来审计把**论文的一项构造证据**误当成了**论文的系统类别**。

---

# 二、不过“一个可扩展 MLIR compiler + 一个 RISC-V 例子”还需要再锋利一点

你的直觉是正确的，但论文不能只停留在这句话，因为：

* MLIR 可扩展基础设施本身已有；
* plugin、dialect、typed op 本身已有；
* RISC-V backend 本身不是 novelty；
* 自动生成高性能 kernel 也已有大量先例；
* QIGen 已经直接生成量化 CPU GEVM/GEPM kernel。

真正有研究价值的命题是：

> **能否在一个碎片化 execution layer 中，同时获得可扩展性和专家级性能？**

这两个目标存在天然张力：

```text
抽象得越统一
→ 越容易扩展
→ 越可能丢失底层性能细节

手写得越专用
→ 越容易达到专家性能
→ 越容易退化成 format × board × backend 的乘积实现
```

Weft 的核心不是简单地“同时有两个优点”，而是提出一个具体的解决方式：

> **在 execution layer 中，将变化轴分开表示，在编译时再利用 capability-conditioned executable knowledge 将它们重新专化成高性能 kernel。**

一句最准确的总结是：

> **先按变化轴解耦，以获得可扩展性；再按能力和场景重组，以恢复专家级性能。**

这才是两柱真正共用的核心。

---

# 三、两根柱完全不需要改变

## 柱一：可扩展性来自能力驱动、类型化的 execution-layer 模板

柱一解决的是：

> 碎片化的 operator、format、ISA 版本、向量宽度、自定义扩展、backend 和板级能力，如何不再造成整个代码库的横向修改。

RISC-V 是非常好的压力场景，因为它同时具备：

* RVV 0.7 与 RVV 1.0 等版本差异；
* 不同 VLEN；
* fractional LMUL、寄存器预算等能力差异；
* 标准向量与厂商/研究型扩展；
* RVV、IME、Scalar 等不同实现 family；
* 大量不规则 block-quant 格式；
* decode、prefill、dequant、vec-dot、GEMV、GEMM 等不同 operation/regime。

因此柱一不是“我们用了 MLIR plugin”。

而是：

> **当 operator/format、capability 和 backend 三个变化轴独立扩展时，变化是否主要落在自己的 typed owner，而不是重新形成乘积式实现。**

---

## 柱二：高性能来自可执行知识

柱二解决的是：

> 一个可扩展 compiler 为什么不会只生成正确但平庸的代码？

答案不是“因为我们有公式”这么简单，而是：

> **专家对数据格式、ISA、资源、循环组织和部署路径的知识被编译器真实执行，用于构造 kernel，而不是留在手写 leaf、注释或无限搜索中。**

柱二必须解释两件事：

1. **高性能知识是什么；**
2. **它如何改变最终 kernel。**

---

# 四、生成出来的 kernel 为什么高性能

高性能主要来自四层。

## 1. 语义级专化

编译器知道真实 operator 和格式语义，例如：

* packed low-bit carrier；
* high-bit plane；
* scale、zero、min；
* codebook、grid、ternary；
* block 与 sub-block 拓扑；
* decode 与 prefill；
* GEMV 与 GEMM；
* packing/repack layout。

因此它可以直接生成：

* fused unpack + dot；
* fused decode + scale/fold；
* compressed-domain execution；
* 不产生不必要的中间 dequantized tensor；
* 与实际 GGML ABI 一致的 load/store。

这部分由 (g)、(\omega) 和 mechanisms 承担。

---

## 2. 硬件能力专化

编译器知道：

* VLEN；
* ISA/version；
* fractional LMUL；
* SEW 支持；
* vector-register budget；
* gather、dot、widening 能力；
* IME matrix/vmadot 能力；
* 可用 hart 或 tile shape。

因此同一个 operator/format 在不同 (c) 下可以得到不同：

* LMUL；
* strip width；
* accumulator structure；
* gather anchor；
* tile；
* unroll；
* mechanism family；
* fallback。

这部分由 (c) 与解析公式承担。

---

## 3. operation/regime 专化

decode 和 prefill 的最佳实现通常不同：

* GEMV 更偏内存带宽；
* GEMM/prefill 更关心数据复用和计算密度；
* 不同 layout、shape bucket 和 packing 状态需要不同 loop order；
* 小矩阵、tail 和主循环的资源组织不同。

这就是保留 (\omega) 的意义。

---

## 4. 有限经验修正

即使解析知识已经构造了正确、合法而有竞争力的候选，仍可能有：

* 微架构吞吐差异；
* compiler scheduling 差异；
* 指令组合的非显然效果；
* 两个合法 LMUL 或 loop order 很难稳定闭式排序。

measurement 在这里发挥作用：

```text
不创造 kernel
不创造 mechanism
不改变语义
不改变合法性
只在有限合法候选中修正 winner
```

所以高性能来源可以概括成：

[
\boxed{
\text{高性能}
==========

\text{语义专化}
+
\text{能力专化}
+
\text{资源与场景专化}
+
\text{有限经验修正}
}
]

其中：

* mechanisms 承载语义与算法知识；
* (f^A(g,c,\omega)) 承载能力、资源和场景推导；
* residual measurement 只补静态知识无法可靠排序的部分。

这才是柱二真正需要讲清楚的因果链。

---

# 五、核心公式不用改变，但要把它放回正确位置

保留你们现有公式：

[
\theta_{i,v}=f^A_{i,v}(g,c,\omega)
]

[
K_{v,\omega}(g,c)
=================

Emit_v(c)\circ
\bigoplus_i
\left{
m_{i,v}(g,\omega)
;\text{with};
\theta_{i,v}
\right}
]

但要明确：

> **这不是整个 Weft compiler 的定义；它是整个 compiler 中“execution knowledge 如何构造一个 kernel candidate”的核心。**

完整系统仍是：

```text
MLIR operator / source
→ typed extension proposal
→ family / route selection
→ g / c / ω
→ 主公式构造 kernel candidates
→ legality
→ thin selection
→ selected typed body
→ lowering / emission
→ ABI / runtime / deployment
```

ARS 的源码审计其实也确认当前系统具有完整 dialect、plugin、operation graph、RVV/IME/Scalar、lowering、ABI/runtime 和硬件测试链路。

所以不要再让主公式吞掉整个 compiler，也不要为了 formula framing 把其他层写成“supporting ceremony”。

---

# 六、Knowledge Factorization 应该放在哪里

Knowledge Factorization 可以保留，而且有价值。

但它的地位应当是：

> **两柱之间的设计桥梁，而不是系统身份，也不是第三根柱。**

它表示：

* 为了柱一，把知识沿变化轴分开；
* 为了柱二，在编译时把这些知识重新组合成专化 kernel。

例如：

```text
format/operator facts
+ capability facts
+ reusable mechanisms
+ analytic decisions
+ applicability boundary
+ optional empirical residual
```

这解释了系统如何工作。

但论文不需要变成：

> “我们提出了一种新的 Execution Knowledge Factorization 理论。”

也不需要证明所有逐点 leaf 都必须消失。

更合理的目标是：

> **完整 point implementation 不再是默认扩展单位；已有机制能够表达的组合应由 compiler 构造，真正新的语义仍然可以增加新的最小 mechanism。**

因此：

* q5_1 reconstruction 是一个很好的重构和消融实验；
* Codebook topology 是一个不同机制组合的证据；
* IME 是第二硬件 family 的证据；
* 它们都不是 Weft 的定义。

---

# 七、QIGen 应该怎样影响你们，而不应该怎样影响

QIGen 的论文明确描述的是：

```text
nonuniformly quantized LLM
+ target CPU characteristics
→ 提取 group size / bitwidth
→ 生成 GEVM/GEPM microkernels
→ 搜索 blocking/unroll
→ 组合成 layer-specific kernel
→ JIT / PyTorch model
```



它的生成器内部确实使用按 bitwidth 的完整 `switch(bits)` 生成 unpack/dot body，然后按 `(M,K,g,b)` 查找并组合 microkernel。

所以 QIGen 击穿的是这些主张：

* 首个量化 CPU kernel generator；
* 首次由量化描述和 CPU 特征生成 kernel；
* 首次组合非均匀量化 group；
* 首次解析模型加有限搜索；
* 首次跨 ISA retarget。

但 QIGen 的论文技术对象仍然是特定的非均匀量化 GEVM/GEPM generator；它自己也把实验重点放在 kernel/layer 时间，并排除了完整 inference pipeline 的正交实现细节。

因此 QIGen 应当是：

> **柱二中“量化 kernel generation”部分的强近邻。**

它不应当被提升为：

> **决定整个 Weft 必须变成 microkernel-leaf-below factorization 系统的分类器。**

“QIGen 有一个 bitwidth switch，所以我们必须把 leaf 再拆一层”可以是一个有价值的差分实验，但不应成为整篇论文唯一的生存理由。

真正更大的差异是：

| QIGen                          | Weft                                                              |
| ------------------------------ | ----------------------------------------------------------------- |
| 专门的非均匀量化线性 kernel generator    | MLIR execution-layer 算子 compiler                                  |
| 主要是 GEVM/GEPM                  | dequant、vec-dot、GEMV、GEMM、contraction 等                           |
| 量化模型 + CPU → per-model kernels | MLIR operator + facts/capability/context → backend implementation |
| x86/ARM SIMD generator         | 碎片化 RISC-V，含 RVV 版本、VLEN 和 custom family                          |
| 主要回答量化 kernel 如何生成             | 同时回答硬件/格式/后端变化如何扩展，以及为何仍有专家性能                                     |
| kernel/layer 评价为主              | compiler extension、kernel、真实 ggml、强对手与 e2e                        |

所以 related work 要正面承认重叠，但 system identity 不应被它劫持。

---

# 八、我建议定稿的六律

六律继续是两柱的操作规律，不是六项 novelty。

## 律一：变化归位律

新 operator、格式、能力、机制、解析知识、经验知识和 backend 各有主要 typed owner。

这条回答：

> 新东西到底写在哪里？

---

## 律二：知识因式分解律

高性能知识应尽量拆成：

```text
typed facts
+ reusable mechanisms
+ analytic decisions
+ applicability boundary
+ optional empirical residual
```

而不是默认保存为完整 `format × board × regime` kernel。

真正不可复用的新语义可以增加新 mechanism，不要求强行“零 leaf”。

---

## 律三：和积扩展律

人写知识主要沿独立变化轴增长；compiler 负责把这些轴组合成实际实现。

理想目标是：

[
\text{手写知识规模接近各轴之和}
]

而：

[
\text{可生成覆盖接近各轴组合}
]

它不是需要证明的数学渐进定理，而应由：

* 新格式修改范围；
* 新 capability 修改范围；
* 新 backend 修改范围；
* mechanism/rule fan-out；
* 旧代码是否需要横向修改；

来实证。

---

## 律四：公式承重律

一个公式只有真实改变以下至少一项，才算柱二知识：

* mechanism choice/composition；
* typed plan/body 参数；
* resource shape；
* legal candidate requirements；
* loop/layout/schedule；
* analytic prior。

它的输出必须被真实 construction 或 lowering 消费。

仅包装常数、生成 reason 或查 winner 不算公式贡献。

---

## 律五：单向专化律

固定方向：

```text
facts / capability / context
→ analytic construction
→ applicability / legality
→ bounded selection
→ selected typed body
→ realization / emission
```

禁止：

* emitter 重新选择算法；
* measurement 创造 compute；
* selector 绕过 legality；
* metadata 重建 computation。

这条是两柱的共同防火墙。

---

## 律六：证据回流律

正确性、性能、compiler wall、hardware wall 和 e2e wash 应归因回：

* facts 是否充分；
* mechanism 是否缺失；
* formula 是否错误；
* capability 是否不足；
* boundary 是否需要收缩；
* residual 是否必要。

失败不应默认生成新的 format/board 特判。

---

## 六律与两柱

| 六律     | 柱一 | 柱二 |
| ------ | -: | -: |
| 变化归位   | 核心 | 支撑 |
| 知识因式分解 | 核心 | 核心 |
| 和积扩展   | 核心 | 支撑 |
| 公式承重   | 支撑 | 核心 |
| 单向专化   | 核心 | 核心 |
| 证据回流   | 支撑 | 核心 |

---

# 九、真正可辩护的 novelty 应该是什么

不是：

* extensible MLIR 本身；
* kernel generator 本身；
* formula 本身；
* Knowledge Factorization 这个词本身；
* q5_1 reconstruction 本身。

而是：

> **Weft 提出并实现一个 capability-driven typed execution layer，用来解决 operator/format 与碎片化硬件能力、实现 family 之间的乘积专化问题；该层在保持变化局部性的同时，执行专家知识构造 capability- and context-specialized kernel，从而在真实 RISC-V 软件栈中兼顾可扩展性和专家级性能。**

英文可以是：

> **Weft is an extensible, capability-driven MLIR execution-layer compiler that reconciles modular support for fragmented operator and hardware variants with expert-level kernel performance through executable, context-conditioned specialization knowledge.**

三个论文贡献可以包装为：

### C1：Execution-layer compiler architecture

一个 typed、capability-driven 的 MLIR execution layer，把：

* operator/format facts；
* capability；
* mechanism；
* backend realization；

分成可独立扩展的 owner。

### C2：Executable high-performance construction

以主公式为核心，让专家知识真实构造：

* mechanism；
* 参数；
  -资源计划；
* legal candidate；
* typed body；

measurement 只修正有限残差。

### C3：RISC-V realization and evaluation

在真实 RISC-V 碎片化环境中展示：

* RVV 版本与 VLEN 差异；
* RVV、IME、Scalar family；
* 多种 quantized operators/formats；
* ggml consumer；
* 强对手；
* kernel 与 e2e；
* 扩展实验与性能因果消融。

两柱仍然只有两根；C3 是证据，不是第三柱。

---

# 十、接下来代码应该怎么做

你当前“系统完善 / 代码重构”的自我定位是对的。

不需要重新立项，也不需要把整个仓库改成 factorization framework。

## 第一优先级：把 compiler 骨架讲清楚

建立一个面向维护者的明确地图：

| 新内容              | 应进入哪里                                    |
| ---------------- | ---------------------------------------- |
| 新 MLIR operator  | operator/dialect/front door              |
| 新格式或布局           | typed family facts                       |
| 新硬件能力            | canonical capability                     |
| 新的可复用执行模式        | family mechanism / typed body op         |
| 新公式              | family-local construction/decision owner |
| 新合法性或 fallback   | family boundary                          |
| 新经验 winner       | residual sidecar                         |
| 新 backend/family | plugin-local realization                 |
| 最终指令/EmitC       | emitter/lowerer                          |

这是当前最值得做的系统完善。

尤其要解决：

> “新增一个公式不知道写到哪里。”

答案应该稳定为：

```text
family-local construction / decision layer
```

它：

1. 读取 typed (g/c/\omega)；
2. 输出 plan/body 中真正承重的字段或 candidate；
3. 在 emitter 之前运行；
4. 被 typed construction 真实消费。

不需要全局 Formula IR。

---

## 第二优先级：清理代表性 late decisions

不要求全仓一次完成。

挑选 2–3 条最能说明两柱的路径：

### RVV flat repack

用 q4/q5 family 整理共享 facts、mechanisms 和 GEMV/GEMM construction。q5_1 可以作为“新增组合写在哪里”的示范，但不是整篇论文的命门。

### Codebook 或 KQuant

用不同的 scale/decode topology，证明骨架不仅适用于 flat affine q4/q5。

### IME

把 emitter 内的 batched/wide deployment 决策前移到 typed body，证明新 hardware family 可以局部接入，而不是改 RVV core。

这三项分别服务：

* 格式扩展；
* 机制拓扑；
* 硬件 family 扩展。

---

## 第三优先级：不要为了审计而大修

暂时不要建立：

* CM0–CM4 日常治理系统；
* universal point-authority verifier；
* global Formula service；
* 深层 provider hierarchy；
* 通用 mechanism AST；
* 全仓零 point leaf 战役；
* prospective holdout 作为普通开发流程。

ARS 可以继续作为内部审计材料，但不应该变成开发架构。

---

# 十一、接下来实验怎么排

现有实验大部分保留。

真正需要补的是两柱各自的**直接证据**。

## 柱一实验：到底是否更可扩展

设计三种扩展事件：

### A. 新格式/组合

增加一个格式变体或已有 mechanism 的新组合，记录：

* 新增了哪些 facts；
* 是否新增 mechanism；
* 是否修改旧格式；
* 是否修改 backend；
* 是否新增完整 kernel body。

### B. 新 capability

加入一个新的 capability 点，例如：

* 新 VLEN；
* 不同 fractional LMUL；
* 不同 RVV version；
* 不同 register budget。

记录是否只修改 capability/decision，而不逐格式修改。

### C. 新 family/backend

用 IME 展示：

* core 是否无需 family-name branch；
* family 是否有自己的 mechanisms 和 realization；
* 相同 operator facts 是否能进入另一条 backend chain。

柱一的主要指标不是单纯 LOC，而是：

* 修改 owner 数；
* 修改旧实现数；
* 新增 cross-product branch 数；
* mechanism/rule fan-out；
* 同一 compiler contract 是否跨 family 成立。

---

## 柱二实验：为什么它高性能

至少做四臂消融：

```text
generic / fixed-default implementation
shared mechanisms + fixed parameters
shared mechanisms + analytic formula
shared mechanisms + analytic formula + residual
```

然后回答：

1. 哪个 mechanism 带来什么性能；
2. 哪个 capability 变化导致什么 plan flip；
3. 哪个 formula 避免了错误资源选择；
4. measurement 到底改变了多少 winner；
5. analytic-only 已经达到多少性能；
6. residual 又补了多少；
7. 与 ggml/手写强对手相比如何；
8. kernel 收益是否传到 e2e。

这样柱二不是“最后跑得快”，而是：

> **我们能解释为什么快，以及性能知识具体经过哪条 compiler 因果链进入 kernel。**

---

# 十二、论文结构建议

## 1. 问题

碎片化 operator/format 与碎片化 RISC-V hardware 造成乘积式手写 kernel。

## 2. 核心矛盾

可扩展抽象容易损失性能；专家特化容易损失可扩展性。

## 3. Weft architecture

capability-driven typed MLIR execution layer。

## 4. 柱一

typed owners、plugin/family、局部扩展、和积覆盖。

## 5. 柱二

主公式、mechanisms、capability/context specialization、legality、thin residual。

## 6. RISC-V realization

RVV、IME、Scalar；量化 formats；dequant、dot、GEMV/GEMM。

## 7. Evaluation

* 扩展性；
* 性能因果；
* 强对手；
* 部署/e2e；
* 负结果。

## 8. Related work

QIGen、SPIRAL、Exo、TVM、Ladder 等分别覆盖部分能力；Weft 的差额是碎片化 execution-layer 中 extensibility–performance tension 的系统解决。

---

# 最终裁决

你的原始方向没有错：

> **Weft 就应该是一个可扩展的 MLIR 算子 compiler，以复杂、碎片化的 RISC-V execution layer 作为具体实例，并证明它不但容易扩展，而且生成的 kernel 具有专家级性能。**

需要强化的不是把它改造成“量化知识因式分解系统”，而是把这句话变成一个清楚的科研命题：

> **现有系统通常在可扩展性和高性能之间二选一。Weft 通过 capability-driven typed execution layer，将变化轴解耦，再通过可执行专家知识重新专化，从而同时获得局部扩展和高性能 kernel。**

Knowledge Factorization 保留为内部设计方法：

> **分开，是为了可扩展；重组，是为了高性能。**

q5_1、Codebook、IME、residual density 和 point-leaf deletion 都是证明这个命题的实验工具，不是 Weft 本身。

下面这段可以直接成为下一轮 agent prompt 的总纲：

```text
Weft is an extensible MLIR execution-layer operator compiler, not a
quantization-factorization system or a point-authority verification
framework.

Its research problem is the extensibility–performance tension caused by
the cross product of operator/format variants, fragmented hardware
capabilities, and backend families.

Pillar 1 remains:
extensibility comes from capability-driven typed execution-layer
templates and local ownership of independent change axes.

Pillar 2 remains:
high performance comes from executable expert knowledge that constructs
mechanisms, resource plans, legal candidates, and typed kernel bodies;
measurement only corrects a bounded residual among legal candidates.

Knowledge Factorization is the bridge between the pillars:
knowledge is separated along change axes for extensibility and recomposed
at compile time for performance. It is not a third pillar and does not
redefine the system.

QIGen invalidates broad claims such as “first quantized CPU kernel
generator,” but it does not redefine Weft’s category. QIGen is a strong
neighbor for the quantized GEVM/GEPM generation slice; Weft remains a
multi-operator, multi-family MLIR execution-layer compiler evaluated on
fragmented RISC-V hardware.

q5_1 reconstruction, Codebook mechanism composition, IME authority
cleanup, and residual-density experiments are evidence for the two
pillars, not the identity of the project.

The current phase remains system completion and code refactoring:
clarify typed owners, establish a clear family-local location for new
formulas, move representative code-affecting decisions before emission,
preserve the existing compiler/ABI/runtime/benchmark assets, and add
targeted extensibility and performance-causality experiments.
```
