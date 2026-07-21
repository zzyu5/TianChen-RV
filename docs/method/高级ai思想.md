不改两根柱子，而且**不需要把系统推翻重做**。

这次调整应当是：

> **论文核心发生一次明显增强；代码层主要是系统完善、知识归位和模块重构；已经完成的大部分正确性、性能与部署实验继续有效。**

你不喜欢 `Reason Completeness`、静态 verifier、源码依赖检查这些东西，我认可全部拿掉。它们既不是当前最需要解决的问题，也容易把项目带向形式化验证、语言安全或代码审计，偏离你们真正想做的事情。

真正应该保留的是早期版本抓到、后来被 Codex 弄丢的那个核心：

> **高性能 kernel 不是若干无关实现点，而是可以被分解和重建的 execution knowledge family。**

早期版本已经把“源语言是 kernel family”和机制、公式、能力条件化分解放在中心，但又附加了“全空间闭式”“点不入源机检”等过度承重内容。 后来又把 novelty 压到“闭式全空间”和“语法级机检”两条腿上，导致研究逐渐被自身的形式承诺绑架。

正确路线不是继续这个 verifier 方向，也不是退回普通 candidate selector，而是：

# 两柱不变，一核重立：Knowledge Factorization

---

## 一、最直接的几个裁决

### 1. 两根柱完全不改

仍然是：

### 柱一：可扩展性来自能力驱动、类型化的 execution-layer 模板

新格式、新能力、新机制、新后端和新知识进入各自 typed owner，避免重新退化成逐格式、逐板、逐后端手写。

### 柱二：高性能来自可执行知识

专家知识必须真实参与：

* mechanism 构造；
* plan 参数推导；
* capability/context 专化；
* 候选生成与选择；
* 适用边界；
* 最终真实性能。

一个字都不需要改变。

---

### 2. 新增的不是第三根柱，而是两柱共同的核心

新的核心叫：

> **Execution Knowledge Factorization**

中文可以稳定写成：

> **执行知识因式分解**

它回答的是：

> 专家手写 kernel 中混在一起的格式知识、执行机制、硬件适配、参数推导、经验选择和负边界，能否被拆成少量可独立复用、可组合、可重新执行的知识成分，从而重建一整个 kernel family，而不是继续保存大量逐点实现？

它同时支撑两柱：

* 对柱一，factorization 让变化被局部吸收；
* 对柱二，factorization 让专家知识真正参与 kernel 构造，而不是只存在于文档、注释或测量 winner 中。

所以新的论文结构不是“三柱”，而是：

```text
两柱
  ↑
Knowledge Factorization
  ↑
六条操作规律
```

---

### 3. 系统代码不需要大改

研究叙事变化很大，代码变化只应是**小到中等规模重构**。

不需要新增：

* Formula IR；
* 通用表达式 AST；
* verifier；
* dependency type system；
* runtime adaptive framework；
* 新的 correctness proof；
* 新 autotuner；
* 巨型全局 Formula service。

现有这些都继续保留：

* typed format/mechanism facts；
* canonical capability；
* bounded context (\omega)；
* plugin-local provider；
* typed plan/body；
* emitter/lowerer；
* legality；
* measurement sidecar；
* fallback；
* ggml integration；
* 当前 benchmark 和 e2e 路径。

需要做的是把这些已有部分按照 Knowledge Factorization **重新归位和命名**，并消除一些让用户不知道“新公式该写在哪里”的混乱。

---

# 二、Knowledge Factorization 到底是什么

一个 execution family 的知识不再被看成若干散乱代码，而被分成四类：

[
\boxed{
\mathcal E_f
============

\left\langle
\mathcal M_f,,
\mathcal F_f,,
\mathcal B_f,,
\mathcal R_f
\right\rangle
}
]

其中：

| 成分             | 含义                                 |
| -------------- | ---------------------------------- |
| (\mathcal M_f) | 可复用的 typed mechanisms              |
| (\mathcal F_f) | 根据 (g,c,\omega) 构造机制参数与 plan 的解析知识 |
| (\mathcal B_f) | 适用域、合法性、资源边界、fallback 和负结果         |
| (\mathcal R_f) | 解析知识无法可靠排序时留下的有限经验残差               |

这里的关键不是四个模块，而是四种**知识角色**。

物理代码完全可以只分成五六个文件，不需要做大架构。

---

## 三、主公式应该如何写

保留 09 中更完整的变量：

* (g)：格式、编码、数据布局和机制事实；
* (c)：canonical capability；
* (\omega)：有限静态使用上下文，包括 op、regime、layout、必要 shape bucket；
* (v)：一个具名的实现 family/candidate。

主公式不再只被解释成“候选构造公式”，而是：

> **因式分解后的知识如何重建一个 typed execution plan。**

[
\boxed{
P_v(g,c,\omega)
===============

\operatorname{Compose}*v
\left(
\left{
m*{i,v}
\left(
g,\omega;
\theta_{i,v}
\right)
\right}*i
\right),
\qquad
\theta*{i,v}
============

f^A_{i,v}(g,c,\omega)
}
]

其中：

* (m_{i,v}) 是共享 typed mechanism；
* (f^A_{i,v}) 是可执行知识函数；
* (\theta_{i,v}) 是 mechanism 或 plan 真正消费的 typed decision；
* `Compose` 是有序的 typed plan 构造，不必声称满足交换律；
* (P_v) 是可以被 emitter 消费的完整 typed plan。

合法候选：

[
\mathcal V(g,c,\omega)
======================

\left{
P_v(g,c,\omega)
\mid
L_v(g,c,\omega)=1
\right}
]

最终：

[
\boxed{
K^*(g,c,\omega)
===============

Emit
\left(
Select_{\mathcal R}
\left(
\mathcal V(g,c,\omega)
\right)
\right)
}
]

这里：

* Knowledge Factorization 负责产生 (\mathcal V)；
* legality 负责过滤；
* (\mathcal R) 只修正解析知识剩余的排序残差；
* emitter 机械实现最终 plan。

这样保留了 09 版本中真实可信的 legality 和 bounded selection，但不再让它们抢走论文核心。

---

# 四、这和 09 的本质区别

09 的逻辑是：

> 我们可以规范地构造有限候选，然后规范地选一个。

新的逻辑是：

> **我们将专家 execution knowledge 因式分解，使少量共享机制与知识函数能够重建跨格式、跨能力、跨场景的 kernel family；候选选择只是因式分解之后的残差处理。**

两者的差异是：

| 09                  | 新版本                      |
| ------------------- | ------------------------ |
| 候选是基本对象             | 知识因式分解是基本对象              |
| provider 构造 plan    | 知识函数重建实现 family          |
| measurement 是重要系统组成 | measurement 是剩余残差        |
| 重点是治理               | 重点是知识复用与组合               |
| 证明系统整洁              | 证明实现空间具有共享结构             |
| 容易接近 autotuner      | 更接近新的知识化 kernel compiler |

这就是 novelty 被重新抬起来的地方。

---

# 五、六律应该如何修改

不再保留“语法级机检”“无邻居 verifier”“Reason Completeness”等内容。

六律改成真正服务 Knowledge Factorization 的六条。

## 律一：变化归位律

格式、能力、机制、上下文、经验和后端变化分别进入明确的 typed owner。

目的：

* 保持柱一；
* 用户能够理解系统结构；
* 新增内容时知道该放到哪里。

---

## 律二：知识因式分解律

一个高性能实现不得只被保存为完整代码点，而应尽可能分解为：

```text
shared mechanism
+ analytic formula
+ applicability boundary
+ optional empirical residual
```

这里的“不得”是设计方向，不要求做静态 verifier。

它的意义是：

* 机制可以跨格式复用；
* 公式可以跨能力重新执行；
* 边界可以诚实描述；
* 测量不必重新编码整个乘积空间。

---

## 律三：族先于点律

系统首先描述 kernel family，再在具体 (g,c,\omega) 上实例化。

逐点知识允许存在，但只能被视为 residual，而不是默认知识形态。

理想知识规模接近：

[
O\left(
|\mathcal M|
+
|\mathcal F|
+
|\mathcal B|
+
|\mathcal R|
+
|G|+|C|+|\Omega|
\right)
]

而不是所有实现逻辑都接近：

[
O(|G|\times|C|\times|\Omega|)
]

这里不需要声称已经证明渐进复杂度，只需要通过实验展示：

* 一条机制覆盖多少组合；
* 一条公式影响多少组合；
* 新增格式或能力时旧代码修改多少；
* residual 是否随组合数量快速膨胀。

---

## 律四：公式构成律

一个公式只有真实改变 typed plan，才算执行知识。

合格公式必须：

* 消费真实的 (g,c,\omega) 字段；
* 输出 typed mechanism choice 或 plan 参数；
* 进入 typed body；
* 被 emitter/lowerer 实际消费；
* 有明确适用域；
* 能通过正确性或性能实验体现作用。

不算公式的东西：

* 把现有常数换成 provider；
* 隐藏的 winner lookup；
* 只生成 reason 字符串；
* 不影响 computation 的 optional field；
* 为了统计“公式数量”而制造的伪参数。

这条可以保留 09 中最好的部分，但它现在服务于 factorization，而不是单独成为 novelty。

---

## 律五：解析主导、测量修残律

解析知识负责：

* mechanism；
* 候选结构；
* typed plan；
* 合法域；
* 资源边界；
* cold-start prior。

测量只负责：

* 在已构造、已合法的有限候选之间修正排序。

测量不得成为默认实现知识，也不应被包装成第二根柱。

论文需要实测：

* 有多少决策由解析知识直接确定；
* 有多少决策被 measurement flip；
* 每增加格式或能力，measurement row 是否快速增长；
* measured residual 是否真正小于整个产品空间。

---

## 律六：证据回流律

正确性、性能和负结果必须回到知识因式分解中的一个位置：

* 机制不够；
* 公式适用域不对；
* capability 不充分；
* context 缺字段；
* measurement residual；
* compiler/HW boundary；
* kernel win 但 e2e wash。

这不是 verification framework。

它只是要求系统不要每遇到一个失败就再写一处特判，而是判断：

> 失败说明哪一块知识分解还不正确？

---

# 六、两柱与新六律的关系

| 规律           | 柱一：可扩展性 | 柱二：可执行知识 |
| ------------ | ------: | -------: |
| 律一：变化归位      |      核心 |       支撑 |
| 律二：知识因式分解    |      核心 |       核心 |
| 律三：族先于点      |      核心 |       支撑 |
| 律四：公式构成      |      支撑 |       核心 |
| 律五：解析主导、测量修残 |      支撑 |       核心 |
| 律六：证据回流      |      支撑 |       核心 |

Knowledge Factorization 是两柱的共同接头，而不是第三根柱。

---

# 七、代码究竟需要怎么改

## 基本不动的部分

现有这些代码和成果继续使用：

* typed facts；
* capability snapshot；
* dequant providers；
* typed mechanisms；
* typed body；
* plugin lifecycle；
* measured LMUL/SP4/loop-order；
* winner hit/miss/prior；
* clean-room 与 Scalar no-V；
* ggml 部署；
* 强对手；
* e2e；
* negative/boundary records。

这些刚好可以重新映射为：

| 已有资产                         | Knowledge Factorization 中的位置 |
| ---------------------------- | ---------------------------- |
| typed mechanisms/body        | (\mathcal M)                 |
| analytic providers           | (\mathcal F)                 |
| capability/context decisions | (\mathcal F) 的输入             |
| fallback/legality/wall       | (\mathcal B)                 |
| LMUL/SP4/loop-order winner   | (\mathcal R)                 |
| emitter                      | plan realization             |
| ggml/强对手/e2e                 | factorized knowledge 的外部证据   |

所以不是推翻已有系统，而是重新讲清楚它。

---

## 应该进行的重构

建议每个 family 至少形成下面的逻辑结构：

```text
family/
  facts
  mechanisms
  formulas
  boundaries
  residual_selection
  plans
  emitter
```

不一定必须七个物理目录，也可以合并。

但概念上必须能回答：

### 新增格式事实写在哪里？

写入 `facts`。

### 新增硬件能力写在哪里？

写入 canonical capability。

### 新增执行机制写在哪里？

写入 `mechanisms`。

### 新增解析公式写在哪里？

写入 family-local `formulas/provider`。

### 公式输出写到哪里？

写入 typed plan/body。

### 某些合法候选无法解析排序怎么办？

写入 `residual_selection`。

### 某个场景不适用或无收益怎么办？

写入 `boundaries`。

### emitter 应该干什么？

只实现 typed plan，不重新组织专家知识。

这正是你说的：

> 使用者真正能够看懂，新公式来了知道写在哪里。

---

## 一个最小公式条目

不需要 Formula IR，只需要统一的小型接口或文档格式：

```text
Formula:
  name
  family
  inputs: g / c / ω 中实际使用的字段
  output: typed plan 字段或 candidate choice
  domain: 适用场景
  fallback: 不适用时怎样处理
  residual: 是否允许 measurement 修正
  evidence: 相关正确性和性能实验
```

它可以是：

* C++ function；
* Python provider；
* typed builder method；
* plugin-local rule。

关键是位置和职责统一，而不是一定有某个新的类。

---

# 八、已经做完的实验会不会浪费

基本不会。

## 已有正确性实验

继续证明：

* factorized knowledge 生成的 plan 是正确的；
* 多种格式和路径能够由共享机制承载。

## 已有性能实验

继续证明：

* factorization 不是只产生可运行代码；
* 解析知识和 residual selection 最终能产生高性能实现。

## 已有 capability 实验

可以重新解释为：

> 同一机制和知识函数在不同 capability 下重新实例化，产生不同但正确的 plan。

这正是 Knowledge Factorization 的核心证据。

## 已有 measurement 实验

可以重新解释为：

> 解析构造后还剩下哪些不能稳定闭式排序的经验残差。

它们不再抢 novelty，反而帮助测量 factorization 的边界。

## 已有 ggml 和 e2e 实验

继续证明：

> 因式分解后的知识能够真正进入部署软件，而不只是停留在一个 compiler microbenchmark 中。

---

# 九、还需要补哪些实验

不需要重新做一整套实验，只需补几项直接证明 Knowledge Factorization 的实验。

## 1. Knowledge census

把现有承重决策分类：

* mechanism；
* analytic formula；
* boundary；
* measured residual；
* 尚未归位的 baked decision。

这不是 verifier，只是论文和重构所需的清点。

最终报告：

```text
多少机制
多少公式
多少 residual
多少格式/能力组合
多少尚未归位决策
```

---

## 2. 规则扇出实验

展示一个机制或公式覆盖多少组合，例如：

* 一个 repack mechanism 覆盖多个量化格式；
* 一个 VLEN 公式覆盖多个 capability；
* 一个 loop structure 被多个 op/regime 复用。

核心指标是：

[
Fanout(r)
=========

\text{规则 }r\text{ 被重新执行的有效组合数}
]

---

## 3. 新组合迁移实验

不一定要新增完整第三后端。

可以设计：

```text
已有：
g₁×c₁
g₁×c₂
g₂×c₁

观察：
规则能否直接生成 g₂×c₂
```

这能证明系统不是把已经测试过的点重新包装起来。

---

## 4. Factorization 消融

比较：

1. 当前完整 factorized system；
2. 去掉关键 formula，使用固定值；
3. 使用逐格式/逐板 lookup；
4. 纯 measurement selection。

观察：

* 正确性；
* 性能；
* 新能力上的 cold start；
* 需要的 point rows；
* 扩展修改量。

---

## 5. Residual 规模

报告：

* 多少选择由 analytic prior 直接决定；
* 多少发生 measured flip；
* 哪些决策无法被当前 formulas 解释；
* residual 是否随产品空间增长保持有限。

这会让“测量只是残差”成为实证结果，而不是口号。

---

# 十、论文 novelty 应当如何正式表述

建议核心主张：

> **Weft factorizes execution-layer expertise into reusable typed mechanisms, capability- and context-conditioned analytic knowledge, explicit applicability boundaries, and a bounded empirical residual. This factorization allows a compiler to reconstruct high-performance kernel families across formats, usage regimes, and hardware capabilities, rather than maintaining pointwise implementations or relying on unconstrained search.**

中文：

> **Weft 将 execution-layer 专家知识因式分解为可复用的类型化机制、由能力与静态上下文条件化的解析知识、明确的适用边界以及有限经验残差。借助这种因式分解，编译器能够跨格式、使用场景和硬件能力重建高性能 kernel family，而不是继续维护逐点实现或依赖无约束搜索。**

这句话不会把项目写成：

* verifier；
* Formula IR；
* autotuner；
* runtime adaptive compiler；
* 形式化证明系统。

它抓住的就是：

> **知识如何被拆开、复用、组合并重新执行。**

---

# 十一、对整体改动规模的最终判断

## 论文定位

**改动较大。**

从：

> 公式构造候选，加上 legality 和 bounded measurement selection

提升为：

> execution knowledge factorization 重建 kernel family，selection 只是 residual。

这是必要的 novelty 升级。

## 软件架构

**改动中等，主要是重构和归位。**

* 不改主 pipeline；
* 不改大多数 typed structures；
* 不改 emitter 输出；
* 不改 benchmark harness；
* 不改部署路径；
* 不加 verifier；
* 不新建通用 IR。

主要动作是：

* 统一 formula/provider 位置；
* 把承重决策从散落位置归到 formulas、mechanisms、boundaries 或 residual；
* 形成清晰的 family-local 知识结构；
* 给用户一个明确的新增知识路径。

## 已有实验

**绝大多数保留，只需重新组织叙事和补少量 factorization 实验。**

所以你对当前阶段的定义仍然是准确的：

> **系统完善与代码重构阶段。**

不是重新立项，也不是推翻已有工程。

---

下面这份可以直接交给 agent，作为新文档创建和 prompt 修改的定案输入。

# 定案令：两柱不变，以 Execution Knowledge Factorization 重建论文核心

## 一、总裁决

本轮不是推翻现有系统，不是新增第三根柱，也不是转向 verifier、形式化验证、Formula IR、通用 autotuner 或 runtime adaptive compiler。

两根柱的文字与地位保持不变：

### 柱一：可扩展性来自能力驱动、类型化的 execution-layer 模板

新格式、新能力、新机制、新后端、新上下文和新知识进入各自 typed owner，避免重新退化为逐格式、逐板和逐后端手写。

### 柱二：高性能来自可执行知识

专家知识必须真实参与 mechanism、typed plan、参数推导、能力与上下文专化、合法候选构造、有限选择、适用边界以及最终部署性能。

本轮新增和强化的是两柱共同的形式核心：

> **Execution Knowledge Factorization，执行知识因式分解。**

它不是第三根柱，而是两柱同时成立的共同机制。

---

## 二、必须纠正的研究方向偏移

后续文档不得再把论文核心降格为：

> typed candidate construction + legality + bounded measurement selection。

上述结构可以保留，但只能是系统实现流程，不能成为论文的主要 novelty。

论文真正研究的是：

> 专家手写 kernel 中混合在一起的格式知识、执行机制、硬件能力适配、参数推导、使用场景、适用边界和经验选择，能否被因式分解为少量可复用、可组合、可重新执行的知识成分，从而重建跨格式、跨能力和跨场景的 kernel family，而不是继续保存大量逐点实现。

不得再声称“公式变化不影响 novelty”。主公式定义了论文的研究对象、源语言和高性能来源，公式变化会直接改变 novelty。

---

## 三、明确删除的方向

从主论文点、架构核心和后续 prompt 中删除以下方向：

* Reason Completeness；
* No Hidden Specialization verifier；
* 源码 dependency/effect verifier；
* “乘积空间在源码中无名”的语法级机检 novelty；
* 通用 Formula IR；
* 通用表达式 AST；
* 形式化证明系统；
* 将 grep census 或静态审计包装成主要贡献；
* 将所有专家知识都宣称为全空间闭式；
* 将 legality、freshness、qualification 或 measurement governance 包装成主要 novelty。

这些内容最多可作为普通工程纪律、未来完善项或 artifact hygiene，不得改变论文中心。

---

## 四、新的知识模型

一个 execution family 的知识表示为：

[
\mathcal E_f
============

\left\langle
\mathcal M_f,,
\mathcal F_f,,
\mathcal B_f,,
\mathcal R_f
\right\rangle
]

其中：

* (\mathcal M_f)：可复用 typed mechanisms；
* (\mathcal F_f)：根据格式、能力和静态上下文构造 mechanism 参数与 typed plan 的 analytic knowledge；
* (\mathcal B_f)：合法性、资源边界、适用域、fallback 和负边界；
* (\mathcal R_f)：解析知识无法可靠排序时留下的有限 empirical residual。

Knowledge Factorization 的核心主张不是“系统有四个物理模块”，而是：

> 点状专家实现能够被拆成这些具有不同作用、不同复用范围和不同增长方式的知识成分。

禁止提前建立巨型全局服务或深层类树。知识可以继续采用 plugin-local function、provider、typed builder 和 sidecar data 实现。

---

## 五、形式核心

保留以下变量：

* (g)：格式、编码、布局和 mechanism facts；
* (c)：canonical capability snapshot；
* (\omega)：有限静态使用上下文，包括 op、layout、regime 和必要 shape bucket；
* (v)：有限、具名的 implementation family/candidate。

因式分解后的知识构造 typed execution plan：

[
P_v(g,c,\omega)
===============

\operatorname{Compose}*v
\left(
\left{
m*{i,v}
\left(
g,\omega;
\theta_{i,v}
\right)
\right}*i
\right),
\qquad
\theta*{i,v}
============

f^A_{i,v}(g,c,\omega)
]

其中：

* (m_{i,v}) 是 typed mechanism；
* (f^A_{i,v}) 是 analytic knowledge function；
* (\theta_{i,v}) 是 mechanism、plan 或 body 实际消费的 typed decision；
* `Compose` 表示有序的 typed plan construction，不强加结合交换代数；
* (P_v) 必须是 emitter/lowerer 能直接消费的完整 typed plan。

合法候选：

[
\mathcal V(g,c,\omega)
======================

\left{
P_v(g,c,\omega)
\mid
L_v(g,c,\omega)=1
\right}
]

最终实现：

[
K^*(g,c,\omega)
===============

Emit
\left(
Select_{\mathcal R}
\left(
\mathcal V(g,c,\omega)
\right)
\right)
]

解释优先级：

1. Knowledge Factorization 负责重建候选 family；
2. legality 过滤非法 plan；
3. empirical residual 只修正合法候选之间的剩余排序；
4. emitter 实现已构造和已选择的 typed plan。

不得把 selector 或 measurement 写成与知识因式分解平行的第二项 novelty。

---

## 六、新六律

### 律一：变化归位律

每类变化进入主要 typed owner：

* 新格式、编码、布局 → typed facts；
* 新能力 → canonical capability；
* 新机制 → typed mechanism；
* 新解析知识 → family-local formula/provider；
* 新经验知识 → residual sidecar；
* 新静态上下文 → bounded (\omega)；
* 新边界 → boundary record；
* 新后端/family → plugin-local realization。

目标是使使用者能够明确知道新增内容应写在哪里。

### 律二：知识因式分解律

高性能实现不应只以完整代码点保存，而应尽可能分解为：

```text
shared mechanism
+ analytic formula
+ applicability boundary
+ optional empirical residual
```

该律是设计和研究命题，不要求建立语法 verifier。

### 律三：族先于点律

系统首先描述 kernel family，再在具体 (g,c,\omega) 上实例化。

逐点规则和 measurement row 可以存在，但只能作为经验残差，不得成为默认知识形态。

论文应通过规则扇出、组合覆盖、扩展修改量和 residual 数量验证这一规律，而不是只写渐进复杂度公式。

### 律四：公式构成律

一个 analytic formula 只有真实构造 typed plan，才算执行知识。

必须满足：

* 输入来自真实 typed (g,c,\omega) 字段；
* 输出决定 mechanism choice、plan 参数、候选结构、资源边界或 analytic prior；
* 输出进入 typed plan/body；
* emitter/lowerer 真实消费；
* 有明确适用域和 fallback；
* 有正确性或性能证据。

把现有常数包装进 provider、隐藏 winner lookup、只生成 reason 字符串或制造无承重 optional field，不算公式贡献。

### 律五：解析主导、测量修残律

analytic knowledge 负责：

* mechanisms；
* candidate structure；
* typed plan；
* legality/resource boundary；
* cold-start prior。

measurement 只在解析知识已经构造出的有限合法候选之间修正排序。

论文必须测量 analytic coverage、winner flip、residual density 和 residual 随产品空间增长的趋势。

### 律六：证据回流律

正确性、性能和负结果必须归因到：

* mechanism；
* formula；
* capability；
* context；
* boundary；
* empirical residual；
* compiler/HW wall；
* e2e wash。

失败不得默认转化为新的逐格式或逐板特判。它应当帮助修改知识因式分解及其适用范围。

---

## 七、软件重构边界

本轮定位仍然是：

> **系统完善与代码重构。**

不得发起新的大系统战役。

保留现有：

* typed facts/context；
* capability；
* providers；
* typed plans/body；
* plugin lifecycle；
* legality；
* measurement sidecar；
* emitter/lowerer；
* ggml 接入；
* correctness/performance/e2e harness。

主要重构动作：

1. 明确每个 family 的 facts、mechanisms、formulas、boundaries、residual selection、plans 和 emitter；
2. 将散落的 code-affecting decision 逐步归入上述位置；
3. 为公式建立统一的小型描述格式；
4. 清理 emitter 中重复或二次决策；
5. 建立一个面向维护者的 knowledge index，回答“新增公式应该写在哪里”；
6. 不建立全局 Formula service，不提前抽象第二实例尚未需要的接口。

建议的最小公式条目：

```text
Formula:
  name
  family
  inputs
  output
  domain
  fallback
  optional residual key
  evidence
```

它可以由普通 C++/Python function 或 plugin-local provider 实现。

---

## 八、现有资产的重新归类

* typed mechanisms/body → (\mathcal M)；
* analytic providers、VLEN/版本/capability 决策 → (\mathcal F)；
* legality、fallback、no-benefit、compiler/HW wall → (\mathcal B)；
* LMUL、SP4、loop-order 等 measured selection → (\mathcal R)；
* emitter → typed plan realization；
* ggml、强对手、e2e → factorized knowledge 的部署和性能证据。

不得因新 framing 推翻或重做已有实验。

---

## 九、实验重组

已有实验继续保留：

1. 正确性：证明 factorized knowledge 生成的实现正确；
2. capability specialization：证明同一知识在不同 capability 上重新实例化；
3. 多格式/多路径：证明 shared mechanisms 和 formulas 具有扇出；
4. 强对手：证明生成实现具有真实竞争力；
5. ggml：证明系统可部署；
6. e2e：证明局部收益是否穿透系统；
7. negative result：标定 factorization 的边界。

新增实验应尽量复用已有数据，只增加：

### A. Knowledge census

统计 mechanisms、analytic formulas、boundaries、empirical residual 和尚未归位的 baked decisions。

### B. Rule fan-out

统计一条机制或公式覆盖多少格式、能力和上下文组合。

### C. Cross-combination transfer

保留某个未直接实现的 (g\times c) 或 (g\times c\times\omega) 组合，观察已有知识能否直接生成正确且有竞争力的实现。

### D. Factorization ablation

比较：

* 完整 factorized system；
* 去除关键 analytic formula；
* 使用固定值或 point lookup；
* 更依赖 measurement 的版本。

### E. Residual density

统计：

* analytic prior 直接决定的点；
* measurement flip 的点；
* 必须逐点处理的 residual；
* residual 是否随产品空间增长保持有限。

第二 family/IME 的复用是强证据，但不得为了形式完美而推翻已完成主线。它应作为强化实验和系统完善目标。

---

## 十、论文 novelty 定案

论文核心不得再表述为：

* typed candidate builder；
* legality-first selector；
* measurement governance；
* lookup-driven autotuner；
* runtime adaptive compiler；
* Formula IR；
* verifier。

建议 headline：

> **Weft factorizes execution-layer expertise into reusable typed mechanisms, capability- and context-conditioned analytic knowledge, explicit applicability boundaries, and a bounded empirical residual. This factorization enables a compiler to reconstruct high-performance kernel families across formats, usage regimes, and hardware capabilities.**

中文：

> **Weft 将 execution-layer 专家知识因式分解为可复用的类型化机制、由能力与静态上下文条件化的解析知识、明确的适用边界以及有限经验残差，使编译器能够跨格式、使用场景和硬件能力重建高性能 kernel family。**

论文的三项主要贡献建议写成：

1. **Knowledge model**：提出 execution knowledge factorization，将点状专家实现拆成 mechanisms、analytic knowledge、boundaries 和 empirical residual；
2. **Compiler realization**：用 capability-driven typed execution layer 将这些知识重新实例化为真实 kernel family；
3. **Empirical study**：在真实格式、硬件和部署路径上测量 factorization coverage、rule fan-out、cross-combination transfer、residual density 与最终性能。

---

## 十一、后续 agent 行为约束

后续 agent 在新建文档、修改 prompt 和重构代码时必须遵守：

* 两柱文字与地位不变；
* Knowledge Factorization 是两柱共同核心，不是第三柱；
* 六律可以继续修改，但必须服务 factorization；
* 不重新引入 Reason Completeness、No Hidden Specialization 或 verifier novelty；
* 不把 legality、measurement schema、freshness 或审计工具抬成主贡献；
* 不以防止 overclaim 为由把 family-level knowledge 重新降成普通 candidate construction；
* 不声称公式变化不影响 novelty；
* 不建立大一统 Formula IR；
* 不推翻现有实验和代码路径；
* 以系统完善、知识归位和代码重构为当前阶段；
* 所有新文档必须明确回答：

  1. 专家知识被因式分解成了什么；
  2. 一个知识单元能够覆盖多少实现组合；
  3. 哪些部分由 analytic knowledge 产生；
  4. 哪些部分仍是 empirical residual；
  5. 用户新增格式、能力、机制、公式和 residual 时分别写在哪里。

本轮成功标准不是文档更复杂，而是任何新维护者读完后都能清楚理解：

> 系统拥有哪些知识成分，它们如何共同生成 kernel family，以及新增一条公式究竟应该进入哪个位置。

这份定案的核心变化可以概括成一句：

> **两柱不动；删除 verifier 路线；保留 09 的可信系统骨架；恢复早期 family-level ambition；用 Knowledge Factorization 作为真正的 novelty 中心。**
