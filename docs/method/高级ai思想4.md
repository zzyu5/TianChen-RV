# 最终判断

> 文档性质：教师方向讨论稿，经项目侧校准后作为 V2 的思想来源，不是稳定 spec。
> 最终系统边界以
> [《项目全景与 Spec 重构前方法基线 V2》](./项目全景与Spec重构前方法基线v2.md)
> 与 `.trellis/spec/` 为准。本文中的“应当”描述目标态，不自动等于当前代码已经做到。

## 项目侧校准

1. `post-graph、pre-schedule` 的 `pre-schedule` 指 canonical source problem 尚未携带
   owner execution schedule；Weft 的 owner-local construction 正是负责产生 LMUL、tile、
   warp、pipeline 等执行计划，不是说 Weft 不生成 schedule。
2. canonical source 是一份 `P=(S,g,ω)` typed contract，不强制新建通用高层 tensor/tile
   IR；现有 named source op 与 adapter 可以共同实现它。
3. source/problem、target-bound selection/deployment domain、typed construction owner、
   final typed result、artifact lowerer 是五个不同工位。GPU 必须绑定自己的 domain，并从
   source problem 由 GPU owner 构造 typed body，不能从 RVV body 或 `flat_*` plan 再发射。
4. 新增 `S` 只是把系统外部 operator semantics 说清；进入具名 operator/owner 后，原
   `θ=f^A(g,c,ω)` 与 `K` 构造式、两柱和六律均不改写。
5. 同一 target domain 内可以有多个 owner-qualified candidates，例如 X60 上的
   RVV/IME/Scalar；不同 domain（例如 RISC-V 与 future NVIDIA）不得进入同一 selector。

这份讨论的方向可作为 V2 输入，但必须按上面的项目边界校准：

> **Weft 的目标身份是一个基于 MLIR 的、有限领域的自动算子编译器；它拥有高于目标
> 执行细节、远低于完整模型图的 typed source/problem contract，以及 RVV、IME、Scalar、
> future GPU 等 owner-specific 的低层执行 IR。**

所以答案不是三选一：

* 不是从“低层 MLIR”彻底改成“直接吃完整高层模型”；
* 不是把现有低层 MLIR 丢掉；
* 也不应主要定位成 Triton 式、由用户编写执行程序的 DSL。

更准确的关系是：

\[
\boxed{
\text{Weft 在系统边界上是 operator compiler，}
\quad
\text{在内部架构上是 execution-layer compiler。}
}
\]

这两句话并不矛盾。

上传的定义把 Weft 称为“分层可编程的、有限领域的 operator compiler”，普通使用者表达语义，性能专家提供构造知识，后端开发者提供能力和实现；这个总体判断是对的。

---

# 一、为什么“高层还是低层”一直说不清

因为“IR 高低”实际上混合了两个不同维度。

## 维度一：计算语义有多高层

大致是：

```text
整个模型图
→ 融合后的算子
→ 循环和数据流
→ 向量/线程操作
→ 目标指令
```

## 维度二：对具体目标承诺了多少

大致是：

```text
目标无关
→ construction family 已绑定
→ tile / LMUL / warp 已选
→ intrinsic / asm 已选
```

一个 IR 可以：

* 在语义上很具体；
* 但在执行上仍然目标无关。

例如：

```text
quantized_contraction
+ q5_1
+ decode
+ M/N/K
```

它已经比普通 `matmul` 更懂业务语义，但还没有决定：

* RVV LMUL；
* IME tile；
* GPU warp 数；
* shared-memory layout；
* tensor core 还是 SIMT。

所以它既不是“非常高层的模型 IR”，也不是“低层目标 kernel”。

它是一个：

> **语义充分，但实现未定的算子 IR。**

这正是 Weft 最合适的输入位置。

---

# 二、Weft 最合适的位置：post-graph，pre-family-schedule

以前使用过的：

> post-operator execution layer

也容易误导，因为听起来 operator 已经完全 lower 完了。

更准确的是：

> **post-graph, pre-family-schedule operator compiler**

中文：

> **位于图级编译之后、family-specific 执行映射之前的算子编译器。**

完整位置是：

```text
模型/图前端
Torch / StableHLO / TOSA / GGML / 其他前端
                    │
                    │ 图融合、shape/layout 传播、算子识别
                    ▼
      语义充分的 operator problem
             P = (S, g, ω)
                    │
                    │ Weft 自动构造
                    │ + target capability c
                    ▼
        family-specific execution IR
       RVV body / IME body / GPU body
                    │
                    │ mechanical lowering
                    ▼
       LLVM / EmitC / NVVM / ROCDL / asm
```

其中：

* (S)：算什么；
* (g)：数据和格式语义是什么；
* (\omega)：shape、layout、decode/prefill 等静态场景；
* (c)：目标能力。

上传定义中给出的系统级接口：

\[
K=\operatorname{Compile}(S,g,\omega,c)
\]

以及“输入是尚未决定具体实现的语义程序”这一判断，是非常准确的。

---

# 三、原来的“底层 MLIR”没有错，但它不应再代表整个系统

这里要把三种输入区分开。

## 1. 整个 Weft compiler 的规范输入

应该是：

\[
P=(S,g,\omega)
\]

即 operator semantics、representation facts 和 static context。

这是普通编译入口。

例如：

```mlir
%y = weft.quantized_contraction %activation, %weight
    {
      weight_format = #weft.format<q5_1>,
      regime = #weft.regime<decode>
    }
```

这里没有：

* LMUL；
* RVV intrinsic；
* IME MAC leaf；
* GPU block tile；
* warp 数；
* pipeline stages。

---

## 2. 公式构造阶段的输入

当 operator (S) 已经确定后，进入你们现有核心：

\[
g,c,\omega
\rightarrow
m_i,\theta_i
\rightarrow
typed\ body
\]

也就是说：

* 系统级接口是 (\operatorname{Compile}(S,g,\omega,c))；
* 论文主公式可以继续是 (K_{v,\omega}(g,c))；
* 因为在主公式所在的 family 中，operator (S) 已经由 source op、route 或 family 固定。

**不需要为了增加 source 层而修改原主公式。**

---

## 3. backend lowerer 的输入

这是你们原来强调的低层 MLIR：

```text
TypedRepackGemvBody
TypedDequantizeBody
IME tile region
RVV selected body
未来的 GPU typed body
```

它已经包含：

* mechanism topology；
* resource decision；
* loop/dataflow；
* selected parameters；
* family-specific semantics。

Lowerer 再把它机械变成 intrinsic、EmitC、LLVM 或 asm。

这层仍然非常重要，而且正是 Weft 的 execution-layer 核心。

所以正确结论是：

> **原来的低层 MLIR 不需要删除；它应该从“整个 Weft 的唯一输入”重新定位为“Weft 自动构造出来的内部执行 IR”。**

---

# 四、如果一直只接受低层 MLIR，会发生什么

假设用户输入已经是：

```text
RVV loop
LMUL=m2
strip_width=16
unroll=4
某种 decode op sequence
```

那么最重要的执行决策已经被上游或用户做完了。

Weft 剩下的是：

```text
低层 IR
→ intrinsic / EmitC / asm
```

这仍然是有价值的 backend compiler，但它更接近：

* target dialect；
* lowering framework；
* kernel DSL backend；
* code generation infrastructure。

此时很难强力主张：

> 编译器自动决定了为什么这个 kernel 高性能。

因为 LMUL、tile、loop topology 和 mechanism 顺序已经在输入里。

这会直接削弱柱二。

---

# 五、如果直接吃非常高层的模型，又会发生什么

另一个极端是：

```text
完整 PyTorch / StableHLO 模型
→ Weft
→ 所有 kernel 和 runtime
```

这会要求你们同时承担：

* 图优化；
* fusion；
* shape inference；
* memory planning；
* distributed/runtime；
* 全算子 coverage；
* 模型级调度。

这就接近完整 TVM、IREE 或 MLC 范围，既不必要，也不符合当前资产。

所以 Weft 不应直接变成完整模型编译器。

上游图编译器只需把一个算子或小型 fused region 交给 Weft。

---

# 六、Weft 不是“高层”或“低层”二选一，而是一个多层 compiler

建议把整个 Weft 明确拆成三层，而不是让一个 dialect 同时承担所有角色。

## 第一层：Weft Source / Problem IR

作用：

> 描述待编译的算子问题。

内容包括：

* operator；
* operand roles；
* format；
* representation；
* shape；
* layout；
* regime；
* 必要静态 policy。

目标无关，不包含 backend schedule。

可以由多种前门产生：

```text
当前 Weft named source op
GGML adapter
Linalg + 保留下来的格式语义
TTIR adapter
StableHLO/TOSA adapter
```

上传定义对此给出的结论也很准确：不一定发明新的文本语言，但必须明确 Weft 究竟接受什么作为“待编译程序”；不同前门最终规范化成 operator semantics、representation facts 和 static context。

---

## 第二层：Weft Execution Construction

作用：

> 根据目标能力自动决定如何执行。

输入：

\[
S,g,\omega,c
\]

构造：

* bound construction family 内的 candidate 与执行结构；
* mechanisms；
* loop/dataflow；
* LMUL/tile；
* vector/thread mapping；
* resource plan；
* legal candidates；
* typed body。

这是两柱和主公式真正所在的层。

---

## 第三层：Family-Specific Execution IR

例如：

```text
weft_rvv.*
weft_ime.*
future weft_gpu.*
weft_scalar.*
```

作用：

> 保存已经确定的执行方案。

此时：

* RVV body 已经知道 LMUL、strip、vector mechanism；
* IME body 已经知道 tile、MAC deployment；
* future GPU body 在真正实现后必须已经知道 block/warp/shared-memory/MMA plan。

然后 lowerer 只物化它。

---

# 七、那 Weft 到底是不是 DSL？

严格说，需要分三个答案。

## 从技术定义上说

只要你们定义了：

```mlir
weft.quantized_contraction
weft.dequantize_row
weft.quantized_vec_dot
```

这当然是一种 domain-specific language 或 domain-specific IR。

所以说 Weft **包含一个领域专用 source dialect**，没有问题。

---

## 从系统类别上说

不建议把 Weft 的 headline 写成：

> Weft is a DSL.

因为“DSL”很容易让人联想到：

* 用户自己写 kernel；
* 用户指定 block/tile；
* 用户控制 schedule；
* compiler 帮忙 lower。

这会把你们拉向 Triton、TileLang 或 Exo 的比较框架。

你们真正想表达的是：

> 用户描述算子问题，compiler 自动构造执行实现。

所以主类别应是：

> **automatic operator compiler**

而不是：

> user-scheduled kernel DSL。

---

## 最准确的说法

> **Weft is an automatic operator compiler with an MLIR-based domain-specific source IR.**

中文：

> **Weft 是一个自动算子编译器，它使用基于 MLIR 的领域专用语义 IR 作为规范输入。**

所以：

* Weft **有** DSL/方言；
* Weft **不只是** DSL；
* Weft 更不是让普通用户写底层 schedule 的 kernel DSL。

---

# 八、“分层可编程”这个定义好在哪里，又需要怎样修正

上传定义中最好的部分，是区分了三种编程者：

| 角色       | 写什么                                           |
| -------- | --------------------------------------------- |
| 上层程序员/前端 | operator、shape、format、layout 语义               |
| 性能专家     | mechanism、公式、适用域、候选                           |
| 后端开发者    | capability、family-local lowering、目标 primitive |



这确实是你们系统很重要的性质。

但“分层可编程”应该是：

> **系统的扩展模型**

而不是：

> **普通用户必须同时学习三层语言。**

普通使用者只需要第一层。

性能专家和后端开发者是在扩展 compiler，不是在编写每一个程序实例。

因此可以区分：

### 普通 compilation

```text
operator instance
+ format/layout
+ target profile
→ kernel
```

不修改 compiler。

### Compiler extension

```text
新格式 facts
新 mechanism
新 formula
新 backend
```

修改局部 owner。

这是柱一真正需要证明的两种扩展性。上传定义也指出，只有运行时覆盖扩展而没有 compiler 演化扩展，会像封闭 generator；只有扩展框架而没有自动编译覆盖，又会像普通框架。

---

# 九、这对当前代码意味着多大变化

## 不需要大改的部分

全部保留：

* 当前 MLIR dialect；
* typed family body；
* RVV/IME/Scalar；
* capability；
* formula；
* legality；
* selector；
* lowering；
* EmitC；
* ABI/runtime；
* benchmark 和实验。

ARS 的正式裁决也明确保留了完整 execution-layer compiler 定位，并把 capability、typed mechanisms、body construction、backend、ABI 和 runtime 都视为 compiler 本体。

---

## 真正需要补清楚的部分

### 1. 指定一个 canonical source boundary

不一定新建一套大 dialect。

可以直接把现有几个 named source op 定为规范入口：

```text
quantized_contraction
quantized_vec_dot
dequantize_row
reduction
```

定义清楚它们携带哪些：

* (S)；
* (g)；
* (\omega)。

---

### 2. 把低层 typed body 标成内部 IR

直接书写：

```text
TypedRepackGemvBody
IME pre-realized body
RVV selected body
```

可以继续用于：

* lit test；
* backend unit test；
* debugging；
* 专家实验。

但不应把它们当作证明“自动算子编译”的规范用户入口。

---

### 3. 至少让代表性 production path 真正从 source op 到 typed body

也就是：

```text
source op
→ normalized S/g/ω
→ capability c
→ formula/mechanisms
→ family body
→ emission
```

而不是测试直接手写 final body。

---

### 4. 未来 adapter 只负责前端转换

例如：

```text
LinalgToWeft
TTIRToWeft
GGMLToWeft
```

adapter 不决定：

* LMUL；
* tile；
* backend schedule；
* winner。

这些仍属于 Weft。

---

# 十、对两柱的影响

## 柱一不变，但更清楚

柱一现在回答：

> 同一个 source problem 能否跨不同 format、capability 和 construction family 自动编译；
> 新增格式、能力、family 或 artifact lowerer 时，compiler 修改是否保持局部？

source 层稳定，family 层局部扩展。

---

## 柱二不变，而且被加强

柱二现在回答：

> source 中没有 LMUL、tile、warp、pipeline 等性能实现；这些决定如何由 compiler-owned expert knowledge 自动产生，并最终带来性能？

只有 source 不预先包含 schedule，才能清楚证明：

> 高性能确实来自 Weft，而不是来自用户已经写好的低层 IR。

---

# 十一、主公式也不需要改变

系统级接口可以写：

\[
P=(S,g,\omega)
\]

\[
\operatorname{Compile}_f(P,c_f)\rightarrow K_f
\]

进入某个 operator/family 后，继续使用你们原有公式：

\[
\theta_{i,v}=f^A_{i,v}(g,c,\omega)
\]

\[
K_{v,\omega}(g,c)
=
Emit_v(c)\circ
\bigoplus_i
\left{
m_{i,v}(g,\omega)
\text{ with }
\theta_{i,v}
\right}
\]

这里的 `Emit_v(c)` 是原候选构造式内部的 family-local 组合/实现投影，不是 artifact
emitter。最终 EmitC/NVVM/ROCDL/object packaging 在完整 typed `K` 形成之后机械进行。

两者不是两个竞争的理论核心：

* 第一式定义 compiler 的外部问题；
* 第二式定义内部 execution construction。

这是一种**包含关系**。

---

# 十二、最终应当固定的定位

## 最准确的中文定义

> **Weft 的目标身份是一个基于 MLIR 的、有限领域的自动算子到 kernel 编译器。它接收经过图级处理后、语义充分但尚未决定执行实现的算子 IR，将算子语义、数据表示和静态上下文与目标能力结合，自动构造 RVV、IME 等当前 owner 以及 future GPU owner 的 family-specific 低层执行 IR，并进一步生成高性能 kernel。当前仓库尚未构造 GPU body。**

## 更短的版本

> **Weft 是一个以领域专用语义 IR 为输入、以 family-specific execution IR 为内部核心的自动算子编译器。**

## 英文

> **Weft is an MLIR-based automatic operator-to-kernel compiler with a late, target-independent semantic source IR and family-specific low-level execution IRs.**

---

# 十三、三句必须禁止混淆

### 不要说

> Weft 是一个低层 MLIR。

这只能描述某个内部 dialect，不能描述整个系统。

### 不要说

> Weft 直接从完整高层模型生成所有东西。

这超出了范围。

### 不要只说

> Weft 是一个 DSL。

这会让人误以为普通用户负责写执行程序。

应该说：

> **Weft 是 operator compiler；它有一个领域专用 source dialect，并内部构造低层 execution dialect。**

因此，这次不是把原项目从低层彻底改成高层，而是把原先混在一起的两个层次正式分开：

\[
\boxed{
\text{语义输入层}
;\xrightarrow{\text{Weft 自动构造}};
\text{低层执行 IR}
}
\]

原来的低层 MLIR继续存在，并且仍是实现核心；新明确的只是：**它应该是编译结果的一部分，而不是要求普通使用者提前写好的全部输入。**
