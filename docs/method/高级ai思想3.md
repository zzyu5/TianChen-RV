# 可以接 GPU，而且这其实可能是最自然的“第二实例”

> 文档性质：教师方向讨论稿，经项目侧校准后作为 V2 的思想来源，不是稳定 spec。
> 最终术语、当前/目标状态和工程契约以
> [《项目全景与 Spec 重构前方法基线 V2》](./项目全景与Spec重构前方法基线v2.md)
> 与 `.trellis/spec/` 为准。

## 项目侧校准：先固定六个不许混用的对象

本文后续出现的宽泛 “backend” 一词，应按语境拆成：

| 对象 | 含义 |
|---|---|
| execution paradigm / realization grouping | RISC-V 与 GPU 这类跨计算范式实例分组 |
| selection/deployment domain | target/profile 显式绑定的单设备、地址空间与 ABI/runtime 可共存作用域；不同 domain 不进入同一 selector |
| typed construction owner | RVV、IME、Scalar、future NVIDIA-GPU、future AMD-GPU 等拥有公式与 typed body 的 owner |
| capability profile | domain capability environment 及其 owner-local projection，例如不同 GPU generation/profile |
| final typed construction result | selected owner 的公式与 mechanisms 已经构造完成的 computation body/plan |
| artifact lowerer/backend | 机械消费 typed result 并交给 EmitC/LLVM/NVVM/ROCDL 等 toolchain 的物化层 |

还需固定四条校准：

1. GPU 不重新解释两柱、六律或原主公式；它只是未来第二执行范式上的验证域；
2. 原公式中的 `Emit_v(c)` 是候选构造式内部的组合/实现投影，不是当前 C++
   `emitter`、artifact registry 或 NVVM lowering；
3. RISC-V/GPU 不直接充当同一个 selector 中的 candidates。AOT target/profile 先绑定
   selection/deployment domain，之后只在该域内收集 owner-qualified candidates；同一 X60
   域中的 RVV/IME/Scalar 可以选择和 fallback，正如同一 NVIDIA 域中的 SIMT/MMA 可以选择；
4. 本文描述的是目标态。当前仓库尚无 GPU capability、formula、typed body、artifact、
   runtime 或硬件证据，不能写成“已经支持 GPU”。

下文保留教师讨论稿的原始措辞，但按此校准阅读：宽泛的 “RISC-V/GPU family” 指
selection/deployment domain 或 realization grouping；RVV/IME/Scalar/NVIDIA owner 指 typed
construction owner。若下文出现“先绑定 RVV/IME exact family”或把 candidate origin 当作
binding，应由本节与稳定 spec 覆盖，不能据此取消同一 target domain 内的跨-owner 选择。

但先把最关键的边界说清楚：

> **Weft 可以扩展成同时支持 RISC-V 和 GPU 的 execution-layer 算子编译器。**
>
> **这不等于把项目改造成另一个完整 TVM。**

你老师提出的理解，我认为比前面围绕 QIGen 不断收窄的 framing 更接近项目的本质：

> **在碎片化硬件生态中，如何同时获得可扩展性和高性能？**

RISC-V 是第一个、也是目前最完整的实例；GPU 可以成为第二个计算范式明显不同的实例。

这不会改变两柱，反而会使两柱更容易被证明。

---

# 一、真正的研究问题是什么

可以把项目的总问题稳定成：

> **如何设计一个可扩展的 execution-layer 算子编译器，使 operator、数据格式、硬件能力、
> construction family 与 artifact 路径可以独立演化，同时仍允许每个目标执行专家级专化，
> 生成有竞争力的 kernel？**

这里存在一个非常清楚的矛盾：

```text
统一抽象做得越强
→ 扩展越容易
→ 越可能抹平底层硬件差异
→ 性能下降

目标特化做得越深
→ 性能越高
→ 越容易退化成
  operator × format × target profile × construction family 的逐点实现
```

Weft 要回答的不是“能不能生成 kernel”，而是：

> **怎样先把变化轴解耦，以获得可扩展性；再在编译时把这些轴按照目标能力重新组合，以恢复高性能。**

这就是两柱真正共用的核心。

ARS 的正式编辑裁决其实仍保留了这个定位：Weft 是完整 execution-layer compiler，capability、typed mechanism、body construction、backend、ABI 和 runtime 都是 compiler 本体，而不是公式到 emitter 的玩具管线。

---

# 二、两根柱的定义不改，GPU 只增加未来验证域

## 柱一：可扩展性来自能力驱动、类型化的 execution-layer 模板

沿用项目既有定义。GPU 目标态额外检验的是：

> operator、format、target capability、construction family 和 artifact 的变化分别进入自己的
> typed owner，而不是形成完整乘积。

在 RISC-V 上，碎片化来自：

* RVV 版本；
* VLEN；
* LMUL 和 fractional LMUL；
* 寄存器预算；
* 标准扩展和厂商扩展；
* RVV、IME、Scalar 等不同执行 family；
* 不规则量化格式。

在 GPU 上，碎片化来自另一套轴：

* NVIDIA、AMD、Intel 等 family；
* 不同 GPU generation；
* warp/subgroup 大小；
* shared memory 和寄存器容量；
* tensor-core/MMA 数据类型和 tile；
* async copy、barrier、TMA 等能力；
* 不同 memory layout 和 swizzle；
* 不同 wave/warp execution model。

所以 GPU 不是与现有设计冲突，而是一个更强的柱一压力测试。

---

## 柱二：高性能来自可执行知识

柱二的定义同样不改。GPU 目标态中的专家知识内容与 RISC-V 不同，但仍须由原有
`g/c/ω` 构造关系真实生成 code-affecting typed result。

RISC-V 上的知识可能是：

* LMUL；
* strip width；
* vector gather；
* widening；
* accumulator arrangement；
* vector-register budget。

GPU 上的知识则是：

* block tile；
* warp tile；
* threads per block；
* pipeline stages；
* shared-memory layout；
* global-to-shared staging；
* async copy；
* tensor-core 或 SIMT path；
* warp reduction；
* split-K；
* fused dequant placement。

统一的是：

```text
facts + capability + context
→ 构造高性能执行方案
```

不统一的是每个 construction family 的具体机制、公式、资源模型和 typed body。

这点非常重要：

> **跨 execution paradigm 共享的是构造责任边界，不是同一套 kernel 模板、同一个 Plan
> 类或同一套调参公式。**

现有 ARS 审计也已经明确：RVV 与 IME 应共享 `g/c/ω` 的角色和构造—选择—实现的因果边界，但不应被迫共享同一个 C++ Plan、相同 op 类型或相同资源模型。GPU 也应按同样原则接入。

---

# 三、RISC-V 和 GPU 为什么都属于“结构化碎片”

你提到 RISC-V 的扩展虽然碎片化，但通常有扩展规范、能力字段和一定的蓝本。这个观察非常重要。

可以把目标硬件分成：

\[
\text{稳定的 family skeleton}
+
\text{随目标变化的 capability}
\]

## RISC-V 的 skeleton

```text
scalar/vector register model
vector length
SEW/LMUL
load/store
reduction
optional dot/matrix extensions
```

具体能力由 (c_{\mathrm{rvv}}) 表达。

## GPU 的 skeleton

```text
grid
→ block/workgroup
→ warp/subgroup
→ thread
+ global/workgroup/private memory
+ barrier/shuffle/MMA
```

具体能力由 (c_{\mathrm{gpu}}) 表达。

MLIR 本身已经以这种方式组织 GPU：通用 `gpu` dialect 表示 launch、thread/block、memory space、barrier、shuffle、subgroup 等中层概念；`nvgpu` 和 `amdgpu` 再表示 NVIDIA、AMD 的专用能力。([MLIR][1])

这恰好说明：

> **“通用 family 结构 + typed target capability + family-local 专化”是一个自然的跨 RISC-V/GPU 建模方式。**

---

# 四、接入 GPU 后，完整架构应该是什么

建议采用一个“窄腰”结构：

```text
               多种 source/frontends
   GGML source ops / Linalg / TTIR / StableHLO
                         │
                         ▼
              canonical execution problem
             operator semantics S
           + format/layout facts g
           + bounded static context ω
                         │
                         │  target profile c_f
                         ▼
             family-local construction
       ┌────────────────┴────────────────┐
       │                                 │
   RISC-V family                     GPU family
 RVV / IME / Scalar          NVIDIA / AMD / other
       │                                 │
 RVV mechanisms                 GPU mechanisms
 RVV formulas                   GPU formulas
 RVV boundaries                 GPU boundaries
       │                                 │
       ▼                                 ▼
 typed RVV/IME body              typed GPU body
       │                                 │
       ▼                                 ▼
 EmitC/LLVM/...        gpu/vector/nvgpu/amdgpu
                                         │
                                         ▼
                                  NVVM / ROCDL / object
```

当前 compiler 已经有 proposal/selection、typed family IR、capability、artifact lowering、ABI/runtime
和历史 evidence 资产，但 physical canonical source coverage、target/profile 驱动的 domain binding
以及重构后 current-artifact evidence 链仍未完整闭合。GPU 未来应新增一条 domain/owner-local
construction 与 artifact 分支，而不是重新定义上层系统。

---

# 五、哪些东西跨 RISC-V/GPU 共享，哪些不能共享

| 层次                      | 应共享                                                  | 不应强行共享                         |
| ----------------------- | ---------------------------------------------------- | ------------------------------ |
| Source semantics        | matmul、matvec、dequant、dot、reduction 等                | 目标线程/寄存器细节                     |
| Format facts (g)        | bit packing、scale、zero/min、codebook、layout           | RVV intrinsic 或 GPU MMA 名称     |
| Static context (\omega) | op、decode/prefill、shape、layout、regime                | GPU stages 或 RVV LMUL          |
| Capability role (c)     | “目标能做什么”的 typed 概念                                   | 一张包含所有平台 optional 字段的巨型 struct |
| Pipeline contract       | construct → legality → select → typed body → artifact | 同一个 physical Plan              |
| Measurement rule        | 只修正合法候选排序                                            | 相同 measurement key/schema      |
| Final typed body        | 完整、可被 artifact lowerer 机械消费                           | 相同 body op、相同资源模型              |
| Formula形式               | (f^A(g,c,\omega))                                    | 相同公式内容                         |

最准确的一句话是：

> **统一的是问题分解方式和责任边界；不统一的是专家知识内容和最终 kernel 形态。**

---

# 六、现有主公式如何作用于未来 GPU family

不需要修改公式；只在正文中说明公式总是在一个已经绑定的 construction family 作用域内
求值。不要另写 GPU 版主公式，也不要给原公式增加新的选择变量。

项目既有公式原样保留：

\[
\theta_{i,v}=f^A_{i,v}(g,c,\omega)
\]

\[
K_{v,\omega}(g,c)
=
Emit_v(c)\circ
\bigoplus_i
\left\{
m_{i,v}(g,\omega)
\text{ with }
\theta_{i,v}
\right\}
\]

在目标态中，外层 target/domain binding 先得到 `(d,C_d)`，域内 owner 再投影 `c_o`，所以式中的
`c` 就是当前 owner 的 `c_o`，`v/m/f^A/θ/K` 的 owner 也都在 `o` 内。这里尤其要避免名称误导：
`Emit_v(c)` 表示把已确定
的 mechanisms 与参数组成完整候选实现 `K` 的构造投影；最终 C/object/cubin/hsaco 的
packaging 属于其后的 artifact lowerer。由此明确：

* capability 是 family-local 的；
* mechanism 是 family-local 的；
* formula 是 owner-local 的；
* typed construction result 是 owner-local 的；
* artifact lowering 不能反向决定上述任何内容。

RISC-V 和 GPU 也不应被同一个性能 selector 当作候选，除非未来真的实现并验证跨设备
runtime dispatch。

正常 AOT 场景是：

```text
target profile
→ 先绑定 selection/deployment domain
→ 域内 owners 各自构造/合法化候选
→ 只在同域 owner-qualified 合法候选中选择
```

---

# 七、GPU kernel 为什么会高性能

仅仅把一个 loop lower 到 `gpu.launch`，不会自动高性能。

MLIR 官方 GPU lowering pipeline明确要求输入已经是显式并行的 GPU IR；它负责将已有 GPU 结构转换和序列化到 NVVM 等目标，但不会替编译器自动决定并行映射。([MLIR][1])

因此，Weft 的 GPU construction 层需要真正构造以下内容。

## 1. 并行映射

从：

\[
op,\ shape,\ regime
\]

推导：

* grid shape；
* block shape；
* warp/subgroup 分工；
* 每线程拥有的数据；
* reduction mapping。

例如：

```text
decode / M=1
→ warp-level quantized matvec
→ 多 warp 分列或分 reduction

prefill / M较大
→ block-tiled GEMM
→ warp-level MMA 或 SIMT tile
```

---

## 2. 内存层次

从 format 和 capability 推导：

* packed weights 是否直接从 global memory 解码；
* activation 是否进入 shared memory；
* weight tile 是否进入 shared memory；
* double buffering；
* async copy stage 数；
* shared-memory layout/swizzle；
* coalesced load width。

---

## 3. 计算机制

根据 format 与 GPU 能力选择：

* fused low-bit unpack + SIMT dot；
* unpack 后进入 tensor-core path；
* native int4/fp8/fp4 MMA；
* codebook gather；
* warp shuffle reduction；
* shared-memory reduction；
* vectorized epilogue。

---

## 4. 资源计划

公式输出的 (\theta_{\mathrm{gpu}}) 可能包括：

```text
block_m
block_n
block_k
warps_per_block
threads_per_block
pipeline_stages
vector_width
shared_memory_bytes
register fragments
mma shape
split_k
decode placement
```

合法性根据：

```text
shared memory capacity
register pressure
supported MMA dtype/shape
alignment
warp size
maximum threads
K/group divisibility
```

过滤候选。

---

## 5. 有限经验修正

解析知识通常可以构造少量合理候选，例如：

```text
tile 64×64 / 4 warps / 2 stages
tile 64×128 / 8 warps / 3 stages
tile 128×128 / 8 warps / 4 stages
```

measurement 只负责：

> 在这些已经合法、语义相同的方案里选最快者。

不能由 measurement 决定：

* 是否需要 high-bit plane；
* 是否有 zero/min fold；
* 使用什么 operator；
* 是否需要 shared-memory reduction；
* kernel 的完整算法。

所以柱二在 GPU 上仍是：

\[
\boxed{
\text{高性能}
=

\text{格式语义专化}
+
\text{并行与内存专化}
+
\text{硬件机制专化}
+
\text{有限经验修正}
}
\]

---

# 八、GPU 后端最好怎样使用 MLIR

最实用的路径不是重新发明 GPU IR。

建议：

```text
Weft GPU construction
→ gpu + vector
→ nvgpu 或 amdgpu
→ nvvm 或 rocdl
→ binary
```

MLIR 的 `nvgpu` dialect 本来就是 GPU/Vector 与 NVVM 之间的桥梁，包含 async copy、ldmatrix、barrier 等 NVIDIA 专用结构；`amdgpu` 则为 MFMA、WMMA、swizzle 和 AMD 专用 memory/data movement 提供 typed wrappers。([MLIR][2])

Weft 应负责：

* 选择 tile；
* 选择 warp/block mapping；
* 构造 shared-memory pipeline；
* 选择 MMA/SIMT mechanism；
* 形成 typed GPU body。

标准 MLIR 后端负责：

* 降低 GPU intrinsic；
* 转成 NVVM/ROCDL；
* 目标序列化；
* binary/offload packaging。

这与现有 RISC-V 设计很一致：

> **Weft 决定执行结构；目标 dialect/toolchain 负责最终物化。**

---

# 九、它和 TVM 到底是什么关系

这里需要把“类似 TVM”分成两个含义。

## 含义一：研究问题类似 TVM

也就是：

> 一个系统能否跨多种硬件生成高性能 operator kernel？

这是可以接受的，也是你老师希望看到的。

## 含义二：项目规模等同 TVM

也就是同时实现：

* model graph IR；
* graph fusion；
* tensor program IR；
* general scheduling language；
* search infrastructure；
* CPU/GPU/accelerator codegen；
* runtime；
* deployment；
* 整个 ML frontend ecosystem。

这个方向不适合当前项目。

TVM 当前包含高层 Relax、TensorIR/TIRx、规则式 GPU scheduling、MetaSchedule 搜索、CPU/GPU codegen 和 runtime；其官方架构也明确 GPU 生成依赖 schedule/thread binding，并提供搜索式优化。([Apache TVM][3])

2026 年 TVM 又推出了面向快速演化 GPU/accelerator kernel 的 TIRx；Triton也已经是成熟的高吞吐 GPU kernel 语言和编译器，并进一步提供暴露 layout、shared memory、warp specialization 的低层 Gluon 模型。([Apache TVM][4])

因此不能把论文写成：

> “我们提出了一个新的通用 CPU/GPU tensor compiler。”

这个竞争面太宽，而且会失去已有 RISC-V 和量化系统资产的优势。

更合适的定位是：

> **Weft 提供 TVM-like 的跨目标性能可移植目标，但停留在更窄的 post-graph、
> pre-family-schedule operator execution layer，专门解决碎片化 capability、复杂格式和
> family-local 专家知识的组合问题。**

一句话：

> **TVM-like portability ambition，not TVM-like full-stack scope。**

---

# 十、GPU 对 novelty 的影响

GPU 不会自动产生 novelty。

已有系统已经覆盖：

* CPU/GPU tensor compiler；
* GPU kernel DSL；
* hardware-aware low-precision compilation；
* analytic scheduling；
* autotuning；
* typed extensible dialect。

所以不能说：

> “我们增加了 GPU，因此成为通用新编译器。”

GPU 的真正作用是**强化证据**：

## 对柱一

证明相同的 canonical operator/format/context：

```text
S + g + ω
```

可以在不修改 RISC-V family 的情况下，接入一个完全不同的 GPU family。

## 对柱二

证明同一套构造哲学不仅能表达：

```text
VLEN / LMUL / vector gather
```

还能表达：

```text
warp / block / shared memory / MMA
```

并在两类目标上都达到强对手性能。

## 对整体 novelty

最后的主张可以变成：

> **一种统一但不抹平硬件差异的 execution-layer contract：共享 operator/data semantics 与
> 专化责任边界，同时允许 family-local 专家知识构造完全不同的高性能 kernel。**

我认为这比把 novelty 收缩为“深入 QIGen 的 bitwidth leaf 以下”更符合整个项目的身份。06 文档中的 QIGen 差分仍可保留为 RISC-V/量化 slice 的 related-work 边界，但不应成为整个 compiler 的定义。该文档本身也声明它不修改两柱和系统定位，只调整特定 novelty 边界。

---

# 十一、接入 GPU 需要改多少系统

## 公共架构改动：中等

需要把 capability 从“主要面向 RISC-V 的字段集合”收敛成：

```text
TargetCapability {
    family
    common target facts
    family-local typed payload
}
```

例如：

```text
RVVCapability
IMECapability
NvidiaGpuCapability
AmdGpuCapability
```

不应建立一个包含：

```text
vlen?
lmul?
warp_size?
shared_memory?
mfma?
tensor_core?
```

的巨型 optional struct。

还需要把 source boundary 稳定成：

```text
operator semantics S
+ format/layout facts g
+ static context ω
```

这些本来就应该与 target family 无关。

---

## GPU construction family 与 artifact 工作：较大

真正高性能的 GPU family 不是“加一个 emitter”即可完成。

它需要：

* GPU capability projection；
* GPU mechanism catalog；
* GPU formulas；
* resource/legality model；
* typed GPU body construction；
* GPU lowering；
* runtime launch 和 artifact；
* GPU correctness/performance harness。

所以更准确的判断是：

> **不会推翻现有 compiler，但会新增一个真正的 construction family 及其 GPU artifact/
> runtime。**

公共核心不应大改，backend 工程量会相当可观。

---

# 十二、最合理的 GPU 最小实例

不要一开始做“任意 GPU、任意算子”。

建议选择一个 GPU family，例如 NVIDIA，并复用现有最成熟的 source operation：

```text
quantized contraction / matvec / matmul
```

然后覆盖两个 regime。

## GPU witness A：decode

```text
M=1 或很小
weight-only quantized matvec
fused packed decode + dot
warp-level reduction
```

这更偏带宽受限。

## GPU witness B：prefill

```text
M较大
tiled quantized GEMM
shared-memory staging
SIMT 或 tensor-core path
```

这更偏计算和复用。

使用同一个：

* source op；
* format facts；
* q4/q5/KQuant semantic definition；
* output semantics。

但是生成完全不同的：

* RVV body；
* GPU decode body；
* GPU prefill body。

这就是一个非常漂亮的跨 backend 证据。

---

# 十三、GPU 实验必须怎样设计

## 柱一实验

增加 GPU construction family 时记录：

* source dialect 是否修改；
* format facts 是否修改；
* RVV/IME 是否修改；
* core 是否出现 GPU-name 分支；
* 新增工作是否主要落在 GPU capability、mechanism、formula 和 lowering；
* 同一个格式能否直接进入 GPU family。

最有力的结果是：

```text
新 GPU target
→ 新 GPU capability
→ 新 GPU mechanisms/formulas
→ 新 GPU realization

旧 RVV/IME path 不修改
旧 format semantics 不复制
```

---

## 柱二实验

对同一个 GPU operator 做：

```text
A. generic/fixed GPU implementation
B. shared mechanisms + fixed tile
C. shared mechanisms + analytic GPU formula
D. analytic formula + bounded residual
```

测量：

* kernel correctness；
* generated body shape；
* occupancy/resource；
* memory traffic；
* kernel latency；
* 与强 GPU baseline 的性能；
* decode/prefill 分开；
* analytic-only 与 residual 的增益。

这样可以回答：

> GPU 为什么快？

而不是只报告一个最终数字。

---

# 十四、两柱六律如何覆盖 GPU

六律也不需要改，只需明确跨 family 的读法。

## 律一：变化归位

新 GPU target 进入 GPU family，不进入 RVV emitter。

## 律二：知识因式分解

operator/format semantics 与 GPU execution mechanisms 分开。

## 律三：和积扩展

多个格式和多个 GPU capability 由机制与公式组合，不逐点复制完整 kernel。

## 律四：公式承重

GPU formula 必须真实决定：

* tile；
* warp/block；
* stages；
* shared memory；
* MMA/SIMT；
* legal plan。

## 律五：单向专化

```text
S/g/ω + c_gpu
→ GPU construction
→ legal plan
→ selected GPU body
→ gpu/nvgpu/amdgpu lowering
```

GPU lowerer 不再重新选 tile。

## 律六：证据回流

性能失败应回到：

* tile formula；
* memory mechanism；
* capability；
* boundary；
* residual；

而不是新增某块 GPU 的完整手写 point kernel。

---

# 十五、我建议最终把项目定位成这样

## 中文定位

> **Weft 是一个面向碎片化硬件生态的、能力驱动的 MLIR execution-layer 算子编译器。它将稳定的 operator 与数据语义和 owner-local 的目标能力、执行机制及专家知识分离，并在编译时重新专化，以同时获得扩展局部性和高性能 kernel。RISC-V 是当前主要系统实例，GPU 是 V2 计划验证的第二执行范式目标。**

## 英文定位

> **Weft is a capability-driven MLIR execution-layer compiler for extensible, high-performance operator specialization across fragmented hardware ecosystems. It separates stable operator and data semantics from family-local target capabilities and executable expert knowledge, then recomposes them into target-specialized kernels.**

## 三项贡献

### C1：Extensible execution-layer architecture

operator、format、capability、construction family 与 artifact 的变化保持局部。

### C2：Executable family-local specialization

公式与 mechanisms 构造 target-specific typed body；measurement 只修正合法残差。

### C3：Cross-paradigm realization

在碎片化 RISC-V 和至少一个 GPU family 上展示：

* 相同 source contract；
* 不同 family-local construction；
* 高性能 kernel；
* 强对手和真实部署证据。

---

# 最终建议

你老师提出的方向值得接受，但应采取下面这个版本：

> **把 Weft 从“RISC-V 专用 compiler”提升为“碎片化 execution hardware 的可扩展、高性能算子编译方法”；RISC-V 仍是当前主实例，GPU 是后续要聚焦实现和验证的第二 execution-paradigm realization。**

不要采取下面这个版本：

> **重新实现一个覆盖全模型、全算子、全 CPU/GPU、带通用搜索的 TVM。**

最合适的工程策略是：

1. 保留现有两柱、六律、公式和 RISC-V 资产；
2. 稳定 `S + g + ω + c_f` 的窄腰边界；
3. 新增一个真正 family-local 的 GPU construction；
4. 使用标准 MLIR `gpu/vector/nvgpu` 或 `amdgpu` 做后续物化；
5. 在一个量化算子的 decode 与 prefill 两个 regime 上做到强 GPU 性能；
6. 用“新增 GPU 不破坏 RISC-V”和“GPU 公式确实带来性能”分别证明两柱。

最核心的一句话是：

\[
\boxed{
\text{统一构造契约}
\quad+\quad
\text{非统一的 family-local 专家知识}
\quad=\quad
\text{可扩展且高性能}
}
\]

这应当成为接下来整个项目的总纲。

[1]: https://mlir.llvm.org/docs/Dialects/GPU/?utm_source=chatgpt.com "'gpu' Dialect - MLIR"
[2]: https://mlir.llvm.org/docs/Dialects/NVGPU/?utm_source=chatgpt.com "'nvgpu' Dialect - MLIR"
[3]: https://tvm.apache.org/docs/arch/index.html?utm_source=chatgpt.com "Design and Architecture — Apache TVM"
[4]: https://tvm.apache.org/2026/06/22/tirx?utm_source=chatgpt.com "TIRx: An Open Compiler Stack for Evolving Frontier ML Kernels"
