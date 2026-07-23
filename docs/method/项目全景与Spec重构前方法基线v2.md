# 项目全景与 Spec 重构前方法基线 V2

> 状态：2026-07-23 内部方法基线。
>
> 本文综合《高级 AI 思想 3》《高级 AI 思想 4》与当前代码/spec 事实，定义 Weft
> 引入 GPU 之后应采用的完整系统抽象，以及真正实现 GPU 之前必须完成的横向重构。
> 它不是 GPU 已实现声明，也不覆盖两柱、六律和既有主公式的权威。
>
> 术语校准：本文中的 `pre-schedule` 指 canonical source problem 尚未携带
> family-specific execution schedule；Weft construction 本身负责产生 LMUL、tile、warp、
> pipeline 等 schedule。`backend` 只在指 artifact lowerer/toolchain 时使用；拥有公式、
> legality 与 typed body 的对象统一称 construction family。

---

## 0. V2 到底改变什么

V2 不是给 Weft-RV 末端增加一个 GPU emitter，也不是把项目扩张成覆盖完整模型图、通用
Tensor IR、全设备搜索和部署的另一个 TVM。

V2 正式明确三件此前没有被充分分开的事：

1. Weft 的规范输入是图级处理之后、执行映射之前的 **operator problem**，不是要求普通
   使用者预先写好的 RVV/IME/未来 GPU 低层执行体；
2. RISC-V 与 GPU 共享的是 problem decomposition 和 construction contract，不共享具体
   mechanism、Plan 类、资源模型或低层 IR；
3. family construction 与 artifact lowering 是两层。一个 family 必须先构造自己的最终
   typed body，之后才选择适合它的 EmitC/LLVM/NVVM/ROCDL 等机械物化路径。

保持不变的内容更多：

- 两根柱不变；
- 六律不变；
- `θ=f^A(g,c,ω)` 与 `K` 构造式不变；
- legality 先于 selector，measurement 只能修正合法候选排序；
- `flat_*` 仍是 RVV formula 产生的最终计算 plan，emitter 直接消费；
- `ConstructedWeak` 与 delete-leaf strong reconstruction 的诚实边界不变；
- RISC-V 仍是当前最完整、最深入、拥有真实证据的旗舰 realization。

V2 的总研究问题是：

> 如何设计一个基于 MLIR 的自动 operator-to-kernel compiler，使 operator semantics、
> representation、静态场景、目标能力与 execution family 可以独立演化，同时允许每个
> family 以自己的可执行专家知识完成深度目标专化，生成有竞争力的 kernel？

这里的共同方法是：

```text
先分离变化轴
  → 保持扩展局部性
  → 绑定目标 family 与 typed capability
  → 由 family-local 知识重新组合
  → 构造目标专化 typed body
  → 机械物化为 family artifact
```

---

## 1. 系统身份：post-graph、pre-schedule 的自动算子编译器

### 1.1 外部身份与内部核心

Weft 在系统边界上是：

> **位于图级处理之后、family-specific 执行映射之前的自动 operator-to-kernel compiler。**

Weft 在内部架构上是：

> **以 family-local execution construction 与低层 typed execution IR 为核心的
> execution-layer compiler。**

这两句话是包含关系，不是二选一。

```text
模型/图前端
  ├─ Torch / StableHLO / TOSA
  ├─ Linalg 或其它结构化算子 IR
  └─ GGML/llama.cpp adapter
          │
          │ 图融合、shape/layout 传播、算子识别由上游负责
          ▼
canonical operator problem P=(S,g,ω)
          │
          │ Weft + target request
          ▼
family-local execution construction
          │
          ▼
RVV / IME / Scalar / future GPU typed execution body
          │
          ▼
family artifact lowering + ABI/runtime
```

因此，原有低层 `weft_rvv.*`、IME body 与其它 family IR 不删除。它们被准确定位为 Weft
自动构造出来的内部执行 IR，而不是整个系统唯一的规范用户输入。

### 1.2 Weft 不是什么

Weft 不负责：

- 完整模型图优化、全模型 memory planning 或分布式图调度；
- 新建一个覆盖任意 tensor 程序的高层 tensor/tile IR；
- 要求普通用户手写 tile、LMUL、warp、pipeline stage 的 kernel DSL；
- 通用搜索式或在线 autotuning；
- 自动得到所有 family、所有 operator 的全局最优 kernel；
- 把不同设备放进尚不存在的跨设备 runtime selector。

Weft 可以包含领域专用 source ops/dialect，但主类别不是“用户调度 DSL”，而是：

> **an MLIR-based automatic operator-to-kernel compiler with a domain-specific
> semantic source contract and family-specific execution IRs.**

### 1.3 总方法与 realization 命名

建议固定以下关系：

```text
Weft                         总方法与 compiler architecture
├─ Weft-RV                   RISC-V 旗舰 realization suite
│  ├─ RVV                    construction family
│  ├─ IME                    construction family
│  └─ Scalar                 construction family
└─ Weft-GPU                  GPU 跨范式 realization grouping（目标态）
   ├─ NVIDIA-GPU             future construction family
   │  ├─ H100 profile
   │  └─ 5090D profile
   └─ AMD-GPU                future construction family
```

`Weft-RV`/`Weft-GPU` 是 execution-paradigm realization 分组，不是 formula candidate，也
不是一个必须吞掉所有 vendor 差异的巨型 plugin。本文中的 family `f` 指真正拥有
capability projection、formula、legality 与 typed construction result 的 construction
family，例如 RVV、IME、Scalar、未来 NVIDIA-GPU 或 AMD-GPU。Artifact contract/lowerer
与 family construction 相接，但不是 family identity 或 compute 的判定来源。

当前仓库和工程名可以继续使用 TianchenRV/Weft-RV；无需在架构重构前做大规模改名。

---

## 2. 四层而不是一个含混的“高/低层 IR”

### 2.1 Source / Problem 层

这一层描述“要编译什么”，而不是“怎样在某个目标上执行”。它包含：

- operator semantics 与 operand roles；
- format/encoding/packing/layout；
- shape/geometry；
- decode/prefill 等有限静态 regime；
- 必要的静态 policy。

它不得包含：

- RVV LMUL、strip width 或 intrinsic；
- IME tile/helper；
- GPU block/warp tile、shared-memory stage 或 MMA 指令；
- measurement winner；
- artifact 名、ABI 拼写或 backend route id。

多个前门可以规范化到同一 problem contract：

```text
GGML named source op ─┐
Linalg + descriptor ─┼─→ canonical execution problem
其它 kernel IR adapter┘
```

adapter 只恢复语义和 representation facts，不决定 family schedule。

### 2.2 Family Binding 与 Capability 层

目标请求先绑定 construction family，并得到该 family 的 typed capability projection。
该步骤决定“由哪个知识域构造”，不决定“该知识域内用哪个 candidate”。

family binding 必须依据显式 target/profile 与 typed applicability，经 registry/interface 完成；
不得在 artifact emitter 中按 family 名、route id 或输出格式二次分派。

### 2.3 Execution Construction 层

family-local formula 根据 problem 与 capability 构造：

- mechanisms 与组合拓扑；
- candidate/typed plan；
- legality/resource requirements；
- analytic prior；
- optional measurement key；
- selected final typed body。

这是两柱、六律与主公式真正承重的层。

### 2.4 Artifact / Runtime 层

final typed body 已经包含全部 code-affecting 计算决定。artifact backend 只负责：

- 机械 lowering；
- 目标 dialect/toolchain 交接；
- symbol/module/object packaging；
- ABI 与 launch contract；
- runtime glue。

当前 RISC-V family 主要使用 EmitC/C/C++/object 路径。未来 NVIDIA-GPU 可以使用
`gpu/vector/nvgpu/nvvm`，AMD-GPU 可以使用 `gpu/vector/amdgpu/rocdl`。Artifact 形态不同，
不意味着重新定义 construction。

---

## 3. V2 窄腰的正式定义

### 3.1 Canonical Execution Problem

定义：

\[
P=(S,g,\omega)
\]

其中：

- `S`：operator semantics。它规定数学/离散语义、operand roles 与结果语义；
- `g`：typed representation facts。它规定 format、encoding、packing、layout、
  逻辑 shape/geometry 及与表示相关的事实，但不携带目标执行选择；
- `ω`：bounded static context。它只包含不属于 representation 本身、但真实承重的
  regime、必要 usage/shape bucket、静态 memory form 或 policy，不包含 runtime data
  distribution 与 winner memory。某个 shape 事实只能在 `g` 或 `ω` 有一个 owner；family
  projection 可以派生视图，但不得双写成两个决定来源。

`S` 的引入不改变原主公式。进入一个具名 operator/family formula 时，`S` 已由 source op
和 formula domain 固定，原公式继续只显式写 `g/c/ω`。

### 3.2 Target Binding 不是 selector

令 `t` 为显式编译目标请求：

\[
\operatorname{Bind}(P,t)=(f,c_f)
\]

其中：

- `f` 是唯一 construction family；
- `c_f` 是该 family 的 typed capability projection；
- 不适用、缺失或冲突的目标事实导致具名 unsupported/reject；
- `Bind` 不产生 formula candidate，不读取 measurement winner，也不选择逐点 leaf。

正常 AOT 模式是：

```text
target/profile request
  → bind one construction family + c_f
  → construct/legalize/select inside f
```

当前不定义 CPU/GPU 跨设备 runtime selection。将来若真有跨设备 dispatch，它必须另有
observer、cost、transfer、artifact 共存和 runtime 证据，不能偷塞进本轮 selector。

### 3.3 原主公式原样保留

在已绑定的 family `f` 内，`c` 取该 family 的 `c_f`。核心构造式仍是：

\[
\boxed{
\theta_{i,v}=f^A_{i,v}(g,c,\omega)
}
\]

\[
\boxed{
K_{v,\omega}(g,c)
=
Emit_v(c)\circ
\bigoplus_i
\left\{
m_{i,v}(g,\omega)
\;\text{with}\;
\theta_{i,v}
\right\}}
\]

V2 只增加作用域说明：

- `v` 是 family `f` 内的有限候选；
- `m`、`f^A`、`θ`、legality 与 `Emit_v` 都由 `f` 的 typed owner 持有；
- `c` 是 `c_f`，不是一个同时塞入 RVV/GPU/IME optional 字段的巨型对象；
- `K` 是已经可由 final typed body 表达的完整候选实现，不是 family/leaf 标签；
- `Emit_v` 是构造式中把已确定 mechanisms/parameters 组成完整候选 `K` 的 family-local
  构造投影。它不是 C++ emitter、artifact registry 或 NVVM/ROCDL backend；最终
  object/cubin/hsaco packaging 属于下游 artifact lowering，不获得新的 compute authority。

没有必要另立一套“GPU 主公式”，也不把 family 名写进原公式正文。若需要讨论多个 family，
可以把 `f` 作为外层作用域标记，但它不是新的 selector 变量。

### 3.4 合法域与薄 selector 继续原义

\[
\mathcal V_f(g,c_f;\omega)
=
\left\{
v\in\mathcal A_f(g,c_f;\omega)
\mid
L^f_v(g,c_f,\omega)=1
\right\}
\]

family-local selector 只允许：

```text
qualified/fresh/selection-valid winner 且仍在合法集
  → 选择该 candidate
否则合法集非空
  → analytic prior
否则
  → named fallback or reject
```

Selector 不产生 mechanism、不补 `θ`、不构造 body、不选择 artifact backend。Measurement
也不能创造 candidate、扩大合法域或把 NVIDIA-GPU body 翻译为 RVV body。

### 3.5 完整编译关系

可以把外层系统接口写成：

\[
\operatorname{Compile}(P,t)
=
\operatorname{LowerArtifact}_f
\left(
\operatorname{Construct}_f(P,c_f)
\right),
\quad
(f,c_f)=\operatorname{Bind}(P,t)
\]

这里 `Construct_f` 内部就是既有 construct → legality → bounded select → final body
语义；`LowerArtifact_f` 只机械消费 construction-qualified final body。

---

## 4. 跨 family 共享什么，不共享什么

| 对象 | 跨 family 共享 | family-local |
|---|---|---|
| Problem contract | `S/g/ω` 的角色、类型纪律与规范化责任 | family-specific applicability projection |
| Capability | identity/provenance/relation/query 的最小协议 | RVV、IME、NVIDIA、AMD 的具体字段与资源模型 |
| Construction | construct→legality→bounded select→final body 的因果顺序 | mechanisms、formula、candidate/plan 类型、prior |
| Measurement | correctness、lineage、qualification、freshness、只修正合法排序 | key、resource metrics、opponent、device/board fields |
| Final body | typed、完整、足以让 lowerer 机械工作 | `weft_rvv`、IME、Scalar、future GPU execution IR |
| Artifact | typed ownership、fail-closed、ABI/runtime provenance | EmitC/object、NVVM/cubin、ROCDL/hsaco 等 |

最重要的原则是：

> **统一的是问题分解、作用域和 authority 顺序；非统一的是专家知识内容与 kernel 形态。**

因此禁止：

- universal capability struct；
- universal physical Plan；
- universal Formula IR/result；
- 让 GPU 读取 RVV `flat_*` plan 再“换一种方式发射”；
- 让同一 common lowerer 解释所有 family 的 compute；
- 为形式统一给每个 family 强加 provider/verifier 中间层。

`flat_*` 是 RVV family 的最终 computation plan，不是 V2 的跨 family IR。未来 GPU 必须从
同一 `P+c_f` construction contract 构造自己的 GPU typed plan/body。

---

## 5. Capability、Family 与 Target Profile

### 5.1 最小公共 capability contract

公共层只需要表达：

```text
capability identity
family ownership
typed relation / provides / implies / conflicts
provenance and conflict/missing policy
family projection/query entry
artifact/runtime availability facts when load-bearing
```

具体字段留在 family payload：

```text
RVVCapability       VLEN/ELEN/SEW/vreg/fractional-LMUL/ISA/toolchain
IMECapability       tile/MAC/data-type/runtime/toolchain
NvidiaCapability    warp/register/shared-memory/MMA/async-copy/artifact
AmdCapability       wave/register/LDS/MFMA/WMMA/artifact
```

不建立含 `vlen?`、`warp_size?`、`mfma?`、`tensor_core?` 等全部 optional 字段的公共巨型
struct。Family formula 只消费自己声明的 typed projection。

### 5.2 Profile 与 family 的关系

H100 与 5090D 应是同一 NVIDIA-GPU construction family 下的不同 capability profile，
而不是两个复制完整 compiler 的 family。它们共享 family skeleton、mechanism categories、
typed GPU body contract 与 NVIDIA artifact/runtime；差异进入 capability 与 family-local
applicability/legality/formula。

AMD-GPU 若使用显著不同的 wave/MFMA/artifact 语义，可以是独立 construction family，同时
复用上层 problem contract 与相同 authority 顺序。

---

## 6. GPU 是完整 construction family，不是 emission branch

### 6.1 GPU family 必须拥有的内容

一个真正的 GPU family 至少拥有：

- typed GPU capability projection；
- GPU mechanism catalog/basis；
- family-local formulas；
- legality 与 resource model；
- bounded candidate/selection space；
- final typed GPU body；
- GPU artifact lowering；
- launch ABI/runtime；
- correctness、resource 与 performance evidence。

仅仅把 loop 包进 `gpu.launch`、登记外部 CUDA kernel 名或增加一个 `emit_gpu` 分支，都不算
GPU construction。

### 6.2 GPU construction 的承重内容

GPU formula 可以决定：

- grid/block/warp/thread mapping；
- block/warp tile 与 per-thread ownership；
- shared-memory/register/global placement；
- coalescing、swizzle、double/multi buffering；
- async copy、barrier 与 pipeline stages；
- SIMT、MMA/Tensor Core、fused unpack/dequant、reduction topology；
- split-K、persistent strategy 与 epilogue placement。

legality/resource model至少检查：

- threads/warps limits；
- shared-memory 与 register usage；
- supported MMA dtype/shape；
- alignment、K/group divisibility、layout compatibility；
- async-copy/barrier generation requirements；
- artifact/runtime capability compatibility。

这些决定必须落入 typed GPU body；lowerer 不再按 format、device name 或 default tile 重选。

### 6.3 GPU artifact 路径

建议复用标准 MLIR 生态，而不是重新发明 GPU IR：

```text
future Weft NVIDIA construction
  → typed GPU execution body
  → gpu/vector/nvgpu
  → nvvm
  → PTX/cubin/device module
  → CUDA launch ABI

future Weft AMD construction
  → typed GPU execution body
  → gpu/vector/amdgpu
  → rocdl
  → hsaco/device module
  → HIP/ROCm launch ABI
```

Weft 决定执行结构；标准 MLIR/LLVM toolchain 负责目标指令和 artifact 物化。GPU 不应被
强迫经过 EmitC，正如 RISC-V 不应被迫经过 NVVM。

### 6.4 当前状态边界

截至 2026-07-23：

- 仓库没有 NVIDIA/AMD GPU construction family；
- 没有 GPU typed body、GPU formula、GPU artifact/runtime 或 GPU 性能证据；
- H100/5090D 是后续 implementation profile，不是当前 supported target；
- GPU 前置的 artifact-neutral family construction rebase 已实施；本轮仍未实现任何 GPU
  family 或 GPU artifact。

因此可以说“V2 architecture 显式容纳 GPU family”，不能说“Weft 已支持 GPU”。

---

## 7. V2 不改写两柱与六律

### 7.1 两根柱沿用既有定义

#### P1：能力驱动、类型化、可复用的扩展架构

operator semantics、representation、static context、target capability、construction family、
typed execution body、artifact/runtime 各有唯一 typed owner，不形成完整笛卡尔积实现。

GPU 是 P1 的强压力测试：若新增 NVIDIA-GPU 需要改写 RVV formula、复制 source semantics、
或在 common emitter 中加 CUDA 分支，说明扩展局部性未成立。

#### P2：family-local 可执行专家知识

高性能知识仍以 mechanism、formula、legality、bounded selection 和 typed body 构造存在。
RISC-V 与 GPU 的知识内容不同，但都必须从 `g/c/ω` 真实产生 code-affecting 结构，不能只
返回标签让 emitter 取回手写 kernel。

### 7.2 六律沿用既有定义，GPU 只作为未来检验域

下列内容是既有六律在 V2 系统边界上的检查项，不是新版本或重新解释：

1. **变化有唯一 typed owner**：`S`、`g`、`ω`、`c_f`、family formula/body、artifact/runtime
   分别归位；
2. **机制源码无经验点**：RVV/GPU mechanism 不嵌入具体 board/device winner；
3. **和积增长**：新增 operator、format、profile、family 主要增加各自 owner，而非复制
   operator×format×device×regime 完整 kernel；
4. **理由可执行、变量充分**：承重决定真实消费声明的 `S/g/ω/c_f` 并产生 typed 结果；
5. **单向专化**：problem→binding→construction→legality→selection→typed body→artifact，
   后层不重新解释；
6. **合格的双向知识积累**：win/loss/wall/wash 更新 formula、capability、measurement 或
   boundary，不产生逐设备 point 特判。

`S` 只补足系统外部 operator semantics 的住址；它不进入原主公式改写 `g/c/ω`，也不把
六律改造成新理论。未来 GPU 的作用只是检查同一不变量能否跨范式成立。

---

## 8. V2 的贡献组织

两柱仍是价值命题，C1/C2/C3 只是论文组织。

### C1：Extensible Execution-Layer Architecture

贡献是 post-graph/pre-schedule 的 typed narrow waist，使 source semantics、representation、
capability、construction family、typed body 与 artifact/runtime 能独立演化。

### C2：Executable Family-Local Specialization

贡献是 formula-centered construct → legality → bounded select → final typed body 方法；
measurement 只修正合法残差，artifact lowerer 无 compute authority。

### C3：Cross-Paradigm Realization

目标贡献是在碎片化 RISC-V 与至少一个 GPU construction family 上证明：

- 同一 source/problem contract；
- 不同 family capability、mechanism、formula、body 与 artifact；
- 各自具有竞争力且可解释的 kernel；
- 新 family 不破坏旧 family。

当前只有 RISC-V 旗舰 realization 拥有完整工程与硬件证据。C3 的 GPU 部分是明确的未来
实现和评价目标，不得由架构文档提前写成已完成贡献。

---

## 9. 当前代码基础与 artifact-neutral rebase 状态

本节必须同时写目标态和当前态，二者不得互相冒充：

| 边界 | 当前已经成立 | 边界 / 尚未完成 |
|---|---|---|
| family / artifact authority | family 先返回 exact typed body，artifact lowerer 机械消费 | 不代表所有 leaf 已能由 mechanisms/formula 重建 |
| RVV body | `flat_*` 与 runtime control 由 construction 拥有，旧 route-provider/protocol 已退出 | 仍有 `ConstructedWeak` 完整 leaf |
| verification | typed dialect 检查结构/类型，capability check 检查已绑定配置 | verifier 不得重放公式或重新选择 compute |
| GPU | V2 lifecycle 已为独立 GPU family 留出正确工位 | 尚无 GPU family、body、artifact、runtime 或证据 |
| evidence | 本地 compiler/build 路径可验证 | current-artifact 真硬件 A/B 尚未闭合 |

### 9.1 已经成立的基础

当前项目已经完成一轮重要横向收口：

- current production formula/construction authority 已进入 family-local typed owner；
- quantize 与 dequantize 进入相同 top-level construction cut；
- RVV `flat_*` 是 formula 产生的最终计算 plan，emitter 不读取旧
  `kind/format/fold_model` 重选；
- shared EmitC conversion 只有一个 `applyPartialConversion` harness；
- direct pass、registry、translate 与 artifact 路径 fail closed；
- `ExtensionPlugin` base construction fail closed，所有 live production family 显式实现
  variant-scoped `constructFormulaPlans(FamilyConstructionRequest, FamilyConstructionResult)`；
- target/profile 在 construction 前绑定 origin family 与 typed `c_f`；
- family construction 返回 exact typed operation/root 或 explicit unsupported；公共编排只
  检查结果存在、仍属于绑定 kernel/variant，并把该 exact result 传给 artifact query；
- `TypedBackendEmissionDriver` 没有 construction hook，constructed-only API 不扫描 module
  重发现 body，只消费已传入的 final typed body/plan；
- `emitc.func` 只属于 current EmitC artifact completion gate；
- mixed-family body 在 standalone materialization 前拒绝；
- Demo、Toy、Template 与 TensorExtLite 的 construction manifest、typed-role replay、route
  provider、通用 readiness verifier、role/status/interface 字符串镜像和 metadata-only
  `lowering_boundary` 已退出 production；它们保留 family-local typed body、legality 与纯
  artifact ABI/callee 常量，artifact 常量不参与 compute；
- RVV construction 现在以
  `RVVBodyRuntimeControl {sew, lmul, policy, runtimeAVLValue}` 明确拥有 exact-body runtime
  control；artifact lowering 不从 ABI
  顺序、route metadata 或 provider state 补出 VL/config 决定；
- 旧 `lib/Plugin/RVV/EmitC` route-provider 栈、`RVVConstructionProtocol`、
  `RVVContractionRouteIdentity`、各 route-family plan-owner mirror 与旧 CMake provider target
  已物理删除；RVV 构建只链接 artifact-neutral `WeftRVVConstruction`；
- segment2 的真实 typed-body construction 已回到自己的 realization owner；contraction
  realization 直接消费 formula 的 relation/layout/resource plan，不再经 primitive-facts 或
  intrinsic-metadata mirror 重放；
- provider/formula replay validator 已删除；保留的 dialect `verify()` 只检查 typed body 的
  局部结构、类型、operand role、policy 与语义关系，construction-time capability check 只
  检查已经绑定的 body config，二者都不是第二 compute authority；
- `ConstructedWeak` 与 strong reconstruction 的界线保持诚实。

这里的“artifact-neutral”不仅指 lifecycle/caller 已迁出 EmitC driver，也指确定性 family
不再用 artifact route/manifest 判定 construction 完成。但这仍只是 authority 与结构收敛：
typed body 是否已经包含可由 mechanisms/formula 重建的完整执行知识，仍须按
delete-leaf reconstruction 单独验证。

这些资产必须保留，后续 A/B 闭环、GPU family 或强重建工作都不能恢复旧 emitter
authority。

### 9.2 当前仍需完成的 A/B 主线

Artifact-neutral 主体切换、caller closure、旧 RVV provider/protocol 物理清场与相应本地
回归已经完成；剩余工作不能被误写成 construction 仍在 EmitC driver，也不能因此跳过
RISC-V 旗舰 realization 的方法闭环：

1. `P=(S,g,ω)` ownership 已可枚举，但真实 code-affecting `g/c/ω` 与 mechanism/formula
   仍分散在若干 family leaf、front door、schedule 与 conversion 中；
2. `ConstructedWeak` final leaf 仍需多 topology 的 delete-leaf reconstruction 才能升级 strong
   construction；
3. Scalar 等路径若 formula 只生成参数字典、而完整算法仍住 artifact leaf，仍须提升为
   family-local typed mechanism/body；确定性 typed root 不能替代这项证明；
4. formula causal fan-out、capability counterfactual、analytic-only 与 bounded residual 的
   作用边界仍需直接实验；
5. 重构后的 current artifact 必须重新经过 correctness、deployed symbol、strong opponent 与
   e2e paired regression，不能继承历史 leaf 的性能结论。

因此现在也不能把 GPU 简化为：

- 伪装成 EmitC backend；或
- 绕开 current artifact-neutral construction lifecycle；或
- 在新 emitter 中重新解释 source metadata。

三种都违反 V2 第一性原理。

---

## 10. GPU 前置的 artifact-neutral 横向重构（已完成）

首个 task 定义为 **artifact-neutral family construction rebase**，而不是 GPU
implementation；主体提交已完成结构切换、caller closure 与完整回归。本节保留其架构结果，
不改写为 GPU 完成声明。

### 10.1 重构目标

建立以下唯一主链：

```text
canonical problem/source entry
  → typed target/family binding
  → artifact-neutral family construction lifecycle
  → legality + optional bounded selection
  → construction-qualified final typed body
  → family artifact driver
       ├─ current EmitC/C/object
       └─ future NVVM/ROCDL/other
```

### 10.2 横向范围

不是先迁一个 family。最终合入必须同时覆盖：

- RVV、IME、Scalar、Demo、Toy、Template、TensorExtLite；
- Offload 的 explicit unsupported；
- 所有 registered source front door；
- public pass、direct conversion、materialization、translate、artifact export；
- formula catalog 与 construction-entry inventory；
- selected-body realization、ABI 与 artifact candidate handoff。

### 10.3 关键动作

1. 定义可枚举的 typed problem/source contract，明确 `S/g/ω` owner；
2. 把 target/family binding 放在 construction 前，并使其不依赖 EmitC registry；
3. 让 family construction 成为 plugin/family 的必实现 artifact-neutral lifecycle；
4. 删除 `ExtensionPlugin` 的隐式成功 construction；unsupported family 显式拒绝；
5. 将 construction 与 qualification 从 `TypedBackendEmissionDriver` 的 EmitC 专属职责中
   横向迁出；EmitC driver 只消费 construction-qualified final body；
6. 让 public/direct/translate/artifact caller 先调用同一 family construction seam，再进入
   对应 artifact driver；
7. 保持当前 EmitC 路径的唯一 conversion harness 与 fail-closed 性质；
8. 从 common contract 移除“成功必须产生 `emitc.func`”之类 artifact-specific 假设；该门
   留在 EmitC artifact backend 内；
9. 删除旧双入口、默认 no-op、兼容 adapter 与 emitter-side recovery；
10. 用 current families 做全路径回归，不创建 GPU dialect、GPU plan、CUDA/ROCm runtime。

### 10.4 这个 task 不做什么

- 不实现 NVIDIA/AMD plugin；
- 不新增 GPU capability 字段；
- 不创建 `weft_gpu` dialect；
- 不生成 `gpu.launch`、NVVM、PTX 或 cubin；
- 不做 GPU benchmark；
- 不把 current RISC-V performance 重新归因；
- 不建立 universal verifier/provider/Formula IR；
- 不改变 `flat_*`、六律、原主公式和 thin selector。

### 10.5 完成判据

本次 task 已同时满足：

- 每个 production source entry 都能说明自己的 `S/g/ω`；
- target/family binding 在 construction 前完成，common artifact code 不按 family 名分支；
- 每个 live family 有唯一 artifact-neutral construction owner；
- construction completion 不依赖 EmitC 类型或 `emitc.func`；
- EmitC registry 只代表一个 artifact class，不代表所有 family construction；
- 所有 current paths 从 source/direct input 到 artifact 仍共享 construction-before-emission；
- missing/ambiguous/mixed/unconstructed/unsupported 全部 fail closed；
- quantize/dequantize 对称性和 RVV `flat_*` plan authority 保持；
- current full test、catalog/registry、source/direct/artifact negative tests 全绿；
- 没有 production compatibility middle path。

完成该 task 只证明“系统已经具备正确接入 GPU family 的结构”，不证明 GPU 已支持，也不
表示科研主线应立即转向 GPU。当前 A/B 横向 task 先闭合 RISC-V 旗舰 realization 的执行
知识因式分解、强重建、formula causality、current artifact correctness 与真实性能。
其中 deterministic family 的 route/manifest qualification 已在该 task 的第一项横向清理中
退役；这项完成不代表其余 leaf 已 strong，也不授权恢复新的通用 verifier/provider。

---

## 11. A/B 闭环之后的 GPU realization 方向

在 artifact-neutral rebase 与 RISC-V 旗舰 A/B 闭环完成后，GPU 才按真正 family 接入：

1. NVIDIA capability profiles；
2. source/problem applicability；
3. GPU mechanisms 与 formula；
4. resource legality；
5. final typed GPU body；
6. NVGPU/NVVM artifact backend；
7. CUDA launch ABI/runtime；
8. correctness、resource、strong opponent 与真实 workload evidence。

实施时可以先覆盖 H100/5090D 上已有 operator domain 中的代表性 decode、prefill、
dequantize、quantized dot/matvec/matmul 或 reduction。实现顺序是工程计划，不是系统定义的
上界；也不能为了“最小 demo”把 GPU 降级成 external-kernel registry。

---

## 12. Evaluation 的长期组织

### 12.1 C1：扩展局部性

检查新增 NVIDIA-GPU family 与 H100→5090D profile 时：

- source/problem semantics 是否复用；
- RVV/IME/Scalar 是否无需修改；
- core/common 是否无 GPU/device-name branch；
- 变化是否集中在 GPU capability、mechanism、formula、body、artifact/runtime；
- 新 profile 是否主要表现为 capability 与 family-local规则变化，而非复制完整 compiler。

### 12.2 C2：construction 因果

对同一 operator 分离观察 fixed/default、mechanism+fixed plan、analytic formula 与
analytic+qualified measurement。记录 candidate/legal set、selected plan、final body、
resource 与 artifact，证明性能决定来自 compiler-owned knowledge 而非 source 已带 schedule。

### 12.3 C3：跨范式 realization

跨范式不直接比较 RVV 与 GPU 的绝对延迟，而比较同一方法在两个 execution paradigms 的
成立方式：

| 维度 | RISC-V | GPU |
|---|---|---|
| problem contract | `S/g/ω` | `S/g/ω` |
| capability | RVV/IME/Scalar | NVIDIA/AMD profile |
| mechanism | vector/matrix/scalar | SIMT/MMA/shared memory |
| legality | VLEN/vreg/extension | register/shared memory/MMA |
| typed body | RVV/IME/Scalar | future GPU body |
| artifact | native C/object | device module/cubin/hsaco |
| runtime | native ABI | launch ABI |
| opponent | llama.cpp/hand kernel | Triton/CUTLASS/vendor/library |

所有性能主张仍需要对应真硬件证据。架构可表达性、compile-lit 与 catalog 完整性不能替代
GPU correctness/performance。

---

## 13. 禁止的错误方向

1. 在 current EmitC registry 里加一个名为 GPU 的 emitter；
2. 把 RVV typed body 或 `flat_*` plan 翻译成 GPU；
3. 用一个含所有 family optional 字段的 capability/plan；
4. 让 artifact lowerer 重新选择 tile、LMUL、warp、stage、MMA 或 mechanism；
5. 把 H100/5090D device name 写入 core/common decision；
6. 只登记外部 CUDA kernel 名便声称 automatic GPU compiler；
7. source descriptor 携带完整 GPU schedule 或 kernel template；
8. measurement 创造 algorithm、candidate 或 legality；
9. 为 GPU 新增通用高层 tensor/tile IR；
10. 在没有 transfer/observer/runtime 证据时提前做 CPU/GPU 跨设备 selector；
11. 用架构目标冒充当前实现；
12. 借 V2 恢复 compatibility bridge、旧 direct emitter 或第二 compute authority。

---

## 14. Spec 迁移原则

### Canon

- 系统类别明确为 post-graph/pre-schedule automatic operator compiler；
- RISC-V 是当前旗舰 realization；
- GPU 是明确的第二 execution paradigm 目标，而非当前完成事实；
- C3 调整为 cross-paradigm realization；
- 两柱、六律和主公式不变。

### Architecture

- 增加 canonical execution problem 与 target/family binding 正本；
- construction contract 与 artifact contract 分离；
- source body 与 family execution body 分层；
- 当前 EmitC 详细规则保留为 RISC-V artifact realization；
- GPU family architecture 作为目标态，不混入当前 family census。

### Measurement / Evidence

统一治理原则，保留 family-specific schema：

- 统一 correctness、lineage、qualification、freshness、provenance；
- RISC-V 保持 board/engine/regime；
- GPU 未来增加 device/architecture/launch/resource/opponent；
- 不强迫两类硬件使用同一物理 row schema。

### Issues / Tasks

- ISSUE-131 保留 rebase 前 construction lifecycle 被 EmitC 绑住的历史问题，当前已关闭；
- 第一个 task 已完成 artifact-neutral horizontal rebase；
- 当前 task 是 executable-knowledge A/B horizontal closure，不实现 GPU；
- 当前 task 已完成 RVV exact-body/runtime-control 与旧 provider/protocol 清场这一结构阶段，
  但仍保持 `in_progress`；
- GPU implementation 仍须之后另建，不作为 A/B task 的隐藏子项。

---

## 15. V2 的正式定位

### 中文

> Weft 是一个面向碎片化硬件生态的、能力驱动的 MLIR 自动算子到 kernel 编译器。它在
> 图级处理之后接收语义充分但尚未决定执行映射的 operator problem，将稳定的算子与数据
> 表示同 family-local 的目标能力、执行机制和可执行专家知识分离，再在编译时构造成
> target-specialized typed execution body，并由 family artifact/runtime 机械物化。RISC-V
> 是当前最完整的旗舰 realization；GPU 是 V2 明确引入的第二执行范式目标。

### English

> **Weft is a capability-driven, MLIR-based automatic operator-to-kernel compiler
> for extensible high-performance specialization across fragmented hardware
> ecosystems. It accepts a semantically complete but execution-undetermined
> operator problem after graph-level compilation, then combines stable operator
> and representation semantics with family-local target capabilities and
> executable expert knowledge to construct target-specialized typed execution
> bodies and mechanically lower them into family artifacts. RISC-V is the
> flagship realization, while GPU is the second execution paradigm introduced
> by the V2 architecture.**

最终总纲是：

\[
\boxed{
\text{稳定的执行问题窄腰}
+
\text{family-local capability 与可执行知识}
+
\text{artifact-neutral construction}
=
\text{可扩展且高性能的跨范式专化}
}
\]

不是：

```text
RISC-V compiler + GPU emitter
```

而是：

```text
one construction philosophy
├─ RISC-V flagship realization
└─ future GPU construction families
```
