# 06｜QIGen 全文差分与最终再收敛

> **状态**：2026-07-21 全文复审；关闭前四阶段中“仅看摘要与 artifact”的不确定性。
>
> **边界**：本文件不修改两柱、六律、canon 或用户确认的主公式。它只修正 novelty 边界、q5_1 的证据角色，以及后续实现与评价应真正回答的问题。
>
> **当前时态**：以下“应做到”“需要证明”均是目标态。当前 Weft-RV 尚未普遍做到删除完整逐点实现后，仅由 `g/c/ω + mechanisms + formula` 重建实例。

## 0. 再审裁决

取得并逐页审查 [QIGen 正式全文](../QIGen_A_Kernel_Generator_for_Inference_on_Nonuniformly_Quantized_Large_Language_Models.pdf) 后，原审计需要再收紧一层：

1. **QIGen 已经跨过“非均匀 group 能否由生成器组合成 layer kernel”这条边界。** 它不是按完整矩阵名字选回一份 monolithic leaf，而是枚举 `(M,K,g,b)`，生成对应 GEVM/GEPM microkernel，再按每层 group 序列组合。
2. **QIGen 仍保留更深一层的完整语义 leaf。** 其 microkernel generator 对 2/3/4/5/6/8 位分别用显式 `switch(bits)` 编码完整 unpack、shift、mask、dot 过程。它因式分解了 group/microkernel composition，但没有把 bit-decode microkernel 本身继续拆成正交语义事实与可复用机制。
3. **“Execution Knowledge Factorization”不能作为独立知识模型 novelty。** `M/F/B/R` 继续只是内部解释语言；生成、组合、解析模型加有限搜索、按 CPU 特化和跨 ISA 移植，都已有直接先例。
4. **Weft 只剩一个更窄、且仍需代码证明的系统假说：** 能否把因式分解推进到 QIGen microkernel leaf 以下，覆盖不止 `(g,b,s,z)` 仿射 group 的异质 GGML 语义拓扑，并在完整 post-MLIR execution-layer compiler 中由能力与上下文构造 typed body、限制经验残差权限。
5. **q5_1 不再承担主要科研差额。** 它仍是合理的第一项代码重构和 point-authority-erasure 冒烟证人，但 QIGen 已经覆盖 5-bit 仿射 group、GEVM/GEPM 和 group composition；因此 q5_1 单独成功也不能支撑论文 novelty。

编辑裁决仍是 **Major Revision，方向保留，泛化 novelty 进一步降级**。变化的是对先例的认识，不是重新定义两柱、六律或公式。

## 1. 正文实际描述的 QIGen

### 1.1 输入、构造与输出链

QIGen 的正文给出了一条真实 generator 链，而不是简单 dispatch：

```text
ExLlamaV2 nonuniformly quantized model
  + target CPU characteristics
  → per-layer group sequence (g_i, b_i)
  → quantizer-output translation to W / S / Z
  → enumerate unique (M, K, g, b) tuples
  → analytical cache-blocking model
  → bounded grid search for register/unroll parameters
  → generate one LLVM GEVM/GEPM microkernel per tuple
  → pack matrices in a static blocked/Z-order layout
  → compose layer-specific kernels from the group sequence
  → JIT compile and assign kernels to layers
```

其中：

- `W/S/Z` 分别承载量化权重、scale 与 zero；quantizer 到该内部形式的 translation layer 是手工实现的；
- generation 覆盖 `n=1` 的 GEVM，以及通常 `n≤8` 的小批量 GEPM/GEMM；
- 相邻且 `(g,b)` 相同的 group 可以合并；
- cache model 先决定主要 blocking，再对小型参数空间做 grid search；论文明确说大模型 tuning 仍可能持续数日；
- `(M,K,g,b) → microkernel` 映射被保存并供 layer generator 复用。

因此，QIGen 已经直接覆盖：

- known workload 中 unique tuple 的去重生成；
- 非均匀 group 序列的 layer-specific composition；
- GEVM 与小批量 GEPM 两种 regime；
- 解析模型与经验搜索的结合；
- storage packing 与生成 kernel 的协同设计；
- 由量化模型和目标 CPU 特征产生专化 LLVM kernel。

### 1.2 它的真实 leaf boundary

QIGen 的一般 packing 可以按 bit width 计算布局，但生成 microkernel 的核心过程仍按 bitwidth 分支：

```text
switch(bits):
  case 2: complete 2-bit unpack / mask / dot body
  case 3: complete 3-bit unpack / mask / dot body
  case 4: complete 4-bit unpack / mask / dot body
  case 5: complete 5-bit unpack / mask / dot body
  case 6: complete 6-bit unpack / mask / dot body
  case 8: complete 8-bit body
```

这不是对 QIGen 的贬低，而是精确定位其因式分解粒度：

```text
QIGen 已分解：整层矩阵 → group tuples → 可复用 microkernels → layer composition
QIGen 未展示：bit/scale/layout 语义事实 → 正交 mechanisms → 无完整 per-bit body 的 microkernel construction
```

此外，正文中的实际 domain 仍较集中：

- group size 要满足 `g mod 32 = 0`，实际量化器使用 32/64/128；
- 量化模型以每 group 的 bitwidth、scale 和 zero 为核心，属于仿射 group quantization；
- 论文主 kernel 是非均匀权重下的 GEVM/GEPM；
- 它没有在正文中展示 KQuant 式量化 scale/min superblock、codebook/grid、ternary、q1 等异质编码拓扑的统一构造。

最后一项只能用来限定本次直接比较，不能反向声称 QIGen 永远不能扩展到这些格式。

### 1.3 retargeting 与评价边界

QIGen 将大量 architecture-neutral 工作交给 LLVM。正文还报告了 x86 到 ARM 的实现移植：因为两端 signed/unsigned dot 语义不同，需要手工平移输入并预计算权重列和，作者称单人不足一天完成。

必须准确区分：

- 这是 **ARM 实现/移植证据**；
- 正文性能实验只在 Intel Xeon Silver 4410Y 上进行，**没有 ARM 性能结果**；
- 评价以 kernel 与 layer latency 为主，明确排除完整端到端 inference 实现细节；
- TPOT/TTFT 是由 layer 结果推算的 simulated/projected 指标，不是真实 runtime token 运行。

所以 Weft 不能把“可 retarget”本身当 novelty；若最终给出真实 llama.cpp/ggml consumer 路径、真实 token correctness 与 TTFT/TPOT，那是更强的部署证据，而不是自动产生新的理论贡献。

## 2. 与 Weft-RV 的精确差分

| 维度 | QIGen 正文已经做到 | Weft-RV 当前事实 | Weft-RV 需要做到的目标 |
|---|---|---|---|
| 输入知识 | `(M,K,g,b)`、scale/zero、CPU 特征、量化器 translation | typed facts/capability/context 分散存在 | 让独立 `g/c/ω` 真正决定 mechanism 与 typed body |
| 构造粒度 | 生成 per-tuple microkernel，再按 group 组合 layer | 多条路径仍保留 per-format/per-regime builder | 继续拆掉完整 per-format/per-bit semantic body authority |
| 编码域 | 非均匀仿射 group，显式支持 2/3/4/5/6/8 bit | GGML flat、KQuant、grid/codebook、ternary、q1 等资产 | 以少量正交 mechanisms 覆盖多种真实编码拓扑 |
| 操作域 | GEVM 与小批量 GEPM | dequant、contraction/repack、多个 backend 路径存在但成熟度不一 | 在语义不同的 operation/family 上展示同一因果纪律 |
| 能力/合法性 | CPU 参数、cache/vector 约束承重；正文未把它们写成独立统一 boundary 模型 | capability 与 legality 已有局部真实作用，也有重复 authority | 固定 `g/ω` 改 `c` 时，合法候选与 typed plan 可预测变化 |
| 经验知识 | 每个所需 unique tuple 经模型加搜索调优 | 已有 analytic 与 measurement 基础 | 证明 analytic-only 正确，measurement 只作稀疏合法 winner 修正 |
| 生成证据 | known tuples 与 arbitrary supported group sequence 可生成 | 尚无严格 point-authority erasure 后的 family reconstruction | 删除完整 authority 后仍由 facts/mechanisms/formula 构造 |
| 系统证据 | Intel kernel/layer，TPOT/TTFT 为模拟 | 有 runtime、bench 与真板基础资产 | 重构后补真实 production path、硬件与端到端结果 |

这里最重要的不是列出 QIGen “没有”的名词，而是避免三种错误比较：

1. QIGen 没用 `M/F/B/R` 命名，不代表对应功能不存在；
2. Weft 有 typed dialect/provider/verifier，不代表构造因果关系更强；
3. Weft 支持 RISC-V/IME/Scalar，不代表自动比 QIGen 的 CPU retargeting 更有 novelty。

## 3. novelty 的再次收敛

### 3.1 已经不能使用的宽泛主张

除前四阶段已排除的内容外，全文现在明确排除：

- “首次把非均匀量化 groups 组合成专化 CPU kernel”；
- “首次对量化 LLM 同时生成 GEMV/GEMM regime”；
- “首次由已知格式组合跨点生成，而不是完整矩阵 leaf lookup”；
- “首次以解析模型缩小搜索，再用有限经验调优”；
- “首次让一个量化 kernel generator 在 CPU ISA 间移植”。

`knowledge factorization`、`mechanism composition`、`bounded search` 和 `retargeting` 都只能描述系统设计，不能单独充当贡献标题。

### 3.2 仍值得检验的最窄假说

当前更稳健的目标表述是：

> Weft-RV 研究能否在 extensible post-MLIR execution layer 中，把异质 GGML block-quant 的编码拓扑、scale/bias/layout 语义、目标 capability 与有限 static context 分解为独立 typed facts 和可复用 mechanisms，由既定解析构造公式在显式合法域内生成多个 operation/backend 的 typed implementation；在删除完整逐点 compute authority 后仍保持正确、可部署且有竞争力，而 measurement 只修正稀疏的合法性能残差。

这与 QIGen 的真正差额不再是“有无 generator”，而是三项同时成立：

1. **更深粒度的语义因式分解**：低于完整 per-bit/per-format microkernel leaf；
2. **更异质的知识域**：不只覆盖仿射 `(g,b,s,z)` group，而覆盖多种真实 GGML encoding/scale topology；
3. **更强的因果与系统证据**：authority erasure、跨 operation/backend、受限 residual 和真实部署共同成立。

即使三项都做到，最终也应作为一个窄的 compiler systems contribution 与 SPIRAL、Exo、Ladder、QIGen 等直接比较，而不是宣称新的通用编译理论。

## 4. q5_1 的新位置

### 4.1 为什么路线仍从 q5_1 开始

q5_1 仍适合当第一项实现重构，因为它让现有缺口非常可见：

```text
low nibble
+ optional high-bit plane
+ affine scale/offset fold
→ typed GEMV/GEMM body
```

删除 Q51 GEMV/GEMM builders、dispatch、等价 registry/provider row 和 emitter fallback 后，仍由独立 facts 与共享 mechanisms 构造，可以检验：

- 完整 point authority 是否真的被移除；
- construction 是否承重，而不只是换了函数名；
- GEMV/GEMM 是否共享语义机制而非复制 body；
- emitter 是否只机械实现已构造结构；
- analytic-only correctness 是否成立。

这些都是第一轮代码重构必须学会的能力。

### 4.2 为什么它不再是主要科研证人

QIGen 已经支持 5-bit 仿射 group、GEVM/GEPM，并把 group microkernel 组合进 layer。虽然 q5_1 的 GGML 物理布局与 QIGen 内部格式不必相同，但仅证明“low bits + fifth bit + scale/zero 能组合生成”已经不足以形成明显方法差额。

因此 q5_1 的准确角色应改为：

> **首个 point-authority-erasure 实现里程碑与本地结构重建证人。**

它不是：

- 主要 novelty witness；
- unseen generalization 证人；
- 整个异质 block-quant domain 已因式分解的证据；
- 对六律的单点验收。

### 4.3 后续确认性证据需要跨 topology

在 q5_1 跑通之后，论文级证据应覆盖多个语义拓扑，且至少包含 QIGen 仿射 group 模型之外的类型，例如：

- KQuant：量化 scale/min 与 superblock 层级；
- codebook/grid：索引、查表/gather 与 codebook 语义；
- ternary 或 q1：非普通多 bit affine decode；
- 另一个真正独立的 operation 或 backend 因果链。

这不是现在替论文 agent指定最终实验点。它只说明：若所有确认性证据都停留在 flat affine q4/q5 组合，QIGen 全文会让 novelty 差额仍然过薄。

## 5. 对代码路线的影响

QIGen 不推翻 flat-repack 第一刀，但改变这项工作的目的和后续顺序。

### 5.1 第一刀：用 q5_1 学会删除完整 authority

保留既有主公式，把 flat q4/q5 的知识拆到真正独立的轴：

- payload plane topology；
- bit significance、signedness 与 centering policy；
- scale/offset topology；
- block/stride/ABI layout；
- operation regime；
- capability/resource facts。

再由共享 mechanisms 构造 typed body，而不是按格式名选完整 recipe。允许保留不可约 primitive，但不能把“Q51 pipeline”换名成一个新 primitive。

### 5.2 第二刀：穿过 QIGen 的 affine microkernel 边界

q5_1 完成后，不应只继续横向合并更多 flat per-bit builder。需要选择至少一个不同知识拓扑，验证机制词汇是否真能表达：

- 分层 scale/min decode；
- codebook/grid lookup；
- 非标准符号或低熵编码；
- 与 dequant/contraction 不同的 operation semantics。

如果每种 topology 最终仍需要一份完整 family body，那么正确结论是“完成了 flat family 重构”，而不是把局部成功外推成通用 knowledge factorization。

### 5.3 measurement 的真正差额

QIGen 对每个所需 unique tuple 做模型加搜索。因此 Weft 只有在以下事实被量化后，才可说 empirical knowledge 更像 residual：

- 无 measurement row 时仍能构造正确实现；
- row 只在多个已合法候选中选 winner；
- row 不含 body、decode、scale、layout 或 operation script；
- 报告 residual coverage、实际 intervention 和 performance gain；
- product space 扩张时，经验依赖没有随 point 数近似等比例增长。

“搜索空间有限”或“winner table 很小”本身都不足以与 QIGen 区分。

## 6. 评价设计的最终优先级

后续确认性证据按科研辨识力排序：

1. **多 topology 的 point-authority erasure**：不只删除 q5_1 leaf，还要覆盖至少一种非仿射-flat 语义拓扑；
2. **真实 producer 与运行闭环**：从真实 workload/source front door 到 runtime，不由测试直接手写目标 body；
3. **capability counterfactual**：固定 `g/ω` 改变 `c`，合法候选与 typed structure 按公式预测变化；
4. **rule/mechanism causal fan-out**：修改一个共享语义规则，预期实例同步变化，非目标实例不变；
5. **analytic-only 与 residual 干预**：证明 measurement 不承担 compute；
6. **重构后硬件与真实端到端**：kernel attribution、correctness、prefill/decode 和真实 runtime 指标共同报告。

prospective holdout **不是普通 compiler 重构的强制手续**。只有论文要明确主张 unseen/generalization 时，才需要冻结机制/公式后选择未参与规则设计的点；若论文主张只是“在所定义 domain 内消除逐点完整 authority并构造 family”，多 topology 的严格删除与因果实验已经是更直接的主证据。

## 7. 最终编辑结论

QIGen 全文关闭了原审计最大的外部不确定性，也让剩余问题更清楚：

```text
不是：Weft 会不会从量化描述生成并组合 CPU kernel
因为：QIGen 已经会

而是：Weft 能否把异质 GGML 执行语义继续分解到完整 microkernel leaf 以下，
      让既定公式、真实 capability 与共享 mechanisms 在完整 compiler 中
      删除逐点 authority 后仍构造多个 topology/op/backend，
      且 measurement 只修正稀疏性能残差
```

这仍是值得做的研究目标，但它比“Execution Knowledge Factorization”口号窄得多，也必须由代码和实验而非命名成立。

本次再收敛的最终约束是：

- 两柱、六律、主公式不动；
- q5_1 继续作为第一实现里程碑，不作为主要 novelty 证人；
- `M/F/B/R` 保留为职责解释，不升级为理论贡献；
- 后续 spec 只更新 current/target、QIGen 差分和多 topology 证据需求；
- 当前仍写“要做到”，不写成“已经做到”。
