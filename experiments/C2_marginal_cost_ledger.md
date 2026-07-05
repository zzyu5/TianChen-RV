# C2 边际成本 ledger — block-quant 家族 typed 构造

**这是 C2(泛化代价 → 边际成本规律)的代码实证台账。** 每格从 monolith/dispatch-wired
提升到 constructed(typed 模式库原语)的边际成本逐格记录。**递变的边际成本序列
(建原语一次贵 → 复用参数化便宜 → 泛化新轴有界成本 → 摊销)= 论文级 C2 序列。**

口径:Δhand-LOC = 编译器源码(`lib/`+`include/`)net,在该格 flip commit 处量;
"tier" 由 [net-new capability vs reuse] 定,非单看 LOC 符号。byte-exact 是硬门(不可绕),
正-LOC 在需要新能力时授权(step-not-slope)。

---

## Phase 1 — Flat block-dot 家族(single-block)

| 格式 | 里程碑 | net-new | reuse | tier |
|---|---|---|---|---|
| q8_0 | 多会话 scaffold(step1-4)+ emit(step5) | 建 `typed_flat_block_dot_loop_body` op + 循环层 from scratch | — (首建) | **HIGH** 建原语 |
| q4_0 | 1 workflow | packed-i4 offset-binary decode brick | flat loop op | MEDIUM |
| q4_1 | 1 workflow | unsigned-nibble + min brick(net-new) | flat loop op | MEDIUM |
| q5_0 | 1 workflow | five-bit qh source brick(net-new) | flat loop op | MEDIUM |
| q5_1 | 中档 | q5_0 qh ∪ q4_1 min 并集,复用两 op | flat loop op + 2 brick | **LOW** |

flat 家族 flip 后【整体退役】monolith:commit `1185729a`,**Δhand-LOC −1797**(单一 typed 表征)。

## Phase 2 — Super-block K-quant 家族(当前 C2 序列)

| 格式 | commit | 里程碑 | net-new capability | reuse | ΔC_construct | Δhand-LOC(flip 处) | tier |
|---|---|---|---|---|---|---|---|
| **q4_K** | 0d68f2eb→81b61908→82589f30 | **3** | 建 super-block 双累加器 loop op + emitter + front-door **from scratch** + 6 fine brick | — (首个 super-block) | 8→9 | +935 建机器(scaffold+emit),flip-retire commit −237 | **HIGH** 建原语 |
| **q5_K** | 386d3d4d | **1** | qh optional attr(1 个)+ emitter switch + 砖门格式化 {144,176} | super-block op + 全 6 砖 + fold + 8 CORE helper | 9→10 | **−374** | **LOW** ~免费参数复用 |
| **q6_K** | a489e950→e0034acc | **2** | 单累加器 fold_model-键控 arity 泛化 loop op + no-min fold_model + 单累加器 emit + 新 front-door 链 | super-block op(泛化)+ aux32 core + q4_K fold 语义 | 10→11 | **+213**(retire −335 被新-arity +548 盖过) | **MEDIUM** 泛化新轴 |
| **q2_K** | 37e589f4→e96c113e | **2** | net-new q2_K 整数核(2-bit unpack + plain-nibble scale/min + 16×16 标量 dot)+ **第三累加器 arity(标量)** scalar_scale_min | super-block op(arity 泛化)+ 标量 fold 机制 | 11→12 | **+388** | **HIGH** 结构距离远(近 q4_K 首建;est 曾误判 LOW-MEDIUM) |
| **q3_K** | afa03ce1 | **1** | net-new q3_K decode brick(3-bit hmask 条件减 + 6-bit signed packed scale)| **q6_K 单向量 arity + no-min fold + emit 路全部** + 16×16 | 12→13 | **+16** | **LOW** 结构已被 q6_K 覆盖(表面难=hmask,但结构=symmetric no-min 同 q6_K;est 曾误判 HIGH) |

## 序列解读(C2 headline)

**K-quant sprint 完成(C_construct 8→13,5 超块全 constructed):**
**q4_K HIGH(建 super-block 原语)→ q5_K LOW(~免费参数:qh attr,−374)→ q6_K MEDIUM
(泛化新轴:累加器 arity 单向量,+213)→ q2_K HIGH(净新整数核 + 第三 arity 标量,+388)
→ q3_K LOW(复用 q6_K arity,只加 hmask decode brick,+16)。**

**★★ 最锐利的 C2 发现:边际成本 ∝ 结构距离,【非表面复杂度】。** q3_K 表面上最难
(hmask 符号翻、家族里唯一未拆的最粗 monolith),实测【最便宜】(+16,K-quant 最小边际)
——因其【结构】(symmetric no-min、单向量)已被 q6_K 覆盖,hmask 差异只是局部小 decode
brick。反之 q2_K 表面不起眼但结构真远(2-bit + plain-nibble + 标量 fold,需第三 arity)
→ 最贵(+388)。**"最难看的格式最便宜、不起眼的格式最贵"精确证伪"成本∝表面难度",
证实"成本∝到已覆盖原语空间的结构距离"。** 这是 C2 的干净可引用刻画。

**规律 = 边际成本 ∝ 与已覆盖原语空间的【结构距离】(非单调递减):**
①建一个【新原语】贵(q4_K 3 里程碑从零);②用【参数】复用它 ~免费甚至净负(q5_K 只加
qh attr → −374,结构距离≈0);③沿【新轴泛化】原语(累加器 arity 向量)有【实但有界】
成本(q6_K +213,复用整数核);④结构【真正远】的格式(q2_K:2-bit + plain-nibble +
16×16 + 标量 fold,与已覆盖的 nibble/bit-dance/向量-fold 全不同轴)成本回到近首建
(+388),因整数核 + 第三 arity 都真新。**这不是回退——是 C2 的完整刻画:复用便宜、
泛化有界、结构远则贵;贵在【一次】,该格的 16×16 reduce + 2-bit 机制随后摊销到 q3_K。**

原语 `typed_super_block_block_dot_loop_body` 现参数化于三轴:**stride(格式)/ qh 平面
(q5_K 1-bit)/ 累加器 arity·fold 形态(q6_K 单/双)**。每个新格式要么复用某轴参数(便宜),
要么加一轴(有界成本、随后摊销)。**这正是 [K-2] 模式库 + C2:泛化代价有界且随原语
参数空间饱和而递减。** q6_K 的 +213 不是回退——是"泛化新轴"这一 tier 的诚实成本,
与 q5_K 的 −374 一起构成完整的边际成本谱。

*关联:六态 coverage(schema/coverage-sixstate.v1.json,C_construct 计数)、
T7 覆盖率 Fig.1、full-refactor program 台账。数值/perf 主张见 T3/T8(K-quant perf
= 骨架收口战役,排队第二)。*
