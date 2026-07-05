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
| q2_K | — | PENDING | 2-bit unpack 新;复用 q4_1 ScalePlusMin(有 min)| super-block op(双累加器)+ ? | 11→12 | — | (est LOW-MEDIUM) |
| q3_K | — | PENDING | 6-bit 有符号 packed scale + hmask 符号翻(最难)| super-block op + ? | 12→13 | — | (est HIGH,复用近零) |

## 序列解读(C2 headline)

**q4_K HIGH(建 super-block 原语)→ q5_K LOW(~免费参数:qh attr,−374)→ q6_K MEDIUM
(泛化新轴:累加器 arity,+213)。**

规律:①建一个【新原语】贵(q4_K 3 里程碑从零);②用【参数】复用它 ~免费甚至净负
(q5_K 只加 qh attr → −374);③沿【新轴泛化】原语(累加器 arity)有【实但有界】成本
(q6_K +213),且该泛化随后【摊销】到共享该轴的后续格式。

原语 `typed_super_block_block_dot_loop_body` 现参数化于三轴:**stride(格式)/ qh 平面
(q5_K 1-bit)/ 累加器 arity·fold 形态(q6_K 单/双)**。每个新格式要么复用某轴参数(便宜),
要么加一轴(有界成本、随后摊销)。**这正是 [K-2] 模式库 + C2:泛化代价有界且随原语
参数空间饱和而递减。** q6_K 的 +213 不是回退——是"泛化新轴"这一 tier 的诚实成本,
与 q5_K 的 −374 一起构成完整的边际成本谱。

*关联:六态 coverage(schema/coverage-sixstate.v1.json,C_construct 计数)、
T7 覆盖率 Fig.1、full-refactor program 台账。数值/perf 主张见 T3/T8(K-quant perf
= 骨架收口战役,排队第二)。*
