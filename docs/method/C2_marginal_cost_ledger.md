# C3′ 模式库构造经济学 ledger — block-quant decode 格式 typed 构造

> **贡献归属(2026-07-10 正名)**:本 ledger 属 **C3′(能力键控优化模式库)的构造经济学 /
> 成熟编译器覆盖轴**,量的是 **decode 格式**(q4_K/q5_K/…/iq*)从 monolith→constructed 的
> **构造边际成本**。**这不是 C2**:C2 = 独立 **extension family**(RVV/IME/scalar/zvfh)的
> 逐家族接入成本,锚 = IME ≈2484 行(见 `experiments/active/visibility/T2-ledger-anchor.md`)。**[LED-2] 第二点已落**(X-SCALAR 家族#3 = 1501 raw / 1148 cloc-approx,见文末「附录:C2 extension-family 接入成本」),但**仍非完整曲线**(2 点、缺 ≥1 点;zvfh 子扩展点未落)。
> decode 格式的能力谓词闭包全 ∩ rvv.*≠∅,按 core-invariants [F-6] 不是 independent family,不进 C2 分母。
> **数值一个不改,仅正名贡献标签 + "家族"→"decode 格式/op 谱"用词。**

**这是 C3′(能力键控优化模式库 → 模式库构造经济学)的代码实证台账。** 每格从 monolith/dispatch-wired
提升到 constructed(typed 模式库原语)的构造边际成本逐格记录。**递变的构造成本序列
(建原语一次贵 → 复用参数化便宜 → 泛化新轴有界成本 → 摊销)= 论文级 C3′ 构造谱系。**

口径:Δhand-LOC = 编译器源码(`lib/`+`include/`)net,在该格 flip commit 处量;
"tier" 由 [net-new capability vs reuse] 定,非单看 LOC 符号。byte-exact 是硬门(不可绕),
正-LOC 在需要新能力时授权(step-not-slope)。

---

## Phase 1 — Flat block-dot decode 格式(single-block)

| 格式 | 里程碑 | net-new | reuse | tier |
|---|---|---|---|---|
| q8_0 | 多会话 scaffold(step1-4)+ emit(step5) | 建 `typed_flat_block_dot_loop_body` op + 循环层 from scratch | — (首建) | **HIGH** 建原语 |
| q4_0 | 1 workflow | packed-i4 offset-binary decode brick | flat loop op | MEDIUM |
| q4_1 | 1 workflow | unsigned-nibble + min brick(net-new) | flat loop op | MEDIUM |
| q5_0 | 1 workflow | five-bit qh source brick(net-new) | flat loop op | MEDIUM |
| q5_1 | 中档 | q5_0 qh ∪ q4_1 min 并集,复用两 op | flat loop op + 2 brick | **LOW** |

flat decode 格式 flip 后【整体退役】monolith:commit `1185729a`,**Δhand-LOC −1797**(单一 typed 表征)。

## Phase 2 — Super-block K-quant decode 格式(当前 C3′ 构造序列)

| 格式 | commit | 里程碑 | net-new capability | reuse | ΔC_construct | Δhand-LOC(flip 处) | tier |
|---|---|---|---|---|---|---|---|
| **q4_K** | 0d68f2eb→81b61908→82589f30 | **3** | 建 super-block 双累加器 loop op + emitter + front-door **from scratch** + 6 fine brick | — (首个 super-block) | 8→9 | +935 建机器(scaffold+emit),flip-retire commit −237 | **HIGH** 建原语 |
| **q5_K** | 386d3d4d | **1** | qh optional attr(1 个)+ emitter switch + 砖门格式化 {144,176} | super-block op + 全 6 砖 + fold + 8 CORE helper | 9→10 | **−374** | **LOW** ~免费参数复用 |
| **q6_K** | a489e950→e0034acc | **2** | 单累加器 fold_model-键控 arity 泛化 loop op + no-min fold_model + 单累加器 emit + 新 front-door 链 | super-block op(泛化)+ aux32 core + q4_K fold 语义 | 10→11 | **+213**(retire −335 被新-arity +548 盖过) | **MEDIUM** 泛化新轴 |
| **q2_K** | 37e589f4→e96c113e | **2** | net-new q2_K 整数核(2-bit unpack + plain-nibble scale/min + 16×16 标量 dot)+ **第三累加器 arity(标量)** scalar_scale_min | super-block op(arity 泛化)+ 标量 fold 机制 | 11→12 | **+388** | **HIGH** 结构距离远(近 q4_K 首建;est 曾误判 LOW-MEDIUM) |
| **q3_K** | afa03ce1 | **1** | net-new q3_K decode brick(3-bit hmask 条件减 + 6-bit signed packed scale)| **q6_K 单向量 arity + no-min fold + emit 路全部** + 16×16 | 12→13 | **+16** | **LOW** 结构已被 q6_K 覆盖(表面难=hmask,但结构=symmetric no-min 同 q6_K;est 曾误判 HIGH) |

## 序列解读(C3′ 构造经济学 headline)

**K-quant sprint 完成(C_construct 8→13,5 超块全 constructed):**
**q4_K HIGH(建 super-block 原语)→ q5_K LOW(~免费参数:qh attr,−374)→ q6_K MEDIUM
(泛化新轴:累加器 arity 单向量,+213)→ q2_K HIGH(净新整数核 + 第三 arity 标量,+388)
→ q3_K LOW(复用 q6_K arity,只加 hmask decode brick,+16)。**

**★★ 最锐利的 C3′ 构造发现:构造成本 ∝ 结构距离,【非表面复杂度】。** q3_K 表面上最难
(hmask 符号翻、decode 格式里唯一未拆的最粗 monolith),实测【最便宜】(+16,K-quant 最小边际)
——因其【结构】(symmetric no-min、单向量)已被 q6_K 覆盖,hmask 差异只是局部小 decode
brick。反之 q2_K 表面不起眼但结构真远(2-bit + plain-nibble + 标量 fold,需第三 arity)
→ 最贵(+388)。**"最难看的格式最便宜、不起眼的格式最贵"精确证伪"成本∝表面难度",
证实"成本∝到已覆盖原语空间的结构距离"。** 这是 C3′ 构造经济学的干净可引用刻画。

**规律 = 边际成本 ∝ 与已覆盖原语空间的【结构距离】(非单调递减):**
①建一个【新原语】贵(q4_K 3 里程碑从零);②用【参数】复用它 ~免费甚至净负(q5_K 只加
qh attr → −374,结构距离≈0);③沿【新轴泛化】原语(累加器 arity 向量)有【实但有界】
成本(q6_K +213,复用整数核);④结构【真正远】的格式(q2_K:2-bit + plain-nibble +
16×16 + 标量 fold,与已覆盖的 nibble/bit-dance/向量-fold 全不同轴)成本回到近首建
(+388),因整数核 + 第三 arity 都真新。**这不是回退——是 C3′ 构造经济学的完整刻画:复用便宜、
泛化有界、结构远则贵;贵在【一次】,该格的 16×16 reduce + 2-bit 机制随后摊销到 q3_K。**

原语 `typed_super_block_block_dot_loop_body` 现参数化于三轴:**stride(格式)/ qh 平面
(q5_K 1-bit)/ 累加器 arity·fold 形态(q6_K 单/双)**。每个新格式要么复用某轴参数(便宜),
要么加一轴(有界成本、随后摊销)。**这正是 [K-2] 模式库 + C3′:构造代价有界且随原语
参数空间饱和而递减。** q6_K 的 +213 不是回退——是"泛化新轴"这一 tier 的诚实成本,
与 q5_K 的 −374 一起构成完整的边际成本谱。

*关联:六态 coverage(schema/coverage-sixstate.v1.json,C_construct 计数)、
T7 覆盖率 Fig.1、full-refactor program 台账。数值/perf 主张见 T3/T8(K-quant perf
= 骨架收口战役,排队第二)。*

---

## 第二谱:decode_model=lookup(IQ/码本 decode 格式)+ repack —— 跨轴复证同一 C3′ 构造规律(2026-07-06)

K-quant 谱证了 super-block 算术 decode 谱系内的构造边际成本律。第二谱在【全新 decode 轴 + 全新 loop-shape】上复证:**建原语贵、参数复用便宜**跨 decode 谱系成立。

| 格式 | 里程碑 | 新增 vs 复用 | 边际成本 tier |
|---|---|---|---|
| **q4_0 repack GEVM** | 7(loop-shape/整数核/fold/全臂/front-door/退役/flip) | **第三个 distinct typed-loop shape**(per-strip lane-wise f32 VECTOR 累加器,flat=scalar/super-block=horizontal 都表达不了)=greenfield 原语 | 最贵(新 loop-shape 从零,repack 族基础) |
| **iq4_nl**(flat codebook) | 3(M1 byte-exact/M2M3 flip) | 复用 flat block-dot 族外壳 + codebook 砖(CodebookTableBroadcast/GatherXI8Product 已存);net-new=emitter codebook sub-case + front-door 分支 | 中(flat 族在、codebook decode 新轴) |
| **iq1_s**(super-block 2048-grid vluxei16) | 3 + foundation(M1 scaffold/M2 byte-exact/M3 flip) | **新 grid-core 砖(vluxei16 2048-ternary)+ 新 fold_model scalar_delta_grid + super-block grid loop-body branch + selector**=grid 族 greenfield 基础 | 贵(grid 基础从零,net −475 LOC 退役 monolith) |
| **iq1_m**(grid+delta) | **1 workflow** | ★REUSE 整个 iq1_s scalar-delta-grid scaffold 零边际(loop op/fold/verifier/selector/emitter dispatch/front-door skeleton);唯一 net-new=distinct 核砖 + body code-move | **最便宜(参数复用、结构距离≈0)** |

**★跨族复证**:iq1_s→iq1_m = q4_K→q5_K 的精确同构。**首个 grid-family 成员付 greenfield 基础(iq1_s:新砖+fold+branch+selector),次个成员(iq1_m,同 2048-grid 只 re-param 核)一个 workflow 就 flip=零边际复用。** 证 C3′ 构造经济学"成本∝结构距离到已覆盖原语空间"跨【三个不同 decode/loop-shape 谱系】(K-quant 算术 super-block / IQ grid super-block / repack GEVM)成立——不是 K-quant 偶然,是模式库的普适经济学。**预测**:iq2(+sign-plane 砖变体)/iq3(+i32 grid 砖变体)成本 = iq1_m tier(便宜、复用 grid scaffold)+ 一个 brick-variant 增量(有界,像 q6_K 的 +213 泛化 tier);iq2/iq3 用【不同 grid 内容/宽】,故比 iq1_m(同 grid)略贵、但远低于 iq1_s greenfield。

*关联:C_construct=17(session 起 13:q4_0 repack=14/iq4_nl=15/iq1_s=16/iq1_m=17);travel-decision-ledger.md F8/F11/F14/F15。*

---

## 附录:C2 extension-family 接入成本(真 C2 · [LED-2],**非** C3′ 构造经济学)

> **区分**:上文全部是 C3′ decode 格式**构造**边际成本(闭包 ∩ rvv.*≠∅,同一 RVV 家族内的
> op 谱扩展)。**本附录才是 C2** = 独立 **extension family** 的**逐家族接入**成本(每家族一个数据点,
> 锚 = IME 首接入 slice)。行数取 `family_ledger.py`(stdlib recompute,`schema/family-dirs.v1.json`
> 驱动),口径:code_dirs raw wc-l 排除 tests + CMakeLists;cloc-approx = block-comment-stripped 近似。

| # | 家族 | code_LOC raw / cloc-approx | test_LOC(单列) | family_kind | 边际刻画 |
|---|---|---|---|---|---|
| 1（锚）| **IME**(Spacemit X60 IME 矩阵扩展) | **2484 / ~1866** | 1003 | integrated-attached extension-family | 首接入 slice = C2 绝对锚（[C2-1]/[C2-4] 可复现，founding-slice @git `54465ee7`；`T2-ledger-anchor.md` 双数口径）。**★锚 drift 已查（先查后改·2026-07-11 用户裁）：`--family IME` 现报 raw 5153 = 同 7 文件从 founding-slice(2484)长起(delta +2669)= C3′ 家族内扩展(更多 MMA op 变体+发射路)、非首接入成本、非计数口径差(同工具同 raw-wc-l)。C2 边际曲线锚永远钉 founding-slice 2484;live 5153 是当前家族体积(informational)。禁静默替换 2484。** |
| 2 | **X-SCALAR**（向量缺席独立标量家族 #3） | **1501 / 1148** | 1401 | independent extension-family（闭包 ∩ rvv.* = ∅） | [LED-2] 第二点。方向递减（1501 < 2484 ≈ 60% #2）；1148 = **[C2-1] 轨二 independent 家族的诚实 emitter 价格（非超标）**——早先「<300」是 *integrated 子扩展* 预估被误挂到 *independent 家族*，双轨修订（2026-07-11 用户裁）已撤销该单点目标与「超标」定性。 |

**★C2 第二点的诚实刻画（[NG-4]，不吹成曲线）**:

1. **两点 ≠ 单调递减曲线**。IME 2484 → X-SCALAR 1501 只是 2 个真数据点(仍 <3),**缺 ≥1 点**
   (zvfh 向量族**子扩展**粒度点未落 —— 且它是 *sub-extension* 粒度,与家族级点不同 KIND)。
   「1/≥3 曲线缺失」标注继续保留:landed X-SCALAR 把「1 点」变「2 点」,**未成完整曲线**。

2. **`<300` = 双轨修订【已裁·2026-07-11 用户】：1148 是独立家族的诚实价格·符合双轨预期(非超标)。**
   早先 canon [C2-1]「标量家族 <300 行」是对 *integrated 子扩展* 的量级预估被**误挂到 independent 家族**。
   用户裁决 [C2-1] 改为双轨：**integrated 子扩展 <300(zvfh 将验证) / independent family = 真实 emitter 成本
   (X-SCALAR 1148 cloc-approx 为锚)**。"超标"定性**撤销**；本 ledger 登记的 1148/1501 是 factual 实测,
   现由双轨预期正确刻画(不再是"未 met 的 <300")。数值一个不改,仅撤销定性框架。

3. **结构根因(比 <300 直觉更精确的 C2 主张)= [F-6] 独立性禁止复用抬高了独立家族成本。**
   X-SCALAR 主体是**净新 pure-C emitter**(`ScalarBackendEmissionDriver.cpp` = 878 raw,单文件最大头),
   **不能复用 RVV EmitC 发射机器**(复用会把 rvv.* 拉进能力闭包、破坏 F-6 独立性)。这与
   「integrated 子扩展(如 zvfh,复用宿主家族机器)= 便宜」形成对比。
   → **C2 曲线的正确刻画不是「随家族序号单调递减」,而是「integrated 子扩展廉价 / independent
   家族付真 emitter 成本」([L-2] integrated vs independent-attached 术语的经济学体现)**。这比
   原「<300 / 1-of-10-of-#2」直觉**更精确、更可辩护**。审稿人预答([C2-5])的「1/10 of #2」对
   independent 家族**不成立**（对 integrated 子扩展成立）。**★此精确版已 codify 为 canon [C2-1]
   双轨律（2026-07-11 用户裁），[C2-1]/[C2-5]/[X-SCALAR] 行同批修订、"超标"定性撤销。**

*证据/复算:`python3 .trellis/scripts/family_ledger.py report --family Scalar`(raw 1501 / cloc-approx 1148 /
test 1401);`experiments/active/result-tables/T2_C2_ledger_marginal_cost.csv` seq 14;
排期报告 `docs/reports/2026-07-11-X-SCALAR-AUDIT后续-排期报告.md` 甲.5(hand-est ~1131);
`schema/family-dirs.v1.json` Scalar 条目。快照 HEAD=2d2d8a1b。*
