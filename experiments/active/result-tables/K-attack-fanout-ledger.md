# K线 手调攻坚 · 扇出台账（机制入库）

> 《测试与收尾总令-开测篇》§二.4「每攻克一条 = 机制入库 + 扇出台账行 + XFER 正/负预测」·§七终审交付②。
> 源 = 攻坚环解剖阶段（task `07-18-k-attack-dissect`·trellis-check 已核承重数）。
> **状态（2026-07-18）**：机制① re-scope（满展开→VLEN256 宽化）·**iq3_xxs@k1 首个扇出成功 = proven WIN 1.38（板测·byte-exact·trellis-check 复核·但 NOT deployed → [ISSUE-105]）**（task `07-18-k-mech1-vlen-leaf`）；②–⑤ 仍候选未施工。

## 机制队列（按扇出降序 · §二.4 手调列队）

| # | 机制 | 命中格 | XFER 正预测 | XFER 负预测 |
|---|---|---|---|---|
| **①** | ~~VLEN 专化满展开 grid-gather leaf~~ → **【诊断证伪·待 re-scope·ISSUE-102】** 现 leaf **已全直线展开**（3 结构循环外皆展开）⟹「满展开」= no-op（再展开 = re-roll trap）。vsetvli storm(5397·两板逐一相同) + spill(77) 是**板不变静态属性·非门**（同 leaf 同 storm：rvv WIN vs k1 LOSS）。**真 k1 lever = VLEN256 width-widening**（宽 LMUL/大 AVL 填满 256b·对手 vl256 已达·非架构不可达） | **8 格**：iq1_s/iq1_m/iq3_xxs/iq3_s @rvv,k1 | **iq3 系天花板可达（双证）**：iq3_s@rvv **1.34 WIN**·iq3_xxs@rvv **0.95** near-parity（同 leaf 板不变·对手 vl256 亦达此形态）⟹ VLEN256 宽化天花板存在 | **满展开机制被反汇编证 no-op**（bottleneck ≠ 循环形态·= AVL=8 半宽欠用 VLEN256）·**k1 iq3 现状 0.61/0.65**（引既有 measure.log·非新测）。**★re-scope 工程已决续推**（宽化 = 诊断已定唯一 lever·能力键控参数级 [K-10]·非新结构 plan·先例 Win-K1-VLEN vl=16 满宽）；ISSUE-102 待裁仅**机制①称谓标签**·**不阻塞工程**（k1 iq3 翻 0.8 由 §二 授权）。**★iq3_xxs@k1 首个扇出成功（proven·NOT deployed）**：同叶 `march=rv64gcv_zvl256b`（half_lanes 8→16）板测 **0.65 LOSS→1.38 WIN** vs 部署手调 `_vl256`（**手调档硬赢**·CROSSOP 系统账）·G1 byte-exact PASS（4-arm 真隔离）·objdump 真宽（AVL `zero,8`→`zero,16`·vset 5397→2515·gather 1024→512·**非 re-roll**·2 seed 1.3838/1.3782）·**未改源码**（宽化器 `deriveRepackHalfLanes` 已存·真叶 = `RVVToEmitCBlockQuantLinear.cpp:24897`·非 GridCodebook.cpp）。**★deployed-vs-proven**：标准 k1 分支 march 无 zvl256b ⟹ **deployed 叶仍 VLEN128=0.65·master 维持 具名-X 0.6474 不标 PASS**·部署 gated on **[ISSUE-105]**（GEN_SEAL 对 k1 发 VLEN128 = 半宽根因·per-board fixture 待裁）|
| **②** | [CASE-KQUANT-GCC-CODEGEN]（gcc→clang codegen 杠杆·**仅排序不翻正**） | q4_K/q5_K@rvv decode + 全 rvv-gcc K-quant decode 泛化面 | gcc→clang 恢复 q4_K **0.066→0.361**（~5.5×）· q5_K **0.128→0.496**（~4×） | **clang 残差仍 <0.8**（0.361/0.496）·此杠杆单独不够翻 PASS·须叠加机制③ |
| **③** | [MECH-WEIGHT-RECONSTRUCTION-BOUND]（super-block 位重建原语·(SEW,LMUL,VLEN) 参数化·真成本中心） | q4_K/q5_K@rvv decode M=1 · vec_dot q4_K@rvv · 全 K-quant decode 位重建面 | q5_K@rvv prefill PASS 1.067 + repack seal 1.50–1.62×·q4_K prefill 1.114（有行摊销即赢）=有天花板 | **翻 0.8 未证**：rvv VLEN128 q4_K repack 即使 prefill 也只 0.94×parity（真 hand-brick WIN 锁 k1·不泛化 rvv）·M=1 无行摊销更难·**须丙板测** |
| **④** | [GAP-DEQ-ZERO-VECTOR-EMISSION]（author 真向量化 dequant emit·复用 PR-31） | **9 格**（dequant 轴·cold 未接线） | 可攻坚发射体缺口（指令数内禀档·非物理地板） | **cold 未测**·翻正幅度须板测（丙）·当前无数不宣称幅度 |
| **⑤** | nvfp4 FP4 codebook-gather 向量化（vrgather 16-entry LUT·mxfp4 路已存） | dequant·nvfp4@k1（cold 0.743 SCALAR LOSS） | mxfp4 姊妹同结构 **3.47× WIN**·rvv 侧 nvfp4 clang18 已 autovec 2.17× PASS·正锚 | 4×16 sub-block scale 粒度封顶（VLEN256 半用）→目标仅 ≥0.8 弱赢（opp=scalar-ref 便宜档·非硬赢） |

## 施工序（§二.4）

**机制① 最高扇出优先**（8 格·iq3 正锚证天花板可达）→ ②③（K-quant·须叠加+板测）→ ④（dequant·R 线前置）→ ⑤（nvfp4·弱赢）。
**攻坚候选的「翻 0.8」多数须板测（丙）证实** —— gated on bench harness 基建（各族 harness·ISSUE-099）。

## ★诚实边界

- **机制①：iq3_xxs@k1 首个板测证实**（proven WIN 1.38·手调档硬赢·byte-exact·trellis-check 复核）——但 **proven ≠ deployed**（deployed 叶仍 VLEN128=0.65·[ISSUE-105]）⟹ **master census 不因此翻 PASS**（deployed 现实保留）。②–⑤ 仍**候选未施工·翻正幅度未板测**。
- **★流水线 lever（真高扇出·[ISSUE-105]）**：iq3_xxs@k1 gap 真因 = GEN_SEAL 对 k1 发 VLEN128 fixture（半宽欠用）·非机制/算法。部署 proven 赢（→ deployed PASS）须 per-board k1-VLEN256 fixture = **判据级·须裁**。iq3_s@k1/iq1 系**同病·likely 同 lever·未证**（禁按此外推计数·逐格须 byte-exact+板测）。
- 便宜档（vs generic/scalar 大倍数）**禁称硬赢**·仅降披露列。CROSSOP（repack-GEMM vs dispatched vec_dot）一律标系统账 framing。
- 架构不可达格见 [ISSUE-100]（`vec_dot·nvfp4@rvv`·完整攻坚环证实）。
