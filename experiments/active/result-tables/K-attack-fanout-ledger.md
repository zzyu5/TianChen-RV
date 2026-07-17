# K线 手调攻坚 · 扇出台账（机制入库）

> 《测试与收尾总令-开测篇》§二.4「每攻克一条 = 机制入库 + 扇出台账行 + XFER 正/负预测」·§七终审交付②。
> 源 = 攻坚环解剖阶段（task `07-18-k-attack-dissect`·trellis-check 已核承重数）。**诊断阶段·机制候选未施工。**

## 机制队列（按扇出降序 · §二.4 手调列队）

| # | 机制 | 命中格 | XFER 正预测 | XFER 负预测 |
|---|---|---|---|---|
| **①** | **VLEN 专化满展开 grid-gather leaf**（消灭 generic-aux32 + vsetvli storm·对手已有此形态·可复制） | **8 格**：iq1_s/iq1_m/iq3_xxs/iq3_s @rvv,k1 | **iq3 系天花板可达**：iq3_s@rvv **1.34 WIN**（seal 真值·正锚）· iq3_xxs@rvv **0.95** near-parity（runs row.csv）⟹ k1 侧翻 0.8 有可达天花板 | **iq1 系弱先验·翻 0.8 未证**：iq1_s cold 0.47 双板皆输·无 parity 锚·我方 aux32 21k 全族最大·对手 vl128 leaf 不轻（2048 项 grid）· **k1 iq3 侧**对手 chip-tuned vl256 满展开·未证 |
| **②** | [CASE-KQUANT-GCC-CODEGEN]（gcc→clang codegen 杠杆·**仅排序不翻正**） | q4_K/q5_K@rvv decode + 全 rvv-gcc K-quant decode 泛化面 | gcc→clang 恢复 q4_K **0.066→0.361**（~5.5×）· q5_K **0.128→0.496**（~4×） | **clang 残差仍 <0.8**（0.361/0.496）·此杠杆单独不够翻 PASS·须叠加机制③ |
| **③** | [MECH-WEIGHT-RECONSTRUCTION-BOUND]（super-block 位重建原语·(SEW,LMUL,VLEN) 参数化·真成本中心） | q4_K/q5_K@rvv decode M=1 · vec_dot q4_K@rvv · 全 K-quant decode 位重建面 | q5_K@rvv prefill PASS 1.067 + repack seal 1.50–1.62×·q4_K prefill 1.114（有行摊销即赢）=有天花板 | **翻 0.8 未证**：rvv VLEN128 q4_K repack 即使 prefill 也只 0.94×parity（真 hand-brick WIN 锁 k1·不泛化 rvv）·M=1 无行摊销更难·**须丙板测** |
| **④** | [GAP-DEQ-ZERO-VECTOR-EMISSION]（author 真向量化 dequant emit·复用 PR-31） | **9 格**（dequant 轴·cold 未接线） | 可攻坚发射体缺口（指令数内禀档·非物理地板） | **cold 未测**·翻正幅度须板测（丙）·当前无数不宣称幅度 |
| **⑤** | nvfp4 FP4 codebook-gather 向量化（vrgather 16-entry LUT·mxfp4 路已存） | dequant·nvfp4@k1（cold 0.743 SCALAR LOSS） | mxfp4 姊妹同结构 **3.47× WIN**·rvv 侧 nvfp4 clang18 已 autovec 2.17× PASS·正锚 | 4×16 sub-block scale 粒度封顶（VLEN256 半用）→目标仅 ≥0.8 弱赢（opp=scalar-ref 便宜档·非硬赢） |

## 施工序（§二.4）

**机制① 最高扇出优先**（8 格·iq3 正锚证天花板可达）→ ②③（K-quant·须叠加+板测）→ ④（dequant·R 线前置）→ ⑤（nvfp4·弱赢）。
**攻坚候选的「翻 0.8」多数须板测（丙）证实** —— gated on bench harness 基建（各族 harness·ISSUE-099）。

## ★诚实边界

- 这些是**候选机制·未施工**。翻正幅度**无一已板测证实**（除 iq3 系正锚是既有 seal）。
- 便宜档（vs generic/scalar 大倍数）**禁称硬赢**·仅降披露列。
- 架构不可达格见 [ISSUE-100]（`vec_dot·nvfp4@rvv`·完整攻坚环证实）。
