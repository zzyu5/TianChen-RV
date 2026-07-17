# PRD · K线 · P2 八格按部署对手落账（ISSUE-004/005 销案）

> **权威** = 《测试与收尾总令-开测篇》§二.2「落账清欠先行：P2 已测 8 格按部署对手落账（vs 通用参考降披露列）」。
> **性质**：落账清欠（settle existing measured data），**非新测量** —— 数据已满足 bench 五步等价（见二）。

## 一、目标

把 P2 已测的 8 个 board-cell（`gemm_tile|{iq1_s,iq1_m,iq3_xxs,iq3_s}` × `{rvv,k1}`）从「争议-pending」**落账**为按**部署对手**的 verdict。核心 = **ISSUE-004 对手政策统一为部署事实**。

## 二、数据源与五步等价自证

**源** = `experiments/active/g8-stage3-attack/P2-grid4-raw/`（前役 seal + measure）。**已满足 bench 五步等价**（逐格 seal/measure 机核）：

| bench 五步 | P2 数据对应 |
|---|---|
| ① 静板预飞 | `PRE_STRAY=0` · `cpu_md5_before` |
| ② 双方板端 clang-18 | `BUILD … CC=clang version 18.1.8`（rvv）/ `Bianbu clang 18.1.8`（k1）—— 双板双方 |
| ③ 对拍三验 | `ABI-GATE PASS` · `T2 ours/OPP vs oracle: mism=0/8192` |
| ④ cold 计时 | `GEMM_PREFILL … reps=25 … 2-seed cold` · IQR ≤1.02% |
| ⑤ 入行 | 本 task 补（写主表 + 部署对手身份） |

## 三、★ISSUE-004 对手政策统一（本 task 的实质）

**问题**（ISSUE-049 已诊断）：同一谓词「无同算子 repack GEMM」在表内**两套政策**：
- K-quant 支 → 选**部署的** `_vl128`（CROSSOP）、认下惨输
- iq 支 → 选**未部署的** scalar-ref、记 PASS ← **违反政策**

**落账 = iq 支比照 K-quant 支，改用部署对手**：
- rvv → `ggml_vec_dot_<fmt>_q8_K_vl128`（VLEN 分发 thunk 在 VLEN128 实跳的部署路径·反汇编坐实）
- k1 → `_vl256`（同理）
- 「vs 通用参考（generic）」**降披露列**（保留但不作 verdict 依据）

## 四、落账映射（部署对手 cold ratio → verdict）

| 格 | rvv ratio | rvv verdict | k1 ratio | k1 verdict |
|---|---|---|---|---|
| iq1_s | 0.67 | 具名-X | 0.59 | 具名-X |
| iq1_m | 0.57 | 具名-X | 0.59 | 具名-X |
| iq3_xxs | 0.95 | **PASS**(≥0.8) | 0.65 | 具名-X |
| iq3_s | 1.34 | **PASS** | 0.61 | 具名-X |

⟹ **部署对手下 8 格输 6**；仅 `iq3_xxs@rvv`(0.95) + `iq3_s@rvv`(1.34) PASS。
**★便宜档禁称硬赢**：对 generic 的 2.71–15.99× 是便宜档，**不入硬赢**，仅降披露列。
**★具名-X 须带墙证据**：每个具名-X 格记逐指令墙分类（对手 = VLEN 专化 full-unroll·我方 = generic aux32 未专化）。

## 四·补 ★落账标的订正（我初稿写错，已核实）

**gemm_tile 的 iq 行【不在板 CSV】** —— 板 CSV 键 = `op|format|shape|arity`，iq 系只有 `vec_dot|dequant`。
`gemm_tile|iq*` 行**住在 recon 内嵌 dict**（`recon_master_rebuild.py:125-131`），现值 = `(S, "ggml scalar-ref(fallback)", …)`。

**落账标的 = recon dict 的 4 条数据条目**（iq1_s/iq1_m/iq3_xxs/iq3_s）。
- **边界厘清**：改 dict 的**数据条目**（对手身份 + verdict 字符串）= **落账**（§二.2 授权）；
  改 recon 的**算法/parse/schema/tier 判定逻辑** = **禁**。
- **先例形态** `recon:132`：`("gemm_tile","iq4_nl"):{"rvv":(V,"ggml_vec_dot_iq4_nl_q8_0_vl128(CROSSOP)","measured anchor 0.217×…")}` ——
  证明「部署对手 + measured ratio」落账 recon **已支持**（非新 schema）。落账后 4 格比照此形。

**★tier 分档张力（[ISSUE-048]·必问·保守默认）**：部署对手 `_vl128` 是 **VLEN 专化手调核**（rvv 109 向量指令），
**非标量** ⟹ 归 手调/通用向量，非「标量类」。但改 tier = 分档口径 = 必问。
**保守默认**：本 task **只落对手身份 + verdict**；tier 变更**登记 ISSUE-048 待裁，本 task 不自改 tier**。

## 五、验收标准

1. **8 board-cell 落账**：recon dict `recon:125-131` 四条 → 部署对手身份（`ggml_vec_dot_<fmt>_q8_K_vl128`/`_vl256`·CROSSOP）+ verdict（cold ratio ≥0.8 PASS 否则具名-X）。**cold ratio 从 `P2-grid4-raw/*_measure.log` 的 `ratio_cold_X` 逐格取真值，禁照抄本 prd 的数**。
2. **数字变动可审计**：recon 重跑后 master CSV 会变（8 格从 pending → 具名-X/PASS，**这是预期落账变动，非守恒**）。**记录落账前后头条对照**（标量类计数会降，因这些格原以 scalar-ref 记 PASS）。
3. **禁注入逗号**：板 CSV `f[29]`/`f[35]` 禁含 `,`（前役坏行前科）；用 em-dash。
4. **pending 下降 8**（20 → 12）。
5. **四门绿** + **sealed 资产不动量自证**（9/83·hand-brick 2·certified 不动）。
6. **触碰集**：板 CSV（数字权威·令文 §二.2 授权）。**recon 脚本逻辑不改**（只改数据）。

## 六、遗留 / 交接

- **iq3_s 板间翻转**（rvv 1.34 赢 / k1 0.61 输·同一 leaf）落账后成活证 → 挂 K 线手调攻坚 ① VLEN 专化满展开 leaf。
- 落账后 8 格不再 pending，但 6 个具名-X 是「架构不可达/待攻坚」候选 → 进 K 线手调攻坚环（非静默认输·带墙证据）。
- **交接 trellis-check**：核对手身份是**真部署路径**（反汇编）、cold ratio 从 seal 逐格取值无误、头条变动符合落账预期、sealed 不动、无逗号坏行。
