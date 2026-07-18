# Research: non-grid dequant 全表（B 线第一块收割侦察）

- **Query**: non-grid dequant ~15 格逐格 {verdict/cold · de-lottery owned emit 是否已落 · harness 是否 ready · 成色}，排除 grid（gather 墙 ISSUE-107）
- **Scope**: internal（只读 survey · 零板攻 · 不测速度）
- **Date**: 2026-07-19

## 权威源

| 源 | 路径 |
|---|---|
| 主表（逐板 verdict/cold） | `experiments/master/T3_master_rebuild.csv`（dequantize_row 行 = L32–L55） |
| verdict/cold 重铸 dict | `.trellis/scripts/recon_master_rebuild.py`（`CLANG_WORLD` L261–280 · `COLD` L170+ · `disp()` L282–293） |
| 杠杆台账（F6/F7 分族） | `experiments/active/result-tables/named-X-leverage-census.md`（§2.2 F6 · §四 F7） |
| harness 现状 | `tools/bench/cells/dequantize_row.sh`（case L56–79 · k1 VOID L91–92） |
| owned-emit 派发点 | `lib/Conversion/RVV/RVVToEmitCForwardElementwise.cpp`（`emitTypedDequantizeRowLoopBody` L3628–3777） |
| 已证 owned 范本 | `experiments/active/r-dequant/kernels/q8_0_dequant.c`（5 owned intrinsic · gather=0） |

---

## 一、24 dequant 格分类（先厘清 grid 边界）

主表 dequantize_row 共 **24 个 format**。分四类：

| 类 | formats | 计数 | 处置 |
|---|---|---|---|
| **A. 已 owned（de-lottery 已落·收割完）** | q4_0 · q4_1 · q5_0 · q5_1 · q8_0 | 5 | 完成（本会话已证·非本次收割目标） |
| **B. 域外（永久·非分母）** | q1_0 | 1 | 排除（Weft-internal · CSV L37） |
| **C. GRID（gather 墙·ISSUE-107·排除）** | iq3_xxs · iq3_s · iq1_m〔ISSUE-107 具名〕+ iq2_xxs · iq2_xs · iq2_s · iq1_s〔同族结构墙〕 | 7 | **排除**（见 §三 grid 边界订正） |
| **D. NON-GRID 收割面（未 owned·de-lottery 候选）** | q2_K q3_K q4_K q5_K q6_K · iq4_nl iq4_xs mxfp4 nvfp4 · tq1_0 tq2_0 | **11** | **本次收割目标** |

> **★「~15 格」订正**：行动书 r3 的「~15」= 18 in-domain 未 owned 格 − 3 显式 grid（iq3_xxs/iq3_s/iq1_m）= 15。但那 15 里 **4 格（iq1_s/iq2_xxs/iq2_xs/iq2_s）结构上仍是 grid-table gather**（census §2.2 note「grid 族(iq3_s/iq2*)同墙」+ 发射器 IQ-grid 组 L3740–3745）——**真正「无 gather 墙」的干净收割面 = 11 格**（K-quant 5 + tiny-codebook 4 + ternary 2）。这是本侦察对 scope 的关键收窄，须 main 采纳。

---

## 二、NON-GRID 收割全表（11 格 · 板 = rvv）

> 板：**rvv 为收割板**（q8_0 范式在 rvv 先证）。**@k1 dequant 部署全 gated ISSUE-105 半宽**（harness 对 k1 直接 `HARNESS-VOID`·`dequantize_row.sh` L91–92）——k1 cold 列仅作 lottery 基线参考，非本次收割对象。
> owned 列 = 该格 dequant emit 是否已换 owned 真向量 body（对照发射器 L3717–3776）。
> harness 列 = `dequantize_row.sh` 是否已认该 fmt（case L56–79 现仅认 iq3_xxs/iq3_xxs_grid/q8_0/q4_0/q5_0/q4_1/q5_1）。

### 2.1 K-quant super-block（QK_K=256 · 位重建 · 无 gather）— 5 格

| format | rvv verdict | rvv cold | k1 cold(lottery) | owned emit? | harness? | 成色/结构 |
|---|---|---|---|---|---|---|
| q2_K | PASS | 空(stale-T3) | 0.828 | ✗ 未落（scalar forwarder） | ✗ blocked | 2-bit 超块·16 sub-block·便宜档 opp autovec |
| q3_K | PASS | 空 | 3.227 | ✗ | ✗ blocked | 3-bit 超块·便宜档 |
| **q4_K** | **具名-X** | 空 | 4.315 | ✗ | ✗ blocked | **census F7**·4-bit 超块·便宜档 |
| **q5_K** | **具名-X** | 空 | 5.127 | ✗ | ✗ blocked | **census F7**·5-bit 超块·便宜档 |
| q6_K | PASS | 空 | 1.115 | ✗ | ✗ blocked | 6-bit 超块·便宜档 |

### 2.2 tiny-codebook（16-entry LUT · vrgather 寄存器常驻 · 非内存 gather）— 4 格

| format | rvv verdict | rvv cold | k1 cold(lottery) | owned emit? | harness? | 成色/结构 |
|---|---|---|---|---|---|---|
| iq4_nl | PASS | 空 | 3.867 | ✗ | ✗ blocked | flat fp16 scale + 16-entry 非线性码本·便宜档 |
| **iq4_xs** | **具名-X** | 空 | 3.151 | ✗ | ✗ blocked | **census F7**·超块 signed-6 scale + 16-entry 码本·便宜档 |
| mxfp4 | PASS | 空 | 3.13 | ✗ | ✗ blocked | FP4 e2m1 + E8M0 共享指数·**vrgather 正锚 3.47× WIN**（机制⑤·ledger L17） |
| nvfp4 | PASS | **2.1736** | 0.716(具名-X) | ✗ | ✗ blocked | FP4 + 4×UE4M3 sub-scale·rvv 已 lottery-PASS·便宜档 |

### 2.3 ternary super-block（无 gather）— 2 格

| format | rvv verdict | rvv cold | k1 cold(lottery) | owned emit? | harness? | 成色/结构 |
|---|---|---|---|---|---|---|
| tq1_0 | PASS | 空 | 4.285 | ✗ | ✗ blocked | base-3 packed ternary·便宜档 |
| **tq2_0** | **具名-X** | 空 | 4.232 | ✗ | ✗ blocked | **census F7**·2-bit ternary（近 nibble）·便宜档 |

### 2.4 全表读数

- **owned emit 已落 = 0/11**（全 11 格 dequant 仍走 `emitGgmlDequantizeRowExtended` scalar forwarder·发射器 L3729–3757·kernels 目录只有 5 block + iq3_xxs）。
- **harness ready = 0/11**（`dequantize_row.sh` case L56–79 全不认 → `HARNESS-VOID bad fmt`·L78）——**全 11 格 harness-blocked（ISSUE-099-like）**。
- **rvv cold 数据**：10/11 空（stale-T3·census §五 note「cold 空须 owned emit 落地后板测重入账」）；**唯一有 cold = nvfp4@rvv 2.1736**（lottery-PASS）。
- **rvv 现-具名-X = 4 格 = 恰 census F7**（q4_K/q5_K/iq4_xs/tq2_0·§四 F7 L176）——最高「惨状」优先，但**按扇出排批时分散在 3 个 family**（见 batch-plan）。

---

## 三、★grid 边界订正（排除面·gather 墙 ISSUE-107）

发射器把 IQ 族分两组（`emitTypedDequantizeRowLoopBody` L3740–3757）：

- **IQ grid-table 组**（`emitDequantizeRowIQGridBodyShared` L3578）：iq2_xxs · iq2_xs · iq2_s · iq3_xxs · iq3_s —— 2048/256-entry grid·`vluxei` 内存 indexed gather。
- **ternary-grid / codebook 组**（`emitDequantizeRowCodebookGridBodyShared` L3611）：iq1_s · iq1_m（iq1s_grid 三值 grid）+ iq4_nl/iq4_xs/mxfp4/nvfp4（16-entry 码本）+ tq1_0/tq2_0。

**grid gather 墙 = 大 grid 内存 gather（vluxei）**，ISSUE-107 board-tested 证毕（性能与测量.md L365–366）：iq3_xxs@rvv **3 owned 变体全 <0.8**（naive 0.181 / 标量-load 0.33 / HW-gather 0.36）·honest-null·杠杆清单空(construction)。同墙结构推断覆盖 **iq3_s / iq2* / iq1_m / iq1_s**（census §2.2 F6 note「grid 族(iq3_s/iq2*)同墙」）。

**⟹ 排除 7 格**：iq3_xxs · iq3_s · iq1_m（ISSUE-107 具名）+ iq2_xxs · iq2_xs · iq2_s · iq1_s（同族结构墙·大 grid gather）。这 7 格 **不是干净收割**（owned 会撞 gather 天花板·[L-8]-vs-PASS tradeoff 待用户战略裁 ISSUE-107/106）。

**tiny-codebook（16-entry）≠ grid 墙**：16 项平凡装入单向量寄存器→`vrgather.vv` 寄存器置换是正确原语（对比大 grid 的 vluxei 内存 gather）。mxfp4 已有 **3.47× WIN 正锚**（ledger 机制⑤ L17）证 vrgather 码本路可达。故 iq4_nl/iq4_xs/mxfp4/nvfp4 **留在收割面**（非 gather 墙·需 vrgather 适配·见 applicability 文件）。

---

## 四、成色钉死（禁美化·如实）

de-lottery 对手 = **部署的 `dequantize_row_<fmt>` = scalar-source host-autovec = codegen-lottery = opp-immaturity 便宜档**（`dequantize_row.sh` L4–6 契约·对手法 §〇.1）。

- 收割价值 = **关 ISSUE-002 该格敞口**（dequant 24 行「由机制构造」[L-8] 强义存疑·随 de-lottery 逐格关·性能与测量.md L18–24）+ **[L-8] 强义 construction 真赢**（owned 真向量·非 autovec 抽签·ISSUE-001 反转 L10–16）。
- **非 perf 硬赢**：q8_0/q5_0 的 2.33×/5.96× 大倍数 = 对手 autovec 差（§三.12 opp-immaturity）·非我方核质量硬碾手调对手·**禁称重大 WIN / 禁外推 grid/K-quant**（CLANG_WORLD dict L279 明文）。
- 预期 verdict：memory-bound streaming（K-quant/ternary/codebook decode 皆 store-heavy 无 reduction）→ 多为 **PASS→PASS 或 具名-X→PASS**·成色 = 便宜档·**赢要标便宜档**。

## Caveats / Not Found

- **rvv cold 10/11 空**：owned emit 未落地前无冷计时·verdict token 来自 stale-T3（census §五 L189）。本表 verdict 列为 owned-emit-前的 lottery 状态·**落地后须板测重入账**（不测速度是本侦察纪律·真数留 implement 批）。
- **K-quant dequant ≠ K-quant vec_dot**：F1「weight-reconstruction floor」EXHAUSTED 墙是 **vec_dot（reduction）** 的（census §四 F1 L170）·**dequant 是 streaming 无 q8_K 归约**·该墙不适用——K-quant dequant 结构上应像 q8_0 干净 de-lottery（详 applicability 文件 §一）。禁把 vec_dot 墙误植到 dequant。
- **byte-exact 风险（FMA-contract）**：K-quant/codebook decode 含多重乘（d × sub-scale × quant）·可能有 `-ffp-contract=on` vfmadd-vs-vfmul+vfadd 歧义（同 q4_1/q5_1 FMA-风险集·blockquant PRD §二）·须逐格证 contraction 一致·未证不硬凑（applicability 文件 §三）。
