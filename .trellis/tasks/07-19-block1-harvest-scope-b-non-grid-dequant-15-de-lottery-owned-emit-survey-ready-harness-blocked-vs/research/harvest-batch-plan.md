# Research: 收割批计划（按扇出排序 · non-grid dequant de-lottery）

- **Query**: 按「入库/模式复制能扇几格」排序收割批（非单格惨状）·分直接复制批 / harness-建后批 / 需适配批·便宜档标·harness-blocked 批标需求
- **Scope**: internal（survey · 零代码改）
- **Date**: 2026-07-19

## 扇出机理（先看发射器的「shared body」结构）

de-lottery 的扇出来自**发射器共享 body 函数**——author 一个 owned 真向量 body 可同时覆盖同族多格：

| 共享 body 函数 | 现状 | 覆盖 formats | 源 |
|---|---|---|---|
| `emitDequantizeRowNibbleVectorBody` | **已 owned（已收割）** | q4_0/q4_1/q5_0/q5_1（1 body 扇 4 格） | `RVVToEmitCForwardElementwise.cpp:2966` · 派发 L3717–3728 |
| `emitDequantizeRowQ8_0VectorBody` | **已 owned（已收割）** | q8_0 | L3387 · 派发 L3772 |
| `emitDequantizeRowKQuantBodyShared` | ✗ 仍 scalar forwarder | q2_K/q3_K/q4_K/q5_K/q6_K（5 格待扇） | L3554 → `emitGgmlDequantizeRowExtended` |
| `emitDequantizeRowCodebookGridBodyShared` | ✗ 仍 scalar forwarder | iq4_nl/iq4_xs/mxfp4/nvfp4（+ grid iq1_s/iq1_m）· tq1_0/tq2_0 | L3611 → scalar |

> **★关键结论**：**已有的「直接复制」扇出面（flat nibble + q8_0）已被本会话收割完**（5 格）。剩余 11 格全在 **K-quant super-block / tiny-codebook / ternary** 三个**结构不同**的族——**没有一格是 flat-nibble body 的纯直接复制**。收割方式 = 每族 author 一个新 shared 超块/码本/ternary body（family 内首格 = 适配·同族余格 = family-内直接复制）。

---

## 收割批（按扇出降序 · 全 harness-blocked）

> ★**所有 3 批都 harness-blocked**：`dequantize_row.sh` case（L56–79）只认 5 block + iq3_xxs·11 收割格全触 `HARNESS-VOID bad fmt`（L78）；`experiments/active/r-dequant/kernels/` 只有 5 block + iq3_xxs 两 kernel——**无「harness-ready 可即打」批**。每批须先建 harness（新 LEAFC owned kernel + DRVC driver + GSED fault + OPPSYM + oracle 对拍）= ISSUE-099-like 前置。

### 批 1 — K-quant super-block（扇出 = 5 · 最高 · 含 2 census-F7）

- **格**：q2_K · q3_K · **q4_K**(F7) · **q5_K**(F7) · q6_K（@rvv）
- **扇出**：author 1 个超块 owned body（`emitDequantizeRowKQuantBodyShared` L3554 换 owned·参数化 bit-width 2/3/4/5/6）→ 扇 5 格。**最高扇出**。
- **模式**：**需适配**（QK_K=256 超块 · per-sub-block 6-bit scale + fp16 super-scale + 2/3/4/5/6-bit 位解包·结构 ≠ flat QK=32 block）；family 内首格适配后·余 4 格随 bit-width 复制。
- **墙风险**：**低**（dequant 是 streaming·无 q8_K 归约·**F1 vec_dot「weight-reconstruction floor」墙不适用**·见 applicability §一）；byte-exact 须证 d×sub-scale×quant 的 fp-contract 一致（FMA-风险）。
- **成色**：便宜档（opp = `dequantize_row_qX_K` autovec）·价值 = 关 ISSUE-002 敞口 + [L-8]·k1 lottery 基线 0.83–5.13（不测·参考）。
- **harness 需求**：5 owned kernel + 5 driver + oracle（ggml `dequantize_row_qX_K`·住 libggml-base）+ case 扩 5 条。

### 批 2 — tiny-codebook 16-entry（扇出 = 4 · 含 1 census-F7 · 有正锚）

- **格**：iq4_nl · **iq4_xs**(F7) · mxfp4 · nvfp4（@rvv）
- **扇出**：author 1 个 vrgather 16-entry 码本 body（`emitDequantizeRowCodebookGridBodyShared` L3611 换 owned）→ 扇 4 格。
- **模式**：**需适配**（vrgather 寄存器常驻码本 = 与 flat nibble 不同原语·**非大 grid gather 墙**）；**mxfp4 已有 3.47× WIN 正锚**（ledger 机制⑤ L17·vrgather 路已存）——适配置信最高的一族。
- **墙风险**：**低-中**（16 项 vrgather 是正确原语·非 vluxei 内存 gather）；nvfp4 vec_dot@rvv 侧曾诊出「码本 TABLE spill」(F5 L174)·但那是 vec_dot·dequant streaming 应更干净；scale 粒度（mxfp4 E8M0 共享 / nvfp4 4×UE4M3 / iq4_xs 超块 signed-6 / iq4_nl flat fp16）逐格异 = 适配点。
- **成色**：便宜档；nvfp4@rvv 已 lottery-PASS 2.1736（de-lottery = 换 owned·关敞口·非涨数）。
- **harness 需求**：4 owned kernel（含 vrgather 码本 decl）+ 4 driver + oracle + case 扩 4 条。

### 批 3 — ternary super-block（扇出 = 2 · 含 1 census-F7）

- **格**：tq1_0 · **tq2_0**(F7)（@rvv）
- **扇出**：author ternary body → 扇 2 格。**最低扇出**。
- **模式**：**需适配**·且 **tq1_0（base-3 packed）与 tq2_0（2-bit）解包不同构** → 近似两次适配·family 内复用弱。tq2_0（2-bit）最近 nibble（适配最轻）·tq1_0（base-3）最重。
- **墙风险**：低（ternary unpack + scale·无 gather·无 reduction）；byte-exact fp-contract 须证。
- **成色**：便宜档；k1 lottery 4.23–4.29（不测·参考）。
- **harness 需求**：2 owned kernel + 2 driver + oracle + case 扩 2 条。

---

## 排批建议（供 main 派 implement）

| 批 | 族 | 扇出 | census-F7 覆盖 | 适配难度 | 正锚 | harness 状态 |
|---|---|---|---|---|---|---|
| **1** | K-quant super-block | **5** | q4_K, q5_K（2） | 中（超块位解包·family 内复用强） | 无（但 dequant 无 F1 墙） | blocked（建 5 kernel+oracle） |
| **2** | tiny-codebook 16-entry | 4 | iq4_xs（1） | 中（vrgather 码本适配） | **mxfp4 3.47× WIN** | blocked（建 4 kernel+码本 decl） |
| **3** | ternary super-block | 2 | tq2_0（1） | 中-高（tq1_0 base-3 不复用 tq2_0） | 无 | blocked（建 2 kernel） |

**推荐次序 = 批1 → 批2 → 批3**：
1. **批1 扇出最高（5）**·覆盖 2 个 census-F7·family 内 bit-width 参数化复用强·dequant 无 F1 墙 → 单位 harness 投入回报最高。
2. **批2 次之（4）**·有 mxfp4 vrgather 正锚降适配风险·覆盖 iq4_xs（F7）。
3. **批3 最后（2）**·扇出最低且 tq1_0/tq2_0 不同构·family 复用弱。

> **★注**：4 个 census-F7「惨状格」(q4_K/q5_K/iq4_xs/tq2_0) 分散在 3 个 family——**按 family 攻（非按单格惨状）自然一并覆盖**。批1 一举收 q4_K+q5_K 两个 F7·是最优先。

## Caveats / Not Found

- **无「直接复制即打」批**：flat-nibble 的直接扇出面已被 5 block 收割穷尽·剩 11 格全需 family 级新 body 适配 + harness 建设。所谓「收割 = 复制已证模式」在此处 = **复制 q8_0 的〈de-lottery 方法学〉（owned emit + byte-exact + 便宜档成色）**·非复制其〈body 代码〉。
- **harness 建设成本 = 收割真前置**：每格 owned kernel + driver + oracle 对拍 + fault 探针（`dequantize_row.sh` 契约 L43–55）——批1 = 5 套·非零工程。这是「最大最便宜」中「便宜」的边界：**便宜 = 无 gather 墙/无算法墙（几乎必 PASS）·非 = 零工程**。
- **成色统一钉死**：3 批全 = 便宜档 opp-immaturity（对手 autovec scalar-C）·价值 = ISSUE-002 敞口 + [L-8] construction·**禁称硬赢·禁外推**（CLANG_WORLD dict L279 · §三.12）。
