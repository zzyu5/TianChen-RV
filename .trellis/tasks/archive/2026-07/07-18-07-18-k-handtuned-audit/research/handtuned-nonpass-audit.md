# Research: 手调档非 PASS 逐格审计（裁决 2·穷举·8 类归类）

- **Query**: 机算穷举扫 `experiments/master/T3_master_rebuild.csv`，筛手调档非 PASS 格，逐格表 + 8 类归类闭合。只出表不改 verdict。
- **Scope**: internal（主表 + spec + 仓内证据）
- **Date**: 2026-07-18

---

## 方法（穷举·非窗口·连提取代码）

主表 = `experiments/master/T3_master_rebuild.csv`（**108 数据行** + header；`wc -l` = 109）。
每行含两板 5 列（`*_tier / *_disp / *_cold / *_opp_sym / *_note`）。一个「格|板」= (行 × 板)。

**入选谓词**（(row,board) 命中 iff）：
`disp 不以 "PASS" 起头`  ∧  `( tier=="手调"  OR  opp_sym 含手调标记 )`，排除 `tier∈{N/A-hw, 域外}`。
手调标记 = `["_vl128","_vl256","8x8","16x1","8x4","IME","hand-brick"]`。

提取代码（即席复跑）：
```python
import csv
CSV="experiments/master/T3_master_rebuild.csv"
HT=["_vl128","_vl256","8x8","16x1","8x4","IME","hand-brick"]
ht=lambda s:any(m in s for m in HT); ispass=lambda d:d.startswith("PASS")
rows=list(csv.DictReader(open(CSV)))               # 108 行
for r in rows:
  cid=f"{r['op']}|{r['format']}"+(f"@{r['engine']}" if r['engine'] else "")+(f"/{r['regime']}" if r['regime'] else "")
  for b,tk,dk,ck,ok in [("rvv","rvv_tier","rvv_disp","rvv_cold","rvv_opp_sym"),
                        ("k1","k1_tier","k1_disp","k1_cold","k1_opp_sym")]:
    tier,disp,cold,opp=r[tk].strip(),r[dk].strip(),r[ck].strip(),r[ok].strip()
    if tier in("N/A-hw","域外") or ispass(disp): continue
    if tier=="手调" or ht(opp): print(cid,b,tier,disp,cold,opp)
```

**独立交叉验证**：`tier=="手调"` 非 PASS 命中 = **36**，与主表配套工件 `T3_master_rowclue.txt` 独立机算数（rvv 手调 24−8PASS=**16** + k1 手调 29−9PASS=**20** = 36）**逐一吻合**。加 10 个 `tier≠手调 但 opp 手调标记`（术语碰撞·见下）= **总 46 命中**。

---

## 一、逐格表（全量 46·按归类分组）

列 = `格 | 板 | 现tier | 现verdict(disp) | 现cold | 对手符号 | 归类 | 依据/走环?`

### CAT 1 · 对手作废（2 格·均 pending-fold·**非 loss**）
| 格 | 板 | tier | verdict | cold | 对手符号 | 依据 |
|---|---|---|---|---|---|---|
| gemm\|iq4_nl@rvv/decode | rvv | 通用向量 | pending-fold | — | ggml_vec_dot_iq4_nl_q8_0_vl128(CROSSOP) | ISSUE-064 |
| gemm\|iq4_nl@rvv/prefill | rvv | 通用向量 | pending-fold | — | ggml_vec_dot_iq4_nl_q8_0_vl128(CROSSOP) | ISSUE-064 |

### CAT 2 · 数据脏（**0 格**·见下方分析）

### CAT 3 · 法条明禁（1 格·pending·**禁计失败**）
| 格 | 板 | tier | verdict | cold | 对手符号 | 依据 |
|---|---|---|---|---|---|---|
| gemm\|q8_0@ime | k1 | 手调 | pending(IME结构·该板无合法对手…) | — | stock IME q8_0 (vendor) | ISSUE-029 |

### CAT 4 · march 残缺致半宽（4 格·gated 裁决 1 / ISSUE-105）
| 格 | 板 | tier | verdict | cold | 对手符号 | 依据/走环? |
|---|---|---|---|---|---|---|
| gemm\|iq1_s@rvv | k1 | **标量类** | 具名-X | 0.5919 | ..._iq1_s_q8_K_vl256(CROSSOP) | mech① · iq1系同病未证 |
| gemm\|iq1_m@rvv | k1 | **标量类** | 具名-X | 0.5945 | ..._iq1_m_q8_K_vl256(CROSSOP) | mech① · iq1系同病未证 |
| gemm\|iq3_xxs@rvv | k1 | **标量类** | 具名-X | 0.6474 | ..._iq3_xxs_q8_K_vl256(CROSSOP) | mech① · **proven 0.65→1.38**(未部署) |
| gemm\|iq3_s@rvv | k1 | **标量类** | 具名-X | 0.6136 | ..._iq3_s_q8_K_vl256(CROSSOP) | mech① · 板间翻转 rvv1.34/k1 0.61 |

### CAT 5 · 覆盖缺口（可按批变绿·ISSUE-019）（8 格）
| 格 | 板 | tier | verdict | cold | 对手符号 | 依据/走环? |
|---|---|---|---|---|---|---|
| vec_dot\|q2_K | rvv | 手调 | 具名-X | 0.342 | ..._q2_K_q8_K_vl128 | oracle 待补·未逐格环 |
| vec_dot\|q2_K | k1 | 手调 | 具名-X | 0.684 | ..._q2_K_q8_K_vl256 | oracle 待补·未逐格环 |
| vec_dot\|q3_K | rvv | 手调 | 具名-X | 0.258 | ..._q3_K_q8_K_vl128 | oracle 待补·未逐格环 |
| vec_dot\|q3_K | k1 | 手调 | 具名-X | 0.522 | ..._q3_K_q8_K_vl256 | oracle 待补·未逐格环 |
| vec_dot\|q4_K | k1 | 手调 | 具名-X | 0.592 | ..._q4_K_q8_K_vl256 | @k1 march半宽·未逐格环 |
| vec_dot\|q6_K | k1 | 手调 | 具名-X | 0.549 | ..._q6_K_q8_K_vl256 | @k1 march半宽·未逐格环 |
| vec_dot\|iq4_nl | k1 | 手调 | 具名-X | 0.697 | ..._iq4_nl_q8_0_vl256 | ISSUE-019 iq4_nl·k1 半宽 |
| gemm\|iq4_nl@rvv/prefill | k1 | 通用向量 | 具名-X | 0.6025 | ..._iq4_nl_q8_0_vl256(CROSSOP) | ISSUE-019 iq4_nl·k1 prefill |

### CAT 6 · 墙记指错成本中心（7 格·ISSUE-014·成本中心留白·留攻击队列）
| 格 | 板 | tier | verdict | cold | 对手符号 | 现墙记(须订正) |
|---|---|---|---|---|---|---|
| gemm\|q2_K@rvv/decode | rvv | 手调 | 具名-X(decode-M1) | 0.0685 | ..._q2_K_q8_K_vl128(CROSSOP) | **"结构墙"** + fold@M=1 |
| gemm\|q3_K@rvv/decode | rvv | 手调 | 具名-X(decode-M1) | 0.083 | ..._q3_K_q8_K_vl128(CROSSOP) | **"结构墙"** + fold@M=1 |
| gemm\|q3_K@rvv/decode | k1 | 手调 | 具名-X(decode-M1) | 0.4252 | ..._q3_K_q8_K_vl256(CROSSOP) | fold@M=1 不 amortize |
| gemm\|q4_K@rvv/decode | rvv | 手调 | 具名-X(decode-M1) | 0.361 | ..._q4_K_q8_K_vl128(CROSSOP) | **"指令数内禀墙"** + 对手结构优势具名 |
| gemm\|q5_K@rvv/decode | k1 | 手调 | 具名-X(decode-M1) | 0.6806 | ggml_gemm_q5_K_8x4_q8_K(repack) | fold@M=1·成色升级 DEFERRED |
| gemm\|q6_K@rvv/decode | rvv | 手调 | 具名-X(decode-M1) | 0.0535 | ..._q6_K_q8_K_vl128(CROSSOP) | **"结构墙"** + fold@M=1 |
| gemm\|q6_K@rvv/decode | k1 | 手调 | 具名-X(decode-M1) | 0.3771 | ggml_gemm_q6_K_16x1_q8_K(repack) | fold@M=1 不 amortize |

> 注：ISSUE-014 的「8 格 K-quant decode」= 上 7 格（手调）+ `q5_K@rvv/decode`（**tier=通用向量·出手调scope**·0.1273·"结构墙"）。

### CAT 7 · 真墙·已走完整环（4 格·带 objdump/perf/真硅）
| 格 | 板 | tier | verdict | cold | 对手符号 | 墙类型(三档) |
|---|---|---|---|---|---|---|
| vec_dot\|q4_K | rvv | 手调 | 具名-X | 0.189 | ..._q4_K_q8_K_vl128 | 对手结构优势具名(register-resident)·ISSUE-109 |
| gemm\|iq4_nl@rvv/decode | k1 | 通用向量 | 具名-X(decode-M1) | 0.2493 | ..._iq4_nl_q8_0_vl256(CROSSOP) | 对手结构优势具名(vrgather vs vluxei16) |
| gemm\|q4_0@ime | k1 | 手调 | 具名-X(IME-kernel-sym) | 0.196 | stock IME q4_0 (vendor) | 对手结构优势具名(vendor scale-fold epilogue 融进 vmadot) |
| gemm\|q4_K@ime | k1 | 手调 | 具名-X(IME-kernel-sym) | 0.049 | stock IME q4_K (vendor) | 对手结构优势具名(vendor 单核·format-keyed 边界) |

### CAT 8 · 未攻 / pending（20 格·欠账·非认输）
| 格 | 板 | tier | verdict | cold | 对手符号 | 依据 |
|---|---|---|---|---|---|---|
| vec_dot\|q6_K | rvv | 手调 | 具名-X | 0.18 | ..._q6_K_q8_K_vl128 | ★baseline+oracle 已建·**全环未走**(q6k task 未启动) |
| vec_dot\|iq1_s | rvv | 手调 | 具名-X | 0.365 | ..._iq1_s_q8_K_vl128 | grid vecdot·census only |
| vec_dot\|iq1_s | k1 | 手调 | 具名-X | 0.361 | ..._iq1_s_q8_K_vl256 | grid vecdot·census only |
| vec_dot\|iq1_m | rvv | 手调 | 具名-X | 0.152 | ..._iq1_m_q8_K_vl128 | grid vecdot·census only |
| vec_dot\|iq1_m | k1 | 手调 | 具名-X | 0.25 | ..._iq1_m_q8_K_vl256 | grid vecdot·census only |
| vec_dot\|iq2_xxs | rvv | 手调 | 具名-X | 0.697 | ..._iq2_xxs_q8_K_vl128.isra.0 | grid vecdot·census only |
| vec_dot\|iq2_xxs | k1 | 手调 | 具名-X | 0.612 | ..._vl256 | grid vecdot·census only |
| vec_dot\|iq2_xs | rvv | 手调 | 具名-X | 0.529 | ..._iq2_xs_q8_K_vl128.isra.0 | grid vecdot·census only |
| vec_dot\|iq2_xs | k1 | 手调 | 具名-X | 0.474 | ..._vl256 | grid vecdot·census only |
| vec_dot\|iq2_s | k1 | 手调 | 具名-X | 0.562 | ..._vl256 | grid vecdot·census only(rvv侧PASS 1.92) |
| vec_dot\|iq3_xxs | rvv | 手调 | 具名-X | 0.156 | ..._iq3_xxs_q8_K_vl128.isra.0 | grid vecdot·census only |
| vec_dot\|iq3_xxs | k1 | 手调 | 具名-X | 0.192 | ..._vl256 | grid vecdot·census only |
| vec_dot\|iq3_s | rvv | 手调 | 具名-X | 0.213 | ..._iq3_s_q8_K_vl128.isra.0 | grid vecdot·census only |
| vec_dot\|iq3_s | k1 | 手调 | 具名-X | 0.2 | ..._vl256 | grid vecdot·census only |
| vec_dot\|mxfp4 | rvv | 手调 | **pending-fold** | — | ..._mxfp4_q8_0_vl128 | pending 未定 verdict·非 loss |
| vec_dot\|mxfp4 | k1 | 手调 | **pending-fold** | — | ..._mxfp4_q8_0_vl256 | pending 未定 verdict·非 loss |
| vec_dot\|tq1_0 | rvv | 手调 | 具名-X | 0.207 | ..._tq1_0_q8_K_vl128.isra.0 | ternary vecdot·census only |
| vec_dot\|tq1_0 | k1 | 手调 | 具名-X | 0.606 | ..._vl256 | ternary vecdot·census only |
| gemm\|iq1_s@rvv | rvv | **标量类** | 具名-X | 0.6643 | ..._iq1_s_q8_K_vl128(CROSSOP) | rvv CROSSOP·mech① lever(VLEN256)在rvv N/A |
| gemm\|iq1_m@rvv | rvv | **标量类** | 具名-X | 0.5713 | ..._iq1_m_q8_K_vl128(CROSSOP) | rvv CROSSOP·mech① lever(VLEN256)在rvv N/A |

---

## 二、归类计数

| CAT | 名称 | 计数 |
|---|---|---|
| 1 | 对手作废 | 2（均 pending-fold·非 loss） |
| 2 | 数据脏 | **0** |
| 3 | 法条禁 | 1（pending·非 loss） |
| 4 | march 半宽 | 4 |
| 5 | 覆盖缺口(批变绿) | 8 |
| 6 | 墙记指错成本中心 | 7 |
| 7 | 真墙已走环 | 4 |
| 8 | 未攻/pending | 20 |
| **总** | | **46** |

- **手调档非 PASS 总数 = 46**（tier=手调 36 + opp-marker-only 10）。其中 具名-X（真 loss 标签）= **41**，pending/pending-fold（未定 verdict·非 loss）= **5**。
- **不该算输（CAT 1/2/3）= 3 格**。

---

## 三、★ CAT 1/2/3「不该算输」清单 + 关键发现

**结论：CAT 1/2/3 共 3 格，但 3 格全部已是 pending / pending-fold（无一是 具名-X loss）⟹ 摘掉它们不改变 具名-X loss 计数（41 不变）。** 「不该算输」的三条法条出口早已在这 3 格生效（未被误计为失败），审计只是复核确认。

### CAT 1（对手作废·ISSUE-064）★未核项已核 = 无实害
**★该核 VLEN128 上游是否本就不该被选中 → 答案：是，VLEN gate 已挡，且主表对手早已重定，无 phantom loss。**
仓内铁证 `experiments/archive/g5/g5-wiring/M2-iq4_nl/evidence.md:16`：
> dispatch `case 128: { break; } // TODO`（`:4680`）→ get_tensor_traits 返 nullptr → 板上 iq4_nl **恒走 stock block-dot**（`ggml_vec_dot_iq4_nl_q8_0`）。`case 256` 才 route trait。

- ISSUE-064 的数值错核 = iq4_nl **16x1 GEVM/GEMM repack 双核**（avl=16·VLEN128 下 vl 钳 8·511-512/512 mismatch）。
- 该 16x1 核在 rvv(VLEN128) 被 `case 128: break` **门死·从不派发**（部署恒 block-dot）。
- 主表 iq4_nl gemm@rvv 两格对手符号 = `ggml_vec_dot_iq4_nl_q8_0_vl128`（**block-dot 家族·tier=通用向量**·`_vl128` 后缀是命名产物·非 16x1 repack）⟹ **主表用的已是部署对手（block-dot），不是那个数值错的 16x1 核**。
- ⟹ **ISSUE-064 对本审计 loss 计数 = 0 影响**：坏核既未被 gate 选中、也未被主表当对手。两格现为 pending-fold（无 verdict·非 loss）。
- **[报用户裁]**：iq4_nl gemm@rvv 两格 tier=通用向量、对手是 block-dot（非手调）⟹ 严格说不属「手调档」；仅因 opp 符号含 `_vl128` 命名而入扫。建议主会话确认对手符号措辞（block-dot 不加 `_vl128` 后缀以免误读为手调）。

### CAT 2（数据脏·ISSUE-010）= 0 格命中手调 non-PASS
ISSUE-010 的 3 格（iq4_xs 57.21% / q5_K 44.32% / tq2_0 28.63% IQR）在主表状态：
- `vec_dot|iq4_xs` = **PASS** 双板（1.31/1.196）· `vec_dot|q5_K` = **PASS** 双板（0.843/1.065·tier=通用向量）· `vec_dot|tq2_0` = **PASS** 双板（0.975/0.879）。
- ⟹ 三格皆 PASS 或非手调 tier，**均不在手调档非 PASS 集** ⟹ CAT 2 对本审计命中 = **0**。（ISSUE-010 关切的是 PASS 判读脆弱，与「非 PASS 摘输」正交。）

### CAT 3（法条禁·ISSUE-029）= 1 格
`gemm|q8_0@ime`(k1)：厂商 IME 唯一 GEMM 核 = i8×i4（源码 `if constexpr` 仅 q4_0/q4_1/q4_K），q8_0 是 8-bit 权重进不了 i4 通路 → 结构上不存在合法 IME 对手。现 disp = pending（已按 ISSUE-029 写法·未误计失败）。**禁计我方失败，亦禁计胜利。**

---

## 四、★ 物理墙 / 结构墙标签逐格复核（[§三.15] 收严）

**扫全表 `{结构墙, 物理墙, 架构不可达, 内存墙, roofline, 内禀墙}` 标签**（机算·非窗口），手调 scope 内 4 格命中，**均为单侧 at-wall → 按 §三.15 属误标**：

| 格 | 板 | 现标签 | 单侧证据 | §三.15 复核 |
|---|---|---|---|---|
| gemm\|q2_K@rvv/decode | rvv | "结构墙·单世界clang也输" | 我方 vsetvl922 spill·对手 block-dot 更快 | **误标**：单侧·应 pending 或「对手结构优势具名」 |
| gemm\|q3_K@rvv/decode | rvv | "结构墙·单世界clang" | 我方 vsetvl2225 spill·对手不撞同墙 | **误标**：单侧 |
| gemm\|q4_K@rvv/decode | rvv | "指令数内禀墙" + 对手结构优势具名 | vsetvl1387·对手 native-vec block-dot 3.6×快 | 「对手结构优势具名」部分**合规**；"内禀墙"单侧部分误标 |
| gemm\|q6_K@rvv/decode | rvv | "结构墙·单世界clang" | 我方 vsetvl2461 spill·对手更快 | **误标**：单侧 |

- ISSUE-014 承重腿③（存在性证明）：同 M=1/同格式/同硅，对手 0.22ms vs 我方 0.60ms ⟹ **存在实现把同 fold 做快 2.77×** ⟹ 对手**不在同一 roofline 墙**上 ⟹ 单侧 at-wall 坐实 ⟹ "结构墙/内禀墙" 标签不成立（§三.15）。
- **合规样本**（非误标）：`iq4_nl gemm@k1/decode` 墙 = "codebook-gather-bound·对手 register-resident vrgather"（具名对手结构优势·双侧 objdump·§三.15 合规）；`q4_K vec_dot@rvv`(ISSUE-109) 墙 = "对手 register-resident·0 scratch"（具名对手结构优势·合规）；三个 IME 格墙 = "vendor scale-fold 融进 vmadot"（具名 vendor 结构优势·合规）。
- **手调 scope 内无一格标"物理墙 / 架构不可达"**（`dequantize_row|iq3_xxs@rvv` 有"架构不可达"字样但 note 实为「**【非架构不可达】**」且 tier=标量类·出 scope）。

---

## 五、★ q4_K / q6_K vec_dot 墙记核（特别核项 a·区分两墙）

**任务提问：K-quant vec_dot 现墙记是 stall-bound scratch-roundtrip（ISSUE-109·perf 证）还是 fold@M=1（ISSUE-014·非物理地板）—— 是否混淆？**

**结论：两者是不同 op 上的不同机制，未混淆；但 CAT 6 的 decode 现墙记（fold@M=1）确是错的成本中心，须订正。**

| 维度 | ISSUE-109（vec_dot） | ISSUE-014（gemm-decode） |
|---|---|---|
| 命中 op | K-quant **vec_dot**（q4_K@rvv 已证） | 8 格 K-quant **decode**（gemm） |
| 墙机制 | **weight-reconstruction scratch store→load roundtrip**（aux8[256]·8×vse8+68×vle8）+ scalar min-term serial MAC | 判定 = fold@M=1 **非物理地板**（纯算术地板 0.889–0.941 全 >0.8） |
| perf 证据 | **IPC 0.08·96.2% backend-idle·2.2B cache-miss = stall-bound**（板测 2026-07-18） | 四腿证 fold 在比值中相消·成本中心 = 未证·**留白** |
| 旋钮 | `integer_core_lmul` mf2/m1/m2 EXHAUSTED（best 0.186<<0.8） | — |
| lever | 结构级 **register-fusion**（独立大构造·待施工） | 攻击队列·待逐格走环 |

- **q4_K vec_dot@rvv**：完整攻坚环 **board-tested**（task `07-18-k-kquant-vecdot-attack` 已归档 · ISSUE-109 · `K-attack-fanout-ledger.md` 机制③行）⟹ **CAT 7 成立**。墙 = 对手结构优势具名（register-resident），**≠ fold@M=1**，**≠ 架构不可达**（in-principle 可达·register-fusion lever 存在）。
- **q6_K vec_dot@rvv**：★**PRD「q4_K/q6_K vec_dot 已走环」只对 q4_K 成立**。q6_K 现状（仓内核对）：
  - **baseline + oracle 已建**（`k-vecdot-harness/MANIFEST.md`：q4_K/q6_K oracle 板测 byte-exact ALL=true·baseline 具名-X 0.202）。
  - **完整攻坚环（旋钮扫 + IPC perf stat）未走**：task `07-18-07-18-k-q6k-vecdot-attack` 的 `implement.jsonl` / `check.jsonl` **仍为 `_example` 占位**（未启动）；`K-attack-fanout-ledger.md`(2026-07-18) 机制③行只记 q4_K board-tested，未列 q6_K。
  - ⟹ 归 **CAT 8**（环未完成·infra 已就绪·排队第一）。翻正预期 = 经环具名同 q4_K weight-reconstruction floor（诚实预期·禁外推）。
- **decode 群现墙记订正需求（CAT 6）**：主表 decode note 写「指令数内禀-fold@M=1不amortize」被 ISSUE-014 证伪（fold 单独产不出任何已观测 named-X·纯算术地板全 >0.8）。**成本中心须留白**（禁写"权重位重建"作定论）；旁证 `g7-l1-kernelsym-fullfill/rvv-batch/evidence.md:113` 将 q2/q3/q6_K@rvv 归 "weight-recon/spill-bound"（与 ISSUE-109 vec_dot 墙同族·但 decode 面 **未逐格 perf 证·禁外推**）。

---

## 六、★ 术语碰撞（ISSUE-007 / ISSUE-088）逐格标记 [报用户裁]

10 格 `tier≠手调 但 opp 符号手调标记`，tier 字段与对手身份**冲突**，须主会话/用户裁 tier：

| 格 | 板 | 现 tier | 对手符号 | 冲突 |
|---|---|---|---|---|
| gemm\|iq1_s@rvv | rvv,k1 | 标量类 | ..._vl128/vl256(CROSSOP) | note 自述「tier=S 暂留[ISSUE-007 待裁]」 |
| gemm\|iq1_m@rvv | rvv,k1 | 标量类 | ..._vl128/vl256(CROSSOP) | 同上 |
| gemm\|iq3_xxs@rvv | k1 | 标量类 | ..._vl256(CROSSOP) | 同上 |
| gemm\|iq3_s@rvv | k1 | 标量类 | ..._vl256(CROSSOP) | 同上 |
| gemm\|iq4_nl@rvv/{decode,prefill} | rvv,k1 | 通用向量 | ..._vl128/vl256(CROSSOP) | 对手实为 block-dot·`_vl128` 命名产物 |

> 这些是 CROSSOP（我方 repack-GEMM vs 对手 dispatched vec_dot）；tier 字段判「对手强度」由符号映射机判为标量类/通用向量，但对手符号带 `_vl` 手调标记 ⟹ ISSUE-007「暂留」。**保守默认**：入手调审计（PRD「对手=手调」定义），但 tier 归属**报用户裁**·禁 agent 自改。

---

## 七、「真待攻」结构（禁预判认输·仅排序）

- **不该算输（CAT 1/2/3）= 3 格**，且 3 格全为 pending/pending-fold ⟹ 摘掉后 **具名-X loss 计数 41 不变**（三条法条出口早已生效）。
- **已具名路 / 已走环（CAT 4/5/6/7）= 23 格**：有既定机制或已走环——
  - CAT 4（4 格）：mech① VLEN256 宽化·iq3_xxs@k1 **proven 0.65→1.38**·gated 裁决1/ISSUE-105。
  - CAT 5（8 格）：ISSUE-019 覆盖缺口·可按批变绿（先例封印 k1 满宽 1.197×）。
  - CAT 6（7 格）：ISSUE-014 墙记订正 + 逐格走环（成本中心留白）。
  - CAT 7（4 格）：已走完整环·具名墙（q4_K vec_dot / iq4_nl@k1 decode / q4_0@ime / q4_K@ime）·register-fusion / vendor-fusion lever 待裁。
- **真·未走环（CAT 8）= 20 格**：q6_K vec_dot@rvv（infra 就绪·排队第一）+ grid/ternary vec_dot 15 格 + mxfp4 pending×2 + iq1@rvv gemm×2。

> 以上仅排序，**不作「N 格认输」结论**（PRD §三·禁预判）。

---

## Caveats / Not Found

- **对手法.md 不存在**：PRD 引「对手法.md」实为 `.trellis/spec/canon/对手与档位.md`（已读·§一.1/1.5/1.9/三.13/15/二.11 全覆盖）。
- **cold 数字异源**：q6_K vec_dot@rvv 主表 0.18 vs census `vecdot-rvv/raw/rvv_run.log` cold_best 0.2601(M=1) vs harness baseline 0.202——异日期/方法·未 reconcile（非本审计域·只登记）。
- **归类为提案**：8 类存在真实重叠（CAT 5↔6 于 K-quant decode；CAT 5↔7 于 K-quant vec_dot；CAT 4↔5 于 iq/iq4_nl 半宽）。每格给**主归类 + overlap 注**，最终 tier/verdict 变更一律**报用户裁**（PRD「只出表不改 verdict」）。
- **iq4_nl 16x1 gate 证据来自 archive**（`g5-wiring/M2-iq4_nl/evidence.md`·非上游 ggml 源直读；本仓未 vendored 上游 dispatch .cpp）。gate 行 `:4680 case 128: break // TODO` 为该 evidence 转录·未即时复核上游源。
- **未改 verdict / tier / 墙记**：本文件为纯审计产出；research/ 外 0 改动·0 造数。
