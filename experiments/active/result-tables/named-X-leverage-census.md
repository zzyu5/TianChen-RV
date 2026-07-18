# §六 具名-X 杠杆台账（named-X leverage census）

> **交付格式素材**（《测试与收尾总令-开测篇》§六 + 补充令二 §六）。案头产物：**0 造数 · 只读现有 cold/note · 禁 build/bench/ssh/commit**。
> **数字权威** = `experiments/master/T3_master_rebuild.csv`（逐板拆 verdict cell）；**note/verdict 依据** = `.trellis/scripts/recon_master_rebuild.py` 各 dict（`TIER` / `COLD` / `GEMM_DECODE` / `P2_GRID4` / `CLANG_WORLD` / `IME_KERNELSYM`）；**lever 现状** = `experiments/active/result-tables/K-attack-fanout-ledger.md` + `.trellis/spec/issues/*`。
> **机算谓词**（可复跑）：读 CSV，取 `rvv_disp`/`k1_disp` 以 `具名-X` 起头的 board-cell（disp 是受控 token·非 note 子串·避 note 污染）。
> **四类**（补充令二 §六）：**(a) live-lever** 具名未试杠杆 / **(b) board-vacant honest-null** 杠杆真空（板测或结构墙推断） / **(c) gated** 部署 gated（ISSUE-105 / D-2a） / **(d) NEEDS-LEVER** 无 lever 名 ∧ 非 board-vacant ∧ 非 gated（§一/§六 违规·须暴露）。

---

## 一、首节两数

- **具名-X 总格数（逐板拆后）= 55 board-cells**（37 主表行 · 其中 18 行双板皆 X ⟹ 37+18=55；`grep -c 具名-X` = 44 只是**含 X 的行数**，非 board-cell 数，勿引作格数）。
  - 分布：vec_dot 25 · gemm-decode 11 · gemm-prefill/'' 9（内 IME 2 · P2-grid4 6 · iq4_nl-prefill 1）· dequant 10。
- **分类计数 = live-lever(a) 38 / board-vacant(b) 5 / gated(c) 1 / ★NEEDS-LEVER(d) 11**。
- **本轮消灭具名-X = 0**（本交付 = 台账盘点·非攻坚轮；消灭数属 K/R 线攻坚 task）。
- **★头条缺口 = 11 格 NEEDS-LEVER(d)**——无具名 live-lever ∧ 非 board-vacant ∧ 非 gated，是 §一「负结果不作『不用打』依据」/ §六「禁写已尽力除非杠杆清单已空」的**待暴露软认输面**。见 §三。

> 校验：38+5+1+11 = 55 ✓。

---

## 二、主表（一行一 board-cell · NEEDS-LEVER(d) 置顶）

列义：cold（`—`=CSV cold 空·多为 stale-T3 verdict 无冷计时）| verdict token | 类 | lever 名 | 源 ISSUE/dict | 成色注（对手身份·板测 vs 结构推断·成色）。

### 2.0 ★NEEDS-LEVER(d) — 11 格（§一/§六 缺口·置顶）

| op\|format | board | cold | 类 | 缺口 / 候选 lever | 源 | 成色注 |
|---|---|---|---|---|---|---|
| vec_dot\|iq1_s | rvv | 0.365 | **d** | 无 @rvv lever（VLEN256 是 k1-only·mechanism① 命中格列 iq1_s@rvv 但杠杆对 rvv 空转；gemm iq1_s@rvv 亦 0.6643 X ⟹ 非 system-covered） | TIER 手调 · mech① | 对手 vl128 grid-gather 手调 STRONG；grid 但 rvv 已满宽·无宽化杠杆 |
| vec_dot\|iq2_xxs | rvv | 0.697 | **d** | 无 registered @rvv lever（iq2 族不在 ISSUE-019/020/021/102 scope） | TIER 手调 | 对手 vl128.isra 手调（objdump 226/vec73/mac24）；iq2 grid @rvv 无杠杆名 |
| vec_dot\|iq2_xs | rvv | 0.529 | **d** | 同上（iq2 grid @rvv 无杠杆名） | TIER 手调 | 对手 vl128.isra 手调 |
| vec_dot\|tq1_0 | rvv | 0.207 | **d** | 无 registered lever（ternary·三值·非 grid·非 K-quant super-block·无任何 ISSUE 命中） | TIER 手调 | 对手 tq1_0_q8_K_vl128.isra 手调（objdump 114/vec45/mac14 ternary） |
| vec_dot\|tq1_0 | k1 | 0.606 | **d** | 无 registered lever（k1 或有 VLEN256 半宽 co-factor·但 tq 未 registered·未板测） | TIER 手调 | 对手 _vl256 手调；tq 结构未挂任何 lever |
| gemm\|iq1_s | rvv | 0.6643 | **d** | 无 @rvv lever（CROSSOP repack-GEMM·selector 已 decline·部署走 vec_dot；repack-leaf 无杠杆名） | P2_GRID4 | 部署对手 vl128 full-unroll（≈65× 指令量·vset storm）；便宜档 vs generic 15.2×【降披露】 |
| gemm\|iq1_m | rvv | 0.5713 | **d** | 无 @rvv lever（CROSSOP repack@rvv；ISSUE-020 只覆盖 vec_dot iq1_m·不覆盖 gemm repack） | P2_GRID4 | 部署对手 vl128 full-unroll（≈91×）；便宜档 vs generic 9.01×【降披露】 |
| gemm\|iq2_xxs | rvv | 0.60 | **d** | 无 lever（scalar-ref fallback CROSSOP·decode M=1·prefill 同格 PASS 但不 cover decode·[禁继承]） | GEMM_DECODE | decode near-parity·rvv named-X（单世界 clang）·便宜档-scalar-ref |
| gemm\|iq4_xs | rvv | 0.63 | **d** | 无 lever（scalar-ref fallback CROSSOP·decode M=1） | GEMM_DECODE | decode near-parity·便宜档-scalar-ref |
| gemm\|q4_0@ime | k1 | 0.196 | **d** | 无 registered lever（候选=scale-fold epilogue 融入 vmadot MAC·未挂 ISSUE） | IME_KERNELSYM | 对手 vendor 手调 IME `gemm_kernel_i8i4`（真硅 vmadot）·输 5.1×·**e2e q4_0@ime=tie-stock 1.0088× 不受影响**（赛道≠e2e·禁互推） |
| gemm\|q4_K@ime | k1 | 0.049 | **d** | 无 registered lever（同 epilogue-fusion·gap 随格式复杂度扩张 [PAT-1] 最锋利） | IME_KERNELSYM | 对手 vendor IME·输 20×·**e2e q4_K@ime=黄 0.909× 不受影响**·C3′ format-keyed 边界素材 |

### 2.1 gated(c) — 1 格

| op\|format | board | cold | 类 | lever | 源 | 成色注 |
|---|---|---|---|---|---|---|
| gemm\|iq3_xxs | k1 | 0.6474 | **c** | **VLEN256 width-widening**（proven·部署无通道） | ISSUE-105 / ISSUE-102 · K-ledger 机制① | **PROVEN WIN 1.38**（同叶 `march=rv64gcv_zvl256b` half_lanes 8→16·byte-exact·2seed 1.3838/1.3782）；deployed 叶仍 VLEN128=0.65 ⟹ master 维持 具名-X 0.6474·部署 gated on per-board fixture（GEN_SEAL 对 k1 发 VLEN128）·手调档硬赢·CROSSOP 系统账 |

### 2.2 board-vacant honest-null(b) — 5 格（grid dequant HW-gather 天花板·ISSUE-107）

| op\|format | board | cold | 类 | lever 状态 | 源 | 成色注 |
|---|---|---|---|---|---|---|
| dequant\|iq3_xxs | rvv | 0.36 | **b** | **杠杆清单空（construction 轴）** | ISSUE-107（裁决3·board-measured） | **3 owned 变体全 <0.8**：naive 0.181 / 候选②标量-load 0.33（clang-O3 re-gather 非连续 grid）/ HW-gather 0.36（best）；对手仅 via host-autovec-of-scalar-C=codegen 抽签达 parity；opp-immaturity·非 perf 硬赢 |
| dequant\|iq3_xxs | k1 | 0.526 | **b** | 同墙（structural-inference·owned 未测@k1） | ISSUE-107（grid 族同墙） | cold=stale autovec-lottery（非 owned·note 仅「scalar autovec」）；grid HW-gather 天花板板不变·结构推断 |
| dequant\|iq3_s | rvv | — | **b** | 同墙（structural-inference·未板测） | ISSUE-107（grid 族 iq3_s/iq2* 同墙） | cold 空·stale-T3；grid-codebook 索引查表撞同天花板·结构推断（禁外推具体数字） |
| dequant\|iq3_s | k1 | 0.665 | **b** | 同墙（structural-inference） | ISSUE-107 | cold=stale autovec；grid 同墙结构推断 |
| dequant\|iq1_m | rvv | — | **b** | 同墙（structural-inference·未板测） | ISSUE-107（grid 族） | cold 空·stale-T3；iq1 grid-codebook·同 HW-gather 天花板结构推断 |

> **honest-null 合法性**（补充令二 §六）：ISSUE-107 裁「grid dequant owned-construction 轴 = 真墙·杠杆清单空 for construction」——**唯一板测格 iq3_xxs@rvv 三变体证毕**，同族 iq3_s/iq1_m 是**结构墙推断**（非逐格板测·标清）。**非架构不可达**（格可 via lottery-PASS·但非 owned·[L-8]-vs-PASS 权衡待用户战略裁·ISSUE-107/106）。

### 2.3 live-lever(a) — 38 格（按 lever 族分块）

**(a-1) K-quant vec_dot weight-reconstruction floor — lever = `vwredsum.vs` per-sub-block 归约（ISSUE-109）· tier=手调**

| op\|format | board | cold | 板测状态 | 成色注 |
|---|---|---|---|---|
| vec_dot\|q4_K | rvv | 0.189 | **BOARD-TESTED** | register-fusion 已施工=EXHAUSTED（roundtrip 真消·IPC 2.1×·cache-miss 33×↓·byte-exact·但 cold 0.152 略慢·墙没动=latency/dependency-bound）⟹ 新 lever `vwredsum.vs` **未试**（清单非空·NOT 架构不可达）·对手 vl128 手调 STRONG register-resident |
| vec_dot\|q6_K | rvv | 0.18 | **BOARD-TESTED** | floor 泛化 2nd 数据点（q6_K 无 min-term 仍同 floor·best m1 0.214·IPC 0.10-0.12·stall-bound）·vwredsum.vs 未试 |
| vec_dot\|q2_K | rvv | 0.342 | structural-inference | 同 weight-reconstruction floor·未逐格证（ISSUE-109 命中面·禁外推）·对手 vl128 手调 |
| vec_dot\|q3_K | rvv | 0.258 | structural-inference | 同上 |
| vec_dot\|q2_K | k1 | 0.684 | structural-inference | 同 floor（板不变结构）+ k1 VLEN256 半宽 co-factor·lever 未板测@k1 |
| vec_dot\|q3_K | k1 | 0.522 | structural-inference | 同上 |
| vec_dot\|q4_K | k1 | 0.592 | structural-inference | 同上 |
| vec_dot\|q6_K | k1 | 0.549 | structural-inference | 同上 |

**(a-2) K-quant gemm-decode M=1 — lever = 真成本中心诊断待做（ISSUE-014·成本中心留白·非物理墙）[+ISSUE-109 位重建面 · +VLEN256 for k1]**

| op\|format | board | cold | tier | 成色注 |
|---|---|---|---|---|
| gemm\|q2_K/decode | rvv | 0.0685 | 手调 | 对手结构优势具名·fold@M=1 经 ISSUE-014 四腿证非物理地板（纯算术地板 0.889>0.8）·真成本中心**留白待诊断**·单侧 at-wall·C3′ 负 |
| gemm\|q3_K/decode | rvv | 0.083 | 手调 | 同（cost-center 留白·ISSUE-014） |
| gemm\|q4_K/decode | rvv | 0.361 | 手调 | genuine-C3′-negative·对手 native-vec block-dot 3.6× 快·成本中心留白（clang 世界值·CLANG_WORLD） |
| gemm\|q5_K/decode | rvv | 0.1273 | 通用向量 | 对手 q5_K native-vec（无 vl-spec）·cost-center 留白 |
| gemm\|q6_K/decode | rvv | 0.0535 | 手调 | 同（ISSUE-014 留白） |
| gemm\|q3_K/decode | k1 | 0.4252 | 手调 | 指令数内禀 fold@M=1 不 amortize（16-sub-block）+ VLEN256 半宽 co-factor·C3′ 负 |
| gemm\|q5_K/decode | k1 | 0.6806 | 手调 | 对手 repack `ggml_gemm_q5_K_8x4`·+VLEN256·C3′ 负 |
| gemm\|q6_K/decode | k1 | 0.3771 | 手调 | 对手 repack `q6_K_16x1`·+VLEN256·C3′ 负 |

**(a-3) k1 grid/iq 半宽欠用 — lever = VLEN256 width-widening（ISSUE-019/102）· 部署 gated ISSUE-105（结构推断·未逐格证）**

| op\|format | board | cold | 成色注 |
|---|---|---|---|
| vec_dot\|iq3_xxs | k1 | 0.192 | 同叶 proven 面（gemm 侧 1.38 WIN）·vec_dot micro 未单独封·VLEN256 结构推断·gated ISSUE-105 |
| vec_dot\|iq3_s | k1 | 0.2 | 同 VLEN256 半宽病·扇出未证·gated ISSUE-105 |
| vec_dot\|iq1_s | k1 | 0.361 | VLEN256 structural-inference·gated |
| vec_dot\|iq2_xxs | k1 | 0.612 | VLEN256 structural-inference·**iq2 未在 ISSUE-019 显式 scope·建议扩** |
| vec_dot\|iq2_xs | k1 | 0.474 | 同上（iq2 未 registered·结构推断） |
| vec_dot\|iq2_s | k1 | 0.562 | 同上（iq2 未 registered·结构推断） |
| vec_dot\|iq4_nl | k1 | 0.697 | ISSUE-019 iq4_nl 在 scope + ISSUE-021/026 码本·VLEN256·gated |
| gemm\|iq3_s | k1 | 0.6136 | VLEN256 扇出（proven iq3_xxs 的姊妹·likely 同 lever·未证）·gated ISSUE-105 |
| gemm\|iq1_s | k1 | 0.5919 | VLEN256 structural-inference·gated |
| gemm\|iq1_m | k1 | 0.5945 | VLEN256 structural-inference·gated（+ISSUE-020 for vec_dot 侧） |
| gemm\|iq4_nl/decode | k1 | 0.2493 | **ISSUE-021 codebook-gather-bound C4b**（tiny-codebook·16 项 vluxei vs 对手 vrgather·**禁以性能名义立项**·selector 已正确 decline·部署=block-dot）+ VLEN256 |
| gemm\|iq4_nl/prefill | k1 | 0.6025 | 同 ISSUE-021/019·codebook + VLEN256 |

**(a-4) iq3 vec_dot @rvv — system-covered（gemm 同格 PASS/WIN·mechanism①·micro M=1 artifact）**

| op\|format | board | cold | 成色注 |
|---|---|---|---|
| vec_dot\|iq3_xxs | rvv | 0.156 | gemm iq3_xxs@rvv = **PASS 0.9484**（同叶·系统账已 cover）·vec_dot M=1 无行摊销=micro artifact·rvv 已满宽 |
| vec_dot\|iq3_s | rvv | 0.213 | gemm iq3_s@rvv = **WIN 1.3483**·系统账已 cover·micro M=1 artifact |

**(a-5) iq1_m tiny-reduction — lever = sum2 per-group 符号和向量化（ISSUE-020·已就绪）**

| op\|format | board | cold | 成色注 |
|---|---|---|---|
| vec_dot\|iq1_m | rvv | 0.152 | 真瓶颈=32×8-element tiny vwredsum（隔离实测）·sum2 per-group 机制**代数证过+oracle 已建·发射器未改**·已就绪 |
| vec_dot\|iq1_m | k1 | 0.25 | 同 ISSUE-020 + k1 VLEN256 co-factor |

**(a-6) nvfp4 窄位宽 — lever = 窄位宽整数乘加（ISSUE-100·待施工·float re-roll 已证尽）+ FP4 codebook（mechanism⑤）**

| op\|format | board | cold | 成色注 |
|---|---|---|---|
| vec_dot\|nvfp4 | rvv | 0.565 | ISSUE-100：float 4-way re-roll+hoist 环已走尽（G2 板测 NULL 1.00×）·**残余「窄位宽整数乘加」lever 未试** ⟹ 回攻击队列（task `07-18-nvfp4-narrowint`）·对手 ggml 0 处 nvfp4 override=覆盖空洞非选弱 |
| dequant\|nvfp4 | k1 | 0.716 | **mechanism⑤** FP4 codebook-gather（vrgather 16-entry LUT·mxfp4 姊妹路已存 3.47× WIN 正锚）+ ISSUE-021 码本参数化·标量类硬门 |

**(a-7) 非-grid dequant owned 真向量 emit — lever = R线§四.1 de-lottery owned emit（ISSUE-001/mech④·非 grid 无 gather 墙）**

| op\|format | board | cold | 成色注 |
|---|---|---|---|
| dequant\|q4_K | rvv | — | K-quant dequant·非 grid·R线§四.1 owned emit 未落地（cold 空·stale-T3·interim per §二.5）·正锚 q8_0 0.838→2.33 owned PASS（非 grid 无天花板） |
| dequant\|q5_K | rvv | — | 同（non-grid·R线§四.1 owned emit 未落地·interim） |
| dequant\|iq4_xs | rvv | — | iq4_xs=16-entry LUT（非 grid tiny-codebook）·R线§四.1 owned emit + 码本·interim |
| dequant\|tq2_0 | rvv | — | ternary·非 grid·R线§四.1 owned emit·interim |

> **(a-7) 数据薄弱标注**：这 4 格 CSV cold 空、verdict 来自 stale-T3 token、note 仅「scalar autovec」——**verdict 依据薄·须 R线 owned emit 落地后板测重入账**（§二.5「dequant 落地前标 interim·不进头条」）。lever 名成立（R线§四.1），故列 (a) 而非 (d)。

---

## 三、★NEEDS-LEVER 缺口清单（最重要·§一/§六 待暴露软认输面）

**11 格无具名 live-lever ∧ 非 board-vacant honest-null ∧ 非 gated**。逐格列出 + 建议下一步（该补什么 lever / 该 board-test 什么）：

| # | 格 | cold | 为何是缺口 | 建议下一步 |
|---|---|---|---|---|
| 1 | vec_dot\|iq1_s@rvv | 0.365 | grid-gather 手调对手；VLEN256（mechanism①）是 k1-only·对 rvv 空转；gemm iq1_s@rvv 亦 0.6643 X ⟹ **非** system-covered（区别 iq3） | board-attack iq1_s@rvv vec_dot 找墙（tiny-reduction? gather? 是否同 iq1_m ISSUE-020 sum2 面）→ 定 live-lever 或 honest-null |
| 2 | vec_dot\|iq2_xxs@rvv | 0.697 | iq2 族**完全不在** ISSUE-019/020/021/102 任何 scope；@rvv 无宽化杠杆 | 登记 iq2 grid vec_dot lever（扇出/gather/tiny-reduction 解剖）→ ISSUE 立条 |
| 3 | vec_dot\|iq2_xs@rvv | 0.529 | 同 #2（iq2 grid @rvv 无杠杆名） | 同 #2（iq2 族一并解剖） |
| 4 | vec_dot\|tq1_0@rvv | 0.207 | **ternary 三值**·非 grid·非 K-quant super-block·非 iq codebook·**无任何 ISSUE 命中**·对手 vl128 手调 STRONG | board-attack tq1_0 vec_dot 解剖（ternary unpack↔compute floor?）→ 登记 lever |
| 5 | vec_dot\|tq1_0@k1 | 0.606 | ternary @k1·或有 VLEN256 半宽 co-factor·但 tq 未 registered·未板测 | 反汇编 tq1_0 leaf@k1 AVL 是否半宽 → 若是纳入 VLEN256 族·否则独立登记 |
| 6 | gemm\|iq1_s@rvv | 0.6643 | CROSSOP repack-GEMM@rvv·selector 已 decline（部署走 vec_dot）·repack-leaf 无杠杆名·非部署路 | 裁：登记 repack-leaf lever **或** 归 CROSSOP-非部署（系统账=vec_dot 格·免独立攻）→ ISSUE-021 §同域律候选 |
| 7 | gemm\|iq1_m@rvv | 0.5713 | 同 #6（CROSSOP@rvv）·ISSUE-020 只覆盖 vec_dot iq1_m·不覆盖 gemm repack | 同 #6 |
| 8 | gemm\|iq2_xxs@rvv/decode | 0.60 | scalar-ref fallback CROSSOP·decode M=1·同格 prefill PASS 但 [禁继承] 不 cover decode·无 lever 名 | 裁：decode M=1 便宜档-scalar-ref 是否值得攻（near-parity 0.60）→ 若不攻须**具名** honest-null（现无墙证据·不得默认「不用打」） |
| 9 | gemm\|iq4_xs@rvv/decode | 0.63 | 同 #8（scalar-ref CROSSOP decode·无 lever 名） | 同 #8 |
| 10 | gemm\|q4_0@ime | 0.196 | vendor IME 正面度量负结果·候选 lever=scale-fold epilogue 融入 vmadot MAC·**未挂 ISSUE**（ISSUE-029 只管 q8_0@ime） | 登记 IME epilogue-fusion lever ISSUE·**或** 明示「kernel-sym 赛道≠e2e·e2e tie 已 cover·kernel-sym 负结果具名冻结」 |
| 11 | gemm\|q4_K@ime | 0.049 | 同 #10·gap 随格式复杂度扩张（[PAT-1] format-keyed 边界最锋利·输 20×）·未挂 ISSUE | 同 #10（IME epilogue-fusion·e2e 黄 0.909 已 cover·登记 or 具名冻结） |

**缺口性质分三簇**（供 main 排批）：
- **簇 A（真新 lever 缺·2-5,10-11）**：iq2 grid vec_dot@rvv（#2,#3）、ternary vec_dot（#4,#5）、IME epilogue-fusion（#10,#11）——**格式族从未挂任何 ISSUE lever**·须登记新条或 board-attack 后具名 honest-null。
- **簇 B（regime/路由归属未清·1,6-9）**：iq1_s vec_dot@rvv（#1）、iq CROSSOP gemm@rvv（#6-9）——**lever 存否取决于是否算部署路**（selector decline repack·系统账可能已 cover）·须裁「独立攻 vs CROSSOP-非部署具名」。
- **共性**：全 11 格现状 = 有 verdict（具名-X）但**无「未试杠杆」名**·若不补则落入补充令二 §六禁写的「已尽力/经环具名」软认输。

---

## 四、杠杆族聚合（55 格 → 9 族 + 1 缺口簇 · 每族一攻坚入口）

| 族 | lever（攻坚入口） | 覆盖格数 | 现状 | 源 |
|---|---|---|---|---|
| **F1** | **min-term + scale bit-dance 向量化**（K-quant vec_dot·独立 Emission Plan；~~register-fusion~~ / ~~vwredsum.vs~~ 均板测 EXHAUSTED） | 8（q2_K/q3_K/q4_K/q6_K ×2 板） | **两 lever EXHAUSTED·真墙再订正=整核 scalar-heavy**（register-fusion 消 aux8 + vwredsum 消 serial 链·cold 都没动·ours 55 向量 vs 对手 105·min-term/scale 我方标量对手向量化）·**新 lever = 向量化 min-term/scale（未试·清单非空·对手存在性证可达）**·best m1 0.186；q4_K/q6_K@rvv 2 板测·余 6 结构推断 | ISSUE-109 · K-ledger 机制③ |
| **F2** | **真成本中心诊断**（K-quant gemm-decode M=1·fold 非物理地板·真 driver 待诊断） | 8（q2_K/q3_K/q4_K/q5_K/q6_K decode） | **待诊断**（成本中心留白·ISSUE-014 四腿证非物理墙·k1 侧叠 VLEN256/repack 对手） | ISSUE-014（[+109/102]） |
| **F3** | **VLEN256 width-widening**（k1 半宽欠用·宽 LMUL/大 AVL 填满 256b） | 13（iq*@k1 vec_dot+gemm+iq4_nl±） | **1 proven-gated**（iq3_xxs@k1 gemm 1.38 WIN·gated ISSUE-105）+ **12 结构推断扇出**（未逐格证·iq2 未在显式 scope·建议扩） | ISSUE-019/102/105 · 机制① |
| **F4** | **tiny-reduction sum2 per-group**（iq1_m·符号和向量化批处理） | 2（iq1_m vec_dot ×2） | **已就绪**（代数证过+oracle 已建·发射器未改） | ISSUE-020 |
| **F5** | **窄位宽整数乘加 + FP4 codebook-gather**（nvfp4·vrgather 16-LUT） | 2（nvfp4 vec_dot@rvv + nvfp4 dequant@k1） | **queued**（ISSUE-100 float 路已证尽·整数杠杆未试·task 已排；mxfp4 3.47× 正锚·**禁性能名义立项** ISSUE-021） | ISSUE-100/021/025 · 机制⑤ |
| **F6** | **grid HW-gather 天花板 = honest-null**（grid dequant owned-construction 轴杠杆空） | 5（iq3_xxs/iq3_s/iq1_m dequant） | **honest-null**（iq3_xxs@rvv board-measured 3 变体·余结构推断）·非架构不可达·[L-8]-vs-PASS 待战略裁 | ISSUE-107（连 106） |
| **F7** | **R线 dequant de-lottery owned 真向量 emit**（非 grid·无 gather 墙） | 4（q4_K/q5_K/iq4_xs/tq2_0 dequant@rvv） | **待施工**（R线§四.1·正锚 q8_0 0.838→2.33 PASS·4 格 cold 空·interim·须落地重测） | R线§四.1 · ISSUE-001/002 · 机制④ |
| **F9** | **iq3 vec_dot@rvv = system-covered**（同叶 gemm@rvv PASS/WIN·vec_dot M=1 micro artifact） | 2（iq3_xxs/iq3_s vec_dot@rvv） | **已 cover**（gemm iq3_xxs@rvv 0.9484 PASS / iq3_s@rvv 1.3483 WIN·系统账已达·micro M=1 无行摊销） | 机制① · 系统账 regime-split |
| **(F8)** | **iq4_nl codebook 参数化 C4b**（tiny-codebook gather·判别键=codebook 尺寸） | 0 新（2 格 iq4_nl gemm@k1 **已计入 F3**·此为叠加 lever) | **待施工**（禁性能名义·selector 已 decline·C3′ 能力键控判别键正例） | ISSUE-021/022/026 |
| **★D** | **NEEDS-LEVER 缺口簇**（无 registered lever） | **11**（见 §三） | **GAP·须暴露**（簇 A 真新 lever 缺 / 簇 B regime 归属未清） | §一/§六 违规面 |

> **交叠与计数**：非-(d) 44 格无交叠分解 = F1 8 + F2 8 + F3 13 + F4 2 + F5 2 + F6 5 + F7 4 + F9 2 = **44**；(F8) 的 iq4_nl gemm@k1 2 格已含于 F3（同格叠加 lever·不重计）。加 D 11 = **55**（校验以 §一 55 为准）。F3(VLEN256) 在 iq4_nl@k1（叠 F8）与 K-quant decode k1 侧（叠 F2）上是**共存 co-factor**·归族按主攻坚入口。

---

## 五、纪律与可追溯性声明

- **0 造数**：无新 cold / 无新计时；全部 cold 引自 CSV（`rvv_cold`/`k1_cold`）·note 引自 recon dict·lever 状态引自 ISSUE/ledger 实文。
- **禁外推**：board-measured（q4_K/q6_K@rvv floor·iq3_xxs@rvv dequant·iq3_xxs@k1 proven）与 **structural-inference**（同族未逐格证）**逐行标清**；结构墙推断保留·具体 cold 数字**不外推**至未测格。
- **cold 空格**：10 格 dequant 中 6 格 CSV cold 空（q4_K/q5_K/iq1_m/iq3_s@rvv/iq4_xs/tq2_0@rvv）= stale-T3 verdict token 无冷计时·标 interim·**须 main 补**（R线 owned emit 落地）。
- **可追溯锚**：CSV 行键 `op|format`（+`@engine`/`/regime`）· recon dict 名（`TIER`/`COLD`/`GEMM_DECODE`/`P2_GRID4`/`CLANG_WORLD`/`IME_KERNELSYM`）· ISSUE 号 · `K-attack-fanout-ledger.md` 机制①–⑤。
- **复跑谓词**：`python3 .trellis/scripts/recon_master_rebuild.py`（重铸 CSV）；本台账 board-cell 集 = CSV 中 `rvv_disp`/`k1_disp` 以 `具名-X` 起头者（55）。
