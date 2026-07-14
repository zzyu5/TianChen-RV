# G7 §L1 k1-batch — 货架A 全量 micro (K-quant@k1/VLEN256) + ★G 决胜局

> **性质**：**第二赛道 kernel-axis** 对称 micro A/B（T9 kernel-sym 台账派生·C3′ 证据·成色核实）。
> **与 perf-covered 系统账（9/83·recon 权威）永不混算**·**非 e2e beat·非 sealed 8-gate Win**·[NG-4] 纪律。
> **板**：k1 / SpacemiT-X60 / VLEN256 (vlenb=32) / core3 (L2 512KiB shared 0-3) / **clang-18.1.8 出货对称域** / gov=performance 1.6GHz。
> **约束遵守**：无 git · 无 schema/T8/ROADMAP 改动 · 无 rvv · board 纯 micro scratch（`/tmp/q4k_vl16_hb/` + `/tmp/kq236_k1/`）· 主树/build/governor/stock-lib(只读) 全未动 = **零 restore 需求**（纯 micro）。
> **kernel-axis micro**（e2e 冻结令）·对称 clang-18 双证·部署编译器 spill 轴(G1 精化②)·永不混算第二赛道·byte-exact。

---

## ★B — G 决胜局（q4_K@k1 SEALED vl=16 核 vs 真 16x1 hand-brick·kernel-axis micro 直测）

### 0. 争议 / 缘起（五复核遗留·[VERIFY-LADDER] canon line 129）
- **vl=8 kernel-sym 核**（`s6_q4K.c` md5 `90d454da`·vwmacc 2240）已由 b29c269c 证伪 vs 真 16x1 hand-brick = **0.622× LOSS**（hand-brick 快 1.61×）。
- **sealed Win-K1-VLEN 的 vl=16 核**（`s6_q4K_vl16_sealed.c` md5 `e437fd3b`·vwmacc 1120）**e2e 已证 1.085×/1.0876× byte-exact 击败真 hand-brick**·但 **kernel-axis micro 从未直测**（投影 ~1.2×）。
- 本决胜局 = **复用 `q4k_handbrick_driver.c`·换 ours .o 为 vl=16 sealed 核**·对称 clang-18·N=12·T-N·裁定双轴 or e2e-only。

### 1. 对手身份（机判·同 vl=8 resolve）
- 真 hand-brick `ggml_gemm_q4_K_16x1_q8_K`：`nm -D` → `00000000000abe58 T ggml_gemm_q4_K_16x1_q8_K`（k1 真出货默认·case256·VLEN256-native vl=16 专调）。
- 对称编译：stock `arch/riscv/repack.cpp` 由 `clang++-18 -O3 -march=rv64gcv_zfh_zvfh_zicbop_zihintpause`（compile_commands.json 证）；ours 亦 clang-18 → **对称·无 [CASE-COMPILER-ASYMMETRY]**。

### 2. objdump 结构 seal（vl=16 vs vl=8·机制根因）
| 核 | vsetvli | spill | vwmacc | textB | md5 |
|---|--:|--:|--:|--:|---|
| **ours vl=16 sealed -O3** | 35 | 3 | **1120** | 12928 | e437fd3b |
| **ours vl=16 sealed -O2** | 36 | 2 | **1120** | 12842 | e437fd3b |
| ours vl=8（对照·resolve） | 70 | 4 | 2240 | 25362/25304 | 90d454da |
| 真 16x1 hand-brick（stock lib） | ~34 | ~14 | ~80 | — | — |

**机制**：vl=16 sealed 核 **vwmacc 减半（2240→1120）·textB 减半（25KB→12.8KB）** = 真吃满 VLEN256 全宽（16 f32 lane·每 op 2× 元素→半 op 数）。vl=8 核只用 8 lane→2× op 数·输 hand-brick；vl=16 满宽调度追平并超过 hand-brick 的 vl=16 策略。**这正是 vl=8 输→vl=16 赢的分轴根因**（[VERIFY-LADDER] 双核分立·测错核之戒）。libcall-free（zfh/zvfh native fp16·0 `__extendhfsf2`）。

### 3. micro A/B（同 vx/vy·同 shape K=2048 nr=64 nc=512·对称 clang-18·N=12·3 seed·HOT+COLD·O2+O3）
**correctness cross-check**（证两侧算同一 GEMM·非空转）：`nbad=0/32768`（2 seed）· `nbad=1/32768`（seed 0xBEEF01·row23 near-zero cancel·max_rel 0.13 单元素·benign·同 vl=8）· `max_abs≈2.9e-3` bounded-ULP（FMA reassoc noise）· ours[0..last] == 16x1[0..last] → **vl=16 与真 16x1 byte-layout drop-in·输出 bounded-ULP 等价·全 nr=64 行匹配**。A/B 量的是真·全工作量。

**timing（12 ratio·全 HOT∧COLD∧O2∧O3∧3seed）**：
| 统计 | ours/16x1 ratio | ours GMAC/s | 16x1 GMAC/s |
|---|--:|--:|--:|
| 全 12 median | **≈ 1.197×** | ~7.13–7.20 | ~5.96–5.99 |
| min | 1.1928 | | |
| max | 1.2017 | | |
| range | <0.9% (IQR/round 多 <0.3%·1 cold seed 1.5%) | | |

- HOT ratios: 1.1946 1.1938 1.1970 1.2017 1.2010 1.1996 · COLD ratios: 1.1928 1.1938 1.1952 1.1987 1.2003 1.1994。
- **全 12 > parity**（sign-test p=2⁻¹²）·非重叠分布（ours cv & opp cv < 0.3%·ratio ≫ 1·min-A/B > 1）→ 决定性 **WIN**·非噪声。
- HOT≈COLD（1.197 vs 1.196·NULL 区分·compute-bound repack-GEMM + L2<tile）·匹配 §6 hot≈cold 铁律。

### 4. ★裁决（双轴 or e2e-only）
**vl=16 vs 16x1 = ≥parity（实为 WIN ~1.197× · kernel-axis micro）→ sealed = 【双轴 verified hand-brick win】。**

- Win-K1-VLEN（vl=16 sealed 核）现 **kernel-axis micro（~1.197×）∧ e2e（1.085×/1.0876× byte-exact）皆击败真出货 16x1 hand-brick**（VLEN256-native·对称 clang-18·真部署路径）。**成色升级**：这是**首个双轴 verified hand-brick win**。
- 与 §1.1 kernel-sym 台账 **不冲突·双核分立**：台账计数用的是 **vl=8 核**（q4_K@k1 vs block-dot 3.106×·vs 真 hand-brick 0.622× LOSS·不变）；**sealed vl=16 核**是分立核·本决胜局证其 vs 真 hand-brick 亦赢。
- 投影 ~1.2× **兑现**（实测 1.197×·精准命中）。收窄 T9 §1.1 line 43 遗留「vl=16 kernel-axis micro 待补测决定双轴 or e2e-only」= **RESOLVED → 双轴**。

---

## A — K-quant@k1 kernel-sym（q2_K/q3_K/q6_K @k1/VLEN256·hot/cold·变体穷举）

- **ours**：PLAIN 前门 export（kq_export_q6q2q3·md5 q2=`6c9322f6` q3=`f4cfb641` q6=`0f14791e`·symbol `tcrv_emitc_ggml_repack_gemm_qX_K_q8_K_kernel_...`）·-O2 canonical march。
- **对手（机判·public T 符号·nm -D）**：`ggml_vec_dot_q2_K_q8_K`(0xa034c) · `ggml_vec_dot_q3_K_q8_K`(0xa15da) · `ggml_vec_dot_q6_K_q8_K`(0xa21d2) = factory block-dot（**弱—中对手·weight-bound**·机判非手写类目）。对称 clang-18。
- **协议**：同形状 GEMM·K=2048·nc=512·hot(single tile median-of-12)/cold(P=8 pool 沖 L2)·nr∈{64,16,4} 穷举·3 seed。

### ★objdump SEAL（部署编译器 spill 轴·G1 精化②·机制根因）
| ours 核 -O2 | vsetvli | **spill** | vwmacc | textB | 形态 |
|---|--:|--:|--:|--:|---|
| q2_K PLAIN | 77 | **627** | 2304 | 51276 | full-unroll |
| q3_K PLAIN | 13 | **1058** | 2176 | 78870 | full-unroll |
| q6_K PLAIN | 21 | **971** | 2304 | 78288 | full-unroll |
| （对照 q4_K sealed vl=16） | 36 | **2** | 1120 | 12842 | S6+满宽 |

**机制根因**：q2/q3/q6 PLAIN = **全展开·regfile spill 627/1058/971**（v-reg 预算爆·[GAP-P1] register-pressure trap·性能宪章规则1 = K-quant repack 慢=全展开 regfile spill·非指令微质量）。对照 q4_K sealed vl=16 spill=2·q5_K@k1 S6-tiled spill 收拢 → 那两格才 ≥parity。**q2/q3/q6 的 S6 tiling：q3/q6 = NULL（未收拢 spill·故留 PLAIN）·q2 有 tiled 变体（md5 9c5bac28·spill 收拢）但本测【未含】**（仅 PLAIN·见变体穷举注）。

### hot/cold + 折中态 + 穷举（板测·N=12·T-N·load-gate·loadavg_begin=2.48）
| 格@k1 | 对手符号(机判) | 折中态 | HOT nr64 | COLD nr64 | nr16 | nr4 | cold≥parity | regime |
|---|---|---|--:|--:|--:|--:|:--:|---|
| **q2_K** | ggml_vec_dot_q2_K_q8_K | full-unroll spill 627 | **0.602×** | 0.601× | 0.625× | 0.551× | ❌ | weight-bound（spill）|
| **q3_K** | ggml_vec_dot_q3_K_q8_K | full-unroll spill 1058 | **0.344×** | 0.344× | 0.345× | 0.340× | ❌ | weight-bound（spill）|
| **q6_K** | ggml_vec_dot_q6_K_q8_K | full-unroll spill 971 | **0.304×**(nr64) | 0.304×(nr64) | —※ | —※ | ❌ | weight-bound（spill）|

※ q6_K nr16/nr4 = **未测**（板 kq236 nohup 于 q6_K nr64 后经主会话裁定 kill·q6_K nr64 三 seed 全 ~0.304× <parity 已足证 verdict·nr16/nr4 同族同机制 projected <parity·不影响判定）。

**★A 总结论（三格全 <parity·不入 kernel-sym ≥parity 计数）**：q2_K 0.55–0.62× · q3_K 0.34× · q6_K 0.30× · **全 nr 全 HOT≈COLD**（IQR <0.3%·`sink=nan` 仅 anti-DCE 累加器溢出·非空转·timing 真·correctness 构造 oracle 另封）。**VLEN256 【不救】 weight-bound full-unroll spill 格**（同 rvv 向：rvv q2_K 0.386×/q3_K ~0.18×/q6_K ~0.18× 对称-gcc → k1 0.60/0.34/0.30× 对称-clang·皆 <parity·**跨板不可比·各自对称域内 <parity 一致**）。**与 q5_K@k1 翻正（1.916× ≥parity）的关键差 = q5_K 是 S6-tiled（spill 收拢）· q2/q3/q6 是 PLAIN full-unroll（spill 627/1058/971 爆·[GAP-P1]）**——**判别键 = 核形态（tiled vs full-unroll spill）·非板/VLEN**。folding：q3_K/q6_K 更慢（0.34/0.30 < q2_K 0.60）= 更高 spill（1058/971 > 627）+ 更重 weight-recon（q3 3-bit subtractive-hmask / q6 6-bit dual-plane·per-element decode 更贵）。

### 变体穷举（令 §二.4）
- **合法 codegen 候选集（VLEN256）= VLA mf/mf2 单一**（VLEN-invariant·t4a 证 k1 vtype 描述符 == rvv128·[repack-winA-always-mf2]）→ selector 平凡·无宽度旋钮可扫。
- **可枚举变体轴 = ① nr-shape {64,16,4}**（本测·全 <parity·nr↓ 未翻）· **② tile 形态**：q2_K PLAIN(本测) vs S6-tiled(md5 9c5bac28·spill 收拢·**未测·selector 现选此 tiled 者**)·q3/q6 S6=NULL（PLAIN 即 shipped）。→ **q2_K 若测 tiled 变体或可改善（spill 收拢）·但 q3/q6 无此杠杆（tiling NULL）**·主会话可续 q2_K-tiled@k1 补测。

---

## 污染 / restore
- **board**：仅写 scratch `/tmp/q4k_vl16_hb/` + `/tmp/kq236_k1/`（ephemeral·driver+.o+bins+run.log）。**主树 / build / stock ggml .so（只读链接）/ governor 全未改** = **零 restore**（纯 micro）。stock lib md5 未动（只读 nm/objdump/-l 链接）。
- **ours md5**：sealed vl=16 = `e437fd3b`（板上 md5 双证·no regen）· q2/q3/q6 = kq_export cached（板上 md5 双证）。ours .o 全 libcall-free。
