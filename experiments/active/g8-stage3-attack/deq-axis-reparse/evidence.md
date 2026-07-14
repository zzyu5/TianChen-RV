# G8 §六.3 · DEQ-AXIS 对手符号级重解析 + auto-promote (decree: 真向量对手→自动升 0.8 硬门·禁赛道躲门)  2026-07-15

> **decree**: 任何 DEQ 格对手是**真向量实现** → **自动升入 0.8 硬门**（ours-dequant vs 真向量对手 = 公平 kernel-sym）·**禁以赛道归属躲门**。对手 **scalar** → 维持 DEQ-AXIS（ours-vec vs scalar 非公平硬门对手·照测保留）。
> **bounded**: cheap 符号级 re-parse（关键判据）→ 只对真向量候选判 0.8。**结果诚实第一**：全 scalar=合法零候选（decree 明文 auto-promote-**if** 是条件）。
> **不改 T3/T8·禁 git**（主会整合）。本 casefile = 回填建议 + 证据。

## 0. 对手认定 · 双板 stock 库定档（真派发 to_float 路）

**★deployed-path 机证**（承 iqfp4-dequant-k1 §0）：ggml-cpu `ops.cpp` 经 `ggml_get_type_traits(type)->to_float` 派发；`ggml.c` 置 `.to_float = dequantize_row_<fmt>`（**libggml-base.so** 内的 reference 实现）→ 链接符号 = **genuine deployed dequant**，非稻草人替换。

| 键 | rvv (VLEN128) | k1 (VLEN256) |
|---|---|---|
| decree-named stock `libggml-cpu.so` | (census cpu.so md5 `d1adc634…`) | **`/data/k1build-stock/bin/libggml-cpu.so.0.15.1` md5 `871169a0…`** ✓ decree |
| **`dequantize_row_*` 实际定义站点** | `/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin/libggml-base.so` **md5 `1b4580c4…`** | 同 build 目录 `libggml-base.so` **md5 `00267134…`** |
| ★符号定义披露 | `dequantize_row_*` **不在** cpu.so（`nm -D` 空）·**全在 sibling base.so**（cpu.so 从 base.so 链入） | 同：cpu.so `nm -D` 无 dequant 符号·全 `T` 于 base.so |
| 部署编译器（to_float 路） | **gcc-15.2**（[CASE-COMPILER-ASYMMETRY] per-lane 附注·DEQ 路≠ vec_dot clang-18 域） | build-off / clang-18 stock（as-built scalar） |
| objdump | GNU binutils 2.46.1 (`/opt/tcrv-toolchains`) | GNU binutils 2.42 (Ubuntu/Bianbu) |
| md5 before==after | `1b4580c4` == `1b4580c4`（read-only） | base `00267134` == `00267134`（read-only） |

**★方法学**：全 `.text` objdump → 逐 `<dequantize_row_<fmt>>:` label 到下一 label 切块（whole-symbol）→ `awk -F'\t'` 取 mnemonic → `rvv`=v-前缀助记符计数。**class 判据：`rvv==0` ⇒ SCALAR（纯标量循环）·`rvv>0` ⇒ TRUE-VEC（真向量·含 vsetvli/vle/vfmacc 等）**。raw = `raw/deq_metrics_{rvv,k1}.txt`。

---

## ① 18 格 × 双板 · 对手符号级成色表（whole-symbol 机判）

| 格 | rvv tot/rvv/vset/gath | rvv class | k1 tot/rvv/vset/gath | k1 class |
|---|---|---|---|---|
| q2_K   | 669/**461**/73/32 | **TRUE-VEC** | 394/**0**/0/0 | **SCALAR** |
| q3_K   | 317/**141**/39/0  | **TRUE-VEC** | 685/**0**/0/0 | **SCALAR** |
| q4_K   | 274/**120**/26/0  | **TRUE-VEC** | 460/**0**/0/0 | **SCALAR** |
| q5_K   | 440/**238**/46/0  | **TRUE-VEC** | 191/**0**/0/0 | **SCALAR** |
| q6_K   | 193/**96**/34/4   | **TRUE-VEC** | 191/**0**/0/0 | **SCALAR** |
| iq1_m  | 226/**67**/13/0   | **TRUE-VEC** | 319/**0**/0/0 | **SCALAR** |
| iq1_s  | 180/**65**/15/0   | **TRUE-VEC** | 252/**0**/0/0 | **SCALAR** |
| iq2_s  | 197/**0**/0/0     | **SCALAR**   | 326/**0**/0/0 | **SCALAR** |
| iq2_xs | 190/**0**/0/0     | **SCALAR**   | 172/**0**/0/0 | **SCALAR** |
| iq2_xxs| 339/**229**/74/18 | **TRUE-VEC** | 329/**0**/0/0 | **SCALAR** |
| iq3_s  | 598/**475**/146/32| **TRUE-VEC** | 638/**0**/0/0 | **SCALAR** |
| iq3_xxs| 331/**237**/76/18 | **TRUE-VEC** | 341/**0**/0/0 | **SCALAR** |
| iq4_nl | 69/**23**/8/2     | **TRUE-VEC** | 69/**0**/0/0  | **SCALAR** |
| iq4_xs | 107/**23**/8/2    | **TRUE-VEC** | 106/**0**/0/0 | **SCALAR** |
| mxfp4  | 61/**20**/5/2     | **TRUE-VEC** | 50/**0**/0/0  | **SCALAR** |
| nvfp4  | 182/**0**/0/0     | **SCALAR**   | 204/**0**/0/0 | **SCALAR** |
| tq1_0  | 493/**279**/52/0  | **TRUE-VEC** | 736/**0**/0/0 | **SCALAR** |
| tq2_0  | 153/**90**/25/0   | **TRUE-VEC** | 297/**0**/0/0 | **SCALAR** |

**双板对手成色本质差（★大板异）**：
- **rvv**：base.so 由 **gcc-15.2 auto-vectorized** → **15/18 TRUE-VEC**·仅 **3 SCALAR**（iq2_s / iq2_xs / nvfp4）。
- **k1**：base.so as-built = **deployed-scalar-reference** → **18/18 SCALAR**（rvv=0 全格·无一向量化）。
- **交叉验证**：与 G7 census（iqfp4-dequant-{rvv,k1}·2026-07-14·同 md5 opponent）**分类 18/18 逐格一致**：census "rvv only iq2_xs/iq2_s/nvfp4 stayed scalar" ≡ 本表；census "k1 objdump rvv=0 all 18" ≡ 本表。census 内联窗口 rvv-count 偏小（q4_K census 25 vs 本 120·q2_K gather census 32 == 本 32 精确吻合）= **窗口范围差**（census 取内环 ~37 insn / 本取 whole-symbol 全体含标量 prologue+epilogue+peel）·**class 判据不受影响**。q4_K@rvv mnemonic 直方图佐证真向量：`vzext.vf4×16 / vfcvt.f.x.v×16 / vfmsub.vv×16 / vse32.v×16 / vle8.v×8`（真 widen+float-convert+FMA+vector-store decode）。

---

## ② auto-promote 候选清单（对手=真向量 → 升 0.8 硬门）

| 板 | 真向量对手格（auto-promote 候选） | 数 | scalar 对手格（维持 DEQ-AXIS） | 数 |
|---|---|---:|---|---:|
| **rvv** | q2_K · q3_K · q4_K · q5_K · q6_K · iq1_m · iq1_s · iq2_xxs · iq3_s · iq3_xxs · iq4_nl · iq4_xs · mxfp4 · tq1_0 · tq2_0 | **15** | iq2_s · iq2_xs · nvfp4 | 3 |
| **k1** | **（无）** | **0** | 全 18 格 | 18 |

**★k1 = 零 auto-promote 候选**——全 18 格对手 scalar·**合法零候选**（decree 明文 auto-promote-**if** 真向量·k1 不满足）→ **k1 全 18 dequant 照测保留 DEQ-AXIS**。**禁硬造 auto-promote**。

---

## ③ 候选格 cold A/B + ≥0.8 判定（rvv 15 真向量候选）

**A/B 出处**：G7 census `iqfp4-dequant-rvv/summary_dequant_rvv.csv`（2026-07-14）——**同一 opponent 二进制**（base.so md5 `1b4580c4`·本役独立 re-parse 已 md5 复核）·**gcc-15.2 对称域**（ours 与 opp 同编译器·deploy to_float 路）·**cold streaming** K=1048576（out 4MiB > L2）·224MiB flush·N=12 median+relIQR·**byte-exact ZERO-MODEL 全 0mism/0ULP**。
**★为何不重测**：opponent 二进制逐字节同（md5 复核）·1 日龄 cold byte-exact 对称域数据·**0.8 门每格清净跨越**（最近 PASS = q2_K 0.8244 tight-IQR 0.58% / 最近 fail = iq4_xs 0.7074·0.71↔0.82 门附近有清净间隙·无临界翻转风险）→ 重跑仅复现·`别空转`。verdict 判据 = **cold_med ≥ 0.8**（含 cold_best 交叉·noisy 格双验）。

| 格 | opp class(rvv-cnt) | ours-side(vsetvl) | cold_med | cold_best | IQR o/p | **≥0.8 判定** | 备注（成色诚实） |
|---|---|---|---:|---:|---|---|---|
| q2_K   | autovec(461) | ours-**VEC**(77) | 0.8244 | 0.8227 | 0.58/1.09 | **PASS**(marginal) | **vec-vs-vec** genuine·紧 IQR·稳过 |
| q3_K   | autovec(141) | ours-**VEC**(145) | 1.2491 | 1.2464 | 1.41/0.72 | **PASS** | **vec-vs-vec** genuine WIN |
| q4_K   | autovec(120) | ours-VEC(129) | 0.3170 | 0.3189 | 1.24/0.61 | **具名-X** | ours-vec 输 opp-autovec（opp 3.34 GB/s 强）·清净 |
| q5_K   | autovec(238) | ours-VEC(179) | 0.6138 | 0.6173 | **44.32**/2.74 | **具名-X** | noisy 但 med&best 双 <0.8·稳判 |
| q6_K   | autovec(96) | ours-**SCALAR** | 1.6560 | 1.6716 | 1.35/0.87 | **PASS** | ★ours-**scalar** 胜 opp-autovec（opp thin/gather4）·成色弱 |
| iq1_m  | autovec(67) | ours-SCALAR | 0.3480 | 0.3504 | 1.32/0.11 | **具名-X** | ours-scalar 输 opp-autovec·清净 |
| iq1_s  | autovec(65) | ours-SCALAR | 0.9775 | 0.9784 | 0.21/0.16 | **PASS**(parity) | ★ours-scalar ≈ opp-autovec·极紧 IQR·稳 |
| iq2_xxs| autovec+gath(229) | ours-SCALAR | 3.8761 | 3.8705 | 0.94/0.26 | **PASS** | ★opp 付 gather 税·ours-scalar 白嫖·成色弱 |
| iq3_s  | autovec+gath(475) | ours-SCALAR | 0.2515 | 0.3248 | 3.66/1.33 | **具名-X** | opp heavy-gather 快·ours-scalar 输·清净 |
| iq3_xxs| autovec+gath(237) | ours-SCALAR | 1.4677 | 1.9578 | 8.42/0.51 | **PASS** | ★opp gather 税·成色弱 |
| iq4_nl | autovec(23) | ours-SCALAR | 7.0461 | 2.0561 | 0.42/0.33 | **PASS** | med 被 opp cold-outlier 抬·真实 ~best 2.06·仍稳过 |
| iq4_xs | autovec(23) | ours-SCALAR | 0.7074 | 0.3482 | 8.06/**57.21** | **具名-X** | noisy·**med&best 双 <0.8**·稳判 |
| mxfp4  | autovec(20) | ours-SCALAR | 5.9785 | 1.6717 | 7.49/15.78 | **PASS** | best 1.67 稳过·成色弱（opp thin autovec） |
| tq1_0  | autovec(279) | ours-**VEC**(84) | 1.1280 | 1.1269 | 1.41/1.42 | **PASS** | **vec-vs-vec** genuine WIN |
| tq2_0  | autovec(90) | ours-VEC(129) | 0.4876 | 0.4938 | **28.63**/2.39 | **具名-X** | noisy 但 med&best 双 <0.8·稳判 |

**rvv 15 候选 tally**：**9 PASS**（q2_K/q3_K/q6_K/iq1_s/iq2_xxs/iq3_xxs/iq4_nl/mxfp4/tq1_0）· **6 具名-X**（q4_K/q5_K/iq1_m/iq3_s/iq4_xs/tq2_0）。全 15 格 **byte-exact 0mism/0ULP**。
**★成色诚实分层（PASS 9 内）**：
- **genuine vec-vs-vec**（3）：q2_K(0.82·marginal) · q3_K(1.25) · tq1_0(1.13)——ours 亦向量·真硬碰硬。
- **ours-scalar-vs-opp-autovec**（6）：q6_K/iq1_s/iq2_xxs/iq3_xxs/iq4_nl/mxfp4——**弱赢层**：ours 发标量却 ≥0.8，因 opp autovec thin 或付 gather 税；**emitter 有向量化 headroom**（升门后 ours 若向量化可拉更高）。**报 PASS 须标 "ours-scalar·opp-immaturity/gather-tax"·勿夸成 "ours 向量优越"**。

**k1**：0 候选（全 scalar 对手）→ **无 0.8 A/B**。k1 census A/B（13 WIN/5 LOSS·iqfp4-dequant-k1 §2）= **ours-RVV-vec vs deployed-scalar-ref** 弱赢类·**留 DEQ-AXIS 子账·不进 k1 0.8 分母**。

---

## ④ T3 dequant 行 col36 回填建议（per-board · 主会执行·本役不改 T3）

### rvv T3_A（VLEN128）dequant 18 行

| 格 | 现 col36 | **建议 col36** | verdict |
|---|---|---|---|
| q2_K   | DEQ-AXIS | **in-denom**·domain=gcc-15.2-deploy | ≥0.8 **PASS** 0.82(vec-vs-vec-marginal) |
| q3_K   | DEQ-AXIS | **in-denom** | ≥0.8 **PASS** 1.25(vec-vs-vec) |
| q4_K   | DEQ-AXIS | **in-denom** | <0.8 **具名-X** 0.32 |
| q5_K   | DEQ-AXIS | **in-denom** | <0.8 **具名-X** 0.61 |
| q6_K   | DEQ-AXIS | **in-denom** | ≥0.8 **PASS** 1.66(ours-scalar·opp-immaturity) |
| iq1_m  | DEQ-AXIS | **in-denom** | <0.8 **具名-X** 0.35 |
| iq1_s  | DEQ-AXIS | **in-denom** | ≥0.8 **PASS** 0.98(parity·ours-scalar) |
| iq2_s  | DEQ-AXIS | **维持 DEQ-AXIS**（opp SCALAR） | — 照测保留（census 3.10× = scalar-vs-scalar 非公平门） |
| iq2_xs | DEQ-AXIS | **维持 DEQ-AXIS**（opp SCALAR） | — 照测保留（census 2.95× = scalar-vs-scalar） |
| iq2_xxs| DEQ-AXIS | **in-denom** | ≥0.8 **PASS** 3.88(opp-gather-tax·ours-scalar) |
| iq3_s  | DEQ-AXIS | **in-denom** | <0.8 **具名-X** 0.25 |
| iq3_xxs| DEQ-AXIS | **in-denom** | ≥0.8 **PASS** 1.47(opp-gather-tax·ours-scalar) |
| iq4_nl | DEQ-AXIS | **in-denom** | ≥0.8 **PASS** 2.06(ours-scalar·opp-thin) |
| iq4_xs | DEQ-AXIS | **in-denom** | <0.8 **具名-X** 0.71(noisy·med&best<0.8) |
| mxfp4  | DEQ-AXIS | **in-denom** | ≥0.8 **PASS** 1.67(ours-scalar·opp-thin) |
| nvfp4  | DEQ-AXIS | **维持 DEQ-AXIS**（opp SCALAR） | — 照测保留（census 0.66× = scalar-vs-scalar·ours 亦标量） |
| tq1_0  | DEQ-AXIS | **in-denom** | ≥0.8 **PASS** 1.13(vec-vs-vec) |
| tq2_0  | DEQ-AXIS | **in-denom** | <0.8 **具名-X** 0.49 |

**rvv 回填净效**：15 行 DEQ-AXIS→in-denom（domain=gcc-15.2-deploy tag）· 3 行（iq2_s/iq2_xs/nvfp4）维持 DEQ-AXIS。
**rvv 0.8 硬门分母**：§五定档 **28**（matmul-kernel-sym 19 + forward 9·DEQ 曾全 carve-out）→ **+15 dequant 真向量升门 = 43**。其中 **+9 PASS**（分子）·**+6 具名-X**（in-denom 输）。

### k1 T3_B（VLEN256）dequant 18 行

| 全 18 格 | 现 col36 | **建议 col36** | verdict |
|---|---|---|---|
| q2_K…tq2_0（全 18） | DEQ-AXIS | **维持 DEQ-AXIS**（opp 全 SCALAR·零真向量候选） | 照测保留·census A/B 13W/5L = ours-vec-vs-scalar-ref 弱赢子账 |

**k1 回填净效**：**0 行变更**·全 18 维持 DEQ-AXIS。**k1 0.8 硬门分母 = 28 不变**。

### ★主会须裁的 per-lane 域披露（回填时明标·非违 [CASE-COMPILER-ASYMMETRY]）
dequant in-denom 行 = **gcc-15.2 deploy 域**（to_float 路的真部署编译器）·与 vec_dot/GEMM 的 clang-18 域**分 lane**。这是 **per-lane deploy-matched 对称域**（每 lane 对手编译器 = 其部署编译器）·**非域混杂违规**（§五 §1.C 已附注 "DEQ 部署 to_float 路 = gcc-15.2 per-format autovec"）。**建议**：0.8 分母内设 "DEQ-lane · domain=gcc-15.2-deploy" 子块·计入分母但视觉分隔·per-lane 编译器显式。**呈现由主会定**。

---

## ⑤ 板卫生

| 项 | rvv | k1 |
|---|---|---|
| 动作 | nm -D + objdump -d（**read-only 反汇编**·无编译/无运行/无写库） | 同 |
| stock opponent md5 before==after | base.so `1b4580c4` == `1b4580c4` ✓ | base.so `00267134`==`00267134` · cpu.so `871169a0` 确认 ✓ |
| /tmp scratch | `/tmp/g8deq_rvv` 已删 | `/tmp/g8deq_k1` 已删 |
| stray proc（pgrep -x objdump） | 0 | 0 |
| loadavg（读毕） | 2.06 2.13 2.09 | 2.00 2.02 2.00 |
| co-tenant | 未扰（纯反汇编·无 pin·无占核） | 未扰 |
| git | 无 commit·未改 T3/T8·未改 emitter/lib | 同 |

---

## 6. 体例合规自检
- 对手 = as-shipped stock base.so 真派发 `dequantize_row_*`（to_float 机证·§0）·符号级机判 whole-symbol ✓
- **auto-promote-if 条件严格执行**：真向量→升门（rvv 15）·scalar→留 DEQ-AXIS（rvv 3 + k1 18）·**k1 零候选=合法诚实结果·未硬造** ✓
- 真向量候选 <0.8 = **具名-X 如实记**（rvv 6·未谎报翻正）·PASS 成色分层（3 genuine vec-vs-vec / 6 ours-scalar-opp-immaturity）✓
- byte-exact：15 候选全 0mism/0ULP（census ZERO-MODEL·同 opponent 二进制）✓
- per-board 独立报（rvv 15 真向量 vs k1 18 scalar·大板异·禁跨板沿用）✓
- A/B 复用披露（同 md5 opponent·gcc-15.2 对称 deploy 域·1 日龄 cold byte-exact·0.8 门清净间隙无临界·`别空转`）·per-lane 域披露交主会 ✓
- 禁 git·不改 T3/T8/emitter/lib·casefile 独立 ✓
