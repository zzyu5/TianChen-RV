# G8 阶段三 · §六 · k1 板全量 cold A/B 重测 (0.8 硬门 baseline)  2026-07-14

> **性质**：§五 对手重解析定档 (`aab4168c`) 之后的 **cold A/B 实测**。clang-18 对称域·N≥10·median·冷启动 flush·load-gate·core0-3 pin·byte-exact 门。回填 `T3_B_board_B_rvv1.0_vlen256.csv` perf 列（禁另建账本）。
> **禁 git · 不改 emitter/lib（纯测量·复用已 seal kernel）· e2e 冻结（kernel-axis cold A/B 非 e2e）**。

## 0. 环境 · 对手树 · provenance (k1 flag 处置)

| 键 | 值 |
|---|---|
| board | Spacemit X60 · VLEN256 (vlenb=32) · 8c · gov=performance · 1.6GHz · sv39 · **ime present** |
| ours 编译器 | `/usr/bin/clang` = Bianbu clang 18.1.8 (`11bb4`) |
| ours march (定稿) | `-O2 -march=rv64gcv_zfh_zvfh_zba_zbb_zbc_zbs -mabi=lp64d -ffp-contract=on` (G8 stage1 canonical) |
| **对手** | as-shipped **真派发** kernel·**clang-18-built stock `libggml-cpu.so`**（bclass seal 实证 stock=clang-18·= 对称域）·逐格 §五 定档符号 |
| stock .so **provenance=so-hash**（★flag②：树无 .git 不可 pin） | `/data/k1build-stock/bin/libggml-cpu.so` **md5=871169a0123139692177468b3c8578be**（symlink→.so.0·Jun26）· base md5=00267134 |
| ★flag① **-fno-integrated-as** | 该 flag 治 **对手-TU 重编**（quants.c policy-less inline-asm）。本役 **DISPATCH as-shipped stock .so**（不重编 quants.c）+ ours 纯 intrinsic（无 inline-asm）→ integrated-as·**codegen-neutral**。flag 已如实处置（非载重于本 dispatch 路径） |
| pin / hygiene | core0-3 pick-idle (每役 load-gate ≥70%·实测 pin core0/core1 idle 97-100%) · co-tenant(多 user) 未扰 · loadavg 2.4→2.9(全程<3·8c) |
| IME carve-out | q4_0/q8_0/q4_K @ime 3 格 **不进 0.8 分母**（gcc-13 `xsmtvdotii1p0`·clang-18 拒·[CASE-COMPILER-ASYMMETRY]） |

**★口径注**：ours=-O2 finalized-march·对手=as-shipped stock .so（-O3 stock-recipe 预建）。O2(ours)/O3(opp) 轻度 **不利 ours**（保守·如实披露）。vec_dot 轴 verdict 与 g7-census(-O3 stock-march) 逐格复现（<1% 漂移）→ recipe-robust。forward-op 轴用 O3 stock-march（内部对称·匹配 stock-.so 导出对手）。

## 1. ★0.8 达标表 (k1·28 格·headline·cold M=1 median·ratio=ours/对手)

> byte-exact 门：全测格 **0 mismatch / 0 ULP**（INT & f32·逐 VERIFY 行 raw 实证）。≥0.8=PASS·<0.8=FAIL。

### 1.A FLAT vec_dot (5) — 对手=单符号 inline (无 VLEN 专化)
| 格 | cold_m1 | cold_m8 | hot | 0.8 | 对手成色 |
|---|---:|---:|---:|:--:|---|
| q4_0 | **0.986** | 0.989 | 0.982 | **PASS** | light-vec·**parity-by-adoption**(tautological·发射 ggml 自身 block-dot) |
| q4_1 | **1.008** | 1.008 | 1.006 | **PASS** | light-vec·parity-by-adoption |
| q5_0 | **0.384** | 0.385 | 0.386 | **FAIL** | 单符号 native-vec(§五 74/12 light-vec·objdump-full-slice)·ours qh5bit block-dot 发散 |
| q5_1 | **0.398** | 0.403 | 0.403 | **FAIL** | 同上 |
| q8_0 | **0.984** | 0.985 | 0.979 | **PASS** | light-vec·parity-by-adoption |

**★flag③ q5_0/q5_1 count 钉死**：真派发对手 = **单符号** `ggml_vec_dot_q5_0_q8_0`/`q5_1_q8_1`（nm 仅 `_generic` sib·**无 `_vl128/_vl256`**·FLAT 格无 VLEN 专化坐实）。旧 "80/32 better-vec vs 74/12 light-vec" 之差 = **objdump 方法学 artifact**（本役 awk-body 截断法得 tot10·§五 full-.text-slice 得 74/12·§五 方法学坑已记档）→ 采 §五 **74/12 light-vec**。**成色无关结论**：ours LOSS 0.38-0.40×（板异确认：rvv 同格亦 LOSS 0.26-0.27×·qh 5-bit block-dot 发射发散·非 ggml 本体）。

### 1.B K-quant vec_dot (5) — 对手=vl256 手调 (q5_K 唯一未手调)
| 格 | cold_m1 | cold_m8 | hot | 0.8 | 对手成色 |
|---|---:|---:|---:|:--:|---|
| q2_K | **0.684** | 0.680 | 0.677 | **FAIL** | hand-tuned `_vl256` 强·weight-recon floor |
| q3_K | **0.522** | 0.535 | 0.533 | **FAIL** | hand-tuned `_vl256` 强 |
| q4_K | **0.592** | 0.583 | 0.579 | **FAIL** | hand-tuned `_vl256` 强·**vec_dot 轴天然 weight-recon-bound·beat 归 GEMM 轴 (Win-K1-VLEN)** |
| q5_K | **1.065** | 1.032 | 1.033 | **PASS** | native-vec-heavy·**NO vl256 spec**(唯一未手调)·⚠**赢=对手 immaturity·非赢手调**·DISCLOSE·非 tautological |
| q6_K | **0.549** | 0.545 | 0.540 | **FAIL** | hand-tuned `_vl256` 强 |

### 1.C K-quant GEMM (5) — ★大板异·对手=REAL same-op hand-brick·**PENDING follow-up**
| 格 | cold | 0.8 | 对手成色 (§五 定档·per-board) |
|---|---|:--:|---|
| q2_K gemm | **PENDING** | — | REAL hand-brick `ggml_gemm/gemv_q2_K_{8x8,16x1}_q8_K`(强·stock .so 实存) |
| q3_K gemm | **PENDING** | — | **cross-op**(q3_K 唯一无 repack·vs vl256 block-dot) |
| q4_K gemm | **PENDING** | (prior-seal **PASS**) | REAL 16x1 `ggml_gemv_q4_K_16x1_q8_K`(decode)/`gemm_16x1`(prefill)·**anchor=Win-K1-VLEN sealed 1.197× kernel vs 真 16x1 gemv**(既有·非本役) |
| q5_K gemm | **PENDING** | — | REAL hand-brick `gemm/gemv_q5_K_{8x8,8x4}`(强·无 16x1) |
| q6_K gemm | **PENDING** | — | REAL hand-brick `gemm/gemv_q6_K_{8x8,8x4}`(强·无 16x1) |

**★为何 PENDING（诚实·非空转）**：K-quant GEMM@k1 cold A/B 需 (1) **ours repack-GEMM kernel 重导出**（编译器 emit·源已清·非 prebuilt·任何 .so 无 `weft_emitc/tcrv_emitc.*repack_gemm_qK` 符号）+ (2) **hand-brick 对手 harness**（16x1/8x8 交织布局 + q8_K 重排激活·直调 `ggml_gemv/gemm_*`）。二者皆重-build·本会话不硬撑超时。§五 已 objdump 坐实真派发对手身份+metric（gemv rvv209 等）·q4_K 有既有 sealed anchor(1.197× PASS)。→ **#1 follow-up**（下一板会话·batch1-kquant gemm driver 改 k1 hand-brick 口径）。**大板异 per-board 记**：rvv K-quant GEMM 全 cross-op(零 repack)·k1 4/5 REAL hand-brick·**过 0.8 k1 远难于 rvv**·禁沿用 rvv 对手成色。

### 1.D iq/fp4 vec_dot (4) — 对手=vl256 gather 手调 (nvfp4 generic 弱)
| 格 | cold_m1 | cold_m8 | hot | 0.8 | 对手成色 |
|---|---:|---:|---:|:--:|---|
| iq1_s | **0.361** | 0.365 | 0.364 | **FAIL** | RVV-gather `_vl256` 强·gather-trap |
| iq1_m | **0.250** | 0.245 | 0.243 | **FAIL** | RVV-gather `_vl256` 强 |
| iq4_nl | **0.697** | 0.689 | 0.688 | **FAIL** | codebook-gather `_vl256`·近 0.7 |
| nvfp4 | **1.086** | 1.086 | 1.087 | **PASS** | generic 近标量(rvv2/无 vl-spec)·⚠**赢=对手 immaturity**·DISCLOSE·marginal |

### 1.E forward-op (9) — anchor n=4096·cold median
| 算子 | cold | hot | 0.8 | 对手成色 / 机制 |
|---|---:|---:|:--:|---|
| softmax | **1.004** | 1.013 | **PASS** | native-vec·vcpop-short-circuit **FIXED**(9a0b5fa4)·0-ULP-vs-opp |
| rms_norm | **1.336** | 1.179 | **PASS** | **1-pass fusion vs opp 2-pass**(memcpy+scale)·WIN·0 ULP |
| rope | **0.982** | 0.982 | **PASS** | mostly-scalar sin/cos·parity |
| silu | **1.012** | 1.017 | **PASS** | native-vec·vcpop **FIXED**·parity·ULP2 |
| gelu | **0.247** | 0.239 | **FAIL** | ★**STRUCTURAL** f16-LUT(GGML_GELU_FP16) vs ours tanhf·**ours +2500× acc**·**非公平速度 A/B**(结构差·flag) |
| add | **1.209** | 1.176 | **PASS** | **wide-m8 vs opp-m2 autovec WIN**·0 ULP bit-exact |
| mul | **1.189** | 1.175 | **PASS** | wide-m8 vs opp-m2 WIN·0 ULP |
| scale | **0.917** | 0.918 | **PASS** | native-vec·scheduling near-parity·0 ULP |
| cpy | **1.108** | 0.894 | **PASS**(cold) | autovec/memcpy·cold-WIN/hot-LOSS flip·0 ULP |

## 2. ★达标汇总 (k1·28 格·0.8 硬门)

| 桶 | 格数 | PASS | FAIL | PENDING |
|---|---:|---:|---:|---:|
| FLAT vec_dot | 5 | 3 (q4_0/q4_1/q8_0) | 2 (q5_0/q5_1) | — |
| K-quant vec_dot | 5 | 1 (q5_K⚠) | 4 (q2/q3/q4/q6_K) | — |
| K-quant GEMM | 5 | (q4_K prior-seal) | — | **5** (harness follow-up) |
| iq/fp4 vec_dot | 4 | 1 (nvfp4⚠) | 3 (iq1_s/iq1_m/iq4_nl) | — |
| forward-op | 9 | 8 | 1 (gelu·结构差) | — |
| **合计** | **28** | **13 measured PASS**(+q4_K prior-seal) | **10 FAIL** | **5 PENDING**(GEMM) |

- **本役实测 23/28 格**（byte-exact 全 0/0）：**13 PASS · 10 FAIL**。
- **5 GEMM PENDING**（q4_K 有既有 seal PASS anchor 1.197×）。
- ⚠ **成色披露**：2 PASS 靠 **对手 immaturity**（q5_K 无 vl256 spec·nvfp4 generic 近标量·非赢手调）·5 PASS 靠 **parity-by-adoption**(FLAT tautological)·8 forward PASS 中 gelu 若含则为结构差(已判 FAIL)。**真-硬碰硬赢强手调 = 0**（matmul 轴天花板 = weight-recon floor + 强 `_vl256`/gather 手调·结构性）·真机制 WIN 集中在 **forward-op**(rms_norm 1-pass·add/mul wide-m8·softmax/silu vcpop-fix)。

## 3. 输格清单 (<0.8·供攻坚·优先靶单)

| 格 | cold | 归因 / 出口 |
|---|---:|---|
| iq1_m@k1 | 0.250 | gather-trap vs vl256 手调·**Exit A**(结构·hardware gather 天花板) `[GAP-IQ-GATHER-VS-VECTORIZED-RVV]` |
| iq1_s@k1 | 0.361 | 同上 |
| q5_0/q5_1@k1 | 0.384/0.398 | qh 5-bit block-dot 发射发散·**Exit**(deployed 走 GEVM leaf 绿·此 secondary 路 low-value) `[GAP-Q5x-QH-BLOCKDOT-EMIT]` |
| q3_K@k1 | 0.522 | weight-recon floor vs `_vl256` 手调·**Exit A+C**(真 beat 归 GEMM 轴) `[GAP-KQUANT-VECDOT-VS-NATIVE-RVV]` |
| q6_K@k1 | 0.549 | 同上 |
| q4_K@k1 vec_dot | 0.592 | 同上(参照·真 beat 在 GEMM 轴 Win-K1-VLEN) |
| q2_K@k1 | 0.684 | 同上 |
| iq4_nl@k1 | 0.697 | codebook-gather vs `_vl256`·近 0.7 |
| gelu@k1 | 0.247 | ★结构差 LUT-vs-tanhf·**非公平速度**·ours 赢精度 2500×·收手(Exit structural) |

**★参照 §五 flag（vec_dot@k1 四格 0.54-0.68×）**：本役 q2_K 0.684/q3_K 0.522/q4_K 0.592/q6_K 0.549 — 与参照带一致(K-quant vec_dot weight-recon floor)。

## 4. DEQ-AXIS 18 状态 (照测·test-only-not-in-denom·不进 headline 0.8 分母)

**★DONE（非 flag follow-up）**·cold median·byte-exact 全 18 格 0/0·对手=**deployed-scalar-ref**(to_float·clang-18 scalar·rvv=0·较弱赢类·如实标)：

> 约定：ratio = ours-relative-speed (opp_ns/ours_ns)·>1 ours 快=WIN·<1 ours 慢=LOSS（与 T-CENSUS k1 13WIN/5LOSS 一致）。

| 结果 | 数 | 明细 |
|---|---:|---|
| **WIN(>1)** | **13** | q3_K 3.23·q4_K 4.31·q5_K 5.13·q6_K 1.12·iq1_s 2.89·iq1_m 3.55·iq2_xxs 2.47·iq2_s 1.81·iq4_nl 3.87·iq4_xs 3.15·mxfp4 3.13·tq1_0 4.29·tq2_0 4.23 |
| **LOSS(<1)** | **5** | q2_K 0.828(ours 亦 emit 标量 vsetvl=0)·iq2_xs 0.923·iq3_xxs 0.526·iq3_s 0.665·nvfp4 0.716(ours scalar/ldexpf) |

（DEQ opponent=deployed-scalar-ref·非 headline·体例独立子账·完整 18 格 raw 见 `raw/iqfp4_dequant_run.log`）

## 5. 体例合规自检
- 对手 = 该板 as-shipped **真派发** kernel·§五 定档符号·clang-18 对称域(stock=clang-18 build·bclass seal 实证) ✓
- q4_K 稻草人前车：K-quant GEMM 对手 = REAL hand-brick(§五 大板异·per-board·未沿用 rvv cross-op) ✓
- 对手 immaturity 赢如实披露(q5_K/nvfp4)·parity-by-adoption tautological 标注(FLAT) ✓
- 结构差非公平速度 flag(gelu LUT) ✓
- **provenance=so-hash**(md5 871169a0·树无 .git·flag① 处置) ✓
- byte-exact 门 0/0 全测格·失配=0(无停格) ✓
- 禁 git·不改 emitter/lib·e2e 冻结·回填 T3_B(非另建账本) ✓
- board 卫生：load-gate≥70%·core0-3 pin·pkill 清·scratch 清·co-tenant 未扰·loadavg 记录·restore ✓
- **PENDING 诚实标**（K-quant GEMM 5·需 ours 重导出+hand-brick harness·非假装测过） ✓

## 附：raw 指针
- `raw/vecdot_run.log` + `raw/vecdot_build_seal.txt`（FLAT 5 + K-quant vec_dot 5·cold M=1/M=8·byte-exact）
- `raw/iqfp4_dequant_run.log` + `raw/iqfp4_dequant_build_seal.txt`（iq/fp4 vec_dot 4 + DEQ 18）
- `raw/bclass_forward_run.log` + `raw/bclass_forward_build_seal.txt`（forward-op 9·shape sweep 512-16384·anchor 4096）
- 回填表：`../../result-tables/T3_B_board_B_rvv1.0_vlen256.csv`（perf 列·46 行·备份 `/tmp/T3_B_backup_pre_g8s6.csv`）
