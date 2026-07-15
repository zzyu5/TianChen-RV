# A2 batch2 — ⑦ pending-fold 折账 + ② iq/tq/fp4 gemm 标量仗（GEMM 轴·双板·clang-18 对称域）

> **任务**：线 A 板测执行员 A2-batch2 = **⑦ pending-fold 折账（18 项·多为存量 cold 折入）** + **② iq/tq/fp4 gemm 12 格标量仗**。承 A2-batch1（iq/tq vec_dot·harness recipe 复用）。
> **赛道**：**GEMM 轴 kernel-sym**（kernel-axis MICRO·同编译器/flags/march 对称）。**NOT e2e·NOT perf-covered 9/83·不入系统账**。[NG-4]。
> **口径铁线**：cold 唯一·**clang-18 双板对称**（k1 shipped=clang-18=部署域；rvv shipped=gcc-15 → clang-18=对称-micro·gcc-15=部署-clean）·**N≥20 中位+2-seed**·**禁一切继承**（decode=M=1 GEVM·prefill=nr16 GEMM·两串行 bug 前科）·每格对手身份探针·预注册判读（cold≥0.8=PASS / <0.8=具名-X+墙）·0 样本不造数·**便宜档禁称硬赢·接线≠转绿·预判不作结论**。
> **测于**：2026-07-16 · k1(VLEN256·SpacemiT-X60·core2 idle=100%·gov=performance 1.6GHz) 真板 pilot；rvv/k1 存量数据源逐格标 provenance。主树/build/stock `.so`/governor 未改·0 stray·无 git。

---

## 0. ★净结论（折账 split：折入 vs 转补测）

**⑦ pending-fold 18 项分类：**

| 类 | 项 | 处置 | 依据 |
|---|---|---|---|
| **FOLD-IN（clang-18 clean sym·PREFILL）** | q4_1/q5_0/q5_1@k1 gemm prefill | ✅ **折入·本 batch N=20 2-seed 真板确认**（升级存量 N=12） | 本役 k1 pilot（§2）·clang-18 clean sym·byte-exact PASS |
| **FOLD-IN（vs-block-dot·PREFILL·存量 N=12·建议刷 N≥20）** | q4_0@k1 gemm prefill · q8_0@k1(GEVM·memory-leaning) | ✅ 折入·flag N=12 | `be524605`（§6 货架A·clang-18·vs-block-dot 干净对） |
| **FOLD-CAVEAT（PREFILL·rvv·clang-micro opp-insensitive）** | q4_0/q4_1/q5_0/q5_1/q8_0@rvv gemm prefill | ✅ 折入(双域并列)·**clean clang-18-opp 重建为严格路(队列)** | T9 `668f1d45`：gcc-15 sym(clean·部署) + clang-18(clang-ours vs gcc-opp·opp block-dot 编译器不敏感 ~1.002× 已 T9§5 证) |
| **转补测（DECODE=M=1 GEVM·全格缺·构造）** | q4_0/q4_1/q8_0 @双板 decode · q5_0/q5_1@rvv decode · iq4_nl@双板 decode | ⛔ **转补测**（存量 cold 全 prefill nr≥4·**无 M=1 GEVM harness**·禁继承） | 见 §6 构造队列 |
| **DECODE 已部署（免测·登记）** | q5_0/q5_1@k1 decode | ✅ 折入(PASS-DEPLOYED) | C1 per-format measured-gate `d109d6ed2`·kernel 1.76/2.24×·独立验证 `a6fdf1a3`·已在 T3 |
| **转补测（IME kernel-sym·串行 bug 前科）** | q4_0@ime / q8_0@ime / q4_K@ime @k1 | ⛔ **转补测**（kernel-sym 从未测·**禁继承 native cold**·q4_K@ime bug 前科·e2e 存量另账） | 见 §6·onw2 有 e2e 非 kernel-sym |
| **转补测（iq4_nl gemm·clang-18 域）** | iq4_nl@双板 gemm（prefill+decode） | ⛔ **转补测**（anchor 0.217×/0.505× 是 gcc-15 域·cold-remeasure 只做 iq4_nl *vecdot* 1.150·非 gemm） | 见 §6 |

**② 12 格标量仗**：**全部 = BLOCKED-ON-CONSTRUCTION 本役**（需 ①export 7 格 gemm repack 核 + ②**构造 ggml scalar-ref GEMM 对手**（现无·§3）+ ③2 板 N≥20 harness）。5 格（iq1_s/iq1_m/iq3_xxs/iq3_s/nvfp4）**无独立 gemm repack**（block-dot 单路·regime-split 未命中）。**如实报阻塞·0 样本不造数·预判不作结论**（§3）。

**★本役真板产出（genuine·非折账）= k1 FLAT gemm prefill N=20 2-seed 确认**（q4_1 7.11× / q5_0 2.33× / q5_1 2.57× cold·byte-exact PASS·§2）——3 格从"继承 N=12"升为"真板 N≥20 2-seed 确认"·兑现铁线。

---

## 1. ⑦ pending-fold 折账逐项（存量 provenance 逐格机判）

### 1.1 FLAT gemm PREFILL（nr16 GEMM·可折入·存量 cold）

**存量数据源逐格 provenance（cold nr16 锚）：**

| 格·板·regime | 存量 cold(nr16) | 源·编译器域 | 对手符号(机判) | 对手成色 | N | predreg |
|---|---:|---|---|---|:--:|:--:|
| q4_0@rvv prefill | **6.224**(gcc) / 5.852(clang) | T9 `668f1d45` | ggml_vec_dot_q4_0_q8_0 (VLEN128 gate-off→block-dot) | light-vec 弱(便宜·routing白嫖) | 12 | PASS |
| q4_1@rvv prefill | **6.105**(gcc) / 6.325(clang) | T9 `668f1d45` | ggml_vec_dot_q4_1_q8_1 (CROSSOP block-dot) | light-vec 弱(便宜) | 12 | PASS |
| q5_0@rvv prefill | **1.232**(gcc) / 1.199(clang) | T9 `668f1d45` | ggml_vec_dot_q5_0_q8_0 (block-dot) | better-vec 中(相对硬) | 12 | PASS |
| q5_1@rvv prefill | **1.412**(gcc) / 1.402(clang) | T9 `668f1d45` | ggml_vec_dot_q5_1_q8_1 (block-dot) | better-vec 中(相对硬) | 12 | PASS |
| q8_0@rvv prefill | **4.752**(gcc) / 4.059(clang) | T9 `668f1d45` | ggml_vec_dot_q8_0_q8_0 (上游VLEN128破损→block-dot) | light-vec 弱(便宜·cold>hot 对手退化) | 12 | PASS |
| q4_1@k1 prefill | **7.101/7.125**(N=20·本役) | **§2 本役 pilot** | ggml_vec_dot_q4_1_q8_1 @0x9f774 (block-dot·0 repack) | light-vec 弱(便宜) | **20×2seed** | **PASS** |
| q5_0@k1 prefill | **2.325/2.328**(N=20·本役) | **§2 本役 pilot** | ggml_vec_dot_q5_0_q8_0 @0x9f81c (block-dot) | block-dot 中(相对硬) | **20×2seed** | **PASS** |
| q5_1@k1 prefill | **2.566/2.570**(N=20·本役) | **§2 本役 pilot** | ggml_vec_dot_q5_1_q8_1 @0x9f942 (block-dot) | block-dot 中(相对硬) | **20×2seed** | **PASS** |
| q4_0@k1 prefill | **3.50**(be524605·vs-block-dot) | `be524605` §6 | factory block-dot | light-vec 弱(便宜) | 12⚠ | PASS |

- **rvv 双域纪律**：rvv shipped=gcc-15 → **gcc-15 sym = 部署-clean 数**（主账·合部署现实）；**clang-18 = 对称-micro**（clang-ours vs gcc-opp·**opp block-dot 编译器不敏感 ~1.002×·T9§5 双证**）。两域对全 5 格 hot∧cold 均 ≥parity（verdict 对编译器身份稳健）。**严格 clang-18 clean-opp（opp 亦 clang-18 重编）= 队列**（§6·同 batch1 `-fno-integrated-as` 构造·本役板时投 k1 turnkey 未做 rvv opp 重建）。
- **k1 clean sym**：k1 shipped=clang-18 → clang-18 sym = **部署域有效·clean**。q4_1/q5_0/q5_1 本役 N=20 2-seed 真板确认（§2）。q4_0@k1 存量 be524605 N=12（建议随后刷 N≥20）。
- **成色分层（令六 lint·计数≠成色）**：**0 verified hand-brick**。q4_0/q4_1/q8_0 = light-vec 弱对手（big multiple 4–7×·**便宜档·成色低**）；q5_0/q5_1 = better-vec/block-dot 中对手（modest 1.2–2.6×·相对硬但仍非 hand-brick）。**FLAT gemm 对手全 block-dot 单实现折中/破损**（k1 `OPP_REPACK_SYMS=0`·rvv VLEN128 gate-off/破损）。

### 1.2 FLAT gemm DECODE（M=1 GEVM）— 转补测（存量全缺）

- **存量 cold 全是 prefill 形**（nr∈{4,8,16,64}·M=nr≥4）。**无任何 M=1 GEVM harness**（cold-remeasure/T9/batch1 全 prefill 或 vecdot-M1-block-dot·**非 repack-GEVM M=1**）。
- **★关键区分（禁混）**：FLAT `gemm_tile` decode = 我方 **repack-GEVM leaf @ M=1** vs opp block-dot @ M=1 —— **≠** cold-remeasure §六 已测的 FLAT **vec_dot** M1-GEVM（那是 block-dot emit 路·q4_0 0.938/q4_1 1.027/q5_0 0.451/q5_1 0.446/q8_0 0.965·已在 T3 vec_dot 行）。两核分立·**禁用 vec_dot M1 数冒充 gemm decode**。
- **唯一已定 decode**：**q5_0/q5_1@k1 decode = PASS-DEPLOYED**（C1 部署 repack-GEVM·kernel 1.760/2.241×·e2e 2×·独立验证 `a6fdf1a3`/`w91jl99ia`·已在 T3）。**q8_0@k1 存量 §6 "GEVM" 1.48×**（memory-leaning·e2e decode 1.21× clean·可作 q8_0@k1 decode 弱预测·但 N=12⚠·shape 非严格 M=1·建议补 M=1 GEVM 确认）。
- **转补测清单**：q4_0/q4_1/q8_0 @双板 decode · q5_0/q5_1@rvv decode · （q8_0@k1 decode 补确认）。**构造需求见 §6**。

### 1.3 IME gemm 3（q4_0@ime/q8_0@ime/q4_K@ime @k1）— 转补测

- **kernel-sym 从未测**：T3 IME 行（engine=ime）此前 **串行 bug 误继承 native gemm cold**（`q4_K@ime` 误显 =q4_K@k1 native·已修为 pending·memory 记 "一.1 IME 数据串行 bug"）。**禁继承 native cold**（q4_K@ime bug 前科）。
- **存量 = e2e（非 kernel-sym·另账）**：onw2 有 G6-A e2e 成色 —— q4_0@ime **tie-stock 1.0088×**（未真 beat）· q8_0@ime **beat-stock 2.233×**（赢弱 vendor·无内存税）· q4_K@ime **黄 0.909×**（super-block 重 fold epilogue 稀释）。**e2e≠kernel-sym·禁互推**。
- **对手身份**：stock vendor IME kernel（chip-specific dispatch·成色=手调·主表 canon 已归 手调 pending-fold）。kernel-sym A/B = 我方-IME-核 vs stock-vendor-IME-核（需 IME microbench 隔离·构造·§6）。
- **N/A-hw 机判**：IME@rvv 列 = N/A-hw（ime.present unsatisfiable on rvv·机判·唯一合法不对称）·**不建 rvv IME 行**。

### 1.4 iq4_nl gemm — 转补测（clang-18 域）

- **anchor**：prefill **0.217×** / decode **0.505×**（g5-wiring M2·**rvv gcc-15 sym**·八门全过）。
- **cold-remeasure 未覆盖 gemm**：`aab4168c` §六 只做 iq4_nl **vec_dot**（cold 1.150× PASS·native-vec 中对手）·**非 gemm repack**。gemm 轴 cold 在 clang-18 域**未测**。
- **驱动存在**：`tools/e2e-harness/board/iq4nl_gemm_paired_driver.c`（rvv/VLEN128·ours repack-GEMM codebook vluxei16 vs opp 真派发 block-dot·cross-op）→ 复用可测·**转补测**（§6·clang-18 域·k1 turnkey / rvv 需 opp 重建）。

---

## 2. ★本役真板 pilot：k1 FLAT gemm prefill N=20 2-seed（genuine·byte-exact）

**目的**：3 格 clean clang-18 sym 折入格从"继承 N=12"升为**真板 N≥20 2-seed 确认**（兑现铁线·禁继承）。**turnkey**（.inc + driver + stock lib 全在·k1 shipped=clang-18 天然对称）。

- **build seal**：clang-18.1.8 `-O3 -march=rv64gcv_zfh_zvfh_zicbop_zihintpause`·ours .inc md5 `736a716c`/`7b1ac4e6`/`03866f50`（**== casefile abb26043 逐字**）·stock lib md5 `871169a0` **before==after UNCHANGED**（只读·双证）。
- **对手探针（机判）**：`ggml_vec_dot_q4_1_q8_1@0x9f774` / `q5_0_q8_0@0x9f81c` / `q5_1_q8_1@0x9f942`（== T9 地址）·`OPP_REPACK_SYMS=0`（block-dot 单实现折中·k1 零 q4_1/q5_0/q5_1 repack）。ours objdump：vsetvl 18/13/21·vwmacc **8**（紧凑 vl=16·无 spill 病）。
- **协议**：K=2048·nc=512·pool=8（ws 5.2–6.3MB >> 512KiB L2·k1 无 L3=真 cold）·hiters=6·**rounds=20**·**2-seed cold {0x1357,0xACE2}**·within-proc paired·load-gate core2 idle=100%。

| 格@k1 | cold nr16 s1 | cold nr16 s2 | ours relIQR% | opp relIQR% | nr4 cold | nr64 cold | GATE(nbad) | HOT nr16 | predreg |
|---|---:|---:|---:|---:|---:|---:|:--:|---:|:--:|
| **q4_1** | **7.1008** | **7.1254** | 0.15–0.32 | 0.15–0.25 | 7.009 | 7.076 | PASS(0) | 7.15 | **PASS** |
| **q5_0** | **2.3251** | **2.3279** | 0.14–0.17 | 0.16–0.20 | 2.305 | 2.301 | PASS(0) | 2.33 | **PASS** |
| **q5_1** | **2.5659** | **2.5700** | 0.15–0.21 | 0.18–0.21 | 2.575 | 2.545 | PASS(0) | 2.57 | **PASS** |

- **byte-exact**：relerr_ours ≈ relerr_opp（3.5e-5–8.8e-5·f32 跨块 reduction ORDER 差·**nbad=0 全格**·正确 kernel 特征）。
- **2-seed 复现**：ratio 双 seed 差 <0.5%·relIQR <0.5% 全格 → **verdict 决定性·非噪声**。**与 casefile abb26043（7.109/2.326/2.571）吻合**（≠继承·真板重跑确认）。
- **成色（诚实·便宜标便宜）**：q4_1 = light-vec 弱对手（7× big multiple·**便宜档·非硬赢**）；q5_0/q5_1 = block-dot 中对手（2.3–2.6× modest·相对硬·**仍非 verified hand-brick**）。cold≈hot（k1 无 L3·compute-bound·缓存态二阶·同 k1-batch NULL 铁律）。**regime = PREFILL only**（decode M=1 GEVM 未测·§1.2）。

---

## 3. ② iq/tq/fp4 gemm 12 格标量仗 — BLOCKED-ON-CONSTRUCTION（如实报·0 造数）

**12 格**：iq1_s / iq1_m / iq2_xxs / iq2_xs / iq2_s / iq3_xxs / iq3_s / iq4_xs / mxfp4 / nvfp4 / tq1_0 / tq2_0 @gemm（对手 = ggml scalar-ref 兜底·§〇.2）。

**本役无法产出真数据·阻塞逐项（禁降级禁造数）：**

1. **对手不存在于现 harness**：GEMM 轴 **same-op repack 对手 absent**（§一.A·零 iq/tq repack GEMM）→ 兜底对手 = **ggml scalar-ref**（generic-C dequant→scalar dot / `_generic` vec_dot in GEVM loop）。此 scalar-ref GEMM 对手 **现无构造**（batch1/cold-remeasure 用的是 dispatched `_vlNNN` 手调 block-dot·**非 scalar-ref**）→ 须**新构造 fair scalar-ref GEMM 对手**（非 as-shipped·须明标兜底口径）。
2. **我方 gemm 核**：7 格有 repack gemm emit（iq2_xxs/iq2_xs/iq2_s/iq4_xs/mxfp4/tq1_0/tq2_0）→ 须 weft-opt export（fixture 存·`build-weft/bin/weft-opt` 可再生）；**5 格无独立 gemm repack**（iq1_s/iq1_m/iq3_xxs/iq3_s/nvfp4 = block-dot 单路·regime-split 未命中·`gemm`=`vec_dot` 同核）→ 这 5 格 "gemm 标量仗" = block-dot 路 vs scalar-ref。
3. **driver + 2 板 + N≥20 2-seed + 正确门**：per-format paired driver（iq4xs_gemm_paired_driver.c / iq4nl_gemm_paired_driver.c 可为模板·但对手需换 scalar-ref）。

**★预注册判读（写下·不作结论·[禁预判当结论]）**：兜底对手 = ggml **标量参考** → ours 向量 gemm vs opp 标量 = **便宜档·预期大倍数 PASS**（§〇.4 便宜档·**大倍数禁称硬赢**·标量对手成色低）。⚠ **接线≠转绿·预判不作结论**——须真测方定 verdict（0 样本不造数）。

**→ 构造队列见 §6.B（本役板时投 ⑦ turnkey 确认·② 标量仗留下一役·honesty first）。**

---

## 4. 成色纪律（诚实第一·便宜标便宜）

- **⑦ FLAT gemm 折入全格 = 0 verified hand-brick**：对手全 block-dot 单实现折中/破损（k1 `OPP_REPACK_SYMS=0`·rvv VLEN128 gate-off）。q4_0/q4_1/q8_0=light-vec 弱（**便宜档·big multiple 4–7× 非硬赢**）；q5_0/q5_1=better-vec/block-dot 中（modest·相对硬·仍非 hand-brick）。
- **② 标量仗 = 便宜档**（兜底 scalar-ref 对手）·**大倍数禁称硬赢**·未测不作结论。
- **计数纪律**：⑦ 折入 = kernel-axis GEMM 覆盖面（C3′ 证据）·**NOT e2e·NOT perf-covered 9/83·禁互推·禁写"加速 N kernel"于 e2e 语境**。matmul kernel-sym ≥parity 计数**不因 cold/regime 折账变**（cold=hot robustness·非新格）。
- **真硬碰硬空间（对照·非本折账格）**：hand-brick 真赢只在 q4_K/q2_K@k1（vs 真 16x1 RVV hand-brick·byte-verified·memory 记 2 格）。本 FLAT gemm 折账**均非**该档。

---

## 5. T3 回填清单（★留主会话机算入库·本 agent 不动 T3·区分 decode/prefill·禁继承）

**主表 `T3_master_rebuild.csv`**（op,format,engine,regime,group,rvv_tier,rvv_disp,rvv_cold,rvv_opp_sym,rvv_note,k1_tier,k1_disp,k1_cold,k1_opp_sym,k1_note）——**gemm_tile PREFILL 行折入（decode 行留 pending·禁继承）：**

| op | format | regime | rvv_disp | rvv_cold | k1_disp | k1_cold | 成色/域 tag |
|---|---|---|---|---:|---|---:|---|
| gemm_tile | q4_0 | **prefill** | PASS | **6.224**(gcc-deploy)/5.852(clang-micro) | PASS | **3.50**(be524605·N12⚠) | 便宜·light-vec·rvv 双域·k1 clang18-clean |
| gemm_tile | q4_1 | **prefill** | PASS | **6.105**(gcc)/6.325(clang) | PASS | **7.11**(N20·2seed·本役) | 便宜·light-vec·k1 clang18-clean-confirmed |
| gemm_tile | q5_0 | **prefill** | PASS | **1.232**(gcc)/1.199(clang) | PASS | **2.327**(N20·2seed·本役) | 相对硬·better-vec/block-dot 中·k1 confirmed |
| gemm_tile | q5_1 | **prefill** | PASS | **1.412**(gcc)/1.402(clang) | PASS | **2.568**(N20·2seed·本役) | 相对硬·k1 confirmed |
| gemm_tile | q8_0 | **prefill** | PASS | **4.752**(gcc)/4.059(clang) | PASS | **1.48**(be524605·GEVM·memory-leaning·N12⚠) | 便宜·light-vec·k1 数是 GEVM/decode-leaning⚠ |

**36-col T3_A(rvv)/T3_B(k1) 关键列**（cold_ratio=col30·opponent_grade=col31·opponent_symbol=col32·compiler_axis=col34·hardgate_0p8=col36）：上表 cold 值填 col30；opponent_grade=light-vec(q4_0/q4_1/q8_0)/better-vec(q5_0/q5_1@rvv)/block-dot(q5_0/q5_1@k1)；opponent_symbol=`ggml_vec_dot_<fmt>`；compiler_axis：**rvv=`gcc-deploy-footnote`(clean 6.224 等) + `clang18-micro`(opp-insensitive·caveat)**；k1=`clang18-sym=MAIN`；hardgate_0p8=`in-denom`·PASS。ledger_account=`matmul-kernel-sym`。artifact 指针=`A2-batch2-gemm-fold-raw/k1_flat_gemm_prefill_N20_2seed.log`（k1 3 格）/ T9 `668f1d45`(rvv) / `be524605`(k1 q4_0/q8_0)。

**★DECODE 行（gemm_tile·regime=decode）= 保持 pending·禁填**（除 q5_0/q5_1@k1 已 PASS-DEPLOYED）。q8_0@k1 decode 可暂记 `1.48(GEVM·N12·memory-leaning·待 M=1 确认)` 弱预测·标 ⚠。

**IME 3 行 / iq4_nl 行 / 12 标量仗行 = 保持 pending-fold / pending-真**（本役未测·§6 队列）。

---

## 6. 转补测 / 构造队列（★board 板时排序·主会话/下一役承接）

### 6.A ⑦ 剩余（DECODE + IME + iq4_nl·构造需求）
| 项 | 缺 | 构造需求 | 板 | turnkey? |
|---|---|---|---|:--:|
| **FLAT gemm DECODE M=1 GEVM**（q4_0/q4_1/q8_0@双板·q5_0/q5_1@rvv） | 无 M=1 GEVM harness | repack-GEVM leaf 核 export（q5_0/q5_1 `weft_emitted_gevm_*.inc` 存·余需 weft-opt export）+ **M=1 GEVM cold driver**（现 driver 全 nr≥4） | rvv+k1 | 否(需 driver) |
| **q8_0@k1 decode 确认** | §6 数是 GEVM 但 shape 非严格 M=1·N12 | 同上·M=1 GEVM | k1 | 半 |
| **IME kernel-sym 3**（q4_0/q8_0/q4_K@ime） | kernel-sym 从未测·禁继承 native | IME microbench 隔离核 A/B（我方-IME vs stock-vendor-IME）·q4_K@ime bug 前科须小心 | k1 | 否(需 IME harness) |
| **iq4_nl gemm clang-18**（prefill+decode） | anchor 是 gcc-15 域·cold-remeasure 只做 vecdot | `iq4nl_gemm_paired_driver.c` 复用·k1 天然 clang-18·rvv 需 opp clang-18 重建 | rvv+k1 | k1 半-turnkey |
| **rvv FLAT gemm clean clang-18-opp** | T9 clang 列是 clang-ours vs gcc-opp | opp libggml `-fno-integrated-as` clang-18 重编（同 batch1）→ 严格双域 | rvv | 半(需 opp 重建) |

### 6.B ② 12 格标量仗（构造·下一役）
- ① weft-opt export 7 格 gemm repack 核（iq2_xxs/iq2_xs/iq2_s/iq4_xs/mxfp4/tq1_0/tq2_0）；② **构造 ggml scalar-ref GEMM 兜底对手**（generic-C·须明标非 as-shipped）；③ per-format paired driver（模板 iq4xs/iq4nl_gemm_paired_driver.c·换对手为 scalar-ref）；④ 2 板 N≥20 2-seed + ZERO-MODEL 正确门。5 格（iq1_s/iq1_m/iq3_xxs/iq3_s/nvfp4）block-dot 单路（gemm=vecdot 同核）。**便宜档·预期大倍数·禁称硬赢·未测不作结论。**

---

## 7. 污染纪律 + restore（k1 pilot）

- **cold 协议**：pool=8 tile POOL（ws 5.2–6.3MB >> 512KiB L2·k1 无 L3→真 DRAM cold）·N=20 median+relIQR·2-seed·within-proc paired·warmup dropped。
- **restore 双证**：k1 `/data/k1build-stock/bin/libggml-cpu.so.0` md5 `871169a0…` **before==after UNCHANGED**（只读·我方仅 build ours .o + link driver·未触 .so）·主树/build/governor 未改·**无 git add/commit**。
- **stray**：测后 `pgrep flatcold_k1`=0·scratch `/tmp/g8_a2b2_k1` 已 `rm -rf`（GONE 双证）。load-gate core2 idle=100%·loadavg begin 2.29→end 2.92（我方单核 bench 自负荷·非污染）。
- **disjoint-pin**：taskset -c 2（L2 shared 0-3·idlest）。

## durable files
- `A2-batch2-gemm-fold.md`（本文）
- `A2-batch2-gemm-fold-raw/k1_flat_gemm_prefill_N20_2seed.log`（k1 3 格 build seal + 对手探针 + 12 行 cold A/B raw·byte-exact·2-seed）
- 折入存量源（不拷·登记指针）：T9 `668f1d45`（rvv FLAT gemm 5·gcc+clang 双账）· `abb26043`/k1-flat-kernelsym（k1 q4_1/q5_0/q5_1 存量·本役 N=20 确认）· `be524605` §6（k1 q4_0/q8_0 vs-block-dot）· C1 deploy `d109d6ed2`/`a6fdf1a3`（q5_0/q5_1@k1 decode PASS-DEPLOYED）
