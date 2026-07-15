# A2 batch6 — ② iq/tq/fp4 gemm 12 格标量仗（GEMM 轴·scalar-ref 对手·双板·便宜档·真板测）

> **任务**：线 A·A2-batch6 = **② iq/tq/fp4 gemm 12 格标量仗**（承 batch2 §3 BLOCKED-ON-CONSTRUCTION·本役构造之）。我方 gemm repack-GEMM kernel vs **构造的 ggml scalar-ref**（`ggml_vec_dot_<fmt>_..._generic`·ggml 自带标量参考·同 op repack GEMM 缺席→标量兜底）。
> **赛道**：**GEMM-prefill 轴 kernel-sym MICRO**（同编译器/flags/march 对称·rvv 双域）。**NOT e2e·NOT perf-covered 9/83·不入系统账**。[NG-4]。
> **口径铁线**：cold 唯一·**便宜档**（scalar 对手·big multiple 预期·**★禁称硬赢**）·k1 clang-18 部署对称 + rvv 双域（gcc-15.2 deploy-clean MAIN / clang-18 micro footnote·Amdahl 同域律）·**N=25 中位+relIQR+2-seed**·**禁一切继承**·对手身份探针（generic 符号机判）·**ZERO-MODEL 正确门**（ours vs ggml 权威 generic·独立于 ours 内部）·预注册判读（cold≥0.8=PASS/<0.8=named-X+墙）·**0 样本不造数·接线≠转绿·预判不作结论**。
> **测于**：2026-07-16 · k1(VLEN256·SpacemiT-X60·clang-18.1.8) + rvv(VLEN128·gcc-15.2 + clang-18.1.8)。主树/build/stock `.so`/governor 未改。

---

## 0. ★净结论

**12 格 = 7 可测（有 repack-GEMM 干净 export）+ 5 BLOCKED（无 repack·block-dot 单路）：**

| 类 | 格 | 处置 |
|---|---|---|
| **可测（clean weft-opt export·prefill GEMM + decode GEVM 均有 leaf）** | iq2_xxs / iq2_xs / iq2_s / iq4_xs / mxfp4 / tq1_0 / tq2_0 | ✅ 本役双板真测·**prefill(§2.1-2.3) + decode M=1 GEVM(§2.4)** |
| **BLOCKED（无 repack-GEMM/GEVM·regime-split 未命中·gemm=vec_dot 同核）** | iq1_s / iq1_m / iq3_xxs / iq3_s / nvfp4 | ⛔ 如实报 blocked（无 repack leaf export·§4） |

**★覆盖**：7 格 × 2 regime（prefill GEMM nr16 + decode M=1 GEVM）× 3 编译器域（k1-clang / rvv-gcc / rvv-clang）= **42 datapoint 真测·全 byte-exact**。

**★正确性（ZERO-MODEL·全 7 格双板 PASS）**：ours repack-GEMM vs ggml **权威 generic scalar** 逐 (row,col) 全 byte-exact/near-exact（k1 seed 0x1357：**6 格 maxrel=0.000e+00 bit-exact** + iq4_xs 1.76e-5 f32 fold-order·nbad=0/256 全格）。**数值正确性坐实**（对手=ggml 自带参考·独立于 ours 内部=真 ZERO-MODEL）。

**★成色（诚实第一·★codegen 双侧主导·三域·非简单便宜档）**：对手 = **构造 ggml scalar-ref**（`_generic`）·**便宜档**（禁称硬赢）。**关键实测 = ratio 由【双侧编译器 codegen】主导**（三域·§2）：
- **k1（clang-18 部署 MAIN）：7/7 PASS cold 2.1–10.3×**。但 opp generic 被 **clang-18 编成臃肿 autovec**（iq2_xs 1372 insns·373ms）→ 大倍数**半是"对手被 clang 编胖"假象·非我方核硬赢**（同源 generic：clang-stock 373ms vs gcc-stock 33ms = 11×）。
- **rvv gcc-15.2（部署 MAIN·Amdahl 同域律）：7/7 named-X cold 0.35–0.75×·ours 全输**（iq2_xs/iq2_s/mxfp4 **输给 gcc 纯标量 ref vec=0**）= **我方 kernel gcc-15.2 codegen-death**。
- **rvv clang-18（micro footnote·消歧✅）：7/7 PASS 2.5–4.8×·同 kernel 换 clang 提速 6.8–11.2×**（iq4_xs 71.6→9.4ms·tq1_0 35.5→3.2ms）→ **决定性证 rvv-gcc 全输 = 编译器 artifact·非算法·非 VLEN128·[CASE-COMPILER-ASYMMETRY]/[CASE-KQUANT-GCC-CODEGEN] 家族扩到 iq/tq repack-GEMM**。
- **★单一诚实数（apples best-vs-best·ours-clang vs gcc-tight-opp）= 2.5–4.8×**（便宜档下真 repack-over-scalar 优势·**禁称硬赢**）。**0 verified hand-brick**·**NOT matmul ≥parity headline**（scalar-ref 不入系统账·同 batch3 ④ SANITY 定性）。
- **★本 batch 唯一 genuine 正结果 = 7 格×3域 byte-exact 正确性坐实**（我方 iq/tq/fp4 repack-GEMM kernel 真板数值正确·C3′ 覆盖）+ **[CASE-CC-ASYM] 6.8–11.2× codegen 档案证**。

---

## 1. harness 构造实录（durable·可复现）

**驱动**：`A2-batch6-iqtq-gemm-scalar-raw/iqtq_gemm_scalar_driver.cpp`（新造）。
- **OURS** = weft repack-GEMM leaf（front-door export：`weft-opt <fixture> --weft-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`·7 格 md5 见 `kernels/GEN_SEAL.txt`）。7-arg ABI `(n=K, s, vx, vy, nr, nc, bs)`。weight = x16-interleaved（block_<fmt>x16）· activation = block_q8_Kx4（f32 d[4]@+0·qs@+16 pos*4+row·stride 1168）；mxfp4 = block_q8_0x4（fp16 d[4]@+0·qs@+8·stride 136）。
- **OPP（scalar-ref 兜底）** = stock `libggml-cpu.so` 真派发 `ggml_vec_dot_<fmt>_q8_K_generic`（mxfp4=`_q8_0_generic`）逐 (row,col) over PLAIN blocks = **scalar-ref GEMM**。这 = ggml **自带 generic 标量参考**（同 op repack GEMM 在 ggml 缺席→标量兜底·§〇.2）·**便宜档**（非手调 block-dot）。符号机判探针入 build_seal。
- **repack（plain→x16）**：byte-exact·复用 `tools/oracle-repack/oracle_repack_<fmt>.cpp` 的 make_block 寻址（qs/scale/sign 重排·fp16 d 落真 kernel 布局；oracle 的 float-d 仅 int-cert 用·本役用真 fp16-d 布局）。plain block = ggml-common.h 权威布局（board 取证）。
- **GATE（ZERO-MODEL）** = ours 输出 vs generic 输出 per-cell（generic = ggml 权威参考·独立于 ours）·maxrel floor@1.0·gate nbad(rel>5e-3)==0。
- **cold** = 32MiB flush per rep·CLOCK_MONOTONIC·median+relIQR·2-seed·core-pinned·ratio=opp_med/ours_med。

**export 清单（7 格·byte-exact 可再生·GEN_SEAL.txt）**：
`rvv-to-emitc-repack-gemm-{iq4-xs,iq2-xxs,iq2-xs,iq2-s,mxfp4,tq1-0,tq2-0}-q8-{K,0}.mlir`。全 clean export（symbol `weft_emitc_ggml_repack_gemm_<fmt>_kernel_...`）。

---

## 2. 逐格结果（cold / 判读 / 成色 / 探针）

### 2.0 正确性 GATE（VERIFY·K=2048 nr=4 nc=64·seed 0x1357·全 7 格双板）

| 格 | maxrel (k1 / rvv-gcc / rvv-clang) | nbad | fp |
|---|---|---:|:--:|
| iq4_xs | 1.76e-5 / 1.76e-5 / 1.76e-5（nc512: ≤4.9e-4） | 0/(256·8192) | near-exact(fold-order) |
| iq2_xxs | 0.0 / 0.0 / 0.0 | 0 | **bit-exact** |
| iq2_xs | 0.0 / 0.0 / 0.0 | 0 | **bit-exact** |
| iq2_s | 0.0 / 0.0 / 0.0 | 0 | **bit-exact** |
| mxfp4 | 0.0 / 0.0 / 0.0 | 0 | **bit-exact** |
| tq1_0 | 0.0 / 0.0 / 0.0 | 0 | **bit-exact** |
| tq2_0 | 0.0 / 0.0 / 0.0 | 0 | **bit-exact** |

**全 7 格×3 域（k1-clang / rvv-gcc / rvv-clang）PASS**·6 格 bit-exact(0.0) + iq4_xs 1.76e-5～4.9e-4（f32 跨块 reduction order·正确核特征·nbad=0/8192 at nc=512）。**数值正确坐实**（vs ggml 权威 generic·独立 ZERO-MODEL）。repack（plain→x16）+ activation q8_Kx4 interleave 全 byte-exact by construction。

### 2.1 k1（VLEN256·clang-18 部署对称·cold nr16 prefill·K=2048 nr=16 nc=512 reps=25 2-seed）

| 格 | ours_ms(s1/s2) | opp_ms(s1/s2) | **ratio_cold(s1/s2)** | opp generic codegen | **predreg** | 成色（便宜档·诚实） |
|---|---:|---:|---:|---|:--:|---|
| iq4_xs | 21.64/21.72 | 45.78/45.65 | **2.11/2.10** | clang-autovec 762ins | **PASS** | 便宜·opp clang-胖·非硬赢 |
| iq2_xxs | 29.71/29.70 | 90.06/89.34 | **3.03/3.01** | clang-autovec 318ins | **PASS** | 便宜·非硬赢 |
| iq2_xs | 36.20/36.29 | 373.29/373.18 | **10.31/10.28** | clang-autovec 1372ins | **PASS** | 便宜·opp clang 编巨胖(1372ins)·大倍数假象·**禁称硬赢** |
| iq2_s | 36.42/38.17 | 361.56/361.51 | **9.93/9.47** | clang-autovec 1311ins | **PASS** | 同上·opp clang 巨胖·s2 ours iqr5.8%(co-tenant) |
| mxfp4 | 20.22/20.09 | 46.60/45.99 | **2.31/2.29** | clang-autovec 139ins | **PASS** | 便宜·非硬赢 |
| tq1_0 | 7.97/7.95 | 51.59/51.18 | **6.47/6.44** | clang-autovec 595ins | **PASS** | 便宜·非硬赢 |
| tq2_0 | 5.56/5.52 | 26.50/26.31 | **4.76/4.76** | clang-autovec 409ins | **PASS** | 便宜·非硬赢·ours vsetvl=8 最简 |

**k1 tally：7/7 PASS**（cold 2.1–10.3×·2-seed 差 <2% 稳）。**★便宜档诚实**：big multiple = ①scalar-ref 弱 + ②opp 被 clang-18 编成臃肿 autovec（iq2_xs opp 1372 insns=巨慢）双重打折。**禁称硬赢**·计数不入 matmul headline。stock md5 `871169a0` before==after·STRAY=0。

### 2.2 rvv（VLEN128·gcc-15.2 deploy-clean=MAIN·K=2048 nr=16 nc=512 reps=25 2-seed）

| 格 | ours_ms(s1/s2) | opp_ms(s1/s2) | **ratio_cold(s1/s2)** | opp generic codegen | **predreg** | 墙/成色（诚实） |
|---|---:|---:|---:|---|:--:|---|
| iq4_xs | 71.63/71.29 | 35.90/35.88 | **0.50/0.50** | gcc-compact 137ins(vec55) | **named-X** | ours 输·gcc opp 编紧凑·[CASE-COMPILER-ASYMMETRY] |
| iq2_xxs | 77.88/78.84 | 39.46/39.41 | **0.51/0.50** | gcc-compact 320ins(vec224) | **named-X** | ours 输 |
| iq2_xs | 87.96/88.09 | 33.20/33.63 | **0.38/0.38** | gcc **PURE-SCALAR 368ins vec=0** | **named-X** | ★ours 输给 gcc-**纯标量** ref·ours gcc-codegen 差 |
| iq2_s | 89.55/88.80 | 33.03/33.03 | **0.37/0.37** | gcc **PURE-SCALAR 364ins vec=0** | **named-X** | ★同上·输纯标量 |
| mxfp4 | 25.45/25.45 | 19.17/19.04 | **0.75/0.75** | gcc **PURE-SCALAR 270ins vec=0** | **named-X** | ★输纯标量·near-parity |
| tq1_0 | 35.46/35.48 | 12.49/12.96 | **0.35/0.37** | gcc-compact 478ins(vec222) | **named-X** | ours 输 |
| tq2_0 | 21.94/21.91 | 9.89/9.84 | **0.45/0.45** | gcc-compact 129ins(vec74) | **named-X** | ours 输 |

**rvv-gcc tally：7/7 named-X**（cold 0.35–0.75×·2-seed 差 <5% 稳）。**★关键诚实负结果**：ours repack-GEMM 在 rvv 部署域（gcc-15.2）**全输 scalar-ref**·其中 **iq2_xs/iq2_s/mxfp4 输给 gcc 编的【纯标量】ref（vec=0）**=我方 kernel gcc-15.2 codegen 差（vsetvl 4104–5374·同 batch4 q4_K gcc-death 类·[CASE-KQUANT-GCC-CODEGEN] 家族）。**rvv-clang-18 micro 消歧**（§2.3·测我方核 clang 是否恢复→若恢复=纯 compiler artifact）。stock md5 `d1adc634` before==after。

### 2.3 rvv（VLEN128·clang-18 对称-micro footnote·[CASE-COMPILER-ASYMMETRY] 消歧·ours=clang·opp=同 gcc-15 stock generic）

| 格 | ours_ms(s1/s2) | opp_ms(s1/s2) | **ratio_cold(s1/s2)** | **★ours gcc→clang 提速** | predreg |
|---|---:|---:|---:|---:|:--:|
| iq4_xs | 9.38/9.09 | 36.12/35.89 | **3.85/3.95** | 71.6→9.4 = **7.6×** | **PASS** |
| iq2_xxs | 10.74/10.68 | 39.40/39.38 | **3.67/3.69** | 77.9→10.7 = **7.3×** | **PASS** |
| iq2_xs | 13.03/13.03 | 33.66/33.45 | **2.58/2.57** | 88.0→13.0 = **6.8×** | **PASS** |
| iq2_s | 13.01/13.08 | 33.16/33.50 | **2.55/2.56** | 89.5→13.0 = **6.9×** | **PASS** |
| mxfp4 | 7.27/7.26 | 19.09/19.12 | **2.63/2.63** | 25.4→7.3 = **3.5×** | **PASS** |
| tq1_0 | 3.16/3.16 | 11.99/12.50 | **3.80/3.96** | 35.5→3.2 = **11.2×** | **PASS** |
| tq2_0 | 2.05/2.02 | 9.91/9.89 | **4.84/4.89** | 21.9→2.0 = **10.7×** | **PASS** |

**rvv-clang tally：7/7 PASS**（cold 2.5–4.8×）。**★决定性 [CASE-COMPILER-ASYMMETRY] 实证**：**同 rvv/VLEN128 硬件·同 kernel·仅换编译器**→ ours 提速 **6.8–11.2×**（iq）/ 3.5×（mxfp4）。**rvv-gcc named-X = 我方 kernel gcc-15.2 codegen-death**（非算法·非 VLEN128·非对手）·**[CASE-KQUANT-GCC-CODEGEN] 家族扩展到 iq/tq repack-GEMM**。opp（gcc-15 stock generic）跨两役不变（~33–39ms iq·印证 opp 编译器身份固定=部署 gcc）。**全 byte-exact**（GATE nc=512·nbad=0/8192·iq4_xs 1.7e-4–4.9e-4 fold-order·余 0.0 bit-exact）。

**★三域对照（同一 kernel·honesty 关键）**：apples-vs-apples = **ours-clang vs gcc-tight-opp**（rvv-clang §2.3）= **2.5–4.8× 真 repack-over-scalar 优势（便宜档·非硬赢）**。k1 的 2.1–10.3× 中 iq2_xs/iq2_s 的 10× **是 clang-18 把 opp generic 编成 11× 慢（373ms vs gcc 33ms）的假象**·非我方核强。rvv-gcc 的 0.35–0.75× = 我方核 gcc-death。**→ 单一诚实数 = 2.5–4.8×（best-vs-best·便宜档·禁称硬赢）**。

---

### 2.4 DECODE（M=1 GEVM·7 格×3 域·活化=plain 单 q8 向量·nc=512·reps=25·2-seed）

> **性质**：M=1 GEVM = 我方 repack-GEVM leaf（同 x16 weight·activation=**plain 单 q8_K/q8_0 向量**·非交织·5-arg ABI）vs generic 逐列 scalar-ref。memory-bound decode regime。

| 格 | k1(clang·s1/s2) | rvv-gcc(MAIN·s1/s2) | rvv-clang(micro·s1/s2) | correctness |
|---|---:|---:|---:|:--:|
| iq4_xs | **0.93/0.94** PASS | **0.62/0.65** named-X | 1.32/1.50 PASS | 1.6e-5 nbad=0 |
| iq2_xxs | **1.13/1.13** PASS | **0.61/0.60** named-X | 1.25/1.25 PASS | 0.0 |
| iq2_xs | **3.81/3.82** PASS | 0.97/1.02 PASS | 2.03/2.05 PASS | 0.0 |
| iq2_s | **3.57/3.54** PASS | 1.04/1.03 PASS | 2.01/2.06 PASS | 0.0 |
| mxfp4 | **0.94/0.95** PASS | 1.52/1.51 PASS | 1.01/1.01 PASS | 0.0 |
| tq1_0 | **1.99/2.00** PASS | 1.72/1.62 PASS | 0.79/0.80 named-X | 0.0 |
| tq2_0 | **1.59/1.59** PASS | 1.25/1.13 PASS | 1.04/1.03 PASS | 0.0 |

**decode tally**：k1 **7/7 PASS**(0.93–3.81×)· rvv-gcc **5 PASS/2 named-X**· rvv-clang **6 PASS/1 named-X**(tq1_0 0.79 boundary)。**★decode = memory-bound·全域 parity-leaning**（比值集中 0.6–3.8×·多在 0.8–2×）·**编译器 asymmetry 远弱于 prefill**（GEVM leaf vsetvl 258–2093·不触 GEMM 的 gcc-death 5000+）→ rvv-gcc decode 5/7 恢复 PASS（对比 prefill 全输）。**全 byte-exact**（verify nc=64 + nc=512 GATE·nbad=0 全）。**同便宜档·禁称硬赢**。stock md5 双板 before==after·stray=0·scratch GONE。

---

## 3. 对手成色（机判·便宜档·★但 codegen 双侧主导）

generic = ggml 自带 **参考实现**（arch fallback·非手调 `_vlNNN`）·**便宜档**（同 op repack-GEMM 缺席→标量兜底·big multiple 预期·**禁称硬赢**·同 batch3 ④ product_reduce SANITY 定性）。**★但实测 generic 的 codegen 随编译器剧变（非稳定 scalar）**·objdump 逐格：

| generic 符号 | k1（clang-18 stock·insns/vec） | rvv（gcc-15 stock·insns/vec） | 判 |
|---|---|---|---|
| iq4_xs | 762 / 202（bloated autovec） | 137 / 55（compact） | clang 编胖 5.6× |
| iq2_xs | 1372 / 696（巨胖 autovec） | 368 / **0 纯标量** | ★clang 编巨胖·gcc 纯标量却更快 |
| iq2_s | 1311 / 693 | 364 / **0 纯标量** | 同上 |
| mxfp4 | 139 / 27 | 270 / **0 纯标量** | — |
| iq2_xxs | 318 / 56 | 320 / 224 | — |
| tq1_0 | 595 / 297 | 478 / 222 | — |
| tq2_0 | 409 / 152 | 129 / 74 | gcc 编紧凑 3.2× |

**★成色诚实结论**：opp 是"标量参考"但其速度不由 scalar 性质定·由编译器定（同 iq2_xs generic：clang-stock 373ms vs gcc-stock 33ms = **11× 差**·同源 C）。故 **k1 的大倍数（iq2_xs 10.3×）主要是 clang 把 opp 编巨胖的假象·非我方核硬赢**；**rvv-gcc 的全输是我方核 gcc-death**（§2.3·clang 恢复 6.8–11.2×）。**apples-vs-apples（ours-clang vs gcc-tight-opp·rvv §2.3）= 2.5–4.8× = 便宜档下真 repack-over-scalar 优势·仍禁称硬赢**。**0 verified hand-brick**。

---

## 4. BLOCKED 清单（5 格·如实报·0 造数）

| 格 | 缺 | 依据 |
|---|---|---|
| iq1_s / iq1_m / iq3_xxs / iq3_s / nvfp4 | **无 repack-GEMM emitc fixture** | `test/Conversion/RVV/` 无 `rvv-to-emitc-repack-gemm-<fmt>-*.mlir`（仅 block-dot `*-super-block-block-dot-*` / typed-scaffold / dequant）→ gemm = vec_dot 同核（regime-split 未命中）·无独立 repack leaf 可 export → **BLOCKED-ON-CONSTRUCTION**（同 batch2 §3 判定·本役证实）。这 5 格 gemm 标量仗 = block-dot 路 vs scalar-ref（非 repack-GEMM）·须另立（block-dot 已在 batch1 vec_dot 轴测手调对手·非本役 scalar 轴）。 |

---

## 5. T3 回填清单（★留主会/机算入库·本 agent 不动 T3·gemm·**scalar-ref 类档·便宜档·禁继承·prefill**）

> 均为 **新行**（gemm_tile·regime=prefill·op-format×engine=rvv-plugin）。opponent = **scalar-ref（ggml `_generic`）**·**便宜档**·**★建议 test-only-not-in-denom**（同 batch3 ④ product_reduce·scalar-ref 非 as-shipped 手调 framework·不入 0.8 硬门 matmul 分母·§7 policy 待主会裁）。区分 **rvv 三域**：gcc-15.2=deploy-MAIN·clang-18=micro-footnote。

### 5.1 T3_A（rvv·VLEN128）— 双域并列（gcc deploy-clean MAIN / clang micro）

| measurement_row_key | cold_ratio gcc(s1/s2) | cold_ratio clang(s1/s2) | opponent_grade | opponent_symbol | ledger_account | hardgate_0p8 | correctness |
|---|---|---|---|---|---|---|---|
| gemm\|iq4_xs\|prefill\|nr16\|q8_K | **0.501/0.503**(gcc-death) | 3.85/3.95 | scalar-ref(generic·gcc-compact/clang-bloat) | ggml_vec_dot_iq4_xs_q8_K_generic | scalar-ref(cheap)·[CASE-CC-ASYM] | **test-only-not-in-denom**(便宜·gcc named-X/clang PASS) | byte-exact 1.7–4.9e-4 nbad=0 |
| gemm\|iq2_xxs\|prefill\|nr16\|q8_K | **0.507/0.500** | 3.67/3.69 | scalar-ref(generic) | ggml_vec_dot_iq2_xxs_q8_K_generic | scalar-ref(cheap) | test-only-not-in-denom | byte-exact 0.0 nbad=0 |
| gemm\|iq2_xs\|prefill\|nr16\|q8_K | **0.377/0.382** | 2.58/2.57 | scalar-ref(gcc **纯标量**/clang 巨胖) | ggml_vec_dot_iq2_xs_q8_K_generic | scalar-ref(cheap) | test-only-not-in-denom | byte-exact 0.0 nbad=0 |
| gemm\|iq2_s\|prefill\|nr16\|q8_K | **0.369/0.372** | 2.55/2.56 | scalar-ref(gcc **纯标量**) | ggml_vec_dot_iq2_s_q8_K_generic | scalar-ref(cheap) | test-only-not-in-denom | byte-exact 0.0 nbad=0 |
| gemm\|mxfp4\|prefill\|nr16\|q8_0 | **0.753/0.748** | 2.63/2.63 | scalar-ref(gcc **纯标量**) | ggml_vec_dot_mxfp4_q8_0_generic | scalar-ref(cheap) | test-only-not-in-denom | byte-exact 0.0 nbad=0 |
| gemm\|tq1_0\|prefill\|nr16\|q8_K | **0.352/0.365** | 3.80/3.96 | scalar-ref(generic) | ggml_vec_dot_tq1_0_q8_K_generic | scalar-ref(cheap) | test-only-not-in-denom | byte-exact 0.0 nbad=0 |
| gemm\|tq2_0\|prefill\|nr16\|q8_K | **0.451/0.449** | 4.84/4.89 | scalar-ref(generic) | ggml_vec_dot_tq2_0_q8_K_generic | scalar-ref(cheap) | test-only-not-in-denom | byte-exact 0.0 nbad=0 |

**rvv compiler_axis**：`gcc15.2-deploy=MAIN`（named-X 全 7·gcc-death）+ `clang18-micro=footnote`（PASS 全 7·6.8–11.2× 提速证 codegen artifact）。opp 恒 = gcc-15 stock generic（跨域固定）。

### 5.2 T3_B（k1·VLEN256·clang-18 部署对称=MAIN）

| measurement_row_key | cold_ratio(s1/s2) | opponent_grade | opponent_symbol | ledger_account | hardgate_0p8 | correctness |
|---|---|---|---|---|---|---|
| gemm\|iq4_xs\|prefill\|nr16\|q8_K | 2.11/2.10 | scalar-ref(generic·clang-bloat 762ins) | ggml_vec_dot_iq4_xs_q8_K_generic | scalar-ref(cheap) | **test-only-not-in-denom**(便宜·opp clang 编胖) | byte-exact 1.76e-5 nbad=0 |
| gemm\|iq2_xxs\|prefill\|nr16\|q8_K | 3.03/3.01 | scalar-ref(clang-bloat) | ggml_vec_dot_iq2_xxs_q8_K_generic | scalar-ref(cheap) | test-only-not-in-denom | byte-exact 0.0 |
| gemm\|iq2_xs\|prefill\|nr16\|q8_K | 10.31/10.28 | scalar-ref(clang 巨胖 1372ins) | ggml_vec_dot_iq2_xs_q8_K_generic | scalar-ref(cheap) | test-only-not-in-denom(★大倍数=clang-bloat 假象) | byte-exact 0.0 |
| gemm\|iq2_s\|prefill\|nr16\|q8_K | 9.93/9.47 | scalar-ref(clang 巨胖 1311ins) | ggml_vec_dot_iq2_s_q8_K_generic | scalar-ref(cheap) | test-only-not-in-denom(★同上) | byte-exact 0.0 |
| gemm\|mxfp4\|prefill\|nr16\|q8_0 | 2.31/2.29 | scalar-ref(generic) | ggml_vec_dot_mxfp4_q8_0_generic | scalar-ref(cheap) | test-only-not-in-denom | byte-exact 0.0 |
| gemm\|tq1_0\|prefill\|nr16\|q8_K | 6.47/6.44 | scalar-ref(generic) | ggml_vec_dot_tq1_0_q8_K_generic | scalar-ref(cheap) | test-only-not-in-denom | byte-exact 0.0 |
| gemm\|tq2_0\|prefill\|nr16\|q8_K | 4.76/4.76 | scalar-ref(generic) | ggml_vec_dot_tq2_0_q8_K_generic | scalar-ref(cheap) | test-only-not-in-denom | byte-exact 0.0 |

### 5.3 BLOCKED 行（5 格·prefill+decode 均保持 pending·禁填）
`{gemm,gevm}|{iq1_s,iq1_m,iq3_xxs,iq3_s,nvfp4}` = **pending-BLOCKED**（无 repack-GEMM/GEVM export·§4）。

### 5.4 DECODE（M=1 GEVM·✅ 本役也测·7 格×3 域·§2.4）
7 格 iq/tq/fp4 **有独立 GEVM repack leaf export**（`rvv-to-emitc-repack-gemv-<fmt>`·5-arg `(K,s,vx,vy,nc)`·weight=同 x16 repack·activation=**plain 单 q8_K/q8_0 向量**）→ decode 已测。T3 `gemm_tile|<fmt>|regime=decode` 行（scalar-ref 类·test-only-not-in-denom）：

| format | rvv gcc(s1/s2) | rvv clang(s1/s2) | k1(s1/s2) | 判读 |
|---|---|---|---|---|
| iq4_xs | 0.62/0.65 named-X | 1.32/1.50 | 0.93/0.94 | 近 parity·gcc named-X |
| iq2_xxs | 0.61/0.60 named-X | 1.25/1.25 | 1.13/1.13 | gcc named-X·余 PASS |
| iq2_xs | 0.97/1.02 | 2.03/2.05 | 3.81/3.82 | PASS(k1 opp clang 胖) |
| iq2_s | 1.04/1.03 | 2.01/2.06 | 3.57/3.54 | PASS |
| mxfp4 | 1.52/1.51 | 1.01/1.01 | 0.94/0.95 | PASS·parity-leaning |
| tq1_0 | 1.72/1.62 | 0.79/0.80 named-X | 1.99/2.00 | rvv-clang boundary named-X |
| tq2_0 | 1.25/1.13 | 1.04/1.03 | 1.59/1.59 | PASS |

**decode tally**：k1 **7/7 PASS**(0.93–3.81×)· rvv-gcc **5 PASS/2 named-X**(iq4_xs/iq2_xxs 0.6)· rvv-clang **6 PASS/1 named-X**(tq1_0 0.79 boundary)。**全 byte-exact**（verify+nc512 GATE nbad=0）。**★decode memory-bound·全域 parity-leaning**（0.6–3.8×·多数 0.8–2×）·**编译器 asymmetry 远弱于 prefill**（GEVM leaf vsetvl 258–2093 << GEMM 5000+·不触 gcc-death）。**同便宜档·禁称硬赢**。

**★须主会/用户裁的 policy（canon 级·本 agent 不自决）**：
1. **scalar-ref gemm 是否入 0.8 硬门 matmul 分母**：opp=ggml `_generic`（as-shipped 但为 fallback·非手调 kernel）→ 建议 **test-only-not-in-denom**（便宜档·同 batch3 ④）。**但注意**：rvv-gcc 部分 opp 是 gcc-autovec（iq4_xs/iq2_xxs/tq1_0/tq2_0 vec>0）·若按 batch3 "opp 真向量→升门" decree 则须升门（此时 rvv-gcc named-X 入分母=负结果登记）。**建议主会明裁 scalar-ref-generic 归 test-only（不升门）**·避免与手调 `_vlNNN`（batch1 vec_dot 轴）混口径。
2. **rvv MAIN 域**：Amdahl 同域律→rvv deploy=gcc-15.2=**named-X 全 7**（若入分母=7 负结果）·clang-micro=footnote。**本 batch 唯一 genuine 正结果 = 7 格双板 byte-exact 正确性 + [CASE-CC-ASYM] 6.8–11.2× codegen 证**。

---

## 6. 污染纪律 + restore（✅ 双板收尾）

- **cold 协议**：32MiB flush per rep·N=25 median+relIQR·2-seed·within-proc·core-pinned（k1 core0-1 idle=100%·rvv core8 idle=100%·gov=performance）。K=2048 nr=16 nc=512（q8_K/8 superblock·mxfp4 q8_0/64 block·weight repacked 278–557KB·flush 保 cold）。
- **stock 只读·双证**：k1 `871169a0` before==after==now UNCHANGED · rvv gcc-15 `d1adc634` before==after==now UNCHANGED（仅编 ours .o + link·未触 .so）。
- **stray**：`pgrep -x bench`=0 双板。**scratch**：`/tmp/g8_a2b6_{k1,rvv,rvv_gcc,rvv_clang}` 全 `rm -rf` GONE 双板。
- **域**：k1=clang-18 部署对称（MAIN）· rvv=gcc-15.2 deploy-clean（MAIN·Amdahl 同域律）+ clang-18 micro（footnote·消歧）。per-lane deploy-matched·非域混杂。
- **git**：无 add/commit·未改 T3/T8/T9/emitter/lib·casefile 独立。

## durable files
- `A2-batch6-iqtq-gemm-scalar.md`（本文）
- `A2-batch6-iqtq-gemm-scalar-raw/iqtq_gemm_scalar_driver.cpp`（7-格 paired 驱动·GEMM(prefill)+GEVM(decode·`-DDECODE_ONLY`)·ZERO-MODEL generic gate）
- `A2-batch6-iqtq-gemm-scalar-raw/kernels/{<fmt>_gemm.c,<fmt>_gevm.c}`（7×2 weft-opt 导出 repack leaf + GEN_SEAL.txt·md5 provenance）
- `A2-batch6-iqtq-gemm-scalar-raw/run_iqtq_{gemm,gevm}_{k1,rvv}.sh`（双板 prefill+decode build/cold 脚本·可复现·rvv dual-domain）
- `A2-batch6-iqtq-gemm-scalar-raw/logs/`（k1 + rvv-gcc + rvv-clang · prefill + decode · build seal + cold raw · 3域×2regime）
