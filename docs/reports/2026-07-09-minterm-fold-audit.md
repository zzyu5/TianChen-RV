# 2026-07-09 min-fold 血缘 + cert 语料审计 (G3-minterm-fix M0)

**性质**: 纯 read + doc 审计扫描,先于动刀。零 lib/schema/T8 改。别 commit(由用户提交)。
**背景**: M2c 决定性发现(`experiments/active/t4b-m2c-dispatch/`,snapshot commit 53666846) —
q4_K repack-GEMM 共享 min fold `kquant_dmin_bsums_min` 在 VLEN128 有 **parity-alternating min-term bug**:
真 ggml_mul_mat dispatch 下,dmin=0 语料 rel 4.5e-7 干净,dmin≠0 语料 rel **1.998e-01 / 3.143e+00 /
1.201e+00**(~5%/super-block 到 314% full-output),逐列 R/correct = **0.947 / 0.637 / 0.935 / 0.637**
(偶列 ~0.94、奇列 ~0.64,按权重列 parity 交替)。旧 cert 全用 dmin=0 / identity / 自比 语料 → 漏抓。
领先诊断(**未证/HYPOTHESIS**):VLEN128 两-8-lane-strip(half_lanes=8)min-term 物化,在 ggml 的
16-lane VLEN256 `ggml_gemm_q4_K_16x1_q8_K` 无对应形态,故 M2b PHASE1 source-diff-vs-ggml 结构上抓不到。

---

## 0. 结论速览(自然汇报四问)

1. **共享 `kquant_dmin_bsums_min` 的格式**: **q4_K / q5_K / q2_K**(全部 `hasMin=true`),在 **repack GEMM
   与 repack GEVM 两条路径**都共享。q2_K 判定 = **共享**(`kQ2KDecodeFacts.foldModel="kquant_dmin_bsums_min"`,
   源码注释明写 "REUSES the existing q4_K KQuantDecodeFacts fold WITHOUT any framework generalization")。
   **q6_K / q3_K 不共享**(走 `kquant_single_scale_no_min`,`hasMin=false`,根本无 min-term)。
2. **GEVM/vec_dot 侧 min 路径判定**: repack GEVM(`emitRepackKQuantGemvBodyQ{4,2,5}K`)**有 min-term,
   且与 GEMM 结构同构受累**(同 `vwmacc_vx`-per-strip bsums-min,VLEN128 同样 numHalves=2/half=8 两-strip
   物化)—— 但 GEVM 用 **plain-q8_K** bsums 配对(`a.bsums[2*sub]+a.bsums[2*sub+1]`),GEMM 用
   **interleaved-q8_Kx4** 配对(`a.bsums[gsub*8+m]+a.bsums[gsub*8+m+4]`);M2c 只直测了 GEMM 的 interleaved
   列-parity 签名,GEVM 从未经真 dispatch dmin≠0 直测 → **suspect / pending-audit,不得认 byte-exact**。
   **另两条 min 路径不受累**:(a) block-dot 超块路径(`RVVToEmitCKQuant.cpp` 的 `q4_k_min_term` op)是
   **标量 order-free 归约** `sumi=Σ bsums[j]·mins[j/2]`,非 strip 物化,结构不同、无 parity;(b) dequant-row
   路径(`dequantize-row-q4-k` 等)是**逐元素** `d1·q − m1` 权重重建,**无 activation bsums fold**,无关。
3. **cert 语料审计**: 除 **M2c 一个**(真 dispatch + dmin≠0 + 逐元素独立 oracle,恰好抓到缺陷)外,
   **所有 K-quant min cert 语料 under-tested**:INT/byte-exact 行全 `dmin=0`;NORM 行虽 dmin≠0 但用**自比
   (PRE≡POST / untiled≡tiled)或非-dispatch 独立 harness**,不复现真 ggml interleaved-q8_Kx4 dispatch 的
   min 算术,故一致漏抓;emit-golden lit(FileCheck 文本)结构上**无法**抓任何数值缺陷(与本审计正交)。
4. **受累范围**: q4_K(**已证** defective,GEMM)+ q5_K/q2_K(共享 fold,**inferred / pending**),
   × {GEMM, GEVM} × VLEN128。T8 已有 2026-07-09 裁决一.1 收窄块(行 26–45)对应,本审计补齐 **GEVM 与 q2_K
   NORM cert 的 hollow 判定** + **cert 逐项语料表**。

---

## 1. Fold 血缘清单(格式 × 路径 × 是否共享该 fold × VLEN128 两-strip 物化)

血缘源:`lib/Plugin/RVV/RVVLowerQuantContraction.cpp`(`KQuantDecodeFacts` 常量,行 340–480)。
Emitter 分派:`lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp`(GEVM 门 1761、GEMM 门 2475)。

| 格式 | foldModel | hasMin | nSub | GEMM emitter (行) | GEVM emitter (行) | 共享 `kquant_dmin_bsums_min`? | VLEN128 两-8-lane-strip min 物化? |
|---|---|:--:|:--:|---|---|:--:|:--:|
| **q4_K** | `kquant_dmin_bsums_min` | ✔ | 8 | `emitRepackKQuantGemmBodyQ4K` (6108) | `emitRepackKQuantGemvBodyQ4K` (5023) | **是(canonical)** | **是**(numHalves=2, half=8) |
| **q2_K** | `kquant_dmin_bsums_min` | ✔ | 16 | `…GemmBodyQ2K` (9853) | `…GemvBodyQ2K` (9335) | **是**(整体复用 q4_K fold,零 framework 泛化) | **是** |
| **q5_K** | `kquant_dmin_bsums_min` | ✔ | 8 | `…GemmBodyQ5K` (7425) | `…GemvBodyQ5K` (6838) | **是**(复用 q4_K fold + qh 5th-bit plane) | **是** |
| q6_K | `kquant_single_scale_no_min` | ✗ | 16 | `…GemmBodyQ6K` (8681) | `…GemvBodyQ6K` (8185) | **否(无 min-term)** | N/A |
| q3_K | `kquant_single_scale_no_min` | ✗ | 16 | `…GemmBodyQ3K` (15442) | `…GemvBodyQ3K` | **否(无 min-term)** | N/A |

> `hasMin=false` 的 q6_K/q3_K:偏移全走单累加器 no-min fold(`weightDminByteOffset=0`,
> `gevm/gemmActivationBsumsByteOffset=0`),根本不读 bsums、不算 min-term → 与本缺陷无关。

### 1a. 各 min 路径的算术结构(用于同构判定)

| 路径 | 文件:行 | min-term 结构 | bsums 配对 | 两-strip 物化 | parity 受累 |
|---|---|---|---|:--:|:--:|
| **repack GEMM** | BlockQuantLinear.cpp 6659–6684 / 6749–6782 | `bsums_acc[m] += bsum_pair·min_strip`(vwmacc_vx,per strip),尾 `sumf_c −= dmin_x·d_y_c·bsums_c`(vfnmsac) | **interleaved q8_Kx4** `a.bsums[gsub·8+m]+a.bsums[gsub·8+m+4]` | 是(h=0 列0-7 / h=1 列8-15) | **是(M2c 已证)** |
| **repack GEVM** | BlockQuantLinear.cpp 5421–5443 / 5498–5537 | 同上 vwmacc_vx-per-strip + vfnmsac 尾 fold | **plain q8_K** `a.bsums[2·sub]+a.bsums[2·sub+1]` | 是(numHalves=2) | **suspect(未直测)** |
| block-dot 超块 | RVVToEmitCKQuant.cpp 1390–1537(`q4_k_min_term` op) | **标量 order-free** `sumi=Σ bsums[j]·mins[j/2]`;`sumf −= dmin·(float)sumi`(单 emitc.expression) | 标量逐-j,无 strip | **否** | 否(结构不同) |
| dequant-row | test/…/rvv-to-emitc-ggml-dequantize-row-q4-k.mlir(+ q2-k / q5-k) | 逐元素权重重建 `d1·q − m1`(get_scale_min_k4) | **无 activation bsums** | 否 | 否 |

**判定**:缺陷住在 **repack 家族(GEMM+GEVM)**的 per-strip vwmacc bsums-min 物化。block-dot 与 dequant 的
min 是**另两种结构**(标量归约 / 逐元素),不共享该 fold、不受两-strip parity 影响。

---

## 2. GEVM/vec_dot & dequant min 路径判定(自然汇报第 2 问细化)

- **repack GEVM(q4_K/q5_K/q2_K)= 同构受累、pending-audit**。分派门 `if (loopBody.getFoldModel() ==
  "kquant_dmin_bsums_min")`(BlockQuantLinear.cpp:1761)与 GEMM 门 2475 **同一 foldModel**;GEVM body 的
  min-term(5421 `min_bsums_fold` step + 5498 尾 fold)与 GEMM 逐行同构(同 `vwmaccVX32` + 尾 `vfnmsac`,
  同 numHalves=2/half=8 两-strip)。差异仅在 bsums 配对布局(plain vs interleaved)。M2c 只经 mul_mat(nr≥4)
  直测了 GEMM interleaved 列-parity;**GEVM 无任何真-dispatch dmin≠0 数值 cert**(见 §3),故:
  **若** 根因是「两-8-lane-strip 物化」(领先诊断),GEVM **同样受累**;**若** 根因是 GEMM interleaved-bsums
  列索引,GEVM 可能幸免。二者皆未证 → **GEVM min-term 归为 pending-audit,修复前不得认 byte-exact**。
- **block-dot 超块 `q4_k_min_term`(RVVToEmitCKQuant.cpp)= 不受累(结构不同)**。它是**标量整数 order-free
  归约**(注释:"integer multiply/add is associative/order-free",行 1399/1464),不做 SEW8 strip 物化、
  不分 8-lane half,无 parity 交替风险。但注意:它同样**未经真 dispatch dmin≠0 数值直测**(M2c log 行 43
  "TCRV Q4_K BLOCK-DOT ENGAGED" 仅在 INT N=160 形状出现,非被数值比对路径)→ 风险低但非零,建议一并纳入
  未来 dmin≠0 直测清单(非本缺陷共享面)。
- **dequant-row(q4_k/q2_k/q5_k)= 不受累**。逐元素 `d1·q − m1`,**无 activation bsums fold**,与
  「min×Σactivation 折叠」是不同数学;不经 repack 路径。

---

## 3. 全表 cert 语料审计(逐 cert 标注是否 exercise 每一算术项)

**判据**:一个 cert 若要抓住本缺陷,其语料必须同时满足 (a) **dmin≠0**(min-term 激活);(b) **mins≠0**
非退化 scale;(c) **经真 ggml_mul_mat interleaved-q8_Kx4 dispatch**(或等价复现该 bsums 布局);(d) 参照系是
**逐元素/逐列独立** oracle(而非与被测 kernel 共享同一 bsums 读法的自比)。缺 (a) 或 (c)(d) 者降 **partial**。

| # | cert / 出处 | 轴 | 语料:dmin≠0 | mins≠0 / 非退化 scale | 符号覆盖 | 真-dispatch interleaved? | 参照系 | **判定** |
|---|---|---|:--:|:--:|:--:|:--:|---|---|
| 1 | **M2c** `t4b-m2c-dispatch/`(53666846) | 真 dispatch 数值 | ✔ (NORM `x->dmin=DHALF[…]`) | ✔ | ✔(ri8 −127..127 类) | **✔ ggml_quantize_mat_q8_K_4x1** | **逐元素独立** `oracle_dot2`(`mins+=m0·a8`) | **FULL**(唯一充分语料;**是 failure-cert**,抓到 rel 5–314% + 列-parity 0.947/0.637) |
| 2 | M0 tracer `t4b-m0-q4k-tracer/m0_q4k_result.log`(039133ea) | probe+数值 | INT=`dmin=0`;NORM=✔ | 部分(NORM adversarial) | 部分 | **✗ 非真 mul_mat**(golden kernel + 探针,link libggml 但非 dispatch) | vs stock-ggml(NORM worst rel **2.4e-5**) | **PARTIAL**:INT byte-exact 仅 dmin=0;NORM 虽 min-active 却 **worst 2.4e-5** ≪ M2c 2e-1 → 语料不复现真-dispatch min 算术,漏抓 |
| 3 | M1 repacker cert `t4b-m1-repacker/`(q4_K+q5_K) | 数值 | INT=`dmin=0`;NORM=✔ | 部分 | 部分 | **✗ 独立 harness** | vs stock-ggml(NORM worst rel **1.18e-4**) | **PARTIAL**:同 #2 结构;q5_K INT byte-exact 也仅 dmin=0 |
| 4 | q4_K L1-candidate `kquant-l1-q4k-q5k-repack-prefill/`(T8 行 46) | 吞吐+correctness | 未 re-verify(引 039133ea) | — | — | ✗ | "construction oracle NOT re-verified this run" | **PARTIAL**:correctness 全托 039133ea 构造 oracle(该 oracle 已被 M2c 证漏抓) |
| 5 | q5_K L1-candidate(T8 行 47) | 吞吐+correctness | 引 039133ea | — | — | ✗ | 同上 | **PARTIAL**(同 #4;q5_K 共享 fold) |
| 6 | q2_K L1 SILICON numeric `kquant-l1-q6q2q3-repack/`(T8 行 49) | silicon 数值 | INT=`dmin=0`;NORM=✔(dual d/dmin+bsums-min) | ✔ 声明 incl. 折叠 | 部分 | **✗ 独立 harness(非 mul_mat)** | INDEPENDENT per-block ref;NORM `maxAbsErr/rms=**5.77e-7**` | **PARTIAL / ★HOLLOW**:NORM 声明「incl dual d/dmin + bsums-min」却仅 5.77e-7 → 与 q4_K 同 fold 却漏抓,证明其独立 ref **与 kernel 共享 bsums 读法**或语料列-对称;T8 已标 q2_K `PENDING-AUDIT` |
| 7 | q6_K L1 SILICON numeric(T8 行 48) | silicon 数值 | NORM(无 min-term) | — | — | ✗ | per-block ref NORM 6.83e-7 | **N/A 本缺陷**(q6_K no-min);记录其 harness 与 #6 同源、同样非-dispatch |
| 8 | l1-tile-s1 `tile_s1_findings.md`§0 | 自比 identity | int=unit scales;norm=finite | 退化(int regime d=dmin=1.0) | — | ✗ | **PRE(golden)≡POST(S1) 自比** | **PARTIAL**:自比只证 tiling 保序不变量,**无法**抓两变体共享的缺陷;绝对正确性托 039133ea |
| 9 | l1-tile-s6 `tile_s6_findings.md`§0(T8 行 53) | 自比 identity | int=unit scales;norm=finite | 退化 | — | ✗ | **S1≡S6 自比** | **PARTIAL**:同 #8;M2c 正是证明 S1/S6 **同时**带缺陷、自比 0-mismatch 仍漏 |
| 10 | l1-reroll / l1-pipeline q4_K | 自比 identity | — | — | — | ✗ | untiled≡reordered 自比 | **PARTIAL**(相对不变量;correctness 引 039133ea bounded-norm,未 re-verify) |
| 11 | l1-t3-q5k `tile_q5k_t3_findings.md`§0(T8 行 57) | 独立 oracle + 自比 | ✔ bsums 激活(mean 186.1) | ✔;NOMIN/PERM/ROWROT 反例均触发 | ✔(ROWROT 验 4-row interleave) | **✗ 独立 harness `oracle_q5K.cpp`** | INDEPENDENT scalar q5_K;WORST **8.0e-7** | **PARTIAL / ★SUSPECT**:反例守卫最强(NOMIN 149291×、ROWROT 6.15M×)却仍 8e-7 → 其 harness 未复现真-dispatch interleaved 列-parity;M2c 明推 q5_K 带同缺陷 → 语料不充分 |
| 12 | l1-t3-q2k `tile_q2k_t3_findings.md`§0(T8 行 55) | 独立 verify + 自比 | INT=`dmin=0`;NORM 6.6e-7 | 部分 | — | ✗ | INT_mismatch=0 + untiled≡tiled | **PARTIAL**(同 #6/#9;q2_K 共享 fold,PENDING-AUDIT) |
| 13 | k1-VLEN256 t4a `run_k1_identity_timing.sh`(T8 行 60) | 自比 identity | int/norm 两模 | — | — | ✗(且 **VLEN256** 16-lane 单-strip) | golden≡S6 @VLEN256 自比 | **PARTIAL / 无关**:VLEN256 单-strip 本就无两-strip 缺陷(M2c:"NO counterpart in 16-lane VLEN256"),故 identical 属预期、**不构成 VLEN128 语料证据** |
| 14 | emit-golden lit(见 §3a 清单) | FileCheck 文本 | — | — | — | — | 无(检查 emitted-C 文本形状) | **N/A-numerics**:结构上无法抓任何数值缺陷,与本审计正交;**不得当 correctness 证据** |

### 3a. 归入 #14(emit-golden,数值无关)的 lit 清单

`test/Conversion/RVV/`:`rvv-to-emitc-repack-gemm-q{4,5,2}-K-q8-K.mlir`、`rvv-to-emitc-repack-gemv-q{4,5,2}-K-q8-K.mlir`、
`rvv-emit-quant-contraction-q{4,5,2}-K-repack-gemm-prefill-vlen128.mlir`、`rvv-emit-identity-quant-contraction-q{4,5,2}-K-repack-vlen128.mlir`、
`rvv-to-emitc-q4-k-min-term.mlir`、`rvv-to-emitc-q4-k-scale-min-bit-dance.mlir`、`rvv-to-emitc-q4-k-sums-fold-scale-d.mlir`、
`rvv-to-emitc-q4-k-scaled-dot.mlir`、`rvv-to-emitc-q4-k-horizontal-fold.mlir`、`rvv-to-emitc-q4-k-nibble-unpack.mlir`;
`test/Dialect/RVV/q4-k-min-term-dataflow.mlir`、`q4-k-sums-fold-scale-d-dataflow.mlir` 等对应 dataflow;
`test/Target/RVV/q{4,5,2}-k-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir`(block-dot 路径,§1a 第3行,另结构)。
→ 全部只锁 emitted-code 形状,**任一都无 dmin≠0 数值断言**;它们「PASS」不等于 min-term 正确。

---

## 4. 受累格式 / cert 清单(动刀 & 重-cert 的 touch-set 输入)

### 4a. 受累格式 × 路径

| 格式 | repack GEMM VLEN128 | repack GEVM VLEN128 | 证据等级 |
|---|:--:|:--:|---|
| q4_K | **DEFECTIVE(已证)** | **suspect(未直测)** | M2c 真-dispatch rel 5–314%,列-parity 0.947/0.637 |
| q5_K | **inferred-defective** | **suspect** | 共享 fold;M2c 明推「expected to carry SAME defect;do NOT certify until fixed」 |
| q2_K | **inferred-defective / pending** | **suspect** | 共享 fold(整体复用);T8 标 `q2_K-PENDING-AUDIT`;从未经 M2c 直测 |

**不受累(不收窄)**:q6_K / q3_K(`kquant_single_scale_no_min`,无 min-term,且本就 LOSS);
iq4_nl / iq4_xs 码本(无 fold);q4_0 / q5_0 / q8_0 flat(无 min);block-dot 超块 `q4_k_min_term`(标量
order-free,§1a 第3行);dequant-row(逐元素,无 bsums)。

### 4b. 需降级 / 重-cert 的 cert 清单

- **降 partial(byte-exact 仅 dmin=0 corpus + 自比不变量,非绝对正确性)**:M0 tracer、M1 repacker cert、
  q4_K/q5_K/q2_K L1-candidate、q2_K/q6_K L1 SILICON numeric、l1-tile-s1/s6/reroll/pipeline identity 门、
  l1-t3-q5k(`oracle_q5K.cpp` NORM)、l1-t3-q2k、k1-VLEN256-t4a identity。**★特别标注**:#6(q2_K SILICON
  NORM 5.77e-7)与 #11(q5_K oracle NORM 8.0e-7)是**声称 min-active 却 hollow** 的两个 —— 其独立 ref 或与
  kernel 共享 bsums 读法、或语料列-对称,不复现真-dispatch interleaved 列-parity。
- **N/A-numerics(结构上无关,别当 correctness)**:§3a 全部 emit-golden lit。
- **唯一充分 cert**:**M2c**(且是 failure-cert)。→ 任何 q4_K/q5_K/q2_K repack GEMM **及 GEVM** 的 byte-exact
  重-cert **必须**经真 ggml interleaved-activation dispatch、dmin≠0、逐元素独立 oracle,并**分别覆盖 GEMM
  的 interleaved-q8_Kx4 与 GEVM 的 plain-q8_K 两种 bsums 布局**(GEVM 尤其空白)。

### 4c. 与 T8 既有裁决的对账

T8(`experiments/active/result-tables/T8_winloss_gap_ledger.csv` 行 26–45)已有 2026-07-09 裁决一.1 收窄块,
覆盖:① byte-exact = dmin=0-corpus only + PRE≡POST identity;② vs-opponent 边际数 pending-重测;③ q2_K
pending-audit;不收窄 [XFER-1] register-cliff 分类 / spill 计数(tiling PRE≡POST 相对不变量)。
**本审计补齐三点**:(i) **GEVM 侧**同构受累判定(T8 收窄块只列 GEMM 行,未显式点名 GEVM min-term);
(ii) #6/#11 两个 **NORM「min-active」cert 实为 hollow** 的逐项定性(非仅 INT dmin=0 行);
(iii) block-dot `q4_k_min_term` 与 dequant-row 两条 min 路径的**不受累**结构证据(排除误伤面)。
