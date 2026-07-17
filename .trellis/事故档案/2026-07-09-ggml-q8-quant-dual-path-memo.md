# ggml 激活量化双路径备忘 (mat-quant vs row-quant) — G3-cert-hardening

**性质**: 纯 doc / 方法学备忘 (T4b M4 验收依据 + 论文素材)。零 lib/schema/board 改。别 commit(由用户提交)。
**血缘**: M2c 决定性重测(commit `53666846`)判 q4_K/q5_K repack-GEMM min-term "5% latent bug" → M1
逐环 instrumented bisect(commit `8ede2779`,cell `experiments/active/t4b-m1-minterm-bisect/`)ZERO-MODEL
证伪:**kernel 对其被真 dispatch 喂入的激活 q8 = BIT-EXACT**,"5%" 是 **cert-harness 激活量化路径失配**伪影。
最早苗头见 commit `2ee8b1d1`("min-term 惯例失配发现 — M0/M1 未测 ggml 真 quantize_mat")。M4 终审在跑;
**本备忘不恢复任何吞吐/byte-exact 口径(等 M4);只定性双路径差 + 钉死验收 oracle 惯例。**

---

## 0. 一句话

ggml 对同一段 f32 激活,GEMM-path 与 RISC-V scalar-path 用**两套不同的 q8_K 量化惯例**;两者都是合法的
q8_K 近似,但**逐 byte 不相等**。我方 repack-GEMM 走 GEMM-path,故其**唯一同源验收 oracle = mat-quant**;
用 row-quant / 独立 lrintf 标量 oracle 去验 GEMM kernel = **喂错激活**,会把 ggml-side 的惯例差**误记成 kernel 缺陷**
(M2c 就是这样误诊的)。

---

## 1. 两条激活量化路径 (定性)

| 路径 | ggml 函数 | 谁走它 | 我方 repack-GEMM 关系 |
|---|---|---|---|
| **mat-quant** | `ggml_quantize_mat_q8_K_4x1` (generic; `ggml-cpu/ggml-cpu-quants` GEMM 前处理) | `ggml_mul_mat` 的 **repack/GEMM** 分支(nr≥4 交织成 `block_q8_Kx4`) | **同源**:真 dispatch 喂给我方 kernel 的正是这份 q8 |
| **row-quant** | `quantize_row_q8_K` (RISC-V arch SIMD 版) | **scalar-path** block-dot fallback(`ggml_vec_dot_q4_K_q8_K`) | 我方 kernel **不吃**它(M1: recompute-from-ROW-q8 mismatches=64/63) |

两路径对同一 f32 激活产出的 q8 **不相等**(M1 直测:`qs diffs 780/2048`,`bsums diffs 64/128`)。差由**两个正交
成分**构成:

### (a) SIGN-FLIP(良性,dot 内抵消)

正-abs-max block 上,RISC-V arch row-quant 取 `d = +1/iscale`,generic mat-quant 取 `d = −1/iscale`
(符号选择惯例不同)。但 `q8 · a_d`(量化码 × 反量化 scale)复原**同一个数值**;而 q4_K·q8_K 的两项
(main `Σ q4·q8` 与 min `dmin·Σ bsums`)对激活符号**同时翻转、点积内相消**——sign-flip-invariant。
∴ 该成分对最终 dot **零净效应**,纯记账噪声。

### (b) ROUND-HALF(真 ±1 q8 噪,近零输出放大)

半整值 x.5 的取整惯例不同:generic `nearest_int` 把 `−63.5 → −63`(round-half-away/toward per libm),
arch SIMD 走 RNE 得 `−64`。约 **~1.5% 元素**落在半整点 → 产生**真 ±1 的 q8 码差**,主要压在 **main 项**。
- **绝对**尺度:R-vs-F(row vs mat)**max ABS err = 12.5 @ F=3662 = 0.34%**(该点相对)。
- **相对**尺度:在**近零、抵消主导**的 super-block 输出上,同样的 ±1 主项差被相消放大成
  **~5–20% RELATIVE**(M2c 观测的 "rel 1.998e-01 / 3.143e+00 / 1.201e+00" + 逐列 parity 0.947/0.637
  就是这个放大在**列奇偶**上的投影,非 kernel 的两-8-lane-strip 缺陷)。
- 结论:**max ABS 0.34% 是真实上界**;>100% 的相对数是**近零分母**产物,不是能量级误差。

> 领先诊断纠偏:M2c HYPOTHESIS 把 parity-alternating 归因到 VLEN128 两-8-lane-strip(half_lanes=8)min 物化。
> M1 q8-FREE 证据(`min_int(oracle-weight-min × KERNEL-自己-bsums) vs kernel v46 = 0 mismatches`,两 block 全
> row/col)**证伪**该假设:min VALUE unpack + bsums pairing + accumulation **结构正确**,d/dmin fp16 = 0 mismatch。
> objdump min-fold 指令级亦正确:`vsetivli zero,8,e32,m2` → `vfmul.vf v30,v6,fa5`(dmin·a_d)→
> `vfmacc.vv v2,v4,v30`(main)→ `vfnmsac.vv v2,v30,v16`(减 dmin·a_d·min_acc)。列-parity 是
> **mat-vs-row 激活差**在交织列布局上的投影,不是 kernel 缺陷。

---

## 2. ★裁定条目 — 验收 oracle 惯例(钉死)

> **我方 repack-GEMM 的一切数值验收 oracle = mat-quant(`ggml_quantize_mat_q8_K_4x1`,与真 dispatch 同源)。**

依据 = M1 ZERO-MODEL(`m1_bisect_result.log`):
- `kernel v47 MAIN vs recompute-from-mat-q8 = 0`,`kernel v46 MIN vs recompute-from-mat-q8 = 0`,
  `kernel captured bsums vs ggml mat-quant bsums = 0` → **kernel 对 mat-quant 激活 bit-exact 计算 q4_K·q8_K**。
- `recompute-from-ROW-q8 mismatches = 64/63` → kernel **不吃** row-quant;拿 row-quant 当 oracle = 系统性喂错。

推论(cert-hardening 三要件之②"输入路径同源"):任何 q4_K/q5_K/q2_K repack GEMM **及 GEVM** 的正确性 cert 必须
- **oracle=mat-quant**(或等价:vs ggml 自己的 repack GEMM,若该格 ggml 有 shipping repack kernel),且
- **分别覆盖** GEMM 的 interleaved-`q8_Kx4` 与 GEVM 的 plain-`q8_K` 两种 bsums 布局(GEVM 侧 M2c 从未直测 = 空白)。

**正交声明**:mat-quant vs row-quant 的分歧(符号选择 + round-half)是 **ggml-side numerics 惯例问题**,
落在激活量化器里,**与本编译器 / 我方 kernel 正交**。我方 kernel 对其被喂的 q8 已 bit-exact;不需改 emitter。

> ★口径纪律:本裁定只钉**验收 oracle 惯例**(方法学),**不**恢复 q4_K/q5_K/q2_K 的 byte-exact / 吞吐兑现
> 口径——那些仍持 M2c-NARROWED(byte-exact=dmin=0-corpus-only、吞吐 pending),等 M4 定案由用户恢复。

---

## 3. T4b M4 seal 的 e2e 贪心 token 一致性 — 不受该差影响 (论证)

M4 seal 关心的是**整模型 e2e 贪心解码 token 序列**在"我方 repack-GEMM path"与"参照 path"下是否一致。双路径
激活量化差(§1)**不威胁**该一致性,理由三点:

1. **两路径皆为 valid q8_K 近似**。mat-quant 与 row-quant 都是同一 f32 激活的 q8_K 量化,**都在 q8_K 量化容差之内**
   (q8_K 本身对 f32 的量化噪声与二者之差同数量级);二者之差(§1b:max ABS 0.34%)**小于** q8_K→f32 的固有量化台阶。
2. **贪心 argmax 对 ~0.3% ABS 扰动鲁棒**。整模型 logit 向量上的 `argmax` 只在**近似并列**(top-1/top-2 gap <
   扰动幅度)才可能翻;~0.3% ABS / sub-±1-q8 的激活噪只在极罕见近平局处才有翻 token 风险,且**该风险对
   任意两个合法 q8_K 量化器之间都存在**,不是我方 kernel 引入的。
3. **kernel 无净误差,差 100% 归因 ggml-side 惯例**。M1 已证 kernel 对 mat-quant bit-exact;故"我方 path vs
   stock path"的任何 token 分歧,**要么**来自 mat-vs-row 激活惯例(§1,可通过强制两 path 同用 mat-quant 消除、
   ggml-side、正交),**要么**来自算法差本身,**都不是** kernel 数值缺陷。

**M4 seal 干净协议(推荐)**:让被比两侧**同喂 mat-quant 激活**(我方 repack-GEMM vs ggml 自己的 repack GEMM,
或 block-dot 喂 mat-quant q8)→ 由 §2 的 kernel bit-exactness,token 逐位相等即可判 seal PASS。**若**只能 vs
stock-as-shipped(row-quant block-dot),则贪心 token 若有分歧,须先归因到 mat-vs-row 惯例(强制同 quant 复测即消)
才判,不得记为 kernel 失败。

---

## 4. 用途 / 落点

- **T4b M4 验收依据**:§2 钉死 oracle=mat-quant;§3 给 M4 seal 的 token 判据 + 干净协议。M4 定案前口径不恢复。
- **论文方法学素材**:双路径量化惯例差 = "定罪与翻案同等严谨"的教材(M2c failure-cert 抓到真现象→误归因;
  M1 ZERO-MODEL bit-exact 逆转→正确归因 ggml-side 惯例);ZERO-MODEL(recompute-from-实际-dispatch-q8)
  = 数值终审的标准工具(排除 oracle 失配这一类空心/错源 cert)。
- **cert-hardening 三要件之②**(输入路径同源)的**规范实例**:本备忘就是"② 违反 → 误诊 → 修正 oracle 惯例"的
  完整案例;并入 T8 `[CASE-MINTERM]` 卷宗 + `[CERT-3REQ]` 补标块。

### 证据指针

| 环节 | 出处 | commit |
|---|---|---|
| M2c 误诊(决定性 outcome-b) | `experiments/active/t4b-m2c-dispatch/m2c_dispatch_result.log` | `53666846` |
| 惯例失配最早苗头 | (T4b M2 journal:"M0/M1 未测 ggml 真 quantize_mat") | `2ee8b1d1` |
| M0 fold 血缘 + cert 语料审计 | `docs/reports/2026-07-09-minterm-fold-audit.md` | `4f8d815c` |
| **M1 ZERO-MODEL 逆转(本备忘证据核心)** | `experiments/active/t4b-m1-minterm-bisect/m1_bisect_result.log` + `MANIFEST.md` + `m1_probe.cpp` | `8ede2779` |
| M4 终审 | (pending) | — |
