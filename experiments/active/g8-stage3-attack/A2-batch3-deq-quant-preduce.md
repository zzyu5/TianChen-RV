# A2 batch3 — ③ quantize 3 + ④ product_reduce 3 + ⑤ dequant FLAT-5（双板·per-lane deploy-matched 对称·真板测）

> **任务**：线 A 板测执行员 A2-batch3 = **③ quantize 3**（`quantize_row/q8_0·q8_1·q8_K`·向量仗）+ **④ product_reduce 3**（`q4_0_nibble·offset_binary_n3·codebook_n3`·标量 sanity）+ **⑤ dequant 未测行 FLAT-5**（`dequantize_row/q4_0·q4_1·q5_0·q5_1·q8_0`·标量仗）。承 A2-batch1/batch2（harness recipe 复用）。
> **赛道**：kernel-axis MICRO（同编译器/flags/march 对称）。**NOT e2e·NOT perf-covered 9/83·不入系统账**。[NG-4]。dequant = **[DEQ-AXIS] 独立子账**·不进 matmul headline。
> **口径铁线**：cold 唯一·**per-lane deploy-matched 对称**（k1 shipped=clang-18=部署域；rvv shipped=gcc-15.2 → gcc-15.2=部署-clean·**avoid clang-micro-on-rvv artifact** [CASE-COMPILER-ASYMMETRY]·Amdahl 同域律）·**N≥20 中位+2-seed**·**禁一切继承**（FLAT-5 dequant 存量零·quantize/product_reduce 从未测）·每格对手身份探针（objdump 三判据）·预注册判读（cold≥0.8=PASS / <0.8=具名-X+墙）·0 样本不造数·**便宜档禁称硬赢·接线≠转绿·预判不作结论**·byte-exact ZERO-MODEL 正确门。
> **测于**：2026-07-16 · rvv(VLEN128·core8-15 idle-pick·gcc-15.2) + k1(VLEN256·SpacemiT-X60·clang-18) · 主树/build/stock `.so`/governor 未改·0 stray·无 git。

---

## 0. ★进度看板（✅ 三子项双板全完成·全 byte-exact·2-seed·N=24·卫生 clean）

| 子项 | 状态 | rvv (gcc-15.2) | k1 (clang-18) |
|---|---|---|---|
| ⑤ dequant FLAT-5 | ✅ 完成双板 | vec-vs-vec·2 PASS/3 X·升 0.8 门 | opp scalar·5 WIN [DEQ-AXIS]·维持 |
| ③ quantize 3 | ✅ 完成双板 | 3 PASS parity-roofline | 3 PASS parity-roofline |
| ④ product_reduce 3 | ✅ 完成双板 | 3 PASS-sanity(cheap) | 2 PASS/1 X(cheap) |

**净**：0 verified hand-brick·0 net-new matmul beat·全 byte-exact ZERO-MODEL·stock 库 md5 before==after·scratch GONE·无 git。

---

## 1. 域政策（★per-lane deploy-matched·记录留痕）

**为何非 "clang-18 双板对称"**：③⑤ 对手 = **stock-lib as-shipped**（`dequantize_row_*` @ base.so · `quantize_row_q8_*` @ cpu.so）·对手编译器由 stock 库固定 = 该板部署编译器（rvv=gcc-15.2 / k1=clang-18）。"ours" 用**同板编译器**编 → **per-lane deploy-matched 对称**（每 lane 对手编译器 = 其部署编译器）。这 = census/deq-axis-reparse DEQ 既定约定·且 = **Amdahl 同域律正门**（rvv 用 clang-micro 喂部署 = garbage-in·[CASE-COMPILER-ASYMMETRY] 前车）。**rvv 记 gcc-15.2 部署域·非 clang-18-micro**·k1 记 clang-18 部署域。逐格明标 per-lane 编译器。

（结果逐格填于下方 §2/§3/§4·增量写。）

---

## 2. ⑤ dequant FLAT-5（q4_0/q4_1/q5_0/q5_1/q8_0·[DEQ-AXIS]·byte-exact ZERO-MODEL 全 0mism/0ULP）

> **性质**：streaming dequant（读 quant 块→写 K f32·K=1048576·out=4MiB > L2·224MiB flush·N=24·2-seed）。**FLAT-5 dequant 存量零**（census 18 格 = K-quant/iq/fp4/tq·**FLAT-5 从未测**）→ 真板补测。对手 = stock base.so 真派发 `dequantize_row_<fmt>`（to_float 部署路·机判 whole-symbol）。
> **★板异（成色本质差·同 census/deq-axis-reparse 规律）**：**rvv opp = gcc-15.2 全 TRUE-VEC**（autovec·真向量对手→按 decree auto-promote **升 0.8 硬门**·vec-vs-vec 公平）；**k1 opp = clang-18 全 SCALAR**（deployed-scalar-ref·**维持 DEQ-AXIS**·不升 0.8·ours-vec/scalar vs scalar-ref 弱赢类）。

### 2.1 rvv（VLEN128·gcc-15.2 部署域·vec-vs-vec·0.8 硬门内）

| 格 | opp class(rvv-ins) | ours(vsetvl/rvv-ins) | cold_med s1 | cold_med s2 | cold_best s1/s2 | o_iqr% | **predreg** | 成色 |
|---|---|---|---:|---:|---:|---:|:--:|---|
| q4_0 | TRUE-VEC autovec 11 | 17/16 | 0.7078 | 0.6975 | 0.698/0.702 | 1.2–2.2 | **具名-X** | opp autovec 更快(5.5 GB/s)·清净<0.8·vec-vs-vec |
| q4_1 | TRUE-VEC autovec 11 | 17/16 | 1.3870 | 1.1807 | 1.384/1.215 | 5.1–6.6 | **PASS(WIN)** | ★vec-vs-vec 真赢·noisy o_iqr·双 seed 均>1.0 |
| q5_0 | TRUE-VEC autovec 34 | 17/16 | 0.6183 | 0.6182 | 0.618/0.621 | 2.0–2.6 | **具名-X** | opp heavy autovec(34rvv·qh 5-bit)·清净<0.8 |
| q5_1 | TRUE-VEC autovec 32 | 17/16 | 0.7449 | 0.7018 | 0.638/0.557 | 0.2–2.0 | **具名-X** | med&best 双 seed 均<0.8·稳判 |
| q8_0 | TRUE-VEC autovec 13 | 17/16 | 0.8392 | 0.8372 | 0.842/0.842 | 0.4–0.8 | **PASS(marginal)** | tight IQR·稳过·vec-vs-vec near-parity |

**rvv tally**：**2 PASS**（q4_1 WIN 1.18–1.39× · q8_0 marginal 0.84×）· **3 具名-X**（q4_0 0.70 / q5_0 0.62 / q5_1 0.70–0.74）。全 vec-vs-vec genuine·byte-exact。**墙（3 X）= 对手结构优势具名**（gcc-15.2 autovec `dequantize_row_*` thin+紧凑·q5_0/q5_1 opp 34/32 rvv 强）。

### 2.2 k1（VLEN256·clang-18 部署域·opp SCALAR→[DEQ-AXIS] 维持·不升 0.8）

| 格 | opp class | ours-side(机判) | cold_med s1 | cold_med s2 | ratio_hot | **verdict [DEQ-AXIS]** | 成色（诚实） |
|---|---|---|---:|---:|---:|:--:|---|
| q4_0 | SCALAR(6ins) | **SCALAR**(0 vec) | 1.4311 | 1.4433 | 1.43 | **WIN 1.43×** | scalar-vs-scalar·ours 紧凑标量·弱赢 |
| q4_1 | SCALAR(9ins) | **SCALAR**(0 vec) | 1.3913 | 1.4007 | 1.36 | **WIN 1.40×** | scalar-vs-scalar·弱赢 |
| q5_0 | SCALAR(6ins) | light-vec(7) | 4.1677 | 4.2723 | 4.17 | **WIN 4.17×** | ours-light-vec vs opp-scalar·**opp-immaturity**(deployed scalar-ref)·大倍数非硬赢 |
| q5_1 | SCALAR(8ins) | light-vec(7) | 4.4943 | 4.3468 | 4.53 | **WIN 4.49×** | 同上·opp-immaturity·大倍数非硬赢 |
| q8_0 | SCALAR(6ins) | **SCALAR**(0 vec) | 1.0649 | 1.0684 | 1.06 | **WIN 1.07×** | scalar-vs-scalar·marginal·弱赢 |

**k1 tally**：**5 WIN**（DEQ-AXIS·vs deployed-scalar-ref）。**成色诚实**：q4_0/q4_1/q8_0 = scalar-vs-scalar（ours 紧凑标量·弱赢 1.07–1.43×）；q5_0/q5_1 = ours-light-vec vs opp-scalar（4.2–4.5× 大倍数·**opp-immaturity·非硬赢**）。**opp 全 SCALAR → 按 auto-promote decree 维持 DEQ-AXIS 子账·不进 k1 0.8 硬门分母**（同 census k1 全 18 格）。

### 2.3 ⑤ 卫生 + 域披露
- **byte-exact**：全 10 格-板点 0mism/0ULP（ZERO-MODEL·各板独立 stock base.so oracle）。
- **stock 库只读**：rvv base.so md5 `1b4580c4` before==after · k1 base.so md5 `00267134` before==after · **未改 .so**（仅编 ours .o + link driver）。
- **stray**：`pgrep -x dequant_flat5`=NONE 双板（run 内 STRAY=3 = `-f` 误匹配 ssh/tee cmdline·假阳·`-x` 实测 0）。load-gate rvv core8 idle=100%/k1 core1 idle=100%。
- **域**：rvv=gcc-15.2 部署（vec-vs-vec·0.8 硬门）· k1=clang-18 部署（scalar 对手·DEQ-AXIS）。per-lane deploy-matched·**非域混杂**（§1）。

---

## 3. ③ quantize 3（quantize_row_q8_0/q8_1/q8_K·向量仗·byte-exact 输出 ZERO-MODEL 全 0mism）

> **性质**：streaming quantize（读 K f32→写 quant 块·K=1048576·in=4MiB f32 > L2·224MiB flush·N=24·2-seed·**data-dependent**·byte-exact = 比对**输出 quant 字节** bit-for-bit vs stock oracle）。对手 = stock cpu.so 真派发 `quantize_row_q8_<fmt>`（arch/riscv/quants.c 手写 `__riscv_v` intrinsic·机判 whole-symbol·**双板全 TRUE-VEC·inline·非 dispatch-to-generic**）。**从未测**·真板补测。
> **★双板全 vec-vs-vec**：ours 发 ggml 精确 RVV 法（`vfredmax` amax + f32→i16→i8 RNE narrow + int8 store·EMIT fixture 锁）→ byte-exact by construction。opp 亦 TRUE-VEC → 公平 kernel-sym 向量仗·**升 0.8 硬门**。

### 3.1 rvv（VLEN128·gcc-15.2 部署域）

| 格 | opp(rvv-ins·inline) | ours(vsetvl/real-vec) | cold_med s1 | cold_med s2 | o_iqr% | our/opp GB/s | **predreg** |
|---|---|---|---:|---:|---:|---|:--:|
| q8_0 | TRUE-VEC 6 | 6/15 | 1.0016 | 0.9980 | 0.06–0.14 | 2.83/2.82 | **PASS(parity)** |
| q8_1 | TRUE-VEC 5 | 6/12 | 0.9993 | 1.0002 | 1.24–1.72 | 2.09/2.09 | **PASS(parity)** |
| q8_K | TRUE-VEC 9 | 12/23 | 1.0013 | 0.9996 | 5.4–14.5 | 2.22/2.22 | **PASS(parity)** |

### 3.2 k1（VLEN256·clang-18 部署域）

| 格 | opp(rvv-ins·inline) | ours(vsetvl/real-vec) | cold_med s1 | cold_med s2 | o_iqr% | our/opp GB/s | **predreg** |
|---|---|---|---:|---:|---:|---|:--:|
| q8_0 | TRUE-VEC 4 | 4/10 | 0.9987 | 1.0013 | 0.34–1.34 | 1.61/1.61 | **PASS(parity)** |
| q8_1 | TRUE-VEC 6 | 6/12 | 1.0000 | 1.0009 | 0.33–0.90 | 1.47/1.47 | **PASS(parity)** |
| q8_K | TRUE-VEC 7 | 11/24 | 1.0080 | 0.9997 | 0.62–1.82 | 1.86/1.86 | **PASS(parity)** |

### 3.3 ③ 结论 + 成色（诚实）
- **6/6 格-板点 PASS·全 byte-exact 0mism·全 vec-vs-vec**。
- **★成色 = memory-bound roofline parity（"对手贴墙 parity=满分"·非 beat）**：quantize = streaming（读 4MiB f32 + 写 ~1.1MiB quant）·compute（amax+RNE narrow）藏于内存流下 → cold A/B **贴死 1.00×**（0.998–1.008 全格·our_GB/s==opp_GB/s 逐格）。两侧同贴 DRAM 墙（rvv 2.1–2.8 GB/s·k1 1.5–1.9 GB/s）→ ratio≈1.0 与码质无关。**干净竞争性 PASS（parity·非硬赢）**·ours emit 竞争力足（发 ggml 精确 RVV 法·byte-exact·未付额外税）。
- **计数纪律**：kernel-axis 向量仗 parity·**NOT e2e·NOT perf-covered·不入 matmul headline**。parity-competitive（对手 TRUE-VEC·非 immaturity）·**非独立 beat**。
- **卫生**：rvv cpu.so md5 `d1adc634` before==after · k1 cpu.so md5 `871169a0` before==after·未改 .so。stray `pgrep -x quantize3`=0。load-gate rvv core8 / k1 core7 idle=100%。

---

## 4. ④ product_reduce 3（q4_0_nibble/offset_binary_n3/codebook_n3·标量 sanity·预注便宜档）

> **性质·SANITY 层**：三个 weft product-reduce **整数子原语**（内部 leaf·非独立算子）。**ggml 无 standalone product_reduce framework 对手**（T-CENSUS §一.F ❌ 缺条件②）→ 公平参照 = **构造的 scalar-ref oracle**（计算**同一整数数学**·driver 内 `sc_*`）。**vs-scalar = 便宜档·大倍数预期·NON hard-gate**（标量对手·非 as-shipped framework 核）。
> **口径**：NB=262144 blocks × 16-lane（每 block→1 int32 reduction·per-block kernel call·streaming·224MiB flush·N=24·2-seed）。byte-exact ZERO-MODEL = `out_ours[b]==out_scalar[b]` 整数精确（纯整数·无 overflow·任意归约序等价）。
> **原语数学（机读 emitc·oracle 逐字镜像）**：nibble=`acc+=(b&0xF)*a[i]+(b>>4)*c[i]`（q8_1 unsigned-nibble）· offbin=5th bit from qh[2..3]/[4..5]·`val=(nib|bit<<4)-16`（q5_0 five-bit·n=16 单 strip 保 byte-exact 跨 VLEN）· codebook=`val=iq4_nl_kvalues[nib]`（q8_0 codebook-gather vrgather）。

### 4.1 逐格 cold（ours-vec / scalar-ref·>1=ours 更快·byte-exact 全 0mism）

| op | 板 | ours(vsetvl/vec_core) | cold_med s1 | cold_med s2 | cold_best | ratio_hot | **predreg[SANITY·cheap]** |
|---|---|---|---:|---:|---:|---:|:--:|
| nibble | rvv | 11/10 | 2.7217 | 2.7006 | 2.75–2.80 | 3.65 | **PASS-sanity 2.7×** |
| offbin | rvv | 11/9 | 0.8580 | 0.8507 | 0.857 | 1.26 | **PASS-sanity 0.85×(marginal)** |
| codebook | rvv | 17/14 | 1.9950 | 1.9607 | 1.96–2.00 | 2.04 | **PASS-sanity 2.0×** |
| nibble | k1 | 6/5 | 1.1636 | 1.1683 | 1.17 | 1.14 | **PASS-sanity 1.17×** |
| offbin | k1 | 10/9 | 0.7961 | 0.7967 | 0.796 | 0.66 | **具名-X 0.80×**(双 seed 均<0.8·边界) |
| codebook | k1 | 8/7 | 1.0961 | 1.0956 | 1.10 | 1.07 | **PASS-sanity 1.10×** |

### 4.2 ④ 结论 + 成色（诚实·便宜标便宜）
- **全 6 格-板点 byte-exact vs scalar-ref**（int_mismatch=0·ZERO-MODEL·数值正确性坐实）。
- **★成色 = 便宜档 SANITY·非硬赢**（对手 = 构造 scalar-ref·非 as-shipped framework）。tally：**rvv 3 PASS-sanity**（nibble 2.7× / codebook 2.0× / offbin 0.85× marginal）· **k1 2 PASS-sanity + 1 具名-X**（offbin 0.80×）。
- **★关键诚实发现（leaf-granularity 天花板）**：16-lane leaf block 粒度**欠用宽 VLEN**（k1 VLEN256 e8m1 max vl=32·仅用 16→半浪费）→ vector-vs-scalar 优势 **rvv 中(2×)·k1 弱(1.1×)**·**offbin 双板最差(rvv 0.85/k1 0.80)**：per-call `vid`+`qh` 重构+`vsetvl` setup 在 16-lane 粒度不摊销 → vector 版反被 scalar-ref 追平/反超（k1 hot 0.66×）。=**"micro leaf 粒度 vector setup 税" 教材·非 emitter 缺陷**（真用嵌入更大循环内联·此处 per-block call 是 sanity 隔离测量的固有开销）。
- **计数纪律**：SANITY 层·**NOT e2e·NOT perf-covered·NOT kernel-sym matmul headline·不入任何系统账**（T-CENSUS §一.F 判 ❌ 缺框架对手·本测坐实"vs-scalar sanity 层"·大倍数禁称硬赢）。
- **卫生**：无 ggml lib 链接（scalar-ref 内置）·无 stock 库触碰。stray `pgrep -x preduce3`=0。load-gate rvv core8 / k1 core2 idle=100%。

---

## 5. ★净结论汇总（三子项·双板·全 byte-exact·2-seed·N=24）

| 子项 | rvv (gcc-15.2 部署域) | k1 (clang-18 部署域) | 赛道/账 |
|---|---|---|---|
| **⑤ dequant FLAT-5** | vec-vs-vec·**2 PASS**(q4_1 WIN 1.18–1.39×·q8_0 0.84 marginal)·**3 具名-X**(q4_0 0.70/q5_0 0.62/q5_1 0.70)·**升 0.8 硬门**(opp 真向量 autovec·decree) | opp 全 SCALAR·**5 WIN [DEQ-AXIS]**(q4_0/q4_1/q8_0 scalar-vs-scalar 1.07–1.43×·q5_0/q5_1 light-vec-vs-scalar 4.2–4.5× opp-immaturity)·**维持 DEQ-AXIS**(不升门) | kernel-sym MICRO·**[DEQ-AXIS]**·NOT matmul headline |
| **③ quantize 3** | vec-vs-vec·**3 PASS parity**(q8_0/q8_1/q8_K 全 0.998–1.002×) | vec-vs-vec·**3 PASS parity**(全 0.999–1.008×) | kernel-sym MICRO·**parity-by-roofline**(memory-bound·非 beat) |
| **④ product_reduce 3** | SANITY·**3 PASS-sanity**(nibble 2.7×/codebook 2.0×/offbin 0.85× marginal) | SANITY·**2 PASS + 1 具名-X**(nibble 1.17×/codebook 1.10×/offbin 0.80× boundary) | SANITY 层·**vs 构造 scalar-ref**·NOT 任何系统账 |

**★成色总纲（诚实第一·计数≠成色）**：
- **0 verified hand-brick win**·**0 net-new matmul kernel-sym ≥parity**（三子项均非 matmul 轴 real beat）。
- **③ quantize = 干净竞争 parity**（对手 TRUE-VEC·双板贴 DRAM 墙·ratio≈1.00·"对手贴墙 parity=满分"·非 immaturity·**非独立 beat**）。
- **⑤ dequant** = per-board 大异（rvv vec-vs-vec 真硬碰硬·q4_1 唯一干净 WIN·q8_0 near-parity·3 输 gcc autovec 强对手；k1 全赢 scalar-ref = 弱赢类 DEQ-AXIS·大倍数 = opp-immaturity 非硬赢）。
- **④ product_reduce** = 便宜档 sanity（scalar 对手·大倍数禁称硬赢·offbin 双板 leaf-granularity vector-setup 税反追平/输）。
- **全 byte-exact ZERO-MODEL**（dequant/quantize 输出 0mism/0ULP vs stock oracle·product_reduce int 0mism vs scalar-ref）= 数值正确性坐实。

---

## 6. T3 回填清单（★36-col schema·本 agent 不改 T3·主会执行·标量类档·禁继承·per-lane 域）

> 均为 **新行**（T3_A/T3_B 现无 FLAT-5 dequant / quantize / product_reduce 行·census 18 dequant 不含 FLAT-5）。按 T3 36-col schema 追加。关键列填下（余列 = `n_a` 或 casefile 指针）。

### 6.1 T3_A（rvv·VLEN128·gcc-15.2 域）

| measurement_row_key | cold_ratio(s1/s2) | opponent_grade | opponent_symbol | ledger_account | compiler_axis | hardgate_0p8 | correctness |
|---|---|---|---|---|---|---|---|
| dequant\|q4_0\|streaming\|arity1\|f32 | 0.708/0.698 | native-vec(autovec11) | dequantize_row_q4_0 | DEQ-AXIS | gcc15.2-deploy-lane | **in-denom·具名-X**(vec-vs-vec·真向量升门) | byte-exact 0/0 |
| dequant\|q4_1\|streaming\|arity1\|f32 | 1.387/1.181 | native-vec(autovec11) | dequantize_row_q4_1 | DEQ-AXIS | gcc15.2-deploy-lane | **in-denom·PASS(WIN)**(noisy o_iqr 5–6.6%) | byte-exact 0/0 |
| dequant\|q5_0\|streaming\|arity1\|f32 | 0.618/0.618 | native-vec(autovec34) | dequantize_row_q5_0 | DEQ-AXIS | gcc15.2-deploy-lane | **in-denom·具名-X** | byte-exact 0/0 |
| dequant\|q5_1\|streaming\|arity1\|f32 | 0.745/0.702 | native-vec(autovec32) | dequantize_row_q5_1 | DEQ-AXIS | gcc15.2-deploy-lane | **in-denom·具名-X**(med&best 双 seed<0.8) | byte-exact 0/0 |
| dequant\|q8_0\|streaming\|arity1\|f32 | 0.839/0.837 | native-vec(autovec13) | dequantize_row_q8_0 | DEQ-AXIS | gcc15.2-deploy-lane | **in-denom·PASS(marginal)** | byte-exact 0/0 |
| quantize\|q8_0\|streaming\|arity1\|f32 | 1.002/0.998 | native-vec(inline6) | quantize_row_q8_0 | forward-op/quant-stream | gcc15.2-deploy-lane | **in-denom·PASS(parity-roofline)** | byte-exact-out 0mism |
| quantize\|q8_1\|streaming\|arity1\|f32 | 0.999/1.000 | native-vec(inline5) | quantize_row_q8_1 | forward-op/quant-stream | gcc15.2-deploy-lane | **in-denom·PASS(parity-roofline)** | byte-exact-out 0mism |
| quantize\|q8_K\|streaming\|arity1\|f32 | 1.001/1.000 | native-vec(inline9) | quantize_row_q8_K | forward-op/quant-stream | gcc15.2-deploy-lane | **in-denom·PASS(parity-roofline)** | byte-exact-out 0mism |
| product_reduce\|q4_0_nibble\|leaf\|arity2\|i8 | 2.722/2.701 | **scalar-ref(constructed)** | (无框架对手·sc_nibble oracle) | SANITY(no-framework-opp) | gcc15.2 | **test-only-not-in-denom**(便宜档·标量) | byte-exact-int 0mism |
| product_reduce\|offset_binary_n3\|leaf\|arity2\|i8 | 0.858/0.851 | scalar-ref(constructed) | (无框架对手·sc_offbin) | SANITY(no-framework-opp) | gcc15.2 | **test-only-not-in-denom**(便宜·marginal) | byte-exact-int 0mism |
| product_reduce\|codebook_n3\|leaf\|arity2\|i8 | 1.995/1.961 | scalar-ref(constructed) | (无框架对手·sc_codebook·vrgather) | SANITY(no-framework-opp) | gcc15.2 | **test-only-not-in-denom**(便宜档) | byte-exact-int 0mism |

### 6.2 T3_B（k1·VLEN256·clang-18 域）

| measurement_row_key | cold_ratio(s1/s2) | opponent_grade | opponent_symbol | ledger_account | compiler_axis | hardgate_0p8 | correctness |
|---|---|---|---|---|---|---|---|
| dequant\|q4_0\|streaming\|arity1\|f32 | 1.431/1.443 | scalar-ref(deployed) | dequantize_row_q4_0 | DEQ-AXIS | clang18-deploy | **维持 DEQ-AXIS**(opp scalar·不升门·弱赢) | byte-exact 0/0 |
| dequant\|q4_1\|streaming\|arity1\|f32 | 1.391/1.401 | scalar-ref(deployed) | dequantize_row_q4_1 | DEQ-AXIS | clang18-deploy | **维持 DEQ-AXIS**(弱赢) | byte-exact 0/0 |
| dequant\|q5_0\|streaming\|arity1\|f32 | 4.168/4.272 | scalar-ref(deployed) | dequantize_row_q5_0 | DEQ-AXIS | clang18-deploy | **维持 DEQ-AXIS**(opp-immaturity·大倍数非硬赢) | byte-exact 0/0 |
| dequant\|q5_1\|streaming\|arity1\|f32 | 4.494/4.347 | scalar-ref(deployed) | dequantize_row_q5_1 | DEQ-AXIS | clang18-deploy | **维持 DEQ-AXIS**(opp-immaturity) | byte-exact 0/0 |
| dequant\|q8_0\|streaming\|arity1\|f32 | 1.065/1.068 | scalar-ref(deployed) | dequantize_row_q8_0 | DEQ-AXIS | clang18-deploy | **维持 DEQ-AXIS**(marginal 弱赢) | byte-exact 0/0 |
| quantize\|q8_0\|streaming\|arity1\|f32 | 0.999/1.001 | native-vec(inline4) | quantize_row_q8_0 | forward-op/quant-stream | clang18-deploy=MAIN | **in-denom·PASS(parity-roofline)** | byte-exact-out 0mism |
| quantize\|q8_1\|streaming\|arity1\|f32 | 1.000/1.001 | native-vec(inline6) | quantize_row_q8_1 | forward-op/quant-stream | clang18-deploy=MAIN | **in-denom·PASS(parity-roofline)** | byte-exact-out 0mism |
| quantize\|q8_K\|streaming\|arity1\|f32 | 1.008/1.000 | native-vec(inline7) | quantize_row_q8_K | forward-op/quant-stream | clang18-deploy=MAIN | **in-denom·PASS(parity-roofline)** | byte-exact-out 0mism |
| product_reduce\|q4_0_nibble\|leaf\|arity2\|i8 | 1.164/1.168 | scalar-ref(constructed) | (无框架对手·sc_nibble) | SANITY(no-framework-opp) | clang18 | **test-only-not-in-denom**(便宜档) | byte-exact-int 0mism |
| product_reduce\|offset_binary_n3\|leaf\|arity2\|i8 | 0.796/0.797 | scalar-ref(constructed) | (无框架对手·sc_offbin) | SANITY(no-framework-opp) | clang18 | **test-only-not-in-denom**(便宜·<0.8 具名-X) | byte-exact-int 0mism |
| product_reduce\|codebook_n3\|leaf\|arity2\|i8 | 1.096/1.096 | scalar-ref(constructed) | (无框架对手·sc_codebook) | SANITY(no-framework-opp) | clang18 | **test-only-not-in-denom**(便宜档) | byte-exact-int 0mism |

**★须主会/用户裁的 policy（canon 级·本 agent 不自决）**：
1. **quantize 的 ledger_account 归属**：quantize_row 是 f32→quant streaming（quant-stream 族·类比 dequant DEQ-AXIS 的镜像）。建议独立 **quant-stream 子账**或并入 DEQ-AXIS 类（parity-by-roofline·非 matmul）。**呈现由主会定**（我暂记 `forward-op/quant-stream`·非 matmul headline）。
2. **quantize 是否入 0.8 硬门分母**：对手 TRUE-VEC（真向量·非 scalar）→ 按 DEQ auto-promote decree 逻辑**应升 0.8 硬门**（6/6 PASS-parity）。但成色 = roofline-parity（非 beat）→ 计 in-denom-PASS 但**须标 parity-by-roofline·非独立 beat**（避免当 win 外推）。**建议主会裁**：quantize 6 in-denom(PASS-parity) 增 0.8 分母（rvv +3 / k1 +3）·分子 +3 各（parity 视 PASS）。
3. **⑤ dequant FLAT-5 rvv 升门后分母增量**：rvv +5 dequant 入 0.8 硬门（真向量对手·+2 PASS[q4_1/q8_0] +3 具名-X）→ rvv 0.8 分母 43→**48**（deq-axis-reparse §④ 定档 43 + FLAT-5 5）。k1 FLAT-5 维持 DEQ-AXIS(0 升门·同 census)。
4. **product_reduce**：**test-only-not-in-denom**（无框架对手·SANITY·不入 0.8 硬门·[NG-4] 便宜档）——无争议·直接登记。

---

## 7. 污染纪律 + restore（双板·per-item）

- **cold 协议**：dequant/quantize K=1048576（out/in 4MiB > L2·224MiB flush·N=24 median+relIQR·2-seed·within-proc paired）· product_reduce NB=262144×16-lane（in 12.6MiB > L2·同 flush/N/seed）。
- **stock 库只读·双证**：rvv base.so `1b4580c4` / cpu.so `d1adc634` · k1 base.so `00267134` / cpu.so `871169a0`——**全 before==after UNCHANGED**（仅编 ours .o + link driver·未触 .so）。
- **stray**：`pgrep -x {dequant_flat5,quantize3,preduce3}`=0 双板全 item（run 内 dequant STRAY=3 = `-f` 误匹 ssh/tee cmdline 假阳·`-x` 实测 0）。
- **scratch**：`/tmp/g8_a2b3_{dq,qz,pr}_{rvv,k1}` 全 `rm -rf` GONE 双板。
- **load-gate**：rvv core8-15 idle-pick(全 100%)·gov=performance 2.6GHz；k1 core0-7 idle-pick(全 100%)·gov=performance 1.6GHz。loadavg 全程 2.1–2.4（co-tenant 未扰·单核 bench 自负荷）。
- **git**：无 add/commit·未改 T3/T8/T9/emitter/lib·casefile 独立。**per-lane deploy-matched 域**（rvv gcc-15.2 / k1 clang-18·§1·Amdahl 同域律·非 clang-micro-on-rvv artifact）。

## durable files
- `A2-batch3-deq-quant-preduce.md`（本文）
- `A2-batch3-deq-quant-preduce-raw/`：
  - `GEN_SEAL.txt`（11 exported kernel md5 provenance + weft-opt HEAD + recipe）
  - `dequant_flat5_driver.c` / `run_dequant_flat5.sh` + `{rvv,k1}_dequant_flat5.log` + `_seal.txt`
  - `quantize3_driver.c` / `run_quantize3.sh` + `{rvv,k1}_quantize3.log` + `_seal.txt`
  - `preduce3_driver.c`（内含 scalar-ref oracle）/ `run_preduce3.sh` + `{rvv,k1}_preduce3.log` + `_seal.txt`
  - `kernels_dequant/`（5 FLAT `.dq.c`）· `kernels_quantize/`（3 `.qz.c`）· `kernels_preduce/`（3 `.pr.c`）
  - `{rvv,k1}_{dequant,quantize,preduce}_console.log`（build seal + opp probe + 逐格 A/B raw）
