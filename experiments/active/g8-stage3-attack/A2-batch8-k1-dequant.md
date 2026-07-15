# A2 batch8 — k1 dequant K-quant/iq/fp4/tq 照测补齐（18 格·clang-18 deploy·per-lane·[DEQ-AXIS]·真板测）

> **任务**：线 A · A2-batch8 · k1 dequant 补齐板测员 = 填 **k1 侧 18 格 dequant 标量类 pending**：`dequantize_row/{q2_K,q3_K,q4_K,q5_K,q6_K,iq1_s,iq1_m,iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq4_nl,iq4_xs,mxfp4,nvfp4,tq1_0,tq2_0}` @k1（0.8 cold·我方 weft-emitted dequant kernel vs stock `dequantize_row_*` 对手）。承 A2-batch3 harness（同 driver·仅 FLUSH 32→224MiB 升 batch3 标准 + 2-seed + K=1048576 + N=24）。rvv 侧这些格 DEQ auto-promote 早已做（census）·本役只补 **k1 侧**。
> **赛道**：kernel-axis MICRO（同编译器/flags/march 对称·clang-18 deploy）。**NOT e2e·NOT perf-covered·不入 matmul headline**。[NG-4]。dequant = **[DEQ-AXIS] 独立子账**。
> **口径铁线**：cold 唯一·**k1 clang-18 deploy 域**（per-lane deploy-matched·opp=stock base.so 由 k1 出货编译器 clang-18 固定）·**N≥20 中位（reps=24）+ 2-seed（0x1357/0xACE2）**·**禁一切继承**（重编 ours .o + 重测·不复用 g7-census 数字）·每格对手身份探针（objdump 三判据·tri-confirmed）·预注册判读（cold≥0.8 双 seed=PASS / <0.8=具名-X+墙）·0 样本不造数·**便宜档禁称硬赢（opp 全 SCALAR·big multiple=弱赢/opp-immaturity 非 hand-brick）·接线≠转绿·预判不作结论**·byte-exact ZERO-MODEL 正确门。
> **测于**：2026-07-16 · k1（VLEN256·SpacemiT-X60·clang-18.1.8-Bianbu·core7 idle-pick·gov=performance 1.6GHz）· 主树/build/stock `.so`/governor 未改·0 stray·无 git。

---

## 0. ★进度看板（✅ 18 格全完成·全 byte-exact·2-seed·N=24·卫生 clean）

| 量 | 值 |
|---|---|
| 格数 | **18/18 完成**（全 2-seed·reps=24·224MiB flush·K=1048576） |
| byte-exact | **18/18 全 0mism/0ULP**（ZERO-MODEL vs 独立 stock ggml base.so oracle·54 VERIFY-OK） |
| 预注册判读（≥0.8 双 seed=PASS） | **15 PASS / 3 具名-X**（具名-X = iq3_xxs 0.54 / iq3_s 0.68 / nvfp4 0.74） |
| 对手身份 | **18/18 SCALAR**（rvv=0·deployed-scalar-reference·tri-confirmed objdump） |
| 账 | **全 [DEQ-AXIS]·test-only-not-in-denom**（opp 全 scalar → 不 auto-promote 进 0.8 硬门·同 census / batch3 §2.2 k1-lane） |
| 成色 | **0 verified hand-brick win**·全"ours-vec/scalar vs deployed-scalar-ref"=弱赢类（big multiple=opp-immaturity 非硬赢） |
| 卫生 | stock base.so md5 `00267134` before==after·STRAY_x=0·core7 idle=100%·无 git |

**净**：0 verified hand-brick·0 net-new matmul beat·全 byte-exact·全 DEQ-AXIS 弱赢/near-parity/gather-bound-loss·**照测补齐 = 在 batch3 更严 cold 协议（224MiB flush + 2-seed + N=24 + K=1048576）下复现 g7-census 结论**（逐格同判·见 §5 对照）。

---

## 1. 域政策（★per-lane clang-18 deploy·opp scalar → DEQ-AXIS 维持·记录留痕）

- **为何 clang-18 单板对称**：对手 = **stock base.so as-shipped** `dequantize_row_<fmt>`（deployed `ggml_get_type_traits(type)->to_float` 真派发路·k1 出货编译器 = clang-18-Bianbu 固定编译）。"ours" 用**同板 clang-18** 编 → **per-lane deploy-matched 对称**（对手编译器 = 部署编译器 = clang-18）。= census/deq-axis-reparse DEQ 既定约定 + Amdahl 同域律正门（不喂 clang-micro-on-rvv artifact·k1 本就是 clang 出货板·无不对称问题）。
- **opp 全 SCALAR → 维持 DEQ-AXIS·不升 0.8 硬门**：decree 二 "auto-promote-if-真向量对手" 的触发条件 = 对手真向量（rvv>0）。k1 dequant 全 18 opp = **SCALAR（rvv=0·deployed scalar-ref）→ 不满足 auto-promote → 维持 [DEQ-AXIS] test-only-not-in-denom**（同 batch3 §2.2 k1-lane；对照 rvv 侧 opp = gcc-15.2 autovec 真向量 → 已 auto-promote 进 rvv 0.8 硬门）。
- **成色纪律**：opp = deployed **scalar reference**（未向量化）→ ours-RVV-vec vs opp-scalar 的赢是**比"vs hand-tuned RVV"弱一档**（opp-immaturity）。**big multiple（q5_K 5.15× / tq2_0 4.59× / iq4_nl 3.81×）禁称硬赢**（对手贴地板·非贴墙）。诚实披露。

---

## 2. 对手身份探针（★18/18 SCALAR·tri-confirmed·objdump 机判 whole-symbol + authoritative --disassemble=）

**stock libggml-base.so md5 `00267134a3e86cd4fb83e9a2cbf5185a`**（== 部署，未改）。opp = `ggml_get_type_traits(type)->to_float = dequantize_row_<fmt>`（deployed to_float 真路，非 straw-man）。

| 格 | opp rvv_ins | class | 格 | opp rvv_ins | class |
|---|---:|---|---|---:|---|
| q2_K | 0 | SCALAR | iq2_s | 0 | SCALAR |
| q3_K | 0 | SCALAR | iq3_xxs | 0 | SCALAR |
| q4_K | 0 | SCALAR | iq3_s | 0 | SCALAR |
| q5_K | 0 | SCALAR | iq4_nl | 0 | SCALAR |
| q6_K | 0 | SCALAR | iq4_xs | 0 | SCALAR |
| iq1_s | 0 | SCALAR | mxfp4 | 0 | SCALAR |
| iq1_m | 0 | SCALAR | nvfp4 | 0 | SCALAR |
| iq2_xxs | 0 | SCALAR | tq1_0 | 0 | SCALAR |
| iq2_xs | 0 | SCALAR | tq2_0 | 0 | SCALAR |

- **三判据一致（rvv=0 铁定 SCALAR）**：① batch8 whole-file awk 探针（18/18 rvv=0）② authoritative `objdump --disassemble=dequantize_row_<fmt>`（抽测 8 格 q2_K/q4_K/q5_K/q6_K/iq3_s/iq4_nl/mxfp4/tq2_0 = 全 rvv=0）③ g7-census raw/k1_opponent_classification.txt（18/18 rvv=0 "scalar reference"）。→ **SCALAR class 三源确证**。
- **★探针 ins-count 方法学订正（诚实留痕）**：g7-census classification 文件记的 opp ins（q4_K=460 / q2_K=394 / q5_K=191）为**方法学 artifact 偏大**；batch8 两法（whole-file awk 与 --disassemble=）均给短体（q4_K ins=15~18 / q2_K 9~12 / q5_K ~17，rvv=0），彼此吻合。**判定键 = rvv_ins（=0）三源一致 → SCALAR 铁定**；ins-count 绝对值不作判据（仅 rvv=0/>0 定档）。

**★ours-side 自探针（build_seal·哪些 ours 是向量/标量·决定成色分类）**：
| ours-vec（rvv>0） | rvv_ins/gather | ours-SCALAR（rvv=0） |
|---|---|---|
| q3_K 36 · q4_K 21 · q5_K 24 | 中向量 | **q2_K 0** |
| iq1_s 64 · iq1_m 64 · iq2_xxs 64 · iq2_s 64 | 向量 | **q6_K 0** |
| iq2_xs 160/32 · iq3_xxs 160/64 · iq3_s 160/32 | **heavy-gather-vec** | **nvfp4 0（ldexpf）** |
| iq4_xs 56/16 · iq4_nl 6/2 · mxfp4 6/2 · tq1_0 9 · tq2_0 16 | light~中向量 | |

---

## 3. 逐格 cold（★2-seed·N=24·>1=ours 更快·byte-exact 全 0mism/0ULP·verdict = ≥0.8 双 seed PASS）

| 格 | ours-side | opp | cold_med s1 | cold_med s2 | best s1/s2 | o_iqr% s1/s2 | our/opp GB/s | **verdict[0.8]** | 成色（诚实·[DEQ-AXIS]·opp scalar） |
|---|---|---|---:|---:|---:|---:|---|:--:|---|
| q2_K | SCALAR | scalar | 0.8313 | 0.8287 | 0.830/0.830 | 0.9/1.3 | 0.53/0.63 | **PASS(边界)** | scalar-vs-scalar·ours **略慢**(0.83×)·near-parity 边界·**非 win** |
| q3_K | vec36 | scalar | 1.9624 | 1.9706 | 1.958/1.959 | 0.3/1.2 | 1.03/0.53 | **PASS** | ours-vec vs opp-scalar·弱赢 1.97×·opp-immaturity |
| q4_K | vec21 | scalar | 2.7587 | 2.7584 | 2.793/2.764 | 2.4/1.5 | 2.08/0.75 | **PASS** | ours-vec·弱赢 2.76×·opp-immaturity |
| q5_K | vec24 | scalar | 5.1537 | 5.1845 | 5.226/5.293 | 3.0/6.6 | 2.66/0.52 | **PASS** | ours-vec·**big-mult 5.15×=opp-immaturity·非硬赢** |
| q6_K | SCALAR | scalar | 1.0954 | 1.0851 | 1.095/1.097 | 2.7/4.0 | 0.48/0.44 | **PASS** | scalar-vs-scalar·marginal 弱赢 1.09×(opp 6-bit 重标量) |
| iq1_s | vec64 | scalar | 3.0109 | 3.0075 | 3.050/3.061 | 1.2/1.0 | 1.60/0.53 | **PASS** | ours-vec·弱赢 3.01×·opp-immaturity |
| iq1_m | vec64 | scalar | 3.5131 | 3.5095 | 3.575/3.564 | 1.5/2.3 | 1.73/0.49 | **PASS** | ours-vec·弱赢 3.51×·opp-immaturity |
| iq2_xxs | vec64 | scalar | 1.8387 | 1.8361 | 1.848/1.833 | 1.7/0.3 | 1.05/0.57 | **PASS** | ours-vec·弱赢 1.84×·opp-immaturity |
| iq2_xs | heavy-gather160/32 | scalar | 0.9492 | 0.9529 | 0.964/0.962 | 14.7/2.2 | 0.48/0.50 | **PASS(边界)** | ours-gather-重 vs opp-scalar·near-parity 0.95×·**ours 略慢**·gather 代价 |
| iq2_s | vec64 | scalar | 1.7002 | 1.7021 | 1.721/1.705 | 0.8/1.8 | 0.99/0.58 | **PASS** | ours-vec·弱赢 1.70×·opp-immaturity |
| iq3_xxs | heavy-gather160/64 | scalar | 0.5363 | 0.5349 | 0.539/0.540 | 4.0/1.5 | 0.31/0.58 | **具名-X** | **gather-bound LOSS 0.54×**·ours 重 vrgather 慢于 opp-scalar |
| iq3_s | heavy-gather160/32 | scalar | 0.6764 | 0.6754 | 0.677/0.686 | 5.2/6.4 | 0.40/0.59 | **具名-X** | **gather-bound LOSS 0.68×**·同 iq3_xxs |
| iq4_nl | light-vec-gather6/2 | scalar | 3.7825 | 3.8383 | 3.708/3.676 | 2.6/2.4 | 2.22/0.59 | **PASS** | ours-light-vec·**big-mult 3.81×=opp-immaturity** |
| iq4_xs | vec-gather56/16 | scalar | 3.2638 | 3.3044 | 3.039/3.200 | 1.8/2.3 | 1.95/0.60 | **PASS** | ours-vec·弱赢 3.28×·opp-immaturity |
| mxfp4 | light-vec-gather6/2 | scalar | 3.4734 | 3.4578 | 3.343/3.268 | 1.7/2.0 | 2.17/0.63 | **PASS** | ours-light-vec·**big-mult 3.47×=opp-immaturity** |
| nvfp4 | SCALAR(ldexpf) | scalar | 0.7431 | 0.7431 | 0.734/0.736 | 0.6/1.4 | 0.32/0.43 | **具名-X** | **ours-scalar-ldexpf LOSS 0.74×**·ldexpf 路慢于 opp-scalar |
| tq1_0 | light-vec9 | scalar | 2.2655 | 2.3461 | 2.262/2.268 | 0.4/3.6 | 1.04/0.46 | **PASS** | ours-light-vec·弱赢 2.30×·opp-immaturity |
| tq2_0 | vec16 | scalar | 4.5482 | 4.6228 | 4.670/4.660 | 4.8/3.9 | 3.10/0.68 | **PASS** | ours-vec·**big-mult 4.59×=opp-immaturity** |

**★2-seed 复现性**：seed1/seed2 逐格贴合（q2_K 0.831/0.829·q5_K 5.15/5.18·tq2_0 4.55/4.62·nvfp4 0.743/0.743）。o_iqr 多 <5%（唯 iq2_xs s1_iqr=14.7% 为瞬态·s2=2.2% 且 best 0.964/0.962 贴死 → median 稳、判读不变）。全 verdict 双 seed 同侧。

---

## 4. ★tally + 成色总纲（诚实第一·计数≠成色）

**预注册判读 tally（≥0.8 双 seed=PASS）**：**15 PASS / 3 具名-X**。
- **具名-X（<0.8·墙）**：**iq3_xxs 0.54 / iq3_s 0.68**（= ours heavy-gather vrgather-bound·慢于 opp-scalar·墙 = gather 重解码 > 标量顺序解码）· **nvfp4 0.74**（= ours-scalar-ldexpf·ldexpf 慢于 opp-scalar path·墙 = ours 未向量化且 ldexpf 昂贵）。
- **15 PASS 内再分**：① **13 格 ours 真快于 opp**（q3_K/q4_K/q5_K/q6_K/iq1_s/iq1_m/iq2_xxs/iq2_s/iq4_nl/iq4_xs/mxfp4/tq1_0/tq2_0）—— **全弱赢类**（vs deployed-scalar-ref·big-mult=opp-immaturity）；② **2 格边界 PASS 但 ours 实际略慢**（q2_K 0.83× / iq2_xs 0.95× · ≥0.8 门槛内但 <1.0 = near-parity ours 略逊·**非 win**）。

**★成色总纲（0 硬赢·全弱赢/near-parity/gather-loss）**：
- **0 verified hand-brick win**·**0 net-new matmul kernel-sym ≥parity**（本役全 dequant 轴·非 matmul）。
- **全 18 opp = deployed SCALAR reference（rvv=0·三源确证）** → 所有"ours 更快"= **弱赢类（opp-immaturity）**·big multiple（q5_K 5.15/tq2_0 4.59/iq4_nl 3.81/mxfp4 3.47/iq4_xs 3.28/iq1_m 3.51）**明标 opp-immaturity·禁称硬赢**（对手贴地板·非贴墙·非 hand-tuned RVV）。
- **3 具名-X 是真诚实负结果**：ours 向量/标量 kernel 在 iq3_xxs/iq3_s（heavy-gather）与 nvfp4（scalar-ldexpf）上**输给 deployed opp-scalar**——gather-bound 与 ldexpf-cost 两类 emitter 成熟度墙（C3′ 负结果·[DEQ-AXIS] 记录）。
- **全 byte-exact ZERO-MODEL**（18/18 · 0mism/0ULP vs 独立 stock ggml base.so oracle·54 VERIFY-OK）= 数值正确性坐实（各格 decode-formula 与部署 to_float bit-for-bit 一致）。
- **计数纪律**：**全 [DEQ-AXIS]·test-only-not-in-denom·NOT e2e·NOT perf-covered·NOT matmul headline·不入任何 0.8 硬门系统账**（opp scalar·未 auto-promote·同 census / batch3 §2.2 k1-lane）。

---

## 5. ★照测补齐对照（batch8 更严协议 vs g7-census·逐格同判 = 复现坐实）

> g7-census（32MiB flush·1-seed·reps=12·K=131072）→ batch8（**224MiB flush·2-seed·reps=24·K=1048576**）。判读逐格一致 = 结论在更严 cold 协议下复现。

| 格 | g7 cold(1-seed) | batch8 cold(2-seed med) | 判读一致? | 格 | g7 | batch8 | 一致? |
|---|---:|---:|:--:|---|---:|---:|:--:|
| q2_K | 0.807 | 0.831/0.829 | ✓ 边界 | iq2_s | 1.704 | 1.700/1.702 | ✓ |
| q3_K | 1.940 | 1.962/1.971 | ✓ | iq3_xxs | 0.526 | 0.536/0.535 | ✓ X |
| q4_K | 2.781 | 2.759/2.758 | ✓ | iq3_s | 0.666 | 0.676/0.675 | ✓ X |
| q5_K | 5.141 | 5.154/5.185 | ✓ | iq4_nl | 3.755 | 3.783/3.838 | ✓ |
| q6_K | 1.132 | 1.095/1.085 | ✓ | iq4_xs | 3.277 | 3.264/3.304 | ✓ |
| iq1_s | 2.917 | 3.011/3.008 | ✓ | mxfp4 | 3.168 | 3.473/3.458 | ✓ |
| iq1_m | 3.633 | 3.513/3.510 | ✓ | nvfp4 | 0.740 | 0.743/0.743 | ✓ X |
| iq2_xxs | 1.839 | 1.839/1.836 | ✓ | tq1_0 | 2.233 | 2.266/2.346 | ✓ |
| iq2_xs | 0.926 | 0.949/0.953 | ✓ 边界 | tq2_0 | 4.223 | 4.548/4.623 | ✓ |

**18/18 判读一致**（含 3 具名-X 与 2 边界）。deeper 224MiB flush 微幅抬高部分大倍数格（tq2_0 4.22→4.59·mxfp4 3.17→3.47·= 更深 cold 令 opp-scalar 相对更慢），但**判档不变**。→ 照测补齐坐实 g7-census 结论·且升 batch3 标准（2-seed/N24/224MiB）。

---

## 6. T3 回填清单（★36-col schema·k1·VLEN256·clang-18 域·标量类·per-lane·禁继承·本 agent 不改 T3·主会执行）

> **现状**：T3_B 已有这 18 dequant 行（来自 2026-07-14 g8-stage3-opponent-reparse·单 seed coldm1 + **opponent_grade 误标 native-vec**）。**本 batch8 回填 = 校正 3 处**：① cold_ratio 升 N=24 2-seed 值；② opponent_grade `native-vec`→**`scalar`**（objdump 三源确证 rvv=0·校正 reparse 误标）；③ verdict 明标弱赢/opp-immaturity/gather-loss。**均 in-place 更新既有行**（非新行·row_key 不变）。

**k1-lane（T3_B·VLEN256·clang-18 域）· 关键列填下（余列见 casefile 指针）**：

| measurement_row_key | cold_ratio(s1/s2·2-seed·N24) | opponent_grade | opponent_symbol | ledger_account | compiler_axis | hardgate_0p8 | correctness | 成色注 |
|---|---|---|---|---|---|---|---|---|
| dequant\|q2_K\|streaming\|arity1\|f32 | 0.831/0.829 | **scalar** | dequantize_row_q2_K | DEQ-AXIS | clang18-deploy(per-lane) | **test-only-not-in-denom**(opp scalar·维持DEQ·PASS边界) | byte-exact 0/0 | scalar-vs-scalar·ours略慢·near-parity |
| dequant\|q3_K\|streaming\|arity1\|f32 | 1.962/1.971 | **scalar** | dequantize_row_q3_K | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(弱赢1.97×·opp-immaturity) | byte-exact 0/0 | ours-vec vs scalar-ref |
| dequant\|q4_K\|streaming\|arity1\|f32 | 2.759/2.758 | **scalar** | dequantize_row_q4_K | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(弱赢2.76×·opp-immaturity) | byte-exact 0/0 | ours-vec vs scalar-ref |
| dequant\|q5_K\|streaming\|arity1\|f32 | 5.154/5.185 | **scalar** | dequantize_row_q5_K | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(big-mult 5.15×=opp-immaturity·非硬赢) | byte-exact 0/0 | ours-vec vs scalar-ref |
| dequant\|q6_K\|streaming\|arity1\|f32 | 1.095/1.085 | **scalar** | dequantize_row_q6_K | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(scalar-vs-scalar marginal 1.09×) | byte-exact 0/0 | 双方 scalar·ours 略快 |
| dequant\|iq1_s\|streaming\|arity1\|f32 | 3.011/3.008 | **scalar** | dequantize_row_iq1_s | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(弱赢3.01×·opp-immaturity) | byte-exact 0/0 | ours-vec vs scalar-ref |
| dequant\|iq1_m\|streaming\|arity1\|f32 | 3.513/3.510 | **scalar** | dequantize_row_iq1_m | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(弱赢3.51×·opp-immaturity) | byte-exact 0/0 | ours-vec vs scalar-ref |
| dequant\|iq2_xxs\|streaming\|arity1\|f32 | 1.839/1.836 | **scalar** | dequantize_row_iq2_xxs | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(弱赢1.84×·opp-immaturity) | byte-exact 0/0 | ours-vec vs scalar-ref |
| dequant\|iq2_xs\|streaming\|arity1\|f32 | 0.949/0.953 | **scalar** | dequantize_row_iq2_xs | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(near-parity 0.95×·ours略慢·gather代价) | byte-exact 0/0 | ours-gather-重 vs scalar-ref |
| dequant\|iq2_s\|streaming\|arity1\|f32 | 1.700/1.702 | **scalar** | dequantize_row_iq2_s | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(弱赢1.70×·opp-immaturity) | byte-exact 0/0 | ours-vec vs scalar-ref |
| dequant\|iq3_xxs\|streaming\|arity1\|f32 | 0.536/0.535 | **scalar** | dequantize_row_iq3_xxs | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(**LOSS 0.54× gather-bound·具名-X**) | byte-exact 0/0 | ours-heavy-gather 慢于 scalar-ref |
| dequant\|iq3_s\|streaming\|arity1\|f32 | 0.676/0.675 | **scalar** | dequantize_row_iq3_s | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(**LOSS 0.68× gather-bound·具名-X**) | byte-exact 0/0 | ours-heavy-gather 慢于 scalar-ref |
| dequant\|iq4_nl\|streaming\|arity1\|f32 | 3.783/3.838 | **scalar** | dequantize_row_iq4_nl | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(big-mult 3.81×=opp-immaturity) | byte-exact 0/0 | ours-light-vec vs scalar-ref |
| dequant\|iq4_xs\|streaming\|arity1\|f32 | 3.264/3.304 | **scalar** | dequantize_row_iq4_xs | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(弱赢3.28×·opp-immaturity) | byte-exact 0/0 | ours-vec vs scalar-ref |
| dequant\|mxfp4\|streaming\|arity1\|f32 | 3.473/3.458 | **scalar** | dequantize_row_mxfp4 | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(big-mult 3.47×=opp-immaturity) | byte-exact 0/0 | ours-light-vec vs scalar-ref |
| dequant\|nvfp4\|streaming\|arity1\|f32 | 0.743/0.743 | **scalar** | dequantize_row_nvfp4 | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(**LOSS 0.74× ours-scalar-ldexpf·具名-X**) | byte-exact 0/0 | ours-scalar-ldexpf 慢于 scalar-ref |
| dequant\|tq1_0\|streaming\|arity1\|f32 | 2.266/2.346 | **scalar** | dequantize_row_tq1_0 | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(弱赢2.30×·opp-immaturity) | byte-exact 0/0 | ours-light-vec vs scalar-ref |
| dequant\|tq2_0\|streaming\|arity1\|f32 | 4.548/4.623 | **scalar** | dequantize_row_tq2_0 | DEQ-AXIS | clang18-deploy | **test-only-not-in-denom**(big-mult 4.59×=opp-immaturity) | byte-exact 0/0 | ours-vec vs scalar-ref |

**★须主会/用户裁的 policy（canon 级·本 agent 不自决）**：
1. **opponent_grade 校正**：T3_B 现有 18 dequant 行的 `opponent_grade` 多误标 `native-vec`（reparse 从 rvv-lane 误抄）。**k1-lane 实测三源 = 全 SCALAR（rvv=0）**。建议 in-place 改为 `scalar`。**校正非扩分母**（这 18 行本就 test-only-not-in-denom·不入 0.8 头条）。
2. **cold_ratio 升级**：现有 coldm1（单 seed·32MiB flush·g7）→ batch8 N=24 2-seed 224MiB 值。判档不变（§5 逐格一致）·仅精度升。
3. **k1 dequant 全 18 = DEQ-AXIS test-only-not-in-denom**（opp scalar·未 auto-promote）·**无争议**（同 census / batch3 §2.2 k1-lane）。**不改 0.8 头条分母**。rvv 侧这些格已 auto-promote（opp gcc-15.2 autovec 真向量）另账·本役不触。

---

## 7. 污染纪律 + restore（k1·per-item）

- **cold 协议**：K=1048576（out=4MiB f32 >> k1 512K L2·in=最大 q6_K 860KB）·**224MiB flush**（>> 任何 cache·batch3 std·较 g7 32MiB 更深）·**N=24 median + relIQR**·**2-seed（0x1357/0xACE2）**·within-proc paired（每 rep flush→ours 计时·flush→opp 计时）。
- **byte-exact ZERO-MODEL**：18/18 格 · 0mism/0ULP（各格独立 stock base.so oracle·54 VERIFY-OK 无一 MISMATCH）。数值正确性坐实。
- **stock 库只读·双证**：base.so md5 `00267134a3e86cd4fb83e9a2cbf5185a` **before==after UNCHANGED**（仅编 ours .o + link driver·未触 .so）。
- **stray**：`pgrep -x dequant_census`=**0**（STRAY_x=0）。
- **scratch**：`/tmp/g8_a2b8_dq_k1`（remote）保留原始 log/seal/.o 供复核（可 `rm -rf`·未污主树）。
- **load-gate**：core7 idle-pick=100% at gate·gov=performance 1.6GHz。loadavg 2.47→3.61（run 中 co-tenant 抬升·但**核 7 pin·单核 streaming bench·2-seed 逐格贴合证未被扰**·o_iqr 多 <5%）。
- **git**：无 add/commit·未改 T3/T8/T9/emitter/lib·casefile 独立。**per-lane clang-18 deploy 域**（k1 出货 = clang·无 compiler-asymmetry artifact）。
- **build 修记（诚实）**：初次 harness 用 `ssh k1 "...$OBJS..."` 触 zsh 无自动 word-split → link 把 18 .o 当单 arg（DLINK_FAIL）。**编译/探针/verify 未受影响**（[A]/[B] 全成）·经 `finish_k1_run.sh`（硬列 18 .o·nohup 抗 ssh 断连）relink + 全测完成。**测量数据无污染**（同 driver/kernels/flags·仅 link 命令行修正）。

## durable files（`A2-batch8-k1-dequant-raw/`）
- `A2-batch8-k1-dequant.md`（本文·上级目录）
- `dequant_census_driver.c`（承 g7-census·仅 FLUSH 32→224MiB 升 batch3 std）
- `run_k1_dequant_census.sh`（初版·zsh link bug 记录）· `finish_k1_run.sh`（relink 修正版·实际产数）
- `kernels_dequant/`（18 `.dq.c` weft-emitted·+ `GEN_SEAL.txt` 提供 md5 provenance·HEAD=27fdc898·recipe = weft-opt materialize-dequantize-row-stream-front-door）
- `k1_dequant_census.log`（全 raw·54 VERIFY + 36 CENSUS）· `k1_dequant_census_seal.txt`（build + ours 自探针 + opp 探针）
- `opp_probe_reconcile.txt`（authoritative --disassemble= 8-格 rvv=0 复核）· `parse.py`（2-seed 解析）
