# T-VALIDITY — 编译器对称性 validity 底账 ([CASE-COMPILER-ASYMMETRY] Stage-1 sweep)

> 生成 2026-07-10 · 分支 `refactor/full-refactor-m1` · 纯文档/元数据段（零 build / 零板 / 零 perf 测量）。
> **口径冻结**：这是**分类诊断表**，**不是** canon 重写。不改 T8/8-gate doc 的既有 verdict、不改 memory；
> 不对称格只标 **「撤回候选 pending Stage-0 定案」**，绝不标「已撤回」。canon 修订是 Stage-2 的事。
>
> **案由**：q4_K micro 1.884× 被查出 = clang-17(ours) vs gcc-15(shipped-block-dot) 的编译器不对称测量
> （同源 md5 90d454da，clang 71 vsetvli/3 spill vs gcc 820/742），e2e 编译器对称（两侧 gcc-15）= 0.334× LOSS。
> 根因 doc = `docs/reports/2026-07-10-q4k-micro-vs-e2e-regression-mechanism.md` + `T-PERF1b_q4k_e2e_prefill_regression.md`。
> 卷宗 = `docs/reports/2026-07-10-CASE-COMPILER-ASYMMETRY-casefile.md`。
>
> **协议本要求（实验宪法 §1）**：perf 指纹**必须记录双方 compiler + flags**；指纹变则同指纹格自动 STALE。
> 本表逐格提取 {ours-compiler, opp-compiler, board, march/flags, opponent 身份} 并四分类。

## 四分类图例
- **对称-gcc** = ours 与 opp 都 gcc-15 → **幸存**。
- **对称-clang** = ours 与 opp 都 clang（含 internal-A/B 都是我方 clang 变体、vs-clang-proxy）→ **幸存于 clang-域**（+ allocator/emitter-maturity 注记；非 ggml-as-shipped beat）。
- **不对称** = ours-compiler ≠ opp-compiler（典型 = clang-ours vs gcc-shipped-block-dot）→ **撤回候选 pending Stage-0 定案**；论文承重者排入 Stage-1 重测队列。
- **指纹缺失** = compiler 身份未记录 → **纪律缺口**，降级 partial。

## 底账表（每 perf 格一行；split-cell 拆 internal-A/B 与 vs-opponent 两腿）

| # | cell / 腿 | 数字 | ours-compiler | opp-compiler | 对手身份 | board/march | 分类 | 承重? | 处置 |
|---|---|---|---|---|---|---|---|---|---|
| **A. q8_0 flat block-dot（fair 已重测）** |||||||||
| 1 | `q8_0-deferred-ordered-fold-P2c` (T8 r2) | 0.812× (deferred/factory=1.232) LOSS | clang-20 -O2 zfh | clang-20 -O2 zfh | ggml factory block-dot | rvv VLEN128 | 对称-clang | N | 幸存（clang-域；已 fair） |
| 2 | `q8_0-capability-keyed-fill` (T8 r3) | k1/VLEN256 +2.5% win / VLEN128 loss | clang-20 zfh | clang-20 zfh | ggml factory block-dot (m2) | k1 VLEN256 / rvv VLEN128 | 对称-clang | N | 幸存（clang-域；board-conditional；**allocator/emitter-sched 注记**：+9.9%纯fill已 DISSOLVE→+4.37%C-结构） |
| 3 | `q8_0-item4-fcvt-reschedule` (T8 r4) | k1/VLEN256 +4.37% | clang-18 -O2 | clang-18 -O2 | ggml factory block-dot | k1 VLEN256 | 对称-clang | N | 幸存（clang-域；C-structure-lever；objdump 双板 fcvt 真移） |
| **B. q5_K deferred（fair 已重测）** |||||||||
| 4 | `q5_K-2c-deferred-reduce-vs-aux8` (T8 r5) | k1 +2.7~4.8% / rvv −15~17.5% | clang-17(rvv)/clang-18(k1) -O2 | 同（PREFLIGHT(3) same-compiler OK 实证） | ggml q5_K block-dot (softest, plain intrinsics) | k1 VLEN256 / rvv VLEN128 | 对称-clang | N | 幸存（clang-域；board-split；指纹已实证同编译器 `ondevice-q5_K/*_ab_raw.txt`） |
| **C. q4_0 e2e（routing）** |||||||||
| 5 | `q4_0-repack-gemm-rvv-vlen128-prefill-5.92x` (T8 r16) | e2e prefill 5.9195× + decode 1.91× | gcc-15.2.0 fullmarch | gcc-15.2.0 fullmarch | stock upstream ggml block-dot (git f3e1828) | rvv VLEN128 | **对称-gcc** | Y（头条） | **幸存**（两树同 git f3e1828 同 gcc-15；★byte-identical kernel 共模；已 canon 修正=routing 白嫖非 kernel 质量） |
| 6 | `q4_0-gevm-k1-vlen256-decode-P1-roofline-GAP` (T8 r6) | k1/VLEN256 decode 0.854× LOSS | clang-18(k1) 两树 | clang-18(k1) 两树 | e2e-factory (stock tree) | k1 VLEN256 | 对称-clang | N | 幸存（bandwidth-bound roofline GAP，非 codegen；对 compiler 稳健） |
| 7 | `q4_0-gemm-k1-vlen256-prefill-parity` (T8 r7) | k1/VLEN256 prefill 1.0022× parity | clang-18(k1) 两树 | clang-18(k1) 两树 | e2e-factory (stock tree) | k1 VLEN256 | 对称-clang | N | 幸存（compute-bound parity，符合 VLEN-flip 预期） |
| **D. format-micro batch2c（fair 已重测，8 格；SUPERSEDES batch2b-INVALID）** |||||||||
| 8 | `format-micro-iq3_s-…-batch2c` (T8 r8) | 0.2804× LOSS | gcc-15.2.0 -O3 | gcc-15.2.0 -O3 | ggml factory vl128 SIMD | rvv VLEN128 | **对称-gcc** | N | **幸存**（byte-identical march 双侧 gcc-15；missing_pattern 真差） |
| 9 | `format-micro-iq2_s-…-batch2c` (T8 r9) | 0.4772× LOSS | gcc-15.2.0 -O3 | gcc-15.2.0 -O3 | ggml factory vl128 | rvv VLEN128 | **对称-gcc** | N | **幸存** |
| 10 | `format-micro-iq2_xs-…-batch2c` (T8 r10) | 0.3293× LOSS | gcc-15.2.0 -O3 | gcc-15.2.0 -O3 | ggml factory vl128 | rvv VLEN128 | **对称-gcc** | N | **幸存** |
| 11 | `format-micro-iq2_xxs-…-batch2c` (T8 r11) | 0.7796× LOSS | gcc-15.2.0 -O3 | gcc-15.2.0 -O3 | ggml factory vl128 | rvv VLEN128 | **对称-gcc** | N | **幸存** |
| 12 | `format-micro-iq3_xxs-…-batch2c` (T8 r12) | 0.1583× LOSS (worst) | gcc-15.2.0 -O3 | gcc-15.2.0 -O3 | ggml factory vl128 | rvv VLEN128 | **对称-gcc** | N | **幸存** |
| 13 | `format-micro-iq4_xs-…-batch2c` (T8 r13) | 0.7178× LOSS (batch2b 1.4556×>1 EVAPORATED) | gcc-15.2.0 -O3 | gcc-15.2.0 -O3 | ggml factory vl128 | rvv VLEN128 | **对称-gcc** | N | **幸存**（★batch2b>1 曾是 compiler-asymmetry，fair 后蒸发 = 本案先例） |
| 14 | `format-micro-tq2_0-…-batch2c` (T8 r14) | 0.2311× LOSS | gcc-15.2.0 -O3 | gcc-15.2.0 -O3 | ggml factory vl128 | rvv VLEN128 | **对称-gcc** | N | **幸存** |
| 15 | `format-micro-tq1_0-…-batch2c` (T8 r15) | 0.5505× LOSS | gcc-15.2.0 -O3 | gcc-15.2.0 -O3 | ggml factory vl128 | rvv VLEN128 | **对称-gcc** | N | **幸存** |
| **E. GAP-SB pair-batching（内部 A/B + vs-scalar-generic clean；vs-SIMD-gcc15 已 REFUSE）** |||||||||
| 16a | `iq2_xxs-gapsb-flip` internal A/B (T8 r17) | PRE/POST 3.52× | clang-17 | clang-17 (自比 PRE/POST) | 我方 PRE-变体 | rvv VLEN128 | 对称-clang | N | 幸存（clang-域；[NG-4] 内部度量） |
| 16b | `iq2_xxs-gapsb-flip` vs-scalar-generic (T8 r17) | 2.570× WIN | clang-17 | **clang-17** (PREFLIGHT(0) OK 实证) | ggml GENERIC scalar ref | rvv VLEN128 | 对称-clang | N | 幸存（clang-域；★vs-SIMD-dispatch gcc-15 腿已正确 REFUSE=纪律生效正例） |
| 17a | `iq2iq3-gapsb-family` internal A/B (T8 r18) | 1.09~1.29× | clang-17 | clang-17 (自比) | 我方 PRE-变体 | rvv VLEN128 | 对称-clang | N | 幸存（clang-域） |
| 17b | `iq2iq3-gapsb-family` vs-generic (T8 r18) | iq2 win / iq3 loss | clang-17 | clang-17 (PREFLIGHT(0)) | ggml GENERIC scalar ref | rvv VLEN128 | 对称-clang | N | 幸存（clang-域；无 family flip） |
| **F. GAP-P1 VLEN256 fix-confirm（内部 LMUL A/B，无外部对手）** |||||||||
| 18 | `q4_0-gemm-k1-vlen256-GAP-P1-fix-REGRESSION` (T8 r19) | m1/mf2 = 0.367× (2.72× 回归) | clang-18 (llvm-objdump-18) | clang-18 (自比 m1 vs mf2) | 我方 LMUL 变体 | k1 VLEN256 | 对称-clang | N | 幸存（clang-域；internal-AB；widen-to-m1 FALSIFIED） |
| 19 | `q8_0-gevm-k1-vlen256-mf2-vs-m1-VINDICATED` (T8 r20) | wide/m1 = 1.111× | clang-18 | clang-18 (自比 3-way) | 我方 strip/LMUL 变体 | k1 VLEN256 | 对称-clang | N | 幸存（clang-域；per-format 键确认） |
| **G. GAP-RP tq2_0 spill-fix（全 gcc-15）** |||||||||
| 20a | `tq2_0-gaprp-spill-fix` internal A/B (T8 r21) | ~1.92× (spill 4→0) | gcc-15.2.0 -O3 | gcc-15.2.0 -O3 (PRE/POST 自比) | 我方 PRE-变体 | rvv VLEN128 | **对称-gcc** | N | **幸存**（byte-exact spill 消除） |
| 20b | `tq2_0-gaprp` vs-generic (T8 r21) | POST 3.88× WIN | gcc-15.2.0 -O3 | gcc-15.2.0 -O3 | ggml GENERIC scalar | rvv VLEN128 | **对称-gcc** | N | **幸存** |
| 20c | `tq2_0-gaprp` vs-SIMD-factory (T8 r21) | still ~0.45× LOSS | gcc-15.2.0 -O3 | gcc-15.2.0 -O3 (batch2c) | ggml factory vl128 | rvv VLEN128 | **对称-gcc** | N | **幸存**（对手侧= batch2c 对称基线） |
| **H. L3 memory-axis 融合（内部 A/B，无 ggml 对手）** |||||||||
| 21 | `g2-fuse-rms-norm-mul-memory-axis` (T8 r23) | 1.308× | clang-17.0.6 -O3 | clang-17.0.6 -O3 (fused/unfused 自比) | 我方 unfused two-pass | rvv VLEN128 | 对称-clang | N | 幸存（clang-域；[NG-4] 内部 memory-axis；byte-attributed 0.1%） |
| 22 | `fmtprop-rms-norm-mul-quantize` (T8 r24) | 1.139× | clang-17.0.6 -O3 | clang-17.0.6 -O3 (自比) | 我方 unfused two-pass | rvv VLEN128 | 对称-clang | N | 幸存（clang-域；[NG-4] 内部；256MiB 消除 99.9%） |
| 23 | `g2-fuse-WHOLE-MODEL-e2e-NOT-SIGNIFICANT` (T8 r65) | ~0 (prefill −0.11%/decode −0.27%) | gcc-15.2.0 fullmarch | gcc-15.2.0 (**同一 binary** getenv toggle) | ggml 自己 fusion ON/OFF | rvv VLEN128 | **对称-gcc** | N | **幸存**（同二进制=最强对称；Amdahl-diluted null） |
| **I. K-quant L1-candidate vs-opponent throughput（★不对称：clang-ours vs gcc-shipped-block-dot）** |||||||||
| 24 | `q4_K-repack-gemm-…-prefill-L1-candidate` (T8 r60) | vs-opp 0.94-0.97× (parity) | clang-17 clean libcall-free | **gcc-shipped** (board libggml-cpu.so `_vl128` block-dot) | ggml factory block-dot | rvv VLEN128 | **不对称** | Y | **撤回候选** pending Stage-0（opponent-absence 结构事实另存，见注②） |
| 25 | `q5_K-repack-gemm-…-prefill-L1-candidate` (T8 r61) | vs-opp 1.50-1.62× ours-faster | clang-17 | **gcc-shipped** (ggml `_generic` UNTUNED block-dot) | ggml factory generic | rvv VLEN128 | **不对称** | Y | **撤回候选** pending Stage-0（vs UNTUNED，最软对手） |
| 26 | `q6_K-repack-gemm-…-prefill-L1-candidate` (T8 r62) | vs-opp 0.18-0.19× LOSS | clang-17 | **gcc-shipped** block-dot | ggml factory block-dot | rvv VLEN128 | 不对称（**moot=LOSS**） | N | 撤回候选（LOSS 非 win；retraction moot；结构 opening + SILICON numeric 幸存） |
| 27 | `q2_K-repack-gemm-…-prefill-L1-candidate` (T8 r63) | vs-opp 0.20-0.21× LOSS | clang-17 | **gcc-shipped** `_vl128` block-dot | ggml factory block-dot | rvv VLEN128 | 不对称（**moot=LOSS**） | N | 撤回候选（LOSS 非 win；moot） |
| 28 | `q3_K-repack-gemm-…-prefill-L1-candidate` (T8 r64) | vs-opp 0.176× LOSS | clang-17 | **gcc-shipped** `_vl128` block-dot | ggml factory block-dot | rvv VLEN128 | 不对称（**moot=LOSS**） | N | 撤回候选（LOSS 非 win；moot） |
| **J. K-quant tiling S1/S6/T3（split：internal-A/B 对称-clang 幸存 · vs-opponent 不对称撤回候选）** |||||||||
| 29a | `q4_K-tile-S1-hstrip` internal A/B (T8 r66) | post/pre +52.4% | clang-17.0.6 -O2/-O3 | clang-17 (PRE/POST 自比) | 我方 S1-pre | rvv VLEN128 | 对称-clang | N | 幸存（clang-域；spill 84→23 byte-exact） |
| 29b | `q4_K-tile-S1` vs-opponent (T8 r66) | parity 0.962→1.473× | clang-17 | **gcc-shipped** `_vl128` block-dot | ggml factory block-dot | rvv VLEN128 | **不对称** | Y | **撤回候选** pending Stage-0 |
| 30a | `q4_K-tile-S6-minfold` internal A/B (T8 r67) | post-S6/pre-S1 +27.9% | clang-17.0.6 -O2/-O3 | clang-17 (S6/S1 自比) | 我方 S1 | rvv VLEN128 | 对称-clang | N | 幸存（clang-域；spill 23→0 cliff byte-exact） |
| 30b | `q4_K-tile-S6` vs-opponent **=micro 1.884×** (T8 r67) | 1.473→**1.884×** | clang-17.0.6 -O2 | **gcc-15 shipped** `_vl128` block-dot | ggml factory block-dot | rvv VLEN128 | **不对称** | **Y（案由核心）** | **撤回候选** pending Stage-0（★= regression doc 已证 = clang-vs-gcc artifact；对称-gcc e2e=0.334×） |
| 31a | `q6_K-tile-T3-NULL` internal A/B (T8 r68) | −1.5%/−2.3% (NULL) | clang-17 -O2/-O3 | clang-17 (tiled/untiled 自比) | 我方 untiled | rvv VLEN128 | 对称-clang | N | 幸存（clang-域；family-lever-boundary NULL） |
| 31b | `q6_K-tile-T3` vs-opponent (T8 r68) | 0.184→0.181× LOSS | clang-17 | **gcc-shipped** block-dot | ggml factory block-dot | rvv VLEN128 | 不对称（**moot=NULL/LOSS**） | N | 撤回候选（NULL/LOSS；moot） |
| 32a | `q2_K-tile-S6` internal A/B (T8 r69) | tiled/untiled 6.653× | clang-17 -O2/-O3 | clang-17 (自比) | 我方 untiled | rvv VLEN128 | 对称-clang | N | 幸存（clang-域；spill 619→7 cliff byte-exact） |
| 32b | `q2_K-tile-S6` vs-opponent (T8 r69) | 0.212→**1.413×** | clang-17 | **gcc-shipped** `_vl128` block-dot | ggml factory block-dot | rvv VLEN128 | **不对称** | **Y** | **撤回候选** pending Stage-0 |
| 33a | `q3_K-tile-T3-NULL` internal A/B (T8 r70) | +8.0% (非 cliff lever) | clang-17 -O2/-O3 | clang-17 (自比) | 我方 untiled | rvv VLEN128 | 对称-clang | N | 幸存（clang-域；[XFER-1] weight-bound NULL 预测命中） |
| 33b | `q3_K-tile-T3` vs-opponent (T8 r70) | 0.176→0.191× LOSS | clang-17 | **gcc-shipped** `_vl128` block-dot | ggml factory block-dot | rvv VLEN128 | 不对称（**moot=NULL/LOSS**） | N | 撤回候选（+8% 不翻，still LOSS；moot） |
| 34a | `q5_K-tile-T3-S6` internal A/B (T8 r71) | tiled/untiled +41.7% | clang-17 -O2/-O3 | clang-17 (自比) | 我方 untiled | rvv VLEN128 | 对称-clang | N | 幸存（clang-域；spill 155→105 cliff；[XFER-1] min-fold survives qh） |
| 34b | `q5_K-tile-T3-S6` vs-opponent (T8 r71) | 1.547→**2.193×** | clang-17 | **gcc-shipped** `_generic` block-dot | ggml factory generic | rvv VLEN128 | **不对称** | **Y** | **撤回候选** pending Stage-0（vs UNTUNED generic） |
| **K. K-quant k1 VLEN256 board-B（★行内自陈 "same asymmetry as board A"）** |||||||||
| 35 | `q4_K-repack-gemm-S6-k1-VLEN256-BOARD-B` (T8 r74) | k1/VLEN256 median **3.106×** | clang-18.1.8-Bianbu -O2 zfh | **factory-as-shipped** (gcc) block-dot | ggml factory block-dot (VLEN256 branch) | k1 VLEN256 | **不对称** | **Y** | **撤回候选** pending Stage-0（★行内明记 "ours=clang-18 vs factory-as-shipped (same asymmetry as board A)"；关 8-gate ③⑤ 的对象即此不对称） |
| 36 | `q5_K-repack-gemm-k1-VLEN256-BOARD-B` (T8 r75) | k1/VLEN256 median **1.916×** | clang-18.1.8-Bianbu -O2 zfh | **factory-as-shipped** (gcc) block-dot | ggml factory block-dot | k1 VLEN256 | **不对称** | **Y** | **撤回候选** pending Stage-0（同 r74 自陈不对称） |
| **L. dequantize gate4（历史 finale；cell 不在 active 树，指纹从 memory 复原）** |||||||||
| 37 | dequantize vs true-scalar **10.8×** (memory `[[perf-finale-metric3-selector-capability]]`) | 10.8× | 系统 clang-17（gate4 惯例） | **未记录**（scalar ref 编译器身份缺） | true-scalar ref | rvv (gate4) VLEN128 | **指纹缺失** | 待核 | partial-降级（协议要求记双方 compiler+flags 但 scalar 侧缺；cell 未在 active 树，需定位存档） |
| 38 | dequantize vs clang-autovec-naive-RVV **3.3×** (memory) | 3.3× | 系统 clang-17 | **clang**（autovec proxy 显名 clang） | clang-autovec-naive-RVV | rvv (gate4) VLEN128 | 对称-clang | 待核 | 幸存（clang-域 proxy；但需回定位 cell 复验指纹） |
| 39 | dequantize/clamp 0.5× vs autovec (canon 执行总纲v2 K-2b) | 0.5× LOSS | clang | clang autovec | clang-autovec-naive-RVV | rvv VLEN128 | 对称-clang | N | 幸存（clang-域；= 缺 wide-clamp 双区能力，非编译器；moot=LOSS） |
| **M. q4_K e2e（对称真相 + 集成正确性）** |||||||||
| 40 | `q4_K-e2e-prefill-REGRESSION` (T-PERF1b) | A/Bq4kOFF **0.334×** (pp128) / 0.335× (pp512) | gcc-15.2.0 -O3 (deploy) | gcc-15.2.0 -O3 (Bq4kOFF block-dot) | A-build block-dot 回退（同二进制） | rvv VLEN128 | **对称-gcc** | Y（诚实真相） | **幸存**（这是编译器对称下的诚实 LOSS；= 1.884× 的对称对照真相） |
| 41 | `q4_K-e2e-integration-CORRECT-full-construct` (T8 r181) | 无 perf Δ（PPL 12.008≈12.05） | gcc-15.2.0 (deploy) | stock ggml block-dot f3e1828 | 正确性参考（非 perf 对手） | rvv VLEN128 | N/A（correctness-only） | N | 不适用（无 perf 数；e2e=correct-proven/perf-pending board-gated） |
| **N. 无 perf 数（登记以求完备）** |||||||||
| 42 | `iq4_nl-frontdoor-tile-NOOP` (T8 r72) | 无（structural no-op） | clang-17 (read-only) | — | — | rvv VLEN128 | N/A（无 perf） | N | 不适用（byte-exact 构造；codebook-already-lean） |
| 43 | `iq4_xs-frontdoor-tile-NOOP` (T8 r73) | 无（structural no-op） | clang-17 (read-only) | — | — | rvv VLEN128 | N/A（无 perf） | N | 不适用（byte-exact 构造） |
| 44 | `iq4_xs-tq1_0-GAP-NUM-relaxed-not-materialized` (T8 r22) | 无（structural block） | — | — | — | — | N/A（无 perf） | N | 不适用（relaxed body 不存在，无可计时对照） |

## 四分类计数（含 split-cell 两腿；N/A 不计入四类）

| 分类 | 计数 | 内容 |
|---|---:|---|
| **对称-gcc（幸存）** | **14** | batch2c ×8 (8-15) + q4_0-e2e-5.92× (5) + tq2_0-GAP-RP 三腿 (20a/b/c) + g2-wholemodel-e2e (23) + q4_K-e2e-0.334× (40) |
| **对称-clang（幸存 clang-域）** | **19** | q8_0 ×3 (1-3) + q5_K-deferred (4) + q4_0-k1 ×2 (6-7) + GAP-SB 四腿 (16a/b,17a/b) + GAP-P1 ×2 (18-19) + g2/fmtprop (21-22) + 六 tiling internal-A/B (29a,30a,31a,32a,33a,34a) + dequantize-autovec-3.3× (38) + clamp-0.5× (39) |
| **不对称（撤回候选 pending Stage-0）** | **13** | K-quant L1-candidate vs-opp ×5 (24-28) + 六 tiling vs-opp (29b,30b,31b,32b,33b,34b) + k1-board-B ×2 (35-36) |
| **指纹缺失（纪律缺口 partial-降级）** | **1** | dequantize vs true-scalar 10.8× (37；scalar 侧编译器身份未记录 + cell 未在 active 树) |

> 注：对称-clang 与对称-gcc 的 split-cell（tiling / GAP-SB / GAP-RP）**同一 cell 两腿分属两类**——internal-A/B 幸存、vs-opponent 撤回候选。数计入各腿。
> N/A（无 perf / correctness-only）= 4 行（41-44）。

## 承重-不对称格清单（★Stage-1 重测队列 = 论文承重、须同工具链双侧重测）

论文 C3′「吞吐兑现」当前记 4（`paper-evidence-index.md:87`）：q4_0 e2e 5.9×（**幸存**=对称-gcc）+ q4_K/q5_K/q2_K S6 kernel-轴。后三者的 **vs-opponent 腿全部不对称**，是承重撤回候选：

1. **`q4_K-tile-S6` vs-opponent 1.884×** (T8 r67 / 30b) — ★案由核心；regression doc 已证 = clang-vs-gcc artifact，对称-gcc e2e = 0.334×。
2. **`q2_K-tile-S6` vs-opponent 1.413×** (T8 r69 / 32b) — clang-ours vs gcc-shipped `_vl128`。
3. **`q5_K-tile-T3-S6` vs-opponent 2.193×** (T8 r71 / 34b) — clang-ours vs gcc-shipped `_generic`（对手 UNTUNED）。
4. **`q4_K-S6-k1-VLEN256-BOARD-B` 3.106×** (T8 r74 / 35) — 行内自陈 "same asymmetry as board A"；关 q4_K 8-gate ③⑤ 依据此不对称。
5. **`q5_K-k1-VLEN256-BOARD-B` 1.916×** (T8 r75 / 36) — 同上不对称。
6. **`q5_K-L1-candidate` 1.50-1.62×** (T8 r61 / 25) — vs UNTUNED generic，clang-ours vs gcc-shipped。

> LOSS/NULL 的不对称腿（q6_K/q3_K tiling + q4_K/q6_K/q2_K/q3_K L1-candidate 输/平）= 撤回 moot（本非 claimed win），不进重测队列，但仍标不对称待 Stage-0 统一定案。

## 指纹缺失格清单（纪律缺口）

1. **dequantize vs true-scalar 10.8×** (item 37) — scalar 参考侧编译器 + flags **未记录**；evidence cell 未在 active 树（历史 finale，从 memory 复原）。**处置**：Stage-1 需回定位存档 cell、补记 scalar 侧 {compiler,flags}；若 scalar 亦 clang-17 则升 对称-clang，否则不对称。vs-autovec-3.3× 腿（item 38）指纹可从 memory 判 clang-proxy 但同样需回定位 cell 复验。

## 纪律缺口立案（pre-flight assertion 覆盖漏洞）

**规范工具链政策**：实验宪法 §1「指纹含双方 compiler+flags」+ 工具链政策（执行总纲v2/实验总纲v1 §1 第9/10条：clang-20 + 板全能力 march）+ preflight gate3「same-compiler」。

**assertion 在哪生效、哪失效**：
- ✅ **format-micro 线生效**：batch2b（ours clang-17 -O2 vs factory **gcc-12.3.1 -O3**）被 preflight/宪法 §1 判 **INVALID（protocol-defect: compiler-asymmetry）**（T3_A CSV 顶部 INVALID banner + T8 r8-15 "SUPERSEDES batch2b-INVALID"），重建为 batch2c **gcc-15.2.0 双侧对称** → 8 格 fair 重测；GAP-SB 更进一步**主动建 clang-17 scalar ref** 并**正确 REFUSE** vs-SIMD-dispatch(gcc-15) 腿。**这是 pre-flight 生效的正例。**
- ❌ **K-quant repack tiling harness 失效**：S1/S6/T3 及 L1-candidate 的 **vs-opponent** 腿一律取「board 自己 shipped libggml-cpu.so 的 REAL-dispatched block-dot」（gcc 编译）作对手，ours 侧 clang-17，**preflight same-compiler gate 未对 vs-opponent 腿执行**——只对 internal-A/B（PRE/POST 都 ours clang）验了对称，把 shipped-gcc 对手当默认可接受。1.884×/1.413×/2.193× 因此带 clang-vs-gcc codegen artifact 混入。
- ❌ **k1 VLEN256 board-B / gate4 kernel-axis harness 失效**：r74/r75 行内**自陈** "ours=clang-18 vs factory-as-shipped (same asymmetry as board A)" 却仍用于**关 8-gate ③⑤**；即 assertion 识别到不对称、但**未阻断该数进 gate 台账**。dequantize gate4 finale 更早于本纪律固化，scalar 侧指纹未记。

**共同失效模式**（与 g1-hygiene 九.2/九.4/九.5 同构=「守卫写好了却没接电」）：pre-flight same-compiler 断言**存在且在 format-micro 证明有效**，但**未接到 repack-tiling / kernel-axis / gate4 harness 的 vs-opponent 计时腿**；于是 clang-ours-vs-gcc-shipped 的不对称测量畅通进 T8 + 8-gate 台账。**修复方向（Stage-2，非本表职权）**：把 same-compiler gate 提为**所有 vs-opponent 计时腿的 fail-closed 前置**（对 shipped-gcc 对手要么 REFUSE 如 GAP-SB、要么两侧同工具链重建如 batch2c）。

---

*本文件为纯分类诊断底账（Stage-1 sweep 产物），未 commit；touch-set 仅本文件 + casefile。不改 T8/8-gate/memory 既有 verdict。所有「撤回候选」pending [CASE-COMPILER-ASYMMETRY] Stage-0 终审定案。*
