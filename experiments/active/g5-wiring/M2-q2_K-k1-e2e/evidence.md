# G5 · q2_K @ k1 · e2e transduction (our-emitted VLA S6-tiled repack vs STOCK hand-brick)

- **campaign**: G5-wiring / perf-covered 拉绿板线 (q5_K@k1 后第 3 个 K-quant e2e 尝试)
- **board**: k1 / SpacemiT X60 / VLEN256 (vlenb=32) / 8-core / Bianbu clang-18.1.8 / gov=performance 1.6GHz
- **date**: 2026-07-13 · **status**: ACTIVE · board-measured · NOT committed (user commits)
- **role**: replace the k1 STOCK hand-tuned RVV q2_K 16x1 repack with OUR compiler-emitted VLA
  (S6-tiled GEMM + plain GEVM), measure e2e transduction on k1.

## ★★ X-0 前置核查 — 与 q5_K 的决定性差异 (opponent identity 纠偏)
裁二.3 X-0 三问 + 前置 opponent 核查暴露一个**任务前提纠偏**:

**q2_K@k1 的对手 = STOCK 手调 RVV repack, 不是 block-dot.** k1 的 ggml tree (`get_tensor_traits`)
对 q2_K **case256 直接 `return &q2_K_16x1_q8_K`** (repack.cpp:4637), 并在 `arch/riscv/repack.cpp`
携带真 RVV 手调核 `ggml_gemm_q2_K_16x1_q8_K` (568/1496 行, N_ROWS_TILE=4/N_COLS_TILE=16, 紧凑
rolled loop). 这与 q5_K **相反** (q5_K case256 只有 NEON 子分支 → riscv 上 nullptr → block-dot·净新).
∴ q2_K@k1 = **q4_K Win-K1-VLEN 同型对局** (our-emit vs 真出货 hand-brick·绿 1.085×), **非** q5_K
的 net-new-vs-block-dot. 任务预注册的「对手 stock q2_K block-dot」措辞对 k1 **失效**; 正门对局 =
our-emit vs stock hand-brick (更强对手·薄胜或负·非 block-dot 的肥胜).

**X-0 三问 (板 objdump seal, clang-18 k1 VLEN256):**
1. **瓶颈/lean 核查**: OUR GEMM = **lean** (vsetvli=86 · vsetivli=1 · spill=6 · vwmacc=2304 · text 26982B)
   — byte-matches 已证 l1-t3-q2k S6-tiled 形 (vwmacc 2304 / spill≈7 / text≈27KB), **NOT** q6_K 的
   2995-vsetivli giant. **过门 (lean gate PASS)**. OUR GEVM: vsetvli=81 · spill=22 · vwmacc=576 (plain·未 tile).
   STOCK GEMM (对手): vsetvli=54 · vwmacc=16 · spill=4 (紧凑 rolled loop); STOCK GEVM: vsetvli=30 · vwmacc=4 · spill=0.
   ⇒ 真对局 = 我方全展开 27KB register-resident 核 vs SpacemiT 紧凑 rolled loop (胜负未知·须 e2e 判).
2. **键值板测**: k1 = clang-18 对称 (kernel-axis == system-axis·双账本对称). 区别 rvv=gcc-15 域.
3. **Amdahl**: q2_K prefill 热路·Amdahl 高·期望 = clang-domain register-resident 核 ≥parity 紧凑 stock loop.

## correctness (先过·硬门) — GREEN @ bounded-ULP · MIRAGE ruled out
我方 emitted 核期望 **STRAIGHT 16-way interleave** scale layout (`scales[s*16+c]`), STOCK repacker
产 **permuted "Sequential-Parallel"** layout — 不兼容, 故 (照 q5_K) ON 变体部署 **我方 make_block_q2_Kx16
(straight) + 我方核** (自洽对), OFF = 全 stock (自洽对). 两侧各自正确, 唯一 A/B 差 = q2_K compute.

- **Silicon verify** (`kquant_repack_verify_q2K`, 独立 scalar q2_K dequant-matmul 参考, 从原始
  pre-repack block 解码, isum − summs fold): 当前 export (weft_emitc, md5 d9dee831) + 我方 straight-pack:
  **INT_mismatch_total=0 全 8 shapes** (GEVM nr=1 + GEMM nr∈{4,8,16}, 双 seed 0xC0FFEE/20260708),
  NORM worst_ulp=3402 worst_norm=6.618e-07 (dual fp16 d/dmin adversarial-NORM 路·INT 字节精确). 与 l1-t3
  worst_ulp=3402 **完全一致** ⇒ 部署的即已证 S6-tiled 核. **MIRAGE trap 关** (pack_w == make_block_q2_Kx16 straight).
- **e2e perplexity** (argmax-immune, 265-word corpus, -c 128, 2 chunks): ON **17.2702 ± 4.50** vs
  OFF **17.2078 ± 4.50** (Δ 0.36%, well within ±4.50 error bars ⇒ statistically indistinguishable).
  per-chunk ON [1]15.3427 [2]17.2702 vs OFF [1]15.3503 [2]17.2078. ON banner (GEMM+GEVM) confirms our kernel engaged.
- **greedy coherence**: first-pass greedy 生成连贯正确文本 (p3 "2 + 2 = 4.", p1 "…the city of Paris…",
  p2 "…a young girl named Lily…") — a layout/ABI bug 会给 incoherent/NaN, 此为定性正确接线 (clean
  byte-identical greedy 因该 llama-cli conversational-output 污染无法机械提取, ppl-match 是更强的
  argmax-immune 替代).
- **Verdict**: correctness = **GREEN @ bounded-ULP** — kernel bit-exact-INTEGER vs 独立 oracle +
  e2e coherent + ppl-matched (Δ0.36% within error) + MIRAGE ruled out. 同 q5_K standing.

## deployed kernel + net-new wiring (deploy_patch_q2_K_emitted.py) — 2 tracked files, reversible
- **repack.cpp (GEN)**: `make_block_q2_Kx16` body 换为 STRAIGHT 16-way interleave (scales[s*16+c]=in[c].scales[s];
  qs 与 stock 同为 straight; d/dmin verbatim) — 匹配我方核·byte-exact vs oracle. (struct block_q2_Kx16 stride
  1344·repack template·gemv/gemm<block_q2_K,1,16,Q8_K> template·q2_K_16x1_q8_K trait·case256 dispatch
  **均已在 stock 存在 → 不改**.)
- **arch/riscv/repack.cpp**: `#include` 2 emitted `.inc` (weft_emitc_ symbol) + 换 `ggml_gemm_q2_K_16x1_q8_K`
  / `ggml_gemv_q2_K_16x1_q8_K` body 为调用我方 emitted VLA 核 (banner on first engage).
- **repack.h**: 无改动 (stock block_q2_Kx16 == 我方 layout).

## build + seal (g5_q2_K_build_seal.sh) — dedicated /data/build-k1-q2k, clang-18 SYMMETRIC
Both OFF & ON built from the SAME k1 tcrv tree, recompiling ONLY repack.cpp.o + arch/riscv/repack.cpp.o
(exact compile_commands flags) + relink ⇒ ONLY A/B diff = q2_K make_block + q2_K arch kernel bodies.
- **OFF** lib md5 `14b6add6...`  q2_K weft_emit symbols = **0** (stock hand-brick baseline; == q5_K OFF md5, 同一 stock 树)
- **ON**  lib md5 `5b036155...`  q2_K weft_emit symbols = **2** (T exported): `..._gemm_q2_K_q8_K_kernel...`,
  `..._gemv_q2_K_q8_K_kernel...`; banners present (gevm=1, gemm=1); ON != OFF.
- objdump seal (ON): OUR GEMM vsetvli=86 vwmacc=2304 spill=6 (VLA shapes e16,m1×41 · e8,mf2×40 · e32,m2×5);
  OUR GEVM vsetvli=81 vwmacc=576 spill=22. STOCK (OFF, opponent): GEMM vsetvli=54 vwmacc=16 spill=4; GEVM vsetvli=30 vwmacc=4 spill=0.
- source RESTORED byte-exact (GEN=3cac40aa HDR=57851439 ARCH=c3c101fd = baseline).

## provisioning
q2_K model built ON-BOARD (决策卡④): k1-local `llama-quantize --allow-requantize` requantized
`/data/tinyllama-q8_0.gguf` → `/data/tinyllama-1.1b-Q2_K_M.gguf` (411 MiB, sha256 `7ef48ade...`, 3.14 BPW).
Q2_K ftype: q2_K for most 2D tensors (+ q4_K/q6_K attn/output few·common-mode OFF/ON).

## e2e phase-split (prefill pp64 / decode tg32) — paired A/B, physical .so swap, clang-18 symmetric
Config: taskset -c 0-3, threads=4 (shared board — 4-core pin avoids 8-thread oversubscription, per
kernel-axis k1 methodology; paired interleaved ratio robust to steady load). PP=64 TG=32 REPS=6 PASSES=2
⇒ 12 samples/side. gov=performance 1.6GHz locked.

| phase | ON (our-emit) t/s | OFF (stock hand-brick) t/s | ratio ON/OFF | verdict |
|-------|------------------:|---------------------------:|-------------:|---------|
| **PREFILL (pp64)** | 7.3979 (relIQR 0.28%) | 8.4741 (relIQR 0.19%) | **0.873×** | **<parity LOSS (对手更强)** |
| **DECODE (tg32)**  | 4.1314 (relIQR 0.27%) | 4.7225 (relIQR 0.59%) | **0.875×** | **<parity LOSS (对手更强)** |

n=12/side (2 passes × 6 reps). relIQR <0.6% all sides = clean uncontended 4-thread measurement.
Kernel==system ledger (clang-18 symmetric, single tree). loadavg during run 3.8→8.3 (my 4 threads +
board baseline); the LOSS is NOT env-induced: OFF is the FAST side and is clean (relIQR 0.19%/0.59%),
so contention (which would slow OFF, inflating our ratio) did not manufacture the loss — our-emit is
genuinely slower. Both passes agreed (n=12 stable). ([决定性实验教训] caveat = env-crash makes OFF
崩 → false LOSS; here OFF is the clean fast winner, so no env-crash artifact.)

## VERDICT (裁·q5_K 定式 pre-registered dual-exit — LOSS branch)
- **q2_K@k1 = 黄-对手更强 (维持)**: 「k1·q2_K·e2e prefill **0.873×** / decode **0.875×** (both <parity)·
  kernel账==system账(clang-18 对称)·**对手 = STOCK 手调 RVV q2_K 16x1 repack (真出货 hand-brick·case256
  fires·NOT block-dot)**·我方 compiler-emitted 全展开 27KB S6-tiled 核 e2e 输给 SpacemiT 紧凑 rolled loop
  (vwmacc 2304-vs-16·spill 6-vs-4·text 27KB-vs-tiny)」. correctness GREEN@bounded-ULP (silicon
  byte-exact-INTEGER + ppl-matched Δ0.36% + MIRAGE ruled out) ⇒ 干净 perf LOSS, 非 broken kernel.
- **不 flip 绿·perf-covered 维持 7/83.** q2_K registry cell 保持 黄-对手更强, symmetric_account_pointer
  追加 k1 clang 实测数.

### interpretation — q4_K Win-K1-VLEN 的 CONTRAST (决定性负结果)
- q2_K@k1 与 q4_K@k1 (Win-K1-VLEN) **同型对局** (both = our-emit vs 真出货 stock hand-brick RVV repack),
  但**结果相反**: q4_K emitted 核 **胜** hand-brick 1.085× (薄胜·绿); q2_K emitted 核 **负** 0.873×.
  差别 = emitted kernel 形: q2_K 的全展开 27KB GEMM (2304 vwmacc) 在 k1 上比 stock 紧凑 rolled loop
  (16 vwmacc·小 text) 慢 ~13% (both prefill+decode). 全展开的 register-resident hot loop 没能补偿 I-cache
  压力 / stock 紧凑循环在该微架构上更优.
- **CLOSES the registry fix-hypothesis**: q2_K 旧 yellow account 的 "fixable via k1/VLEN256 部署" 已**测试**
  — k1/VLEN256 clang-domain 部署仍负 (对手是 hand-brick 非 block-dot). clang-domain flip 对 q5_K 有效
  (弱 block-dot 对手) 但对 q2_K 无效 (强 hand-brick 对手). 这不是 gcc-death (那是 rvv/VLEN128); 是我方
  emitted 核质量 vs 竞品手调核在 clang-对称域的干净 e2e 负结果.
- **micro↛e2e**: l1-t3-q2k 曾测我方 tiled-vs-block-dot 1.413× (kernel-axis·rvv·对手=弱 block-dot); 那个
  micro 数不迁移到 k1-e2e-vs-hand-brick (对手身份不同·[CASE-COMPILER-ASYMMETRY]/perf 宪法三层·L1 path 赢 ≠
  vs-competitor-repack 赢). 正门对手 = ggml 真跑的 stock repack.

## [PERF-1] eight-gate walk (LOSS branch — 记账用)
① correctness: PASS @ bounded-ULP (silicon byte-exact-INTEGER 全 8 shapes + ppl Δ0.36% within ±4.50 + coherent + MIRAGE ruled out)
② VLEN-flip / dual-board: e2e k1/VLEN256 ONLY (rvv/VLEN128 = 分离 gcc-spill 域·block-dot 对手) → PARTIAL
③ objdump seal: PASS (nm 2 weft_emitc q2_K 符号 T; OUR gemm vwmacc=2304 spill=6 lean; STOCK opponent sealed vwmacc=16 spill=4)
④ micro AND e2e: our-emit e2e LOSS (prefill 0.873× / decode 0.875×) — NOT a win; l1-t3 micro-vs-block-dot 1.413× 不迁移到 vs-hand-brick
⑤ dual-board e2e verified: k1-only → PARTIAL
⑥ discipline: PASS (DVFS lock 1.6GHz, taskset 4-core pin, paired interleaved, tight relIQR <0.6%, LOSS not env-induced verified)
⑦ mechanism attribution: PASS (LOSS = 全展开 27KB emitted 核 vs stock 紧凑 rolled loop·I-cache/code-size·clang-对称域)
⑧ wording: PASS (bounded k1/VLEN256/clang-18/both-phase·our-emit vs 真出货 hand-brick·NOT block-dot·NOT a beat)
⇒ 干净的 vs-hand-brick e2e LOSS (correctness GREEN·perf 对手更强)·非 sealed 结果·非 broken kernel.

## board restore (verified clean)
- source tree restored byte-exact: repack.cpp=3cac40aa, repack.h=57851439, arch/riscv/repack.cpp=c3c101fd (all baseline)
- no `weft_emitted_*q2_K*.inc` / `tcrv_emitted_*q2_K*.inc` leftover in arch/riscv/
- dedicated build live lib = OFF-pristine (14b6add6); stock /data/k1build-stock untouched (871169a0)
- /data/build-k1-q5k (q5_K cell) untouched; main tree UNTOUCHED (no git action)

## ★同批捎带归档 — q6_K@k1 X-0 = 不过门 (裁二.3·板批 defer)
q6_K@k1 X-0 前置核查 = **不过门·不投 k1 板线** (预判 LOSS). 依据: q6_K = weight-reconstruction-bound
(T8 row 68 XFER-1 二类 NULL·q4_K S6 lever 不迁移·spill 913→949 升); emitted vl=8 = 2995 vsetivli +
6065 e8mf2 巨大展开·无调度 = emitter 指令数不成熟·板+编译器无关 (redundant vsetivli 在发射 IR 里·
clang 删不掉); rvv e2e 已实测 0.07× (docs/reports/2026-07-12-perf-covered-q6_K-yellow-kernel-axis.md).
现有具名杠杆均不适用·新杠杆未具名. q6_K 保持 黄-对手更强 (any-board·k1 预判同 LOSS)·〇追认.2 已定 q6_K FINAL.
