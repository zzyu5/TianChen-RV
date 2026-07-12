# G5-M2 · q5_K @ k1 · e2e transduction (net-new our-kernel dispatch)

- **campaign**: G5-wiring / [裁四.2 pre-registered「k1 唯一新绿精准点」]
- **board**: k1 / SpacemiT X60 / VLEN256 (vlenb=32) / 8-core / Bianbu clang-18.1.8 / gov=performance 1.6GHz
- **date**: 2026-07-12 · **status**: ACTIVE · board-measured · NOT committed (user commits)
- **role**: convert q5_K@k1 from block-dot to OUR net-new riscv 1,16 repack (compiler-emitted VLA kernel),
  measure e2e transduction. **Distinct from [WORK-ITEM-K1-KQUANT-E2E]** (that tested k1's *stock* q4_K
  case256 repack; k1 ships NO q5_K repack, so this is our-kernel + net-new dispatch = new perf-covered
  candidate if ≥parity).

## why this is net-new (our kernel), not a stock toggle
The k1 ggml tree ships q4_K/q2_K 1,16 riscv repack (case256 route) but q5_K's `get_tensor_traits`
has ONLY NEON sub-branches (`q5_K_8x8/8x4` under `has_neon && matmul_int8/dotprod`). On the riscv
board those never fire => `nullptr` at every VLEN => stock `ggml_vec_dot_q5_K_q8_K` block-dot. The
`<block_q5_K,1,16,GGML_TYPE_Q8_K>` riscv trait our emitted kernel targets was ABSENT => NET-NEW.
Opponent = stock q5_K block-dot (objdump-confirmed the only q5_K mul_mat on riscv), NOT SELF.

## deployed kernel (OUR compiler-emitted, reused — no regen)
- q5_K repack **GEMM** `gemm_q5_K_q8_K.kernel.c` md5 **ba30ba54** (kernel-axis-proven: k1 VLEN256 1.916×
  vs factory block-dot, `kquant-k1-vlen256-kernel-axis-t4a`)
- q5_K repack **GEVM** `gemv_q5_K_q8_K.kernel.c` md5 **c445b89e**
- both VLA (dynamic `__riscv_vsetvl_e32m1` + avl=8 f32m2 chunks) => valid at VLEN>=128; k1 VLEN256 native.
- entry ABI (verified against source): GEMM `(n,s,vx,vy,nr,nc,bs)` store row-stride=bs(v7); GEVM `(n,s,vx,vy,nc)`.

## net-new wiring (deploy_patch_q5_K_emitted.py) — 3 tracked files, reversible
- **repack.h**: `struct block_q5_Kx16` (stride 2816: d[16]@0 dmin[16]@32 scales[192]@64 qh[512]@256 qs[2048]@768)
  + 2 arch-body decls.
- **repack.cpp (GEN)**: `make_block_q5_Kx16` (d/dmin/scale-region byte-IDENTICAL to proven `make_block_q4_Kx16`
  + qh 5th-bit plane straight interleave + qs pushed to @768; derived from `kquant_repacker.h` /
  `RVVLowerQuantContraction kQ5KDecodeFacts`) + `repack_q5_K_to_q5_K_16_bl` + `repack<block_q5_K,1,16>` +
  `gemv/gemm<block_q5_K,1,16,Q8_K>` templates + `q5_K_16x1_q8_K` trait + riscv dispatch branch (case256=ON).
- **arch/riscv/repack.cpp**: `#include` the 2 emitted `.inc` + `ggml_gemv/gemm_q5_K_16x1_q8_K` bodies
  (call OUR emitted VLA kernel; banner on first engage).

## build + seal (g5_m2_q5_K_build_seal.sh) — dedicated /data/build-k1-q5k, clang-18 SYMMETRIC
Both OFF & ON built from the SAME k1 tcrv tree, recompiling ONLY repack.cpp.o + arch/riscv/repack.cpp.o
(exact `compile_commands.json` flags) + relink => the ONLY A/B difference is the q5_K dispatch.
- **OFF** lib md5 `14b6add6...`  q5_K tcrv symbols = **0** (block-dot baseline)
- **ON**  lib md5 `a408563e...`  q5_K tcrv symbols = **2** (T exported): `..._gemm_q5_K_q8_K_kernel...`,
  `..._gemv_q5_K_q8_K_kernel...`; banners present (gevm=1, gemm=1); ON != OFF.
- objdump seal (ON): GEMM `vwmacc=2240` (== kernel-axis seal, SAME kernel deployed) VLA shapes
  `e16,m1×26 · e8,mf2×24 · e32,m2×2`; GEVM `vwmacc=560` `e16,m1×28 · e8,mf2×24 · e32,m2×5`.

## correctness (certificate three-requirements + MIRAGE de-risk)
Opponent-oracle B = stock block-dot (ggml's OWN reference, independent of our repack). Same input path
(same gguf, same q8_K activation quant, physical .so swap in one tree). Corpus = 4 prompts + ppl corpus.
- **greedy A==B (--simple-io -st, temp0 top-k1)**: **3/4 BYTE-IDENTICAL**
  - p1 "The capital of France is" → both "Yes, the capital of France is Paris." ✅
  - p2 "Once upon a time" → both "…in a small village nestled in the heart of the countryside…" ✅
  - p3 "Q: What is 2 + 2? A:" → A "The answer to the question \"What is 2 + 2?\" is 4." vs B "The answer
    to the question is 4." — **bounded-ULP coherent divergence** (both correct answer 4; single argmax flip
    cascaded) ⚠️
  - p4 "The quick brown fox jumps" → both "…is a common idiom that means \"something happens quickly…" ✅
- **e2e perplexity (argmax-immune numeric)**: ON **17.9683 ± 3.809** vs OFF **17.9095 ± 3.785** (Δ 0.33%,
  well within ±3.8 error bars => statistically indistinguishable). ON banner confirms our kernel engaged.
- **MIRAGE ruled out**: coherent output + matched ppl. A layout/ABI bug would give PPL in the hundreds/
  thousands (cf. vl16 PPL 822057) or NaN — this is definitively correct wiring.
- **Verdict**: correctness = **GREEN at bounded-ULP level** — kernel proven bit-exact-INTEGER + bounded-ULP-
  norm vs independent oracle (kernel-axis, k1 VLEN256 ULP0 int / 8e-7 norm); e2e coherent + ppl-matched.
  NOT strict byte-identical over the corpus (our repack fp-fold ≠ block-dot fold — expected, unlike the
  q4_K WORK-ITEM where A=stock-repack shared ggml's exact fold). This is the honest correctness standing.

## provisioning
q5_K model built ON-BOARD (决策卡④): k1-local `llama-quantize` (built from pristine tree) requantized
`/data/tinyllama-q8_0.gguf` → `/data/tinyllama-1.1b-Q5_K_M.gguf` (783 MB, sha256 `6002d505...`). No rvv
round-trip needed. Q5_K_M mixes q5_K (most tensors) + q6_K (output) + q4_K.

## e2e phase-split (prefill pp64 / decode tg32) — paired A/B, physical .so swap, ONE tree, clang-18 symmetric
Config: taskset -c 0-3, threads=4 (shared board — 4-core pin avoids the 8-thread oversubscription that
contaminated the first attempt at baseline load ~4; paired interleaved ratio robust to steady load, per
kernel-axis k1 methodology). PP=64 TG=32 REPS=6 PASSES=2 => 12 samples/side. gov=performance 1.6GHz locked.
loadavg during clean run ~6.9-7.1 (my 4 threads + board baseline ~3), freq_khz=1600000 all sides.

| phase | ON (our repack) t/s | OFF (block-dot) t/s | ratio ON/OFF | verdict |
|-------|--------------------:|--------------------:|-------------:|---------|
| **PREFILL (pp64)** | 4.5046 (relIQR 0.19%) | 2.7454 (relIQR 0.09%) | **1.641×** | **≥parity WIN (GREEN)** |
| **DECODE (tg32)**  | 1.7672 (relIQR 0.13%) | 2.4244 (relIQR 0.18%) | **0.729×** | **<parity LOSS (YELLOW-roofline)** |

n=12/side (2 passes × 6 reps). Pass-agreement: prefill 1.637×(p1)→1.641×(both), decode 0.728×→0.729×
=> load steady, ratio trustworthy. relIQR <0.2% both sides = clean uncontended 4-thread measurement.
Kernel==system ledger (clang-18 symmetric, single tree, one llama-bench ELF).

## [PERF-1] eight-gate walk (prefill axis)
① correctness: PASS @ bounded-ULP (bit-exact-integer cert + e2e coherent + ppl 0.33% within error; 3/4 greedy byte-identical) · NOT strict-byte-identical (our fold ≠ block-dot fold)
② VLEN-flip / dual-board: e2e k1/VLEN256 ONLY (kernel-axis already dual-board; rvv e2e = separate gcc-spill regime) → PARTIAL
③ objdump seal: PASS (nm 2 tcrv symbols exported; vwmacc=2240 == kernel-axis seal; VLA vtype sealed)
④ micro AND e2e: PASS prefill (micro 1.916× AND e2e 1.641× both positive) · decode e2e loss (GEVM not micro headline)
⑤ dual-board e2e verified: k1-only → PARTIAL
⑥ discipline: PASS (DVFS lock 1.6GHz, taskset pin, paired interleaved, board-load gate honored — dirty 8-thread run discarded + retried clean, tight IQR)
⑦ mechanism attribution: PASS (prefill win = repack-GEMM compute transduction; decode loss = interleaved-layout bandwidth roofline for nr=1)
⑧ wording: PASS (bounded k1/VLEN256/clang-18/prefill-phase; our-repack vs stock block-dot; net-new dispatch; NOT universal beat)
=> prefill is a legit **e2e transduction GREEN-candidate** (①③④⑥⑦⑧ pass, ②⑤ partial=single-board e2e), NOT a sealed universal 8-gate Win.

### interpretation (split result — matches pre-registered dual-exit)
- **Prefill GREEN 1.637×**: our net-new q5_K repack GEMM transduces its kernel-axis win (1.916× micro) to
  e2e prefill (compute-bound, benefits from the interleaved repack GEMM). Directionally consistent with the
  WORK-ITEM q4_K@k1 prefill 2.644× (q5_K weaker because the qh 5th-bit plane leaves a residual spill floor
  the q4_K min-fold does not — per l1-t3-q5k objdump seal). **k1 ships NO q5_K repack => our-kernel net-new
  path => new perf-covered point on the PREFILL axis.**
- **Decode YELLOW 0.728×**: memory-bound GEVM (nr=1, weights streamed once/token); the block_q5_Kx16
  interleaved layout (stride 2816, +qh plane) costs more bandwidth than plain block_q5_K block-dot for the
  single-row case => roofline loss. Consistent with the campaign law "kernel/prefill wins don't transplant
  to memory-bound decode" ([[kernel-wins-dont-transplant-to-e2e]]). Pre-registered outcome: decode回退→黄-物理墙.

## VERDICT (裁四.2 pre-registered dual-exit — SPLIT)
- **PREFILL = GREEN**: 「k1·q5_K·prefill·e2e **1.641×** (≥parity)·kernel账==system账(clang-18 对称)·对手 stock
  q5_K block-dot·**our-kernel 净新 dispatch·perf-covered 6/83→7/83 (prefill axis)·k1 唯一新绿点·C3′ 绿路径家族
  扩展(q4_K 后第2 K-quant e2e 传导·our kernel)**」. 成色: our compiler-emitted repack GEMM (ba30ba54) vs
  ggml's OWN stock block-dot fallback (k1 ships zero q5_K repack => genuinely net-new path, not a white-label
  stock toggle). e2e transduction GREEN-candidate (②⑤ single-board e2e), NOT a sealed universal 8-gate Win.
- **DECODE = YELLOW-物理墙(roofline)**: 「k1·q5_K·decode·e2e **0.729×** (<parity)·memory-bound GEVM: block_q5_Kx16
  interleaved stride-2816 (+qh plane) streams more bytes/token than plain block_q5_K block-dot for nr=1 =>
  bandwidth roofline loss·micro↛e2e 卷宗 (prefill-win ↛ decode)」. Pre-registered decode回退→黄-物理墙 met.
- **correctness = GREEN @ bounded-ULP** (kernel bit-exact-integer cert + e2e coherent + ppl-matched + MIRAGE ruled out).

## board restore (verified clean)
- source tree restored byte-exact: repack.cpp=3cac40aa, repack.h=57851439, arch/riscv/repack.cpp=c3c101fd (all baseline)
- stock lib untouched: /data/k1build-stock/bin/libggml-cpu.so.0.15.1 = 871169a0
- no `tcrv_emitted_*_q5_K.inc` leftover in arch/riscv/
- main build/ and build-ime untouched; dedicated build = /data/build-k1-q5k (live lib left at OFF-pristine)
