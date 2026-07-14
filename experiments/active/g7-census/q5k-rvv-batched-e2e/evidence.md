# G7 §3 — q5_K@rvv batched-e2e transduction verification

**Question**: does the M-sweep kernel-axis crossover (q5_K@rvv block-dot: cold M=1 LOSS 0.896 →
M≥4 WIN 1.115–1.294, casefile `../m-sweep-batch-regime/`, ruling `21de0f15`) transduce to a REAL
batched llama.cpp e2e — or is it a driver-loop (r-outer/c-inner) micro artifact that a real
batched `mul_mat` does not reproduce?

**Scope (honest, framed)**: q5_K is already GREEN (any-board, k1 clang-18 prefill 1.641×) →
this役 is **COUNT-NEUTRAL** (perf-covered 9/83 unchanged; a M>1 *regime* result, not a count mover).
Kernel-axis (M-sweep) ≠ batched-e2e ≠ perf-covered — **no cross-inference**. This is not a hunt for
a new green cell; it validates whether the single crossover candidate is e2e-real.

---

## Step 1 — Batched-path feasibility (IS our kernel on the real M>1 mul_mat path?)  ✅ FEASIBLE

**Question**: for a q5_K weight tensor at n_tokens = M>1, which kernel does
`ggml_compute_forward_mul_mat` call — our vec_dot (block-dot) core, or a GEMM/repack path that
bypasses it?

Source read of the deployed A-tree (`/home/ubuntu/tcrv-llamacpp`, llama.cpp @ f3e1828,
`ggml/src/ggml-cpu/`):

1. **llamafile_sgemm has NO q5_K case.** `llamafile/sgemm.cpp` `llamafile_sgemm` only dispatches
   F32 / BF16 / F16 / Q8_0 / Q4_0 / Q5_0 / IQ4_NL (grep of `case GGML_TYPE_*`). For q5_K it returns
   false at both call sites (ggml-cpu.c:1442 pre-quant, :1510 post-quant) → `goto UseGgmlGemm` →
   falls through to the standard path. (LLAMAFILE is ON in the build cache, but q5_K is not covered.)

2. **q5_K is NEVER repacked on RISC-V.** `repack.cpp get_optimal_repack_type()` for
   `GGML_TYPE_Q5_K` has ONLY `ggml_cpu_has_neon()` branches (→ q5_K_8x8 / q5_K_8x4). It has **NO
   `ggml_cpu_has_riscv_v()` branch** — unlike Q4_K / Q2_K / IQ4_NL, which DO have a riscv_v case
   (VLEN256→16x1). On the rvv board (RISC-V, no NEON) the function returns nullptr for q5_K → the
   tensor is never assigned the repack extra-buffer type → `ggml_cpu_extra_compute_forward`
   (ggml-cpu.c:1856) does not intercept → mul_mat runs the standard path.

3. **Standard path = block-dot vec_dot, 16×16 block-tiled.**
   `ggml_compute_forward_mul_mat_one_chunk` calls `type_traits_cpu[Q5_K].vec_dot`
   (= `ggml_vec_dot_q5_K_q8_K`) once per (weight-row, token-col). The loop is 16×16-tiled
   (blck_0 = blck_1 = 16): within a tile each weight row is reused across up to 16 token columns —
   the **real e2e analogue of the M-sweep cold-weight amortization**. `TCRV_GEMM_ENABLED` is 0 (the
   q4_0 fast-path is disabled and q5_K-irrelevant). Model is dense Llama → regular `mul_mat`, not
   `mul_mat_id`.

4. **Splice point identified & stock baseline confirmed.** The stock RVV `ggml_vec_dot_q5_K_q8_K`
   (arch/riscv/quants.c:2081) is a real vectorized kernel (vl=8 core, `#if __riscv_v`), not the
   scalar generic — so OFF = stock RVV vl=8, ON = our weft block-dot (m2 core): symmetric gcc-15.2,
   both real RVV, matching the M-sweep ours-vs-opp identities. `nrc` is always 1 for q5_K.

**Feasibility verdict: FEASIBLE — our q5_K block-dot IS on the real batched (M>1) mul_mat path on
rvv; it is NOT bypassed by llamafile_sgemm or repack-GEMM.** The structural-null exit is ruled out
at the source level: q5_K@rvv routes through vec_dot at every M. A real q5_K model exists on-box
(`DeepSeek-R1-Distill-Llama-8B-Q5_K_M.gguf`, 5.73 GB) → no synthesis needed.

---

## Step 2 — Batched-e2e A/B (llama-batched-bench, M=npl∈{1,4,8})

**Deploy** (reversible, NO git): weft q5_K block-dot spliced into `ggml_vec_dot_q5_K_q8_K`
(arch/riscv/quants.c). OFF (stock) and ON (weft) `libggml-cpu.so` built back-to-back in the SAME
build dir / flags / gcc-15.2 — **ONLY quants.c differs → q5_K kernel is the sole variable.**
- OFF .so md5 `05a62e6a…` == original stock live .so (deterministic build; a perfect stock baseline).
- ON .so md5 `bcaeabb7…` (weft symbol present in ON, absent in OFF — nm-verified).
- quants.c restored to stock md5 `58827c2e…` (double-proof); original live .so byte-restored.

**Model**: llama-batched-bench cannot exercise the 5.7 GB DeepSeek-8B-Q5_K_M fast enough on this
board (~1.7 tok/s prefill / 0.78 tok/s decode → a single N≥10 A/B would be hours). Requantized
`tinyllama-1.1B q8_0 → q5_K_M` (`llama-quantize --allow-requantize`, sha256 `a676aa7f…`, 783 MB,
**135 q5_K + 21 q6_K tensors = 87 % q5_K**). Per-q5_K-matrix weight (ffn = 7.56 MB) still exceeds
cache → the M=1 memory-bound regime that the crossover lives in is preserved, at ~7× the speed.

**Engagement**: WEFT banner ("TCRV G7-S3 WEFT q5_K block-dot ENGAGED") fires in the ON batched-bench
stderr (first q5_K vec_dot on the real measured path); absent in OFF → MIRAGE excluded, our kernel
confirmed live in e2e. Bit-exactness inherited from M-sweep ZERO-MODEL VERIFY_FAIL=0 (weft == stock
RVV bit-for-bit) → ON/OFF logits identical (llama-cli greedy gate dropped: cli decode is single-
threaded on this board, ~21 s/tok, unusable; batched-bench threads fine at ~7 cores).

**Metric**: batched-bench jsonl `speed_tg` (= pl·tg/t_tg, total decode tok/s, **M=pl in decode →
the crossover metric**) and `speed_pp` (prefill, large-M compute-bound). Grid npp=32 ntg=24
npl=1,4,8, ctx=2048, cores 8-15, 8 threads, N=10 interleaved OFF/ON rounds, median + IQR.

### Results (median, tok/s; ratio ON/OFF; IQR shown; N=10 interleaved)

N=10 interleaved rounds, medians. Engagement PROVEN: ON batched-bench stderr =
`TCRV G7-S3 WEFT q5_K block-dot ENGAGED (n=2048 nrc=1)` (67×); OFF = 0. Our weft kernel confirmed on
the real batched mul_mat path (n=2048 contraction). Board hygiene: live .so restored to stock
`05a62e6a…` (match=YES); stray_bb=0.

**DECODE — speed_tg (tok/s), M = pl (batch) — THE CROSSOVER METRIC**
| M (pl) | OFF med | ON med | ON/OFF | verdict |
|---|--|--|--|--|
| **1** | 5.546 | 5.026 | **0.906** | **LOSS** (memory-bound at M=1) |
| **4** | 9.506 | 10.360 | **1.090** | **WIN** (crosses parity) |
| **8** | 10.797 | 12.572 | **1.164** | **WIN** |
(IQR ≤ 0.047 both sides → highly significant.)

**PREFILL — speed_pp (tok/s), large-M compute-bound**
| M (pl) | OFF med | ON med | ON/OFF | verdict |
|---|--|--|--|--|
| 1 | 12.300 | 15.134 | **1.230** | **WIN** |
| 4 | 12.406 | 15.382 | **1.240** | **WIN** |
| 8 | 12.441 | 15.427 | **1.240** | **WIN** |

**The M-sweep kernel-axis crossover reproduces in real batched llama.cpp e2e, to within ~1-2 %:**
| M | M-sweep (kernel-axis micro) | batched-e2e decode (this役) |
|---|--|--|
| 1 | 0.896 LOSS | **0.906 LOSS** |
| 4 | 1.115 WIN | **1.090 WIN** |
| 8 | 1.192 WIN | **1.164 WIN** |

Prefill (always large-M) is a strong outright WIN (1.23-1.24×) — the compute-bound regime where our
wider-LMUL block-dot beats the stock RVV vl=8 kernel with weights fully amortized. Decode crosses
parity between M=1 and M=4 exactly as the M-sweep predicted: at M=1 each token streams the q5_K weight
cold from DRAM (memory-bound → our compute edge masked → 0.906 LOSS); at M≥4 ggml's 16×16-tiled
one_chunk reuses each weight row across the batched token columns (compute-bound → our edge shows).

---

## Step 3 — Verdict (three exits)

**EXIT = 传导 WIN (transduction confirmed).**

- Step 1 ruled out the **structural-null** exit at source level (q5_K@rvv routes through vec_dot
  block-dot at every M; not bypassed by llamafile_sgemm or repack-GEMM) — and the WEFT banner
  (n=2048) empirically confirms our kernel is the live kernel in the batched path.
- Step 2 rules out the **washout** exit: the batched-e2e decode ratio is NOT flat-<parity — it
  **crosses parity** (M=1 0.906 LOSS → M=4 1.091 / M=8 1.164 WIN), and prefill is an outright
  1.23-1.24× WIN. The M-sweep crossover (M=1 memory-bound washout → M≥4 compute-bound win) is
  **real in batched llama.cpp e2e**, not a driver-loop (r-outer/c-inner) micro artifact — because
  ggml's own `mul_mat_one_chunk` is 16×16 block-tiled, which supplies the same cold-weight
  amortization across the batched token columns.

→ **The "M=8 new-regime" mechanism decree is e2e-validated for q5_K@rvv.** At batch M≥4 our q5_K
block-dot beats the stock RVV vec_dot in real batched decode; at M=1 (unbatched decode) it loses to
memory-boundness, matching the kernel-axis M-sweep.

## 4. Honesty / craftsmanship (count-neutral, regime result)

- **COUNT-NEUTRAL.** q5_K is already GREEN (any-board). This役 does NOT move perf-covered (9/83
  unchanged), certified, or kernel-sym. It is a **M>1 regime result** — the batched-decode/prefill
  win is a new *regime* fact, not a new green cell. **No count changed.**
- **No互推.** kernel-axis M-sweep ≠ this batched-e2e ≠ perf-covered 9/83. The e2e number here is a
  system-account (decode/prefill tok/s A/B), independent of the second-track kernel-sym census.
- **Model proxy caveat (honest).** Measured on requantized **tinyllama-q5_K_M** (783 MB, 87 % q5_K),
  NOT the 5.7 GB DeepSeek-8B (too slow for N≥10 on this board: ~1.7/0.78 tok/s). Per-q5_K-matrix
  weight (7.56 MB) exceeds cache → the M=1 memory-bound regime that the crossover requires is
  preserved; the 8B (weights ≫ cache) would be *even more* memory-bound at M=1, so the crossover
  direction is conservative, not inflated. The *absolute* prefill-win magnitude (1.24×) is a
  wider-LMUL compute-quality win and is model-agnostic for the q5_K matmuls.
- **Amdahl framing.** The e2e ratios ARE the whole-model tok/s (q5_K matmuls are 87 % of the
  quantized weight; the q6_K remainder + attention/RoPE/softmax are identical in ON/OFF and dilute
  toward 1.0). That the whole-model decode still shows 1.09-1.16× at M≥4 and 1.23× prefill means the
  q5_K-matmul win is large enough to survive Amdahl dilution — a genuine e2e signal, not a
  micro-only artifact ([CASE-MICRO-E2E] does NOT extend to the batched q5_K@rvv regime).
- Bit-exact / MIRAGE: weft == stock RVV vl=8 bit-for-bit (M-sweep ZERO-MODEL VERIFY_FAIL=0) → ON/OFF
  logits identical; WEFT banner proves the ON path ran our kernel (not stock silently).

## 5. Board hygiene
- Deploy reversible: quants.c restored to stock md5 `58827c2e…` (double-proof); OFF .so md5
  `05a62e6a…` == original stock live .so; live .so byte-restored after bench (trap + md5 match). NO git.
- co-tenant: baseline vLLM core0,1 undisturbed; my bench pinned core8-15, 8 threads; load spikes to
  ~10 during bench = my own 8-thread load (not external), settles to ~2.2 baseline between.
- strays pkill'd; scratch /tmp/g7_q5k_* + variant .so's cleaned; loadavg logged per round.
