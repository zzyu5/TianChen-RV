# G7 §3 — q5_K@rvv batched-e2e transduction verification (casefile)

Verifies whether the M-sweep kernel-axis crossover (q5_K@rvv block-dot: M=1 LOSS 0.896 →
M≥4 WIN 1.12–1.29, casefile `../m-sweep-batch-regime/`, ruling `21de0f15`) transduces to a
REAL batched llama.cpp e2e, or is a driver-loop (r-outer/c-inner) micro artifact.

**Scope discipline**: q5_K is already GREEN (any-board, via k1 clang-18 prefill 1.641×) →
this役 is COUNT-NEUTRAL (perf-covered 9/83 unchanged, a M>1 *regime* result, not a count mover).
Second-track (kernel-axis) ≠ perf-covered ≠ batched-e2e — NO cross-inference.

## Files
- `evidence.md` — feasibility (Step 1) + batched-e2e A/B (Step 2) + verdict (Step 3, three exits).
- `board/harness/`
  - `weft_q5k_blockdot.inc` — OUR weft-emitted q5_K q8_K block-dot, transformed extern-C→file-scope
    static for in-tree splice (source: `../vecdot-rvv/kernels/q5_K.kernel.c`). Byte-exact vs stock
    RVV vl=8 (M-sweep VERIFY_FAIL=0).
  - `deploy_patch_q5k_weft.py` — reversible in-place splice into `arch/riscv/quants.c`
    (`ggml_vec_dot_q5_K_q8_K`); md5 baseline check `58827c2e…`; include + intercept-call + one-shot
    banner. Mirrors G5-M2 iq4_nl deploy precedent.
  - `build_ab_q5k.sh` — builds OFF (stock) + ON (weft) `libggml-cpu.so` variants in the tcrv-llamacpp
    A-tree (gcc-15.2, VLEN128); ONLY quants.c differs; restores quants.c (md5 double-proof) and the
    original live .so (byte-identical). NO git.
  - `run_bench_q5k_v2.sh` — **used for the sealed run.** llama-batched-bench A/B, M=npl∈{1,4,8},
    N=10 interleaved rounds, atomic in-place .so swap (impl.so uses DT_RPATH → LD_LIBRARY_PATH cannot
    override), core8-15 pinned, load-gated, restores live .so. Engagement = WEFT banner from the ON
    bench stderr (fires on the real path); bit-exactness inherited from M-sweep VERIFY_FAIL=0.
  - `run_bench_q5k.sh` — v1 (superseded): included a llama-cli greedy correctness gate; dropped
    because llama-cli batch-1 decode is single-threaded on this board (~21 s/tok, unusable) while
    llama-batched-bench threads fine (~7 cores).
  - `parse_bench.py` — per-M median speed_tg/speed_pp + IQR + ON/OFF ratio.
- `board/raw/rvv_tinyllama_q5k_batched_ab_N10.log` — the sealed N=10 A/B jsonl + banner + hygiene.

## Board / provenance
- rvv: openEuler riscv64, VLEN128, gcc-15.2.0 (`/opt/tcrv-toolchains`), core8-15, co-tenant vLLM core0,1.
- A-tree: `/home/ubuntu/tcrv-llamacpp/build-gcc15-rv64gcv` (llama.cpp @ f3e1828).
- **Model (sealed run)**: `/home/ubuntu/models/tinyllama-q5_K_M.gguf` (783 MB, sha256 `a676aa7f…`,
  135 q5_K + 21 q6_K tensors), requantized from `tinyllama-q8_0` via `llama-quantize --allow-requantize`.
  Chosen because the 5.7 GB DeepSeek-8B-Q5_K_M (`/home/ubuntu/models/…Q5_K_M.gguf`) runs too slow for
  N≥10 on this board (feasibility-probed: 1.7 tok/s prefill / 0.78 tok/s decode). Per-q5_K-matrix
  weight (7.56 MB) still exceeds cache → M=1 memory-bound crossover regime preserved.
- OFF .so md5 `05a62e6a…` (== original stock live .so, deterministic) · ON .so md5 `bcaeabb7…`
  (weft symbol present; OFF absent). quants.c stock md5 `58827c2e…` restored post-build; live .so
  restored to `05a62e6a…` (match=YES); weft .inc removed; scratch cleared; no strays. NO git.
