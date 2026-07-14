# Opponent-probe (L1 crux) — does ggml have a WORKING q4_K/q5_K repack path @ VLEN128?

**Answer: NO — for BOTH q4_K and q5_K. Both are L1 path-win candidates (same precedent as q4_0).**
Established two ways: (1) exact source of the board's own llama.cpp, (2) a runtime dispatch probe run
at the LIVE `__riscv_vlenb()` on the board.

Opponent: `/home/ubuntu/llama.cpp-upstream-native`, tag **b9692**, commit **f3e1828**, file
`ggml/src/ggml-cpu/repack.cpp` :: `ggml::cpu::repack::get_tensor_traits()`.

## ⚠ memory correction

The prior memory ("ggml q4_K/q5_K repack hardcoded VLEN256"; "opponent q4_K@128 factually wrong") is
now REFINED against current source: this llama.cpp **does** ship a RISC-V `ggml_gemm_q4_K_16x1_q8_K`
(arch/riscv/repack.cpp:983) — but it is **explicitly VLEN256-gated** in the trait selector, and q5_K
has **no RISC-V repack at all**. So "no WORKING repack @ VLEN128" holds for both, for different reasons.

## The two gates

**q4_K** — `get_tensor_traits()` RISC-V branch (repack.cpp:4616-4624):
```
if (ggml_cpu_has_riscv_v()) {
  switch (__riscv_vlenb() * 8) {
    case 128:  { break; } // TODO      <-- VLEN128 falls through: NO trait returned
    case 256:  { if (ne[1] % 16 == 0) return &q4_K_16x1_q8_K; break; }
    ...
  }
}
```
At VLEN128 the case is a TODO no-op → the function returns nullptr → the tensor is **not repacked** →
`ggml_compute_forward_mul_mat` uses the standard per-(row,col) block-dot `ggml_vec_dot_q4_K_q8_K`.
Belt-and-suspenders: even the gated-off `ggml_gemm_q4_K_16x1_q8_K` hardcodes `vfmv_v_f_f32m2(...,16)`
(16 > VLMAX=8 at VLEN128) → it would be numerically WRONG at VLEN128 — which is precisely why it is
`case 256` only.

**q5_K** — the `GGML_TYPE_Q5_K` branch (repack.cpp:4644+) has ONLY neon (`matmul_int8` / `dotprod`)
sub-branches; there is **no `ggml_cpu_has_riscv_v()` branch**. On RISC-V it returns nullptr at EVERY
VLEN → q5_K is never repacked → block-dot `ggml_vec_dot_q5_K_q8_K` fallback.

## Runtime confirmation (board, LIVE VLEN=128) — `dispatch_probe_raw.txt`

```
Q4_K  -> NULLPTR(block-dot fallback) [case 128 = TODO no-op]
Q5_K  -> NULLPTR(block-dot fallback) [no riscv branch in selector]
```
(Also NULLPTR for Q4_0/Q2_K/Q8_0/IQ4_NL at 128, and Q6_K — the whole K-quant repack family is
VLEN256-or-absent on RISC-V.)

## What the opponent ACTUALLY runs (the fair prefill baseline)

`nm libggml-cpu.so`: `ggml_vec_dot_q4_K_q8_K` dispatches to a **hand-tuned `_vl128` variant**;
`ggml_vec_dot_q5_K_q8_K` has only a **`_generic`** implementation (no vl128 variant). This asymmetry
predicts — and matches — the prefill result: parity on q4_K (tuned opponent), a gap on q5_K (untuned).

## Prefill paired result (`prefill_paired.txt/.csv`) — [NG-4] candidate, NOT a beat

| fmt | ours GMAC/s | opponent (real dispatched) GMAC/s | ratio | reading |
|---|---:|---:|---:|---|
| q4_K | ~4.21–4.25 | ~4.37–4.47 (vl128-tuned block-dot) | **0.94–0.97×** | ≈ PARITY (ours marginally slower) |
| q5_K | ~2.16–2.19 | ~1.34–1.47 (generic block-dot) | **1.50–1.62×** | ours faster |

(K=2048, nc=512, nr∈{16,64,256}, core 8 @ 2.6 GHz, best-of-5.)

## Verdict + discipline

- **L1 STRUCTURAL path-win candidate: CONFIRMED for both q4_K and q5_K** — opponent has no working
  repack @ VLEN128; we do (VLEN128-correct `half_lanes=8` two-strip repack GEMM, oracle-GREEN at
  039133ea). This is the same precedent as q4_0.
- **Throughput**: q5_K shows a ~1.5–1.6× gap; q4_K is ≈parity — an HONEST caveat that the structural
  path-win does NOT imply a throughput win where ggml's block-dot is already VLEN128-tuned.
- **[NG-4] beat still FORBIDDEN**: opponent-absence is now proven, but the eight [PERF-1] gates are
  NOT walked; the paired is single-core, opponent = single-thread block-dot loop (kernel-axis proxy,
  not ggml threaded mul_mat), throughput timing (our correctness = construction oracle, not re-verified
  this run). These are candidate datapoints, not a win.
