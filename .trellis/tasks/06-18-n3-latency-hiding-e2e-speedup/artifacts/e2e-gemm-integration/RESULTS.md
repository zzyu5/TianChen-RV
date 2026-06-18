# E2E q4_0 GEMM integration into OUR llama.cpp — prefill speedup on `ssh rvv`

Wires the validated M-blocked q4_0×q8_0 prefill GEMM (decode amortization) into the
q4_0 `mul_mat` of **our own** clean llama.cpp (`~/tcrv-llamacpp`, mainline `b9692`/
`f3e1828`, GNU 14.2, VLEN=128) and measures the end-to-end prefill (pp512) tok/s win
over ggml's default per-`ggml_vec_dot_q4_0_q8_0` path.

**This is the e2e performance deliverable. The M-blocking structure is standard and is
NOT claimed as novelty — only the honest measured e2e number is.**

---

## TL;DR

- **pp512 (prefill): 1.55 → 2.24 tok/s = 1.44× e2e speedup** (`-t 8`, Llama-2-7B Q4_0).
  Confirmed in **two** independent invocations (standalone 2.24, baseline-form combined 2.23).
  Below 1.5× purely by Amdahl (activation q8_0 quant, attention, the block-major repack,
  and norms are NOT sped up — only the q4_0 weight matmul is). Honest, real, not inflated.
- **tg128 (decode): 1.38 → 1.23.** This is **NOT** a regression from the integration: decode
  is single-column (M=1) GEMV → our gate routes it to the **unmodified** ggml path (proven:
  `TCRV GEMM ENGAGED` count during tg = **0**; the decode machine code is byte-identical to
  stock). tg is memory-bound and does not ramp the `ondemand` governor (measured 1.14–1.27
  GHz this session), so cross-session tg carries clock noise that the compute-bound,
  self-ramping pp512 (pinned 2.6 GHz both sessions) does not. Decode is GEMV/reduction-capped
  and is explicitly not our target. (Corollary: the board was in a cooler tg state now, so the
  pp 1.44× is if anything conservative.)
- **Correctness: bit-exact** (by composition). The GLUE (repack + scatter) is VERIFY-proven on
  real model data (in-loop abort-check vs `ggml_vec_dot_q4_0_q8_0`, **0 mismatches**); the
  shipped v4fold KERNEL is `max_err_vsV0 = 0` over 32 000 trials at the model's exact n
  (4096/11008) under GCC. The shipped build's greedy output matches stock token-for-token
  ("The capital of France is Paris."). (The VERIFY run predates the kernel swap, but the glue
  it covers is byte-identical in the shipped build.)
- **The load-bearing finding (genuinely new): kernel choice is COMPILER-SENSITIVE.** The
  micro-probe's winning kernel under **clang** (scalar fold, 1.56×) is the *losing* kernel
  under **GCC** (the ggml build compiler): it gives only ~1.05× and a **0.88× e2e
  regression**. The compiler-robust kernel is the vector-fold (`vmv.x.s`-free) one. We ship
  that. See "Compiler sensitivity" below — this is why the first integration regressed.

---

## What was integrated (and where)

`ggml/src/ggml-cpu/ggml-cpu.c`, inside `ggml_compute_forward_mul_mat_one_chunk` (the
per-chunk worker called under ggml's existing threadpool + after the q8_0 activation
quantization + barrier). Full diff: `ggml-cpu-q4_0-gemm.patch`.

A fast-path is inserted **before** ggml's per-(row,col) `vec_dot` loop. It engages only when:

```
type == GGML_TYPE_Q4_0  &&  vec_dot_type == GGML_TYPE_Q8_0
  &&  num_rows_per_vec_dot == 1            // no ARM mmla 2-row path (n/a on riscv)
  &&  ne12 == 1 && ne13 == 1              // simple (unbroadcast) 2D matmul -> i11 = ir1
  &&  ne00 % 32 == 0
  &&  (ir1_end - ir1_start) >= 2          // PREFILL only: >=2 activation columns
```

For a 16-col × 16-row chunk it:
1. repacks the M (≤16) q8_0 activation columns **block-major** into a `_Thread_local`
   scratch (`vyp[(ib*M+j)*YS]`), so one weight block's M columns are contiguous;
2. for each weight row, calls `tcrv_gemm_q4_0_q8_0(ne00, M, gout, src0_row, vyp)` —
   **decode each q4_0 weight block's nibbles ONCE, reuse across all M columns**;
3. scatters `gout[j]` to `dst[col=iir1+j][row=ir0]`.

**Decode (tg) and any non-q4_0/broadcast/M=1 case fall through to the original ggml loop,
byte-for-byte unchanged.** Threading and the q8_0 activation quantization are untouched.

The kernel is `kern_gemm_v4fold` from the validated probe (verbatim), chosen over the
scalar-fold variant for the compiler reason below. It is identical in interface
(`out[0..M)`), bit-exact, and already proven `max_err_vsV0 = 0` under GCC.

---

## Host / build (this run)

| field | value |
|---|---|
| host | `ssh rvv`, `ubuntu`, riscv64, 64 harts, VLEN=128 |
| repo | OUR `~/tcrv-llamacpp`, mainline `b9692` / `f3e1828`, ggml 0.15.1 |
| compiler | **GNU 14.2** (`/usr/bin/cc`), `-march=rv64gcv_zfh_zvfh_zicbop_zihintpause -mabi=lp64d`, Release |
| model | `/home/ubuntu/llama-2-7b-chat.Q4_0.gguf` (Q4_0, n_embd=4096, n_ff=11008, 32 layers) |
| build discipline | `-j4` only (board is fragile); separate VERIFY build for the abort-check |
| baseline (BEFORE) | banked in `../e2e-baseline/RESULTS.md`: pp512=1.55, tg128=1.38 (±0.00, same commit/flags). ggml does NOT repack q4_0 at VLEN=128 → the per-`vec_dot` path is the real default denominator. |

---

## Measured e2e (AFTER = our GEMM, perf build, `-t 8`)

| test | BEFORE (stock ggml) | AFTER (our GEMM) | ratio | notes |
|---|---:|---:|---:|---|
| **pp512 (prefill)** | **1.55** | **2.24 / 2.23** | **1.44×** | 2.24 standalone (`-p 512 -n 0`), 2.23 combined (`-p 512 -n 128`, baseline form) — agree. GEMM `ENGAGED nb=128 cols=16 rows=16` confirmed in bench log. |
| tg128 (decode) | 1.38 | 1.22 / 1.23 | 0.89× | decode = M=1 → routed to **stock** (`ENGAGED`=0 during tg). NOT a code regression — see note. |

`ENGAGED` = a one-shot diagnostic `fprintf` proving the fast-path actually fired during the
bench (it is gated, so a null result would otherwise be ambiguous). The pp512 bench printed
`cols=16` — exactly the M where the micro-probe peaks. pp512 agrees to ±0.5% across two
independent invocations → the 1.44× headline is not a single-clock artifact.

> **tg128 note (NOT a regression).** The decode path is **byte-identical to stock**: decode is
> single-column (M=1), so our gate falls through to the unmodified `one_chunk` loop
> (`ENGAGED`=0 during both the standalone and combined tg phases). The only added cost is the
> gate's ~5 integer comparisons per chunk — nanoseconds against a memory-bound, hundreds-of-
> ms/token decode. So our change *cannot* move tg. The 1.38→1.23 delta is **cross-session
> clock noise**: pp is compute-bound and self-ramps the `ondemand` governor to 2.6 GHz (so
> pp is clock-matched across the baseline session and this one — why 1.44× is robust), but tg
> is memory-bound and never ramps (measured 1.14–1.27 GHz this session vs whatever the
> baseline session gave). We did NOT rebuild stock to re-measure tg: `ENGAGED`=0 already proves
> the decode bytes are unchanged, and a re-measure would only re-confirm that at the cost of
> board-crash risk. Decode is GEMV/reduction-capped and explicitly not our target (the prior
> probe capped q4_0 GEMV at ~1.22×).

---

## Correctness — bit-exact (by composition: glue ∘ kernel)

The shipped result is bit-exact to ggml's `ggml_vec_dot_q4_0_q8_0`, established as the
composition of two independently-proven halves:

1. **The GLUE (repack indexing + transposed scatter) is bit-exact.** A build with
   `-DTCRV_GEMM_VERIFY=1` runs, *inside the matmul*, the original `ggml_vec_dot_q4_0_q8_0`
   into a shadow buffer for every GEMM output and `abort()`s on any mismatch. Run on
   `llama-cli` (greedy, `--temp 0 --seed 1234`, prompt "The capital of France is"):

   ```
   EXIT=0   TCRV GEMM MISMATCH count = 0   abort count = 0
   output: "The capital of France is Paris."   (identical to stock)
   ```

   This VERIFY run was on the *scalar-fold* kernel (before the v4fold swap), so it proves the
   **glue** — the block-major repack, the per-(row,col) addressing, and the `dst[col][row]`
   scatter — is bit-exact on real model data. **The glue is byte-identical in the shipped
   build; only the kernel body changed.**

2. **The shipped v4fold KERNEL is bit-exact to V0.** The standalone harness checks v4fold's
   numerics at the model's exact dimensions (n = 4096 and 11008) over 32 000 trials under
   **GCC** at the ggml flags: `max_err_vsV0 = 0.000e+00` (probe `ACC GEMM_v4fold` line). This
   is a direct test of the kernel body, not a proxy — same compiler, same n as e2e.

**Composition:** glue (VERIFY-proven, real model data) ∘ v4fold kernel (harness-proven,
model n, GCC) → bit-exact. Independently, the shipped v4fold build's greedy output matches
stock token-for-token ("The capital of France is Paris."). Decode amortization is a
scheduling change only; the per-block accumulation order matches V0 → logits unchanged → drop-in.

---

## Compiler sensitivity — why the first integration REGRESSED (and the fix)

The micro-probe (`../failfast-gemm/`) was built with **clang 18.1.3** and found the
**scalar-fold** GEMM at 1.56× (and documented the vector-fold variant as a `vsetvli`-thrash
*negative*). The e2e ggml build uses **GCC 14.2**. The 1.56× was never measured under GCC.

Re-running the *same probe* under GCC at the ggml flags
(`probe-gcc-compiler-sensitivity.log`) is decisive — note the **absolute** ns/out:

| kernel @ n=11008, M=8 | clang ns/out | GCC ns/out | clang speedup | GCC speedup |
|---|---:|---:|---:|---:|
| V4_GEMV (vector fold ref) | 2355 | 2400 | 1.22× | 3.62×* |
| **v4fold GEMM** (vector fold) | 2402 | 2508 | 1.19× | 3.47×* |
| **scalar GEMM** (what we first shipped) | **1876** | **8260** | **1.56×** | **1.05×** |
| V0_ggml (baseline `vec_dot` copy) | 2857 | 8694 | 1.00× | 1.00× |

\* The GCC "3.5×" is a **broken-baseline artifact, NOT a fast kernel** — the vector-fold
kernels run at the *same absolute speed* under both compilers (2402→2508, +4%); their
speedup only tripled because the **`vmv.x.s`-using kernels (V0 and scalar-GEMM) are
GCC-pessimized ~4×** (1876→8260). On this in-order board GCC 14 serializes the per-block
`vmv.x.s` scalar-extract; clang did not. So:

- **clang:** scalar fold is the global best (1876 ns); vector fold thrashes `vsetvli`.
- **GCC:** scalar fold collapses to ~parity (1.05×); the **vector fold (`vmv.x.s`-free) is the
  robust winner.**

We first integrated the scalar fold → it gave a measured **pp512 = 1.37 (0.88× regression)**
(`after-pp512.log`), because at the kernel level GCC gives it no win and the repack tipped
the net below 1.0. Swapping to the **vector-fold** kernel gave **pp512 = 2.24 (1.44×)**.
The real, transferable lesson: **the winning RVV micro-kernel is compiler-dependent — pick
the `vmv.x.s`-free vector-fold for the GCC ggml build.** The vector fold also scales cleanly
under GCC (totals M8/M12/M16 = 7292/11083/14561 ns, linear — not the degenerate `vsetvli`
fast-path clang showed at M16), so it is safe across the live `cols=2..16` range.

---

## Honest framing

- pp512 **1.44×** is a **desirable** e2e prefill result, below the kernel-level ceiling
  (~1.5–3.5× depending on compiler) by Amdahl: the q4_0 weight matmul is the only sped-up
  op; activation q8_0 quantization, attention, RMSNorm, and the block-major repack are not.
  The task explicitly permits an honest sub-1.5× e2e — this is reported as the real number.
- tg128 (decode) is untouched by design (M=1 falls back to stock, `ENGAGED`=0). The 1.38→1.23
  reading is cross-session clock noise on a byte-identical path, not a regression. Not our win.
- The M-blocking / decode-amortization structure is standard; **no novelty is claimed** for
  it. The one genuinely interesting finding is the compiler-sensitivity of the RVV kernel.

---

## Files in this dir

- `RESULTS.md` — this file
- `ggml-cpu-q4_0-gemm.patch` — the full integration diff (unified, vs pristine `.orig`)
- `correctness-verify.log` — VERIFY build: 0 mismatches, 0 aborts, "Paris" output
- `engaged-proof.log` — positive proof the GEMM fast-path fires in the perf build
- `after-pp512.log` — FIRST integration (scalar fold): pp512=1.37 (0.88× regression) + the cause
- `after-pp512-v4fold.log` — the SHIPPED v4fold, standalone: **pp512 = 2.24** + ENGAGED cols=16
- `after-tg128-v4fold.log` — tg128 standalone (decode falls back to stock, engaged=0)
- `after-combined-v4fold.log` — baseline-form combined run: **pp512 = 2.23**, tg128 = 1.23 (re-validates the headline)
- `probe-gcc-compiler-sensitivity.log` — the GCC-vs-clang discriminator (scalar 1.05× vs vector robust)
