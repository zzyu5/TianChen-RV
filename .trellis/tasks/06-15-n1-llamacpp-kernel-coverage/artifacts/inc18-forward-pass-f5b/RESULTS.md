# INC-18 F5b — ggml `ggml_vec_soft_max_f32` (the attention op: reduction + transcendental combined)

Date: 2026-06-16. Board: `ssh rvv` (riscv64, rv64gcv, VLEN=128, `__riscv_v_min_vlen
128`, `__riscv_zvl128b`, `__riscv_v_intrinsic 12000`, clang 18.1.3).

## What landed

A STRUCTURED byte-exact compiler-emitted drop-in for ggml's vectorized BARE
`ggml_vec_soft_max_f32` (the attention softmax core: `y[i] = e^{x[i]-max}`,
RETURNING the f64 sum `Σ_i e^{x[i]-max}`; vec.cpp:531 + the `__riscv_v_intrinsic`
path vec.cpp:584-592). F5b is the COMBINED rung — it fuses F5's vectorized exp
transcendental with a NEW reduction shape (the f64 widening reduce).

## Scope correction (primary-source) — faithful to ggml's BARE function

The task IMPLEMENT steps listed (1) a max reduction and (4) a normalize. Reading
the ggml source, those are the **caller** `ggml_compute_forward_soft_max_f32`
(ops.cpp:5400 `ggml_vec_max_f32` + ops.cpp:5415-5416 `sum=1/sum;
ggml_vec_scale_f32`), NOT the function being replaced. The bare
`ggml_vec_soft_max_f32` (vec.cpp:531):
- **takes `max` as an INPUT** (it does not compute it),
- writes only `y[i] = e^{x[i]-max}` (no normalize),
- **RETURNS** the f64 sum.

So the faithful drop-in matches that exact signature/semantics:
`double ggml_vec_soft_max_f32(int n, float *y, const float *x, float max)`. The
deliverable is validated byte-exact against THAT literal function. (The full
row-softmax is a thin scalar-max prologue + an F1-scale epilogue around this
core; cheap to add if wanted, but not what `ggml_vec_soft_max_f32` is.)

- New op `tcrv_rvv.ggml_vec_soft_max_f32` (y out, x in, max, n; fail-closed
  verifier I7; `kind` bounded to `"ggml_vec_soft_max_f32"`, no SEW/LMUL/policy
  attr at the I5 boundary; pinned at m2 — no knob this cut). It is the ONLY
  forward-pass op whose C function RETURNS a scalar (the f64 sum), so the
  lowering builds the function with a `double` result (guarded tightly on the
  soft_max body so EVERY other kernel builds void byte-identically — additivity).
- Lowering (`emitGgmlVecSoftMaxF32`, lib/Conversion/RVV/RVVToEmitC.cpp): the m2
  f32 strip loop (`vsetvl_e32m2(n-i)` / `vle32` / … / `vse32`) carrying a
  loop-carried f64m1 sum accumulator, exactly ggml's vec.cpp:584-592.

## The two reuses (F5's exp + F3's reduction discipline)

1. **F5's exp polynomial — REUSED, not re-derived.** F5's node-for-node
   `ggml_v_expf_m2` chain was factored into a shared member helper
   `emitGgmlVExpfM2(rewriter, X, bodyVL, …)`. silu now calls it (proven
   byte-IDENTICAL re-render of the committed F5 artifact — the 68-intrinsic exp
   sequence is bit-for-bit the same in silu and soft_max). soft_max feeds it
   `vfsub_vf(vle32(x), max)` (x[i]-max). So `y[i] = e^{x[i]-max}` inherits F5's
   already-HW-proven bit-exactness.
2. **F3's reduction discipline (PRINCIPLE) — but the MECHANISM is ggml's exact
   f64 WIDENING reduce, NOT F3's scalar fold.** F3 (rms_norm) accumulates in
   double via a scalar-ascending fold. ggml's soft_max accumulates in
   `ggml_float`=double too, but via the WIDENING reduce
   `vfwredusum_vs_f32m2_f64m1` into a SINGLE f64m1 accumulator carried across
   strips (vec.cpp:585/590/592). The lowering reuses F3's loop-carried-accumulator
   STRUCTURE (an `emitc.variable` lvalue + `emitc.assign`, since `emitc.for` has
   no iter_args) but the value type is the opaque vector `vfloat64m1_t` and the
   per-strip op is the widening reduce — ggml's EXACT method. This is the one
   genuinely-new structural element in F5b.

## raw() = 0 + structured proof

The whole kernel is STRUCTURED emitc nodes — the exp polynomial is a
`call_opaque` chain, the f64 accumulator a `variable`/`assign`/`load`, the reduce
a `call_opaque`, the return a `cast`+`return`. Of the lowered IR: total
`emitc.verbatim` that are NOT `// `-prefixed comment lines = **0**. Every value
(the strip `for`, every intrinsic `call_opaque`, every constant `literal`, the
pointer `add`/`cast`, the `sub` remaining-AVL, the accumulator
`variable`/`assign`/`load`) is a node in the IR graph. The provenance verbatims
are comment lines only. The emitted C function signature is the faithful
`double tcrv_emitc_…(size_t n, float* y, const float* x, float max)`.

## Byte-exact HW evidence (ssh rvv) — `board_validation_stdout.txt`

The UNMODIFIED compiler-emitted `soft_max_kernel.cpp` compiled `-O2
-march=rv64gcv` and run vs a VERBATIM transcription of ggml's BARE vectorized
`ggml_vec_soft_max_f32` (vec.cpp:584-592 + `ggml_v_expf_m2` vec.h:1324-1360), the
deployment oracle:

```
INC-18 F5b ggml_vec_soft_max_f32: y[] 4601/4601 bit-exact, sum 4601/4601 bit-exact (vs ggml's VECTORIZED soft_max)
NC1 wrong-exp-method (libm expf): 711/768 correctly DIFFER
NC2 wrong-sum-precision (f32 accumulate, not f64 vfwredusum): 859/864 rows the SUM correctly DIFFERS
DISCRIMINATION: NC1 exp-polynomial control SHARP (711/768 differ); NC2 sum-precision control caught 859/864 rows (the f64 widening reduce is the discriminator)
RESULT: PASS (byte-exact y[] AND returned sum vs ggml VECTORIZED soft_max; NC1 discriminates the exact exp polynomial, NC2 the exact f64 widening reduce -> ggml's method matched node-for-node)
```

- **EXACT, not near-exact.** Both outputs are BIT-IDENTICAL (memcmp of the float
  bits for y[], memcmp of the double bits for the returned sum) over 24 sizes
  (incl. attention dims 256/512/1024/2048/4096 + strip-boundary tails) × 8 value
  distributions × 24 seeds + a named 32-element edge row = **4601/4601 on BOTH
  y[] and the sum**. The distributions include large/small/uniform (all-equal)/
  spiky (one huge lane) and — critically for attention — rows with **-inf masked
  entries** (~30% of lanes -inf). The `max` passed to both kernel and oracle is
  ggml's own scalar `ggml_vec_max_f32` fold (vec.h:1541), so -inf masked rows
  behave exactly like real attention (`max` ignores -inf via `MAX(-INF,x)`; the
  masked lane's `exp(-inf - max)` flows through the polynomial's |n|>192
  underflow path to exactly 0.0f on both sides).
- **Why EXACT** (the two cruxes both pinned by negative controls):
  - **NC1** (711/768 differ): the SAME soft_max computed via libm `expf` (a
    different exp than ggml's minimax polynomial) diverges from ggml's vectorized
    soft_max. So the y[]/sum byte-compare DISCRIMINATES ggml's exact polynomial
    from a merely-close exp — we matched ggml's METHOD (the minimax chain
    node-for-node, reused verbatim from F5), not a tolerance. (~7% agree on short
    rows where the polynomial and libm happen to round identically in every lane;
    majority-differ gate holds.)
  - **NC2** (859/864 differ): ggml's EXACT y[] but the sum accumulated in **f32**
    instead of the f64 widening reduce. The f32 accumulation loses low bits and
    diverges from ggml's f64 sum on long rows. So the **returned-sum byte-match
    pins ggml's EXACT f64 widening reduce** (`vfwredusum_vs_f32m2_f64m1` carried
    across strips), not merely "a sum of the same y[]" — the new structural
    element is verified, separate from the polynomial.
- The negative controls are DISCRIMINATION EVIDENCE; the kernel verdict gates on
  `y[]==total && sum==total`, not on a control being sharp (so a correct kernel
  cannot read FAIL if a control happens dull on the sampled rows).

## The hardest part + how resolved

**The returned f64 sum's byte-exactness** (the genuinely-new risk; F5 already
proved the y[] polynomial). A naive impl would (a) call libm `expf` — caught by
NC1, or (b) sum in f32 / a scalar fold — caught by NC2. The resolution is to
replicate ggml's EXACT method: the f64m1 widening-reduce accumulator
(`vfwredusum_vs_f32m2_f64m1`) carried across strips as a loop-carried
opaque-vector lvalue (since `emitc.for` has no iter_args), seeded
`vfmv_v_f_f64m1(0, 1)`, extracted `(double)vfmv_f_s_f64m1` for the return. Because
the per-strip `val` is bit-identical to ggml's (F5-proven) and the reduce
instruction + accumulator chain are identical, the returned double is bit-exact
— confirmed 4601/4601.

## Build / lit

- Full clean rebuild GREEN.
- `check-tianchenrv`: 624/627 pass; the 3 failures are the SAME pre-existing
  DOCUMENTED reds (`rvv-generated-bundle-abi-e2e-self-test` + the two
  `computed-masked-strided-input-widening-dot-reduce-add` dry-run tests) — none
  reference soft_max; the F5b change is purely ADDITIVE (the new soft_max lit
  test is the +1 green: 627 = 626 prior + 1).
- **Additivity proven**: F5 silu + F3 rms_norm re-render BYTE-IDENTICAL to their
  committed artifacts (the shared exp-helper extract is a no-op for silu's bits);
  F1 scale + the q4_0 block-dot quant-dot emitc lit PASS. The shared function-type
  result-type edit (line ~835) is guarded on the soft_max body, so every other
  kernel builds a void function byte-identically.

## Files

- include/TianChenRV/Dialect/RVV/IR/RVVOps.td — `GgmlVecSoftMaxF32Op` def.
- lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp — `GgmlVecSoftMaxF32Op::verify()`.
- lib/Conversion/RVV/RVVToEmitC.cpp — `emitGgmlVExpfM2` shared exp helper
  (factored out of silu), `isGgmlVecSoftMaxF32Body` recognizer, the soft_max-only
  `double` result-type guard, dispatch wiring, `emitGgmlVecSoftMaxF32` emitter
  (the f64 widening-reduce accumulator + shared exp).
- test/Conversion/RVV/rvv-to-emitc-ggml-vec-soft-max-f32.mlir — the F5b lit test.
- artifacts/inc18-forward-pass-f5b/{rvv-to-emitc-ggml-vec-soft-max-f32.mlir,
  soft_max_kernel.cpp, inc18_validate.cpp, board_validation_stdout.txt, RESULTS.md}.

## What's left (the forward-pass family)

With F1 (scale) + F3 (rms_norm) + F5 (silu) + F5b (soft_max) landed byte-exact,
the remaining ops to complete the forward-pass op set are:
- **F4** activation quantizers (`quantize_row_q8_0`/`q8_1`): reduction
  (`vfredmax`/`vfredusum`) + scale + f32→i16→i8 narrowing-convert — closes the
  f32→quant-dot bridge so a forward pass feeds our dot kernels.
- **F6** `rope`: f32 elementwise rotation (reuses F1/F2 lanes) + a verbatim
  scalar libm `cosf`/`sinf` angle cache (a DIFFERENT exactness axis — libm-linked,
  not a vectorized polynomial). The COMPOSITION rung.
Then the forward-pass op set is complete.
