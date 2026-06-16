# INC-20 F6 — ggml NORMAL `rope` (rotary position embedding, the LAST forward-pass primitive)

Date: 2026-06-16. Board: `ssh rvv` (riscv64, rv64gcv, VLEN=128, `__riscv_v_min_vlen
128`, `__riscv_zvl128b`, `__riscv_zfhmin`, `__riscv_zfh`, `__riscv_v_intrinsic 12000`,
clang 18.1.3).

## What landed

A STRUCTURED byte-exact compiler-emitted drop-in for ggml's NORMAL rope
(`ggml_compute_forward_rope_f32`, the `GGML_ROPE_TYPE_NORMAL` variant). F6 is the
COMPOSITION rung — the LAST forward-pass primitive. With it the forward-pass
primitive set (scale / rms_norm / silu / soft_max / quantize / rope) is COMPLETE.

**Variant chosen: NORMAL** (NOT NEOX), verified from primary source. llama.cpp
`llama_model_rope_type` maps `LLM_ARCH_LLAMA -> LLAMA_ROPE_TYPE_NORM`
(llama-model.cpp:2387-2423, comment "a normal RoPE, operating on pairs of
CONSECUTIVE head values"). So NORMAL *is* the rope llama-2 uses — matching the
mission ("the actual llama.cpp kernel"). NORMAL =
`rotate_pairs<float>(n_dims, n_offset=1, cache, src, dst, scale=1)`: `ic = i0/1 =
i0`, so x0 = src[i0], x1 = src[i0+1] (consecutive), cos = cache[i0], sin =
cache[i0+1].

- New op `tcrv_rvv.ggml_rope_norm_f32` (x `const float *` in, y `float *` out,
  theta_base `float`, theta_scale `float`, n_dims; fail-closed verifier I7; `kind`
  bounded to `"ggml_rope_norm_f32"`; no SEW/LMUL/policy attr at the I5 boundary;
  no `strip_lmul` knob — cos/sin are scalar libm so the loop is scalar, matching
  silu's no-knob precedent).
- Lowering (`emitGgmlRopeNormF32`, lib/Conversion/RVV/RVVToEmitC.cpp): a SINGLE
  scalar per-pair loop reproducing ggml's exact structure.

## The op + lowering (theta recurrence + cos/sin + rotation) — files & math

The plain path (the params llama-2 inference uses): `ext_factor=0` (skips the
yarn ramp + the `logf` mscale), `freq_scale=1`, `attn_factor=1`,
`freq_factors=NULL`, forward so `sin_sign=+1`, `n_dims=ne0` = the full head_dim
(no passthrough channel). The math then collapses to ggml's bare recurrence:

```
theta = theta_base;                              // = pos (the integer position) as f32
for p in 0 .. n_dims/2-1:                         // i0 = 2*p
  cos_p = cosf(theta);  sin_p = sinf(theta);      // SCALAR libm (one call/pair)
  x0 = x[2p];  x1 = x[2p+1];                       // CONSECUTIVE pair (NORMAL)
  y[2p]   = x0*cos_p - x1*sin_p;
  y[2p+1] = x0*sin_p + x1*cos_p;
  theta *= theta_scale;                           // ITERATIVE f32 recurrence
```

- **theta recurrence** (the hardest part, axis 1): the ITERATIVE f32 `theta *=
  theta_scale` (ops.cpp:5711/5719), NOT a closed-form `theta = pos *
  theta_scale^p` and NOT double — reproducing ggml's exact accumulated f32
  rounding. Emitted as a loop-carried `emitc.variable` lvalue + `emitc.assign`
  (emitc.for has no iter_args, exactly the F3 rms_norm scalar-accumulator
  pattern). `theta_base` (= pos as f32) and `theta_scale` (= `powf(freq_base,
  -2/n_dims)`) are PRECOMPUTED runtime f32 INPUTS — hoisting the `powf` out, so
  the kernel makes NO `powf` call and theta_scale is bit-identical to ggml's.
- **cos/sin** (the hardest part, axis 2 — the libm axis): `cosf(theta)` /
  `sinf(theta)` via `emitc.call_opaque` (the sanctioned opaque seam — NOT a raw
  string, NOT a vectorized polynomial). This is a DIFFERENT byte-exactness axis
  from F5's exp (a vectorized minimax we replicate node-for-node): rope's
  transcendental is scalar libm, so exactness depends on linking the SAME libm.
- **rotation**: each output's `a*b - c*d` GROUPED into ONE `emitc.expression`
  (the F3 rms_norm / INC-2a emitc.expression FMA-fix discipline), so
  mlir-translate renders ONE C statement (`y[2p] = x0*cos - x1*sin;`)
  TOKEN-IDENTICAL to ggml's source (ops.cpp:5808-5809). The loads (`x[2p]`,
  `x[2p+1]`) happen into SSA temps BEFORE any store, so the in-place `x==y`
  aliasing ggml allows is safe.

Files:
- include/TianChenRV/Dialect/RVV/IR/RVVOps.td — `GgmlRopeNormF32Op` def.
- lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp — `GgmlRopeNormF32Op::verify()`
  (fail-closed: kind, the `const float *` x / `float *` y / `float` theta_base /
  `float` theta_scale ABI, the m1 result, the with_vl policy).
- lib/Conversion/RVV/RVVToEmitC.cpp — `isGgmlRopeNormF32Body` recognizer, the
  dispatch wiring, `emitGgmlRopeNormF32` emitter.
- test/Conversion/RVV/rvv-to-emitc-ggml-rope-norm-f32.mlir — the F6 lit test.
- NO `RVVDialect.cpp` ABI-allowlist change needed (all ctypes `const float *` /
  `float *` / `float` already exist from F1/F3).

## raw() = 0 + structured proof

The whole kernel is STRUCTURED emitc nodes — the per-pair loop is `emitc.for`,
the angle recurrence theta is an `emitc.variable` lvalue + `emitc.assign`, cos/sin
are `emitc.call_opaque`, the consecutive-pair loads are `cast`→`subscript`→`load`,
each rotation output is ONE `emitc.expression` (two `emitc.mul` + a `emitc.sub`/
`emitc.add` yielded) assigned to a `subscript` lvalue. Of the lowered IR:
**`emitc.raw` / `raw(` = 0**, and the non-comment `emitc.verbatim` count = **0**
(every verbatim is a `// `-prefixed provenance comment line). Every value is a
node in the IR graph. The committed `ggml_rope_norm_f32_kernel.cpp` has `raw( = 0`.

## Byte-exact HW evidence (ssh rvv) — `board_validation_stdout.txt`

The UNMODIFIED compiler-emitted `ggml_rope_norm_f32_kernel.cpp` compiled `-O2
-march=rv64gcv` and run vs a VERBATIM transcription of ggml's NORMAL rope
(`ggml_rope_cache_init` + `rotate_pairs<float>` n_offset=1 scale=1), the deployment
oracle, over head_dim=128 × {pos 0,1,2,7,31,128,511,2047,4095,8191,32767} × 6 value
distributions = 66 rows. **Built once PER `-ffp-contract` flag** (so the reference
exercises ggml's REAL fused behavior at that flag), all four modes:

```
-ffp-contract=default : 66/66 rows f32 BIT-EXACT ;  NC-neox 60/66 DIFFER (SHARP) ;  RESULT PASS
-ffp-contract=on      : 66/66 rows f32 BIT-EXACT ;  NC-neox 60/66 DIFFER (SHARP) ;  RESULT PASS
-ffp-contract=off     : 66/66 rows f32 BIT-EXACT ;  NC-neox 60/66 DIFFER (SHARP) ;  RESULT PASS
-ffp-contract=fast    : 66/66 rows f32 BIT-EXACT ;  NC-neox 60/66 DIFFER (SHARP) ;  RESULT PASS
```

- **EXACT (with same-libm), under EVERY contraction mode.** The f32 output
  (memcmp of all 128 lanes × 66 rows) is BIT-IDENTICAL to ggml's NORMAL rope.
  This is the q4_0 bar (INC-2a achieved `default/on/off/fast`): because each
  rotation is ONE C statement token-identical to ggml's single source expression,
  clang makes the IDENTICAL contraction decision under every flag — so the kernel
  is byte-exact **independent of llama.cpp's build flag** (and llama.cpp sets no
  `-ffp-contract`/`-ffast-math`/`-Ofast` override, so its default `on` is the live
  config — covered).
- **The honest libm dependency.** cos/sin are scalar libm. The validation links
  the SAME libm (the board's) for BOTH our kernel and the ggml reference in one
  TU, so the angle cache is bit-identical and the comparison is apples-to-apples
  EXACT — not a tolerance. A deployment that links a DIFFERENT libm would be
  libm-tolerance bound on the angles ONLY (the rotation itself is exact f32 once
  cos/sin agree). This is the ONE forward-pass op whose byte-exactness depends on
  the LINKED libm, not on the emitted instructions — stated honestly, exactly as
  research/forward-pass-scoping.md (F6) flagged. theta_scale is precomputed on the
  host and passed identically to both, so there is no `powf` divergence either.

## Negative controls (discrimination — `board_validation_stdout.txt`)

- **NC-neox** (the split-half NEOX pairing `x0=x[i], x1=x[i+n_dims/2]` instead of
  NORMAL's consecutive pair) is the FLAG-ROBUST discriminator: **60/66 rows
  correctly DIFFER under every contraction mode** (the 6 non-differing are pos=0,
  where theta=0 ⇒ cos=1,sin=0 ⇒ the rotation is identity regardless of pairing).
  This proves the f32 byte-compare DISCRIMINATES the pairing/variant — matching
  ggml's NORMAL consecutive indexing is load-bearing, not vacuous.
- **NC-fma** (the rotation FORCED unfused via `fmaf`, one rounding) is reported as
  CONTEXT, not the verdict gate: it differs from the ggml ref **only when the ref
  is itself unfused** (50/66 at `-ffp-contract=off`; 0/66 at on/fast/default where
  the ref fuses). That on/off split CONFIRMS the contraction mechanism — our
  grouped expression contracts exactly as ggml's source does at each flag.

The kernel verdict gates on `ours == ggml ref` over all rows AND the flag-robust
NC-neox being sharp, at every contraction mode.

## lit + reds

- Full clean rebuild GREEN.
- `check-tianchenrv`: **629 total, 626 pass, 3 fail** — the 3 failures are the
  SAME pre-existing DOCUMENTED reds (`rvv-generated-bundle-abi-e2e-self-test` +
  the two `…computed-masked-strided-input-widening-dot-reduce-add…dry-run` tests);
  none reference rope. The F6 change is purely ADDITIVE (the new F6 lit test is
  the +1 green: 629 = 628 prior + 1).
- **Sibling additivity**: F1 scale / F3 rms_norm / F5 silu / F5b soft_max / F4
  quantize lit tests all re-render and PASS unchanged; the q4_0/q8_0/q4_1/q6_K
  block-dot + the quant-dot siblings are byte-untouched (my change adds a new op
  def + verifier + recognizer/dispatch/emitter + lit test — it touches NO sibling
  emission path, and needed NO `RVVDialect.cpp` ABI-allowlist change).

## The hardest part (the theta recurrence + the libm cos/sin axis) + how resolved

1. **The theta recurrence.** ggml accumulates `theta *= theta_scale` ITERATIVELY
   in f32 (not closed-form, not double), so each pair's theta carries the
   accumulated f32 rounding of all prior multiplies. RESOLUTION: emit the exact
   iterative recurrence (a loop-carried emitc.variable theta + per-pair `theta =
   theta * theta_scale`), and HOIST `theta_scale = powf(freq_base, -2/n_dims)` to
   a precomputed runtime input (so theta_scale is bit-identical to ggml's and the
   kernel makes no powf call). The 66/66 bit-exactness at large pos (32767, where
   theta has accumulated 63 multiplies) confirms the recurrence matches.
2. **The libm cos/sin axis.** Unlike F5's exp (a vectorized polynomial replicated
   node-for-node ⇒ bit-exact from the instructions), rope's cos/sin are SCALAR
   libm whose result depends on the LINKED libm. RESOLUTION (option (a) from the
   scoping research): emit the angle prologue as the SAME scalar `cosf`/`sinf`
   call_opaque nodes ggml uses, link the same libm for the comparison ⇒ bit-exact;
   surface the libm dependency honestly (different libm ⇒ tolerance on the angles
   only). The rotation is exact f32 once cos/sin agree.
3. **The FP-contraction subtlety (caught by measurement, fixed).** A first cut
   emitted the two products as SEPARATE statements and was byte-exact only at
   `-ffp-contract=off` (16/66 at the default `on`, because ggml's single-expression
   rotation fuses under `on` while separate statements don't). RESOLUTION: GROUP
   each `a*b - c*d` into ONE emitc.expression (token-identical to ggml's source) ⇒
   identical contraction under every flag ⇒ 66/66 under default/on/off/fast (the
   q4_0 bar), independent of the build flag.

## The forward-pass primitive set is now COMPLETE

With F6 (rope) landed byte-exact, the forward-pass primitive set our compiler
emits byte-exact (STRUCTURED, raw()=0, ssh-rvv-validated) is:

| F | op | kind | shape |
|---|---|---|---|
| F1 | `ggml_vec_scale_f32` | elementwise | y[i] *= v (bare f32 multiply) |
| F3 | `ggml_rms_norm_f32` | reduction | Σx² scalar-double → 1/√ → scale |
| F5 | `ggml_vec_silu_f32` | vec-transcendental | x·σ(x), exact `ggml_v_expf_m2` poly |
| F5b | `ggml_vec_soft_max_f32` | transcend.+reduce | e^{x-max}, f64 widening reduce sum |
| F4 | `quantize_row_q8_0` | f32→quant bridge | amax reduce → scale → f32→i16→i8 |
| F6 | `ggml_rope_norm_f32` | composition | iterative θ + scalar-libm cos/sin + f32 rotation |

**What a FULL forward pass still needs (out of the primitive set, all trivial /
small increments):**
- **the trivial elementwise residuals**: `ggml_vec_add_f32` (z[i]=x[i]+y[i], the
  residual add) and `ggml_vec_mul_f32` (the FFN gate elementwise mul) — both are
  bare two-buffer elementwise (reuse F1's strip with `vfadd_vv`/`vfmul_vv`, two
  input pointers); `ggml_vec_mad_f32` (y[i]+=v·x[i], an FMA — reuse the
  emitc.expression FMA discipline F6 just exercised). These have a RVV intrinsic
  path only for scale/mad; add/mul are scalar one-liners in ggml — trivial,
  unconditionally byte-exact (no reduction, no transcendental, the contraction is
  handled by the F6/INC-2a expression discipline for mad).
- **the q8_1 quantizer** (`quantize_row_q8_1`): F4's amax/scale/narrow + an extra
  `vfredusum` block sum stored as `block_q8_1.s` (stride 36); a small increment
  INHERITING F4's reduction + narrowing-convert machinery, needed when the
  q4_1/q5_1 dots need their q8_1 activation.

The dot kernels (q4_0/q8_0/q4_1/q6_K block-dot + the q4_0 GEMM tile) are already
committed. So a full llama-2 f32 forward pass is: the F-family primitives above
(F1–F6 done) + add/mul/mad residuals + the q8_0/q8_1 quant bridges (F4 done, q8_1
pending) feeding the committed quant-dot kernels.

## Files

- include/TianChenRV/Dialect/RVV/IR/RVVOps.td
- lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp
- lib/Conversion/RVV/RVVToEmitC.cpp
- test/Conversion/RVV/rvv-to-emitc-ggml-rope-norm-f32.mlir
- artifacts/inc20-forward-pass-f6/{rvv-to-emitc-ggml-rope-norm-f32.mlir,
  ggml_rope_norm_f32_kernel.cpp, inc20_validate.cpp, board_validation_stdout.txt,
  RESULTS.md}.
