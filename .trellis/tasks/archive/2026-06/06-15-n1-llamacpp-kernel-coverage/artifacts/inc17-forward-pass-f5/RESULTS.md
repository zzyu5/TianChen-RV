# INC-17 F5 — ggml `ggml_vec_silu_f32` (the first forward-pass VECTORIZED TRANSCENDENTAL)

Date: 2026-06-16. Board: `ssh rvv` (riscv64, rv64gcv, VLEN=128, `__riscv_v_min_vlen
128`, `__riscv_zvl128b`, `__riscv_v_intrinsic 12000`, clang 18.1.3).

## What landed

A STRUCTURED byte-exact compiler-emitted drop-in for ggml's vectorized
`ggml_vec_silu_f32` (the FFN activation `y[i] = x[i]·sigmoid(x[i])`,
`sigmoid(x) = 1/(1+e^{-x})`). The crux — and the reason F5 is the hardest rung —
is that ggml computes `e^x` with a fully VECTORIZED MINIMAX POLYNOMIAL
(`ggml_v_expf_m2`, vec.h:1324-1360) built entirely from `__riscv_v` intrinsics
(NO libm `expf` call), so byte-exactness is achievable by replicating that
intrinsic chain node-for-node.

- New op `tcrv_rvv.ggml_vec_silu_f32` (x in, y out, n; fail-closed verifier I7;
  `kind` bounded to `"ggml_vec_silu_f32"`, no SEW/LMUL/policy attr at the I5
  boundary; pinned at m2 — no `strip_lmul` knob this cut).
- Lowering (`emitGgmlVecSiluF32`, lib/Conversion/RVV/RVVToEmitC.cpp): an m2 f32
  strip loop (`vsetvl_e32m2(n-i)` / `vle32` / … / `vse32`, the F1/F3 machinery)
  whose body is `ggml_v_silu_m2` = `vfneg → ggml_v_expf_m2 → vfadd 1.0f → vfdiv`,
  with `ggml_v_expf_m2` expanded node-for-node as STRUCTURED emitc `call_opaque`
  nodes (one per `__riscv_v` intrinsic, the one sanctioned opaque seam).

## The exp polynomial replicated (the constants/order from ggml, node-for-node)

`ggml_v_expf_m2(x)` (Cephes/ARM-limited-style minimax), every constant the exact
ggml hex-float bit pattern:
1. round trick: `r = vfmv_v_f(0x1.8p23f)`; `z = vfmacc_vf(r, 0x1.715476p+0f, x)`
   (× log2(e)); `n = vfsub_vv(z, r)` (the rounded integer part).
2. two-term Cayley range reduction:
   `b = vfnmsac_vf(vfnmsac_vf(x, 0x1.62e4p-1f, n), 0x1.7f7d1cp-20f, n)`.
3. integer exponent: `e = vsll_vx(vreinterpret_f32→u32(z), 23)`;
   `k = vreinterpret_u32→f32(vadd_vx(e, 0x3f800000))`  (≙ `2^n`).
4. degree-5 Estrin polynomial in `b` (coeffs `0x1.ffffecp-1f`, `0x1.fffdb6p-2f`,
   `0x1.555e66p-3f`, `0x1.573e2ep-5f`, `0x1.0e4020p-7f`) via `vfmul_vf`/`vfmacc`.
5. overflow/underflow fixup masks (`126.0f`, `192.0f`, `0x82000000`,
   `0x7f000000`) merged with `vmerge_vvm`/`vmerge_vxm`.
Then silu: `1 + exp(-x)` (`vfadd_vf … 1.0f`) and `x / (1+exp(-x))` (`vfdiv_vv`,
numerator = the ORIGINAL x, not neg_x).

## The hardest part + how resolved (the data-dependent branch)

ggml's `ggml_v_expf_m2` has `if (!__riscv_vcpop_m_b16(c, vl)) return fast; …slow`
— a DATA-DEPENDENT branch yielding a vector value. The resolution: **emit the
slow-path `vmerge` value graph UNCONDITIONALLY** (a straight-line chain, no
`emitc.if`, no `vcpop`). This is byte-exact because the `vcpop` is a PURE
PERFORMANCE short-circuit: the fast path returns `vfmacc_vv(k, j, k) = k + j*k`,
and the slow path's c-false / |n|≤192 lanes return `vfmacc_vv(k, k, j) = k + k*j`
— bit-identical (fp multiply commutes; RISC-V yields the canonical NaN
regardless of operand order). For any input where ggml takes the slow path, we
emit ggml's slow path verbatim. So same output bits for EVERY input — confirmed
empirically below, including every NaN/inf/denormal/saturation edge case. (This
corrected the research doc, which over-cautiously planned an `emitc.if`; only the
value graph matters, not the control flow.)

## raw() = 0 + structured proof

The exp polynomial is a STRUCTURED `call_opaque` chain, NOT a raw string. Of the
lowered IR: total `emitc.verbatim` that are NOT `// `-prefixed comment lines = 0;
exported `silu_kernel.cpp` has 0 raw/`__asm`/multi-statement-string lines. Every
value (the strip `for`, every intrinsic `call_opaque`, every constant `literal`,
the pointer `add`/`cast`, the `sub` remaining-AVL) is a node in the IR graph. The
provenance verbatims are comment lines only (`// tcrv_emitc.source_op=…`).

## Byte-exact HW evidence (ssh rvv) — `board_validation_stdout.txt`

The UNMODIFIED compiler-emitted `silu_kernel.cpp` (from `tcrv-opt
--tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp`, re-render
byte-IDENTICAL to the committed artifact) compiled `-O2 -march=rv64gcv` and run
vs a VERBATIM transcription of ggml's VECTORIZED `ggml_vec_silu_f32` /
`ggml_v_silu_m2` / `ggml_v_expf_m2` (the deployment oracle — bit-exact vs IT, not
the scalar libm sigmoidf):

```
INC-17 F5 ggml_vec_silu_f32: 2641/2641 bit-exact cases PASS (vs ggml's VECTORIZED silu)
NC wrong-exp-method (libm expf vs ggml minimax): 967/1056 correctly DIFFER
NC agreement breakdown by dist: d0(~0)=26 d1(moderate)=28 d2(saturation)=35; of all
89 agreeing rows, 0 are fully-saturated (every lane |x|>=104, exp not the discriminator)
RESULT: PASS (byte-exact vs ggml VECTORIZED silu; the libm-exp NC discriminates ->
ggml's exact minimax exp polynomial is matched node-for-node, not just its tolerance)
```

- **2641/2641 bit-exact** over 22 sizes (incl. llama-2-7b FFN dim 11008) × 5
  value distributions × 24 seeds + a 32-element named edge row: the polynomial-
  dense region [-2,2], the reduction range [-20,20], the SATURATION TAILS
  [-120,120] (large +ve → x, large -ve → 0, crossing the 88.38 +inf flush and the
  -103.97 zero flush), the OVERFLOW KNOTS [85,95]/[-100,-110] (the |n|>126/192
  `vmerge` fixup boundaries), raw-bit denormals/inf/NaN, and explicit
  ±inf/NaN/±0/denormal/FLT_MAX/±126/±192 edges. The saturation tails and the
  polynomial knots — the cases a naive impl gets wrong — are bit-exact.
- **Negative control 967/1056 correctly DIFFER**: the SAME silu function computed
  via libm `expf` (a different exp implementation than ggml's minimax) diverges
  from ggml's vectorized silu on the polynomial-active region. This proves the
  byte-compare DISCRIMINATES "matches ggml's exact polynomial" from "merely
  close": our kernel matches ggml's vectorized silu bit-for-bit while the
  libm-exp silu does not → we matched ggml's METHOD (the minimax polynomial
  node-for-node), not just the silu function to some tolerance. (Of the 89 NC
  cases that agree — instrumented, not guessed: d0=26 / d1=28 / d2=35, and 0 of
  them fully-saturated — agreement is NOT the saturated tails but small-n rows
  where ggml's polynomial and libm's expf happen to round to the same f32 bits in
  every lane; the whole-row memcmp passes by coincidence on short rows. The gate
  is majority-differ; 91.5% differ, so the discrimination holds regardless of why
  the minority agree.)

## Build / lit

- Full clean rebuild GREEN.
- `check-tianchenrv`: 623/626 pass; the 3 failures are the SAME pre-existing
  DOCUMENTED reds recorded in the F1/F3 RESULTS (`rvv-generated-bundle-abi-e2e-
  self-test` + the two `computed-masked-strided-input-widening-dot-reduce-add`
  dry-run tests) — none reference silu; the F5 change is purely ADDITIVE.
- F1 (`ggml_vec_scale_f32`) + F3 (`ggml_rms_norm_f32`) + the quant-dot siblings
  re-render byte-IDENTICAL (their lit tests are in the 623 green); the silu
  kernel re-renders byte-IDENTICAL (determinism verified).

## Files

- include/TianChenRV/Dialect/RVV/IR/RVVOps.td — `GgmlVecSiluF32Op` def.
- lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp — `GgmlVecSiluF32Op::verify()`.
- lib/Conversion/RVV/RVVToEmitC.cpp — `isGgmlVecSiluF32Body` recognizer, dispatch
  wiring, `emitGgmlVecSiluF32` emitter (the node-for-node exp polynomial).
- test/Conversion/RVV/rvv-to-emitc-ggml-vec-silu-f32.mlir — the F5 lit test.
- artifacts/inc17-forward-pass-f5/{rvv-to-emitc-ggml-vec-silu-f32.mlir,
  silu_kernel.cpp, inc17_validate.cpp, board_validation_stdout.txt, RESULTS.md}.

## What's left (the forward-pass family)

- **F4** activation quantizers (`quantize_row_q8_0`/`q8_1`): reduction
  (`vfredmax`/`vfredusum`) + scale + f32→i16→i8 narrowing-convert — closes the
  f32→quant-dot bridge so a forward pass feeds our dot kernels.
- **F5b** `soft_max` (`ggml_vec_soft_max_f32`): REUSES this exact exp polynomial
  + F3's scalar-double reduction (`y[i]=e^{x[i]-max}`, returns Σy in a double) —
  the combined reduction+transcendental rung, now both hazards are de-risked.
- **F6** `rope`: f32 elementwise rotation (reuses F1/F2 lanes) + a verbatim
  scalar libm `cosf`/`sinf` angle cache (a DIFFERENT exactness axis — libm-linked,
  not a vectorized polynomial).
