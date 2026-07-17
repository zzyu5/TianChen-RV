# Forward-pass non-dot op family — scoping (INC-15, deliverable A)

Primary source, 2026-06-16. All line references are into the shallow clone at
`/home/kingdom/phdworks/llama.cpp` (ggml @ this checkout). Board = `ssh rvv`:
VLEN=128, `rv64gcv+zvfh`, clang 18.1.3, `__riscv_v_intrinsic == 12000`,
`__riscv_v_min_vlen == 128` (so `Zvl128b` ⇒ VLEN ≥ 128 mandated, same as the
quant-dot work).

## Why this is a NEW op family (not the quant-dot family we already cover)

Everything we cover today (`q4_0`/`q8_0`/`q4_1`/`q6_K` block-dot + a q4_0 GEMM
tile) is **block-quantized integer dot-product**: AoS block loop, packed-int
load, `vwmul`/`vwmacc`/`vwredsum` integer widening chain, per-block fp16 scale,
single fp32 `*s` output. The forward pass needs an entirely different shape:

- **dtype** f32 throughout (not packed int8/int4), loaded with `vle32`.
- **per-lane f32 arithmetic** `vfmul`/`vfadd`/`vfmacc`/`vfdiv` (not integer
  widening).
- **f32 reductions** `vfredusum`/`vfredmax` into a scalar (rms_norm, soft_max),
  and `ggml` accumulates those in **`ggml_float` = double** (a precision
  subtlety the integer dot never had).
- **transcendentals** `exp`/`sigmoid` via a hand-written **vectorized minimax
  polynomial** (`ggml_v_expf_m2`), the genuinely hard tail.
- **memory shape** flat unit-stride buffers (`y[i] *= v`, in some cases the
  same buffer read-and-written in place) — not AoS strided blocks.

Proving the compiler emits *these* structured, byte-exact is what shows it is a
**general** capability-driven f32 vector compiler, not a quant-dot specialist.

## The op set a llama-2-7b forward pass actually invokes

Mapped from `ggml/src/ggml-cpu/{vec.h,vec.cpp,ops.cpp}`. For each: the exact
math, where the RVV path lives, and which sub-family it belongs to.

### Sub-family 1 — f32 elementwise (no reduction, no transcendental)

| op | math | RVV path | hand-written `__riscv_v`? |
|---|---|---|---|
| `ggml_vec_scale_f32(n,y,v)` | `y[i] *= v` | vec.h:703 | YES — `vsetvl_e32m8` strip, `vle32`/`vfmul_vf`/`vse32` (vec.h:733-739) |
| `ggml_vec_mad_f32(n,y,x,v)` | `y[i] += v*x[i]` | vec.h:319 | YES — `vfmadd_vf_f32m8(ax,v,ay)` (vec.h:400-407) — **FMA** |
| `ggml_vec_add_f32(n,z,x,y)` | `z[i]=x[i]+y[i]` | vec.h:89 | macro layer only (`GGML_SIMD`); no RVV branch → scalar tail on the board |
| `ggml_vec_mul_f32` | `z[i]=x[i]*y[i]` | vec.h:127 | scalar one-liner (no SIMD) |
| `ggml_vec_sub_f32`/`div_f32` | `±`,`/` | vec.h:112/133 | scalar one-liner |

Key fact: **`scale` and `mad` are the only two with a real RVV intrinsic path**
in vec.h; both anchor at `m8`, both use a `vsetvl_e32m8(n-i)` VLEN-robust strip.
`scale` is a bare multiply (no FMA); `mad` uses `vfmadd` (an FMA → fp-contract
matters, exactly the issue solved on q4_0 with the emitc.expression FMA fix).

### Sub-family 2 — f32 reduction (scalar/double accumulate)

| op | math | source | RVV path |
|---|---|---|---|
| `rms_norm` | `mean=Σx²/n; scale=1/√(mean+eps); y[i]=x[i]*scale*(w[i]?)` | ops.cpp:3758 (`ggml_compute_forward_rms_norm_f32`) | **NO vector path** — `Σx²` is a scalar `ggml_float`(double) loop (ops.cpp:3791-3795, marked "worth switching to explicit SIMD?"); then scalar `1/sqrtf` and a `ggml_vec_scale_f32` to apply `scale` |
| `ggml_vec_sum_f32` | `Σx[i]` | vec.h | `ggml_float`(double) scalar accumulate |
| `ggml_vec_max_f32` | `max x[i]` | vec.h | used as the soft_max `max` argument |

rms_norm is structurally **reduction → scalar rsqrt → elementwise scale**. Its
Σx² is *double*-accumulated and *scalar* in ggml — so a byte-exact RVV reduction
must reproduce ggml's exact (scalar, ascending, double) fold, OR we match ggml's
*result* only to its own scalar tolerance. This is the first place where "match
ggml's method" forces a decision (see Hardest-part below).

### Sub-family 3 — transcendental (exp / sigmoid) — the hard tail

| op | math | source | RVV path |
|---|---|---|---|
| `ggml_vec_silu_f32(n,y,x)` | `y[i]=x[i]·σ(x[i])`, `σ(x)=1/(1+e^{-x})` | vec.cpp:380 | YES — `vsetvl_e32m2` strip, `ggml_v_silu_m2` (vec.h:1363) |
| `ggml_vec_soft_max_f32(n,y,x,max)` | `y[i]=e^{x[i]-max}`, returns `Σy` | vec.cpp:531 | YES — `vsetvl_e32m2` strip, `ggml_v_expf_m2`, `vfredusum` into a **double** sum |
| `ggml_v_expf_m2(x,vl)` | minimax `e^x` | vec.h:1324 | YES — **fully vectorized polynomial**, NO libm call |

`ggml_v_silu_m2` = `neg → ggml_v_expf_m2 → +1 → vfdiv` (vec.h:1363-1368).
`ggml_v_expf_m2` is a 14-instruction vectorized minimax (Cephes-style): a
`0x1.8p23f` round trick, a 2-term Cayley reduction (`vfnmsac` ×2), a degree-?
polynomial in `vfmacc`, an integer exponent `vsll<<23 + 0x3f800000`
reinterpret, and a masked overflow/underflow fixup (`vcpop_m`, `vmerge`). **It
calls no libm function** — every step is an `__riscv_v` intrinsic (vec.h:1324-
1360). That is the decisive fact for byte-exactness (see below).

### Sub-family 3b — rope (rotary position embedding) — transcendental, but SCALAR-libm

| op | math | source | RVV path |
|---|---|---|---|
| `ggml_compute_forward_rope_f32` | per (pos, dim-pair i0): `x0' = x0·cosθ − x1·sinθ`, `x1' = x0·sinθ + x1·cosθ` | ops.cpp:`ggml_compute_forward_rope_f32` | the ROTATION is f32 elementwise (4 mul + 1 sub + 1 add per pair); NO `__riscv_v` branch -> scalar on the board |
| `rope_yarn` / `ggml_rope_cache_init` | `θ = freq_scale·θ_extrap` (+ yarn ramp); `cache[i0]=cosf(θ)·mscale`, `cache[i0+1]=sinf(θ)·mscale·sin_sign` | ops.cpp:5690-5720 | the angles use **scalar libm `cosf`/`sinf`** (ops.cpp:5704-5705), precomputed ONCE per position into a scalar `cache[]` |

**Primary-source finding (decisive for return-item 6):** rope's transcendental is
`cosf`/`sinf` from **libm, scalar**, evaluated once per (position, dim-pair) into a
precomputed `cache[]`; the per-element work that touches the vector is just the f32
elementwise rotation that READS the cache. So rope's transcendental byte-exactness
is a **different axis** from exp/sigmoid: exp is a *vectorized polynomial* we can
replicate bit-for-bit, but `cosf`/`sinf` are *libm calls* whose result depends on
the libm the kernel links against. A compiler-emitted rope must therefore either
(a) emit the angle setup as the SAME scalar `cosf`/`sinf(cache)` calls (so it links
against the SAME libm and matches bit-for-bit) and vectorize only the rotation, or
(b) accept near-exact (libm-tolerance) for the angle. (a) is clean and is the plan:
the rotation lanes are f32 elementwise (sub-family 1 machinery: vle32 / vfmul /
vfmsub / vfmadd), the cache stays a scalar `cosf`/`sinf` epilogue/prologue exactly
as ggml does. Rope is NOT a vectorized-transcendental problem — it is f32
elementwise rotation + a verbatim scalar libm angle cache.

### Sub-family 4 — the f32 → quant bridge (activation quantizers)

The dot kernels we already cover consume `block_q8_0`/`block_q8_1`/`block_q8_K`.
Those blocks are produced at runtime from the f32 activations by:

| op | source | RVV path | what it does |
|---|---|---|---|
| `quantize_row_q8_0(x,vy,k)` | riscv/quants.c:32 | YES — per 32-block: `vle32_m8`, `vfabs`, `vfredmax`(amax), `d=amax/127`, `vfmul_vf(id)`, `vfncvt_x_f_w_i16m4`, `vncvt_x_x_w_i8m2`, `vse8` | f32 → block_q8_0 |
| `quantize_row_q8_1(x,vy,k)` | riscv/quants.c:73 | YES — same + a `vfredusum` block sum stored as `block_q8_1.s` | f32 → block_q8_1 |
| `quantize_row_q8_K` | ggml-cpu-quants.c (generic) | macro/scalar | f32 → block_q8_K (K-quant) |

These are a clean **reduction (`vfredmax`/`vfredusum`) + scale + f32→int
narrowing-convert** family. They are the structural counterpart to our dequant
epilogue (which goes int→f32). They matter because without them the dot kernels
have no inputs in a real forward pass — but each is self-contained and
byte-exact-testable against ggml's own RVV quantizer.

## Recommended first target + increment plan

**F1 = `ggml_vec_scale_f32`** (IMPLEMENTED in INC-15-B). Rationale: it is the
*only* forward-pass op with **zero numeric ambiguity** — a bare per-lane fp32
multiply, so (a) no FMA ⇒ `-ffp-contract` cannot perturb it (the exact hazard
that needed the emitc.expression fix on q4_0), and (b) no cross-lane reduction
⇒ LMUL / tail / strip-count are all correctness-free. Byte-exactness is
therefore *unconditional* and independent of the LMUL knob. A failing
bit-compare can only mean a load/store/tail *bug*, never a rounding gap — the
cleanest possible foundation for the f32 strip-loop lowering machinery, which
F2–F5 inherit.

Increment order (each ssh-rvv byte-exact-validated, each inherits the prior
machinery):

- **F1 — `scale` (elementwise, bare multiply).** [DONE] Establishes the f32
  `vsetvl_e32m<L>` strip loop + `vle32`/`vfmul_vf`/`vse32` + in-place
  read-and-write buffer ABI. Anchored at `m8` to match ggml faithfully.
- **F2 — `mad` (+ `add`/`mul`).** Two-buffer elementwise. **This is where the
  FMA-contraction decision lands**: `mad` uses ggml's `vfmadd_vf` (a fused
  multiply-add). To be byte-exact we must reproduce ggml's RVV `vfmadd` path,
  not a `mul`+`add` pair (different rounding). Reuse the q4_0 emitc.expression
  FMA-fix discipline so `-ffp-contract` is pinned. `add`/`mul` are pure
  elementwise (reuse F1's strip with `vfadd_vv`/`vfmul_vv`, two input pointers).
- **F3 — `rms_norm` (first reduction).** Σx² reduction → scalar rsqrt → reuse F1
  `scale` to apply. Brings in `vfredusum` (the f32 cousin of the `vredsum` we
  already emit in the dot kernels) + a scalar `1/sqrtf`. The reduction-fold
  decision (match ggml's scalar-double Σ exactly vs a vectorized `vfredusum`
  with its own fold order) is made here, on a NON-transcendental op, before the
  hard tail.
- **F4 — the activation quantizers `quantize_row_q8_0`/`q8_1`.** Reduction
  (`vfredmax`/`vfredusum`) + scale + f32→i16→i8 narrowing-convert. Closes the
  f32→quant-dot bridge so a forward pass can actually feed our dot kernels.
- **F5 — `silu` + `soft_max` (vectorized-transcendental tail).** The
  `ggml_v_expf_m2` minimax polynomial as a structured intrinsic chain, then silu
  (`neg/exp/+1/div`) and soft_max (`sub/exp/vfredusum`). The hardest rung for the
  *vectorized* transcendental; deferred so F1–F4 prove the f32 machinery first.
- **F6 — `rope` (f32 rotation + scalar-libm sin/cos cache).** The per-pair rotation
  (`x0·cosθ ∓ x1·sinθ`) is f32 elementwise (reuses F1/F2 vfmul/vfmsub/vfmadd lanes);
  the angle cache stays a verbatim scalar `cosf`/`sinf` prologue exactly as ggml's
  `ggml_rope_cache_init`, so it links against the SAME libm and matches bit-for-bit.
  rope is NOT a vectorized-transcendental problem (its sin/cos are scalar libm, NOT
  a polynomial), so it is ordered last as the COMPOSITION rung — it reuses the most
  prior machinery, not because its transcendental is harder to make exact.

rms_norm (F3) intentionally precedes soft_max (F5) because soft_max is reduction
**and** transcendental *combined* — F3 isolates the reduction risk, F5 the
transcendental risk, so neither rung carries two new hazards at once. rope (F6) is
the composition rung (F1/F2 elementwise rotation + a scalar libm angle cache), not
a new vectorized-transcendental hazard.

## The single hardest part of the f32 forward-pass family

**The transcendental byte-exactness of `exp`/`sigmoid` (`ggml_v_expf_m2`).**
A naive implementation would call libm `expf` (a different polynomial than
ggml's), giving *near*-exact but not bit-identical results — and we could only
claim a tolerance, not byte-exactness.

**Primary-source finding that makes it tractable:** ggml's RVV `ggml_v_expf_m2`
(vec.h:1324-1360) is a **fully vectorized minimax polynomial built entirely from
`__riscv_v` intrinsics — it makes NO libm call.** Every step
(`vfmacc`/`vfnmsac`/`vfmv`/`vsll`/`vadd`/`vreinterpret`/`vmfgt`/`vmerge`/
`vcpop_m`) is a vector intrinsic with a fixed bit pattern constant. Therefore
the increment plan handles it by **replicating that exact intrinsic chain
node-for-node** as structured emitc `call_opaque` nodes (each intrinsic is one
sanctioned-opaque node, exactly as the dot kernels emit `vwmul` etc.), with the
identical magic constants (`0x1.8p23f`, `0x1.715476p+0f`, `0x1.62e4p-1f`,
`0x1.7f7d1cp-20f`, the polynomial coefficients, `0x3f800000`, the `126.0f`/
`192.0f`/`0x82000000` overflow fixup). Because the compiler emits the *same*
instructions ggml's source compiles to, the result is **bit-identical** — full
byte-exactness is achievable, not merely tolerance-bounded.

Three honest residual subtleties the plan must respect:
1. The overflow/underflow fixup branch (`if (!vcpop_m(c, vl))`) is data-
   dependent control flow inside the polynomial. The structured lowering keeps
   it as the SAME conditional (an emitc.if on the `vcpop` result), so the path
   taken matches ggml for every input — including the NaN/inf/denormal/large
   edge cases the task requires.
2. soft_max/rms_norm accumulate the reduction in `ggml_float` (= **double**).
   F3/F5 must fold in double (or prove the f32-vs-double difference is below
   ggml's own claimed tolerance) — a reduction-precision decision, separate from
   the polynomial. F1 (scale) has neither hazard, which is exactly why it is the
   chosen foundation.
3. **rope (F6) is a SEPARATE transcendental axis.** Its sin/cos are **scalar
   libm `cosf`/`sinf`** (not a vectorized polynomial like exp), evaluated once
   per position into a scalar `cache[]`. So rope's exactness story differs from
   exp: it is NOT "replicate the polynomial" but "emit the SAME scalar libm calls
   in the angle prologue (so the same libm is linked) and vectorize only the f32
   rotation that reads the cache." If the deployment links the same libm, rope is
   bit-exact; otherwise it is libm-tolerance bound on the angles only. This is the
   one forward-pass op whose byte-exactness depends on the LINKED libm, not on the
   emitted instructions — surfaced here so F6 plans the scalar-cache-verbatim
   approach (option a) rather than a polynomial replication that does not exist
   in ggml's rope path.
</content>
</invoke>
