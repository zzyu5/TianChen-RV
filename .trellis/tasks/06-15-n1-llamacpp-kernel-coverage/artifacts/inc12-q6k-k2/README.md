# INC-12 — q6_K super-block K2 (full block dot) — byte-exact ssh-rvv evidence

The second q6_K rung: the COMPLETE ggml `ggml_vec_dot_q6_K_q8_K`, emitted as
STRUCTURED emitc by our compiler, producing the fp32 `*s` output **byte-exact**
vs ggml's own `_generic` reference on real `ssh rvv`. K1 (commit `ae479367`)
proved the integer `aux32[8]` core; K2 adds the deferred two-level fp32 fold +
the ggml ABI for a deployable drop-in.

## What is proven (K2 scope)

The C **our compiler emits** for the new typed op
`tcrv_rvv.q6_k_q8_k_block_dot` reproduces the fp32 dot-product output `*s`
**byte-exact (exact IEEE-754 bit equality)** vs
`ggml_vec_dot_q6_K_q8_K_generic` (`llama.cpp/ggml/src/ggml-cpu/quants.c:800-853`)
on real `ssh rvv` (riscv64, clang 18.1.3, rv64gcv, VLEN=128), under
**`-ffp-contract=off`** (the regime where both reference and kernel are
unfused), over the full named n set {256, 512, 2048, 4096, 25600} + 6 named
edge cases + 2000 random n (NB∈1..32). **0 failures.** Negative control
**200/200 discriminating.**

## The fp32 fold (the K2 byte-exactness risk) — exactly matching `_generic`

Per super-block, after the K1 integer core produces `aux32[8]`:
- `d = GGML_FP16_TO_FP32(x[i].d) * y[i].d` — the fp16 weight scale (byte 208,
  read via the `(float)*(const _Float16 *)` seam) times the fp32 activation
  scale (byte 0, a plain `const float` load).
- `sums[l] += d * aux32[l]` — 8 INDEPENDENT fp32 lane-accumulators, emitted as
  `vfcvt` (i32→f32, RNE = C's `(float)aux32[l]`) then a **SEPARATE** `vfmul`
  (by `d`) then a **SEPARATE** `vfadd` into `sums`. **NEVER** a fused
  `vfmacc`/`vfmadd` — that diverges from `_generic`'s separate scalar mul/add
  under `-ffp-contract=off` (the INC-2a FMA class). The 8 lanes are independent,
  so vectorizing the fold body is byte-safe.
- `sums` (a `vfloat32m2`) is declared+zeroed **ONCE** outside the super-block
  loop (carried across super-blocks; unlike `aux32`, which resets each block).

The final horizontal sum is **SEQUENTIAL**, ascending l=0..7: `sums` is
`vse32`-stored into a `float[8]` scratch, then summed by `float sumf = 0.0f;
sumf += sums[0]; ...; sumf += sums[7]` — 8 scalar `emitc.add`, mirroring
`_generic`'s `float sumf = 0; for(l) sumf += sums[l]` INCLUDING the leading
`0.0f +` (so byte-exactness holds even at the `-0.0` edge). **NEVER** a vector
`vfredusum` (fp add is non-associative). Lane l ↔ index l is preserved by the
`vse32` store and pinned by the K1 `aux32[l] ↔ l` ordering.

## raw() = 0 / STRUCTURED proof

`grep -c 'raw(' tcrv_emitted_kernel.cpp` → **0**. Every value is a typed emitc
node (the `riscv_vector.h` intrinsics go through `emitc.call_opaque`, the one
sanctioned opaque seam, exactly as the q4_0/q8_0/q4_1/K1 siblings). The
sequential fp32 horizontal sum is 8 structured `emitc.add` ops over `sums8[l]`
loads (`v9[0]..v9[7]` in the emitted C), **not** a vector reduce — verify:

    grep 'vfredu\|vfredo' tcrv_emitted_kernel.cpp   # -> (none)
    grep 'vfmacc\|vfmadd' tcrv_emitted_kernel.cpp   # -> (none)

## How it reuses K1

K1's super-block integer core (the 6-bit ql+qh unpack into `aux8[256]` + the
nested sub-block int8-scaled i32 accumulation into `aux32`) was factored into a
shared helper `emitQ6_KSuperBlockAux32Core` (returns the per-super-block aux32
lvalue). K1 (`emitQ6_KQ8_KAux32Partial`) calls it then `vse32`-stores aux32;
K2 (`emitQ6_KQ8_KBlockDot`) calls it then folds aux32 into the fp32 `sums`.
This is a **pure extraction**: K1's emitted C is **byte-identical** to its
committed `inc11-q6k-k1/tcrv_emitted_kernel.cpp` (verified by diff, modulo
comments), and the K1 conversion + dialect lit tests stay green.

## The `=fast` reference-fusion artifact (honest)

Under `-ffp-contract=fast`, the **reference** `_generic` fuses
`sums[l] += d*aux32[l]` to fma (and reorders integer-domain scalars), so it
diverges from our contraction-immune kernel by 1–few ULP (~half the random
trials). This is a **reference-compilation artifact, not a kernel bug**:
`ref_band.cpp` shows `_generic(=off)` vs `_generic(=fast)` themselves diverge
**1013/2000** (max |ulp diff| 2048) — the same ~half rate — so the reference is
the moving part. Our kernel matches `_generic(=off)` exactly and is the
fusion-stable implementation. Do NOT fuse the kernel to chase `=fast` parity
(that re-breaks `=off`). See `ssh_rvv_stdout.txt` section (2)/(3).

## Deployable drop-in (ggml ABI)

`tcrv_q6_k_q8_k.cpp` DEFINES a REAL external `extern "C" void
ggml_vec_dot_q6_K_q8_K(int n, float* s, size_t bs, const void* vx, size_t bx,
const void* vy, size_t by, int nrc)` — the EXACT 8-arg ggml vec_dot symbol — as a
thin ABI bridge (drops the unused bs/bx/by/nrc, casts the block pointers, widens
`int n`→size_t) to our 4-arg emitted kernel, mirroring inc3's `tcrv_q4_integ.h`
delegation role. Because the bridge is a non-inline function in its own TU, it
emits a genuine link-overriding symbol (NOT a header-local inline copy): linked
alongside `tcrv_emitted_kernel.cpp`, callers' `ggml_vec_dot_q6_K_q8_K` resolves
to our kernel. The emitted kernel carries the 4 live q6_K ABI operands
(vx/vy/s/n); ggml's trailing stride/count args are unused by the q6_K `_generic`
math. `inc12_abi_linkcompat.cpp` calls THROUGH that 8-arg entry (dummy
bs/bx/by/nrc) → **bit-exact vs `_generic`, 1005/1005**; `nm` confirms a real
`T ggml_vec_dot_q6_K_q8_K` (defined text symbol) in the linked binary
(`ssh_rvv_stdout.txt` section (4)).

## Files

- `tcrv_emitted_kernel.cpp` — the UNMODIFIED kernel C our compiler emits.
- `inc12_validate.cpp` — byte-exact harness (oracle = VERBATIM `_generic`, fp16
  shimmed to an exact `_Float16` cast; full n set + edges + 2000 random + a
  perturbed negative control; bit-equality on `*s`).
- `tcrv_q6_k_q8_k.cpp` + `tcrv_q6_k_q8_k.h` — the deployable 8-arg ggml ABI
  drop-in (a real external link-overriding `ggml_vec_dot_q6_K_q8_K`).
- `inc12_abi_linkcompat.cpp` — the link-compat test (calls through the 8-arg
  entry; built as 3 separate TUs: kernel + bridge + test).
- `ref_band.cpp` — the `=off`-vs-`=fast` reference-fusion band characterization.
- `ssh_rvv_stdout.txt` — the raw board stdout (all four runs + the nm proof).

## Residuals (honest)

- **VLEN ≥ 128 pinned** (inherited from K1): the unpack uses `vsetvl_e8m2(32)`
  and the sub-block half uses `e8mf2(8)`/`e32m2`; no VLEN<128 re-strip. Correct
  on rv64gcv ⇒ Zvl128b; a zve32x/zve64x re-strip is a capability item.
- **Performance not yet measured.** K2 is a byte-exact correctness rung; the
  N3 measurement tuner (K3) inheriting q6_K's candidate shapes is the NEXT step
  (not built here).
- The fp16 `d` read mechanism `(float)*(const _Float16 *)` is board-proven
  (q4_0/q4_1 use it); K2 is the first q6_K rung to read a NONZERO fp16 `d`, and
  the `=off` match confirms it lands bit-exact (fp16→fp32 widening never rounds).

## Reproduce on the board

    scp tcrv_emitted_kernel.cpp inc12_validate.cpp tcrv_q6_k_q8_k.h \
        tcrv_q6_k_q8_k.cpp inc12_abi_linkcompat.cpp ref_band.cpp rvv:~/inc12_q6k_k2/
    ssh rvv 'cd ~/inc12_q6k_k2 && \
      clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off \
        tcrv_emitted_kernel.cpp inc12_validate.cpp -o v_off && ./v_off && \
      clang++ -O2 -march=rv64gcv -mabi=lp64d -ffp-contract=off \
        tcrv_emitted_kernel.cpp tcrv_q6_k_q8_k.cpp inc12_abi_linkcompat.cpp -o v_link && \
      nm v_link | grep ggml_vec_dot_q6_K_q8_K && ./v_link'

## Regenerate the emitted kernel C

    build/bin/tcrv-opt \
      test/Conversion/RVV/rvv-to-emitc-q6-k-q8-k-block-dot.mlir \
      --tcrv-rvv-lower-to-emitc \
      | /usr/lib/llvm-20/bin/mlir-translate --mlir-to-cpp
