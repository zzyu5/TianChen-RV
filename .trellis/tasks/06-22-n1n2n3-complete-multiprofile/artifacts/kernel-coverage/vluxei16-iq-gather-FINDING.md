# FINDING — vluxei16 IQ-gather revectorization, FIRST kernel (iq1_s)

Implements DESIGN step 2: convert iq1_s's SCALAR codebook gather to a hardware
`__riscv_vluxei16` indexed vector load — the same gather ggml uses. Evidence below
is from `ssh rvv` (Sophgo SG2044, RVV1.0, VLEN128, clang-18,
`-O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast`, `taskset -c 2`).

## 1. The change (file:lines)

`lib/Conversion/RVV/RVVToEmitCTernaryBinary.cpp :: emitIQ1SQ8KBlockDot`:
- Added wide emit types (`vuint16mf2_t`, `vint64m2_t`, `vint8m2_t`, `vint16m4_t`,
  `uint16_t*`, `const int64_t*`) in the type-decl block.
- Replaced the inner 4-group `l`-loop (the 4× scalar `vle8`/`vwmul i16m2`/
  `vwredsum` micro-vectors) with ONE 32-lane body per sub-block:
  - KEEP the 4 scalar 11-bit index computations `idx = qs[ib*4+l] |
    (((qh>>3l)&7)<<8)` byte-exact (untouched), each `<<3` and stored into a
    `uint16_t tmp[4]` stack array.
  - `__riscv_vle16_v_u16mf2(&tmp[0], 4)` → index vector (EMUL = (16/64)*2 = mf2).
  - `__riscv_vluxei16_v_i64m2((const int64_t*)tcrv_iq1s_grid, vidx, 4)` → gather
    4 grid u64 entries → `__riscv_vreinterpret_v_i64m2_i8m2` = 32 signed ternary
    grid bytes.
  - `__riscv_vle8_v_i8m2(q8Group, 32)` → `__riscv_vwmul_vv_i16m4` →
    `__riscv_vwredsum_vs_i16m4_i32m1(p, lacc, 32)` — ONE reduction per sub-block.
  - Removed now-dead `coreLmul/wideLmul/i8CoreType/i16WideType/groupLanes`, and
    the dead `(const int8_t*)` grid byte-view cast.
- KEPT UNCHANGED (byte-exact): `ls`, `delta`, `sumi += ls*lsum`,
  `sumi1 += ls*delta*(bsums...)`, `sumf += d*((float)sumi + 0.125f*(float)sumi1)`,
  `*s = sumf`. The grid table decl (`static const uint64_t tcrv_iq1s_grid[2048]`)
  is untouched.

Two emit-correctness traps avoided (advisor-flagged, both verified empirically):
1. Index EMUL: `vluxei16_v_i64m2` requires a `vuint16mf2_t` index, NOT m1
   (EMUL = (EEW_idx/SEW_data)*LMUL_data = (16/64)*2 = 1/2). Loaded via
   `vle16_v_u16mf2`.
2. NO captured `vsetvl` for the 32-lane body — `vsetvl_e8m1(32)` returns 16 at
   VLEN128 and would silently drop groups 2-3. Literal vls passed (4 to the
   gather, 32 to vle8/vwmul/vwredsum).

`tcrv-opt` rebuilt FORCED/CLEAN (touched the .cpp, rebuilt the target). No ODS/.td
touched. Emit + `mlir-translate-20 --mlir-to-cpp` clean (793-line `ours.cpp`, down
from 986 scalar). The emitted C was eyeballed locally (array decl, 4 stores,
`&tmp[0]` decay, gather chain) before hardware.

## 2. Byte-exact oracle — PASS (bit-identical)

The integer gather is order-free and the FP/delta fold is untouched, so the new
emit must produce output BIT-IDENTICAL to the prior scalar emit (NOT tolerance).
Oracle: both emits compiled into one binary with distinct symbols, run over the
harness test vectors (8 seeds × 6 sizes), result `memcmp`'d (raw float bits):

```
BYTE-EXACT-ORACLE cases=48 mismatches=0 -> PASS (bit-identical)
```

48/48 cases bit-identical. The revectorization is byte-exact-correct.
(Also: micro harness AGREEMENT vs ggml `max_rel_norm=0.000e+00` for both emits.)

## 3. objdump engagement proof — vluxei16 PRESENT (yes)

```
$ objdump -d ours_vlux.o | grep vluxei16     # our NEW kernel
  fc: vluxei16.v v16,(s9),v11
 158: vluxei16.v v22,(s9),v11
 ... (8 total, one per sub-block)
count = 8
$ objdump -d ours_scalar.o | grep vluxei16   # prior SCALAR kernel
count = 0
```

The emitted binary actually contains the hardware indexed gather (8×, one per
sub-block); the scalar baseline has none. Engagement proven, not assumed.

## 4. Micro ratio vs ggml — gap CLOSED 7.4x → 2.3x, NOT yet parity (HONEST)

`blockdot-bench/iq1_s` micro on `ssh rvv` (timing_n=32*QK_K, iters=2000, reps=200,
best-of-min, `taskset -c 2`), `ours.cpp` = the OUR emit, ggml = shipped `_vl128`:

| emit | ours (ns) | ggml (ns) | RATIO ggml/ours | gap (ggml faster by) |
|---|---|---|---|---|
| SCALAR (prior) | 25156.5 | 3415.1 | 0.136 | 7.4x |
| VLUX (new)     |  8973.9 | 3899.5 | 0.435 | 2.3x |

- Our kernel got **2.8x faster** (25156 → 8974 ns); the gap to ggml narrowed from
  **7.4x to 2.3x**.
- **This is NOT parity.** The advisor-locked expectation was a tie (~1.0x). We
  adopted ggml's exact hardware gather (`vluxei16`, objdump-confirmed) and closed
  most of the maturity gap, but ggml's `_vl128` iq1_s is still **2.3x faster**.
- Defensible claim: **"we now emit the same hardware indexed gather ggml uses
  (`vluxei16`, byte-exact-correct), collapsing the iq1_s scalar-gather maturity
  gap from 7.4x to 2.3x."** NOT "we beat ggml on iq1_s" and NOT "parity".

### Why still 2.3x (the residual, honest mechanism — NOT "we used vluxei16")
Our emit kept the conservative per-sub-block shape; ggml's `_vl128` is wider:
1. **Gather width.** ggml gathers 8 u64 entries = 64 i8 = TWO sub-blocks per
   `vluxei16_v_i64m4` and `vget`-splits, then `vwmul i16m8` over 128 lanes. We do
   `i64m2`/`i8m2`/`i16m4` over 32 lanes (ONE sub-block) — half ggml's gather LMUL
   and a quarter its product LMUL, so ~2x more gather/product/reduce issue.
2. **Scalar index construction.** We keep the 4 index derivations + 4 `tmp[]`
   stores on the SCALAR unit (the byte-exact-safe choice); ggml builds the index
   vector in-register (`0x0009000600030000` shift + `vrgather` + `vor vzext(qs)`,
   quants.c:2790-2799), so its index path is also vectorized.
The remaining 2.3x is gather-LMUL + the still-scalar index assembly, NOT the
gather instruction itself (that is now identical). Both are named, scoped
follow-ons (DESIGN step 2-wide / vectorized-index), not regressions.

## 4b. Conformance test updated (the emit's own lit test)

The emit's lit test `test/Conversion/RVV/rvv-to-emitc-iq1-s-q8-k-block-dot.mlir`
(`// RUN: tcrv-opt %s --tcrv-rvv-lower-to-emitc | FileCheck %s`) had CHECK lines
generated from the OLD scalar emit (`vsetvl_e8m1` / `vle8_v_i8m1` /
`vwmul_vv_i16m2` / `vwredsum_vs_i16m2` / the `(uint64->int8)` byte-view cast). Those
were updated to the new gather shape: `uint16_t tmp[4]` (`emitc.array<4x..uint16_t>`),
the `<<3` byte-offset + `(uint16_t)` cast + array `assign`, then
`vle16_v_u16mf2` / `(const uint64_t*->const int64_t*)` cast / `vluxei16_v_i64m2` /
`vreinterpret_v_i64m2_i8m2` / `vle8_v_i8m2` / `vwmul_vv_i16m4` /
`vwredsum_vs_i16m4_i32m1` / `vmv_x_s`. Result: that test now **PASSES**.

Test status (all on the dev host, post-change):
- `rvv-to-emitc-iq1-s-q8-k-block-dot.mlir` (emit): PASS.
- `Dialect/RVV/iq1-s-q8-k-block-dot-dataflow.mlir` (op pre-lowering): PASS
  (unaffected, as expected).
- `rvv-to-emitc-iq1-m-q8-k-block-dot.mlir` (sibling emit): PASS (untouched).
- Full `check-tianchenrv`: 703/706 pass. The 3 failures
  (`Scripts/rvv-generated-bundle-abi-e2e-*computed-masked-strided-input-widening-
  dot-reduce-add*` + the abi-e2e self-test, a `vlse16` strided-load op) are
  **PRE-EXISTING and unrelated** — verified by stashing this .cpp change, rebuilding
  tcrv-opt, and re-running those 3: they fail identically on clean source. They have
  zero reference to iq1_s/ternary/vluxei.

The committed bench source
`artifacts/kernel-coverage/blockdot-bench/iq1_s/ours.cpp` was regenerated to the
new emit (793 lines, 8× `vluxei16_v_i64m2`).

## 5. Scope / honesty notes
- Compute-side micro only. e2e decode is memory-bound (the memory wall) → this
  does NOT transport to an e2e decode win; NOT claimed (MEMORY:
  kernel-wins-dont-transplant-to-e2e).
- Byte-exact gate is the real correctness proof; the tolerance micro is not.
- No git commit (lead commits).

## Reproduce
- emit: `build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-iq1-s-q8-k-block-dot.mlir
  --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp` (dev host, forced
  clean tcrv-opt).
- oracle (rvv): `clang-18 -O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast
  byte_exact_oracle.cpp k_scalar.cpp k_vlux.cpp -o oracle && taskset -c 2 ./oracle`
  (k_scalar/k_vlux = the two emits with distinct symbol names).
- micro (rvv): `clang-18 -O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast -I.
  harness.cpp <emit>.cpp ggml_ref.cpp -o bench && taskset -c 2 ./bench`.
- objdump: `objdump -d ours_vlux.o | grep vluxei16`.
- Staged on rvv: `~/vlux-iq1s/`.
