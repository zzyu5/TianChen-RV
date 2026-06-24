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

---

# FINDING — vluxei16 IQ-gather revectorization, SECOND kernel (iq1_m)

Extends the proven iq1_s vluxei16 pattern to iq1_m (same 2048-entry uint64
iq1s_grid, signed ternary bytes, NO sign plane). Same `ssh rvv` regime (SG2044,
RVV1.0, VLEN128, clang-18, `-O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast`,
`taskset -c 2`).

## 1. The change (file:lines)

`lib/Conversion/RVV/RVVToEmitCTernaryBinary.cpp :: emitIQ1MQ8KBlockDot`
(method ~:613–1400):
- Added gather emit types (`vuint16mf4_t`, `vint64m1_t`, the `uint16_t*` /
  `const int64_t*` ptr types; reused `vint8m1_t`/`vint16m2_t` for the half).
- iq1_m's grid dot MUST split into TWO halves (groups {0,1}→sum1[0] @ ls1,
  groups {2,3}→sum1[1] @ ls2) because the halves carry DIFFERENT per-half scales,
  so iq1_s's single per-sub-block reduce can't be copied literally. The byte-exact,
  shape-agnostic restructure (advisor-locked, option B over A):
  - KEEP the 4 scalar 11-bit index computations
    `idx = qs[ib*4+l] | ((qhSel<<sh)&0x700)` byte-exact (untouched), each `<<3`
    and stored into a `uint16_t idxoff[4]` stack array.
  - Per HALF h=0,1: `__riscv_vle16_v_u16mf4(&idxoff[2h], 2)` (index EMUL =
    (16/64)*1 = 1/4 → **mf4**, NOT mf2 — the lower-LMUL gotcha vs iq1_s) →
    `__riscv_vluxei16_v_i64m1((const int64_t*)tcrv_iq1m_grid, vidx, 2)` (gather 2
    grid u64) → `__riscv_vreinterpret_v_i64m1_i8m1` = 16 ternary i8 → `vle8` the 16
    q8 (`q8Base + ib*32 + h*16`) → `__riscv_vwmul_vv_i16m2` →
    `__riscv_vwredsum_vs_i16m2_i32m1` → `sum1[h]` ASSIGNED ONCE (not `+=`; the
    whole half is one shot). Byte-exact: a 16-lane half reduce = Σ(group 2h) +
    Σ(group 2h+1), integer-identical to the original two 8-lane reduces summed
    into sum1[h]. i64m1 = ALWAYS 2 u64 = 16 bytes regardless of VLEN, so NO
    VLEN128-only `vget`-split lane assumption (avoids the VLEN-SHAPE-MATCH trap).
- KEPT UNCHANGED (byte-exact): the per-GROUP sum2/delta path (it is grid-
  INDEPENDENT — the gather doesn't touch it): `vsetvl_e8m1`/`vle8_v_i8m1`/
  `vwredsum_vs_i8m1_i16m1` Σq8 + `delta[l]` per group into sum2[l/2]. Also UNCHANGED:
  the packed iq1m_scale fp16 reconstruct, `d`, ls1/ls2, the per-super-block fold
  `sumf += d*((float)sumi1 + 0.125f*(float)sumi2)`, `*s = sumf`. Removed the now-
  dead `(const int8_t*)` grid byte-view cast (replaced by the in-loop
  `(const int64_t*)` gather base).
- Lit test `test/Conversion/RVV/rvv-to-emitc-iq1-m-q8-k-block-dot.mlir` CHECK
  lines updated to the new gather shape (idxoff array, `<<3`+u16 cast, the
  `vle16_u16mf4`/`vluxei16_v_i64m1`/`vreinterpret_v_i64m1_i8m1` chain + the kept
  per-group `vwredsum_vs_i8m1_i16m1` delta path). PASSES.
- `tcrv-opt` rebuilt FORCED/CLEAN (touched the .cpp). No ODS/.td touched. Emit +
  `mlir-translate-20 --mlir-to-cpp` clean (1780-line `ours.cpp`, 16× `vluxei16`).
- Full `check-tianchenrv`: 707/710. The same 3 pre-existing unrelated failures
  (`rvv-generated-bundle-abi-e2e-*computed-masked-strided-input-widening-dot-
  reduce-add*` + abi-e2e self-test, a `vlse16` strided-load op; zero reference to
  iq1_m/ternary/vluxei). iq1_m + iq1_s emit tests both pass.

## 2. Byte-exact oracle — PASS (bit-identical)

Old-scalar-emit iq1_m vs new-vluxei-emit iq1_m, distinct symbols, same binary,
8 seeds × 6 sizes, raw float bits `memcmp`'d:

```
BYTE-EXACT-ORACLE cases=48 mismatches=0 -> PASS (bit-identical)
```

48/48 bit-identical. The revectorization is byte-exact-correct. (Also: micro
harness AGREEMENT vs ggml `max_rel_norm=0.000e+00` for BOTH emits.)

## 3. objdump engagement proof — vluxei16 PRESENT (yes)

```
$ objdump -d k_vlux.o | grep -c vluxei16     # new kernel
16    # (8 sub-blocks × 2 halves, i64m1 gathers)
   1d8: vluxei16.v v12,(s7),v11
   20c: vluxei16.v v11,(s7),v9
   ...
$ objdump -d k_scalar.o | grep -c vluxei16   # prior scalar kernel
0
```

The new binary contains the hardware indexed gather (16×); the scalar baseline has
none. Engagement proven.

## 4. Micro ratio vs ggml — kernel 1.50x faster, gap NARROWED ~9.3x → ~5.8x, NOT parity (HONEST)

`blockdot-bench/iq1_m` micro (timing_n=32*QK_K, iters=2000, reps=200, best-of-min,
`taskset -c 2`), ours = OUR emit, ggml = shipped `_vl128`:

| emit | ours (ns) | ggml (ns) | RATIO ggml/ours | gap (ggml faster by) |
|---|---|---|---|---|
| SCALAR (prior) | 58907.2 | 6354.9 | 0.108 | ~9.3x |
| VLUX (new)     | 39256.1 | 6765.7 | 0.155–0.172 | ~5.8x |

- **The robust, noise-free number: our kernel got 1.50x faster (58907 → 39256 ns).**
  This is our-vs-our (same harness, same input), not subject to ggml-baseline drift.
- The gap to ggml NARROWED from ~9.3x to ~5.8x. **These gap ratios are APPROXIMATE**:
  the two rows used slightly different ggml best-of-min baselines (6355 vs 6766 ns,
  ggml's own run variance), so the gap numbers mix baselines — read them as "the
  remaining gap roughly halved," not as an exact 9.3→5.8.
- **This is NOT parity** (advisor-locked expectation was maturity/gap-closing, not a
  tie and NOT "we beat ggml"). We adopted ggml's exact hardware gather (`vluxei16`,
  objdump-confirmed, byte-exact).
- Note: iq1_m's residual is LARGER than iq1_s's 2.3x. This is NOT a regression vs
  iq1_s — it's a different kernel with more non-gather machinery (see below). Do NOT
  cross-compare the two residuals as if same-axis; each is "ours vs ggml on THAT
  kernel."
- Defensible claim: **"we now emit the same hardware indexed gather ggml uses
  (`vluxei16`, byte-exact-correct); the iq1_m kernel is 1.50x faster and the
  remaining gap to ggml roughly halved (~9.3x → ~5.8x)."**

### Why the residual is larger than iq1_s's 2.3x (honest mechanism — NOT "vluxei16")
The gather instruction is now identical to ggml's; the residual is everything else.
iq1_m adds TWO iq1_m-specific residual sources over iq1_s (un-isolated — listed, not
quantitatively ranked; isolating gather-width from delta would need an i64m2-gather
variant, beyond this maturity scope):
1. **Narrower gather + a reduce ggml avoids.** We gather 2 u64 = 16 i8 per
   `vluxei16_v_i64m1` (m1) + one `vwredsum` per half. This is NARROWER than even
   iq1_s's own gather (`i64m2`, 4 entries), and far narrower than ggml's iq1_m
   `vluxei16_v_u64m4` (m4 = 4× our LMUL, 8 u64 = 64 i8), which `vget`-splits,
   `vwmul i16m4` over 32 lanes, and accumulates via `vwmacc` with NO per-block
   reduce (quants.c:3216-3239). So both more gather issue AND a reduce-per-half we
   pay and ggml doesn't. (The narrower i64m1 was the shape-agnostic byte-exact
   choice for the per-half scale split; an i64m2+vget variant is the named wider
   follow-on.)
2. **Scalar index assembly.** We keep the 4 index derivations + 4 `idxoff[]` stores
   on the SCALAR unit (byte-exact-safe). ggml builds the whole index vector
   in-register (`vzext`/`vsll`/`vor`/`vand` over u16m2, quants.c:3200-3203).
3. **Per-group scalar/Σq8 delta path (iq1_m-SPECIFIC, untouched here).** Our delta
   term is 4 per-group `vle8`/`vwredsum_i8m1` Σq8 reduces × scalar `delta[l]`. ggml
   vectorizes the delta as a full ±1 `delta` vector (`vmsgtu` mask → `vmerge`) and
   folds it with a second `vwmul i16m4`/`vwmacc` (quants.c:3207-3239) — no per-group
   reduce. This path is grid-INDEPENDENT, so the gather change correctly does not
   touch it. All three are named, scoped follow-ons (DESIGN step 2-wide /
   vectorized-index / vectorized-delta), not regressions.

## 5. Scope / honesty notes (iq1_m)
- Compute-side micro only; e2e decode is memory-bound → NOT an e2e win, NOT claimed
  (MEMORY: kernel-wins-dont-transplant-to-e2e).
- Byte-exact gate is the correctness proof; tolerance micro is not.
- No git commit (lead commits).
- Staged on rvv: `~/vlux-iq1m/` (k_scalar/k_vlux/oracle, ours_scalar/ours_vlux/
  harness/ggml_ref for micro).

---

# FINDING — vluxei16 IQ-gather revectorization, THIRD+FOURTH kernels (iq3_xxs + iq3_s)

Extends the proven iq1_s/iq1_m vluxei16 pattern to the iq3 family. Both done, both
byte-exact. Same `ssh rvv` regime (SG2044, RVV1.0, VLEN128, clang-18,
`-O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast`, `taskset -c 2`).

## 0. SIGN-PLANE BLOCKER CHECK — NOT signs64-blocked (the load-bearing fact)

The DESIGN doc flagged iq2 as BLOCKED: its *vectorized* sign-fold needs the
`keven_signs_q2xs` (signs64) table, which the op does NOT carry. **iq3 is NOT
blocked** — verified against the emitters:
- `emitIQ3XXSQ8KBlockDot` reads its sign byte as `tcrv_iq3xxs_ksigns[(aux32>>7l)&127]`
  — the u8 selector table the op DOES carry via `getKsigns()` (`RVVToEmitCGridCodebook
  .cpp` grid+ksigns decls).
- `emitIQ3SQ8KBlockDot` reads explicit per-group sign bytes DIRECTLY from the weight
  signs region (`xb+74`) — NO table at all.
Neither needs signs64. The task is to vectorize only the integer grid GATHER; the
scalar-broadcast ksigns/kmask/vmsne/vneg/vmerge sign-fold STAYS UNCHANGED (it is
byte-exact with itself — that is the gate, not ggml's mechanism). ggml's iq3 happens
to use signs64 + `vrsub`, but we keep our existing sign path. → BOTH iq3_xxs and
iq3_s done; no op-attr extension needed.

## 1. The change (file:lines)

`lib/Conversion/RVV/RVVToEmitCGridCodebook.cpp`:
- `emitIQ3XXSQ8KBlockDot` (grid is `uint32[256]`, 4 bytes/entry) and
  `emitIQ3SQ8KBlockDot` (grid `uint32[512]`) — SAME revectorization in both.
- Added gather emit types (`vuint16mf2_t`, `uint16_t*`, `const int32_t*`; reused
  `vint8m1_t`/`vint16m2_t`/`vint32m1_t`). Added `groupLanes = 8`. Removed now-dead
  `entryLanes`.
- Replaced `gridOf4Pass` (ONE 4-lane scalar pass: `vle8(grid_i8+idx*4, 4)` + sign-fold
  + `vwmul i16m2` + `vwredsum`, called TWICE per sign group = 8 passes/sub-block) with
  `gridOf4Group` (ONE 8-lane body per sign group = 4 groups/sub-block):
  - KEEP the scalar grid indices byte-exact: iq3_xxs `idx1=qg[2l]`, `idx2=qg[2l+1]`;
    iq3_s `idx1/idx2` qh-9th-bit-injected (`qs[...] | ((qh<<sh)&256)`) — UNCHANGED.
    Each `<<2` (byte offset = idx*4 for the u32 grid) into a `uint16_t tmp[2]`.
  - `__riscv_vle16_v_u16mf2(&tmp[0], 2)` → index (EMUL = (16/32)*m1 = mf2).
  - `__riscv_vluxei16_v_i32m1((const int32_t*)tcrv_iq3{xxs,s}_grid, vidx, 2)` → gather
    the 2 grid u32 entries → `__riscv_vreinterpret_v_i32m1_i8m1` = the 8 grid bytes
    (mirror iq1_s's signed-i64 gather; each grid byte ≤62 (xxs)/≤15 (s) < 128, so the
    i8 reinterpret = ggml's identical u8 grid values).
  - sign-fold UNCHANGED ops, vl 4→8, FULL 8-bit kmask `{1,2,4,8,16,32,64,128}` (was
    split kmaskLo/kmaskHi for the two 4-lane passes): `vmv_v_x_u8m1(signs)` / `vand` /
    `vmsne_vx_u8m1_b8` / `vneg_v_i8m1` / `vmerge_vvm_i8m1`.
  - `vle8_v_i8m1(q8Group, 8)` → `vwmul_vv_i16m2` → `vwredsum_vs_i16m2_i32m1` into the
    same `sumiAcc`.
- KEPT UNCHANGED (byte-exact FP/scale/sign): the ksigns/explicit-signs read, `ls`,
  `bsum`/`sumi`, the per-super-block fold, iq3_xxs's trailing `*s = 0.25f*sumf`,
  iq3_s's `*s = sumf` (no factor). Grid table decls untouched.

Lane-mapping byte-exactness proof: group lanes 0..3 = grid1 / q8[0..3] / kmask
{1,2,4,8} (= old pass A), lanes 4..7 = grid2 / q8[4..7] / kmask {16,32,64,128} (= old
pass B). Per-lane products byte-identical; the i32 `vwredsum` is order-free, so
collapsing the two 4-lane reductions into one 8-lane reduce per group is byte-safe.
EMUL gotcha (advisor-flagged): `vluxei16_v_i32m1` wants a `vuint16mf2_t` index; gather
+ `vle16` take vl=2 (entries), sign-fold + q8 + vwmul + vwredsum take vl=8 (lanes) —
literals, NO captured `vsetvl`.

`tcrv-opt`/`tcrv-translate` rebuilt FORCED/CLEAN (`rm -f build/bin/tcrv-opt
build/bin/tcrv-translate && ninja bin/tcrv-opt bin/tcrv-translate`). No ODS/.td
touched. Emit clean: iq3_xxs 1663-line + iq3_s 1667-line `ours.cpp`, each with 32
real `vluxei16_v_i32m1` (8 sub-blocks × 4 groups).

## 2. Byte-exact oracle — PASS (bit-identical), BOTH kernels

Old-scalar-emit vs new-vluxei-emit, distinct symbols (`iq3{xxs,s}_scalar` /
`iq3{xxs,s}_vlux`) in one binary, 8 seeds × 6 sizes, raw float bits `memcmp`'d:

```
BYTE-EXACT-ORACLE [iq3_xxs] cases=48 mismatches=0 -> PASS (bit-identical)
BYTE-EXACT-ORACLE [iq3_s]   cases=48 mismatches=0 -> PASS (bit-identical)
TOTAL mismatches=0 -> ALL PASS
```

96/96 bit-identical. The revectorization is byte-exact-correct on both. (Also: micro
harness AGREEMENT vs ggml `max_rel_norm=0.000e+00` for all four emits.)

## 3. objdump engagement proof — vluxei16 PRESENT (yes)

```
$ for f in iq3xxs_scalar iq3xxs_vlux iq3s_scalar iq3s_vlux; do
    objdump -d $f.o | grep -c vluxei16; done
iq3xxs_scalar vluxei16=0
iq3xxs_vlux   vluxei16=32
iq3s_scalar   vluxei16=0
iq3s_vlux     vluxei16=32
```

The new binaries contain the hardware indexed gather (32× each = 8 sub-blocks × 4
groups); the scalar baselines have none. Engagement proven.

## 4. Micro ratio vs ggml — gaps CLOSED, NOT parity (HONEST)

Measured in ONE combined binary (scalar + vlux + ggml, symmetric warmup, same layout
→ the our-vs-our ratio is confound-free), `timing_n=32*QK_K`, iters=2000, reps=200,
best-of-min, `taskset -c 2`. ggml = shipped `_vl128`:

| kernel | scalar (ns) | vlux (ns) | ggml (ns) | our-vs-our (vlux faster) | gap scalar→ggml | gap vlux→ggml |
|---|---|---|---|---|---|---|
| iq3_xxs | 123830 | 45089 | 5540 | **2.75x** | 22.35x | **8.14x** |
| iq3_s   |  47143 | 40389 | 8011 | **1.17x** |  5.88x | **5.04x** |

- **Robust number = our-vs-our** (same harness/input, not subject to ggml-baseline
  drift): iq3_xxs **2.75x faster**, iq3_s **1.17x faster**. In-family with iq1_s 2.8x
  / iq1_m 1.5x.
- The gap to ggml CLOSED: iq3_xxs 22.35x → **8.14x**; iq3_s 5.88x → **5.04x**.
- **This is NOT parity** (advisor-locked expectation = maturity / gap-closing, NOT a
  tie and NOT "we beat ggml on iq3"). We adopted ggml's exact hardware gather
  (`vluxei16`, objdump-confirmed, byte-exact).
- Defensible claim: **"we now emit the same hardware indexed gather ggml uses
  (`vluxei16`, byte-exact-correct); iq3_xxs is 2.75x faster (gap 22.4x→8.1x) and iq3_s
  1.17x faster (gap 5.9x→5.0x)."**

### ggml-baseline-per-binary contamination (caught + handled, like iq1_m)
The SPLIT vlux bench (vlux ours linked with ggml, separate binary) reported ggml
~564000 ns — a reproducible 100x per-binary timing contamination of the ggml ref when
the vlux kernel precedes it in the warmup (NOT random noise: best-of-min over 200 reps,
reproduced across runs; the vlux `ours` itself timed FAST + stable at 45577 across 3
runs, so vector state is fine — contamination only slows, it didn't touch ours). The
vlux-free SCALAR binary gave a clean ggml = 5511 ns (twice, byte-identical ggml source),
and the COMBINED binary above reproduces clean ggml = 5540/8011 — those are the valid
baselines. The 564000 / 12.4x ratio is a measurement artifact and is NOT reported.

### Why iq3_s gains less than iq3_xxs (honest mechanism — NOT "vluxei16")
iq3_s's scalar baseline was already ~2.6x faster than iq3_xxs's (47k vs 124k ns), so
less of its time was in the gather → less to recover. Both residuals are NOT the gather
(now identical to ggml's), but: (1) narrow 2-entry `i32m1` gather LMUL vs ggml's
wider multi-entry gather (iq3_xxs `u32m4`/16-entry, iq3_xxs/ggml_ref.cpp:52; iq3_s
`u32m2`/8-entry, iq3_s/ggml_ref.cpp:65); (2) scalar index assembly (iq3_s also pays the qh-9th-bit
injection on the scalar unit; ggml builds the index vector in-register); (3) scalar
sign-byte derivation. All named, scoped follow-ons (DESIGN step 2-wide / 32-lane
vectorized-index), not regressions. The 32-lane ggml shape (one gather over all 8
group entries + a vrgather-assembled 32-lane sign mask) is the named follow-on, not
this step.

## 5. Scope / honesty notes (iq3)
- Compute-side micro only; e2e decode is memory-bound → NOT an e2e win, NOT claimed
  (MEMORY: kernel-wins-dont-transplant-to-e2e). Compute-side gather → e2e decode NULL.
- Byte-exact gate is the correctness proof; the tolerance micro is not.
- No git commit (lead commits).
- Staged on rvv: `~/vlux-iq3/` (iq3{xxs,s}_{scalar,vlux}.cpp + byte_exact_oracle_iq3
  + combined bench + ggml_{xxs,s}); `~/blockdot-cov/iq3_{xxs,s}/ours_vlux.cpp` is the
  new emit for the split micro harness.

## Reproduce
- emit: `build/bin/tcrv-opt test/Conversion/RVV/rvv-to-emitc-iq3-{xxs,s}-q8-k-block-dot
  .mlir --tcrv-rvv-lower-to-emitc | mlir-translate-20 --mlir-to-cpp` (dev host, forced
  clean tcrv-opt).
- oracle (rvv): `clang-18 -O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast
  byte_exact_oracle_iq3.cpp iq3xxs_scalar.cpp iq3xxs_vlux.cpp iq3s_scalar.cpp
  iq3s_vlux.cpp -o oracle && taskset -c 2 ./oracle`.
- micro (rvv, combined): `clang-18 ... combined_bench_iq3.cpp iq3{xxs,s}_{scalar,vlux}
  .cpp ggml_{xxs,s}.cpp -o combined && taskset -c 2 ./combined`.
- objdump: `objdump -d iq3xxs_vlux.o | grep -c vluxei16` → 32.
