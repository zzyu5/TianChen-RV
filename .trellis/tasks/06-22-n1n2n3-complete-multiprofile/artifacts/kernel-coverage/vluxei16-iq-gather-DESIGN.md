# DESIGN — vluxei16 IQ-quant grid/sign gather (emitter-maturity target)

READ-ONLY scoping doc. No `lib/` edits made. All file:line refs verified against the
tree at commit `189176c7`. Primary sources: our IQ emitters + ggml riscv
`quants.c`.

## Verdict in one line
FEASIBLE, but the task's "rewire to an existing primitive" framing is **too strong**.
This is **per-kernel revectorization** of each IQ grid emitter onto an already-present
intrinsic-**name-builder** (`riscvIndexedMemoryIntrinsicName`, which can spell
`vluxei16`) emitted as a direct `CallOpaqueOp` — the SAME emit style the IQ/FP4
methods already use. It is NOT a one-call swap and NOT a reuse of the masked-indexed-
load Op infrastructure or the FP4 `vrgather`. **First kernel: `iq1_s`** (simplest:
signed-ternary grid, NO sign plane).

---

## Q1. WHERE the IQ grid/sign emitter produces the SCALAR gather

Three IQ emitter source files, all under `lib/Conversion/RVV/`:

| kernel(s) | file | scalar-gather site |
|---|---|---|
| iq2_xxs / iq2_xs / iq2_s | `RVVToEmitCGridCodebook.cpp` | iq2_xxs grid+sign group: **lines 392–509** (entry `emitIQ2XXSQ8KBlockDot` @29) |
| iq3_xxs / iq3_s | `RVVToEmitCGridCodebook.cpp` | `emitIQ3XXSQ8KBlockDot` @629, `emitIQ3SQ8KBlockDot` @1284 |
| iq1_s / iq1_m | `RVVToEmitCTernaryBinary.cpp` | iq1_s ternary group: **lines 385–453** (entry `emitIQ1SQ8KBlockDot` @29) |

**The scalar shape (iq2_xxs, `RVVToEmitCGridCodebook.cpp:395–438`)** — per group `l`,
ONE element of work, scalar index computed and used in scalar pointer arithmetic:

```c
int idx   = a[l];                                  // line 396: loadByteAsInt — scalar
int signs = tcrv_iq2xxs_ksigns[(aux1 >> 7*l)&127]; // 397-418: SCALAR table subscript+load
__riscv_vsetvl_e8m1(8);                            // 422-431: m1, vl=8 (ONE group)
const int8_t *grid = grid_i8 + idx*8;              // 433-438: SCALAR ptr arith (idx*8)
grid_v = __riscv_vle8_v_i8m1(grid, 8);             // 440-449: contiguous 8-byte load
// ... bcast(signs)/vand kmask/vmsne/vneg/vmerge sign-apply (459-507) ...
p = __riscv_vwmul_vv_i16m2(grid_signed, q8_v, 8);  // 509-520
sumiAcc = __riscv_vwredsum_vs_i16m2_i32m1(p, sumiAcc, 8); // 522-534
```

**The scalar shape (iq1_s, `RVVToEmitCTernaryBinary.cpp:388–453`)** — same skeleton,
NO sign plane (ternary grid IS signed):

```c
int idx = qs[ib*4+l] | (((qhw>>(3*l))&7)<<8);   // 392-414: SCALAR 11-bit index assembly
__riscv_vsetvl_e8m1(8);                          // 416-426
const int8_t *grid = grid_i8 + idx*8;            // 428-434: SCALAR ptr arith
grid_v = __riscv_vle8_v_i8m1(grid, 8);           // 435-445: 8-byte contiguous load
p = __riscv_vwmul_vv_i16m2(grid_v, q8_v, 8);     // 455-464: NO sign-apply
lacc = __riscv_vwredsum_vs_i16m2_i32m1(p, lacc, 8); // 466-...
```

The cost is the **scalar index + per-group m1/vl=8 micro-vector**: 4 separate 8-lane
loads/products/reductions per sub-block, index math on the scalar unit. This is the
5–22x loss vs ggml's single 32-lane gather+product per sub-block.

---

## Q2. The existing vluxei16/vrgather primitive + whether FP4 path is reusable

### The vluxei16 name-builder ALREADY EXISTS and is EEW-parameterized
`lib/Conversion/RVV/RVVToEmitCSupport.cpp:191–200`:
```cpp
std::string riscvIndexedMemoryIntrinsicName(llvm::StringRef mnemonic,
                                            unsigned indexEEW,
                                            llvm::StringRef dtype,
                                            llvm::StringRef lmul) {
  os << "__riscv_" << mnemonic << indexEEW << "_v_" << dtype << lmul;  // unmasked
}
```
Called as `riscvIndexedMemoryIntrinsicName("vluxei", 16, "u64", "m4")` →
`"__riscv_vluxei16_v_u64m4"` — **exactly ggml's spelling**, unmasked 3-arg form
`(base, idxVec, vl)`. This is the correct reuse target.

### The MASKED indexed-load infra is the WRONG consumer (do not use it)
`riscvMaskedIndexedLoadIntrinsicName` (Support.cpp:289) emits the `_tumu` masked
variant; its driver `emitMaskedIndexedLoad` (`RVVToEmitC.cpp:4711–4714`) is gated
`eew != 32 → notifyMatchFailure` and needs a `MaskedIndexedLoadOp` + mask authority.
ggml's IQ gather is **unmasked, EEW=16**. So we do NOT route through the masked Op
path — we emit the unmasked `vluxei16` as a **direct `CallOpaqueOp`**, which is exactly
how the IQ/FP4 methods already build every intrinsic (e.g. iq1_s `vle8` at
TernaryBinary.cpp:442). **The reuse is the name-builder + the direct-CallOpaqueOp
pattern, not the masked-Op machinery.**

### FP4's vrgather is NOT a reusable primitive for the grid gather
FP4 (`RVVToEmitCCodebookFp4.cpp:231–243`) gathers a **16-entry** codebook held **in a
vector register** via `__riscv_vrgather_vv_i8m1(values, idx, vl)` (in-register
permute). The IQ grids are **256–2048 u64 entries in MEMORY** (iq2xxs_grid[256],
iq1s_grid[2048]) — they cannot live in a vreg, so they need `vluxei16` (memory
indexed load), a **different instruction class**. FP4 is a *style* reference only:
"emit a gather as a CallOpaqueOp, feed the product/reduce tail." Even its tail differs
(FP4 = `emitOffsetBinaryProductFromDecodedValue` vwmacc-asymmetric; IQ = symmetric
`vwmul`+`vwredsum`). **The IQ grid emitter CANNOT call the FP4 gather primitive.**

---

## Q3. ggml's win = vluxei16 over the uint64 grid/sign table — CONFIRMED

`/home/kingdom/phdworks/llama.cpp/ggml/src/ggml-cpu/arch/riscv/quants.c`. Table types
confirmed in `ggml-common.h:550` (`GGML_TABLE_BEGIN(uint64_t, iq2xxs_grid, 256)`);
`keven_signs_q2xs` and `iq1s_grid` are likewise `uint64_t[]`.

- **iq2_xxs (vl128, quants.c:4536, 4579–4586):**
  ```c
  const uint64_t *grid64  = (const uint64_t *)iq2xxs_grid;
  const uint64_t *signs64 = (const uint64_t *)keven_signs_q2xs;
  vidx_q2 = (u16) raw_byte << 3;   // element idx*8 = byte offset into u64 base
  vq2_64 = __riscv_vluxei16_v_u64m2(grid64,  vidx_q2, 4);   // GATHER grid entries
  vs2_64 = __riscv_vluxei16_v_u64m2(signs64, vidx_s2, 4);   // GATHER sign words (signs64!)
  ```
- **iq1_s (vl128, quants.c:2800):**
  ```c
  vuint16m4_t index = __riscv_vsll_vx_u16m4(qh_index, 3, 32);  // 11-bit idx << 3
  vint8m4_t grid0 = __riscv_vreinterpret_v_i64m4_i8m4(
      __riscv_vluxei16_v_i64m4((const int64_t*)iq1s_grid, grid_index0, 8));  // 4 entries=32 i8
  lsum0 = __riscv_vwmul_vv_i16m8(grid0, q80, 128);            // NO sign-apply (signed grid)
  ```

So ggml's win is precisely `__riscv_vluxei16` over the u64 grid (and, for iq2/iq3, a
second `vluxei16` over the u64 `keven_signs_q2xs` sign table), with **u16 byte-offset
indices = grid_index << 3**. The index reinterpret to `i8`/`u8` unpacks each u64 entry
into 8 int8 grid values in-lane. No `_tumu`, no mask, EEW=16.

**Index-width: NO blocker.** Largest index is iq1_s's 11-bit grid (max 2047);
`2047<<3 = 16376 < 65535`, fits u16. iq2/iq3 use 8-bit indices (`raw<<3 = max 2040`).

---

## Q4. PRECISE ordered build-incremental edit plan

**FIRST KERNEL: `iq1_s`** (`RVVToEmitCTernaryBinary.cpp::emitIQ1SQ8KBlockDot`). Why
first (discriminated on two verified facts):
1. **No sign plane.** Op doc (`RVVOps.td:7454+`) and emit confirm: ternary grid bytes
   are `{0x00,0x01,0xff}={0,+1,-1}`, byte-viewed as `const int8_t *` they ARE the
   signed value — NO ksigns / kmask / vmsne / vneg / vmerge. The vectorized body is
   pure `gather → vwmul → vwredsum`. iq2_xxs would additionally need the
   `keven_signs_q2xs` (signs64) table, which the op does **NOT carry** (it carries only
   `ksigns` = the u8 *selector*, `DenseI32ArrayAttr:$ksigns`, RVVOps.td:6951) — so
   iq2_xxs's vectorized sign-fold is a strictly harder, separately-scoped problem.
2. ggml's iq1_s vl128 reference (quants.c:2800–2835) is fully in hand.

**Oracle discipline (verified, governs how much ggml we may copy):**
- The micro harness (`blockdot-bench/iq1_s/harness.cpp:66`) is **relative-tolerance**
  vs ggml (`rel = fabs(ro-rg)/denom`), NOT bit-exact.
- The project gate (MEMORY: byte-exact fingerprint) is **bit-exact vs the CURRENT
  scalar emit**. The integer grid dot (`vwmul`→`vwredsum`, seeded from the running
  i32m1) is **order-free**, so revectorizing it is byte-safe. **Do NOT import ggml's
  FP accumulation order** (`sumf += d*(...)`, `*s = 0.125f*sumf`) — the iq1_s
  per-super-block fp32 fold and the delta-bsum term (`sumi1`, `IQ1S_DELTA`) must stay
  byte-identical to the current emit. Revectorize ONLY the inner integer grid gather.

**The new emitted-C target (replaces the 4-group `l`-loop, TernaryBinary.cpp:385–~480)**
— ONE 32-lane body per sub-block, KEEPING the already-written scalar index math and
our delta/scale/accumulate order. The index is the only subtlety, so we leave it on the
**validated scalar path** and only vectorize the gather+product+reduce:
```c
// Reuse the EXISTING 4 scalar idx computations (TernaryBinary.cpp:392-414, byte-exact):
//   idx_l = qs[ib*4+l] | (((qhw>>(3*l))&7)<<8),  l=0..3.   Pack into a u16 stack array.
uint16_t tmp[4] = { (u16)(idx0<<3), (u16)(idx1<<3), (u16)(idx2<<3), (u16)(idx3<<3) };
__riscv_vsetvl_e8m1(32);                          // 4 entries * 8 i8 = 32 lanes (m2)
vuint16m1_t vidx = __riscv_vle16_v_u16m1(tmp, 4); // 4 byte-offsets (idx*8), fits u16
vint8m2_t   grid = __riscv_vreinterpret_v_i64m2_i8m2(
    __riscv_vluxei16_v_i64m2((const int64_t*)tcrv_iq1s_grid, vidx, 4)); // GATHER 4 entries
vint8m2_t   q8v  = __riscv_vle8_v_i8m2(q8Group, 32);                    // full 32-lane sub-block
vint16m4_t  p    = __riscv_vwmul_vv_i16m4(grid, q8v, 32);
lacc = __riscv_vwredsum_vs_i16m4_i32m1(p, lacc, 32); // SAME seed/accumulate as today
// delta/scale (sumi += ls*lacc; sumi1 += delta*bsums...) UNCHANGED — byte-exact.
```
**Width note (corrected):** ONE sub-block = 32 lanes → gather 4 u64 entries with
`vluxei16_v_i64m2` (vl=4) → reinterpret `i8m2` (VLMAX=32 at VLEN=128) → `vwmul`→`i16m4`
→ `vwredsum i16m4→i32m1`. Do **NOT** copy ggml's `i64m4`/`i8m4` verbatim: ggml gathers
**8 entries = 64 i8 = TWO sub-blocks** and `vget`-splits them to keep the two
per-sub-block reductions (each with its own `ls`) separate — see the byte-exact guard
below. We keep ONE `vwredsum` per sub-block, which makes the split unnecessary.

**Byte-exact guard (integer order-freeness is per-sub-block only):** each sub-block has
its own scale `ls`, folded as `sumi += ls_b * lacc_b`. NEVER merge reductions across
sub-blocks — the 32-lane (one-sub-block) body preserves this automatically. This is
exactly why ggml's 64-lane two-sub-block gather must `vget`-split; our 32-lane choice
sidesteps it.

Putting the index math in a `uint16_t tmp[4]` keeps all index derivation on the already-
verified scalar path while still collapsing the 4 micro-reductions into one — that
collapse IS the win. ggml's full-vector index construction (the `0x0009000600030000`
shift + `vrgather` + `vor vzext(qs)` sequence, quants.c:2790-2799) is a LATER, harder-
to-byte-verify optimization, not step 2.

**ORDERED, each step COMPILES (timeout-resistant small increments):**

1. **(name-builder reuse, optional 1-entry warm-up)** Add a local helper in
   `emitIQ1SQ8KBlockDot` that builds the unmasked vluxei callee via
   `riscvIndexedMemoryIntrinsicName("vluxei", 16, "i64", lmul)` and emits it as a
   `CallOpaqueOp` with `(grid_base_i64, idxVec, vl)`. Optionally wire it to gather
   **ONE** entry per group (vl=1, reinterpret i64m1→i8m1, 8 lanes) as a drop-in for the
   current `vle8(grid_i8+idx*8)`, keeping the per-group loop, just to prove the gather
   spelling compiles + verifies. This step is **largely redundant** with step 2 (the
   scalar index is kept either way), so it may be skipped — go straight to step 2 if
   confident. Build `tcrv-opt` (forced/clean per MEMORY), re-emit iq1_s, byte-exact vs
   current `ours.cpp`.

2. **(collapse the 4 reductions — the actual win)** KEEP the 4 EXISTING scalar idx
   computations (TernaryBinary.cpp:392-414, byte-exact, untouched). Pack the 4
   `idx<<3` byte-offsets into a `uint16_t tmp[4]`, `vle16_v_u16m1(tmp,4)`. ONE
   `vluxei16_v_i64m2 → i8m2` gather of all 4 entries (32 i8), ONE `vle8_v_i8m2(q8Group,
   32)`, ONE `vwmul_vv_i16m4`, ONE `vwredsum_vs_i16m4_i32m1` seeded by the same `lacc`.
   Drop the inner `l`-loop. Delta/scale/`sumi`/`sumi1` fold UNCHANGED. ONE reduction
   per sub-block (per the byte-exact guard — `ls` is per-sub-block). Build, byte-exact
   vs current emit (integer order-free within the sub-block → must match), then
   micro-time vs ggml on `ssh rvv`. NOTE the corrected widths: `i64m2`/`i8m2`/`i16m4`,
   NOT ggml's `i64m4`/`i8m4` (those gather two sub-blocks).

3. **(oracle + perf seal)** Run `blockdot-bench/iq1_s` byte-exact (vs scalar emit) +
   micro vs ggml on real RVV. Record speedup. Expect the 5–22x loss to collapse toward
   ggml parity (ggml's own win IS this exact gather).

4. **(extract shared tail)** Only after iq1_s lands: factor a
   `emitVluxei16GridGatherProductReduce(base_i64, idxVec, q8, vl, accSeed)` helper into
   the IQ emit support so iq1_m / iq3 can reuse the gather+product+reduce tail. Keep
   per-kernel index-derivation + sign-representation local (they differ).

5. **(iq3_xxs / iq3_s, then the sign-plane kernels)** Port iq3 (8-bit grid index,
   still has a ksigns sign plane → the sign-fold stays scalar-broadcast OR a second
   vluxei16 over signs64 if the op is extended to carry it). iq2_xxs/iq2_xs/iq2_s LAST:
   they need `keven_signs_q2xs` (signs64) for the *vectorized* sign gather, which the
   op does not currently carry — that is a separate op-attr extension, scoped
   independently.

**Oracle-verify correctness (both axes, each step):**
- (a) **byte-exact** vs the current scalar emit: re-emit iq1_s `.cpp`, diff numeric
  output for the same input — the integer gather is order-free so it MUST match
  bit-for-bit; the FP fold is untouched.
- (b) **micro-time** vs ggml: `blockdot-bench/iq1_s` harness on `ssh rvv` (tolerance
  check + cycles), forced/clean `tcrv-opt` rebuild per the MEMORY incremental-build
  hazard.

---

## Q5. RISK + effort

- **Scope: NOT all 7 at once.** Do **iq1_s first** (1 kernel), prove byte-exact +
  micro-win, THEN extract the shared gather→product→reduce tail, THEN per-kernel.
  Realistic grouping by difficulty:
  - **Easy (no/inline sign):** iq1_s, iq1_m — grid is signed; gather→vwmul→reduce.
  - **Medium (sign plane stays scalar OR needs signs64):** iq3_xxs, iq3_s.
  - **Hard (need keven_signs_q2xs op-attr extension for vectorized sign gather):**
    iq2_xxs, iq2_xs, iq2_s.
- **Shared vs per-kernel:** the gather+product+reduce **tail** is a shared helper
  (step 4). The **index derivation** (8-bit vs 11-bit-with-qh) and **sign
  representation** (none / scalar-broadcast / signs64-gather) are per-kernel — that, not
  the gather call, is the cost driver.
- **Effort:** iq1_s alone ≈ a focused day (3–4 compiling increments). iq1_m + both iq3
  ≈ +1–2 days reusing the tail. The iq2 family ≈ separate task (op-attr extension to
  carry `keven_signs_q2xs` + verifier + emit), do not bundle.
- **Blockers / risks:**
  1. **None on index width** — verified `2047<<3 < 65535` fits u16 EEW.
  2. **iq2 sign table not carried** — `$ksigns` is the u8 selector, NOT signs64. The
     *vectorized* sign-fold for iq2 needs an op-attr extension; until then iq2 either
     stays scalar-sign or is deferred. (This is why iq1_s, with no sign plane, is the
     clean first kernel.)
  3. **Byte-exactness trap** — keep the FP/delta fold order identical to the current
     emit; only the order-free integer gather is revectorized. A passing tolerance
     micro is NOT proof of the byte-exact gate.
  4. **Build hazard (MEMORY)** — ODS isn't touched (emit-method-local), so the
     RVVOps.cpp.inc regen hazard doesn't bite, BUT `tcrv-opt` relink is unreliable:
     gate the final fingerprint on a forced/clean rebuild, and require `ssh rvv`
     evidence for the perf claim.
