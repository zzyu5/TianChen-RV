# Track B G3 (q4_K super-block core) — bounded feasibility verdict (2026-06-29)

**VERDICT: bounded auto-construction NOT cleanly reachable for q4_K** (unlike q4_0/codebook). No code change — feasibility verdict only (the discipline: don't invent a synthetic byte-exact target).

## Why q4_K does NOT factor (4 structural reasons, code-grounded)
The q4_K integer core = `emitQ4_KSuperBlockAux32Core` (`lib/Conversion/RVV/RVVToEmitCKQuant.cpp:842-1265`):
1. **Unpack = plain-mask + scratch-decoupled, NOT the offset-binary fused op.** q4_K does `q4&0xF` / `q4>>4` in u8 domain, **no offset-binary bias**, writes scratch `aux8[256]` (`:870-967`). G1's reusable `tcrv_rvv.packed_i4_offset_binary_x_i8_product` (offset-binary register-fused, `vxor 0x88`+sign-extend) is **NOT reusable** → q4_K needs a q4_K-specific unpack op.
2. **uint6 scale FUSED into the dot, inseparable from the bit-dance.** scale enters as `vwmacc_vx(aux32, scale, p, vl)` (`:1175-1184`); `scale` comes from the 6-bit bit-dance (region B, excluded by bounded scope). Byte-exact scaled MAC REQUIRES the bit-dance.
3. **Dot accumulates into an 8-lane `aux32` VECTOR, not a scalar reduce.** No clean `standalone_reduce`-to-i32-scalar like G1/G2's `...→product→reduce→store`; collapse happens much later in the fp fold.
4. **Region A is INLINE** (no factored helper, grep-confirmed; not separately lit-pinned) and uses **literal-vl `vsetvl_e8m2(32)`** vs the front-door scaffold's **runtime-n** `tcrv_rvv.setvl` → no existing q4_K emit path to byte-match at the bounded core.

## Capability / flip
q4_K = **NO VLEN flip** (q4_0-style; not in any schedule autotuner; monolithic op lowers at default `mf2` anchor, region A hard-pinned `u8m2/i8m2`; dormant `integer_core_lmul` knob is emitter-sealed, not gearbox-selected). No q4_K "gearbox to ride" at bounded-core.

## Full-G3 needs NEW generic ODS vocab (multi-day, DEFERRED)
plain-nibble-unpack-to-scratch op (+ factored emitter) · 6-bit scale/min bit-dance as generic ops · scaled-`vwmacc`-into-vector op · wide `aux32` accumulator + register fold-back · the q4_K MIN term (`sumf -= dmin·Σbsums·mins`) · super-block outer loop · per-block fp16 d/dmin cast-deref reads · deferred two-level fp32 fold (hairiest byte-exact seam, fp non-associativity). Overlaps the full-G1 missing-vocab catalog in `g1-q40-design.md`.

## Implication for the rung order
G1 (nibble, reuses existing op) and G2 (codebook, new ops) were tractable bounded cores; **q4_K is the inflection point where bounded-core stops working** — the most-used quant needs the full new vocabulary, not a shortcut. Next-session: either build the full q4_K vocab (multi-day) or pivot to G5 (cm-shape-through-TrackB, the beat) on the families that DO factor.
