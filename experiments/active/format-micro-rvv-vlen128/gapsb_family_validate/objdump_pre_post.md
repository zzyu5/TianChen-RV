# [GAP-SB] family pair-batching — real machine-code objdump (PRE vs POST)

De-risk question (family leg): the iq2_xxs cell proved the EMIT-level "gather 16->8" proxy
holds on silicon. Does the SAME machine-code drop occur for the 4 family members that were
pair-batched at 685ab5a1 — and, for iq3_{s,xxs}, does the extra **AVL=2 (vl=2) storm fix**
actually remove the vl=2 / fractional-LMUL config churn on real machine code?

## provenance
- PRE  = ours @ **0c64477c** (before ALL [GAP-SB] pair-batching)
- POST = ours @ **685ab5a1** (裁决三-[GAP-SB] family roll: iq2_xs/iq2_s/iq3_xxs/iq3_s pair-batch + iq3 AVL=2 fix)
- both PRE and POST are post the 0ca224f7 zfh merge => both hardware-fp16 (0 __extendhfsf2), A/B not softfloat-confounded.
- exported per fmt (main tree + build/ UNTOUCHED, no git stash) via cached detached-worktree binaries:
    tcrv-opt <git-show sha:test/Target/RVV/<dashfmt>-q8-k-super-block-block-dot-full-pipeline-export-e2e.mlir> \
      --tcrv-rvv-materialize-<dashfmt>-q8-k-block-dot-source-front-door --tcrv-materialize-emission-plans \
    | tcrv-translate --tcrv-export-target-artifact > <fmt>.o
  PRE  binaries: .worktrees/cache/0c64477ca5e6b1bd9e7e2dff697f58d2c1e8dc38/{tcrv-opt,tcrv-translate}
  POST binaries: .worktrees/cache/685ab5a1b52d5a77a62c62e886cf2129c580223f/{tcrv-opt,tcrv-translate}
- disassembly: llvm-objdump-20 -d --mattr=+v,+zvfh (llvm-20 matches the clang-20 codegen family that
  tcrv-translate embeds). Raw disasm kept per target: raw/disasm_<fmt>_{PRE,POST}.txt. Counts: raw/objdump_counts.csv.
- counting keys (mnemonic tokens, register operands excluded):
    vluxei16.v            = the indexed GATHER (grid + derived-sign) — the [GAP-SB] halving/quartering target
    vsetivli ", 0x2,"     = an AVL=2 (vl=2) config — the iq3 "AVL=2 storm" (2 elements per vector op)
    vsetivli/vsetvli mf*  = a FRACTIONAL-LMUL config (mf2/mf4/mf8) — the narrow-lane churn the storm rode on
    vwmul.vv / vwredsum.vs= the GEARBOX DISCRIMINANT (widening product + reduction) — MUST be unchanged
                            PRE->POST => the integer fold order / value semantics are byte-exact-preserved.

## per-format machine-code counts (whole exported .o == kernel + its constructed grid body)

| fmt      | vluxei16 GATHER | vle16 index | **AVL=2 (vl=2)** | fracLMUL vset | ALL vset (churn) | vwmul / vwredsum (GEARBOX) |
|----------|-----------------|-------------|------------------|---------------|------------------|-----------------------------|
| iq3_s    | **32 -> 4**     | 32 -> 4     | **46 -> 0**  ★   | 14 -> 0       | 143 -> 47 (-67%) | 32 -> 32 / 32 -> 32  (UNCHANGED) |
| iq3_xxs  | **32 -> 4**     | 32 -> 4     | **129 -> 6** (-95%) | 97 -> 6    | 254 -> 52 (-80%) | 32 -> 32 / 32 -> 32  (UNCHANGED) |
| iq2_xs   | **32 -> 8**     | 32 -> 8     | 23 -> 0  ★       | 13 -> 1       | 75 -> 26  (-65%) | 16 -> 16 / 16 -> 16  (UNCHANGED) |
| iq2_s    | **32 -> 8**     | 32 -> 8     | 37 -> 9  (-76%)  | 30 -> 12      | 109 -> 43 (-61%) | 16 -> 16 / 16 -> 16  (UNCHANGED) |

★ = AVL=2 (vl=2) storm FULLY ELIMINATED (46->0 for iq3_s, 23->0 for iq2_xs).

## reading
1. **GATHER quartered/halved on real machine code (proxy CONFIRMED for the whole family).**
   iq3_{s,xxs}: 32 -> 4 gathers (the "iq3 32->4" the task named). iq2_{xs,s}: 32 -> 8 ("iq2 32->8").
   (Whole-object gather count; consistent with the iq2_xxs cell's 16->8 measured the same way.)
2. **iq3 AVL=2 (vl=2) storm removed — the real killer.** PRE iq3_s emits 46 `vsetivli zero, 0x2, e32/e16, m1/mf2`
   configs (process only 2 elements at a time); iq3_xxs emits 129. POST replaces them with a single wide
   `vsetivli zero, 0x8, e16, m2` per pair (vl=8) => iq3_s AVL=2 46 -> **0**, iq3_xxs 129 -> **6** (the residual 6
   are unrelated e32 tail configs, not the per-sub-block gather storm). This is the machine-code proof that the
   685ab5a1 "iq3 AVL=2 修" landed: the vl=2 fractional-LMUL churn iq3_s carried (its 3.57x-loss cause) is gone.
3. **GEARBOX byte-exact-preserved.** vwmul.vv + vwredsum.vs are IDENTICAL PRE->POST for every format
   (iq3: 32/32, iq2: 16/16). The widening-product + reduction — i.e. the integer fold that defines the value —
   never changed; only the decode/gather/config shell around it was pair-batched. Consistent with STRICT byte-exact.
4. **Net config churn collapses** (all-vset -61%..-80%): halved gathers + eliminated vl=2 storm dominate; unlike the
   iq2_xxs cell, these 4 add NO vslide1down re-split (slides = 0 both sides) — the pair is assembled by wider
   config alone, so the machine-code reduction here is large (not eaten back by shuffle overhead).

=> proxy(count) -> is real on the family. Whether it nets to FASTER on silicon is decided by the A/B micro
   (derisk_cell.md sec.2): counts predict a decisive win for the storm-carrying iq3 pair.
