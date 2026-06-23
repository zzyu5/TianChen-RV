# Repack Win-A axis reconciliation (2026-06-24) — authoritative; supersedes confounded framings

Two q8_0-repack experiments this session gave apparently-opposite headlines ("5.5× WIDE-loses
inversion" on rvv VLEN128; "1.95× WIDE-wins, limitation refuted" on K1 VLEN256). They measured
**different axes**. This note reconciles them and labels what is and isn't an RVV1.0 auto-tune.

## The linchpin fact (settled — code + empirical + byte-identical proof)
**On RVV1.0 the repack ALWAYS runs the mf2 fractional core.** `integer_core_lmul="m1"` is set ONLY
inside `if (isRVV0p7)` branches (`RVVRepackStripWidthMaterialization.cpp:124/136/152/168/185`); repack
is NOT in `RVVScheduleDescriptorRegistry` (no measured-best LMUL path). Empirical: the real RVV1.0 stamp
`--tcrv-rvv-materialize-repack-strip-width=march=rv64gcv` on BOTH q4_0 and q8_0 repack GEVM emits
`half_lanes=8`, **integer_core_lmul absent (mf2)**, 0 occurrences of m1. (K1 also showed the real
`rv64gcv_zvl256b` stamp is byte-identical to a hand-set mf2/hl=16 kernel.) The m1 "whole-LMUL" arm is the
**RVV0.7 (xtheadvector) form**, never auto-emitted on RVV1.0.

## Three axes tangled in the word "WIDE"
1. **strip count** at fixed mf2 (hl=8 two-strip vs hl=16 one-strip) — the ONLY clean RVV1.0 tune the
   repack exposes (VLEN-derived: `deriveRepackHalfLanes` 128→8, 256→16).
2. **integer_core_lmul** mf2-vs-m1 — m1 is the RVV0.7 form, NOT an RVV1.0 selection (linchpin above).
3. **VLEN** 128 vs 256, which forces axis 1 (16 e16m1 lanes don't fit one register at VLEN128).
The rvv "5.5×" varied axes 1+2 together (mf2/hl=8 vs **m1**/hl=16); the K1 "1.95×" varied only axis 1
(mf2/hl=8 vs mf2/hl=16). Neither "inversion" nor "limitation refuted" is the right headline.

## q8_0 repack Win-A — the honest, axis-labeled cells
- **Strip-count tune at mf2 (the real RVV1.0 auto-tune):** at VLEN256 the gearbox selects hl=16 (one
  16-lane strip), **1.95× faster** than hl=8 (the gearbox selects the faster arm). At VLEN128 hl=8 is
  forced (16 lanes don't fit one register). Byte-exact (norm 0), oracle PASS. `[q8_0-repack-winA-oracle-FINDING.md]`
- **mf2 vs RVV0.7-m1 form (VLEN128, NOT an RVV1.0 tune):** the mf2 fractional core is **5.5× faster**
  than the m1 whole-LMUL form for q8_0 — i.e. the RVV0.7 form is *worse* here (q8's full-int8 →
  i32-per-position serial `vwadd_wv_i32m4` chain vs mf2's two-ILP i32m2 chains). This is the OPPOSITE of
  q4 (where the m1 form is faster), but it is an **ISA-generation kernel comparison**, not an RVV1.0
  auto-selection. NOT "inversion," NOT "mis-selection."

**Do NOT write** "the tune correctly diverges by quant type" / "inversion" / "mis-selection at VLEN256" —
all presume a tune choice (m1 on RVV1.0) the gearbox does not make.

## SEPARATE one-check audit owed (q4_0 / q4_1 repack Win-A cells)
The same linchpin means the matrix's q4_0 ("WIDE m1 2.11–2.21×") and q4_1 ("WIDE m1 1.80×") repack Win-A
cells are **also mf2-vs-RVV0.7-m1-form numbers, not RVV1.0 auto-tune ablations**. Likely honest reframe
(per advisor, NOT yet verified): the m1 repack kernel is genuinely faster for q4 and was demonstrated e2e
via the **hand-placed `.inc`-swap**, but the auto-gearbox emits mf2 on RVV1.0, so auto-*selection* of that
winner is a separate gap — the same selection-vs-kernel distinction gap-6 already tracks. **AUDIT DONE (2026-06-24):** `winA-e2e/FINDING.md:32-34` confirms the q4_0 e2e WIDE arm = m1 emitted via
`march=rv64gc_xtheadvector` (isRVV0p7 stamp → integer_core_lmul=m1), compiled rv64gcv_zvfh for VLEN128. So the
m1 kernel is HAND-PLACED (not auto-selected); it is genuinely faster for q4 (2.1× micro, **1.10× t16 prefill /
NULL decode** e2e — the micro win largely washes out at e2e, the memory wall again), and auto-SELECTING m1 on
RVV1.0 is an unbuilt gap. Cells now carry the resolved caveat. This is winning-LMUL-CONFIG-differs-by-quant
(motivation for a quant-aware repack cost model), NOT the ledger §5 winning-ALGORITHM-differs-by-PROFILE target.
