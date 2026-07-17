# Win-A enrich brick #1: q8_0 block-dot LMUL flips by REAL VLEN fact (DONE host-side)

## What shipped
The q8_0 block-dot integer-core LMUL anchor is now selected by the **real VLEN
capability fact** (`deriveMinimumVLEN` bits), not a 1-bit `hasZvl128b` boolean. The
optimal anchor FLIPS with VLEN, and the flip drives **byte-different emitted C**
(non-NULL — the opposite of a Win-C structural NULL):

| march | minimumVLEN | selected shape | emitted core |
|---|---|---|---|
| `rv64gcv` | 128 | **m2**, f2, elided | `vsetvl_e8m2` / `vwredsum_vs_i16m4` / `vint8m2_t` |
| `rv64gcv_zvl256b` | 256 | **m1**, f2, elided | `vsetvl_e8m1` / `vwredsum_vs_i16m2` / `vint8m1_t` |
| `rv64gc_zve32x` | 0 | m2, f2, robust | (elided pruned; unchanged) |

## Mechanism (all selection-only; NO algorithm/layout/repack change)
- **Single structural quantity = strip VLMAX = VLEN·LMUL/SEW**, computed against the
  anchor's ACTUAL emitted vsetvl spelling (m1/m2 strip e8<lmul>; mf4 strips e32m1 →
  LMUL m1, SEW 32). It drives BOTH:
  - per-block reduction count = `ceil(blockLen / VLMAX)` → m1 goes 2→1 at VLEN256,
  - elided-cover legality = `VLMAX ≥ blockLen` → frees m1-elided at VLEN256, and
    SUBSUMES the old `hasZvl128b` boolean (and the per-kernel `elidedLegalAnchor`).
- **Two VLEN reads, separated on purpose** (`RVVGearboxSchedule.h` enumeration):
  - elision LEGALITY uses the RAW guaranteed minimum (0 → every cover pruned),
  - reduction-RANK uses a `max(minimumVLEN,128)` calibration reference, which
    reproduces the committed VLEN=128 reduction counts EXACTLY (so the non-Zvl128b
    anchor ordering + the other 4 kernels' picks are byte-identical; only ≥256 deviates).
- **Cost CORE stays capability-blind** (`computeBlockDotShapeCostCore` unchanged) — VLEN
  enters ONLY the structural reduction count + legality prune, per the anti-lookup spec.
- **Exact-cost-tie tiebreak** = lighter peak-live vreg footprint (m1=6 < m2=9). At
  VLEN256 m1-elided and m2-elided TIE at cost 1050; the footprint (a real resource fact,
  already computed) breaks it to m1 — so the win is attributable to a resource fact, not
  enumeration array order. Defaults to 0 for non-block-dot candidates → no other pick moves.

## Files
- `include/TianChenRV/Plugin/RVV/RVVGearboxSchedule.h` — VLMAX helpers
  (`getRVVLMULFraction`, `getRVVStripVLMAXElements`, `getRVVBlockDotStripLMUL/SEW`),
  VLMAX-driven `enumerateBlockDotShapeCandidates`, `tieBreakVregCost` on the generic
  candidate + the tie consult in `selectGenericMinCostCandidate`, all 5 block-dot
  enumerate fns re-signatured `(minimumVLEN, budget)`.
- `include/TianChenRV/Plugin/RVV/RVVScheduleMaterialization.h` — `enumerate` lambda
  `(minimumVLEN,budget)`, `minimumVLENAttrName` descriptor field, `stampRVVSchedule`
  stamps the SEMANTIC `minimum_vlen` op attr.
- `lib/Plugin/RVV/RVVScheduleDescriptorRegistry.cpp` — runner derives + threads
  `minimumVLEN`; q8_0 descriptor sets `minimumVLENAttrName="minimum_vlen"`.
- `include/TianChenRV/Dialect/RVV/IR/RVVOps.td` — q8_0 op gains optional `minimum_vlen`.
- `lib/Dialect/RVV/IR/RVVDialectWideningOps.cpp` — q8_0 verifier recomputes elided-cover
  VLMAX legality from `minimum_vlen` (default 128 floor) via the SAME formula the gearbox
  uses (single source of truth); whitelists `minimum_vlen`.
- Tests: unit `RVVQ40Q80ShapeSelectionTest.cpp` (+VLMAX-derivation, +VLEN256 flip,
  +nibble VLEN-invariance regression gate), lit divergence `.mlir` (+VLEN256 stamp +
  emit-divergence), dataflow `.mlir` (reject m1+elided@128, accept m1+elided@256).

## Verification status
- **host (x86_64, no qemu-riscv)**: `ninja check-tianchenrv` (builds + runs ALL unit
  binaries against the changed APIs + full lit) = 716/719 PASS. The 3 failures are
  PRE-EXISTING Script/bundle-metadata tests, confirmed identical on a stashed pristine
  tree — unrelated to this change. q8_0 selection unit suite PASS incl. the VLMAX-
  derivation, the VLEN256 flip, and the VLEN-invariance regression gate over ALL 4
  nibble kernels (q4_0/q4_1/q5_0/q5_1 — picks unmoved at both 128 and 256). emitted-C
  divergence proven at lit level (m1 bytes ≠ m2 bytes).
- **BOARD-PENDING (host CANNOT run any RVV oracle — x86, no qemu)**:
  - [ ] **rvv VLEN128**: m2-elided emitted-C vs scalar oracle byte-exact (no regression).
  - [ ] **k1 VLEN256**: m1-elided emitted-C vs scalar oracle byte-exact. THIS IS THE
        LOAD-BEARING ONE — `vsetvl_e8m1(32)` is correct ONLY when VLMAX≥32 (real VLEN256
        silicon). Memory records a prior VLEN256 fold-back silent regression; do NOT
        assume, must run on k1. Drivers: archive `…/06-22-…/k1-vlen256/tune_block_dot.py`.
  - [ ] (optional perf) micro best-of-N: confirm m1 ~+7% over m2 at VLEN256 / m2 ~+7%
        over m1 at VLEN128 (the calibration that motivated the flip).

## Boundary honesty
Pure backend lowering: same op, same algorithm, same block layout, no repack. Only the
LMUL/elision SHAPE moves with the VLEN capability fact. This is the Win-A (Triton-backend-
style능력驱动 lowering) brick the task asked for — NOT option-2 / repack / 选算法.
