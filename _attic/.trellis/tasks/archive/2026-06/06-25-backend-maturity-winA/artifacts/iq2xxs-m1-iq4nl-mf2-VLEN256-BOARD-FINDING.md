# k1 (SpacemiT X60, VLEN256) board seal — 2 new VLEN256-default emits, byte-exact

Verifies the two VLEN256-default block-dot emits the gearbox selects from the REAL VLEN
fact (`--tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b`), against ggml's shipped
`_vl256` body. MEASURE-ONLY: no lib/ change, no tcrv-opt rebuild, existing
`build/bin/tcrv-opt` (HEAD 5c572d5c; main tip in task = 5dc0f20e — same emit).

## Emit recipe (host)
- `build/bin/tcrv-opt test/Conversion/RVV/rvv-iq2-xxs-q8-k-block-dot-autotuner-divergence.mlir
   --tcrv-rvv-materialize-schedule=march=rv64gcv_zvl256b --tcrv-rvv-lower-to-emitc
   | mlir-translate-20 --mlir-to-cpp` → iq2_xxs **m1** form (saved `bench/iq2_xxs/ours_vl256_m1.cpp`).
- same with `rvv-iq4-nl-q8-0-block-dot-autotuner-divergence.mlir` → iq4_nl **mf2** form
  (saved `bench/iq4_nl/ours_vl256_mf2.cpp`).
- Host LMUL grep matches the divergence test's VLEN256 FileCheck blocks exactly.

## Board
`ssh k1`, SpacemiT X60, VLEN256 (vlenb=32; isa lists `zve64x zvfh`). clang-18
`-O3 -march=rv64gcv_zfh_zvfh -ffp-contract=fast`, `taskset -c 0-3`, build+run on `/data`
(`/data/k1micro/iq2_xxs_vl256m1/`, `/data/k1micro/iq4_nl_vl256mf2/`), serial. Harness +
`_vl256` ggml_ref reused verbatim from the existing k1-micro bench (only `ours.cpp`
swapped to the new VLEN256 emit). ratio = ggml_ns / ours_ns; >1 ⇒ ours faster.

NOTE: the harness's `RESULT ggml(real,vl128)` printf label is a stale string; the linked
ggml_ref IS the shipped `_vl256` body (iq2_xxs `vluxei16_v_u64m1` 32-lane; iq4_nl mf2
split-16 2-blocks/iter). So this is a fair VLEN256-native fight.

## Results

| kernel | VLEN256 anchor | byte-exact vs ggml `_vl256` | ours ns | ggml `_vl256` ns | ratio | bug? |
|--------|----------------|-----------------------------|--------:|-----------------:|------:|------|
| iq2_xxs | **m1** (`vluxei16_v_i64m1`) | **0.000e+00** (nonzero=1, nonfinite=0) | 10330.8 | 9711.2 | **0.940** | none |
| iq4_nl  | **mf2** (`vsetvl_e8mf2`/`vrgather_i8mf2`) | **0.000e+00** (nonzero=1, nonfinite=0) | 1598.3 | 1717.9 | **1.075** | none |

- **iq2_xxs m1**: byte-exact PASS. The m1 anchor-flip (from the old VLEN128-shaped m2 emit,
  which showed 0.658 on this board) recovers the bulk of the sub-LMUL loss → 0.940 (near
  parity; ggml `_vl256` is m1-32-lane, ours now matches the anchor). No gather bug — the
  i64m1 gather correctly indexes all 4 grid/sign u64 entries at VLMAX=4.
- **iq4_nl mf2**: byte-exact PASS, ratio 1.075 — IDENTICAL to the prior e65edf76 seal,
  confirming the regenerated emit (after the Track-B "真翻转" codebook change) is byte-for-byte
  unchanged. Parity, marginally above (per VLEN-SHAPE-MATCH framing: adopting ggml's own mf2
  gather → parity, NOT a structural-win claim).

## objdump (on-silicon LMUL evidence, `llvm-objdump-18 -d ours.o`)
- **iq2_xxs**: grid/sign gather at `vsetivli zero, 0x4, e64, m1` (i64m1, 4 entries) + index
  load `e16, mf2`; widened product `e16` core. **NO `e64, m2`, NO `e16, m4`** → the VLEN128
  m2 shape is ABSENT. m1 anchor confirmed on hardware.
- **iq4_nl**: codebook core `vsetivli zero, 0x10, e8, mf2` + `vsetvli zero, zero, e8, mf2`
  (mf2 VLMAX=16 = FULL register at VLEN256) + `vrgather.vv` + widened `e16, m1`. **NO
  `e8, m1`, NO `e16, m2`** → the VLEN128 sub-VLMAX m1 form is ABSENT. mf2 anchor confirmed.

## Verdict
Both VLEN256-default new emits byte-exact vs ggml `_vl256`, anchors confirmed by objdump.
No emitter bug. iq2_xxs m1-flip lands at 0.940 (loss largely recovered); iq4_nl mf2 at
1.075 parity (re-confirms e65edf76).
