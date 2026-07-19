# q8_0 repack-GEVM vs the DEPLOYED ggml opponent — honest verdict

**PRD ask #1**: replace the q8 "vs deployed opponent" pending (master had a
projection 0.937×1.4) with a DIRECT board measurement.

## Finding (NOT a favourable ratio — an honest structural block)

The obvious opponent is ggml's own x16 repack `ggml_gemv_q8_0_16x1_q8_0`
(`.../arch/riscv/repack.cpp`). Its weight struct `block_q8_0x16 = { ggml_half d[16];
int8_t qs[512]; }` (544 B) is BYTE-IDENTICAL to our x16 repack layout, and the
activation `block_q8_0` (34 B) matches — so the SAME weight/act bytes feed both.
Compiled byte-for-byte, compiler-symmetric (clang 17.0.6, `-O3 -march=rv64gcv_zvfh`),
run @rvv (VLEN128) against our DEPLOYED m1 GEVM + a scalar oracle, 2-seed cold reps=15:

| footprint | ours m1 exact | ggml 16x1 exact | ours/ggml |
|---|---|---|---|
| M=256 K=1024 | PASS | **FAIL** | 0.18–0.26 (meaningless) |
| M=2048 K=4096 | PASS | **FAIL** | 0.23–0.25 (meaningless) |
| M=8192 K=4096 (DRAM) | PASS | **FAIL** | 0.19–0.25 (meaningless) |

**The ggml arch `16x1` kernel MISCOMPUTES on VLEN128** (ggml_exact = FAIL, all cells).
It is a **VLEN256-tuned** kernel: every op runs `vl = 16` on `i8mf2`/`i16m1`/`f32m2`,
whose VLMAX at VLEN128 is only 8 (`128 × ½ / 8 = 8`). vsetvli clamps vl to 8, so it
processes only 8 of each 16-column group and stores garbage for the other 8. The
ratio (~4–5× "faster") is therefore INVALID — it is timing a broken partial kernel,
NOT a fair opponent.

## Consequence

On the VLEN128 rvv board, ggml does NOT deploy the arch `16x1` repack (it would
miscompile). The true VLEN128 deployed q8_0 repack opponent is a DIFFERENT variant
(the `4x8`/`4x4` interleave in the generic `repack.cpp`, or the SpacemiT-tuned
`spacemit/repack.cpp`, or the `ggml_vec_dot_q8_0_q8_0` block-dot fallback), each with
its OWN weight layout (block_q8_0x8 / x4, blocklen 4) requiring its own
`ggml_quantize_mat` repack + matched data. Identifying which variant ggml's runtime
dispatch actually selects on VLEN128 — and building a BYTE-EXACT opponent against it
— is the remaining continuation for the q8 vs-opponent verdict.

This is exactly why the master had the q8 vs-opponent as *pending* with only a
projection: the naive x16 opponent is not VLEN128-valid, and no fabricated ratio is
recorded here. The m1-vs-mf2 measured-table flip (both OURS, byte-exact) stands on
its own board evidence (r51f FINDING); this note refutes the naive x16 projection
rather than replacing it with a fake number.

## What IS established

- ggml's x16 repack layout == our x16 layout (byte-identical) — a future VLEN256
  board could measure ours vs `16x1` directly and byte-exact.
- Our DEPLOYED m1 GEVM is byte-exact (3-arm) — the correctness carrier on VLEN128.
