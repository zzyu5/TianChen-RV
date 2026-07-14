# T4b M0 tracer — q4_K single-tensor single-mul_mat finest line (VLEN128 / ssh rvv)

Operational index only (findings live in the task report / journal, not here).

## What
Finest-line feasibility + numeric-equivalence tracer for T4b full-construct e2e integration:
construct ONE q4_K weight tensor's repack (block_q4_Kx16) + ONE q8_K activation interleave
(block_q8_Kx4), run OUR golden emitted repack-GEMM kernel, compare vs the board's OWN stock
ggml integer path (`ggml_vec_dot_q4_K_q8_K`, linked from libggml-cpu.so). NOT a perf probe.

## Files
- `tools/e2e-harness/board/kquant_repack_verify_q4K.c` — the oracle (minimal one-tensor repack
  + interleave + decomposition probes + INT/NORM certs). NOT a general repacker.
- `tools/e2e-harness/board/kquant_repack_verify_q4K.sh` — build+run driver.
- `m0_q4k_result.log` — captured board run (this cell).

## Reproduce
    BOARD=rvv CORE=8 bash tools/e2e-harness/board/kquant_repack_verify_q4K.sh

## Inputs / provenance
- OUR kernel: golden emitted q4_K repack-GEMM `golden_q4K.c` md5 `b0b5beacac233116b6aaa481e46b8ec4`
  (== pipeline PRE / golden emitter lowering, HEAD 039133ea). Pre-generated; board `/tmp`.
  No local tcrv-opt run (line-B binary untouched).
- Reference: stock `ggml_vec_dot_q4_K_q8_K` from
  `/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin/libggml-cpu.so`.
- Board scratch: `/tmp/m0_q4k_tracer/` (nothing to restore; main tree + build/ + ggml A-tree UNTOUCHED).

## Result (this cell's log)
INT cert: byte-exact-integer vs stock ggml, 0 mismatch / 0 ULP, all 5 shapes.
NORM cert: bounded-ULP (worst rel 2.4e-5, fp32 reassociation only) => PASS.

## Key construction fact (block_q4_Kx16 scale region @+64)
NOT ggml on-disk `scales[12]`; it is ggml repack.cpp's CUSTOM 6-bit layout. Given
`get_scale_min_k4(scales[12]) -> sc[0..7],mn[0..7]`:
  lo strip g<4 @ 64+g*16 ; g>=4 @ 128+(g-4)*16 : byte = (mn[g]&0xF)<<4 | (sc[g]&0xF)
  hi strip sb  @ 192+sb*16 : [1:0]=sc[sb]>>4 [3:2]=mn[sb]>>4 [5:4]=sc[sb+4]>>4 [7:6]=mn[sb+4]>>4
(A naive verbatim copy — the q6_K template — yields a 9x-inflated main term.)
