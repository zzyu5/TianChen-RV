# T4b M1 — general offline q4_K/q5_K repacker + per-tensor byte-exact certificate (VLEN128 / ssh rvv)

Operational index only (findings live in the task report / journal, not here).

## What
Generalizes the M0 one-shot q4_K pack (commit 663214e9) into a GENERAL offline K-quant weight
repacker (any tensor shape, q4_K AND q5_K) + a per-tensor byte-exact / bounded-ULP certificate
harness. The q8_K activation interleave is promoted to a SHARED helper (byte-identical q4_K/q5_K).
This is M1 of the T4b full-construct e2e campaign: offline repacker + cert ONLY (NOT a ggml dispatch
patch — that is M2). NOT a perf probe.

## Files
- `tools/e2e-harness/board/kquant_repacker.h` — the GENERAL repacker (header-only): `kqr_repack_q4_K`
  (block_q4_Kx16 stride 2304), `kqr_repack_q5_K` (block_q5_Kx16 stride 2816), `kqr_interleave_q8_K`
  (block_q8_Kx4 stride 1168, shared), `kqr_pack_scales16` (shared 6-bit custom scale split). Arbitrary
  shape (nc%16, nr%4, nb=K/256). Derived from the golden emitted kernels, not trial-and-error.
- `tools/e2e-harness/board/kquant_repack_cert.c` — GENERAL cert: any (format, shape) -> repack ->
  OUR golden kernel vs stock ggml grid. INT byte-exact + NORM bounded-ULP.
- `tools/e2e-harness/board/kquant_repack_tool.c` — standalone offline repacker CLI (host-only, no
  libs/ggml): any shape -> block_qX_Kx16 file + geometry + FNV-1a fingerprint + determinism self-check.
- `tools/e2e-harness/board/kquant_repack_cert.sh` — build+run driver (both golden kernels + cert +
  tool; links board libggml-cpu.so).
- `m1_repack_cert_result.log` — captured board run (this cell).

## Reproduce
    BOARD=rvv CORE=8 bash tools/e2e-harness/board/kquant_repack_cert.sh

## Inputs / provenance
- OUR kernels (pre-generated golden emitter lowering; NO local tcrv-opt run, line-B binary untouched):
  q4_K `golden_q4K.c` md5 `b0b5beacac233116b6aaa481e46b8ec4` (block_q4_Kx16/2304);
  q5_K `gemm_q5_K_q8_K.kernel.c` md5 `ba30ba5450f1b9ac4c6f27a0939d7875` (block_q5_Kx16/2816).
  Both from board `/tmp/kquant_tile_s6_ab/`.
- Reference: stock `ggml_vec_dot_q4_K_q8_K` / `ggml_vec_dot_q5_K_q8_K` from
  `/home/ubuntu/llama.cpp-upstream-native/build-gcc15-rv64gcv/bin/libggml-cpu.so`.
- Board scratch: `/tmp/m1_repack_cert/` (nothing to restore; main tree + build/ + ggml A-tree UNTOUCHED).

## Result (this cell's log)
INT cert (byte-exact-integer vs stock ggml): 0 mismatch / 0 ULP, ALL 7 shapes, BOTH q4_K and q5_K.
NORM cert (adversarial fp16 d+dmin, min*bsums active): bounded-ULP, worst rel 1.18e-4 (pure fp32
reassociation, grows with reduction size, benign) => PASS. Standalone tool: q4_K/q5_K/q8_K over
7 shapes incl. 4096x4096, all deterministic=YES; file roundtrip OK.

## Key construction fact (q5_K block_q5_Kx16 scale/qh/qs geometry, stride 2816)
Header @0..256 (d[16]/dmin[16]/scales) is BYTE-IDENTICAL to q4_K (same custom 6-bit split). The delta:
  qh[32][16]  @256 512B : qh[m] col c @ 256 + m*16 + c  (raw block_q5_K.qh, STRAIGHT verbatim interleave)
  qs[128][16] @768 2048B: qs[i] col c @ 768 + i*16 + c  (raw block_q5_K.qs)  <-- qs pushed to @768
The kernel assembles the 5-bit weight as `nibble | (qh_bit<<4)`; for nibble byte i the qh byte is
qh[i%32] and the bit is 2*(i/32) (low nibble) / 2*(i/32)+1 (high nibble) — exactly ggml's u1/u2<<=2
plane walk. A naive guess (qs@256, qh@2304) is WRONG; qh precedes qs. Determined deterministically by
reading the golden kernel's first+last nibble/qh assembly (offsets @768 nibble, @256 qh; @2304 nibble,
@256 qh bits 6/7), not by trial-and-error.
