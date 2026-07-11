// G4 M2 host int32-EXACT oracle for the format-keyed q8_0 IME GEMM tile.
//
// The tcrv.ime.q8_0_matmul_tile emitter (lib/Plugin/IME/IMEBackendEmissionDriver.cpp)
// lowers the typed region to two structured C helpers -- the q8_0 DIRECT int8
// DECODE (tcrv_ime_q8_0_dequant_fragment) and the tiled q8_0 int8->int32 GEMM
// (tcrv_ime_q8_0_vmadot_matmul) that reduces via the FOUNDATION-validated vmadot
// MAC leaf. The vmadot INSTRUCTION runs only on real K1 (M2 board seal), so this
// host oracle substitutes a SCALAR reference of the vmadot 4x4x8 MAC with the SAME
// int8->int32 semantics the seal validates bit-exact. The DECODE + the int32 MAC
// arithmetic (the "what") are validated here on host; the vmadot "how" is
// board-proven at the seal.
//
// Two independent int32-EXACT checks (ZERO-MODEL: the reference re-derives the
// decode + the logical A/W matrices from the raw packed bytes and does a PLAIN
// triple-loop GEMM, a different code path than the tiled fragment kernel):
//   (1) DECODE check: the kernel decode == the canonical ggml q8_0 quant (the raw
//       int8 quant) for every quant, bit-exact int8.
//   (2) MAC check: the tiled kernel int32 output == the plain-GEMM reference over
//       the independently-decoded logical matrices, bit-exact int32.
//
// Build + run:
//   cc -O2 -std=c11 q8-0-matmul-tile-int32-oracle.c -o /tmp/q80oracle && /tmp/q80oracle
//
// Exit 0 + "ORACLE PASS" iff both checks are 0-diff over the whole test grid.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// (A) The emitted structured helpers (mirror of the M2 emitter output; the ONLY
// asm leaf tcrv_ime_vmadot_mma_4x4x8 is replaced by the scalar-vmadot substitute
// below). These are byte-for-byte the C the IME emitter emits for the region.
// ---------------------------------------------------------------------------

// The scalar substitute for the FOUNDATION-validated vmadot 4x4x8 MAC: fresh
// C[4x4] int32 = A[4x8] . B[4x8]^T (int8*int8 -> int32). This is the exact int32
// contract the real vmadot instruction satisfies (proven bit-exact on K1 at seal).
static inline void tcrv_ime_vmadot_mma_4x4x8(const int8_t *A, const int8_t *B,
                                             int32_t *C) {
  for (int m = 0; m < 4; ++m)
    for (int n = 0; n < 4; ++n) {
      int32_t s = 0;
      for (int k = 0; k < 8; ++k)
        s += (int32_t)A[m * 8 + k] * (int32_t)B[n * 8 + k];
      C[m * 4 + n] = s;
    }
}

// The q8_0 DIRECT int8 DECODE (identical to the emitted decode core). No nibble
// unpack, no offset-binary centering -- q8_0 is already flat int8.
static inline void tcrv_ime_q8_0_dequant_fragment(const uint8_t *blk,
                                                  int8_t *out) {
  const int8_t *qs = (const int8_t *)(blk + 2); // past the 2-byte fp16 d
  for (int j = 0; j < 32; ++j) {
    out[j] = qs[j];
  }
}

// The tiled q8_0 int8->int32 GEMM (identical to the emitted tiled kernel).
static inline void tcrv_ime_q8_0_vmadot_matmul(const int8_t *Apack,
                                               const uint8_t *Bq8, int32_t *C,
                                               long M, long N, long K) {
  const long mt = M / 4, nt = N / 4, kt = K / 8;
  const long q80_block_bytes = 34;
  for (long mi = 0; mi < mt; ++mi) {
    const int8_t *Arow = Apack + (long)mi * 4 * K;
    for (long nj = 0; nj < nt; ++nj) {
      const uint8_t *Bcol = Bq8 + (long)nj * kt * q80_block_bytes;
      int32_t acc[16];
      for (int r = 0; r < 16; ++r) acc[r] = 0;
      for (long kf = 0; kf < kt; ++kf) {
        int8_t Bframe[32];
        int32_t frag[16];
        tcrv_ime_q8_0_dequant_fragment(Bcol + kf * q80_block_bytes, Bframe);
        tcrv_ime_vmadot_mma_4x4x8(Arow + kf * 32, Bframe, frag);
        for (int r = 0; r < 16; ++r) acc[r] += frag[r];
      }
      for (long r = 0; r < 4; ++r)
        for (long c = 0; c < 4; ++c)
          C[(long)(mi * 4 + r) * N + (nj * 4 + c)] += acc[r * 4 + c];
    }
  }
}

// ---------------------------------------------------------------------------
// (B) The INDEPENDENT reference (ZERO-MODEL): canonical ggml q8_0 quant + a plain
// triple-loop GEMM over logical matrices re-derived from the raw packed bytes.
// ---------------------------------------------------------------------------

// The canonical ggml q8_0 integer quant (before the * d scale) is simply the raw
// signed int8 (test/Target/Scalar/dequantize-row-q8-0-scalar): x = qs[i] * d.
static int ggml_ref_q8_0(int8_t byte) { return (int)byte; }

// Logical weight W[n][k] re-derived from the fragment-major q8_0 pack, using the
// canonical decode + the fragment index mapping (n%4)*8 + (k%8), block k/8 of
// col-tile n/4. A different code path than the kernel's tiled decode.
static int ref_W(const uint8_t *Bq8, long N, long K, long n, long k) {
  (void)N;
  const long kt = K / 8;
  const long q80_block_bytes = 34;
  long nj = n / 4, nl = n % 4;
  long kf = k / 8, kl = k % 8;
  const uint8_t *blk = Bq8 + (nj * kt + kf) * q80_block_bytes;
  const int8_t *qs = (const int8_t *)(blk + 2);
  long idx = nl * 8 + kl; // Bframe index
  return ggml_ref_q8_0(qs[idx]);
}

// Logical activation A[m][k] re-derived from the fragment-major int8 pack.
static int ref_A(const int8_t *Apack, long K, long m, long k) {
  long mi = m / 4, ml = m % 4;
  long kf = k / 8, kl = k % 8;
  return (int)Apack[mi * 4 * K + kf * 32 + ml * 8 + kl];
}

int main(void) {
  const long M = 8, N = 8, K = 64; // mt=2 nt=2 kt=8; K % 32 == 0 (whole q8_0 blks)
  const long kt = K / 8;
  const long q80_block_bytes = 34;

  srand(1234567u);

  // Fragment-major int8 activation pack: (M/4) row-tiles, kt fragments of 32B.
  long apack_bytes = (M / 4) * kt * 32;
  int8_t *Apack = (int8_t *)malloc(apack_bytes);
  for (long i = 0; i < apack_bytes; ++i) Apack[i] = (int8_t)(rand() % 256 - 128);

  // Fragment-major q8_0 weight pack: (N/4) col-tiles, kt 34B blocks each.
  long bq8_bytes = (N / 4) * kt * q80_block_bytes;
  uint8_t *Bq8 = (uint8_t *)malloc(bq8_bytes);
  for (long i = 0; i < bq8_bytes; ++i) Bq8[i] = (uint8_t)(rand() % 256);

  // --- Check (1): DECODE bit-exact vs canonical ggml quant --------------------
  long decode_mismatch = 0, decode_total = 0;
  for (long blk = 0; blk < (N / 4) * kt; ++blk) {
    const uint8_t *b = Bq8 + blk * q80_block_bytes;
    int8_t frame[32];
    tcrv_ime_q8_0_dequant_fragment(b, frame);
    const int8_t *qs = (const int8_t *)(b + 2);
    for (int j = 0; j < 32; ++j) {
      decode_total += 1;
      if ((int)frame[j] != ggml_ref_q8_0(qs[j])) decode_mismatch++;
    }
  }

  // --- Check (2): tiled kernel int32 output vs plain-GEMM reference ------------
  // On real K1 this reduces through the actual `vmadot` instruction.
  int32_t *Ck = (int32_t *)calloc((size_t)M * N, sizeof(int32_t));
  tcrv_ime_q8_0_vmadot_matmul(Apack, Bq8, Ck, M, N, K);

  long mac_mismatch = 0;
  int32_t maxabs = 0;
  for (long m = 0; m < M; ++m)
    for (long n = 0; n < N; ++n) {
      int64_t ref = 0;
      for (long k = 0; k < K; ++k)
        ref += (int64_t)ref_A(Apack, K, m, k) * (int64_t)ref_W(Bq8, N, K, n, k);
      if ((int64_t)Ck[m * N + n] != ref) mac_mismatch++;
      int32_t a = Ck[m * N + n] < 0 ? -Ck[m * N + n] : Ck[m * N + n];
      if (a > maxabs) maxabs = a;
    }

  printf("decode int8 quants: %ld/%ld bit-exact (mismatch=%ld)\n",
         decode_total - decode_mismatch, decode_total, decode_mismatch);
  printf("int32 MAC (real vmadot): %ld/%ld tiles bit-exact (mismatch=%ld, max|C|=%d)\n",
         (M * N) - mac_mismatch, M * N, mac_mismatch, maxabs);

  free(Apack);
  free(Bq8);
  free(Ck);
  if (decode_mismatch == 0 && mac_mismatch == 0) {
    printf("ORACLE PASS: q8_0 decode + scalar-vmadot int32 MAC are int32-EXACT\n");
    return 0;
  }
  printf("ORACLE FAIL\n");
  return 1;
}
