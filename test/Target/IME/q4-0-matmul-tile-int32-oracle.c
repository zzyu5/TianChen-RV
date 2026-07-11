// G4 M1a host int32-EXACT oracle for the format-keyed q4_0 IME GEMM tile.
//
// The tcrv.ime.q4_0_matmul_tile emitter (lib/Plugin/IME/IMEBackendEmissionDriver.cpp)
// lowers the typed region to two structured C helpers -- the q4_0 offset-binary
// nibble DECODE (tcrv_ime_q4_0_dequant_fragment) and the tiled q4_0 int8->int32
// GEMM (tcrv_ime_q4_0_vmadot_matmul) that reduces via the FOUNDATION-validated
// vmadot MAC leaf. The vmadot INSTRUCTION runs only on real K1 (M1b board seal),
// so this host oracle substitutes a SCALAR reference of the batched vmadot MAC
// with the SAME int8->int32 semantics the seal validates bit-exact. The DECODE + the
// int32 MAC arithmetic (the "what") are validated here on host; the vmadot "how"
// is board-proven at M1b.
//
// Two independent int32-EXACT checks (ZERO-MODEL: the reference re-derives the
// decode + the logical A/W matrices from the raw packed bytes and does a PLAIN
// triple-loop GEMM, a different code path than the tiled fragment kernel):
//   (1) DECODE check: the kernel decode == the canonical ggml q4_0 quant (nibble
//       - 8) for every nibble, bit-exact int8.
//   (2) MAC check: the tiled kernel int32 output == the plain-GEMM reference over
//       the independently-decoded logical matrices, bit-exact int32.
//
// Build + run:
//   cc -O2 -std=c11 q4-0-matmul-tile-int32-oracle.c -o /tmp/q40oracle && /tmp/q40oracle
//
// Exit 0 + "ORACLE PASS" iff both checks are 0-diff over the whole test grid.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// (A) The emitted structured helpers (mirror of the M1a emitter output; the ONLY
// asm leaf tcrv_ime_vmadot_mac_kloop is replaced by the scalar-vmadot substitute
// below). These are byte-for-byte the C the IME emitter emits for the region.
// ---------------------------------------------------------------------------

// The scalar substitute for the BATCHED register-resident vmadot MAC leaf:
// frag[4x4] int32 = Sum_kf A_kf[4x8] . B_kf[4x8]^T over `kt` contiguous 32B
// fragments. This is the exact int32 contract the real batched vmadot loop
// satisfies (v2/v3 accumulate in-register across the K/8 loop; single vsetvli;
// one store -- proven bit-exact on K1 at M1b). It is int32-identical to the
// per-fragment `acc[r] += vmadot(A_kf,B_kf)[r]` form: same reductions, summed in
// the same kf order.
static inline void tcrv_ime_vmadot_mac_kloop(const int8_t *A, const int8_t *B,
                                             long kt, int32_t *frag) {
  int32_t acc[16];
  for (int r = 0; r < 16; ++r) acc[r] = 0;
  for (long kf = 0; kf < kt; ++kf) {
    const int8_t *Af = A + kf * 32;
    const int8_t *Bf = B + kf * 32;
    for (int m = 0; m < 4; ++m)
      for (int n = 0; n < 4; ++n) {
        int32_t s = 0;
        for (int k = 0; k < 8; ++k)
          s += (int32_t)Af[m * 8 + k] * (int32_t)Bf[n * 8 + k];
        acc[m * 4 + n] += s;
      }
  }
  for (int r = 0; r < 16; ++r) frag[r] = acc[r];
}

// The q4_0 offset-binary nibble DECODE (identical to the emitted decode core).
static inline void tcrv_ime_q4_0_dequant_fragment(const uint8_t *blk,
                                                  int8_t *out) {
  const uint8_t *qs = blk + 2; // past the 2-byte fp16 d
  for (int j = 0; j < 16; ++j) {
    out[j] = (int8_t)((int)(qs[j] & 0x0F) - 8);
    out[j + 16] = (int8_t)((int)(qs[j] >> 4) - 8);
  }
}

// The tiled q4_0 int8->int32 GEMM (identical to the emitted tiled kernel). The
// tile's kt q4_0 blocks are decoded into a contiguous int8 fragment buffer, then
// ONE register-resident batched MAC runs over the K/8 loop (single vsetvli, one
// store) -- int32-identical to the old per-fragment form.
static void tcrv_ime_q4_0_vmadot_matmul(const int8_t *Apack, const uint8_t *Bq4,
                                        int32_t *C, long M, long N, long K) {
  const long mt = M / 4, nt = N / 4, kt = K / 8;
  const long q40_block_bytes = 18;
  for (long mi = 0; mi < mt; ++mi) {
    const int8_t *Arow = Apack + (long)mi * 4 * K;
    for (long nj = 0; nj < nt; ++nj) {
      const uint8_t *Bcol = Bq4 + (long)nj * kt * q40_block_bytes;
      int8_t Bdec[kt * 32];
      for (long kf = 0; kf < kt; ++kf)
        tcrv_ime_q4_0_dequant_fragment(Bcol + kf * q40_block_bytes,
                                       Bdec + kf * 32);
      int32_t frag[16];
      tcrv_ime_vmadot_mac_kloop(Arow, Bdec, kt, frag);
      for (long r = 0; r < 4; ++r)
        for (long c = 0; c < 4; ++c)
          C[(long)(mi * 4 + r) * N + (nj * 4 + c)] += frag[r * 4 + c];
    }
  }
}

// ---------------------------------------------------------------------------
// (B) The INDEPENDENT reference (ZERO-MODEL): canonical ggml q4_0 quant + a plain
// triple-loop GEMM over logical matrices re-derived from the raw packed bytes.
// ---------------------------------------------------------------------------

// The canonical ggml q4_0 integer quant of nibble `lo`/`hi` (before the * d scale):
// (val & 15) - 8 / (val >> 4) - 8 (test/Target/Scalar/dequantize-row-q4-0-scalar).
static int ggml_ref_q4_0_low(uint8_t byte) { return (int)(byte & 15) - 8; }
static int ggml_ref_q4_0_high(uint8_t byte) { return (int)(byte >> 4) - 8; }

// Logical weight W[n][k] re-derived from the fragment-major q4_0 pack, using the
// canonical decode + the fragment index mapping (n%4)*8 + (k%8), block k/8 of
// col-tile n/4. A different code path than the kernel's tiled decode.
static int ref_W(const uint8_t *Bq4, long N, long K, long n, long k) {
  (void)N;
  const long kt = K / 8;
  const long q40_block_bytes = 18;
  long nj = n / 4, nl = n % 4;
  long kf = k / 8, kl = k % 8;
  const uint8_t *blk = Bq4 + (nj * kt + kf) * q40_block_bytes;
  const uint8_t *qs = blk + 2;
  long idx = nl * 8 + kl; // Bframe index
  if (idx < 16) return ggml_ref_q4_0_low(qs[idx]);
  return ggml_ref_q4_0_high(qs[idx - 16]);
}

// Logical activation A[m][k] re-derived from the fragment-major int8 pack.
static int ref_A(const int8_t *Apack, long K, long m, long k) {
  long mi = m / 4, ml = m % 4;
  long kf = k / 8, kl = k % 8;
  return (int)Apack[mi * 4 * K + kf * 32 + ml * 8 + kl];
}

int main(void) {
  const long M = 8, N = 8, K = 64; // mt=2 nt=2 kt=8; K % 32 == 0 (whole q4_0 blks)
  const long kt = K / 8;
  const long q40_block_bytes = 18;

  srand(1234567u);

  // Fragment-major int8 activation pack: (M/4) row-tiles, kt fragments of 32B.
  long apack_bytes = (M / 4) * kt * 32;
  int8_t *Apack = (int8_t *)malloc(apack_bytes);
  for (long i = 0; i < apack_bytes; ++i) Apack[i] = (int8_t)(rand() % 256 - 128);

  // Fragment-major q4_0 weight pack: (N/4) col-tiles, kt 18B blocks each.
  long bq4_bytes = (N / 4) * kt * q40_block_bytes;
  uint8_t *Bq4 = (uint8_t *)malloc(bq4_bytes);
  for (long i = 0; i < bq4_bytes; ++i) Bq4[i] = (uint8_t)(rand() % 256);

  // --- Check (1): DECODE bit-exact vs canonical ggml quant --------------------
  long decode_mismatch = 0, decode_total = 0;
  for (long blk = 0; blk < (N / 4) * kt; ++blk) {
    const uint8_t *b = Bq4 + blk * q40_block_bytes;
    int8_t frame[32];
    tcrv_ime_q4_0_dequant_fragment(b, frame);
    const uint8_t *qs = b + 2;
    for (int j = 0; j < 16; ++j) {
      decode_total += 2;
      if ((int)frame[j] != ggml_ref_q4_0_low(qs[j])) decode_mismatch++;
      if ((int)frame[j + 16] != ggml_ref_q4_0_high(qs[j])) decode_mismatch++;
    }
  }

  // --- Check (2): tiled kernel int32 output vs plain-GEMM reference ------------
  int32_t *Ck = (int32_t *)calloc((size_t)M * N, sizeof(int32_t));
  tcrv_ime_q4_0_vmadot_matmul(Apack, Bq4, Ck, M, N, K);

  long mac_mismatch = 0;
  int32_t maxabs = 0;
  for (long m = 0; m < M; ++m)
    for (long n = 0; n < N; ++n) {
      int64_t ref = 0;
      for (long k = 0; k < K; ++k)
        ref += (int64_t)ref_A(Apack, K, m, k) * (int64_t)ref_W(Bq4, N, K, n, k);
      if ((int64_t)Ck[m * N + n] != ref) mac_mismatch++;
      int32_t a = Ck[m * N + n] < 0 ? -Ck[m * N + n] : Ck[m * N + n];
      if (a > maxabs) maxabs = a;
    }

  printf("decode nibbles: %ld/%ld bit-exact (mismatch=%ld)\n",
         decode_total - decode_mismatch, decode_total, decode_mismatch);
  printf("int32 MAC: %ld/%ld tiles bit-exact (mismatch=%ld, max|C|=%d)\n",
         (M * N) - mac_mismatch, M * N, mac_mismatch, maxabs);

  free(Apack);
  free(Bq4);
  free(Ck);
  if (decode_mismatch == 0 && mac_mismatch == 0) {
    printf("ORACLE PASS: q4_0 decode + int32 vmadot MAC are int32-EXACT\n");
    return 0;
  }
  printf("ORACLE FAIL\n");
  return 1;
}
