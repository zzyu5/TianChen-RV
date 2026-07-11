// G4 M2 K1 SILICON SEAL for the format-keyed q8_0 IME GEMM tile.
//
// This is the board-seal counterpart of the M2 host oracle
// (q8-0-matmul-tile-int32-oracle.c). It is BYTE-FOR-BYTE identical to that
// oracle -- the q8_0 DIRECT int8 DECODE, the tiled matmul, the ZERO-MODEL
// plain-GEMM reference, and the test grid are all unchanged -- with EXACTLY ONE
// difference: the scalar substitute for the batched vmadot MAC leaf
// (tcrv_ime_vmadot_mac_kloop) is replaced by the REAL `vmadot` inline-asm leaf
// emitted VERBATIM by the IME backend emitter
// (lib/Plugin/IME/IMEBackendEmissionDriver.cpp macKloopHelperBody(), the SAME leaf
// the q4_0 tile reuses). So on real K1 silicon the tiled kernel reduces through
// the actual `vmadot` instruction (encoding 0xe210312b) in a register-resident
// K/8 loop, and check (2) below is now a HARDWARE validation of the vmadot
// int8->int32 MAC semantics over q8_0 weights.
//
// The batched vmadot leaf contract (identical to the M2 scalar shim it replaces):
// over `kt` contiguous 4x8 A/B fragments, v2/v3 accumulate frag = Sum_kf
// A_kf . B_kf^T (int32), single vsetvli, one store. Per-fragment:
//   A: (4,8) int8 row-major -> v0 ; B: stored (4,8) int8 -> v1 (== B^T)
//   frag: (4,4) int32 (v2/v3), frag[i][j] += sum_k A_kf[i][k]*B_kf[j][k]
// pinned by `vsetvli e8,m1` => vl=32 at VLEN=256 => the 4x4x8 MAC unit.
//
// Build (SpacemiT/board toolchain that assembles `vmadot`; on this K1 the stock
// binutils encodes it once the version-pinned march token unlocks the opcode):
//   gcc -O2 -std=c11 -march=rv64gcv_xsmtvdotii1p0 -mabi=lp64d \
//       q8-0-matmul-tile-int32-k1seal.c -o /tmp/q80seal
// Run pinned to IME harts (X60: IME present on harts 0-3, hart 4 SIGILLs):
//   taskset -c 0-3 /tmp/q80seal
//
// objdump golden: the leaf must disassemble to `vmadot v2,v0,v1` = 0xe210312b.
// Exit 0 + "SEAL PASS" iff both DECODE (int8) and int32 MAC checks are 0-diff.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// (A) The emitted structured helpers (mirror of the M2 emitter output). The
// asm leaf below is the EMITTER-VERBATIM batched `vmadot` MAC (macKloopHelperBody(),
// signed vmadot), the ONE justified instruction leaf; all surrounding dataflow is
// the same structured C the IME emitter emits for the region.
// ---------------------------------------------------------------------------

// tcrv_ime.asm_leaf=tcrv_ime_vmadot_mac_kloop batched_kloop mac=4x4x8 elem_in=int8 accum=int32 ime_op=vmadot register_resident_accumulate=1 single_vsetvli=1 store_once=1
static inline void tcrv_ime_vmadot_mac_kloop(const int8_t *A, const int8_t *B, long kt, int32_t *frag) {
  __asm__ volatile(
      "vsetvli   t0, zero, e8, m1, ta, ma   \n\t"
      "vmv.v.i   v2, 0                       \n\t"
      "vmv.v.i   v3, 0                       \n\t"
      "mv        t2, %[kt]                   \n\t"
      "mv        t3, %[pa]                   \n\t"
      "mv        t4, %[pb]                   \n\t"
      "1:                                    \n\t"
      "vle8.v    v0, (t3)                    \n\t"
      "vle8.v    v1, (t4)                    \n\t"
      "vmadot    v2, v0, v1                 \n\t"
      "addi      t3, t3, 32                  \n\t"
      "addi      t4, t4, 32                  \n\t"
      "addi      t2, t2, -1                  \n\t"
      "bnez      t2, 1b                      \n\t"
      "vsetvli   t0, zero, e32, m1, ta, ma   \n\t"
      "vse32.v   v2, (%[pf])                 \n\t"
      "addi      t5, %[pf], 32               \n\t"
      "vse32.v   v3, (t5)                    \n\t"
      :
      : [pa] "r"(A), [pb] "r"(B), [kt] "r"(kt), [pf] "r"(frag)
      : "t0", "t2", "t3", "t4", "t5", "v0", "v1", "v2", "v3", "memory");
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

// The tiled q8_0 int8->int32 GEMM (identical to the emitted tiled kernel). The
// tile's kt q8_0 blocks are decoded into a contiguous int8 fragment buffer, then
// ONE register-resident batched MAC runs over the K/8 loop (single vsetvli, one
// store) -- int32-identical to the old per-fragment form.
static void tcrv_ime_q8_0_vmadot_matmul(const int8_t *Apack, const uint8_t *Bq8,
                                        int32_t *C, long M, long N, long K) {
  const long mt = M / 4, nt = N / 4, kt = K / 8;
  const long q80_block_bytes = 34;
  for (long mi = 0; mi < mt; ++mi) {
    const int8_t *Arow = Apack + (long)mi * 4 * K;
    for (long nj = 0; nj < nt; ++nj) {
      const uint8_t *Bcol = Bq8 + (long)nj * kt * q80_block_bytes;
      int8_t Bdec[kt * 32];
      for (long kf = 0; kf < kt; ++kf)
        tcrv_ime_q8_0_dequant_fragment(Bcol + kf * q80_block_bytes,
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
    printf("SEAL PASS: q8_0 decode + real-vmadot int32 MAC are int32-EXACT on K1\n");
    return 0;
  }
  printf("SEAL FAIL\n");
  return 1;
}
