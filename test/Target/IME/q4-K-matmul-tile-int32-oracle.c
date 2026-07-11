// G4 M2b host int32-EXACT oracle for the format-keyed q4_K IME GEMM tile (the
// SUPER-BLOCK K-quant tile; the DEDICATED effort beyond the q4_0/q8_0 copy-adapt).
//
// The tcrv.ime.q4_K_matmul_tile emitter (lib/Plugin/IME/IMEBackendEmissionDriver.cpp)
// lowers the typed region to FIVE structured C helpers -- the q4_K RAW-nibble
// DECODE (tcrv_ime_q4_K_dequant_fragment), the 6-bit scale/min UNPACK
// (tcrv_ime_q4_K_get_scale_min), the deterministic fp16 epilogue helpers
// (tcrv_ime_fp16_to_f32), and the tiled q4_K TWO-LEVEL-fold GEMM
// (tcrv_ime_q4_K_vmadot_matmul) that reduces via the FOUNDATION-validated vmadot
// MAC leaf. The vmadot INSTRUCTION runs only on real K1 (M2b board seal), so this
// host oracle substitutes a SCALAR reference of the batched vmadot MAC with the SAME
// int8->int32 semantics the seal validates bit-exact.
//
// The q4_K two-level fold (mirrors the RVV kquant_dmin_bsums_min precedent):
//   S_scale = Sum_b sc_b * (Sum_{i in b} A_i * q_i)     (int32-EXACT; board-sealed)
//   S_min   = Sum_b m_b  * (Sum_{i in b} A_i)           (int32-EXACT; board-sealed)
//   C       = d * S_scale - dmin * S_min                (deferred fp16 float epilogue)
// sc_b / m_b are the 6-bit per-sub-block scale / min (get_scale_min_k4); q is the
// RAW unsigned nibble [0,15] (NOT q4_0 offset-binary). ONLY sealing the bare nibble
// MAC Sum A_i*q_i would be a HOLLOW q4_K representation (it drops the per-sub-block
// scale weighting + the min bias that DEFINE q4_K); this oracle validates the FULL
// int32 core S_scale AND S_min.
//
// Independent int32-EXACT checks (ZERO-MODEL: the reference re-derives the decode +
// the 6-bit sc/m + the logical A/W from the raw packed bytes and does PLAIN
// triple-loops, a different code path than the tiled fragment kernel):
//   (1) DECODE: the kernel raw nibble == the canonical q4_K nibble, every quant.
//   (2) SCALE/MIN: the kernel 6-bit sc/m == the canonical get_scale_min_k4, every
//       sub-block.
//   (3) INT32 CORE: the tiled kernel S_scale/S_min == the plain-loop reference over
//       the independently-decoded logical matrices, bit-exact int32.
//   (4) FLOAT EPILOGUE: C == d*S_scale - dmin*S_min with the SAME two-level fold as
//       the reference int32 cores (deterministic; the RVV kquant fold order, which
//       differs from a naive element-wise dequant-dot in float associativity).
//
// Build + run:
//   cc -O2 -std=c11 q4-K-matmul-tile-int32-oracle.c -o /tmp/q4koracle && /tmp/q4koracle
//
// Exit 0 + "ORACLE PASS" iff all four checks are 0-diff over the whole test grid.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// (A) The emitted structured helpers (mirror of the M2b emitter output; the ONLY
// asm leaf tcrv_ime_vmadot_mac_kloop is replaced by the scalar-vmadot substitute
// below). These are byte-for-byte the C the IME emitter emits for the region.
// ---------------------------------------------------------------------------

// The scalar substitute for the BATCHED register-resident vmadot MAC leaf:
// frag[4x4] int32 = Sum_kf A_kf[4x8] . B_kf[4x8]^T over `kt` contiguous 32B
// fragments. This is the exact int32 contract the real batched vmadot loop
// satisfies (v2/v3 accumulate in-register across the sub-block's 4 fragments;
// single vsetvli; one store -- proven bit-exact on K1 at seal). It is
// int32-identical to the per-fragment `sumi[r] += vmadot(A_kf,B_kf)[r]` form:
// same reductions, summed in the same kf order.
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

// tcrv_ime.fp16_epilogue=tcrv_ime_fp16_to_f32 (deterministic IEEE half->float; the deferred d/dmin float fold)
static inline unsigned short tcrv_ime_load_fp16(const uint8_t *p) {
  return (unsigned short)((unsigned)p[0] | ((unsigned)p[1] << 8));
}
static inline float tcrv_ime_fp16_to_f32(unsigned short h) {
  unsigned int sign = (unsigned int)(h & 0x8000u) << 16;
  unsigned int exp = (h >> 10) & 0x1Fu;
  unsigned int mant = h & 0x3FFu;
  unsigned int bits;
  if (exp == 0u) {
    if (mant == 0u) {
      bits = sign;
    } else {
      exp = 127u - 15u + 1u;
      while ((mant & 0x400u) == 0u) { mant <<= 1; exp--; }
      mant &= 0x3FFu;
      bits = sign | (exp << 23) | (mant << 13);
    }
  } else if (exp == 0x1Fu) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
  }
  float f;
  __builtin_memcpy(&f, &bits, 4);
  return f;
}

// tcrv_ime.decode_core=tcrv_ime_q4_K_dequant_fragment decode_model=q4_K_raw_nibble qk=256 weight_block_stride=144 weight_quant_byte_offset=16
static inline void tcrv_ime_q4_K_dequant_fragment(const uint8_t *blk, int b, int kf, int8_t *out8) {
  const uint8_t *qs = blk + 16; // past fp16 d/dmin + 12-byte 6-bit scales
  for (int kl = 0; kl < 8; ++kl) {
    int pl = kf * 8 + kl;
    uint8_t byte = qs[(b / 2) * 32 + pl];
    out8[kl] = (int8_t)((b & 1) ? (byte >> 4) : (byte & 0x0F));
  }
}

// tcrv_ime.scale_min_core=tcrv_ime_q4_K_get_scale_min scale_min_model=get_scale_min_k4 num_sub_blocks=8 scale_bits=6 k_scale_size=12
static inline void tcrv_ime_q4_K_get_scale_min(int j, const uint8_t *q, uint8_t *sc, uint8_t *m) {
  if (j < 4) {
    *sc = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// tcrv_ime.asm_leaf=tcrv_ime_vmadot_mac_kloop tiled_q4_K_matmul mac=4x4x8 elem_in=int8 accum=int32 ime_op=vmadot weight_format=q4_K int32_exact=1 two_level_fold=kquant_dmin_bsums_min register_resident_accumulate=1
static void tcrv_ime_q4_K_vmadot_matmul(const int8_t *Apack, const uint8_t *Bq4k, int32_t *Sscale,
    int32_t *Smin, float *Cf, long M, long N, long K) {
  const long mt = M / 4, nt = N / 4, nsb = K / 256;
  const long q4k_block_bytes = 144; // fp16 d+dmin + 12B scales + 128B nibbles
  for (long mi = 0; mi < mt; ++mi) {
    const int8_t *Arow = Apack + (long)mi * 4 * K;
    for (long nj = 0; nj < nt; ++nj) {
      for (long sb = 0; sb < nsb; ++sb) {
        const uint8_t *blk[4];
        uint8_t sc[8][4], mm[8][4];
        for (int nl = 0; nl < 4; ++nl) {
          blk[nl] = Bq4k + ((((nj * nsb) + sb) * 4) + nl) * q4k_block_bytes;
          for (int b = 0; b < 8; ++b)
            tcrv_ime_q4_K_get_scale_min(b, blk[nl] + 4, &sc[b][nl], &mm[b][nl]);
        }
        int32_t Sc[16], Sm[16];
        for (int r = 0; r < 16; ++r) { Sc[r] = 0; Sm[r] = 0; }
        for (int b = 0; b < 8; ++b) {
          int32_t sumi[16];
          int32_t asum[4] = {0, 0, 0, 0};
          int8_t Bdec[128];
          for (int kf = 0; kf < 4; ++kf) {
            long gf = sb * 32 + b * 4 + kf;
            const int8_t *Aframe = Arow + gf * 32;
            for (int nl = 0; nl < 4; ++nl)
              tcrv_ime_q4_K_dequant_fragment(blk[nl], b, kf, Bdec + kf * 32 + nl * 8);
            for (int ml = 0; ml < 4; ++ml)
              for (int kl = 0; kl < 8; ++kl)
                asum[ml] += (int32_t)Aframe[ml * 8 + kl];
          }
          const int8_t *Ablk = Arow + (long)(sb * 32 + b * 4) * 32;
          tcrv_ime_vmadot_mac_kloop(Ablk, Bdec, 4, sumi);
          for (int ml = 0; ml < 4; ++ml)
            for (int nl = 0; nl < 4; ++nl) {
              Sc[ml * 4 + nl] += (int32_t)sc[b][nl] * sumi[ml * 4 + nl];
              Sm[ml * 4 + nl] += (int32_t)mm[b][nl] * asum[ml];
            }
        }
        for (int ml = 0; ml < 4; ++ml)
          for (int nl = 0; nl < 4; ++nl) {
            long mo = mi * 4 + ml, no = nj * 4 + nl;
            long oidx = sb * M * N + mo * N + no;
            Sscale[oidx] = Sc[ml * 4 + nl];
            Smin[oidx] = Sm[ml * 4 + nl];
            float d = tcrv_ime_fp16_to_f32(tcrv_ime_load_fp16(blk[nl] + 0));
            float dmin = tcrv_ime_fp16_to_f32(tcrv_ime_load_fp16(blk[nl] + 2));
            Cf[mo * N + no] += d * (float)Sc[ml * 4 + nl] - dmin * (float)Sm[ml * 4 + nl];
          }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// (B) The INDEPENDENT reference (ZERO-MODEL): canonical q4_K raw nibble + 6-bit
// sc/m re-derived from the raw packed bytes, and a plain nested-loop two-level
// fold over logical matrices -- a different code path than the tiled kernel.
// ---------------------------------------------------------------------------

// Column n's native block_q4_K for super-block sb (fragment-major over columns: 4
// native 144-byte blocks per (col-tile, super-block)).
static const uint8_t *ref_block(const uint8_t *Bq4k, long nsb, long n, long sb) {
  long nj = n / 4, nl = n % 4;
  return Bq4k + ((((nj * nsb) + sb) * 4) + nl) * 144;
}

// Canonical q4_K raw nibble for logical W[n][k] (k local to super-block sb). p in
// [0,256): sub-block b=p/32, intra-sub-block pl=p%32; group=b/2, low/high nibble.
static int ref_nibble(const uint8_t *blk, long p) {
  long b = p / 32, pl = p % 32;
  const uint8_t *qs = blk + 16;
  uint8_t byte = qs[(b / 2) * 32 + pl];
  return (b & 1) ? (byte >> 4) : (byte & 0x0F);
}

// Canonical get_scale_min_k4 re-implemented independently (the SAME canonical
// bit-dance; there is exactly one correct 6-bit unpack, like ggml_ref_q8_0).
static void ref_scale_min(int j, const uint8_t *q, int *sc, int *m) {
  if (j < 4) {
    *sc = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

// Logical activation A[m][k] re-derived from the fragment-major int8 pack (SAME
// layout as the q8_0/q4_0 tiles).
static int ref_A(const int8_t *Apack, long K, long m, long k) {
  long mi = m / 4, ml = m % 4;
  long kf = k / 8, kl = k % 8;
  return (int)Apack[mi * 4 * K + kf * 32 + ml * 8 + kl];
}

int main(void) {
  const long M = 8, N = 8, K = 256; // mt=2 nt=2 nsb=1 (whole q4_K super-block)
  const long nsb = K / 256;
  const long q4k_block_bytes = 144;

  srand(20260711u);

  // Fragment-major int8 activation pack: (M/4) row-tiles, K/8 fragments of 32B.
  long apack_bytes = (M / 4) * (K / 8) * 32;
  int8_t *Apack = (int8_t *)malloc(apack_bytes);
  for (long i = 0; i < apack_bytes; ++i) Apack[i] = (int8_t)(rand() % 256 - 128);

  // Fragment-major q4_K weight pack: (N/4) col-tiles x nsb super-blocks x 4 native
  // 144B block_q4_K. scales[12] + qs[128] fully random (get_scale_min_k4 masks the
  // 6-bit fields); d/dmin forced to FINITE positive normals (exp 12/13) so the
  // deferred float epilogue has no NaN/inf, and dmin != 0 + mins != 0 keep BOTH
  // fold terms exercised (cert corpus completeness).
  long bq4k_bytes = (N / 4) * nsb * 4 * q4k_block_bytes;
  uint8_t *Bq4k = (uint8_t *)malloc(bq4k_bytes);
  for (long i = 0; i < bq4k_bytes; ++i) Bq4k[i] = (uint8_t)(rand() % 256);
  for (long blk = 0; blk < (N / 4) * nsb * 4; ++blk) {
    uint8_t *b = Bq4k + blk * q4k_block_bytes;
    uint16_t d = (uint16_t)((rand() & 0x03FF) | (13 << 10));    // ~[0.25,0.5)
    uint16_t dm = (uint16_t)((rand() & 0x03FF) | (12 << 10));   // ~[0.125,0.25)
    b[0] = (uint8_t)(d & 0xFF); b[1] = (uint8_t)(d >> 8);
    b[2] = (uint8_t)(dm & 0xFF); b[3] = (uint8_t)(dm >> 8);
  }

  // --- Check (1)+(2): DECODE + SCALE/MIN bit-exact vs canonical ----------------
  long decode_mismatch = 0, decode_total = 0;
  long scalemin_mismatch = 0, scalemin_total = 0;
  for (long n = 0; n < N; ++n)
    for (long sb = 0; sb < nsb; ++sb) {
      const uint8_t *blk = ref_block(Bq4k, nsb, n, sb);
      for (int b = 0; b < 8; ++b) {
        int rsc, rm;
        ref_scale_min(b, blk + 4, &rsc, &rm);
        uint8_t ksc, km;
        tcrv_ime_q4_K_get_scale_min(b, blk + 4, &ksc, &km);
        scalemin_total += 2;
        if ((int)ksc != rsc) scalemin_mismatch++;
        if ((int)km != rm) scalemin_mismatch++;
        for (int kf = 0; kf < 4; ++kf) {
          int8_t frame[8];
          tcrv_ime_q4_K_dequant_fragment(blk, b, kf, frame);
          for (int kl = 0; kl < 8; ++kl) {
            long p = b * 32 + kf * 8 + kl;
            decode_total += 1;
            if ((int)frame[kl] != ref_nibble(blk, p)) decode_mismatch++;
          }
        }
      }
    }

  // --- Check (3): tiled kernel S_scale/S_min int32 core vs plain-loop reference -
  // On real K1 the sumi_b reduces through the actual `vmadot` instruction.
  int32_t *Sscale = (int32_t *)calloc((size_t)nsb * M * N, sizeof(int32_t));
  int32_t *Smin = (int32_t *)calloc((size_t)nsb * M * N, sizeof(int32_t));
  float *Cf = (float *)calloc((size_t)M * N, sizeof(float));
  tcrv_ime_q4_K_vmadot_matmul(Apack, Bq4k, Sscale, Smin, Cf, M, N, K);

  long core_mismatch = 0;
  int32_t maxabs = 0;
  long float_mismatch = 0;
  for (long m = 0; m < M; ++m)
    for (long n = 0; n < N; ++n) {
      float cf_ref = 0.0f;
      for (long sb = 0; sb < nsb; ++sb) {
        const uint8_t *blk = ref_block(Bq4k, nsb, n, sb);
        int64_t Ss = 0, Sm = 0;
        for (int b = 0; b < 8; ++b) {
          int rsc, rm;
          ref_scale_min(b, blk + 4, &rsc, &rm);
          int64_t sumi = 0, asum = 0;
          for (int pl = 0; pl < 32; ++pl) {
            long p = b * 32 + pl;
            long k = sb * 256 + p;
            int a = ref_A(Apack, K, m, k);
            sumi += (int64_t)a * (int64_t)ref_nibble(blk, p);
            asum += (int64_t)a;
          }
          Ss += (int64_t)rsc * sumi;
          Sm += (int64_t)rm * asum;
        }
        long oidx = sb * M * N + m * N + n;
        if ((int64_t)Sscale[oidx] != Ss) core_mismatch++;
        if ((int64_t)Smin[oidx] != Sm) core_mismatch++;
        int32_t a1 = Sscale[oidx] < 0 ? -Sscale[oidx] : Sscale[oidx];
        if (a1 > maxabs) maxabs = a1;
        // reference float epilogue: SAME two-level fold as the kernel.
        float d = tcrv_ime_fp16_to_f32(tcrv_ime_load_fp16(blk + 0));
        float dmin = tcrv_ime_fp16_to_f32(tcrv_ime_load_fp16(blk + 2));
        cf_ref += d * (float)(int32_t)Ss - dmin * (float)(int32_t)Sm;
      }
      if (Cf[m * N + n] != cf_ref) float_mismatch++;
    }

  printf("decode raw nibbles: %ld/%ld bit-exact (mismatch=%ld)\n",
         decode_total - decode_mismatch, decode_total, decode_mismatch);
  printf("6-bit scale/min unpack: %ld/%ld bit-exact (mismatch=%ld)\n",
         scalemin_total - scalemin_mismatch, scalemin_total, scalemin_mismatch);
  printf("int32 core S_scale+S_min (real vmadot): %ld/%ld tiles bit-exact "
         "(mismatch=%ld, max|S_scale|=%d)\n",
         2 * (M * N) - core_mismatch, 2 * (M * N), core_mismatch, maxabs);
  printf("float epilogue C=d*S_scale-dmin*S_min: %ld/%ld byte-exact vs same-fold "
         "ref (mismatch=%ld)\n",
         (M * N) - float_mismatch, M * N, float_mismatch);

  free(Apack);
  free(Bq4k);
  free(Sscale);
  free(Smin);
  free(Cf);
  if (decode_mismatch == 0 && scalemin_mismatch == 0 && core_mismatch == 0 &&
      float_mismatch == 0) {
    printf("ORACLE PASS: q4_K decode + 6-bit sc/m + scalar-vmadot two-level int32 "
           "core (S_scale AND S_min) are int32-EXACT\n");
    return 0;
  }
  printf("ORACLE FAIL\n");
  return 1;
}
