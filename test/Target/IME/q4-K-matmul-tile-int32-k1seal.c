// G4 M2b K1 SILICON SEAL for the format-keyed q4_K IME GEMM tile (the SUPER-BLOCK
// K-quant tile; the DEDICATED effort with the TWO-LEVEL 6-bit scale/min fold).
//
// This is the board-seal counterpart of the M2b host oracle
// (q4-K-matmul-tile-int32-oracle.c). It is BYTE-FOR-BYTE identical to that oracle
// -- the q4_K raw-nibble DECODE, the 6-bit scale/min UNPACK, the fp16 epilogue
// helpers, the tiled two-level-fold matmul, the ZERO-MODEL plain-loop reference,
// and the test grid are all unchanged -- with EXACTLY ONE difference: the scalar
// substitute for the batched vmadot MAC leaf (tcrv_ime_vmadot_mac_kloop) is
// replaced by the REAL `vmadot` inline-asm leaf emitted VERBATIM by the IME
// backend emitter (lib/Plugin/IME/IMEBackendEmissionDriver.cpp macKloopHelperBody(),
// the SAME leaf the q4_0/q8_0 tiles reuse). So on real K1 silicon the per-sub-block
// sumi_b reduces through the actual `vmadot` instruction (encoding 0xe210312b) in a
// register-resident 4-fragment loop, and the int32 core check below is a HARDWARE
// validation of the q4_K TWO-LEVEL fold (S_scale = Sum_b sc_b*sumi_b AND S_min =
// Sum_b m_b*asum_b) -- NOT the hollow bare-nibble MAC.
//
// The batched vmadot leaf contract (identical to the scalar shim it replaces):
// over the sub-block's 4 contiguous 4x8 A/B fragments, v2/v3 accumulate
// sumi_b = Sum_kf A_kf . B_kf^T (int32), single vsetvli, one store. Per-fragment:
//   A: (4,8) int8 row-major -> v0 ; B: stored (4,8) int8 -> v1 (== B^T)
//   sumi_b: (4,4) int32 (v2/v3), sumi_b[i][j] += sum_k A_kf[i][k]*B_kf[j][k]
// pinned by `vsetvli e8,m1` => vl=32 at VLEN=256 => the 4x4x8 MAC unit.
//
// Build (SpacemiT/board toolchain that assembles `vmadot`; on this K1 the stock
// binutils encodes it once the version-pinned march token unlocks the opcode):
//   gcc -O2 -std=c11 -march=rv64gcv_xsmtvdotii1p0 -mabi=lp64d \
//       q4-K-matmul-tile-int32-k1seal.c -o /tmp/q4kseal
// Run pinned to IME harts (X60: IME present on harts 0-3, hart 4 SIGILLs):
//   taskset -c 0-3 /tmp/q4kseal
//
// objdump golden: the leaf must disassemble to `vmadot v2,v0,v1` = 0xe210312b.
// Exit 0 + "SEAL PASS" iff DECODE + SCALE/MIN + the two-level int32 core (S_scale
// AND S_min) + the deferred float epilogue are all 0-diff on real silicon.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// (A) The emitted structured helpers (mirror of the M2b emitter output). The asm
// leaf below is the EMITTER-VERBATIM batched `vmadot` MAC (macKloopHelperBody(),
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

static const uint8_t *ref_block(const uint8_t *Bq4k, long nsb, long n, long sb) {
  long nj = n / 4, nl = n % 4;
  return Bq4k + ((((nj * nsb) + sb) * 4) + nl) * 144;
}

static int ref_nibble(const uint8_t *blk, long p) {
  long b = p / 32, pl = p % 32;
  const uint8_t *qs = blk + 16;
  uint8_t byte = qs[(b / 2) * 32 + pl];
  return (b & 1) ? (byte >> 4) : (byte & 0x0F);
}

static void ref_scale_min(int j, const uint8_t *q, int *sc, int *m) {
  if (j < 4) {
    *sc = q[j] & 63;
    *m = q[j + 4] & 63;
  } else {
    *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
    *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
  }
}

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

  long apack_bytes = (M / 4) * (K / 8) * 32;
  int8_t *Apack = (int8_t *)malloc(apack_bytes);
  for (long i = 0; i < apack_bytes; ++i) Apack[i] = (int8_t)(rand() % 256 - 128);

  long bq4k_bytes = (N / 4) * nsb * 4 * q4k_block_bytes;
  uint8_t *Bq4k = (uint8_t *)malloc(bq4k_bytes);
  for (long i = 0; i < bq4k_bytes; ++i) Bq4k[i] = (uint8_t)(rand() % 256);
  for (long blk = 0; blk < (N / 4) * nsb * 4; ++blk) {
    uint8_t *b = Bq4k + blk * q4k_block_bytes;
    uint16_t d = (uint16_t)((rand() & 0x03FF) | (13 << 10));
    uint16_t dm = (uint16_t)((rand() & 0x03FF) | (12 << 10));
    b[0] = (uint8_t)(d & 0xFF); b[1] = (uint8_t)(d >> 8);
    b[2] = (uint8_t)(dm & 0xFF); b[3] = (uint8_t)(dm >> 8);
  }

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
    printf("SEAL PASS: q4_K decode + 6-bit sc/m + real-vmadot two-level int32 core "
           "(S_scale AND S_min) are int32-EXACT on K1\n");
    return 0;
  }
  printf("SEAL FAIL\n");
  return 1;
}
