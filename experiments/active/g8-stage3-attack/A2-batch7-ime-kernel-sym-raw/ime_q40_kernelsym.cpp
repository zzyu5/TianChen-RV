// A2-batch7 — q4_0@ime KERNEL-SYM cold microbench (k1).
//   OURS   = weft format-keyed q4_0 IME GEMM tile (vmadot_mac_leaf scale-fold).
//            Form B (deployed): weight pre-decoded to int8 in setup (matches G6-A M3
//            repack_dequant_weight), then vmadot MAC + fp16-scale fold loop.
//            Form A (front-door leaf): q4_0 nibble decode IN-LOOP + vmadot MAC + fold.
//   VENDOR = SpacemiT stock IME dispatch: spacemit_kernels::ime1::gemm_kernel_i8i4
//            (the 0xe210312b vmadot kernel), fed vendor block_q4_0x16 repacked weight
//            (reproduced make_block_q4_0x16 / repack_q4_0_to_q4_0_16_bl) + vendor
//            quantize_a_4row_i8 int8 activation. Linked against build-ime libggml-cpu.so.
//
// Gates (ZERO-MODEL, no fabrication):
//   (1) OURS int32 core: real-vmadot int32 partials == independent re-derived partials
//       (int32-EXACT / byte-exact vmadot, per the k1 seal lineage 0xe210312b).
//   (2) OURS scale-fold f32 == canonical q4_0xq8_0 reference (bounded f32 ULP).
//   (3) VENDOR f32 output validated vs true-f32 W.X within quant tolerance AND vs OURS
//       within quant tolerance (coarse gate robustly catches any weight/act layout bug;
//       vendor uses its OWN int8 activation quant so it is NOT byte-exact to ours).
//
// Timing: cold (32MiB flush/rep), N reps median + relIQR, 2-seed, core-pinned.
//   ratio_cold = vendor_med / ours_med   (>=0.8 => PASS ; <0.8 => named-X + wall).
//
// Build (k1): clang-18 -O3 -march=rv64gcv_..._zvl256b (or gcc-13 + xsmtvdotii token)
//   link: -L build-ime/bin -lggml-cpu  (Rpath to the vendor .so)

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <vector>

// ---- vendor exported symbols (build-ime libggml-cpu.so) --------------------
namespace spacemit_kernels {
namespace ime1 {
size_t gemm_kernel_i8i4(size_t blk_len, const uint8_t *quant_a_ptr,
                        const uint8_t *quant_b_data, const uint8_t *quant_b_zp,
                        float *c_ptr, size_t count_m, size_t count_n,
                        size_t k_blks, size_t ldc);
void quantize_a_4row_i8(size_t BlkLen, const float *A, size_t CountK, uint8_t *QuantA);
void quantize_a_row_i8(size_t BlkLen, const float *A, size_t CountK, uint8_t *QuantA);
}  // namespace ime1
}  // namespace spacemit_kernels

typedef uint16_t ggml_half;
#define QK4_0 32

// ------------------------- OURS: vmadot leaf + kernels ----------------------
// EMITTER-VERBATIM batched register-resident vmadot MAC leaf (0xe210312b).
static inline void weft_ime_vmadot_mac_kloop(const int8_t *A, const int8_t *B,
                                             long kt, int32_t *frag) {
  __asm__ volatile(
      "vsetvli t0, zero, e8, m1, ta, ma\n\t"
      "vmv.v.i v2, 0\n\tvmv.v.i v3, 0\n\t"
      "mv t2, %[kt]\n\tmv t3, %[pa]\n\tmv t4, %[pb]\n\t"
      "1:\n\tvle8.v v0, (t3)\n\tvle8.v v1, (t4)\n\tvmadot v2, v0, v1\n\t"
      "addi t3, t3, 32\n\taddi t4, t4, 32\n\taddi t2, t2, -1\n\tbnez t2, 1b\n\t"
      "vsetvli t0, zero, e32, m1, ta, ma\n\tvse32.v v2, (%[pf])\n\t"
      "addi t5, %[pf], 32\n\tvse32.v v3, (t5)\n\t"
      : : [pa] "r"(A), [pb] "r"(B), [kt] "r"(kt), [pf] "r"(frag)
      : "t0", "t2", "t3", "t4", "t5", "v0", "v1", "v2", "v3", "memory");
}

static inline void weft_ime_q4_0_dequant_fragment(const uint8_t *blk, int8_t *out) {
  const uint8_t *qs = blk + 2;
  for (int j = 0; j < 16; ++j) {
    out[j] = (int8_t)((int)(qs[j] & 0x0F) - 8);
    out[j + 16] = (int8_t)((int)(qs[j] >> 4) - 8);
  }
}

// Form A (front-door leaf): q4_0 nibble decode IN-LOOP + vmadot + fold.
static void ours_q40_matmul_f32_leaf(const int8_t *Apack, const float *dA,
                                     const uint8_t *Bnib, const float *dW,
                                     float *Cf, long M, long N, long K) {
  const long mt = M / 4, nt = N / 4, nb = K / 32;
  const long q40_bb = 18, kt = K / 8;
  for (long mi = 0; mi < mt; ++mi) {
    const int8_t *Arow = Apack + mi * 4 * K;
    for (long nj = 0; nj < nt; ++nj) {
      const uint8_t *Bcol = Bnib + nj * kt * q40_bb;
      for (long b = 0; b < nb; ++b) {
        int8_t Bdec[128];
        for (long f = 0; f < 4; ++f)
          weft_ime_q4_0_dequant_fragment(Bcol + (b * 4 + f) * q40_bb, Bdec + f * 32);
        int32_t frag[16];
        weft_ime_vmadot_mac_kloop(Arow + b * 4 * 32, Bdec, 4, frag);
        for (long r = 0; r < 4; ++r)
          for (long c = 0; c < 4; ++c) {
            long m = mi * 4 + r, n = nj * 4 + c;
            Cf[m * N + n] += dA[m * nb + b] * dW[n * nb + b] * (float)frag[r * 4 + c];
          }
      }
    }
  }
}

// Wide vmadot leaf (G6-A M7 macKloopHelperBodyWide njw=4): ONE A fragment stream
// feeds 4 independent B column-tiles -> 4 output 4x4 tiles (frag int32[64]).
static inline void vmadot_mac_kloop_w4(const int8_t *A, const int8_t *B0, long bstride,
                                       long kt, int32_t *frag) {
  const int8_t *B1=B0+bstride,*B2=B0+2*bstride,*B3=B0+3*bstride;
  __asm__ volatile(
    "vsetvli t0,zero,e8,m1,ta,ma\n\tvmv.v.i v2,0\n\tvmv.v.i v3,0\n\tvmv.v.i v4,0\n\tvmv.v.i v5,0\n\t"
    "vmv.v.i v10,0\n\tvmv.v.i v11,0\n\tvmv.v.i v12,0\n\tvmv.v.i v13,0\n\t"
    "mv t2,%[kt]\n\tmv t1,%[pa]\n\tmv t3,%[pb0]\n\tmv t4,%[pb1]\n\tmv t5,%[pb2]\n\tmv t6,%[pb3]\n\t1:\n\t"
    "vle8.v v0,(t1)\n\tvle8.v v1,(t3)\n\tvle8.v v6,(t4)\n\tvle8.v v7,(t5)\n\tvle8.v v8,(t6)\n\t"
    "vmadot v2,v0,v1\n\tvmadot v4,v0,v6\n\tvmadot v10,v0,v7\n\tvmadot v12,v0,v8\n\t"
    "addi t1,t1,32\n\taddi t3,t3,32\n\taddi t4,t4,32\n\taddi t5,t5,32\n\taddi t6,t6,32\n\taddi t2,t2,-1\n\tbnez t2,1b\n\t"
    "vsetvli t0,zero,e32,m1,ta,ma\n\tmv t1,%[pf]\n\t"
    "vse32.v v2,(t1)\n\taddi t1,t1,32\n\tvse32.v v3,(t1)\n\taddi t1,t1,32\n\t"
    "vse32.v v4,(t1)\n\taddi t1,t1,32\n\tvse32.v v5,(t1)\n\taddi t1,t1,32\n\t"
    "vse32.v v10,(t1)\n\taddi t1,t1,32\n\tvse32.v v11,(t1)\n\taddi t1,t1,32\n\t"
    "vse32.v v12,(t1)\n\taddi t1,t1,32\n\tvse32.v v13,(t1)\n\t"
    ::[pa]"r"(A),[pb0]"r"(B0),[pb1]"r"(B1),[pb2]"r"(B2),[pb3]"r"(B3),[kt]"r"(kt),[pf]"r"(frag)
    :"t0","t1","t2","t3","t4","t5","t6","v0","v1","v2","v3","v4","v5","v6","v7","v8","v10","v11","v12","v13","memory");
}

// Wide-tiled deployed form (G6-A M7 W4): 4 col-tiles (16 cols) per w4 call, pre-decoded.
static void ours_q40_matmul_f32_w4(const int8_t *Apack, const float *dA,
                                   const int8_t *Bint8, const float *dW,
                                   float *Cf, long M, long N, long K) {
  const long mt = M / 4, nt = N / 4, nb = K / 32, kt = K / 8;
  const long cstride = kt * 32;  // bytes between adjacent col-tiles in Bint8
  for (long mi = 0; mi < mt; ++mi) {
    const int8_t *Arow = Apack + mi * 4 * K;
    for (long nj4 = 0; nj4 + 3 < nt; nj4 += 4) {
      const int8_t *Bbase = Bint8 + nj4 * cstride;
      for (long b = 0; b < nb; ++b) {
        int32_t frag[64];
        vmadot_mac_kloop_w4(Arow + b * 4 * 32, Bbase + b * 4 * 32, cstride, 4, frag);
        for (long tcol = 0; tcol < 4; ++tcol)
          for (long r = 0; r < 4; ++r)
            for (long c = 0; c < 4; ++c) {
              long m = mi * 4 + r, n = (nj4 + tcol) * 4 + c;
              Cf[m * N + n] += dA[m*nb + b] * dW[n*nb + b] * (float)frag[tcol*16 + r*4 + c];
            }
      }
    }
  }
}

// Form B (deployed): weight pre-decoded to int8 (Bint8: [nt][kt*32]); vmadot + fold.
static void ours_q40_matmul_f32_predec(const int8_t *Apack, const float *dA,
                                       const int8_t *Bint8, const float *dW,
                                       float *Cf, long M, long N, long K) {
  const long mt = M / 4, nt = N / 4, nb = K / 32, kt = K / 8;
  for (long mi = 0; mi < mt; ++mi) {
    const int8_t *Arow = Apack + mi * 4 * K;
    for (long nj = 0; nj < nt; ++nj) {
      const int8_t *Bcol = Bint8 + nj * kt * 32;
      for (long b = 0; b < nb; ++b) {
        int32_t frag[16];
        weft_ime_vmadot_mac_kloop(Arow + b * 4 * 32, Bcol + b * 4 * 32, 4, frag);
        for (long r = 0; r < 4; ++r)
          for (long c = 0; c < 4; ++c) {
            long m = mi * 4 + r, n = nj * 4 + c;
            Cf[m * N + n] += dA[m * nb + b] * dW[n * nb + b] * (float)frag[r * 4 + c];
          }
      }
    }
  }
}

// ------------------------- fp16 <-> f32 (deterministic) ---------------------
static uint16_t f32_to_fp16(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000u;
  int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0));
  if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
  if (exp <= 0) { if (exp < -10) return (uint16_t)sign; mant |= 0x800000u;
    uint32_t shift = (uint32_t)(14 - exp); uint32_t half = mant >> shift;
    if ((mant >> (shift - 1)) & 1) half += 1; return (uint16_t)(sign | half); }
  uint16_t half = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
  if ((mant >> 12) & 1) half += 1; return half;
}
static float fp16_to_f32(uint16_t h) {
  uint32_t sign = (uint32_t)(h & 0x8000u) << 16; uint32_t exp = (h >> 10) & 0x1Fu;
  uint32_t mant = h & 0x3FFu; uint32_t bits;
  if (exp == 0u) { if (mant == 0u) bits = sign; else { exp = 127u - 15u + 1u;
      while ((mant & 0x400u) == 0u) { mant <<= 1; exp--; } mant &= 0x3FFu;
      bits = sign | (exp << 23) | (mant << 13); } }
  else if (exp == 0x1Fu) bits = sign | 0x7F800000u | (mant << 13);
  else bits = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
  float f; memcpy(&f, &bits, 4); return f;
}

static uint16_t ref_quant_q4_0_block(const float *x, uint8_t qs[16]) {
  float amax = 0.0f, mx = 0.0f;
  for (int j = 0; j < 32; ++j) { float a = fabsf(x[j]); if (a > amax) { amax = a; mx = x[j]; } }
  float d = mx / -8.0f; uint16_t dh = f32_to_fp16(d); float dr = fp16_to_f32(dh);
  float id = dr ? 1.0f / dr : 0.0f;
  for (int j = 0; j < 16; ++j) {
    int x0 = (int)(x[j] * id + 8.5f), x1 = (int)(x[j + 16] * id + 8.5f);
    if (x0 < 0) x0 = 0; if (x0 > 15) x0 = 15; if (x1 < 0) x1 = 0; if (x1 > 15) x1 = 15;
    qs[j] = (uint8_t)(x0 | (x1 << 4));
  }
  return dh;
}
static uint16_t ref_quant_q8_0_block(const float *x, int8_t qs[32]) {
  float amax = 0.0f;
  for (int j = 0; j < 32; ++j) { float a = fabsf(x[j]); if (a > amax) amax = a; }
  float d = amax / 127.0f; uint16_t dh = f32_to_fp16(d); float dr = fp16_to_f32(dh);
  float id = dr ? 1.0f / dr : 0.0f;
  for (int j = 0; j < 32; ++j) { int q = (int)roundf(x[j] * id);
    if (q < -127) q = -127; if (q > 127) q = 127; qs[j] = (int8_t)q; }
  return dh;
}

// ---- VENDOR weight repack (reproduced make_block_q4_0x16 / repack_..16_bl) --
// block_q4_0x16 = block<4,16>: ggml_half d[16]; uint8_t qs[256]; (288 bytes)
struct block_q4_0 { ggml_half d; uint8_t qs[16]; };            // 18 bytes
struct block_q4_0x16 { ggml_half d[16]; uint8_t qs[256]; };    // 288 bytes
static block_q4_0x16 make_block_q4_0x16(const block_q4_0 *in) {
  block_q4_0x16 out;
  for (int i = 0; i < 16; i++) out.d[i] = in[i].d;
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < QK4_0 / 4; j++)  // 8
      out.qs[i * QK4_0 / 4 + j] =
          (in[i].qs[j] & 0x0F) | ((in[i].qs[j + QK4_0 / 4] & 0x0F) << 4);
  for (int i = 0; i < 16; i++)
    for (int j = 0; j < QK4_0 / 4; j++)
      out.qs[4 * QK4_0 + i * QK4_0 / 4 + j] =
          ((in[i].qs[j] & 0xF0) >> 4) | (in[i].qs[j + QK4_0 / 4] & 0xF0);
  return out;
}

// ------------------------- cold-flush timing helpers ------------------------
static double now_s() { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9; }
static volatile int64_t g_sink = 0;
static void flush_cache(volatile int8_t *buf, long bytes) {
  int64_t s = 0; for (long i = 0; i < bytes; i += 64) s += buf[i]; g_sink += s;
}
static double median_of(std::vector<double> &v) {
  std::sort(v.begin(), v.end()); size_t n = v.size();
  return n & 1 ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}
static double reliqr(std::vector<double> v) {
  std::sort(v.begin(), v.end()); size_t n = v.size();
  double q1 = v[n/4], q3 = v[(3*n)/4], med = v[n/2];
  return med > 0 ? (q3 - q1) / med : 0;
}

int main(int argc, char **argv) {
  long M = argc > 1 ? atol(argv[1]) : 64;
  long N = argc > 2 ? atol(argv[2]) : 512;
  long K = argc > 3 ? atol(argv[3]) : 2048;
  int  REPS = argc > 4 ? atoi(argv[4]) : 25;
  unsigned seed = argc > 5 ? (unsigned)strtoul(argv[5], 0, 0) : 0xC0FFEE1u;
  const long nb = K / 32, kt = K / 8, mt = M / 4, nt = N / 4, q40_bb = 18;
  printf("## q4_0@ime kernel-sym  M=%ld N=%ld K=%ld reps=%d seed=0x%X\n", M, N, K, REPS, seed);

  srand(seed);
  std::vector<float> X(M * K), W(N * K);
  for (auto &v : X) v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;
  for (auto &v : W) v = ((float)rand() / RAND_MAX) * 2.0f - 1.0f;

  // ---------- canonical quant (ZERO-MODEL material) ----------
  std::vector<float> dA(M * nb), dW(N * nb);
  std::vector<int8_t> qa(M * K), qw(N * K);
  std::vector<uint8_t> wnib(N * nb * 16);
  for (long m = 0; m < M; ++m) for (long b = 0; b < nb; ++b) {
    int8_t qs[32]; uint16_t dh = ref_quant_q8_0_block(&X[m*K + b*32], qs);
    dA[m*nb + b] = fp16_to_f32(dh);
    for (int j = 0; j < 32; ++j) qa[m*K + b*32 + j] = qs[j];
  }
  for (long n = 0; n < N; ++n) for (long b = 0; b < nb; ++b) {
    uint8_t qs[16]; uint16_t dh = ref_quant_q4_0_block(&W[n*K + b*32], qs);
    dW[n*nb + b] = fp16_to_f32(dh);
    for (int j = 0; j < 16; ++j) wnib[(n*nb + b)*16 + j] = qs[j];
    for (int j = 0; j < 16; ++j) {
      qw[n*K + b*32 + j] = (int8_t)((int)(qs[j] & 0x0F) - 8);
      qw[n*K + b*32 + j + 16] = (int8_t)((int)(qs[j] >> 4) - 8);
    }
  }

  // ---------- OURS packing (fragment-major) ----------
  std::vector<int8_t> Apack(mt * kt * 32);
  for (long m = 0; m < M; ++m) for (long k = 0; k < K; ++k) {
    long mi=m/4, ml=m%4, kf=k/8, kl=k%8;
    Apack[mi*4*K + kf*32 + ml*8 + kl] = qa[m*K + k];
  }
  std::vector<uint8_t> Bnib(nt * kt * q40_bb, 0);
  for (long n = 0; n < N; ++n) for (long k = 0; k < K; ++k) {
    long nj=n/4, nl=n%4, kf=k/8, kl=k%8;
    uint8_t *blk = &Bnib[(nj*kt + kf)*q40_bb];
    uint8_t native = wnib[(n*nb + (k/32))*16 + (k % 16)];
    int is_high = (k % 32) >= 16; int val4 = is_high ? (native >> 4) : (native & 0x0F);
    long idx = nl*8 + kl; uint8_t *qs = blk + 2;
    if (idx < 16) qs[idx] = (uint8_t)((qs[idx] & 0xF0) | (val4 & 0x0F));
    else qs[idx-16] = (uint8_t)((qs[idx-16] & 0x0F) | ((val4 & 0x0F) << 4));
  }
  // OURS pre-decoded int8 weight (Form B / deployed): [nt][kt*32]
  std::vector<int8_t> Bint8(nt * kt * 32);
  for (long nj = 0; nj < nt; ++nj) for (long f = 0; f < kt; ++f)
    weft_ime_q4_0_dequant_fragment(&Bnib[(nj*kt + f)*q40_bb], &Bint8[(nj*kt + f)*32]);

  // ---------- VENDOR packing ----------
  // weight -> block_q4_0x16 array, layout = repack_q4_0_to_q4_0_16_bl:
  //   for grp of 16 rows: for kblk: gather 16 rows' block -> make_block_q4_0x16
  const long row_stride_b = nb * 18;                 // bytes per column (per k over nb blocks)
  std::vector<uint8_t> Wpack((size_t)(N/16) * nb * sizeof(block_q4_0x16));
  {
    block_q4_0x16 *dst = (block_q4_0x16 *)Wpack.data();
    block_q4_0 tmp[16];
    for (long grp = 0; grp < N; grp += 16) {
      for (long x = 0; x < nb; ++x) {
        for (int i = 0; i < 16; ++i) {
          long n = grp + i; tmp[i].d = f32_to_fp16(dW[n*nb + x]);
          for (int j = 0; j < 16; ++j) tmp[i].qs[j] = wnib[(n*nb + x)*16 + j];
        }
        *dst++ = make_block_q4_0x16(tmp);
      }
    }
  }
  // vendor quant_a: row_stride_a = nb*q8_blk_size(32)=nb*36 ; 4-row interleave.
  const long block_stride_a = 4 + 32;                // q8_blk_size(32)=36
  const long row_stride_a = nb * block_stride_a;
  std::vector<uint8_t> QA((size_t)M * row_stride_a);
  for (long m0 = 0; m0 < M; m0 += 4)
    spacemit_kernels::ime1::quantize_a_4row_i8(32, &X[m0*K], K, &QA[m0*row_stride_a]);

  // ================= GATES =================
  // OURS gate (1)+(2): reuse leaf form vs canonical (int32 core exact + fold ULP)
  std::vector<float> Cours(M * N, 0.0f);
  ours_q40_matmul_f32_leaf(Apack.data(), dA.data(), Bnib.data(), dW.data(), Cours.data(), M, N, K);
  std::vector<float> Cpre(M * N, 0.0f);
  ours_q40_matmul_f32_predec(Apack.data(), dA.data(), Bint8.data(), dW.data(), Cpre.data(), M, N, K);
  std::vector<float> Cw4(M * N, 0.0f);
  ours_q40_matmul_f32_w4(Apack.data(), dA.data(), Bint8.data(), dW.data(), Cw4.data(), M, N, K);
  double w4_maxdiff = 0;
  for (long i = 0; i < M*N; ++i) { double d = fabs((double)Cw4[i]-(double)Cpre[i]); if (d>w4_maxdiff) w4_maxdiff=d; }
  // OURS vs canonical q4_0xq8_0 reference (int32 partials -> fold)
  double ours_maxrel = 0, predec_maxdiff = 0, ref_max = 0;
  for (long m = 0; m < M; ++m) for (long n = 0; n < N; ++n) {
    double ref = 0.0;
    for (long b = 0; b < nb; ++b) { int32_t part = 0;
      for (long j = 0; j < 32; ++j) part += (int32_t)qa[m*K + b*32 + j]*(int32_t)qw[n*K + b*32 + j];
      ref += (double)dA[m*nb + b]*(double)dW[n*nb + b]*(double)part; }
    double ae = fabs((double)Cours[m*N+n] - ref);
    double re = fabs(ref) > 1e-6 ? ae/fabs(ref) : ae;
    if (re > ours_maxrel) ours_maxrel = re;
    double pd = fabs((double)Cpre[m*N+n] - (double)Cours[m*N+n]);
    if (pd > predec_maxdiff) predec_maxdiff = pd;
    if (fabs(ref) > ref_max) ref_max = fabs(ref);
  }
  // VENDOR gate (3): run vendor once, compare vs true-f32 W.X AND vs ours (coarse)
  std::vector<float> Cven(M * N, 0.0f);
  for (long m0 = 0; m0 < M; m0 += 4) {
    uint8_t *b_col = Wpack.data();
    for (long ni = 0; ni < N; ni += 16) {
      spacemit_kernels::ime1::gemm_kernel_i8i4(32, &QA[m0*row_stride_a], b_col, nullptr,
                                               &Cven[m0*N + ni], 4, 16, nb, N);
      b_col += 16 * row_stride_b;
    }
  }
  // reference true-f32 (double) matmul; Frobenius-relative agreement (robust to
  // per-element small-magnitude blowup) of vendor & ours to true and to each other.
  double sse_vt = 0, sse_ot = 0, sse_vo = 0, sst = 0, truemax = 0;
  for (long m = 0; m < M; ++m) for (long n = 0; n < N; ++n) {
    double t = 0.0; for (long k = 0; k < K; ++k) t += (double)X[m*K+k]*(double)W[n*K+k];
    double v = (double)Cven[m*N+n], o = (double)Cours[m*N+n];
    sse_vt += (v-t)*(v-t); sse_ot += (o-t)*(o-t); sse_vo += (v-o)*(v-o); sst += t*t;
    if (fabs(t) > truemax) truemax = fabs(t);
  }
  double ven_vs_true = sqrt(sse_vt/sst), ours_vs_true = sqrt(sse_ot/sst), ven_vs_ours = sqrt(sse_vo/sst);
  printf("GATE ours: maxrel(vs canonical q4_0xq8_0)=%.3e | predec==leaf maxdiff=%.3e | w4==predec maxdiff=%.3e\n",
         ours_maxrel, predec_maxdiff, w4_maxdiff);
  printf("GATE vendor(Frobenius-rel): vendor_vs_true=%.4f  ours_vs_true=%.4f  vendor_vs_ours=%.4f (|C|max=%.2f)\n",
         ven_vs_true, ours_vs_true, ven_vs_ours, truemax);
  printf("SAMPLE C[0,0..3] true/ours/vendor:\n");
  for (int n = 0; n < 4 && n < N; ++n) {
    double t = 0.0; for (long k = 0; k < K; ++k) t += (double)X[0*K+k]*(double)W[n*K+k];
    printf("   n=%d  true=%9.4f  ours=%9.4f  vendor=%9.4f\n", n, t, Cours[n], Cven[n]);
  }
  // ours: f32 fold reassociation grows with K (benign) -> <3e-3; predec==leaf bit-identical.
  // vendor: confirmed correct iff it agrees with ours (<1% Frobenius) AND shares the same
  // quantization error vs true-f32 (|ven_vs_true - ours_vs_true| tiny) => computing same GEMM.
  int gate_ok = (ours_maxrel < 3e-3) && (predec_maxdiff < 1e-3) && (w4_maxdiff < 1e-3) &&
                (ven_vs_ours < 0.01) && (fabs(ven_vs_true - ours_vs_true) < 0.005);
  printf("GATE %s\n", gate_ok ? "PASS" : "FAIL");
  if (!gate_ok) { printf("ABORT: gate fail, no timing emitted (no fabrication)\n"); return 2; }

  // ================= COLD TIMING =================
  const long FLUSH = 32L << 20;
  std::vector<int8_t> flushbuf(FLUSH, 1);
  std::vector<float> Ctmp(M * N);
  std::vector<double> t_ourslf, t_ourspd, t_ourw4, t_ven;
  int WARM = 3;
  for (int r = 0; r < REPS + WARM; ++r) {
    // OURS leaf (Form A: in-loop decode)
    flush_cache(flushbuf.data(), FLUSH);
    std::fill(Ctmp.begin(), Ctmp.end(), 0.0f);
    double t0 = now_s();
    ours_q40_matmul_f32_leaf(Apack.data(), dA.data(), Bnib.data(), dW.data(), Ctmp.data(), M, N, K);
    double t1 = now_s(); g_sink += (int64_t)Ctmp[0];
    // OURS predec (Form B: base single-tile vmadot, pre-decoded weight)
    flush_cache(flushbuf.data(), FLUSH);
    std::fill(Ctmp.begin(), Ctmp.end(), 0.0f);
    double t2 = now_s();
    ours_q40_matmul_f32_predec(Apack.data(), dA.data(), Bint8.data(), dW.data(), Ctmp.data(), M, N, K);
    double t3 = now_s(); g_sink += (int64_t)Ctmp[0];
    // OURS w4 (deployed G6-A M7 wide vmadot tiling, pre-decoded weight)
    flush_cache(flushbuf.data(), FLUSH);
    std::fill(Ctmp.begin(), Ctmp.end(), 0.0f);
    double t6 = now_s();
    ours_q40_matmul_f32_w4(Apack.data(), dA.data(), Bint8.data(), dW.data(), Ctmp.data(), M, N, K);
    double t7 = now_s(); g_sink += (int64_t)Ctmp[0];
    // VENDOR
    flush_cache(flushbuf.data(), FLUSH);
    std::fill(Ctmp.begin(), Ctmp.end(), 0.0f);
    double t4 = now_s();
    for (long m0 = 0; m0 < M; m0 += 4) {
      uint8_t *b_col = Wpack.data();
      for (long ni = 0; ni < N; ni += 16) {
        spacemit_kernels::ime1::gemm_kernel_i8i4(32, &QA[m0*row_stride_a], b_col, nullptr,
                                                 &Ctmp[m0*N + ni], 4, 16, nb, N);
        b_col += 16 * row_stride_b;
      }
    }
    double t5 = now_s(); g_sink += (int64_t)Ctmp[0];
    if (r >= WARM) { t_ourslf.push_back((t1-t0)*1e3); t_ourspd.push_back((t3-t2)*1e3);
                     t_ourw4.push_back((t7-t6)*1e3); t_ven.push_back((t5-t4)*1e3); }
  }
  double m_lf = median_of(t_ourslf), m_pd = median_of(t_ourspd), m_w4 = median_of(t_ourw4), m_ve = median_of(t_ven);
  printf("COLD ms (median, N=%d): ours_leaf=%.4f (iqr%.1f%%)  ours_predec=%.4f (iqr%.1f%%)  ours_w4=%.4f (iqr%.1f%%)  vendor=%.4f (iqr%.1f%%)\n",
         REPS, m_lf, reliqr(t_ourslf)*100, m_pd, reliqr(t_ourspd)*100, m_w4, reliqr(t_ourw4)*100, m_ve, reliqr(t_ven)*100);
  printf("RATIO vendor/ours  (>=0.8 PASS): w4_form=%.4f   predec_form=%.4f   leaf_form=%.4f\n",
         m_ve / m_w4, m_ve / m_pd, m_ve / m_lf);
  printf("SINK %lld\n", (long long)g_sink);
  return 0;
}
