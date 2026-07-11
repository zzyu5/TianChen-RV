// G4-M3 / T5b : IME paradigm-lever measurement harness (q4_0@ime tracer cell).
//
// Measures the PARADIGM LEVER (matrix range vs vector range) x LAYOUT across an
// M sweep {1,2,4,...,512} on real K1 (SpacemiT X60) silicon, and reports the
// crossover M* at which the matrix range (vmadot MMA) starts to >= the vector
// range (RVV widening MAC).  [NG-4]: methodology framing only -- this is the
// "when does the capability-keyed matrix-range selection pay off" curve, NOT an
// "IME is X times faster" headline.  Double ledger + adversary-identity probe are
// emitted with every row.
//
// 2x2 factor (paradigm x layout); IME@vector-layout is N/A (the IME range shares
// the vector regfile and requires the matrix tile layout -- experiment master
// v1 §3), so the grid collapses to 3 measured cells:
//
//                 layout = tiled (matrix-optimal)   layout = vector-optimal (K-contig)
//   paradigm=IME  Cell1  vmadot 4x4x8               (N/A -- documented, not a gap)
//   paradigm=RVV  Cell2  vector reduce on tiles     Cell3  col-outer vwmacc
//
//   Cell1 vs Cell2 = PURE paradigm lever (same tiled layout, matrix vs vector).
//   Cell2 vs Cell3 = PURE layout lever   (vector paradigm, tiled vs vector-opt).
//   Cell1 vs Cell3 = best-vs-best        => the operational crossover M* -> P7.
//
// All three cells consume PRE-DECODED weights (q4_0 nibble decode is done, once,
// OUTSIDE the timed region) so the timed region is pure int8->int32 MAC compute:
// that isolates the range/layout lever from the (identical, memory-bound) decode
// cost.  Every cell writes the SAME C[M][N] and is checked 0-diff against an
// independent ZERO-MODEL plain-GEMM reference re-derived from the logical
// matrices before any timing runs (ZERO-MODEL correctness discipline).
//
// The vmadot leaf is EMITTER-VERBATIM from q4-0-matmul-tile-int32-k1seal.c
// (lib/Plugin/IME/IMEBackendEmissionDriver.cpp macHelperBody, encoding
// 0xe210312b); the RVV cells use standard rv64gcv RVV-1.0 intrinsics.  BOTH
// ranges are compiled by the SAME toolchain into the SAME binary => the KERNEL
// account is compiler-symmetric by construction (the paradigm lever is the only
// free variable).  The opponent for this measurement is OUR OWN RVV-vector
// construction, not a vendor baseline (this is a range-vs-range ablation, not a
// vs-ggml beat; no eight-gate release is claimed here).
//
// Build (k1 SpacemiT gcc assembles vmadot once the march token unlocks it):
//   gcc -O2 -std=c11 -march=rv64gcv_xsmtvdotii1p0 -mabi=lp64d \
//       q4-0-paradigm-lever-t5b-k1.c -o /tmp/t5b
//   taskset -c 0-3 /tmp/t5b            # pin to IME harts 0-3 (hart>=4 SIGILLs)

#define _POSIX_C_SOURCE 199309L
#include <riscv_vector.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define NCOL 256L // N (output features), fixed
#define KDIM 256L // K (contraction), fixed; %32==0 (q4_0 blks), %8==0 (tiles)
#define NREP 15   // paired samples per (cell,M); median reported (N>=10 SOP)

static inline uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

// ---------------------------------------------------------------------------
// (Cell1) IME matrix range: the emitter-verbatim vmadot 4x4x8 MAC leaf.
// A: (4,8) int8 row-major -> v0 ; B: stored (4,8) int8 -> v1 ; C:(4,4) int32.
// C[i][j] = sum_k A[i][k]*B[j][k].  vsetvli e8,m1 => vl=32 @ VLEN256 = 4x4x8 MAC.
// ---------------------------------------------------------------------------
static inline void ime_vmadot_mma_4x4x8(const int8_t *A, const int8_t *B,
                                        int32_t *C) {
  __asm__ volatile(
      "vsetvli   t0, zero, e8, m1, ta, ma   \n\t"
      "vle8.v    v0, (%[pa])                \n\t"
      "vle8.v    v1, (%[pb])                \n\t"
      "vmv.v.i   v2, 0                      \n\t"
      "vmv.v.i   v3, 0                      \n\t"
      "vmadot    v2, v0, v1                 \n\t"
      "vsetvli   t0, zero, e32, m1, ta, ma  \n\t"
      "vse32.v   v2, (%[pc])                \n\t"
      "addi      t1, %[pc], 32              \n\t"
      "vse32.v   v3, (t1)                   \n\t"
      :
      : [pa] "r"(A), [pb] "r"(B), [pc] "r"(C)
      : "t0", "t1", "v0", "v1", "v2", "v3", "memory");
}

// (Cell1b) IME matrix range, register-resident batched leaf: v2/v3 accumulate
// across the whole K/8 fragment loop (vmadot is a MAC into v2), ONE vsetvli
// e8->e32 toggle at the end, single store.  This is the un-pipelined leaf's
// first maturity lever (drop per-fragment zeroing/store/scalar-acc + the e8<->e32
// vsetvli toggle).  Same 0xe210312b vmadot; if vmadot did not accumulate the
// ZERO-MODEL gate would flag it.
static inline void ime_vmadot_kloop(const int8_t *A, const int8_t *B, long kt,
                                    int32_t *frag) {
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
      "vmadot    v2, v0, v1                  \n\t"
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

static void gemm_ime_matrix_batched(const int8_t *Atile, const int8_t *Btile,
                                    int32_t *C, long M, long N, long K) {
  const long mt = M / 4, nt = N / 4, kt = K / 8;
  for (long mi = 0; mi < mt; ++mi) {
    const int8_t *Arow = Atile + mi * 4 * K;
    for (long nj = 0; nj < nt; ++nj) {
      const int8_t *Bcol = Btile + nj * kt * 32;
      int32_t frag[16];
      ime_vmadot_kloop(Arow, Bcol, kt, frag);
      for (long r = 0; r < 4; ++r)
        for (long c = 0; c < 4; ++c)
          C[(mi * 4 + r) * N + (nj * 4 + c)] = frag[r * 4 + c];
    }
  }
}

// Cell1 GEMM: fragment-major int8 tiles (Atile[mt][kt][ml*8+kl],
// Btile[nt][kt][nl*8+kl]).  Writes C directly (each output written once).
static void gemm_ime_matrix(const int8_t *Atile, const int8_t *Btile,
                            int32_t *C, long M, long N, long K) {
  const long mt = M / 4, nt = N / 4, kt = K / 8;
  for (long mi = 0; mi < mt; ++mi) {
    const int8_t *Arow = Atile + mi * 4 * K;
    for (long nj = 0; nj < nt; ++nj) {
      const int8_t *Bcol = Btile + nj * kt * 32;
      int32_t acc[16];
      for (int r = 0; r < 16; ++r) acc[r] = 0;
      for (long kf = 0; kf < kt; ++kf) {
        int32_t frag[16];
        ime_vmadot_mma_4x4x8(Arow + kf * 32, Bcol + kf * 32, frag);
        for (int r = 0; r < 16; ++r) acc[r] += frag[r];
      }
      for (long r = 0; r < 4; ++r)
        for (long c = 0; c < 4; ++c)
          C[(mi * 4 + r) * N + (nj * 4 + c)] = acc[r * 4 + c];
    }
  }
}

// ---------------------------------------------------------------------------
// (Cell2) RVV vector range on the SAME matrix-tiled layout: 8-element widening
// multiply + widening reduce per (ml,nl) fragment dot, accumulated over kt.
// Same layout as Cell1 => Cell1-vs-Cell2 isolates the paradigm lever.
// ---------------------------------------------------------------------------
static void gemm_rvv_tiled(const int8_t *Atile, const int8_t *Btile, int32_t *C,
                           long M, long N, long K) {
  const long mt = M / 4, nt = N / 4, kt = K / 8;
  const size_t vl8 = __riscv_vsetvl_e8m1(8);
  const size_t vl16 = __riscv_vsetvl_e16m2(8);
  for (long mi = 0; mi < mt; ++mi) {
    const int8_t *Arow = Atile + mi * 4 * K;
    for (long nj = 0; nj < nt; ++nj) {
      const int8_t *Bcol = Btile + nj * kt * 32;
      int32_t acc[16];
      for (int r = 0; r < 16; ++r) acc[r] = 0;
      for (long kf = 0; kf < kt; ++kf) {
        const int8_t *Af = Arow + kf * 32;
        const int8_t *Bf = Bcol + kf * 32;
        for (int ml = 0; ml < 4; ++ml) {
          vint8m1_t a = __riscv_vle8_v_i8m1(Af + ml * 8, vl8);
          for (int nl = 0; nl < 4; ++nl) {
            vint8m1_t b = __riscv_vle8_v_i8m1(Bf + nl * 8, vl8);
            vint16m2_t prod = __riscv_vwmul_vv_i16m2(a, b, vl8);
            vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, 1);
            vint32m1_t red =
                __riscv_vwredsum_vs_i16m2_i32m1(prod, zero, vl16);
            acc[ml * 4 + nl] += __riscv_vmv_x_s_i32m1_i32(red);
          }
        }
      }
      for (long r = 0; r < 4; ++r)
        for (long c = 0; c < 4; ++c)
          C[(mi * 4 + r) * N + (nj * 4 + c)] = acc[r * 4 + c];
    }
  }
}

// ---------------------------------------------------------------------------
// (Cell3) RVV vector range on the vector-optimal layout: col-outer vwmacc.
// A row-major int16 [M][K] ; W as int16 [K][N] (N contiguous, N in lanes).
// acc[n] += a * W[k][n] over K, no per-element reduction.  Vector paradigm best.
// ---------------------------------------------------------------------------
static void gemm_rvv_veclayout(const int16_t *Arow, const int16_t *Bkn,
                               int32_t *C, long M, long N, long K) {
  for (long m = 0; m < M; ++m) {
    const int16_t *arow = Arow + m * K;
    for (long nb = 0; nb < N;) {
      size_t vl = __riscv_vsetvl_e16m2(N - nb);
      vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
      for (long k = 0; k < K; ++k) {
        vint16m2_t w = __riscv_vle16_v_i16m2(Bkn + k * N + nb, vl);
        acc = __riscv_vwmacc_vx_i32m4(acc, arow[k], w, vl);
      }
      __riscv_vse32_v_i32m4(C + m * N + nb, acc, vl);
      nb += (long)vl;
    }
  }
}

// (Cell3b) RVV vector range, vector-optimal layout WITH 4-row register blocking
// (MR=4): each W[k] lane-vector is loaded ONCE and fused into 4 row-accumulators,
// so weight reads are amortized across 4 rows exactly as the IME tile amortizes
// them.  This is the vector paradigm's FAIR best (removes the per-row weight
// re-read confound) => Cell1b-vs-Cell3b is the fair paradigm crossover.
static void gemm_rvv_veclayout_mr4(const int16_t *Arow, const int16_t *Bkn,
                                   int32_t *C, long M, long N, long K) {
  long m0 = 0;
  for (; m0 + 4 <= M; m0 += 4) {
    const int16_t *a0 = Arow + (m0 + 0) * K, *a1 = Arow + (m0 + 1) * K,
                  *a2 = Arow + (m0 + 2) * K, *a3 = Arow + (m0 + 3) * K;
    for (long nb = 0; nb < N;) {
      size_t vl = __riscv_vsetvl_e16m2(N - nb);
      vint32m4_t c0 = __riscv_vmv_v_x_i32m4(0, vl), c1 = __riscv_vmv_v_x_i32m4(0, vl),
                 c2 = __riscv_vmv_v_x_i32m4(0, vl), c3 = __riscv_vmv_v_x_i32m4(0, vl);
      const int16_t *bp = Bkn + nb;
      for (long k = 0; k < K; ++k) {
        vint16m2_t w = __riscv_vle16_v_i16m2(bp, vl);
        bp += N;
        c0 = __riscv_vwmacc_vx_i32m4(c0, a0[k], w, vl);
        c1 = __riscv_vwmacc_vx_i32m4(c1, a1[k], w, vl);
        c2 = __riscv_vwmacc_vx_i32m4(c2, a2[k], w, vl);
        c3 = __riscv_vwmacc_vx_i32m4(c3, a3[k], w, vl);
      }
      __riscv_vse32_v_i32m4(C + (m0 + 0) * N + nb, c0, vl);
      __riscv_vse32_v_i32m4(C + (m0 + 1) * N + nb, c1, vl);
      __riscv_vse32_v_i32m4(C + (m0 + 2) * N + nb, c2, vl);
      __riscv_vse32_v_i32m4(C + (m0 + 3) * N + nb, c3, vl);
      nb += (long)vl;
    }
  }
  for (; m0 < M; ++m0) { // remainder rows (M%4): per-row
    const int16_t *arow = Arow + m0 * K;
    for (long nb = 0; nb < N;) {
      size_t vl = __riscv_vsetvl_e16m2(N - nb);
      vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
      for (long k = 0; k < K; ++k) {
        vint16m2_t w = __riscv_vle16_v_i16m2(Bkn + k * N + nb, vl);
        acc = __riscv_vwmacc_vx_i32m4(acc, arow[k], w, vl);
      }
      __riscv_vse32_v_i32m4(C + m0 * N + nb, acc, vl);
      nb += (long)vl;
    }
  }
}

// (Cell3c) Same as Cell3b (MR=4 vector-opt) but reading INT8 weights (widen on
// load with vsext), so the vector path's weight-byte traffic EXACTLY matches the
// IME tile's int8 traffic.  Cell1b-vs-Cell3c is the airtight paradigm isolation:
// same weight bytes, same 4-row blocking -- only free variable is the MAC range
// (vmadot int8 matrix-MAC vs vector vsext+vwmacc).
static void gemm_rvv_veclayout_mr4_i8(const int16_t *Arow, const int8_t *Bkn8,
                                      int32_t *C, long M, long N, long K) {
  long m0 = 0;
  for (; m0 + 4 <= M; m0 += 4) {
    const int16_t *a0 = Arow + (m0 + 0) * K, *a1 = Arow + (m0 + 1) * K,
                  *a2 = Arow + (m0 + 2) * K, *a3 = Arow + (m0 + 3) * K;
    for (long nb = 0; nb < N;) {
      size_t vl = __riscv_vsetvl_e16m2(N - nb);
      vint32m4_t c0 = __riscv_vmv_v_x_i32m4(0, vl), c1 = __riscv_vmv_v_x_i32m4(0, vl),
                 c2 = __riscv_vmv_v_x_i32m4(0, vl), c3 = __riscv_vmv_v_x_i32m4(0, vl);
      const int8_t *bp = Bkn8 + nb;
      for (long k = 0; k < K; ++k) {
        vint8m1_t w8 = __riscv_vle8_v_i8m1(bp, vl);
        vint16m2_t w = __riscv_vsext_vf2_i16m2(w8, vl);
        bp += N;
        c0 = __riscv_vwmacc_vx_i32m4(c0, a0[k], w, vl);
        c1 = __riscv_vwmacc_vx_i32m4(c1, a1[k], w, vl);
        c2 = __riscv_vwmacc_vx_i32m4(c2, a2[k], w, vl);
        c3 = __riscv_vwmacc_vx_i32m4(c3, a3[k], w, vl);
      }
      __riscv_vse32_v_i32m4(C + (m0 + 0) * N + nb, c0, vl);
      __riscv_vse32_v_i32m4(C + (m0 + 1) * N + nb, c1, vl);
      __riscv_vse32_v_i32m4(C + (m0 + 2) * N + nb, c2, vl);
      __riscv_vse32_v_i32m4(C + (m0 + 3) * N + nb, c3, vl);
      nb += (long)vl;
    }
  }
  for (; m0 < M; ++m0) {
    const int16_t *arow = Arow + m0 * K;
    for (long nb = 0; nb < N;) {
      size_t vl = __riscv_vsetvl_e16m2(N - nb);
      vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, vl);
      for (long k = 0; k < K; ++k) {
        vint8m1_t w8 = __riscv_vle8_v_i8m1(Bkn8 + k * N + nb, vl);
        vint16m2_t w = __riscv_vsext_vf2_i16m2(w8, vl);
        acc = __riscv_vwmacc_vx_i32m4(acc, arow[k], w, vl);
      }
      __riscv_vse32_v_i32m4(C + m0 * N + nb, acc, vl);
      nb += (long)vl;
    }
  }
}

// ---------------------------------------------------------------------------
// ZERO-MODEL independent reference: plain triple-loop GEMM on the LOGICAL
// matrices (re-derived, not captured from any cell).  C[m][n]=sum_k A[m][k]W[n][k].
// ---------------------------------------------------------------------------
static void gemm_ref(const int32_t *Alog, const int32_t *Wlog, int32_t *C,
                     long M, long N, long K) {
  for (long m = 0; m < M; ++m)
    for (long n = 0; n < N; ++n) {
      int64_t s = 0;
      for (long k = 0; k < K; ++k)
        s += (int64_t)Alog[m * K + k] * (int64_t)Wlog[n * K + k];
      C[m * N + n] = (int32_t)s;
    }
}

static long cmp_rows(const int32_t *X, const int32_t *R, long rows, long N) {
  long mism = 0;
  for (long i = 0; i < rows * N; ++i)
    if (X[i] != R[i]) mism++;
  return mism;
}

// adaptive repeat: pick R so one timed sample is >= target_ns (clock-noise floor)
static uint64_t time_gemm(void (*run)(void), long *out_R, uint64_t target_ns) {
  long R = 1;
  uint64_t t;
  for (;;) {
    uint64_t t0 = now_ns();
    for (long r = 0; r < R; ++r) run();
    t = now_ns() - t0;
    if (t >= target_ns || R >= (1L << 22)) break;
    R *= 2;
  }
  *out_R = R;
  return t / (uint64_t)R; // ns per GEMM
}

// global run closures (avoid function-ptr arg plumbing)
static const int8_t *g_Atile, *g_Btile, *g_Bkn8;
static const int16_t *g_Arow16, *g_Bkn16;
static int32_t *g_C;
static long g_Mp, g_M, g_N, g_K;
static void run_cell1(void) { gemm_ime_matrix(g_Atile, g_Btile, g_C, g_Mp, g_N, g_K); }
static void run_cell1b(void) { gemm_ime_matrix_batched(g_Atile, g_Btile, g_C, g_Mp, g_N, g_K); }
static void run_cell2(void) { gemm_rvv_tiled(g_Atile, g_Btile, g_C, g_Mp, g_N, g_K); }
static void run_cell3(void) { gemm_rvv_veclayout(g_Arow16, g_Bkn16, g_C, g_M, g_N, g_K); }
static void run_cell3b(void) { gemm_rvv_veclayout_mr4(g_Arow16, g_Bkn16, g_C, g_M, g_N, g_K); }
static void run_cell3c(void) { gemm_rvv_veclayout_mr4_i8(g_Arow16, g_Bkn8, g_C, g_M, g_N, g_K); }

static uint64_t umedian(uint64_t *a, int n) {
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
      if (a[j] < a[i]) { uint64_t t = a[i]; a[i] = a[j]; a[j] = t; }
  return a[n / 2];
}

int main(void) {
  const long N = NCOL, K = KDIM;
  const long Ms[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
  const int nM = (int)(sizeof(Ms) / sizeof(Ms[0]));
  const long Mmax = 512, Mpmax = 512; // padded max
  srand(20260711u);

  // Logical matrices at max M (padded rows are zero -> harmless for ref/IME).
  int32_t *Alog = (int32_t *)malloc((size_t)Mpmax * K * sizeof(int32_t));
  int32_t *Wlog = (int32_t *)malloc((size_t)N * K * sizeof(int32_t));
  for (long i = 0; i < Mpmax * K; ++i) Alog[i] = (rand() % 256) - 128; // int8 range
  for (long i = 0; i < N * K; ++i) Wlog[i] = (rand() % 16) - 8;        // q4_0 range

  // Derived layouts (built at Mpmax; cells index only what they need).
  int8_t *Atile = (int8_t *)malloc((size_t)(Mpmax / 4) * (K / 8) * 32);
  int8_t *Btile = (int8_t *)malloc((size_t)(N / 4) * (K / 8) * 32);
  int16_t *Arow16 = (int16_t *)malloc((size_t)Mpmax * K * sizeof(int16_t));
  int16_t *Bkn16 = (int16_t *)malloc((size_t)K * N * sizeof(int16_t));
  int8_t *Bkn8 = (int8_t *)malloc((size_t)K * N);
  int32_t *C = (int32_t *)malloc((size_t)Mpmax * N * sizeof(int32_t));
  int32_t *Cref = (int32_t *)malloc((size_t)Mpmax * N * sizeof(int32_t));

  const long ktM = K / 8;
  for (long mi = 0; mi < Mpmax / 4; ++mi)
    for (long kf = 0; kf < ktM; ++kf)
      for (int ml = 0; ml < 4; ++ml)
        for (int kl = 0; kl < 8; ++kl)
          Atile[(mi * ktM + kf) * 32 + ml * 8 + kl] =
              (int8_t)Alog[(mi * 4 + ml) * K + kf * 8 + kl];
  for (long nj = 0; nj < N / 4; ++nj)
    for (long kf = 0; kf < ktM; ++kf)
      for (int nl = 0; nl < 4; ++nl)
        for (int kl = 0; kl < 8; ++kl)
          Btile[(nj * ktM + kf) * 32 + nl * 8 + kl] =
              (int8_t)Wlog[(nj * 4 + nl) * K + kf * 8 + kl];
  for (long m = 0; m < Mpmax; ++m)
    for (long k = 0; k < K; ++k) Arow16[m * K + k] = (int16_t)Alog[m * K + k];
  for (long k = 0; k < K; ++k)
    for (long n = 0; n < N; ++n) {
      Bkn16[k * N + n] = (int16_t)Wlog[n * K + k];
      Bkn8[k * N + n] = (int8_t)Wlog[n * K + k];
    }

  g_Atile = Atile; g_Btile = Btile; g_Arow16 = Arow16; g_Bkn16 = Bkn16;
  g_Bkn8 = Bkn8; g_C = C; g_N = N; g_K = K;

  // ---- fingerprint / adversary-identity probe (printed once) ----
  size_t vl_probe = __riscv_vsetvl_e8m1(1024); // VLEN/8 at e8m1
  printf("# G4-M3 T5b q4_0@ime paradigm-lever  phase=matmul(prefill M-sweep)+GEVM(M=1)\n");
  printf("# board=k1(SpacemiT X60)  VLEN=%zubit  harts_pinned=0-3(taskset)\n",
         vl_probe * 8);
  printf("# format=q4_0@ime  N=%ld K=%ld  gates=structural-sealed(0perf)  ledger=KERNEL(compiler-symmetric)+SYSTEM\n",
         N, K);
  printf("# paradigms: IME-matrix(vmadot 0xe210312b) vs RVV-vector(rvv-1.0 intrinsics)\n");
  printf("# opponent-identity: SELF(our RVV-vector construction) -- range-vs-range ablation, NOT vs-ggml, NO eight-gate release\n");
  printf("# both-ranges-compiler: k1 SpacemiT gcc -O2 -march=rv64gcv_xsmtvdotii1p0 (SAME binary => kernel-symmetric)\n");
  printf("# k1-shipped-baseline-compiler: clang-18 (not exercised here; no vs-shipped claim)\n");
  printf("# NG-4: methodology framing (paradigm lever / crossover M*), no 'IME X-times' headline, no e2e beat\n");

  // ---- correctness gate: every cell 0-diff vs ZERO-MODEL, before any timing --
  int any_bad = 0;
  for (int im = 0; im < nM; ++im) {
    long M = Ms[im];
    long Mp = ((M + 3) / 4) * 4;
    gemm_ref(Alog, Wlog, Cref, M, N, K);
    gemm_ime_matrix(Atile, Btile, C, Mp, N, K);
    long b1 = cmp_rows(C, Cref, M, N);
    gemm_ime_matrix_batched(Atile, Btile, C, Mp, N, K);
    long b1b = cmp_rows(C, Cref, M, N);
    gemm_rvv_tiled(Atile, Btile, C, Mp, N, K);
    long b2 = cmp_rows(C, Cref, M, N);
    gemm_rvv_veclayout(Arow16, Bkn16, C, M, N, K);
    long b3 = cmp_rows(C, Cref, M, N);
    gemm_rvv_veclayout_mr4(Arow16, Bkn16, C, M, N, K);
    long b3b = cmp_rows(C, Cref, M, N);
    gemm_rvv_veclayout_mr4_i8(Arow16, Bkn8, C, M, N, K);
    long b3c = cmp_rows(C, Cref, M, N);
    if (b1 || b1b || b2 || b3 || b3b || b3c) {
      printf("# CORRECTNESS-FAIL M=%ld  ime=%ld ime-batched=%ld rvv-tiled=%ld rvv-vec=%ld rvv-vec-mr4=%ld rvv-vec-mr4-i8=%ld\n",
             M, b1, b1b, b2, b3, b3b, b3c);
      any_bad = 1;
    }
  }
  if (any_bad) { printf("ZERO-MODEL FAIL -- timing aborted\n"); return 1; }
  printf("# ZERO-MODEL: all cells x all M are int32-EXACT vs plain-GEMM reference (0-diff)\n");

  // ---- measurement-hygiene self-check: 3x repeat spread on a fixed config ----
  g_M = 64; g_Mp = 64;
  uint64_t sc[3]; long Rsc;
  for (int i = 0; i < 3; ++i) sc[i] = time_gemm(run_cell3, &Rsc, 300000ull);
  uint64_t smin = sc[0], smax = sc[0];
  for (int i = 1; i < 3; ++i) { if (sc[i] < smin) smin = sc[i]; if (sc[i] > smax) smax = sc[i]; }
  double spread = 100.0 * (double)(smax - smin) / (double)smin;
  printf("# self-check(M=64,Cell3,3x): %llu/%llu/%llu ns  spread=%.2f%%  %s\n",
         (unsigned long long)sc[0], (unsigned long long)sc[1],
         (unsigned long long)sc[2], spread,
         spread <= 5.0 ? "OK(<=5%)" : "NOISY(record+interpret with care)");

  // ---- header ----
  printf("# ratio column = ns_per_gemm / Cell1b(register-resident IME) = fair matrix baseline; <1 => faster than batched-IME\n");
  printf("M,M_ime_padded,cell,paradigm,layout,ns_per_gemm_median,macs,ratio_vs_Cell1b\n");

  // ---- paired-alternation timing sweep ----
  for (int im = 0; im < nM; ++im) {
    long M = Ms[im];
    long Mp = ((M + 3) / 4) * 4;
    uint64_t t1[NREP], t1b[NREP], t2[NREP], t3[NREP], t3b[NREP], t3c[NREP];
    long R;
    for (int r = 0; r < NREP; ++r) {
      g_M = M; g_Mp = Mp;
      t1[r] = time_gemm(run_cell1, &R, 300000ull);  // IME-matrix @ tiled (leaf)
      t1b[r] = time_gemm(run_cell1b, &R, 300000ull); // IME-matrix @ tiled (batched)
      t2[r] = time_gemm(run_cell2, &R, 300000ull);  // RVV-vector @ tiled
      t3[r] = time_gemm(run_cell3, &R, 300000ull);  // RVV-vector @ vector-opt (per-row)
      t3b[r] = time_gemm(run_cell3b, &R, 300000ull); // RVV-vector @ vector-opt (MR=4,i16)
      t3c[r] = time_gemm(run_cell3c, &R, 300000ull); // RVV-vector @ vector-opt (MR=4,i8)
    }
    uint64_t m1 = umedian(t1, NREP), m1b = umedian(t1b, NREP);
    uint64_t m2 = umedian(t2, NREP), m3 = umedian(t3, NREP);
    uint64_t m3b = umedian(t3b, NREP), m3c = umedian(t3c, NREP);
    long macs = M * N * K;
    // ratio column = ns / Cell1b (register-resident IME) = fair matrix baseline
    printf("%ld,%ld,Cell1,IME-matrix-leaf,tiled,%llu,%ld,%.3f\n", M, Mp,
           (unsigned long long)m1, macs, (double)m1 / (double)m1b);
    printf("%ld,%ld,Cell1b,IME-matrix-batched,tiled,%llu,%ld,1.000\n", M, Mp,
           (unsigned long long)m1b, macs);
    printf("%ld,%ld,Cell2,RVV-vector,tiled,%llu,%ld,%.3f\n", M, Mp,
           (unsigned long long)m2, macs, (double)m2 / (double)m1b);
    printf("%ld,%ld,Cell3,RVV-vector,vector-opt-perrow,%llu,%ld,%.3f\n", M, Mp,
           (unsigned long long)m3, macs, (double)m3 / (double)m1b);
    printf("%ld,%ld,Cell3b,RVV-vector,vector-opt-mr4-i16w,%llu,%ld,%.3f\n", M, Mp,
           (unsigned long long)m3b, macs, (double)m3b / (double)m1b);
    printf("%ld,%ld,Cell3c,RVV-vector,vector-opt-mr4-i8w,%llu,%ld,%.3f\n", M, Mp,
           (unsigned long long)m3c, macs, (double)m3c / (double)m1b);
    fflush(stdout);
  }

  free(Alog); free(Wlog); free(Atile); free(Btile);
  free(Arow16); free(Bkn16); free(Bkn8); free(C); free(Cref);
  return 0;
}
