// widen_kernels.c -- STANDALONE m1-chain vs mf2-chain widening int8 dot-reduce
// kernels for the W2 §③ widen-to-m1 board re-measurement ([GAP-P1] boundary).
//
// This mirrors the repack-GEVM widening chain the RVV plugin's
// selectRepackAccumulatorLMUL reasons over (RVVLowerQuantContraction.cpp), but
// is a SELF-CONTAINED experiment kernel: it does NOT include or link any plugin
// code, and it changes NO selector. It only supplies the board number the
// [GAP-P1] iron rule says is missing (the m1-vs-mf2 per-format crossover is
// board-MEASURED, never projected).
//
// Two chains (both register-pressure-LEGAL at unroll=1, budget=32 vregs):
//   * mf2 chain: e8mf2 load -> vwmul i16m1 product -> vwadd.wv i32m2 accumulate.
//                peak-live = footprint(i16m1)=1 + footprint(i32m2)=2 = 3 vregs.
//   * m1  chain: e8m1  load -> vwmul i16m2 product -> vwadd.wv i32m4 accumulate.
//                peak-live = footprint(i16m2)=2 + footprint(i32m4)=4 = 6 vregs.
// Both leave the 32-vreg file with huge headroom at unroll=1 => SPILL-FREE by
// the rvvRegisterPressureLegal inequality; the only open question is the runtime
// crossover, which is what this measures.
//
// Two formats:
//   * q8-like:  int8 weight  x int8 activation  (no unpack).
//   * q4-like:  int4 weight (nibble, offset-8) unpacked to int8, x int8 act.
//
// Byte-exactness: every RVV kernel returns the IDENTICAL i32 sum as the scalar
// reference (checked in harness.c). The two chains differ only in the LMUL of
// the widening pipeline, not in the arithmetic.

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

// ---- scalar reference (the byte-exact oracle) ----------------------------

int32_t dot_q8_scalar(const int8_t *w, const int8_t *a, size_t n) {
  int32_t acc = 0;
  for (size_t i = 0; i < n; ++i)
    acc += (int32_t)w[i] * (int32_t)a[i];
  return acc;
}

// q4 weight: two nibbles per byte, offset-8 (q4_0 convention: value = nibble-8).
// n is the number of int8 activations; the packed weight has n/2 bytes.
int32_t dot_q4_scalar(const uint8_t *wpacked, const int8_t *a, size_t n) {
  int32_t acc = 0;
  for (size_t i = 0; i < n; ++i) {
    uint8_t byte = wpacked[i >> 1];
    int32_t nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
    int32_t w = nib - 8;
    acc += w * (int32_t)a[i];
  }
  return acc;
}

// ---- q8-like: mf2 chain --------------------------------------------------

int32_t dot_q8_mf2(const int8_t *w, const int8_t *a, size_t n) {
  vint32m2_t acc = __riscv_vmv_v_x_i32m2(0, __riscv_vsetvlmax_e32m2());
  size_t i = 0;
  while (i < n) {
    size_t vl = __riscv_vsetvl_e8mf2(n - i);
    vint8mf2_t vw = __riscv_vle8_v_i8mf2(w + i, vl);
    vint8mf2_t va = __riscv_vle8_v_i8mf2(a + i, vl);
    vint16m1_t prod = __riscv_vwmul_vv_i16m1(vw, va, vl); // i8xi8 -> i16m1
    acc = __riscv_vwadd_wv_i32m2(acc, prod, vl);          // i32m2 += i16m1
    i += vl;
  }
  vint32m1_t z = __riscv_vmv_v_x_i32m1(0, 1);
  vint32m1_t s = __riscv_vredsum_vs_i32m2_i32m1(acc, z, __riscv_vsetvlmax_e32m2());
  return __riscv_vmv_x_s_i32m1_i32(s);
}

// ---- q8-like: m1 chain ---------------------------------------------------

int32_t dot_q8_m1(const int8_t *w, const int8_t *a, size_t n) {
  vint32m4_t acc = __riscv_vmv_v_x_i32m4(0, __riscv_vsetvlmax_e32m4());
  size_t i = 0;
  while (i < n) {
    size_t vl = __riscv_vsetvl_e8m1(n - i);
    vint8m1_t vw = __riscv_vle8_v_i8m1(w + i, vl);
    vint8m1_t va = __riscv_vle8_v_i8m1(a + i, vl);
    vint16m2_t prod = __riscv_vwmul_vv_i16m2(vw, va, vl); // i8xi8 -> i16m2
    acc = __riscv_vwadd_wv_i32m4(acc, prod, vl);          // i32m4 += i16m2
    i += vl;
  }
  vint32m1_t z = __riscv_vmv_v_x_i32m1(0, 1);
  vint32m1_t s = __riscv_vredsum_vs_i32m4_i32m1(acc, z, __riscv_vsetvlmax_e32m4());
  return __riscv_vmv_x_s_i32m1_i32(s);
}

// ---- q4-like: unpack nibble (offset-8) to int8, then the SAME chain -------
// The unpack is a scalar-load + nibble split into an int8 scratch, then the same
// widening chain. This keeps the chain identical; the format differs only by the
// added unpack traffic (which shifts the compute/memory balance).

int32_t dot_q4_mf2(const uint8_t *wpacked, const int8_t *a, int8_t *scratch,
                   size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint8_t byte = wpacked[i >> 1];
    int32_t nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
    scratch[i] = (int8_t)(nib - 8);
  }
  return dot_q8_mf2(scratch, a, n);
}

int32_t dot_q4_m1(const uint8_t *wpacked, const int8_t *a, int8_t *scratch,
                  size_t n) {
  for (size_t i = 0; i < n; ++i) {
    uint8_t byte = wpacked[i >> 1];
    int32_t nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
    scratch[i] = (int8_t)(nib - 8);
  }
  return dot_q8_m1(scratch, a, n);
}
