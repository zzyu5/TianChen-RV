// INC-1 byte-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new
// tcrv_rvv.packed_i4_offset_binary_x_i8_product op computes the SAME per-block
// integer partial `sumi` as ggml's own reference
// (ggml_vec_dot_q4_0_q8_0_generic, llama.cpp/ggml/src/ggml-cpu/quants.c:174),
// byte-exact (exact i32 equality), over many random single Q4_0 x Q8_0 blocks
// plus the named edge cases.
//
// One Q4_0 block = 16 packed bytes qs[0..15] (uint8: low/high nibble per byte).
// One Q8_0 block = 32 int8 q8[0..31].
//   sumi = sum_{j=0..15} ((qs[j]&0x0F)-8)*q8[j] + ((qs[j]>>4)-8)*q8[j+16].
// (INTEGER only -- no fp16 scales in INC-1; those are INC-2.)
//
// The kernel under test is the UNMODIFIED, compiler-emitted tcrv_emitted_kernel.cpp.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>

// ---- The kernel our compiler emitted (verbatim, extern "C") -------------------
extern "C" void
tcrv_emitc_rvv_q4_0_q8_0_integer_core_kernel_rvv_q4_0_q8_0_integer_core(
    const int8_t *w, const int8_t *qlo, const int8_t *qhi, const int32_t *acc,
    int32_t *out, size_t n);

// ---- ggml's own integer partial (the ground truth) ---------------------------
// Mirrors ggml_vec_dot_q4_0_q8_0_generic's inner loop EXACTLY for one block.
// CRITICAL: qs is uint8_t so & 0x0F / >> 4 are LOGICAL (offset-binary decode).
static int32_t ggml_reference_sumi(const uint8_t qs[16], const int8_t q8[32]) {
  int sumi0 = 0;
  int sumi1 = 0;
  for (int j = 0; j < 16; ++j) {
    const int v0 = (qs[j] & 0x0F) - 8; // low  nibble, decoded [-8,7]
    const int v1 = (qs[j] >> 4) - 8;   // high nibble, decoded [-8,7]
    sumi0 += v0 * q8[j];               // low  half  <-> q8[0..15]
    sumi1 += v1 * q8[j + 16];          // high half  <-> q8[16..31]
  }
  return sumi0 + sumi1;
}

// ---- Run the compiler-emitted kernel on one block ----------------------------
// The kernel iterates n=16 mf4 chunks: w = the 16 packed q4 bytes (reinterpreted
// as int8 -- the xor-0x88 in the kernel makes the source signedness irrelevant
// to the bit pattern), qlo = q8[0..15], qhi = q8[16..31], acc = {0}, out = sumi.
static int32_t tcrv_kernel_sumi(const uint8_t qs[16], const int8_t q8[32]) {
  const int8_t *w = reinterpret_cast<const int8_t *>(qs);
  const int8_t *qlo = q8;
  const int8_t *qhi = q8 + 16;
  int32_t acc = 0;
  int32_t out = 0;
  tcrv_emitc_rvv_q4_0_q8_0_integer_core_kernel_rvv_q4_0_q8_0_integer_core(
      w, qlo, qhi, &acc, &out, 16);
  return out;
}

static unsigned g_rng = 0x12345678u;
static unsigned next_rand() {
  // xorshift32 (deterministic, host-independent)
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}

static int check_block(const uint8_t qs[16], const int8_t q8[32],
                       const char *label) {
  int32_t ref = ggml_reference_sumi(qs, q8);
  int32_t got = tcrv_kernel_sumi(qs, q8);
  if (ref != got) {
    printf("FAIL [%s]: ref=%d tcrv=%d\n", label, ref, got);
    return 1;
  }
  return 0;
}

int main() {
  int failures = 0;
  int checked = 0;

  // ---- Named edge cases --------------------------------------------------------
  {
    uint8_t qs[16];
    int8_t q8[32];

    // All q4 nibbles 0x00 -> every decoded weight = -8.
    for (int i = 0; i < 16; ++i) qs[i] = 0x00;
    for (int i = 0; i < 32; ++i) q8[i] = (int8_t)(i - 16);
    failures += check_block(qs, q8, "all-q4-0x00"); ++checked;

    // All q4 nibbles 0xFF -> every decoded weight = +7.
    for (int i = 0; i < 16; ++i) qs[i] = 0xFF;
    for (int i = 0; i < 32; ++i) q8[i] = (int8_t)(127 - i * 5);
    failures += check_block(qs, q8, "all-q4-0xFF"); ++checked;

    // q4 = 0x88 -> both nibbles decode to 0 (the offset-binary zero).
    for (int i = 0; i < 16; ++i) qs[i] = 0x88;
    for (int i = 0; i < 32; ++i) q8[i] = 127;
    failures += check_block(qs, q8, "all-q4-0x88-zero"); ++checked;

    // Mixed nibbles, q8 saturated at the int8 extremes (+127 / -128).
    for (int i = 0; i < 16; ++i) qs[i] = (uint8_t)((i * 17) & 0xFF);
    for (int i = 0; i < 32; ++i) q8[i] = (int8_t)((i & 1) ? 127 : -128);
    failures += check_block(qs, q8, "mixed-q4-q8-extremes"); ++checked;

    // q8 all -128 (the asymmetric extreme), q4 a sweep 0x0F..0xF0.
    for (int i = 0; i < 16; ++i) qs[i] = (uint8_t)((i << 4) | (15 - i));
    for (int i = 0; i < 32; ++i) q8[i] = -128;
    failures += check_block(qs, q8, "q8-all-neg128"); ++checked;
  }

  // ---- Many random blocks (incl. random edge values) ---------------------------
  const int N_RANDOM = 4000;
  for (int b = 0; b < N_RANDOM; ++b) {
    uint8_t qs[16];
    int8_t q8[32];
    for (int i = 0; i < 16; ++i) qs[i] = (uint8_t)(next_rand() & 0xFF);
    for (int i = 0; i < 32; ++i) q8[i] = (int8_t)(next_rand() & 0xFF);
    char label[32];
    snprintf(label, sizeof(label), "random#%d", b);
    failures += check_block(qs, q8, label);
    ++checked;
  }

  printf("INC-1 byte-exact check: %d blocks checked, %d failures\n", checked,
         failures);
  if (failures == 0)
    printf("RESULT: PASS (sumi exactly equal vs ggml _generic for all blocks)\n");
  else
    printf("RESULT: FAIL\n");
  return failures == 0 ? 0 : 1;
}
