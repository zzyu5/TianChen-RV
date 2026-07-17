// INC-11 (K-quant K1) byte-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new typed op
// tcrv_rvv.q6_k_q8_k_aux32_partial computes the SAME per-super-block INTEGER
// state aux32[8] as ggml's own reference (ggml_vec_dot_q6_K_q8_K_generic,
// llama.cpp/ggml/src/ggml-cpu/quants.c:800) computes right BEFORE the fp32
// d-multiply, byte-exact (exact i32 equality), over many random super-blocks
// plus the named edge cases. (The fp32 two-level fold is K2 -- out of scope.)
//
// Block byte layouts (ggml-common.h, QK_K = 256):
//   block_q6_K (210 bytes): ql[128] @0 | qh[64] @128 | scales[16] @192 | d(fp16) @208
//   block_q8_K (292 bytes): d(fp32) @0 | qs[256] @4 | bsums[16] @260
// K1 reproduces aux32 -- the INTEGER state -- so neither d is read here.
//
// The kernel under test is the UNMODIFIED, compiler-emitted tcrv_emitted_kernel.cpp.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstddef>

// ---- The kernel our compiler emitted (verbatim, extern "C") -------------------
// void f(size_t n, int32_t *aux32_out, const uint8_t *vx, const uint8_t *vy);
extern "C" void
tcrv_emitc_ggml_vec_dot_q6_K_q8_K_aux32_kernel_ggml_vec_dot_q6_K_q8_K_aux32(
    size_t n, int32_t *aux32_out, const uint8_t *vx, const uint8_t *vy);

// ---- ggml's own integer partial (the ground truth) ---------------------------
// Mirrors ggml_vec_dot_q6_K_q8_K_generic's per-super-block integer state aux32
// EXACTLY (quants.c:820-847), for one super-block (n = 256). The unpack writes an
// element-ordered aux8[256] biased -32; the per-sub-block int8-scaled i32 dot
// accumulates into aux32[0..7]. We expose aux32 (the K1 target) directly.
static void ggml_reference_aux32(const uint8_t *q6k, const uint8_t *q8k,
                                 int32_t aux32[8]) {
  const uint8_t *q4 = q6k + 0;        // ql @ 0
  const uint8_t *qh = q6k + 128;      // qh @ 128
  const int8_t *scales = (const int8_t *)(q6k + 192); // scales @ 192
  const int8_t *q8 = (const int8_t *)(q8k + 4);       // qs @ 4

  int8_t aux8[256];
  int16_t aux16[8];
  std::memset(aux32, 0, 8 * sizeof(int32_t));

  int8_t *a = aux8;
  const uint8_t *q4p = q4;
  const uint8_t *qhp = qh;
  for (int j = 0; j < 256; j += 128) {
    for (int l = 0; l < 32; ++l) {
      a[l + 0] =
          (int8_t)((q4p[l + 0] & 0xF) | (((qhp[l] >> 0) & 3) << 4)) - 32;
      a[l + 32] =
          (int8_t)((q4p[l + 32] & 0xF) | (((qhp[l] >> 2) & 3) << 4)) - 32;
      a[l + 64] =
          (int8_t)((q4p[l + 0] >> 4) | (((qhp[l] >> 4) & 3) << 4)) - 32;
      a[l + 96] =
          (int8_t)((q4p[l + 32] >> 4) | (((qhp[l] >> 6) & 3) << 4)) - 32;
    }
    a += 128;
    q4p += 64;
    qhp += 32;
  }
  a = aux8;
  const int8_t *q8p = q8;
  int is = 0;
  for (int j = 0; j < 256 / 16; ++j) {
    int scale = scales[is++];
    for (int l = 0; l < 8; ++l) aux16[l] = q8p[l] * a[l];
    for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
    q8p += 8; a += 8;
    for (int l = 0; l < 8; ++l) aux16[l] = q8p[l] * a[l];
    for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
    q8p += 8; a += 8;
  }
}

static unsigned g_rng = 0x6b5f1c27u;
static unsigned next_rand() {
  // xorshift32 (deterministic, host-independent)
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}

// One super-block: nb = 1, so the kernel writes aux32[0..7] for ib = 0.
static int check_block(const uint8_t *q6k, const uint8_t *q8k,
                       const char *label) {
  int32_t ref[8];
  int32_t got[8];
  std::memset(got, 0, sizeof(got));
  ggml_reference_aux32(q6k, q8k, ref);
  tcrv_emitc_ggml_vec_dot_q6_K_q8_K_aux32_kernel_ggml_vec_dot_q6_K_q8_K_aux32(
      256, got, q6k, q8k);
  for (int l = 0; l < 8; ++l) {
    if (ref[l] != got[l]) {
      printf("FAIL [%s]: lane %d ref=%d tcrv=%d\n", label, l, ref[l], got[l]);
      return 1;
    }
  }
  return 0;
}

// Build one random super-block pair into the exact ggml byte layout.
static void fill_random(uint8_t *q6k, uint8_t *q8k) {
  for (int i = 0; i < 128; ++i) q6k[i] = (uint8_t)(next_rand() & 0xFF); // ql
  for (int i = 0; i < 64; ++i) q6k[128 + i] = (uint8_t)(next_rand() & 0xFF); // qh
  for (int i = 0; i < 16; ++i) q6k[192 + i] = (uint8_t)(next_rand() & 0xFF); // scales
  q6k[208] = 0; q6k[209] = 0; // d (unused by K1)
  // q8_K: d (unused), qs[256], bsums (unused)
  for (int i = 0; i < 4; ++i) q8k[i] = 0;
  for (int i = 0; i < 256; ++i) q8k[4 + i] = (uint8_t)(next_rand() & 0xFF); // qs
  for (int i = 0; i < 32; ++i) q8k[260 + i] = 0; // bsums
}

int main() {
  int failures = 0;
  int checked = 0;
  uint8_t q6k[210];
  uint8_t q8k[292];

  auto set_q6_all_nibble = [&](uint8_t lowNib, uint8_t qhByte) {
    for (int i = 0; i < 128; ++i) q6k[i] = (uint8_t)((lowNib << 4) | lowNib);
    for (int i = 0; i < 64; ++i) q6k[128 + i] = qhByte;
  };
  auto set_scales = [&](int8_t v) {
    for (int i = 0; i < 16; ++i) q6k[192 + i] = (uint8_t)v;
  };
  auto set_q8 = [&](int8_t v) {
    for (int i = 0; i < 256; ++i) q8k[4 + i] = (uint8_t)v;
  };

  // ---- Named edge cases --------------------------------------------------------
  std::memset(q6k, 0, sizeof(q6k));
  std::memset(q8k, 0, sizeof(q8k));

  // q6 all -32: ql nibble 0x0, qh 2-bit 0 -> q6 = 0 -> -32. scales +1, q8 +1.
  set_q6_all_nibble(0x0, 0x00); set_scales(1); set_q8(1);
  failures += check_block(q6k, q8k, "q6-all-minus32 scale+1 q8+1"); ++checked;

  // q6 all +31: ql nibble 0xF, qh all-ones byte (each 2-bit field = 3) -> q6=63 -> +31.
  set_q6_all_nibble(0xF, 0xFF); set_scales(1); set_q8(1);
  failures += check_block(q6k, q8k, "q6-all-plus31 scale+1 q8+1"); ++checked;

  // q6 all +31, scales at the int8 extremes (+127 / -128 alternating), q8 +127.
  set_q6_all_nibble(0xF, 0xFF);
  for (int i = 0; i < 16; ++i) q6k[192 + i] = (uint8_t)((i & 1) ? 127 : -128);
  set_q8(127);
  failures += check_block(q6k, q8k, "q6+31 scales-extremes q8+127"); ++checked;

  // q6 all -32, q8 all -128 (the asymmetric int8 extreme), scales -128.
  set_q6_all_nibble(0x0, 0x00); set_scales(-128); set_q8(-128);
  failures += check_block(q6k, q8k, "q6-32 scale-128 q8-128"); ++checked;

  // q6 all +31, q8 all -128, scales +127 (max positive product magnitude).
  set_q6_all_nibble(0xF, 0xFF); set_scales(127); set_q8(-128);
  failures += check_block(q6k, q8k, "q6+31 scale+127 q8-128"); ++checked;

  // Mixed sweep: ql/qh/scales/q8 a structured ramp.
  for (int i = 0; i < 128; ++i) q6k[i] = (uint8_t)((i * 17) & 0xFF);
  for (int i = 0; i < 64; ++i) q6k[128 + i] = (uint8_t)((i * 29) & 0xFF);
  for (int i = 0; i < 16; ++i) q6k[192 + i] = (uint8_t)((i * 9) - 64);
  for (int i = 0; i < 256; ++i) q8k[4 + i] = (uint8_t)((i & 1) ? 127 : -128);
  failures += check_block(q6k, q8k, "mixed-ramp q8-extremes"); ++checked;

  // ---- Many random super-blocks ------------------------------------------------
  const int N_RANDOM = 4000;
  for (int b = 0; b < N_RANDOM; ++b) {
    fill_random(q6k, q8k);
    char label[32];
    snprintf(label, sizeof(label), "random#%d", b);
    failures += check_block(q6k, q8k, label);
    ++checked;
  }

  // ---- Multi-super-block runs (exercise the outer loop + out+ib*8 indexing) ----
  // n = NB*256: the kernel must write NB independent aux32[8] groups; the
  // reference recomputes each super-block's aux32 and compares the whole array.
  for (int NB = 2; NB <= 8; ++NB) {
    uint8_t *vx = (uint8_t *)malloc((size_t)NB * 210);
    uint8_t *vy = (uint8_t *)malloc((size_t)NB * 292);
    int32_t *ref = (int32_t *)malloc((size_t)NB * 8 * sizeof(int32_t));
    int32_t *got = (int32_t *)calloc((size_t)NB * 8, sizeof(int32_t));
    for (int ib = 0; ib < NB; ++ib) {
      fill_random(vx + (size_t)ib * 210, vy + (size_t)ib * 292);
      ggml_reference_aux32(vx + (size_t)ib * 210, vy + (size_t)ib * 292,
                           ref + (size_t)ib * 8);
    }
    tcrv_emitc_ggml_vec_dot_q6_K_q8_K_aux32_kernel_ggml_vec_dot_q6_K_q8_K_aux32(
        (size_t)NB * 256, got, vx, vy);
    int local = 0;
    for (int l = 0; l < NB * 8; ++l)
      if (ref[l] != got[l]) {
        printf("FAIL [multi NB=%d]: idx %d ref=%d tcrv=%d\n", NB, l, ref[l],
               got[l]);
        ++local;
      }
    failures += local ? 1 : 0;
    ++checked;
    free(vx); free(vy); free(ref); free(got);
  }

  printf("INC-11 q6_K K1 byte-exact check: %d super-blocks checked, %d failures\n",
         checked, failures);
  if (failures == 0)
    printf("RESULT: PASS (aux32[8] exactly equal vs ggml _generic for all "
           "super-blocks)\n");
  else
    printf("RESULT: FAIL\n");
  return failures == 0 ? 0 : 1;
}
