// INC-23 (q4_K K4a) byte-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new typed op
// tcrv_rvv.q4_k_q8_k_aux_partial computes the SAME per-super-block INTEGER state
// as ggml's own reference (ggml_vec_dot_q4_K_q8_K_generic,
// llama.cpp/ggml/src/ggml-cpu/quants.c:644) computes right BEFORE the fp32
// d/dmin fold, byte-exact:
//   (1) aux32[8] -- the per-sub-block uint6-scaled i32 accumulator (exact i32
//       equality), AND
//   (2) scales[8] + mins[8] -- the 16 decoded 6-bit scale/min bytes from the
//       get_scale_min_k4 / utmp/kmask bit-dance (exact byte equality).
// over many random super-blocks + the named edge cases. (The fp32 fold + the min
// term + the ABI are K4b -- out of scope.)
//
// Block byte layouts (ggml-common.h, QK_K = 256):
//   block_q4_K (144 bytes): d(fp16) @0 | dmin(fp16) @2 | scales[12] @4 | qs[128] @16
//   block_q8_K (292 bytes): d(fp32) @0 | qs[256] @4 | bsums[16] @260
// K4a reproduces aux32 + the decoded scales/mins -- so neither fp16/fp32 d is read.
//
// The kernel under test is the UNMODIFIED, compiler-emitted tcrv_emitted_kernel.cpp.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstddef>

// ---- The kernel our compiler emitted (verbatim, extern "C") -------------------
// void f(size_t n, int32_t *aux32_out, uint8_t *scalemin_out,
//        const uint8_t *vx, const uint8_t *vy);
extern "C" void
tcrv_emitc_ggml_vec_dot_q4_K_q8_K_aux_kernel_ggml_vec_dot_q4_K_q8_K_aux(
    size_t n, int32_t *aux32_out, uint8_t *scalemin_out, const uint8_t *vx,
    const uint8_t *vy);

// ---- ggml's own integer partial (the ground truth) ---------------------------
// Mirrors ggml_vec_dot_q4_K_q8_K_generic's per-super-block integer state EXACTLY
// (quants.c:674-714), for one super-block (n = 256). The unpack writes an
// element-ordered aux8[256] (plain 4-bit, NO bias); the get_scale_min_k4 /
// utmp/kmask bit-dance decodes 8 scales + 8 mins; the per-sub-block uint6-scaled
// i32 dot accumulates into aux32[0..7]. We expose aux32 AND scales/mins (the K4a
// target) directly.
static void ggml_reference_partial(const uint8_t *q4k, const uint8_t *q8k,
                                   int32_t aux32[8], uint8_t scales_out[8],
                                   uint8_t mins_out[8]) {
  const uint8_t *q4 = q4k + 16;                 // qs @ 16
  const int8_t *q8 = (const int8_t *)(q8k + 4); // qs @ 4

  static const uint32_t kmask1 = 0x3f3f3f3f;
  static const uint32_t kmask2 = 0x0f0f0f0f;
  static const uint32_t kmask3 = 0x03030303;

  uint32_t utmp[4];
  const uint8_t *scales = (const uint8_t *)&utmp[0];
  const uint8_t *mins = (const uint8_t *)&utmp[2];

  int8_t aux8[256];
  int16_t aux16[8];
  std::memset(aux32, 0, 8 * sizeof(int32_t));

  // The plain 4-bit nibble unpack (quants.c:677-682), element-ordered, NO bias.
  int8_t *a = aux8;
  const uint8_t *q4p = q4;
  for (int j = 0; j < 256 / 64; ++j) {
    for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4p[l] & 0xF);
    a += 32;
    for (int l = 0; l < 32; ++l) a[l] = (int8_t)(q4p[l] >> 4);
    a += 32;
    q4p += 32;
  }

  // The 6-bit scale/min bit-dance (quants.c:685-690).
  std::memcpy(utmp, q4k + 4, 12); // scales[12] @ 4
  utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
  const uint32_t uaux = utmp[1] & kmask1;
  utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
  utmp[2] = uaux;
  utmp[0] &= kmask1;

  // The per-sub-block uint6-scaled i32 dot (quants.c:697-714): 8 sub-blocks of
  // 32, processed as 4 halves of 8.
  a = aux8;
  const int8_t *q8p = q8;
  int is = 0;
  for (int j = 0; j < 256 / 32; ++j) {
#ifdef NEGCTRL
    // Negative control: SIGN-extend the 6-bit scale (the q6_K bug) instead of
    // zero-extending the q4_K unsigned scale. A correct kernel must DIVERGE from
    // this, proving the byte-exact check discriminates the uint6 model.
    int32_t scale = (int32_t)(int8_t)(scales[is++] << 2) >> 2;
#else
    int32_t scale = scales[is++];
#endif
    for (int q = 0; q < 4; ++q) {
      for (int l = 0; l < 8; ++l) aux16[l] = q8p[l] * a[l];
      for (int l = 0; l < 8; ++l) aux32[l] += scale * aux16[l];
      q8p += 8;
      a += 8;
    }
  }

  for (int l = 0; l < 8; ++l) scales_out[l] = scales[l];
  for (int l = 0; l < 8; ++l) mins_out[l] = mins[l];
}

static unsigned g_rng = 0x6b5f1c27u;
static unsigned next_rand() {
  // xorshift32 (deterministic, host-independent)
  g_rng ^= g_rng << 13;
  g_rng ^= g_rng >> 17;
  g_rng ^= g_rng << 5;
  return g_rng;
}

// One super-block: nb = 1, so the kernel writes aux32[0..7] + scalemin[0..15].
static int check_block(const uint8_t *q4k, const uint8_t *q8k,
                       const char *label) {
  int32_t ref_aux32[8];
  uint8_t ref_scales[8], ref_mins[8];
  int32_t got_aux32[8];
  uint8_t got_scalemin[16];
  std::memset(got_aux32, 0, sizeof(got_aux32));
  std::memset(got_scalemin, 0, sizeof(got_scalemin));
  ggml_reference_partial(q4k, q8k, ref_aux32, ref_scales, ref_mins);
  tcrv_emitc_ggml_vec_dot_q4_K_q8_K_aux_kernel_ggml_vec_dot_q4_K_q8_K_aux(
      256, got_aux32, got_scalemin, q4k, q8k);
  int fail = 0;
  for (int l = 0; l < 8; ++l) {
    if (ref_aux32[l] != got_aux32[l]) {
      printf("FAIL [%s]: aux32 lane %d ref=%d tcrv=%d\n", label, l, ref_aux32[l],
             got_aux32[l]);
      fail = 1;
    }
  }
  for (int l = 0; l < 8; ++l) {
    if (ref_scales[l] != got_scalemin[l]) {
      printf("FAIL [%s]: scale[%d] ref=%u tcrv=%u\n", label, l, ref_scales[l],
             got_scalemin[l]);
      fail = 1;
    }
  }
  for (int l = 0; l < 8; ++l) {
    if (ref_mins[l] != got_scalemin[8 + l]) {
      printf("FAIL [%s]: min[%d] ref=%u tcrv=%u\n", label, l, ref_mins[l],
             got_scalemin[8 + l]);
      fail = 1;
    }
  }
  return fail;
}

// Build one random super-block pair into the exact ggml byte layout.
static void fill_random(uint8_t *q4k, uint8_t *q8k) {
  for (int i = 0; i < 4; ++i) q4k[i] = 0;                                  // d,dmin (unused)
  for (int i = 0; i < 12; ++i) q4k[4 + i] = (uint8_t)(next_rand() & 0xFF); // scales[12]
  for (int i = 0; i < 128; ++i) q4k[16 + i] = (uint8_t)(next_rand() & 0xFF); // qs[128]
  for (int i = 0; i < 4; ++i) q8k[i] = 0;                                  // d (unused)
  for (int i = 0; i < 256; ++i) q8k[4 + i] = (uint8_t)(next_rand() & 0xFF); // qs[256]
  for (int i = 0; i < 32; ++i) q8k[260 + i] = 0;                           // bsums (unused)
}

int main() {
  int failures = 0;
  int checked = 0;
  uint8_t q4k[144];
  uint8_t q8k[292];

  auto set_q4_nibble = [&](uint8_t nib) {
    for (int i = 0; i < 128; ++i) q4k[16 + i] = (uint8_t)((nib << 4) | nib);
  };
  auto set_scales_raw = [&](uint8_t v) {
    for (int i = 0; i < 12; ++i) q4k[4 + i] = v;
  };
  auto set_q8 = [&](int8_t v) {
    for (int i = 0; i < 256; ++i) q8k[4 + i] = (uint8_t)v;
  };

  // ---- Named edge cases --------------------------------------------------------
  std::memset(q4k, 0, sizeof(q4k));
  std::memset(q8k, 0, sizeof(q8k));

  // q4 all 0, scales packed all 0x00, q8 0 -> aux32 all 0, scales/mins 0.
  set_q4_nibble(0x0); set_scales_raw(0x00); set_q8(0);
  failures += check_block(q4k, q8k, "q4-0 scales-0x00 q8-0"); ++checked;

  // q4 all 15 (unsigned, no bias), scales packed all 0xFF (exercises the >>6,
  // &kmask corners), q8 +127.
  set_q4_nibble(0xF); set_scales_raw(0xFF); set_q8(127);
  failures += check_block(q4k, q8k, "q4-15 scales-0xFF q8+127"); ++checked;

  // q4 all 15, scales packed all 0x3F (6-bit max in low bytes), q8 -128.
  set_q4_nibble(0xF); set_scales_raw(0x3F); set_q8(-128);
  failures += check_block(q4k, q8k, "q4-15 scales-0x3F q8-128"); ++checked;

  // q4 all 0, q8 -128, scales 0xAA (cross-byte/cross-word bit pattern).
  set_q4_nibble(0x0); set_scales_raw(0xAA); set_q8(-128);
  failures += check_block(q4k, q8k, "q4-0 scales-0xAA q8-128"); ++checked;

  // q4 all 15, q8 +127, distinct per-byte packed scales (each scale byte unique
  // -> exercises the cross-byte shuffle, not just a uniform pattern).
  set_q4_nibble(0xF);
  for (int i = 0; i < 12; ++i) q4k[4 + i] = (uint8_t)(0x80 + i * 7);
  set_q8(127);
  failures += check_block(q4k, q8k, "q4-15 scales-distinct q8+127"); ++checked;

  // Mixed sweep: qs/scales/q8 a structured ramp.
  for (int i = 0; i < 128; ++i) q4k[16 + i] = (uint8_t)((i * 17) & 0xFF);
  for (int i = 0; i < 12; ++i) q4k[4 + i] = (uint8_t)((i * 23) & 0xFF);
  for (int i = 0; i < 256; ++i) q8k[4 + i] = (uint8_t)((i & 1) ? 127 : -128);
  failures += check_block(q4k, q8k, "mixed-ramp q8-extremes"); ++checked;

  // ---- Many random super-blocks ------------------------------------------------
  const int N_RANDOM = 4000;
  for (int b = 0; b < N_RANDOM; ++b) {
    fill_random(q4k, q8k);
    char label[32];
    snprintf(label, sizeof(label), "random#%d", b);
    failures += check_block(q4k, q8k, label);
    ++checked;
  }

  // ---- Multi-super-block runs (exercise the outer loop + per-block indexing) ----
  // n = NB*256: the kernel must write NB independent aux32[8] groups AND NB
  // independent scalemin[16] groups; the reference recomputes each super-block's
  // state and compares the whole arrays (scales/mins DIFFER per block).
  for (int NB = 2; NB <= 8; ++NB) {
    uint8_t *vx = (uint8_t *)malloc((size_t)NB * 144);
    uint8_t *vy = (uint8_t *)malloc((size_t)NB * 292);
    int32_t *ref_aux32 = (int32_t *)malloc((size_t)NB * 8 * sizeof(int32_t));
    uint8_t *ref_sm = (uint8_t *)malloc((size_t)NB * 16);
    int32_t *got_aux32 = (int32_t *)calloc((size_t)NB * 8, sizeof(int32_t));
    uint8_t *got_sm = (uint8_t *)calloc((size_t)NB * 16, 1);
    for (int ib = 0; ib < NB; ++ib) {
      fill_random(vx + (size_t)ib * 144, vy + (size_t)ib * 292);
      uint8_t sc[8], mn[8];
      ggml_reference_partial(vx + (size_t)ib * 144, vy + (size_t)ib * 292,
                             ref_aux32 + (size_t)ib * 8, sc, mn);
      for (int l = 0; l < 8; ++l) ref_sm[ib * 16 + l] = sc[l];
      for (int l = 0; l < 8; ++l) ref_sm[ib * 16 + 8 + l] = mn[l];
    }
    tcrv_emitc_ggml_vec_dot_q4_K_q8_K_aux_kernel_ggml_vec_dot_q4_K_q8_K_aux(
        (size_t)NB * 256, got_aux32, got_sm, vx, vy);
    int local = 0;
    for (int l = 0; l < NB * 8; ++l)
      if (ref_aux32[l] != got_aux32[l]) {
        printf("FAIL [multi NB=%d]: aux32 idx %d ref=%d tcrv=%d\n", NB, l,
               ref_aux32[l], got_aux32[l]);
        ++local;
      }
    for (int l = 0; l < NB * 16; ++l)
      if (ref_sm[l] != got_sm[l]) {
        printf("FAIL [multi NB=%d]: scalemin idx %d ref=%u tcrv=%u\n", NB, l,
               ref_sm[l], got_sm[l]);
        ++local;
      }
    failures += local ? 1 : 0;
    ++checked;
    free(vx); free(vy); free(ref_aux32); free(ref_sm); free(got_aux32);
    free(got_sm);
  }

  printf("INC-23 q4_K K4a byte-exact check: %d super-blocks checked, %d "
         "failures\n",
         checked, failures);
  if (failures == 0)
    printf("RESULT: PASS (aux32[8] + scales[8] + mins[8] exactly equal vs ggml "
           "_generic for all super-blocks)\n");
  else
    printf("RESULT: FAIL\n");
  return failures == 0 ? 0 : 1;
}
