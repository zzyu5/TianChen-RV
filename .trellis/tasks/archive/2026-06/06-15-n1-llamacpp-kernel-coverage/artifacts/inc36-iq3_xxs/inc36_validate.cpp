// INC-36 iq3_xxs x q8_K byte-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.iq3_xxs_q8_k_block_dot op
// (the COMPLETE ggml ggml_vec_dot_iq3_xxs_q8_K GRID-codebook super-block kernel: AoS
// QK_K=256 super-block loop + per-super-block d scale + per-sub-block 4-bit integer
// scale from the packed aux uint32 + GRID-of-4-codebook decode + ksigns SIGN-plane
// application + integer bsum fold + 0.25f + *s store) computes the SAME fp32 result *s
// as ggml's OWN _generic iq3_xxs reference, BIT-FOR-BIT (memcmp of the float bits),
// over random block_iq3_xxs x block_q8_K arrays at n multiples of 256 + named edge
// cases (grid index range, sign patterns, scale extremes, q8 +/-127). THREE negative
// controls (a WRONG grid + WRONG signs + WRONG scale) must FAIL -> proves the grid,
// sign plane, AND scale extraction are the live, load-bearing mechanisms.
//
// iq3_xxs is the iq3 GRID variant of the deep IQ tail -- the GRID-of-4 codebook class.
// Each weight byte INDEXES a 256-entry packed uint32 GRID codebook (iq3xxs_grid, each
// entry = 4 int8 values -- HALF the lane width of iq2_xxs's grid-of-8), the per-element
// SIGN comes from the SAME SIGN PLANE as iq2_xxs (ksigns_iq2xs/kmask), the 4-bit scale
// folds in the INTEGER domain, with a trailing 0.25f. The grid indices live in the
// first 64 bytes of qs[96] (q3); the per-sub-block aux uint32 (sign selectors + scale)
// lives in the trailing 32 bytes (gas = qs + QK_K/4).
//
// REFERENCE: ggml's VERBATIM _generic (quants.c:999-1041): aux32 from gas (one uint32
// per ib32), ls = 2*(aux32>>28)+1, grid1 = (uint8_t*)(iq3xxs_grid + q3[2l+0]), grid2 =
// (uint8_t*)(iq3xxs_grid + q3[2l+1]), signs = ksigns_iq2xs[(aux32>>7l)&127], sumi +=
// grid1[j]*q8[j]*(signs&kmask[j]?-1:1) + grid2[j]*q8[j+4]*(signs&kmask[j+4]?-1:1),
// bsum += sumi*ls, sumf += d*bsum, *s = 0.25f*sumf. FP16->FP32 mirrors the board's
// scalar _Float16 widening (lossless). Whole TU built with the same -ffp-contract.
//
// The kernel under test is the UNMODIFIED, compiler-emitted iq3_xxs kernel
// (iq3_xxs_emitted.cpp) -- every line tagged source_op=tcrv_rvv.iq3_xxs_q8_k_block_dot.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>

#define QK_K 256
static const int IQ3XXS_STRIDE = 98;  // sizeof block_iq3_xxs = 2 + 3*QK_K/8
static const int Q8K_STRIDE = 292;    // sizeof block_q8_K

// ---- block formats (ggml-common.h, QK_K = 256) -------------------------------
//   block_iq3_xxs (98 bytes): d(fp16) @0 | qs[96](uint8) @2 (64 grid idx + 32 aux)
//   block_q8_K   (292 bytes): d(fp32) @0 | qs[256] @4 | bsums[16] @260 (unused)
typedef struct { uint16_t d; uint8_t qs[3*QK_K/8]; } block_iq3_xxs;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;

// ---- ggml's GRID-of-4 codebook + SIGN plane + kmask (ggml-common.h) -----------
static const uint32_t iq3xxs_grid[256] = {0x04040404U, 0x04040414U, 0x04040424U, 0x04040c0cU, 0x04040c1cU, 0x04040c3eU, 0x04041404U, 0x04041414U, 0x04041c0cU, 0x04042414U, 0x04043e1cU, 0x04043e2cU, 0x040c040cU, 0x040c041cU, 0x040c0c04U, 0x040c0c14U, 0x040c140cU, 0x040c142cU, 0x040c1c04U, 0x040c1c14U, 0x040c240cU, 0x040c2c24U, 0x040c3e04U, 0x04140404U, 0x04140414U, 0x04140424U, 0x04140c0cU, 0x04141404U, 0x04141414U, 0x04141c0cU, 0x04141c1cU, 0x04141c3eU, 0x04142c0cU, 0x04142c3eU, 0x04143e2cU, 0x041c040cU, 0x041c043eU, 0x041c0c04U, 0x041c0c14U, 0x041c142cU, 0x041c3e04U, 0x04240c1cU, 0x04241c3eU, 0x04242424U, 0x04242c3eU, 0x04243e1cU, 0x04243e2cU, 0x042c040cU, 0x042c043eU, 0x042c1c14U, 0x042c2c14U, 0x04341c2cU, 0x04343424U, 0x043e0c04U, 0x043e0c24U, 0x043e0c34U, 0x043e241cU, 0x043e340cU, 0x0c04040cU, 0x0c04041cU, 0x0c040c04U, 0x0c040c14U, 0x0c04140cU, 0x0c04141cU, 0x0c041c04U, 0x0c041c14U, 0x0c041c24U, 0x0c04243eU, 0x0c042c04U, 0x0c0c0404U, 0x0c0c0414U, 0x0c0c0c0cU, 0x0c0c1404U, 0x0c0c1414U, 0x0c14040cU, 0x0c14041cU, 0x0c140c04U, 0x0c140c14U, 0x0c14140cU, 0x0c141c04U, 0x0c143e14U, 0x0c1c0404U, 0x0c1c0414U, 0x0c1c1404U, 0x0c1c1c0cU, 0x0c1c2434U, 0x0c1c3434U, 0x0c24040cU, 0x0c24042cU, 0x0c242c04U, 0x0c2c1404U, 0x0c2c1424U, 0x0c2c2434U, 0x0c2c3e0cU, 0x0c34042cU, 0x0c3e1414U, 0x0c3e2404U, 0x14040404U, 0x14040414U, 0x14040c0cU, 0x14040c1cU, 0x14041404U, 0x14041414U, 0x14041434U, 0x14041c0cU, 0x14042414U, 0x140c040cU, 0x140c041cU, 0x140c042cU, 0x140c0c04U, 0x140c0c14U, 0x140c140cU, 0x140c1c04U, 0x140c341cU, 0x140c343eU, 0x140c3e04U, 0x14140404U, 0x14140414U, 0x14140c0cU, 0x14140c3eU, 0x14141404U, 0x14141414U, 0x14141c3eU, 0x14142404U, 0x14142c2cU, 0x141c040cU, 0x141c0c04U, 0x141c0c24U, 0x141c3e04U, 0x141c3e24U, 0x14241c2cU, 0x14242c1cU, 0x142c041cU, 0x142c143eU, 0x142c240cU, 0x142c3e24U, 0x143e040cU, 0x143e041cU, 0x143e0c34U, 0x143e242cU, 0x1c04040cU, 0x1c040c04U, 0x1c040c14U, 0x1c04140cU, 0x1c04141cU, 0x1c042c04U, 0x1c04342cU, 0x1c043e14U, 0x1c0c0404U, 0x1c0c0414U, 0x1c0c1404U, 0x1c0c1c0cU, 0x1c0c2424U, 0x1c0c2434U, 0x1c14040cU, 0x1c14041cU, 0x1c140c04U, 0x1c14142cU, 0x1c142c14U, 0x1c143e14U, 0x1c1c0c0cU, 0x1c1c1c1cU, 0x1c241c04U, 0x1c24243eU, 0x1c243e14U, 0x1c2c0404U, 0x1c2c0434U, 0x1c2c1414U, 0x1c2c2c2cU, 0x1c340c24U, 0x1c341c34U, 0x1c34341cU, 0x1c3e1c1cU, 0x1c3e3404U, 0x24040424U, 0x24040c3eU, 0x24041c2cU, 0x24041c3eU, 0x24042c1cU, 0x24042c3eU, 0x240c3e24U, 0x24141404U, 0x24141c3eU, 0x24142404U, 0x24143404U, 0x24143434U, 0x241c043eU, 0x241c242cU, 0x24240424U, 0x24242c0cU, 0x24243424U, 0x242c142cU, 0x242c241cU, 0x242c3e04U, 0x243e042cU, 0x243e0c04U, 0x243e0c14U, 0x243e1c04U, 0x2c040c14U, 0x2c04240cU, 0x2c043e04U, 0x2c0c0404U, 0x2c0c0434U, 0x2c0c1434U, 0x2c0c2c2cU, 0x2c140c24U, 0x2c141c14U, 0x2c143e14U, 0x2c1c0414U, 0x2c1c2c1cU, 0x2c240c04U, 0x2c24141cU, 0x2c24143eU, 0x2c243e14U, 0x2c2c0414U, 0x2c2c1c0cU, 0x2c342c04U, 0x2c3e1424U, 0x2c3e2414U, 0x34041424U, 0x34042424U, 0x34042434U, 0x34043424U, 0x340c140cU, 0x340c340cU, 0x34140c3eU, 0x34143424U, 0x341c1c04U, 0x341c1c34U, 0x34242424U, 0x342c042cU, 0x342c2c14U, 0x34341c1cU, 0x343e041cU, 0x343e140cU, 0x3e04041cU, 0x3e04042cU, 0x3e04043eU, 0x3e040c04U, 0x3e041c14U, 0x3e042c14U, 0x3e0c1434U, 0x3e0c2404U, 0x3e140c14U, 0x3e14242cU, 0x3e142c14U, 0x3e1c0404U, 0x3e1c0c2cU, 0x3e1c1c1cU, 0x3e1c3404U, 0x3e24140cU, 0x3e24240cU, 0x3e2c0404U, 0x3e2c0414U, 0x3e2c1424U, 0x3e341c04U};
static const uint8_t ksigns_iq2xs[128] = {0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255};
static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

// WRONG grid for negative control 1: each entry's bytes incremented by 1. If the
// grid were NOT load-bearing, swapping it would not change the result.
static uint32_t iq3xxs_grid_wrong[256];
// WRONG signs for negative control 2: ksigns with the sign bit (bit 7) cleared, so
// the negate decision flips for half the patterns.
static uint8_t ksigns_wrong[128];

// ---- board fp16 -> fp32 (mirrors riscv_compute_fp16_to_fp32, lossless) -------
static inline float fp16_to_fp32(uint16_t h) {
  _Float16 hf;
  std::memcpy(&hf, &h, sizeof(uint16_t));
  return (float)hf;
}

// ---- The kernel our compiler emitted (declared; defined in the linked .cpp) --
extern "C" void
tcrv_emitc_ggml_vec_dot_iq3_xxs_q8_K_kernel_ggml_vec_dot_iq3_xxs_q8_K(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);

static void our_kernel(int n, float *s, const void *vx, const void *vy) {
  tcrv_emitc_ggml_vec_dot_iq3_xxs_q8_K_kernel_ggml_vec_dot_iq3_xxs_q8_K(
      (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy);
}

// ---- ggml's VERBATIM _generic (quants.c:999-1041) ----------------------------
// Parameterised by the grid table, the ksigns table, AND a scale-bias flag so the
// negative controls can swap them. The structure is ggml's EXACT inner loops.
static void ggml_generic_param(int n, float *s, const void *vx, const void *vy,
                               const uint32_t *grid_tab, const uint8_t *ksign_tab,
                               int wrong_scale) {
  const block_iq3_xxs *x = (const block_iq3_xxs *)vx;
  const block_q8_K *y = (const block_q8_K *)vy;
  const int nb = n / QK_K;
  uint32_t aux32;
  float sumf = 0.f;
  for (int i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d) * y[i].d;
    const uint8_t *q3 = x[i].qs;
    const uint8_t *gas = x[i].qs + QK_K / 4;
    const int8_t *q8 = y[i].qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
      std::memcpy(&aux32, gas, sizeof(uint32_t));
      gas += sizeof(uint32_t);
      // wrong_scale negative control: drop the "+1".
      const uint32_t ls = wrong_scale ? (2 * (aux32 >> 28))
                                      : (2 * (aux32 >> 28) + 1);
      int32_t sumi = 0;
      for (int l = 0; l < 4; ++l) {
        const uint8_t *grid1 = (const uint8_t *)(grid_tab + q3[2 * l + 0]);
        const uint8_t *grid2 = (const uint8_t *)(grid_tab + q3[2 * l + 1]);
        const uint8_t signs = ksign_tab[(aux32 >> 7 * l) & 127];
        for (int j = 0; j < 4; ++j) {
          sumi += grid1[j] * q8[j + 0] * (signs & kmask_iq2xs[j + 0] ? -1 : 1);
          sumi += grid2[j] * q8[j + 4] * (signs & kmask_iq2xs[j + 4] ? -1 : 1);
        }
        q8 += 8;
      }
      q3 += 8;
      bsum += sumi * ls;
    }
    sumf += d * bsum;
  }
  *s = 0.25f * sumf;
}

static void ggml_ref(int n, float *s, const void *vx, const void *vy) {
  ggml_generic_param(n, s, vx, vy, iq3xxs_grid, ksigns_iq2xs, 0);
}

// ---- RNG -----------------------------------------------------------------------
static uint64_t rng_state = 0x1234567890abcdefULL;
static uint32_t xrng() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (uint32_t)(rng_state >> 32);
}

// Fill a random iq3_xxs weight block + q8_K activation block.
static void fill_random_blocks(uint8_t *xbuf, uint8_t *ybuf, int nb, int q8_mode) {
  for (int i = 0; i < nb; ++i) {
    block_iq3_xxs *x = (block_iq3_xxs *)(xbuf + i * IQ3XXS_STRIDE);
    block_q8_K *y = (block_q8_K *)(ybuf + i * Q8K_STRIDE);
    // fp16 weight scale: a random small positive half.
    float dx = 0.005f + (xrng() & 0xff) * 0.0007f;
    _Float16 hx = (_Float16)dx;
    std::memcpy(&x->d, &hx, 2);
    // 64 grid index bytes (full [0,255] range) + 32 aux bytes (random).
    for (int k = 0; k < 3 * QK_K / 8; ++k) x->qs[k] = (uint8_t)(xrng() & 0xff);
    y->d = 0.01f + (xrng() & 0xff) * 0.0011f;
    for (int k = 0; k < QK_K; ++k) {
      int v;
      if (q8_mode == 1) v = 127;
      else if (q8_mode == 2) v = (k & 1) ? 127 : -127;
      else v = (int)(xrng() % 255) - 127;  // [-127,127]
      y->qs[k] = (int8_t)v;
    }
    for (int k = 0; k < QK_K / 16; ++k) y->bsums[k] = 0;
  }
}

static int g_fail = 0, g_total = 0, g_gen_delta = 0;

static int check_case(const char *name, int n, const uint8_t *xbuf,
                      const uint8_t *ybuf) {
  float s_ours = 0.f, s_ref = 0.f;
  our_kernel(n, &s_ours, xbuf, ybuf);
  ggml_ref(n, &s_ref, xbuf, ybuf);
  g_total++;
  if (std::memcmp(&s_ours, &s_ref, sizeof(float)) != 0) {
    g_fail++;
    g_gen_delta++;
    if (g_fail <= 8)
      std::printf("  FAIL %-28s n=%-6d ours=%.9g ref=%.9g (bits %08x vs %08x)\n",
                  name, n, s_ours, s_ref,
                  *(uint32_t *)&s_ours, *(uint32_t *)&s_ref);
    return 1;
  }
  return 0;
}

int main() {
  // Build the wrong-grid + wrong-signs tables (negative controls).
  for (int i = 0; i < 256; ++i) {
    uint32_t g = iq3xxs_grid[i];
    uint32_t w = 0;
    for (int b = 0; b < 4; ++b) {
      uint8_t by = (uint8_t)((g >> (8 * b)) & 0xff);
      w |= (uint32_t)((uint8_t)(by + 1)) << (8 * b);
    }
    iq3xxs_grid_wrong[i] = w;
  }
  for (int i = 0; i < 128; ++i) ksigns_wrong[i] = ksigns_iq2xs[i] & 0x7f;

  const int MAXB = 64;
  uint8_t *xbuf = (uint8_t *)std::malloc(MAXB * IQ3XXS_STRIDE);
  uint8_t *ybuf = (uint8_t *)std::malloc(MAXB * Q8K_STRIDE);

  // ---- random multiples of 256 -------------------------------------------------
  int ns[] = {256, 512, 768, 1024, 2048, 4096, 8192, 16384};
  for (int rep = 0; rep < 300; ++rep) {
    for (int ni = 0; ni < 8; ++ni) {
      int n = ns[ni];
      int nb = n / QK_K;
      if (nb > MAXB) continue;
      fill_random_blocks(xbuf, ybuf, nb, 0);
      check_case("random", n, xbuf, ybuf);
    }
  }

  // ---- edge: q8 = +127 / q8 = +/-127 ------------------------------------------
  for (int rep = 0; rep < 50; ++rep) {
    fill_random_blocks(xbuf, ybuf, 8, 1);
    check_case("q8=+127", 8 * QK_K, xbuf, ybuf);
    fill_random_blocks(xbuf, ybuf, 8, 2);
    check_case("q8=+/-127", 8 * QK_K, xbuf, ybuf);
  }

  // ---- edge: marching grid indices (exercise the full [0,255] grid range) -----
  // Set the 64 grid index bytes to march 0..255 and the sign selectors to march
  // 0..127, the scale to its extremes (top nibble of the per-sub-block aux uint32).
  for (int extreme = 0; extreme < 2; ++extreme) {
    int nb = 8;
    fill_random_blocks(xbuf, ybuf, nb, 0);
    for (int i = 0; i < nb; ++i) {
      block_iq3_xxs *x = (block_iq3_xxs *)(xbuf + i * IQ3XXS_STRIDE);
      uint8_t *q3 = x->qs;            // 64 grid index bytes
      uint8_t *gas = x->qs + QK_K / 4;  // 32 aux bytes
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        for (int l = 0; l < 8; ++l) {
          q3[ib32 * 8 + l] = (uint8_t)((i * 64 + ib32 * 8 + l) & 0xff);  // march
        }
        // sign selectors marching, scale extreme (0 or 15 in the top nibble).
        uint32_t signsel = (uint32_t)((i * 8 + ib32) & 0x7f);
        uint32_t aux = signsel | (signsel << 7) | (signsel << 14) | (signsel << 21);
        uint32_t scale4 = extreme ? 15u : 0u;
        aux = (aux & 0x0fffffffu) | (scale4 << 28);
        std::memcpy(gas + ib32 * 4, &aux, sizeof(uint32_t));
      }
    }
    check_case(extreme ? "grid-march+scale15" : "grid-march+scale0", nb * QK_K,
               xbuf, ybuf);
  }

  std::printf("iq3_xxs byte-exact: %d/%d passed, %d failed (generic delta %d/%d)\n",
              g_total - g_fail, g_total, g_fail, g_gen_delta, g_total);

  // ---- NEGATIVE CONTROLS: the grid/signs/scale must be load-bearing -----------
  // Each control runs OUR kernel (real tables) vs a WRONG-table reference; the
  // results MUST DIVERGE (a match would mean the mechanism is NOT load-bearing).
  fill_random_blocks(xbuf, ybuf, 8, 0);
  int nbig = 8 * QK_K;
  float s_ours = 0.f;
  our_kernel(nbig, &s_ours, xbuf, ybuf);

  int controls_ok = 0;
  {
    float s_wrong = 0.f;
    ggml_generic_param(nbig, &s_wrong, xbuf, ybuf, iq3xxs_grid_wrong,
                       ksigns_iq2xs, 0);
    int diverges = std::memcmp(&s_ours, &s_wrong, sizeof(float)) != 0;
    std::printf("NEG-CTRL wrong-grid:  ours=%.9g wrong=%.9g -> %s\n", s_ours,
                s_wrong, diverges ? "DIVERGES (PASS)" : "MATCHES (FAIL!)");
    controls_ok += diverges;
  }
  {
    float s_wrong = 0.f;
    ggml_generic_param(nbig, &s_wrong, xbuf, ybuf, iq3xxs_grid, ksigns_wrong, 0);
    int diverges = std::memcmp(&s_ours, &s_wrong, sizeof(float)) != 0;
    std::printf("NEG-CTRL wrong-signs: ours=%.9g wrong=%.9g -> %s\n", s_ours,
                s_wrong, diverges ? "DIVERGES (PASS)" : "MATCHES (FAIL!)");
    controls_ok += diverges;
  }
  {
    float s_wrong = 0.f;
    ggml_generic_param(nbig, &s_wrong, xbuf, ybuf, iq3xxs_grid, ksigns_iq2xs, 1);
    int diverges = std::memcmp(&s_ours, &s_wrong, sizeof(float)) != 0;
    std::printf("NEG-CTRL wrong-scale: ours=%.9g wrong=%.9g -> %s\n", s_ours,
                s_wrong, diverges ? "DIVERGES (PASS)" : "MATCHES (FAIL!)");
    controls_ok += diverges;
  }

  std::free(xbuf);
  std::free(ybuf);

  int ok = (g_fail == 0) && (controls_ok == 3);
  std::printf("RESULT: %s\n", ok ? "ALL PASS" : "FAILURES PRESENT");
  return ok ? 0 : 1;
}
