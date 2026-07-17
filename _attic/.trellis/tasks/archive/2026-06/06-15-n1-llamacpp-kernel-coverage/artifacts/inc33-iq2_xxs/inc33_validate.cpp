// INC-33 iq2_xxs x q8_K byte-exact HW validation harness (ssh rvv, -march=rv64gcv).
//
// Proves: the C our compiler emits for the new tcrv_rvv.iq2_xxs_q8_k_block_dot op
// (the COMPLETE ggml ggml_vec_dot_iq2_xxs_q8_K GRID-codebook super-block kernel: AoS
// QK_K=256 super-block loop + per-super-block d scale + per-sub-block 4-bit integer
// scale + GRID-codebook decode + SIGN-plane application + integer bsum fold + 0.125f
// + *s store) computes the SAME fp32 result *s as ggml's OWN _generic iq2_xxs
// reference, BIT-FOR-BIT (memcmp of the float bits), over random block_iq2_xxs x
// block_q8_K arrays at n multiples of 256 + named edge cases (grid index range, sign
// patterns, scale extremes, q8 +/-127). THREE negative controls (a WRONG grid + WRONG
// signs + WRONG scale) must FAIL -> proves the grid, sign plane, AND scale extraction
// are the live, load-bearing mechanisms.
//
// iq2_xxs is the FIRST member of the deep IQ tail -- the GRID-codebook class. Each
// weight byte indexes a 256-entry packed uint64 GRID codebook (iq2xxs_grid), the
// per-element sign comes from a separate SIGN PLANE (ksigns_iq2xs/kmask), and the
// 4-bit scale folds in the INTEGER domain (q6_K-style bsum), with a trailing 0.125f.
//
// REFERENCE: ggml's VERBATIM _generic (quants.c:855-895): memcpy aux32[2] from q2,
// ls = 2*(aux32[1]>>28)+1, grid = (uint8_t*)(iq2xxs_grid + aux8[l]), signs =
// ksigns_iq2xs[(aux32[1]>>7*l)&127], sumi += grid[j]*q8[j]*(signs&kmask?-1:1),
// bsum += sumi*ls, sumf += d*bsum, *s = 0.125f*sumf. FP16->FP32 mirrors the board's
// scalar _Float16 widening (lossless). Whole TU built with the same -ffp-contract.
//
// The kernel under test is the UNMODIFIED, compiler-emitted iq2_xxs kernel
// (iq2_xxs_emitted.cpp) -- every line tagged source_op=tcrv_rvv.iq2_xxs_q8_k_block_dot.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstddef>
#include <cstring>

#define QK_K 256
static const int IQ2XXS_STRIDE = 66;  // sizeof block_iq2_xxs
static const int Q8K_STRIDE = 292;    // sizeof block_q8_K

// ---- block formats (ggml-common.h, QK_K = 256) -------------------------------
//   block_iq2_xxs (66 bytes): d(fp16) @0 | qs[32](uint16) @2
//   block_q8_K   (292 bytes): d(fp32) @0 | qs[256] @4 | bsums[16] @260 (unused)
typedef struct { uint16_t d; uint16_t qs[QK_K/8]; } block_iq2_xxs;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;

// ---- ggml's GRID codebook + SIGN plane + kmask (ggml-common.h) ----------------
static const uint64_t iq2xxs_grid[256] = {0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL, 0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL, 0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL, 0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL, 0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL, 0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL, 0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL, 0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL, 0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL, 0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL, 0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL, 0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL, 0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL, 0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL, 0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL, 0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL, 0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL, 0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL, 0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL, 0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL, 0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL, 0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL, 0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL, 0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL, 0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL, 0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL, 0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL, 0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL, 0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL, 0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL, 0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL, 0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL, 0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL, 0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL, 0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL, 0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL, 0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL, 0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL, 0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL, 0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL, 0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL, 0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL, 0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL, 0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL, 0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL, 0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL, 0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL, 0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL, 0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL, 0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL, 0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL, 0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL, 0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL, 0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL, 0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL, 0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL, 0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL, 0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL, 0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL, 0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL, 0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL, 0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL, 0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL, 0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL};
static const uint8_t ksigns_iq2xs[128] = {0, 129, 130, 3, 132, 5, 6, 135, 136, 9, 10, 139, 12, 141, 142, 15, 144, 17, 18, 147, 20, 149, 150, 23, 24, 153, 154, 27, 156, 29, 30, 159, 160, 33, 34, 163, 36, 165, 166, 39, 40, 169, 170, 43, 172, 45, 46, 175, 48, 177, 178, 51, 180, 53, 54, 183, 184, 57, 58, 187, 60, 189, 190, 63, 192, 65, 66, 195, 68, 197, 198, 71, 72, 201, 202, 75, 204, 77, 78, 207, 80, 209, 210, 83, 212, 85, 86, 215, 216, 89, 90, 219, 92, 221, 222, 95, 96, 225, 226, 99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111, 240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255};
static const uint8_t kmask_iq2xs[8] = {1, 2, 4, 8, 16, 32, 64, 128};

// WRONG grid for negative control 1: each entry's bytes incremented by 1. If the
// grid were NOT load-bearing, swapping it would not change the result.
static uint64_t iq2xxs_grid_wrong[256];
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
tcrv_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_ggml_vec_dot_iq2_xxs_q8_K(
    size_t n, float *s, const uint8_t *vx, const uint8_t *vy);

static void our_kernel(int n, float *s, const void *vx, const void *vy) {
  tcrv_emitc_ggml_vec_dot_iq2_xxs_q8_K_kernel_ggml_vec_dot_iq2_xxs_q8_K(
      (size_t)n, s, (const uint8_t *)vx, (const uint8_t *)vy);
}

// ---- ggml's VERBATIM _generic (quants.c:855-895) -----------------------------
// Parameterised by the grid table, the ksigns table, AND a scale-bias flag so the
// negative controls can swap them. The structure is ggml's EXACT inner loops.
static void ggml_generic_param(int n, float *s, const void *vx, const void *vy,
                               const uint64_t *grid_tab, const uint8_t *ksign_tab,
                               int wrong_scale) {
  const block_iq2_xxs *x = (const block_iq2_xxs *)vx;
  const block_q8_K *y = (const block_q8_K *)vy;
  const int nb = n / QK_K;
  uint32_t aux32[2];
  const uint8_t *aux8 = (const uint8_t *)aux32;
  float sumf = 0.f;
  for (int i = 0; i < nb; ++i) {
    const float d = fp16_to_fp32(x[i].d) * y[i].d;
    const uint16_t *q2 = x[i].qs;
    const int8_t *q8 = y[i].qs;
    int32_t bsum = 0;
    for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
      std::memcpy(aux32, q2, 2 * sizeof(uint32_t));
      q2 += 4;
      // wrong_scale negative control: drop the "+1" (so ls = 2*(aux>>28), which is
      // wrong vs ggml's 2*(aux>>28)+1).
      const uint32_t ls = wrong_scale ? (2 * (aux32[1] >> 28))
                                      : (2 * (aux32[1] >> 28) + 1);
      int32_t sumi = 0;
      for (int l = 0; l < 4; ++l) {
        const uint8_t *grid = (const uint8_t *)(grid_tab + aux8[l]);
        const uint8_t signs = ksign_tab[(aux32[1] >> 7 * l) & 127];
        for (int j = 0; j < 8; ++j) {
          sumi += grid[j] * q8[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        }
        q8 += 8;
      }
      bsum += sumi * ls;
    }
    sumf += d * bsum;
  }
  *s = 0.125f * sumf;
}

static void ggml_ref(int n, float *s, const void *vx, const void *vy) {
  ggml_generic_param(n, s, vx, vy, iq2xxs_grid, ksigns_iq2xs, 0);
}

// ---- RNG -----------------------------------------------------------------------
static uint64_t rng_state = 0x1234567890abcdefULL;
static uint32_t xrng() {
  rng_state ^= rng_state << 13;
  rng_state ^= rng_state >> 7;
  rng_state ^= rng_state << 17;
  return (uint32_t)(rng_state >> 32);
}

// Fill a random iq2_xxs weight block + q8_K activation block.
static void fill_random_blocks(uint8_t *xbuf, uint8_t *ybuf, int nb, int q8_mode) {
  for (int i = 0; i < nb; ++i) {
    block_iq2_xxs *x = (block_iq2_xxs *)(xbuf + i * IQ2XXS_STRIDE);
    block_q8_K *y = (block_q8_K *)(ybuf + i * Q8K_STRIDE);
    // fp16 weight scale: a random small positive half.
    float dx = 0.005f + (xrng() & 0xff) * 0.0007f;
    _Float16 hx = (_Float16)dx;
    std::memcpy(&x->d, &hx, 2);
    for (int k = 0; k < QK_K / 8; ++k) x->qs[k] = (uint16_t)(xrng() & 0xffff);
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
    uint64_t g = iq2xxs_grid[i];
    uint64_t w = 0;
    for (int b = 0; b < 8; ++b) {
      uint8_t by = (uint8_t)((g >> (8 * b)) & 0xff);
      w |= (uint64_t)((uint8_t)(by + 1)) << (8 * b);
    }
    iq2xxs_grid_wrong[i] = w;
  }
  for (int i = 0; i < 128; ++i) ksigns_wrong[i] = ksigns_iq2xs[i] & 0x7f;

  const int MAXB = 64;
  uint8_t *xbuf = (uint8_t *)std::malloc(MAXB * IQ2XXS_STRIDE);
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
  // Set every aux8 byte (the 4 grid indices per sub-block) to march 0..255 and the
  // sign selectors to march 0..127, the scale to its extremes.
  for (int extreme = 0; extreme < 2; ++extreme) {
    int nb = 8;
    fill_random_blocks(xbuf, ybuf, nb, 0);
    for (int i = 0; i < nb; ++i) {
      block_iq2_xxs *x = (block_iq2_xxs *)(xbuf + i * IQ2XXS_STRIDE);
      for (int ib32 = 0; ib32 < 8; ++ib32) {
        uint32_t aux0 = 0, aux1 = 0;
        for (int l = 0; l < 4; ++l) {
          uint8_t idx = (uint8_t)((i * 64 + ib32 * 4 + l) & 0xff);  // march grid idx
          aux0 |= (uint32_t)idx << (8 * l);
        }
        // sign selectors marching, scale extreme (0 or 15 in the top nibble).
        uint32_t signsel = (uint32_t)((i * 8 + ib32) & 0x7f);
        aux1 = signsel | (signsel << 7) | (signsel << 14) | (signsel << 21);
        uint32_t scale4 = extreme ? 15u : 0u;
        aux1 = (aux1 & 0x0fffffffu) | (scale4 << 28);
        // write aux0/aux1 into qs (little-endian uint16 view).
        uint16_t *q2 = x->qs + ib32 * 4;
        q2[0] = (uint16_t)(aux0 & 0xffff);
        q2[1] = (uint16_t)(aux0 >> 16);
        q2[2] = (uint16_t)(aux1 & 0xffff);
        q2[3] = (uint16_t)(aux1 >> 16);
      }
    }
    check_case(extreme ? "grid-march+scale15" : "grid-march+scale0", nb * QK_K,
               xbuf, ybuf);
  }

  std::printf("iq2_xxs byte-exact: %d/%d passed, %d failed (generic delta %d/%d)\n",
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
    ggml_generic_param(nbig, &s_wrong, xbuf, ybuf, iq2xxs_grid_wrong,
                       ksigns_iq2xs, 0);
    int diverges = std::memcmp(&s_ours, &s_wrong, sizeof(float)) != 0;
    std::printf("NEG-CTRL wrong-grid:  ours=%.9g wrong=%.9g -> %s\n", s_ours,
                s_wrong, diverges ? "DIVERGES (PASS)" : "MATCHES (FAIL!)");
    controls_ok += diverges;
  }
  {
    float s_wrong = 0.f;
    ggml_generic_param(nbig, &s_wrong, xbuf, ybuf, iq2xxs_grid, ksigns_wrong, 0);
    int diverges = std::memcmp(&s_ours, &s_wrong, sizeof(float)) != 0;
    std::printf("NEG-CTRL wrong-signs: ours=%.9g wrong=%.9g -> %s\n", s_ours,
                s_wrong, diverges ? "DIVERGES (PASS)" : "MATCHES (FAIL!)");
    controls_ok += diverges;
  }
  {
    float s_wrong = 0.f;
    ggml_generic_param(nbig, &s_wrong, xbuf, ybuf, iq2xxs_grid, ksigns_iq2xs, 1);
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
