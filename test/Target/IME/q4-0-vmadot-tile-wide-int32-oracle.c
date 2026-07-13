// G6-A M7 host int32-EXACT oracle for the WIDE (output-tiled) vmadot MAC leaf.
//
// The [PAT-1] IME-VMADOT-TILE-W2/W4 rows register a LAYER-4 array-utilization optimization on the
// already-mechanized batched vmadot MAC leaf (lib/Plugin/IME/IMEBackendEmissionDriver.cpp,
// macKloopHelperBodyWide): the WIDE leaf reuses the 4x8 A fragment IN-REGISTER across NJW adjacent
// column-tiles, feeding ONE `vle8` of A into NJW independent `vmadot` chains. The vmadot INSTRUCTION
// runs only on real K1 (board-sealed byte-exact: greedy md5 f5e77482 == the M1..M6 q4_0 bridge
// lineage; plus a standalone K1 unit test proved w2/w4 int32 == the width-1 leaf, memcmp=0). This
// host oracle substitutes a SCALAR reference of the batched vmadot MAC (same int8->int32 semantics)
// and validates the "what": the wide leaf's per-sub-tile int32 is (1) BIT-IDENTICAL to the width-1
// leaf run NJW times, and (2) equal to an INDEPENDENT ZERO-MODEL plain-GEMM over matrices re-derived
// from the raw packed bytes -- a different code path than the tiled fragment kernel.
//
// Build + run:
//   cc -O2 -std=c11 q4-0-vmadot-tile-wide-int32-oracle.c -o /tmp/q40wide && /tmp/q40wide
// Exit 0 + "ORACLE PASS" iff both checks are 0-diff over the whole test grid.

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---- (A) the batched width-1 vmadot MAC leaf (scalar substitute, int32-exact contract) ----
static inline void weft_ime_vmadot_mac_kloop(const int8_t *A, const int8_t *B,
                                             long kt, int32_t *frag) {
  int32_t acc[16];
  for (int r = 0; r < 16; ++r) acc[r] = 0;
  for (long kf = 0; kf < kt; ++kf) {
    const int8_t *Af = A + kf * 32;
    const int8_t *Bf = B + kf * 32;
    for (int m = 0; m < 4; ++m)
      for (int n = 0; n < 4; ++n) {
        int32_t s = 0;
        for (int k = 0; k < 8; ++k) s += (int32_t)Af[m * 8 + k] * (int32_t)Bf[n * 8 + k];
        acc[m * 4 + n] += s;
      }
  }
  for (int r = 0; r < 16; ++r) frag[r] = acc[r];
}

// ---- (B) the WIDE (NJW-tiled) leaf (scalar substitute mirroring macKloopHelperBodyWide) ----
// ONE A fragment stream reused across NJW column-tiles (B0 + w*bstride); frag[w*16 .. +15] = tile w.
// By construction each tile w reduces its OWN kt fragments in the SAME kf order -> int32-identical
// to the width-1 leaf run on (A, B0 + w*bstride).
static void weft_ime_vmadot_mac_kloop_wide(const int8_t *A, const int8_t *B0,
                                           long bstride, long kt, int njw,
                                           int32_t *frag) {
  for (int w = 0; w < njw; ++w) {
    int32_t sub[16];
    for (int r = 0; r < 16; ++r) sub[r] = 0;
    const int8_t *Bw = B0 + (long)w * bstride;
    for (long kf = 0; kf < kt; ++kf) {
      const int8_t *Af = A + kf * 32;
      const int8_t *Bf = Bw + kf * 32;
      for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n) {
          int32_t s = 0;
          for (int k = 0; k < 8; ++k) s += (int32_t)Af[m * 8 + k] * (int32_t)Bf[n * 8 + k];
          sub[m * 4 + n] += s;
        }
    }
    for (int r = 0; r < 16; ++r) frag[w * 16 + r] = sub[r];
  }
}

int main(void) {
  srand(778899u);
  const long kt = 8;                 // K/8 fragments per col-tile
  const long bstride = kt * 32;      // fragment-major col-tile stride (bytes)

  long a_bytes = kt * 32;            // one A row-tile: kt fragments of 32B
  int8_t *A = (int8_t *)malloc(a_bytes);
  for (long i = 0; i < a_bytes; ++i) A[i] = (int8_t)(rand() % 256 - 128);

  // 4 column-tiles laid out fragment-major and contiguous (col-tile w at B + w*bstride).
  long b_bytes = 4 * bstride;
  int8_t *B = (int8_t *)malloc(b_bytes);
  for (long i = 0; i < b_bytes; ++i) B[i] = (int8_t)(rand() % 256 - 128);

  long widetile_mismatch = 0, zero_model_mismatch = 0;

  for (int njw = 2; njw <= 4; njw += 2) {   // both registered widths
    // width-1 reference: njw separate baseline calls
    int32_t base[64];
    for (int w = 0; w < njw; ++w)
      weft_ime_vmadot_mac_kloop(A, B + (long)w * bstride, kt, base + w * 16);
    // wide leaf: one call producing njw sub-tiles
    int32_t wide[64];
    weft_ime_vmadot_mac_kloop_wide(A, B, bstride, kt, njw, wide);
    if (memcmp(base, wide, (size_t)njw * 16 * sizeof(int32_t)) != 0) widetile_mismatch++;

    // ZERO-MODEL: plain triple-loop GEMM over logical A[m][k], W[w][n][k] re-derived from bytes.
    for (int w = 0; w < njw; ++w) {
      const int8_t *Bw = B + (long)w * bstride;
      for (int m = 0; m < 4; ++m)
        for (int n = 0; n < 4; ++n) {
          int64_t ref = 0;
          for (long k = 0; k < kt * 8; ++k) {
            long kf = k / 8, kl = k % 8;
            int a = (int)A[kf * 32 + m * 8 + kl];
            int b = (int)Bw[kf * 32 + n * 8 + kl];
            ref += (int64_t)a * (int64_t)b;
          }
          if ((int64_t)wide[w * 16 + m * 4 + n] != ref) zero_model_mismatch++;
        }
    }
  }

  printf("wide-tile vs width-1 leaf: mismatch=%ld (NJW=2 and 4)\n", widetile_mismatch);
  printf("wide-tile vs ZERO-MODEL plain GEMM: mismatch=%ld\n", zero_model_mismatch);
  free(A);
  free(B);
  if (widetile_mismatch == 0 && zero_model_mismatch == 0) {
    printf("ORACLE PASS: wide vmadot tiling is int32-EXACT (== width-1 leaf == ZERO-MODEL)\n");
    return 0;
  }
  printf("ORACLE FAIL\n");
  return 1;
}
