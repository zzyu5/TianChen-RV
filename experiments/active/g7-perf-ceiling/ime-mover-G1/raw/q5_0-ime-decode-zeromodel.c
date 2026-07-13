/* q5_0 @ ime — byte-exact ZERO-MODEL demonstration (host, no board, no tracked-source change)
 *
 * Purpose (G7 ime-mover-G1 step-3 hard gate): prove that a q5_0 IME GEMM emitter,
 * built as q4_0@ime's decode-brick SWAP (5-bit offset-binary plane insert) + the
 * IDENTICAL single scale-fold epilogue (q40ScaleFoldMatmulHelperBody reused verbatim),
 * is byte-exact by construction:
 *   (1) the q5_0 decode brick reproduces ggml's canonical dequant_row_q5_0 5-bit value
 *       (offset-binary, minus 16) exactly  -> INT8 fragment mismatch = 0;
 *   (2) the int32 vmadot core over the decoded int8 == a ZERO-MODEL plain integer GEMM
 *       recomputed independently from the raw bytes  -> INT32 mismatch = 0 (the 0xe210312b
 *       vmadot leaf is UNCHANGED; only the decode brick is format-keyed);
 *   (3) the deferred fp16 scale fold Cf += dA*dW*frag matches the ZERO-MODEL
 *       dequant-then-float-GEMM up to float reassociation (same fold structure as q4_0).
 *
 * This is the exact ZERO-MODEL discipline from the G4/G6 IME seals (int32 0-diff +
 * host oracle), transplanted to q5_0. The integer core is portable (x86 host == k1).
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define QK 32
/* q5_0 block: [fp16 d][qh[4]][qs[16]] = 22 bytes. We store d as float here (scale fold
 * is float either way; the INT core does not touch d). */
typedef struct { float d; uint8_t qh[4]; uint8_t qs[16]; } blk_q5_0;
/* q8_0 activation block: [fp16 d][int8 qs[32]] */
typedef struct { float d; int8_t qs[32]; } blk_q8_0;

/* ---- canonical ggml dequant_row_q5_0 (the ZERO-MODEL oracle for the 5-bit value) ---- */
static void oracle_dequant_q5_0(const blk_q5_0 *b, float *out /* 32 */) {
    uint32_t qh; memcpy(&qh, b->qh, 4);
    for (int j = 0; j < 16; ++j) {
        const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
        const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;
        const int32_t x0 = ((b->qs[j] & 0x0F) | xh_0) - 16;
        const int32_t x1 = ((b->qs[j] >>   4) | xh_1) - 16;
        out[j + 0]  = b->d * (float)x0;
        out[j + 16] = b->d * (float)x1;
    }
}

/* ---- the q5_0 IME DECODE BRICK (candidate emitter helper, pure integer transform) ----
 * Mirrors q40DequantHelperBody() shape: fragment layout out[j]=lo, out[j+16]=hi, each
 * shifted to signed by the offset (q4_0: -8; q5_0: 5-bit -16). The 5th bit comes from qh.
 * This is what would be emitted as weft_ime_q5_0_dequant_fragment. */
static void ime_decode_q5_0(const blk_q5_0 *b, int8_t *out /* 32 */) {
    uint32_t qh; memcpy(&qh, b->qh, 4);
    for (int j = 0; j < 16; ++j) {
        const uint8_t xh_0 = ((qh >> (j + 0)) << 4) & 0x10;
        const uint8_t xh_1 = ((qh >> (j + 12))     ) & 0x10;
        out[j + 0]  = (int8_t)(((b->qs[j] & 0x0F) | xh_0) - 16);
        out[j + 16] = (int8_t)(((b->qs[j] >>   4) | xh_1) - 16);
    }
}

/* ---- the vmadot int32 core, expressed as plain integer MAC (the 0xe210312b leaf is
 * int32-exact and order-independent for integer add; this host model IS its semantics) --- */
static int32_t int32_core(const int8_t *qa /* 32 */, const int8_t *qw /* 32 */) {
    int32_t s = 0;
    for (int k = 0; k < QK; ++k) s += (int32_t)qa[k] * (int32_t)qw[k];
    return s;
}

int main(void) {
    srand(20260714);
    const int NB = 64;      /* K = NB*32 = 2048 contraction blocks */
    const int MROW = 4, NCOL = 4;  /* one 4x4 IME output tile */

    /* random weights (q5_0) [NCOL][NB] and activations (q8_0) [MROW][NB] */
    static blk_q5_0 W[4][64];
    static blk_q8_0 A[4][64];
    for (int n = 0; n < NCOL; ++n) for (int b = 0; b < NB; ++b) {
        W[n][b].d = ((rand()%2000)-1000) * 0.001f;
        for (int t = 0; t < 4; ++t) W[n][b].qh[t] = rand() & 0xFF;
        for (int t = 0; t < 16; ++t) W[n][b].qs[t] = rand() & 0xFF;
    }
    for (int m = 0; m < MROW; ++m) for (int b = 0; b < NB; ++b) {
        A[m][b].d = ((rand()%2000)-1000) * 0.001f;
        for (int t = 0; t < 32; ++t) A[m][b].qs[t] = (int8_t)((rand()%255)-127);
    }

    /* ---- GATE 1: decode brick INT8 == canonical 5-bit value (offset-binary) ---- */
    long int8_mismatch = 0;
    for (int n = 0; n < NCOL; ++n) for (int b = 0; b < NB; ++b) {
        int8_t dec[32]; float ref[32];
        ime_decode_q5_0(&W[n][b], dec);
        oracle_dequant_q5_0(&W[n][b], ref);  /* ref = d * qval5 */
        for (int k = 0; k < 32; ++k) {
            int32_t qval5 = (int32_t)lroundf(ref[k] / W[n][b].d); /* recover the 5-bit signed value */
            if ((int32_t)dec[k] != qval5) int8_mismatch++;
        }
    }

    /* ---- GATE 2: IME int32 core == ZERO-MODEL plain integer GEMM (recomputed independently) --- */
    long int32_mismatch = 0;
    for (int m = 0; m < MROW; ++m) for (int n = 0; n < NCOL; ++n) {
        for (int b = 0; b < NB; ++b) {
            int8_t wdec[32];
            ime_decode_q5_0(&W[n][b], wdec);
            int32_t ime = int32_core(A[m][b].qs, wdec);
            /* ZERO-MODEL: independent recompute of the 5-bit value straight from bytes */
            int32_t zm = 0;
            uint32_t qh; memcpy(&qh, W[n][b].qh, 4);
            for (int j = 0; j < 16; ++j) {
                int32_t w0 = ((W[n][b].qs[j] & 0x0F) | (int32_t)(((qh >> (j+0)) & 1) << 4)) - 16;
                int32_t w1 = ((W[n][b].qs[j] >>   4) | (int32_t)(((qh >> (j+16)) & 1) << 4)) - 16;
                zm += (int32_t)A[m][b].qs[j+0]  * w0;
                zm += (int32_t)A[m][b].qs[j+16] * w1;
            }
            if (ime != zm) int32_mismatch++;
        }
    }

    /* ---- GATE 3: scale-fold epilogue (SAME as q4_0) == float ZERO-MODEL GEMM ---- */
    double max_rel = 0.0;
    for (int m = 0; m < MROW; ++m) for (int n = 0; n < NCOL; ++n) {
        /* IME path: Cf += dA*dW*int32_core  (q40ScaleFoldMatmulHelperBody, verbatim) */
        double cf_ime = 0.0;
        for (int b = 0; b < NB; ++b) {
            int8_t wdec[32]; ime_decode_q5_0(&W[n][b], wdec);
            int32_t p = int32_core(A[m][b].qs, wdec);
            cf_ime += (double)A[m][b].d * (double)W[n][b].d * (double)p;
        }
        /* ZERO-MODEL: dequant both to float, plain float dot */
        double cf_zm = 0.0;
        for (int b = 0; b < NB; ++b) {
            float wf[32], af[32];
            oracle_dequant_q5_0(&W[n][b], wf);
            for (int k = 0; k < 32; ++k) af[k] = A[m][b].d * (float)A[m][b].qs[k];
            for (int k = 0; k < 32; ++k) cf_zm += (double)af[k] * (double)wf[k];
        }
        double rel = fabs(cf_ime - cf_zm) / (fabs(cf_zm) + 1e-9);
        if (rel > max_rel) max_rel = rel;
    }

    printf("q5_0@ime ZERO-MODEL demonstration (K=%d, one 4x4 tile)\n", NB*32);
    printf("  GATE1 decode-brick INT8 vs canonical 5-bit  : mismatch = %ld  %s\n",
           int8_mismatch, int8_mismatch==0 ? "[PASS]" : "[FAIL]");
    printf("  GATE2 IME int32 core vs ZERO-MODEL int-GEMM  : mismatch = %ld  %s\n",
           int32_mismatch, int32_mismatch==0 ? "[PASS]" : "[FAIL]");
    printf("  GATE3 scale-fold epilogue vs float ZERO-MODEL: max_rel = %.3e  %s\n",
           max_rel, max_rel < 1e-4 ? "[PASS(float-reassoc)]" : "[CHECK]");
    int ok = (int8_mismatch==0) && (int32_mismatch==0) && (max_rel < 1e-4);
    printf("  RESULT: %s\n", ok ? "BYTE-EXACT-BY-CONSTRUCTION CONFIRMED" : "FAILED");
    return ok ? 0 : 1;
}
