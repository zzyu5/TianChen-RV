// Byte-exact white-box numeric harness for the COMPILER-EMITTED q5_0 16x1
// repack GEMV.
//
// Drives the compiler-emitted q5_0 GEMV (ggml_gemv_q5_0_16x1_q8_0, via the thin
// ABI adapter over tcrv-opt's emitted symbol) against a CANONICAL ggml q5_0
// reference, over many n, random inputs. Verdict = matches to fp32 rounding
// (norm = max_abs_err / rms(ref)) < 1e-4.
//
// CRITICAL (the advisor's gate): the reference reconstructs each weight from the
// PLAIN pre-repacked block_q5_0 (canonical ggml dequant ((qs&0x0F)|xh)-16 with
// xh the qh bit), NOT from my block_q5_0x16. So a transpose/invert bug in
// make_block_q5_0x16 CANNOT be mirrored on both sides -- the emitted kernel
// reads my repacked vx, the reference reads the plain weights. Agreement proves
// BOTH the packer and the emitter against canonical ggml.
//
// q5_0 (Family A, symmetric offset-binary -16): each weight =
//   ((qs_4bit) | (qh_1bit << 4)) - 16, low element j takes qh bit j, high
//   element j+16 takes qh bit j+16 (ggml dequantize_row_q5_0 /
//   ggml_vec_dot_q5_0_q8_0). Activation = plain block_q8_0 (stride 34).
//
// block_q5_0 (22 bytes): { fp16 d; uint8 qh[4]; uint8 qs[16]; }
// block_q5_0x16 (352 bytes): { fp16 d[16]; uint8 qs[256]; uint16 qhmask[32]; }
//   d[16]   @0    : 16 inline fp16 scales (copy)
//   qs[256] @32   : 16-way interleaved RAW nibbles (byte i = block(i%16).qs[i/16])
//   qhmask[32] @288: TRANSPOSED bit-packed qh. qhmask[e] bit b = block b's qh
//                    bit for element e (NON-inverted). 32 masks = 64 bytes.
//   stride 352 = 16*22 (byte-exactly the plain q5_0 for the same 512 weights).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define QK5_0 32
#define QK8_0 32

typedef uint16_t ggml_half;

static inline float half_to_float(ggml_half h) {
    uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
    uint32_t f;
    if (exp == 0) {
        if (man == 0) f = sign << 31;
        else { exp = 127 - 15 + 1; while ((man & 0x400) == 0) { man <<= 1; exp--; } man &= 0x3ff; f = (sign<<31)|(exp<<23)|(man<<13); }
    } else if (exp == 0x1f) { f = (sign<<31)|(0xff<<23)|(man<<13); }
    else { f = (sign<<31)|((exp-15+127)<<23)|(man<<13); }
    float r; memcpy(&r, &f, 4); return r;
}
static inline ggml_half float_to_half(float fv) {
    uint32_t x; memcpy(&x, &fv, 4);
    uint32_t sign = (x >> 16) & 0x8000;
    int32_t exp = ((x >> 23) & 0xff) - 127 + 15;
    uint32_t man = x & 0x7fffff;
    if (exp <= 0) return (ggml_half)sign;
    if (exp >= 0x1f) return (ggml_half)(sign | 0x7c00);
    uint32_t h = sign | (exp << 10) | (man >> 13);
    if ((man & 0x1000) && ((man & 0x2fff) || (h & 1))) h++;
    return (ggml_half)h;
}

struct block_q5_0 { ggml_half d; uint8_t qh[4]; uint8_t qs[QK5_0/2]; };  // 22 bytes
struct block_q8_0 { ggml_half d; int8_t qs[QK8_0]; };                    // 34 bytes
struct block_q5_0x16 {
    ggml_half d[16];          // @0
    uint8_t   qs[QK8_0 * 8];  // @32, 256 interleaved RAW nibbles
    uint16_t  qhmask[32];     // @288, transposed bit-packed qh (one mask/element)
};
static_assert(sizeof(block_q5_0x16) == 352, "block_q5_0x16 must be 352 bytes");

extern "C" {
void ggml_gemv_q5_0_16x1_q8_0(int n, float*, size_t, const void*, const void*, int, int);
}

// Repack: 16 plain q5_0 blocks -> one block_q5_0x16. Scales copied; nibbles
// RAW-interleaved (no bake -- the bias lives in the assembled 5-bit field); qh
// TRANSPOSED into 32 per-element 16-bit masks. For element e and block b, bit b
// of qhmask[e] = block b's qh bit e (NON-inverted, matching the block-dot
// reconstruct (nibble | (bit<<4)) - 16).
static block_q5_0x16 make_block(const block_q5_0* in) {
    block_q5_0x16 out;
    for (int i = 0; i < 16; i++) out.d[i] = in[i].d;
    const int end = QK8_0 * 8; // 256
    for (int i = 0; i < end; ++i) {
        int src_id = i % 16, src_offset = i / 16;
        out.qs[i] = in[src_id].qs[src_offset]; // RAW nibble byte
    }
    for (int e = 0; e < 32; ++e) {
        uint16_t m = 0;
        for (int b = 0; b < 16; ++b) {
            uint32_t qh; memcpy(&qh, in[b].qh, sizeof(qh));
            uint32_t bit = (qh >> e) & 1u;       // block b's qh bit for element e
            m |= (uint16_t)(bit << b);
        }
        out.qhmask[e] = m;
    }
    return out;
}

// CANONICAL ggml reference: reconstruct each weight from the PLAIN q5_0 blocks
// (NOT the repacked vx) -- the independent gold. Dots the QUANTIZED q8_0
// activation. sumf[j] = sum_l d_x[j]*d_y * sumi_l  with sumi the ggml q5_0/q8_0
// integer dot over the 32 reconstructed weights.
static void gemv_ref_plain(int n, float* s, const block_q5_0* w, int nb,
                           const block_q8_0* a_ptr, int nc) {
    const int qk = QK5_0;
    const int ncols = 16;
    for (int x = 0; x < nc / ncols; x++) {
        float sumf[16];
        for (int j = 0; j < ncols; j++) sumf[j] = 0.0f;
        for (int l = 0; l < nb; l++) {
            const float d_y = half_to_float(a_ptr[l].d);
            for (int j = 0; j < ncols; j++) {
                const block_q5_0& xb = w[(x * 16 + j) * nb + l];
                uint32_t qh; memcpy(&qh, xb.qh, sizeof(qh));
                int sumi = 0;
                for (int k = 0; k < qk / 2; k++) {
                    const uint8_t xh_0 = ((qh >> (k +  0)) << 4) & 0x10;
                    const uint8_t xh_1 = ((qh >> (k + 12))     ) & 0x10;
                    const int32_t x0 = ((xb.qs[k] & 0x0F) | xh_0) - 16;
                    const int32_t x1 = ((xb.qs[k] >>   4) | xh_1) - 16;
                    sumi += x0 * a_ptr[l].qs[k] + x1 * a_ptr[l].qs[k + qk / 2];
                }
                const float d_x = half_to_float(xb.d);
                sumf[j] += (d_x * d_y) * (float)sumi;
            }
        }
        for (int j = 0; j < ncols; j++) s[x * ncols + j] = sumf[j];
    }
}

static void make_random_q5_0_row(block_q5_0* row, int nblocks, std::mt19937& rng) {
    std::uniform_int_distribution<int> nib(0, 15);
    std::uniform_int_distribution<uint32_t> u32(0, 0xFFFFFFFFu);
    std::uniform_real_distribution<float> sc(0.005f, 0.05f);
    for (int b = 0; b < nblocks; b++) {
        row[b].d = float_to_half(sc(rng));
        uint32_t qh = u32(rng);
        memcpy(row[b].qh, &qh, sizeof(qh));
        for (int k = 0; k < QK5_0/2; k++) row[b].qs[k] = (uint8_t)(nib(rng) | (nib(rng) << 4));
    }
}

int main(int argc, char** argv) {
    int ns[] = {64, 256, 4096, 11008, 14336};
    const int NN = sizeof(ns)/sizeof(ns[0]);
    int trials = (argc > 1) ? atoi(argv[1]) : 100;
    std::mt19937 rng(20260625);
    std::uniform_real_distribution<float> act(-2.0f, 2.0f);

    int ncs[] = {16, 32, 64, 336};
    const int NNC = sizeof(ncs)/sizeof(ncs[0]);
    bool all_ok = true;
    printf("VERIFY EMITTED q5_0_16x1 GEMV  trials=%d  n in {64,256,4096,11008,14336}\n", trials);
    for (int ci = 0; ci < NNC; ci++) {
        const int NC = ncs[ci];
        const int ngrp = NC / 16;
        double max_g = 0, sse_g = 0, maxrel_g = 0;
        long cnt_g = 0;
        for (int ni = 0; ni < NN; ni++) {
            int n = ns[ni];
            int nb = n / QK5_0;
            for (int t = 0; t < trials; t++) {
                // weights: NC rows x nb blocks (plain q5_0), repacked GROUP-major.
                std::vector<block_q5_0> w(NC * nb);
                for (int r = 0; r < NC; r++) make_random_q5_0_row(&w[r * nb], nb, rng);
                std::vector<block_q5_0x16> vx(ngrp * nb);
                for (int g = 0; g < ngrp; g++) {
                    block_q5_0 tmp[16];
                    for (int xb = 0; xb < nb; xb++) {
                        for (int i = 0; i < 16; i++) tmp[i] = w[(g * 16 + i) * nb + xb];
                        vx[g * nb + xb] = make_block(tmp);
                    }
                }
                // ONE activation column (gemv) -> PLAIN q8_0 (block_q8_0, stride 34).
                std::vector<float> af(n);
                for (int i = 0; i < n; i++) af[i] = act(rng);
                std::vector<block_q8_0> vy(nb);
                for (int b = 0; b < nb; b++) {
                    float amax = 0; for (int j = 0; j < QK8_0; j++) amax = std::max(amax, fabsf(af[b*QK8_0+j]));
                    float d = amax / 127.0f, id = d ? 1.0f/d : 0.0f;
                    for (int j = 0; j < QK8_0; j++) { int q = (int)lroundf(af[b*QK8_0+j]*id); vy[b].qs[j] = (int8_t)q; }
                    vy[b].d = float_to_half(d);
                }
                std::vector<float> out_mine(NC), out_ref(NC);
                ggml_gemv_q5_0_16x1_q8_0(n, out_mine.data(), 0, vx.data(), vy.data(), 1, NC);
                gemv_ref_plain(n, out_ref.data(), w.data(), nb, vy.data(), NC);
                for (int j = 0; j < NC; j++) {
                    double v = (double)out_ref[j];
                    double e = fabs((double)out_mine[j] - v);
                    max_g = std::max(max_g, e);
                    sse_g += v*v; cnt_g++;
                    maxrel_g = std::max(maxrel_g, e/(fabs(v)+1e-6));
                }
            }
        }
        double rms_g = sqrt(sse_g/cnt_g);
        bool ok = (max_g/rms_g < 1e-4);
        all_ok = all_ok && ok;
        printf("  NC=%-4d (%2d grp)  max_abs_err=%.3e  rms(ref)=%.3e  norm=%.3e  rel=%.3e  %s\n",
               NC, ngrp, max_g, rms_g, max_g/rms_g, maxrel_g, ok ? "PASS" : "FAIL");
    }
    printf("  VERDICT: %s\n", all_ok ? "PASS (compiler-emitted q5_0 GEMV matches CANONICAL ggml reference within fp32 rounding)" : "FAIL");
    return all_ok ? 0 : 1;
}
