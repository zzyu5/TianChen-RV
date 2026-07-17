// White-box numeric harness for the COMPILER-EMITTED q4_0 16x1 repack GEMM at
// RVV0.7.1 (XuanTie xtheadvector, whole-LMUL i8m1/i16m2/i32m4/f32m4).
//
// Mirrors the GEMV RVV0.7 numeric seal but for the GEMM: drives OUR
// RVV0.7-emitted GEMM symbol (via emitted_adapter_gemm_rvv07.cpp over the
// tcrv-opt symbol) against a VERBATIM-INLINE scalar copy of ggml's own
// ggml_gemm_q4_0_16x1_q8_0_generic + ggml_quantize_mat_q8_0_4x1_generic
// (no ggml linkage — fully self-contained, like the GEMV seal's gemv_ref).
// Verdict = matches to fp32 rounding (norm = max_abs_err / rms(ref)) < 1e-4.
//
// DISCRIMINATOR: this settles q4_0-garbage = GEMM-bug vs llama-wiring.
//   PASS  -> the emitted GEMM is numerically CORRECT on the C920 -> bug is wiring.
//   FAIL  -> the emitted GEMM is the bug -> reports which n/nr/nc/column diverges.
//
// CRITICAL: this sweeps nr (4, 8) AND nc (16, 32, 64) -- multiple ROW-groups and
// COLUMN-groups. The board-confirm harness tested only nr=4/nc=16 (single group
// both ways), which multiplies every cross-group pointer increment
// (vx += x*nb*288, a_ptr += y*nb, s[(y*4+m)*bs + x*16]) by ZERO -> any stride
// bug is invisible -> false PASS. Real llama PREFILL (this exact node) runs
// nr = nrows-(nrows%4) (multi row-group) and nc = per-thread feature chunk
// (multi col-group, 4096/11008/14336 split). So we MUST exercise both axes.
//
// GEMM activation = INTERLEAVED block_q8_0x4 (stride 136, scale +0, quants +8),
// quantized by ggml_quantize_mat_q8_0_4x1_generic -- distinct from the GEMV's
// plain block_q8_0. Reference + quantizer are a MATCHED layout pair (pasted
// verbatim from repack.cpp 52-88 and 2388-2440), which is what makes the
// generic-quantizer-without-ggml-linkage approach self-consistent.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define QK4_0 32
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

struct block_q4_0 { ggml_half d; uint8_t qs[QK4_0/2]; };
struct block_q4_0x16 { ggml_half d[16]; int8_t qs[QK8_0*8]; }; // 288 bytes total
struct block_q8_0x4 { ggml_half d[4]; int8_t qs[QK8_0*4]; };   // 136 bytes total

extern "C" {
// COMPILER-EMITTED RVV0.7 GEMM (via emitted_adapter_gemm_rvv07.cpp).
void ggml_gemm_q4_0_16x1_q8_0(int n, float*, size_t, const void*, const void*, int, int);
}

// Inline of static make_block_q4_0x16(in, 1) from repack.cpp (the ^0x88 bake).
static block_q4_0x16 make_block(const block_q4_0* in) {
    block_q4_0x16 out;
    for (int i = 0; i < 16; i++) out.d[i] = in[i].d;
    const int end = QK4_0 * 8 / 1; // 256
    const uint8_t xor_mask = 0x88;
    for (int i = 0; i < end; ++i) {
        int src_id = i % 16, src_offset = i / 16;
        out.qs[i] = (int8_t)(in[src_id].qs[src_offset] ^ xor_mask);
    }
    return out;
}

// VERBATIM-INLINE scalar quantizer: ggml_quantize_mat_q8_0_4x1_generic
// (repack.cpp:52-88). Produces the INTERLEAVED block_q8_0x4 the GEMM consumes:
// src_id = (j % 4) and src_offset = (j / 4) for blck_size_interleave=1.
// x is 4 rows of length k laid out [row0..k][row1..k][row2..k][row3..k].
static void quantize_mat_q8_0_4x1(const float* x, void* vy, int64_t k) {
    const int nb = (int)(k / QK8_0);
    block_q8_0x4* y = (block_q8_0x4*) vy;
    const int blck_size_interleave = 1;
    float srcv[4][QK8_0];
    float id[4];
    for (int i = 0; i < nb; i++) {
        for (int row_iter = 0; row_iter < 4; row_iter++) {
            float amax = 0.0f;
            for (int j = 0; j < QK8_0; j++) {
                srcv[row_iter][j] = x[row_iter * k + i * QK8_0 + j];
                amax = std::max(amax, fabsf(srcv[row_iter][j]));
            }
            const float d = amax / ((1 << 7) - 1);
            id[row_iter] = d ? 1.0f / d : 0.0f;
            y[i].d[row_iter] = float_to_half(d);
        }
        for (int j = 0; j < QK8_0 * 4; j++) {
            int src_offset = (j / (4 * blck_size_interleave)) * blck_size_interleave;
            int src_id = (j % (4 * blck_size_interleave)) / blck_size_interleave;
            src_offset += (j % blck_size_interleave);
            float x0 = srcv[src_id][src_offset] * id[src_id];
            y[i].qs[j] = (int8_t)roundf(x0);
        }
    }
}

// VERBATIM-INLINE scalar reference: ggml_gemm_q4_0_16x1_q8_0_generic
// (repack.cpp:2388-2440). Activation indexing is block_q8_0x4 interleave:
// qs[k*4 + m] and qs[k*4 + m + qk/2*4] -- DISTINCT from GEMV's qs[k].
// Consumes the SAME repacked vx + interleaved block_q8_0x4 vy the kernel does.
static void gemm_ref(int n, float* s, size_t bs, const void* vx, const void* vy, int nr, int nc) {
    const int qk = QK8_0;
    const int nb = n / qk;
    const int ncols_interleaved = 16;
    const int blocklen = 1;
    float sumf[4][16];
    int sumi;
    for (int y = 0; y < nr / 4; y++) {
        const block_q8_0x4* a_ptr = (const block_q8_0x4*) vy + (y * nb);
        for (int x = 0; x < nc / ncols_interleaved; x++) {
            const block_q4_0x16* b_ptr = (const block_q4_0x16*) vx + (x * nb);
            for (int m = 0; m < 4; m++)
                for (int j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            for (int l = 0; l < nb; l++) {
                for (int k = 0; k < (qk / (2 * blocklen)); k++) {
                    for (int m = 0; m < 4; m++) {
                        for (int j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            for (int i = 0; i < blocklen; ++i) {
                                const int v0 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                const int v1 = (int8_t)(b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * half_to_float(b_ptr[l].d[j]) * half_to_float(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            for (int m = 0; m < 4; m++)
                for (int j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
        }
    }
}

static void make_random_q4_0_row(block_q4_0* row, int nblocks, std::mt19937& rng, int nonneg=0) {
    std::uniform_int_distribution<int> nib(nonneg ? 8 : 0, 15);
    std::uniform_real_distribution<float> sc(0.005f, 0.05f);
    for (int b = 0; b < nblocks; b++) {
        row[b].d = float_to_half(sc(rng));
        for (int k = 0; k < QK4_0/2; k++) row[b].qs[k] = (uint8_t)(nib(rng) | (nib(rng) << 4));
    }
}

int main(int argc, char** argv) {
    int ns[] = {64, 256, 4096, 11008, 14336};
    const int NN = sizeof(ns)/sizeof(ns[0]);
    int trials = (argc > 1) ? atoi(argv[1]) : 200;
    int nonneg = (argc > 2) ? atoi(argv[2]) : 0;
    std::mt19937 rng(20260622);
    std::uniform_real_distribution<float> act(nonneg ? 0.1f : -2.0f, 2.0f);

    // SWEEP both group axes. nr (row-groups, /4) and nc (col-groups, /16) both
    // >1 exercise the cross-group pointer math (a_ptr += y*nb, vx += x*nb*288,
    // s stride (y*4+m)*bs + x*16) that single-group (nr=4,nc=16) multiplies by 0.
    // Real llama prefill runs multi-group on BOTH. (nr,nc) pairs:
    struct { int nr, nc; } cfgs[] = {
        {4, 16},   // single group both (baseline, == board-confirm)
        {4, 32},   // multi col-group  (x=0,1)
        {4, 64},   // multi col-group  (x=0..3)
        {8, 16},   // multi row-group  (y=0,1)
        {8, 64},   // multi BOTH       (y=0,1 ; x=0..3)
    };
    const int NCFG = sizeof(cfgs)/sizeof(cfgs[0]);

    bool all_ok = true;
    printf("VERIFY EMITTED q4_0_16x1 GEMM RVV0.7 (xtheadvector) VLEN=128  trials=%d  n in {64,256,4096,11008,14336}  nonneg=%d\n", trials, nonneg);
    for (int ci = 0; ci < NCFG; ci++) {
        const int NR = cfgs[ci].nr;
        const int NC = cfgs[ci].nc;
        const int nrg = NR / 4;            // row groups
        const int ncg = NC / 16;           // col groups
        double max_g = 0, sse_g = 0, maxrel_g = 0;
        long cnt_g = 0;
        int worst_n = 0, worst_m = -1, worst_j = -1;
        for (int ni = 0; ni < NN; ni++) {
            int n = ns[ni];
            int nb = n / QK4_0;
            for (int t = 0; t < trials; t++) {
                // weights: NC rows x nb blocks (plain q4_0), repacked GROUP-major:
                // col-group g holds rows g*16..g*16+15; vx[g*nb + x] = make_block.
                std::vector<block_q4_0> w(NC * nb);
                for (int r = 0; r < NC; r++) make_random_q4_0_row(&w[r * nb], nb, rng, nonneg);
                std::vector<block_q4_0x16> vx(ncg * nb);
                for (int g = 0; g < ncg; g++) {
                    block_q4_0 tmp[16];
                    for (int x = 0; x < nb; x++) {
                        for (int i = 0; i < 16; i++) tmp[i] = w[(g * 16 + i) * nb + x];
                        vx[g * nb + x] = make_block(tmp);
                    }
                }
                // activations: NR rows -> nrg ROW-GROUPS of 4, each a
                // block_q8_0x4 stream (stride 136). Group y holds rows
                // y*4..y*4+3; vy[y*nb .. ] = quantize_mat_q8_0_4x1(af_y).
                std::vector<block_q8_0x4> vy(nrg * nb);
                std::vector<float> af(4 * n);
                for (int y = 0; y < nrg; y++) {
                    for (int i = 0; i < 4 * n; i++) af[i] = act(rng);
                    quantize_mat_q8_0_4x1(af.data(), vy.data() + y * nb, n);
                }
                std::vector<float> out_mine(NR * NC), out_ref(NR * NC);
                const size_t bs = NC;  // output row stride = nc (one chunk)
                ggml_gemm_q4_0_16x1_q8_0(n, out_mine.data(), bs, vx.data(), vy.data(), NR, NC);
                gemm_ref(n, out_ref.data(), bs, vx.data(), vy.data(), NR, NC);
                for (int m = 0; m < NR; m++) for (int j = 0; j < NC; j++) {
                    double v = (double)out_ref[m*bs+j];
                    double e = fabs((double)out_mine[m*bs+j] - v);
                    if (e > max_g) { max_g = e; worst_n = n; worst_m = m; worst_j = j; }
                    sse_g += v*v; cnt_g++;
                    maxrel_g = std::max(maxrel_g, e/(fabs(v)+1e-6));
                }
            }
        }
        double rms_g = sqrt(sse_g/cnt_g);
        bool ok = (max_g/rms_g < 1e-4);
        all_ok = all_ok && ok;
        printf("  nr=%-2d (%d rg)  nc=%-3d (%d cg)  max_abs_err=%.3e  rms(ref)=%.3e  norm=%.3e  rel=%.3e  %s",
               NR, nrg, NC, ncg, max_g, rms_g, max_g/rms_g, maxrel_g, ok ? "PASS" : "FAIL");
        if (!ok) printf("   <<< worst @ n=%d row=%d col=%d", worst_n, worst_m, worst_j);
        printf("\n");
    }
    printf("  VERDICT: %s\n", all_ok
        ? "PASS (emitted RVV0.7 GEMM matches generic across nr=4/8, nc=16/32/64 within fp32 rounding -> GEMM CORRECT -> q4_0-garbage is WIRING)"
        : "FAIL (emitted RVV0.7 GEMM diverges -> GEMM IS THE BUG; see worst nr/nc/row/col above)");
    return all_ok ? 0 : 1;
}
