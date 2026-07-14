// q3_K x q8_K 16x1-REPACKED GEVM + GEMM numeric oracle.
//
// Gate: does the compiler-emitted q3_K repack kernel compute the CORRECT q3_K
// dequant-matmul result?
//
// The oracle is INDEPENDENT: the REFERENCE decodes the ORIGINAL (pre-repack)
// per-block q3_K exactly as ggml's canonical dequantize_row_q3_K / vec_dot_q3_K_q8_K:
//   3-bit weight VALUE = ((qs[..]>>shift)&3) - ((hmask[..] & (1<<p)) ? 0 : 4)
//       -- the SUBTRACTIVE hmask: the high bit CLEAR subtracts 4, the high bit SET
//          subtracts 0 (equivalently ((qs&3)|(hbit<<2)) - 4, a SIGNED value in
//          [-4,3]);  hmask bit position p = 4*(n/128) + j, the SAME 32-byte hmask
//          plane reused across BOTH super-halves;
//   per-16-element SIGNED 6-bit scale (12 packed scale bytes -> 16 six-bit values,
//       scale_j = unpacked_j - 32), SINGLE super-block d, NO min (a SINGLE
//       accumulator, exactly like q6_K).
// The EMITTER-MODEL reads the REPACKED block_q3_Kx16 and reproduces the emitter's
// EXACT lane-wise integer organization (block-as-lane strips, per-shift-quadrant i16
// partial over 16 positions, i32 scale-weighted fold, single accumulator, NO min).
// The two paths use DIFFERENT data layouts and DIFFERENT arithmetic organization, so
// agreement is strong evidence.
//
// The integer contraction isum is compared BYTE-EXACT (int64 equality) between the
// reference and the emitter-model -- this is the core numeric certificate. The i16
// intermediate is tracked to certify no overflow (a 16-position partial with |w|<=4
// and |q8|<=127 stays 16*4*127 = 8128 < 32767, so q3_K -- unlike q6_K's [-32,31]
// weight -- needs NO 2x8 k-chunk split). The fp result norm is reported informationally.
//
// NEGATIVE CONTROLS perturb the REFERENCE; an integer-mismatch SPIKE proves the
// corresponding q3_K structure was genuinely exercised by the emitter-model:
//   HMASK-OFF : treat every hmask bit as CLEAR in the ref -> the high bit vanishes
//               (weights always -4); the emitter reads the REAL bits -> spike.
//   HMASK-INV : INVERT the subtractive polarity (subtract 4 when the bit is SET, not
//               CLEAR) -> proves the emitter's SUBTRACTIVE direction is the right one.
//   SCALE-ROT : rotate the 16 signed scales by +1 -> a scale permutation bug cannot
//               cancel (distinct-per-sub-block scales required).
//   BIAS-OFF  : drop the -4 subtractive bias -> the signed weight lane is exercised.
//
// Build:  g++ -O2 -std=c++17 oracle_repack_q3K.cpp -o /tmp/oracle_q3k && /tmp/oracle_q3k

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define QK_K 256

// ---- original per-block q3_K (ggml-common.h), 110 bytes ----
struct block_q3_K {
    uint8_t hmask[QK_K / 8];   // +0    high bit plane (32 B, 1 bit/weight)
    uint8_t qs[QK_K / 4];      // +32   low 2-bit plane (64 B, 4 weights/byte)
    uint8_t scales[12];        // +96   packed 6-bit scales (16 of them)
    float   d;                 // super-block scale (float here for the int test;
                               // fp16 does not affect the byte-exact INTEGER isum)
};

// ---- repacked block_q3_Kx16 (this project's block-as-lane convention), 1824 B ----
//   d[16]      @ +0     (32 B, fp16 in the real kernel; float here for the int test)
//   scales[256]@ +32    (256 B: SIGNED int8, byte for sub-block s / column c at
//                        32 + s*16 + c;  value = unpacked_6bit - 32, in [-32,31] --
//                        the 12-byte pack + the -32 bias are folded into the REPACK,
//                        so the emitter's scale side is byte-identical to q6_K's)
//   hmask[512] @ +288   (512 B: byte for hmask index i (0..31) / column c at
//                        288 + i*16 + c)
//   qs[1024]   @ +800   (1024 B: byte for qs index i (0..63) / column c at
//                        800 + i*16 + c)
struct block_q3_Kx16 {
    float   d[16];
    int8_t  scales[256];
    uint8_t hmask[512];
    uint8_t qs[1024];
};

// ---- plain block_q8_K, 292 B (bsums UNUSED by q3_K) ----
struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
};

// ---- unpack the 12 packed q3_K scale bytes into 16 SIGNED int8 (value - 32) ----
// Byte-identical to ggml's canonical scale bit-dance
// (kmask1=0x03030303, kmask2=0x0f0f0f0f).
static void unpack_q3K_scales(const uint8_t packed[12], int8_t out[16]) {
    const uint32_t kmask1 = 0x03030303u, kmask2 = 0x0f0f0f0fu;
    uint32_t aux[4];
    std::memcpy(aux, packed, 12);
    uint32_t tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
    aux[0] = ((aux[0]     ) & kmask2) | (((tmp >> 0) & kmask1) << 4);
    aux[1] = ((aux[1]     ) & kmask2) | (((tmp >> 2) & kmask1) << 4);
    const int8_t *sc = (const int8_t *)aux;
    for (int i = 0; i < 16; ++i) out[i] = (int8_t)((int)sc[i] - 32); // [-32,31]
}

// ---- make_block_q3_Kx16: interleave 16 original q3_K blocks (one per lane/col) ----
static block_q3_Kx16 make_block_q3_Kx16(const block_q3_K *in) {
    block_q3_Kx16 out;
    for (int c = 0; c < 16; ++c) out.d[c] = in[c].d;
    // the packed 12-byte scales are UNPACKED + (-32)-biased at repack time.
    for (int c = 0; c < 16; ++c) {
        int8_t sc[16];
        unpack_q3K_scales(in[c].scales, sc);
        for (int s = 0; s < 16; ++s) out.scales[s * 16 + c] = sc[s];
    }
    for (int i = 0; i < 32; ++i)
        for (int c = 0; c < 16; ++c)
            out.hmask[i * 16 + c] = in[c].hmask[i];
    for (int i = 0; i < 64; ++i)
        for (int c = 0; c < 16; ++c)
            out.qs[i * 16 + c] = in[c].qs[i];
    return out;
}

enum DqMode { DQ_NORMAL = 0, DQ_HMASKOFF = 1, DQ_HMASKINV = 2, DQ_SCALEROT = 3,
              DQ_BIASOFF = 4 };

// ---- REFERENCE integer isum for ONE original q3_K block dotted with ONE q8_K block.
// Mirrors ggml's canonical decode; returns sum_i(scale_i * weight3_i * q8_i) as int64.
static int64_t ref_isum_block(const block_q3_K *x, const block_q8_K *a, DqMode mode) {
    int8_t sc[16];
    unpack_q3K_scales(x->scales, sc);         // already (unpacked - 32)
    if (mode == DQ_SCALEROT) {
        int8_t rot[16];
        for (int i = 0; i < 16; ++i) rot[i] = sc[(i + 1) % 16];
        for (int i = 0; i < 16; ++i) sc[i] = rot[i];
    }
    int bias = (mode == DQ_BIASOFF) ? 0 : 4;

    int64_t isum = 0;
    int is = 0;
    for (int n = 0; n < QK_K; n += 128) {
        int sh = n / 128;                     // super-half 0/1
        const uint8_t *q = x->qs + sh * 32;   // qs advances 32 bytes per super-half
        for (int j = 0; j < 4; ++j) {         // shift group; shift = 2j
            int shift = 2 * j;
            int p = 4 * sh + j;               // hmask bit position 0..7
            for (int grp = 0; grp < 2; ++grp) {
                int scl = sc[is++];
                for (int l = 0; l < 16; ++l) {
                    int low2 = (q[grp * 16 + l] >> shift) & 3;
                    int hset = (x->hmask[grp * 16 + l] >> p) & 1;
                    int sub;
                    if (mode == DQ_HMASKOFF)      sub = bias;                 // bit forced CLEAR
                    else if (mode == DQ_HMASKINV) sub = hset ? bias : 0;      // inverted polarity
                    else                          sub = hset ? 0 : bias;      // SUBTRACTIVE
                    int w = low2 - sub;
                    int base = n + j * 32 + grp * 16 + l;
                    isum += (int64_t)scl * w * a->qs[base];
                }
            }
        }
    }
    return isum;
}

// ---- EMITTER-MODEL integer isum for column c of a repacked group, dotted with a
// per-column activation quant accessor `act(globalPos)`. Reproduces the emitter's
// EXACT lane-wise organization (sh x grp x 4 shift-quadrants, each a 16-position i16
// partial folded scale-weighted into i32). `maxAbsPartial` tracks the i16 magnitude.
template <typename ActFn>
static int64_t mine_isum_col(const block_q3_Kx16 *b, int c, ActFn act,
                             int &maxAbsPartial) {
    int64_t sumi = 0;
    for (int sh = 0; sh < 2; ++sh) {
        for (int grp = 0; grp < 2; ++grp) {
            int scale[4];
            for (int q = 0; q < 4; ++q) {           // shift-quadrant q == shift group
                int sIdx = sh * 8 + q * 2 + grp;
                scale[q] = b->scales[sIdx * 16 + c]; // signed [-32,31]
            }
            int partial[4] = {0, 0, 0, 0};
            for (int l = 0; l < 16; ++l) {
                uint8_t qsb = b->qs[(sh * 32 + grp * 16 + l) * 16 + c];
                uint8_t hmb = b->hmask[(grp * 16 + l) * 16 + c];
                for (int q = 0; q < 4; ++q) {
                    int shift = 2 * q;
                    int p = 4 * sh + q;
                    int low2 = (qsb >> shift) & 3;
                    int hbit = ((hmb >> p) & 1) << 2;
                    int w = (int)(int8_t)((low2 | hbit)) - 4;  // SUBTRACTIVE -> [-4,3]
                    int aq = act(sh * 128 + q * 32 + grp * 16 + l);
                    partial[q] += aq * w;                       // i16 accumulate x16
                }
            }
            for (int q = 0; q < 4; ++q) {
                int ap = std::abs(partial[q]);
                if (ap > maxAbsPartial) maxAbsPartial = ap;
                sumi += (int64_t)scale[q] * partial[q];
            }
        }
    }
    return sumi;
}

// ---- build ONE adversarial original q3_K block (distinct per col/blk) ----
static void build_q3K_block(block_q3_K *x, std::mt19937 &rng, int col, int blk) {
    std::uniform_int_distribution<int> byte(0, 255);
    x->d = 0.013f + 0.0006f * ((col + 3 * blk) % 11);
    // distinct-per-sub-block packed scales: fill the 12 scale bytes with varied
    // values so the 16 unpacked 6-bit scales differ across sub-blocks.
    for (int i = 0; i < 12; ++i)
        x->scales[i] = (uint8_t)((17 + 29 * i + 7 * col + 13 * blk) & 0xFF);
    for (int i = 0; i < QK_K / 4; ++i) x->qs[i] = (uint8_t)byte(rng);    // full 8-bit qs
    for (int i = 0; i < QK_K / 8; ++i) x->hmask[i] = (uint8_t)byte(rng); // full 8-bit hmask
}

// ---- build a q8_K activation block with per-16-group distinct DC (varied values) ----
static void build_q8K_block(block_q8_K *a, std::mt19937 &rng, int blk) {
    std::uniform_int_distribution<int> q8d(-90, 90);
    a->d = 0.015f + 0.0009f * (blk % 13);
    for (int grp = 0; grp < QK_K / 16; ++grp) {
        int dc = ((grp * 7 + blk * 3) % 21) - 10;
        int sum = 0;
        for (int ii = 0; ii < 16; ++ii) {
            int v = q8d(rng) + dc;
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            a->qs[grp * 16 + ii] = (int8_t)v;
            sum += v;
        }
        a->bsums[grp] = (int16_t)sum;
    }
}

int main() {
    printf("# q3_K x q8_K 16x1-REPACKED oracle (independent scalar).\n");
    printf("# BYTE-EXACT integer isum: reference (original q3_K) vs emitter-model "
           "(repacked block_q3_Kx16).\n\n");

    std::mt19937 rng(20260708u);
    struct Shape { int nc; int n; int nr; };
    std::vector<Shape> shapes = {
        {16, 256, 1}, {16, 4096, 1}, {32, 256, 4}, {32, 2560, 4},
        {256, 256, 1}, {256, 4096, 8}, {160, 2560, 4},
    };

    bool anyBug = false;
    int globalMaxPartial = 0;
    double worstNorm = 0.0;

    for (auto sh : shapes) {
        const int nc = sh.nc, n = sh.n, nr = sh.nr;
        const int nb = n / QK_K, ng = nc / 16;

        std::vector<block_q3_K> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_q3K_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);

        std::vector<block_q3_Kx16> vx((size_t)ng * nb);
        std::vector<block_q3_K> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_q3_Kx16(tmp16.data());
            }

        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        long long mismatch = 0;
        int maxPartial = 0;
        double sumsqRef = 0, maxAbsErr = 0;
        for (int r = 0; r < nr; ++r) {
            for (int g = 0; g < ng; ++g) {
                for (int col = 0; col < 16; ++col) {
                    int gcol = g * 16 + col;
                    double refF = 0, mineF = 0;
                    for (int l = 0; l < nb; ++l) {
                        const block_q3_K *xo = &orig[(size_t)gcol * nb + l];
                        const block_q8_K *a = &act[(size_t)r * nb + l];
                        int64_t ri = ref_isum_block(xo, a, DQ_NORMAL);
                        const block_q3_Kx16 *b = &vx[(size_t)g * nb + l];
                        auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                        int64_t mi = mine_isum_col(b, col, actFn, maxPartial);
                        if (ri != mi) mismatch++;
                        refF  += (double)xo->d * (double)a->d * (double)ri;
                        mineF += (double)b->d[col] * (double)a->d * (double)mi;
                    }
                    double e = std::fabs(refF - mineF);
                    if (e > maxAbsErr) maxAbsErr = e;
                    sumsqRef += refF * refF;
                }
            }
        }
        int ncnt = nr * nc;
        double rms = std::sqrt(sumsqRef / ncnt);
        double norm = rms > 0 ? maxAbsErr / rms : maxAbsErr;
        if (norm > worstNorm) worstNorm = norm;
        if (maxPartial > globalMaxPartial) globalMaxPartial = maxPartial;
        bool bug = (mismatch != 0);
        if (bug) anyBug = true;
        printf("  shape nr=%-3d nc=%-4d n=%-6d : int-mismatch=%-6lld  "
               "maxAbsPartial=%-6d (i16 ok<=32767)  fp-norm=%.3e  %s\n",
               nr, nc, n, mismatch, maxPartial, norm,
               bug ? "INT-BUG" : "BYTE-EXACT");
    }

    // ---- GEMM interleaved-activation addressing: pack 4 plain q8_K rows into one
    // block_q8_Kx4 (qs at +16, byte for flat pos p / column c = p*4 + c) and confirm
    // the emitter-model's interleaved read is BYTE-EXACT to the per-row plain dot.
    {
        const int nc = 32, n = 2560;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_q3_K> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_q3K_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);
        std::vector<block_q3_Kx16> vx((size_t)ng * nb);
        std::vector<block_q3_K> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_q3_Kx16(tmp16.data());
            }
        std::vector<block_q8_K> row(4 * nb);
        for (int r = 0; r < 4; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8K_block(&row[(size_t)r * nb + l], rng, 900 + r * 7 + l);
        std::vector<int8_t> qsx4((size_t)nb * QK_K * 4);
        for (int l = 0; l < nb; ++l)
            for (int p = 0; p < QK_K; ++p)
                for (int c = 0; c < 4; ++c)
                    qsx4[(size_t)l * QK_K * 4 + p * 4 + c] =
                        row[(size_t)c * nb + l].qs[p];
        long long mism = 0, total = 0;
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col) {
                int gcol = g * 16 + col;
                for (int c = 0; c < 4; ++c)
                    for (int l = 0; l < nb; ++l) {
                        int64_t ref = ref_isum_block(&orig[(size_t)gcol * nb + l],
                                                     &row[(size_t)c * nb + l],
                                                     DQ_NORMAL);
                        const int8_t *qsl = &qsx4[(size_t)l * QK_K * 4];
                        auto actFn = [&](int pos) { return (int)qsl[pos * 4 + c]; };
                        int dummy = 0;
                        int64_t mine = mine_isum_col(&vx[(size_t)g * nb + l], col,
                                                     actFn, dummy);
                        total++;
                        if (ref != mine) mism++;
                    }
            }
        printf("\n# GEMM interleaved block_q8_Kx4 (qs @+16, pos*4+c): "
               "int-mismatch = %lld / %lld  %s\n",
               mism, total, mism ? "INT-BUG" : "BYTE-EXACT");
        if (mism) anyBug = true;
    }

    // ---- NEGATIVE CONTROLS on one shape (nr=4, nc=32, n=2560) ----
    printf("\n# negative controls (perturb the REFERENCE; a large integer-mismatch\n"
           "# fraction proves the q3_K feature is genuinely exercised):\n");
    {
        const int nc = 32, n = 2560, nr = 4;
        const int nb = n / QK_K, ng = nc / 16;
        std::vector<block_q3_K> orig((size_t)nc * nb);
        for (int g = 0; g < ng; ++g)
            for (int col = 0; col < 16; ++col)
                for (int l = 0; l < nb; ++l)
                    build_q3K_block(&orig[(size_t)(g * 16 + col) * nb + l], rng,
                                    g * 16 + col, l);
        std::vector<block_q3_Kx16> vx((size_t)ng * nb);
        std::vector<block_q3_K> tmp16(16);
        for (int g = 0; g < ng; ++g)
            for (int l = 0; l < nb; ++l) {
                for (int col = 0; col < 16; ++col)
                    tmp16[col] = orig[(size_t)(g * 16 + col) * nb + l];
                vx[(size_t)g * nb + l] = make_block_q3_Kx16(tmp16.data());
            }
        std::vector<block_q8_K> act((size_t)nr * nb);
        for (int r = 0; r < nr; ++r)
            for (int l = 0; l < nb; ++l)
                build_q8K_block(&act[(size_t)r * nb + l], rng, r * 131 + l);

        const char *names[4] = {"HMASK-OFF (force high bit CLEAR)",
                                "HMASK-INV (invert subtractive polarity)",
                                "SCALE-ROT (rotate signed scales +1)",
                                "BIAS-OFF  (drop -4 subtractive bias)"};
        DqMode modes[4] = {DQ_HMASKOFF, DQ_HMASKINV, DQ_SCALEROT, DQ_BIASOFF};
        for (int m = 0; m < 4; ++m) {
            long long total = 0, mism = 0;
            for (int r = 0; r < nr; ++r)
                for (int g = 0; g < ng; ++g)
                    for (int col = 0; col < 16; ++col) {
                        int gcol = g * 16 + col;
                        for (int l = 0; l < nb; ++l) {
                            const block_q3_K *xo = &orig[(size_t)gcol * nb + l];
                            const block_q8_K *a = &act[(size_t)r * nb + l];
                            int dummy = 0;
                            int64_t refP = ref_isum_block(xo, a, modes[m]);
                            auto actFn = [&](int pos) { return (int)a->qs[pos]; };
                            int64_t mine = mine_isum_col(&vx[(size_t)g * nb + l],
                                                         col, actFn, dummy);
                            total++;
                            if (refP != mine) mism++;
                        }
                    }
            printf("  %-40s : perturbed-ref vs emitter int-mismatch = %lld / %lld "
                   "(%.0f%%)  %s\n",
                   names[m], mism, total, 100.0 * mism / total,
                   mism > 0 ? "EXERCISED" : "!! NOT EXERCISED");
        }
    }

    printf("\nWORST_FP_NORM %.3e   GLOBAL_MAX_i16_PARTIAL %d (bound 32767)   "
           "VERDICT %s\n",
           worstNorm, globalMaxPartial, anyBug ? "INT-BUG" : "BYTE-EXACT-INTEGER");
    return anyBug ? 1 : 0;
}
