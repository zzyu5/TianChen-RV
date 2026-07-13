// [G7-L3 q5_0] Interleaver unit test (correctness-critical MIRAGE de-risk).
// Validates make_block_q5_0x16 (net-new transposed-qh interleaver) + the
// compiler-EMITTED vl=16 (VLEN256) GEVM kernel against an INDEPENDENT scalar q5_0 oracle.
// If the qh transpose bit-order is wrong, emitted output diverges from oracle.
// Build (board, VLEN128): source /opt/tcrv-toolchains/env.sh then
//   g++ -O2 -march=rv64gcv_zvfh ut_q5_0_interleaver.cpp -o ut && ./ut
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

typedef _Float16 ggml_half;
#define QK5_0 32
#define QK8_0 32

// ---- original ggml block layouts ----
struct block_q5_0 { ggml_half d; uint8_t qh[4]; uint8_t qs[QK5_0/2]; };   // 22B
struct block_q8_0 { ggml_half d; int8_t  qs[QK8_0];   };                  // 34B

// ---- NET-NEW interleaved weight block (field order MUST place qh @288) ----
// d[16]@0 (32B), qs[256]@32, qh[64]@288  => stride 352 (matches emitted kernel)
struct block_q5_0x16 { ggml_half d[16]; uint8_t qs[256]; uint8_t qh[64]; };
static_assert(sizeof(block_q5_0x16) == 352, "block_q5_0x16 must be 352B");

// ============ CANDIDATE INTERLEAVER (piece #3, correctness-critical) ============
static block_q5_0x16 make_block_q5_0x16(block_q5_0 * in, unsigned int blck_size_interleave) {
    block_q5_0x16 out;
    for (int i = 0; i < 16; i++) out.d[i] = in[i].d;

    // nibbles: 16 bytes/col * 16 cols = 256, interleave=1 (byte c at [k*16+c])
    const int end = (QK5_0/2) * 16 / blck_size_interleave;   // 256
    if (blck_size_interleave == 1) {
        for (int i = 0; i < end; ++i) {
            int src_id = i % 16, src_offset = i / 16, dst_offset = i;
            out.qs[dst_offset] = in[src_id].qs[src_offset];  // NO xor for q5_0
        }
    } else { std::fprintf(stderr, "unsupported interleave\n"); std::abort(); }

    // qh transpose: original qh is uint32/col (bit e = 5th bit of elem e).
    // interleaved qh_lo[k] (u16 @0..31) bit c = bit k     of col c's qh (k=0..15)
    //             qh_hi[k] (u16 @32..63) bit c = bit k+16 of col c's qh (k=0..15)
    uint32_t qh_col[16];
    for (int c = 0; c < 16; c++) std::memcpy(&qh_col[c], in[c].qh, sizeof(uint32_t));
    for (int k = 0; k < 16; k++) {
        uint16_t lo = 0, hi = 0;
        for (int c = 0; c < 16; c++) {
            lo |= (uint16_t)(((qh_col[c] >> k)        & 1u) << c);
            hi |= (uint16_t)(((qh_col[c] >> (k + 16)) & 1u) << c);
        }
        std::memcpy(out.qh +  0 + k*2, &lo, 2);
        std::memcpy(out.qh + 32 + k*2, &hi, 2);
    }
    return out;
}

// ============ INDEPENDENT ORACLE: standard q5_0 dequant + q8_0 dot ============
static float oracle_dot(const block_q5_0 & w, const block_q8_0 & a) {
    uint32_t qh; std::memcpy(&qh, w.qh, 4);
    int isum = 0;
    for (int j = 0; j < QK5_0/2; j++) {
        const uint8_t xh_0 = ((qh >> (j +  0)) << 4) & 0x10;   // 5th bit elem j
        const uint8_t xh_1 = ((qh >> (j + 12)) >> 0) & 0x10;   // 5th bit elem j+16
        const int w0 = ((w.qs[j] & 0x0F) | xh_0) - 16;
        const int w1 = ((w.qs[j] >>   4) | xh_1) - 16;
        isum += w0 * a.qs[j] + w1 * a.qs[j + 16];
    }
    return (float)isum * (float)w.d * (float)a.d;
}

// emitted GEVM kernel
#include "weft_emitted_gevm_q5_0.inc"

int main() {
    const int nb = 8;                 // blocks
    const int nc = 16;                // one 16-col group
    const int n  = nb * QK5_0;        // total elements = 256
    std::mt19937 rng(0xC0FFEE);
    std::uniform_int_distribution<int> qd(0, 255), ad(-127, 127);
    std::uniform_real_distribution<float> dd(0.001f, 0.05f);

    // weight source, column-major: src[col*nb + blk]
    std::vector<block_q5_0> src(nc * nb);
    for (auto & b : src) {
        b.d = (ggml_half)dd(rng);
        for (int i = 0; i < 4; i++)  b.qh[i] = (uint8_t)qd(rng);
        for (int i = 0; i < 16; i++) b.qs[i] = (uint8_t)qd(rng);
    }
    // activation q8_0, nb blocks
    std::vector<block_q8_0> act(nb);
    for (auto & a : act) {
        a.d = (ggml_half)dd(rng);
        for (int i = 0; i < 32; i++) a.qs[i] = (int8_t)ad(rng);
    }

    // repack: dst[blk] = interleave of {src[col*nb+blk] : col 0..15}
    std::vector<block_q5_0x16> packed(nb);
    for (int blk = 0; blk < nb; blk++) {
        block_q5_0 tmp[16];
        for (int col = 0; col < 16; col++) tmp[col] = src[col*nb + blk];
        packed[blk] = make_block_q5_0x16(tmp, 1);
    }

    // oracle: s_ref[col] = sum_blk oracle_dot(src[col*nb+blk], act[blk])
    std::vector<float> s_ref(nc, 0.0f);
    for (int col = 0; col < nc; col++)
        for (int blk = 0; blk < nb; blk++)
            s_ref[col] += oracle_dot(src[col*nb + blk], act[blk]);

    // emitted kernel
    std::vector<float> s_emit(nc, 0.0f);
    weft_emitc_ggml_vec_dot_q5_0_q8_0_kernel_ggml_vec_dot_q5_0_q8_0(
        (size_t)n, s_emit.data(), (size_t)nc,
        (const uint8_t*)packed.data(), (size_t)0,
        (const uint8_t*)act.data(),   (size_t)0, (int32_t)0);

    // compare
    int fails = 0; float max_rel = 0.0f;
    for (int col = 0; col < nc; col++) {
        float ref = s_ref[col], emt = s_emit[col];
        float rel = std::fabs(ref - emt) / (std::fabs(ref) + 1e-6f);
        if (rel > max_rel) max_rel = rel;
        int bad = rel > 1e-3f;
        if (bad) fails++;
        std::printf("  col %2d  ref=% .6f  emit=% .6f  rel=%.2e %s\n",
                    col, ref, emt, rel, bad ? "  <<< MISMATCH" : "");
    }
    std::printf("\nRESULT: %s  (fails=%d/%d  max_rel=%.3e)\n",
                fails == 0 ? "GREEN interleaver correct" : "RED qh bit-order WRONG",
                fails, nc, max_rel);
    return fails == 0 ? 0 : 1;
}
