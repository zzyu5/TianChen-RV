// [G7-L3 q5_0] GEMM (prefill) unit test: make_block_q5_0x16 + EMITTED vl=16 (VLEN256) GEMM
// vs independent scalar q5_0 oracle. Same interleaver as GEVM test; block_q8_0x4
// activation (stride 136, elem e row m = qs[e*4+m]).
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

typedef _Float16 ggml_half;
#define QK5_0 32
#define QK8_0 32
struct block_q5_0 { ggml_half d; uint8_t qh[4]; uint8_t qs[QK5_0/2]; };
struct block_q8_0 { ggml_half d; int8_t  qs[QK8_0]; };
struct block_q8_0x4 { ggml_half d[4]; int8_t qs[128]; };      // 136B
struct block_q5_0x16 { ggml_half d[16]; uint8_t qs[256]; uint8_t qh[64]; };
static_assert(sizeof(block_q5_0x16) == 352, "352");
static_assert(sizeof(block_q8_0x4) == 136, "136");

static block_q5_0x16 make_block_q5_0x16(block_q5_0 * in, unsigned int blck_size_interleave) {
    block_q5_0x16 out;
    for (int i = 0; i < 16; i++) out.d[i] = in[i].d;
    const int end = (QK5_0/2) * 16 / blck_size_interleave;
    for (int i = 0; i < end; ++i) { int s = i%16, o = i/16; out.qs[i] = in[s].qs[o]; }
    uint32_t qh_col[16];
    for (int c = 0; c < 16; c++) std::memcpy(&qh_col[c], in[c].qh, 4);
    for (int k = 0; k < 16; k++) {
        uint16_t lo = 0, hi = 0;
        for (int c = 0; c < 16; c++) {
            lo |= (uint16_t)(((qh_col[c] >> k)        & 1u) << c);
            hi |= (uint16_t)(((qh_col[c] >> (k + 16)) & 1u) << c);
        }
        std::memcpy(out.qh + 0 + k*2, &lo, 2);
        std::memcpy(out.qh + 32 + k*2, &hi, 2);
    }
    return out;
}
static float oracle_dot(const block_q5_0 & w, const block_q8_0 & a) {
    uint32_t qh; std::memcpy(&qh, w.qh, 4);
    int isum = 0;
    for (int j = 0; j < QK5_0/2; j++) {
        const uint8_t xh0 = ((qh >> (j)) << 4) & 0x10;
        const uint8_t xh1 = ((qh >> (j + 12))) & 0x10;
        const int w0 = ((w.qs[j] & 0x0F) | xh0) - 16;
        const int w1 = ((w.qs[j] >>   4) | xh1) - 16;
        isum += w0 * a.qs[j] + w1 * a.qs[j + 16];
    }
    return (float)isum * (float)w.d * (float)a.d;
}
#include "weft_emitted_gemm_q5_0.inc"

int main() {
    const int nb = 8, nc = 16, nr = 4, bs = 16, n = nb*QK5_0;
    std::mt19937 rng(0xBADF00D);
    std::uniform_int_distribution<int> qd(0,255), ad(-127,127);
    std::uniform_real_distribution<float> dd(0.001f,0.05f);

    std::vector<block_q5_0> src(nc*nb);
    for (auto & b : src) { b.d=(ggml_half)dd(rng); for(int i=0;i<4;i++)b.qh[i]=(uint8_t)qd(rng); for(int i=0;i<16;i++)b.qs[i]=(uint8_t)qd(rng); }

    // per (row, block) q8_0, then pack into block_q8_0x4
    std::vector<block_q8_0> arow(nr*nb);   // arow[r*nb+blk]
    for (auto & a : arow) { a.d=(ggml_half)dd(rng); for(int i=0;i<32;i++)a.qs[i]=(int8_t)ad(rng); }
    std::vector<block_q8_0x4> apack(nb);
    for (int blk=0; blk<nb; blk++) {
        for (int m=0;m<4;m++) apack[blk].d[m] = arow[m*nb+blk].d;
        for (int e=0;e<32;e++) for(int m=0;m<4;m++) apack[blk].qs[e*4+m] = arow[m*nb+blk].qs[e];
    }
    std::vector<block_q5_0x16> packed(nb);
    for (int blk=0; blk<nb; blk++) { block_q5_0 t[16]; for(int c=0;c<16;c++) t[c]=src[c*nb+blk]; packed[blk]=make_block_q5_0x16(t,1); }

    std::vector<float> s_ref(nr*bs, 0.0f);
    for (int r=0;r<nr;r++) for(int c=0;c<nc;c++){ float acc=0; for(int blk=0;blk<nb;blk++) acc += oracle_dot(src[c*nb+blk], arow[r*nb+blk]); s_ref[r*bs+c]=acc; }

    std::vector<float> s_emit(nr*bs, 0.0f);
    weft_emitc_ggml_gemm_q5_0_q8_0_kernel_ggml_gemm_q5_0_q8_0(
        (size_t)nr, (size_t)bs, (size_t)n, s_emit.data(), (size_t)nc,
        (const uint8_t*)packed.data(), (const uint8_t*)apack.data());

    int fails=0; float maxrel=0;
    for (int r=0;r<nr;r++) for(int c=0;c<nc;c++){
        float ref=s_ref[r*bs+c], emt=s_emit[r*bs+c];
        float rel=std::fabs(ref-emt)/(std::fabs(ref)+1e-6f);
        if(rel>maxrel)maxrel=rel;
        if(rel>1e-3f){fails++; std::printf("  r%d c%2d ref=% .5f emit=% .5f rel=%.2e  <<<\n",r,c,ref,emt,rel);}
    }
    std::printf("RESULT: %s  (fails=%d/%d  max_rel=%.3e)\n",
                fails==0?"GREEN GEMM correct":"RED GEMM WRONG", fails, nr*nc, maxrel);
    return fails==0?0:1;
}
