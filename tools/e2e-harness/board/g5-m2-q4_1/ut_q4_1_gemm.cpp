// [G5-M2 q4_1] GEMM (prefill) unit test: make_block_q4_1x16 + EMITTED vl=8 GEMM vs
// independent scalar q4_1 oracle (UNSIGNED 4-bit nibble + min; NO qh). block_q8_1x4
// activation (d[4]@0, s[4]@8, qs[128]@16, stride 144, elem e row m = qs[e*4+m]).
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

typedef _Float16 ggml_half;
#define QK4_1 32
#define QK8_1 32
struct block_q4_1 { ggml_half d; ggml_half m; uint8_t qs[QK4_1/2]; };
struct block_q8_1 { ggml_half d; ggml_half s; int8_t qs[QK8_1]; };
struct block_q8_1x4 { ggml_half d[4]; ggml_half s[4]; int8_t qs[128]; };   // 144B
struct block_q4_1x16 { ggml_half d[16]; ggml_half m[16]; uint8_t qs[256]; };
static_assert(sizeof(block_q4_1x16) == 320, "320");
static_assert(sizeof(block_q8_1x4) == 144, "144");

static block_q4_1x16 make_block_q4_1x16(block_q4_1 * in, unsigned int blck_size_interleave) {
    block_q4_1x16 out;
    for (int i = 0; i < 16; i++) {
        const uint8_t * raw = (const uint8_t *) &in[i];
        std::memcpy(&out.d[i], raw + 0, 2);
        std::memcpy(&out.m[i], raw + 2, 2);
    }
    const int end = (QK4_1/2) * 16 / blck_size_interleave;
    for (int i = 0; i < end; ++i) { int s = i%16, o = i/16; out.qs[i] = in[s].qs[o]; }
    return out;
}
static float oracle_dot(const block_q4_1 & w, const block_q8_1 & a) {
    int isum = 0;
    for (int j = 0; j < QK4_1/2; j++) {
        const int w0 = (w.qs[j] & 0x0F);
        const int w1 = (w.qs[j] >>   4);
        isum += w0 * a.qs[j] + w1 * a.qs[j + 16];
    }
    return (float)isum * (float)w.d * (float)a.d + (float)w.m * (float)a.s;
}
#include "tcrv_emitted_gemm_q4_1.inc"

int main() {
    const int nb = 8, nc = 16, nr = 4, bs = 16, n = nb*QK4_1;
    std::mt19937 rng(0xBADF00D);
    std::uniform_int_distribution<int> qd(0,255), ad(-127,127);
    std::uniform_real_distribution<float> dd(0.001f,0.05f);

    std::vector<block_q4_1> src(nc*nb);
    for (auto & b : src) {
        b.d=(ggml_half)dd(rng); b.m=(ggml_half)(dd(rng)-0.025f);
        for(int i=0;i<16;i++)b.qs[i]=(uint8_t)qd(rng);
    }
    std::vector<block_q8_1> arow(nr*nb);   // arow[r*nb+blk]
    for (auto & a : arow) {
        a.d=(ggml_half)dd(rng); int sum=0;
        for(int i=0;i<32;i++){a.qs[i]=(int8_t)ad(rng); sum+=a.qs[i];}
        a.s=(ggml_half)((float)a.d*(float)sum);
    }
    std::vector<block_q8_1x4> apack(nb);
    for (int blk=0; blk<nb; blk++) {
        for (int m=0;m<4;m++) { apack[blk].d[m] = arow[m*nb+blk].d; apack[blk].s[m] = arow[m*nb+blk].s; }
        for (int e=0;e<32;e++) for(int m=0;m<4;m++) apack[blk].qs[e*4+m] = arow[m*nb+blk].qs[e];
    }
    std::vector<block_q4_1x16> packed(nb);
    for (int blk=0; blk<nb; blk++) { block_q4_1 t[16]; for(int c=0;c<16;c++) t[c]=src[c*nb+blk]; packed[blk]=make_block_q4_1x16(t,1); }

    std::vector<float> s_ref(nr*bs, 0.0f);
    for (int r=0;r<nr;r++) for(int c=0;c<nc;c++){ float acc=0; for(int blk=0;blk<nb;blk++) acc += oracle_dot(src[c*nb+blk], arow[r*nb+blk]); s_ref[r*bs+c]=acc; }

    std::vector<float> s_emit(nr*bs, 0.0f);
    tcrv_emitc_ggml_gemm_q4_1_q8_1_kernel_ggml_gemm_q4_1_q8_1(
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
