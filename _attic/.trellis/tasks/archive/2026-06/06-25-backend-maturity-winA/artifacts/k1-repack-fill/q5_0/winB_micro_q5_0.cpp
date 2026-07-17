// Win-B micro: OUR compiler-emitted q5_0 repack GEVM (16 weight columns at once,
// block-as-lane, NO per-block vredsum) vs ggml's REAL shipped RVV q5_0 block-dot
// (ggml_vec_dot_q5_0_q8_0, the per-block vwredsum body, vlenb==16/VLEN128 path)
// run ONCE PER COLUMN -- the genuine ggml VLEN128 fallback for a 16-column GEVM
// (ggml ships NO q5_0 repack). Same logical work (nc=16 columns x nb blocks dot
// one q8_0 activation); the ratio is the Win-B number.
//
// Honest frame: ggml ships no q5_0 repack, so the baseline IS its heavy block-dot
// (5-bit qh unpack + per-block vredsum + scattered reads). q5_0 is exactly the
// regime where the repack out-streams the block-dot at VLEN128 (the q4_0
// 1.22->2.6x pattern) -- but MEASURE it. NAME the regime: VLEN128, SG2044,
// clang-18 -O3, taskset -c 2, baseline = ggml_vec_dot_q5_0_q8_0 per-column.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define QK 32
#define WB 22   // plain block_q5_0 bytes
#define YB 34   // plain block_q8_0 bytes

extern "C" {
// ggml's real shipped RVV q5_0 block-dot (one column).
void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy);
// OUR compiler-emitted q5_0 repack GEVM (16 columns, repacked vx).
void ggml_gemv_q5_0_16x1_q8_0(int n, float *s, size_t bs, const void *vx, const void *vy, int nr, int nc);
}

typedef uint16_t ggml_half;
struct block_q5_0   { ggml_half d; uint8_t qh[4]; uint8_t qs[QK/2]; };          // 22
struct block_q8_0   { ggml_half d; int8_t qs[QK]; };                            // 34
struct block_q5_0x16 { ggml_half d[16]; uint8_t qs[QK*8]; uint16_t qhmask[32]; }; // 352

static block_q5_0x16 make_block(const block_q5_0* in) {
    block_q5_0x16 out;
    for (int i = 0; i < 16; i++) out.d[i] = in[i].d;
    for (int i = 0; i < QK*8; ++i) out.qs[i] = in[i % 16].qs[i / 16];
    for (int e = 0; e < 32; ++e) {
        uint16_t m = 0;
        for (int b = 0; b < 16; ++b) {
            uint32_t qh; memcpy(&qh, in[b].qh, sizeof(qh));
            m |= (uint16_t)(((qh >> e) & 1u) << b);
        }
        out.qhmask[e] = m;
    }
    return out;
}

static uint32_t rng = 0x2468ace0u;
static uint32_t xr() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static uint16_t rfp16() { uint16_t s=(xr()&1)<<15,e=(uint16_t)(xr()%31),m=(uint16_t)(xr()&0x3FF); return s|(e<<10)|m; }

static double now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }

int main() {
    const int NC = 16;                 // one repacked column-group
    const int n = 64 * QK, nb = n / QK;
    // Plain weights: NC rows x nb blocks. Activation: 1 column, nb q8_0 blocks.
    block_q5_0* w = (block_q5_0*)malloc((size_t)NC * nb * sizeof(block_q5_0));
    block_q8_0* a = (block_q8_0*)malloc((size_t)nb * sizeof(block_q8_0));
    for (int r = 0; r < NC; ++r)
        for (int b = 0; b < nb; ++b) {
            block_q5_0& blk = w[r*nb+b];
            uint16_t d = rfp16(); memcpy(&blk.d, &d, 2);
            uint32_t qh = xr(); memcpy(blk.qh, &qh, 4);
            for (int k = 0; k < QK/2; ++k) blk.qs[k] = (uint8_t)(xr() & 0xFF);
        }
    for (int b = 0; b < nb; ++b) {
        uint16_t d = rfp16(); memcpy(&a[b].d, &d, 2);
        for (int k = 0; k < QK; ++k) a[b].qs[k] = (int8_t)(xr() & 0xFF);
    }
    // Repack the NC=16 rows into one block_q5_0x16 stream (nb blocks).
    block_q5_0x16* vx = (block_q5_0x16*)malloc((size_t)nb * sizeof(block_q5_0x16));
    { block_q5_0 tmp[16];
      for (int b = 0; b < nb; ++b) { for (int i=0;i<16;i++) tmp[i]=w[i*nb+b]; vx[b]=make_block(tmp); } }

    // Verify the two produce the same 16 outputs (sanity, not the gate).
    float out_ours[16], out_ggml[16];
    ggml_gemv_q5_0_16x1_q8_0(n, out_ours, 0, vx, a, 1, NC);
    for (int c = 0; c < NC; ++c) kern_ggml(n, &out_ggml[c], (const uint8_t*)(w + c*nb), (const uint8_t*)a);
    double maxrel = 0; for (int c=0;c<NC;c++){ double dn=fabs(out_ggml[c]); if(dn<1e-6)dn=1e-6; double rl=fabs(out_ours[c]-out_ggml[c])/dn; if(rl>maxrel)maxrel=rl; }
    printf("SANITY max_rel=%.3e (ours-GEVM vs ggml-per-col)\n", maxrel);

    const int iters = 4000, reps = 200;
    // warm
    for (int i=0;i<iters;i++){ float s16[16]; ggml_gemv_q5_0_16x1_q8_0(n,s16,0,vx,a,1,NC); for(int c=0;c<NC;c++) kern_ggml(n,&s16[c],(const uint8_t*)(w+c*nb),(const uint8_t*)a); }
    double ours_best=1e18, ggml_best=1e18;
    for (int r=0;r<reps;r++){
        double t0=now(); for(int i=0;i<iters;i++){ float s16[16]; ggml_gemv_q5_0_16x1_q8_0(n,s16,0,vx,a,1,NC); } double po=(now()-t0)/iters;
        if(po<ours_best)ours_best=po;
        double t1=now(); for(int i=0;i<iters;i++){ float s16[16]; for(int c=0;c<NC;c++) kern_ggml(n,&s16[c],(const uint8_t*)(w+c*nb),(const uint8_t*)a); } double pg=(now()-t1)/iters;
        if(pg<ggml_best)ggml_best=pg;
    }
    printf("RESULT ours(repack-GEVM,16col) %.1f ns\n", ours_best);
    printf("RESULT ggml(block-dot x16col,vl128) %.1f ns\n", ggml_best);
    printf("RATIO ggml/ours %.3f  (>1 = repack WIN)\n", ggml_best / ours_best);
    free(w); free(a); free(vx);
    return 0;
}
