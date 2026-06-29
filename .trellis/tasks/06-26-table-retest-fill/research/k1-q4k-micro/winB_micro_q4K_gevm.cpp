// Win-B2 micro (decode GEVM): OUR compiler-emitted q4_K repack GEVM (VLEN256-native
// h16 single mf2 strip, block-as-lane, 16 cols/group) vs ggml's REAL shipped
// vectorized ggml_gemv_q4_K_16x1_q8_K (the SHIPPING VLEN256 decode path on k1 -- ggml
// repacks q4_K -> 16x1 GEMV at VLEN>=256, so this is the methodology-correct opponent,
// NOT block-dot, NOT _generic).  Win-B2 success criterion = PARITY.
//
// Both kernels read IDENTICAL buffers:
//   vx = block_q4_Kx16 stream (ng groups x nb blocks, group-major then block-major)
//   vy = plain block_q8_K stream (nb blocks, with bsums)  ->  write nc contiguous floats.
// (Confirmed from ggml arch/riscv/repack.cpp: a_ptr=(block_q8_K*)vy, b_ptr=(q4_Kx16*)vx+x*nb,
//  vse32 s+x*16; bs UNUSED.)  So the SAME vx/vy go to both -- a clean apples-to-apples
//  matched-repack comparison (no per-column handicap like the q5_0 block-dot baseline).
//
// Gates (BEFORE any perf):
//  - OURS already byte-exact via banked oracle (WORST_NORM 7.07e-7, reproduced this session).
//  - This harness ALSO runs an INDEPENDENT scalar q4_K dequant-matmul reference at the
//    smallest shape and asserts BOTH norm(ours-ref) < 1e-4 AND norm(ggml-ref) < 1e-4
//    (self-contained gate for the LINKED ggml symbol too), plus per-shape SANITY ours~ggml.
//
// Regime: k1 (SpacemiT X60), VLEN256, clang-18 -O3, taskset -c 0-3, best-of-reps min.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <random>

#define QK_K 256
#define K_SCALE_SIZE 12

struct block_q4_K {
    _Float16 d;
    _Float16 dmin;
    uint8_t  scales[K_SCALE_SIZE];
    uint8_t  qs[QK_K/2];
};
static_assert(sizeof(block_q4_K) == 4 + 12 + 128, "q4_K size");

struct block_q4_Kx16 {
    _Float16 d[16];
    _Float16 dmin[16];
    uint8_t  scales[192];
    uint8_t  qs[2048];
};
static_assert(sizeof(block_q4_Kx16) == 32 + 32 + 192 + 2048, "q4_Kx16 size 2304");

struct block_q8_K {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K/16];
};
static_assert(sizeof(block_q8_K) == 4 + 256 + 32, "q8_K size 292");

// ---- ggml get_scale_min_k4 (verbatim) ----
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) { *d = q[j] & 63; *m = q[j + 4] & 63; }
    else { *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4); *m = (q[j+4] >> 4) | ((q[j-0] >> 6) << 4); }
}

static void dequantize_block_q4_K(const block_q4_K * x, float * y) {
    const uint8_t * q = x->qs;
    const float d   = (float) x->d;
    const float min = (float) x->dmin;
    uint8_t scv[8], mnv[8];
    for (int sb = 0; sb < 8; sb++) get_scale_min_k4(sb, x->scales, &scv[sb], &mnv[sb]);
    int sb = 0;
    for (int jj = 0; jj < QK_K; jj += 64) {
        const float d1 = d * scv[sb];   const float m1 = min * mnv[sb];
        const float d2 = d * scv[sb+1]; const float m2 = min * mnv[sb+1];
        for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
        for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
        q += 32; sb += 2;
    }
}

// ---- make_block_q4_Kx16 (repack.cpp:2913, blck=1) (verbatim) ----
static block_q4_Kx16 make_block_q4_Kx16(const block_q4_K * in) {
    block_q4_Kx16 out;
    for (int i = 0; i < 16; i++) out.d[i]    = in[i].d;
    for (int i = 0; i < 16; i++) out.dmin[i] = in[i].dmin;
    const int end = QK_K * 8; // blck=1
    for (int i = 0; i < end; ++i) { int src_id = i % 16; int src_offset = i / 16; out.qs[i] = in[src_id].qs[src_offset]; }
    uint8_t s[128], m[128];
    for (int i = 0; i < 4; i++) for (int j = 0; j < 16; j++) {
        s[i * 16 + j] = in[j].scales[i] & 63;
        m[i * 16 + j] = in[j].scales[i + 4] & 63;
    }
    for (int i = 0; i < 4; i++) for (int j = 0; j < 16; j++) {
        s[64 + i * 16 + j] = ((in[j].scales[i] & 192) >> 2) | (in[j].scales[i+8] & 15);
        m[64 + i * 16 + j] = ((in[j].scales[i + 4] & 192) >> 2) | ((in[j].scales[i+8] & 240) >> 4);
    }
    for (int i = 0; i < 128; i++) out.scales[i] = (s[i] & 15) | ((m[i] & 15) << 4);
    for (int i = 0; i < 64; i++)
        out.scales[128 + i] = ((s[i] & 48) >> 4) | ((m[i] & 48) >> 2) | (s[64 + i] & 48) | ((m[64 + i] & 48) << 2);
    return out;
}

// ---- adversarial original q4_K block (verbatim from oracle) ----
static void build_q4K_block(block_q4_K * b, std::mt19937 & rng, int col, int blk) {
    std::uniform_int_distribution<int> nib(0, 15);
    b->d    = (_Float16)(0.012f + 0.0007f * ((col + 3*blk) % 11));
    b->dmin = (_Float16)(0.008f + 0.0005f * ((2*col + blk) % 7));
    uint8_t ls[8], lm[8];
    for (int sb = 0; sb < 8; sb++) {
        ls[sb] = (uint8_t)((7 + 6*sb + 2*col + blk) & 63);
        lm[sb] = (uint8_t)((5 + 5*sb + col + 3*blk) & 63);
        if (ls[sb] == 0) ls[sb] = 1;
    }
    memset(b->scales, 0, K_SCALE_SIZE);
    for (int sb = 0; sb < 8; sb++) {
        uint8_t l_s = ls[sb], l_m = lm[sb];
        if (sb < 4) { b->scales[sb] = l_s; b->scales[sb+4] = l_m; }
        else { b->scales[sb+4] = (l_s & 0xF) | ((l_m & 0xF) << 4); b->scales[sb-4] |= ((l_s >> 4) << 6); b->scales[sb-0] |= ((l_m >> 4) << 6); }
    }
    for (int i = 0; i < QK_K/2; i++) b->qs[i] = (uint8_t)(nib(rng) | (nib(rng) << 4));
}

// ---- kernels under test ----
extern "C" void tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc);     // OURS, 5 args
extern "C" void ggml_gemv_q4_K_16x1_q8_K(
    int n, float* s, size_t bs, const void* vx, const void* vy, int nr, int nc); // ggml real (.so), 7 args
extern "C" void ggml_src_gemv_q4_K_16x1_q8_K(
    int n, float* s, size_t bs, const void* vx, const void* vy, int nr, int nc); // ggml SRC, our flags

static double now_ns(){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1e9 + ts.tv_nsec; }

// build vx/vy/ref for shape (nc,n).  ref via independent scalar dequant-matmul from pre-repack blocks.
static void build_shape(int nc, int n, std::mt19937 & rng,
                        std::vector<block_q4_Kx16> & vx, std::vector<block_q8_K> & act,
                        std::vector<float> & ref) {
    const int nb = n / QK_K, ng = nc / 16;
    std::vector<block_q4_K> orig((size_t)nc * nb);
    for (int g = 0; g < ng; g++) for (int col = 0; col < 16; col++) for (int l = 0; l < nb; l++)
        build_q4K_block(&orig[(size_t)(g*16+col)*nb + l], rng, g*16+col, l);
    vx.assign((size_t)ng * nb, {});
    { std::vector<block_q4_K> tmp16(16);
      for (int g = 0; g < ng; g++) for (int l = 0; l < nb; l++) {
        for (int col = 0; col < 16; col++) tmp16[col] = orig[(size_t)(g*16+col)*nb + l];
        vx[(size_t)g*nb + l] = make_block_q4_Kx16(tmp16.data());
      } }
    act.assign(nb, {});
    std::uniform_int_distribution<int> q8d(-90, 90);
    for (int l = 0; l < nb; l++) {
        block_q8_K & a = act[l];
        a.d = 0.015f + 0.0009f * (l % 13);
        for (int grp = 0; grp < QK_K/16; grp++) {
            int dc = ((grp * 7 + l * 3) % 21) - 10; int sum = 0;
            for (int ii = 0; ii < 16; ii++) { int v = q8d(rng) + dc; if (v>127) v=127; if (v<-128) v=-128; a.qs[grp*16+ii]=(int8_t)v; sum+=v; }
            a.bsums[grp] = (int16_t)sum;
        }
    }
    ref.assign(nc, 0.0f);
    std::vector<float> w(QK_K);
    for (int g = 0; g < ng; g++) for (int col = 0; col < 16; col++) {
        double acc = 0.0;
        for (int l = 0; l < nb; l++) {
            dequantize_block_q4_K(&orig[(size_t)(g*16+col)*nb + l], w.data());
            const block_q8_K & a = act[l];
            double dot = 0.0;
            for (int i = 0; i < QK_K; i++) dot += (double)w[i] * (double)a.qs[i];
            acc += (double)a.d * dot;
        }
        ref[g*16 + col] = (float)acc;
    }
}

static double norm_vs(const std::vector<float>& a, const std::vector<float>& b){
    double max_abs=0, sumsq=0; int nc=(int)a.size();
    for (int i=0;i<nc;i++){ double e=fabs((double)a[i]-(double)b[i]); if(e>max_abs)max_abs=e; sumsq+=(double)b[i]*(double)b[i]; }
    double rms=sqrt(sumsq/nc); return rms>0?max_abs/rms:max_abs;
}

int main(){
    struct Shape { int nc; int n; };
    std::vector<Shape> shapes = {
        {16, 256},        // tiny -- full INDEPENDENT-ref gate runs here
        {4096, 4096},     // decode GEVM: attn/ffn-down out=4096 contract=4096
        {11008, 4096},    // ffn up/gate out=11008 contract=4096
        {4096, 11008},    // ffn down out=4096 contract=11008
        {14336, 4096},    // larger-model out dim
    };
    std::mt19937 rng(20260629u);

    printf("# q4_K repack GEVM micro -- OURS (VLEN256-native h16) vs ggml REAL ggml_gemv_q4_K_16x1_q8_K\n");
    printf("# board=k1 VLEN256, clang-18 -O3, taskset -c 0-3, min-of-reps. SANITY bar norm<1e-4.\n");
    printf("# RATIO = ggml_ns/ours_ns  ( >1 => OUR repack faster ; ~1 => PARITY ; <1 => ggml faster )\n\n");

    bool gate_fail = false;
    for (size_t si=0; si<shapes.size(); si++) {
        int nc=shapes[si].nc, n=shapes[si].n;
        std::vector<block_q4_Kx16> vx; std::vector<block_q8_K> act; std::vector<float> ref;
        build_shape(nc, n, rng, vx, act, ref);

        std::vector<float> out_ours(nc, 0.0f), out_ggml(nc, 0.0f);
        const uint8_t* pvx = reinterpret_cast<const uint8_t*>(vx.data());
        const uint8_t* pvy = reinterpret_cast<const uint8_t*>(act.data());

        tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K((size_t)n, out_ours.data(), pvx, pvy, (size_t)nc);
        ggml_gemv_q4_K_16x1_q8_K(n, out_ggml.data(), (size_t)nc*sizeof(float), vx.data(), act.data(), 1, nc);

        double sane = norm_vs(out_ours, out_ggml);            // ours vs ggml (same kernel engaged?)
        double n_ours_ref = norm_vs(out_ours, ref);           // ours vs independent scalar ref
        double n_ggml_ref = norm_vs(out_ggml, ref);           // ggml vs independent scalar ref
        const char* gv = (sane<1e-4 && n_ours_ref<1e-4 && n_ggml_ref<1e-4) ? "PASS" : "FAIL";
        if (std::string(gv)=="FAIL") gate_fail=true;
        printf("[gate %5dx%-6d] norm(ours-ggml)=%.3e  norm(ours-ref)=%.3e  norm(ggml-ref)=%.3e  %s\n",
               nc, n, sane, n_ours_ref, n_ggml_ref, gv);

        // ---- adaptive iters: target ~30ms inner, reps=80 min ----
        double t0=now_ns();
        tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K((size_t)n, out_ours.data(), pvx, pvy, (size_t)nc);
        double one_call = now_ns()-t0; if (one_call < 1.0) one_call = 1.0;
        int iters = (int)(30e6 / one_call); if (iters < 1) iters = 1; if (iters > 200000) iters = 200000;
        const int reps = 80;

        // warm
        for (int i=0;i<iters;i++) {
            tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K((size_t)n, out_ours.data(), pvx, pvy, (size_t)nc);
            ggml_gemv_q4_K_16x1_q8_K(n, out_ggml.data(), (size_t)nc*sizeof(float), vx.data(), act.data(), 1, nc);
        }
        std::vector<float> out_src(nc, 0.0f);
        double ob=1e30, gb=1e30, sb=1e30;
        for (int r=0;r<reps;r++){
            double a0=now_ns(); for(int i=0;i<iters;i++) tcrv_emitc_ggml_repack_gemv_q4_K_q8_K_kernel_ggml_repack_gemv_q4_K_q8_K((size_t)n, out_ours.data(), pvx, pvy, (size_t)nc);
            double po=(now_ns()-a0)/iters; if (po<ob) ob=po;
            double b0=now_ns(); for(int i=0;i<iters;i++) ggml_gemv_q4_K_16x1_q8_K(n, out_ggml.data(), (size_t)nc*sizeof(float), vx.data(), act.data(), 1, nc);
            double pg=(now_ns()-b0)/iters; if (pg<gb) gb=pg;
            double c0=now_ns(); for(int i=0;i<iters;i++) ggml_src_gemv_q4_K_16x1_q8_K(n, out_src.data(), (size_t)nc*sizeof(float), vx.data(), act.data(), 1, nc);
            double ps=(now_ns()-c0)/iters; if (ps<sb) sb=ps;
        }
        double src_ref = norm_vs(out_src, ref); // ggml-src must also be correct
        printf("  shape %5dx%-6d  iters=%-7d  ours=%10.1f ns  ggml.so=%10.1f ns  ggmlSRC=%10.1f ns  | RATIO ggml.so/ours=%.3f  ggmlSRC/ours=%.3f  (srcRefNorm=%.1e)\n\n",
               nc, n, iters, ob, gb, sb, gb/ob, sb/ob, src_ref);
    }
    printf("GATE %s\n", gate_fail ? "FAIL" : "PASS");
    return gate_fail ? 1 : 0;
}
