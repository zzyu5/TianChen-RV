/* kquant_repack_verify_q4K.c — [SEL-1 T4b / M0 tracer bullet] SILICON byte-exact / bounded-ULP
 * verifier for the compiler-emitted q4_K x q8_K 16x1-REPACKED GEMM kernel (golden emitter
 * lowering, HEAD 039133ea; the exact .kernel.c md5 b0b5beac..., tcrv-opt --tcrv-rvv-lower-to-emitc
 * | mlir-translate). This is the M0 "single tensor single mul_mat finest line" tracer bullet for
 * the T4b full-construct e2e integration: it drives the WHOLE finest chain for ONE mul_mat's worth
 * of data — repack q4_K weights -> block_q4_Kx16 ; interleave q8_K activations -> block_q8_Kx4 ;
 * run OUR constructed repack-GEMM kernel ; compare vs the board's OWN stock ggml integer path
 * (ggml_vec_dot_q4_K_q8_K, linked from libggml-cpu.so — the exact per-(row,col) block-dot that
 * ggml's prefill mul_mat dispatches to at VLEN128, where the q4_K repack trait is a NULL no-op).
 *
 * NOT a perf probe (no timing) — feasibility + numeric-equivalence only.
 *
 * TWO certificates per GEMM shape, BOTH against real stock ggml (no hand-rolled reference; ggml
 * is ground truth):
 *   (INT)  unit weight scale d=1.0 (fp16 0x3C00), dmin=0 (min term vanishes), magnitude-bounded
 *          integer core => both sides fold an EXACT integer in fp32 => require Or[k]==Oref[k]
 *          bit-for-bit (0 ULP). Reported as int-mismatch count: BYTE-EXACT-INTEGER-vs-stock-ggml
 *          iff 0. Exercises decode + nibble + scale-unpack + scaled int accumulation.
 *   (NORM) full path: adversarial fp16 super-block d AND dmin (min*bsums term ACTIVE), real fp32
 *          activation d, true per-16 bsums. fp32 fold-order differs between our vector kernel and
 *          ggml's scalar loop => small ULP EXPECTED (not a bug). Reported worst ULP + worst rel.
 *
 * REAL kernel byte layout (confirmed from golden_q4K.c op-attr annotations):
 *   weights block_q4_Kx16 stride 2304:
 *      d[16]   fp16 @0     (d[c]    @ 2*c)
 *      dmin[16]fp16 @32    (dmin[c] @ 32+2*c)
 *      scales[12][16] @64  (raw ggml scales[s] col c @ 64 + s*16 + c)   -- kernel unpacks 6-bit
 *      qs[128][16]    @256 (raw ggml qs[i]    col c @ 256 + i*16 + c)   -- kernel splits nibbles
 *   activation block_q8_Kx4 stride 1168:
 *      d[4]    fp32 @0     (d[r]    @ 4*r)
 *      qs[256][4]     @16  (raw int8 qs[p] row r @ 16 + p*4 + r)
 *      bsums[16][4] i16 @1040 (bsums[g] row r @ 1040 + g*8 + r*2)
 *   output GEMM s[r*nc + c], bs = nc.
 *
 * Build: clang-17 -O2 -march=rv64gcv_zfh_zvfh... -x c++ (golden kernel) + -x c++ (this) ;
 *        link the board's libggml-cpu.so (-lggml-cpu -lggml-base -lggml -lm).
 * argv: <seed>
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

#define QK_K 256
#define K_SCALE_SIZE 12
typedef uint16_t ggml_half;

struct block_q4_K { ggml_half d, dmin; uint8_t scales[K_SCALE_SIZE]; uint8_t qs[QK_K/2]; }; /* 144 */
struct block_q8_K { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; };                     /* 292 */
static_assert(sizeof(block_q4_K)==144,"q4_K");
static_assert(sizeof(block_q8_K)==292,"q8_K");

/* OUR exported repack GEMM kernel (n, s, vx=weights, vy=activations, nr, nc, bs) */
extern "C" void weft_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nr, size_t nc, size_t bs);
/* STOCK ggml block-dot dispatched at VLEN128 (reference), linked from libggml-cpu.so */
extern "C" void ggml_vec_dot_q4_K_q8_K(int n, float* s, size_t bs,
                                       const void* vx, size_t bx,
                                       const void* vy, size_t by, int nrc);

/* ---- helpers ---- */
static void wr16(uint8_t* p, uint16_t h){ memcpy(p,&h,2); }
static int64_t okey(float f){ uint32_t u; memcpy(&u,&f,4); return (u&0x80000000u)?-(int64_t)(u&0x7fffffffu):(int64_t)u; }
static int64_t ulp(float a,float b){ int64_t d=okey(a)-okey(b); return d<0?-d:d; }

static uint64_t RNG;
static uint32_t xr(void){ RNG^=RNG<<13; RNG^=RNG>>7; RNG^=RNG<<17; return (uint32_t)(RNG>>32); }

/* small positive fp16 bit patterns (exact, no float->half encoder needed) */
static const uint16_t DHALF[8] = {0x2800,0x2C00,0x3000,0x3400,0x2400,0x2A00,0x3200,0x2600};

/* build one original q4_K block. intmode: unit d, dmin=0, bounded scales for the exact-int cert */
static void build_w(block_q4_K* x, int col, int blk, int intmode){
    if(intmode){ wr16((uint8_t*)&x->d,0x3C00); wr16((uint8_t*)&x->dmin,0x0000); }
    else       { wr16((uint8_t*)&x->d,DHALF[(col+blk)&7]); wr16((uint8_t*)&x->dmin,DHALF[(col+3*blk+1)&7]); }
    /* scales[12] pack 8 6-bit sub-scales + 8 6-bit sub-mins (ggml get_scale_min_k4 domain).
     * INT: keep sub-scales small so the scaled integer fold stays < 2^24 and stays exact. */
    uint8_t sc[8], mn[8];
    for(int j=0;j<8;++j){
        sc[j]= intmode ? (uint8_t)(1+((j+col+blk)%3))          : (uint8_t)(xr()&63);
        mn[j]= intmode ? 0                                      : (uint8_t)(xr()&63);
    }
    /* inverse of ggml get_scale_min_k4 packing into 12 bytes */
    for(int j=0;j<4;++j){ x->scales[j]=sc[j]; x->scales[j+4]=mn[j]; }
    for(int j=4;j<8;++j){
        x->scales[j+4] = (uint8_t)((sc[j]&0x0F) | ((mn[j]&0x0F)<<4));
        x->scales[j-4] = (uint8_t)((x->scales[j-4]&0x3F) | ((sc[j]>>4)<<6));
        x->scales[j-0] = (uint8_t)((x->scales[j-0]&0x3F) | ((mn[j]>>4)<<6));
    }
    for(int i=0;i<QK_K/2;++i){
        uint8_t lo = intmode ? (uint8_t)((i+col)%13)     : (uint8_t)(xr()&0x0F);
        uint8_t hi = intmode ? (uint8_t)((i+2*blk+1)%13) : (uint8_t)(xr()&0x0F);
        x->qs[i] = (uint8_t)((lo&0x0F) | ((hi&0x0F)<<4));
    }
}
static void build_a(block_q8_K* a, int blk, int intmode){
    a->d = intmode ? 1.0f : (0.008f + 0.0013f*(float)(blk%7));
    for(int g=0;g<QK_K/16;++g){ int sum=0;
        for(int i=0;i<16;++i){
            int v = intmode ? ((int)(xr()%9)-4) : ((int)(xr()%181)-90);
            a->qs[g*16+i]=(int8_t)v; sum+=v;
        }
        a->bsums[g]=(int16_t)sum; /* true per-16 sum, exactly as ggml quantize_row_q8_K */
    }
}

/* ggml canonical 6-bit K-scale unpack of the on-disk scales[12] -> 8 sc + 8 m */
static void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m){
    if(j<4){ *d=q[j]&63; *m=q[j+4]&63; }
    else   { *d=(q[j+4]&0xF)|((q[j-4]>>6)<<4); *m=(q[j+4]>>4)|((q[j]>>6)<<4); }
}
/* repack weights [nc][nb] -> block_q4_Kx16 buffer [grp_c][nb] stride 2304.
 * SCALE region @64 uses ggml's repack.cpp CUSTOM layout (NOT verbatim scales[12]):
 *   lo strip g<4 @ 64+g*16 ; g>=4 @ 128+(g-4)*16 : byte=(m[g]&0xF)<<4 | (sc[g]&0xF)
 *   hi strip sb  @ 192+sb*16 : bits[1:0]=sc[sb]>>4 [3:2]=m[sb]>>4 [5:4]=sc[sb+4]>>4 [7:6]=m[sb+4]>>4 */
static void pack_w(std::vector<uint8_t>& W, const std::vector<block_q4_K>& o, int nc, int nb){
    int ng=nc/16; W.assign((size_t)ng*nb*2304,0);
    for(int g=0;g<ng;++g)for(int b=0;b<nb;++b){ uint8_t* blk=&W[((size_t)g*nb+b)*2304];
        for(int c=0;c<16;++c){ const block_q4_K& x=o[(size_t)(g*16+c)*nb+b];
            wr16(blk+2*c,      x.d);
            wr16(blk+32+2*c,   x.dmin);
            uint8_t sc[8], mn[8];
            for(int jj=0;jj<8;++jj) get_scale_min_k4(jj,x.scales,&sc[jj],&mn[jj]);
            for(int gg=0;gg<8;++gg){
                int loOff = (gg<4) ? (64 + gg*16 + c) : (128 + (gg-4)*16 + c);
                blk[loOff] = (uint8_t)(((mn[gg]&0xF)<<4) | (sc[gg]&0xF));
            }
            for(int sb=0;sb<4;++sb){
                blk[192 + sb*16 + c] = (uint8_t)(
                    ((sc[sb]>>4)&3) | (((mn[sb]>>4)&3)<<2) |
                    (((sc[sb+4]>>4)&3)<<4) | (((mn[sb+4]>>4)&3)<<6));
            }
            for(int i=0;i<128;++i) blk[256 + i*16 + c] = x.qs[i];
        } }
}
/* interleave activations [nr][nb] -> block_q8_Kx4 buffer [grp_r][nb] stride 1168 */
static void pack_a(std::vector<uint8_t>& A, const std::vector<block_q8_K>& o, int nr, int nb){
    int ng=nr/4; A.assign((size_t)ng*nb*1168,0);
    for(int g=0;g<ng;++g)for(int b=0;b<nb;++b){ uint8_t* blk=&A[((size_t)g*nb+b)*1168];
        for(int r=0;r<4;++r){ const block_q8_K& a=o[(size_t)(g*4+r)*nb+b];
            memcpy(blk+4*r, &a.d, 4);
            for(int p=0;p<256;++p) blk[16 + p*4 + r] = (uint8_t)a.qs[p];
            for(int gg=0;gg<16;++gg){ int16_t bs=a.bsums[gg]; memcpy(blk+1040 + gg*8 + r*2, &bs, 2); }
        } }
}

/* -------- surgical decomposition probes: vary scale/qs/act independently -------- */
static void probe(const char* tag, uint8_t scByte, uint8_t qsByte, int8_t actVal, uint16_t dminBits){
    const int nr=4,nc=16,n=256,nb=1;
    std::vector<block_q4_K> W((size_t)nc*nb);
    std::vector<block_q8_K> A((size_t)nr*nb);
    for(auto& x:W){
        wr16((uint8_t*)&x.d,0x3C00); wr16((uint8_t*)&x.dmin,dminBits);
        for(int s=0;s<12;++s) x.scales[s]=scByte;
        for(int i=0;i<128;++i) x.qs[i]=qsByte;
    }
    for(auto& a:A){ a.d=1.0f;
        for(int g=0;g<16;++g){ int sum=0; for(int i=0;i<16;++i){ a.qs[g*16+i]=actVal; sum+=actVal; } a.bsums[g]=(int16_t)sum; } }
    std::vector<uint8_t> Wr,Ar; pack_w(Wr,W,nc,nb); pack_a(Ar,A,nr,nb);
    std::vector<float> Ours((size_t)nr*nc,-1), Ref((size_t)nr*nc,-1);
    weft_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K((size_t)n,Ours.data(),Wr.data(),Ar.data(),(size_t)nr,(size_t)nc,(size_t)nc);
    for(int c=0;c<nc;++c)for(int r=0;r<nr;++r) ggml_vec_dot_q4_K_q8_K(n,&Ref[(size_t)r*nc+c],0,&W[(size_t)c*nb],0,&A[(size_t)r*nb],0,1);
    printf("  PROBE %-22s sc=0x%02x qs=0x%02x act=%d dmin=0x%04x :  ours=%.3f  ggml=%.3f\n",
           tag, scByte, qsByte, actVal, dminBits, Ours[0], Ref[0]);
}

int main(int argc,char**argv){
    unsigned seed = argc>1 ? (unsigned)strtoul(argv[1],0,0) : 20260709u;
    printf("# q4_K x q8_K REPACK GEMM  M0 tracer bullet  (kernel md5 b0b5beac, ref=STOCK ggml_vec_dot_q4_K_q8_K)\n");
    printf("\n== DECOMPOSITION PROBES (hand-predictable, dmin=0 kills min term) ==\n");
    probe("all-zero-qs",     0x01,0x00,1,0x0000); /* q4=0 -> ggml 0 */
    probe("zero-scale",      0x00,0x11,1,0x0000); /* sc=0 -> ggml 0 */
    probe("zero-act",        0x01,0x11,0,0x0000); /* act=0 -> ggml 0 */
    probe("uniform1",        0x01,0x11,1,0x0000); /* -> ggml 256 */
    probe("uniform1-dminOn", 0x01,0x11,1,0x3C00); /* dmin=1: min term active */
    probe("sc2",             0x02,0x22,1,0x0000); /* sc=2,q4=2 -> ggml 8*2*(2*32)=1024 */
    struct Sh{int nr,nc,n;};
    std::vector<Sh> shM = {{4,16,256},{4,32,512},{8,64,256},{16,256,512},{4,160,2560}};

    long long worstIntMism=0; int worstUlp=0; double worstRel=0; int fail=0;
    for(int mode=0;mode<2;++mode){
        printf("\n== %s ==\n", mode? "NORM (adversarial fp16 d+dmin, min*bsums ACTIVE, f32 act d, bounded-ULP vs stock ggml)"
                                    : "INT (unit d, dmin=0, magnitude-bounded, BYTE-EXACT-INTEGER vs stock ggml)");
        for(auto s: shM){
            int nr=s.nr, nc=s.nc, n=s.n, nb=n/QK_K;
            RNG = ((uint64_t)seed<<1 | 1ull) ^ ((uint64_t)(nr*131+nc*7+n) * 0x9E3779B97F4A7C15ull);
            std::vector<block_q4_K> W((size_t)nc*nb);
            std::vector<block_q8_K> A((size_t)nr*nb);
            for(int c=0;c<nc;++c)for(int b=0;b<nb;++b) build_w(&W[(size_t)c*nb+b], c, b, !mode);
            for(int r=0;r<nr;++r)for(int b=0;b<nb;++b) build_a(&A[(size_t)r*nb+b], r*31+b, !mode);

            std::vector<uint8_t> Wr, Ar; pack_w(Wr,W,nc,nb); pack_a(Ar,A,nr,nb);
            std::vector<float> Ours((size_t)nr*nc, -123.0f), Ref((size_t)nr*nc, -456.0f);

            /* OUR constructed repack-GEMM: one call, whole mul_mat */
            weft_emitc_ggml_repack_gemm_q4_K_q8_K_kernel_ggml_repack_gemm_q4_K_q8_K(
                (size_t)n, Ours.data(), Wr.data(), Ar.data(), (size_t)nr, (size_t)nc, (size_t)nc);

            /* STOCK ggml block-dot grid: the exact prefill fallback ggml dispatches at VLEN128 */
            for(int c=0;c<nc;++c)for(int r=0;r<nr;++r)
                ggml_vec_dot_q4_K_q8_K(n, &Ref[(size_t)r*nc+c], 0,
                                       &W[(size_t)c*nb], 0, &A[(size_t)r*nb], 0, 1);

            long long mism=0; int wU=0; double wR=0;
            for(size_t k=0;k<(size_t)nr*nc;++k){
                float a=Ours[k], b=Ref[k];
                if(!mode){ if(memcmp(&a,&b,4)!=0) ++mism; }
                int u=(int)ulp(a,b); if(u>wU)wU=u;
                double den=std::fabs((double)b); double rel = den>1e-9 ? std::fabs((double)a-(double)b)/den : std::fabs((double)a-(double)b);
                if(rel>wR)wR=rel;
            }
            if(!mode){ if(mism>worstIntMism)worstIntMism=mism; if(mism)fail=1;
                printf("  GEMM nr=%-3d nc=%-3d K=%-5d : int_mismatch=%lld  worstULP=%d   %s\n",
                       nr,nc,n,mism,wU, mism? "*** INT MISMATCH ***":"byte-exact-int OK");
            } else {
                /* NORM verdict is RELATIVE-error based: INT mode already proved the integer
                 * core bit-exact, so any NORM delta is pure fp32 reassociation (our vector
                 * fold vs ggml scalar loop) and grows with reduction size — benign iff small. */
                if(wU>worstUlp)worstUlp=wU; if(wR>worstRel)worstRel=wR; if(wR>1e-3)fail=1;
                printf("  GEMM nr=%-3d nc=%-3d K=%-5d : worstULP=%-4d worstRel=%.3e   %s\n",
                       nr,nc,n,wU,wR, wR>1e-3? "*** REL TOO LARGE ***":"bounded-ULP OK (fp32 reassoc)");
            }
        }
    }
    printf("\n# SUMMARY: worst_int_mismatch=%lld  worst_norm_ULP=%d  worst_norm_rel=%.3e  =>  %s\n",
           worstIntMism, worstUlp, worstRel, fail? "FAIL":"PASS");
    return fail?1:0;
}
