/* kquant_repack_verify_q6K.c -- SILICON numerical verifier for the compiler-emitted
 * q6_K x q8_K 16x1-REPACKED GEVM + GEMM kernels (HEAD 924dc31f, front-door lowered
 * tcrv-opt --tcrv-rvv-lower-to-emitc | mlir-translate). This UPGRADES the construction
 * oracle (which was a C++ emitter-MODEL vs reference) to a HARDWARE-RUN confirmation:
 * the ACTUAL exported .kernel.c is compiled by board clang-17 and executed on rvv silicon,
 * its fp32 output compared to an INDEPENDENT reference decoded from the ORIGINAL (pre-repack)
 * per-block q6_K exactly as ggml's canonical dequantize_row_q6_K / vec_dot_q6_K_q8_K.
 *
 * TWO certificates per shape, for BOTH the GEVM (plain q8_K activation) and the GEMM
 * (block_q8_Kx4 interleaved activation) kernel:
 *   (INT)  unit fp scales (d_w=1.0 fp16, d_a=1.0 fp32) + magnitude-bounded integer core
 *          => the kernel's fp32 fold is an EXACT integer == the int64 reference isum.
 *          Reported as int-mismatch count (BYTE-EXACT-INTEGER on silicon iff 0).
 *   (NORM) full adversarial fp16 super-block d + fp32 activation d => f64 reference;
 *          reported as worst ULP(kernel_fp32, (float)f64ref) + worst relative norm
 *          (bounded-norm: fp32 FMA fold vs f64, small ULP is expected, NOT a bug).
 *
 * REAL kernel byte layout (from the conversion fixture op attrs), weights block_q6_Kx16
 * stride 3360:  d[16] fp16 @0 | scales[256] i8 @32 | qh[1024] @288 | ql[2048] @1312.
 *   interleave: scales[32 + s*16 + c] qh[288 + i*16 + c] ql[1312 + i*16 + c], col c 0..15.
 * activation GEVM plain block_q8_K stride 292:  d f32 @0 | qs @4 | bsums @260 (UNUSED q6).
 * activation GEMM block_q8_Kx4 stride 1168:     d[4] f32 @0 | qs @16 (p*4+c) | bsums @1040.
 * output GEMM s[r*nc + c] (bs=nc); output GEVM s[c] (nr=1).
 *
 * Build: clang-17 -O2 -march=rv64gcv_zfh_zvfh... -x c++ (kernels) + -x c (this) ; link -lm.
 * argv: <seed>
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>

#define QK_K 256

/* ---- original per-block q6_K, 210 B ---- */
struct block_q6_K { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; float d; };
/* ---- plain block_q8_K, 292 B ---- */
struct block_q8_K { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; };

/* ---- exported kernels (extern "C" symbols from the .kernel.c) ---- */
extern "C" void tcrv_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nr, size_t nc, size_t bs);
extern "C" void tcrv_emitc_ggml_repack_gemv_q6_K_q8_K_kernel_ggml_repack_gemv_q6_K_q8_K(
    size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc);

/* ---- fp16 helpers ---- */
static void wr16(uint8_t* p, uint16_t h){ memcpy(p, &h, 2); }
static double h2d(uint16_t h){ int s=(h>>15)&1,e=(h>>10)&0x1F,m=h&0x3FF; double sg=s?-1:1;
    if(e==0)return sg*ldexp((double)m,-24); if(e==0x1F)return m?(double)NAN:sg*(double)INFINITY;
    return sg*ldexp((double)(m|0x400),e-25); }
static int64_t okey(float f){ uint32_t u; memcpy(&u,&f,4); return (u&0x80000000u)?-(int64_t)(u&0x7fffffffu):(int64_t)u; }
static int64_t ulp(float a,float b){ int64_t d=okey(a)-okey(b); return d<0?-d:d; }

/* ---- REFERENCE integer isum (scale-weighted) for ONE original q6_K block . q8_K block,
 * ggml canonical decode (6-bit = (ql&0xF | ((qh>>sh)&3)<<4) - 32, per-16 signed scale). ---- */
static int64_t ref_isum_block(const block_q6_K* x, const block_q8_K* a){
    int64_t isum=0;
    for(int n=0;n<QK_K;n+=128){
        const uint8_t* ql=x->ql+(n/128)*64; const uint8_t* qh=x->qh+(n/128)*32;
        const int sb=(n/128)*8;
        for(int l=0;l<32;++l){ int is=l/16;
            int q1=((ql[l+ 0]&0xF)|(((qh[l]>>0)&3)<<4))-32;
            int q2=((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4))-32;
            int q3=((ql[l+ 0]>> 4)|(((qh[l]>>4)&3)<<4))-32;
            int q4=((ql[l+32]>> 4)|(((qh[l]>>6)&3)<<4))-32;
            int base=n+l;
            isum+=(int64_t)x->scales[sb+is+0]*q1*a->qs[base+ 0];
            isum+=(int64_t)x->scales[sb+is+2]*q2*a->qs[base+32];
            isum+=(int64_t)x->scales[sb+is+4]*q3*a->qs[base+64];
            isum+=(int64_t)x->scales[sb+is+6]*q4*a->qs[base+96];
        }
    }
    return isum;
}

static std::mt19937 rng;
/* build one original q6_K block; `intmode` bounds scale magnitude for the exact-int cert */
static void build_w(block_q6_K* x,int col,int blk,int intmode){
    std::uniform_int_distribution<int> byte(0,255);
    for(int i=0;i<16;++i){ int v; if(intmode){ v=((3+i+col+blk)%7)-3; if(v==0)v=1; }
        else { v=((7+5*i+2*col+3*blk)%63)-31; if(v==0)v=1; } x->scales[i]=(int8_t)v; }
    for(int i=0;i<QK_K/2;++i)x->ql[i]=(uint8_t)byte(rng);
    for(int i=0;i<QK_K/4;++i)x->qh[i]=(uint8_t)byte(rng);
    x->d=1.0f;
}
static void build_a(block_q8_K* a,int blk,int intmode){
    std::uniform_int_distribution<int> q8(intmode?-7:-90, intmode?7:90);
    a->d=1.0f;
    for(int g=0;g<QK_K/16;++g){ int dc=intmode?0:(((g*7+blk*3)%21)-10); int sum=0;
        for(int i=0;i<16;++i){ int v=q8(rng)+dc; if(v>127)v=127; if(v<-128)v=-128;
            a->qs[g*16+i]=(int8_t)v; sum+=v; } a->bsums[g]=(int16_t)sum; }
}
/* small positive fp16 bits (super-block d) for the NORM pass */
static uint16_t rand_dhalf(){ std::uniform_int_distribution<int> e(7,11),m(0,0x3FF);
    return (uint16_t)((e(rng)<<10)|m(rng)); }

/* pack weights [nc][nb] -> repacked byte buffer [grp_c][nb] stride 3360 (fp16 d) */
static void pack_w(std::vector<uint8_t>& W,const std::vector<block_q6_K>& o,
                   const std::vector<uint16_t>& dh,int nc,int nb){
    int ng=nc/16; W.assign((size_t)ng*nb*3360,0);
    for(int g=0;g<ng;++g)for(int b=0;b<nb;++b){ uint8_t* blk=&W[((size_t)g*nb+b)*3360];
        for(int c=0;c<16;++c){ const block_q6_K& x=o[(size_t)(g*16+c)*nb+b];
            wr16(blk+c*2, dh[(size_t)(g*16+c)*nb+b]);
            for(int s=0;s<16;++s) blk[32+s*16+c]=(uint8_t)x.scales[s];
            for(int i=0;i<64;++i) blk[288+i*16+c]=x.qh[i];
            for(int i=0;i<128;++i) blk[1312+i*16+c]=x.ql[i]; } }
}

int main(int argc,char**argv){
    rng.seed(argc>1?(unsigned)strtoul(argv[1],0,0):20260708u);
    long vlen=(long)0; /* filled by kernel side implicitly; printed by shell */
    printf("# q6_K x q8_K REPACK GEVM+GEMM SILICON verify (kernel emitted @924dc31f)\n");
    struct Sh{int nr,nc,n;};
    std::vector<Sh> shG={{1,16,256},{1,32,512},{1,256,256},{1,160,2560}};  /* GEVM nr=1 */
    std::vector<Sh> shM={{4,32,512},{8,64,256},{16,256,256},{4,160,2560}}; /* GEMM nr%4 */
    long long anyInt=0; double worstNorm=0; int worstUlp=0;

    for(int mode=0;mode<2;++mode){ /* 0=INT exact, 1=NORM */
        printf("\n== %s ==\n", mode?"NORM (adversarial fp16 d, f64 ref, bounded-ULP)":"INT (unit d, magnitude-bounded, byte-exact-integer)");
        /* ---------- GEVM ---------- */
        for(auto s:shG){ int nc=s.nc,n=s.n,nb=n/QK_K,ng=nc/16;
            std::vector<block_q6_K> o((size_t)nc*nb);
            std::vector<uint16_t> dh((size_t)nc*nb);
            for(int g=0;g<ng;++g)for(int c=0;c<16;++c)for(int b=0;b<nb;++b){
                build_w(&o[(size_t)(g*16+c)*nb+b],g*16+c,b,!mode);
                dh[(size_t)(g*16+c)*nb+b]= mode? rand_dhalf():0x3C00; }
            std::vector<block_q8_K> act(nb);
            for(int b=0;b<nb;++b){ build_a(&act[b],b,!mode); if(mode) act[b].d=0.01f+0.002f*b; }
            std::vector<uint8_t> W; pack_w(W,o,dh,nc,nb);
            /* GEVM activation = plain block_q8_K[nb] (stride 292, qs@4) */
            std::vector<uint8_t> Y((size_t)nb*292,0);
            for(int b=0;b<nb;++b){ uint8_t* blk=&Y[(size_t)b*292];
                memcpy(blk,&act[b].d,4);
                for(int p=0;p<QK_K;++p) blk[4+p]=(uint8_t)act[b].qs[p]; }
            std::vector<float> out(nc,9.0f);
            tcrv_emitc_ggml_repack_gemv_q6_K_q8_K_kernel_ggml_repack_gemv_q6_K_q8_K(
                (size_t)n,out.data(),W.data(),Y.data(),(size_t)nc);
            long long mism=0; int wU=0; double wN=0,mAE=0,ssR=0; long cntN=0;
            for(int c=0;c<nc;++c){ double ref=0; int64_t refi=0;
                for(int b=0;b<nb;++b){ int64_t ii=ref_isum_block(&o[(size_t)c*nb+b],&act[b]);
                    refi+=ii; ref += h2d(dh[(size_t)c*nb+b])*(double)act[b].d*(double)ii; }
                float ko=out[c];
                if(!mode){ if((double)ko!=(double)refi) mism++; }
                else { int u=(int)ulp(ko,(float)ref); if(u>wU)wU=u;
                    double e=fabs((double)ko-ref); if(e>mAE)mAE=e; ssR+=ref*ref; cntN++; } }
            if(!mode){ anyInt+=mism; printf("  GEVM nr=1  nc=%-4d n=%-5d int-mismatch=%-4lld %s\n",
                nc,n,mism,mism?"INT-BUG":"BYTE-EXACT-INT"); }
            else{ double nrm=(ssR>0&&cntN>0)?mAE/sqrt(ssR/cntN):mAE; if(nrm>worstNorm)worstNorm=nrm; if(wU>worstUlp)worstUlp=wU;
                printf("  GEVM nr=1  nc=%-4d n=%-5d worst_ulp=%-6d norm=%.3e (maxAbsErr/rms)\n",nc,n,wU,nrm); }
        }
        /* ---------- GEMM ---------- */
        for(auto s:shM){ int nr=s.nr,nc=s.nc,n=s.n,nb=n/QK_K,ng=nc/16,gr=nr/4;
            std::vector<block_q6_K> o((size_t)nc*nb);
            std::vector<uint16_t> dh((size_t)nc*nb);
            for(int g=0;g<ng;++g)for(int c=0;c<16;++c)for(int b=0;b<nb;++b){
                build_w(&o[(size_t)(g*16+c)*nb+b],g*16+c,b,!mode);
                dh[(size_t)(g*16+c)*nb+b]= mode? rand_dhalf():0x3C00; }
            std::vector<block_q8_K> act((size_t)nr*nb);
            for(int r=0;r<nr;++r)for(int b=0;b<nb;++b){ build_a(&act[(size_t)r*nb+b],r*131+b,!mode);
                if(mode) act[(size_t)r*nb+b].d=0.01f+0.002f*((r+b)%9); }
            std::vector<uint8_t> W; pack_w(W,o,dh,nc,nb);
            /* GEMM activation = block_q8_Kx4[gr][nb] stride 1168: d[4]f32@0 qs@16(p*4+c) */
            std::vector<uint8_t> Y((size_t)gr*nb*1168,0);
            for(int g=0;g<gr;++g)for(int b=0;b<nb;++b){ uint8_t* blk=&Y[((size_t)g*nb+b)*1168];
                for(int c=0;c<4;++c){ float d=act[(size_t)(g*4+c)*nb+b].d; memcpy(blk+c*4,&d,4); }
                for(int p=0;p<QK_K;++p)for(int c=0;c<4;++c)
                    blk[16+p*4+c]=(uint8_t)act[(size_t)(g*4+c)*nb+b].qs[p]; }
            std::vector<float> out((size_t)nr*nc,9.0f);
            tcrv_emitc_ggml_repack_gemm_q6_K_q8_K_kernel_ggml_repack_gemm_q6_K_q8_K(
                (size_t)n,out.data(),W.data(),Y.data(),(size_t)nr,(size_t)nc,(size_t)nc);
            long long mism=0; int wU=0; double wN=0,mAE=0,ssR=0; long cntN=0;
            for(int r=0;r<nr;++r)for(int c=0;c<nc;++c){ double ref=0; int64_t refi=0;
                for(int b=0;b<nb;++b){ int64_t ii=ref_isum_block(&o[(size_t)c*nb+b],&act[(size_t)r*nb+b]);
                    refi+=ii; ref += h2d(dh[(size_t)c*nb+b])*(double)act[(size_t)r*nb+b].d*(double)ii; }
                float ko=out[(size_t)r*nc+c];
                if(!mode){ if((double)ko!=(double)refi) mism++; }
                else { int u=(int)ulp(ko,(float)ref); if(u>wU)wU=u;
                    double e=fabs((double)ko-ref); if(e>mAE)mAE=e; ssR+=ref*ref; cntN++; } }
            if(!mode){ anyInt+=mism; printf("  GEMM nr=%-2d nc=%-4d n=%-5d int-mismatch=%-4lld %s\n",
                nr,nc,n,mism,mism?"INT-BUG":"BYTE-EXACT-INT"); }
            else{ double nrm=(ssR>0&&cntN>0)?mAE/sqrt(ssR/cntN):mAE; if(nrm>worstNorm)worstNorm=nrm; if(wU>worstUlp)worstUlp=wU;
                printf("  GEMM nr=%-2d nc=%-4d n=%-5d worst_ulp=%-6d norm=%.3e (maxAbsErr/rms)\n",nr,nc,n,wU,nrm); }
        }
    }
    (void)vlen;
    printf("\nVERDICT q6_K: INT_mismatch_total=%lld  NORM worst_ulp=%d worst_norm=%.3e  => %s\n",
        anyInt,worstUlp,worstNorm,
        (anyInt==0 && worstNorm<1e-2)?"SILICON BYTE-EXACT-INTEGER + BOUNDED-NORM":"CHECK");
    return anyInt?1:0;
}
