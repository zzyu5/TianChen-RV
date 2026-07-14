/* kquant_repack_verify_q2K.c -- SILICON numerical verifier for the compiler-emitted
 * q2_K x q8_K 16x1-REPACKED GEVM + GEMM kernels (HEAD 924dc31f). q2_K is the MIN-TERM
 * format: 2-bit UNSIGNED weight, 4-bit packed scale/min per sub-block, DUAL super-block
 * d/dmin, MIN folded via the activation bsums. Upgrades the construction oracle (C++
 * emitter-MODEL) to a HARDWARE-RUN confirmation vs a reference decoded from the ORIGINAL
 * per-block q2_K exactly as ggml's canonical vec_dot_q2_K_q8_K.
 *
 * REAL weight byte layout block_q2_Kx16 stride 1344:
 *   d[16] fp16 @0 | dmin[16] fp16 @32 | scales[256] @64 (packed byte s*16+c) | qs[1024] @320 (i*16+c,i0..63).
 * activation GEVM plain q8_K 292: d f32@0 | qs@4 | bsums@260 (i16[16]).
 * activation GEMM block_q8_Kx4 1168: d[4]@0 | qs@16 (p*4+c) | bsums@1040 ((g16*4+c) i16).
 * fold: d*a->d*isum - dmin*a->d*summs.  output GEMM s[r*nc+c]; GEVM s[c].  argv: <seed>
 */
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#define QK_K 256
struct block_q2_K { uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; float d; float dmin; };
struct block_q8_K { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; };

extern "C" void weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K(
    size_t n,float*s,const uint8_t*vx,const uint8_t*vy,size_t nr,size_t nc,size_t bs);
extern "C" void weft_emitc_ggml_repack_gemv_q2_K_q8_K_kernel_ggml_repack_gemv_q2_K_q8_K(
    size_t n,float*s,const uint8_t*vx,const uint8_t*vy,size_t nc);

static void wr16(uint8_t*p,uint16_t h){ memcpy(p,&h,2);}
static double h2d(uint16_t h){ int s=(h>>15)&1,e=(h>>10)&0x1F,m=h&0x3FF;double sg=s?-1:1;
    if(e==0)return sg*ldexp((double)m,-24); if(e==0x1F)return m?(double)NAN:sg*(double)INFINITY;
    return sg*ldexp((double)(m|0x400),e-25);}
static int64_t okey(float f){uint32_t u;memcpy(&u,&f,4);return (u&0x80000000u)?-(int64_t)(u&0x7fffffffu):(int64_t)u;}
static int64_t ulp(float a,float b){int64_t d=okey(a)-okey(b);return d<0?-d:d;}

/* REFERENCE (ggml canonical vec_dot_q2_K_q8_K): isum(scale main) + summs(min) integers */
static void ref_block(const block_q2_K* x,const block_q8_K* a,int64_t& isum,int64_t& summs){
    int scale[16],mins[16];
    for(int s=0;s<16;++s){ scale[s]=x->scales[s]&0x0F; mins[s]=(x->scales[s]>>4)&0x0F; }
    isum=0; int is=0;
    for(int k=0;k<QK_K/128;++k){ const uint8_t* q2=x->qs+k*32; const int8_t* q8=a->qs+k*128; int shift=0;
        for(int j=0;j<4;++j){ int d0=scale[is]; int sl=0;
            for(int l=0;l<16;++l) sl+=q8[j*32+l]*((q2[l]>>shift)&3); isum+=(int64_t)d0*sl;
            int d1=scale[is+1]; sl=0;
            for(int l=16;l<32;++l) sl+=q8[j*32+l]*((q2[l]>>shift)&3); isum+=(int64_t)d1*sl;
            is+=2; shift+=2; } }
    summs=0; for(int s=0;s<16;++s) summs+=(int64_t)a->bsums[s]*mins[s];
}
static std::mt19937 rng;
static void build_w(block_q2_K* x){ std::uniform_int_distribution<int> byte(0,255);
    for(int i=0;i<QK_K/16;++i)x->scales[i]=(uint8_t)byte(rng);
    for(int i=0;i<QK_K/4;++i)x->qs[i]=(uint8_t)byte(rng); x->d=1.0f; x->dmin=1.0f; }
static void build_a(block_q8_K* a,int blk,int intmode){
    std::uniform_int_distribution<int> q8(intmode?-3:-90,intmode?3:90); a->d=1.0f;
    for(int g=0;g<QK_K/16;++g){ int dc=intmode?0:(((g*7+blk*3)%21)-10); int sum=0;
        for(int i=0;i<16;++i){ int v=q8(rng)+dc; if(v>127)v=127; if(v<-128)v=-128;
            a->qs[g*16+i]=(int8_t)v; sum+=v; } a->bsums[g]=(int16_t)sum; } }
static uint16_t rand_dhalf(){ std::uniform_int_distribution<int> e(7,11),m(0,0x3FF);
    return (uint16_t)((e(rng)<<10)|m(rng)); }
static void pack_w(std::vector<uint8_t>& W,const std::vector<block_q2_K>& o,
                   const std::vector<uint16_t>& dh,const std::vector<uint16_t>& dmh,int nc,int nb){
    int ng=nc/16; W.assign((size_t)ng*nb*1344,0);
    for(int g=0;g<ng;++g)for(int b=0;b<nb;++b){ uint8_t* blk=&W[((size_t)g*nb+b)*1344];
        for(int c=0;c<16;++c){ const block_q2_K& x=o[(size_t)(g*16+c)*nb+b];
            wr16(blk+c*2,dh[(size_t)(g*16+c)*nb+b]); wr16(blk+32+c*2,dmh[(size_t)(g*16+c)*nb+b]);
            for(int s=0;s<16;++s) blk[64+s*16+c]=x.scales[s];
            for(int i=0;i<64;++i) blk[320+i*16+c]=x.qs[i]; } } }

int main(int argc,char**argv){
    rng.seed(argc>1?(unsigned)strtoul(argv[1],0,0):20260708u);
    printf("# q2_K x q8_K REPACK GEVM+GEMM SILICON verify (kernel emitted @924dc31f)\n");
    struct Sh{int nr,nc,n;};
    std::vector<Sh> shG={{1,16,256},{1,32,512},{1,256,256},{1,160,2560}};
    std::vector<Sh> shM={{4,32,512},{8,64,256},{16,256,256},{4,160,2560}};
    long long anyInt=0; double worstNorm=0; int worstUlp=0;
    for(int mode=0;mode<2;++mode){
        printf("\n== %s ==\n",mode?"NORM (adversarial fp16 d/dmin, f64 ref, bounded-ULP)":"INT (unit d/dmin, magnitude-bounded, byte-exact-integer)");
        for(auto s:shG){ int nc=s.nc,n=s.n,nb=n/QK_K,ng=nc/16;
            std::vector<block_q2_K> o((size_t)nc*nb);
            std::vector<uint16_t> dh((size_t)nc*nb),dmh((size_t)nc*nb);
            for(int g=0;g<ng;++g)for(int c=0;c<16;++c)for(int b=0;b<nb;++b){
                build_w(&o[(size_t)(g*16+c)*nb+b]);
                dh[(size_t)(g*16+c)*nb+b]=mode?rand_dhalf():0x3C00;
                dmh[(size_t)(g*16+c)*nb+b]=mode?rand_dhalf():0x3C00; }
            std::vector<block_q8_K> act(nb);
            for(int b=0;b<nb;++b){ build_a(&act[b],b,!mode); if(mode)act[b].d=0.01f+0.002f*b; }
            std::vector<uint8_t> W; pack_w(W,o,dh,dmh,nc,nb);
            std::vector<uint8_t> Y((size_t)nb*292,0);
            for(int b=0;b<nb;++b){ uint8_t* blk=&Y[(size_t)b*292]; memcpy(blk,&act[b].d,4);
                for(int p=0;p<QK_K;++p) blk[4+p]=(uint8_t)act[b].qs[p];
                for(int s=0;s<16;++s){ int16_t v=act[b].bsums[s]; memcpy(blk+260+s*2,&v,2); } }
            std::vector<float> out(nc,9.0f);
            weft_emitc_ggml_repack_gemv_q2_K_q8_K_kernel_ggml_repack_gemv_q2_K_q8_K(
                (size_t)n,out.data(),W.data(),Y.data(),(size_t)nc);
            long long mism=0; int wU=0; double wN=0,mAE=0,ssR=0; long cntN=0;
            for(int c=0;c<nc;++c){ double ref=0; int64_t refi=0;
                for(int b=0;b<nb;++b){ int64_t ii,ss; ref_block(&o[(size_t)c*nb+b],&act[b],ii,ss);
                    refi+=ii-ss; ref+=h2d(dh[(size_t)c*nb+b])*(double)act[b].d*(double)ii
                                     -h2d(dmh[(size_t)c*nb+b])*(double)act[b].d*(double)ss; }
                float ko=out[c];
                if(!mode){ if((double)ko!=(double)refi)mism++; }
                else{ int u=(int)ulp(ko,(float)ref); if(u>wU)wU=u;
                    double e=fabs((double)ko-ref); if(e>mAE)mAE=e; ssR+=ref*ref; cntN++; } }
            if(!mode){ anyInt+=mism; printf("  GEVM nr=1  nc=%-4d n=%-5d int-mismatch=%-4lld %s\n",nc,n,mism,mism?"INT-BUG":"BYTE-EXACT-INT"); }
            else{ double nrm=(ssR>0&&cntN>0)?mAE/sqrt(ssR/cntN):mAE; if(nrm>worstNorm)worstNorm=nrm; if(wU>worstUlp)worstUlp=wU;
                printf("  GEVM nr=1  nc=%-4d n=%-5d worst_ulp=%-6d norm=%.3e (maxAbsErr/rms)\n",nc,n,wU,nrm); } }
        for(auto s:shM){ int nr=s.nr,nc=s.nc,n=s.n,nb=n/QK_K,ng=nc/16,gr=nr/4;
            std::vector<block_q2_K> o((size_t)nc*nb);
            std::vector<uint16_t> dh((size_t)nc*nb),dmh((size_t)nc*nb);
            for(int g=0;g<ng;++g)for(int c=0;c<16;++c)for(int b=0;b<nb;++b){
                build_w(&o[(size_t)(g*16+c)*nb+b]);
                dh[(size_t)(g*16+c)*nb+b]=mode?rand_dhalf():0x3C00;
                dmh[(size_t)(g*16+c)*nb+b]=mode?rand_dhalf():0x3C00; }
            std::vector<block_q8_K> act((size_t)nr*nb);
            for(int r=0;r<nr;++r)for(int b=0;b<nb;++b){ build_a(&act[(size_t)r*nb+b],r*131+b,!mode);
                if(mode)act[(size_t)r*nb+b].d=0.01f+0.002f*((r+b)%9); }
            std::vector<uint8_t> W; pack_w(W,o,dh,dmh,nc,nb);
            std::vector<uint8_t> Y((size_t)gr*nb*1168,0);
            for(int g=0;g<gr;++g)for(int b=0;b<nb;++b){ uint8_t* blk=&Y[((size_t)g*nb+b)*1168];
                for(int c=0;c<4;++c){ float d=act[(size_t)(g*4+c)*nb+b].d; memcpy(blk+c*4,&d,4); }
                for(int p=0;p<QK_K;++p)for(int c=0;c<4;++c)
                    blk[16+p*4+c]=(uint8_t)act[(size_t)(g*4+c)*nb+b].qs[p];
                for(int s=0;s<16;++s)for(int c=0;c<4;++c){ int16_t v=act[(size_t)(g*4+c)*nb+b].bsums[s];
                    memcpy(blk+1040+(s*4+c)*2,&v,2); } }
            std::vector<float> out((size_t)nr*nc,9.0f);
            weft_emitc_ggml_repack_gemm_q2_K_q8_K_kernel_ggml_repack_gemm_q2_K_q8_K(
                (size_t)n,out.data(),W.data(),Y.data(),(size_t)nr,(size_t)nc,(size_t)nc);
            long long mism=0; int wU=0; double wN=0,mAE=0,ssR=0; long cntN=0;
            for(int r=0;r<nr;++r)for(int c=0;c<nc;++c){ double ref=0; int64_t refi=0;
                for(int b=0;b<nb;++b){ int64_t ii,ss; ref_block(&o[(size_t)c*nb+b],&act[(size_t)r*nb+b],ii,ss);
                    refi+=ii-ss; ref+=h2d(dh[(size_t)c*nb+b])*(double)act[(size_t)r*nb+b].d*(double)ii
                                     -h2d(dmh[(size_t)c*nb+b])*(double)act[(size_t)r*nb+b].d*(double)ss; }
                float ko=out[(size_t)r*nc+c];
                if(!mode){ if((double)ko!=(double)refi)mism++; }
                else{ int u=(int)ulp(ko,(float)ref); if(u>wU)wU=u;
                    double e=fabs((double)ko-ref); if(e>mAE)mAE=e; ssR+=ref*ref; cntN++; } }
            if(!mode){ anyInt+=mism; printf("  GEMM nr=%-2d nc=%-4d n=%-5d int-mismatch=%-4lld %s\n",nr,nc,n,mism,mism?"INT-BUG":"BYTE-EXACT-INT"); }
            else{ double nrm=(ssR>0&&cntN>0)?mAE/sqrt(ssR/cntN):mAE; if(nrm>worstNorm)worstNorm=nrm; if(wU>worstUlp)worstUlp=wU;
                printf("  GEMM nr=%-2d nc=%-4d n=%-5d worst_ulp=%-6d norm=%.3e (maxAbsErr/rms)\n",nr,nc,n,wU,nrm); } }
    }
    printf("\nVERDICT q2_K: INT_mismatch_total=%lld  NORM worst_ulp=%d worst_norm=%.3e  => %s\n",
        anyInt,worstUlp,worstNorm,(anyInt==0&&worstNorm<1e-2)?"SILICON BYTE-EXACT-INTEGER + BOUNDED-NORM":"CHECK");
    return anyInt?1:0;
}
