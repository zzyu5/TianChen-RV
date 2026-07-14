/* g6b_q2k_identity_ab.c -- [G6-B / GAP-EMIT-UNROLL] direct byte-exact A/B identity gate.
 * Runs OUR q2_K repack GEMM in BOTH schedules (UNROLLED full static unroll vs ROLLED
 * compact runtime-loop main term) on the SAME random inputs and memcmp's the full
 * nr*nc fp32 output byte-for-byte. Packing is copied VERBATIM from
 * kquant_repack_verify_q2K.c (the same q2_Kx16 repack + q8_Kx4 interleave ABI). NORM
 * mode (random fp16 d/dmin) exercises the full fp16-scale/min fold path; INT mode the
 * integer path. Any mismatch => the rolled schedule is NOT byte-exact. Self-contained
 * (no ggml). argv: <seed>.
 */
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

#define QK_K 256
typedef struct { uint16_t d, dmin; uint8_t scales[QK_K/16]; uint8_t qs[QK_K/4]; } block_q2_K; /* d/dmin as fp16 bits */
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;

extern "C" void tcrv_emitc_q2K_gemm_UNROLLED(
    size_t n,float*s,const uint8_t*vx,const uint8_t*vy,size_t nr,size_t nc,size_t bs);
extern "C" void tcrv_emitc_q2K_gemm_ROLLED(
    size_t n,float*s,const uint8_t*vx,const uint8_t*vy,size_t nr,size_t nc,size_t bs);

static void wr16(uint8_t*p,uint16_t h){ memcpy(p,&h,2);}
static std::mt19937 rng;
static void build_w(block_q2_K* x){ std::uniform_int_distribution<int> byte(0,255);
    for(int i=0;i<QK_K/16;++i)x->scales[i]=(uint8_t)byte(rng);
    for(int i=0;i<QK_K/4;++i)x->qs[i]=(uint8_t)byte(rng); }
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
    printf("# G6-B q2_K GEMM A/B IDENTITY: UNROLLED vs ROLLED (byte-for-byte fp32 output)\n");
    struct Sh{int nr,nc,n;};
    std::vector<Sh> shM={{4,32,512},{8,64,256},{16,256,256},{4,160,2560}};
    long long total_bytes=0,mism_bytes=0,mism_floats=0;
    for(int mode=0;mode<2;++mode){
        printf("== %s ==\n",mode?"NORM (random fp16 d/dmin -- full fp fold path)":"INT (unit d/dmin)");
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
                for(int ss=0;ss<16;++ss)for(int c=0;c<4;++c){ int16_t v=act[(size_t)(g*4+c)*nb+b].bsums[ss];
                    memcpy(blk+1040+(ss*4+c)*2,&v,2); } }
            std::vector<float> out_u((size_t)nr*nc,9.0f), out_r((size_t)nr*nc,7.0f);
            tcrv_emitc_q2K_gemm_UNROLLED((size_t)n,out_u.data(),W.data(),Y.data(),(size_t)nr,(size_t)nc,(size_t)nc);
            tcrv_emitc_q2K_gemm_ROLLED  ((size_t)n,out_r.data(),W.data(),Y.data(),(size_t)nr,(size_t)nc,(size_t)nc);
            size_t nbytes=(size_t)nr*nc*sizeof(float);
            long long mb=0,mf=0;
            const uint8_t* pu=(const uint8_t*)out_u.data(); const uint8_t* pr=(const uint8_t*)out_r.data();
            for(size_t i=0;i<nbytes;++i) if(pu[i]!=pr[i]) mb++;
            for(size_t i=0;i<(size_t)nr*nc;++i) if(memcmp(&out_u[i],&out_r[i],4)!=0) mf++;
            total_bytes+=nbytes; mism_bytes+=mb; mism_floats+=mf;
            printf("  GEMM nr=%-2d nc=%-4d n=%-5d bytes=%-8zu  mism_bytes=%-6lld mism_floats=%-6lld %s\n",
                   nr,nc,n,nbytes,mb,mf,(mb==0)?"BYTE-IDENTICAL":"DIVERGENT");
        }
    }
    printf("\nVERDICT G6-B q2_K A/B: total_bytes=%lld  mism_bytes=%lld  mism_floats=%lld  => %s\n",
        total_bytes,mism_bytes,mism_floats,(mism_bytes==0)?"ROLLED == UNROLLED (BYTE-EXACT)":"DIVERGENT-BUG");
    return mism_bytes?1:0;
}
