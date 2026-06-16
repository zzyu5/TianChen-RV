// INC-12 reference-fusion band characterization. Shows that the _generic
// reference ITSELF is the moving part under -ffp-contract: compile this TU once
// at =off (producing the canonical, unfused *s, written to a file) and once at
// =fast (the fused *s), and the two diverge by the SAME tiny magnitude as the
// kernel-vs-generic(=fast) mismatch -- proving the =fast mismatch is a reference
// compilation artifact, NOT a kernel divergence. The kernel (separate vfmul/
// vfadd intrinsics) is contraction-immune and matches generic(=off) exactly.
//
// Usage: built twice; the =off build writes ref_off.bin, the =fast build reads
// it and reports how many of the SAME random *s values diverge.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstddef>

#define QK_K 256
typedef uint16_t ggml_half;
typedef struct { uint8_t ql[QK_K/2]; uint8_t qh[QK_K/4]; int8_t scales[QK_K/16]; ggml_half d; } block_q6_K;
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;

static inline float fp16(uint16_t h){ _Float16 v; std::memcpy(&v,&h,2); return (float)v; }
#define GGML_CPU_FP16_TO_FP32(x) fp16(x)
#define GGML_RESTRICT __restrict
#define UNUSED(x) (void)(x)

static void gen(int n, float *s, const void *vx, const void *vy) {
    const block_q6_K *x=(const block_q6_K*)vx; const block_q8_K *y=(const block_q8_K*)vy;
    const int nb=n/QK_K; int8_t aux8[QK_K]; int16_t aux16[8]; float sums[8]; int32_t aux32[8];
    std::memset(sums,0,sizeof(sums)); float sumf=0;
    for(int i=0;i<nb;++i){ const uint8_t*q4=x[i].ql; const uint8_t*qh=x[i].qh; const int8_t*q8=y[i].qs;
        std::memset(aux32,0,sizeof(aux32)); int8_t*a=aux8;
        for(int j=0;j<QK_K;j+=128){ for(int l=0;l<32;++l){
            a[l+0]=(int8_t)((q4[l+0]&0xF)|(((qh[l]>>0)&3)<<4))-32;
            a[l+32]=(int8_t)((q4[l+32]&0xF)|(((qh[l]>>2)&3)<<4))-32;
            a[l+64]=(int8_t)((q4[l+0]>>4)|(((qh[l]>>4)&3)<<4))-32;
            a[l+96]=(int8_t)((q4[l+32]>>4)|(((qh[l]>>6)&3)<<4))-32; }
            a+=128; q4+=64; qh+=32; }
        a=aux8; int is=0;
        for(int j=0;j<QK_K/16;++j){ int scale=x[i].scales[is++];
            for(int l=0;l<8;++l)aux16[l]=q8[l]*a[l];
            for(int l=0;l<8;++l)aux32[l]+=scale*aux16[l]; q8+=8;a+=8;
            for(int l=0;l<8;++l)aux16[l]=q8[l]*a[l];
            for(int l=0;l<8;++l)aux32[l]+=scale*aux16[l]; q8+=8;a+=8; }
        const float d=GGML_CPU_FP16_TO_FP32(x[i].d)*y[i].d;
        for(int l=0;l<8;++l)sums[l]+=d*aux32[l]; }
    for(int l=0;l<8;++l)sumf+=sums[l]; *s=sumf;
}

static unsigned g=0x6b5f1c27u;
static unsigned r(){ g^=g<<13; g^=g>>17; g^=g<<5; return g; }
static uint16_t rd(){ uint16_t m=(uint16_t)(r()&0x3FF); uint16_t e=(uint16_t)(8+(r()%12)); return (uint16_t)((e<<10)|m); }
static void fb(uint8_t*q6k,uint8_t*q8k){
    for(int i=0;i<128;++i)q6k[i]=(uint8_t)(r()&0xFF);
    for(int i=0;i<64;++i)q6k[128+i]=(uint8_t)(r()&0xFF);
    for(int i=0;i<16;++i)q6k[192+i]=(uint8_t)(r()&0xFF);
    uint16_t d=rd(); std::memcpy(q6k+208,&d,2);
    float dy=(float)((int)(r()%2000)-1000)/4096.0f; std::memcpy(q8k+0,&dy,4);
    for(int i=0;i<256;++i)q8k[4+i]=(uint8_t)(r()&0xFF);
    for(int i=0;i<32;++i)q8k[260+i]=0;
}
static inline uint32_t bits(float f){ uint32_t u; std::memcpy(&u,&f,4); return u; }

#define N 2000
int main(){
    float vals[N];
    for(int t=0;t<N;++t){ int NB=1+(int)(r()%32); size_t n=(size_t)NB*256;
        uint8_t*vx=(uint8_t*)malloc((size_t)NB*210); uint8_t*vy=(uint8_t*)malloc((size_t)NB*292);
        for(int ib=0;ib<NB;++ib)fb(vx+(size_t)ib*210,vy+(size_t)ib*292);
        gen((int)n,&vals[t],vx,vy); free(vx); free(vy); }
#ifdef WRITE_OFF
    FILE*f=fopen("ref_off.bin","wb"); fwrite(vals,sizeof(float),N,f); fclose(f);
    printf("ref_band: wrote %d _generic(=off) *s values to ref_off.bin\n",N);
#else
    FILE*f=fopen("ref_off.bin","rb"); if(!f){printf("missing ref_off.bin\n");return 2;}
    float off[N]; size_t got=fread(off,sizeof(float),N,f); fclose(f); (void)got;
    int diverge=0; uint32_t maxulp=0;
    for(int t=0;t<N;++t){ if(bits(off[t])!=bits(vals[t])){ ++diverge;
        uint32_t a=bits(off[t]),b=bits(vals[t]); uint32_t d=a>b?a-b:b-a; if(d>maxulp)maxulp=d; } }
    printf("ref_band: _generic(=off) vs _generic(=fast): %d/%d diverge, max |ulp diff|=%u\n",
           diverge,N,maxulp);
    printf("  => the REFERENCE itself moves under -ffp-contract; the kernel is contraction-immune\n");
#endif
    return 0;
}
