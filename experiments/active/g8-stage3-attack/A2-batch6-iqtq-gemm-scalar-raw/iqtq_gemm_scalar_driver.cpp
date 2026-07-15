// A2-batch6 iq/tq/fp4 GEMM prefill scalar-ref paired A/B driver.
//   OURS = weft repack-GEMM kernel (front-door export via weft-opt --weft-rvv-lower-to-emitc),
//          x16-interleaved weight + block_q8_Kx4 (or q8_0x4 for mxfp4) interleaved activation.
//   OPP  = ggml_vec_dot_<fmt>_q8_K_generic  (ggml's own SCALAR reference, from stock libggml-cpu.so),
//          called per (row,col) over PLAIN blocks == a SCALAR-REF GEMM.  This is the CONSTRUCTED
//          ggml scalar-ref opponent (same-op repack GEMM absent in ggml).  ==> CHEAP TIER by
//          construction: a big multiple is EXPECTED and is NOT a hard win (scalar opponent).
//   GATE = OURS output vs OPP-generic output (generic IS ggml's authoritative reference => ZERO-MODEL:
//          the reference is independent of OURS internals). integer decode is identical; only f32
//          cross-block reduction ORDER differs => a correct kernel agrees to ~1e-4 rel.
//   ratio_cold = opp_med / ours_med  (>=0.8 => PASS ; <0.8 => named-X).
// cold: 32MiB flush, CLOCK_MONOTONIC, median+relIQR, core-pinned, 2-seed.
// [NG-4] kernel-axis GEMM MICRO datapoint, NOT e2e, NOT perf-covered, NOT a hard-gate win.
//  argv: <fmt> <K> <nr(mult4)> <nc(mult16)> <reps> [verify_only=0] [seed]
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <time.h>

typedef uint16_t ggml_half;

// ---- fp16 <-> f32 (RNE, small finite range) ----
static uint16_t f32_to_f16(float f){ uint32_t x; memcpy(&x,&f,4);
    uint32_t sign=(x>>16)&0x8000u; int32_t exp=(int32_t)((x>>23)&0xff)-127+15; uint32_t man=x&0x7fffffu;
    if(exp<=0){ if(exp<-10) return (uint16_t)sign; man|=0x800000u; uint32_t sh=(uint32_t)(14-exp);
        uint16_t r=(uint16_t)(man>>sh); if((man>>(sh-1))&1) r++; return (uint16_t)(sign|r); }
    if(exp>=31) return (uint16_t)(sign|0x7c00u);
    uint16_t r=(uint16_t)(sign|((uint32_t)exp<<10)|(man>>13)); if((man>>12)&1) r++; return r; }

// ---- OURS exported repack-GEMM kernels (7-arg: n,s,vx,vy,nr,nc,bs) ----
extern "C" {
#ifndef DECODE_ONLY
void weft_emitc_ggml_repack_gemm_iq4_xs_q8_K_kernel_ggml_repack_gemm_iq4_xs_q8_K (size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
void weft_emitc_ggml_repack_gemm_iq2_xxs_q8_K_kernel_ggml_repack_gemm_iq2_xxs_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
void weft_emitc_ggml_repack_gemm_iq2_xs_q8_K_kernel_ggml_repack_gemm_iq2_xs_q8_K  (size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
void weft_emitc_ggml_repack_gemm_iq2_s_q8_K_kernel_ggml_repack_gemm_iq2_s_q8_K    (size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
void weft_emitc_ggml_repack_gemm_mxfp4_q8_0_kernel_ggml_repack_gemm_mxfp4_q8_0    (size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
void weft_emitc_ggml_repack_gemm_tq1_0_q8_K_kernel_ggml_repack_gemm_tq1_0_q8_K    (size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
void weft_emitc_ggml_repack_gemm_tq2_0_q8_K_kernel_ggml_repack_gemm_tq2_0_q8_K    (size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t);
#endif
// OURS decode/GEVM leaves (5-arg: n,s,vx,vy,nc)  weight=x16 (same repack), activation=PLAIN block_q8_K/q8_0 single vector
void weft_emitc_ggml_repack_gemv_iq4_xs_q8_K_kernel_ggml_repack_gemv_iq4_xs_q8_K (size_t,float*,const uint8_t*,const uint8_t*,size_t);
void weft_emitc_ggml_repack_gemv_iq2_xxs_q8_K_kernel_ggml_repack_gemv_iq2_xxs_q8_K(size_t,float*,const uint8_t*,const uint8_t*,size_t);
void weft_emitc_ggml_repack_gemv_iq2_xs_q8_K_kernel_ggml_repack_gemv_iq2_xs_q8_K  (size_t,float*,const uint8_t*,const uint8_t*,size_t);
void weft_emitc_ggml_repack_gemv_iq2_s_q8_K_kernel_ggml_repack_gemv_iq2_s_q8_K    (size_t,float*,const uint8_t*,const uint8_t*,size_t);
void weft_emitc_ggml_repack_gemv_mxfp4_q8_0_kernel_ggml_repack_gemv_mxfp4_q8_0    (size_t,float*,const uint8_t*,const uint8_t*,size_t);
void weft_emitc_ggml_repack_gemv_tq1_0_q8_K_kernel_ggml_repack_gemv_tq1_0_q8_K    (size_t,float*,const uint8_t*,const uint8_t*,size_t);
void weft_emitc_ggml_repack_gemv_tq2_0_q8_K_kernel_ggml_repack_gemv_tq2_0_q8_K    (size_t,float*,const uint8_t*,const uint8_t*,size_t);
// OPP scalar-ref (ggml generic)
void ggml_vec_dot_iq4_xs_q8_K_generic (int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_iq2_xxs_q8_K_generic(int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_iq2_xs_q8_K_generic (int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_iq2_s_q8_K_generic  (int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_mxfp4_q8_0_generic  (int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_tq1_0_q8_K_generic  (int,float*,size_t,const void*,size_t,const void*,size_t,int);
void ggml_vec_dot_tq2_0_q8_K_generic  (int,float*,size_t,const void*,size_t,const void*,size_t,int);
}

// ---- format table ----
enum Fmt { F_IQ4XS, F_IQ2XXS, F_IQ2XS, F_IQ2S, F_MXFP4, F_TQ1, F_TQ2, NFMT };
struct FInfo { const char* name; int qk; int wblk; int wgrp; int ablk; int agrp; int act_q8_1; };
// act_q8_1: 0 = q8_K (f32 d, stride 1168) ; 1 = q8_0 (fp16 d, stride 136)
static const FInfo FT[NFMT] = {
  {"iq4_xs", 256,136,2176,292,1168,0},
  {"iq2_xxs",256, 66,1184,292,1168,0},
  {"iq2_xs", 256, 74,1824,292,1168,0},
  {"iq2_s",  256, 82,1824,292,1168,0},
  {"mxfp4",   32, 17, 272, 34, 136,1},
  {"tq1_0",  256, 54, 864,292,1168,0},
  {"tq2_0",  256, 66,1056,292,1168,0},
};
static int parse_fmt(const char* s){ for(int i=0;i<NFMT;i++) if(!strcmp(s,FT[i].name)) return i; return -1; }

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static uint8_t rb(void){ return (uint8_t)(xr()&0xff); }
static int8_t  rq(void){ return (int8_t)((int)(xr()%255)-127); }
static uint16_t rwscale(void){ float f=0.010f+(float)(xr()%40)*0.001f; return f32_to_f16(f); } // ~[0.01,0.05] fp16
static float    rascale(void){ return 0.010f+(float)(xr()%40)*0.001f; }                          // ~[0.01,0.05] f32

// ---- PLAIN weight block fill (raw bytes, canonical ggml layout, fp16 d) ----
static void fill_wblk(int f, uint8_t* p){
    switch(f){
    case F_IQ4XS:{ *(ggml_half*)p = rwscale(); *(uint16_t*)(p+2)=(uint16_t)(xr()); // scales_h
        for(int i=0;i<4;i++) p[4+i]=rb(); for(int i=0;i<128;i++) p[8+i]=rb(); } break;
    case F_IQ2XXS:{ *(ggml_half*)p=rwscale(); uint16_t* q=(uint16_t*)(p+2); for(int i=0;i<32;i++) q[i]=(uint16_t)xr(); } break;
    case F_IQ2XS:{ *(ggml_half*)p=rwscale(); uint16_t* q=(uint16_t*)(p+2); for(int i=0;i<32;i++) q[i]=(uint16_t)xr(); for(int i=0;i<8;i++) p[66+i]=rb(); } break;
    case F_IQ2S:{ *(ggml_half*)p=rwscale(); for(int i=0;i<64;i++) p[2+i]=rb(); for(int i=0;i<8;i++) p[66+i]=rb(); for(int i=0;i<8;i++) p[74+i]=rb(); } break;
    case F_MXFP4:{ p[0]=(uint8_t)(122 + (xr()%5)); for(int i=0;i<16;i++) p[1+i]=rb(); } break; // e in [122,126] => 2^-6..2^-2
    case F_TQ1:{ for(int i=0;i<48;i++) p[i]=rb(); for(int i=0;i<4;i++) p[48+i]=rb(); *(ggml_half*)(p+52)=rwscale(); } break;
    case F_TQ2:{ for(int i=0;i<64;i++) p[i]=rb(); *(ggml_half*)(p+64)=rwscale(); } break;
    }
}
// ---- PLAIN q8_K / q8_0 activation block fill ----
static void fill_ablk(int f, uint8_t* p){
    if(FT[f].act_q8_1){ // q8_0 (34): fp16 d + int8 qs[32]
        *(ggml_half*)p = f32_to_f16(rascale()); int8_t* q=(int8_t*)(p+2); for(int i=0;i<32;i++) q[i]=rq();
    } else { // q8_K (292): f32 d + int8 qs[256] + int16 bsums[16]
        *(float*)p = rascale(); int8_t* q=(int8_t*)(p+4);
        int16_t bs[16]; for(int g=0;g<16;g++){ int s=0; for(int i=0;i<16;i++){ int8_t v=rq(); q[g*16+i]=v; s+=v; } bs[g]=(int16_t)s; }
        memcpy(p+260, bs, 32);
    }
}

// ---- WEIGHT repack: plain[16 cols][at wblk] -> x16 group buffer g (wgrp bytes), fp16 d ----
static void repack_wgrp(int f, uint8_t* g, const uint8_t* cols /*16 * wblk*/, int wblk){
    #define IN(c) (cols + (size_t)(c)*wblk)
    ggml_half* d16 = (ggml_half*)g;
    switch(f){
    case F_TQ2:{ for(int c=0;c<16;c++) d16[c]=*(const ggml_half*)(IN(c)+64);
        for(int c=0;c<16;c++){ const uint8_t* qs=IN(c); for(int i=0;i<64;i++) g[32 + i*16 + c]=qs[i]; } } break;
    case F_TQ1:{ for(int c=0;c<16;c++) d16[c]=*(const ggml_half*)(IN(c)+52);
        for(int c=0;c<16;c++){ const uint8_t* pc=IN(c);
            for(int i=0;i<48;i++) g[32 + i*16 + c]=pc[i];         // qs @ +32 (768B)
            for(int i=0;i<4;i++)  g[800 + i*16 + c]=pc[48+i]; } } break; // qh @ +800 (64B)
    case F_MXFP4:{ for(int c=0;c<16;c++) g[c]=IN(c)[0];            // e[16] @ +0
        for(int c=0;c<16;c++){ const uint8_t* qs=IN(c)+1; for(int i=0;i<16;i++) g[16 + i*16 + c]=qs[i]; } } break;
    case F_IQ4XS:{ for(int c=0;c<16;c++){ const uint8_t* pc=IN(c);
            d16[c]=*(const ggml_half*)pc; uint16_t sh=*(const uint16_t*)(pc+2);
            g[32+c]=(uint8_t)(sh&0xFF); g[48+c]=(uint8_t)((sh>>8)&0xFF);
            for(int p=0;p<4;p++) g[64 + p*16 + c]=pc[4+p];         // scales_l @ +64
            const uint8_t* qs=pc+8; for(int k=0;k<128;k++) g[128 + k*16 + c]=qs[k]; } } break; // qs @ +128
    case F_IQ2XXS:{ for(int c=0;c<16;c++){ const uint8_t* pc=IN(c); d16[c]=*(const ggml_half*)pc;
            const uint16_t* q=(const uint16_t*)(pc+2);
            for(int ib=0;ib<8;ib++){ uint32_t a0=(uint32_t)q[4*ib]|((uint32_t)q[4*ib+1]<<16);
                uint32_t a1=(uint32_t)q[4*ib+2]|((uint32_t)q[4*ib+3]<<16);
                ((int8_t*)g)[32 + ib*16 + c]=(int8_t)(2*(a1>>28)+1);        // ls @ +32
                for(int l=0;l<4;l++){ g[160 + (ib*4+l)*16 + c]=(uint8_t)((a0>>(8*l))&0xff);   // gidx @ +160
                    g[672 + (ib*4+l)*16 + c]=(uint8_t)((a1>>(7*l))&127); } } } } break;        // ssel @ +672
    case F_IQ2XS:{ for(int c=0;c<16;c++){ const uint8_t* pc=IN(c); d16[c]=*(const ggml_half*)pc;
            const uint16_t* q=(const uint16_t*)(pc+2); const uint8_t* sc=pc+66;
            for(int ib=0;ib<8;ib++){ int s=sc[ib];
                ((int8_t*)g)[32 + (ib*2+0)*16 + c]=(int8_t)(2*(s&0xf)+1);   // ls1 @ +32
                ((int8_t*)g)[32 + (ib*2+1)*16 + c]=(int8_t)(2*(s>>4)+1);    // ls2
                for(int l=0;l<4;l++){ uint16_t w=q[4*ib+l];
                    *(uint16_t*)(g + 288 + ((ib*4+l)*16 + c)*2)=(uint16_t)(w&511);  // gidx u16 @ +288
                    g[1312 + (ib*4+l)*16 + c]=(uint8_t)(w>>9); } } } } break;         // ssel u8 @ +1312
    case F_IQ2S:{ for(int c=0;c<16;c++){ const uint8_t* pc=IN(c); d16[c]=*(const ggml_half*)pc;
            const uint8_t* qs=pc+2; const uint8_t* qh=pc+66; const uint8_t* sc=pc+74;
            for(int ib=0;ib<8;ib++){ int s=sc[ib]; int h=qh[ib];
                ((int8_t*)g)[32 + (ib*2+0)*16 + c]=(int8_t)(2*(s&0xf)+1);
                ((int8_t*)g)[32 + (ib*2+1)*16 + c]=(int8_t)(2*(s>>4)+1);
                for(int l=0;l<4;l++){ int gg=4*ib+l; int high2=(h>>(2*l))&3;
                    *(uint16_t*)(g + 288 + ((ib*4+l)*16 + c)*2)=(uint16_t)(qs[gg]|(high2<<8));  // gidx u16
                    g[1312 + (ib*4+l)*16 + c]=qs[32+gg]; } } } } break;                          // ssel explicit
    }
    #undef IN
}
// ---- ACTIVATION interleave: plain[4 rows][ablk] -> x4 group (agrp) ----
static void repack_agrp(int f, uint8_t* g, const uint8_t* rows /*4 * ablk*/, int ablk){
    if(FT[f].act_q8_1){ // q8_0x4: fp16 d[4] @ +0, qs @ +8 as pos*4+row
        for(int r=0;r<4;r++){ const uint8_t* p=rows+(size_t)r*ablk; ((ggml_half*)g)[r]=*(const ggml_half*)p;
            const int8_t* q=(const int8_t*)(p+2); for(int i=0;i<32;i++) ((int8_t*)g)[8 + i*4 + r]=q[i]; }
    } else { // q8_Kx4: f32 d[4] @ +0, qs @ +16 as pos*4+row (bsums region +1040 left zero, unread)
        for(int r=0;r<4;r++){ const uint8_t* p=rows+(size_t)r*ablk; ((float*)g)[r]=*(const float*)p;
            const int8_t* q=(const int8_t*)(p+4); for(int i=0;i<256;i++) ((int8_t*)g)[16 + i*4 + r]=q[i]; }
    }
}
#ifndef DECODE_ONLY
static void our_run(int f,int K,int nr,int nc,float* Or,const uint8_t* Wr,const uint8_t* Ar){
    size_t n=(size_t)K,R=(size_t)nr,C=(size_t)nc,bs=(size_t)nc;
    switch(f){
    case F_IQ4XS: weft_emitc_ggml_repack_gemm_iq4_xs_q8_K_kernel_ggml_repack_gemm_iq4_xs_q8_K(n,Or,Wr,Ar,R,C,bs); break;
    case F_IQ2XXS:weft_emitc_ggml_repack_gemm_iq2_xxs_q8_K_kernel_ggml_repack_gemm_iq2_xxs_q8_K(n,Or,Wr,Ar,R,C,bs); break;
    case F_IQ2XS: weft_emitc_ggml_repack_gemm_iq2_xs_q8_K_kernel_ggml_repack_gemm_iq2_xs_q8_K(n,Or,Wr,Ar,R,C,bs); break;
    case F_IQ2S:  weft_emitc_ggml_repack_gemm_iq2_s_q8_K_kernel_ggml_repack_gemm_iq2_s_q8_K(n,Or,Wr,Ar,R,C,bs); break;
    case F_MXFP4: weft_emitc_ggml_repack_gemm_mxfp4_q8_0_kernel_ggml_repack_gemm_mxfp4_q8_0(n,Or,Wr,Ar,R,C,bs); break;
    case F_TQ1:   weft_emitc_ggml_repack_gemm_tq1_0_q8_K_kernel_ggml_repack_gemm_tq1_0_q8_K(n,Or,Wr,Ar,R,C,bs); break;
    case F_TQ2:   weft_emitc_ggml_repack_gemm_tq2_0_q8_K_kernel_ggml_repack_gemm_tq2_0_q8_K(n,Or,Wr,Ar,R,C,bs); break;
    }
}
#endif
static void our_gevm(int f,int K,int nc,float* Or,const uint8_t* Wr,const uint8_t* Aplain){
    size_t n=(size_t)K,C=(size_t)nc;
    switch(f){
    case F_IQ4XS: weft_emitc_ggml_repack_gemv_iq4_xs_q8_K_kernel_ggml_repack_gemv_iq4_xs_q8_K(n,Or,Wr,Aplain,C); break;
    case F_IQ2XXS:weft_emitc_ggml_repack_gemv_iq2_xxs_q8_K_kernel_ggml_repack_gemv_iq2_xxs_q8_K(n,Or,Wr,Aplain,C); break;
    case F_IQ2XS: weft_emitc_ggml_repack_gemv_iq2_xs_q8_K_kernel_ggml_repack_gemv_iq2_xs_q8_K(n,Or,Wr,Aplain,C); break;
    case F_IQ2S:  weft_emitc_ggml_repack_gemv_iq2_s_q8_K_kernel_ggml_repack_gemv_iq2_s_q8_K(n,Or,Wr,Aplain,C); break;
    case F_MXFP4: weft_emitc_ggml_repack_gemv_mxfp4_q8_0_kernel_ggml_repack_gemv_mxfp4_q8_0(n,Or,Wr,Aplain,C); break;
    case F_TQ1:   weft_emitc_ggml_repack_gemv_tq1_0_q8_K_kernel_ggml_repack_gemv_tq1_0_q8_K(n,Or,Wr,Aplain,C); break;
    case F_TQ2:   weft_emitc_ggml_repack_gemv_tq2_0_q8_K_kernel_ggml_repack_gemv_tq2_0_q8_K(n,Or,Wr,Aplain,C); break;
    }
}
static void opp_cell(int f,int K,float* out,const void* wc,const void* ar){
    switch(f){
    case F_IQ4XS: ggml_vec_dot_iq4_xs_q8_K_generic(K,out,0,wc,0,ar,0,1); break;
    case F_IQ2XXS:ggml_vec_dot_iq2_xxs_q8_K_generic(K,out,0,wc,0,ar,0,1); break;
    case F_IQ2XS: ggml_vec_dot_iq2_xs_q8_K_generic(K,out,0,wc,0,ar,0,1); break;
    case F_IQ2S:  ggml_vec_dot_iq2_s_q8_K_generic(K,out,0,wc,0,ar,0,1); break;
    case F_MXFP4: ggml_vec_dot_mxfp4_q8_0_generic(K,out,0,wc,0,ar,0,1); break;
    case F_TQ1:   ggml_vec_dot_tq1_0_q8_K_generic(K,out,0,wc,0,ar,0,1); break;
    case F_TQ2:   ggml_vec_dot_tq2_0_q8_K_generic(K,out,0,wc,0,ar,0,1); break;
    }
}

static double now_ms(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3+t.tv_nsec/1e6; }
static const size_t FLUSH_BYTES=32ull*1024*1024;
static uint8_t* g_flush=nullptr; static volatile uint64_t g_sink=0;
static void cold_flush(){ uint64_t a=0; for(size_t i=0;i<FLUSH_BYTES;i+=64){ g_flush[i]=(uint8_t)(g_flush[i]+1); a+=g_flush[i]; } g_sink+=a; }
static double med(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }
static double reliqr(std::vector<double> v){ std::sort(v.begin(),v.end()); size_t n=v.size(); double m=v[n/2]; return m>0?100.0*(v[(3*n)/4]-v[n/4])/m:0; }

int main(int argc,char**argv){
    if(argc<6){ fprintf(stderr,"usage: %s <fmt> K nr nc reps [verify_only] [seed]\n",argv[0]); return 2; }
    int f=parse_fmt(argv[1]); if(f<0){ fprintf(stderr,"bad fmt %s\n",argv[1]); return 2; }
    int K=atoi(argv[2]),nr=atoi(argv[3]),nc=atoi(argv[4]),reps=atoi(argv[5]);
    bool vo=argc>6&&atoi(argv[6])!=0; unsigned seed=argc>7?(unsigned)strtoul(argv[7],0,0):0xC0FFEE1u;
    bool decode=argc>8&&!strcmp(argv[8],"gevm");   // decode: M=1 GEVM, plain single q8 activation vector
#ifdef DECODE_ONLY
    decode=true;
#endif
    rng=(uint64_t)seed|1ull;
    if(decode) nr=1;                               // M=1
    int qk=FT[f].qk; if(K%qk||(!decode&&nr%4)||nc%16){ fprintf(stderr,"K%%%d, nr%%4(gemm), nc%%16 required\n",qk); return 2; }
    if(reps>200)reps=200; if(reps<1)reps=1;
    int nb=K/qk, grpc=nc/16, grpr=nr/4;
    int wblk=FT[f].wblk, ablk=FT[f].ablk, wgrp=FT[f].wgrp, agrp=FT[f].agrp;

    std::vector<uint8_t> Wp((size_t)nc*nb*wblk), Ap((size_t)nr*nb*ablk);
    std::vector<uint8_t> Wr((size_t)grpc*nb*wgrp,0), Ar((size_t)grpr*nb*agrp,0);
    std::vector<float> Or((size_t)nr*nc,0.f), Oo((size_t)nr*nc,0.f);
    g_flush=(uint8_t*)aligned_alloc(64,FLUSH_BYTES); memset(g_flush,1,FLUSH_BYTES);

    for(int c=0;c<nc;c++)for(int l=0;l<nb;l++) fill_wblk(f, &Wp[((size_t)c*nb+l)*wblk]);
    for(int r=0;r<nr;r++)for(int l=0;l<nb;l++) fill_ablk(f, &Ap[((size_t)r*nb+l)*ablk]);
    // repack weight
    std::vector<uint8_t> cols((size_t)16*wblk);
    for(int gc=0;gc<grpc;gc++)for(int l=0;l<nb;l++){
        for(int c=0;c<16;c++) memcpy(&cols[(size_t)c*wblk], &Wp[((size_t)(gc*16+c)*nb+l)*wblk], wblk);
        repack_wgrp(f, &Wr[((size_t)gc*nb+l)*wgrp], cols.data(), wblk); }
    const char* MODE = decode?"gevm":"gemm";
    if(!decode){ // GEMM: interleave activation into q8_Kx4/q8_0x4
        std::vector<uint8_t> rows((size_t)4*ablk);
        for(int gr=0;gr<grpr;gr++)for(int l=0;l<nb;l++){
            for(int r=0;r<4;r++) memcpy(&rows[(size_t)r*ablk], &Ap[((size_t)(gr*4+r)*nb+l)*ablk], ablk);
            repack_agrp(f, &Ar[((size_t)gr*nb+l)*agrp], rows.data(), ablk); }
    }
    // Ap for decode = single M=1 plain activation vector (nb blocks), fed PLAIN to OURS gevm AND opp.
#ifdef DECODE_ONLY
    #define RUN_OURS() our_gevm(f,K,nc,Or.data(),Wr.data(),Ap.data())
#else
    #define RUN_OURS() do{ if(decode) our_gevm(f,K,nc,Or.data(),Wr.data(),Ap.data()); else our_run(f,K,nr,nc,Or.data(),Wr.data(),Ar.data()); }while(0)
#endif
    #define RUN_OPP()  do{ for(int c=0;c<nc;c++)for(int r=0;r<nr;r++) opp_cell(f,K,&Oo[(size_t)r*nc+c], &Wp[((size_t)c*nb)*wblk], &Ap[((size_t)r*nb)*ablk]); }while(0)

    RUN_OURS(); RUN_OPP();

    double maxrel=0; int nbad=0;
    for(int r=0;r<nr;r++)for(int c=0;c<nc;c++){ double a=Or[(size_t)r*nc+c], b=Oo[(size_t)r*nc+c];
        double den=std::fabs(b)>1.0?std::fabs(b):1.0; double re=std::fabs(a-b)/den;
        if(re>maxrel)maxrel=re; if(re>5e-3)nbad++; }
    int gate=(nbad==0);
    printf("[%s.%s] GATE K=%d nr=%d nc=%d seed=0x%x | ours vs generic: maxrel=%.3e nbad=%d/%d sample ours=%.5f gen=%.5f => %s\n",
        FT[f].name,MODE,K,nr,nc,seed,maxrel,nbad,nr*nc,Or[0],Oo[0], gate?"OK":"*** MISMATCH ***");
    if(vo){ free(g_flush); return gate?0:1; }

    std::vector<double> tR(reps),tO(reps);
    for(int p=0;p<reps;p++){ cold_flush(); double t0=now_ms(); RUN_OURS(); tR[p]=now_ms()-t0; }
    for(int p=0;p<reps;p++){ cold_flush(); double t0=now_ms(); RUN_OPP(); tO[p]=now_ms()-t0; }
    double rM=med(tR),oM=med(tO);
    printf("[%s.%s] COLD K=%d nr=%d nc=%d reps=%d seed=0x%x | ours_ms=%.4f(iqr%.2f) opp_ms=%.4f(iqr%.2f) | ratio_cold=%.4f %s\n",
        FT[f].name,MODE,K,nr,nc,reps,seed,rM,reliqr(tR),oM,reliqr(tO),oM/rM,(oM/rM>=0.8?"PASS":"named-X"));
    free(g_flush); return gate?0:1;
}
