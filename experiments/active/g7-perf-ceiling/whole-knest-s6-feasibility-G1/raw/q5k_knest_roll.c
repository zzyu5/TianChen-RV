// ============================================================================
// whole-K-nest S6 feasibility G1 -- q5_K GEVM (M=1) ROLLED vs FULL-UNROLL A/B.
//
// PURPOSE: isolate the *loop-structure* effect (whole-nest roll vs full static
// unroll) on a BYTE-IDENTICAL q5_K decode algorithm, compiled with the deployment
// compiler (SpacemiT gcc-15.2, -march=rv64gcv_zvfh -O3). The per-element inner
// WORK is a single shared macro ELEM_STEP(...) so both variants execute the
// EXACT same sequence of vwmacc / decode ops in the EXACT same order (asc j,pair,
// k,ii) -- only the loop nest is rolled (runtime emitc.for-equivalent) vs unrolled
// (#pragma GCC unroll 64). => byte-exact by construction; the A/B driver proves it
// numerically (memcmp full fp32 output on random input).
//
// Accumulator = SINGLE WIDE resident bank carried across the super-block K loop
// as SSA locals (sumi/bsum/sumf, vint32m2/vfloat32m2) -- never materialized to a
// stack scratch tile (the [GAP-EMIT-KQUANT-GEVM-TILE-ROUNDTRIP] pattern). numHalves
// =1 single strip (mf2 core u8mf2->i16m1->i32m2->f32m2); v-insn COUNT is VLEN-
// invariant per q5k-knest-G1 calibration. Structurally faithful to
// emitRepackKQuantGemvBodyQ5K.  NO board run required for the .s account.
// ============================================================================
#include <riscv_vector.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#define HALF 16
#define OFF_DMIN 32
#define OFF_SCALES 64
#define OFF_QH 256
#define OFF_QS 768
#define WSTRIDE 2816
#define ASTRIDE 292
#define OFF_AQS 4
#define OFF_ABSUMS 260

static inline vuint8mf2_t ldU8(const uint8_t*b,long o){return __riscv_vle8_v_u8mf2(b+o,HALF);}
static inline vint16m1_t liftI16(vuint8mf2_t u){
  return __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(u,HALF));
}
static inline void unpack_sm(const uint8_t*bl,int j,int sb,vint16m1_t*sc,vint16m1_t*mn){
  long loB=OFF_SCALES+j*64+sb*16, hiB=OFF_SCALES+128+sb*16;
  vuint8mf2_t lo=ldU8(bl,loB), hi=ldU8(bl,hiB);
  vuint8mf2_t scLo=__riscv_vand_vx_u8mf2(lo,0x0F,HALF);
  vuint8mf2_t mnLo=__riscv_vsrl_vx_u8mf2(lo,4,HALF);
  vuint8mf2_t scHi,mnHi;
  if(j==0){ scHi=__riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(hi,0x03,HALF),4,HALF);
            mnHi=__riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(hi,0x0C,HALF),2,HALF); }
  else    { scHi=__riscv_vand_vx_u8mf2(hi,0x30,HALF);
            mnHi=__riscv_vsrl_vx_u8mf2(__riscv_vand_vx_u8mf2(hi,0xC0,HALF),2,HALF); }
  *sc=liftI16(__riscv_vor_vv_u8mf2(scHi,scLo,HALF));
  *mn=liftI16(__riscv_vor_vv_u8mf2(mnHi,mnLo,HALF));
}

// per-element inner work -- SHARED by both variants (identical op sequence).
// decodes one weight element (lo+hi arms), applies qh 5th-bit (CURRENT OLD-style
// recon), and vwmacc's into the i16 partials sLo/sHi.
#define ELEM_STEP(bl,al,qsB,aLoB,aHiB,sLoBit,sHiBit,i,sLo,sHi) do {                 \
    vuint8mf2_t packed=ldU8(bl,qsB+(i)*16);                                         \
    vuint8mf2_t loNibU=__riscv_vand_vx_u8mf2(packed,0x0F,HALF);                     \
    vuint8mf2_t hiNibU=__riscv_vsrl_vx_u8mf2(packed,4,HALF);                        \
    vuint8mf2_t qh=ldU8(bl,OFF_QH+(i)*16);                                          \
    vuint8mf2_t _ls=((sLoBit)==0)?(qh):__riscv_vsrl_vx_u8mf2(qh,(sLoBit),HALF);     \
    vuint8mf2_t _lb=__riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(_ls,0x01,HALF),4,HALF);\
    vuint8mf2_t _hs=__riscv_vsrl_vx_u8mf2(qh,(sHiBit),HALF);                        \
    vuint8mf2_t _hb=__riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(_hs,0x01,HALF),4,HALF);\
    vint8mf2_t nLo=__riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vor_vv_u8mf2(loNibU,_lb,HALF));\
    vint8mf2_t nHi=__riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vor_vv_u8mf2(hiNibU,_hb,HALF));\
    int aLo=*(const int8_t*)((al)+(aLoB)+(i)), aHi=*(const int8_t*)((al)+(aHiB)+(i));\
    sLo=__riscv_vwmacc_vx_i16m1(sLo,aLo,nLo,HALF);                                  \
    sHi=__riscv_vwmacc_vx_i16m1(sHi,aHi,nHi,HALF);                                  \
  } while(0)

// ---- ROLLED whole-nest form: every K-nest loop is a runtime loop (#pragma unroll 1)
// with a single wide resident accumulator bank (sumi/bsum/sumf) carried across the
// super-block loop.  This is the whole-K-nest S6 restructuring under evaluation.
void q5k_rolled(size_t n,float*s,const uint8_t*vx,const uint8_t*vy,size_t nc){
  size_t nb=n/256, ncg=nc/16;
  for(size_t x=0;x<ncg;++x){
    vfloat32m2_t sumf=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
    const uint8_t*bG=vx+x*nb*WSTRIDE;
    #pragma GCC unroll 1
    for(size_t l=0;l<nb;++l){
      const uint8_t*bl=bG+l*WSTRIDE; const uint8_t*al=vy+l*ASTRIDE;
      float aD=*(const float*)al;
      vfloat32m2_t dminD=__riscv_vfmul_vf_f32m2(__riscv_vfwcvt_f_f_v_f32m2(
        __riscv_vle16_v_f16m1((const _Float16*)(bl+OFF_DMIN),HALF),HALF),aD,HALF);
      vint32m2_t sumi=__riscv_vmv_v_x_i32m2(0,HALF);
      vint32m2_t bsum=__riscv_vmv_v_x_i32m2(0,HALF);
      #pragma GCC unroll 1
      for(int j=0;j<2;++j){
        vint16m1_t sc0,sc1,sc2,sc3,mn0,mn1,mn2,mn3;
        unpack_sm(bl,j,0,&sc0,&mn0); unpack_sm(bl,j,1,&sc1,&mn1);
        unpack_sm(bl,j,2,&sc2,&mn2); unpack_sm(bl,j,3,&sc3,&mn3);
        #pragma GCC unroll 1
        for(int sb=0;sb<4;++sb){
          int g=j*4+sb;
          int bp=*(const int16_t*)(al+OFF_ABSUMS+g*4)+*(const int16_t*)(al+OFF_ABSUMS+g*4+2);
          vint16m1_t mnv=(sb==0?mn0:sb==1?mn1:sb==2?mn2:mn3);
          bsum=__riscv_vwmacc_vx_i32m2(bsum,bp,mnv,HALF);
        }
        #pragma GCC unroll 1
        for(int pair=0;pair<2;++pair){
          int sbLo=pair*2, sbHi=pair*2+1;
          long qsB=OFF_QS+j*1024+pair*512;
          long aLoB=OFF_AQS+j*128+sbLo*32, aHiB=OFF_AQS+j*128+sbHi*32;
          int sLoBit=j*4+sbLo, sHiBit=j*4+sbHi;
          vint16m1_t scL=(sbLo==0?sc0:sbLo==2?sc2:sc0);
          vint16m1_t scH=(sbHi==1?sc1:sbHi==3?sc3:sc1);
          #pragma GCC unroll 1
          for(int k=0;k<2;++k){
            vint16m1_t sLo=__riscv_vmv_v_x_i16m1(0,HALF);
            vint16m1_t sHi=__riscv_vmv_v_x_i16m1(0,HALF);
            #pragma GCC unroll 1
            for(int ii=0;ii<16;++ii){
              int i=k*16+ii;
              ELEM_STEP(bl,al,qsB,aLoB,aHiB,sLoBit,sHiBit,i,sLo,sHi);
            }
            sumi=__riscv_vwmacc_vv_i32m2(sumi,scL,sLo,HALF);
            sumi=__riscv_vwmacc_vv_i32m2(sumi,scH,sHi,HALF);
          }
        }
      }
      vfloat32m2_t d0=__riscv_vfmul_vf_f32m2(__riscv_vfwcvt_f_f_v_f32m2(
        __riscv_vle16_v_f16m1((const _Float16*)(bl),HALF),HALF),aD,HALF);
      sumf=__riscv_vfmacc_vv_f32m2(sumf,__riscv_vfcvt_f_x_v_f32m2(sumi,HALF),d0,HALF);
      sumf=__riscv_vfnmsac_vv_f32m2(sumf,dminD,__riscv_vfcvt_f_x_v_f32m2(bsum,HALF),HALF);
    }
    __riscv_vse32_v_f32m2(s+x*16,sumf,HALF);
  }
}

// ---- FULL-UNROLL form: identical algorithm, inner K-nest loops FULLY unrolled
// (#pragma GCC unroll 64).  Represents the CURRENT deployed emission structure.
void q5k_unrolled(size_t n,float*s,const uint8_t*vx,const uint8_t*vy,size_t nc){
  size_t nb=n/256, ncg=nc/16;
  for(size_t x=0;x<ncg;++x){
    vfloat32m2_t sumf=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
    const uint8_t*bG=vx+x*nb*WSTRIDE;
    for(size_t l=0;l<nb;++l){
      const uint8_t*bl=bG+l*WSTRIDE; const uint8_t*al=vy+l*ASTRIDE;
      float aD=*(const float*)al;
      vfloat32m2_t dminD=__riscv_vfmul_vf_f32m2(__riscv_vfwcvt_f_f_v_f32m2(
        __riscv_vle16_v_f16m1((const _Float16*)(bl+OFF_DMIN),HALF),HALF),aD,HALF);
      vint32m2_t sumi=__riscv_vmv_v_x_i32m2(0,HALF);
      vint32m2_t bsum=__riscv_vmv_v_x_i32m2(0,HALF);
      #pragma GCC unroll 64
      for(int j=0;j<2;++j){
        vint16m1_t sc0,sc1,sc2,sc3,mn0,mn1,mn2,mn3;
        unpack_sm(bl,j,0,&sc0,&mn0); unpack_sm(bl,j,1,&sc1,&mn1);
        unpack_sm(bl,j,2,&sc2,&mn2); unpack_sm(bl,j,3,&sc3,&mn3);
        #pragma GCC unroll 64
        for(int sb=0;sb<4;++sb){
          int g=j*4+sb;
          int bp=*(const int16_t*)(al+OFF_ABSUMS+g*4)+*(const int16_t*)(al+OFF_ABSUMS+g*4+2);
          vint16m1_t mnv=(sb==0?mn0:sb==1?mn1:sb==2?mn2:mn3);
          bsum=__riscv_vwmacc_vx_i32m2(bsum,bp,mnv,HALF);
        }
        #pragma GCC unroll 64
        for(int pair=0;pair<2;++pair){
          int sbLo=pair*2, sbHi=pair*2+1;
          long qsB=OFF_QS+j*1024+pair*512;
          long aLoB=OFF_AQS+j*128+sbLo*32, aHiB=OFF_AQS+j*128+sbHi*32;
          int sLoBit=j*4+sbLo, sHiBit=j*4+sbHi;
          vint16m1_t scL=(sbLo==0?sc0:sbLo==2?sc2:sc0);
          vint16m1_t scH=(sbHi==1?sc1:sbHi==3?sc3:sc1);
          #pragma GCC unroll 64
          for(int k=0;k<2;++k){
            vint16m1_t sLo=__riscv_vmv_v_x_i16m1(0,HALF);
            vint16m1_t sHi=__riscv_vmv_v_x_i16m1(0,HALF);
            #pragma GCC unroll 64
            for(int ii=0;ii<16;++ii){
              int i=k*16+ii;
              ELEM_STEP(bl,al,qsB,aLoB,aHiB,sLoBit,sHiBit,i,sLo,sHi);
            }
            sumi=__riscv_vwmacc_vv_i32m2(sumi,scL,sLo,HALF);
            sumi=__riscv_vwmacc_vv_i32m2(sumi,scH,sHi,HALF);
          }
        }
      }
      vfloat32m2_t d0=__riscv_vfmul_vf_f32m2(__riscv_vfwcvt_f_f_v_f32m2(
        __riscv_vle16_v_f16m1((const _Float16*)(bl),HALF),HALF),aD,HALF);
      sumf=__riscv_vfmacc_vv_f32m2(sumf,__riscv_vfcvt_f_x_v_f32m2(sumi,HALF),d0,HALF);
      sumf=__riscv_vfnmsac_vv_f32m2(sumf,dminD,__riscv_vfcvt_f_x_v_f32m2(bsum,HALF),HALF);
    }
    __riscv_vse32_v_f32m2(s+x*16,sumf,HALF);
  }
}

#ifdef DRIVER
// A/B byte-exact driver: random weight + activation buffers, run both, memcmp fp32.
// fp16 super-scales (d @0, dmin @OFF_DMIN) and activation float-d are set finite to
// avoid nan false-mismatch; every integer field is random.
static uint16_t f16_one(void){ return 0x3C00; } // 1.0 in IEEE fp16
int main(int argc,char**argv){
  unsigned seed = (argc>1)?(unsigned)strtoul(argv[1],0,0):0xC0FFEEu;
  srand(seed);
  size_t nb=3, ncg=1, nc=ncg*16, n=nb*256;
  size_t wbytes=ncg*nb*WSTRIDE, abytes=nb*ASTRIDE;
  uint8_t*vx=(uint8_t*)malloc(wbytes); uint8_t*vy=(uint8_t*)malloc(abytes);
  for(size_t i=0;i<wbytes;++i) vx[i]=(uint8_t)rand();
  for(size_t i=0;i<abytes;++i) vy[i]=(uint8_t)rand();
  // finite fp16 super-scales per weight super-block
  for(size_t x=0;x<ncg;++x) for(size_t l=0;l<nb;++l){
    uint8_t*bl=vx+(x*nb+l)*WSTRIDE;
    uint16_t h=f16_one(); memcpy(bl,&h,2); memcpy(bl+OFF_DMIN,&h,2);
  }
  // finite activation float-d per block
  for(size_t l=0;l<nb;++l){ float d=0.5f; memcpy(vy+l*ASTRIDE,&d,4); }
  float*sr=(float*)malloc(nc*sizeof(float)); float*su=(float*)malloc(nc*sizeof(float));
  memset(sr,0,nc*sizeof(float)); memset(su,0,nc*sizeof(float));
  q5k_rolled  (n,sr,vx,vy,nc);
  q5k_unrolled(n,su,vx,vy,nc);
  int mm=memcmp(sr,su,nc*sizeof(float));
  int nzr=0; for(size_t i=0;i<nc;++i) if(sr[i]!=0.0f) nzr++;
  printf("seed=0x%X nb=%zu nc=%zu : rolled-vs-unrolled memcmp=%d (0=byte-exact) nonzero_outputs=%d/%zu\n",
         seed,nb,nc,mm,nzr,nc);
  printf("  s[0..3] rolled  = %g %g %g %g\n", sr[0],sr[1],sr[2],sr[3]);
  printf("  s[0..3] unroll  = %g %g %g %g\n", su[0],su[1],su[2],su[3]);
  return mm==0?0:1;
}
#endif
