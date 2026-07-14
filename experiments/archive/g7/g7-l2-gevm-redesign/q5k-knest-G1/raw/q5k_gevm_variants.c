// q5_K GEVM decode variants -- ROLLED, single-strip (numHalves=1), structurally
// faithful to emitRepackKQuantGemvBodyQ5K (RVVToEmitCBlockQuantLinear.cpp:8517).
// mf2 core: u8mf2 -> i16m1 -> i32m2 -> f32m2 (identical intrinsic NAMES to the
// deployed VLEN128 numHalves=2 path; only the vl literal `half` differs -- so the
// per-strip v-insn COUNT is invariant to VLEN). The two functions are BYTE-IDENTICAL
// except the inner qh 5th-bit reconstruction:
//   cur_q5k   = CURRENT decode  : per-lane vsrl+vand+vsll+vor  (OLD-style)
//   knest_q5k = KNEST decode    : vand+vmsne+vadd_mu (stock native-mask technique
//               moved into the transposed super-block envelope)
// Every other op (6-bit scale/min unpack, min-fold, nibble extract, vwmacc chains,
// end-block f32 fold, accumulator residency) is IDENTICAL. RVV arrays are illegal in
// gcc, so numHalves=1 + fully-explicit sub-block scale vars. NO board run.
#include <riscv_vector.h>
#include <stdint.h>

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
// 6-bit scale/min unpack for sub-block sb of super-half j (IDENTICAL in both variants).
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

// The ONLY difference: RECON(loNib_u8,hiNib_u8,qh, sLoBit,sHiBit) -> nLo,nHi (i8mf2).
#define RECON_CUR(loNibU,hiNibU,qh,sLoBit,sHiBit,nLo,nHi) do {                     \
    vuint8mf2_t _ls=(sLoBit==0)?(qh):__riscv_vsrl_vx_u8mf2((qh),(sLoBit),HALF);    \
    vuint8mf2_t _lb=__riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(_ls,0x01,HALF),4,HALF); \
    vuint8mf2_t _hs=__riscv_vsrl_vx_u8mf2((qh),(sHiBit),HALF);                     \
    vuint8mf2_t _hb=__riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(_hs,0x01,HALF),4,HALF); \
    (nLo)=__riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vor_vv_u8mf2((loNibU),_lb,HALF)); \
    (nHi)=__riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vor_vv_u8mf2((hiNibU),_hb,HALF)); \
  } while(0)

#define RECON_KNEST(loNibU,hiNibU,qh,sLoBit,sHiBit,nLo,nHi) do {                   \
    vint8mf2_t _ln=__riscv_vreinterpret_v_u8mf2_i8mf2((loNibU));                   \
    vint8mf2_t _hn=__riscv_vreinterpret_v_u8mf2_i8mf2((hiNibU));                   \
    vbool16_t _lm=__riscv_vmsne_vx_u8mf2_b16(__riscv_vand_vx_u8mf2((qh),(uint8_t)(1u<<(sLoBit)),HALF),0,HALF); \
    vbool16_t _hm=__riscv_vmsne_vx_u8mf2_b16(__riscv_vand_vx_u8mf2((qh),(uint8_t)(1u<<(sHiBit)),HALF),0,HALF); \
    (nLo)=__riscv_vadd_vx_i8mf2_mu(_lm,_ln,_ln,16,HALF);                           \
    (nHi)=__riscv_vadd_vx_i8mf2_mu(_hm,_hn,_hn,16,HALF);                           \
  } while(0)

#define GEVM_BODY(RECON)                                                          \
  size_t nb=n/256, ncg=nc/16;                                                     \
  for(size_t x=0;x<ncg;++x){                                                      \
    vfloat32m2_t sumf=__riscv_vfmv_v_f_f32m2(0.0f,HALF);                          \
    const uint8_t*bG=vx+x*nb*WSTRIDE;                                             \
    for(size_t l=0;l<nb;++l){                                                     \
      const uint8_t*bl=bG+l*WSTRIDE; const uint8_t*al=vy+l*ASTRIDE;               \
      float aD=*(const float*)al;                                                 \
      vfloat32m2_t dminD=__riscv_vfmul_vf_f32m2(__riscv_vfwcvt_f_f_v_f32m2(       \
        __riscv_vle16_v_f16m1((const _Float16*)(bl+OFF_DMIN),HALF),HALF),aD,HALF);\
      vint32m2_t sumi=__riscv_vmv_v_x_i32m2(0,HALF);                              \
      vint32m2_t bsum=__riscv_vmv_v_x_i32m2(0,HALF);                              \
      for(int j=0;j<2;++j){                                                       \
        vint16m1_t sc0,sc1,sc2,sc3,mn0,mn1,mn2,mn3;                               \
        unpack_sm(bl,j,0,&sc0,&mn0); unpack_sm(bl,j,1,&sc1,&mn1);                 \
        unpack_sm(bl,j,2,&sc2,&mn2); unpack_sm(bl,j,3,&sc3,&mn3);                 \
        for(int sb=0;sb<4;++sb){                                                  \
          int g=j*4+sb;                                                           \
          int bp=*(const int16_t*)(al+OFF_ABSUMS+g*4)+*(const int16_t*)(al+OFF_ABSUMS+g*4+2);\
          vint16m1_t mnv=(sb==0?mn0:sb==1?mn1:sb==2?mn2:mn3);                     \
          bsum=__riscv_vwmacc_vx_i32m2(bsum,bp,mnv,HALF);                         \
        }                                                                         \
        for(int pair=0;pair<2;++pair){                                           \
          int sbLo=pair*2, sbHi=pair*2+1;                                        \
          long qsB=OFF_QS+j*1024+pair*512;                                        \
          long aLoB=OFF_AQS+j*128+sbLo*32, aHiB=OFF_AQS+j*128+sbHi*32;            \
          int sLoBit=j*4+sbLo, sHiBit=j*4+sbHi;                                   \
          vint16m1_t scL=(sbLo==0?sc0:sbLo==2?sc2:sc0);                           \
          vint16m1_t scH=(sbHi==1?sc1:sbHi==3?sc3:sc1);                           \
          for(int k=0;k<2;++k){                                                   \
            vint16m1_t sLo=__riscv_vmv_v_x_i16m1(0,HALF);                         \
            vint16m1_t sHi=__riscv_vmv_v_x_i16m1(0,HALF);                         \
            for(int ii=0;ii<16;++ii){                                            \
              int i=k*16+ii;                                                      \
              vuint8mf2_t packed=ldU8(bl,qsB+i*16);                               \
              vuint8mf2_t loNibU=__riscv_vand_vx_u8mf2(packed,0x0F,HALF);         \
              vuint8mf2_t hiNibU=__riscv_vsrl_vx_u8mf2(packed,4,HALF);            \
              vuint8mf2_t qh=ldU8(bl,OFF_QH+i*16);                                \
              vint8mf2_t nLo,nHi;                                                 \
              RECON(loNibU,hiNibU,qh,sLoBit,sHiBit,nLo,nHi);                      \
              int aLo=*(const int8_t*)(al+aLoB+i), aHi=*(const int8_t*)(al+aHiB+i);\
              sLo=__riscv_vwmacc_vx_i16m1(sLo,aLo,nLo,HALF);                      \
              sHi=__riscv_vwmacc_vx_i16m1(sHi,aHi,nHi,HALF);                      \
            }                                                                     \
            sumi=__riscv_vwmacc_vv_i32m2(sumi,scL,sLo,HALF);                      \
            sumi=__riscv_vwmacc_vv_i32m2(sumi,scH,sHi,HALF);                      \
          }                                                                       \
        }                                                                         \
      }                                                                           \
      vfloat32m2_t d0=__riscv_vfmul_vf_f32m2(__riscv_vfwcvt_f_f_v_f32m2(          \
        __riscv_vle16_v_f16m1((const _Float16*)(bl),HALF),HALF),aD,HALF);         \
      sumf=__riscv_vfmacc_vv_f32m2(sumf,__riscv_vfcvt_f_x_v_f32m2(sumi,HALF),d0,HALF);\
      sumf=__riscv_vfnmsac_vv_f32m2(sumf,dminD,__riscv_vfcvt_f_x_v_f32m2(bsum,HALF),HALF);\
    }                                                                             \
    __riscv_vse32_v_f32m2(s+x*16,sumf,HALF);                                       \
  }

void cur_q5k  (size_t n,float*s,const uint8_t*vx,const uint8_t*vy,size_t nc){ GEVM_BODY(RECON_CUR) }
void knest_q5k(size_t n,float*s,const uint8_t*vx,const uint8_t*vy,size_t nc){ GEVM_BODY(RECON_KNEST) }
