// q3_K GEVM decode variants -- ROLLED, single-strip, structurally faithful to
// emitRepackKQuantGemvBodyQ3K (RVVToEmitCBlockQuantLinear.cpp:16735). q3_K is the
// NO-MIN K-quant (x = a*q, no dmin/bsums fold) whose 3rd-bit "hmask" plane is the
// STRUCTURAL ANALOG of q5_K's qh 5th-bit plane -- a SINGLE-bit plane, so the stock
// native-mask technique (vmseq + vadd_mu, bias folded) applies exactly.
// mf2 core: u8mf2 -> i16m1 -> i32m2 -> f32 (identical intrinsic NAMES to the deployed
// VLEN128 path; only the vl literal `HALF` differs, so per-strip v-insn count is
// VLEN-invariant). The two functions are BYTE-IDENTICAL except the hmask recon:
//   cur_q3k   = CURRENT decode : base=vand0x03 ; hb=vand0x01 ; vsll<<2 ; vor ; vsub-4
//   knest_q3k = KNEST  decode  : mask0=vmseq(hplane,0) ; vadd_mu(base,base,-4)  (stock
//               native-mask technique moved into the super-block envelope; bias folded)
// Everything else (base extract, sub-block scale mul, vwmacc chains, d fold, residency)
// is IDENTICAL. RVV arrays illegal in gcc so fully-explicit vars. NO board run.
#include <riscv_vector.h>
#include <stdint.h>

#define HALF 16
#define OFF_SCALES 2       // fp16 d at [0], 6-bit packed scales region after
#define OFF_HMASK  256
#define OFF_QS     512
#define WSTRIDE    1824    // block_q3_Kx16 group stride (from paired driver)
#define ASTRIDE    292
#define OFF_AQS    4

static inline vuint8mf2_t ldU8(const uint8_t*b,long o){return __riscv_vle8_v_u8mf2(b+o,HALF);}

// hmask recon is the ONLY difference. RECON(baseU, hplane) -> q (i8mf2).
//  baseU  = 2-bit base already vand'd to 0x03 (u8mf2)
//  hplane = per-element 3rd-bit plane byte (repacked: bit in position `sBit`)
#define RECON_CUR(baseU,hplane,sBit,q) do {                                        \
    vuint8mf2_t _hs=(sBit==0)?(hplane):__riscv_vsrl_vx_u8mf2((hplane),(sBit),HALF); \
    vuint8mf2_t _hb=__riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(_hs,0x01,HALF),2,HALF); \
    vuint8mf2_t _v =__riscv_vor_vv_u8mf2((baseU),_hb,HALF);                         \
    (q)=__riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(_v),4,HALF);       \
  } while(0)

#define RECON_KNEST(baseU,hplane,sBit,q) do {                                      \
    vint8mf2_t _bn=__riscv_vreinterpret_v_u8mf2_i8mf2((baseU));                     \
    vbool16_t _m0=__riscv_vmseq_vx_u8mf2_b16(                                       \
        __riscv_vand_vx_u8mf2((hplane),(uint8_t)(1u<<(sBit)),HALF),0,HALF);         \
    (q)=__riscv_vadd_vx_i8mf2_mu(_m0,_bn,_bn,-4,HALF);                              \
  } while(0)

#define GEVM_BODY(RECON)                                                          \
  size_t nb=n/256, ncg=nc/16;                                                     \
  for(size_t x=0;x<ncg;++x){                                                      \
    vfloat32m2_t sumf=__riscv_vfmv_v_f_f32m2(0.0f,HALF);                          \
    const uint8_t*bG=vx+x*nb*WSTRIDE;                                             \
    for(size_t l=0;l<nb;++l){                                                     \
      const uint8_t*bl=bG+l*WSTRIDE; const uint8_t*al=vy+l*ASTRIDE;               \
      float aD=*(const float*)al;                                                 \
      vint32m2_t sumi=__riscv_vmv_v_x_i32m2(0,HALF);                              \
      for(int g=0;g<8;++g){       /* 8 sub-blocks per super-block */              \
        /* 6-bit sub-block scale (i16 broadcast; identical both variants) */      \
        vint16m1_t scv=__riscv_vreinterpret_v_u16m1_i16m1(                        \
          __riscv_vzext_vf2_u16m1(ldU8(bl,OFF_SCALES+g*16),HALF));                \
        vint16m1_t acc=__riscv_vmv_v_x_i16m1(0,HALF);                             \
        for(int ii=0;ii<4;++ii){   /* 4 elem-steps per sub-block strip */         \
          int i=g*4+ii;                                                           \
          vuint8mf2_t baseU=__riscv_vand_vx_u8mf2(ldU8(bl,OFF_QS+i*16),0x03,HALF); \
          vuint8mf2_t hplane=ldU8(bl,OFF_HMASK+i*16);                             \
          vint8mf2_t q;                                                           \
          RECON(baseU,hplane,(g&7),q);                                            \
          int a=*(const int8_t*)(al+OFF_AQS+i);                                   \
          acc=__riscv_vwmacc_vx_i16m1(acc,a,q,HALF);                              \
        }                                                                         \
        sumi=__riscv_vwmacc_vv_i32m2(sumi,scv,acc,HALF);                          \
      }                                                                           \
      vfloat32m2_t d0=__riscv_vfmul_vf_f32m2(__riscv_vfwcvt_f_f_v_f32m2(          \
        __riscv_vle16_v_f16m1((const _Float16*)(bl),HALF),HALF),aD,HALF);         \
      sumf=__riscv_vfmacc_vv_f32m2(sumf,__riscv_vfcvt_f_x_v_f32m2(sumi,HALF),d0,HALF);\
    }                                                                             \
    __riscv_vse32_v_f32m2(s+x*16,sumf,HALF);                                       \
  }

void cur_q3k  (size_t n,float*s,const uint8_t*vx,const uint8_t*vy,size_t nc){ GEVM_BODY(RECON_CUR) }
void knest_q3k(size_t n,float*s,const uint8_t*vx,const uint8_t*vy,size_t nc){ GEVM_BODY(RECON_KNEST) }
