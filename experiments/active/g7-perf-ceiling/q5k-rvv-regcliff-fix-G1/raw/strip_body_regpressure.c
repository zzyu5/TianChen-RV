/* strip_body_regpressure.c -- FULL q5_K GEMM S1 h-strip body (ONE strip h, the hot
 * per-block contraction nest WITH accumulators), THREE variants byte-structurally
 * faithful to RVVToEmitCBlockQuantLinear.cpp:9673-10045 (emitRepackKQuantGemmBodyQ5K),
 * isolating the REGISTER-PRESSURE question (does qh-recon reduction clear the spill
 * cliff, or is the cliff dominated by the accumulator+unroll structure?).
 *
 * Deployment domain: VLEN128, coreLmul=mf2 -> half=8, columnsPerPass=4 (activationInterleave).
 * chain i8mf2 -> i16m1 -> i32m2 -> f32m2 (f16 scale m1). numHalves=2 @VLEN128 (this file
 * emits ONE strip; the deployed function emits TWO -- absolute board spill counts BOTH).
 *
 * Variants (ONLY the qh 5th-bit recon differs; the q5_K delta vs q4_K per :9928-9935):
 *   Q4K       : NO qh plane (baseline; deployed clang-18 spill=4). nLo=loNib, nHi=hiNib.
 *   Q5K_OLD   : deployed OLD qh recon (:9942-9951)
 *               loSel=vsrl(qh,s); loBit=vsll(vand(loSel,1),4); nLo=vor(loNib,loBit)  (4 op/arm, VREG intermediates)
 *   Q5K_KNEST : sub-block native-mask (q5k-knest-G1 design, byte-exact proven, decode-only)
 *               m=vmsne(vand(qh,1<<s),0); nLo=vadd_mu(m,loNib,loNib,16)               (3 op/arm, mask in v0/mask-reg)
 *
 * Hot live set reproduced faithfully: 4 f32m2 sumf (carried across block loop) + 4 i32m2 sumi
 * (per block) + 8 i16m1 sLo/sHi partials + shared weight decode + qh recon transients.
 * scale/min/bsums panels staged to STACK arrays (vse16/vle16 = DELIBERATE tile-materialization,
 * NOT spill; the emitter's :9731 S6 move). d/dmin widen DEFERRED to fold (:9985). This lets the
 * clang "# Folded Spill/Reload" comments cleanly separate TRUE RA spill from panel materialization.
 *
 * gcc-15.2 / clang-20  -march=rv64gcv_zvfh -mabi=lp64d -O3   (zvl128b = VLEN128).
 */
#include <riscv_vector.h>
#include <stdint.h>
#include <stddef.h>

#define HALF 8                 /* VLEN128 e8mf2 -> vl = 8 */
#define NSUB 8                 /* 8 sub-blocks / super-block */
#define NSUPERH 2              /* QK_K/128 */
#define SUBPERSUPER 4          /* 4 sub-blocks / super-half */
#define COLS 4                 /* columnsPerPass = activationInterleave */

/* weight ABI offsets (block_q5_Kx16 stride 2816; the SHARED qh slot @256, nibbles @256... */
#define OFF_D        0
#define OFF_DMIN     32
#define OFF_SCALES   64
#define OFF_QH       256       /* weightQhOffset (SHARED slot, MIN fold) */
#define OFF_QS       256       /* nibble bytes (per the leaf's weightQuantOffset=256) */
#define WSTRIDE      2816
#define ASTRIDE      1168
#define OFF_A_QUANT  16
#define OFF_A_BSUMS  1040

/* ---- shared: unpack ONE super-half's 4 sub-block 6-bit scale/min -> panels (:9848-9867) ---- */
static inline void unpack_scale_min(const uint8_t* bl, int j, int h,
                                    int16_t* scalePanel, int16_t* minPanel){
  size_t vl = HALF;
  for(int sb=0; sb<SUBPERSUPER; ++sb){
    int loByte = OFF_SCALES + j*64 + sb*16 + h*HALF;
    int hiByte = OFF_SCALES + 128 + sb*16 + h*HALF;
    vuint8mf2_t lo = __riscv_vle8_v_u8mf2(bl + loByte, vl);
    vuint8mf2_t hi = __riscv_vle8_v_u8mf2(bl + hiByte, vl);
    vuint8mf2_t scalesLo = __riscv_vand_vx_u8mf2(lo, 0x0F, vl);
    vuint8mf2_t minsLo   = __riscv_vsrl_vx_u8mf2(lo, 4, vl);
    vuint8mf2_t scalesHi, minsHi;
    if(j==0){
      scalesHi = __riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(hi,0x03,vl),4,vl);
      minsHi   = __riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(hi,0x0C,vl),2,vl);
    } else {
      scalesHi = __riscv_vand_vx_u8mf2(hi,0x30,vl);
      minsHi   = __riscv_vsrl_vx_u8mf2(__riscv_vand_vx_u8mf2(hi,0xC0,vl),2,vl);
    }
    vuint8mf2_t scU8 = __riscv_vor_vv_u8mf2(scalesHi, scalesLo, vl);
    vuint8mf2_t mnU8 = __riscv_vor_vv_u8mf2(minsHi, minsLo, vl);
    vint16m1_t scI16 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(scU8, vl));
    vint16m1_t mnI16 = __riscv_vreinterpret_v_u16m1_i16m1(__riscv_vzext_vf2_u16m1(mnU8, vl));
    __riscv_vse16_v_i16m1(scalePanel + sb*HALF, scI16, vl);
    __riscv_vse16_v_i16m1(minPanel   + sb*HALF, mnI16, vl);
  }
}

/* qh recon mode selector at compile time */
#ifndef QHMODE
#define QHMODE 1    /* 0 = Q4K (no qh), 1 = Q5K_OLD, 2 = Q5K_KNEST */
#endif

/* ==== the S1 h-strip body: ONE strip h over nb blocks, 4-column output ==== */
void strip_body(const uint8_t* b, const uint8_t* a, float* out,
                size_t nb, size_t out_stride, int h
#if QHMODE==3
                , const uint8_t* qhmask   /* re-transposed per-(i,sub) 8-col bit plane */
#endif
                ){
  size_t vl = HALF;
  /* per-strip f32 accumulators, carried ACROSS the block loop (SSA register, :9678) */
  vfloat32m2_t sumf0 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t sumf1 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t sumf2 = __riscv_vfmv_v_f_f32m2(0.0f, vl);
  vfloat32m2_t sumf3 = __riscv_vfmv_v_f_f32m2(0.0f, vl);

  for(size_t l=0; l<nb; ++l){
    const uint8_t* bl = b + l*WSTRIDE;
    const uint8_t* al = a + l*ASTRIDE;
    float aD[COLS];
    for(int c=0;c<COLS;++c) aD[c] = *(const float*)(al + c*4);

    /* PER-BLOCK stack panels (:9752-9766) -- DELIBERATE materialization, NOT spill */
    int16_t scalePanel[SUBPERSUPER*HALF];
    int16_t minPanel[SUBPERSUPER*HALF];
    int32_t bsumsPanel[COLS*HALF];

    /* per-block i32 main accumulators, SSA register (:9826-9833) */
    vint32m2_t sumi0 = __riscv_vmv_v_x_i32m2(0, vl);
    vint32m2_t sumi1 = __riscv_vmv_v_x_i32m2(0, vl);
    vint32m2_t sumi2 = __riscv_vmv_v_x_i32m2(0, vl);
    vint32m2_t sumi3 = __riscv_vmv_v_x_i32m2(0, vl);

    for(int j=0;j<NSUPERH;++j){
      unpack_scale_min(bl, j, h, scalePanel, minPanel);

      /* MIN term -> bsumsPanel (:9876-9896) */
      for(int sb=0; sb<SUBPERSUPER; ++sb){
        int gsub = j*SUBPERSUPER + sb;
        vint16m1_t minStrip = __riscv_vle16_v_i16m1(minPanel + sb*HALF, vl);
        for(int c=0;c<COLS;++c){
          int16_t bs0 = *(const int16_t*)(al + OFF_A_BSUMS + (gsub*8 + c)*2);
          int16_t bs1 = *(const int16_t*)(al + OFF_A_BSUMS + (gsub*8 + c + 4)*2);
          int32_t bsPair = (int32_t)bs0 + (int32_t)bs1;
          vint32m2_t cur = (j==0 && sb==0) ? __riscv_vmv_v_x_i32m2(0, vl)
                                           : __riscv_vle32_v_i32m2(bsumsPanel + c*HALF, vl);
          __riscv_vse32_v_i32m2(bsumsPanel + c*HALF,
                                __riscv_vwmacc_vx_i32m2(cur, bsPair, minStrip, vl), vl);
        }
      }

      /* MAIN term (:9906-9977) */
      for(int pair=0; pair<SUBPERSUPER/2; ++pair){
        int sbLo = pair*2, sbHi = pair*2+1;
        int qsPairBase = OFF_QS + j*1024 + pair*512;
        int aLoBase = OFF_A_QUANT + j*512 + sbLo*128;
        int aHiBase = OFF_A_QUANT + j*512 + sbHi*128;
        for(int k=0;k<2;++k){
          /* per-column i16 partials for this 16-elem chunk */
          vint16m1_t sLo0=__riscv_vmv_v_x_i16m1(0,vl), sLo1=__riscv_vmv_v_x_i16m1(0,vl),
                     sLo2=__riscv_vmv_v_x_i16m1(0,vl), sLo3=__riscv_vmv_v_x_i16m1(0,vl);
          vint16m1_t sHi0=__riscv_vmv_v_x_i16m1(0,vl), sHi1=__riscv_vmv_v_x_i16m1(0,vl),
                     sHi2=__riscv_vmv_v_x_i16m1(0,vl), sHi3=__riscv_vmv_v_x_i16m1(0,vl);
          int sLoBit = j*SUBPERSUPER + sbLo;
          int sHiBit = j*SUBPERSUPER + sbHi;
          for(int ii=0; ii<16; ++ii){
            int i = k*16 + ii;
            vuint8mf2_t packed = __riscv_vle8_v_u8mf2(bl + qsPairBase + i*16 + h*HALF, vl);
            vuint8mf2_t loNib = __riscv_vand_vx_u8mf2(packed, 0x0F, vl);
            vuint8mf2_t hiNib = __riscv_vsrl_vx_u8mf2(packed, 4, vl);
            vint8mf2_t nLo, nHi;
#if QHMODE==0
            /* Q4K: no qh plane */
            nLo = __riscv_vreinterpret_v_u8mf2_i8mf2(loNib);
            nHi = __riscv_vreinterpret_v_u8mf2_i8mf2(hiNib);
#elif QHMODE==1
            /* Q5K_OLD (deployed) */
            vuint8mf2_t qhs = __riscv_vle8_v_u8mf2(bl + OFF_QH + i*16 + h*HALF, vl);
            vuint8mf2_t loSel = (sLoBit==0) ? qhs : __riscv_vsrl_vx_u8mf2(qhs, sLoBit, vl);
            vuint8mf2_t loBit = __riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(loSel,0x01,vl),4,vl);
            vuint8mf2_t hiSel = __riscv_vsrl_vx_u8mf2(qhs, sHiBit, vl);
            vuint8mf2_t hiBit = __riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(hiSel,0x01,vl),4,vl);
            nLo = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vor_vv_u8mf2(loNib, loBit, vl));
            nHi = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vor_vv_u8mf2(hiNib, hiBit, vl));
#elif QHMODE==2
            /* Q5K_KNEST (native-mask, byte-exact proven) */
            vuint8mf2_t qhs = __riscv_vle8_v_u8mf2(bl + OFF_QH + i*16 + h*HALF, vl);
            vbool16_t loM = __riscv_vmsne_vx_u8mf2_b16(__riscv_vand_vx_u8mf2(qhs,(1u<<sLoBit),vl),0,vl);
            vbool16_t hiM = __riscv_vmsne_vx_u8mf2_b16(__riscv_vand_vx_u8mf2(qhs,(1u<<sHiBit),vl),0,vl);
            nLo = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vadd_vx_u8mf2_mu(loM, loNib, loNib, 16, vl));
            nHi = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vadd_vx_u8mf2_mu(hiM, hiNib, hiNib, 16, vl));
#endif
#if QHMODE==3
            /* RETRANS (re-transposed layout, HIGHER-RISK informed-decline): mask via vlm direct */
            vbool16_t loM = __riscv_vlm_v_b16(qhmask + (i*8 + sLoBit)*(HALF/8), vl);
            vbool16_t hiM = __riscv_vlm_v_b16(qhmask + (i*8 + sHiBit)*(HALF/8), vl);
            nLo = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vadd_vx_u8mf2_mu(loM, loNib, loNib, 16, vl));
            nHi = __riscv_vreinterpret_v_u8mf2_i8mf2(__riscv_vadd_vx_u8mf2_mu(hiM, hiNib, hiNib, 16, vl));
#endif
            int8_t aLo0 = *(const int8_t*)(al + aLoBase + (i*4+0));
            int8_t aLo1 = *(const int8_t*)(al + aLoBase + (i*4+1));
            int8_t aLo2 = *(const int8_t*)(al + aLoBase + (i*4+2));
            int8_t aLo3 = *(const int8_t*)(al + aLoBase + (i*4+3));
            int8_t aHi0 = *(const int8_t*)(al + aHiBase + (i*4+0));
            int8_t aHi1 = *(const int8_t*)(al + aHiBase + (i*4+1));
            int8_t aHi2 = *(const int8_t*)(al + aHiBase + (i*4+2));
            int8_t aHi3 = *(const int8_t*)(al + aHiBase + (i*4+3));
            sLo0 = __riscv_vwmacc_vx_i16m1(sLo0, aLo0, nLo, vl);
            sHi0 = __riscv_vwmacc_vx_i16m1(sHi0, aHi0, nHi, vl);
            sLo1 = __riscv_vwmacc_vx_i16m1(sLo1, aLo1, nLo, vl);
            sHi1 = __riscv_vwmacc_vx_i16m1(sHi1, aHi1, nHi, vl);
            sLo2 = __riscv_vwmacc_vx_i16m1(sLo2, aLo2, nLo, vl);
            sHi2 = __riscv_vwmacc_vx_i16m1(sHi2, aHi2, nHi, vl);
            sLo3 = __riscv_vwmacc_vx_i16m1(sLo3, aLo3, nLo, vl);
            sHi3 = __riscv_vwmacc_vx_i16m1(sHi3, aHi3, nHi, vl);
          }
          vint16m1_t scLo = __riscv_vle16_v_i16m1(scalePanel + sbLo*HALF, vl);
          vint16m1_t scHi = __riscv_vle16_v_i16m1(scalePanel + sbHi*HALF, vl);
          sumi0 = __riscv_vwmacc_vv_i32m2(__riscv_vwmacc_vv_i32m2(sumi0, scLo, sLo0, vl), scHi, sHi0, vl);
          sumi1 = __riscv_vwmacc_vv_i32m2(__riscv_vwmacc_vv_i32m2(sumi1, scLo, sLo1, vl), scHi, sHi1, vl);
          sumi2 = __riscv_vwmacc_vv_i32m2(__riscv_vwmacc_vv_i32m2(sumi2, scLo, sLo2, vl), scHi, sHi2, vl);
          sumi3 = __riscv_vwmacc_vv_i32m2(__riscv_vwmacc_vv_i32m2(sumi3, scLo, sLo3, vl), scHi, sHi3, vl);
        }
      }
    }

    /* end-of-block fold (:9981-10017): d/dmin widen loaded HERE (deferred) */
    vfloat32m2_t dminF = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1((const _Float16*)(bl+OFF_DMIN+h*HALF*2), vl), vl);
    vfloat32m2_t dF    = __riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1((const _Float16*)(bl+OFF_D   +h*HALF*2), vl), vl);
    /* fold column c: sumf_c += cvt(sumi_c)*(dF*aD_c) ; sumf_c -= (dminF*aD_c)*cvt(bsums_c) */
#define FOLD(SUMF, SUMI, C) do{ \
    vfloat32m2_t d0 = __riscv_vfmul_vf_f32m2(dF, aD[C], vl); \
    SUMF = __riscv_vfmacc_vv_f32m2(SUMF, __riscv_vfcvt_f_x_v_f32m2(SUMI, vl), d0, vl); \
    vfloat32m2_t dmin0 = __riscv_vfmul_vf_f32m2(dminF, aD[C], vl); \
    vint32m2_t bsV = __riscv_vle32_v_i32m2(bsumsPanel + (C)*HALF, vl); \
    SUMF = __riscv_vfnmsac_vv_f32m2(SUMF, dmin0, __riscv_vfcvt_f_x_v_f32m2(bsV, vl), vl); \
  }while(0)
    FOLD(sumf0, sumi0, 0);
    FOLD(sumf1, sumi1, 1);
    FOLD(sumf2, sumi2, 2);
    FOLD(sumf3, sumi3, 3);
#undef FOLD
  }

  /* per-column store */
  __riscv_vse32_v_f32m2(out + 0*out_stride + h*HALF, sumf0, vl);
  __riscv_vse32_v_f32m2(out + 1*out_stride + h*HALF, sumf1, vl);
  __riscv_vse32_v_f32m2(out + 2*out_stride + h*HALF, sumf2, vl);
  __riscv_vse32_v_f32m2(out + 3*out_stride + h*HALF, sumf3, vl);
}
