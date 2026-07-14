#include <riscv_vector.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#define HALF 8
#define WSTRIDE 3360
#define ASTRIDE 1168
#define OFF_QL 1312
#define OFF_QH 288
#define OFF_SCALES 32
#define OFF_AQ 16
static const int Q_STREAM[4]={0,1,0,1};
static const int Q_HIGH[4]  ={0,0,1,1};
static const int Q_SHIFT[4] ={0,2,4,6};
static inline vuint8mf2_t ldU8(const uint8_t*b,long o){return __riscv_vle8_v_u8mf2(b+o,HALF);}
static inline vint8mf2_t  ldI8(const uint8_t*b,long o){return __riscv_vle8_v_i8mf2((const int8_t*)(b+o),HALF);}
static inline vint8mf2_t asmW(vuint8mf2_t ql,vuint8mf2_t qh,int hi,int sh){
  vuint8mf2_t nib=hi?__riscv_vsrl_vx_u8mf2(ql,4,HALF):__riscv_vand_vx_u8mf2(ql,0x0F,HALF);
  vuint8mf2_t sel=(sh==0)?qh:__riscv_vsrl_vx_u8mf2(qh,sh,HALF);
  vuint8mf2_t qhb=__riscv_vsll_vx_u8mf2(__riscv_vand_vx_u8mf2(sel,0x03,HALF),4,HALF);
  vuint8mf2_t raw=__riscv_vor_vv_u8mf2(nib,qhb,HALF);
  return __riscv_vsub_vx_i8mf2(__riscv_vreinterpret_v_u8mf2_i8mf2(raw),32,HALF);
}
static inline vint16m1_t liftScale(vint8mf2_t s8){return __riscv_vsext_vf2_i16m1(s8,HALF);}

void q6k_gemm_rolled(size_t n,float*s,const uint8_t*vx,const uint8_t*vy,size_t nr,size_t nc,size_t bs){
  size_t nb=n/256, nrg=nr/4, ncg=nc/16;
  for(size_t y=0;y<nrg;++y){
    const uint8_t*aG=vy+y*nb*ASTRIDE;
    for(size_t x=0;x<ncg;++x){
      const uint8_t*bG=vx+x*nb*WSTRIDE;
      vfloat32m2_t sumf_0_0=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
      vfloat32m2_t sumf_0_1=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
      vfloat32m2_t sumf_1_0=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
      vfloat32m2_t sumf_1_1=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
      vfloat32m2_t sumf_2_0=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
      vfloat32m2_t sumf_2_1=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
      vfloat32m2_t sumf_3_0=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
      vfloat32m2_t sumf_3_1=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
      #pragma GCC unroll 1
      for(size_t l=0;l<nb;++l){
    const uint8_t* bl=bG+l*WSTRIDE;
    const uint8_t* al=aG+l*ASTRIDE;
    float aD0=*(const float*)(al+0);
    float aD1=*(const float*)(al+4);
    float aD2=*(const float*)(al+8);
    float aD3=*(const float*)(al+12);
    vfloat32m2_t dF0=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1((const _Float16*)(bl+0),HALF),HALF);
    vfloat32m2_t dF1=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1((const _Float16*)(bl+16),HALF),HALF);
    vint32m2_t sumi_0_0=__riscv_vmv_v_x_i32m2(0,HALF);
    vint32m2_t sumi_0_1=__riscv_vmv_v_x_i32m2(0,HALF);
    vint32m2_t sumi_1_0=__riscv_vmv_v_x_i32m2(0,HALF);
    vint32m2_t sumi_1_1=__riscv_vmv_v_x_i32m2(0,HALF);
    vint32m2_t sumi_2_0=__riscv_vmv_v_x_i32m2(0,HALF);
    vint32m2_t sumi_2_1=__riscv_vmv_v_x_i32m2(0,HALF);
    vint32m2_t sumi_3_0=__riscv_vmv_v_x_i32m2(0,HALF);
    vint32m2_t sumi_3_1=__riscv_vmv_v_x_i32m2(0,HALF);
    _Pragma("GCC unroll 1")
    for(int j=0;j<2;++j){
      _Pragma("GCC unroll 1")
      for(int sh=0;sh<2;++sh){
        vint16m1_t sc_0_0=liftScale(ldI8(bl,OFF_SCALES+(j*8+0*2+sh)*16+0));
        vint16m1_t sc_0_1=liftScale(ldI8(bl,OFF_SCALES+(j*8+1*2+sh)*16+0));
        vint16m1_t sc_0_2=liftScale(ldI8(bl,OFF_SCALES+(j*8+2*2+sh)*16+0));
        vint16m1_t sc_0_3=liftScale(ldI8(bl,OFF_SCALES+(j*8+3*2+sh)*16+0));
        vint16m1_t sc_1_0=liftScale(ldI8(bl,OFF_SCALES+(j*8+0*2+sh)*16+8));
        vint16m1_t sc_1_1=liftScale(ldI8(bl,OFF_SCALES+(j*8+1*2+sh)*16+8));
        vint16m1_t sc_1_2=liftScale(ldI8(bl,OFF_SCALES+(j*8+2*2+sh)*16+8));
        vint16m1_t sc_1_3=liftScale(ldI8(bl,OFF_SCALES+(j*8+3*2+sh)*16+8));
        _Pragma("GCC unroll 1")
        for(int k=0;k<2;++k){
          vint16m1_t acc_0_0_0=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_0_0_1=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_0_0_2=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_0_0_3=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_0_1_0=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_0_1_1=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_0_1_2=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_0_1_3=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_1_0_0=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_1_0_1=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_1_0_2=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_1_0_3=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_1_1_0=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_1_1_1=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_1_1_2=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_1_1_3=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_2_0_0=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_2_0_1=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_2_0_2=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_2_0_3=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_2_1_0=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_2_1_1=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_2_1_2=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_2_1_3=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_3_0_0=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_3_0_1=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_3_0_2=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_3_0_3=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_3_1_0=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_3_1_1=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_3_1_2=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_3_1_3=__riscv_vmv_v_x_i16m1(0,HALF);
          _Pragma("GCC unroll 1")
          for(int p=0;p<8;++p){
            int ll=sh*16+k*8+p;
            vuint8mf2_t qlA0=ldU8(bl,OFF_QL+(j*64+ll)*16+0);
            vuint8mf2_t qlB0=ldU8(bl,OFF_QL+(j*64+32+ll)*16+0);
            vuint8mf2_t qh0=ldU8(bl,OFF_QH+(j*32+ll)*16+0);
            vint8mf2_t wq_0_0=asmW(qlA0,qh0,Q_HIGH[0],Q_SHIFT[0]);
            vint8mf2_t wq_0_1=asmW(qlB0,qh0,Q_HIGH[1],Q_SHIFT[1]);
            vint8mf2_t wq_0_2=asmW(qlA0,qh0,Q_HIGH[2],Q_SHIFT[2]);
            vint8mf2_t wq_0_3=asmW(qlB0,qh0,Q_HIGH[3],Q_SHIFT[3]);
            vuint8mf2_t qlA1=ldU8(bl,OFF_QL+(j*64+ll)*16+8);
            vuint8mf2_t qlB1=ldU8(bl,OFF_QL+(j*64+32+ll)*16+8);
            vuint8mf2_t qh1=ldU8(bl,OFF_QH+(j*32+ll)*16+8);
            vint8mf2_t wq_1_0=asmW(qlA1,qh1,Q_HIGH[0],Q_SHIFT[0]);
            vint8mf2_t wq_1_1=asmW(qlB1,qh1,Q_HIGH[1],Q_SHIFT[1]);
            vint8mf2_t wq_1_2=asmW(qlA1,qh1,Q_HIGH[2],Q_SHIFT[2]);
            vint8mf2_t wq_1_3=asmW(qlB1,qh1,Q_HIGH[3],Q_SHIFT[3]);
            int aq_0_0=*(const int8_t*)(al+OFF_AQ+(j*128+0*32+ll)*4+0);
            acc_0_0_0=__riscv_vwmacc_vx_i16m1(acc_0_0_0,aq_0_0,wq_0_0,HALF);
            acc_0_1_0=__riscv_vwmacc_vx_i16m1(acc_0_1_0,aq_0_0,wq_1_0,HALF);
            int aq_0_1=*(const int8_t*)(al+OFF_AQ+(j*128+1*32+ll)*4+0);
            acc_0_0_1=__riscv_vwmacc_vx_i16m1(acc_0_0_1,aq_0_1,wq_0_1,HALF);
            acc_0_1_1=__riscv_vwmacc_vx_i16m1(acc_0_1_1,aq_0_1,wq_1_1,HALF);
            int aq_0_2=*(const int8_t*)(al+OFF_AQ+(j*128+2*32+ll)*4+0);
            acc_0_0_2=__riscv_vwmacc_vx_i16m1(acc_0_0_2,aq_0_2,wq_0_2,HALF);
            acc_0_1_2=__riscv_vwmacc_vx_i16m1(acc_0_1_2,aq_0_2,wq_1_2,HALF);
            int aq_0_3=*(const int8_t*)(al+OFF_AQ+(j*128+3*32+ll)*4+0);
            acc_0_0_3=__riscv_vwmacc_vx_i16m1(acc_0_0_3,aq_0_3,wq_0_3,HALF);
            acc_0_1_3=__riscv_vwmacc_vx_i16m1(acc_0_1_3,aq_0_3,wq_1_3,HALF);
            int aq_1_0=*(const int8_t*)(al+OFF_AQ+(j*128+0*32+ll)*4+1);
            acc_1_0_0=__riscv_vwmacc_vx_i16m1(acc_1_0_0,aq_1_0,wq_0_0,HALF);
            acc_1_1_0=__riscv_vwmacc_vx_i16m1(acc_1_1_0,aq_1_0,wq_1_0,HALF);
            int aq_1_1=*(const int8_t*)(al+OFF_AQ+(j*128+1*32+ll)*4+1);
            acc_1_0_1=__riscv_vwmacc_vx_i16m1(acc_1_0_1,aq_1_1,wq_0_1,HALF);
            acc_1_1_1=__riscv_vwmacc_vx_i16m1(acc_1_1_1,aq_1_1,wq_1_1,HALF);
            int aq_1_2=*(const int8_t*)(al+OFF_AQ+(j*128+2*32+ll)*4+1);
            acc_1_0_2=__riscv_vwmacc_vx_i16m1(acc_1_0_2,aq_1_2,wq_0_2,HALF);
            acc_1_1_2=__riscv_vwmacc_vx_i16m1(acc_1_1_2,aq_1_2,wq_1_2,HALF);
            int aq_1_3=*(const int8_t*)(al+OFF_AQ+(j*128+3*32+ll)*4+1);
            acc_1_0_3=__riscv_vwmacc_vx_i16m1(acc_1_0_3,aq_1_3,wq_0_3,HALF);
            acc_1_1_3=__riscv_vwmacc_vx_i16m1(acc_1_1_3,aq_1_3,wq_1_3,HALF);
            int aq_2_0=*(const int8_t*)(al+OFF_AQ+(j*128+0*32+ll)*4+2);
            acc_2_0_0=__riscv_vwmacc_vx_i16m1(acc_2_0_0,aq_2_0,wq_0_0,HALF);
            acc_2_1_0=__riscv_vwmacc_vx_i16m1(acc_2_1_0,aq_2_0,wq_1_0,HALF);
            int aq_2_1=*(const int8_t*)(al+OFF_AQ+(j*128+1*32+ll)*4+2);
            acc_2_0_1=__riscv_vwmacc_vx_i16m1(acc_2_0_1,aq_2_1,wq_0_1,HALF);
            acc_2_1_1=__riscv_vwmacc_vx_i16m1(acc_2_1_1,aq_2_1,wq_1_1,HALF);
            int aq_2_2=*(const int8_t*)(al+OFF_AQ+(j*128+2*32+ll)*4+2);
            acc_2_0_2=__riscv_vwmacc_vx_i16m1(acc_2_0_2,aq_2_2,wq_0_2,HALF);
            acc_2_1_2=__riscv_vwmacc_vx_i16m1(acc_2_1_2,aq_2_2,wq_1_2,HALF);
            int aq_2_3=*(const int8_t*)(al+OFF_AQ+(j*128+3*32+ll)*4+2);
            acc_2_0_3=__riscv_vwmacc_vx_i16m1(acc_2_0_3,aq_2_3,wq_0_3,HALF);
            acc_2_1_3=__riscv_vwmacc_vx_i16m1(acc_2_1_3,aq_2_3,wq_1_3,HALF);
            int aq_3_0=*(const int8_t*)(al+OFF_AQ+(j*128+0*32+ll)*4+3);
            acc_3_0_0=__riscv_vwmacc_vx_i16m1(acc_3_0_0,aq_3_0,wq_0_0,HALF);
            acc_3_1_0=__riscv_vwmacc_vx_i16m1(acc_3_1_0,aq_3_0,wq_1_0,HALF);
            int aq_3_1=*(const int8_t*)(al+OFF_AQ+(j*128+1*32+ll)*4+3);
            acc_3_0_1=__riscv_vwmacc_vx_i16m1(acc_3_0_1,aq_3_1,wq_0_1,HALF);
            acc_3_1_1=__riscv_vwmacc_vx_i16m1(acc_3_1_1,aq_3_1,wq_1_1,HALF);
            int aq_3_2=*(const int8_t*)(al+OFF_AQ+(j*128+2*32+ll)*4+3);
            acc_3_0_2=__riscv_vwmacc_vx_i16m1(acc_3_0_2,aq_3_2,wq_0_2,HALF);
            acc_3_1_2=__riscv_vwmacc_vx_i16m1(acc_3_1_2,aq_3_2,wq_1_2,HALF);
            int aq_3_3=*(const int8_t*)(al+OFF_AQ+(j*128+3*32+ll)*4+3);
            acc_3_0_3=__riscv_vwmacc_vx_i16m1(acc_3_0_3,aq_3_3,wq_0_3,HALF);
            acc_3_1_3=__riscv_vwmacc_vx_i16m1(acc_3_1_3,aq_3_3,wq_1_3,HALF);
          }
          {
            sumi_0_0=__riscv_vwmacc_vv_i32m2(sumi_0_0,sc_0_0,acc_0_0_0,HALF);
            sumi_0_0=__riscv_vwmacc_vv_i32m2(sumi_0_0,sc_0_1,acc_0_0_1,HALF);
            sumi_0_0=__riscv_vwmacc_vv_i32m2(sumi_0_0,sc_0_2,acc_0_0_2,HALF);
            sumi_0_0=__riscv_vwmacc_vv_i32m2(sumi_0_0,sc_0_3,acc_0_0_3,HALF);
          }
          {
            sumi_0_1=__riscv_vwmacc_vv_i32m2(sumi_0_1,sc_1_0,acc_0_1_0,HALF);
            sumi_0_1=__riscv_vwmacc_vv_i32m2(sumi_0_1,sc_1_1,acc_0_1_1,HALF);
            sumi_0_1=__riscv_vwmacc_vv_i32m2(sumi_0_1,sc_1_2,acc_0_1_2,HALF);
            sumi_0_1=__riscv_vwmacc_vv_i32m2(sumi_0_1,sc_1_3,acc_0_1_3,HALF);
          }
          {
            sumi_1_0=__riscv_vwmacc_vv_i32m2(sumi_1_0,sc_0_0,acc_1_0_0,HALF);
            sumi_1_0=__riscv_vwmacc_vv_i32m2(sumi_1_0,sc_0_1,acc_1_0_1,HALF);
            sumi_1_0=__riscv_vwmacc_vv_i32m2(sumi_1_0,sc_0_2,acc_1_0_2,HALF);
            sumi_1_0=__riscv_vwmacc_vv_i32m2(sumi_1_0,sc_0_3,acc_1_0_3,HALF);
          }
          {
            sumi_1_1=__riscv_vwmacc_vv_i32m2(sumi_1_1,sc_1_0,acc_1_1_0,HALF);
            sumi_1_1=__riscv_vwmacc_vv_i32m2(sumi_1_1,sc_1_1,acc_1_1_1,HALF);
            sumi_1_1=__riscv_vwmacc_vv_i32m2(sumi_1_1,sc_1_2,acc_1_1_2,HALF);
            sumi_1_1=__riscv_vwmacc_vv_i32m2(sumi_1_1,sc_1_3,acc_1_1_3,HALF);
          }
          {
            sumi_2_0=__riscv_vwmacc_vv_i32m2(sumi_2_0,sc_0_0,acc_2_0_0,HALF);
            sumi_2_0=__riscv_vwmacc_vv_i32m2(sumi_2_0,sc_0_1,acc_2_0_1,HALF);
            sumi_2_0=__riscv_vwmacc_vv_i32m2(sumi_2_0,sc_0_2,acc_2_0_2,HALF);
            sumi_2_0=__riscv_vwmacc_vv_i32m2(sumi_2_0,sc_0_3,acc_2_0_3,HALF);
          }
          {
            sumi_2_1=__riscv_vwmacc_vv_i32m2(sumi_2_1,sc_1_0,acc_2_1_0,HALF);
            sumi_2_1=__riscv_vwmacc_vv_i32m2(sumi_2_1,sc_1_1,acc_2_1_1,HALF);
            sumi_2_1=__riscv_vwmacc_vv_i32m2(sumi_2_1,sc_1_2,acc_2_1_2,HALF);
            sumi_2_1=__riscv_vwmacc_vv_i32m2(sumi_2_1,sc_1_3,acc_2_1_3,HALF);
          }
          {
            sumi_3_0=__riscv_vwmacc_vv_i32m2(sumi_3_0,sc_0_0,acc_3_0_0,HALF);
            sumi_3_0=__riscv_vwmacc_vv_i32m2(sumi_3_0,sc_0_1,acc_3_0_1,HALF);
            sumi_3_0=__riscv_vwmacc_vv_i32m2(sumi_3_0,sc_0_2,acc_3_0_2,HALF);
            sumi_3_0=__riscv_vwmacc_vv_i32m2(sumi_3_0,sc_0_3,acc_3_0_3,HALF);
          }
          {
            sumi_3_1=__riscv_vwmacc_vv_i32m2(sumi_3_1,sc_1_0,acc_3_1_0,HALF);
            sumi_3_1=__riscv_vwmacc_vv_i32m2(sumi_3_1,sc_1_1,acc_3_1_1,HALF);
            sumi_3_1=__riscv_vwmacc_vv_i32m2(sumi_3_1,sc_1_2,acc_3_1_2,HALF);
            sumi_3_1=__riscv_vwmacc_vv_i32m2(sumi_3_1,sc_1_3,acc_3_1_3,HALF);
          }
        }
      }
    }
    sumf_0_0=__riscv_vfmacc_vv_f32m2(sumf_0_0,__riscv_vfcvt_f_x_v_f32m2(sumi_0_0,HALF),__riscv_vfmul_vf_f32m2(dF0,aD0,HALF),HALF);
    sumf_0_1=__riscv_vfmacc_vv_f32m2(sumf_0_1,__riscv_vfcvt_f_x_v_f32m2(sumi_0_1,HALF),__riscv_vfmul_vf_f32m2(dF1,aD0,HALF),HALF);
    sumf_1_0=__riscv_vfmacc_vv_f32m2(sumf_1_0,__riscv_vfcvt_f_x_v_f32m2(sumi_1_0,HALF),__riscv_vfmul_vf_f32m2(dF0,aD1,HALF),HALF);
    sumf_1_1=__riscv_vfmacc_vv_f32m2(sumf_1_1,__riscv_vfcvt_f_x_v_f32m2(sumi_1_1,HALF),__riscv_vfmul_vf_f32m2(dF1,aD1,HALF),HALF);
    sumf_2_0=__riscv_vfmacc_vv_f32m2(sumf_2_0,__riscv_vfcvt_f_x_v_f32m2(sumi_2_0,HALF),__riscv_vfmul_vf_f32m2(dF0,aD2,HALF),HALF);
    sumf_2_1=__riscv_vfmacc_vv_f32m2(sumf_2_1,__riscv_vfcvt_f_x_v_f32m2(sumi_2_1,HALF),__riscv_vfmul_vf_f32m2(dF1,aD2,HALF),HALF);
    sumf_3_0=__riscv_vfmacc_vv_f32m2(sumf_3_0,__riscv_vfcvt_f_x_v_f32m2(sumi_3_0,HALF),__riscv_vfmul_vf_f32m2(dF0,aD3,HALF),HALF);
    sumf_3_1=__riscv_vfmacc_vv_f32m2(sumf_3_1,__riscv_vfcvt_f_x_v_f32m2(sumi_3_1,HALF),__riscv_vfmul_vf_f32m2(dF1,aD3,HALF),HALF);
      }
      __riscv_vse32_v_f32m2(s+(y*4+0)*bs+x*16+0,sumf_0_0,HALF);
      __riscv_vse32_v_f32m2(s+(y*4+0)*bs+x*16+8,sumf_0_1,HALF);
      __riscv_vse32_v_f32m2(s+(y*4+1)*bs+x*16+0,sumf_1_0,HALF);
      __riscv_vse32_v_f32m2(s+(y*4+1)*bs+x*16+8,sumf_1_1,HALF);
      __riscv_vse32_v_f32m2(s+(y*4+2)*bs+x*16+0,sumf_2_0,HALF);
      __riscv_vse32_v_f32m2(s+(y*4+2)*bs+x*16+8,sumf_2_1,HALF);
      __riscv_vse32_v_f32m2(s+(y*4+3)*bs+x*16+0,sumf_3_0,HALF);
      __riscv_vse32_v_f32m2(s+(y*4+3)*bs+x*16+8,sumf_3_1,HALF);
    }
  }
}

void q6k_gemm_unrolled(size_t n,float*s,const uint8_t*vx,const uint8_t*vy,size_t nr,size_t nc,size_t bs){
  size_t nb=n/256, nrg=nr/4, ncg=nc/16;
  for(size_t y=0;y<nrg;++y){
    const uint8_t*aG=vy+y*nb*ASTRIDE;
    for(size_t x=0;x<ncg;++x){
      const uint8_t*bG=vx+x*nb*WSTRIDE;
      vfloat32m2_t sumf_0_0=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
      vfloat32m2_t sumf_0_1=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
      vfloat32m2_t sumf_1_0=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
      vfloat32m2_t sumf_1_1=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
      vfloat32m2_t sumf_2_0=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
      vfloat32m2_t sumf_2_1=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
      vfloat32m2_t sumf_3_0=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
      vfloat32m2_t sumf_3_1=__riscv_vfmv_v_f_f32m2(0.0f,HALF);
      #pragma GCC unroll 1
      for(size_t l=0;l<nb;++l){
    const uint8_t* bl=bG+l*WSTRIDE;
    const uint8_t* al=aG+l*ASTRIDE;
    float aD0=*(const float*)(al+0);
    float aD1=*(const float*)(al+4);
    float aD2=*(const float*)(al+8);
    float aD3=*(const float*)(al+12);
    vfloat32m2_t dF0=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1((const _Float16*)(bl+0),HALF),HALF);
    vfloat32m2_t dF1=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1((const _Float16*)(bl+16),HALF),HALF);
    vint32m2_t sumi_0_0=__riscv_vmv_v_x_i32m2(0,HALF);
    vint32m2_t sumi_0_1=__riscv_vmv_v_x_i32m2(0,HALF);
    vint32m2_t sumi_1_0=__riscv_vmv_v_x_i32m2(0,HALF);
    vint32m2_t sumi_1_1=__riscv_vmv_v_x_i32m2(0,HALF);
    vint32m2_t sumi_2_0=__riscv_vmv_v_x_i32m2(0,HALF);
    vint32m2_t sumi_2_1=__riscv_vmv_v_x_i32m2(0,HALF);
    vint32m2_t sumi_3_0=__riscv_vmv_v_x_i32m2(0,HALF);
    vint32m2_t sumi_3_1=__riscv_vmv_v_x_i32m2(0,HALF);
    _Pragma("GCC unroll 64")
    for(int j=0;j<2;++j){
      _Pragma("GCC unroll 64")
      for(int sh=0;sh<2;++sh){
        vint16m1_t sc_0_0=liftScale(ldI8(bl,OFF_SCALES+(j*8+0*2+sh)*16+0));
        vint16m1_t sc_0_1=liftScale(ldI8(bl,OFF_SCALES+(j*8+1*2+sh)*16+0));
        vint16m1_t sc_0_2=liftScale(ldI8(bl,OFF_SCALES+(j*8+2*2+sh)*16+0));
        vint16m1_t sc_0_3=liftScale(ldI8(bl,OFF_SCALES+(j*8+3*2+sh)*16+0));
        vint16m1_t sc_1_0=liftScale(ldI8(bl,OFF_SCALES+(j*8+0*2+sh)*16+8));
        vint16m1_t sc_1_1=liftScale(ldI8(bl,OFF_SCALES+(j*8+1*2+sh)*16+8));
        vint16m1_t sc_1_2=liftScale(ldI8(bl,OFF_SCALES+(j*8+2*2+sh)*16+8));
        vint16m1_t sc_1_3=liftScale(ldI8(bl,OFF_SCALES+(j*8+3*2+sh)*16+8));
        _Pragma("GCC unroll 64")
        for(int k=0;k<2;++k){
          vint16m1_t acc_0_0_0=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_0_0_1=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_0_0_2=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_0_0_3=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_0_1_0=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_0_1_1=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_0_1_2=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_0_1_3=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_1_0_0=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_1_0_1=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_1_0_2=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_1_0_3=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_1_1_0=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_1_1_1=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_1_1_2=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_1_1_3=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_2_0_0=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_2_0_1=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_2_0_2=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_2_0_3=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_2_1_0=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_2_1_1=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_2_1_2=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_2_1_3=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_3_0_0=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_3_0_1=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_3_0_2=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_3_0_3=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_3_1_0=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_3_1_1=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_3_1_2=__riscv_vmv_v_x_i16m1(0,HALF);
          vint16m1_t acc_3_1_3=__riscv_vmv_v_x_i16m1(0,HALF);
          _Pragma("GCC unroll 64")
          for(int p=0;p<8;++p){
            int ll=sh*16+k*8+p;
            vuint8mf2_t qlA0=ldU8(bl,OFF_QL+(j*64+ll)*16+0);
            vuint8mf2_t qlB0=ldU8(bl,OFF_QL+(j*64+32+ll)*16+0);
            vuint8mf2_t qh0=ldU8(bl,OFF_QH+(j*32+ll)*16+0);
            vint8mf2_t wq_0_0=asmW(qlA0,qh0,Q_HIGH[0],Q_SHIFT[0]);
            vint8mf2_t wq_0_1=asmW(qlB0,qh0,Q_HIGH[1],Q_SHIFT[1]);
            vint8mf2_t wq_0_2=asmW(qlA0,qh0,Q_HIGH[2],Q_SHIFT[2]);
            vint8mf2_t wq_0_3=asmW(qlB0,qh0,Q_HIGH[3],Q_SHIFT[3]);
            vuint8mf2_t qlA1=ldU8(bl,OFF_QL+(j*64+ll)*16+8);
            vuint8mf2_t qlB1=ldU8(bl,OFF_QL+(j*64+32+ll)*16+8);
            vuint8mf2_t qh1=ldU8(bl,OFF_QH+(j*32+ll)*16+8);
            vint8mf2_t wq_1_0=asmW(qlA1,qh1,Q_HIGH[0],Q_SHIFT[0]);
            vint8mf2_t wq_1_1=asmW(qlB1,qh1,Q_HIGH[1],Q_SHIFT[1]);
            vint8mf2_t wq_1_2=asmW(qlA1,qh1,Q_HIGH[2],Q_SHIFT[2]);
            vint8mf2_t wq_1_3=asmW(qlB1,qh1,Q_HIGH[3],Q_SHIFT[3]);
            int aq_0_0=*(const int8_t*)(al+OFF_AQ+(j*128+0*32+ll)*4+0);
            acc_0_0_0=__riscv_vwmacc_vx_i16m1(acc_0_0_0,aq_0_0,wq_0_0,HALF);
            acc_0_1_0=__riscv_vwmacc_vx_i16m1(acc_0_1_0,aq_0_0,wq_1_0,HALF);
            int aq_0_1=*(const int8_t*)(al+OFF_AQ+(j*128+1*32+ll)*4+0);
            acc_0_0_1=__riscv_vwmacc_vx_i16m1(acc_0_0_1,aq_0_1,wq_0_1,HALF);
            acc_0_1_1=__riscv_vwmacc_vx_i16m1(acc_0_1_1,aq_0_1,wq_1_1,HALF);
            int aq_0_2=*(const int8_t*)(al+OFF_AQ+(j*128+2*32+ll)*4+0);
            acc_0_0_2=__riscv_vwmacc_vx_i16m1(acc_0_0_2,aq_0_2,wq_0_2,HALF);
            acc_0_1_2=__riscv_vwmacc_vx_i16m1(acc_0_1_2,aq_0_2,wq_1_2,HALF);
            int aq_0_3=*(const int8_t*)(al+OFF_AQ+(j*128+3*32+ll)*4+0);
            acc_0_0_3=__riscv_vwmacc_vx_i16m1(acc_0_0_3,aq_0_3,wq_0_3,HALF);
            acc_0_1_3=__riscv_vwmacc_vx_i16m1(acc_0_1_3,aq_0_3,wq_1_3,HALF);
            int aq_1_0=*(const int8_t*)(al+OFF_AQ+(j*128+0*32+ll)*4+1);
            acc_1_0_0=__riscv_vwmacc_vx_i16m1(acc_1_0_0,aq_1_0,wq_0_0,HALF);
            acc_1_1_0=__riscv_vwmacc_vx_i16m1(acc_1_1_0,aq_1_0,wq_1_0,HALF);
            int aq_1_1=*(const int8_t*)(al+OFF_AQ+(j*128+1*32+ll)*4+1);
            acc_1_0_1=__riscv_vwmacc_vx_i16m1(acc_1_0_1,aq_1_1,wq_0_1,HALF);
            acc_1_1_1=__riscv_vwmacc_vx_i16m1(acc_1_1_1,aq_1_1,wq_1_1,HALF);
            int aq_1_2=*(const int8_t*)(al+OFF_AQ+(j*128+2*32+ll)*4+1);
            acc_1_0_2=__riscv_vwmacc_vx_i16m1(acc_1_0_2,aq_1_2,wq_0_2,HALF);
            acc_1_1_2=__riscv_vwmacc_vx_i16m1(acc_1_1_2,aq_1_2,wq_1_2,HALF);
            int aq_1_3=*(const int8_t*)(al+OFF_AQ+(j*128+3*32+ll)*4+1);
            acc_1_0_3=__riscv_vwmacc_vx_i16m1(acc_1_0_3,aq_1_3,wq_0_3,HALF);
            acc_1_1_3=__riscv_vwmacc_vx_i16m1(acc_1_1_3,aq_1_3,wq_1_3,HALF);
            int aq_2_0=*(const int8_t*)(al+OFF_AQ+(j*128+0*32+ll)*4+2);
            acc_2_0_0=__riscv_vwmacc_vx_i16m1(acc_2_0_0,aq_2_0,wq_0_0,HALF);
            acc_2_1_0=__riscv_vwmacc_vx_i16m1(acc_2_1_0,aq_2_0,wq_1_0,HALF);
            int aq_2_1=*(const int8_t*)(al+OFF_AQ+(j*128+1*32+ll)*4+2);
            acc_2_0_1=__riscv_vwmacc_vx_i16m1(acc_2_0_1,aq_2_1,wq_0_1,HALF);
            acc_2_1_1=__riscv_vwmacc_vx_i16m1(acc_2_1_1,aq_2_1,wq_1_1,HALF);
            int aq_2_2=*(const int8_t*)(al+OFF_AQ+(j*128+2*32+ll)*4+2);
            acc_2_0_2=__riscv_vwmacc_vx_i16m1(acc_2_0_2,aq_2_2,wq_0_2,HALF);
            acc_2_1_2=__riscv_vwmacc_vx_i16m1(acc_2_1_2,aq_2_2,wq_1_2,HALF);
            int aq_2_3=*(const int8_t*)(al+OFF_AQ+(j*128+3*32+ll)*4+2);
            acc_2_0_3=__riscv_vwmacc_vx_i16m1(acc_2_0_3,aq_2_3,wq_0_3,HALF);
            acc_2_1_3=__riscv_vwmacc_vx_i16m1(acc_2_1_3,aq_2_3,wq_1_3,HALF);
            int aq_3_0=*(const int8_t*)(al+OFF_AQ+(j*128+0*32+ll)*4+3);
            acc_3_0_0=__riscv_vwmacc_vx_i16m1(acc_3_0_0,aq_3_0,wq_0_0,HALF);
            acc_3_1_0=__riscv_vwmacc_vx_i16m1(acc_3_1_0,aq_3_0,wq_1_0,HALF);
            int aq_3_1=*(const int8_t*)(al+OFF_AQ+(j*128+1*32+ll)*4+3);
            acc_3_0_1=__riscv_vwmacc_vx_i16m1(acc_3_0_1,aq_3_1,wq_0_1,HALF);
            acc_3_1_1=__riscv_vwmacc_vx_i16m1(acc_3_1_1,aq_3_1,wq_1_1,HALF);
            int aq_3_2=*(const int8_t*)(al+OFF_AQ+(j*128+2*32+ll)*4+3);
            acc_3_0_2=__riscv_vwmacc_vx_i16m1(acc_3_0_2,aq_3_2,wq_0_2,HALF);
            acc_3_1_2=__riscv_vwmacc_vx_i16m1(acc_3_1_2,aq_3_2,wq_1_2,HALF);
            int aq_3_3=*(const int8_t*)(al+OFF_AQ+(j*128+3*32+ll)*4+3);
            acc_3_0_3=__riscv_vwmacc_vx_i16m1(acc_3_0_3,aq_3_3,wq_0_3,HALF);
            acc_3_1_3=__riscv_vwmacc_vx_i16m1(acc_3_1_3,aq_3_3,wq_1_3,HALF);
          }
          {
            sumi_0_0=__riscv_vwmacc_vv_i32m2(sumi_0_0,sc_0_0,acc_0_0_0,HALF);
            sumi_0_0=__riscv_vwmacc_vv_i32m2(sumi_0_0,sc_0_1,acc_0_0_1,HALF);
            sumi_0_0=__riscv_vwmacc_vv_i32m2(sumi_0_0,sc_0_2,acc_0_0_2,HALF);
            sumi_0_0=__riscv_vwmacc_vv_i32m2(sumi_0_0,sc_0_3,acc_0_0_3,HALF);
          }
          {
            sumi_0_1=__riscv_vwmacc_vv_i32m2(sumi_0_1,sc_1_0,acc_0_1_0,HALF);
            sumi_0_1=__riscv_vwmacc_vv_i32m2(sumi_0_1,sc_1_1,acc_0_1_1,HALF);
            sumi_0_1=__riscv_vwmacc_vv_i32m2(sumi_0_1,sc_1_2,acc_0_1_2,HALF);
            sumi_0_1=__riscv_vwmacc_vv_i32m2(sumi_0_1,sc_1_3,acc_0_1_3,HALF);
          }
          {
            sumi_1_0=__riscv_vwmacc_vv_i32m2(sumi_1_0,sc_0_0,acc_1_0_0,HALF);
            sumi_1_0=__riscv_vwmacc_vv_i32m2(sumi_1_0,sc_0_1,acc_1_0_1,HALF);
            sumi_1_0=__riscv_vwmacc_vv_i32m2(sumi_1_0,sc_0_2,acc_1_0_2,HALF);
            sumi_1_0=__riscv_vwmacc_vv_i32m2(sumi_1_0,sc_0_3,acc_1_0_3,HALF);
          }
          {
            sumi_1_1=__riscv_vwmacc_vv_i32m2(sumi_1_1,sc_1_0,acc_1_1_0,HALF);
            sumi_1_1=__riscv_vwmacc_vv_i32m2(sumi_1_1,sc_1_1,acc_1_1_1,HALF);
            sumi_1_1=__riscv_vwmacc_vv_i32m2(sumi_1_1,sc_1_2,acc_1_1_2,HALF);
            sumi_1_1=__riscv_vwmacc_vv_i32m2(sumi_1_1,sc_1_3,acc_1_1_3,HALF);
          }
          {
            sumi_2_0=__riscv_vwmacc_vv_i32m2(sumi_2_0,sc_0_0,acc_2_0_0,HALF);
            sumi_2_0=__riscv_vwmacc_vv_i32m2(sumi_2_0,sc_0_1,acc_2_0_1,HALF);
            sumi_2_0=__riscv_vwmacc_vv_i32m2(sumi_2_0,sc_0_2,acc_2_0_2,HALF);
            sumi_2_0=__riscv_vwmacc_vv_i32m2(sumi_2_0,sc_0_3,acc_2_0_3,HALF);
          }
          {
            sumi_2_1=__riscv_vwmacc_vv_i32m2(sumi_2_1,sc_1_0,acc_2_1_0,HALF);
            sumi_2_1=__riscv_vwmacc_vv_i32m2(sumi_2_1,sc_1_1,acc_2_1_1,HALF);
            sumi_2_1=__riscv_vwmacc_vv_i32m2(sumi_2_1,sc_1_2,acc_2_1_2,HALF);
            sumi_2_1=__riscv_vwmacc_vv_i32m2(sumi_2_1,sc_1_3,acc_2_1_3,HALF);
          }
          {
            sumi_3_0=__riscv_vwmacc_vv_i32m2(sumi_3_0,sc_0_0,acc_3_0_0,HALF);
            sumi_3_0=__riscv_vwmacc_vv_i32m2(sumi_3_0,sc_0_1,acc_3_0_1,HALF);
            sumi_3_0=__riscv_vwmacc_vv_i32m2(sumi_3_0,sc_0_2,acc_3_0_2,HALF);
            sumi_3_0=__riscv_vwmacc_vv_i32m2(sumi_3_0,sc_0_3,acc_3_0_3,HALF);
          }
          {
            sumi_3_1=__riscv_vwmacc_vv_i32m2(sumi_3_1,sc_1_0,acc_3_1_0,HALF);
            sumi_3_1=__riscv_vwmacc_vv_i32m2(sumi_3_1,sc_1_1,acc_3_1_1,HALF);
            sumi_3_1=__riscv_vwmacc_vv_i32m2(sumi_3_1,sc_1_2,acc_3_1_2,HALF);
            sumi_3_1=__riscv_vwmacc_vv_i32m2(sumi_3_1,sc_1_3,acc_3_1_3,HALF);
          }
        }
      }
    }
    sumf_0_0=__riscv_vfmacc_vv_f32m2(sumf_0_0,__riscv_vfcvt_f_x_v_f32m2(sumi_0_0,HALF),__riscv_vfmul_vf_f32m2(dF0,aD0,HALF),HALF);
    sumf_0_1=__riscv_vfmacc_vv_f32m2(sumf_0_1,__riscv_vfcvt_f_x_v_f32m2(sumi_0_1,HALF),__riscv_vfmul_vf_f32m2(dF1,aD0,HALF),HALF);
    sumf_1_0=__riscv_vfmacc_vv_f32m2(sumf_1_0,__riscv_vfcvt_f_x_v_f32m2(sumi_1_0,HALF),__riscv_vfmul_vf_f32m2(dF0,aD1,HALF),HALF);
    sumf_1_1=__riscv_vfmacc_vv_f32m2(sumf_1_1,__riscv_vfcvt_f_x_v_f32m2(sumi_1_1,HALF),__riscv_vfmul_vf_f32m2(dF1,aD1,HALF),HALF);
    sumf_2_0=__riscv_vfmacc_vv_f32m2(sumf_2_0,__riscv_vfcvt_f_x_v_f32m2(sumi_2_0,HALF),__riscv_vfmul_vf_f32m2(dF0,aD2,HALF),HALF);
    sumf_2_1=__riscv_vfmacc_vv_f32m2(sumf_2_1,__riscv_vfcvt_f_x_v_f32m2(sumi_2_1,HALF),__riscv_vfmul_vf_f32m2(dF1,aD2,HALF),HALF);
    sumf_3_0=__riscv_vfmacc_vv_f32m2(sumf_3_0,__riscv_vfcvt_f_x_v_f32m2(sumi_3_0,HALF),__riscv_vfmul_vf_f32m2(dF0,aD3,HALF),HALF);
    sumf_3_1=__riscv_vfmacc_vv_f32m2(sumf_3_1,__riscv_vfcvt_f_x_v_f32m2(sumi_3_1,HALF),__riscv_vfmul_vf_f32m2(dF1,aD3,HALF),HALF);
      }
      __riscv_vse32_v_f32m2(s+(y*4+0)*bs+x*16+0,sumf_0_0,HALF);
      __riscv_vse32_v_f32m2(s+(y*4+0)*bs+x*16+8,sumf_0_1,HALF);
      __riscv_vse32_v_f32m2(s+(y*4+1)*bs+x*16+0,sumf_1_0,HALF);
      __riscv_vse32_v_f32m2(s+(y*4+1)*bs+x*16+8,sumf_1_1,HALF);
      __riscv_vse32_v_f32m2(s+(y*4+2)*bs+x*16+0,sumf_2_0,HALF);
      __riscv_vse32_v_f32m2(s+(y*4+2)*bs+x*16+8,sumf_2_1,HALF);
      __riscv_vse32_v_f32m2(s+(y*4+3)*bs+x*16+0,sumf_3_0,HALF);
      __riscv_vse32_v_f32m2(s+(y*4+3)*bs+x*16+8,sumf_3_1,HALF);
    }
  }
}

#ifdef DRIVER
static uint16_t f16_one(void){return 0x3C00;}
int main(int argc,char**argv){
  unsigned seed=(argc>1)?(unsigned)strtoul(argv[1],0,0):0xC0FFEEu; srand(seed);
  size_t nb=3,nrg=1,ncg=1,nr=nrg*4,nc=ncg*16,n=nb*256,bs=nc;
  size_t wbytes=ncg*nb*WSTRIDE, abytes=nrg*nb*ASTRIDE;
  uint8_t*vx=(uint8_t*)malloc(wbytes),*vy=(uint8_t*)malloc(abytes);
  for(size_t i=0;i<wbytes;++i)vx[i]=(uint8_t)rand();
  for(size_t i=0;i<abytes;++i)vy[i]=(uint8_t)rand();
  for(size_t x=0;x<ncg;++x)for(size_t l=0;l<nb;++l){uint8_t*bl=vx+(x*nb+l)*WSTRIDE;uint16_t h=f16_one();memcpy(bl,&h,2);}
  for(size_t y=0;y<nrg;++y)for(size_t l=0;l<nb;++l){float d=0.5f;for(int c=0;c<4;++c)memcpy(vy+(y*nb+l)*ASTRIDE+c*4,&d,4);}
  float*sr=(float*)calloc(nr*nc,4),*su=(float*)calloc(nr*nc,4);
  q6k_gemm_rolled(n,sr,vx,vy,nr,nc,bs);
  q6k_gemm_unrolled(n,su,vx,vy,nr,nc,bs);
  int mm=memcmp(sr,su,nr*nc*4); int nz=0; for(size_t i=0;i<nr*nc;++i) if(sr[i]!=0.0f)nz++;
  printf("seed=0x%X nb=%zu nr=%zu nc=%zu : memcmp=%d (0=byte-exact) nonzero=%d/%zu\n",seed,nb,nr,nc,mm,nz,nr*nc);
  return mm==0?0:1;
}
#endif
