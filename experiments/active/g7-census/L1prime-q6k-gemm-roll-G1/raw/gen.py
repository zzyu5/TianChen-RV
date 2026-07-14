#!/usr/bin/env python3
# Generate q6_K GEMM whole-K-nest roll A/B with EXPLICIT named vector vars
# (RVV sizeless types forbid arrays). c/q/h are compile-time expanded; the
# super-block element nest j/sh/k/p is emitted as real C loops whose unroll is
# toggled by NEST(pragma). Rolled keeps the 32 i16 partials + 8 i32 sumi + 8 f32
# sumf RESIDENT across the runtime p-loop (whole-K-nest roll, no tile narrowing).
AI, NH = 4, 2

HEAD = r'''#include <riscv_vector.h>
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
'''

def body(nest_pragma):
    L=[]
    a=L.append
    a('    const uint8_t* bl=bG+l*WSTRIDE;')
    a('    const uint8_t* al=aG+l*ASTRIDE;')
    for c in range(AI): a(f'    float aD{c}=*(const float*)(al+{c*4});')
    for h in range(NH): a(f'    vfloat32m2_t dF{h}=__riscv_vfwcvt_f_f_v_f32m2(__riscv_vle16_v_f16m1((const _Float16*)(bl+{h*8*2}),HALF),HALF);')
    for c in range(AI):
        for h in range(NH): a(f'    vint32m2_t sumi_{c}_{h}=__riscv_vmv_v_x_i32m2(0,HALF);')
    a(f'    {nest_pragma}')
    a('    for(int j=0;j<2;++j){')
    a(f'      {nest_pragma}')
    a('      for(int sh=0;sh<2;++sh){')
    # scale strips sc[h][q]
    for h in range(NH):
        for q in range(4):
            a(f'        vint16m1_t sc_{h}_{q}=liftScale(ldI8(bl,OFF_SCALES+(j*8+{q}*2+sh)*16+{h*8}));')
    a(f'        {nest_pragma}')
    a('        for(int k=0;k<2;++k){')
    # 32 i16 partials RESIDENT across p-loop
    for c in range(AI):
        for h in range(NH):
            for q in range(4):
                a(f'          vint16m1_t acc_{c}_{h}_{q}=__riscv_vmv_v_x_i16m1(0,HALF);')
    a(f'          {nest_pragma}')
    a('          for(int p=0;p<8;++p){')
    a('            int ll=sh*16+k*8+p;')
    # decode wq[h][q] : shared across the 4 columns (decoded ONCE per element)
    for h in range(NH):
        a(f'            vuint8mf2_t qlA{h}=ldU8(bl,OFF_QL+(j*64+ll)*16+{h*8});')
        a(f'            vuint8mf2_t qlB{h}=ldU8(bl,OFF_QL+(j*64+32+ll)*16+{h*8});')
        a(f'            vuint8mf2_t qh{h}=ldU8(bl,OFF_QH+(j*32+ll)*16+{h*8});')
        for q in range(4):
            stream = 'qlA' if [0,1,0,1][q]==0 else 'qlB'
            a(f'            vint8mf2_t wq_{h}_{q}=asmW({stream}{h},qh{h},Q_HIGH[{q}],Q_SHIFT[{q}]);')
    # macc into 32 partials, weight shared across columns
    for c in range(AI):
        for q in range(4):
            a(f'            int aq_{c}_{q}=*(const int8_t*)(al+OFF_AQ+(j*128+{q}*32+ll)*4+{c});')
            for h in range(NH):
                a(f'            acc_{c}_{h}_{q}=__riscv_vwmacc_vx_i16m1(acc_{c}_{h}_{q},aq_{c}_{q},wq_{h}_{q},HALF);')
    a('          }')  # end p
    # scale fold into sumi
    for c in range(AI):
        for h in range(NH):
            terms=''.join(f'\n            sumi_{c}_{h}=__riscv_vwmacc_vv_i32m2(sumi_{c}_{h},sc_{h}_{q},acc_{c}_{h}_{q},HALF);' for q in range(4))
            a(f'          {{{terms}\n          }}')
    a('        }')  # end k
    a('      }')  # end sh
    a('    }')  # end j
    # end-of-block fold
    for c in range(AI):
        for h in range(NH):
            a(f'    sumf_{c}_{h}=__riscv_vfmacc_vv_f32m2(sumf_{c}_{h},__riscv_vfcvt_f_x_v_f32m2(sumi_{c}_{h},HALF),__riscv_vfmul_vf_f32m2(dF{h},aD{c},HALF),HALF);')
    return '\n'.join(L)

def func(name, nest_pragma):
    L=[]; a=L.append
    a(f'void {name}(size_t n,float*s,const uint8_t*vx,const uint8_t*vy,size_t nr,size_t nc,size_t bs){{')
    a('  size_t nb=n/256, nrg=nr/4, ncg=nc/16;')
    a('  for(size_t y=0;y<nrg;++y){')
    a('    const uint8_t*aG=vy+y*nb*ASTRIDE;')
    a('    for(size_t x=0;x<ncg;++x){')
    a('      const uint8_t*bG=vx+x*nb*WSTRIDE;')
    for c in range(AI):
        for h in range(NH): a(f'      vfloat32m2_t sumf_{c}_{h}=__riscv_vfmv_v_f_f32m2(0.0f,HALF);')
    a('      #pragma GCC unroll 1')
    a('      for(size_t l=0;l<nb;++l){')
    a(body(nest_pragma))
    a('      }')
    for c in range(AI):
        for h in range(NH): a(f'      __riscv_vse32_v_f32m2(s+(y*4+{c})*bs+x*16+{h*8},sumf_{c}_{h},HALF);')
    a('    }')
    a('  }')
    a('}')
    return '\n'.join(L)

DRIVER = r'''
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
'''

with open('q6k_gemm_roll_gen.c','w') as f:
    f.write(HEAD+'\n')
    f.write(func('q6k_gemm_rolled','_Pragma("GCC unroll 1")')+'\n\n')
    f.write(func('q6k_gemm_unrolled','_Pragma("GCC unroll 64")')+'\n')
    f.write(DRIVER)
print("generated q6k_gemm_roll_gen.c")
