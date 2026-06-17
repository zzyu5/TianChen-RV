#include <cstdint>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <vector>
#include <riscv_vector.h>
static const int QK=32,Q4=18,Q8=34;
static inline float h2f(uint16_t h){_Float16 x;memcpy(&x,&h,2);return (float)x;}
static inline uint16_t f2h(float f){_Float16 x=(_Float16)f;uint16_t h;memcpy(&h,&x,2);return h;}
extern "C" void tcrv_emitc_gemm_m4_gemm_m4(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t,size_t,size_t);
extern "C" void tcrv_emitc_gemm_m6_gemm_m6(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t,size_t,size_t);
static float vd(int n,const uint8_t*vx,const uint8_t*vy){int nb=n/QK;float s=0;size_t vl=16;
 for(int ib=0;ib<nb;ib++){const uint8_t*xb=vx+(size_t)ib*Q4,*yb=vy+(size_t)ib*Q8;
  vuint8m1_t tx=__riscv_vle8_v_u8m1(xb+2,vl);vint8m1_t y0=__riscv_vle8_v_i8m1((const int8_t*)(yb+2),vl),y1=__riscv_vle8_v_i8m1((const int8_t*)(yb+18),vl);
  vuint8m1_t a=__riscv_vand_vx_u8m1(tx,0x0F,vl),l=__riscv_vsrl_vx_u8m1(tx,4,vl);
  vint8m1_t v0=__riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(a),8,vl),v1=__riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(l),8,vl);
  vint16m2_t p=__riscv_vwmul_vv_i16m2(v0,y0,vl);p=__riscv_vwmacc_vv_i16m2(p,v1,y1,vl);
  vint32m1_t sd=__riscv_vmv_v_x_i32m1(0,1),r=__riscv_vwredsum_vs_i16m2_i32m1(p,sd,vl);int32_t si=__riscv_vmv_x_s_i32m1_i32(r);
  s=s+((float)si*h2f(*(const uint16_t*)xb))*h2f(*(const uint16_t*)yb);}return s;}
static uint64_t R=0x123456789abcdef0ULL;static uint32_t xr(){R^=R<<13;R^=R>>7;R^=R<<17;return R>>32;}
static void fw(uint8_t*r,int nb){for(int i=0;i<nb;i++){uint8_t*b=r+(size_t)i*Q4;*(uint16_t*)b=f2h(((int)(xr()%2000)-1000)/4096.0f);for(int j=0;j<16;j++)b[2+j]=xr()&0xFF;}}
static void fc(uint8_t*c,int nb){for(int i=0;i<nb;i++){uint8_t*b=c+(size_t)i*Q8;*(uint16_t*)b=f2h(((int)(xr()%2000)-1000)/8192.0f);for(int j=0;j<32;j++)b[2+j]=(int8_t)(xr()&0xFF);}}
static double ns(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
typedef void(*fn)(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t,size_t,size_t);
static void bench(fn f,const char*tag,int n,int nr,int nc,int reps){int nb=n/QK;size_t bx=(size_t)nb*Q4,by=(size_t)nb*Q8,bs=nc;
 std::vector<uint8_t>W((size_t)nr*bx),A((size_t)nc*by);std::vector<float>o((size_t)nr*nc);
 for(int r=0;r<nr;r++)fw(W.data()+(size_t)r*bx,nb);for(int c=0;c<nc;c++)fc(A.data()+(size_t)c*by,nb);
 f(n,o.data(),W.data(),A.data(),by,nr,nc,bx,bs);double bg=1e30,bb=1e30;volatile float sk=0;
 for(int r=0;r<reps;r++){double t=ns();f(n,o.data(),W.data(),A.data(),by,nr,nc,bx,bs);double e=ns()-t;if(e<bg)bg=e;sk+=o[0];}
 for(int r=0;r<reps;r++){double t=ns();for(int i=0;i<nr;i++)for(int c=0;c<nc;c++)o[(size_t)i*nc+c]=vd(n,W.data()+(size_t)i*bx,A.data()+(size_t)c*by);double e=ns()-t;if(e<bb)bb=e;sk+=o[0];}
 double O=(double)nr*nc;printf("  [%s] n=%d nr=%d nc=%d: GEMM %.1f | vec_dot %.1f | speedup %.3fx\n",tag,n,nr,nc,bg/O,bb/O,bb/bg);}
int main(){
 bench(tcrv_emitc_gemm_m4_gemm_m4,"M4",4096,8,4,4000);
 bench(tcrv_emitc_gemm_m4_gemm_m4,"M4",4096,8,8,4000);
 bench(tcrv_emitc_gemm_m6_gemm_m6,"M6",4096,8,6,4000);
 bench(tcrv_emitc_gemm_m6_gemm_m6,"M6",4096,8,12,3000);
 return 0;}
