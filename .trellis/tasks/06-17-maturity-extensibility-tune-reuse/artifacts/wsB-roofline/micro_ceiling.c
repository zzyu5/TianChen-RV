/* WS-B roofline anchors: peak integer widening-MAC throughput + memory bandwidth.
 * Real ssh rvv (riscv64, VLEN=128). rdcycle primary; MONOTONIC for freq.
 * Compute anchor: vwmul i8xi8 -> i16 then vwadd.wv -> i32 (the emitted byte dot core) at LMUL m2 (src) / m8 (acc) and m1/m4.
 * Accumulator sweep A=1,2,4,8 directly tests the latency-hiding knob:
 *   A>1 throughput == A=1  => per-iter MAC chain NOT latency-bound (knob no help).
 * RVV vector types are sizeless -> accumulators are named vars (no arrays).
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <riscv_vector.h>

static inline uint64_t rdcycle(void){ uint64_t c; asm volatile("rdcycle %0":"=r"(c)); return c; }
static double freq_ghz=0;

/* All compute variants stream over a tiny in-L1 tile (strips*64 B) so we measure
 * pure ALU throughput. Each strip = one VL of MACs per accumulator.
 * To prevent the multiplied product from being identical each rep (and to keep
 * the load on the path), we re-load x/y from the L1 tile every strip. */

/* m2 source: vl=64 i8/strip, i32m8 acc. */
static uint64_t mac_m2_a1(const int8_t*x,const int8_t*y,size_t strips,uint64_t iters,int32_t*sink){
  size_t vl=__riscv_vsetvl_e8m2(64);
  vint32m8_t a0=__riscv_vmv_v_x_i32m8(0,vl);
  uint64_t c0=rdcycle();
  for(uint64_t it=0;it<iters;it++) for(size_t s=0;s<strips;s++){
    vint8m2_t vx=__riscv_vle8_v_i8m2(x+s*64,vl), vy=__riscv_vle8_v_i8m2(y+s*64,vl);
    a0=__riscv_vwadd_wv_i32m8(a0,__riscv_vwmul_vv_i16m4(vx,vy,vl),vl);
  }
  uint64_t c1=rdcycle();
  vint32m1_t z=__riscv_vmv_v_x_i32m1(0,1);
  *sink+=__riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m8_i32m1(a0,z,vl));
  return c1-c0;
}
static uint64_t mac_m2_a2(const int8_t*x,const int8_t*y,size_t strips,uint64_t iters,int32_t*sink){
  size_t vl=__riscv_vsetvl_e8m2(64);
  vint32m8_t a0=__riscv_vmv_v_x_i32m8(0,vl),a1=__riscv_vmv_v_x_i32m8(0,vl);
  uint64_t c0=rdcycle();
  for(uint64_t it=0;it<iters;it++) for(size_t s=0;s+1<strips;s+=2){
    vint8m2_t x0=__riscv_vle8_v_i8m2(x+s*64,vl), y0=__riscv_vle8_v_i8m2(y+s*64,vl);
    vint8m2_t x1=__riscv_vle8_v_i8m2(x+(s+1)*64,vl), y1=__riscv_vle8_v_i8m2(y+(s+1)*64,vl);
    a0=__riscv_vwadd_wv_i32m8(a0,__riscv_vwmul_vv_i16m4(x0,y0,vl),vl);
    a1=__riscv_vwadd_wv_i32m8(a1,__riscv_vwmul_vv_i16m4(x1,y1,vl),vl);
  }
  uint64_t c1=rdcycle();
  a0=__riscv_vadd_vv_i32m8(a0,a1,vl);
  vint32m1_t z=__riscv_vmv_v_x_i32m1(0,1);
  *sink+=__riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m8_i32m1(a0,z,vl));
  return c1-c0;
}
static uint64_t mac_m2_a4(const int8_t*x,const int8_t*y,size_t strips,uint64_t iters,int32_t*sink){
  size_t vl=__riscv_vsetvl_e8m2(64);
  vint32m8_t a0=__riscv_vmv_v_x_i32m8(0,vl),a1=__riscv_vmv_v_x_i32m8(0,vl),a2=__riscv_vmv_v_x_i32m8(0,vl),a3=__riscv_vmv_v_x_i32m8(0,vl);
  uint64_t c0=rdcycle();
  for(uint64_t it=0;it<iters;it++) for(size_t s=0;s+3<strips;s+=4){
    vint8m2_t x0=__riscv_vle8_v_i8m2(x+s*64,vl), y0=__riscv_vle8_v_i8m2(y+s*64,vl);
    vint8m2_t x1=__riscv_vle8_v_i8m2(x+(s+1)*64,vl), y1=__riscv_vle8_v_i8m2(y+(s+1)*64,vl);
    vint8m2_t x2=__riscv_vle8_v_i8m2(x+(s+2)*64,vl), y2=__riscv_vle8_v_i8m2(y+(s+2)*64,vl);
    vint8m2_t x3=__riscv_vle8_v_i8m2(x+(s+3)*64,vl), y3=__riscv_vle8_v_i8m2(y+(s+3)*64,vl);
    a0=__riscv_vwadd_wv_i32m8(a0,__riscv_vwmul_vv_i16m4(x0,y0,vl),vl);
    a1=__riscv_vwadd_wv_i32m8(a1,__riscv_vwmul_vv_i16m4(x1,y1,vl),vl);
    a2=__riscv_vwadd_wv_i32m8(a2,__riscv_vwmul_vv_i16m4(x2,y2,vl),vl);
    a3=__riscv_vwadd_wv_i32m8(a3,__riscv_vwmul_vv_i16m4(x3,y3,vl),vl);
  }
  uint64_t c1=rdcycle();
  a0=__riscv_vadd_vv_i32m8(__riscv_vadd_vv_i32m8(a0,a1,vl),__riscv_vadd_vv_i32m8(a2,a3,vl),vl);
  vint32m1_t z=__riscv_vmv_v_x_i32m1(0,1);
  *sink+=__riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m8_i32m1(a0,z,vl));
  return c1-c0;
}
/* m1 source: vl=32 i8/strip, i32m4 acc -> can fit A=8 in vreg file */
static uint64_t mac_m1_a1(const int8_t*x,const int8_t*y,size_t strips,uint64_t iters,int32_t*sink){
  size_t vl=__riscv_vsetvl_e8m1(32);
  vint32m4_t a0=__riscv_vmv_v_x_i32m4(0,vl);
  uint64_t c0=rdcycle();
  for(uint64_t it=0;it<iters;it++) for(size_t s=0;s<strips;s++){
    vint8m1_t vx=__riscv_vle8_v_i8m1(x+s*32,vl), vy=__riscv_vle8_v_i8m1(y+s*32,vl);
    a0=__riscv_vwadd_wv_i32m4(a0,__riscv_vwmul_vv_i16m2(vx,vy,vl),vl);
  }
  uint64_t c1=rdcycle();
  vint32m1_t z=__riscv_vmv_v_x_i32m1(0,1);
  *sink+=__riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(a0,z,vl));
  return c1-c0;
}
static uint64_t mac_m1_a4(const int8_t*x,const int8_t*y,size_t strips,uint64_t iters,int32_t*sink){
  size_t vl=__riscv_vsetvl_e8m1(32);
  vint32m4_t a0=__riscv_vmv_v_x_i32m4(0,vl),a1=__riscv_vmv_v_x_i32m4(0,vl),a2=__riscv_vmv_v_x_i32m4(0,vl),a3=__riscv_vmv_v_x_i32m4(0,vl);
  uint64_t c0=rdcycle();
  for(uint64_t it=0;it<iters;it++) for(size_t s=0;s+3<strips;s+=4){
    vint8m1_t x0=__riscv_vle8_v_i8m1(x+s*32,vl), y0=__riscv_vle8_v_i8m1(y+s*32,vl);
    vint8m1_t x1=__riscv_vle8_v_i8m1(x+(s+1)*32,vl), y1=__riscv_vle8_v_i8m1(y+(s+1)*32,vl);
    vint8m1_t x2=__riscv_vle8_v_i8m1(x+(s+2)*32,vl), y2=__riscv_vle8_v_i8m1(y+(s+2)*32,vl);
    vint8m1_t x3=__riscv_vle8_v_i8m1(x+(s+3)*32,vl), y3=__riscv_vle8_v_i8m1(y+(s+3)*32,vl);
    a0=__riscv_vwadd_wv_i32m4(a0,__riscv_vwmul_vv_i16m2(x0,y0,vl),vl);
    a1=__riscv_vwadd_wv_i32m4(a1,__riscv_vwmul_vv_i16m2(x1,y1,vl),vl);
    a2=__riscv_vwadd_wv_i32m4(a2,__riscv_vwmul_vv_i16m2(x2,y2,vl),vl);
    a3=__riscv_vwadd_wv_i32m4(a3,__riscv_vwmul_vv_i16m2(x3,y3,vl),vl);
  }
  uint64_t c1=rdcycle();
  a0=__riscv_vadd_vv_i32m4(__riscv_vadd_vv_i32m4(a0,a1,vl),__riscv_vadd_vv_i32m4(a2,a3,vl),vl);
  vint32m1_t z=__riscv_vmv_v_x_i32m1(0,1);
  *sink+=__riscv_vmv_x_s_i32m1_i32(__riscv_vredsum_vs_i32m4_i32m1(a0,z,vl));
  return c1-c0;
}

/* Memory bandwidth: stream vle8m8 over buf >> LLC. Accumulate ELEMENTWISE in
 * the loop (vadd_vv, no cross-lane reduce on the critical path) and reduce ONCE
 * after the loop, so the loop is load-bound, not reduce-bound. */
static uint64_t stream_load(const int8_t*buf,size_t nbytes,int32_t*sink){
  size_t vl=__riscv_vsetvl_e8m8(128); /* vl = VLMAX e8m8 on VLEN128 = 128 B/strip */
  vint8m8_t acc=__riscv_vmv_v_x_i8m8(0,vl);
  uint64_t c0=rdcycle();
  for(size_t off=0;off+128<=nbytes;off+=128){
    vint8m8_t v=__riscv_vle8_v_i8m8(buf+off,vl);
    acc=__riscv_vadd_vv_i8m8(acc,v,vl); /* elementwise, lane-local; keeps load live */
  }
  uint64_t c1=rdcycle();
  vint16m1_t z=__riscv_vmv_v_x_i16m1(0,1);
  *sink+=__riscv_vmv_x_s_i16m1_i16(__riscv_vwredsum_vs_i8m8_i16m1(acc,z,vl));
  return c1-c0;
}

static int cmp64(const void*a,const void*b){uint64_t x=*(const uint64_t*)a,y=*(const uint64_t*)b;return (x>y)-(x<y);}
static uint64_t median(uint64_t*v,int n){qsort(v,n,sizeof(uint64_t),cmp64);return v[n/2];}
static uint64_t vmin(uint64_t*v,int n){uint64_t m=v[0];for(int i=1;i<n;i++)if(v[i]<m)m=v[i];return m;}

int main(void){
  size_t strips=8;
  int8_t*x=aligned_alloc(64,strips*64),*y=aligned_alloc(64,strips*64);
  for(size_t i=0;i<strips*64;i++){x[i]=(int8_t)(i*7+1);y[i]=(int8_t)(i*3-2);}
  int32_t sink=0; uint64_t iters=2000000; int reps=11;

  /* WARMUP to ramp the ondemand governor up to its sustained frequency, then
   * measure the sustained freq DURING a long hot vwadd.wv kernel run (rdcycle
   * vs CLOCK_MONOTONIC bracketing the same op the kernels use). This reflects
   * the clock the kernels actually see, not an idle-floor busy loop. */
  /* ACTUAL VL per strip (VLMAX clamps AVL): print to confirm element count. */
  size_t vl_m2=__riscv_vsetvl_e8m2(64), vl_m1=__riscv_vsetvl_e8m1(32);
  printf("vl_e8m2(AVL=64)=%zu  vl_e8m1(AVL=32)=%zu  (these are MACs/strip)\n",vl_m2,vl_m1);

  /* Hard warmup so the ondemand governor reaches its sustained (max) freq before
   * the freq read AND before all timed sections; take freq during a long hot run. */
  for(int w=0;w<8;w++) (void)mac_m2_a1(x,y,strips,iters,&sink);
  struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0); uint64_t f0=rdcycle();
  for(int w=0;w<6;w++) (void)mac_m2_a1(x,y,strips,iters,&sink);
  uint64_t f1=rdcycle(); clock_gettime(CLOCK_MONOTONIC,&t1);
  double ns=(t1.tv_sec-t0.tv_sec)*1e9+(t1.tv_nsec-t0.tv_nsec); freq_ghz=(double)(f1-f0)/ns;
  printf("freq_GHz_sustained=%.4f (measured during hot vwadd kernel, ondemand governor)\n",freq_ghz);

  struct{const char*name;uint64_t(*fn)(const int8_t*,const int8_t*,size_t,uint64_t,int32_t*);int mps;}
  comp[]={{"mac_m2_a1",mac_m2_a1,(int)vl_m2},{"mac_m2_a2",mac_m2_a2,(int)vl_m2},
          {"mac_m1_a1",mac_m1_a1,(int)vl_m1},{"mac_m1_a4",mac_m1_a4,(int)vl_m1}};
  printf("\n# COMPUTE CEILING vwmul+vwadd.wv i8->i32 (emitted byte dot core)  strips=%zu iters=%llu reps=%d\n",strips,(unsigned long long)iters,reps);
  printf("# variant       min_cyc   med_cyc   cyc/strip  MACs/cyc   GMAC/s\n");
  for(unsigned k=0;k<sizeof(comp)/sizeof(comp[0]);k++){
    uint64_t m[16]; for(int r=0;r<reps;r++) m[r]=comp[k].fn(x,y,strips,iters,&sink);
    uint64_t mn=vmin(m,reps),md=median(m,reps);
    double ts=(double)iters*strips, cps=(double)mn/ts, macs=ts*comp[k].mps;
    double mpc=macs/(double)mn, g=mpc*freq_ghz;
    printf("%-12s %10llu %9llu  %8.4f  %8.3f  %8.2f\n",comp[k].name,(unsigned long long)mn,(unsigned long long)md,cps,mpc,g);
  }
  free(x);free(y);

  size_t MB=256,nbytes=MB*1024*1024;
  int8_t*big=aligned_alloc(64,nbytes);
  for(size_t i=0;i<nbytes;i++) big[i]=(int8_t)(i*131+7);
  printf("\n# MEMORY BW streaming vle8m8 over %zu MB (>> LLC)\n",MB);
  for(int w=0;w<3;w++) (void)stream_load(big,nbytes,&sink); /* page-in + freq ramp to sustained */
  int breps=9; uint64_t bm[16];
  for(int r=0;r<breps;r++) bm[r]=stream_load(big,nbytes,&sink);
  uint64_t bmn=vmin(bm,breps),bmd=median(bm,breps);
  printf("bytes=%zu min_cyc=%llu med_cyc=%llu  B/cyc=%.3f  GB/s(min)=%.2f  GB/s(med)=%.2f  sink=%d\n",
    nbytes,(unsigned long long)bmn,(unsigned long long)bmd,(double)nbytes/bmn,
    (double)nbytes/bmn*freq_ghz,(double)nbytes/bmd*freq_ghz,sink);
  free(big);
  return 0;
}
