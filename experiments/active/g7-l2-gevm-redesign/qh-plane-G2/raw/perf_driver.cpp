// G7 L2 qh-plane G2 cold GEVM M=1 micro: NEW REDESIGN-B repack-GEVM (1 call over
// the whole N-row matrix) vs stock ggml_vec_dot_q5_0_q8_0 block-dot (N per-row
// calls). Identical MAC work (N x nb x 32) and identical 22 B/row/block weight
// footprint. Wall-time (CLOCK_MONOTONIC) is the memory-bound verdict; run with
// `perf stat` per mode for cycles/instructions/IPC/cache-misses. Default N=4096
// K=4096 => 11.5 MB weight (memory-wall decode).
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <algorithm>
extern "C" void kern_new(size_t,float*,size_t,const uint8_t*,size_t,const uint8_t*,size_t,int32_t);
extern "C" void kern_old(size_t,float*,size_t,const uint8_t*,size_t,const uint8_t*,size_t,int32_t);
extern "C" void ggml_vec_dot_q5_0_q8_0(int,float*,size_t,const void*,size_t,const void*,size_t,int);
static void fill_rand(uint8_t*p,size_t n){ for(size_t i=0;i<n;i++)p[i]=(uint8_t)(rand()&0xff); }
static const uint16_t H1 = 0x3c00; // _Float16 1.0
static double now_ms(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e3 + t.tv_nsec/1e6; }
int main(int argc,char**argv){
  int N      = argc>1?atoi(argv[1]):4096;   // rows (mult of 16)
  int K      = argc>2?atoi(argv[2]):4096;   // contraction (mult of 32)
  int rounds = argc>3?atoi(argv[3]):15;
  int mode   = argc>4?atoi(argv[4]):0;      // 0=both(ratio) 1=new-only 2=stock-only
  srand(argc>5?(unsigned)atoi(argv[5]):777u);
  int nb = K/32;
  size_t wnewB   = (size_t)(N/16)*nb*352;
  size_t wstockB = (size_t)N*nb*22;
  size_t actB    = (size_t)nb*34;
  uint8_t* wnew   = (uint8_t*)malloc(wnewB);
  uint8_t* wstock = (uint8_t*)malloc(wstockB);
  uint8_t* act    = (uint8_t*)malloc(actB);
  float* outn = (float*)malloc((size_t)N*4);
  float* outs = (float*)malloc((size_t)N*4);
  fill_rand(wnew,wnewB); fill_rand(wstock,wstockB); fill_rand(act,actB);
  for(int g=0; g<N/16; ++g){ uint8_t* blk = wnew + (size_t)g*nb*352;
    for(int b=0;b<nb;b++){ uint16_t* d=(uint16_t*)(blk+(size_t)b*352); for(int j=0;j<16;j++) d[j]=H1; } }
  for(int r=0;r<N;r++){ uint8_t* row = wstock + (size_t)r*nb*22;
    for(int b=0;b<nb;b++){ *(uint16_t*)(row+(size_t)b*22)=H1; } }
  for(int b=0;b<nb;b++){ *(uint16_t*)(act+(size_t)b*34)=H1; }
  // A m2 buffer alias for OLD (same repacked layout as NEW).
  if(rounds>128) rounds=128;
  double tN[128], tO[128], tS[128]; int rn=0, ro=0, rs=0;
  double sink=0;
  if(mode==0 || mode==1){
    for(int t=0;t<rounds;t++){ double a=now_ms(); kern_new((size_t)32*nb,outn,N,wnew,0,act,0,0); tN[rn++]=now_ms()-a; }
    for(int r=0;r<N;r++) sink+=outn[r];
  }
  if(mode==0 || mode==3){
    for(int t=0;t<rounds;t++){ double a=now_ms(); kern_old((size_t)32*nb,outs,N,wnew,0,act,0,0); tO[ro++]=now_ms()-a; }
    for(int r=0;r<N;r++) sink+=outs[r];
  }
  if(mode==0 || mode==2){
    for(int t=0;t<rounds;t++){ double a=now_ms();
      for(int r=0;r<N;r++) ggml_vec_dot_q5_0_q8_0(K,&outs[r],0,wstock+(size_t)r*nb*22,0,act,0,0);
      tS[rs++]=now_ms()-a; }
    for(int r=0;r<N;r++) sink+=outs[r];
  }
  auto med=[&](double*v,int n){ std::sort(v,v+n); return v[n/2]; };
  auto mn =[&](double*v,int n){ double m=v[0]; for(int i=1;i<n;i++) if(v[i]<m)m=v[i]; return m; };
  auto reliqr=[&](double*v,int n){ double q1=v[n/4],q3=v[(3*n)/4],md=v[n/2]; return md>0?(q3-q1)/md:0; };
  printf("N=%d K=%d nb=%d rounds=%d weightMB=%.1f mode=%d (sink=%.3g)\n",N,K,nb,rounds,wnewB/1e6,mode,sink);
  double nMin=0,oMin=0,sMin=0,nMed=0,oMed=0,sMed=0;
  if(rn){ nMin=mn(tN,rn); nMed=med(tN,rn); double rq=reliqr(tN,rn);
    printf("NEW  (REDESIGN-B repack-GEVM): ms_med=%.3f ms_min=%.3f relIQR=%.3f\n",nMed,nMin,rq); }
  if(ro){ oMin=mn(tO,ro); oMed=med(tO,ro); double rq=reliqr(tO,ro);
    printf("OLD  (per-lane-expand repack-GEVM): ms_med=%.3f ms_min=%.3f relIQR=%.3f\n",oMed,oMin,rq); }
  if(rs){ sMin=mn(tS,rs); sMed=med(tS,rs); double rq=reliqr(tS,rs);
    printf("STOCK(ggml block-dot 16x): ms_med=%.3f ms_min=%.3f relIQR=%.3f\n",sMed,sMin,rq); }
  if(rn && rs) printf("RATIO stock/new  (>=1 NEW faster): med=%.4f min=%.4f  [path: repack-GEVM vs block-dot]\n", sMed/nMed, sMin/nMin);
  if(rn && ro) printf("RATIO old/new    (>=1 NEW faster): med=%.4f min=%.4f  [DECODE-ISOLATED: REDESIGN-B vs per-lane-expand, same footprint]\n", oMed/nMed, oMin/nMin);
  if(ro && rs) printf("RATIO stock/old  (>=1 OLD faster): med=%.4f min=%.4f  [path only, OLD decode]\n", sMed/oMed, sMin/oMin);
  free(wnew);free(wstock);free(act);free(outn);free(outs);
  return 0;
}
