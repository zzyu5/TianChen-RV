#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
extern void ggml_gemm_q5_K_8x8_q8_K(int,float*,size_t,const void*,const void*,int,int);
extern void ggml_gemm_q6_K_8x8_q8_K(int,float*,size_t,const void*,const void*,int,int);
extern void ggml_gemm_q5_K_8x4_q8_K(int,float*,size_t,const void*,const void*,int,int);
extern void ggml_gemm_q6_K_8x4_q8_K(int,float*,size_t,const void*,const void*,int,int);
static double now_ns(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static int cmpd(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return x<y?-1:x>y?1:0;}
#define FL (224u*1024u*1024u)
static uint8_t* fb; static volatile double snk;
static void flush(void){uint64_t s=0;for(size_t i=0;i<FL;i+=64){fb[i]^=(uint8_t)i;s+=fb[i];}snk+=s;}
static void probe(const char* name,void(*fn)(int,float*,size_t,const void*,const void*,int,int),int K,int nr,int nc,int wblk8,int reps){
  int nb=K/256;int grp_c=(nc+7)/8;
  size_t wbytes=(size_t)grp_c*nb*wblk8+65536;
  size_t abytes=(size_t)((nr+7)/8)*nb*(8*292)+65536;
  uint8_t* W=aligned_alloc(64,wbytes);uint8_t* A=aligned_alloc(64,abytes);float* O=aligned_alloc(64,(size_t)nr*nc*4);
  memset(W,0x08,wbytes);   /* every fp16 halfword=0x0808 ~4.9e-4 finite; every int8=8 small */
  memset(A,0x08,abytes);   /* activation q8: d(fp32)=0x08080808 ~1.6e-34 tiny-normal; qs int8=8 */
  double macs=(double)nr*nc*K;double* t=malloc(sizeof(double)*reps);
  flush();fn(K,O,(size_t)nc,W,A,nr,nc);
  for(int p=0;p<reps;p++){flush();double t0=now_ns();fn(K,O,(size_t)nc,W,A,nr,nc);t[p]=now_ns()-t0;snk+=O[0];}
  qsort(t,reps,sizeof(double),cmpd);double med=t[reps/2];
  printf("PROBEB %-14s wblk8=%d : opp_med_ns=%.0f opp_gmacs=%.4f\n",name,wblk8,med,macs/med);
  free(W);free(A);free(O);free(t);
}
int main(int c,char**v){int K=c>1?atoi(v[1]):2048,nr=c>2?atoi(v[2]):64,nc=c>3?atoi(v[3]):512,reps=c>4?atoi(v[4]):12;
  fb=aligned_alloc(64,FL);memset(fb,1,FL);
  probe("q5_K_8x8",ggml_gemm_q5_K_8x8_q8_K,K,nr,nc,1408,reps);
  probe("q6_K_8x8",ggml_gemm_q6_K_8x8_q8_K,K,nr,nc,1680,reps);
  probe("q5_K_8x4",ggml_gemm_q5_K_8x4_q8_K,K,nr,nc,1408,reps);
  probe("q6_K_8x4",ggml_gemm_q6_K_8x4_q8_K,K,nr,nc,1680,reps);
  (void)snk;return 0;}
