/* gelu_f16lut_driver.c — G.0.3 gelu SAME-PRECISION-TIER rematch (JUDGMENT-SUSPENDED release).
 * OURS = weft-emitted f16-LUT gelu variant (weft_gelu_f16lut_scalar seam; front-door emit,
 *        gelu_precision="f16lut"). OPP = ggml as-shipped GGML_GELU_FP16 f16 table (opp_gelu.cpp).
 * BOTH sides now share the f16 numeric tier => byte-exact A/B (gate = ours-vs-opp 0 ULP).
 * COLD protocol identical to bfwd_micro_driver.c: pool streamed cold from DRAM, median-of-rounds.
 * Also reports the precision DUAL-COLUMN: ours & opp relerr vs the fp64 tanh oracle (both ~f16 tier).
 * argv: <rounds> <target_mb> <seed>   (sweeps the 8 census shapes internally; single instance).
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

extern "C" void weft_emitc_gelu_f32_kernel_gelu_f32(size_t,const float*,float*); /* OURS f16lut */
extern "C" void weft_gelu_f16lut_init(void);                                     /* OURS table build (untimed) */
extern "C" void opp_ggml_vec_gelu_f32(int n, float* y, const float* x);          /* OPP  f16lut */
extern "C" void opp_gelu_init(void);                                             /* OPP  table build (untimed) */

static uint64_t rng;
static uint32_t xr(void){ rng^=rng<<13; rng^=rng>>7; rng^=rng<<17; return (uint32_t)(rng>>32); }
static float rf(void){ return ((float)(xr()&0xffffff)/(float)0xffffff)*2.0f-1.0f; } /* [-1,1) — census dist */
static double now_ns(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec*1e9+(double)t.tv_nsec; }
static int cmp_d(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return x<y?-1:x>y?1:0; }
static void stats(double* v,int n,double*med,double*iqrpct){ qsort(v,n,sizeof(double),cmp_d);
  *med=(n&1)?v[n/2]:0.5*(v[n/2-1]+v[n/2]); double q1=v[n/4],q3=v[(3*n)/4]; *iqrpct=(*med>0)?100.0*(q3-q1)/(*med):0.0; }
static uint32_t ulp_diff(float a,float b){ if(a==b) return 0; uint32_t ua,ub; memcpy(&ua,&a,4); memcpy(&ub,&b,4);
    if((ua>>31)!=(ub>>31)) return 0xffffffu; return ua>ub?ua-ub:ub-ua; }

int main(int argc,char**argv){
    int rounds = argc>1?atoi(argv[1]):12;
    double tmb  = argc>2?atof(argv[2]):8.0;
    uint64_t seed0 = argc>3?strtoull(argv[3],0,0):0xC0FFEEull;
    if(rounds<10)rounds=10;
    opp_gelu_init();          /* OPP  table build (once, off the timed path) */
    weft_gelu_f16lut_init();  /* OURS table build (once, off the timed path — symmetric) */
    int Ns[]={512,1024,2048,3072,4096,5120,8192,16384};
    printf("# G.0.3 gelu f16-LUT rematch  rounds=%d target_mb=%.1f seed=0x%llx  (bytes_moved=2*n*4 read+write)\n",
           rounds,tmb,(unsigned long long)seed0);
    for(int k=0;k<8;k++){ int n=Ns[k]; rng=(seed0+ (uint64_t)k*0x9E3779B97F4A7C15ull)|1ull;
        size_t set_bytes=(size_t)n*sizeof(float)+ (size_t)n*sizeof(float); /* in + out */
        int P=(int)((tmb*1e6)/(double)set_bytes); if(P<8)P=8; if(P>4096)P=4096;
        float**X=(float**)malloc(P*sizeof(float*)),**OU=(float**)malloc(P*sizeof(float*)),**OO=(float**)malloc(P*sizeof(float*));
        for(int p=0;p<P;p++){ X[p]=(float*)aligned_alloc(64,(size_t)n*4); OU[p]=(float*)aligned_alloc(64,(size_t)n*4); OO[p]=(float*)aligned_alloc(64,(size_t)n*4);
            for(int i=0;i<n;i++) X[p][i]=rf(); }
        /* byte-exact gate + precision dual-column on set 0 */
        weft_emitc_gelu_f32_kernel_gelu_f32((size_t)n,X[0],OU[0]);
        opp_ggml_vec_gelu_f32(n,OO[0],X[0]);
        uint32_t mu=0; double orel=0,prel=0;
        for(int i=0;i<n;i++){ uint32_t u=ulp_diff(OU[0][i],OO[0][i]); if(u>mu)mu=u;
            double ox=X[0][i]; double oref=0.5*ox*(1.0+tanh(0.7978845608028654*ox*(1.0+0.044715*ox*ox)));
            double den=fabs(oref)>1e-6?fabs(oref):1e-6;
            double ro=fabs((double)OU[0][i]-oref)/den, rp=fabs((double)OO[0][i]-oref)/den;
            if(ro>orel)orel=ro; if(rp>prel)prel=rp; }
        int gate = (mu==0); /* SAME-TIER byte-exact */
        /* warm */
        for(int w=0;w<2;w++){ for(int p=0;p<P;p++) weft_emitc_gelu_f32_kernel_gelu_f32((size_t)n,X[p],OU[p]);
                              for(int p=0;p<P;p++) opp_ggml_vec_gelu_f32(n,OO[p],X[p]); }
        /* COLD median-of-rounds */
        double *to=(double*)malloc(rounds*sizeof(double)), *tp=(double*)malloc(rounds*sizeof(double));
        volatile double sink=0;
        for(int rd=0;rd<rounds;rd++){
            double t0=now_ns(); for(int p=0;p<P;p++) weft_emitc_gelu_f32_kernel_gelu_f32((size_t)n,X[p],OU[p]); double t1=now_ns(); to[rd]=(t1-t0)/(double)P; sink+=OU[0][0];
            double t2=now_ns(); for(int p=0;p<P;p++) opp_ggml_vec_gelu_f32(n,OO[p],X[p]); double t3=now_ns(); tp[rd]=(t3-t2)/(double)P; sink+=OO[0][0];
        }
        double om,oi,pm,pi; stats(to,rounds,&om,&oi); stats(tp,rounds,&pm,&pi);
        double bmv=2.0*n*4; double totmb=(double)P*set_bytes/1e6;
        printf("COLD n=%-6d P=%-4d ws_MB=%5.1f GATE=%s ours_vs_opp_maxulp=%u "
               "ours_ns=%.1f(iqr%.2f%%) opp_ns=%.1f(iqr%.2f%%) ours_GBs=%.3f opp_GBs=%.3f "
               "ratio_ours_over_opp=%.4f | ours_rel_fp64=%.3e opp_rel_fp64=%.3e sink=%.1f\n",
               n,P,totmb,gate?"PASS":"FAIL",mu,om,oi,pm,pi,bmv/om,bmv/pm,pm/om,orel,prel,(double)sink);
        for(int p=0;p<P;p++){ free(X[p]);free(OU[p]);free(OO[p]); } free(X);free(OU);free(OO);free(to);free(tp);
    }
    return 0;
}
