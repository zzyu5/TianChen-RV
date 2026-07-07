/* tq2_0_ab_driver.c — [GAP-RP] tq2_0 register-spill-fix PRE/POST A/B + vs-generic (rvv/VLEN128).
 *
 * 裁决二.4: does eliminating the LMUL-over-widen regfile spill (commit 3186919d, "keep m4 / kill
 * overlap / serialize the 2 chunks") transduce to a silicon speedup?  Same read structure as [GAP-SB]:
 *   A/B      = ours-PRE-median / ours-POST-median   (isolated spill-fix delta; >1 => POST faster)
 *   vs-gen   = generic-median  / ours-POST-median   (>1 => ours faster than the ggml scalar ref)
 * PRE  = tq2_0.o exported @0ad9cf6d (has the spill), POST = @3186919d (spill gone). Both via the
 * cached detached-worktree tcrv-opt/tcrv-translate (main tree + build/ UNTOUCHED).
 *
 * Cache-cold: an oversized pool of (weight tq2_0 + activation q8_K) block pairs > 3xL3, swept once
 * per round so every super-block streams cold from DRAM. N rounds => median + IQR + min; FNV
 * fingerprint of the outputs (PRE==POST proves byte-exact by construction; vs generic = numeric xcheck).
 *
 * Env: WSET_MIB (default 256), ROUNDS (default 16). Links: driver.o + PRE.o + POST.o + generic.o,
 *      symbols objcopy-renamed to *_PRE / *_POST / vecdot_tq2_0_generic (see tq2_0_ab.sh).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#define QK_K 256
typedef uint16_t ggml_half;
typedef struct { uint8_t qs[QK_K/4]; ggml_half d; }                  block_tq2_0;  /* 66  (qs@0, d@64) */
typedef struct { float d; int8_t qs[QK_K]; int16_t bsums[QK_K/16]; } block_q8_K;   /* 292 */
_Static_assert(sizeof(block_tq2_0)==66, "tq2_0");
_Static_assert(sizeof(block_q8_K)==292, "q8_K");

/* ours: constructed block-dot kernels (non-PIC, extern C), PRE and POST renamed at link time */
extern void tcrv_tq2_0_block_dot_PRE (size_t n, float *s, const uint8_t *vx, const uint8_t *vy);
extern void tcrv_tq2_0_block_dot_POST(size_t n, float *s, const uint8_t *vx, const uint8_t *vy);
/* generic: ggml scalar reference ggml_vec_dot_tq2_0_q8_K_generic, renamed.
 * Guarded: -DNO_GENERIC builds the PRE/POST A/B alone (no ggml dependency). */
#ifndef NO_GENERIC
extern void vecdot_tq2_0_generic(int n, float *s, size_t bs, const void *vx, size_t bx,
                                 const void *vy, size_t by, int nrc);
#endif

static uint64_t rng = 0x243F6A8885A308D3ULL;
static uint32_t xr(void){ rng ^= rng<<13; rng ^= rng>>7; rng ^= rng<<17; return (uint32_t)(rng>>32); }
static uint16_t rf16(void){ /* finite moderate f16: exp in [10..17], sign 0 */
    uint16_t e = 10 + (xr() % 8); uint16_t m = xr() & 0x3FF; return (uint16_t)((e<<10) | m);
}
static double now_ns(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts);
    return (double)ts.tv_sec*1e9 + (double)ts.tv_nsec; }
static int cmpd(const void*a,const void*b){ double x=*(const double*)a,y=*(const double*)b; return (x>y)-(x<y); }
static void stats(double*v,int k,double*med,double*iqr,double*mn){
    double*s=malloc(sizeof(double)*k); memcpy(s,v,sizeof(double)*k); qsort(s,k,sizeof(double),cmpd);
    *mn=s[0]; *med=(k&1)?s[k/2]:0.5*(s[k/2-1]+s[k/2]); *iqr=s[(3*k)/4]-s[k/4]; free(s);
}

static block_tq2_0 *W;   /* weights */
static block_q8_K   *A;   /* activations */
static size_t NB;         /* pool blocks */

static uint64_t sweep(int side, double *out_ns){
    /* one cold sweep over the whole pool; side 0=PRE 1=POST 2=generic; returns fnv(outputs) */
    uint64_t h = 1469598103934665603ULL;
    volatile float sink = 0;
    double t0 = now_ns();
    for (size_t b = 0; b < NB; b++){
        float r;
        const uint8_t *vx = (const uint8_t *)&W[b];
        const uint8_t *vy = (const uint8_t *)&A[b];
        if      (side==0) tcrv_tq2_0_block_dot_PRE (QK_K,&r,vx,vy);
        else if (side==1) tcrv_tq2_0_block_dot_POST(QK_K,&r,vx,vy);
#ifndef NO_GENERIC
        else              vecdot_tq2_0_generic(QK_K,&r,0,vx,0,vy,0,1);
#else
        else              { r = 0; }
#endif
        uint32_t bits; memcpy(&bits,&r,4); h ^= bits; h *= 1099511628211ULL; sink += r;
    }
    *out_ns = now_ns() - t0;
    (void)sink;
    return h;
}

int main(void){
    size_t wset_mib = getenv("WSET_MIB") ? strtoull(getenv("WSET_MIB"),0,10) : 256;
    int rounds = getenv("ROUNDS") ? atoi(getenv("ROUNDS")) : 16;
    size_t pair = sizeof(block_tq2_0) + sizeof(block_q8_K); /* 358 B */
    NB = (wset_mib * 1024ull * 1024ull) / pair;
    W = malloc(NB * sizeof(block_tq2_0));
    A = malloc(NB * sizeof(block_q8_K));
    if (!W || !A){ fprintf(stderr,"alloc fail NB=%zu\n",NB); return 2; }
    for (size_t b=0;b<NB;b++){
        for (int i=0;i<QK_K/4;i++) W[b].qs[i] = (uint8_t)xr();
        W[b].d = rf16();
        A[b].d = 1.0f + (float)(xr()%64)/64.0f;
        int32_t gs[QK_K/16]; memset(gs,0,sizeof(gs));
        for (int i=0;i<QK_K;i++){ int8_t q=(int8_t)((int)(xr()%255)-127); A[b].qs[i]=q; gs[i/16]+=q; }
        for (int i=0;i<QK_K/16;i++) A[b].bsums[i]=(int16_t)gs[i];
    }
    double ws_mib = (double)(NB*pair)/(1024.0*1024.0);
#ifdef NO_GENERIC
    int have_gen = 0;
#else
    int have_gen = 1;
#endif
    /* warm (fault-in) once, untimed */
    double dump; sweep(0,&dump); sweep(1,&dump); if (have_gen) sweep(2,&dump);
    double *tp=malloc(sizeof(double)*rounds), *to=malloc(sizeof(double)*rounds), *tg=malloc(sizeof(double)*rounds);
    uint64_t hp=0,ho=0,hg=0;
    for (int r=0;r<rounds;r++){
        double a,b,c=0;
        hp=sweep(0,&a); ho=sweep(1,&b); if (have_gen) hg=sweep(2,&c);
        tp[r]=a; to[r]=b; tg[r]=c;
    }
    double mp,ip,np, mo,io,no, mg,ig,ng;
    stats(tp,rounds,&mp,&ip,&np); stats(to,rounds,&mo,&io,&no); stats(tg,rounds,&mg,&ig,&ng);
    /* ns per block */
    double npb_pre=mp/NB, npb_post=mo/NB, npb_gen=mg/NB;
    printf("TQ2AB wset_MiB=%.1f pool_blocks=%zu rounds=%d L3_MiB=64 hygiene=%s\n",
           ws_mib, NB, rounds, ws_mib>192.0?"COLD(ws>3xL3)":"WARN");
    if (have_gen) {
        printf("TQ2AB ns_per_block  PRE=%.3f (iqr%.3f) POST=%.3f (iqr%.3f) GENERIC=%.3f (iqr%.3f)\n",
               npb_pre, ip/NB, npb_post, io/NB, npb_gen, ig/NB);
        printf("TQ2AB A/B(PRE/POST)=%.4f  vs-generic(GEN/POST)=%.4f  vs-generic(GEN/PRE)=%.4f\n",
               mp/mo, mg/mo, mg/mp);
        printf("TQ2AB fnv PRE=0x%016llx POST=0x%016llx GENERIC=0x%016llx  PRE==POST:%s  POST==GEN:%s\n",
               (unsigned long long)hp,(unsigned long long)ho,(unsigned long long)hg,
               hp==ho?"YES":"NO", ho==hg?"YES":"NO");
    } else {
        printf("TQ2AB ns_per_block  PRE=%.3f (iqr%.3f) POST=%.3f (iqr%.3f) GENERIC=n_a\n",
               npb_pre, ip/NB, npb_post, io/NB);
        printf("TQ2AB A/B(PRE/POST)=%.4f  vs-generic=n_a(NO_GENERIC build)\n", mp/mo);
        printf("TQ2AB fnv PRE=0x%016llx POST=0x%016llx  PRE==POST:%s\n",
               (unsigned long long)hp,(unsigned long long)ho, hp==ho?"YES":"NO");
    }
    (void)npb_gen;
    return 0;
}
