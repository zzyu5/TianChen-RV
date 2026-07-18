// iq4nl_gemm_prefill_p1r.cpp — P1-remainder #5 (rvv) / #7 (k1): iq4_nl gemm_tile PREFILL, nr=16 GEMM.
//
// WHY THIS FILE EXISTS (P1 前轮 VOID 成因, 机核过):
//   tools/e2e-harness/board/iq4nl_gemm_paired_driver.c:59-60 filled OUR weights (Wr) and the
//   OPPONENT's weights (Wo) with two INDEPENDENT fill_rand() streams (only the fp16 scales were
//   both forced to 0x2C00). The two sides therefore never held the same data, and the file
//   contains zero oracle / zero memcmp / zero mismatch counter. PREREG §4.0 makes byte-exact
//   ZERO-MODEL a HARD PRECONDITION for reporting any perf number => that harness is STRUCTURALLY
//   incapable of satisfying the correctness gate. This driver replaces it.
//
// ZERO-MODEL construction ([K-5b] 证书三要件, 禁空心证书):
//   ① SINGLE SOURCE OF TRUTH: one plain `W` (block_iq4_nl, nc*nb) + one plain `A` (block_q8_0,
//      nr*nb). EVERY consumer is a pure function of those two arrays:
//        - packed[] = make_x16(W)      -> block_iq4_nlx16 288B  (ours + OPP-S)
//        - apack[]  = make_q8x4(A)     -> block_q8_0x4   136B, qs@8 (ours + OPP-S)
//        - OPP-X consumes plain W / plain A directly.
//      Both sides are the SAME data. No second RNG stream anywhere.
//   ② PER-(r,c) ORACLE recomputed from PLAIN W/A ONLY, zero reuse of intermediates: oracle_block()
//      re-derives every arithmetic term (nibble split, codebook lookup, int MAC, fp16 scale
//      product) with its own ORACLE_KV table and its own scalar loop. It never reads packed/apack
//      and never calls the leaf or ggml. It is not a captured intermediate.
//   ③ THREE-WAY GATE: ours / OPP-X / OPP-S each independently vs the oracle.
//
// TOLERANCE (principled, not arbitrary): kernels accumulate nb per-block f32 contributions; the
//   standard forward error bound is |fl(sum)-sum| <= nb*eps/(1-nb*eps) * sum|term_i|. We gate at
//   tol = TOLK * nb * FLT_EPSILON * amag, amag = sum_l |contribution_l| computed by the oracle in
//   double. worst_tolratio is PRINTED so the headroom is auditable (a hollow gate would show
//   ratio ~1; a live tight gate shows ratio << 1 while faults blow past 1).
//
// FAULT INJECTION (proves the gate is not hollow — see run script):
//   -DINJECT=1  oracle constant fault  (ORACLE_KV[3] -65 -> -64)  => ALL THREE must go RED
//   -DINJECT=2  DUT output fault       (ours[mid] += 1.0f)        => OURS ONLY must go RED
//   -DINJECT=3  no driver change; built against a leaf whose codebook byte was flipped
//                                                                 => OURS ONLY must go RED
//
// argv: <K(mult32)> <nr(mult4)> <nc(mult16)> <reps> <seed> [verify_only]
// PREREG §4.3 measure shape: K=2048 nr=16 nc=512 N>=20 2-seed cold. THIS STAGE = verify_only.
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cfloat>
#include <vector>
#include <random>
#include <algorithm>
#include <time.h>
#include <riscv_vector.h>

#ifndef INJECT
#define INJECT 0
#endif
#ifndef TOLK
#define TOLK 4.0
#endif
#ifndef DIAG
#define DIAG 0
#endif

typedef _Float16 ggml_half;
#define QK4_NL 32

struct block_iq4_nl    { ggml_half d;     uint8_t qs[QK4_NL/2]; };  // 18B  plain weight block
struct block_q8_0      { ggml_half d;     int8_t  qs[QK4_NL];   };  // 34B  plain activation block
struct block_iq4_nlx16 { ggml_half d[16]; uint8_t qs[256];      };  // 288B x16 interleaved weight
struct block_q8_0x4    { ggml_half d[4];  int8_t  qs[128];      };  // 136B x4 interleaved act, qs@8
static_assert(sizeof(block_iq4_nl)==18,    "iq4_nl 18");
static_assert(sizeof(block_q8_0)==34,      "q8_0 34");
static_assert(sizeof(block_iq4_nlx16)==288,"iq4_nlx16 288");
static_assert(sizeof(block_q8_0x4)==136,   "q8_0x4 136");

// ---------------------------------------------------------------------------
// ORACLE — independent. Own codebook table, own scalar recompute. Shares no
// decode implementation with the leaf or with ggml.
// ---------------------------------------------------------------------------
static int8_t ORACLE_KV[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};

// Recompute ONE (row-block, col-block) contribution from PLAIN inputs only.
// Returns the contribution; *absmag receives |contribution| for the error bound.
static double oracle_block(const block_iq4_nl& w, const block_q8_0& a, double* absmag){
    long isum = 0;
    for (int j = 0; j < QK4_NL/2; j++){
        int lo = ORACLE_KV[ w.qs[j] & 0x0F ];   // low  nibble -> element j
        int hi = ORACLE_KV[ w.qs[j] >> 4    ];   // high nibble -> element j+16
        isum += (long)lo * (long)a.qs[j] + (long)hi * (long)a.qs[j + QK4_NL/2];
    }
    double t = (double)isum * (double)(float)w.d * (double)(float)a.d;
    *absmag = fabs(t);
    return t;
}

// ---------------------------------------------------------------------------
// Interleavers — the ONLY producers of the packed forms, both fed from W/A.
// make_x16      : o.qs[j*16 + c] = in[c].qs[j]            (== ggml block_iq4_nlx16)
// make_q8x4     : o.qs[e*4 + r] (e<16) / o.qs[64+(e-16)*4 + r] (e>=16) = in[r].qs[e]
//                 (== ggml block_q8_0x4 as consumed by ggml_gemm_iq4_nl_16x1_q8_0:
//                  a_ptr[l].qs[i*4+rr] for the low nibbles, qs[64+i*4+rr] for the high)
// ---------------------------------------------------------------------------
static block_iq4_nlx16 make_x16(const block_iq4_nl* in){
    block_iq4_nlx16 o;
    for (int c = 0; c < 16; c++) o.d[c] = in[c].d;
    for (int i = 0; i < 256; i++){ int src = i % 16, off = i / 16; o.qs[i] = in[src].qs[off]; }
    return o;
}
static block_q8_0x4 make_q8x4(const block_q8_0* in){
    block_q8_0x4 o;
    for (int r = 0; r < 4; r++) o.d[r] = in[r].d;
    for (int r = 0; r < 4; r++)
        for (int e = 0; e < QK4_NL; e++){
            int idx = (e < 16) ? (e*4 + r) : (64 + (e-16)*4 + r);
            o.qs[idx] = in[r].qs[e];
        }
    return o;
}

// OURS: front-door lowered repack GEMM leaf. ABI (current HEAD, unified grid4-family order):
//   (size_t nr, size_t bs, size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc)
// (was (n,s,vx,vy,nr,nc,bs) at the P1-remainder commit; the emitter now emits the grid4 order.)
extern "C" void weft_emitc_ggml_repack_gemm_iq4_nl_q8_0_kernel_ggml_repack_gemm_iq4_nl_q8_0(
    size_t nr, size_t bs, size_t n, float* s, const uint8_t* vx, const uint8_t* vy, size_t nc);
// OPP-X (CROSSOP): as-shipped block-dot, per (r,c), nrc=1. VLEN dispatch thunk -> _vl128/_vl256.
extern "C" void ggml_vec_dot_iq4_nl_q8_0(int n, float* s, size_t bs, const void* vx, size_t bx,
                                         const void* vy, size_t by, int nrc);
// OPP-S (same-operator): as-shipped hand-written RVV repack-GEMM, arch/riscv/repack.cpp:1252.
extern "C" void ggml_gemm_iq4_nl_16x1_q8_0(int n, float* s, size_t bs, const void* vx,
                                           const void* vy, int nr, int nc);

#ifndef FLUSH_MB
#define FLUSH_MB 224
#endif
static const size_t FLUSH = (size_t)FLUSH_MB*1024*1024;
static uint8_t* g_flush = nullptr; static volatile uint64_t g_sink = 0;
static void cold_flush(){ uint64_t a=0; for(size_t i=0;i<FLUSH;i+=64){ g_flush[i]=(uint8_t)(g_flush[i]+1); a+=g_flush[i]; } g_sink+=a; }
static double now_ns(){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return t.tv_sec*1e9+t.tv_nsec; }
static double med(std::vector<double> v){ std::sort(v.begin(),v.end()); return v[v.size()/2]; }
static double riqr(std::vector<double> v){ std::sort(v.begin(),v.end()); size_t n=v.size();
    double m=v[n/2]; return m>0?100.0*(v[3*n/4]-v[n/4])/m:0.0; }

struct Verdict { int mism; double worst_tolratio; double max_rel; int exact; };

static Verdict gate(const std::vector<float>& x, const std::vector<double>& ora,
                    const std::vector<double>& tol, int n){
    Verdict v{0, 0.0, 0.0, 0};
    for (int i = 0; i < n; i++){
        double e  = fabs((double)x[i] - ora[i]);
        double tr = (tol[i] > 0.0) ? e / tol[i] : (e > 0.0 ? 1e30 : 0.0);
        double rl = e / (fabs(ora[i]) + 1e-6);
        if (tr > 1.0) v.mism++;
        if (tr > v.worst_tolratio) v.worst_tolratio = tr;
        if (rl > v.max_rel)        v.max_rel = rl;
        if (e == 0.0)              v.exact++;
    }
    return v;
}

int main(int argc, char** argv){
    if (argc < 6){ fprintf(stderr,"usage: %s K nr nc reps seed [verify_only]\n", argv[0]); return 2; }
    int K = atoi(argv[1]), nr = atoi(argv[2]), nc = atoi(argv[3]), reps = atoi(argv[4]);
    unsigned seed = (unsigned)strtoul(argv[5], 0, 0);
    bool vo = argc > 6 && atoi(argv[6]);
    if (K % QK4_NL){ fprintf(stderr,"K must be mult of 32\n"); return 2; }
    if (nr % 4)    { fprintf(stderr,"nr must be mult of 4\n");  return 2; }
    if (nc % 16)   { fprintf(stderr,"nc must be mult of 16\n"); return 2; }
    if (reps < 10) reps = 10;
    int nb = K / QK4_NL, ng = nc / 16, nrg = nr / 4, no = nr * nc;
    long vlen = (long)__riscv_vlenb() * 8;

#if INJECT==1
    ORACLE_KV[3] = -64;   // *** DELIBERATE ORACLE-CONSTANT FAULT (anti-hollow proof) ***
#endif

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int>   qd(0,255);
    std::uniform_real_distribution<float> dd(0.001f,0.05f);
    std::uniform_int_distribution<int>   ad(-127,127);

    // ===== SINGLE SOURCE OF TRUTH: plain W, plain A. Nothing else is randomised. =====
    std::vector<block_iq4_nl> W((size_t)nc*nb);
    for (auto& b : W){ b.d=(ggml_half)dd(rng); for(int i=0;i<16;i++) b.qs[i]=(uint8_t)qd(rng); }
    std::vector<block_q8_0> A((size_t)nr*nb);
    for (auto& a : A){ a.d=(ggml_half)dd(rng); for(int i=0;i<32;i++) a.qs[i]=(int8_t)ad(rng); }

    // ===== derived views (pure functions of W / A) =====
    std::vector<block_iq4_nlx16> packed((size_t)ng*nb);
    for (int g=0; g<ng; g++) for (int l=0; l<nb; l++){
        block_iq4_nl t[16];
        for (int c=0;c<16;c++) t[c] = W[(size_t)(g*16+c)*nb + l];
        packed[(size_t)g*nb + l] = make_x16(t);
    }
    std::vector<block_q8_0x4> apack((size_t)nrg*nb);
    for (int y=0; y<nrg; y++) for (int l=0; l<nb; l++){
        block_q8_0 t[4];
        for (int r=0;r<4;r++) t[r] = A[(size_t)(y*4+r)*nb + l];
        apack[(size_t)y*nb + l] = make_q8x4(t);
    }

    std::vector<float> ours(no,0.f), oX(no,0.f), oS(no,0.f);

    weft_emitc_ggml_repack_gemm_iq4_nl_q8_0_kernel_ggml_repack_gemm_iq4_nl_q8_0(
        (size_t)nr, (size_t)nc, (size_t)K, ours.data(),
        (const uint8_t*)packed.data(), (const uint8_t*)apack.data(), (size_t)nc);
#if INJECT==2
    ours[no/2] += 1.0f;   // *** DELIBERATE DUT-OUTPUT FAULT (anti-hollow proof) ***
#endif
    for (int r=0;r<nr;r++) for (int c=0;c<nc;c++)
        ggml_vec_dot_iq4_nl_q8_0(K, &oX[(size_t)r*nc+c], 0,
                                 W.data()+(size_t)c*nb, 0, A.data()+(size_t)r*nb, 0, 1);
    ggml_gemm_iq4_nl_16x1_q8_0(K, oS.data(), (size_t)nc, (const void*)packed.data(),
                               (const void*)apack.data(), nr, nc);

    // ===== ORACLE: per-(r,c), recomputed from PLAIN W/A only =====
    std::vector<double> ora(no,0.0), tol(no,0.0);
    for (int r=0;r<nr;r++) for (int c=0;c<nc;c++){
        double s=0.0, amag=0.0;
        for (int l=0;l<nb;l++){ double m; s += oracle_block(W[(size_t)c*nb+l], A[(size_t)r*nb+l], &m); amag += m; }
        size_t i=(size_t)r*nc+c; ora[i]=s;
        tol[i] = (double)TOLK * (double)nb * (double)FLT_EPSILON * amag;
    }

    Verdict vR = gate(ours,ora,tol,no), vX = gate(oX,ora,tol,no), vS = gate(oS,ora,tol,no);
    // informational cross-check: do the two independently-computed kernels agree bit-for-bit?
    int be_RX=0, be_RS=0;
    for (int i=0;i<no;i++){ if(ours[i]==oX[i]) be_RX++; if(ours[i]==oS[i]) be_RS++; }

    printf("ZERO-MODEL K=%d nr=%d nc=%d VLEN=%ld seed=0x%X INJECT=%d TOLK=%.1f outputs=%d\n",
           K,nr,nc,vlen,seed,INJECT,(double)TOLK,no);
    printf("  ours  vs oracle: mism=%d/%d worst_tolratio=%.3e max_rel=%.3e\n", vR.mism,no,vR.worst_tolratio,vR.max_rel);
    printf("  OPP-X vs oracle: mism=%d/%d worst_tolratio=%.3e max_rel=%.3e\n", vX.mism,no,vX.worst_tolratio,vX.max_rel);
    printf("  OPP-S vs oracle: mism=%d/%d worst_tolratio=%.3e max_rel=%.3e\n", vS.mism,no,vS.worst_tolratio,vS.max_rel);
    printf("  xcheck: ours==OPP-X bitexact %d/%d | ours==OPP-S bitexact %d/%d\n", be_RX,no,be_RS,no);
    // PREREG §4.0 / §4.5: the three gates have DIFFERENT consequences. Do not conflate them.
    printf("  GATE-OURS = %-8s (%s)\n", vR.mism==0?"PASS":"FAIL",
           vR.mism==0?"correct kernel; perf reportable":"VOID-CORRECTNESS => perf numbers forbidden");
    printf("  GATE-OPPX = %-8s (%s)\n", vX.mism==0?"PASS":"FAIL",
           vX.mism==0?"CROSSOP opponent numerically valid on this board":"OPP-X invalid on this board => OPP-X ratio forbidden");
    printf("  GATE-OPPS = %-8s (%s)\n", vS.mism==0?"PASS":"FAIL",
           vS.mism==0?"PREREG §3.3 layout gate PASS: make_x16/make_q8x4 == ggml block_iq4_nlx16/block_q8_0x4, functionally proven"
                     :"PREREG §3.3 => VOID-S: same-operator opponent numbers forbidden; report OPP-X only, tagged CROSSOP");
#if DIAG
    // Diagnostic only (never gates): where do OPP-S mismatches live? Histogram by lane-in-group
    // (c%16) tells a half-width-VLMAX story (lanes 8..15 dead) apart from a whole-kernel story.
    {
        int hist[16]={0}, tot[16]={0}, zero_lane[16]={0};
        for (int r=0;r<nr;r++) for (int c=0;c<nc;c++){
            size_t i=(size_t)r*nc+c; int lane=c%16; tot[lane]++;
            double e=fabs((double)oS[i]-ora[i]);
            if (tol[i]>0.0 && e/tol[i] > 1.0) hist[lane]++;
            if (oS[i]==0.0f) zero_lane[lane]++;
        }
        printf("  DIAG OPP-S mism by lane(c%%16): ");
        for (int l=0;l<16;l++) printf("%d:%d/%d ", l, hist[l], tot[l]);
        printf("\n  DIAG OPP-S untouched(==0.0f) by lane: ");
        for (int l=0;l<16;l++) printf("%d:%d ", l, zero_lane[l]);
        printf("\n  DIAG sample lane0 r0c0: oS=%.6f ora=%.6f ours=%.6f | lane8 r0c8: oS=%.6f ora=%.6f ours=%.6f\n",
               (double)oS[0], ora[0], (double)ours[0], (double)oS[8], ora[8], (double)ours[8]);
    }
#endif
    if (vo) return (vR.mism==0 && vX.mism==0) ? 0 : 1;   // OPP-S failure = VOID-S, not a cell VOID
    if (vR.mism != 0){ fprintf(stderr,"VOID-CORRECTNESS: ours mismatch => perf numbers forbidden\n"); return 1; }

    // ===== COLD paired A/B/C, within-proc interleave, flush before EACH timed region =====
    g_flush=(uint8_t*)aligned_alloc(64,FLUSH); if(!g_flush){ fprintf(stderr,"OOM flush\n"); return 3; }
    memset(g_flush,1,FLUSH);
    std::vector<double> tR,tX,tS; tR.reserve(reps); tX.reserve(reps); tS.reserve(reps);
    std::vector<float> bR(no),bX(no),bS(no);
    for (int p=0;p<reps;p++){
        cold_flush(); double t0=now_ns();
        weft_emitc_ggml_repack_gemm_iq4_nl_q8_0_kernel_ggml_repack_gemm_iq4_nl_q8_0(
            (size_t)nr,(size_t)nc,(size_t)K,bR.data(),
            (const uint8_t*)packed.data(),(const uint8_t*)apack.data(),(size_t)nc);
        tR.push_back(now_ns()-t0);
        cold_flush(); double t1=now_ns();
        for (int r=0;r<nr;r++) for (int c=0;c<nc;c++)
            ggml_vec_dot_iq4_nl_q8_0(K,&bX[(size_t)r*nc+c],0,W.data()+(size_t)c*nb,0,A.data()+(size_t)r*nb,0,1);
        tX.push_back(now_ns()-t1);
        cold_flush(); double t2=now_ns();
        ggml_gemm_iq4_nl_16x1_q8_0(K,bS.data(),(size_t)nc,(const void*)packed.data(),
                                   (const void*)apack.data(),nr,nc);
        tS.push_back(now_ns()-t2);
    }
    double rM=med(tR), xM=med(tX), sM=med(tS);
    printf("REPS_OURS seed=0x%X :",seed); for(double v:tR) printf(" %.0f",v); printf("\n");
    printf("REPS_OPPX seed=0x%X :",seed); for(double v:tX) printf(" %.0f",v); printf("\n");
    printf("REPS_OPPS seed=0x%X :",seed); for(double v:tS) printf(" %.0f",v); printf("\n");
    printf("GEMM_PREFILL K=%d nr=%d nc=%d reps=%d VLEN=%ld seed=0x%X | ours_med_ns=%.0f(iqr%.2f%%) "
           "oppX_med_ns=%.0f(iqr%.2f%%) oppS_med_ns=%.0f(iqr%.2f%%) | ratio_cold_X=%.4f ratio_cold_S=%.4f sink=%llu\n",
           K,nr,nc,reps,vlen,seed,rM,riqr(tR),xM,riqr(tX),sM,riqr(tS),xM/rM,sM/rM,
           (unsigned long long)g_sink);
    free(g_flush); return 0;
}
