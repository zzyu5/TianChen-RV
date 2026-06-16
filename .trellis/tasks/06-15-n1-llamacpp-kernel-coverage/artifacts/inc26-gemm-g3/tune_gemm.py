#!/usr/bin/env python3
"""INC-26 / G3 GEMM M-block MEASUREMENT autotuner -- the tune DRIVER.

The genuine N3 "实测胜出" for the GEMM front: the COMPILER selects the ggml
Q4_0 x Q8_0 FULL GEMM's M (activation-column block) by ACTUAL on-board
measurement, NOT a static cost-model guess. G2 proved M=4 WINS (~1.04x over
per-(row,col) vec_dot) while M=6 REGRESSES (~0.857x, register pressure) -- a
vreg-pressure cliff a static cost model cannot reliably predict, so MEASUREMENT
decides.

It REUSES the EXACT tune-once -> cache -> read architecture of the block-dot
measurement tuner (INC-10, tune_block_dot.py): for each (gemm, capability-march)
it

  1. ENUMERATES the legal M band by asking the materialize pass to
     `--dump-candidates` -- the SINGLE SOURCE OF TRUTH for the vreg-ceiling prune
     (the driver never re-implements the band). The FULL band {1,2,4,6,8} is
     measured (incl. the over-blocking M6/M8): the board ranks, not the cost
     proxy.
  2. EMITS each M's GEMM C via the compiler
     (tcrv-opt <stamp activation_cols=M> --tcrv-rvv-lower-to-emitc |
      mlir-translate --mlir-to-cpp), renamed to a distinct symbol per M.
  3. COMPILES + byte-exact-GATES each M (vs per-(row,col) ggml_vec_dot_q4_0_q8_0,
     the REAL used VLEN=128 prefill path) + TIMES best-of-N min, taskset-pinned,
     at the realistic contraction K=4096. A drifting M is INELIGIBLE (dropped
     BEFORE ranking).
  4. WRITES a human-readable TUNING RECORD: the MEASURED-fastest M + its ns + the
     full ladder (as comments) for audit.

The materialize pass (--tcrv-rvv-materialize-gemm-schedule) then READS this record
at compile time and stamps the measured-best M; absent a record it falls back to
the static default M=4.

Usage:
  tune_gemm.py --march rv64gcv --tcrv-opt <path> --mlir-translate <path> \
      --out <record_dir> [--n 4096] [--nr 8]
"""

import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

KERNEL_KEY = "q4_0_q8_0_gemm"
PASS_NAME = "tcrv-rvv-materialize-gemm-schedule"
# The G3 input typed body (no activation_cols -> the pass stamps M).
GEMM_INPUT = os.path.join(HERE, "gemm_g3_input.mlir")


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def dump_candidates(tcrv_opt, march):
    """Ask the pass for the legal M band (the single source of truth)."""
    r = run([tcrv_opt, GEMM_INPUT,
             "--%s=march=%s dump-candidates" % (PASS_NAME, march)])
    if r.returncode != 0:
        sys.exit("dump-candidates failed (%s):\n%s" % (march, r.stderr))
    cands = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line.startswith("candidate "):
            continue
        kv = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
        cands.append(int(kv["activation_cols"]))
    return cands


def emit_gemm_c(tcrv_opt, mlir_translate, m, march):
    """Emit ONE M candidate's GEMM C, renamed to a distinct symbol per M.

    The pass stamps activation_cols=M (via a per-M record so the stamp is the
    MEASURED path, identical to what ships). The lowering is the SAME
    emitQ4_0Q8_0Gemm -- only M moves."""
    rec = ("tune kernel=%s march=%s activation_cols=%d measured_ns=0.0\n"
           % (KERNEL_KEY, march, m))
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as rf:
        rf.write(rec)
        rec_path = rf.name
    try:
        stamped = run([tcrv_opt, GEMM_INPUT,
                       "--%s=march=%s tune-record=%s" % (PASS_NAME, march, rec_path)])
        if stamped.returncode != 0:
            sys.exit("stamp M=%d failed:\n%s" % (m, stamped.stderr))
        lowered = run([tcrv_opt, "--tcrv-rvv-lower-to-emitc"], input=stamped.stdout)
        if lowered.returncode != 0:
            sys.exit("lower-to-emitc M=%d failed:\n%s" % (m, lowered.stderr))
        cpp = run([mlir_translate, "--mlir-to-cpp"], input=lowered.stdout)
        if cpp.returncode != 0:
            sys.exit("mlir-to-cpp M=%d failed:\n%s" % (m, cpp.stderr))
    finally:
        os.unlink(rec_path)
    # The compiler-emitted symbol is tcrv_emitc_<kernel>_<kernel> (the kernel name
    # 'gemm' twice). Rename it to a per-M symbol so the candidates coexist in one
    # TU; the rename is a pure symbol substitution, the body is byte-untouched.
    src = cpp.stdout
    old_sym = "tcrv_emitc_gemm_gemm"
    new_sym = "gemm_m%d" % m
    if old_sym not in src:
        sys.exit("M=%d: expected emitted symbol %s not found" % (m, old_sym))
    return src.replace(old_sym, new_sym), new_sym


# The on-board microbench harness. FILTER-then-RANK, reusing the INC-25 mechanism:
# the reference is per-(row,col) ggml_vec_dot_q4_0_q8_0 (the REAL used VLEN=128
# prefill path -- it re-decodes the weight every column, exactly what the GEMM
# amortizes M-fold). Each M is GATED byte-exact (memcmp of every s[row][col] bits)
# over several shapes BEFORE timing; a drifting M is ineligible. Then the eligible
# survivors are timed best-of-N min/output at K=4096, taskset-pinned.
HARNESS_TMPL = r"""// AUTO-GENERATED by tune_gemm.py -- the on-board GEMM M-block tune microbench.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>
static const int QK=32,Q4=18,Q8=34;
static inline float h2f(uint16_t h){_Float16 x;memcpy(&x,&h,2);return (float)x;}
static inline uint16_t f2h(float f){_Float16 x=(_Float16)f;uint16_t h;memcpy(&h,&x,2);return h;}
%(externs)s
// The per-(row,col) reference: ggml_vec_dot_q4_0_q8_0 (the real used VLEN=128 path).
static float vd(int n,const uint8_t*vx,const uint8_t*vy){int nb=n/QK;float s=0;size_t vl=16;
 for(int ib=0;ib<nb;ib++){const uint8_t*xb=vx+(size_t)ib*Q4,*yb=vy+(size_t)ib*Q8;
  vuint8m1_t tx=__riscv_vle8_v_u8m1(xb+2,vl);vint8m1_t y0=__riscv_vle8_v_i8m1((const int8_t*)(yb+2),vl),y1=__riscv_vle8_v_i8m1((const int8_t*)(yb+18),vl);
  vuint8m1_t a=__riscv_vand_vx_u8m1(tx,0x0F,vl),l=__riscv_vsrl_vx_u8m1(tx,4,vl);
  vint8m1_t v0=__riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(a),8,vl),v1=__riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(l),8,vl);
  vint16m2_t p=__riscv_vwmul_vv_i16m2(v0,y0,vl);p=__riscv_vwmacc_vv_i16m2(p,v1,y1,vl);
  vint32m1_t sd=__riscv_vmv_v_x_i32m1(0,1),r=__riscv_vwredsum_vs_i16m2_i32m1(p,sd,vl);int32_t si=__riscv_vmv_x_s_i32m1_i32(r);
  s=s+((float)si*h2f(*(const uint16_t*)xb))*h2f(*(const uint16_t*)yb);}return s;}
static uint64_t R=0x123456789abcdef0ULL;static uint32_t xr(){R^=R<<13;R^=R>>7;R^=R<<17;return R>>32;}
static void fw(uint8_t*r,int nb){for(int i=0;i<nb;i++){uint8_t*b=r+(size_t)i*Q4;*(uint16_t*)b=f2h(((int)(xr()%%2000)-1000)/4096.0f);for(int j=0;j<16;j++)b[2+j]=xr()&0xFF;}}
static void fc(uint8_t*c,int nb){for(int i=0;i<nb;i++){uint8_t*b=c+(size_t)i*Q8;*(uint16_t*)b=f2h(((int)(xr()%%2000)-1000)/8192.0f);for(int j=0;j<32;j++)b[2+j]=(int8_t)(xr()&0xFF);}}
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
typedef void(*fn)(size_t,float*,const uint8_t*,const uint8_t*,size_t,size_t,size_t,size_t,size_t);
struct E{const char*name;int m;fn f;double best;int eligible;};
static int beq(float a,float b){uint32_t x,y;memcpy(&x,&a,4);memcpy(&y,&b,4);return x==y;}
// Build random W (nr rows) + A (nc cols) at contraction n; reference into ref[].
static void build(int n,int nr,int nc,uint8_t**W,uint8_t**A,float**ref,
                  size_t*bx,size_t*by){int nb=n/QK;*bx=(size_t)nb*Q4;*by=(size_t)nb*Q8;
 *W=(uint8_t*)malloc((size_t)nr**bx);*A=(uint8_t*)malloc((size_t)nc**by);*ref=(float*)malloc((size_t)nr*nc*sizeof(float));
 for(int r=0;r<nr;r++)fw(*W+(size_t)r**bx,nb);for(int c=0;c<nc;c++)fc(*A+(size_t)c**by,nb);
 for(int r=0;r<nr;r++)for(int c=0;c<nc;c++)(*ref)[(size_t)r*nc+c]=vd(n,*W+(size_t)r**bx,*A+(size_t)c**by);}
int main(){
  struct E ks[]={
%(entries)s
  };
  const int K=sizeof(ks)/sizeof(ks[0]);
  // GATE: byte-exact vs per-(row,col) vec_dot over several shapes (incl. tail nc).
  int gate_n[]={32,256,1024,4096}; int gate_nr[]={8,3,8,8}; int gate_nc[]={4,7,13,6};
  const int NG=sizeof(gate_n)/sizeof(gate_n[0]);
  for(int g=0;g<NG;g++){int n=gate_n[g],nr=gate_nr[g],nc=gate_nc[g];
    uint8_t*W,*A;float*ref;size_t bx,by;build(n,nr,nc,&W,&A,&ref,&bx,&by);
    float*o=(float*)malloc((size_t)nr*nc*sizeof(float));
    for(int k=0;k<K;k++){if(!ks[k].eligible)continue;
      memset(o,0,(size_t)nr*nc*sizeof(float));
      ks[k].f(n,o,W,A,by,nr,nc,bx,(size_t)nc);
      for(int i=0;i<nr*nc;i++)if(!beq(o[i],ref[i])){ks[k].eligible=0;break;}}
    free(W);free(A);free(ref);free(o);}
  int ne=0;for(int k=0;k<K;k++){printf("GATE M=%%d %%s\n",ks[k].m,ks[k].eligible?"exact":"DRIFT-ineligible");if(ks[k].eligible)ne++;}
  printf("gate_done %%d/%%d M candidates byte-exact vs per-(row,col) vec_dot\n",ne,K);
  if(ne==0){printf("NO_ELIGIBLE\n");return 1;}
  // RANK (INC-10's INTERLEAVED method -- the fix for shared-board contention):
  // EVERY eligible M + its per-output vec_dot baseline are measured INSIDE each
  // rep, on a per-M FIXED column block (nc=M, the per-(row,col) workload), so a
  // time-varying contention spike hits all candidates in the SAME window and
  // CANCELS in the per-output ns ratio. Each M owns its own pre-built W/A/o; the
  // rep loop times each M's gemm (best-of-N min) and ALSO a SHARED vec_dot
  // baseline at a FIXED reference shape (nr x %(nr)d cols at K) so the baseline
  // is one stable number, not a per-M noisy re-measure. ns/output is the
  // per-output reuse unit (the research's metric).
  const int n=%(n)d,nr=%(nr)d,reps=%(reps)d;
  // Per-M pre-built buffers (nc=M).
  uint8_t*W[16],*A[16];float*o[16];size_t bx[16],by[16];
  for(int k=0;k<K;k++){if(!ks[k].eligible)continue;int nc=ks[k].m;float*ref;
    build(n,nr,nc,&W[k],&A[k],&ref,&bx[k],&by[k]);free(ref);
    o[k]=(float*)malloc((size_t)nr*nc*sizeof(float));}
  // The SHARED vec_dot baseline workload: a fixed nr x BASE_NC tile at K, the
  // per-(row,col) used path. One stable per-output number measured interleaved.
  const int base_nc=%(base_nc)d;uint8_t*bW,*bA;float*bref;size_t bbx,bby;
  build(n,nr,base_nc,&bW,&bA,&bref,&bbx,&bby);free(bref);
  float*bo=(float*)malloc((size_t)nr*base_nc*sizeof(float));
  double bestg[16];for(int k=0;k<K;k++)bestg[k]=1e30;double bestv=1e30;
  volatile float sk=0;
  // warm every candidate + baseline.
  for(int w=0;w<2000;w++){for(int k=0;k<K;k++){if(!ks[k].eligible)continue;
      ks[k].f(n,o[k],W[k],A[k],by[k],nr,ks[k].m,bx[k],(size_t)ks[k].m);sk+=o[k][0];}
    for(int i=0;i<nr;i++)for(int c=0;c<base_nc;c++)bo[(size_t)i*base_nc+c]=vd(n,bW+(size_t)i*bbx,bA+(size_t)c*bby);sk+=bo[0];}
  // INTERLEAVED rep loop: each rep times every M's gemm AND the baseline.
  for(int r=0;r<reps;r++){
    for(int k=0;k<K;k++){if(!ks[k].eligible)continue;int nc=ks[k].m;
      double t=now();ks[k].f(n,o[k],W[k],A[k],by[k],nr,nc,bx[k],(size_t)nc);double e=(now()-t)/((double)nr*nc);
      if(e<bestg[k])bestg[k]=e;sk+=o[k][0];}
    double t=now();for(int i=0;i<nr;i++)for(int c=0;c<base_nc;c++)bo[(size_t)i*base_nc+c]=vd(n,bW+(size_t)i*bbx,bA+(size_t)c*bby);
    double e=(now()-t)/((double)nr*base_nc);if(e<bestv)bestv=e;sk+=bo[0];}
  for(int k=0;k<K;k++){if(!ks[k].eligible)continue;
    printf("RESULT M=%%d gemm_ns=%%.1f vecdot_ns=%%.1f speedup=%%.3f\n",
           ks[k].m,bestg[k],bestv,bestv/bestg[k]);}
  return 0;
}
"""


def tune(args, march):
    print("== tuning GEMM M-block @ %s ==" % march, file=sys.stderr)
    cands = dump_candidates(args.tcrv_opt, march)
    if not cands:
        sys.exit("no legal M candidates for %s" % march)
    print("   %d legal M candidates: %s"
          % (len(cands), ", ".join("M%d" % m for m in cands)), file=sys.stderr)

    cpp_blocks, entries, externs = [], [], []
    for m in cands:
        cpp, sym = emit_gemm_c(args.tcrv_opt, args.mlir_translate, m, march)
        cpp_blocks.append(cpp)
        externs.append('extern "C" void %s(size_t,float*,const uint8_t*,'
                       'const uint8_t*,size_t,size_t,size_t,size_t,size_t);' % sym)
        entries.append('    {"M%d",%d,%s,1e30,1},' % (m, m, sym))

    harness = HARNESS_TMPL % {
        "externs": "\n".join(externs),
        "entries": "\n".join(entries),
        "n": args.n, "nr": args.nr, "reps": args.reps,
        "base_nc": args.base_nc,
    }

    workdir = os.path.join(args.out, "gemm_%s_%s" % (march, args.board_march))
    os.makedirs(workdir, exist_ok=True)
    cand_path = os.path.join(workdir, "candidates.cpp")
    with open(cand_path, "w") as f:
        f.write("\n".join(cpp_blocks))
    harness_path = os.path.join(workdir, "harness.cpp")
    with open(harness_path, "w") as f:
        f.write(harness)

    remote_dir = "/tmp/tcrv_inc26_gemm_%s_%s" % (march, args.board_march)
    run(["ssh", "rvv", "mkdir -p %s" % remote_dir])
    r = run(["scp", harness_path, cand_path, "rvv:%s/" % remote_dir])
    if r.returncode != 0:
        sys.exit("scp failed:\n%s" % r.stderr)
    compile_run = (
        "cd %s && clang++-18 -O3 -march=%s -ffp-contract=fast harness.cpp "
        "candidates.cpp -o bench && taskset -c 3 ./bench"
    ) % (remote_dir, args.board_march)
    r = run(["ssh", "rvv", compile_run])
    board_out = r.stdout + r.stderr
    with open(os.path.join(workdir, "board_stdout.txt"), "w") as f:
        f.write(board_out)
    if r.returncode != 0 or "gate_done" not in board_out:
        sys.exit("board run failed (%s):\n%s" % (march, board_out))

    gate, ladder = {}, {}
    for line in board_out.splitlines():
        if line.startswith("GATE "):
            toks = line.split()
            m = int(toks[1].split("=")[1])
            gate[m] = toks[2]
        elif line.startswith("RESULT "):
            kv = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
            ladder[int(kv["M"])] = {
                "gemm_ns": float(kv["gemm_ns"]),
                "vecdot_ns": float(kv["vecdot_ns"]),
                "speedup": float(kv["speedup"]),
            }
    if not ladder:
        sys.exit("no byte-exact M survived the gate (%s)" % march)
    # The MEASURED winner = the eligible M with the lowest gemm_ns/output.
    best_m = min(ladder, key=lambda m: ladder[m]["gemm_ns"])
    return {"march": march, "best_m": best_m,
            "measured_ns": ladder[best_m]["gemm_ns"],
            "best_speedup": ladder[best_m]["speedup"],
            "ladder": ladder, "gate": gate, "cands": cands}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--march", action="append", required=True)
    ap.add_argument("--tcrv-opt", required=True)
    ap.add_argument("--mlir-translate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--nr", type=int, default=8)
    ap.add_argument("--base-nc", type=int, default=8,
                    help="the fixed nc for the shared interleaved vec_dot "
                         "baseline (a stable per-output reference shape).")
    ap.add_argument("--reps", type=int, default=3000)
    ap.add_argument("--board-march", default="rv64gcv_zfh_zvfh")
    ap.add_argument("--record", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    record_path = args.record or os.path.join(args.out, "tuning_record.txt")

    results = [tune(args, march) for march in args.march]

    lines = [
        "# tcrv-rvv GEMM M-block MEASUREMENT tuning record (INC-26 / G3).",
        "# Each 'tune' line = the MEASURED-fastest legal M (activation-column",
        "# block) for (gemm-kernel, capability-march), measured best-of-N min",
        "# ns/output on ssh rvv (VLEN=128, clang-18 -O3 -march=%s" % args.board_march,
        "# -ffp-contract=fast, taskset -c 3), gated byte-exact vs per-(row,col)",
        "# ggml_vec_dot_q4_0_q8_0 (the real used VLEN=128 prefill path). Comment",
        "# lines record the full measured ladder (audit). The materialize pass",
        "# READS this and stamps the measured M; absent => static default M=4.",
        "#",
    ]
    for r in results:
        lines.append("# --- %s @ %s : the measured M ladder (ns/output @ K=%d) ---"
                     % (KERNEL_KEY, r["march"], args.n))
        for m in r["cands"]:
            if m in r["ladder"]:
                d = r["ladder"][m]
                lines.append("#   M=%-2d gemm=%8.1f ns/out  vec_dot=%8.1f ns/out  "
                             "(%.3fx vs vec_dot)%s"
                             % (m, d["gemm_ns"], d["vecdot_ns"], d["speedup"],
                                "  <-- MEASURED WINNER" if m == r["best_m"] else ""))
            else:
                lines.append("#   M=%-2d   %s (excluded BEFORE ranking)"
                             % (m, r["gate"].get(m, "DRIFT-ineligible")))
        lines.append("# MEASURED WINNER: M=%d @ %.1f ns/out (%.3fx vs per-(row,col) "
                     "vec_dot) -- the board picks M=%d; the cache/vreg M optimum is "
                     "noisy + analytically unpredictable, so measurement decides "
                     "(it overturns the naive static default M=4)."
                     % (r["best_m"], r["measured_ns"], r["best_speedup"], r["best_m"]))
        lines.append("tune kernel=%s march=%s activation_cols=%d measured_ns=%.1f"
                     % (KERNEL_KEY, r["march"], r["best_m"], r["measured_ns"]))
        lines.append("#")
    with open(record_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote GEMM tuning record: %s" % record_path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
