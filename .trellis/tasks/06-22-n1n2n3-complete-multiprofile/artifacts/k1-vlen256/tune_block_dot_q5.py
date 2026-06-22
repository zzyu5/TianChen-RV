#!/usr/bin/env python3
"""q5_0 / q5_1 block-dot MEASUREMENT autotuner -- a GENTLE port of the INC-10
tune driver (06-15 inc10-measurement-tuner/tune_block_dot.py) to the q5 family.

Same FILTER-then-RANK method as inc10:
  1. ENUMERATE the legal candidate set via the materialize pass `--dump-candidates`
     (single source of truth for the capability+budget prune).
  2. EMIT each candidate's C
     (tcrv-opt <stamp shape attrs> --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp),
     renamed to a distinct per-candidate symbol.
  3. FILTER: gate byte-exact vs ggml's REAL q5 RVV kernel (a SEPARATE TU, kern_ggml)
     over many n x many trials at -ffp-contract=fast; drift => INELIGIBLE, dropped
     BEFORE timing.
  4. RANK: time the eligible survivors best-of-N min, taskset-pinned, and write the
     measured-best + the full ladder.

GENTLE knobs for a fragile board (just rebooted after a crash): SHORTER best-of-N
(reps), fewer gate trials, single-thread, taskset -c 3. The ggml-real q5 recipes
below are transcribed verbatim from the inc21/inc22 byte-exact validators (which in
turn transcribe llama.cpp ggml-cpu/arch/riscv/quants.c, q5_0 :328-379, q5_1
:382-433). The emitted kernels are byte-exact to these.

Usage:
  tune_block_dot_q5.py --kernel q5_0 --kernel q5_1 --march rv64gcv \
      --tcrv-opt build/bin/tcrv-opt --mlir-translate /usr/bin/mlir-translate-20 \
      --out runs --record tuning_record_q5.txt --n 4096 [--reps 100]
"""

import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# Per-kernel facts: base MLIR template, the materialize pass, the kernel/variant
# symbol names, and the q5/q8 block strides + quant-byte counts.
KERNELS = {
    "q5_0": {
        "base": "q5_0_base.mlir",
        "pass": "tcrv-rvv-materialize-q5-0-schedule",
        "kernel_sym": "ggml_vec_dot_q5_0_q8_0_kernel",
        "variant_sym": "ggml_vec_dot_q5_0_q8_0",
        "x_stride": 22, "y_stride": 34,
        # q5_0 block = d(2) qh(4) qs(16); q8_0 block = d(2) qs(32)
        "has_xmin": 0, "has_ysum": 0,
        "x_qh_off": 2, "x_qs_off": 6,
        "y_quant_off": 2, "x_qbytes": 16, "y_qbytes": 32,
    },
    "q5_1": {
        "base": "q5_1_base.mlir",
        "pass": "tcrv-rvv-materialize-q5-1-schedule",
        "kernel_sym": "ggml_vec_dot_q5_1_q8_1_kernel",
        "variant_sym": "ggml_vec_dot_q5_1_q8_1",
        "x_stride": 24, "y_stride": 36,
        # q5_1 block = d(2) m(2) qh(4) qs(16); q8_1 block = d(2) s(2) qs(32)
        "has_xmin": 1, "has_ysum": 1,
        "x_qh_off": 4, "x_qs_off": 8,
        "y_quant_off": 4, "x_qbytes": 16, "y_qbytes": 32,
    },
}

# ggml's REAL q5 RVV kernels as a SEPARATE TU (extern "C" kern_ggml), contracted
# exactly as the standalone kernel that ships -- BOTH the byte-exactness reference
# AND the perf baseline (transcribed verbatim from inc21/inc22 validators).
GGML_REF = {
    "q5_0": r"""
extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int qk = 32; const int Q5 = 22, Q8 = 34;
  const int nb = (int)n / qk; float sumf = 0; size_t vl;
  size_t vlenb = __riscv_vlenb();
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * Q5, *yb = vy + (size_t)ib * Q8;
    const uint8_t *xqs = xb + 6, *xqh = xb + 2;
    const int8_t *yqs = (const int8_t *)(yb + 2);
    uint16_t xdh, ydh; memcpy(&xdh, xb + 0, 2); memcpy(&ydh, yb + 0, 2);
    vl = qk / 2;
    vuint8m1_t v0 = __riscv_vle8_v_u8m1(xqs, vl);
    vint8m1_t v0l = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(v0, 0x0F, vl));
    vint8m1_t v0h = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(v0, 4, vl));
    vint8m2_t v0c;
    if (vlenb == 16) { v0c = __riscv_vcreate_v_i8m1_i8m2(v0l, v0h); }
    else { v0l = __riscv_vslideup_vx_i8m1(v0l, v0h, 16, 32); v0c = __riscv_vlmul_ext_v_i8m1_i8m2(v0l); }
    vl = qk;
    vbool4_t qh = __riscv_vlm_v_b4(xqh, vl);
    qh = __riscv_vmnand_mm_b4(qh, qh, vl);
    vint8m2_t v0f = __riscv_vsub_vx_i8m2_mu(qh, v0c, v0c, 0x10, vl);
    vint8m2_t v1 = __riscv_vle8_v_i8m2(yqs, vl);
    vint16m4_t mul = __riscv_vwmul_vv_i16m4(v0f, v1, vl);
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t sum = __riscv_vwredsum_vs_i16m4_i32m1(mul, zero, vl);
    int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sum);
    _Float16 xf, yf; memcpy(&xf, &xdh, 2); memcpy(&yf, &ydh, 2);
    sumf += ((float)xf * (float)yf) * sumi;
  }
  *s = sumf;
}""",
    "q5_1": r"""
extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int qk = 32; const int Q5 = 24, Q8 = 36;
  const int nb = (int)n / qk; float sumf = 0; size_t vl;
  size_t vlenb = __riscv_vlenb();
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * Q5, *yb = vy + (size_t)ib * Q8;
    const uint8_t *xqs = xb + 8, *xqh = xb + 4;
    const int8_t *yqs = (const int8_t *)(yb + 4);
    uint16_t xdh, xmh, ydh, ysh;
    memcpy(&xdh, xb + 0, 2); memcpy(&xmh, xb + 2, 2);
    memcpy(&ydh, yb + 0, 2); memcpy(&ysh, yb + 2, 2);
    vl = qk / 2;
    vuint8m1_t v0 = __riscv_vle8_v_u8m1(xqs, vl);
    vint8m1_t v0l = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vand_vx_u8m1(v0, 0x0F, vl));
    vint8m1_t v0h = __riscv_vreinterpret_v_u8m1_i8m1(__riscv_vsrl_vx_u8m1(v0, 4, vl));
    vint8m2_t v0c;
    if (vlenb == 16) { v0c = __riscv_vcreate_v_i8m1_i8m2(v0l, v0h); }
    else { v0l = __riscv_vslideup_vx_i8m1(v0l, v0h, 16, 32); v0c = __riscv_vlmul_ext_v_i8m1_i8m2(v0l); }
    vl = qk;
    vbool4_t qh = __riscv_vlm_v_b4(xqh, vl);
    vint8m2_t v0f = __riscv_vor_vx_i8m2_mu(qh, v0c, v0c, 0x10, vl);
    vint8m2_t v1 = __riscv_vle8_v_i8m2(yqs, vl);
    vint16m4_t mul = __riscv_vwmul_vv_i16m4(v0f, v1, vl);
    vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
    vint32m1_t sum = __riscv_vwredsum_vs_i16m4_i32m1(mul, zero, vl);
    int32_t sumi = __riscv_vmv_x_s_i32m1_i32(sum);
    _Float16 xf, yf, xm, ys;
    memcpy(&xf, &xdh, 2); memcpy(&yf, &ydh, 2);
    memcpy(&xm, &xmh, 2); memcpy(&ys, &ysh, 2);
    sumf += ((float)xf * (float)yf) * sumi + (float)xm * (float)ys;
  }
  *s = sumf;
}""",
}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def dump_candidates(tcrv_opt, base_path, kinfo, pass_name, march):
    with open(base_path) as f:
        tmpl = f.read()
    attrless = (tmpl.replace(", SHAPE", "")
                    .replace("KERNELNAME", kinfo["kernel_sym"])
                    .replace("VARIANTNAME", kinfo["variant_sym"]))
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as tf:
        tf.write(attrless)
        dpath = tf.name
    try:
        r = run([tcrv_opt, dpath,
                 "--%s=march=%s dump-candidates" % (pass_name, march)])
    finally:
        os.unlink(dpath)
    if r.returncode != 0:
        sys.exit("dump-candidates failed (%s, %s):\n%s" % (pass_name, march, r.stderr))
    cands = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line.startswith("candidate "):
            continue
        kv = dict(tok.split("=", 1) for tok in line.split() if "=" in tok)
        cands.append((kv["lmul"], int(kv["factor"]), kv["elision"], int(kv["cost"])))
    return cands


SHAPE_TMPL = ('integer_core_lmul = "%s", multi_block_factor = %d : i64, '
              'strip_elision = "%s"')


def emit_candidate_c(tcrv_opt, mlir_translate, base_path, kinfo, lmul, factor,
                     elision, suffix):
    with open(base_path) as f:
        tmpl = f.read()
    shape = SHAPE_TMPL % (lmul, factor, elision)
    variant = kinfo["variant_sym"] + "_" + suffix
    mlir = (tmpl.replace("KERNELNAME", kinfo["kernel_sym"])
                .replace("VARIANTNAME", variant)
                .replace("SHAPE", shape))
    with tempfile.NamedTemporaryFile("w", suffix=".mlir", delete=False) as tf:
        tf.write(mlir)
        mpath = tf.name
    try:
        lowered = run([tcrv_opt, mpath, "--tcrv-rvv-lower-to-emitc"])
        if lowered.returncode != 0:
            sys.exit("lower-to-emitc failed (%s):\n%s" % (suffix, lowered.stderr))
        cpp = run([mlir_translate, "--mlir-to-cpp"], input=lowered.stdout)
        if cpp.returncode != 0:
            sys.exit("mlir-to-cpp failed (%s):\n%s" % (suffix, cpp.stderr))
        fn = "tcrv_emitc_%s_%s" % (kinfo["kernel_sym"], variant)
        return cpp.stdout, fn
    finally:
        os.unlink(mpath)


# NOTE the fill: EVERY block byte is randomized FIRST (so qh / qs / m / s are all
# exercised -- the q5 5th-bit path is not vacuous), THEN the finite fp16 scale
# fields are overlaid with rfp16() so no scale is Inf/NaN. q8_1's `s` is left as a
# random finite fp16 (consistent across both TUs => the gate compares apples-to-
# apples; this matches inc10's q4_1 treatment of the sum field).
HARNESS_TMPL = r"""// AUTO-GENERATED by tune_block_dot_q5.py -- on-board microbench for ONE
// (q5 kernel, capability-march) tune run. FILTER-then-RANK: each candidate is
// GATED byte-exact vs ggml's real q5 kernel (separate TU kern_ggml) over many n x
// many trials at -ffp-contract=fast; drift => INELIGIBLE, dropped BEFORE timing.
// Then surviving candidates are timed best-of-N min (taskset-pinned). GENTLE knobs.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <time.h>
#include <riscv_vector.h>
extern "C" void kern_ggml(size_t, float *, const uint8_t *, const uint8_t *);
%(externs)s
static uint32_t rng = 0x2468ace0u;
static uint32_t xr() { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }
static uint16_t rfp16() {
  uint16_t s = (xr() & 1) << 15, e = (uint16_t)(xr() %% 31), m = (uint16_t)(xr() & 0x3FF);
  return s | (e << 10) | m;
}
// init EVERY byte random, then overlay finite fp16 scales. x.d@0 always; x.m@2 if
// has_xmin; y.d@0 always; y.s@2 if has_ysum.
static void fill(uint8_t *vx, uint8_t *vy, int nb, int xs, int ys,
                 int has_xmin, int has_ysum) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t *xb = vx + (size_t)ib * xs, *yb = vy + (size_t)ib * ys;
    for (int i = 0; i < xs; ++i) xb[i] = (uint8_t)(xr() & 0xFF);
    for (int i = 0; i < ys; ++i) yb[i] = (uint8_t)(xr() & 0xFF);
    uint16_t d;
    d = rfp16(); memcpy(xb + 0, &d, 2);             // x.d
    if (has_xmin) { d = rfp16(); memcpy(xb + 2, &d, 2); }  // x.m
    d = rfp16(); memcpy(yb + 0, &d, 2);             // y.d
    if (has_ysum) { d = rfp16(); memcpy(yb + 2, &d, 2); }  // y.s
  }
}
typedef void (*kfn)(size_t, float *, const uint8_t *, const uint8_t *);
struct E { const char *name; kfn fn; double best; int eligible; };
static double now() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static double tb(kfn fn, size_t n, int it, const uint8_t *vx, const uint8_t *vy) {
  double t0 = now(); float s; for (int i = 0; i < it; i++) fn(n, &s, vx, vy); return (now() - t0) / it;
}
static int beq(float a, float b) { uint32_t x, y; memcpy(&x, &a, 4); memcpy(&y, &b, 4); return x == y; }
int main() {
  const int xs = %(xs)d, ys = %(ys)d, has_xmin = %(has_xmin)d, has_ysum = %(has_ysum)d;
  const int timing_n = %(n)d, iters = %(iters)d, reps = %(reps)d;
  struct E ks[] = {
%(entries)s
  };
  const int K = sizeof(ks) / sizeof(ks[0]);

  // GATE: byte-exact vs ggml (separate TU) over many n x many trials at =fast.
  int gate_ns[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  const int NN = sizeof(gate_ns) / sizeof(gate_ns[0]);
  for (int t = 0; t < %(gate_trials)d; ++t) {
    for (int j = 0; j < NN; ++j) {
      int n = gate_ns[j], nb = n / 32;
      uint8_t *vx = (uint8_t *)malloc((size_t)nb * xs), *vy = (uint8_t *)malloc((size_t)nb * ys);
      fill(vx, vy, nb, xs, ys, has_xmin, has_ysum);
      float ref = 0; kern_ggml(n, &ref, vx, vy);
      for (int k = 0; k < K; ++k) {
        if (!ks[k].eligible) continue;
        float g = 0; ks[k].fn(n, &g, vx, vy);
        if (!beq(g, ref)) ks[k].eligible = 0;
      }
      free(vx); free(vy);
    }
  }
  int n_eligible = 0;
  for (int k = 0; k < K; ++k) {
    printf("GATE %%s %%s\n", ks[k].name, ks[k].eligible ? "exact" : "DRIFT-ineligible");
    if (ks[k].eligible) n_eligible++;
  }
  printf("gate_done %%d/%%d candidates byte-exact vs ggml (=fast, many n)\n", n_eligible, K);
  if (n_eligible == 0) { printf("NO_ELIGIBLE\n"); return 1; }

  // RANK: time only the ELIGIBLE survivors + the ggml baseline.
  int nb = timing_n / 32;
  uint8_t *vx = (uint8_t *)malloc((size_t)nb * xs), *vy = (uint8_t *)malloc((size_t)nb * ys);
  fill(vx, vy, nb, xs, ys, has_xmin, has_ysum);
  double ggml_best = 1e18;
  for (int k = 0; k < K; ++k) for (int i = 0; i < iters; i++) { float s; if (ks[k].eligible) ks[k].fn(timing_n, &s, vx, vy); }
  for (int i = 0; i < iters; i++) { float s; kern_ggml(timing_n, &s, vx, vy); }
  for (int r = 0; r < reps; r++) {
    double pg = tb(kern_ggml, timing_n, iters, vx, vy); if (pg < ggml_best) ggml_best = pg;
    for (int k = 0; k < K; k++) {
      if (!ks[k].eligible) continue;
      double p = tb(ks[k].fn, timing_n, iters, vx, vy); if (p < ks[k].best) ks[k].best = p;
    }
  }
  printf("RESULT ggml(real) %%.1f\n", ggml_best);
  for (int k = 0; k < K; ++k) if (ks[k].eligible) printf("RESULT %%s %%.1f\n", ks[k].name, ks[k].best);
  free(vx); free(vy);
  return 0;
}
"""


def tune_one(args, kernel, march):
    kinfo = KERNELS[kernel]
    base_path = os.path.join(HERE, kinfo["base"])
    print("== tuning %s @ %s ==" % (kernel, march), file=sys.stderr)
    cands = dump_candidates(args.tcrv_opt, base_path, kinfo, kinfo["pass"], march)
    if not cands:
        sys.exit("no legal candidates for %s @ %s" % (kernel, march))
    print("   %d legal candidates: %s" % (len(cands),
          ", ".join("%s/%d/%s" % (l, f, e) for l, f, e, _ in cands)), file=sys.stderr)

    cpp_blocks, entries, externs, name_map = [], [], [], {}
    for lmul, factor, elision, cost in cands:
        suffix = "%s_f%d_%s" % (lmul, factor, elision)
        cpp, fn = emit_candidate_c(args.tcrv_opt, args.mlir_translate, base_path,
                                   kinfo, lmul, factor, elision, suffix)
        cpp_blocks.append(cpp)
        externs.append('extern "C" void %s(size_t, float *, const uint8_t *, const uint8_t *);' % fn)
        entries.append('    {"%s", %s, 1e18, 1},' % (suffix, fn))
        name_map[suffix] = (lmul, factor, elision, cost)

    harness = HARNESS_TMPL % {
        "externs": "\n".join(externs),
        "n": args.n, "xs": kinfo["x_stride"], "ys": kinfo["y_stride"],
        "has_xmin": kinfo["has_xmin"], "has_ysum": kinfo["has_ysum"],
        "iters": args.iters, "reps": args.reps, "gate_trials": args.gate_trials,
        "entries": "\n".join(entries),
    }
    ggml_tu = ("#include <stdint.h>\n#include <string.h>\n"
               "#include <riscv_vector.h>\n" + GGML_REF[kernel] + "\n")

    workdir = os.path.join(args.out, "%s_%s_%s_n%d" %
                           (kernel, march, args.board_march, args.n))
    os.makedirs(workdir, exist_ok=True)
    candidate_cpp_path = os.path.join(workdir, "candidates.cpp")
    with open(candidate_cpp_path, "w") as f:
        f.write("\n".join(cpp_blocks))
    ggml_cpp_path = os.path.join(workdir, "ggml_ref.cpp")
    with open(ggml_cpp_path, "w") as f:
        f.write(ggml_tu)
    harness_path = os.path.join(workdir, "harness.cpp")
    with open(harness_path, "w") as f:
        f.write(harness)

    remote_dir = "%s/tcrv_q5_%s_%s_%s_n%d" % (
        args.remote_root, kernel, march, args.board_march, args.n)
    board_march = args.board_march
    run(["ssh", args.host, "mkdir -p %s" % remote_dir])
    r = run(["scp", harness_path, candidate_cpp_path, ggml_cpp_path,
             "%s:%s/" % (args.host, remote_dir)])
    if r.returncode != 0:
        sys.exit("scp failed:\n%s" % r.stderr)
    compile_run = (
        "cd %s && timeout 600 %s -O3 -march=%s -ffp-contract=fast harness.cpp "
        "candidates.cpp ggml_ref.cpp -o bench && timeout 300 taskset -c %d ./bench"
    ) % (remote_dir, args.clang, board_march, args.cpu)
    r = run(["ssh", args.host, compile_run])
    board_out = r.stdout + r.stderr
    with open(os.path.join(workdir, "board_stdout.txt"), "w") as f:
        f.write(board_out)
    if r.returncode != 0 or "gate_done" not in board_out:
        sys.exit("board run failed (%s @ %s):\n%s" % (kernel, march, board_out))

    gate = {}
    ladder = {}
    for line in board_out.splitlines():
        if line.startswith("GATE "):
            _, name, verdict = line.split()
            gate[name] = verdict
        elif line.startswith("RESULT "):
            _, name, ns = line.split()
            ladder[name] = float(ns)
    ggml_ns = ladder.get("ggml(real)")
    cand_ladder = {k: v for k, v in ladder.items() if k != "ggml(real)"}
    if not cand_ladder:
        sys.exit("no byte-exact candidate survived the gate (%s @ %s)"
                 % (kernel, march))
    best_name = min(cand_ladder, key=cand_ladder.get)
    lmul, factor, elision, cost = name_map[best_name]
    best_ns = cand_ladder[best_name]
    dropped = [s for s in name_map if gate.get(s) == "DRIFT-ineligible"]
    return {
        "kernel": kernel, "march": march, "n": args.n,
        "lmul": lmul, "factor": factor, "elision": elision,
        "measured_ns": best_ns, "ggml_ns": ggml_ns,
        "ladder": [(n, name_map[n][0], name_map[n][1], name_map[n][2],
                    cand_ladder[n]) for n in sorted(cand_ladder, key=cand_ladder.get)],
        "dropped": [(s, name_map[s][0], name_map[s][1], name_map[s][2]) for s in dropped],
        "all_costs": {("%s/%d/%s" % (l, f, e)): c for l, f, e, c in cands},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="append", required=True, choices=list(KERNELS))
    ap.add_argument("--march", action="append", required=True)
    ap.add_argument("--tcrv-opt", required=True)
    ap.add_argument("--mlir-translate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--reps", type=int, default=100, help="best-of-N reps (gentle)")
    ap.add_argument("--iters", type=int, default=2000, help="inner timing iters")
    ap.add_argument("--gate-trials", type=int, default=80, help="byte-exact gate trials")
    ap.add_argument("--board-march", default="rv64gcv_zfh_zvfh")
    ap.add_argument("--host", default="rvv")
    ap.add_argument("--cpu", type=int, default=3, help="taskset -c core")
    ap.add_argument("--clang", default="clang++-18")
    ap.add_argument("--remote-root", default="/root/tcrv-q5-ablation")
    ap.add_argument("--record", default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    record_path = args.record or os.path.join(args.out, "tuning_record_q5.txt")

    results = []
    for kernel in args.kernel:
        for march in args.march:
            results.append(tune_one(args, kernel, march))

    lines = [
        "# tcrv-rvv q5 block-dot MEASUREMENT tuning record (06-18 ablation-q5).",
        "# Each 'tune' line = the MEASURED-fastest legal shape for (kernel,",
        "# capability-march, n), measured best-of-N min on ssh %s (clang -O3" % args.host,
        "# -march=%s -ffp-contract=fast, taskset -c %d, reps=%d, iters=%d)."
        % (args.board_march, args.cpu, args.reps, args.iters),
        "# Comment lines record the full measured ladder + the gate-dropped shapes.",
        "#",
    ]
    for r in results:
        lines.append("# --- %s @ %s n=%d : ggml(real)=%.1f ns ---"
                     % (r["kernel"], r["march"], r["n"], r["ggml_ns"]))
        for name, lmul, factor, elision, ns in r["ladder"]:
            speedup = r["ggml_ns"] / ns if ns else 0.0
            cost = r["all_costs"].get("%s/%d/%s" % (lmul, factor, elision), -1)
            lines.append("#   %-18s %10.1f ns  (%.3fx vs ggml)  cost=%d"
                         % ("%s/%d/%s" % (lmul, factor, elision), ns, speedup, cost))
        for name, lmul, factor, elision in r["dropped"]:
            lines.append("#   %-18s   DRIFT-ineligible (not byte-exact vs ggml at "
                         "-ffp-contract=fast; excluded BEFORE ranking)"
                         % ("%s/%d/%s" % (lmul, factor, elision)))
        lines.append("# MEASURED WINNER: %s/%d/%s @ %.1f ns (%.3fx vs ggml)"
                     % (r["lmul"], r["factor"], r["elision"], r["measured_ns"],
                        r["ggml_ns"] / r["measured_ns"] if r["measured_ns"] else 0.0))
        lines.append("tune kernel=%s march=%s n=%d lmul=%s factor=%d elision=%s measured_ns=%.1f"
                     % (r["kernel"], r["march"], r["n"], r["lmul"], r["factor"],
                        r["elision"], r["measured_ns"]))
        lines.append("#")
    with open(record_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tuning record: %s" % record_path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
