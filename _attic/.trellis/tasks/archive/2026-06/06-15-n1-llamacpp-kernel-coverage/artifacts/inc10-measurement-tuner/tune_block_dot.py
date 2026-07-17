#!/usr/bin/env python3
"""INC-10 measurement-backed block-dot autotuner -- the tune DRIVER.

The genuine N3 "实测胜出": the COMPILER selects each ggml block-dot kernel's shape
by ACTUAL on-board measurement, not a static cost-model guess. This driver runs
OFFLINE, ONCE per (kernel, capability-march). For each kernel it:

  1. ENUMERATES the legal candidate set by asking the materialize pass to
     `--dump-candidates` -- the SINGLE SOURCE OF TRUTH for the capability+budget
     prune (the driver never re-implements the Zvl128b legality rule). The full
     legal set is measured (including the mf4 anchors the cost model deems
     dominated): the board ranks, not the cost model.
  2. EMITS each candidate's C via the compiler
     (tcrv-opt <set the shape> --tcrv-rvv-lower-to-emitc | mlir-translate --mlir-to-cpp),
     renamed to a distinct symbol per candidate.
  3. COMPILES + correctness-gates (byte-exact vs ggml's real kernel) + TIMES
     (best-of-N min, taskset-pinned) on `ssh rvv`.
  4. WRITES a human-readable TUNING RECORD: the MEASURED-fastest shape + its ns +
     the full ladder (as comments) for audit.

The materialize pass then READS this record at compile time and stamps the
measured-best shape; absent a record it falls back to the static cost model.

Usage:
  tune_block_dot.py --kernel q4_1 --march rv64gcv [--march rv64gc_zve32x] \
      --tcrv-opt <path> --mlir-translate <path> --out <record_dir> [--n 4096]
"""

import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# Per-kernel facts: base MLIR template, the ggml-real reference C, the materialize
# pass name, and the data layout for the harness.
KERNELS = {
    "q4_0": {
        "base": "q4_0_base.mlir",
        "pass": "tcrv-rvv-materialize-q4-0-schedule",
        "kernel_sym": "ggml_vec_dot_q4_0_q8_0_kernel",
        "variant_sym": "ggml_vec_dot_q4_0_q8_0",
        "x_stride": 18, "y_stride": 34,
    },
    "q8_0": {
        "base": "q8_0_base.mlir",
        "pass": "tcrv-rvv-materialize-q8-0-schedule",
        "kernel_sym": "ggml_vec_dot_q8_0_q8_0_kernel",
        "variant_sym": "ggml_vec_dot_q8_0_q8_0",
        "x_stride": 34, "y_stride": 34,
    },
    "q4_1": {
        "base": "q4_1_base.mlir",
        "pass": "tcrv-rvv-materialize-q4-1-schedule",
        "kernel_sym": "ggml_vec_dot_q4_1_q8_1_kernel",
        "variant_sym": "ggml_vec_dot_q4_1_q8_1",
        "x_stride": 20, "y_stride": 36,
    },
}

# ggml's REAL hand-written RVV kernels (extracted verbatim from the inc6/inc8/inc9
# microbench rigs -- the SAME direct-FPR fp16 load + recipe ggml ships). Emitted as
# a SEPARATE translation unit (extern "C" kern_ggml) so the compiler contracts it
# under -ffp-contract=fast EXACTLY as the standalone kernel that ships -- NOT as an
# inlined reference (an inlined ref gets contracted differently and would mask the
# real drop-in drift, the bug the single-point n=4096 gate missed). This is BOTH the
# byte-exactness reference AND the perf baseline.
GGML_REF = {
    "q4_0": r"""
extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = (int)n / 32; float sumf = 0; size_t vl = 16;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * 18, *yb = vy + (size_t)ib * 34;
    vuint8m1_t tx = __riscv_vle8_v_u8m1(xb + 2, vl);
    vint8m1_t y0 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2), vl);
    vint8m1_t y1 = __riscv_vle8_v_i8m1((const int8_t *)(yb + 2 + 16), vl);
    vuint8m1_t x_a = __riscv_vand_vx_u8m1(tx, 0x0F, vl);
    vuint8m1_t x_l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);
    vint8m1_t v0 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(x_a), 8, vl);
    vint8m1_t v1 = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(x_l), 8, vl);
    vint16m2_t p = __riscv_vwmul_vv_i16m2(v0, y0, vl);
    p = __riscv_vwmacc_vv_i16m2(p, v1, y1, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m2_i32m1(p, z, vl));
    float dx = (float)*(const _Float16 *)(xb), dy = (float)*(const _Float16 *)(yb);
    sumf += sumi * dx * dy;
  }
  *s = sumf;
}""",
    "q8_0": r"""
extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = (int)n / 32; float sumf = 0; size_t vl = 32;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xb = vx + (size_t)ib * 34, *yb = vy + (size_t)ib * 34;
    const int8_t *xq = (const int8_t *)(xb + 2), *yq = (const int8_t *)(yb + 2);
    float dx = (float)*(const _Float16 *)(xb), dy = (float)*(const _Float16 *)(yb);
    vint8m2_t bx = __riscv_vle8_v_i8m2(xq, vl), by = __riscv_vle8_v_i8m2(yq, vl);
    vint16m4_t m = __riscv_vwmul_vv_i16m4(bx, by, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(__riscv_vwredsum_vs_i16m4_i32m1(m, z, vl));
    sumf += sumi * (dx * dy);
  }
  *s = sumf;
}""",
    "q4_1": r"""
extern "C" void kern_ggml(size_t n, float *s, const uint8_t *vx, const uint8_t *vy) {
  const int nb = (int)n / 32; float sumf = 0; size_t vl = 16;
  for (int ib = 0; ib < nb; ++ib) {
    const uint8_t *xq = vx + ib * 20 + 4; const int8_t *yq = (const int8_t *)(vy + ib * 36 + 4);
    float xd = (float)*(const _Float16 *)(vx + ib * 20), xm = (float)*(const _Float16 *)(vx + ib * 20 + 2);
    float yd = (float)*(const _Float16 *)(vy + ib * 36), ys = (float)*(const _Float16 *)(vy + ib * 36 + 2);
    vuint8m1_t tx = __riscv_vle8_v_u8m1(xq, vl);
    vint8m1_t y0 = __riscv_vle8_v_i8m1(yq, vl), y1 = __riscv_vle8_v_i8m1(yq + 16, vl);
    vuint8m1_t a = __riscv_vand_vx_u8m1(tx, 0x0F, vl), l = __riscv_vsrl_vx_u8m1(tx, 0x04, vl);
    vint8m1_t v0 = __riscv_vreinterpret_v_u8m1_i8m1(a), v1 = __riscv_vreinterpret_v_u8m1_i8m1(l);
    vint16m2_t m1 = __riscv_vwmul_vv_i16m2(v0, y0, vl), m2 = __riscv_vwmacc_vv_i16m2(m1, v1, y1, vl);
    vint32m1_t z = __riscv_vmv_v_x_i32m1(0, vl), r = __riscv_vwredsum_vs_i16m2_i32m1(m2, z, vl);
    int sumi = __riscv_vmv_x_s_i32m1_i32(r);
    sumf += (xd * yd) * sumi + xm * ys;
  }
  *s = sumf;
}""",
}


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def dump_candidates(tcrv_opt, base_path, kinfo, pass_name, march):
    """Ask the pass for the legal candidate set (the single source of truth).

    The dump only needs the op present to enumerate the candidate space, so we
    feed an ATTR-LESS instance of the base op (the SHAPE placeholder removed) --
    a valid block-dot op carrying no shape knobs."""
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
    """Emit ONE candidate's C, renamed to a distinct symbol (per-shape suffix)."""
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


HARNESS_TMPL = r"""// AUTO-GENERATED by tune_block_dot.py -- the on-board microbench for ONE
// (kernel, capability-march) tune run. FILTER-then-RANK: each candidate is
// GATED byte-exact vs ggml's real kernel (a SEPARATE TU, kern_ggml) over MANY n
// and MANY trials under the SAME -ffp-contract=fast the kernel ships with -- a
// candidate that drifts even ~1 ULP is INELIGIBLE (not a valid drop-in) and is
// dropped BEFORE timing. The record therefore can NEVER name a non-exact shape.
// Then the surviving candidates are timed best-of-N min (taskset-pinned).
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
static void fill(uint8_t *vx, uint8_t *vy, int nb, int xs, int ys, int xq, int yq) {
  for (int ib = 0; ib < nb; ++ib) {
    uint8_t *xb = vx + (size_t)ib * xs, *yb = vy + (size_t)ib * ys;
    uint16_t d;
    d = rfp16(); memcpy(xb, &d, 2);
    d = rfp16(); memcpy(yb, &d, 2);
    if (xs == 20) { d = rfp16(); memcpy(xb + 2, &d, 2); }  /* q4_1 weight min */
    if (ys == 36) { d = rfp16(); memcpy(yb + 2, &d, 2); }  /* q8_1 act sum */
    for (int i = 0; i < xq; ++i) xb[(xs - xq) + i] = (uint8_t)(xr() & 0xFF);
    for (int i = 0; i < yq; ++i) yb[(ys - yq) + i] = (uint8_t)(xr() & 0xFF);
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
  const int xs = %(xs)d, ys = %(ys)d, xq = %(xq)d, yq = %(yq)d;
  const int timing_n = %(n)d, iters = 2000, reps = 200;
  struct E ks[] = {
%(entries)s
  };
  const int K = sizeof(ks) / sizeof(ks[0]);

  // GATE: byte-exact vs ggml (separate TU) over many n x many trials at =fast.
  int gate_ns[] = {32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
  const int NN = sizeof(gate_ns) / sizeof(gate_ns[0]);
  for (int t = 0; t < 200; ++t) {
    for (int j = 0; j < NN; ++j) {
      int n = gate_ns[j], nb = n / 32;
      uint8_t *vx = (uint8_t *)malloc((size_t)nb * xs), *vy = (uint8_t *)malloc((size_t)nb * ys);
      fill(vx, vy, nb, xs, ys, xq, yq);
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
  fill(vx, vy, nb, xs, ys, xq, yq);
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

    xq = 16 if kernel in ("q4_0", "q4_1") else 32
    yq = 32
    harness = HARNESS_TMPL % {
        "externs": "\n".join(externs),
        "n": args.n, "xs": kinfo["x_stride"], "ys": kinfo["y_stride"],
        "xq": xq, "yq": yq,
        "entries": "\n".join(entries),
    }
    # ggml's real kernel as its OWN TU (deployment-faithful: contracted exactly as
    # the standalone kernel that ships -- the gate's reference and the baseline).
    ggml_tu = ("#include <stdint.h>\n#include <string.h>\n"
               "#include <riscv_vector.h>\n" + GGML_REF[kernel] + "\n")

    workdir = os.path.join(args.out, "%s_%s_%s" % (kernel, march, args.board_march))
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

    # Compile + run on ssh rvv. The board clang -march is the FULL ISA tier of the
    # silicon (rv64gcv_zfh_zvfh, VLEN=128) -- distinct from the capability-march
    # the record is keyed on. We measure every candidate emitted FOR that
    # capability under the board's real toolchain.
    remote_dir = "/tmp/tcrv_inc10_%s_%s_%s" % (kernel, march, args.board_march)
    board_march = args.board_march
    # Candidates / harness / ggml-ref are SEPARATE translation units (so the
    # per-shape helpers don't collide AND the ggml ref is contracted exactly as
    # the standalone kernel that ships). Ship all three then compile+run.
    run(["ssh", "rvv", "mkdir -p %s" % remote_dir])
    r = run(["scp", harness_path, candidate_cpp_path, ggml_cpp_path,
             "rvv:%s/" % remote_dir])
    if r.returncode != 0:
        sys.exit("scp failed:\n%s" % r.stderr)
    compile_run = (
        "cd %s && clang++-18 -O3 -march=%s -ffp-contract=fast harness.cpp "
        "candidates.cpp ggml_ref.cpp -o bench && taskset -c 3 ./bench"
    ) % (remote_dir, board_march)
    r = run(["ssh", "rvv", compile_run])
    board_out = r.stdout + r.stderr
    with open(os.path.join(workdir, "board_stdout.txt"), "w") as f:
        f.write(board_out)
    if r.returncode != 0 or "gate_done" not in board_out:
        sys.exit("board run failed (%s @ %s):\n%s" % (kernel, march, board_out))

    # Parse the gate verdicts + the timing ladder.
    gate = {}        # suffix -> "exact" / "DRIFT-ineligible"
    ladder = {}      # name -> ns (only eligible survivors + ggml)
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

    # The full ELIGIBLE ladder (sorted fastest-first) + the gate-dropped shapes.
    dropped = [s for s in name_map if gate.get(s) == "DRIFT-ineligible"]
    return {
        "kernel": kernel, "march": march,
        "lmul": lmul, "factor": factor, "elision": elision,
        "measured_ns": best_ns, "ggml_ns": ggml_ns,
        "ladder": [(n, name_map[n][0], name_map[n][1], name_map[n][2],
                    cand_ladder[n]) for n in sorted(cand_ladder, key=cand_ladder.get)],
        "dropped": [(s, name_map[s][0], name_map[s][1], name_map[s][2]) for s in dropped],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", action="append", required=True,
                    choices=list(KERNELS))
    ap.add_argument("--march", action="append", required=True)
    ap.add_argument("--tcrv-opt", required=True)
    ap.add_argument("--mlir-translate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--board-march", default="rv64gcv_zfh_zvfh",
                    help="clang -march for the BOARD compile (the silicon ISA "
                         "tier; distinct from the capability-march the record is "
                         "keyed on). Default rv64gcv_zfh_zvfh -- the real board "
                         "tier so the scalar fp16->fp32 fcvt.s.h is native.")
    ap.add_argument("--record", default=None,
                    help="tuning-record output path (default: <out>/tuning_record.txt)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    record_path = args.record or os.path.join(args.out, "tuning_record.txt")

    results = []
    for kernel in args.kernel:
        for march in args.march:
            results.append(tune_one(args, kernel, march))

    lines = [
        "# tcrv-rvv block-dot MEASUREMENT tuning record (INC-10).",
        "# Each 'tune' line = the MEASURED-fastest legal shape for (kernel,",
        "# capability-march), measured best-of-N min on ssh rvv (VLEN=128,",
        "# clang-18 -O3 -march=%s -ffp-contract=fast, taskset -c 3)." % args.board_march,
        "# Comment lines record the full measured ladder (audit). The materialize",
        "# pass READS this and stamps the measured shape; absent => static fallback.",
        "#",
    ]
    for r in results:
        lines.append("# --- %s @ %s : ggml(real)=%.1f ns ---"
                     % (r["kernel"], r["march"], r["ggml_ns"]))
        for name, lmul, factor, elision, ns in r["ladder"]:
            speedup = r["ggml_ns"] / ns if ns else 0.0
            lines.append("#   %-18s %10.1f ns  (%.3fx vs ggml)"
                         % ("%s/%d/%s" % (lmul, factor, elision), ns, speedup))
        for name, lmul, factor, elision in r["dropped"]:
            lines.append("#   %-18s   DRIFT-ineligible (not byte-exact vs ggml at "
                         "-ffp-contract=fast; excluded BEFORE ranking)"
                         % ("%s/%d/%s" % (lmul, factor, elision)))
        lines.append("# MEASURED WINNER: %s/%d/%s @ %.1f ns (%.3fx vs ggml)"
                     % (r["lmul"], r["factor"], r["elision"], r["measured_ns"],
                        r["ggml_ns"] / r["measured_ns"] if r["measured_ns"] else 0.0))
        lines.append("tune kernel=%s march=%s lmul=%s factor=%d elision=%s measured_ns=%.1f"
                     % (r["kernel"], r["march"], r["lmul"], r["factor"],
                        r["elision"], r["measured_ns"]))
        lines.append("#")
    with open(record_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote tuning record: %s" % record_path)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
