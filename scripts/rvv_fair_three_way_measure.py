#!/usr/bin/env python3
"""Fair 3-way RVV performance diagnostic (N3 性能灯 / P-A).

Builds a *fair* same-target comparison for the no-autovectorizable Gearbox
low-precision contraction kernels and runs it on real ``ssh rvv`` hardware:

  column                | march            | what it is
  ----------------------|------------------|----------------------------------
  genuine-scalar        | rv64gc (no 'v')  | honest scalar; NO vector ISA at all
  autovec-scalar (ref)  | rv64gcv          | SAME scalar C source, lets clang
                        |                  | autovectorize -> explains the old
                        |                  | "regression" number
  naive-RVV             | rv64gcv          | hand-written untuned riscv_vector.h
                        |                  | strip-loop (the bar the tune must beat)
  tuned-RVV             | rv64gcv          | the compiler's actual emitted bundle
                        |                  | (generate_verified_bundle -> .o + .h)

Fairness controls
  * all four variants are linked into ONE binary and timed *interleaved* in the
    same repeat loop on the same input buffers (the board is shared / jump-hosted
    => high variance; cross-run comparison would be apples-to-oranges).
  * the genuine-scalar TU is compiled ``-march=rv64gc`` so the vector ISA is not
    even available; we ``objdump`` that object ON THE BOARD and assert ZERO
    vector ops (no ``v...`` / ``vset``) -> the fairness fix is *verified*, not asserted.
  * correctness guard (scalar as oracle, tol 1e-05) runs for every variant
    before any timing -- a fast-but-wrong kernel is not a win (I8).

This is measurement + baselines (P-A). It does NOT change the tune. The tuned
bundle is whatever ``generate_verified_bundle`` emits today.

All timing is on real ``ssh rvv`` riscv64 (``--ssh-target rvv``). With
``--dry-run`` it generates + compiles-locally-skipped and STOPS before timing
(for harness self-check only); a real diagnostic requires the board.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import rvv_generated_bundle_abi_e2e as abi
import rvv_generated_bundle_same_target_measure as gate4

SCRIPT_NAME = "rvv_fair_three_way_measure"
SCHEMA_VERSION = "rvv-fair-three-way-measure.v1"

# The two no-autovectorizable Gearbox kernels (spec: where the tune legitimately
# matters). Each maps to (op_kind, input MLIR that selects the variant).
GROUPED_INPUT = Path(
    "artifacts/gate4-candidate-feedback-ssh/gate4-candidate-feedback-ssh/"
    "grouped-u2/widening_product_reduce_dequantize_f32/"
    "pre_realized_selected_body_input.mlir"
)
PACKED_I4_INPUT = Path(
    "artifacts/gate4-candidate-feedback-ssh/gate4-candidate-feedback-ssh/"
    "packed-i4/widening_product_reduce_dequantize_f32/"
    "pre_realized_selected_body_input.mlir"
)

DEFAULT_COUNTS = (257, 4096, 65536)
DEFAULT_SCALE = 0.0125
DEFAULT_WARMUPS = 3
DEFAULT_REPEATS = 9
DEFAULT_ITERS = 16

# per-TU compile flags (board default mabi=lp64d)
SCALAR_FLAGS = ("-O2", "-march=rv64gc", "-mabi=lp64d")
VECTOR_FLAGS = ("-O2", "-march=rv64gcv", "-mabi=lp64d")
# autovec-scalar = the SAME scalar source but on the vector march (one flag diff);
# this is the column that explains the prior 0.76x "regression".

# Vector-instruction signatures for the objdump no-vector check.
_VSET_RE = re.compile(r"\bvset[ivl]")
_VOP_RE = re.compile(r"\b(v[a-z][a-z0-9._]*)\b")
# scalar/pseudo mnemonics that begin with 'v' but are NOT vector ops.
_VOP_ALLOW = {"vsetvl", "vsetvli", "vsetivli"}  # these ARE vector; kept for clarity


@dataclass(frozen=True)
class Kernel:
    label: str
    op_kind: str
    input_path: Path
    packed_i4: bool


KERNELS = {
    "grouped-u2": Kernel(
        "grouped-u2", "widening_product_reduce_dequantize_f32", GROUPED_INPUT, False
    ),
    "packed-i4": Kernel(
        "packed-i4", "widening_product_reduce_dequantize_f32", PACKED_I4_INPUT, True
    ),
}


# --------------------------------------------------------------------------- #
# C reference sources (genuine-scalar + naive-RVV) for each kernel family.
# The signature MUST match the tuned bundle's ABI exactly:
#   (const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
#    float scale, float *out, size_t n)
# n unit: grouped -> bytes==elements; packed-i4 -> bytes (each byte = 2 nibbles).
# --------------------------------------------------------------------------- #

# Genuine-scalar reference for the byte (grouped) kernel. Compiled rv64gc.
SCALAR_BYTE_C = r"""
#include <stdint.h>
#include <stddef.h>
/* genuine scalar: int8*int8 -> i32 reduce, then * f32 scale. rv64gc, no vector ISA. */
__attribute__((noinline)) void
ref_scalar_byte(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
                float scale, float *out, size_t n) {
  int32_t sum = acc[0];
  for (size_t i = 0; i < n; ++i)
    sum += (int32_t)lhs[i] * (int32_t)rhs[i];
  out[0] = ((float)sum) * scale;
}
"""

# Genuine-scalar reference for the packed-i4 kernel. Compiled rv64gc.
SCALAR_PACKED_I4_C = r"""
#include <stdint.h>
#include <stddef.h>
static int32_t sx_i4(uint8_t nib) {
  int32_t v = (int32_t)(nib & 0x0fu);
  return v >= 8 ? v - 16 : v;
}
/* genuine scalar packed-i4: two signed i4 per byte (low,high nibble). */
__attribute__((noinline)) void
ref_scalar_packed_i4(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
                     float scale, float *out, size_t n) {
  int32_t sum = acc[0];
  for (size_t i = 0; i < n; ++i) {
    uint8_t lb = (uint8_t)lhs[i], rb = (uint8_t)rhs[i];
    sum += sx_i4(lb & 0x0fu) * sx_i4(rb & 0x0fu);
    sum += sx_i4((lb >> 4) & 0x0fu) * sx_i4((rb >> 4) & 0x0fu);
  }
  out[0] = ((float)sum) * scale;
}
"""

# Naive-RVV (untuned but COMPETENT) reference for the byte kernel. Compiled rv64gcv.
# The textbook RVV dot-product pattern a reasonable (non-tuning) programmer writes:
# accumulate element-wise into a persistent vint32m1 vector accumulator via
# vwmul (i8mf4->i16mf2) + vwadd.wv (i16mf2 into i32m1), and do the horizontal
# reduction (vredsum) ONCE after the loop -- NOT a cross-lane vwredsum every
# iteration (the latency-bound anti-pattern). NO unroll, NO grouping, NO region-tune.
NAIVE_BYTE_C = r"""
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
__attribute__((noinline)) void
naive_rvv_byte(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
               float scale, float *out, size_t n) {
  size_t vl;
  size_t vlmax = __riscv_vsetvlmax_e32m1();
  vint32m1_t vacc = __riscv_vmv_v_x_i32m1(0, vlmax);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e8mf4(n - i);
    vint8mf4_t vl8 = __riscv_vle8_v_i8mf4(lhs + i, vl);
    vint8mf4_t vr8 = __riscv_vle8_v_i8mf4(rhs + i, vl);
    vint16mf2_t vprod = __riscv_vwmul_vv_i16mf2(vl8, vr8, vl);
    /* widen-accumulate i16mf2 lanes into the i32m1 vector accumulator */
    vacc = __riscv_vwadd_wv_i32m1(vacc, vprod, vl);
  }
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, vlmax);
  vint32m1_t vred = __riscv_vredsum_vs_i32m1_i32m1(vacc, vzero, vlmax);
  int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  out[0] = ((float)sum) * scale;
}
"""

# Naive-RVV (untuned but COMPETENT) reference for the packed-i4 kernel. rv64gcv.
# Basic nibble unpack: load bytes as i8mf4, sign-extend low and high nibbles
# (shift-left 4 then arithmetic-shift-right 4 keeps each nibble's sign), two
# vwmul, and -- like a reasonable RVV programmer -- accumulate BOTH products
# element-wise into a persistent vint32m1 accumulator (vwadd.wv) and reduce
# (vredsum) ONCE after the loop, not a per-iteration cross-lane vwredsum.
# NO unroll, NO fused vwmacc, NO region-tune (that is the tuned version's job).
NAIVE_PACKED_I4_C = r"""
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
__attribute__((noinline)) void
naive_rvv_packed_i4(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
                    float scale, float *out, size_t n) {
  size_t vl;
  size_t vlmax = __riscv_vsetvlmax_e32m1();
  vint32m1_t vacc = __riscv_vmv_v_x_i32m1(0, vlmax);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e8mf4(n - i);
    vint8mf4_t lb = __riscv_vle8_v_i8mf4(lhs + i, vl);
    vint8mf4_t rb = __riscv_vle8_v_i8mf4(rhs + i, vl);
    /* sign-extend the low nibble: (b << 4) >>(arith) 4 */
    vint8mf4_t l_lo = __riscv_vsra_vx_i8mf4(__riscv_vsll_vx_i8mf4(lb, 4, vl), 4, vl);
    vint8mf4_t r_lo = __riscv_vsra_vx_i8mf4(__riscv_vsll_vx_i8mf4(rb, 4, vl), 4, vl);
    /* sign-extend the high nibble: b >>(arith) 4 */
    vint8mf4_t l_hi = __riscv_vsra_vx_i8mf4(lb, 4, vl);
    vint8mf4_t r_hi = __riscv_vsra_vx_i8mf4(rb, 4, vl);
    vint16mf2_t p_lo = __riscv_vwmul_vv_i16mf2(l_lo, r_lo, vl);
    vint16mf2_t p_hi = __riscv_vwmul_vv_i16mf2(l_hi, r_hi, vl);
    vacc = __riscv_vwadd_wv_i32m1(vacc, p_lo, vl);
    vacc = __riscv_vwadd_wv_i32m1(vacc, p_hi, vl);
  }
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, vlmax);
  vint32m1_t vred = __riscv_vredsum_vs_i32m1_i32m1(vacc, vzero, vlmax);
  int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  out[0] = ((float)sum) * scale;
}
"""


def kernel_ref_sources(packed_i4: bool) -> tuple[str, str, str, str]:
    """Return (scalar_fn, scalar_src, naive_fn, naive_src) for the kernel."""
    if packed_i4:
        return (
            "ref_scalar_packed_i4",
            SCALAR_PACKED_I4_C,
            "naive_rvv_packed_i4",
            NAIVE_PACKED_I4_C,
        )
    return ("ref_scalar_byte", SCALAR_BYTE_C, "naive_rvv_byte", NAIVE_BYTE_C)


# --------------------------------------------------------------------------- #
# Input initialization (mirrors gate4's signed-widening coverage discipline so
# the correctness guard sees positive, negative and >127 products).
# --------------------------------------------------------------------------- #
BYTE_INIT = r"""
  for (size_t i = 0; i < alloc_n; ++i) {
    lhs[i] = (int8_t)(((i % 6) < 3) ? -((int)(i % 53) + 21) : ((int)(i % 47) + 18));
    rhs[i] = (int8_t)(((i % 5) == 1 || (i % 5) == 4) ? -((int)(i % 43) + 17)
                                                     : ((int)(i % 41) + 23));
  }
"""

# For packed-i4 each byte carries two signed nibbles in [-8,7].
PACKED_I4_INIT = r"""
  for (size_t i = 0; i < alloc_n; ++i) {
    int lo_l = (int)(i % 15) - 7;            /* [-7,7] */
    int hi_l = (int)((i * 3 + 2) % 15) - 7;
    int lo_r = (int)((i * 5 + 1) % 15) - 7;
    int hi_r = (int)((i * 7 + 3) % 15) - 7;
    lhs[i] = (int8_t)((lo_l & 0x0f) | ((hi_l & 0x0f) << 4));
    rhs[i] = (int8_t)((lo_r & 0x0f) | ((hi_r & 0x0f) << 4));
  }
"""


def harness_source(
    *,
    header_file: str,
    tuned_fn: str,
    scalar_fn: str,
    naive_fn: str,
    autovec_fn: str,
    counts: list[int],
    scale: float,
    warmups: int,
    repeats: int,
    iters: int,
    init_body: str,
) -> str:
    counts_c = ", ".join(str(c) for c in counts)
    counts_summary = ",".join(str(c) for c in counts)
    return _HARNESS_TEMPLATE.format(
        header_file=header_file,
        tuned_fn=tuned_fn,
        scalar_fn=scalar_fn,
        naive_fn=naive_fn,
        autovec_fn=autovec_fn,
        counts_c=counts_c,
        counts_summary=counts_summary,
        scale=f"{scale:.9g}f",
        warmups=warmups,
        repeats=repeats,
        iters=iters,
        init_body=init_body,
    )


# The 4-way interleaved timing harness. Variants are timed back-to-back inside
# the SAME repeat loop on the SAME buffers; best-of-N per variant.
_HARNESS_TEMPLATE = r"""
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "{header_file}"

#define WARMUPS {warmups}
#define REPEATS {repeats}
#define ITERS {iters}

typedef void (*kernel_fn)(const int8_t *, const int8_t *, const int32_t *,
                          float, float *, size_t);

/* externally-defined reference kernels (separate TUs / objects) */
void {scalar_fn}(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
void {naive_fn}(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);
void {autovec_fn}(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);

static volatile double tcrv_sink = 0.0;

static unsigned long long now_ns(void) {{
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {{
    perror("clock_gettime");
    exit(97);
  }}
  return (unsigned long long)ts.tv_sec * 1000000000ULL +
         (unsigned long long)ts.tv_nsec;
}}

#define N_VARIANTS 4
static const char *VARIANT_NAMES[N_VARIANTS] = {{
  "genuine-scalar", "autovec-scalar", "naive-rvv", "tuned-rvv"
}};

static int run_case(size_t n, float scale) {{
  const size_t alloc_n = n == 0 ? 1 : n;
  int8_t *lhs = (int8_t *)malloc(alloc_n);
  int8_t *rhs = (int8_t *)malloc(alloc_n);
  int32_t *acc = (int32_t *)malloc(sizeof(int32_t) * 4);
  float *out = (float *)malloc(sizeof(float) * 16);
  if (!lhs || !rhs || !acc || !out) {{ fprintf(stderr, "alloc failed n=%zu\n", n); return 11; }}
{init_body}
  acc[0] = (int32_t)(-137 + (int32_t)n);
  acc[1] = acc[2] = acc[3] = 0;

  kernel_fn fns[N_VARIANTS] = {{
    {scalar_fn}, {autovec_fn}, {naive_fn},
    {tuned_fn}
  }};

  /* correctness guard: scalar oracle, all variants, tol 1e-05, before timing */
  const float tol = 1e-05f;
  out[0] = 0.0f;
  {scalar_fn}(lhs, rhs, acc, scale, out, n);
  float oracle = out[0];
  for (int v = 0; v < N_VARIANTS; ++v) {{
    out[0] = 0.0f;
    fns[v](lhs, rhs, acc, scale, out, n);
    float d = out[0] - oracle; if (d < 0) d = -d;
    if (d > tol) {{
      fprintf(stderr,
              "MISMATCH variant=%s n=%zu got=%.9g oracle=%.9g delta=%.9g tol=%.9g\n",
              VARIANT_NAMES[v], n, out[0], oracle, d, tol);
      free(lhs); free(rhs); free(acc); free(out);
      return 12;
    }}
  }}
  printf("CORRECTNESS n=%zu scale=%.9g ok oracle=%.9g\n", n, scale, oracle);

  /* warmup all variants */
  for (int w = 0; w < WARMUPS; ++w)
    for (int v = 0; v < N_VARIANTS; ++v) {{
      fns[v](lhs, rhs, acc, scale, out, n);
      tcrv_sink += (double)out[0];
    }}

  double best[N_VARIANTS];
  for (int v = 0; v < N_VARIANTS; ++v) best[v] = -1.0;

  for (int r = 0; r < REPEATS; ++r) {{
    for (int v = 0; v < N_VARIANTS; ++v) {{
      unsigned long long start = now_ns();
      for (int it = 0; it < ITERS; ++it) {{
        fns[v](lhs, rhs, acc, scale, out, n);
        tcrv_sink += (double)out[0];
      }}
      double per_iter = (double)(now_ns() - start) / (double)ITERS;
      if (best[v] < 0.0 || per_iter < best[v]) best[v] = per_iter;
    }}
  }}

  for (int v = 0; v < N_VARIANTS; ++v)
    printf("SUMMARY variant=%s n=%zu scale=%.9g best_per_iter_ns=%.3f\n",
           VARIANT_NAMES[v], n, scale, best[v]);
  /* ratios vs both honest baselines (genuine-scalar=0, naive-rvv=2) */
  double scal = best[0], naive = best[2], tuned = best[3];
  printf("RATIO n=%zu scale=%.9g tuned_vs_scalar=%.6f tuned_vs_naive=%.6f "
         "naive_vs_scalar=%.6f autovec_vs_scalar=%.6f\n",
         n, scale,
         tuned > 0 ? scal / tuned : 0.0,
         tuned > 0 ? naive / tuned : 0.0,
         naive > 0 ? scal / naive : 0.0,
         best[1] > 0 ? scal / best[1] : 0.0);

  free(lhs); free(rhs); free(acc); free(out);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{ {counts_c} }};
  const float scale = {scale};
  const size_t nc = sizeof(counts) / sizeof(counts[0]);
  printf("CONFIG variants=genuine-scalar,autovec-scalar,naive-rvv,tuned-rvv "
         "counts={counts_summary} warmups=%d repeats=%d iters=%d "
         "timing=clock_gettime(CLOCK_MONOTONIC_RAW)\n", WARMUPS, REPEATS, ITERS);
  for (size_t i = 0; i < nc; ++i) {{
    int s = run_case(counts[i], scale);
    if (s != 0) return s;
  }}
  printf("PASS three-way measurement counts={counts_summary} sink=%.9g\n", tcrv_sink);
  return 0;
}}
""".lstrip()


def parse_run_stdout(stdout: str) -> dict[str, Any]:
    """Parse SUMMARY + RATIO lines into a per-(variant,n) timing table."""
    summaries: list[dict[str, Any]] = []
    ratios: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if line.startswith("SUMMARY variant="):
            m = re.search(
                r"variant=(\S+) n=(\d+) scale=(\S+) best_per_iter_ns=([\d.]+)", line
            )
            if m:
                summaries.append(
                    {
                        "variant": m.group(1),
                        "n": int(m.group(2)),
                        "scale": float(m.group(3)),
                        "best_per_iter_ns": float(m.group(4)),
                    }
                )
        elif line.startswith("RATIO n="):
            m = re.search(
                r"n=(\d+) scale=(\S+) tuned_vs_scalar=([\d.]+) "
                r"tuned_vs_naive=([\d.]+) naive_vs_scalar=([\d.]+) "
                r"autovec_vs_scalar=([\d.]+)",
                line,
            )
            if m:
                ratios.append(
                    {
                        "n": int(m.group(1)),
                        "scale": float(m.group(2)),
                        "tuned_vs_scalar": float(m.group(3)),
                        "tuned_vs_naive": float(m.group(4)),
                        "naive_vs_scalar": float(m.group(5)),
                        "autovec_vs_scalar": float(m.group(6)),
                    }
                )
    return {"summaries": summaries, "ratios": ratios}


def vector_ops_in_disasm(disasm: str) -> list[str]:
    """Return the set of vector mnemonics found in a disassembly text."""
    found: set[str] = set()
    for line in disasm.splitlines():
        # objdump line: "  <addr>:\t<bytes>\t<mnemonic> <operands>"
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        insn = parts[2].strip()
        mnem = insn.split()[0] if insn else ""
        if not mnem:
            continue
        if _VSET_RE.match(mnem) or (
            mnem.startswith("v") and mnem not in ("vmv.x.s",) and len(mnem) > 1
        ):
            # heuristic: RVV mnemonics are v<letters>... ; plain 'v' alone is not.
            if re.match(r"^v[a-z]", mnem):
                found.add(mnem)
    return sorted(found)


def classify(ratios: list[dict[str, Any]]) -> dict[str, Any]:
    if not ratios:
        return {"verdict": "not-measured"}
    tvs = [r["tuned_vs_scalar"] for r in ratios]
    tvn = [r["tuned_vs_naive"] for r in ratios]
    beats_scalar = all(x > 1.0 for x in tvs)
    beats_naive = all(x > 1.0 for x in tvn)
    if beats_scalar and beats_naive:
        verdict = "win"  # the 灯 is ON for this kernel
    elif beats_scalar or beats_naive:
        verdict = "partial"
    else:
        verdict = "no-win"
    return {
        "verdict": verdict,
        "tuned_vs_scalar_range": f"{min(tvs):.4f}..{max(tvs):.4f}",
        "tuned_vs_naive_range": f"{min(tvn):.4f}..{max(tvn):.4f}",
        "tuned_vs_scalar_median": round(statistics.median(tvs), 4),
        "tuned_vs_naive_median": round(statistics.median(tvn), 4),
        "beats_genuine_scalar": beats_scalar,
        "beats_naive_rvv": beats_naive,
    }


def remote_objdump_scalar(
    *, ssh_target: str, connect_timeout: int, timeout: int, remote_obj: str
) -> dict[str, Any]:
    """objdump the genuine-scalar OBJECT on the board; confirm zero vector ops."""
    # pick whichever disassembler exists on the board
    cmd = (
        "if command -v llvm-objdump >/dev/null 2>&1; then "
        f"  llvm-objdump -d {abi.remote_quote(remote_obj)}; "
        "elif command -v objdump >/dev/null 2>&1; then "
        f"  objdump -d {abi.remote_quote(remote_obj)}; "
        "else echo NO_OBJDUMP; fi"
    )
    rec = abi.run_remote_shell(
        ssh_target, connect_timeout, cmd, timeout, stdout_limit=262144
    )
    disasm = str(rec.get("stdout", ""))
    vops = vector_ops_in_disasm(disasm)
    return {
        "command_record": rec,
        "disasm": disasm,
        "vector_ops_found": vops,
        "is_genuinely_scalar": (len(vops) == 0 and "NO_OBJDUMP" not in disasm),
        "objdump_available": "NO_OBJDUMP" not in disasm,
    }


def probe_rdcycle(
    *, ssh_target: str, connect_timeout: int, timeout: int
) -> dict[str, Any]:
    """Best-effort probe for U-mode rdcycle/rdtime (commonly traps SIGILL)."""
    probe_c = r"""
#include <stdio.h>
#include <stdint.h>
static uint64_t rdcycle(void){uint64_t v;__asm__ volatile("rdcycle %0":"=r"(v));return v;}
static uint64_t rdtime(void){uint64_t v;__asm__ volatile("rdtime %0":"=r"(v));return v;}
int main(void){
  uint64_t c0=rdcycle(); for(volatile int i=0;i<100000;++i); uint64_t c1=rdcycle();
  uint64_t t0=rdtime(); uint64_t t1=rdtime();
  printf("RDCYCLE_OK delta_cycles=%llu\n",(unsigned long long)(c1-c0));
  printf("RDTIME_OK delta_time=%llu\n",(unsigned long long)(t1-t0));
  return 0;
}
"""
    rd = "/tmp/tcrv_rdcycle_probe"
    cmd = (
        f"rm -rf {rd} && mkdir -p {rd} && cd {rd} && "
        f"cat > probe.c <<'EOF'\n{probe_c}\nEOF\n"
        "clang -O2 -march=rv64gcv -mabi=lp64d probe.c -o probe 2>probe_build.err && "
        "(./probe 2>probe_run.err; echo RDPROBE_EXIT=$?) ; "
        f"rm -rf {rd}"
    )
    rec = abi.run_remote_shell(ssh_target, connect_timeout, cmd, timeout)
    out = str(rec.get("stdout", ""))
    return {
        "command_record": rec,
        "rdcycle_available": "RDCYCLE_OK" in out,
        "rdtime_available": "RDTIME_OK" in out,
        "stdout": out,
    }


def build_and_run_remote(
    *,
    kernel: Kernel,
    artifact_dir: Path,
    tuned_object: Path,
    tuned_header: Path,
    tuned_fn: str,
    harness_path: Path,
    scalar_src_path: Path,
    naive_src_path: Path,
    scalar_fn: str,
    naive_fn: str,
    autovec_fn: str,
    ssh_target: str,
    connect_timeout: int,
    timeout: int,
) -> dict[str, Any]:
    remote_dir = f"/tmp/tcrv_fair3way_{kernel.label}_{abi.safe_run_id('p-a')}"
    commands: dict[str, Any] = {"remote_dir": remote_dir}

    setup = abi.run_remote_shell(
        ssh_target,
        connect_timeout,
        f"rm -rf {abi.remote_quote(remote_dir)} && mkdir -p {abi.remote_quote(remote_dir)}",
        timeout,
    )
    commands["setup"] = setup
    abi.require_command_success(setup, "fair3way remote setup")

    # stage all sources + tuned bundle
    scp = abi.scp_base_command(connect_timeout) + [
        str(tuned_object),
        str(tuned_header),
        str(harness_path),
        str(scalar_src_path),
        str(naive_src_path),
        f"{ssh_target}:{remote_dir}/",
    ]
    scp_rec = abi.run_command(scp, timeout=timeout)
    commands["scp"] = scp_rec
    abi.require_command_success(scp_rec, "fair3way artifact staging")

    r_obj = f"{remote_dir}/{tuned_object.name}"
    r_hdr = tuned_header.name  # included relative (harness has -I.)
    r_harness = f"{remote_dir}/{harness_path.name}"
    r_scalar = f"{remote_dir}/{scalar_src_path.name}"
    r_naive = f"{remote_dir}/{naive_src_path.name}"
    r_scalar_o = f"{remote_dir}/scalar.o"
    r_autovec_o = f"{remote_dir}/autovec.o"
    r_naive_o = f"{remote_dir}/naive.o"
    r_harness_o = f"{remote_dir}/harness.o"
    r_bin = f"{remote_dir}/fair3way_{kernel.label}"

    # target profile
    profile_cmd = (
        f"cd {abi.remote_quote(remote_dir)} && "
        "printf 'remote_arch=' && uname -m && "
        "printf 'remote_uname=' && uname -a && "
        "printf 'clang_version=' && clang --version | head -n1 && "
        "printf 'lscpu_begin\\n' && (lscpu | sed -n '1,20p') && printf 'lscpu_end\\n'"
    )
    profile_rec = abi.run_remote_shell(
        ssh_target, connect_timeout, profile_cmd, timeout, stdout_limit=32768
    )
    commands["target_profile"] = profile_rec
    abi.require_command_success(profile_rec, "fair3way target profile")

    scalar_flags = shlex.join(SCALAR_FLAGS)
    vector_flags = shlex.join(VECTOR_FLAGS)
    # autovec object: rename the scalar function symbol so both genuine + autovec
    # coexist in one binary. We compile the SAME scalar source with -march=rv64gcv
    # and -D to remap its function name.
    compile_cmd = (
        f"cd {abi.remote_quote(remote_dir)} && set -e && "
        # genuine scalar: NO vector ISA
        f"clang {scalar_flags} -c {r_scalar} -o {r_scalar_o} && "
        # autovec scalar: SAME source, vector march, function renamed via macro
        f"clang {vector_flags} -D{scalar_fn}={autovec_fn} -c {r_scalar} -o {r_autovec_o} && "
        # naive rvv
        f"clang {vector_flags} -c {r_naive} -o {r_naive_o} && "
        # harness (needs header on include path)
        f"clang {vector_flags} -I {abi.remote_quote(remote_dir)} -c {r_harness} -o {r_harness_o} && "
        # link all + the tuned object
        f"clang {vector_flags} {r_harness_o} {r_scalar_o} {r_autovec_o} {r_naive_o} "
        f"{r_obj} -o {r_bin} && "
        "echo COMPILE_OK"
    )
    compile_rec = abi.run_remote_shell(
        ssh_target, connect_timeout, compile_cmd, timeout, stdout_limit=32768
    )
    commands["compile"] = compile_rec
    abi.require_command_success(compile_rec, "fair3way compile/link")

    # objdump the genuine-scalar object: assert zero vector ops (fairness proof)
    objdump = remote_objdump_scalar(
        ssh_target=ssh_target,
        connect_timeout=connect_timeout,
        timeout=timeout,
        remote_obj=r_scalar_o,
    )
    commands["scalar_objdump_available"] = objdump["objdump_available"]

    # run the timed binary
    run_rec = abi.run_remote_shell(
        ssh_target, connect_timeout, r_bin, timeout,
        stdout_limit=262144, stderr_limit=32768,
    )
    commands["run"] = run_rec
    abi.require_command_success(run_rec, "fair3way timed run")
    stdout = str(run_rec.get("stdout", ""))
    abi.require_contains(stdout, "PASS three-way measurement", "fair3way run output")

    cleanup = abi.run_remote_shell(
        ssh_target, connect_timeout, f"rm -rf {abi.remote_quote(remote_dir)}", timeout
    )
    commands["cleanup"] = cleanup

    # persist raw evidence
    (artifact_dir / "remote_target_profile.txt").write_text(
        str(profile_rec.get("stdout", "")), encoding="utf-8"
    )
    (artifact_dir / "remote_run_stdout.txt").write_text(stdout, encoding="utf-8")
    (artifact_dir / "scalar_objdump.txt").write_text(
        objdump["disasm"], encoding="utf-8"
    )

    parsed = parse_run_stdout(stdout)
    return {
        "ssh_target": ssh_target,
        "remote_dir": remote_dir,
        "target_profile": str(profile_rec.get("stdout", "")),
        "scalar_objdump": {
            "vector_ops_found": objdump["vector_ops_found"],
            "is_genuinely_scalar": objdump["is_genuinely_scalar"],
            "objdump_available": objdump["objdump_available"],
        },
        "parsed": parsed,
        "classification": classify(parsed["ratios"]),
        "commands": commands,
    }


def generate_tuned_bundle(
    *, kernel: Kernel, artifact_dir: Path, config: gate4.MeasurementConfig,
    tcrv_opt: str, tcrv_translate: str, readobj: str | None, timeout: int,
) -> dict[str, Any]:
    """Drive the existing verified generation -> tuned .o + .h (sha-identity contract)."""
    input_path = abi.resolve_repo_relative_path(kernel.input_path)
    if not input_path.exists():
        raise abi.EvidenceError(f"kernel input not found: {kernel.input_path}")
    expectation = gate4.selected_pre_realized_expectation(
        kernel.op_kind, input_path=input_path
    )
    args = argparse.Namespace(dry_run=True, timeout=timeout)
    gen = gate4.generate_verified_bundle(
        args=args,
        run_id=abi.safe_run_id(f"fair3way-{kernel.label}"),
        artifact_dir=artifact_dir,
        expectation=expectation,
        config=config,
        tcrv_opt=tcrv_opt,
        tcrv_translate=tcrv_translate,
        readobj=readobj,
    )
    bundle_checks = gen["bundle_checks"]
    bundle_dir = artifact_dir / kernel.op_kind / "generated_bundle"
    object_path = bundle_dir / bundle_checks["object_file"]
    header_path = bundle_dir / bundle_checks["header_file"]
    return {
        "expectation_function": expectation.function_name,
        "object_path": object_path,
        "header_path": header_path,
        "object_sha256": abi.sha256_file(object_path),
        "header_sha256": abi.sha256_file(header_path),
        "selected_input": str(input_path),
    }


def measure_kernel(
    *, kernel: Kernel, args: argparse.Namespace, artifact_root: Path
) -> dict[str, Any]:
    kdir = artifact_root / kernel.label
    kdir.mkdir(parents=True, exist_ok=True)
    config = gate4.MeasurementConfig(
        counts=list(args.counts),
        dequant_scale_values=[args.scale, args.scale * 2.0],
        warmup_count=args.warmups,
        repeat_count=args.repeats,
        measure_iterations=args.iters,
        compile_flags=list(VECTOR_FLAGS) + ["-I."],
    )
    gen = generate_tuned_bundle(
        kernel=kernel, artifact_dir=kdir / "gen", config=config,
        tcrv_opt=args.tcrv_opt, tcrv_translate=args.tcrv_translate,
        readobj=args.llvm_readobj, timeout=args.timeout,
    )
    tuned_fn = gen["expectation_function"]

    scalar_fn, scalar_src, naive_fn, naive_src = kernel_ref_sources(kernel.packed_i4)
    autovec_fn = scalar_fn + "_autovec"
    init_body = PACKED_I4_INIT if kernel.packed_i4 else BYTE_INIT

    scalar_src_path = kdir / "ref_scalar.c"
    naive_src_path = kdir / "ref_naive_rvv.c"
    harness_path = kdir / "harness.c"
    scalar_src_path.write_text(scalar_src, encoding="utf-8")
    naive_src_path.write_text(naive_src, encoding="utf-8")
    harness_path.write_text(
        harness_source(
            header_file=gen["header_path"].name,
            tuned_fn=tuned_fn,
            scalar_fn=scalar_fn,
            naive_fn=naive_fn,
            autovec_fn=autovec_fn,
            counts=list(args.counts),
            scale=args.scale,
            warmups=args.warmups,
            repeats=args.repeats,
            iters=args.iters,
            init_body=init_body,
        ),
        encoding="utf-8",
    )

    result: dict[str, Any] = {
        "kernel": kernel.label,
        "op_kind": kernel.op_kind,
        "packed_i4": kernel.packed_i4,
        "tuned_function": tuned_fn,
        "tuned_object_sha256": gen["object_sha256"],
        "tuned_header_sha256": gen["header_sha256"],
        "selected_input": gen["selected_input"],
        "scalar_function": scalar_fn,
        "naive_function": naive_fn,
        "harness_path": str(harness_path),
    }

    if args.dry_run:
        result["status"] = "dry-run-generated-only"
        result["ssh_evidence"] = False
        return result

    remote = build_and_run_remote(
        kernel=kernel,
        artifact_dir=kdir,
        tuned_object=gen["object_path"],
        tuned_header=gen["header_path"],
        tuned_fn=tuned_fn,
        harness_path=harness_path,
        scalar_src_path=scalar_src_path,
        naive_src_path=naive_src_path,
        scalar_fn=scalar_fn,
        naive_fn=naive_fn,
        autovec_fn=autovec_fn,
        ssh_target=args.ssh_target,
        connect_timeout=args.connect_timeout,
        timeout=args.timeout,
    )
    result["status"] = "measured"
    result["ssh_evidence"] = True
    result["ssh_target"] = args.ssh_target
    result["scalar_objdump"] = remote["scalar_objdump"]
    result["target_profile"] = remote["target_profile"]
    result["summaries"] = remote["parsed"]["summaries"]
    result["ratios"] = remote["parsed"]["ratios"]
    result["classification"] = remote["classification"]
    return result


def render_markdown(results: list[dict[str, Any]], rdcycle: dict[str, Any] | None) -> str:
    lines = ["# Fair 3-way RVV performance diagnostic (P-A)", ""]
    lines.append(
        "Timing: `clock_gettime(CLOCK_MONOTONIC_RAW)`, best-of-N, interleaved "
        "variants in one binary on real `ssh rvv` riscv64.\n"
    )
    if rdcycle is not None:
        lines.append(
            f"- rdcycle available on board: **{rdcycle.get('rdcycle_available')}**; "
            f"rdtime available: **{rdcycle.get('rdtime_available')}** "
            "(wall-time is the reported timer).\n"
        )
    for r in results:
        lines.append(f"## Kernel: `{r['kernel']}` ({r['op_kind']})")
        lines.append("")
        lines.append(f"- tuned function: `{r['tuned_function']}`")
        lines.append(f"- tuned object sha256: `{r.get('tuned_object_sha256','')}`")
        sj = r.get("scalar_objdump")
        if sj:
            lines.append(
                f"- genuine-scalar object: `-march=rv64gc`; vector ops in disasm: "
                f"**{sj['vector_ops_found'] or 'NONE'}**; genuinely scalar: "
                f"**{sj['is_genuinely_scalar']}** (objdump available: {sj['objdump_available']})"
            )
        summaries = r.get("summaries") or []
        ratios = r.get("ratios") or []
        if summaries:
            # per-n table of best_per_iter_ns by variant
            ns = sorted({s["n"] for s in summaries})
            lines.append("")
            lines.append(
                "| n | genuine-scalar ns | autovec-scalar ns | naive-RVV ns | tuned-RVV ns |"
            )
            lines.append("|---|---|---|---|---|")
            for n in ns:
                row = {s["variant"]: s["best_per_iter_ns"] for s in summaries if s["n"] == n}
                lines.append(
                    f"| {n} | {row.get('genuine-scalar','-'):.1f} "
                    f"| {row.get('autovec-scalar','-'):.1f} "
                    f"| {row.get('naive-rvv','-'):.1f} "
                    f"| {row.get('tuned-rvv','-'):.1f} |"
                )
            lines.append("")
            lines.append("| n | tuned/scalar | tuned/naive | naive/scalar | autovec/scalar |")
            lines.append("|---|---|---|---|---|")
            for rr in sorted(ratios, key=lambda x: x["n"]):
                lines.append(
                    f"| {rr['n']} | {rr['tuned_vs_scalar']:.3f} "
                    f"| {rr['tuned_vs_naive']:.3f} "
                    f"| {rr['naive_vs_scalar']:.3f} "
                    f"| {rr['autovec_vs_scalar']:.3f} |"
                )
            c = r.get("classification", {})
            lines.append("")
            lines.append(
                f"**Verdict**: `{c.get('verdict')}` — tuned vs genuine-scalar "
                f"{c.get('tuned_vs_scalar_range')} (median {c.get('tuned_vs_scalar_median')}), "
                f"tuned vs naive-RVV {c.get('tuned_vs_naive_range')} "
                f"(median {c.get('tuned_vs_naive_median')}). "
                f"beats genuine-scalar: {c.get('beats_genuine_scalar')}; "
                f"beats naive-RVV: {c.get('beats_naive_rvv')}."
            )
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--kernel", action="append", choices=sorted(KERNELS), default=None,
        help="kernel(s) to measure (default: all)",
    )
    p.add_argument("--artifact-root", default="artifacts/p-a-fair-three-way")
    p.add_argument("--ssh-target", default="rvv")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--counts", type=int, nargs="+", default=list(DEFAULT_COUNTS))
    p.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    p.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p.add_argument("--probe-rdcycle", action="store_true")
    p.add_argument("--tcrv-opt", default="build/bin/tcrv-opt")
    p.add_argument("--tcrv-translate", default="build/bin/tcrv-translate")
    p.add_argument("--llvm-readobj", default=None)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--connect-timeout", type=int, default=20)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    labels = args.kernel or sorted(KERNELS)
    artifact_root = abi.resolve_repo_relative_path(Path(args.artifact_root))
    artifact_root.mkdir(parents=True, exist_ok=True)

    rdcycle = None
    if args.probe_rdcycle and not args.dry_run:
        rdcycle = probe_rdcycle(
            ssh_target=args.ssh_target,
            connect_timeout=args.connect_timeout,
            timeout=args.timeout,
        )
        (artifact_root / "rdcycle_probe.json").write_text(
            json.dumps(rdcycle, indent=2), encoding="utf-8"
        )

    results: list[dict[str, Any]] = []
    for label in labels:
        kernel = KERNELS[label]
        print(f"[fair3way] measuring kernel={label} dry_run={args.dry_run}")
        results.append(
            measure_kernel(kernel=kernel, args=args, artifact_root=artifact_root)
        )

    evidence = {
        "schema_version": SCHEMA_VERSION,
        "tool": SCRIPT_NAME,
        "created_at": abi.utc_timestamp(),
        "ssh_target": args.ssh_target,
        "dry_run": bool(args.dry_run),
        "timing_method": "clock_gettime(CLOCK_MONOTONIC_RAW)",
        "counts": list(args.counts),
        "scale": args.scale,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "iters": args.iters,
        "rdcycle_probe": rdcycle,
        "results": results,
    }
    (artifact_root / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    md = render_markdown(results, rdcycle)
    (artifact_root / "three_way_table.md").write_text(md, encoding="utf-8")
    print(f"[fair3way] wrote {artifact_root / 'evidence.json'}")
    print(f"[fair3way] wrote {artifact_root / 'three_way_table.md'}")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
