#!/usr/bin/env python3
"""N3 性能灯 — P-B step 1: hand-written RVV accumulator/LMUL sweep on real ssh rvv.

Goal (the experiment that decides whether P-B step 2 is GO):
  empirically find an RVV configuration for the no-autovectorizable low-precision
  int8 dot-product contraction (the Gearbox widening-product-reduce kernel) that
  MEASURABLY beats BOTH a genuine (non-autovectorized) scalar AND a *competent*
  naive RVV loop on real ``ssh rvv`` hardware -- and attribute the win to the right
  knob (LMUL width vs multiple independent accumulators) so the eventual Gearbox
  enumeration is resource-AWARE rather than a magic constant.

The discriminating design (per the P-A diagnostic + advisor):
  * The P-A naive uses e8mf4 (4 bytes/iter on this 128-bit-VLEN board) -- that is
    UNDER-VECTORIZED, a confound. So the bar is NOT just "beat the mf4 naive"; the
    grid includes **A=1 at every LMUL** so we can separate "win came from LMUL" from
    "win came from multiple accumulators." The real question: does A>1 beat the best
    A=1 config *at the same LMUL*?
  * The exact P-A mf4 single-accumulator naive is kept as an ANCHOR column so this
    tool's naive/scalar ratio reproduces P-A (validates the measurement seam).
  * The widening chain couples LMUL to accumulator register width:
        i8 mf2 -> i16 m1  -> i32 m2     (1 acc = 2 vregs)
        i8 m1  -> i16 m2  -> i32 m4     (1 acc = 4 vregs)
        i8 m2  -> i16 m4  -> i32 m8     (1 acc = 8 vregs)
    A accumulators at i32-LMUL-X cost A*(X vregs) + temps; combos that overflow the
    32-vreg budget are PRUNED and reported (that prune is the resource fact).

Measurement:
  * timing via ``rdcycle`` (zicntr present, U-mode readable on this board) bracketing
    the ITERS loop -- the margins hunted here are tight (few % .. ~20%), so cycle
    counts beat shared-board wall-clock jitter. wall-time (clock_gettime) is also
    recorded as a cross-check.
  * all variants linked into ONE binary, timed interleaved on the SAME buffers,
    warmup + best-of-(repeats x iters).
  * correctness guard (scalar oracle, tol 1e-05) runs for EVERY variant before any
    timing -- a fast-wrong variant is disqualified (I8). Tail-prone strip-mining is
    exercised with a prime size (257).
  * genuine-scalar TU compiled ``-march=rv64gc`` (no 'v'); objdump on the board
    asserts ZERO vector ops -> the scalar bar is verified, not asserted.

Hand variants only. Does NOT touch the Gearbox / conversion / gate4 (P-B step 2).
All timing on real ``ssh rvv`` riscv64; ``--dry-run`` stops before the board.
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

SCRIPT_NAME = "rvv_accumulator_sweep_measure"
SCHEMA_VERSION = "rvv-accumulator-sweep-measure.v1"

DEFAULT_COUNTS = (257, 256, 1024, 4096, 16384, 65536)
DEFAULT_SCALE = 0.0125
DEFAULT_WARMUPS = 3
DEFAULT_REPEATS = 11
DEFAULT_ITERS = 16

SCALAR_FLAGS = ("-O2", "-march=rv64gc", "-mabi=lp64d")
VECTOR_FLAGS = ("-O2", "-march=rv64gcv", "-mabi=lp64d")

# Vector-instruction detector for the genuine-scalar objdump fairness proof.
_VOP_RE = re.compile(r"^v[a-z]")


# --------------------------------------------------------------------------- #
# LMUL ladder for the widening i8 -> i16 -> i32 chain.
#   load_lmul : the i8 load fractional/whole LMUL (the per-iter byte count)
#   prod_lmul : the i16 widened product LMUL
#   acc_lmul  : the i32 accumulator LMUL
#   acc_regs  : vregs occupied by ONE i32 accumulator (the budget unit)
#   bytes_per_strip : i8 elements processed per single strip at this LMUL on VLEN=128
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LmulRung:
    name: str
    load_lmul: str  # e.g. "mf4"
    prod_lmul: str  # e.g. "mf2"
    acc_lmul: str  # e.g. "m1"
    acc_regs: int
    bytes_per_strip_vlen128: int


LMUL_LADDER = {
    "mf4": LmulRung("mf4", "mf4", "mf2", "m1", 1, 4),
    "mf2": LmulRung("mf2", "mf2", "m1", "m2", 2, 8),
    "m1": LmulRung("m1", "m1", "m2", "m4", 4, 16),
    "m2": LmulRung("m2", "m2", "m4", "m8", 8, 32),
}

VREG_BUDGET = 32  # RVV architectural vector register file size.


# --------------------------------------------------------------------------- #
# C emitters. Both kernels share the ABI:
#   void fn(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
#           float scale, float *out, size_t n)
# byte kernel:     n bytes == n elements (int8 * int8).
# packed-i4 kernel: n bytes, each byte = two signed i4 nibbles (low, high).
# --------------------------------------------------------------------------- #

SCALAR_BYTE_C = r"""
#include <stdint.h>
#include <stddef.h>
/* genuine scalar int8 dot. rv64gc -> NO vector ISA (objdump-verified on board). */
__attribute__((noinline)) void
ref_scalar_byte(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
                float scale, float *out, size_t n) {
  int32_t sum = acc[0];
  for (size_t i = 0; i < n; ++i)
    sum += (int32_t)lhs[i] * (int32_t)rhs[i];
  out[0] = ((float)sum) * scale;
}
"""

SCALAR_PACKED_I4_C = r"""
#include <stdint.h>
#include <stddef.h>
static int32_t sx_i4(uint8_t nib) {
  int32_t v = (int32_t)(nib & 0x0fu);
  return v >= 8 ? v - 16 : v;
}
/* genuine scalar packed-i4: two signed i4 per byte (low, high nibble). */
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

# The exact P-A competent-naive ANCHOR (mf4, single i32m1 accumulator, one reduce).
NAIVE_BYTE_ANCHOR_C = r"""
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
__attribute__((noinline)) void
naive_anchor_byte(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
                  float scale, float *out, size_t n) {
  size_t vl;
  size_t vlmax = __riscv_vsetvlmax_e32m1();
  vint32m1_t vacc = __riscv_vmv_v_x_i32m1(0, vlmax);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e8mf4(n - i);
    vint8mf4_t vl8 = __riscv_vle8_v_i8mf4(lhs + i, vl);
    vint8mf4_t vr8 = __riscv_vle8_v_i8mf4(rhs + i, vl);
    vint16mf2_t vprod = __riscv_vwmul_vv_i16mf2(vl8, vr8, vl);
    vacc = __riscv_vwadd_wv_i32m1(vacc, vprod, vl);
  }
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, vlmax);
  vint32m1_t vred = __riscv_vredsum_vs_i32m1_i32m1(vacc, vzero, vlmax);
  int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  out[0] = ((float)sum) * scale;
}
"""

NAIVE_PACKED_I4_ANCHOR_C = r"""
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
__attribute__((noinline)) void
naive_anchor_packed_i4(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
                       float scale, float *out, size_t n) {
  size_t vl;
  size_t vlmax = __riscv_vsetvlmax_e32m1();
  vint32m1_t vacc = __riscv_vmv_v_x_i32m1(0, vlmax);
  for (size_t i = 0; i < n; i += vl) {
    vl = __riscv_vsetvl_e8mf4(n - i);
    vint8mf4_t lb = __riscv_vle8_v_i8mf4(lhs + i, vl);
    vint8mf4_t rb = __riscv_vle8_v_i8mf4(rhs + i, vl);
    vint8mf4_t l_lo = __riscv_vsra_vx_i8mf4(__riscv_vsll_vx_i8mf4(lb, 4, vl), 4, vl);
    vint8mf4_t r_lo = __riscv_vsra_vx_i8mf4(__riscv_vsll_vx_i8mf4(rb, 4, vl), 4, vl);
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


def _emit_byte_variant(fn: str, rung: LmulRung, accs: int) -> str:
    """Emit an int8-dot variant: `accs` independent i32 accumulators, A-way unrolled
    strip-mine at the given LMUL, tree-summed + single vredsum at the end."""
    L, P, A, R = rung.load_lmul, rung.prod_lmul, rung.acc_lmul, rung.acc_regs
    # main loop processes `accs` strips per iteration, each into its own accumulator.
    acc_decls = "\n  ".join(
        f"vint32{A}_t vacc{k} = __riscv_vmv_v_x_i32{A}(0, vl0);" for k in range(accs)
    )
    # body for the k-th strip in an unrolled iteration; base offset i + k*vl
    strip = (
        "    {{ size_t vl = __riscv_vsetvl_e8{L}(rem); "
        "vint8{L}_t a = __riscv_vle8_v_i8{L}(lhs + off, vl); "
        "vint8{L}_t b = __riscv_vle8_v_i8{L}(rhs + off, vl); "
        "vint16{P}_t p = __riscv_vwmul_vv_i16{P}(a, b, vl); "
        "vacc{k} = __riscv_vwadd_wv_i32{A}(vacc{k}, p, vl); "
        "off += vl; rem = (n > off) ? (n - off) : 0; }}"
    )
    strips = "\n".join(
        strip.format(L=L, P=P, A=A, k=k) for k in range(accs)
    )
    # tree-reduce the accumulators into vacc0 (all same i32-LMUL).
    tree = "\n  ".join(
        f"vacc0 = __riscv_vadd_vv_i32{A}(vacc0, vacc{k}, vl0);" for k in range(1, accs)
    )
    return f"""
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
/* {fn}: int8 dot, LMUL={rung.name} (i32 acc {A}, {R} vregs/acc), A={accs} accumulators. */
__attribute__((noinline)) void
{fn}(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
     float scale, float *out, size_t n) {{
  size_t vl0 = __riscv_vsetvlmax_e32{A}();
  {acc_decls}
  size_t off = 0;
  size_t rem = n;
  while (rem > 0) {{
{strips}
  }}
  {tree}
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
  /* reduce the (possibly wide) i32 accumulator to a scalar */
  vint32m1_t vred = __riscv_vredsum_vs_i32{A}_i32m1(vacc0, vzero, vl0);
  int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  out[0] = ((float)sum) * scale;
}}
"""


def _emit_packed_i4_variant(fn: str, rung: LmulRung, accs: int) -> str:
    """Emit a packed-i4 dot variant: unpack low/high signed nibbles, two products,
    `accs` independent i32 accumulators, A-way unrolled strip-mine, single reduce."""
    L, P, A, R = rung.load_lmul, rung.prod_lmul, rung.acc_lmul, rung.acc_regs
    acc_decls = "\n  ".join(
        f"vint32{A}_t vacc{k} = __riscv_vmv_v_x_i32{A}(0, vl0);" for k in range(accs)
    )
    strip = (
        "    {{ size_t vl = __riscv_vsetvl_e8{L}(rem); "
        "vint8{L}_t lb = __riscv_vle8_v_i8{L}(lhs + off, vl); "
        "vint8{L}_t rb = __riscv_vle8_v_i8{L}(rhs + off, vl); "
        "vint8{L}_t l_lo = __riscv_vsra_vx_i8{L}(__riscv_vsll_vx_i8{L}(lb, 4, vl), 4, vl); "
        "vint8{L}_t r_lo = __riscv_vsra_vx_i8{L}(__riscv_vsll_vx_i8{L}(rb, 4, vl), 4, vl); "
        "vint8{L}_t l_hi = __riscv_vsra_vx_i8{L}(lb, 4, vl); "
        "vint8{L}_t r_hi = __riscv_vsra_vx_i8{L}(rb, 4, vl); "
        "vint16{P}_t p_lo = __riscv_vwmul_vv_i16{P}(l_lo, r_lo, vl); "
        "vint16{P}_t p_hi = __riscv_vwmul_vv_i16{P}(l_hi, r_hi, vl); "
        "vacc{k} = __riscv_vwadd_wv_i32{A}(vacc{k}, p_lo, vl); "
        "vacc{k} = __riscv_vwadd_wv_i32{A}(vacc{k}, p_hi, vl); "
        "off += vl; rem = (n > off) ? (n - off) : 0; }}"
    )
    strips = "\n".join(strip.format(L=L, P=P, A=A, k=k) for k in range(accs))
    tree = "\n  ".join(
        f"vacc0 = __riscv_vadd_vv_i32{A}(vacc0, vacc{k}, vl0);" for k in range(1, accs)
    )
    return f"""
#include <stdint.h>
#include <stddef.h>
#include <riscv_vector.h>
/* {fn}: packed-i4 dot, LMUL={rung.name} (i32 acc {A}, {R} vregs/acc), A={accs} accumulators. */
__attribute__((noinline)) void
{fn}(const int8_t *lhs, const int8_t *rhs, const int32_t *acc,
     float scale, float *out, size_t n) {{
  size_t vl0 = __riscv_vsetvlmax_e32{A}();
  {acc_decls}
  size_t off = 0;
  size_t rem = n;
  while (rem > 0) {{
{strips}
  }}
  {tree}
  vint32m1_t vzero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());
  vint32m1_t vred = __riscv_vredsum_vs_i32{A}_i32m1(vacc0, vzero, vl0);
  int32_t sum = acc[0] + __riscv_vmv_x_s_i32m1_i32(vred);
  out[0] = ((float)sum) * scale;
}}
"""


# --------------------------------------------------------------------------- #
# Variant grid: A x LMUL, with budget pruning.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Variant:
    label: str  # human label, e.g. "L=m1_A=4"
    fn: str  # C function name
    rung: str  # LMUL rung key
    accs: int
    acc_regs_total: int
    fits_budget: bool


def build_variant_grid(
    lmuls: list[str], accs_list: list[int], reserve: int
) -> tuple[list[Variant], list[Variant]]:
    """Return (kept, pruned). reserve = vregs reserved for product/load temps."""
    kept: list[Variant] = []
    pruned: list[Variant] = []
    for lk in lmuls:
        rung = LMUL_LADDER[lk]
        for a in accs_list:
            total = a * rung.acc_regs
            # temps: product is at prod_lmul (= acc_regs/2 rounded up, min 1), plus
            # loads; we approximate the live temp pressure as `reserve` extra vregs.
            fits = (total + reserve) <= VREG_BUDGET
            label = f"L={lk}_A={a}"
            fn = f"v_{lk}_a{a}".replace("mf", "f")
            v = Variant(label, fn, lk, a, total, fits)
            (kept if fits else pruned).append(v)
    return kept, pruned


# --------------------------------------------------------------------------- #
# Input initialization (mirror the gate4 signed-widening coverage discipline).
# --------------------------------------------------------------------------- #
BYTE_INIT = r"""
  for (size_t i = 0; i < alloc_n; ++i) {
    lhs[i] = (int8_t)(((i % 6) < 3) ? -((int)(i % 53) + 21) : ((int)(i % 47) + 18));
    rhs[i] = (int8_t)(((i % 5) == 1 || (i % 5) == 4) ? -((int)(i % 43) + 17)
                                                     : ((int)(i % 41) + 23));
  }
"""

PACKED_I4_INIT = r"""
  for (size_t i = 0; i < alloc_n; ++i) {
    int lo_l = (int)(i % 15) - 7;
    int hi_l = (int)((i * 3 + 2) % 15) - 7;
    int lo_r = (int)((i * 5 + 1) % 15) - 7;
    int hi_r = (int)((i * 7 + 3) % 15) - 7;
    lhs[i] = (int8_t)((lo_l & 0x0f) | ((hi_l & 0x0f) << 4));
    rhs[i] = (int8_t)((lo_r & 0x0f) | ((hi_r & 0x0f) << 4));
  }
"""


def harness_source(
    *,
    scalar_fn: str,
    naive_anchor_fn: str,
    variants: list[Variant],
    counts: list[int],
    scale: float,
    warmups: int,
    repeats: int,
    iters: int,
    init_body: str,
) -> str:
    # the timed function pointer table: index 0 = scalar oracle, 1 = naive anchor,
    # then each kept variant.
    fns = [scalar_fn, naive_anchor_fn] + [v.fn for v in variants]
    names = ["genuine-scalar", "naive-mf4-anchor"] + [v.label for v in variants]
    nvar = len(fns)
    externs = "\n".join(
        f"void {f}(const int8_t *, const int8_t *, const int32_t *, float, float *, size_t);"
        for f in fns
    )
    fn_table = ",\n    ".join(fns)
    names_init = ", ".join(f'"{nm}"' for nm in names)
    counts_c = ", ".join(str(c) for c in counts)
    counts_summary = ",".join(str(c) for c in counts)
    return _HARNESS_TEMPLATE.format(
        externs=externs,
        fn_table=fn_table,
        names_init=names_init,
        nvar=nvar,
        counts_c=counts_c,
        counts_summary=counts_summary,
        scale=f"{scale:.9g}f",
        warmups=warmups,
        repeats=repeats,
        iters=iters,
        init_body=init_body,
    )


_HARNESS_TEMPLATE = r"""
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define WARMUPS {warmups}
#define REPEATS {repeats}
#define ITERS {iters}
#define N_VARIANTS {nvar}

typedef void (*kernel_fn)(const int8_t *, const int8_t *, const int32_t *,
                          float, float *, size_t);

{externs}

static const char *VARIANT_NAMES[N_VARIANTS] = {{ {names_init} }};

static volatile double tcrv_sink = 0.0;

static inline uint64_t rdcycle_now(void) {{
  uint64_t v;
  __asm__ volatile("rdcycle %0" : "=r"(v));
  return v;
}}

static unsigned long long now_ns(void) {{
  struct timespec ts;
  if (clock_gettime(CLOCK_MONOTONIC_RAW, &ts) != 0) {{ perror("clock_gettime"); exit(97); }}
  return (unsigned long long)ts.tv_sec * 1000000000ULL + (unsigned long long)ts.tv_nsec;
}}

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
    {fn_table}
  }};

  /* correctness guard: scalar oracle, every variant, tol 1e-05, before timing. */
  const float tol = 1e-05f;
  out[0] = 0.0f;
  fns[0](lhs, rhs, acc, scale, out, n);
  float oracle = out[0];
  for (int v = 0; v < N_VARIANTS; ++v) {{
    out[0] = 0.0f;
    fns[v](lhs, rhs, acc, scale, out, n);
    float d = out[0] - oracle; if (d < 0) d = -d;
    if (d > tol) {{
      fprintf(stderr, "MISMATCH variant=%s n=%zu got=%.9g oracle=%.9g delta=%.9g\n",
              VARIANT_NAMES[v], n, out[0], oracle, d);
      free(lhs); free(rhs); free(acc); free(out);
      return 12;
    }}
  }}
  printf("CORRECTNESS n=%zu ok oracle=%.9g\n", n, oracle);

  for (int w = 0; w < WARMUPS; ++w)
    for (int v = 0; v < N_VARIANTS; ++v) {{ fns[v](lhs, rhs, acc, scale, out, n); tcrv_sink += (double)out[0]; }}

  double best_ns[N_VARIANTS];
  unsigned long long best_cyc[N_VARIANTS];
  for (int v = 0; v < N_VARIANTS; ++v) {{ best_ns[v] = -1.0; best_cyc[v] = 0; }}

  for (int r = 0; r < REPEATS; ++r) {{
    for (int v = 0; v < N_VARIANTS; ++v) {{
      uint64_t c0 = rdcycle_now();
      unsigned long long t0 = now_ns();
      for (int it = 0; it < ITERS; ++it) {{ fns[v](lhs, rhs, acc, scale, out, n); tcrv_sink += (double)out[0]; }}
      unsigned long long dt = now_ns() - t0;
      uint64_t dc = rdcycle_now() - c0;
      double per_ns = (double)dt / (double)ITERS;
      unsigned long long per_cyc = dc / (unsigned long long)ITERS;
      if (best_ns[v] < 0.0 || per_ns < best_ns[v]) best_ns[v] = per_ns;
      if (best_cyc[v] == 0 || per_cyc < best_cyc[v]) best_cyc[v] = per_cyc;
    }}
  }}

  for (int v = 0; v < N_VARIANTS; ++v)
    printf("SUMMARY variant=%s n=%zu best_ns=%.3f best_cyc=%llu\n",
           VARIANT_NAMES[v], n, best_ns[v], best_cyc[v]);

  free(lhs); free(rhs); free(acc); free(out);
  return 0;
}}

int main(void) {{
  const size_t counts[] = {{ {counts_c} }};
  const float scale = {scale};
  const size_t nc = sizeof(counts) / sizeof(counts[0]);
  printf("CONFIG nvar=%d counts={counts_summary} warmups=%d repeats=%d iters=%d "
         "timing=rdcycle+clock_gettime\n", N_VARIANTS, WARMUPS, REPEATS, ITERS);
  for (size_t i = 0; i < nc; ++i) {{ int s = run_case(counts[i], scale); if (s != 0) return s; }}
  printf("PASS accumulator-sweep counts={counts_summary} sink=%.9g\n", tcrv_sink);
  return 0;
}}
""".lstrip()


def parse_run_stdout(stdout: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if line.startswith("SUMMARY variant="):
            m = re.search(
                r"variant=(\S+) n=(\d+) best_ns=([\d.]+) best_cyc=(\d+)", line
            )
            if m:
                out.append(
                    {
                        "variant": m.group(1),
                        "n": int(m.group(2)),
                        "best_ns": float(m.group(3)),
                        "best_cyc": int(m.group(4)),
                    }
                )
    return out


def vector_ops_in_disasm(disasm: str) -> list[str]:
    found: set[str] = set()
    for line in disasm.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        insn = parts[2].strip()
        mnem = insn.split()[0] if insn else ""
        if mnem and _VOP_RE.match(mnem) and mnem not in ("vmv.x.s",):
            found.add(mnem)
    return sorted(found)


def analyze(summaries: list[dict[str, Any]], variants: list[Variant]) -> dict[str, Any]:
    """Compute per-n ratios (cycle-based) vs scalar + naive-anchor + best-A=1, and
    pick the winning config (beats scalar AND naive-anchor across all n)."""
    ns = sorted({s["n"] for s in summaries})
    by_nv: dict[tuple[int, str], dict[str, Any]] = {
        (s["n"], s["variant"]): s for s in summaries
    }
    # A=1 variants per LMUL (for the LMUL-vs-accumulator attribution)
    a1_by_lmul = {v.rung: v.label for v in variants if v.accs == 1}
    var_labels = [v.label for v in variants]

    rows: list[dict[str, Any]] = []
    for n in ns:
        scal = by_nv.get((n, "genuine-scalar"), {}).get("best_cyc")
        naive = by_nv.get((n, "naive-mf4-anchor"), {}).get("best_cyc")
        # best A=1 cycle across all LMULs at this n (the honest "beat naive" bar)
        a1_cycs = [
            by_nv[(n, lbl)]["best_cyc"]
            for lbl in a1_by_lmul.values()
            if (n, lbl) in by_nv and by_nv[(n, lbl)]["best_cyc"] > 0
        ]
        best_a1 = min(a1_cycs) if a1_cycs else None
        row: dict[str, Any] = {"n": n, "scalar_cyc": scal, "naive_cyc": naive,
                               "best_a1_cyc": best_a1, "variants": {}}
        for lbl in var_labels:
            cyc = by_nv.get((n, lbl), {}).get("best_cyc")
            ns_v = by_nv.get((n, lbl), {}).get("best_ns")
            if cyc and cyc > 0:
                row["variants"][lbl] = {
                    "cyc": cyc,
                    "ns": ns_v,
                    "vs_scalar": (scal / cyc) if scal else None,
                    "vs_naive": (naive / cyc) if naive else None,
                    "vs_best_a1": (best_a1 / cyc) if best_a1 else None,
                }
        rows.append(row)

    # winner: a variant whose vs_scalar>1 AND vs_naive>1 for ALL n; prefer the one
    # that also beats best_a1 (the real latency-hiding win), smallest A then LMUL.
    def variant_wins_all(lbl: str) -> dict[str, Any]:
        vs_s, vs_n, vs_a1 = [], [], []
        for row in rows:
            v = row["variants"].get(lbl)
            if not v or v["vs_scalar"] is None or v["vs_naive"] is None:
                return {"beats_scalar": False, "beats_naive": False}
            vs_s.append(v["vs_scalar"]); vs_n.append(v["vs_naive"])
            if v["vs_best_a1"] is not None:
                vs_a1.append(v["vs_best_a1"])
        return {
            "beats_scalar": all(x > 1.0 for x in vs_s),
            "beats_naive": all(x > 1.0 for x in vs_n),
            "beats_best_a1": bool(vs_a1) and all(x > 1.0 for x in vs_a1),
            "min_vs_scalar": round(min(vs_s), 4),
            "min_vs_naive": round(min(vs_n), 4),
            "median_vs_naive": round(statistics.median(vs_n), 4),
            "min_vs_best_a1": round(min(vs_a1), 4) if vs_a1 else None,
            "median_vs_best_a1": round(statistics.median(vs_a1), 4) if vs_a1 else None,
        }

    per_variant = {lbl: variant_wins_all(lbl) for lbl in var_labels}
    winners = [lbl for lbl, w in per_variant.items() if w["beats_scalar"] and w["beats_naive"]]
    # rank winners: the headline winner is the FASTEST clean config (largest
    # median speedup vs the competent naive anchor) -- this is the performance 灯,
    # not the smallest config. Ties break to fewer accumulators then smaller LMUL
    # (the resource-fact preference: fewest vregs for the same speed). The
    # `beats_best_a1` field is reported separately so the attribution (LMUL width vs
    # accumulator count) stays visible regardless of which config tops the table.
    lmul_order = {"mf4": 0, "mf2": 1, "m1": 2, "m2": 3}
    vmeta = {v.label: v for v in variants}

    def rank_key(lbl: str) -> tuple:
        w = per_variant[lbl]
        v = vmeta[lbl]
        # primary: fastest median vs naive (negate -> ascending sort picks largest).
        speed = -(w.get("median_vs_naive") or 0.0)
        return (speed, v.accs, lmul_order.get(v.rung, 9))

    winners_sorted = sorted(winners, key=rank_key)
    return {
        "sizes": ns,
        "rows": rows,
        "per_variant_verdict": per_variant,
        "a1_by_lmul": a1_by_lmul,
        "winners_beating_scalar_and_naive": winners_sorted,
        "winner": winners_sorted[0] if winners_sorted else None,
    }


def build_and_run_remote(
    *,
    kernel_label: str,
    artifact_dir: Path,
    scalar_src: str,
    scalar_fn: str,
    naive_src: str,
    naive_fn: str,
    variant_srcs: dict[str, str],
    harness_src: str,
    ssh_target: str,
    connect_timeout: int,
    timeout: int,
) -> dict[str, Any]:
    remote_dir = f"/tmp/tcrv_accsweep_{kernel_label}_{abi.safe_run_id('p-b1')}"
    commands: dict[str, Any] = {"remote_dir": remote_dir}

    # write all sources locally first
    scalar_path = artifact_dir / "ref_scalar.c"
    naive_path = artifact_dir / "ref_naive_anchor.c"
    harness_path = artifact_dir / "harness.c"
    scalar_path.write_text(scalar_src, encoding="utf-8")
    naive_path.write_text(naive_src, encoding="utf-8")
    harness_path.write_text(harness_src, encoding="utf-8")
    variant_paths: dict[str, Path] = {}
    for fn, src in variant_srcs.items():
        p = artifact_dir / f"var_{fn}.c"
        p.write_text(src, encoding="utf-8")
        variant_paths[fn] = p

    setup = abi.run_remote_shell(
        ssh_target, connect_timeout,
        f"rm -rf {abi.remote_quote(remote_dir)} && mkdir -p {abi.remote_quote(remote_dir)}",
        timeout,
    )
    commands["setup"] = setup
    abi.require_command_success(setup, "accsweep remote setup")

    all_local = [scalar_path, naive_path, harness_path] + list(variant_paths.values())
    scp = abi.scp_base_command(connect_timeout) + [str(p) for p in all_local] + [
        f"{ssh_target}:{remote_dir}/"
    ]
    scp_rec = abi.run_command(scp, timeout=timeout)
    commands["scp"] = scp_rec
    abi.require_command_success(scp_rec, "accsweep artifact staging")

    profile_cmd = (
        f"cd {abi.remote_quote(remote_dir)} && "
        "printf 'remote_arch=' && uname -m && printf 'remote_uname=' && uname -a && "
        "printf 'clang_version=' && clang --version | head -n1 && "
        "printf 'isa=' && (grep -m1 '^isa' /proc/cpuinfo || true)"
    )
    profile_rec = abi.run_remote_shell(
        ssh_target, connect_timeout, profile_cmd, timeout, stdout_limit=32768
    )
    commands["target_profile"] = profile_rec
    abi.require_command_success(profile_rec, "accsweep target profile")

    scalar_flags = shlex.join(SCALAR_FLAGS)
    vector_flags = shlex.join(VECTOR_FLAGS)
    # compile each variant TU + scalar (rv64gc) + naive (rv64gcv) + harness, link all.
    var_objs = " ".join(f"var_{fn}.o" for fn in variant_paths)
    var_compiles = " && ".join(
        f"clang {vector_flags} -c var_{fn}.c -o var_{fn}.o" for fn in variant_paths
    )
    compile_cmd = (
        f"cd {abi.remote_quote(remote_dir)} && set -e && "
        f"clang {scalar_flags} -c {scalar_path.name} -o scalar.o && "
        f"clang {vector_flags} -c {naive_path.name} -o naive.o && "
        f"{var_compiles} && "
        f"clang {vector_flags} -c {harness_path.name} -o harness.o && "
        f"clang {vector_flags} harness.o scalar.o naive.o {var_objs} -o accsweep_bin && "
        "echo COMPILE_OK"
    )
    compile_rec = abi.run_remote_shell(
        ssh_target, connect_timeout, compile_cmd, timeout, stdout_limit=65536
    )
    commands["compile"] = compile_rec
    abi.require_command_success(compile_rec, "accsweep compile/link")

    # objdump the genuine-scalar object: assert zero vector ops.
    objdump_cmd = (
        f"cd {abi.remote_quote(remote_dir)} && "
        "if command -v llvm-objdump >/dev/null 2>&1; then llvm-objdump -d scalar.o; "
        "elif command -v objdump >/dev/null 2>&1; then objdump -d scalar.o; "
        "else echo NO_OBJDUMP; fi"
    )
    objdump_rec = abi.run_remote_shell(
        ssh_target, connect_timeout, objdump_cmd, timeout, stdout_limit=262144
    )
    commands["scalar_objdump"] = {"exit_code": objdump_rec.get("exit_code")}
    disasm = str(objdump_rec.get("stdout", ""))
    vops = vector_ops_in_disasm(disasm)

    run_rec = abi.run_remote_shell(
        ssh_target, connect_timeout, f"cd {abi.remote_quote(remote_dir)} && ./accsweep_bin",
        timeout, stdout_limit=262144, stderr_limit=32768,
    )
    commands["run"] = run_rec
    abi.require_command_success(run_rec, "accsweep timed run")
    stdout = str(run_rec.get("stdout", ""))
    abi.require_contains(stdout, "PASS accumulator-sweep", "accsweep run output")

    cleanup = abi.run_remote_shell(
        ssh_target, connect_timeout, f"rm -rf {abi.remote_quote(remote_dir)}", timeout
    )
    commands["cleanup"] = cleanup

    (artifact_dir / "remote_target_profile.txt").write_text(
        str(profile_rec.get("stdout", "")), encoding="utf-8"
    )
    (artifact_dir / "remote_run_stdout.txt").write_text(stdout, encoding="utf-8")
    (artifact_dir / "scalar_objdump.txt").write_text(disasm, encoding="utf-8")

    return {
        "ssh_target": ssh_target,
        "remote_dir": remote_dir,
        "target_profile": str(profile_rec.get("stdout", "")),
        "scalar_vector_ops_found": vops,
        "scalar_is_genuinely_scalar": (len(vops) == 0 and "NO_OBJDUMP" not in disasm),
        "summaries": parse_run_stdout(stdout),
        "commands": commands,
    }


def measure_kernel(
    *, kernel_label: str, packed_i4: bool, args: argparse.Namespace, artifact_root: Path
) -> dict[str, Any]:
    kdir = artifact_root / kernel_label
    kdir.mkdir(parents=True, exist_ok=True)

    kept, pruned = build_variant_grid(args.lmuls, args.accs, args.reserve)

    if packed_i4:
        scalar_src, scalar_fn = SCALAR_PACKED_I4_C, "ref_scalar_packed_i4"
        naive_src, naive_fn = NAIVE_PACKED_I4_ANCHOR_C, "naive_anchor_packed_i4"
        emit = _emit_packed_i4_variant
        init_body = PACKED_I4_INIT
    else:
        scalar_src, scalar_fn = SCALAR_BYTE_C, "ref_scalar_byte"
        naive_src, naive_fn = NAIVE_BYTE_ANCHOR_C, "naive_anchor_byte"
        emit = _emit_byte_variant
        init_body = BYTE_INIT

    variant_srcs: dict[str, str] = {}
    for v in kept:
        variant_srcs[v.fn] = emit(v.fn, LMUL_LADDER[v.rung], v.accs)

    harness_src = harness_source(
        scalar_fn=scalar_fn,
        naive_anchor_fn=naive_fn,
        variants=kept,
        counts=list(args.counts),
        scale=args.scale,
        warmups=args.warmups,
        repeats=args.repeats,
        iters=args.iters,
        init_body=init_body,
    )

    result: dict[str, Any] = {
        "kernel": kernel_label,
        "packed_i4": packed_i4,
        "vreg_budget": VREG_BUDGET,
        "reserve_temps": args.reserve,
        "kept_variants": [
            {"label": v.label, "fn": v.fn, "lmul": v.rung, "accs": v.accs,
             "acc_regs_total": v.acc_regs_total, "fits_budget": v.fits_budget}
            for v in kept
        ],
        "pruned_variants": [
            {"label": v.label, "lmul": v.rung, "accs": v.accs,
             "acc_regs_total": v.acc_regs_total,
             "reason": f"{v.acc_regs_total}+{args.reserve} temps > {VREG_BUDGET} vregs"}
            for v in pruned
        ],
    }

    if args.dry_run:
        result["status"] = "dry-run-generated-only"
        result["ssh_evidence"] = False
        return result

    remote = build_and_run_remote(
        kernel_label=kernel_label,
        artifact_dir=kdir,
        scalar_src=scalar_src,
        scalar_fn=scalar_fn,
        naive_src=naive_src,
        naive_fn=naive_fn,
        variant_srcs=variant_srcs,
        harness_src=harness_src,
        ssh_target=args.ssh_target,
        connect_timeout=args.connect_timeout,
        timeout=args.timeout,
    )
    result["status"] = "measured"
    result["ssh_evidence"] = True
    result["ssh_target"] = args.ssh_target
    result["target_profile"] = remote["target_profile"]
    result["scalar_vector_ops_found"] = remote["scalar_vector_ops_found"]
    result["scalar_is_genuinely_scalar"] = remote["scalar_is_genuinely_scalar"]
    result["summaries"] = remote["summaries"]
    result["analysis"] = analyze(remote["summaries"], kept)
    return result


def render_markdown(results: list[dict[str, Any]]) -> str:
    lines = ["# RVV accumulator x LMUL sweep (N3 性能灯 / P-B step 1)", ""]
    lines.append(
        "Timing: `rdcycle` (cycle counts, primary) + `clock_gettime(MONOTONIC_RAW)`, "
        "best-of-N, all variants interleaved in one binary on real `ssh rvv` riscv64 "
        "(VLEN=128). Correctness-guarded (scalar oracle, tol 1e-05) before timing.\n"
    )
    for r in results:
        lines.append(f"## Kernel: `{r['kernel']}`")
        lines.append("")
        lines.append(
            f"- genuine-scalar vector ops in disasm: "
            f"**{r.get('scalar_vector_ops_found') or 'NONE'}**; genuinely scalar: "
            f"**{r.get('scalar_is_genuinely_scalar')}**"
        )
        pruned = r.get("pruned_variants") or []
        if pruned:
            lines.append(
                f"- pruned by 32-vreg budget (resource fact): "
                + ", ".join(f"`{p['label']}` ({p['reason']})" for p in pruned)
            )
        an = r.get("analysis")
        if not an:
            lines.append("")
            continue
        rows = an["rows"]
        var_labels = [v["label"] for v in r["kept_variants"]]
        # cycle table
        lines.append("")
        lines.append("### best cycles/iter (rdcycle)")
        header = "| n | scalar | naive-mf4 | " + " | ".join(var_labels) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (3 + len(var_labels)))
        for row in rows:
            cells = [str(row["n"]), str(row.get("scalar_cyc", "-")),
                     str(row.get("naive_cyc", "-"))]
            for lbl in var_labels:
                v = row["variants"].get(lbl)
                cells.append(str(v["cyc"]) if v else "-")
            lines.append("| " + " | ".join(cells) + " |")
        # ratio table vs naive-anchor
        lines.append("")
        lines.append("### speedup vs naive-mf4-anchor (cycle ratio; >1 = faster)")
        lines.append("| n | best-A=1 | " + " | ".join(var_labels) + " |")
        lines.append("|" + "---|" * (2 + len(var_labels)))
        for row in rows:
            best_a1 = row.get("best_a1_cyc")
            naive = row.get("naive_cyc")
            a1_ratio = f"{naive / best_a1:.3f}" if (best_a1 and naive) else "-"
            cells = [str(row["n"]), a1_ratio]
            for lbl in var_labels:
                v = row["variants"].get(lbl)
                cells.append(f"{v['vs_naive']:.3f}" if (v and v["vs_naive"]) else "-")
            lines.append("| " + " | ".join(cells) + " |")
        # verdict
        lines.append("")
        w = an.get("winner")
        if w:
            wv = an["per_variant_verdict"][w]
            lines.append(
                f"**Winner: `{w}`** — beats genuine-scalar (min {wv['min_vs_scalar']}x) "
                f"AND naive-mf4-anchor (min {wv['min_vs_naive']}x, median "
                f"{wv['median_vs_naive']}x) across all n. "
                f"beats best A=1 (latency-hiding win): {wv.get('beats_best_a1')} "
                f"(median vs best-A=1 {wv.get('median_vs_best_a1')}x)."
            )
        else:
            lines.append(
                "**No variant beats BOTH genuine-scalar AND naive-mf4-anchor across "
                "all n.** (See per-n ratios for the n-range where any config leads.)"
            )
        lines.append("")
        lines.append(f"- A=1-per-LMUL bar (attribution): `{an['a1_by_lmul']}`")
        lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--kernel", action="append",
                   choices=["byte", "packed-i4"], default=None,
                   help="kernel(s) to sweep (default: both)")
    p.add_argument("--artifact-root", default="artifacts/p-b1-accumulator-sweep")
    p.add_argument("--ssh-target", default="rvv")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--counts", type=int, nargs="+", default=list(DEFAULT_COUNTS))
    p.add_argument("--lmuls", nargs="+", default=["mf4", "mf2", "m1", "m2"],
                   choices=list(LMUL_LADDER))
    p.add_argument("--accs", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--reserve", type=int, default=6,
                   help="vregs reserved for product/load temps in the budget prune")
    p.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    p.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS)
    p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    p.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--connect-timeout", type=int, default=20)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    kernels = args.kernel or ["byte", "packed-i4"]
    artifact_root = abi.resolve_repo_relative_path(Path(args.artifact_root))
    artifact_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for k in kernels:
        print(f"[accsweep] measuring kernel={k} dry_run={args.dry_run}")
        results.append(
            measure_kernel(
                kernel_label=k, packed_i4=(k == "packed-i4"),
                args=args, artifact_root=artifact_root,
            )
        )

    evidence = {
        "schema_version": SCHEMA_VERSION,
        "tool": SCRIPT_NAME,
        "created_at": abi.utc_timestamp(),
        "ssh_target": args.ssh_target,
        "dry_run": bool(args.dry_run),
        "timing_method": "rdcycle (primary) + clock_gettime(CLOCK_MONOTONIC_RAW)",
        "vlen_bits": 128,
        "counts": list(args.counts),
        "lmuls": args.lmuls,
        "accs": args.accs,
        "scale": args.scale,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "iters": args.iters,
        "results": results,
    }
    (artifact_root / "evidence.json").write_text(
        json.dumps(evidence, indent=2), encoding="utf-8"
    )
    md = render_markdown(results)
    (artifact_root / "sweep_table.md").write_text(md, encoding="utf-8")
    print(f"[accsweep] wrote {artifact_root / 'evidence.json'}")
    print(f"[accsweep] wrote {artifact_root / 'sweep_table.md'}")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
