#!/usr/bin/env python3
"""coverage_metrics.py — E6 [COV-2] four-coverage-metric report.

Reads the committed denominator-of-record (schema/coverage-roster.v1.json) plus the
hand-labeled six-state numerator input (schema/coverage-sixstate.v1.json) and emits
the four coverage metrics as a CI-report artifact (JSON). This is governance /
workflow tooling and is stdlib-only: Python is tooling here, never the compiler
stack (implementation-stack red line). It does NOT read or touch any C++/ODS, and
it does NOT derive strong-vs-weak from code (that is E5's provenance manifest).

The four metrics ([COV-2]); numerator over the denominator of unique roster keys,
where a key's state = the BEST six-state across its variant rows (实验总纲 line 32):

  C_dispatch       keys with state >= dispatch-wired        (structurally derivable)
  C_construct      keys with state >= constructed (STRONG)  (the burn-down headline
                   [L-8]; STRONG only — block-dot vec_dot is 0/24 today; each strong
                   claim carries auto_readout=pending-E5 until E5 lands)
  C_construct_plus keys with state >= constructed-weak      (transition metric,
                   REPORT-not-gate; block-dot vec_dot = 7/24 today)
  C_attr           leveled string ^CT / load / ^RT          (NOT a single ratio,
                   [COV-2]/[D-4])

Determinism ([GOV-3]): the $meta snapshot uses the git HEAD sha + commit date (NOT
wall-clock), so two runs on the same HEAD produce byte-identical output. Numbers go
to the CI report, never into .trellis/spec/ ([GOV-1]).

Subcommands
-----------
report [--out PATH]   compute the four metrics + snapshot $meta; print (or write).
--self-test           hermetic: synthetic roster + six-state fixture; asserts the
                      metric arithmetic incl. the strong-vs-weak split, best-across-
                      variants dedup, and the pending-E5 surfacing.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# --- locations (single source of truth) ------------------------------------
# This file lives at <repo>/.trellis/scripts/coverage_metrics.py
REPO_ROOT = Path(__file__).resolve().parents[2]
ROSTER_JSON = REPO_ROOT / "schema" / "coverage-roster.v1.json"
SIXSTATE_JSON = REPO_ROOT / "schema" / "coverage-sixstate.v1.json"

# --- six-state ladder ([K-4]) ----------------------------------------------
LADDER = ["absent", "emittable", "dispatch-wired",
          "constructed-weak", "constructed", "covered"]
IDX = {s: i for i, s in enumerate(LADDER)}
I_DISPATCH = IDX["dispatch-wired"]     # C_dispatch threshold
I_WEAK = IDX["constructed-weak"]       # C_construct_plus threshold
I_STRONG = IDX["constructed"]          # C_construct threshold (STRONG only)


# --- canonical hashing (mirrors E1 check_schema_gate.py) -------------------
def canonicalize(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compute_hash(obj) -> str:
    return hashlib.sha256(canonicalize(obj).encode("utf-8")).hexdigest()


# --- keys ------------------------------------------------------------------
def kernel_key(entry):
    """Join key for a roster kernel or a six-state row: (op, format, engine, regime).

    engine is part of identity only where present (gemm_tile rvv vs ime); other
    ops carry engine == "". (op, format) alone disambiguates vec_dot/q4_0 from
    product_reduce/q4_0_nibble because op AND format differ.

    regime is part of identity only where present (gemm_tile/q4_0/rvv splits into
    the DECODE GEVM cell and the PREFILL GEMM cell -- two distinct repack routes
    constructed by lowerToRepackGemv / lowerToRepackGemm on the m_regime axis); all
    other keys carry regime == "" so they are byte-unchanged. The roster row and its
    six-state row must agree on regime to join.
    """
    return (entry["op"], entry["format"], entry.get("engine", ""),
            entry.get("regime", ""))


# --- core metric computation (PURE: no git, no files) ----------------------
def _metric_block(idxs):
    """Given the list of best-state indices for a key set, return the 3 ratios."""
    den = len(idxs)

    def r(threshold):
        num = sum(1 for i in idxs if i >= threshold)
        pct = round(100.0 * num / den, 2) if den else 0.0
        return {"num": num, "den": den, "pct": pct}

    return {
        "C_dispatch": r(I_DISPATCH),
        "C_construct": r(I_STRONG),
        "C_construct_plus": r(I_WEAK),
    }


def compute_metrics(roster_kernels, sixstate_rows):
    """Pure metric computation over roster + six-state lists.

    Returns a dict with per-key resolution, denominator, and metric blocks
    (global / by_class / by_op) plus the headline, C_attr, pending-E5 accounting,
    and any roster keys missing a six-state row.
    """
    # six-state lookup: key -> (best_state_index, auto_readout_of_best). The ladder
    # `state` still drives the metric arithmetic; the M4 three-classification
    # (scope / m4_class) rides on SEPARATE fields so the LADDER logic is untouched.
    best = {}
    meta = {}   # key -> {"scope": str|None, "m4_class": str|None}
    for row in sixstate_rows:
        k = kernel_key(row)
        i = IDX.get(row["state"])
        if i is None:
            raise ValueError(f"unknown six-state {row['state']!r} for {k}")
        cur = best.get(k)
        if cur is None or i > cur[0]:
            best[k] = (i, row.get("auto_readout"))
        m = meta.setdefault(k, {"scope": None, "m4_class": None})
        if row.get("scope"):
            m["scope"] = row["scope"]
        if row.get("m4_class"):
            m["m4_class"] = row["m4_class"]

    resolved = []      # per roster key: (key, class, op, idx, auto_readout)
    missing = []
    out_of_domain = []  # roster keys marked scope=out-of-domain (EXCLUDED from denom)
    for kern in roster_kernels:
        k = kernel_key(kern)
        km = meta.get(k, {"scope": None, "m4_class": None})
        # M4 分母正名: out-of-domain cells (bf16/all, flash_attn/tile) are kept in the
        # roster + sixstate for auditability but EXCLUDED from the coverage denominator.
        if km.get("scope") == "out-of-domain":
            out_of_domain.append(list(k))
            continue
        b = best.get(k)
        if b is None:
            missing.append(list(k))
            idx, auto = IDX["absent"], None
        else:
            idx, auto = b
        resolved.append((k, kern["class"], kern["op"], idx, auto))

    def block_for(pred):
        return _metric_block([idx for (_, cls, op, idx, _) in resolved if pred(cls, op)])

    by_class = {c: block_for(lambda cls, op, cc=c: cls == cc)
                for c in ("A", "B", "C")}
    ops = sorted({op for (_, _, op, _, _) in resolved})
    by_op = {op: block_for(lambda cls, o, oo=op: o == oo) for op in ops}
    global_block = block_for(lambda cls, op: True)

    # pending-E5 accounting over the C_construct (strong) numerator
    strong_keys = [(k, auto) for (k, cls, op, idx, auto) in resolved if idx >= I_STRONG]
    strong_pending = [k for (k, auto) in strong_keys if auto == "pending-E5"]

    denom_by_class = {c: by_class[c]["C_dispatch"]["den"] for c in ("A", "B", "C")}

    # M4 三分类终态 (ROADMAP:52): every IN-DENOMINATOR cell resolves to EXACTLY one of
    # {certified, blocked-on-IME, declared-exception}; out-of-domain cells sit OUTSIDE
    # the denominator. certified = best-state >= constructed (STRONG); the rest carry an
    # explicit m4_class marker. reconciliation MUST close: denominator + out_of_domain ==
    # roster_total, and there must be ZERO undefined cells.
    m4 = {"certified": [], "blocked-on-IME": [], "declared-exception": [],
          "undefined": []}
    for (k, cls, op, idx, auto) in resolved:
        if idx >= I_STRONG:
            m4["certified"].append(list(k))
        else:
            cls_m4 = meta.get(k, {}).get("m4_class")
            if cls_m4 in ("blocked-on-IME", "declared-exception"):
                m4[cls_m4].append(list(k))
            else:
                m4["undefined"].append(list(k))
    roster_total = len(resolved) + len(out_of_domain)
    m4_classification = {
        "certified": len(m4["certified"]),
        "blocked_on_IME": len(m4["blocked-on-IME"]),
        "declared_exception": len(m4["declared-exception"]),
        "out_of_domain": len(out_of_domain),
        "denominator": len(resolved),
        "roster_total": roster_total,
        "undefined_cells": m4["undefined"],
        "reconciliation_ok": (
            len(m4["certified"]) + len(m4["blocked-on-IME"])
            + len(m4["declared-exception"]) == len(resolved)
            and len(resolved) + len(out_of_domain) == roster_total
            and len(m4["undefined"]) == 0
        ),
        "note": ("M4 three-classification over the denominator + out-of-domain; "
                 "reconciliation_ok asserts zero undefined cells and denom + "
                 "out_of_domain == roster_total"),
        "blocked_on_IME_cells": m4["blocked-on-IME"],
        "declared_exception_cells": m4["declared-exception"],
        "out_of_domain_cells": out_of_domain,
    }

    return {
        "denominator": {"total": len(resolved), "by_class": denom_by_class},
        "metrics": {
            "global": global_block,
            "by_class": by_class,
            "by_op": by_op,
        },
        "headline": {
            "note": "burn-down headline: block-dot vec_dot C_construct is STRONG-only",
            "vec_dot": by_op.get("vec_dot"),
            "product_reduce": by_op.get("product_reduce"),
        },
        "C_attr": {
            "level": ("^CT: partial (in-IR selection attrs from E4; no attribution "
                      "JSONL yet) | load: 0 | ^RT: 0"),
            "note": "leveled per [COV-2]/[D-4]; NOT a single ratio",
        },
        "C_construct_pending_e5": {
            "strong_keys": len(strong_keys),
            "pending_e5_keys": len(strong_pending),
            "all_strong_pending_e5": len(strong_keys) == len(strong_pending),
            "note": ("weak-vs-strong AUTO-readout is deferred to E5 (provenance "
                     "manifest); the strong labels here are hand-assigned"),
        },
        "missing_sixstate_keys": missing,
        "m4_classification": m4_classification,
    }


# --- git plumbing (only the live `report` path) ----------------------------
def _git(args):
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def build_report():
    roster = json.loads(ROSTER_JSON.read_text(encoding="utf-8"))
    sixstate = json.loads(SIXSTATE_JSON.read_text(encoding="utf-8"))
    result = compute_metrics(roster["kernels"], sixstate["states"])
    result["$meta"] = {
        "report": "coverage_metrics",
        "repo_snapshot": _git(["rev-parse", "HEAD"]),
        "snapshot_commit_date": _git(["show", "-s", "--format=%cI", "HEAD"]),
        "ggml_pin": roster["$meta"].get("ggml_pin"),
        "epoch": roster["$meta"].get("epoch"),
        "roster_sha256": compute_hash(roster),
        "sixstate_sha256": compute_hash(sixstate),
        "spec_boundary": "numbers live in the CI report / 执行总纲, never in .trellis/spec/",
    }
    return result


def cmd_report(args) -> int:
    result = build_report()
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        out = Path(args.out)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}")
    else:
        print(text)
    return 0


# --- self-test -------------------------------------------------------------
def cmd_self_test(_args) -> int:
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    # Synthetic fixture: mirror the real block-dot shape on a tiny universe.
    # 4 vec_dot: 2 dispatch-wired + 2 constructed-weak.
    # 1 product_reduce: constructed (STRONG, pending-E5).
    # 1 quantize_row: absent.
    roster = [
        {"op": "vec_dot", "format": "w1", "class": "A", "bucket": "flat"},
        {"op": "vec_dot", "format": "w2", "class": "A", "bucket": "flat"},
        {"op": "vec_dot", "format": "s1", "class": "A", "bucket": "super"},
        {"op": "vec_dot", "format": "s2", "class": "A", "bucket": "super"},
        {"op": "product_reduce", "format": "d1", "class": "A", "bucket": "decomp"},
        {"op": "quantize_row", "format": "q", "class": "A", "bucket": "quant"},
    ]
    sixstate = [
        {"op": "vec_dot", "format": "w1", "state": "constructed-weak",
         "auto_readout": "pending-E5"},
        {"op": "vec_dot", "format": "w2", "state": "constructed-weak",
         "auto_readout": "pending-E5"},
        {"op": "vec_dot", "format": "s1", "state": "dispatch-wired"},
        {"op": "vec_dot", "format": "s2", "state": "dispatch-wired"},
        {"op": "product_reduce", "format": "d1", "state": "constructed",
         "auto_readout": "pending-E5"},
        {"op": "quantize_row", "format": "q", "state": "absent"},
    ]
    m = compute_metrics(roster, sixstate)
    vd = m["metrics"]["by_op"]["vec_dot"]
    check("vec_dot C_dispatch = 4/4 (all >= dispatch-wired)",
          vd["C_dispatch"]["num"] == 4 and vd["C_dispatch"]["den"] == 4)
    check("vec_dot C_construct = 0/4 (STRONG only; weak does NOT count)",
          vd["C_construct"]["num"] == 0)
    check("vec_dot C_construct_plus = 2/4 (the two constructed-weak)",
          vd["C_construct_plus"]["num"] == 2)
    pr = m["metrics"]["by_op"]["product_reduce"]
    check("product_reduce C_construct = 1/1 (STRONG)",
          pr["C_construct"]["num"] == 1 and pr["C_construct"]["den"] == 1)
    check("global C_construct counts strong only = 1/6",
          m["metrics"]["global"]["C_construct"]["num"] == 1
          and m["metrics"]["global"]["C_construct"]["den"] == 6)
    check("global C_dispatch = 5/6 (only quantize_row absent pulls it down; "
          "the 2 weak vec_dot ARE >= dispatch-wired)",
          m["metrics"]["global"]["C_dispatch"]["num"] == 5
          and m["metrics"]["global"]["C_dispatch"]["den"] == 6)
    check("pending-E5 surfaces: all strong keys pending",
          m["C_construct_pending_e5"]["strong_keys"] == 1
          and m["C_construct_pending_e5"]["all_strong_pending_e5"] is True)
    check("no missing six-state keys in the fixture",
          m["missing_sixstate_keys"] == [])

    # best-state-across-variants dedup: same key, two rows, best (strong) wins.
    roster_v = [{"op": "vec_dot", "format": "x", "class": "A", "bucket": "b"}]
    sixstate_v = [
        {"op": "vec_dot", "format": "x", "state": "dispatch-wired"},
        {"op": "vec_dot", "format": "x", "state": "constructed",
         "auto_readout": "pending-E5"},
    ]
    mv = compute_metrics(roster_v, sixstate_v)
    check("best-across-variants: strong variant lifts the key into C_construct",
          mv["metrics"]["global"]["C_construct"]["num"] == 1)

    # weak-only key must NOT count in C_construct (strong), only in C_construct_plus.
    roster_w = [{"op": "vec_dot", "format": "y", "class": "A", "bucket": "b"}]
    sixstate_w = [{"op": "vec_dot", "format": "y", "state": "constructed-weak",
                   "auto_readout": "pending-E5"}]
    mw = compute_metrics(roster_w, sixstate_w)
    check("weak-only: C_construct = 0/1 but C_construct_plus = 1/1",
          mw["metrics"]["global"]["C_construct"]["num"] == 0
          and mw["metrics"]["global"]["C_construct_plus"]["num"] == 1)

    # gemm_tile engine disambiguation: rvv vs ime are distinct keys.
    roster_g = [
        {"op": "gemm_tile", "format": "q4_0", "class": "A", "engine": "rvv"},
        {"op": "gemm_tile", "format": "q4_0", "class": "A", "engine": "ime"},
    ]
    sixstate_g = [
        {"op": "gemm_tile", "format": "q4_0", "engine": "rvv", "state": "dispatch-wired"},
        {"op": "gemm_tile", "format": "q4_0", "engine": "ime", "state": "absent"},
    ]
    mg = compute_metrics(roster_g, sixstate_g)
    check("gemm_tile rvv/ime are distinct keys (C_dispatch = 1/2)",
          mg["metrics"]["global"]["C_dispatch"]["num"] == 1
          and mg["metrics"]["global"]["C_dispatch"]["den"] == 2)

    # M4 三分类: out-of-domain exclusion + the three-classification reconciliation.
    roster_od = [
        {"op": "vec_dot", "format": "cert", "class": "A"},
        {"op": "gemm_tile", "format": "ime1", "class": "A", "engine": "ime"},
        {"op": "gemm_tile", "format": "de1", "class": "A", "engine": "rvv"},
        {"op": "bf16", "format": "all", "class": "C"},
    ]
    sixstate_od = [
        {"op": "vec_dot", "format": "cert", "state": "constructed",
         "auto_readout": "pending-E5"},
        {"op": "gemm_tile", "format": "ime1", "engine": "ime", "state": "absent",
         "m4_class": "blocked-on-IME"},
        {"op": "gemm_tile", "format": "de1", "engine": "rvv", "state": "absent",
         "m4_class": "declared-exception"},
        {"op": "bf16", "format": "all", "state": "absent",
         "scope": "out-of-domain", "m4_class": "out-of-domain"},
    ]
    mo = compute_metrics(roster_od, sixstate_od)
    m4s = mo["m4_classification"]
    check("out-of-domain EXCLUDED from denominator (4 roster -> 3 denom, 1 out)",
          m4s["denominator"] == 3 and m4s["out_of_domain"] == 1
          and m4s["roster_total"] == 4 and mo["denominator"]["total"] == 3)
    check("M4 three-classification 1 certified + 1 blocked-on-IME + 1 declared-exception",
          m4s["certified"] == 1 and m4s["blocked_on_IME"] == 1
          and m4s["declared_exception"] == 1)
    check("M4 reconciliation closes with ZERO undefined cells",
          m4s["reconciliation_ok"] is True and m4s["undefined_cells"] == [])

    # a non-certified in-denominator cell with NO m4_class surfaces as UNDEFINED.
    roster_u = [{"op": "vec_dot", "format": "x", "class": "A"}]
    sixstate_u = [{"op": "vec_dot", "format": "x", "state": "absent"}]
    mu = compute_metrics(roster_u, sixstate_u)
    check("un-marked non-certified cell surfaces as undefined (reconciliation fails)",
          mu["m4_classification"]["undefined_cells"] == [["vec_dot", "x", "", ""]]
          and mu["m4_classification"]["reconciliation_ok"] is False)

    # determinism of the canonical hash.
    check("canonical hash is order-insensitive",
          compute_hash({"a": 1, "b": 2}) == compute_hash({"b": 2, "a": 1}))

    ok = True
    for name, passed in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(p for _, p in results)}/{len(results)})")
    return 0 if ok else 1


# --- entrypoint ------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true",
                        help="run hermetic self-tests and exit")
    sub = parser.add_subparsers(dest="cmd")

    p_report = sub.add_parser("report", help="[COV-2] compute the four metrics")
    p_report.add_argument("--out", default=None,
                          help="write the JSON report to PATH (default: stdout)")
    p_report.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    if args.self_test:
        return cmd_self_test(args)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
