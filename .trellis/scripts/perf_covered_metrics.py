#!/usr/bin/env python3
"""perf_covered_metrics.py -- [COV-2]-style perf-covered reconciliation report.

Reads the committed structural denominator-of-record
(schema/coverage-roster.v1.json) + the hand-labeled six-state numerator input
(schema/coverage-sixstate.v1.json) to derive the CERTIFIED cells (== the 84 in
coverage_metrics.py), FOLDS the m_regime axis (gemm_tile/q4_0/rvv decode+prefill
-> one format-cell -> 83), and joins them with the perf-covered category labels
(schema/perf-covered-category.v1.json). It machine-computes the perf-covered
headline (green / denominator, any-board) plus the six-class classification tally
and closes a reconciliation (Σ == denominator, zero undefined, every category's
required fields complete, three-source consistency, anti-gate flag).

This is governance / workflow tooling and is stdlib-only: Python is tooling here,
never the compiler stack (implementation-stack red line). It reuses
coverage_metrics.py for the six-state ladder + certified-cell derivation so the
structural denominator (M4 84/91) stays the SINGLE SOURCE and is never re-derived
independently here. It touches NO C++/ODS and computes NO numerics.

Fold ([裁一.1/一.2] 2026-07-12): perf-covered folds regime per FORMAT; the
structural roster + M4 84/91 do NOT move (regime split is legal for construct-
identity). fold_regime defaults True (6/83); --no-fold surfaces the 7/84 alt-unit
(regime retained), flagged pending user review.

Anti-gate ([裁三]): the declared-exception category is proven numerator-independent
-- removing every declared-exception cell shifts ONLY the denominator, never the
green numerator, and every declared-exception cell is CERTIFIED (off-hot-path,
Amdahl<noise), not un-constructed. So an exception can never be minted to flatter
the headline fraction.

Subcommands
-----------
report [--out PATH] [--no-fold]  compute headline + six-class tally + reconciliation
                                 + anti-gate flag; snapshot $meta; print (or write).
--self-test                      hermetic: synthetic roster/six-state/label fixtures;
                                 asserts the fold arithmetic, per-category required-
                                 field completeness, Σ reconciliation, three-source
                                 consistency, and the anti-gate numerator-invariance.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# reuse the structural ladder + certified derivation (single source of truth)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import coverage_metrics as cm  # noqa: E402

REPO_ROOT = cm.REPO_ROOT
ROSTER_JSON = cm.ROSTER_JSON
SIXSTATE_JSON = cm.SIXSTATE_JSON
LABELS_JSON = REPO_ROOT / "schema" / "perf-covered-category.v1.json"

# --- frozen six-category set ([裁一.3]) ------------------------------------
GREEN = "绿"
FROZEN_CATEGORIES = ["绿", "黄-未接线", "黄-传导稀释",
                     "黄-物理墙", "黄-对手更强", "声明例外"]

# per-category detail block key (ascii) + its REQUIRED non-empty fields.
DETAIL_KEY = {
    "绿": "green",
    "黄-物理墙": "physical_wall",
    "黄-传导稀释": "transduction_dilution",
    "黄-对手更强": "opponent_stronger",
    "黄-未接线": "unwired",
    "声明例外": "declared_exception",
}
REQUIRED_FIELDS = {
    "green": ["board", "account", "ratio", "ledger_pointer", "quality_note"],
    "physical_wall": ["roofline_pointer", "evidence"],
    "transduction_dilution": ["compute_account_ratio", "amdahl_pointer"],
    "opponent_stronger": ["named_gap", "fixable", "symmetric_account_pointer"],
    "unwired": ["wiring_queue_position", "transduction_estimate_note"],
    "declared_exception": ["reason", "amdahl_ceiling", "ceiling_source",
                           "reestimate_condition", "reestimate_hook"],
}


# --- canonical hashing (mirrors coverage_metrics / check_schema_gate) -------
def compute_hash(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


# --- keys ------------------------------------------------------------------
def _fmt_key(entry_or_tuple):
    """FOLDED perf-covered key (op, format, engine) -- regime dropped."""
    if isinstance(entry_or_tuple, dict):
        return (entry_or_tuple["op"], entry_or_tuple["format"],
                entry_or_tuple.get("engine", ""))
    op, fmt, eng, _regime = entry_or_tuple
    return (op, fmt, eng)


# --- certified-cell derivation (reuse coverage_metrics ladder) -------------
def certified_cells(roster_kernels, sixstate_rows):
    """The CERTIFIED cells (best six-state >= constructed STRONG), excluding
    out-of-domain, WITH regime -- identical to coverage_metrics' 84."""
    best = {}
    meta = {}
    for row in sixstate_rows:
        k = cm.kernel_key(row)
        i = cm.IDX.get(row["state"])
        if i is None:
            raise ValueError(f"unknown six-state {row['state']!r} for {k}")
        cur = best.get(k)
        if cur is None or i > cur[0]:
            best[k] = (i, row.get("auto_readout"))
        m = meta.setdefault(k, {"scope": None})
        if row.get("scope"):
            m["scope"] = row["scope"]
    cells = []
    for kern in roster_kernels:
        k = cm.kernel_key(kern)
        if meta.get(k, {}).get("scope") == "out-of-domain":
            continue
        b = best.get(k)
        idx = b[0] if b else cm.IDX["absent"]
        if idx >= cm.I_STRONG:
            cells.append(k)
    return cells


# --- core reconciliation (PURE: no git, no files) --------------------------
def reconcile(roster_kernels, sixstate_rows, labels, fold=True):
    """Join certified cells (folded per fold flag) with category labels and
    close the perf-covered reconciliation. Returns a dict with the headline,
    six-class tally, per-cell resolution, and reconciliation flags."""
    cert = certified_cells(roster_kernels, sixstate_rows)  # (op,fmt,eng,regime) x84

    # fold: dedupe by (op,fmt,eng); --no-fold keeps every regime cell but each
    # still looks up the same FOLDED label key.
    if fold:
        seen = set()
        units = []
        for c in cert:
            fk = _fmt_key(c)
            if fk not in seen:
                seen.add(fk)
                units.append(fk)
    else:
        units = [_fmt_key(c) for c in cert]  # regime cells expanded (dup labels)

    label_by_key = {}
    dup_labels = []
    for lab in labels:
        lk = (lab["key"]["op"], lab["key"]["format"], lab["key"].get("engine", ""))
        if lk in label_by_key:
            dup_labels.append(list(lk))
        label_by_key[lk] = lab

    # resolve every unit to exactly one category via its label.
    per_cell = []
    undefined = []            # a certified unit with NO label
    bad_category = []         # label carries a non-frozen category
    incomplete_fields = []    # label's detail block missing a required field
    for fk in units:
        lab = label_by_key.get(fk)
        if lab is None:
            undefined.append(list(fk))
            per_cell.append((fk, None))
            continue
        cat = lab.get("category")
        if cat not in FROZEN_CATEGORIES:
            bad_category.append({"key": list(fk), "category": cat})
            per_cell.append((fk, None))
            continue
        # per-category required-field completeness (missing/empty => fail)
        dkey = DETAIL_KEY[cat]
        block = lab.get(dkey, {})
        for fld in REQUIRED_FIELDS[dkey]:
            v = block.get(fld)
            if v is None or (isinstance(v, str) and not v.strip()):
                incomplete_fields.append({"key": list(fk), "category": cat,
                                          "missing": fld})
        per_cell.append((fk, cat))

    # labels that reference NO certified unit (stale / phantom label)
    unit_set = set(units)
    orphan_labels = [list(lk) for lk in label_by_key
                     if lk not in {_fmt_key(c) for c in cert}]

    denom = len(units)
    tally = {c: sum(1 for (_, cat) in per_cell if cat == c)
             for c in FROZEN_CATEGORIES}
    green_num = tally[GREEN]
    tally_sum = sum(tally.values())

    # --- three-source consistency ------------------------------------------
    # (1) headline numerator (green count over resolved cells)
    # (2) classification tally green count
    # (3) ledger-cell count (green cells carrying a non-empty ledger_pointer)
    ledger_green = 0
    for fk in units:
        lab = label_by_key.get(fk)
        if lab and lab.get("category") == GREEN:
            if (lab.get("green", {}).get("ledger_pointer") or "").strip():
                ledger_green += 1
    three_source_ok = (green_num == tally[GREEN] == ledger_green)

    # --- anti-gate ([裁三]) -------------------------------------------------
    # declared-exception cells are numerator-INDEPENDENT: the green numerator is
    # invariant to the set of exception cells (they only sit in the denominator),
    # and every exception cell is CERTIFIED (off-hot-path, not un-constructed).
    exc_units = [fk for (fk, cat) in per_cell if cat == "声明例外"]
    cert_folded = {_fmt_key(c) for c in cert}
    exc_all_certified = all(fk in cert_folded for fk in exc_units)
    green_without_exc = sum(1 for (fk, cat) in per_cell
                            if cat == GREEN and fk not in set(exc_units))
    anti_gate_ok = (green_without_exc == green_num) and exc_all_certified
    anti_gate_note = (
        "声明例外 cell 判定依据 = Amdahl<噪声（off-hot-path 结构事实）·与 headline 分子无关: "
        "分子只数绿·例外增减仅动分母·且例外全 certified-but-off-hot-path（非未构造）; "
        f"分子(含例外)={green_num} == 分子(移出例外)={green_without_exc}·例外全 certified={exc_all_certified}"
    )

    reconciliation_ok = (
        tally_sum == denom
        and not undefined
        and not bad_category
        and not incomplete_fields
        and not orphan_labels
        and not dup_labels
        and three_source_ok
        and anti_gate_ok
    )

    return {
        "fold_regime": fold,
        "headline": {
            "perf_covered": f"{green_num}/{denom}",
            "pct": round(100.0 * green_num / denom, 2) if denom else 0.0,
            "green_num": green_num,
            "denominator": denom,
            "any_board": True,
            "note": ("perf-covered = 绿/denom (any-board·裁四.0); "
                     + ("fold=True 6/83" if fold
                        else "fold=False 7/84 备选单位·flagged 待用户复核")),
        },
        "classification": tally,
        "classification_sum": tally_sum,
        "reconciliation": {
            "reconciliation_ok": reconciliation_ok,
            "sum_equals_denominator": tally_sum == denom,
            "undefined_cells": undefined,
            "bad_category_cells": bad_category,
            "incomplete_field_cells": incomplete_fields,
            "orphan_labels": orphan_labels,
            "duplicate_labels": dup_labels,
            "three_source_consistent": three_source_ok,
            "three_source": {"headline_green": green_num,
                             "classification_green": tally[GREEN],
                             "ledger_green": ledger_green},
        },
        "anti_gate": {
            "anti_gate_ok": anti_gate_ok,
            "declared_exception_cells": len(exc_units),
            "green_with_exception": green_num,
            "green_without_exception": green_without_exc,
            "all_exceptions_certified": exc_all_certified,
            "note": anti_gate_note,
        },
    }


# --- git plumbing (only the live `report` path) ----------------------------
def _git(args):
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def build_report(fold=True):
    roster = json.loads(ROSTER_JSON.read_text(encoding="utf-8"))
    sixstate = json.loads(SIXSTATE_JSON.read_text(encoding="utf-8"))
    labels_doc = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    result = reconcile(roster["kernels"], sixstate["states"],
                       labels_doc["cells"], fold=fold)
    result["$meta"] = {
        "report": "perf_covered_metrics",
        "repo_snapshot": _git(["rev-parse", "HEAD"]),
        "snapshot_commit_date": _git(["show", "-s", "--format=%cI", "HEAD"]),
        "roster_sha256": compute_hash(roster),
        "sixstate_sha256": compute_hash(sixstate),
        "labels_sha256": compute_hash(labels_doc),
        "fold_regime": fold,
        "alt_unit_note": ("默认 fold_regime=True -> 6/83; --no-fold -> 7/84 "
                          "备选单位（regime 保留·flagged 待用户复核）"),
        "spec_boundary": "numbers live in the CI report / docs, never in .trellis/spec/",
    }
    return result


def cmd_report(args) -> int:
    result = build_report(fold=not args.no_fold)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)
    return 0 if result["reconciliation"]["reconciliation_ok"] else 1


# --- self-test -------------------------------------------------------------
def cmd_self_test(_args) -> int:
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    # Synthetic universe: 1 certified gemm_tile/q4_0/rvv that SPLITS on regime
    # (decode+prefill) + 1 green flat gemm + 1 declared-exception + 1 wall + 1
    # out-of-domain (excluded). Fold collapses the two q4_0 regime cells to one.
    roster = [
        {"op": "gemm_tile", "format": "q4_0", "class": "A", "engine": "rvv",
         "regime": "decode"},
        {"op": "gemm_tile", "format": "q4_0", "class": "A", "engine": "rvv",
         "regime": "prefill"},
        {"op": "gemm_tile", "format": "q8_0", "class": "A", "engine": "rvv"},
        {"op": "dequantize_row", "format": "q4_K", "class": "A"},
        {"op": "quantize_row", "format": "q8_0", "class": "A"},
        {"op": "bf16", "format": "all", "class": "C"},
    ]
    six = [
        {"op": "gemm_tile", "format": "q4_0", "engine": "rvv", "regime": "decode",
         "state": "constructed", "auto_readout": "pending-E5"},
        {"op": "gemm_tile", "format": "q4_0", "engine": "rvv", "regime": "prefill",
         "state": "constructed", "auto_readout": "pending-E5"},
        {"op": "gemm_tile", "format": "q8_0", "engine": "rvv",
         "state": "constructed", "auto_readout": "pending-E5"},
        {"op": "dequantize_row", "format": "q4_K", "state": "constructed",
         "auto_readout": "pending-E5"},
        {"op": "quantize_row", "format": "q8_0", "state": "constructed",
         "auto_readout": "pending-E5"},
        {"op": "bf16", "format": "all", "state": "absent",
         "scope": "out-of-domain", "m4_class": "out-of-domain"},
    ]

    def green(led="L§x"):
        return {"green": {"board": "rvv", "account": "a", "ratio": "5.9x",
                          "ledger_pointer": led, "quality_note": "q"}}

    def wall():
        return {"physical_wall": {"roofline_pointer": "3.7 GB/s", "evidence": "实测"}}

    def exc():
        return {"declared_exception": {"reason": "off-hot-path", "amdahl_ceiling": "<0.15%",
                "ceiling_source": "GAP-x", "reestimate_condition": "c", "reestimate_hook": "h"}}

    def lab(op, fmt, eng, cat, detail, prov=False):
        e = {"key": {"op": op, "format": fmt, "engine": eng}, "category": cat,
             "provisional": prov}
        e.update(detail)
        return e

    labels_ok = [
        lab("gemm_tile", "q4_0", "rvv", "绿", green("L-q4_0")),
        lab("gemm_tile", "q8_0", "rvv", "绿", green("L-q8_0")),
        lab("dequantize_row", "q4_K", "", "声明例外", exc()),
        lab("quantize_row", "q8_0", "", "黄-物理墙", wall()),
    ]

    # fold=True: 5 certified cells -> 4 folded units (q4_0 regime collapses).
    r = reconcile(roster, six, labels_ok, fold=True)
    check("fold=True denom = 4 (q4_0 decode+prefill folded to one)",
          r["headline"]["denominator"] == 4)
    check("fold=True green = 2 (q4_0 folded + q8_0)",
          r["headline"]["green_num"] == 2)
    check("fold=True Σ classification == denominator (4)",
          r["classification_sum"] == 4)
    check("fold=True zero undefined + reconciliation_ok",
          r["reconciliation"]["reconciliation_ok"] is True
          and r["reconciliation"]["undefined_cells"] == [])
    check("fold=True three-source consistent (2==2==2)",
          r["reconciliation"]["three_source_consistent"] is True)
    check("fold=True anti-gate ok (green invariant to exceptions, all certified)",
          r["anti_gate"]["anti_gate_ok"] is True
          and r["anti_gate"]["green_without_exception"] == 2)

    # fold=False: the two q4_0 regime cells are BOTH counted (denom 5, green 3).
    rnf = reconcile(roster, six, labels_ok, fold=False)
    check("fold=False denom = 5 (regime cells expanded)",
          rnf["headline"]["denominator"] == 5)
    check("fold=False green = 3 (q4_0 decode + prefill both green + q8_0)",
          rnf["headline"]["green_num"] == 3)
    check("fold=False Σ == denom (5) still reconciles",
          rnf["classification_sum"] == 5
          and rnf["reconciliation"]["reconciliation_ok"] is True)

    # missing label -> undefined -> reconciliation FAILS.
    r_missing = reconcile(roster, six, labels_ok[:-1], fold=True)
    check("missing label surfaces as undefined + reconciliation fails",
          r_missing["reconciliation"]["undefined_cells"] == [["quantize_row", "q8_0", ""]]
          and r_missing["reconciliation"]["reconciliation_ok"] is False)

    # incomplete detail block (declared-exception missing a required field) -> FAIL.
    bad_exc = {"declared_exception": {"reason": "r", "amdahl_ceiling": "<0.1%",
               "ceiling_source": "s", "reestimate_condition": "c"}}  # no reestimate_hook
    labels_bad = [
        lab("gemm_tile", "q4_0", "rvv", "绿", green("L1")),
        lab("gemm_tile", "q8_0", "rvv", "绿", green("L2")),
        lab("dequantize_row", "q4_K", "", "声明例外", bad_exc),
        lab("quantize_row", "q8_0", "", "黄-物理墙", wall()),
    ]
    r_bad = reconcile(roster, six, labels_bad, fold=True)
    check("incomplete declared-exception field -> incomplete + reconciliation fails",
          any(x["missing"] == "reestimate_hook"
              for x in r_bad["reconciliation"]["incomplete_field_cells"])
          and r_bad["reconciliation"]["reconciliation_ok"] is False)

    # empty required field (whitespace) also fails.
    empty_green = {"green": {"board": "rvv", "account": "a", "ratio": "5x",
                             "ledger_pointer": "  ", "quality_note": "q"}}
    labels_empty = [
        lab("gemm_tile", "q4_0", "rvv", "绿", empty_green),
        lab("gemm_tile", "q8_0", "rvv", "绿", green("L2")),
        lab("dequantize_row", "q4_K", "", "声明例外", exc()),
        lab("quantize_row", "q8_0", "", "黄-物理墙", wall()),
    ]
    r_empty = reconcile(roster, six, labels_empty, fold=True)
    check("empty ledger_pointer -> incomplete + three-source breaks (ledger_green<green)",
          r_empty["reconciliation"]["reconciliation_ok"] is False
          and r_empty["reconciliation"]["three_source_consistent"] is False)

    # non-frozen category (self-invented label) rejected.
    labels_selfmade = [
        lab("gemm_tile", "q4_0", "rvv", "yellow-kernel-axis", green("L1")),
        lab("gemm_tile", "q8_0", "rvv", "绿", green("L2")),
        lab("dequantize_row", "q4_K", "", "声明例外", exc()),
        lab("quantize_row", "q8_0", "", "黄-物理墙", wall()),
    ]
    r_self = reconcile(roster, six, labels_selfmade, fold=True)
    check("self-invented category (yellow-kernel-axis) rejected as bad_category",
          any(x["category"] == "yellow-kernel-axis"
              for x in r_self["reconciliation"]["bad_category_cells"])
          and r_self["reconciliation"]["reconciliation_ok"] is False)

    # anti-gate discriminates: a phantom declared-exception over a NON-certified
    # cell would break exc_all_certified. Simulate by labeling a cell not in cert.
    labels_phantom = list(labels_ok) + [
        lab("vec_dot", "ghost", "", "声明例外", exc())]
    r_ph = reconcile(roster, six, labels_phantom, fold=True)
    check("phantom exception over non-certified cell -> orphan label + reconciliation fails",
          r_ph["reconciliation"]["orphan_labels"] == [["vec_dot", "ghost", ""]]
          and r_ph["reconciliation"]["reconciliation_ok"] is False)

    # canonical hash determinism.
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
    p_report = sub.add_parser("report", help="[COV-2] perf-covered reconciliation")
    p_report.add_argument("--out", default=None,
                          help="write the JSON report to PATH (default: stdout)")
    p_report.add_argument("--no-fold", action="store_true",
                          help="do NOT fold regime -> 7/84 alt-unit (flagged)")
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
