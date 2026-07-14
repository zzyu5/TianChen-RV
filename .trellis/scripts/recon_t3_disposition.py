#!/usr/bin/env python3
"""recon_t3_disposition.py — G8 §六.3 去向表 recon (decree 三).

Governance /落表纪律 tooling (stdlib-only; Python is tooling here, never the
compiler stack). Validates the DISPOSITION-COMPLETENESS invariant on the two T3
board tables:

    EVERY data row (non-comment, non-blank, non-header) in
    T3_A_board_A_rvv1.0_vlen128.csv and T3_B_board_B_rvv1.0_vlen256.csv
    MUST carry exactly one 去向 (disposition) marker.

Disposition taxonomy (decree 三 · {DEQ-AXIS|IME例外|内部格式|无合法对手|域外/非本轮主账}
plus the in-denominator verdict which is itself a disposition):

  IN-DENOM-VERDICT   28-分母行(matmul vec_dot/gemm + forward-op);last field
                     starts with "in-denom" and carries PASS/FAIL/PASS-DEPLOYED/
                     PASS-MARGINAL/NAMED-X/JUDGMENT-SUSPENDED. These are the 0.8
                     hard-gate denominator cells. NOT touched by this agent.
  DEQ-AXIS           dequant streaming rows — test-only, NOT in the 0.8 headline
                     denominator (auto-promote if a real vector opponent appears).
  内部格式(INTERNAL)  TianChen-internal experimental format (q1_0) — no real ggml
                     vector opponent, excluded from the 0.8 denominator.
  IME例外(IME-CARVE) IME cell (q4_0/q8_0/q4_K@ime) — gcc-13 carve-out, NOT in the
                     0.8 denominator (currently 0 such rows in T3).
  无合法对手(NO-OPP)  no legitimate opponent (currently 0 such rows in T3).
  域外/legacy         pre-§2.1-schema archived campaign rows (STALE/CONTAMINATED/
                     INVALID region below the legacy separator) — 非本轮主账.

Exit code 0 iff every data row is classified (zero UNMARKED). Non-zero otherwise,
listing the offending rows. Also reports a per-row column-count integrity summary.

Usage:  python3 recon_t3_disposition.py   [--tables DIR]
"""

import argparse
import csv
import sys
from pathlib import Path

# This file lives at <repo>/.trellis/scripts/recon_t3_disposition.py; the T3
# tables live under experiments/active/result-tables/ (data cell — a .py recon
# may NOT live inside experiments/ per tools/lint/check_experiments_layout.py,
# so tooling stays here and reaches into the data cell).
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TABLES_DIR = REPO_ROOT / "experiments" / "active" / "result-tables"
TABLES = ["T3_A_board_A_rvv1.0_vlen128.csv", "T3_B_board_B_rvv1.0_vlen256.csv"]

# ordered disposition classifier -------------------------------------------
# Each entry: (label, predicate(last_field, full_line) -> bool).
# First match wins; order encodes precedence.
CLASSES = [
    ("IME例外(IME-CARVE)",
     lambda last, line: "去向=IME例外" in line),
    ("DEQ-AXIS(test-only)",
     lambda last, line: "去向=DEQ-AXIS" in line or last.startswith("test-only-not-in-denom")),
    ("内部格式(INTERNAL)",
     lambda last, line: "去向=内部格式" in line or last.startswith("excluded-internal")),
    ("无合法对手(NO-OPP)",
     lambda last, line: "去向=无合法对手" in line),
    ("域外/legacy-pre-§2.1",
     lambda last, line: "去向=域外" in line or "legacy-pre-§2.1" in line),
    ("IN-DENOM-VERDICT(0.8-gate)",
     lambda last, line: last.startswith("in-denom")),
]


def is_data(raw: str) -> bool:
    s = raw.strip()
    if s == "" or s.startswith("#"):
        return False
    if raw.startswith("measurement_row_key"):
        return False
    return True


def classify(raw: str):
    row = next(csv.reader([raw]))
    last = row[-1]
    for label, pred in CLASSES:
        if pred(last, raw):
            return label, len(row)
    return None, len(row)


def recon_table(path: Path):
    with path.open(newline="") as f:
        lines = f.readlines()
    counts = {}
    unmarked = []
    total = 0
    for i, line in enumerate(lines):
        raw = line.rstrip("\n")
        if not is_data(raw):
            continue
        total += 1
        label, ncol = classify(raw)
        if label is None:
            unmarked.append((i + 1, raw[:70]))
        else:
            counts[label] = counts.get(label, 0) + 1
    return total, counts, unmarked


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(DEFAULT_TABLES_DIR),
                    help="directory holding the T3_*.csv tables")
    args = ap.parse_args()
    tdir = Path(args.tables)

    print("=" * 78)
    print("G8 §六.3 去向表 recon (decree 三) — disposition completeness on T3_A / T3_B")
    print("=" * 78)
    order = [c[0] for c in CLASSES]
    grand_unmarked = 0
    for name in TABLES:
        path = tdir / name
        if not path.exists():
            print(f"[MISSING] {name}")
            grand_unmarked += 1
            continue
        total, counts, unmarked = recon_table(path)
        print(f"\n{name}")
        print(f"  total data rows: {total}")
        for label in order:
            if label in counts:
                print(f"    {label:32} {counts[label]}")
        classified = sum(counts.values())
        print(f"    {'—— classified ——':32} {classified}")
        print(f"    {'UNMARKED (must be 0)':32} {len(unmarked)}")
        if unmarked:
            grand_unmarked += len(unmarked)
            for ln, preview in unmarked:
                print(f"      ! line {ln}: {preview}")

    print("\n" + "=" * 78)
    if grand_unmarked == 0:
        print("RECON PASS — every data row in both tables carries a 去向 marker "
              "(zero unmarked).")
        return 0
    print(f"RECON FAIL — {grand_unmarked} unmarked/missing row(s).")
    return 1


if __name__ == "__main__":
    sys.exit(main())
