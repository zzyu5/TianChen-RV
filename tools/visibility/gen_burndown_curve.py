#!/usr/bin/env python3
"""gen_burndown_curve.py -- T7 burn-down double curve (visibility pack, line E).

Renders the paper Fig.1 double series into experiments/visibility/T7-burndown.md:

  * C_construct (STRONG coverage) climbing, read from the FLIP commits (each such
    commit's subject carries a `C_construct N->M` transition), and
  * hand-written source LOC per flip, read from the commit diffstat over
    lib/ + include/ (add - del). This diffstat net IS the project's "delta 手写LOC"
    convention -- validated against the curated commit-subject figures (e.g. q4_K
    -196 = 243 add - 439 del; q5_K -374 = 152 - 526). Retirement flips burn LOC
    down; first-of-a-new-family flips pay a construction cost (the C2 marginal-
    cost story is legible directly in the two series).

Governance / workflow tooling, stdlib-only (Python is tooling, never the compiler
stack). Reads git history only; touches no C++/ODS.

DETERMINISTIC: output is a pure function of the repo's commit history (flip
commits + their diffstats). No wall clock. It changes only when a new flip commit
lands (or history is rewritten). Data points are labelled with the short hash.

Subcommands
-----------
render [--out PATH]   write the T7 markdown (default: experiments/visibility/
                      T7-burndown.md under the repo root).
--check               regenerate to memory, compare with committed artifact;
                      exit non-zero on drift.
--self-test           hermetic: synthetic flip list; asserts the transition
                      parse, the cumulative-LOC arithmetic, the chain-contiguity
                      check, and idempotent rendering. No real git.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# This file lives at <repo>/tools/visibility/gen_burndown_curve.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = REPO_ROOT / "experiments" / "visibility" / "T7-burndown.md"

# `C_construct 13→14` / `C_construct 13->14` (arrow may be unicode or ascii).
FLIP_RE = re.compile(r"C_construct\s*(\d+)\s*(?:→|->)\s*(\d+)")
# curated `Δ手写LOC +388` / `Δ手写LOC 净 −374` (minus may be unicode U+2212).
CURATED_RE = re.compile(r"Δ手写LOC\s*(?:净\s*)?([+\-−]?\s*\d+)")

# LOC-delta diffstat is scoped to the hand-maintained compiler source tree.
SRC_PREFIXES = ["lib", "include"]


# --- git plumbing ----------------------------------------------------------
def _git(args):
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True, capture_output=True, text=True,
    ).stdout


def _norm_int(s):
    return int(s.replace("−", "-").replace(" ", ""))


def collect_flip_commits():
    """Walk all refs; return flip records sorted by the C_construct 'to' value.

    Each record: {short, full, date, c_from, c_to, subject, add, dele, net,
    curated}. `net = add - dele` over lib/+include/. `curated` is the subject's
    delta 手写LOC figure if present (else None).
    """
    out = _git(["log", "--all", "--no-merges",
                "--format=%H%x1f%h%x1f%cI%x1f%s"])
    seen = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        full, short, date, subject = line.split("\x1f")
        m = FLIP_RE.search(subject)
        if not m:
            continue
        c_from, c_to = int(m.group(1)), int(m.group(2))
        # dedup by transition; keep the earliest commit date for a given (from,to).
        k = (c_from, c_to)
        rec = {"full": full, "short": short, "date": date, "c_from": c_from,
               "c_to": c_to, "subject": subject}
        cur = seen.get(k)
        if cur is None or date < cur["date"]:
            seen[k] = rec

    records = sorted(seen.values(), key=lambda r: (r["c_to"], r["date"]))
    for r in records:
        add, dele = diffstat_src(r["full"])
        r["add"], r["dele"], r["net"] = add, dele, add - dele
        cm = CURATED_RE.search(r["subject"])
        r["curated"] = _norm_int(cm.group(1)) if cm else None
    return records


def diffstat_src(full_hash):
    """Sum (added, deleted) lines over lib/+include/ for one commit."""
    args = ["show", "--numstat", "--format=", full_hash, "--"] + SRC_PREFIXES
    out = _git(args)
    add = dele = 0
    for line in out.splitlines():
        parts = line.split("\t")
        if len(parts) != 3:
            continue
        a, d, _path = parts
        if a == "-" or d == "-":  # binary file
            continue
        add += int(a)
        dele += int(d)
    return add, dele


# --- pure analysis ---------------------------------------------------------
def analyze(records):
    """Attach cumulative LOC + a contiguity flag. Pure over the record list."""
    cum = 0
    prev_to = None
    gaps = []
    rows = []
    for r in records:
        if prev_to is not None and r["c_from"] != prev_to:
            gaps.append((prev_to, r["c_from"], r["short"]))
        cum += r["net"]
        rows.append({**r, "cum": cum})
        prev_to = r["c_to"]
    return rows, gaps


# --- pure render -----------------------------------------------------------
def _signed_bar(value, unit, ch_pos="#", ch_neg="-"):
    """Integer-deterministic proportional bar; sign-aware."""
    mag = abs(value) // unit
    return (ch_neg if value < 0 else ch_pos) * mag


def render_markdown(rows, gaps, head_sha):
    lines = []
    lines.append("# T7 -- Burn-down Double Curve (auto-generated)")
    lines.append("")
    lines.append("> DATA CELL (visibility line E). Generated by "
                 "`tools/visibility/gen_burndown_curve.py` from the repo's FLIP "
                 "commit history. Do NOT hand-edit; rerun the generator. "
                 "Deterministic: pure function of commit history, no wall clock.")
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(f"- HEAD at render : `{head_sha}`")
    lines.append(f"- flip commits   : {len(rows)}")
    lines.append("- coverage series: `C_construct N->M` parsed from each flip "
                 "commit subject.")
    lines.append("- LOC series     : commit diffstat (add - del) over `lib/` + "
                 "`include/`. This equals the project's `Δ手写LOC` convention "
                 "(cross-checked against curated commit-subject figures below).")
    lines.append("")

    if rows:
        first, last = rows[0], rows[-1]
        span_net = last["cum"]
        burned = -sum(r["net"] for r in rows if r["net"] < 0)   # LOC removed (>=0)
        added = sum(r["net"] for r in rows if r["net"] > 0)     # LOC added
        n_retire = sum(1 for r in rows if r["net"] < 0)
        n_green = sum(1 for r in rows if r["net"] > 0)
        lines.append("## Headline")
        lines.append("")
        lines.append(f"- C_construct climbed **{first['c_from']} -> "
                     f"{last['c_to']}** across {len(rows)} flips "
                     f"(`{first['short']}` .. `{last['short']}`).")
        sign = "+" if span_net >= 0 else ""
        lines.append(f"- cumulative hand-written source LOC over the campaign: "
                     f"**{sign}{span_net}** (lib/+include/ net).")
        lines.append(f"- burn-down vs construction split: **{n_retire} retirement "
                     f"flips burned down {burned} hand-written LOC** (the "
                     f"maturity/燃减 component); {n_green} greenfield flips added "
                     f"{added} (first-of-family construction cost, still "
                     f"greenfield-heavy). Net = {added} - {burned} = "
                     f"{sign}{span_net}.")
        lines.append("")

    # --- data table --------------------------------------------------------
    lines.append("## Flip series")
    lines.append("")
    lines.append("| # | commit | date | C_construct | src add/del | src net LOC | "
                 "cum src LOC | curated Δ手写LOC |")
    lines.append("|---:|---|---|---|---:|---:|---:|---:|")
    for i, r in enumerate(rows, start=1):
        cur = "" if r["curated"] is None else f"{r['curated']:+d}"
        # a mismatch between curated subject figure and computed diffstat is worth
        # surfacing (informational, never fatal).
        if r["curated"] is not None and r["curated"] != r["net"]:
            cur += f" (!=net {r['net']:+d})"
        lines.append(
            f"| {i} | `{r['short']}` | {r['date'][:10]} | "
            f"{r['c_from']}->{r['c_to']} | {r['add']}/{r['dele']} | "
            f"{r['net']:+d} | {r['cum']:+d} | {cur} |")
    lines.append("")

    if gaps:
        lines.append("> NOTE: non-contiguous C_construct chain (a flip's `from` "
                     "did not match the previous `to`). Earlier flips (pre-campaign "
                     "epochs) may live on other branches / use older subject prose:")
        for prev_to, cur_from, short in gaps:
            lines.append(f">  - gap before `{short}`: previous to={prev_to}, "
                         f"this from={cur_from}")
        lines.append("")

    # --- ascii double curve ------------------------------------------------
    lines.append("## ASCII double curve")
    lines.append("")
    lines.append("```")
    lines.append("C_construct  (# = 1 constructed key)          |  cum hand-written src LOC (# = +50, - = -50)")
    lines.append("-" * 96)
    for r in rows:
        cbar = "#" * r["c_to"]
        lbar = _signed_bar(r["cum"], 50)
        lines.append(f"{r['short']}  C={r['c_to']:>2} {cbar:<26} | "
                     f"{r['cum']:>+6} {lbar}")
    lines.append("```")
    lines.append("")

    # --- interpretation ----------------------------------------------------
    lines.append("## Reading the two series (C2 marginal-cost)")
    lines.append("")
    lines.append("- A flip with **negative src net LOC** RETIRED an opaque monolith "
                 "hand-helper and replaced it with a more compact constructed region "
                 "(the maturity/burn-down thesis: more coverage, less hand code).")
    lines.append("- A flip with **positive src net LOC** is typically a "
                 "first-of-a-new-family entry that pays a one-time construction cost "
                 "(new brick + body code-move) before there is a sibling to reuse; "
                 "the very next sibling in that family then flips cheaply "
                 "(the C2 marginal-cost payoff).")
    lines.append("- This artifact is coverage/representation accounting only. It "
                 "makes NO numerical or perf claim; those live in T3/T6 under their "
                 "certification status.")
    lines.append("")
    return "\n".join(lines) + "\n"


# --- live paths ------------------------------------------------------------
def build_render():
    records = collect_flip_commits()
    rows, gaps = analyze(records)
    head = _git(["rev-parse", "HEAD"]).strip()
    return render_markdown(rows, gaps, head)


def cmd_render(args) -> int:
    text = build_render()
    out = Path(args.out) if args.out else DEFAULT_OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"wrote {out}")
    return 0


def cmd_check(_args) -> int:
    text = build_render()
    if not DEFAULT_OUT.exists():
        print(f"DRIFT: {DEFAULT_OUT} missing (never rendered).", file=sys.stderr)
        return 1
    committed = DEFAULT_OUT.read_text(encoding="utf-8")
    if committed == text:
        print(f"OK: {DEFAULT_OUT.name} matches the flip history (no drift).")
        return 0
    print(f"DRIFT: {DEFAULT_OUT} is stale vs the flip history. "
          f"Run: python3 tools/visibility/gen_burndown_curve.py render",
          file=sys.stderr)
    return 1


# --- self-test -------------------------------------------------------------
def cmd_self_test(_args) -> int:
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    # transition + curated parse
    m = FLIP_RE.search("q8_0 flip C_construct 3→4 (foo)")
    check("arrow transition parses 3->4", m and m.group(1) == "3" and m.group(2) == "4")
    m2 = FLIP_RE.search("ascii C_construct 17->18 finale")
    check("ascii transition parses 17->18", m2 and m2.group(2) == "18")
    cm = CURATED_RE.search("... Δ手写LOC 净 −374 ...")
    check("curated unicode-minus parses to -374", cm and _norm_int(cm.group(1)) == -374)
    cm2 = CURATED_RE.search("... Δ手写LOC +388 ...")
    check("curated plus parses to 388", cm2 and _norm_int(cm2.group(1)) == 388)

    # analyze(): cumulative + contiguity, over a synthetic contiguous chain
    recs = [
        {"short": "aaa", "full": "aaa", "date": "2026-01-01", "c_from": 3,
         "c_to": 4, "subject": "s", "add": 300, "dele": 4, "net": 296, "curated": None},
        {"short": "bbb", "full": "bbb", "date": "2026-01-02", "c_from": 4,
         "c_to": 5, "subject": "s", "add": 10, "dele": 210, "net": -200, "curated": -200},
    ]
    rows, gaps = analyze(recs)
    check("cumulative LOC = 296 then 96", rows[0]["cum"] == 296 and rows[1]["cum"] == 96)
    check("contiguous chain has no gaps", gaps == [])

    # a gap is detected
    recs_gap = [
        {"short": "a", "full": "a", "date": "2026-01-01", "c_from": 3, "c_to": 4,
         "subject": "s", "add": 1, "dele": 1, "net": 0, "curated": None},
        {"short": "b", "full": "b", "date": "2026-01-02", "c_from": 10, "c_to": 11,
         "subject": "s", "add": 1, "dele": 1, "net": 0, "curated": None},
    ]
    _rows2, gaps2 = analyze(recs_gap)
    check("gap detected (to=4 then from=10)", len(gaps2) == 1 and gaps2[0][:2] == (4, 10))

    # signed bar arithmetic
    check("signed bar: +296 -> 5 '#' at unit 50", _signed_bar(296, 50) == "#" * 5)
    check("signed bar: -200 -> 4 '-' at unit 50", _signed_bar(-200, 50) == "-" * 4)

    # idempotent render
    t1 = render_markdown(rows, gaps, "HEADSHA")
    t2 = render_markdown(rows, gaps, "HEADSHA")
    check("render idempotent (two renders byte-identical)", t1 == t2)
    check("short hash labels the data point", "aaa" in t1 and "bbb" in t1)

    ok = True
    for name, passed in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(p for _, p in results)}/{len(results)})")
    return 0 if ok else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check", action="store_true")
    sub = parser.add_subparsers(dest="cmd")
    p_render = sub.add_parser("render", help="write the T7 markdown")
    p_render.add_argument("--out", default=None)
    p_render.set_defaults(func=cmd_render)

    args = parser.parse_args(argv)
    if args.self_test:
        return cmd_self_test(args)
    if args.check:
        return cmd_check(args)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
