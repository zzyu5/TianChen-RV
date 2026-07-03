#!/usr/bin/env python3
"""family_ledger.py — E6 [LED-1] per-family engineering ledger.

Emits, per capability family, the [LED-1] record used to measure the marginal
cost of onboarding a family:

  {code_LOC, test_LOC, table_rows, interface_touchpoints, calendar_days}

This is governance / workflow tooling and is stdlib-only: Python is tooling here,
never the compiler stack (implementation-stack red line). It reads a hand-authored
family->dirs manifest (schema/family-dirs.v1.json; interim until [F-3]/E2b lands
plugins/<family>/) plus git history; it does NOT read or touch any C++/ODS for
behavior.

LOC ([A-5] "LOC 用 cloc"): `cloc` is NOT installed in this environment, so
code_LOC is a stdlib **cloc-approximation** — physical lines that are non-blank
and not a pure `//` line, after deleting `/* ... */` block-comment spans while
PRESERVING newlines (so adjacent code lines are never merged). Validated on the
IME source set: raw wc-l = 2484, cloc-approx = ~1866 vs the [LED-1] target ~1865.
The approximation is byte-reproducible (no external tool, no version skew).

test_LOC ([A-5] tests single-listed, decision 6): via an EXPLICIT per-family
test-file manifest (raw wc-l; a lit test's `// RUN:`/`// CHECK:` directives are
load-bearing, not comments). We pin the file set rather than a path-glob (research
escalation 6). The historically "un-sourced ~659" figure IS reproducible: it is the
6 test/Dialect/IME/ dialect verifier tests (mma*.mlir/matmul.mlir), which the
ime-*.mlir glob missed. The pinned IME manifest now spans all 14 files: 8 conversion
(344) + 6 dialect (659) = 1003 raw. test_LOC is NOT the LED-1 headline (that is
code_LOC); a human may re-scope to lowering-only 344 (see manifest test_files_note).

Subcommands
-----------
report [--family NAME]   print the [LED-1] record(s) + snapshot $meta. Default:
                         all families in the manifest.
--self-test              hermetic: cloc-approx on synthetic strings, calendar-day
                         arithmetic on FIXED dates, table-row/touchpoint regex
                         counting on fixtures. No real git / clock.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

# This file lives at <repo>/.trellis/scripts/family_ledger.py
REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_JSON = REPO_ROOT / "schema" / "family-dirs.v1.json"

_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)


# --- canonical hashing (mirrors E1 check_schema_gate.py) -------------------
def canonicalize(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compute_hash(obj) -> str:
    return hashlib.sha256(canonicalize(obj).encode("utf-8")).hexdigest()


# --- cloc-approximation (PURE: operates on a string) -----------------------
def cloc_approx(text: str) -> int:
    """Count code lines: non-blank, not a pure `//` line, block comments removed.

    Block comments are removed with their content but with newlines PRESERVED, so
    that `code; /* c\\n c */ code;` keeps both physical code lines rather than
    collapsing them (this is what makes the count land on cloc's physical-line
    figure instead of undercounting).
    """
    def _keep_newlines(match):
        return "\n" * match.group(0).count("\n")

    stripped = _BLOCK_COMMENT.sub(_keep_newlines, text)
    count = 0
    for line in stripped.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("//"):
            continue
        count += 1
    return count


def raw_loc(text: str) -> int:
    """Raw physical line count (matches `wc -l` when the file ends in a newline)."""
    return text.count("\n")


# --- filesystem gathering --------------------------------------------------
def _iter_code_files(dirs):
    """Source files under the family's code dirs, excluding CMakeLists.txt."""
    for d in dirs:
        base = REPO_ROOT / d
        if not base.exists():
            continue
        for p in sorted(base.rglob("*")):
            if p.is_file() and p.name != "CMakeLists.txt":
                yield p


def gather_code_loc(dirs):
    raw = 0
    approx = 0
    files = 0
    for p in _iter_code_files(dirs):
        text = p.read_text(encoding="utf-8", errors="replace")
        raw += raw_loc(text)
        approx += cloc_approx(text)
        files += 1
    return {"raw_wc_l": raw, "cloc_approx": approx, "files": files}


def gather_test_loc(test_files):
    total = 0
    present = []
    missing = []
    for rel in test_files:
        p = REPO_ROOT / rel
        if p.exists():
            total += raw_loc(p.read_text(encoding="utf-8", errors="replace"))
            present.append(rel)
        else:
            missing.append(rel)
    return {"raw_wc_l": total, "files": len(present),
            "missing": missing, "single_listed": present}


# --- regex-anchored counts (table_rows, interface_touchpoints) -------------
def count_anchor(anchor):
    """Count regex matches (per line) in the anchor file. anchor = {file, pattern}."""
    p = REPO_ROOT / anchor["file"]
    if not p.exists():
        return {"count": None, "note": f"anchor file missing: {anchor['file']}"}
    pat = re.compile(anchor["pattern"], re.MULTILINE)
    text = p.read_text(encoding="utf-8", errors="replace")
    n = sum(1 for line in text.splitlines() if pat.search(line))
    return {"count": n, "file": anchor["file"], "pattern": anchor["pattern"]}


# --- git plumbing (only the live `report` path) ----------------------------
def _git(args):
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def calendar_days(dirs):
    """git first->last commit-date span (in days) over the family's code dirs."""
    out = _git(["log", "--format=%ad", "--date=short", "--", *dirs])
    dates = sorted(d for d in out.splitlines() if d.strip())
    if not dates:
        return {"days": None, "first": None, "last": None}
    first, last = dates[0], dates[-1]
    return {"days": days_between(first, last), "first": first, "last": last}


def days_between(first: str, last: str) -> int:
    """Whole days between two YYYY-MM-DD dates (last - first), no external deps."""
    import datetime
    f = datetime.date.fromisoformat(first)
    l = datetime.date.fromisoformat(last)
    return (l - f).days


# --- per-family record -----------------------------------------------------
def family_record(fam):
    code = gather_code_loc(fam["code_dirs"])
    tests = gather_test_loc(fam.get("test_files", []))
    table = count_anchor(fam["table_anchor"])
    touch = count_anchor(fam["interface_anchor"])
    cal = calendar_days(fam["code_dirs"])
    rec = {
        "family": fam["family"],
        "code_LOC": {
            "cloc_approx": code["cloc_approx"],
            "raw_wc_l": code["raw_wc_l"],
            "files": code["files"],
            "note": "cloc-approx (cloc not installed); EXCL tests + CMakeLists.txt",
        },
        "test_LOC": {
            "raw_wc_l": tests["raw_wc_l"],
            "files": tests["files"],
            "single_listed": tests["single_listed"],
            "source": ("explicit test-file manifest (decision 6); raw wc-l because "
                       "lit RUN/CHECK directives are load-bearing"),
        },
        "table_rows": table,
        "interface_touchpoints": touch,
        "calendar_days": cal,
    }
    if tests["missing"]:
        rec["test_LOC"]["missing_from_manifest"] = tests["missing"]
    if fam.get("notes"):
        rec["notes"] = fam["notes"]
    return rec


def cmd_report(args) -> int:
    manifest = json.loads(MANIFEST_JSON.read_text(encoding="utf-8"))
    fams = manifest["families"]
    if args.family:
        fams = [f for f in fams if f["family"].lower() == args.family.lower()]
        if not fams:
            print(f"no family {args.family!r} in {MANIFEST_JSON.name}", file=sys.stderr)
            return 1
    report = {
        "$meta": {
            "report": "family_ledger",
            "repo_snapshot": _git(["rev-parse", "HEAD"]),
            "snapshot_commit_date": _git(["show", "-s", "--format=%cI", "HEAD"]),
            "manifest_sha256": compute_hash(manifest),
            "loc_method": "cloc-approximation (cloc absent); install cloc for [A-5] wording",
            "spec_boundary": "numbers live in the CI report / 执行总纲, never in .trellis/spec/",
        },
        "families": [family_record(f) for f in fams],
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


# --- self-test -------------------------------------------------------------
def cmd_self_test(_args) -> int:
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    # cloc-approx on synthetic strings ---------------------------------------
    src = (
        "int a = 1;\n"          # code
        "\n"                     # blank
        "   \n"                  # whitespace-only
        "// a line comment\n"    # pure line comment
        "int b = 2; // trailing\n"   # code with trailing comment -> counts
        "/* full-line block */\n"    # block comment, single line -> 0
        "/* multi\n line\n block */\n"   # 3 physical lines, all comment -> 0
        "int c = 3;\n"           # code
    )
    check("cloc-approx counts only the 3 real code lines",
          cloc_approx(src) == 3)
    check("raw_loc counts all 10 physical lines (block comment spans 3)",
          raw_loc(src) == 10)

    # block comment must PRESERVE newlines (never merge adjacent code) --------
    merged = "int x; /* c\nc */ int y;\n"
    check("block-comment removal preserves both code lines (no merge)",
          cloc_approx(merged) == 2)

    # calendar-day arithmetic on FIXED dates ---------------------------------
    check("days_between IME span 2026-06-23..2026-06-25 = 2",
          days_between("2026-06-23", "2026-06-25") == 2)
    check("days_between same day = 0",
          days_between("2026-07-03", "2026-07-03") == 0)

    # table-row / touchpoint regex counting on a fixture ---------------------
    td_fixture = "def MMAOp : X\ndef MMAUOp : Y\nsomething else\ndef MatMulOp : Z\n"
    pat = re.compile(r"^def [A-Za-z0-9_]+Op", re.MULTILINE)
    n = sum(1 for line in td_fixture.splitlines() if pat.search(line))
    check("op-def regex counts 3 op defs on the fixture", n == 3)

    header_fixture = (
        "StringRef getName() const override;\n"
        "void foo();\n"
        "Error proposeVariants(...) const override;\n"
    )
    opat = re.compile("override")
    o = sum(1 for line in header_fixture.splitlines() if opat.search(line))
    check("override regex counts 2 touchpoints on the fixture", o == 2)

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

    p_report = sub.add_parser("report", help="[LED-1] per-family engineering ledger")
    p_report.add_argument("--family", default=None,
                          help="restrict to one family (default: all in manifest)")
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
