#!/usr/bin/env python3
"""check_retired_index.py -- 判定书 §3 轴B 裁决③ RETIRED-INDEX gate (retired-index-gate).

Fails closed unless the machine-generated single query point
schema/retired-index.generated.json satisfies ALL of:

  (1) FRESH: it byte-equals the in-memory rebuild by gen_retired_index.build_index (nobody
      edited a source ledger -- .td RETIRED NOTE / monolith-retire-whitelist retired_ledger /
      emit-bypass-whitelist retired_ledger -- without regenerating; nobody hand-edited it);
  (2) FOUR REQUIREMENTS non-empty on EVERY entry: format / op / retirement_basis (退役依据) /
      alternative_path (替代路径) / restore_ref (复原指针);
  (3) VEC_DOT COVERAGE: the set of RETIRED vec_dot monolith formats -- derived INDEPENDENTLY
      as {vec_dot rows in schema/coverage-sixstate.v1.json} MINUS {surviving ODS monolith
      formats in schema/monolith-retire-whitelist.v1.json (deliberately_retained +
      pending_retirement)} -- EQUALS the set of vec_dot-axis formats in the index. A retired
      monolith missing from the index (unindexed dead tail) OR an indexed vec_dot format that
      is NOT actually retired (stale) fails closed.

The coverage source (coverage-sixstate) is INDEPENDENT of the index's own sources, so a
future monolith retired without being indexed is caught (its coverage row survives the
retirement; the diff flags it). Stdlib-Python only.

Usage:  python3 tools/gates/check_retired_index.py [--self-test] [-v]
Exit:   0 GREEN ; 1 RED (drift / four-req / coverage) ; 2 setup error.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_retired_index as gen  # noqa: E402

REPO = gen.REPO
INDEX = gen.OUT_DEFAULT
COVERAGE = os.path.join(REPO, "schema/coverage-sixstate.v1.json")
MONOLITH_WL = gen.MONOLITH_WL

FOUR_REQ = ("format", "op", "retirement_basis", "alternative_path", "restore_ref")


# ---------------------------------------------------------------------------
# Pure verdict cores (fed synthetic inputs by --self-test).
# ---------------------------------------------------------------------------
def check_freshness(committed, regenerated):
    """committed / regenerated are parsed index dicts. Drift -> error."""
    if committed == regenerated:
        return []
    errs = []
    ce = {(e["axis"], e["format"]): e for e in committed.get("entries", [])}
    re_ = {(e["axis"], e["format"]): e for e in regenerated.get("entries", [])}
    for k in sorted(set(re_) - set(ce)):
        errs.append(f"STALE-INDEX: entry {k} present in the rebuild but MISSING from the "
                    f"committed index -> regenerate (python3 tools/gates/gen_retired_index.py).")
    for k in sorted(set(ce) - set(re_)):
        errs.append(f"STALE-INDEX: entry {k} committed but GONE from the rebuild -> "
                    f"regenerate (python3 tools/gates/gen_retired_index.py).")
    for k in sorted(set(ce) & set(re_)):
        if ce[k] != re_[k]:
            errs.append(f"STALE-INDEX: entry {k} differs from the rebuild -> regenerate.")
    if not errs:  # $meta differs
        errs.append("STALE-INDEX: $meta differs from the rebuild -> regenerate "
                    "(python3 tools/gates/gen_retired_index.py).")
    return errs


def check_four_requirements(entries):
    errs = []
    for e in entries:
        tag = f"{e.get('axis','?')}/{e.get('format','?')}"
        for r in FOUR_REQ:
            if not (e.get(r) or "").strip():
                errs.append(f"FOUR-REQ-EMPTY: index entry {tag} has empty '{r}' "
                            f"(四要件 must be non-empty).")
    return errs


def check_coverage(index_vec_formats, expected_retired):
    """index_vec_formats / expected_retired are sets. Assert equality."""
    errs = []
    for f in sorted(expected_retired - index_vec_formats):
        errs.append(f"COVERAGE-GAP: vec_dot monolith '{f}' is RETIRED (in coverage-sixstate, "
                    f"not a surviving ODS monolith) but is NOT in the RETIRED-INDEX -> add a "
                    f".td RETIRED NOTE or a monolith-retire-whitelist retired_ledger entry.")
    for f in sorted(index_vec_formats - expected_retired):
        errs.append(f"COVERAGE-STALE: RETIRED-INDEX lists vec_dot '{f}' as retired but it is "
                    f"NOT in the retired set (either a surviving ODS monolith or absent from "
                    f"coverage-sixstate) -> the index/source is stale.")
    return errs


def expected_retired_from(coverage, whitelist):
    """{vec_dot rows in coverage} - {surviving monolith formats in whitelist}."""
    vec = {s["format"] for s in coverage.get("states", []) if s.get("op") == "vec_dot"}
    surviving = {e.get("format") for e in whitelist.get("deliberately_retained", []) or []}
    surviving |= {e.get("format") for e in whitelist.get("pending_retirement", []) or []}
    return vec - surviving


def evaluate(committed, regenerated, coverage, whitelist):
    errs = []
    errs += check_freshness(committed, regenerated)
    errs += check_four_requirements(committed.get("entries", []))
    idx_vec = {e["format"] for e in committed.get("entries", []) if e.get("axis") == "vec_dot"}
    errs += check_coverage(idx_vec, expected_retired_from(coverage, whitelist))
    return (len(errs) == 0), errs


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_real(verbose):
    for p in (INDEX, COVERAGE, MONOLITH_WL):
        if not os.path.isfile(p):
            print(f"[setup-error] missing {p}", file=sys.stderr)
            return 2
    try:
        committed = json.load(open(INDEX, encoding="utf-8"))
        regenerated = gen.build_index(REPO)
        coverage = json.load(open(COVERAGE, encoding="utf-8"))
        whitelist = json.load(open(MONOLITH_WL, encoding="utf-8"))
    except Exception as ex:  # noqa: BLE001
        print(f"[setup-error] {ex}", file=sys.stderr)
        return 2

    ok, errors = evaluate(committed, regenerated, coverage, whitelist)

    if verbose or not ok:
        ents = committed.get("entries", [])
        vec = sorted(e["format"] for e in ents if e.get("axis") == "vec_dot")
        gem = sorted(e["format"] for e in ents if e.get("axis") == "gemm_tile")
        exp = sorted(expected_retired_from(coverage, whitelist))
        print(f"[retired-index] entries: {len(ents)} "
              f"({len(vec)} vec_dot monolith + {len(gem)} gemm_tile repack)")
        print(f"[retired-index] vec_dot indexed : {vec}")
        print(f"[retired-index] vec_dot expected: {exp}")

    if ok:
        print("[retired-index] GREEN: index fresh; four requirements non-empty; every "
              "retired vec_dot monolith indexed.")
        return 0
    print("[retired-index] RED:", file=sys.stderr)
    for e in errors:
        print(f"  ::error:: {e}", file=sys.stderr)
    return 1


def run_self_test():
    """Prove the gate DISCRIMINATES: matching -> GREEN, each violation class -> RED."""
    fails = []

    def check(label, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            fails.append(label)

    def entry(axis, fmt):
        return {"axis": axis, "format": fmt, "op": f"Op{fmt}",
                "retirement_basis": "front door constructs typed body instead",
                "alternative_path": "createTyped...LoopChain", "restore_ref": "git abc123"}

    idx = {"$meta": {"schema": "retired-index.generated.v1"},
           "entries": [entry("vec_dot", "aa"), entry("vec_dot", "bb"),
                       entry("gemm_tile", "aa")]}
    coverage = {"states": [{"op": "vec_dot", "format": "aa"},
                           {"op": "vec_dot", "format": "bb"},
                           {"op": "vec_dot", "format": "live"},
                           {"op": "gemm_tile", "format": "aa"}]}
    wl = {"deliberately_retained": [{"format": "live"}], "pending_retirement": []}

    # matching -> GREEN
    ok, _ = evaluate(idx, idx, coverage, wl)
    check("fresh + four-req + coverage-equal -> GREEN", ok)

    # drift (committed missing an entry the rebuild has) -> RED
    stale = {"$meta": idx["$meta"], "entries": idx["entries"][:1] + [idx["entries"][2]]}
    ok, errs = evaluate(stale, idx, coverage, wl)
    check("stale index (missing entry) -> RED", (not ok) and any("STALE-INDEX" in e for e in errs))

    # empty four-req field -> RED
    bad = json.loads(json.dumps(idx))
    bad["entries"][0]["restore_ref"] = ""
    ok, errs = evaluate(bad, bad, coverage, wl)
    check("empty restore_ref -> RED", (not ok) and any("FOUR-REQ-EMPTY" in e for e in errs))

    # coverage gap: 'bb' retired but not indexed -> RED
    idx_gap = {"$meta": idx["$meta"], "entries": [entry("vec_dot", "aa"),
                                                  entry("gemm_tile", "aa")]}
    ok, errs = evaluate(idx_gap, idx_gap, coverage, wl)
    check("retired-but-unindexed -> RED (coverage gap)",
          (not ok) and any("COVERAGE-GAP" in e for e in errs))

    # coverage stale: index lists a surviving monolith 'live' as retired -> RED
    idx_stale = {"$meta": idx["$meta"],
                 "entries": idx["entries"] + [entry("vec_dot", "live")]}
    ok, errs = evaluate(idx_stale, idx_stale, coverage, wl)
    check("index lists surviving monolith as retired -> RED (coverage stale)",
          (not ok) and any("COVERAGE-STALE" in e for e in errs))

    if fails:
        print(f"[retired-index --self-test] RED: {len(fails)} discrimination(s) failed")
        return 2
    print("[retired-index --self-test] GREEN: gate discriminates all cases")
    return 0


def main(argv):
    verbose = "-v" in argv or "--verbose" in argv
    if "--self-test" in argv:
        return run_self_test()
    return run_real(verbose)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
