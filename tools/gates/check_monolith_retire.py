#!/usr/bin/env python3
"""check_monolith_retire.py -- 裁决九.4 monolith retirement residual gate.

Asserts that EVERY surviving vec_dot MONOLITH block-dot op-def in the ODS is named
in schema/monolith-retire-whitelist.v1.json (either deliberately_retained forever or
pending_retirement under a named batch), and that NO retired monolith verifier/emitter
lingers as a `#if 0` dead-code tomb in the dialect/conversion .cpp. An un-whitelisted
surviving monolith, or a stale whitelist entry, or a `#if 0` verifier tomb, is the
residual this gate fails closed on ('有名有批次': a monolith may linger only if it is
NAMED and carries a retirement batch).

What is a 'vec_dot monolith'?  An ODS op-def whose description says
    Records the COMPLETE ggml `ggml_vec_dot_<fmt>_<act>` (super-)block dot-product as ONE
i.e. the WHOLE ggml vec_dot recorded as ONE opaque op. Bricks/cores
('Records the INTEGER CORE ...' / 'Records the WHOLE per-super-block body ...') and the
GEMM/GEMV/REPACK/PACK family are NOT monoliths and are out of scope.

Fail-closed in BOTH directions:
  * an ODS monolith absent from the whitelist  -> RED (unnamed dead tail);
  * a whitelist entry whose op-def is gone      -> RED (stale — move it to retired_ledger).

Usage:  python3 tools/gates/check_monolith_retire.py [--self-test] [-v]
Exit:   0 GREEN ; 1 RED (residual / whitelist violation) ; 2 setup error.
"""
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ODS = os.path.join(REPO, "include/Weft/Dialect/RVV/IR/RVVOps.td")
WHITELIST = os.path.join(REPO, "schema/monolith-retire-whitelist.v1.json")
# .cpp files where a retired monolith verifier (dialect) or emitter (conversion) could
# be left behind as a `#if 0` tomb. Scanned recursively for real preprocessor #if 0.
TOMB_SCAN_DIRS = [
    os.path.join(REPO, "lib/Dialect/RVV/IR"),
    os.path.join(REPO, "lib/Conversion/RVV"),
]

# The single load-bearing marker: unique to the 8 vec_dot monoliths (bricks say
# 'INTEGER CORE' / 'WHOLE per-super-block body'; GEMM/REPACK say 'GEMM'/'REPACKED').
MONOLITH_MARKER = "Records the COMPLETE ggml `ggml_vec_dot_"

RE_DEF = re.compile(r"^def\s+(\w+)\b")
RE_MNEMONIC = re.compile(r'WEFTRVV_Op<"([^"]+)"')
RE_IF0 = re.compile(r"^\s*#\s*if\s+0\b")
RE_IFANY = re.compile(r"^\s*#\s*if(def|ndef)?\b")
RE_ENDIF = re.compile(r"^\s*#\s*endif\b")
RE_MONO_VERIFY = re.compile(r"GgmlBlockDot\w*::verify")


# ---------------------------------------------------------------------------
# Pure classifier core (fed synthetic inputs by --self-test; real files at runtime).
# ---------------------------------------------------------------------------
def extract_ods_monoliths(text):
    """text (an ODS .td body) -> {mnemonic: op_def_name} for vec_dot monoliths only."""
    lines = text.splitlines()
    defs = [(i, m.group(1)) for i, ln in enumerate(lines) for m in [RE_DEF.match(ln)] if m]
    out = {}
    for k, (i, name) in enumerate(defs):
        j = defs[k + 1][0] if k + 1 < len(defs) else len(lines)
        seg = "\n".join(lines[i:j])
        if MONOLITH_MARKER in seg:
            mm = RE_MNEMONIC.search(seg)
            if mm:
                out[mm.group(1)] = name
    return out


def find_verifier_tombs(text, fname):
    """Return ['fname:lineno', ...] for every `#if 0` block whose body holds a
    retired monolith verify() (GgmlBlockDot*::verify). // comments never match
    (RE_IF0 requires a real preprocessor directive at line start)."""
    lines = text.splitlines()
    n = len(lines)
    findings = []
    idx = 0
    while idx < n:
        if RE_IF0.match(lines[idx]):
            depth, j, body = 1, idx + 1, []
            while j < n and depth > 0:
                if RE_IFANY.match(lines[j]):
                    depth += 1
                elif RE_ENDIF.match(lines[j]):
                    depth -= 1
                if depth > 0:
                    body.append(lines[j])
                j += 1
            btext = "\n".join(body)
            if RE_MONO_VERIFY.search(btext) or ("BlockDot" in btext and "::verify" in btext):
                findings.append(f"{fname}:{idx + 1}")
            idx = j
        else:
            idx += 1
    return findings


def validate_whitelist_shape(wl):
    """Structural validation of the whitelist doc. Returns (errors, live_entries)."""
    errors = []
    for key in ("deliberately_retained", "pending_retirement"):
        if not isinstance(wl.get(key), list):
            errors.append(f"whitelist: missing/!list top-level '{key}'")
    live = []
    for e in wl.get("deliberately_retained", []) or []:
        for f in ("mnemonic", "op_def", "format", "reason"):
            if not e.get(f):
                errors.append(f"deliberately_retained[{e.get('mnemonic','?')}]: missing '{f}'")
        live.append(("deliberately_retained", e))
    for e in wl.get("pending_retirement", []) or []:
        for f in ("mnemonic", "op_def", "format", "reason", "batch"):
            if not e.get(f):
                errors.append(f"pending_retirement[{e.get('mnemonic','?')}]: missing '{f}'")
        live.append(("pending_retirement", e))
    # duplicate mnemonic across the two live lists
    seen = {}
    for _, e in live:
        m = e.get("mnemonic")
        if m in seen:
            errors.append(f"whitelist: duplicate mnemonic '{m}'")
        seen[m] = True
    return errors, live


def evaluate(ods_monoliths, wl, tombs):
    """Core verdict. ods_monoliths: {mnemonic: op_def}. wl: parsed whitelist dict.
    tombs: list of 'file:line'. Returns (ok, errors)."""
    errors, live = validate_whitelist_shape(wl)
    wl_map = {e["mnemonic"]: e for _, e in live if e.get("mnemonic")}
    W = set(wl_map)
    A = set(ods_monoliths)

    for m in sorted(A - W):
        errors.append(
            f"RESIDUAL: ODS vec_dot monolith '{m}' (op {ods_monoliths[m]}) is NOT "
            f"whitelisted -> unnamed dead tail. Add it to "
            f"schema/monolith-retire-whitelist.v1.json (deliberately_retained or "
            f"pending_retirement + batch), or retire the op-def.")
    for m in sorted(W - A):
        errors.append(
            f"STALE-WHITELIST: '{m}' (op {wl_map[m].get('op_def')}) is whitelisted but "
            f"no longer an ODS monolith -> the op was retired; MOVE it to retired_ledger.")
    # op_def name-drift cross-check for the intersection.
    for m in sorted(A & W):
        want = wl_map[m].get("op_def")
        got = ods_monoliths[m]
        if want != got:
            errors.append(f"OP-DEF DRIFT: mnemonic '{m}' is op '{got}' in ODS but "
                          f"whitelist says '{want}'.")
    for t in tombs:
        errors.append(f"VERIFIER-TOMB: retired monolith verify() left as a `#if 0` "
                      f"dead-code block at {t} -> plain-delete it (git history is the archive).")
    return (len(errors) == 0), errors


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def run_real(verbose):
    for p in (ODS, WHITELIST):
        if not os.path.isfile(p):
            print(f"[setup-error] missing {p}", file=sys.stderr)
            return 2
    try:
        ods_text = open(ODS, encoding="utf-8").read()
        wl = json.load(open(WHITELIST, encoding="utf-8"))
    except Exception as ex:  # noqa: BLE001
        print(f"[setup-error] {ex}", file=sys.stderr)
        return 2

    ods_monoliths = extract_ods_monoliths(ods_text)
    tombs = []
    for d in TOMB_SCAN_DIRS:
        if not os.path.isdir(d):
            continue
        for root, _, files in os.walk(d):
            for fn in files:
                if fn.endswith((".cpp", ".cc", ".h")):
                    fp = os.path.join(root, fn)
                    try:
                        tombs += find_verifier_tombs(open(fp, encoding="utf-8").read(),
                                                     os.path.relpath(fp, REPO))
                    except Exception:  # noqa: BLE001
                        pass

    ok, errors = evaluate(ods_monoliths, wl, tombs)

    if verbose or not ok:
        det = wl.get("deliberately_retained", [])
        pen = wl.get("pending_retirement", [])
        print(f"[monolith-retire] ODS vec_dot monoliths: {len(ods_monoliths)}  "
              f"(whitelisted: {len(det)} retained + {len(pen)} pending)")
        for m in sorted(ods_monoliths):
            cls = "?"
            for c, lst in (("retained", det), ("pending", pen)):
                if any(e.get("mnemonic") == m for e in lst):
                    cls = c
            print(f"    {m:24s} {ods_monoliths[m]:28s} [{cls}]")
        print(f"[monolith-retire] verifier `#if 0` tombs: {len(tombs)}")

    if ok:
        print("[monolith-retire] GREEN: every surviving vec_dot monolith is named + "
              "batched; no verifier tombs.")
        return 0
    print("[monolith-retire] RED:", file=sys.stderr)
    for e in errors:
        print(f"  ::error:: {e}", file=sys.stderr)
    return 1


def run_self_test():
    """Prove the classifier DISCRIMINATES before it judges the committed tree:
    matching -> GREEN, each violation class -> RED."""
    fails = []

    def check(label, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            fails.append(label)

    # --- ODS monolith detector isolates monoliths from bricks/GEMM ---
    ods = (
        'def GgmlBlockDotFooQ80Op\n    : WEFTRVV_Op<"foo_q8_0_block_dot", []> {\n'
        '  let description = [{ Records the COMPLETE ggml `ggml_vec_dot_foo_q8_0` '
        'block dot-product as ONE op. }];\n}\n'
        'def GgmlBlockDotFooCoreOp\n    : WEFTRVV_Op<"foo_q8_0_core", []> {\n'
        '  let description = [{ Records the INTEGER CORE of the ggml '
        '`ggml_vec_dot_foo_q8_0`. }];\n}\n'
        'def GgmlGemmFooOp\n    : WEFTRVV_Op<"foo_gemm", []> {\n'
        '  let description = [{ Records the COMPLETE ggml Foo GEMM as ONE. }];\n}\n')
    mono = extract_ods_monoliths(ods)
    check("detector picks the monolith only", mono == {"foo_q8_0_block_dot": "GgmlBlockDotFooQ80Op"})

    wl_ok = {
        "deliberately_retained": [
            {"mnemonic": "foo_q8_0_block_dot", "op_def": "GgmlBlockDotFooQ80Op",
             "format": "foo", "reason": "control"}],
        "pending_retirement": [],
    }
    ok, _ = evaluate(mono, wl_ok, [])
    check("matching whitelist -> GREEN", ok)

    # --- unwhitelisted residual -> RED ---
    ok, errs = evaluate(mono, {"deliberately_retained": [], "pending_retirement": []}, [])
    check("unwhitelisted monolith -> RED", (not ok) and any("RESIDUAL" in e for e in errs))

    # --- stale whitelist entry (op gone from ODS) -> RED ---
    wl_stale = {
        "deliberately_retained": [
            {"mnemonic": "foo_q8_0_block_dot", "op_def": "GgmlBlockDotFooQ80Op",
             "format": "foo", "reason": "control"},
            {"mnemonic": "ghost_block_dot", "op_def": "GhostOp",
             "format": "ghost", "reason": "x"}],
        "pending_retirement": [],
    }
    ok, errs = evaluate(mono, wl_stale, [])
    check("stale whitelist entry -> RED", (not ok) and any("STALE-WHITELIST" in e for e in errs))

    # --- pending_retirement missing batch -> RED (shape) ---
    wl_nobatch = {
        "deliberately_retained": [],
        "pending_retirement": [
            {"mnemonic": "foo_q8_0_block_dot", "op_def": "GgmlBlockDotFooQ80Op",
             "format": "foo", "reason": "flipped"}],  # no 'batch'
    }
    ok, errs = evaluate(mono, wl_nobatch, [])
    check("pending w/o batch -> RED", (not ok) and any("missing 'batch'" in e for e in errs))

    # --- verifier #if 0 tomb detection ---
    tomb_cpp = ("int a;\n#if 0\nmlir::LogicalResult GgmlBlockDotFooQ80Op::verify() {\n"
                "  return success();\n}\n#endif // retired\nint b;\n")
    check("verifier tomb detected", find_verifier_tombs(tomb_cpp, "x.cpp") == ["x.cpp:2"])

    # --- a // comment mentioning #if 0 is NOT a tomb ---
    comment_cpp = "// body was kept as an #if 0 dead-code tomb (prose)\nint c;\n"
    check("// comment '#if 0' not a tomb", find_verifier_tombs(comment_cpp, "y.cpp") == [])

    # --- a legit #if 0 with no verify inside is NOT flagged ---
    benign = "#if 0\nint unused = 1;\n#endif\n"
    check("benign #if 0 not flagged", find_verifier_tombs(benign, "z.cpp") == [])

    # --- tomb fed through evaluate -> RED ---
    ok, errs = evaluate(mono, wl_ok, ["x.cpp:2"])
    check("tomb -> evaluate RED", (not ok) and any("VERIFIER-TOMB" in e for e in errs))

    if fails:
        print(f"[monolith-retire --self-test] RED: {len(fails)} discrimination(s) failed")
        return 2
    print("[monolith-retire --self-test] GREEN: classifier discriminates all cases")
    return 0


def main(argv):
    verbose = "-v" in argv or "--verbose" in argv
    if "--self-test" in argv:
        return run_self_test()
    return run_real(verbose)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
