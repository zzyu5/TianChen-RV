#!/usr/bin/env python3
"""check_frontdoor_provenance.py -- G3 裁决一.3 [F-EMIT] direct-emitter bypass gate.

Terminal state [K-3b]: the repack front door (RVVLowerQuantContraction.cpp
lowerToRepackGemv/lowerToRepackGemm -> weft_rvv.typed_repack_gem{v,m}_loop_body) is the
SOLE emission authority and every gemm_tile is CONSTRUCTED as a typed region (the q4_0
decode/prefill precedent). A hand-written emitRepackGem{v,m}<fmt> wired into the production
dispatch table kBlockDotKernels emits the whole repacked GEVM/GEMM body BY HAND, BYPASSING
that front door -- it is TRANSITIONAL SCAFFOLDING (feeds C_dispatch, never C_construct).

This gate asserts, over the ACTUAL production dispatch path:
  * [F-EMIT] every &VariantToEmitCFunc::emitRepackGem* wired in kBlockDotKernels is NAMED in
    schema/emit-bypass-whitelist.v1.json -> an un-whitelisted bypass is RED;
  * [shrink/sync] every whitelisted emitter symbol is still wired in kBlockDotKernels -> a
    stale entry (the tile retired to the front door) is RED (MOVE it to retired_ledger);
  * [ratchet] baseline_count == len(entries) -> a whitelist grown without a visible
    baseline bump is RED (baseline_count only decreases across git history = shrink-only);
  * [六态] every whitelisted format's gemm_tile rvv cell in schema/coverage-sixstate.v1.json
    is state != 'constructed' (a bypass may NOT claim strong construction, [L-8]) and carries
    provenance='direct-emitter (transitional scaffolding)' + an integer retirement_batch;
  * [六态-reverse] every sixstate gemm_tile rvv cell that carries the direct-emitter
    provenance marker is whitelisted.

The typed-region front-door path (isTypedRepackGem{v,m}LoopBody -> emitTypedRepackGem{v,m}
LoopBody, q4_0 only) is NOT a bypass and is deliberately absent from the whitelist.

Usage:  python3 tools/gates/check_frontdoor_provenance.py [--self-test] [-v]
Exit:   0 GREEN ; 1 RED (bypass / provenance / ratchet violation) ; 2 setup error.
"""
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DISPATCH = os.path.join(REPO, "lib/Conversion/RVV/RVVToEmitC.cpp")
WHITELIST = os.path.join(REPO, "schema/emit-bypass-whitelist.v1.json")
SIXSTATE = os.path.join(REPO, "schema/coverage-sixstate.v1.json")

PROVENANCE_MARKER = "direct-emitter (transitional scaffolding)"
# The address-of-member initializer that ONLY appears inside kBlockDotKernels: a direct
# emitter is wired as `&VariantToEmitCFunc::emitRepackGem{v,m}<fmt>`. The front-door path
# is `emitTypedRepackGem{v,m}LoopBody` (has "Typed" between emit and Repack) and does NOT
# match `emitRepackGem`. Header declarations lack the `&VariantToEmitCFunc::` prefix.
RE_TABLE = re.compile(r"kBlockDotKernels\s*\[\s*\]\s*=\s*\{")
RE_BYPASS_EMITTER = re.compile(r"&VariantToEmitCFunc::(emitRepackGem[vm]\w+)")


# ---------------------------------------------------------------------------
# Pure classifier core (fed synthetic inputs by --self-test; real files at runtime).
# ---------------------------------------------------------------------------
def strip_cpp_comments(text):
    """Drop /* */ and // comments so brace-balancing + symbol extraction see only code.
    Coarse (may touch string literals), but the only consumers are a brace scan and a
    very specific &VariantToEmitCFunc::emitRepackGem* regex, so corruption is harmless."""
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", "", text)
    return text


def extract_dispatch_bypass_emitters(cpp_text):
    """cpp_text (RVVToEmitC.cpp body) -> (set(emitter_symbols), error_or_None).

    Isolates the kBlockDotKernels initializer region by brace-balanced scan (after
    comment strip) and returns every &VariantToEmitCFunc::emitRepackGem* symbol wired in
    it (the production direct-emitter bypass set)."""
    src = strip_cpp_comments(cpp_text)
    m = RE_TABLE.search(src)
    if not m:
        return set(), "kBlockDotKernels dispatch table not found in RVVToEmitC.cpp"
    start = m.end()  # just past the opening '{'
    depth, i, n = 1, start, len(src)
    while i < n and depth > 0:
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return set(), "kBlockDotKernels initializer braces did not balance"
    region = src[start:i - 1]
    return set(RE_BYPASS_EMITTER.findall(region)), None


def validate_whitelist_shape(wl):
    """Structural validation of the whitelist doc. Returns (errors, entries, baseline)."""
    errors = []
    meta = wl.get("$meta") or {}
    baseline = meta.get("baseline_count")
    if not isinstance(baseline, int):
        errors.append("whitelist: $meta.baseline_count missing or not an int")
    if meta.get("shrink_only") is not True:
        errors.append("whitelist: $meta.shrink_only must be true")
    entries = wl.get("entries")
    if not isinstance(entries, list):
        errors.append("whitelist: 'entries' missing or not a list")
        return errors, [], baseline
    seen_fmt, seen_sym = set(), set()
    for e in entries:
        fmt = e.get("format")
        tag = f"entries[{fmt or '?'}]"
        for f in ("op", "format", "engine", "provenance"):
            if not e.get(f):
                errors.append(f"{tag}: missing '{f}'")
        if e.get("op") != "gemm_tile":
            errors.append(f"{tag}: op must be 'gemm_tile'")
        if e.get("engine") != "rvv":
            errors.append(f"{tag}: engine must be 'rvv'")
        if e.get("provenance") != PROVENANCE_MARKER:
            errors.append(f"{tag}: provenance must be '{PROVENANCE_MARKER}'")
        if not isinstance(e.get("retirement_batch"), int):
            errors.append(f"{tag}: retirement_batch missing or not an int")
        ems = e.get("emitters")
        if not isinstance(ems, list) or not ems:
            errors.append(f"{tag}: 'emitters' missing or empty")
            ems = []
        if fmt in seen_fmt:
            errors.append(f"{tag}: duplicate format")
        seen_fmt.add(fmt)
        for s in ems:
            if s in seen_sym:
                errors.append(f"{tag}: duplicate emitter symbol '{s}'")
            seen_sym.add(s)
    return errors, entries, baseline


def evaluate(source_symbols, wl, sixstate_cells):
    """Core verdict.
      source_symbols: set of emitRepackGem* wired in kBlockDotKernels (production path).
      wl:             parsed whitelist dict.
      sixstate_cells: list of gemm_tile+engine=rvv cell dicts from coverage-sixstate.
    Returns (ok, errors)."""
    errors, entries, baseline = validate_whitelist_shape(wl)

    wl_symbols, wl_formats = set(), set()
    for e in entries:
        wl_formats.add(e.get("format"))
        for s in (e.get("emitters") or []):
            wl_symbols.add(s)

    # [F-EMIT] a production-wired direct emitter must be whitelisted.
    for sym in sorted(source_symbols - wl_symbols):
        errors.append(
            f"UN-WHITELISTED-BYPASS: '{sym}' is wired in kBlockDotKernels (a direct "
            f"emitter that BYPASSES the front door) but is NOT in "
            f"schema/emit-bypass-whitelist.v1.json. Add it with a retirement_batch, or "
            f"route the format through the typed_repack front door instead.")
    # [shrink/sync] a whitelisted symbol must still be wired (else the tile retired).
    for sym in sorted(wl_symbols - source_symbols):
        errors.append(
            f"STALE-WHITELIST: '{sym}' is whitelisted but no longer wired in "
            f"kBlockDotKernels -> the tile retired to the front door. MOVE its entry to "
            f"retired_ledger and lower $meta.baseline_count (旁路存量 shrinks).")
    # [ratchet] baseline_count == entry count (shrink-only: baseline only drops in history).
    if isinstance(baseline, int) and baseline != len(entries):
        errors.append(
            f"RATCHET: $meta.baseline_count={baseline} != len(entries)={len(entries)}. "
            f"The whitelist may only SHRINK: adding a bypass requires a visible "
            f"baseline_count bump; retiring one requires lowering it. Keep them equal.")

    # [六态] cross-check the sixstate provenance for each whitelisted format.
    cells_by_fmt = {c.get("format"): c for c in sixstate_cells}
    labeled_bypass_fmts = {
        c.get("format") for c in sixstate_cells
        if c.get("provenance") == PROVENANCE_MARKER
    }
    for fmt in sorted(wl_formats):
        c = cells_by_fmt.get(fmt)
        if c is None:
            errors.append(
                f"SIXSTATE-MISSING: whitelisted format '{fmt}' has no gemm_tile rvv cell "
                f"in schema/coverage-sixstate.v1.json.")
            continue
        if c.get("state") == "constructed":
            errors.append(
                f"OVER-REPORT: gemm_tile '{fmt}' (rvv) is a whitelisted DIRECT-EMITTER "
                f"bypass but is marked state='constructed' in six-state. A bypass feeds "
                f"C_dispatch, never C_construct ([L-8]); label it dispatch-wired.")
        if c.get("provenance") != PROVENANCE_MARKER:
            errors.append(
                f"PROVENANCE-MISSING: gemm_tile '{fmt}' (rvv) lacks "
                f"provenance='{PROVENANCE_MARKER}' in six-state.")
        if not isinstance(c.get("retirement_batch"), int):
            errors.append(
                f"BATCH-MISSING: gemm_tile '{fmt}' (rvv) lacks an integer "
                f"retirement_batch in six-state.")
    # [六态-reverse] a sixstate cell labeled as a bypass must be whitelisted.
    for fmt in sorted(labeled_bypass_fmts - wl_formats):
        errors.append(
            f"SIXSTATE-BYPASS-UNLISTED: gemm_tile '{fmt}' (rvv) carries the direct-emitter "
            f"provenance marker in six-state but is NOT in the emit-bypass whitelist.")

    return (len(errors) == 0), errors


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------
def _load_sixstate_cells(text):
    doc = json.loads(text)
    return [c for c in doc.get("states", [])
            if c.get("op") == "gemm_tile" and c.get("engine") == "rvv"]


def run_real(verbose):
    for p in (DISPATCH, WHITELIST, SIXSTATE):
        if not os.path.isfile(p):
            print(f"[setup-error] missing {p}", file=sys.stderr)
            return 2
    try:
        cpp = open(DISPATCH, encoding="utf-8").read()
        wl = json.load(open(WHITELIST, encoding="utf-8"))
        cells = _load_sixstate_cells(open(SIXSTATE, encoding="utf-8").read())
    except Exception as ex:  # noqa: BLE001
        print(f"[setup-error] {ex}", file=sys.stderr)
        return 2

    source_symbols, perr = extract_dispatch_bypass_emitters(cpp)
    if perr:
        print(f"[setup-error] {perr}", file=sys.stderr)
        return 2

    ok, errors = evaluate(source_symbols, wl, cells)

    if verbose or not ok:
        entries = wl.get("entries", [])
        print(f"[frontdoor-provenance] direct-emitter bypasses wired in kBlockDotKernels: "
              f"{len(source_symbols)} symbols across {len(entries)} whitelisted tiles "
              f"(baseline {wl.get('$meta', {}).get('baseline_count')})")
        for e in sorted(entries, key=lambda x: (x.get("retirement_batch", -1),
                                                x.get("format", ""))):
            print(f"    batch {e.get('retirement_batch')}  {e.get('format'):8s} "
                  f"{','.join(e.get('emitters', []))}")
        constructed = [c.get("format") for c in cells if c.get("state") == "constructed"]
        print(f"[frontdoor-provenance] front-door CONSTRUCTED gemm_tile rvv cells "
              f"(feed C_construct): {constructed}")

    if ok:
        print("[frontdoor-provenance] GREEN: every kBlockDotKernels direct emitter is "
              "whitelisted + dispatch-wired; no over-report; ratchet holds.")
        return 0
    print("[frontdoor-provenance] RED:", file=sys.stderr)
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

    # --- source extractor: pick bypass emitters, ignore front-door + comments ---
    cpp = (
        "some prelude;\n"
        "static constexpr BlockDotKernel kBlockDotKernels[] = {\n"
        "    {&isRepackGemvFooBody, &VariantToEmitCFunc::emitRepackGemvFoo},\n"
        "    {&isRepackGemmFooBody, &VariantToEmitCFunc::emitRepackGemmFoo},\n"
        "    // NOTE: the monolith bar kernel {isBarBody, emitBarBlockDot} was RETIRED\n"
        "    {&isTypedRepackGemvLoopBody, &VariantToEmitCFunc::emitTypedRepackGemvLoopBody},\n"
        "    {&isTypedRepackGemmLoopBody, &VariantToEmitCFunc::emitTypedRepackGemmLoopBody},\n"
        "};\n"
        "void other() { /* &VariantToEmitCFunc::emitRepackGemvGhost in a comment */ }\n"
        "auto x = &VariantToEmitCFunc::emitRepackGemvOutside;\n")  # outside table
    syms, perr = extract_dispatch_bypass_emitters(cpp)
    check("extractor found no parse error", perr is None)
    check("extractor picks bypass emitters only (not Typed front door, not comment, "
          "not outside-table)", syms == {"emitRepackGemvFoo", "emitRepackGemmFoo"})

    wl_ok = {
        "$meta": {"baseline_count": 1, "shrink_only": True},
        "entries": [
            {"op": "gemm_tile", "format": "foo", "engine": "rvv", "retirement_batch": 0,
             "provenance": PROVENANCE_MARKER,
             "emitters": ["emitRepackGemvFoo", "emitRepackGemmFoo"],
             "recognizers": ["isRepackGemvFooBody", "isRepackGemmFooBody"]}],
    }
    cells_ok = [{"op": "gemm_tile", "format": "foo", "engine": "rvv",
                 "state": "dispatch-wired", "provenance": PROVENANCE_MARKER,
                 "retirement_batch": 0}]
    ok, _ = evaluate({"emitRepackGemvFoo", "emitRepackGemmFoo"}, wl_ok, cells_ok)
    check("matching whitelist + sixstate -> GREEN", ok)

    # --- un-whitelisted bypass (source has a symbol the whitelist lacks) -> RED ---
    ok, errs = evaluate({"emitRepackGemvFoo", "emitRepackGemmFoo", "emitRepackGemvNew"},
                        wl_ok, cells_ok)
    check("un-whitelisted bypass -> RED",
          (not ok) and any("UN-WHITELISTED-BYPASS" in e for e in errs))

    # --- stale whitelist entry (symbol no longer in source) -> RED ---
    ok, errs = evaluate({"emitRepackGemvFoo"}, wl_ok, cells_ok)
    check("stale whitelist symbol -> RED",
          (not ok) and any("STALE-WHITELIST" in e for e in errs))

    # --- ratchet: baseline != entry count -> RED ---
    wl_ratchet = json.loads(json.dumps(wl_ok))
    wl_ratchet["$meta"]["baseline_count"] = 5
    ok, errs = evaluate({"emitRepackGemvFoo", "emitRepackGemmFoo"}, wl_ratchet, cells_ok)
    check("baseline != entry count -> RED", (not ok) and any("RATCHET" in e for e in errs))

    # --- over-report: bypass cell marked constructed -> RED ---
    cells_bad = [{"op": "gemm_tile", "format": "foo", "engine": "rvv",
                  "state": "constructed", "provenance": PROVENANCE_MARKER,
                  "retirement_batch": 0}]
    ok, errs = evaluate({"emitRepackGemvFoo", "emitRepackGemmFoo"}, wl_ok, cells_bad)
    check("bypass marked constructed -> RED (OVER-REPORT)",
          (not ok) and any("OVER-REPORT" in e for e in errs))

    # --- sixstate missing provenance marker -> RED ---
    cells_noprov = [{"op": "gemm_tile", "format": "foo", "engine": "rvv",
                     "state": "dispatch-wired", "retirement_batch": 0}]
    ok, errs = evaluate({"emitRepackGemvFoo", "emitRepackGemmFoo"}, wl_ok, cells_noprov)
    check("sixstate lacks provenance -> RED",
          (not ok) and any("PROVENANCE-MISSING" in e for e in errs))

    # --- sixstate labels a bypass the whitelist omits -> RED (reverse) ---
    cells_extra = cells_ok + [{"op": "gemm_tile", "format": "baz", "engine": "rvv",
                               "state": "dispatch-wired", "provenance": PROVENANCE_MARKER,
                               "retirement_batch": 1}]
    ok, errs = evaluate({"emitRepackGemvFoo", "emitRepackGemmFoo"}, wl_ok, cells_extra)
    check("sixstate-labeled bypass not whitelisted -> RED",
          (not ok) and any("SIXSTATE-BYPASS-UNLISTED" in e for e in errs))

    # --- whitelist shape: missing baseline / bad provenance -> RED ---
    wl_shape = {"$meta": {"shrink_only": True}, "entries": []}
    _, errs = evaluate(set(), wl_shape, [])
    check("missing baseline_count -> shape RED",
          any("baseline_count" in e for e in errs))

    if fails:
        print(f"[frontdoor-provenance --self-test] RED: {len(fails)} discrimination(s) failed")
        return 2
    print("[frontdoor-provenance --self-test] GREEN: classifier discriminates all cases")
    return 0


def main(argv):
    verbose = "-v" in argv or "--verbose" in argv
    if "--self-test" in argv:
        return run_self_test()
    return run_real(verbose)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
