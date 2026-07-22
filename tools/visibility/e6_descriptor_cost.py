#!/usr/bin/env python3
"""E6 -- descriptor cost accounting (T2 growth-law DISCLOSURE, NOT a C2 claim).

WHY THIS FILE EXISTS
--------------------
Prior offences on this project: a prose number ("1237 chars") that three
rebuilds reconstructed three different ways; "28 处" that no scope could
reproduce; "≈24" that turned out to be a FILE count reported as a LINE count.
So: every number this line reports is emitted BY THIS TOOL from the source of
truth at a pinned HEAD. There are no prose numbers. Re-run to re-derive.

WHAT A "DESCRIPTOR" IS (the 口径 -- read this before trusting any number)
------------------------------------------------------------------------
The front door declares each quantization format as a `constexpr <Family>Facts
k<Format>DecodeFacts = { ... };` record in
lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp. That initializer block --
and ONLY that block -- is what this tool calls the DESCRIPTOR.

  D1  (PRIMARY, the number that lands in T2)
      The `constexpr ... k<Fmt>DecodeFacts = {` line through its matching `};`,
      INCLUSIVE, counted as raw physical lines (wc -l semantics) at a pinned
      HEAD. Machine-extracted by brace matching -- never by hand.
      D1_code additionally drops blank + comment-ONLY lines, because this
      project has been burned by counting ANNOTATIONS as arithmetic (the P5
      fold_n offence). Both are emitted; D1_raw is the T2 value, D1_code is the
      honesty cross-check.

  D2  (REPORTED, NOT in T2)
      D1 plus the contiguous `//` doc-comment block immediately above it. Real
      authoring cost, but it is prose, and prose length is a writing-style
      artifact, not a structural cost. Kept visible so nobody can accuse this
      line of hiding it.

  SHARED (REPORTED, NOT per-format, NEVER divided)
      The `struct <Family>DecodeFacts { ... };` definition. It is per-FAMILY
      infrastructure amortized over every format in that family. Dividing it
      per-format would manufacture a number. It is reported per family, whole.

EXPLICITLY OUT OF SCOPE (named, not silently dropped)
    - verifier predicates, emitter core bricks, repack oracles, registry
      dispatch arms. They are NOT mechanically separable per-format by a rule
      this line is willing to defend, and the emitter/brick construction cost
      is ALREADY carried by T2's ledger_delta_hand_LOC column. Counting them
      here would double-count against that column.

STOCK vs FLOW -- the 口径 trap that matters most
-----------------------------------------------
T2's `ledger_delta_hand_LOC` is a FLOW: a delta measured AT the flip commit.
`descriptor_LOC` is a STOCK: the descriptor's size AS IT STANDS at a pinned
HEAD. They are different quantities in different units and MUST NOT be added,
differenced, or plotted on one axis. Measuring the descriptor at its flip
commit would need --at-commit reproduction, which is objection B9 (PR-22,
no 承接) -- this line does NOT do B9.

THE RED LINE ([C2-4] / [LED-4])
-------------------------------
"decode-format 前门化的 LOC/工时一律不得记入 C2." This tool does NOT touch that
rule. It writes descriptor numbers ONLY onto rows whose `axis` is C3'-*, and
`check-redline` mechanically PROVES no C2-axis row carries one. Disclosure in
T2 != attribution to C2.

"描述符" IS AN OVERLOADED WORD IN THIS REPO -- READ BEFORE QUOTING ANY NUMBER
----------------------------------------------------------------------------
At least four distinct things are called a 描述符 here:
  (1) k<Fmt>DecodeFacts       front-door decode facts record   <- D1 MEASURES THIS
  (2) RVVFlatBlockDotPlan     formula-owned final flat computation plan, built
                              from the finite typed mechanism leaf domain
                                                              <- `flat-formula`
  (3) N-operand 通用路由描述符  routing descriptor (总纲v3 [B-1])
  (4) 调度描述符               schedule descriptor (LMUL knobs)
D1 covers (1) ONLY. The formula leaf inventory in (2) is a semantic case set,
not a LOC quantity, and must NEVER be summed, averaged, or plotted with D1.
The flat formats have no (1), but they do have a formula-produced final plan;
T2 therefore says "no DecodeFacts record" rather than "no descriptor".

USAGE
    e6_descriptor_cost.py emit            # per-format table + JSON, from source
    e6_descriptor_cost.py flat-formula    # mechanism (2), separately, NOT in T2
    e6_descriptor_cost.py check-redline   # assert descriptor data never on C2 rows
    e6_descriptor_cost.py c2-rows         # the canonical C2 filter (what C2 may eat)
"""

import csv
import json
import os
import re
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO, "lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp")
T2 = os.path.join(REPO, "experiments/active/result-tables/T2_C2_ledger_marginal_cost.csv")

DECL_RE = re.compile(r"^constexpr\s+(\w+)\s+(k\w*DecodeFacts)\s*=\s*\{\s*$")
STRUCT_RE = re.compile(r"^struct\s+(\w*DecodeFacts)\s*\{\s*$")

# Descriptor symbol -> T2 scope_format. ONLY formats that have BOTH a descriptor
# and a T2 ledger row appear here. Formats with a descriptor but no T2 row are
# reported by `emit` as unmapped -- this line does NOT invent ledger rows.
SYM_TO_T2 = {
    "kQ4KDecodeFacts": "q4_K vec_dot",
    "kQ5KDecodeFacts": "q5_K vec_dot",
    "kQ6KDecodeFacts": "q6_K vec_dot",
    "kQ2KDecodeFacts": "q2_K vec_dot",
    "kQ3KDecodeFacts": "q3_K vec_dot",
    "kIq4NlDecodeFacts": "iq4_nl vec_dot",
    "kIq1SDecodeFacts": "iq1_s vec_dot",
    "kIq1MDecodeFacts": "iq1_m vec_dot",
}

# NOTE: the n/a label strings live with the T2 writer, not here. An earlier
# draft kept a NO_DESCRIPTOR = "pre-descriptor-pattern;no-facts-record-exists"
# constant here -- REMOVED because it ASSERTS "no descriptor exists", which is
# FALSE for the flat formats: they have mechanism (2). See the overloaded-word
# section above. A wrong label that looks authoritative is worse than none.


def head_sha():
    return subprocess.check_output(
        ["git", "-C", REPO, "rev-parse", "--short=8", "HEAD"], text=True
    ).strip()


def read_source(path):
    """Read a source file AT THE PIN (the HEAD blob), never the working tree.

    This repo has concurrent writers: another line owns lib/ and rewrites it
    mid-session. Reading the working tree while STAMPING a HEAD pin would make
    the pin a lie and the line numbers un-reproducible -- the stale-line-number
    class of error this project has already been burned by twice. So: measure
    the blob the pin names. Returns (lines, worktree_differs).
    """
    rel = os.path.relpath(path, REPO)
    blob = subprocess.check_output(["git", "-C", REPO, "show", f"HEAD:{rel}"], text=True)
    dirty = subprocess.call(
        ["git", "-C", REPO, "diff", "--quiet", "HEAD", "--", rel]
    ) != 0
    return blob.split("\n"), dirty


def is_comment_only(line):
    s = line.strip()
    return s.startswith("//")


def extract():
    """Brace-match every descriptor block + shared struct out of SRC @ the pin."""
    lines, dirty = read_source(SRC)
    extract.worktree_differs = dirty
    descriptors, structs = [], []

    for i, line in enumerate(lines):
        m = STRUCT_RE.match(line)
        if m:
            j = i
            while j < len(lines) and lines[j] != "};":
                j += 1
            body = lines[i : j + 1]
            structs.append(
                {
                    "family_struct": m.group(1),
                    "line_start": i + 1,
                    "line_end": j + 1,
                    "shared_raw_LOC": len(body),
                    "shared_code_LOC": sum(
                        1 for b in body if b.strip() and not is_comment_only(b)
                    ),
                }
            )
            continue

        m = DECL_RE.match(line)
        if not m:
            continue
        family, sym = m.group(1), m.group(2)
        j = i
        while j < len(lines) and lines[j] != "};":
            j += 1
        if j >= len(lines):
            sys.exit(f"FATAL: unterminated descriptor {sym} at :{i+1}")
        body = lines[i : j + 1]

        # contiguous leading // doc-comment block
        k = i - 1
        while k >= 0 and is_comment_only(lines[k]):
            k -= 1
        doc_lines = i - 1 - k

        descriptors.append(
            {
                "symbol": sym,
                "family_struct": family,
                "line_start": i + 1,
                "line_end": j + 1,
                "D1_raw_LOC": len(body),
                "D1_code_LOC": sum(
                    1 for b in body if b.strip() and not is_comment_only(b)
                ),
                "doc_comment_LOC": doc_lines,
                "D2_raw_LOC": len(body) + doc_lines,
                "t2_scope_format": SYM_TO_T2.get(sym),
            }
        )
    return descriptors, structs


def read_t2():
    with open(T2, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def cmd_emit():
    sha = head_sha()
    descriptors, structs = extract()
    print(f"# E6 descriptor cost -- emitted by {os.path.basename(__file__)}")
    print(f"# source : {os.path.relpath(SRC, REPO)}")
    print(f"# pin    : HEAD={sha}  (STOCK at HEAD, not a FLOW at flip commit;")
    print(f"#          measured on the HEAD BLOB, not the working tree)")
    if getattr(extract, "worktree_differs", False):
        print(f"# ★ NOTE : worktree copy of the source differs from HEAD (concurrent")
        print(f"#          writer). Numbers below are the PIN's. Re-pin when it lands.")
    print(f"# unit   : D1_raw_LOC = raw physical lines (wc -l semantics) of the")
    print(f"#          `constexpr ... k<Fmt>DecodeFacts = {{` .. `}};` block, inclusive.")
    print()
    print("## PER-FORMAT DESCRIPTORS (D1 = the T2 value)")
    hdr = f"{'symbol':<26}{'family_struct':<22}{'lines':<14}{'D1_raw':>7}{'D1_code':>9}{'doc':>6}{'D2_raw':>8}  t2_row"
    print(hdr)
    print("-" * len(hdr))
    for d in sorted(descriptors, key=lambda x: x["line_start"]):
        span = f":{d['line_start']}-{d['line_end']}"
        t2 = d["t2_scope_format"] or "(no T2 ledger row)"
        print(
            f"{d['symbol']:<26}{d['family_struct']:<22}{span:<14}"
            f"{d['D1_raw_LOC']:>7}{d['D1_code_LOC']:>9}{d['doc_comment_LOC']:>6}"
            f"{d['D2_raw_LOC']:>8}  {t2}"
        )
    mapped = [d for d in descriptors if d["t2_scope_format"]]
    print()
    print(f"  descriptors total          : {len(descriptors)}")
    print(f"  mapped to a T2 ledger row  : {len(mapped)}")
    print(f"  descriptor but NO T2 row   : {len(descriptors) - len(mapped)}"
          f"  (reported, NOT invented into T2)")
    print()
    print("## SHARED PER-FAMILY STRUCT DEFS (amortized; NEVER divided per-format)")
    for s in sorted(structs, key=lambda x: x["line_start"]):
        members = [d["symbol"] for d in descriptors if d["family_struct"] == s["family_struct"]]
        print(
            f"  {s['family_struct']:<22} :{s['line_start']}-{s['line_end']}"
            f"  raw={s['shared_raw_LOC']:<4} code={s['shared_code_LOC']:<4}"
            f"  serves {len(members)} format(s)"
        )
    print()
    print("## FORMATS WITH NO DESCRIPTOR (value is N/A, NOT zero)")
    print("   The flat formats (q8_0/q4_0/q4_1 + the retired flat monolith) and the")
    print("   q4_0 repack gemm_tile loop-shape row predate this pattern: no facts")
    print("   record exists for them. Reporting 0 would be a fabricated measurement.")

    out = os.path.join(REPO, "experiments/active/result-tables/e6_descriptor_cost_emitted.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "pin_head": sha,
                "source": os.path.relpath(SRC, REPO),
                "unit": "D1_raw_LOC = raw physical lines of constexpr k<Fmt>DecodeFacts initializer block, inclusive",
                "stock_not_flow": True,
                "descriptors": descriptors,
                "shared_structs": structs,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n[emit] machine-readable -> {os.path.relpath(out, REPO)}")
    return 0


def cmd_c2_rows():
    """THE CANONICAL C2 FILTER. Any C2 contribution claim / cost law / curve that
    reads T2 MUST consume exactly this row set and no other."""
    rows = read_t2()
    hdr, body = rows[0], rows[1:]
    ax, fk, sf, sq = (hdr.index(c) for c in ("axis", "family_kind", "scope_format", "seq"))
    c2 = [r for r in body if r[ax].startswith("C2-")]
    print("# Canonical C2 row filter over T2:  axis.startswith('C2-')")
    print("# (axis is authoritative, NOT family_kind: rows 15/16 are family_kind=")
    print("#  extension-family-forward but axis=C1-*, so a family_kind prefix match")
    print("#  would wrongly sweep C1 rows into C2.)")
    for r in c2:
        print(f"  seq={r[sq]:<3} family_kind={r[fk]:<28} scope_format={r[sf]}")
    print(f"\n  C2-axis rows: {len(c2)} of {len(body)}")
    print("  NOTE: of these, the two comparable 轨二 onboarding points are the IME")
    print("  and X-SCALAR anchors; the substrate + 轨一 sub-ext rows carry explicit")
    print("  in-cell 勿混 warnings. Two points is not a law -- do not draw a curve.")
    return 0


def cmd_check_redline():
    """MECHANICAL PROOF of [C2-4]/[LED-4]: descriptor numbers never reach C2.

    Without this, the red line is only a promise in prose.
    """
    rows = read_t2()
    hdr, body = rows[0], rows[1:]
    if "descriptor_LOC" not in hdr:
        print("SKIP: T2 has no descriptor_LOC column yet.")
        return 0
    ax, dl, ds, sq = (hdr.index(c) for c in ("axis", "descriptor_LOC", "descriptor_scope", "seq"))
    fails = []

    for r in body:
        axis, val = r[ax], r[dl]
        numeric = val.strip().isdigit()
        # A1: no C2-axis row may carry a numeric descriptor LOC.
        if axis.startswith("C2-") and numeric:
            fails.append(f"seq={r[sq]}: C2-axis row carries numeric descriptor_LOC={val}")
        # A2: no C1-axis row may carry one either.
        if axis.startswith("C1-") and numeric:
            fails.append(f"seq={r[sq]}: C1-axis row carries numeric descriptor_LOC={val}")
        # A3: every numeric descriptor LOC sits on a C3' row (the contrapositive).
        if numeric and not axis.startswith("C3'"):
            fails.append(f"seq={r[sq]}: numeric descriptor_LOC on non-C3' axis={axis}")
        # A4: every numeric value must carry a reproducible scope pin.
        if numeric and "@" not in r[ds]:
            fails.append(f"seq={r[sq]}: numeric descriptor_LOC without a pinned scope")

    n_num = sum(1 for r in body if r[dl].strip().isdigit())
    n_c2 = sum(1 for r in body if r[ax].startswith("C2-"))
    print("[check-redline] [C2-4]/[LED-4]: decode-format descriptor LOC must NOT")
    print("                be attributable to C2. Disclosure in T2 != C2 attribution.")
    print(f"  rows                        : {len(body)}")
    print(f"  C2-axis rows                : {n_c2}")
    print(f"  numeric descriptor_LOC rows : {n_num}")
    print(f"  A1 no numeric on C2 axis    : {'PASS' if not [f for f in fails if 'C2-axis' in f] else 'FAIL'}")
    print(f"  A2 no numeric on C1 axis    : {'PASS' if not [f for f in fails if 'C1-axis' in f] else 'FAIL'}")
    print(f"  A3 numeric => C3' axis only : {'PASS' if not [f for f in fails if 'non-C3' in f] else 'FAIL'}")
    print(f"  A4 numeric => pinned scope  : {'PASS' if not [f for f in fails if 'without a pinned' in f] else 'FAIL'}")
    if fails:
        print("\nFAIL:")
        for f in fails:
            print("  " + f)
        return 1
    print("\n  VERDICT: PASS -- red line mechanically held.")
    return 0


FLAT_FORMULA_SRC = os.path.join(
    REPO, "lib/Plugin/RVV/Construction/RVVFlatBlockDotFormula.cpp"
)
FLAT_CASE_RE = re.compile(r"^\s*case\s+RVVFlatBlockDotLeaf::(\w+)\s*:\s*$")


def cmd_flat_formula():
    """List the formula-owned finite flat leaf cases at the pinned HEAD.

    This is a semantic inventory only. It deliberately reports no LOC metric
    and writes nothing into T2: formula cases and DecodeFacts records are
    different evidence units.
    """
    lines, dirty = read_source(FLAT_FORMULA_SRC)
    start = next(
        i for i, line in enumerate(lines)
        if "constructRVVFlatBlockDotFormula(" in line
    )
    depth = 0
    opened = False
    end = None
    for i in range(start, len(lines)):
        depth += lines[i].count("{")
        if lines[i].count("{"):
            opened = True
        depth -= lines[i].count("}")
        if opened and depth == 0:
            end = i
            break
    if end is None:
        sys.exit("FATAL: unterminated constructRVVFlatBlockDotFormula")
    cases = [
        (i + 1, match.group(1))
        for i in range(start, end + 1)
        if (match := FLAT_CASE_RE.match(lines[i]))
    ]
    if not cases:
        sys.exit("FATAL: no RVVFlatBlockDotLeaf cases found")
    names = [name for _, name in cases]
    if len(names) != len(set(names)):
        sys.exit("FATAL: duplicate RVVFlatBlockDotLeaf formula case")

    print("# Flat computation mechanism (2): formula-owned final plan cases")
    print(f"# source : {os.path.relpath(FLAT_FORMULA_SRC, REPO)}:{start+1}-{end+1}")
    print(f"# pin    : HEAD={head_sha()}  (read from the HEAD blob)")
    if dirty:
        print("# NOTE   : worktree differs; re-run after committing to update the pin")
    print("# unit   : finite semantic leaf identity (NOT LOC; NOT a T2 value)")
    for line, name in cases:
        print(f"  {name:<18} :{line}")
    print(f"\n  formula leaf cases: {len(cases)}")
    return 0


def main():
    cmds = {
        "emit": cmd_emit,
        "flat-formula": cmd_flat_formula,
        "check-redline": cmd_check_redline,
        "c2-rows": cmd_c2_rows,
    }
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        return 2
    return cmds[sys.argv[1]]()


if __name__ == "__main__":
    sys.exit(main())
