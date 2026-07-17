#!/usr/bin/env python3
"""check_pat3_registry_diff.py — [PAT-3] dual-board registry-diff=0 gate.

[PAT-3] canon (_attic/docs/canon/Weft-RV_科研目标总纲v2.md:122):
    「同一注册表在双板（VLEN128/VLEN256）**零条目改写、仅键值不同**即通过（diff 注册表 = 0）」
    (the same registry, on both boards, rewrites ZERO entries and differs only in KEY VALUES)

★ WHY THIS GATE IS NOT A FILE-DIFF (anti-hollow rationale — read before editing) ★
`schema/pattern-registry.v1.json` is ONE board-independent file. There is no per-board
copy of it. So the naive reading of "diff 注册表 = 0" — diff two per-board registry files —
has nothing to diff: it would compare the file to itself and print 0 forever. That gate
would be TRIVIALLY GREEN (hollow): it could never go red, so it would certify nothing.
A hollow gate is worse than no gate, because it launders an unproven property into a
green check.

This gate instead RESOLVES the one registry against TWO REAL board capability-fact
instances and asserts the [PAT-3] property on the RESOLUTION:

  G1  no per-board override mechanism  — no row may carry a board-scoped override
      (board_overrides / per_board / vlen128 / vlen256 / boards keys), and no field
      may be unclassified (fail-closed). This is the load-bearing form of "零条目改写":
      an entry can only be rewritten per board if a per-board override channel EXISTS.
  G2  entry-body board-invariance — for every pattern_id, the ENTRY BODY
      (pattern_id/transform/status/milestone/brick) resolves byte-identically on both
      boards. ★HONESTY: given today's schema G2 is ENTAILED by G1 (no override channel
      => resolution is a copy => bodies cannot differ). G2 is therefore NOT independent
      teeth today; it is the invariant G1 protects, and it becomes load-bearing the day
      an override channel is added. It is reported as `entailed`, never as proof.
  G3  transform board-cleanliness — `transform` (the construction pointer = the entry
      proper) must contain NO board-identity literal. NON-VACUOUS: the registry is full
      of board literals (evidence fields legitimately cite board casefiles); G3 asserts
      they are CONFINED OUT of the entry body. It can and does discriminate.
  G4  no board-identity keying — no `requires` / `discriminant_capability_fact` predicate
      may key on a BOARD IDENTITY (board_id / march / board name). Selection must be by
      CAPABILITY key. This is the [PAT-3]/C1 spirit: capability-keyed, not board-branched.
  G5  anti-vacuity — at least one row's KEY VALUES must actually DIFFER across the two
      boards. If every key value resolved identically, the "dual-board" resolution would
      be a no-op copy and G1-G4 would be vacuously green. G5 is the gate's own
      hollowness detector.

TIERS (a field's tier decides whether a board literal in it is legal):
  entry-body : pattern_id, transform, status, milestone, brick   -> board-literal FORBIDDEN
  key        : requires, discriminant_capability_fact            -> capability keys only,
                                                                    board identity FORBIDDEN
  evidence   : metrics_hook, mechanism, measured_negative_boundary, restart_condition,
               phase_note, xfer_note, xfer_classes               -> board literals EXPECTED
               (these cite board casefiles / record board-measured falsifications; a gate
               that banned board literals here would fire ~11 false reds on real evidence)
  Any field outside these tiers => RED (fail-closed): an unclassified field is exactly how
  a per-board override channel would sneak in.

BOARD FACTS ARE DERIVED FROM REAL IN-REPO MEASURED SNAPSHOTS, never invented:
  source = schema/measurement-memory.v1.json  (rows[].snapshot)
  Derivation rules (documented, mechanical, no fabrication):
    rvv.v       <= march contains 'v'            (both boards)
    ime.present <= march contains 'xsmtvdotii'   (opt-in march token; a board has IME if
                   ANY of its measured snapshots was built with the token)
    everything else (rvv.zvfh / rvv.zvfhmin / format_capability / shape_fact:*) is NOT
    derivable from a snapshot record => resolved as `unknown`, NOT fabricated true/false.
    `unknown` is a legal key value: [PAT-3] constrains entry bodies, not predicate truth.

★ SCOPE / WHAT THIS GATE DOES NOT PROVE (honest boundary) ★
This is a STATIC data-level check on the registry as data. It does NOT execute anything on
a board. It does not prove the runtime consumption path emits identical entries on real
silicon. Board execution is out of this gate's scope (board time is owned elsewhere).
Claiming "dual-board verified" from this gate alone would overstate it.

Usage:  python3 tools/gates/check_pat3_registry_diff.py [--self-test] [--registry PATH] [-v]
        --self-test : hermetic negative controls on synthetic registries. Proves each
                      check DISCRIMINATES (goes red on a planted violation) rather than
                      being green-always.
        --registry  : judge an alternate registry file (used by the CI live-injection step
                      to prove the gate reds on a deliberately corrupted copy).
Exit:   0 GREEN / 1 RED / 2 setup RED
"""
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REGISTRY = os.path.join(REPO, "schema", "pattern-registry.v1.json")
MEMORY = os.path.join(REPO, "schema", "measurement-memory.v1.json")

ENTRY_BODY_FIELDS = ("pattern_id", "transform", "status", "milestone", "brick")
KEY_FIELDS = ("requires", "discriminant_capability_fact")
EVIDENCE_FIELDS = (
    "metrics_hook", "mechanism", "measured_negative_boundary",
    "restart_condition", "phase_note", "xfer_note", "xfer_classes",
)
KNOWN_FIELDS = frozenset(ENTRY_BODY_FIELDS + KEY_FIELDS + EVIDENCE_FIELDS)

# Board-scoped override channels. Their PRESENCE is the violation (G1).
OVERRIDE_KEYS = ("board_overrides", "per_board", "boards", "vlen128", "vlen256",
                 "board_id", "per_vlen", "board_specific")

# Board-IDENTITY literals. Deliberately EXCLUDES 'rvv' (that is the ISA/dialect name, not a
# board) and excludes bare 'v' — matching those would fire on every lowering path string.
BOARD_LITERAL = re.compile(
    r"VLEN\s*=?\s*(?:128|256)|vlen(?:128|256)|\bk1\b|\bX60\b|openEuler|SpacemiT|BananaPi"
    r"|\bvl\s*=\s*\d+",
    re.IGNORECASE,
)
# A requires-predicate keying on board identity rather than capability (G4).
BOARD_KEYED_PREDICATE = re.compile(
    r"^\s*(?:board|board_id|march|vlen|platform|soc)\s*[:=]", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# Pure core (fed synthetic inputs by --self-test; real data at runtime)
# ---------------------------------------------------------------------------
def derive_board_facts(snapshots):
    """Real snapshots -> {board_id: {'vlen':int, 'marches':set, 'caps':{id: True/False}}}.

    Mechanical derivation only. A capability not derivable from a snapshot record is left
    OUT of caps (resolved `unknown` downstream) instead of being fabricated.
    """
    boards = {}
    for snap in snapshots:
        bid = snap.get("board_id")
        if not bid:
            continue
        march = str(snap.get("march", ""))
        b = boards.setdefault(bid, {"vlen": snap.get("vlen"), "marches": set(), "caps": {}})
        b["marches"].add(march)
    for bid, b in boards.items():
        marches = b["marches"]
        # rv64gcv... : the 'v' in the base march => RVV present.
        b["caps"]["rvv.v"] = any(re.match(r"rv64g?c?v", m) for m in marches)
        # IME is an opt-in march token; ANY snapshot built with it proves the board has it.
        b["caps"]["ime.present"] = any("xsmtvdotii" in m for m in marches)
    return boards


def resolve(registry, board):
    """Resolve the ONE registry against ONE board's facts -> resolved entries.

    Returns [{'pattern_id', 'entry_body': {...}, 'key_values': {pred: True/False/'unknown'}}]
    Entry body is carried through UNCHANGED: with no per-board override channel (G1) this is
    a copy — which is exactly the [PAT-3] property, and exactly why G2 alone proves nothing.
    """
    out = []
    caps = board["caps"]
    for row in registry.get("patterns", []):
        key_values = {}
        for pred in row.get("requires", []) or []:
            key_values[pred] = caps.get(pred, "unknown")
        dcf = row.get("discriminant_capability_fact")
        if dcf is not None:
            key_values["__discriminant__"] = "present"
        out.append({
            "pattern_id": row.get("pattern_id"),
            "entry_body": {f: row.get(f) for f in ENTRY_BODY_FIELDS if f in row},
            "key_values": key_values,
        })
    return out


def check(registry, boards, verbose=False):
    """Run G1-G5. Returns (findings, stats). findings == [] => GREEN."""
    findings = []
    stats = {}
    rows = registry.get("patterns", [])
    stats["n_patterns"] = len(rows)
    stats["n_boards"] = len(boards)

    if len(boards) != 2:
        findings.append(("G0-setup", f"need exactly 2 boards to diff, got {len(boards)}: "
                                     f"{sorted(boards)}"))
        return findings, stats

    # --- G1: no per-board override channel; no unclassified field (fail-closed) ---------
    for row in rows:
        pid = row.get("pattern_id")
        for k in row:
            if k.lower() in OVERRIDE_KEYS:
                findings.append(("G1-per-board-override",
                                 f"{pid}: field '{k}' is a per-board override channel — an "
                                 f"entry that can be rewritten per board violates [PAT-3]"))
            elif k not in KNOWN_FIELDS:
                findings.append(("G1-unclassified-field",
                                 f"{pid}: field '{k}' is unclassified (not entry-body/key/"
                                 f"evidence) — fail-closed: classify it in this gate first"))
        for f in ENTRY_BODY_FIELDS + KEY_FIELDS:
            v = row.get(f)
            if isinstance(v, dict):
                for k in v:
                    if str(k).lower() in OVERRIDE_KEYS:
                        findings.append(("G1-per-board-override",
                                         f"{pid}.{f}: nested board-scoped key '{k}'"))

    # --- G2: entry-body board-invariance (entailed by G1 today; see module docstring) ---
    bids = sorted(boards)
    res = {bid: resolve(registry, boards[bid]) for bid in bids}
    a, b = res[bids[0]], res[bids[1]]
    ids_a = [e["pattern_id"] for e in a]
    ids_b = [e["pattern_id"] for e in b]
    if ids_a != ids_b:
        only_a = set(ids_a) - set(ids_b)
        only_b = set(ids_b) - set(ids_a)
        findings.append(("G2-entry-set-drift",
                         f"entry SET differs across boards: only-{bids[0]}={sorted(only_a)} "
                         f"only-{bids[1]}={sorted(only_b)}"))
    n_rewritten = 0
    for ea, eb in zip(a, b):
        if ea["entry_body"] != eb["entry_body"]:
            n_rewritten += 1
            diffs = [f for f in set(ea["entry_body"]) | set(eb["entry_body"])
                     if ea["entry_body"].get(f) != eb["entry_body"].get(f)]
            findings.append(("G2-entry-rewritten",
                             f"{ea['pattern_id']}: entry body REWRITTEN across boards "
                             f"(fields {sorted(diffs)}) — [PAT-3] requires zero rewrite"))
    stats["entries_rewritten"] = n_rewritten
    stats["registry_diff"] = n_rewritten  # the literal "diff 注册表" number

    # --- G3: transform board-cleanliness -----------------------------------------------
    ev_hits = 0
    for row in rows:
        for f in EVIDENCE_FIELDS:
            v = row.get(f)
            if v is None:
                continue
            s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            ev_hits += len(BOARD_LITERAL.findall(s))
    stats["board_literals_in_evidence"] = ev_hits
    body_hits = 0
    for row in rows:
        for f in ENTRY_BODY_FIELDS:
            v = row.get(f)
            if not isinstance(v, str):
                continue
            hits = BOARD_LITERAL.findall(v)
            if hits:
                body_hits += len(hits)
                findings.append(("G3-board-literal-in-entry-body",
                                 f"{row.get('pattern_id')}.{f}: board-identity literal "
                                 f"{sorted(set(hits))} in the ENTRY BODY — the construction "
                                 f"pointer is board-specialized, not capability-keyed"))
    stats["board_literals_in_entry_body"] = body_hits

    # --- G4: no board-identity keying ---------------------------------------------------
    for row in rows:
        for pred in row.get("requires", []) or []:
            if BOARD_KEYED_PREDICATE.match(str(pred)) or BOARD_LITERAL.search(str(pred)):
                findings.append(("G4-board-identity-keyed",
                                 f"{row.get('pattern_id')}: requires '{pred}' keys on BOARD "
                                 f"IDENTITY — selection must be by capability key"))

    # --- G5: anti-vacuity — key values must actually differ across the two boards -------
    differing = []
    for ea, eb in zip(a, b):
        if ea["key_values"] != eb["key_values"]:
            differing.append(ea["pattern_id"])
    stats["rows_with_differing_key_values"] = len(differing)
    stats["differing_rows"] = differing
    if not differing:
        findings.append(("G5-vacuous-resolution",
                         "NO row's key values differ across the two boards — the dual-board "
                         "resolution is a no-op copy, so G1-G4 are VACUOUSLY green. This gate "
                         "would be hollow; it reds instead."))

    if verbose:
        for bid in bids:
            print(f"  [board] {bid}: vlen={boards[bid]['vlen']} caps={boards[bid]['caps']}")
        print(f"  [resolve] rows with differing key values: {differing}")
    return findings, stats


# ---------------------------------------------------------------------------
# Self-test: hermetic negative controls (proves each check discriminates)
# ---------------------------------------------------------------------------
def _boards_fixture():
    return derive_board_facts([
        {"board_id": "openEuler-VLEN128", "vlen": 128, "march": "rv64gcv"},
        {"board_id": "SpacemiT-K1-VLEN256", "vlen": 256, "march": "rv64gcv_xsmtvdotii1p0"},
    ])


def _clean_registry():
    """Shaped like the real registry: capability-keyed, board literals only in evidence."""
    return {"patterns": [
        {"pattern_id": "P-A", "milestone": "M", "brick": 1, "requires": ["rvv.v"],
         "transform": "lib/Conversion/RVV/Foo.cpp:emitFoo", "mechanism": "typed body op",
         "metrics_hook": "test/foo.mlir", "status": "mechanized"},
        {"pattern_id": "P-IME", "milestone": "M", "brick": 2,
         "requires": ["ime.present", "rvv.v"],
         "transform": "lib/Conversion/RVV/Ime.cpp:emitIme",
         "mechanism": "FALSIFIED on k1/VLEN256 board casefile",   # evidence: literal LEGAL
         "metrics_hook": "experiments/active/g6/board_seal.txt (K1, VLEN256)",
         "status": "mechanized"},
    ]}


def self_test(verbose=False):
    boards = _boards_fixture()
    cases = []

    # POSITIVE control: a clean, real-shaped registry must be GREEN.
    cases.append(("POS-clean-registry", _clean_registry(), None))

    # NC1 (G1): a per-board override channel is planted.
    r = _clean_registry()
    r["patterns"][0]["board_overrides"] = {"vlen256": {"transform": "other.cpp:emitWide"}}
    cases.append(("NC1-per-board-override", r, "G1-per-board-override"))

    # NC2 (G1 fail-closed): an unclassified field appears.
    r = _clean_registry()
    r["patterns"][0]["some_new_field"] = "anything"
    cases.append(("NC2-unclassified-field", r, "G1-unclassified-field"))

    # NC3 (G3): the ENTRY BODY (transform) is board-specialized.
    r = _clean_registry()
    r["patterns"][0]["transform"] = "if VLEN256 use emitWide else emitNarrow"
    cases.append(("NC3-board-literal-in-entry-body", r, "G3-board-literal-in-entry-body"))

    # NC4 (G4): selection keyed on BOARD IDENTITY instead of a capability.
    r = _clean_registry()
    r["patterns"][0]["requires"] = ["board:k1"]
    cases.append(("NC4-board-identity-keyed", r, "G4-board-identity-keyed"))

    # NC5 (G5): every key value identical on both boards => vacuous dual-board resolution.
    r = {"patterns": [dict(_clean_registry()["patterns"][0])]}   # drop the IME row
    cases.append(("NC5-vacuous-resolution", r, "G5-vacuous-resolution"))

    # NC6 (G2): entry body actually rewritten per board — reachable only via an override
    # channel, so it is driven through a MUTATED RESOLVER to prove G2 discriminates if the
    # schema ever grows one. (Honest: G2 is unreachable on today's schema; see docstring.)
    fails = []
    for name, reg, want in cases:
        findings, _ = check(reg, boards)
        cats = {c for c, _ in findings}
        if want is None:
            ok = not findings
            got = "GREEN" if ok else f"RED {sorted(cats)}"
        else:
            ok = want in cats
            got = f"RED {sorted(cats)}" if findings else "GREEN"
        print(f"  [{'ok' if ok else 'FAIL'}] {name:34s} want="
              f"{want or 'GREEN':34s} got={got}")
        if not ok:
            fails.append(name)

    # NC6: prove G2 fires when entry bodies DO differ, by feeding check() a registry whose
    # resolution is forced to diverge (monkeypatched resolver = the "override channel exists"
    # world). This is a discrimination proof of G2, not a claim about today's schema.
    global resolve
    orig = resolve

    def _forked_resolve(registry, board):
        out = orig(registry, board)
        if board["vlen"] == 256:                      # simulate a per-board entry rewrite
            for e in out:
                if e["pattern_id"] == "P-A":
                    e["entry_body"] = dict(e["entry_body"], transform="WIDE.cpp:emitWide")
        return out

    resolve = _forked_resolve
    try:
        findings, _ = check(_clean_registry(), boards)
        cats = {c for c, _ in findings}
        ok = "G2-entry-rewritten" in cats
        print(f"  [{'ok' if ok else 'FAIL'}] {'NC6-entry-rewritten(forked resolver)':34s} "
              f"want={'G2-entry-rewritten':34s} got=RED {sorted(cats)}")
        if not ok:
            fails.append("NC6-entry-rewritten")
    finally:
        resolve = orig

    print("=" * 78)
    if fails:
        print(f"[pat3-registry-diff --self-test] RED: {len(fails)} discrimination(s) failed: "
              f"{fails}")
        return 1
    print("[pat3-registry-diff --self-test] GREEN: every check discriminates "
          "(6 negative controls red, positive control green)")
    return 0


# ---------------------------------------------------------------------------
def main(argv):
    verbose = "-v" in argv or "--verbose" in argv
    if "--self-test" in argv:
        return self_test(verbose)

    registry_path = REGISTRY
    if "--registry" in argv:
        i = argv.index("--registry")
        if i + 1 >= len(argv):
            print("[pat3-registry-diff] SETUP RED: --registry needs a PATH", file=sys.stderr)
            return 2
        registry_path = argv[i + 1]

    for path in (registry_path, MEMORY):
        if not os.path.exists(path):
            print(f"[pat3-registry-diff] SETUP RED: missing {path}", file=sys.stderr)
            return 2
    registry = json.load(open(registry_path, encoding="utf-8"))
    memory = json.load(open(MEMORY, encoding="utf-8"))

    snapshots = [r["snapshot"] for r in memory.get("rows", []) if "snapshot" in r]
    boards = derive_board_facts(snapshots)
    # Fail-closed: the two boards must come from REAL measured snapshots, never invented.
    v = {bid: b["vlen"] for bid, b in boards.items()}
    if sorted(set(v.values())) != [128, 256]:
        print(f"[pat3-registry-diff] SETUP RED: need real VLEN128+VLEN256 board snapshots in "
              f"{MEMORY}; derived {v}", file=sys.stderr)
        return 2

    print(f"[pat3-registry-diff] registry={os.path.relpath(registry_path, REPO)} "
          f"({len(registry.get('patterns', []))} patterns)")
    print(f"[pat3-registry-diff] boards derived from real snapshots in "
          f"{os.path.relpath(MEMORY, REPO)}: {sorted(boards)}")
    findings, stats = check(registry, boards, verbose=verbose)

    print(f"[pat3-registry-diff] registry diff (entries rewritten across boards) = "
          f"{stats.get('registry_diff')}")
    print(f"[pat3-registry-diff] board literals: entry-body="
          f"{stats.get('board_literals_in_entry_body')} (must be 0) | evidence="
          f"{stats.get('board_literals_in_evidence')} (legal, proves G3 non-vacuous)")
    print(f"[pat3-registry-diff] rows whose KEY VALUES differ across boards = "
          f"{stats.get('rows_with_differing_key_values')} "
          f"{stats.get('differing_rows')} (must be >0, else vacuous)")
    if findings:
        print("=" * 78)
        for cat, msg in findings:
            print(f"  RED [{cat}] {msg}", file=sys.stderr)
        print(f"[pat3-registry-diff] RED: {len(findings)} finding(s)", file=sys.stderr)
        return 1
    print("[pat3-registry-diff] GREEN: zero entries rewritten across VLEN128/VLEN256; "
          "variation confined to key values; entry bodies board-literal-free.")
    print("[pat3-registry-diff] SCOPE: static data-level check — does NOT execute on a "
          "board and does not prove runtime dual-board emission.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
