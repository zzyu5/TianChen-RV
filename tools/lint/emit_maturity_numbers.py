#!/usr/bin/env python3
# tools/lint/emit_maturity_numbers.py — [DEBT-VIS] single source of truth for maturity numbers.
#
# Auto-DERIVES the coverage-maturity numbers that ROADMAP / briefs / the paper cite, so they stop
# drifting apart by hand. Everything here is a pure function of three committed machine sources:
#
#   1. schema/coverage-sixstate.v1.json   — the six-state roster (labeled counts).
#   2. tools/lint/check_construction_manifest_regex.py — the STRICT realized-body manifest
#      classifier (the CERTIFICATION件). We IMPORT its classify_auto_readout so `certified` is the
#      SAME predicate the [DEBT-CERT] gate enforces — one cert logic, not two.
#   3. schema/emit-bypass-whitelist.v1.json — the [F-EMIT] direct-emitter bypass allow-list.
#
# Numbers emitted (the canonical set):
#   C_construct  labeled   = count(state == constructed)                 # account / roster
#   C_construct  certified = of those, count passing the strict checker  # machine-certified (PAPER)
#   dispatch-wired, absent, constructed-weak (opaque-helper bypass baseline)
#   bypass baseline (emit-bypass-whitelist active entries; retired ledger size)
#   mineral-vein (矿脉) = gemm_tile cells still dispatch-wired (repack-GEMM front-door not built)
#
#   IMPORTANT: `certified` is the number for external / paper use. `labeled` is the account total.
#   certified <= labeled ALWAYS; a gap is honest (未机器认证), not a defect to paper over.
#
# Like the cert checker, the schema is read from a COMMITTED ref (default HEAD), NOT the working
# tree — a parallel line may be mid-editing the working-tree schema. Pass --worktree to override.
#
#   emit_maturity_numbers.py                 # human-readable canonical-numbers block @ HEAD
#   emit_maturity_numbers.py --json          # machine-readable snapshot (stdout)
#   emit_maturity_numbers.py --ref <rev>     # a different committed snapshot
#   emit_maturity_numbers.py --worktree      # read the working-tree schema (may be mid-edit)
#   emit_maturity_numbers.py --drift         # compare doc-cited numbers vs machine; exit 1 on drift
#   emit_maturity_numbers.py --self-test     # exercise the pure aggregator

import argparse
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_construction_manifest_regex as cert  # noqa: E402  (the ONE certification classifier)

SCHEMA_REL = "schema/coverage-sixstate.v1.json"
BYPASS_REL = "schema/emit-bypass-whitelist.v1.json"

# Doc anchors scanned by --drift. Each pattern captures the doc's *labeled* C_construct numerator
# (and, where present, the denominator). We compare the numerator against the machine `labeled`.
# A doc that no longer matches its pattern is reported as a reworded/stale anchor (not silently OK).
DOC_CCONSTRUCT_ANCHORS = [
    "docs/ROADMAP.md",
    "docs/reports/2026-07-10-paper-evidence-index.md",
]
# Any "<num>/<den>" or "C_construct ... <num>/<den>" where den is the roster total.
CCONSTRUCT_RE = re.compile(r"C_construct[^0-9]{0,40}?(\d+)\s*/\s*(\d+)")


# ------------------------------------------------------------ pure aggregator
def compute(doc, bypass):
    """Pure. Returns the canonical-numbers dict from an already-loaded schema + bypass doc."""
    states = doc.get("states", [])
    total = len(states)

    def count_state(st):
        return sum(1 for s in states if s.get("state") == st)

    constructed = [s for s in states if s.get("state") == "constructed"]
    certified = 0
    red = []
    red_reason_buckets = {}
    for s in constructed:
        ok, reason, shape = cert.classify_auto_readout(s.get("auto_readout"))
        if ok:
            certified += 1
        else:
            red.append({"op": s.get("op"), "format": s.get("format"), "reason": reason})
            # coarse mechanical bucket for the aggregate view (judgment taxonomy lives in the .md).
            if reason.startswith("auto_readout is"):
                bucket = "manifest-absent (auto_readout=None)"
            elif reason.startswith("envelope mismatch"):
                bucket = "envelope-mismatch (non-E5 form / hand-labeled pending)"
            elif reason.startswith("shape non-compliant"):
                bucket = "shape-non-compliant (unrecognized typed body)"
            elif "opaque" in reason:
                bucket = "opaque-hand-helper"
            else:
                bucket = "other"
            red_reason_buckets[bucket] = red_reason_buckets.get(bucket, 0) + 1

    labeled = len(constructed)
    dispatch_wired = count_state("dispatch-wired")
    absent = count_state("absent")
    weak = count_state("constructed-weak")

    mineral_vein = [
        {"op": s.get("op"), "format": s.get("format")}
        for s in states
        if s.get("op") == "gemm_tile" and s.get("state") == "dispatch-wired"
    ]
    bypass_entries = bypass.get("entries", []) if isinstance(bypass, dict) else []
    bypass_retired = bypass.get("retired_ledger", []) if isinstance(bypass, dict) else []

    def pct(n):
        return round(100.0 * n / total, 1) if total else 0.0

    return {
        "roster_total": total,
        "C_construct": {
            "labeled": labeled,
            "labeled_pct": pct(labeled),
            "certified": certified,
            "certified_pct": pct(certified),
            "red": labeled - certified,
        },
        "dispatch_wired": dispatch_wired,
        "absent": absent,
        "constructed_weak_opaque_baseline": weak,
        "bypass_whitelist_active": len(bypass_entries),
        "bypass_whitelist_retired": len(bypass_retired),
        "mineral_vein_gemm_tile_dispatch_wired": len(mineral_vein),
        "mineral_vein_cells": mineral_vein,
        "red_cells": red,
        "red_reason_buckets": red_reason_buckets,
    }


# ------------------------------------------------------------------- loaders
def repo_root():
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=os.path.dirname(os.path.abspath(__file__)), capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def load_from_ref(root, ref, rel):
    out = subprocess.run(["git", "show", f"{ref}:{rel}"], cwd=root,
                         capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def load_worktree(root, rel):
    with open(os.path.join(root, rel), encoding="utf-8") as f:
        return json.load(f)


def short_ref(root, ref):
    return subprocess.run(["git", "rev-parse", "--short", ref], cwd=root,
                          capture_output=True, text=True, check=True).stdout.strip()


# --------------------------------------------------------------------- render
def render_human(nums, ref_label):
    cc = nums["C_construct"]
    lines = []
    lines.append(f"== coverage-maturity canonical numbers @ {ref_label} "
                 f"(roster {nums['roster_total']}) ==")
    lines.append(f"  C_construct labeled   : {cc['labeled']}/{nums['roster_total']} "
                 f"= {cc['labeled_pct']}%   (account / state==constructed)")
    lines.append(f"  C_construct certified : {cc['certified']}/{nums['roster_total']} "
                 f"= {cc['certified_pct']}%   (PAPER — passes strict manifest checker)")
    lines.append(f"  C_construct RED       : {cc['red']}   (labeled but NOT machine-certified)")
    lines.append(f"  dispatch-wired        : {nums['dispatch_wired']}")
    lines.append(f"  absent                : {nums['absent']}")
    lines.append(f"  constructed-weak (opaque-helper bypass baseline) : "
                 f"{nums['constructed_weak_opaque_baseline']}")
    lines.append(f"  emit-bypass-whitelist : active={nums['bypass_whitelist_active']}  "
                 f"retired={nums['bypass_whitelist_retired']}")
    lines.append(f"  mineral-vein (矿脉, gemm_tile dispatch-wired) : "
                 f"{nums['mineral_vein_gemm_tile_dispatch_wired']}  "
                 f"{[c['format'] for c in nums['mineral_vein_cells']]}")
    if nums["red_reason_buckets"]:
        lines.append("  RED reason buckets (mechanical):")
        for b, n in sorted(nums["red_reason_buckets"].items(), key=lambda kv: -kv[1]):
            lines.append(f"    - {n:3d}  {b}")
    return "\n".join(lines)


def run_drift(root, ref, nums):
    """Compare doc-cited C_construct numerators vs machine `labeled`. Exit 1 on any mismatch."""
    labeled = nums["C_construct"]["labeled"]
    certified = nums["C_construct"]["certified"]
    total = nums["roster_total"]
    print(f"-- drift check: machine labeled={labeled}/{total}, certified={certified}/{total} "
          f"@ {short_ref(root, ref)} --")
    drift = 0
    for rel in DOC_CCONSTRUCT_ANCHORS:
        path = os.path.join(root, rel)
        if not os.path.exists(path):
            print(f"  [SKIP] {rel} (not found)")
            continue
        text = open(path, encoding="utf-8").read()
        hits = CCONSTRUCT_RE.findall(text)
        if not hits:
            print(f"  [WARN] {rel}: no 'C_construct N/M' anchor found (reworded?)")
            drift += 1
            continue
        for num, den in hits:
            num, den = int(num), int(den)
            tag = "OK " if (num == labeled and den == total) else "DRIFT"
            if tag == "DRIFT":
                drift += 1
            print(f"  [{tag}] {rel}: doc cites {num}/{den}, machine labeled {labeled}/{total}")
    if drift:
        print(f"RED: {drift} doc anchor(s) drift from the machine C_construct. "
              "Regenerate the cited numbers from emit_maturity_numbers.py.")
    else:
        print("OK: no C_construct drift in scanned doc anchors.")
    return 1 if drift else 0


# ------------------------------------------------------------------ self-test
def self_test():
    doc = {"states": [
        {"op": "vec_dot", "format": "q8_0", "state": "constructed",
         "auto_readout": ("E5-increment1-auto: constructed (STRONG); realized-body manifest="
                          "typed_flat_block_dot_loop_body+block_fp16_scale_product+load+load+"
                          "widening_product+standalone_reduce+typed_flat_block_dot_loop_yield; "
                          "opaque_helper=false")},
        {"op": "quantize_row", "format": "q8_0", "state": "constructed",
         "auto_readout": "pending-E5-quant-stream: constructed (STRONG); manifest=...; opaque_helper=false"},
        {"op": "gemm_tile", "format": "q4_K", "state": "constructed", "auto_readout": None},
        {"op": "gemm_tile", "format": "iq2_xs", "state": "dispatch-wired", "auto_readout": None},
        {"op": "vec_dot", "format": "mxfp4", "state": "constructed-weak", "auto_readout": None},
        {"op": "dequantize_row", "format": "q1_0", "state": "absent", "auto_readout": None},
    ]}
    bypass = {"entries": [1, 2, 3, 4], "retired_ledger": [1] * 13}
    n = compute(doc, bypass)
    ok = True
    checks = [
        ("roster_total", n["roster_total"], 6),
        ("labeled", n["C_construct"]["labeled"], 3),
        ("certified", n["C_construct"]["certified"], 1),          # only the flat q8_0 passes
        ("red", n["C_construct"]["red"], 2),
        ("dispatch_wired", n["dispatch_wired"], 1),
        ("absent", n["absent"], 1),
        ("weak", n["constructed_weak_opaque_baseline"], 1),
        ("bypass_active", n["bypass_whitelist_active"], 4),
        ("bypass_retired", n["bypass_whitelist_retired"], 13),
        ("vein", n["mineral_vein_gemm_tile_dispatch_wired"], 1),
    ]
    print("-- emit_maturity_numbers aggregator self-test --")
    for name, got, want in checks:
        mark = "PASS" if got == want else "FAIL"
        ok = ok and got == want
        print(f"  [{mark}] {name}: {got} (expect {want})")
    print("SELF-TEST:", "GREEN" if ok else "RED")
    return 0 if ok else 1


# --------------------------------------------------------------------- runner
def main(argv):
    ap = argparse.ArgumentParser(description="DEBT-VIS canonical maturity-number emitter.")
    ap.add_argument("--ref", default="HEAD", help="committed ref to read the schema from")
    ap.add_argument("--worktree", action="store_true",
                    help="read the working-tree schema instead of a committed ref")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON snapshot")
    ap.add_argument("--drift", action="store_true", help="compare doc-cited numbers vs machine")
    ap.add_argument("--self-test", action="store_true", help="exercise the pure aggregator")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    root = repo_root()
    if args.worktree:
        doc = load_worktree(root, SCHEMA_REL)
        try:
            bypass = load_worktree(root, BYPASS_REL)
        except FileNotFoundError:
            bypass = {}
        ref_label = "WORKTREE"
    else:
        doc = load_from_ref(root, args.ref, SCHEMA_REL)
        try:
            bypass = load_from_ref(root, args.ref, BYPASS_REL)
        except subprocess.CalledProcessError:
            bypass = {}
        ref_label = f"{args.ref} ({short_ref(root, args.ref)})"

    nums = compute(doc, bypass)

    if args.drift:
        return run_drift(root, args.ref, nums)
    if args.json:
        print(json.dumps({"ref": ref_label, **nums}, ensure_ascii=False, indent=2))
        return 0
    print(render_human(nums, ref_label))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
