#!/usr/bin/env python3
# =============================================================================
# [SEL-3] T-SEL3-4  offline-profile WRITE-BACK HARMONIZER  (proof-of-mechanism)
# =============================================================================
# Closes the offline-profile write-back loop for the measurement-memory layer:
#
#     board cold-measure  ->  BYTE-EXACT gate (fail-closed)  ->  write library row
#                          ->  recompute group argmin (`selected`)
#
# It transcribes REAL board measurements into schema/measurement-memory.v1.json.
# It is NOT a selector, NOT a loader; lib/ consumes NOTHING from this file.
#
# ------------------------- IRON LINES (NG-1 / [L-4] / I4) ---------------------
#  * NG-1  no-search / no-learning: this tool is an OFFLINE BATCH TRANSCRIBER.
#          It holds NO learnable parameters, runs NO autotune search, fits NO
#          curve / cost model. It maps (real board samples) -> (median, IQR) and
#          writes them. The variant set is a bounded compile-time enum owned by
#          the selector; this tool never invents a variant.
#  * [L-4] no external-tuner benchmark: the oracle is always SELF (the per-cell
#          _generic construction oracle). This tool never imports/compares an
#          external tuner's cost model.
#  * I4    measurement is a CACHE FACT, never an authority: the byte-exact gate
#          (correctness authority) and the library (ranking cache) are SEPARATE.
#          A cold_median here only RANKS; it never decides correctness.
#  * fail-closed: a row is written ONLY past the byte-exact gate. A variant whose
#          product is not bit-identical to its _generic construction oracle is
#          NEVER cached (the timing is discarded, the row is left untouched).
#
# ------------------------------ INTERFACE ------------------------------------
#   writeback     --job JOB.json [--apply]
#         JOB.json describes ONE (declared_instance_hash, kernel, variant, board)
#         cell + its REAL board cold samples + a path to the INDEPENDENT
#         ZERO-MODEL byte-exact gate evidence (the oracle_repack_* board stdout).
#         Default = DRY-RUN (prints the row diff, writes nothing). --apply commits.
#
#   verify-selected --hash H --kernel K --axis {sp4_tiling|loop_order}
#         Recomputes a group's argmin `selected` flags per the schema decision_rule
#         and asserts they match what is stored (the "selected unchanged" check).
#         Read-only; never writes.
#
# Touch set: writes ONLY schema/measurement-memory.v1.json data rows (never $meta
# structure, never lib/, never canon/roster/perf-covered/coverage). No git commit.
# =============================================================================
import argparse, datetime, json, os, re, statistics, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA = os.path.join(ROOT, "schema", "measurement-memory.v1.json")

# The @rvv / @k1 declared-instance hashes = the BOARD capability-fact-set SHA-256
# (kernel-INDEPENDENT: every fixture kernel @a board expands to the SAME capability
# facts -> the SAME hash). The @rvv value is the one the L2 rows + the selector's
# lookupMeasurement already carry (support::computeDeclaredInstanceHash of the @rvv
# capability facts). Used only when a job asks to promote a SEED sentinel hash to
# the real board hash (instance_hash_seed true->false).
BOARD_INSTANCE_HASH = {
    "rvv": "3cd23a4ec9796a3ce1f863cd80c96b894267ab95b45cb0ecfeb856cc643b58c7",
    # k1 real hash not asserted here (no k1 fixture-hash landed in-tree yet); a k1
    # job must supply real_capability_hash explicitly or keep the seed sentinel.
}

N_NOISE_FLOOR = 10  # T-N: >= 10 cold reps required before a median/IQR is trusted.


def load():
    with open(SCHEMA, encoding="utf-8") as f:
        return json.load(f)


def dump(doc):
    with open(SCHEMA, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, ensure_ascii=False)
        f.write("\n")


# ------------------------- byte-exact gate (fail-closed) ----------------------
def check_byte_exact_gate(evidence_path):
    """ZERO-MODEL gate: parse the INDEPENDENT construction-oracle board stdout.

    Returns (passed: bool, reason: str). PASS requires the oracle to report the
    BYTE-EXACT-INTEGER verdict, zero int-mismatch on every shape, and every
    negative control EXERCISED (cert three-requirements: corpus-complete /
    same-input-path / independent-oracle). Anything else => fail-closed.
    """
    if not evidence_path or not os.path.exists(evidence_path):
        return False, f"gate evidence file missing: {evidence_path!r}"
    txt = open(evidence_path, encoding="utf-8", errors="replace").read()
    if "VERDICT BYTE-EXACT-INTEGER" not in txt:
        return False, "oracle verdict is not BYTE-EXACT-INTEGER (fail-closed)"
    # Parse LINE-AWARE: a negative-control line (contains "perturbed" / "EXERCISED")
    # is SUPPOSED to carry a large mismatch; only the POSITIVE certificate lines
    # (shape / GEMM-interleaved) must be exactly 0. Do not conflate the two.
    pos_certs, neg_controls = [], []
    for line in txt.splitlines():
        is_control = ("perturbed" in line) or ("EXERCISED" in line)
        m = re.search(r"int-mismatch\s*=\s*(\d+)", line)
        if is_control:
            neg_controls.append(line)
        elif m:
            pos_certs.append(int(m.group(1)))
    if not pos_certs:
        return False, "no positive int-mismatch certificate lines found (fail-closed)"
    nonzero = [c for c in pos_certs if c != 0]
    if nonzero:
        return False, f"{len(nonzero)} positive certificate(s) with nonzero mismatch (fail-closed)"
    # Every negative control must be genuinely EXERCISED (guards a hollow oracle).
    if not neg_controls:
        return False, "no negative controls present (hollow-oracle guard, fail-closed)"
    if any("NOT EXERCISED" in c for c in neg_controls) or \
       not all("EXERCISED" in c for c in neg_controls):
        return False, "a negative control was NOT EXERCISED (hollow-oracle guard, fail-closed)"
    return True, (f"BYTE-EXACT-INTEGER; {len(pos_certs)} positive certs all 0-mismatch; "
                  f"{len(neg_controls)} negative controls all EXERCISED")


# ------------------------- median / IQR from real samples ---------------------
def median_iqr(samples):
    xs = sorted(float(s) for s in samples)
    n = len(xs)

    def pct(p):  # linear-interpolation percentile (numpy-free)
        if n == 1:
            return xs[0]
        idx = (p / 100.0) * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (idx - lo)

    return statistics.median(xs), (pct(75) - pct(25))


# ------------------------- selection_valid_input rule -------------------------
def compute_selection_valid_input(variant_axis, ratio_semantics, snapshot):
    """[SEL-3-SELVALID] conservative (paired-variant) reading + compiler symmetry.

    A cold_median is a memoized-argmin SELECTION INPUT iff it is (a) a paired
    A/B of our own variants on the same axis, AND (b) compiler-symmetric.
      * deployed_point / any single-point (no sibling variant)  -> False.
      * vs_shipped_opp (cross-compiler on rvv: ours-clang vs gcc-shipped) -> False.
      * ab_paired / vs_generic on a paired L2 axis, same-compiler -> True.
    """
    if variant_axis not in ("sp4_tiling", "loop_order"):
        return False  # no paired sibling variant to select among.
    if ratio_semantics not in ("ab_paired", "vs_generic"):
        return False  # vs_shipped_opp is disclosure-only, never a selection input.
    tc = snapshot.get("toolchain", {})
    if tc.get("ours") and tc.get("opp_shipped") and tc["ours"] != tc["opp_shipped"]:
        return False  # compiler-ASYMMETRIC => disclosure-only.
    return True


# ------------------------- group argmin (decision_rule) -----------------------
def recompute_selected(rows, hash_, kernel, axis):
    """Recompute `selected` for a (hash, kernel, variant_axis) group per the
    schema decision_rule. Returns {variant: selected_bool}.

    * candidates = rows with byte_exact_gate==pass AND selection_valid_input==true.
      (Ineligible rows keep their stored flag untouched; single-candidate /
      coverage groups are trivially `selected` per the schema.)
    * best cold_median wins; on a tie the SIMPLER variant wins.
    * sp4_tiling: cold_median is a GROUP-SHARED A/B ratio, so it always ties ->
      the register_cliff_reached STRUCTURAL gate breaks the tie (s6_tiled iff the
      cliff is reached; else plain). Mirrors the selector header CAVEAT + the
      "更简单者胜 unless the cliff gate flips it" rule.
    * loop_order: cold_median > 1 means col_outer faster -> col_outer wins; the
      loser row carries the same shared ratio and is decided by direction.
    """
    grp = [r for r in rows if r["declared_instance_hash"] == hash_
           and r["kernel"] == kernel and r["variant_axis"] == axis]
    if not grp:
        return {}
    eligible = [r for r in grp if r["byte_exact_gate"] == "pass"
                and r["selection_valid_input"]]
    out = {}
    if len(grp) == 1:
        out[grp[0]["variant"]] = True  # single-candidate: trivially selected.
        return out
    if not eligible:
        # No selection-eligible row: preserve stored flags (nothing to recompute).
        return {r["variant"]: r["selected"] for r in grp}

    simpler = {"sp4_tiling": "plain", "loop_order": "row_outer"}.get(axis)
    if axis == "sp4_tiling":
        s6 = next((r for r in grp if r["variant"] == "s6_tiled"), None)
        cliff = bool(s6 and s6["axis_extras"].get("register_cliff_reached"))
        winner = "s6_tiled" if cliff else "plain"
    elif axis == "loop_order":
        co = next((r for r in grp if r["variant"] == "col_outer"), None)
        ratio = float(co["cold_median"]) if co and co["cold_median"] is not None else 1.0
        winner = "col_outer" if ratio > 1.0 else (simpler or "row_outer")
    else:
        # Generic: best (largest, ours-faster convention) cold_median; tie->simpler.
        best = max(eligible, key=lambda r: (float(r["cold_median"]), r["variant"] == simpler))
        winner = best["variant"]
    for r in grp:
        out[r["variant"]] = (r["variant"] == winner)
    return out


def find_row(rows, hash_, kernel, variant, board_id):
    for r in rows:
        if (r["declared_instance_hash"] == hash_ and r["kernel"] == kernel
                and r["variant"] == variant
                and r["snapshot"].get("board_id") == board_id):
            return r
    return None


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------- writeback ----------------------------------
def cmd_writeback(args):
    job = json.load(open(args.job, encoding="utf-8"))
    print("=" * 78)
    print("[SEL-3 T-SEL3-4] WRITE-BACK HARMONIZER  (NG-1 offline batch transcriber;")
    print("  no search / no learned cost model / no external-tuner benchmark; fail-closed)")
    print("=" * 78)

    board = job["board"]
    board_id = {"rvv": "openEuler-VLEN128", "k1": "SpacemiT-K1-VLEN256"}[board]
    doc = load()
    rows = doc["rows"]
    row = find_row(rows, job["declared_instance_hash"], job["kernel"],
                   job["variant"], board_id)
    if row is None:
        print(f"!! target row not found: {job['declared_instance_hash']} / "
              f"{job['kernel']} / {job['variant']} @ {board_id}")
        return 2
    before = json.loads(json.dumps(row))  # deep copy for diff

    # ---- STEP 1: byte-exact gate (fail-closed COMMANDING gate) ----
    passed, reason = check_byte_exact_gate(job.get("gate_evidence_path"))
    print(f"\n[1] BYTE-EXACT GATE (ZERO-MODEL, fail-closed): "
          f"{'PASS' if passed else 'FAIL'}")
    print(f"    oracle evidence : {job.get('gate_evidence_path')}")
    print(f"    verdict         : {reason}")
    if not passed:
        print("\n!! GATE FAILED -> FAIL-CLOSED: nothing written, timing discarded, "
              "row left untouched.")
        return 3

    # ---- STEP 2: median / IQR from REAL board samples (N >= noise floor) ----
    samples = job["samples"]
    if len(samples) < N_NOISE_FLOOR:
        print(f"\n!! only {len(samples)} samples (< noise floor {N_NOISE_FLOOR}) "
              "-> refuse to write (fail-closed).")
        return 4
    med, iqr = median_iqr(samples)
    print(f"\n[2] REAL BOARD COLD: N={len(samples)}  median={med:.4f}  IQR={iqr:.4f}")
    print(f"    sample semantics: {job.get('sample_semantics')}")

    # ---- prepare the written row ----
    ratio_semantics = job.get("ratio_semantics", row["ratio_semantics"])
    sv = compute_selection_valid_input(row["variant_axis"], ratio_semantics,
                                       row["snapshot"])
    ts = utc_now()

    row["cold_median"] = round(med, 6)
    row["cold_iqr"] = round(iqr, 6)
    row["ratio_semantics"] = ratio_semantics
    row["selection_valid_input"] = sv
    row["byte_exact_gate"] = "pass"
    row["ts"] = ts
    if job.get("opponent_symbol"):
        row["opponent_symbol"] = job["opponent_symbol"]

    # ---- instance-hash promotion (collision-guarded) ----
    # The real declared_instance_hash is the BOARD capability-fact SHA (kernel-
    # independent). For paired L2 axes it uniquely keys the row. For deployed_point
    # rows the SEED sentinel ALSO encodes op/engine/regime -- the ONLY discriminator
    # among sibling deployed rows (primary_key = hash+kernel+variant, and variant is
    # always "deployed"). Promoting the sentinel to the bare board hash would risk a
    # primary-key COLLISION with a sibling deployed row on the same board. So we
    # promote ONLY when it is collision-safe; otherwise we KEEP the sentinel (it is
    # the row's identity) and record the real capability hash in the breadcrumb.
    # (deployed_point keying is registered as a PENDING ruling.)
    hash_promoted = False
    cap_hash = job.get("real_capability_hash")
    # Guard: a job's real_capability_hash must match the known board capability hash
    # (kernel-independent fact) when one is on file -- catches a typo'd hash.
    known = BOARD_INSTANCE_HASH.get(board)
    if cap_hash and known and cap_hash != known:
        print(f"\n!! real_capability_hash {cap_hash!r} != known @{board} board hash "
              f"{known!r} -> refuse (fail-closed).")
        return 5
    if cap_hash and job.get("promote_primary_hash"):
        collides = any(
            r is not row and r["declared_instance_hash"] == cap_hash
            and r["kernel"] == row["kernel"] and r["variant"] == row["variant"]
            for r in rows)
        if collides:
            print(f"    [hash] promotion REFUSED (would collide with a sibling "
                  f"{row['kernel']}/{row['variant']} row) -> keeping sentinel.")
        else:
            row["declared_instance_hash"] = cap_hash
            row["instance_hash_seed"] = False
            hash_promoted = True
    if job.get("source"):
        row["source"] = job["source"]
    # writeback provenance breadcrumb in axis_extras (does not alter axis schema).
    row["axis_extras"]["writeback"] = {
        "harness": "SEL3 T-SEL3-4 sel3_writeback_harmonizer.py",
        "board": board,
        "cold_samples_n": len(samples),
        "cold_samples": [round(float(s), 6) for s in samples],
        "sample_semantics": job.get("sample_semantics"),
        "gate_evidence": job.get("gate_evidence_path"),
        "gate_verdict": "BYTE-EXACT-INTEGER (ZERO-MODEL oracle_repack, 0-mismatch, "
                        "negative controls EXERCISED)",
        "prev_placeholder_cold": before["cold_median"],
        "prev_byte_exact_gate": before["byte_exact_gate"],
        "real_capability_hash": cap_hash,
        "real_capability_hash_promoted_to_primary_key": hash_promoted,
        "real_capability_hash_note": (
            "kernel-independent @-board capability-fact SHA (== the L2 @rvv rows' "
            "3cd23a4e...). NOT promoted to the primary key for deployed_point rows: "
            "the SEED sentinel additionally encodes op/engine/regime, the only "
            "discriminator among sibling deployed rows (see PENDING [SEL-3-DPKEY])."),
        "protocol_note": job.get("protocol_note"),
        "ng1_note": "offline batch transcription of real board samples; no search, "
                    "no learned cost model, no external-tuner benchmark.",
    }

    # ---- STEP 3: recompute group argmin (`selected`) ----
    sel = recompute_selected(rows, row["declared_instance_hash"], row["kernel"],
                             row["variant_axis"])
    selected_changed = []
    for r in rows:
        if (r["declared_instance_hash"] == row["declared_instance_hash"]
                and r["kernel"] == row["kernel"]
                and r["variant_axis"] == row["variant_axis"]
                and r["variant"] in sel):
            if r["selected"] != sel[r["variant"]]:
                selected_changed.append((r["variant"], r["selected"], sel[r["variant"]]))
            r["selected"] = sel[r["variant"]]
    print(f"\n[3] GROUP ARGMIN RECOMPUTE (decision_rule): {sel}")
    print(f"    selected changed: {selected_changed if selected_changed else 'NONE (unchanged)'}")

    # ---- STEP 4: NG-1 restated + row diff ----
    print("\n[4] ROW DIFF (before -> after):")
    for k in ("declared_instance_hash", "instance_hash_seed", "cold_median",
              "cold_iqr", "ratio_semantics", "selection_valid_input",
              "byte_exact_gate", "selected", "ts"):
        b = before.get(k)
        a = row.get(k)
        flag = "" if b == a else "   <== CHANGED"
        print(f"    {k:24}: {b!r:>40}  ->  {a!r}{flag}")

    if not args.apply:
        print("\n(DRY-RUN: nothing written. re-run with --apply to commit.)")
        return 0

    # round-trip guard before commit.
    dump(doc)
    back = load()
    assert back == doc, "ROUND-TRIP MISMATCH after write"
    print(f"\nWROTE {SCHEMA} (round-trip clean).")
    return 0


# ------------------------------ verify-selected -------------------------------
def cmd_verify_selected(args):
    doc = load()
    rows = doc["rows"]
    sel = recompute_selected(rows, args.hash, args.kernel, args.axis)
    if not sel:
        print(f"!! no group for hash={args.hash} kernel={args.kernel} axis={args.axis}")
        return 2
    print(f"[verify-selected] group ({args.kernel}, {args.axis}):")
    ok = True
    for r in rows:
        if (r["declared_instance_hash"] == args.hash and r["kernel"] == args.kernel
                and r["variant_axis"] == args.axis and r["variant"] in sel):
            match = (r["selected"] == sel[r["variant"]])
            ok = ok and match
            print(f"    {r['variant']:10}  stored={r['selected']!s:5}  "
                  f"recomputed={sel[r['variant']]!s:5}  {'OK' if match else 'MISMATCH!!'}")
    print(f"  => {'SELECTED UNCHANGED (recompute == stored)' if ok else 'SELECTED MISMATCH'}")
    return 0 if ok else 5


def main():
    ap = argparse.ArgumentParser(description="[SEL-3] T-SEL3-4 write-back harmonizer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("writeback", help="gate + write one cell + recompute selected")
    w.add_argument("--job", required=True)
    w.add_argument("--apply", action="store_true", help="commit (default dry-run)")
    w.set_defaults(func=cmd_writeback)
    v = sub.add_parser("verify-selected", help="recompute + assert a group's selected")
    v.add_argument("--hash", required=True)
    v.add_argument("--kernel", required=True)
    v.add_argument("--axis", required=True, choices=["sp4_tiling", "loop_order"])
    v.set_defaults(func=cmd_verify_selected)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
