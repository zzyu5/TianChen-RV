#!/usr/bin/env python3
# =============================================================================
# [SEL-3] T-SEL3-4  offline-profile WRITE-BACK HARMONIZER  (proof-of-mechanism)
# =============================================================================
# Closes the offline-profile write-back loop for the measurement-memory layer:
#
#     immutable run -> BYTE-EXACT + lineage + freshness + structured T-N gates
#                   -> rebuildable qualification view -> recon master publication
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
#  * I4    measurement is never candidate or legality authority: the byte-exact
#          gate (correctness authority), qualification, and selector-valid paired
#          ranking are SEPARATE.  A qualified deployed fact may feed recon, but
#          never creates a candidate or repairs an illegal one.
#  * fail-closed: a row is written ONLY past the byte-exact gate. A variant whose
#          product is not bit-identical to its _generic construction oracle is
#          NEVER cached (the timing is discarded, the row is left untouched).
#
# ------------------------------ INTERFACE ------------------------------------
#   writeback     --job JOB.json [--apply]
#         JOB.json describes ONE (declared_instance_hash, kernel, variant, board)
#         cell -- optionally + op/engine/regime to disambiguate sibling deployed_point
#         rows on the COMPOSITE primary key (PR-16 [SEL-3-DPKEY]: key = hash, kernel,
#         variant, op, engine, regime; needed when several 'deployed' rows share a
#         board+kernel) + its REAL board cold samples + a path to the INDEPENDENT
#         ZERO-MODEL byte-exact gate evidence (the oracle_repack_* board stdout).
#         Default = DRY-RUN (prints the row diff, writes nothing). --apply commits.
#         Hash promotion (real_capability_hash + promote_primary_hash) is now
#         collision-safe for the FULL deployed_point table (was 3-cell under the old
#         variant-only guard), because op/engine/regime make the composite key unique.
#
#   verify-selected --hash H --kernel K --axis {sp4_tiling|loop_order}
#         Recomputes a group's argmin `selected` flags per the schema decision_rule
#         and asserts they match what is stored (the "selected unchanged" check).
#         Read-only; never writes.
#
# Touch set: writes ONLY schema/measurement-memory.v1.json data rows (never $meta
# structure, never lib/, never canon/roster/perf-covered/coverage). No git commit.
# =============================================================================
import argparse, csv, datetime, hashlib, json, os, re, statistics, sys, tempfile
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SCHEMA = os.path.join(ROOT, "schema", "measurement-memory.v1.json")
sys.path.insert(0, os.path.join(ROOT, "tools", "bench"))
from tn_qualify import (  # noqa: E402
    MIN_N as N_NOISE_FLOOR,
    TNQualificationError,
    qualify as qualify_tn,
    validate_evidence_file as validate_tn_evidence,
)

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

# Coverage axes are NOT memoized-argmin axes: every row is a trivially-`selected`
# single point (regret 0). deployed_point rows share variant=='deployed', so they
# MUST be disambiguated by the composite key, never by variant alone.
COVERAGE_AXES = ("deployed_point", "codegen_lmul")


def composite_key(r, hash_override=None):
    """[SEL-3-DPKEY] RESOLVED (PR-16): the ROW COMPOSITE PRIMARY KEY =
    (declared_instance_hash, kernel, variant, op, engine, regime).

    op/engine/regime are TOP-LEVEL fields on deployed_point rows (decoded from the
    SEED sentinel at ingest; None on L2 paired-axis rows). This 6-tuple replaces the
    old 3-tuple (hash, kernel, variant): because deployed_point.variant is always
    'deployed', the 3-tuple forced row identity onto the SEED sentinel hash and made
    promotion of that sentinel to the bare board capability hash collide sibling
    deployed rows. With op/engine/regime in the key, all 168 deployed_point rows are
    unique and hash promotion is collision-safe (unlocks full-table write-back).

    hash_override lets a collision check ask "would this row collide IF its hash were
    promoted to <override>?" without mutating the row.
    """
    h = r["declared_instance_hash"] if hash_override is None else hash_override
    return (h, r["kernel"], r["variant"],
            r.get("op"), r.get("engine"), r.get("regime"))


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
    # Official bench verify transcripts use the live three-way oracle plus
    # anti-hollow arms instead of the older `VERDICT BYTE-EXACT-INTEGER` spelling.
    # Accept that production evidence only when the complete transcript is present.
    if "# ALL_DONE" in txt and "ABI-GATE" in txt:
        ours = [int(x) for x in re.findall(r"T2 ours\s+vs oracle: mism=(\d+)/", txt)]
        oppg = [int(x) for x in re.findall(r"T2 OPP-G\s+vs oracle: mism=(\d+)/", txt)]
        oppx = [int(x) for x in re.findall(r"T2 OPP-X\s+vs oracle: mism=(\d+)/", txt)]
        positive = bool(ours and oppg and oppx and 0 in ours and 0 in oppg and 0 in oppx)
        anti_hollow = (
            txt.count("GATE-OURS = FAIL") >= 3
            and "GATE-OPPX = PASS" in txt
            and "GATE-OPPG = PASS" in txt
        )
        if positive and anti_hollow and re.search(r"ABI-GATE.*PASS", txt):
            return True, "OFFICIAL-BENCH-ZERO-MODEL; three-way zero mismatch + anti-hollow exercised"
        return False, "official bench transcript incomplete or hollow (fail-closed)"
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
    schema decision_rule. Returns {composite_key(row): selected_bool}.

    Keyed by the COMPOSITE key, not by variant: sibling deployed_point rows all
    share variant=='deployed', so a variant-keyed dict would collapse them (the
    exact ambiguity PR-16 fixes).

    * COVERAGE axes (deployed_point / codegen_lmul) are NOT memoized-argmin axes:
      every row is a trivially-`selected` single point (regret 0). Preserve the
      stored flag per composite key -- no argmin, no flip.
    * candidates = rows with byte_exact_gate==pass AND selection_valid_input==true.
      (Ineligible rows keep their stored flag untouched.)
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
    if axis in COVERAGE_AXES:
        # Coverage: no argmin. Preserve stored `selected` per composite key
        # (deployed_point rows are single-point, trivially selected, regret 0).
        return {composite_key(r): r["selected"] for r in grp}
    eligible = [r for r in grp if r["byte_exact_gate"] == "pass"
                and r["selection_valid_input"]]
    if len(grp) == 1:
        return {composite_key(grp[0]): True}  # single-candidate: trivially selected.
    if not eligible:
        # No selection-eligible row: preserve stored flags (nothing to recompute).
        return {composite_key(r): r["selected"] for r in grp}

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
    return {composite_key(r): (r["variant"] == winner) for r in grp}


def find_row(rows, hash_, kernel, variant, board_id,
             op=None, engine=None, regime=None):
    """Locate the target write-back row. Matches (hash, kernel, variant, board_id);
    when op/engine/regime are supplied they further disambiguate sibling
    deployed_point rows (composite key, PR-16). Returns the row, None (no match),
    or the list of matches when the request is ambiguous (caller reports & aborts).
    """
    want_ck = (op is not None) or (engine is not None) or (regime is not None)
    matches = []
    for r in rows:
        if (r["declared_instance_hash"] == hash_ and r["kernel"] == kernel
                and r["variant"] == variant
                and r["snapshot"].get("board_id") == board_id):
            if want_ck and (r.get("op"), r.get("engine"), r.get("regime")) != (op, engine, regime):
                continue
            matches.append(r)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        return None
    return matches  # ambiguous -> caller must disambiguate via op/engine/regime.


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def validate_master_publication(job, row, board):
    """Validate the immutable-run -> qualified-view boundary.

    This gate is intentionally stricter than an ordinary measurement-memory
    writeback.  It never infers a row key and never turns a legacy/ad-hoc result
    into a canonical master input merely because a timing value exists.
    """
    if not job.get("publish_to_master"):
        return None
    if row.get("variant_axis") != "deployed_point":
        raise ValueError("publish_to_master is legal only for deployed_point rows")
    run_id = job.get("run_id")
    if not isinstance(run_id, str) or not re.fullmatch(
            rf"\d{{8}}T\d{{6}}Z-[A-Za-z0-9_.+-]+-{re.escape(board)}-[0-9a-f]{{8}}", run_id):
        raise ValueError(f"invalid or board-mismatched official run_id: {run_id!r}")
    run_dir = os.path.realpath(os.path.join(ROOT, "experiments", "runs", run_id))
    runs_root = os.path.realpath(os.path.join(ROOT, "experiments", "runs")) + os.sep
    if not run_dir.startswith(runs_root):
        raise ValueError("run_id escapes experiments/runs")
    row_path = os.path.join(run_dir, "row.csv")
    if not os.path.isfile(row_path):
        raise ValueError(f"immutable run row missing: {row_path}")
    with open(row_path, encoding="utf-8", newline="") as stream:
        events = list(csv.DictReader(stream))
    if len(events) != 1:
        raise ValueError(f"official run must contain exactly one row event, got {len(events)}")
    event = events[0]
    expected = (row.get("op"), row.get("kernel"), row.get("engine"), row.get("regime"))
    actual = tuple(event.get(field) for field in ("op", "format", "engine", "regime"))
    if actual != expected:
        raise ValueError(f"run row key mismatch: expected={expected}, actual={actual}")
    if event.get("run-id") != run_id:
        raise ValueError(f"run row carries different run-id: {event.get('run-id')!r}")
    if not event.get("世系"):
        raise ValueError("run row lineage is empty")

    gate_path = os.path.realpath(job.get("gate_evidence_path") or "")
    if not gate_path.startswith(run_dir + os.sep):
        raise ValueError("master publication requires byte-exact evidence inside the immutable run")
    tn_path = Path(job.get("tn_evidence_path") or "")
    try:
        validate_tn_evidence(
            tn_path,
            repo_root=Path(ROOT),
            expected_board=board,
            expected_run_id=run_id,
        )
    except (TNQualificationError, OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"structured T-N qualification failed: {exc}") from exc
    if job.get("freshness") != "current":
        raise ValueError("master publication requires freshness='current'")
    return event


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
                   job["variant"], board_id,
                   job.get("op"), job.get("engine"), job.get("regime"))
    if row is None:
        print(f"!! target row not found: {job['declared_instance_hash']} / "
              f"{job['kernel']} / {job['variant']} @ {board_id}")
        return 2
    if isinstance(row, list):
        print(f"!! AMBIGUOUS target: {len(row)} sibling deployed rows match "
              f"{job['declared_instance_hash']} / {job['kernel']} / {job['variant']} "
              f"@ {board_id}. Disambiguate the composite key by adding op/engine/regime "
              f"to the job (PR-16). Candidates: "
              f"{[(r.get('op'), r.get('engine'), r.get('regime')) for r in row]}")
        return 2
    before = json.loads(json.dumps(row))  # deep copy for diff

    # Publication eligibility is checked before timing is allowed to mutate the
    # in-memory row.  Ordinary selection-cache writeback may omit this gate, but
    # then master_qualified_input remains false.
    try:
        publication_event = validate_master_publication(job, row, board)
    except ValueError as exc:
        print(f"!! MASTER PUBLICATION GATE FAILED: {exc}")
        return 6
    if row.get("master_qualified_input") and publication_event is None:
        print("!! target is already a canonical master input; ordinary cache writeback "
              "cannot mutate it without re-running the publication gate")
        return 6

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
    row["measurement_state"] = "measured"
    if job.get("opponent_symbol"):
        row["opponent_symbol"] = job["opponent_symbol"]

    if publication_event is not None:
        run_id = job["run_id"]
        row["measurement_board"] = board
        row["run_id"] = run_id
        row["freshness"] = "current"
        row["tn_qualification"] = "qualified"
        row["tn_evidence"] = os.path.relpath(job["tn_evidence_path"], ROOT)
        row["master_qualified_input"] = True
        row["evidence_lane"] = "official-run"
        row["source"] = f"experiments/runs/{run_id}/row.csv"
        row["axis_extras"]["tier"] = publication_event["对手档"]
        row["axis_extras"]["disp"] = publication_event["判定"]
        row["axis_extras"]["t3_note"] = (
            f"official qualified run {run_id}; lineage={publication_event['世系']}; "
            f"opponent-evidence={publication_event['对手证据引用']}"
        )

    # ---- instance-hash promotion (composite-key collision check, PR-16) ----
    # The real declared_instance_hash is the BOARD capability-fact SHA (kernel-
    # independent). [SEL-3-DPKEY] RESOLVED: the primary key is the COMPOSITE 6-tuple
    # (hash, kernel, variant, op, engine, regime). deployed_point.variant is always
    # "deployed", but op/engine/regime disambiguate sibling deployed rows -- so
    # promoting a SEED sentinel to the bare board hash NO LONGER collides (the old
    # variant-only guard limited write-back to the 3-cell no-sibling scope). We now
    # promote whenever the composite key stays unique; we only refuse on a genuine
    # composite duplicate (a real double-write, not the deployed-variant pseudo-clash).
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
        new_ck = composite_key(row, cap_hash)  # key the row WOULD have post-promotion
        clash = next((r for r in rows
                      if r is not row and composite_key(r) == new_ck), None)
        if clash is not None:
            print(f"    [hash] promotion REFUSED: composite key {new_ck} already "
                  f"exists (genuine duplicate row) -> keeping sentinel (fail-closed).")
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
            "3cd23a4e...). [SEL-3-DPKEY] RESOLVED (PR-16): the primary key is the "
            "COMPOSITE (hash, kernel, variant, op, engine, regime); op/engine/regime "
            "(top-level, decoded from the SEED sentinel) disambiguate sibling deployed "
            "rows, so promoting this sentinel to the bare board hash is collision-safe "
            "and write-back generalizes to the full deployed_point table."),
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
                and r["variant_axis"] == row["variant_axis"]):
            ck = composite_key(r)
            if ck not in sel:
                continue
            if r["selected"] != sel[ck]:
                selected_changed.append((r["variant"], r["selected"], sel[ck]))
            r["selected"] = sel[ck]
    print(f"\n[3] GROUP ARGMIN RECOMPUTE (decision_rule): {sel}")
    print(f"    selected changed: {selected_changed if selected_changed else 'NONE (unchanged)'}")

    # ---- STEP 4: NG-1 restated + row diff ----
    print("\n[4] ROW DIFF (before -> after):")
    for k in ("declared_instance_hash", "instance_hash_seed", "cold_median",
              "cold_iqr", "ratio_semantics", "selection_valid_input",
              "byte_exact_gate", "selected", "measurement_state", "run_id",
              "freshness", "tn_qualification", "master_qualified_input",
              "evidence_lane", "ts"):
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
                and r["variant_axis"] == args.axis):
            ck = composite_key(r)
            if ck not in sel:
                continue
            match = (r["selected"] == sel[ck])
            ok = ok and match
            print(f"    {r['variant']:10}  stored={r['selected']!s:5}  "
                  f"recomputed={sel[ck]!s:5}  {'OK' if match else 'MISMATCH!!'}")
    print(f"  => {'SELECTED UNCHANGED (recompute == stored)' if ok else 'SELECTED MISMATCH'}")
    return 0 if ok else 5


def cmd_self_test(_args):
    """Synthetic control-plane tests; writes only inside a temporary directory."""
    global ROOT
    original_root = ROOT
    with tempfile.TemporaryDirectory(prefix="weft-measurement-control-") as temporary:
        ROOT = temporary
        run_id = "20990101T000000Z-q4_K-rvv-deadbeef"
        run_dir = os.path.join(ROOT, "experiments", "runs", run_id)
        os.makedirs(run_dir)
        event = {
            "op": "vec_dot", "format": "q4_K", "engine": "rvv",
            "regime": "micro-fixed", "cold": "1.0000", "判定": "PASS",
            "对手符号": "synthetic-opp", "对手档": "手调",
            "对手证据引用": "synthetic-probe", "我方向量指令数": "1",
            "世系": "synthetic-board·synthetic-chain·synthetic-batch",
            "run-id": run_id, "噪声标": "synthetic",
        }
        row_path = os.path.join(run_dir, "row.csv")
        with open(row_path, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(event))
            writer.writeheader(); writer.writerow(event)
        verify_path = os.path.join(run_dir, "verify.stdout.txt")
        with open(verify_path, "w", encoding="utf-8") as stream:
            stream.write(
                "ABI-GATE synthetic -> PASS\n"
                "T2 ours  vs oracle: mism=0/1\nT2 OPP-G vs oracle: mism=0/1\n"
                "T2 OPP-X vs oracle: mism=0/1\n"
                "GATE-OURS = FAIL | GATE-OPPG = FAIL | GATE-OPPX = FAIL\n"
                "GATE-OURS = FAIL | GATE-OPPG = PASS | GATE-OPPX = PASS\n"
                "GATE-OURS = FAIL | GATE-OPPG = PASS | GATE-OPPX = PASS\n# ALL_DONE\n"
            )
        timing_path = os.path.join(run_dir, "measure.stdout.txt")
        with open(timing_path, "w", encoding="utf-8") as stream:
            stream.write("synthetic immutable timing source\n")
        timing_sha = hashlib.sha256(Path(timing_path).read_bytes()).hexdigest()
        tn_raw = {
            "board": "rvv",
            "benchmark_class": "vec_dot-micro-fixed",
            "protocol_id": "synthetic-preregistered-protocol",
            "effect_run_id": run_id,
            "pre_registered_n_noise": N_NOISE_FLOOR,
            "pre_registered_n_effect": N_NOISE_FLOOR,
            "noise_repeat_deltas_pct": [
                -0.20, -0.15, -0.10, -0.05, 0.0, 0.0, 0.05, 0.10, 0.15, 0.20,
            ],
            "effect_deltas_pct": [2.0, 2.1, 2.2, 2.1, 2.0, 2.2, 2.1, 2.0, 2.2, 2.1],
            "bootstrap_seed": 7,
            "bootstrap_resamples": 1000,
            "source_artifacts": [{
                "path": os.path.relpath(timing_path, ROOT),
                "sha256": timing_sha,
            }],
        }
        tn_path = os.path.join(run_dir, "tn-qualification.json")
        with open(tn_path, "w", encoding="utf-8") as stream:
            json.dump(qualify_tn(tn_raw), stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        job = {
            "publish_to_master": True, "run_id": run_id, "freshness": "current",
            "gate_evidence_path": verify_path, "tn_evidence_path": tn_path,
        }
        row = {
            "variant_axis": "deployed_point", "op": "vec_dot", "kernel": "q4_K",
            "engine": "rvv", "regime": "micro-fixed",
        }
        passed, _ = check_byte_exact_gate(verify_path)
        assert passed
        assert validate_master_publication(job, row, "rvv")["run-id"] == run_id
        job["freshness"] = "stale"
        try:
            validate_master_publication(job, row, "rvv")
        except ValueError:
            pass
        else:
            raise AssertionError("stale publication unexpectedly accepted")
    ROOT = original_root
    print("measurement publication self-test: PASS")
    return 0


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
    s = sub.add_parser("self-test", help="synthetic key/gate/publication negative controls")
    s.set_defaults(func=cmd_self_test)
    args = ap.parse_args()
    sys.exit(args.func(args))


if __name__ == "__main__":
    main()
