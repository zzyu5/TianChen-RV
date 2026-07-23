#!/usr/bin/env python3
"""B1 gate: one master publisher, explicit keys, separated qualification axes."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "tools" / "bench" / "bench"
CURRENT_ARTIFACT_EXPORTER = ROOT / "tools" / "bench" / "export_current_artifact.py"
RECON = ROOT / ".trellis" / "scripts" / "recon_master_rebuild.py"
HARMONIZER = ROOT / ".trellis" / "scripts" / "sel3_writeback_harmonizer.py"
DISPOSITION = ROOT / ".trellis" / "scripts" / "recon_t3_disposition.py"
MEMORY = ROOT / "schema" / "measurement-memory.v1.json"
ROSTER = ROOT / "schema" / "coverage-roster.v1.json"
MASTER = ROOT / "experiments" / "master" / "T3_master_rebuild.csv"
RUNS = ROOT / "experiments" / "runs"
RUNS_LOG = ROOT / "experiments" / "runs.log"
sys.path.insert(0, str(ROOT / "tools" / "bench"))
from measurement_keys import (  # noqa: E402
    MeasurementKeyError,
    key_from_mapping,
    load_master_keys,
    load_roster_keys,
)
from tn_qualify import qualify as qualify_tn  # noqa: E402


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_qualified_override_bridge() -> None:
    """Hermetic official-run -> structured T-N -> recon-input proof."""
    spec = importlib.util.spec_from_file_location("b1_recon", RECON)
    require(spec is not None and spec.loader is not None, "cannot load recon module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory(prefix="weft-b1-qualified-") as directory:
        root = Path(directory)
        run_id = "20990101T000000Z-q4_K-rvv-deadbeef"
        run_dir = root / "experiments" / "runs" / run_id
        run_dir.mkdir(parents=True)
        event = {
            "op": "vec_dot", "format": "q4_K", "engine": "rvv",
            "regime": "micro-fixed", "cold": "1.0", "判定": "PASS",
            "对手符号": "synthetic-opp", "对手档": "手调",
            "对手证据引用": "synthetic-probe", "我方向量指令数": "1",
            "世系": "synthetic-board·synthetic-chain·synthetic-batch",
            "run-id": run_id, "噪声标": "synthetic",
        }
        with (run_dir / "row.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(event), lineterminator="\n")
            writer.writeheader(); writer.writerow(event)
        timing = run_dir / "measure.stdout.txt"
        timing.write_text("synthetic immutable timing source\n", encoding="utf-8")
        raw_tn = {
            "board": "rvv", "benchmark_class": "vec_dot-micro-fixed",
            "protocol_id": "synthetic-preregistered-protocol",
            "effect_run_id": run_id,
            "pre_registered_n_noise": 10, "pre_registered_n_effect": 10,
            "noise_repeat_deltas_pct": [
                -0.20, -0.15, -0.10, -0.05, 0.0, 0.0, 0.05, 0.10, 0.15, 0.20,
            ],
            "effect_deltas_pct": [2.0, 2.1, 2.2, 2.1, 2.0, 2.2, 2.1, 2.0, 2.2, 2.1],
            "bootstrap_seed": 7, "bootstrap_resamples": 1000,
            "source_artifacts": [{
                "path": str(timing.relative_to(root)),
                "sha256": hashlib.sha256(timing.read_bytes()).hexdigest(),
            }],
        }
        tn_path = run_dir / "tn-qualification.json"
        tn_path.write_text(json.dumps(qualify_tn(raw_tn), indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
        qualified = {
            "variant_axis": "deployed_point", "op": "vec_dot", "kernel": "q4_K",
            "engine": "rvv", "regime": "micro-fixed",
            "master_qualified_input": True, "byte_exact_gate": "pass",
            "measurement_state": "measured", "freshness": "current",
            "tn_qualification": "qualified", "evidence_lane": "official-run",
            "measurement_board": "rvv", "run_id": run_id,
            "tn_evidence": str(tn_path.relative_to(root)), "cold_median": 1.0,
            "axis_extras": {"tier": "手调", "disp": "PASS", "t3_note": "synthetic"},
            "opponent_symbol": {"symbol": "synthetic-opp", "caliber": "hand-brick"},
        }
        memory = root / "schema" / "measurement-memory.v1.json"
        memory.parent.mkdir(parents=True)
        memory.write_text(json.dumps({"rows": [qualified]}, ensure_ascii=False),
                          encoding="utf-8")
        module.ROOT = root
        module.MEASUREMENT_MEMORY = memory
        overrides = module.load_qualified_overrides()
        expected = ("vec_dot", "q4_K", "rvv", "micro-fixed", "rvv")
        require(expected in overrides, "qualified official run did not reach recon input")

        qualified["freshness"] = "stale"
        memory.write_text(json.dumps({"rows": [qualified]}, ensure_ascii=False),
                          encoding="utf-8")
        try:
            module.load_qualified_overrides()
        except MeasurementKeyError:
            pass
        else:
            raise AssertionError("stale qualified override unexpectedly reached recon")


def main() -> int:
    bench = BENCH.read_text(encoding="utf-8")
    recon = RECON.read_text(encoding="utf-8")
    harmonizer = HARMONIZER.read_text(encoding="utf-8")
    disposition = DISPOSITION.read_text(encoding="utf-8")
    require("def update_master_cell" not in bench, "retired bench master writer returned")
    require("guarded_open(MASTER_TABLE" not in bench, "bench can write canonical master")
    require("rk == regime or rk ==" not in bench, "empty-regime wildcard returned")
    require("atomic_publish(OUT" in recon and "publisher_lock" in recon,
            "recon publication is not atomic/locked")
    require("validate_tn_evidence" in recon,
            "recon trusts qualification flags without revalidating T-N evidence")
    require('"experiments" / "master"' in disposition,
            "disposition reader default is not canonical master")
    require("active\" / \"result-tables" not in disposition,
            "disposition reader retained old-path fallback")
    require("validate_tn_evidence" in harmonizer,
            "master publication does not validate structured T-N evidence")
    require("required_tn_tokens" not in harmonizer,
            "free-text T-N token shortcut returned")

    bench_self_test = subprocess.run(
        [str(BENCH), "--self-test"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    require(bench_self_test.returncode == 0,
            f"canonical bench contract self-test failed: {bench_self_test.stderr}")
    exporter_self_test = subprocess.run(
        [sys.executable, str(CURRENT_ARTIFACT_EXPORTER), "--self-test"], cwd=ROOT,
        capture_output=True, text=True, check=False,
    )
    require(exporter_self_test.returncode == 0,
            f"current-artifact route/recipe closure failed: {exporter_self_test.stderr}")

    # Dry-run shares the formal full-key signature but is genuinely read-only:
    # no phantom run-id, placeholder row, append-only log pollution, or old
    # one-positional compatibility route.
    before_master = MASTER.read_bytes()
    before_log = RUNS_LOG.read_bytes()
    before_runs = sorted(path.name for path in RUNS.iterdir())
    dry = subprocess.run(
        [str(BENCH), "vec_dot", "q4_K", "--board", "rvv", "--engine", "rvv",
         "--regime", "micro-fixed", "--dry-run"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    require(dry.returncode == 0, f"full-key dry-run failed: {dry.stderr}")
    require("未分配 run-id" in dry.stdout, "dry-run allocated a phantom run-id")
    require(MASTER.read_bytes() == before_master, "dry-run mutated canonical master")
    require(RUNS_LOG.read_bytes() == before_log, "dry-run polluted append-only runs.log")
    require(sorted(path.name for path in RUNS.iterdir()) == before_runs,
            "dry-run created an immutable-run directory")

    grid_prefill = subprocess.run(
        [str(BENCH), "gemm_tile", "iq2_xxs", "--board", "rvv",
         "--engine", "rvv", "--regime", "prefill", "--dry-run"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    require(grid_prefill.returncode == 0,
            f"implemented grid prefill route failed: {grid_prefill.stderr}")
    grid_decode = subprocess.run(
        [str(BENCH), "gemm_tile", "iq2_xxs", "--board", "rvv",
         "--engine", "rvv", "--regime", "decode", "--dry-run"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    require(grid_decode.returncode != 0,
            "grid decode reached a fixed-prefill harness without a true decode workload")
    require(MASTER.read_bytes() == before_master, "regime route checks mutated canonical master")
    require(RUNS_LOG.read_bytes() == before_log, "regime route checks polluted runs.log")
    require(sorted(path.name for path in RUNS.iterdir()) == before_runs,
            "regime route checks created an immutable-run directory")

    shortcut = subprocess.run(
        [str(BENCH), "q4_K", "--board", "rvv", "--dry-run"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    require(shortcut.returncode != 0, "retired one-positional dry-run route returned")
    legacy_writer = subprocess.run(
        [str(BENCH), "vec_dot", "q4_K", "--board", "rvv", "--engine", "rvv",
         "--regime", "micro-fixed", "--update-master"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    require(legacy_writer.returncode != 0, "retired --update-master alias returned")

    roster_keys = load_roster_keys(ROSTER)
    master_keys = load_master_keys(MASTER)
    require(set(master_keys) == set(roster_keys), "master/roster exact-key sets differ")

    doc = json.load(open(MEMORY, encoding="utf-8"))
    deployed = []
    identities = set()
    for index, row in enumerate(doc.get("rows", [])):
        if row.get("variant_axis") != "deployed_point":
            continue
        key = key_from_mapping(
            {"op": row.get("op"), "format": row.get("kernel"),
             "engine": row.get("engine"), "regime": row.get("regime")},
            source=f"{MEMORY}:rows[{index}]",
        )
        board = row.get("measurement_board")
        require(board in ("rvv", "k1"), f"deployed row {index} has invalid board")
        identity = (*key, board)
        require(identity not in identities, f"duplicate deployed identity {identity}")
        identities.add(identity); deployed.append(row)
        for field in ("measurement_state", "freshness", "tn_qualification",
                      "master_qualified_input", "selection_valid_input", "evidence_lane"):
            require(field in row, f"deployed row {identity} lacks {field}")
        if row["master_qualified_input"]:
            require(row.get("byte_exact_gate") == "pass", f"qualified row {identity} not byte-exact")
            require(row.get("freshness") == "current", f"qualified row {identity} is not current")
            require(row.get("tn_qualification") == "qualified", f"qualified row {identity} lacks T-N")
            require(row.get("run_id"), f"qualified row {identity} lacks run-id")
    require(all(not row["selection_valid_input"] for row in deployed),
            "single deployed points became selection-valid")

    check_qualified_override_bridge()

    print("measurement-control-plane: PASS")
    print(f"  exact roster/master keys: {len(master_keys)}")
    print(f"  explicit deployed-point rows: {len(deployed)}")
    print(f"  master-qualified deployed rows: {sum(r['master_qualified_input'] for r in deployed)}")
    print(f"  selection-valid rows (all axes): {sum(r.get('selection_valid_input') is True for r in doc['rows'])}")
    print("  load-bearing regime routing: grid prefill PASS; unimplemented decode fail-closed")
    print("  current-artifact exporter route/recipe closure: PASS")
    print("  hermetic qualified official-run -> recon input: PASS; stale negative: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
