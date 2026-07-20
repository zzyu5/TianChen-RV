#!/usr/bin/env python3
"""Build the descriptive B1 measurement asset matrix from canonical sources.

The CSV emitted by this script is a task-local snapshot, never a result ledger.
Numbers continue to live in the recon-owned T3 master and immutable runs; this
join only makes missing lineage/qualification explicit.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MASTER = ROOT / "experiments" / "master" / "T3_master_rebuild.csv"
MEMORY = ROOT / "schema" / "measurement-memory.v1.json"
sys.path.insert(0, str(ROOT / "tools" / "bench"))
from measurement_keys import key_from_mapping, load_master_keys  # noqa: E402


FIELDS = (
    "op", "format", "engine", "regime", "board", "group",
    "result_lane", "e2e_evidence", "measurement_state", "disposition", "cold",
    "opponent_tier", "opponent_symbol", "correctness", "run_id", "freshness",
    "tn_qualification", "master_qualified", "selection_valid", "evidence_lane",
    "source", "citation_status",
)


def load_memory_rows():
    doc = json.load(open(MEMORY, encoding="utf-8"))
    deployed = {}
    for index, row in enumerate(doc.get("rows", [])):
        if row.get("variant_axis") != "deployed_point":
            continue
        key = key_from_mapping(
            {"op": row.get("op"), "format": row.get("kernel"),
             "engine": row.get("engine"), "regime": row.get("regime")},
            source=f"{MEMORY}:rows[{index}]",
        )
        board = row.get("measurement_board")
        if board not in ("rvv", "k1"):
            raise ValueError(f"{MEMORY}:rows[{index}] invalid board {board!r}")
        identity = (*key, board)
        if identity in deployed:
            raise ValueError(f"duplicate deployed measurement identity {identity}")
        deployed[identity] = row
    selection_valid_total = sum(
        row.get("selection_valid_input") is True for row in doc.get("rows", [])
    )
    return deployed, selection_valid_total


def build_matrix():
    load_master_keys(MASTER)  # full-table closed-key/uniqueness validation
    deployed, selection_valid_total = load_memory_rows()
    with MASTER.open(encoding="utf-8", newline="") as stream:
        master = list(csv.DictReader(stream))
    output = []
    used = set()
    for source in master:
        key = key_from_mapping(source, source=str(MASTER))
        for board in ("rvv", "k1"):
            identity = (*key, board)
            memory = deployed.get(identity)
            if memory:
                used.add(identity)
            cold = source[f"{board}_cold"]
            disposition = source[f"{board}_disp"]
            if memory:
                state = memory.get("measurement_state")
                correctness = memory.get("byte_exact_gate")
                run_id = memory.get("run_id")
                freshness = memory.get("freshness")
                tn = memory.get("tn_qualification")
                master_qualified = bool(memory.get("master_qualified_input"))
                selection_valid = bool(memory.get("selection_valid_input"))
                evidence_lane = memory.get("evidence_lane")
                evidence_source = memory.get("source")
            else:
                state = "n_a" if disposition == "N/A-hw" else "open"
                correctness = "unresolved"
                run_id = None
                freshness = "unresolved"
                tn = "unqualified"
                master_qualified = False
                selection_valid = False
                evidence_lane = "none"
                evidence_source = None
            if master_qualified:
                citation = "qualified-current"
            elif cold:
                citation = "legacy-unlinked"
            elif state == "n_a":
                citation = "n_a"
            else:
                citation = "open"
            output.append({
                "op": key[0], "format": key[1], "engine": key[2], "regime": key[3],
                "board": board, "group": source["group"],
                "result_lane": "T3-kernel-cell", "e2e_evidence": "false",
                "measurement_state": state, "disposition": disposition, "cold": cold,
                "opponent_tier": source[f"{board}_tier"],
                "opponent_symbol": source[f"{board}_opp_sym"],
                "correctness": correctness, "run_id": run_id or "",
                "freshness": freshness, "tn_qualification": tn,
                "master_qualified": str(master_qualified).lower(),
                "selection_valid": str(selection_valid).lower(),
                "evidence_lane": evidence_lane, "source": evidence_source or "",
                "citation_status": citation,
            })
    orphaned = sorted(set(deployed) - used)
    if orphaned:
        raise ValueError(f"measurement-memory deployed rows absent from master: {orphaned}")
    return output, selection_valid_total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", action="store_true", help="emit the full matrix to stdout")
    parser.add_argument("--summary", action="store_true", help="emit machine-readable counts")
    args = parser.parse_args()
    rows, selection_valid_total = build_matrix()
    if args.csv:
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    counts = Counter(row["citation_status"] for row in rows)
    if args.summary or not args.csv:
        print(json.dumps({
            "master_board_cells": len(rows),
            "citation_status": dict(sorted(counts.items())),
            "master_qualified_cells": sum(r["master_qualified"] == "true" for r in rows),
            "selection_valid_rows_all_axes": selection_valid_total,
            "e2e_cells_in_this_matrix": 0,
        }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
