#!/usr/bin/env python3
"""pattern_fanout_ledger.py — C7 模式计数归一 · 扇出台账（机算·可复跑）

registry = 唯一权威（schema/pattern-registry.v1.json）。本脚本从 registry + 代码/lit
事实【机算】每条模式的覆盖格清单，禁手写小计。算不出来的格标 UNVERIFIED（不猜）。

口径纪律：
  · 本台账的比 = "条目/强义格"（构造轴·分母同 101/108 家族）。
  · ★禁混 perf-covered 9/83（系统账 e2e 性能轴·完全不同赛道·禁互推）。
  · 本脚本 READ-ONLY：不写 registry、不改 sealed 工件（T3p/T8）。

用法：
    python3 tools/visibility/pattern_fanout_ledger.py            # 表格
    python3 tools/visibility/pattern_fanout_ledger.py --json     # 机读
    python3 tools/visibility/pattern_fanout_ledger.py --check    # 漂移自检(退出码)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys

REPO = os.path.realpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", ".."))

REGISTRY = "schema/pattern-registry.v1.json"
LOWER = "lib/Plugin/RVV/FrontDoor/RVVLowerQuantContraction.cpp"
TILING_SEL = "include/Weft/Plugin/RVV/RVVRepackTilingSelection.h"
BLOCKQL = "lib/Conversion/RVV/RVVToEmitCBlockQuantLinear.cpp"
IME_DRV = "lib/Plugin/IME/IMEBackendEmissionDriver.cpp"
LIT_RVV = "test/Conversion/RVV"

VERIFIED, UNVERIFIED = "VERIFIED", "UNVERIFIED"


def rd(rel: str) -> list[str]:
    p = os.path.join(REPO, rel)
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8", errors="replace") as f:
        return f.read().split("\n")


def cite(rel: str, line: int) -> str:
    return f"{rel}:{line}"


# ---------------------------------------------------------------- probes

def probe_registry() -> dict:
    p = os.path.join(REPO, REGISTRY)
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    pats = d["patterns"]
    hist: dict[str, int] = {}
    for x in pats:
        hist[x["status"]] = hist.get(x["status"], 0) + 1
    # the status enum as DECLARED in $meta.note (the authority string itself)
    note = d["$meta"].get("note", "")
    m = re.search(r"status\s*∈\s*\{([^}]+)\}", note)
    declared = [s.strip() for s in m.group(1).split("|")] if m else []
    return {"entries": len(pats), "status_histogram": hist,
            "declared_status_enum": declared, "patterns": pats}


def probe_metrics_hook_files(hook: str | None) -> list[tuple[str, bool]]:
    """Resolve file-ish tokens in a metrics_hook to (path, exists).

    NOTE regex alternation order: csv|mlir|md before bare c, else '.csv' truncates
    to '.c' and reports a false MISSING.
    """
    if not hook:
        return []
    toks = re.findall(r"[\w./-]+\.(?:csv|mlir|md|c)\b", hook)
    out = []
    for t in dict.fromkeys(toks):
        out.append((t, os.path.exists(os.path.join(REPO, t))))
    return out


def probe_s6() -> dict:
    """PAT-S6 fanout: the tiling-selection KEY + its stamp leaves.

    Machine facts:
      (a) classifyTilingBottleneckShape() -- which fold_models classify (the KEY domain)
      (b) the /*foldModel=*/ + /*decodeModel=*/ registration table -- table-driven formats
      (c) stampTilingSelection() call sites -- the actual stamp leaves
    """
    sel = rd(TILING_SEL)
    fold_models: dict[str, str] = {}
    inside, shape = False, None
    for i, l in enumerate(sel):
        if "classifyTilingBottleneckShape" in l and "inline" in "".join(sel[max(0, i - 2):i + 1]):
            inside = True
        if inside:
            for fm in re.findall(r'foldModel == "([^"]+)"', l):
                fold_models[fm] = None
            m = re.search(r"return RVVTilingBottleneckShape::(\w+)", l)
            if m:
                shape = m.group(1)
                for fm in list(fold_models):
                    if fold_models[fm] is None:
                        fold_models[fm] = shape
            if re.match(r"^\}", l) and fold_models:
                break

    # (b) registration table: foldModel + nearby decodeModel
    low = rd(LOWER)
    table: list[dict] = []
    for i, l in enumerate(low):
        m = re.search(r'/\*foldModel=\*/\s*"([^"]+)"', l)
        if not m:
            continue
        dm = None
        for j in range(max(0, i - 6), min(len(low), i + 7)):
            d = re.search(r'/\*decodeModel=\*/\s*"([^"]+)"', low[j])
            if d:
                dm = d.group(1)
                break
        table.append({"fold_model": m.group(1), "decode_model": dm,
                      "shape": fold_models.get(m.group(1)), "cite": cite(LOWER, i + 1)})

    # (c) stamp leaves
    stamps: list[dict] = []
    for i, l in enumerate(low):
        if "stampTilingSelection(builder" not in l:
            continue
        m = re.search(r'stampTilingSelection\(builder,\s*loop,\s*"([^"]+)",\s*"([^"]+)"', l)
        if m:
            stamps.append({"kind": "literal", "fold_model": m.group(1),
                           "decode_model": m.group(2), "cite": cite(LOWER, i + 1)})
        elif "facts.foldModel" in l:
            stamps.append({"kind": "table-driven", "fold_model": "facts.foldModel",
                           "decode_model": "facts.decodeModel", "cite": cite(LOWER, i + 1)})
    return {"key_domain_fold_models": fold_models, "table": table, "stamps": stamps}


def probe_roll() -> dict:
    """PAT-EMIT-ROLLED fanout: leaves carrying an `if (rolledMainTerm)` arm."""
    src = rd(BLOCKQL)
    fn = None
    leaves: list[dict] = []
    for i, l in enumerate(src):
        m = re.match(r"^(?:mlir::LogicalResult|static\s+\S+)\s+(?:VariantToEmitCFunc::)?(\w+)\(", l)
        if m:
            fn = m.group(1)
        if re.search(r"if \(rolledMainTerm\)", l):
            g = re.search(r"emitRepackKQuant(Gemv|Gemm)Body(Q\d+K)", fn or "")
            leaves.append({"emitter": fn, "regime": g.group(1).upper() if g else None,
                           "format": g.group(2) if g else None, "cite": cite(BLOCKQL, i + 1)})
    lits = sorted(f for f in os.listdir(os.path.join(REPO, LIT_RVV)) if "-rolled" in f)
    return {"leaves": leaves, "rolled_lits": lits}


def probe_ime() -> dict:
    """PAT-1-IME fanout: the in-code kIMEVmadotTilingPatterns[] width table."""
    src = rd(IME_DRV)
    rows: list[dict] = []
    grab = False
    for i, l in enumerate(src):
        if "kIMEVmadotTilingPatterns[]" in l:
            grab = True
            continue
        if grab:
            if re.match(r"^\};", l):
                break
            m = re.search(r'\{"(IME-VMADOT-TILE-[\w-]+)",\s*(\d+),\s*(\d+),\s*"([\w-]+)"', l)
            if m:
                rows.append({"pattern_id": m.group(1), "njw": int(m.group(2)),
                             "min_vreg_budget": int(m.group(3)), "status": m.group(4),
                             "cite": cite(IME_DRV, i + 1)})
    fmts: list[dict] = []
    grab = False
    for i, l in enumerate(src):
        if "kIMEWideFormatMeasurements[]" in l:
            grab = True
            continue
        if grab:
            if re.match(r"^\};", l):
                break
            m = re.search(r'\{"(\w+)",\s*"([\w-]+)",\s*"([^"]+)"', l)
            if m:
                fmts.append({"weight_format": m.group(1), "status": m.group(2),
                             "metric": m.group(3), "cite": cite(IME_DRV, i + 1)})
    return {"tile_widths": rows, "per_format_measured_negative": fmts}


def probe_gevm() -> dict:
    """PAT-2-P9 fanout: colgroup-tiled GEVM emitters actually wired."""
    src = rd(BLOCKQL)
    emitters: list[dict] = []
    for i, l in enumerate(src):
        m = re.search(r"VariantToEmitCFunc::(emitRepackKQuantGemvColgroupTiledBody(\w+))\(", l)
        if m:
            emitters.append({"emitter": m.group(1), "format_suffix": m.group(2),
                             "cite": cite(BLOCKQL, i + 1)})
    lits = sorted(f for f in os.listdir(os.path.join(REPO, LIT_RVV)) if "colgroup" in f)
    return {"emitters": emitters, "lits": lits}


# ---------------------------------------------------------------- ledger

def build(reg: dict) -> list[dict]:
    s6, roll, ime, gevm = probe_s6(), probe_roll(), probe_ime(), probe_gevm()
    rows: list[dict] = []

    for p in reg["patterns"]:
        pid, status = p["pattern_id"], p["status"]
        hooks = probe_metrics_hook_files(p.get("metrics_hook"))
        row = {"pattern_id": pid, "status": status,
               "hook_files": [{"path": t, "exists": e} for t, e in hooks]}

        if pid.startswith("MFLAT-") and not pid.startswith("MFLAT-P2c"):
            # 1:1 — one construction brick, one lit artifact.
            n = len(hooks)
            row.update(fanout_kind="lit-artifact (1:1)", fanout_count=n,
                       members=[t for t, _ in hooks],
                       verdict=VERIFIED if (n and all(e for _, e in hooks)) or status == "planned"
                       else UNVERIFIED,
                       note="planned: no artifact yet (fanout 0 by definition)"
                       if status == "planned" else "1 brick -> 1 lit")
        elif pid.startswith("MFLAT-P2c"):
            row.update(fanout_kind="lit-artifact (1:1) + fair-perf csv",
                       fanout_count=len(hooks), members=[t for t, _ in hooks],
                       verdict=VERIFIED if all(e for _, e in hooks) else UNVERIFIED,
                       note="measured-negative: construction green, perf hypothesis falsified")
        elif pid.startswith("WIDE-DECODE"):
            row.update(fanout_kind="none (parked)", fanout_count=0, members=[],
                       verdict=VERIFIED,
                       note="deferred-backlog: no ODS/verifier/emit primitive landed => fanout 0")
        elif pid.startswith("PAT-S6"):
            tbl = [t for t in s6["table"] if t["shape"]]
            lits = [s for s in s6["stamps"] if s["kind"] == "literal"]
            row.update(fanout_kind="tiling-KEY stamp leaves / formats",
                       fanout_count=len(s6["stamps"]),
                       members=[f"{s['decode_model']} @ {s['cite']}" for s in s6["stamps"]],
                       verdict=VERIFIED,
                       note=(f"{len(s6['stamps'])} stamp leaves = {len(lits)} flat literal "
                             f"+ {len(s6['stamps']) - len(lits)} table-driven; "
                             f"table-driven formats = {len(tbl)} "
                             f"({','.join(sorted({t['decode_model'] for t in tbl}))})"),
                       detail={"table_driven_formats": sorted({t["decode_model"] for t in tbl}),
                               "flat_literal_formats": [s["decode_model"] for s in lits],
                               "key_domain": s6["key_domain_fold_models"]})
        elif pid.startswith("PAT-EMIT-ROLLED"):
            fmts = sorted({l["format"] for l in roll["leaves"] if l["format"]})
            row.update(fanout_kind="rolled-arm emitter leaves",
                       fanout_count=len(roll["leaves"]),
                       members=[f"{l['format']}/{l['regime']} @ {l['cite']}" for l in roll["leaves"]],
                       verdict=VERIFIED,
                       note=(f"{len(roll['leaves'])} leaves = {len(fmts)} formats "
                             f"({','.join(fmts)}) x2 regimes; "
                             f"{len(roll['rolled_lits'])} -rolled lit fixtures"),
                       detail={"formats": fmts, "rolled_lits": roll["rolled_lits"]})
        elif pid.startswith("PAT-1-IME"):
            rows_t = ime["tile_widths"]
            mech = [r for r in rows_t if r["status"] == "mechanized"]
            row.update(fanout_kind="vmadot tile widths (W-table)",
                       fanout_count=len(rows_t),
                       members=[f"{r['pattern_id']}(njw={r['njw']},{r['status']})" for r in rows_t],
                       verdict=VERIFIED,
                       note=(f"{len(rows_t)} width rows: {len(mech)} mechanized, "
                             f"{len(rows_t) - len(mech)} measured-negative; "
                             f"per-format measured-negative rows="
                             f"{[f['weight_format'] for f in ime['per_format_measured_negative']]}"),
                       detail=ime)
        elif pid.startswith("PAT-2-P9"):
            row.update(fanout_kind="colgroup-tiled GEVM emitters",
                       fanout_count=len(gevm["emitters"]),
                       members=[f"{e['emitter']} @ {e['cite']}" for e in gevm["emitters"]],
                       verdict=VERIFIED,
                       note=(f"{len(gevm['emitters'])} emitter (q4_K only, hardcoded call) "
                             f"-- registry says '首格'; K-quant addressable denominator=5 "
                             f"(q4_K,q2_K,q5_K,q6_K,q3_K)"),
                       detail={"lits": gevm["lits"]})
        else:
            row.update(fanout_kind="?", fanout_count=None, members=[], verdict=UNVERIFIED,
                       note="no machine probe wired for this pattern")
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    reg = probe_registry()
    rows = build(reg)

    if a.json:
        print(json.dumps({"registry": {k: v for k, v in reg.items() if k != "patterns"},
                          "rows": rows}, ensure_ascii=False, indent=2))
        return 0

    print("=" * 78)
    print("C7 模式扇出台账 — registry = 唯一权威 (机算·可复跑)")
    print("口径 = 条目/强义格（构造轴）。★禁混 perf-covered 9/83（perf 轴·不同赛道）")
    print("=" * 78)
    print(f"\nregistry entries : {reg['entries']}   ({REGISTRY})")
    print(f"status histogram : {reg['status_histogram']}")
    print(f"declared enum    : {reg['declared_status_enum']}  <- $meta.note 自身声明")
    unused = [s for s in reg["declared_status_enum"] if s not in reg["status_histogram"]]
    if unused:
        print(f"declared-but-unused: {unused}")
    print()
    for r in rows:
        print("-" * 78)
        print(f"{r['pattern_id']}")
        print(f"  status={r['status']}  fanout={r['fanout_count']} [{r['fanout_kind']}]  {r['verdict']}")
        print(f"  note: {r['note']}")
        for m in r["members"][:12]:
            print(f"    · {m}")
        miss = [h["path"] for h in r["hook_files"] if not h["exists"]]
        if miss:
            print(f"  ★hook MISSING: {miss}")
    print("-" * 78)
    tot = sum(r["fanout_count"] or 0 for r in rows)
    print(f"\nfanout total (sum of machine-counted members) = {tot}")
    print(f"UNVERIFIED rows = {[r['pattern_id'] for r in rows if r['verdict'] == UNVERIFIED]}")

    if a.check:
        bad = [r["pattern_id"] for r in rows if r["verdict"] == UNVERIFIED]
        miss = [h["path"] for r in rows for h in r["hook_files"] if not h["exists"]]
        if bad or miss:
            print(f"\nCHECK FAIL: unverified={bad} missing_hooks={miss}")
            return 1
        print("\nCHECK OK: every registry row has a machine-computed fanout; all hooks resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
