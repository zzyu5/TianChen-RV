#!/usr/bin/env python3
"""gen_retired_index.py -- 判定书 §3 轴B 裁决③: RETIRED-INDEX generator (机检化).

Merges the THREE canonical retirement ledgers into ONE read-only single query point,
schema/retired-index.generated.json, so "which quant-format cells are RETIRED, and by
what four requirements" has a SINGLE machine-checkable answer instead of being scattered
across three de-memoized places:

  (a) the `// NOTE: def <OpDef> (the monolith <fmt> block-dot op) was RETIRED at ...`
      prose markers in the ODS (include/TianChenRV/Dialect/RVV/IR/RVVOps.td)
          -> vec_dot monolith op-def retirements (form B, four-requirement prose);
  (b) schema/monolith-retire-whitelist.v1.json  retired_ledger
          -> vec_dot monolith op-def retirements (form A, structured);
  (c) schema/emit-bypass-whitelist.v1.json      retired_ledger
          -> repack gemm_tile direct-emitter retirements (structured).

Each merged entry carries the FOUR REQUIREMENTS (四要件):
  format / op (op_def or retired direct-emitter symbols) / retirement_basis (退役依据) /
  alternative_path (替代路径 = front-door construction pointer) / restore_ref (复原指针).

An entry keyed (axis, format) that appears in more than one source is MERGED (union of
sources, first-non-empty per field with .td prose preferred as the richest form).

The output is MACHINE-GENERATED and must NOT be hand-edited. tools/lint/check_retired_index.py
(CI job retired-index-gate) regenerates in-memory and fails closed on any drift, plus
asserts coverage (every retired vec_dot monolith is indexed) + four-requirement non-empty.

Usage:  python3 tools/lint/gen_retired_index.py [-o schema/retired-index.generated.json]
"""
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ODS = os.path.join(REPO, "include/TianChenRV/Dialect/RVV/IR/RVVOps.td")
MONOLITH_WL = os.path.join(REPO, "schema/monolith-retire-whitelist.v1.json")
EMIT_BYPASS_WL = os.path.join(REPO, "schema/emit-bypass-whitelist.v1.json")
OUT_DEFAULT = os.path.join(REPO, "schema/retired-index.generated.json")

# A .td RETIRED NOTE marker starts here (the header may wrap onto following // lines).
RE_NOTE_START = re.compile(
    r"^//\s*NOTE:\s*def\s+(\w+)\s+\(the monolith\s+(\S+)\s+block-dot op\)\s+was RETIRED\b")
RE_FLIP = re.compile(r"(\S+ flip \([^)]*\))")
RE_FRONT_DOOR = re.compile(r"the front door (?:now )?constructs?\s+(.+?)\s+instead", re.S)
RE_CODE_MOVE = re.compile(
    r"(?:code-moved into|extracted into) the byte-exact(?:\s+shared anchor)?\s+(\w+)", re.S)


def _join_comment_block(lines, start):
    """From line index `start` (a // comment), gather the contiguous run of // lines and
    return (joined_prose, next_index). // markers and leading whitespace are stripped."""
    body = []
    i = start
    n = len(lines)
    while i < n and lines[i].lstrip().startswith("//"):
        txt = lines[i].lstrip()
        txt = txt[2:] if txt.startswith("//") else txt  # drop the leading //
        body.append(txt.strip())
        i += 1
    return " ".join(t for t in body if t), i


def parse_td_retired_notes(td_text):
    """RVVOps.td -> [entry] for every vec_dot monolith RETIRED NOTE (four-requirement prose)."""
    lines = td_text.splitlines()
    out = []
    i = 0
    n = len(lines)
    while i < n:
        m = RE_NOTE_START.match(lines[i].lstrip())
        if not m:
            i += 1
            continue
        op_def, fmt = m.group(1), m.group(2)
        prose, nxt = _join_comment_block(lines, i)
        i = nxt

        flip = RE_FLIP.search(prose)
        restore_ref = flip.group(1) if flip else f"{fmt} flip"

        fd = RE_FRONT_DOOR.search(prose)
        alt = ("the front door constructs " + fd.group(1).strip()) if fd else ""
        cm = RE_CODE_MOVE.search(prose)
        if cm:
            alt = (alt + " ; emit anchor " + cm.group(1)) if alt else ("emit anchor " + cm.group(1))
        if not alt:
            alt = prose

        # 退役依据: the "was RETIRED at ... instead." sentence (why the op-def could go).
        basis = prose
        wr = prose.find("was RETIRED at")
        if wr != -1:
            tail = prose[wr:]
            end = tail.find(" instead")
            basis = ("op-def " + tail[: end + len(" instead")].rstrip() + ".") if end != -1 \
                else ("op-def " + tail)

        out.append({
            "axis": "vec_dot",
            "format": fmt,
            "op": op_def,
            "retirement_basis": basis,
            "alternative_path": alt,
            "restore_ref": restore_ref,
            "source": ".td RETIRED NOTE (RVVOps.td)",
            "prose": prose,
        })
    return out


def parse_monolith_retired_ledger(wl):
    """monolith-retire-whitelist.v1.json retired_ledger -> [entry] (vec_dot axis)."""
    out = []
    for e in wl.get("retired_ledger", []) or []:
        fmt = e.get("format", "")
        op_def = e.get("op_def", "")
        basis = e.get("retired_at") or e.get("reason") or ""
        alt = e.get("front_door") or e.get("cleanup") or e.get("note") or ""
        restore = e.get("restore_ref") or e.get("retired_at") or ""
        out.append({
            "axis": "vec_dot",
            "format": fmt,
            "op": op_def,
            "retirement_basis": basis,
            "alternative_path": alt,
            "restore_ref": restore,
            "source": "monolith-retire-whitelist.v1.json retired_ledger",
        })
    return out


def _short_campaign(retired_by):
    """First clause of a retired_by campaign string (up to the first ' (' or ':')."""
    cut = len(retired_by)
    for sep in (" (", ": ", ":"):
        p = retired_by.find(sep)
        if p != -1:
            cut = min(cut, p)
    return retired_by[:cut].strip() or retired_by.strip()


def parse_emit_bypass_retired_ledger(wl):
    """emit-bypass-whitelist.v1.json retired_ledger -> [entry] (gemm_tile / repack axis)."""
    out = []
    for e in wl.get("retired_ledger", []) or []:
        fmt = e.get("format", "")
        was = e.get("was_emitters", []) or []
        op = "repack gemm_tile: " + ", ".join(was) if was else ("repack gemm_tile " + fmt)
        retired_by = e.get("retired_by", "")
        alt = e.get("front_door") or e.get("note") or retired_by
        restore = ("campaign " + _short_campaign(retired_by)
                   + " (git-findable via commit message; retirement_batch "
                   + str(e.get("retirement_batch", "?")) + ")") if retired_by else ""
        out.append({
            "axis": "gemm_tile",
            "format": fmt,
            "op": op,
            "retirement_basis": retired_by,
            "alternative_path": alt,
            "restore_ref": restore,
            "source": "emit-bypass-whitelist.v1.json retired_ledger",
        })
    return out


# .td prose is the richest four-requirement form -> preferred first in the merge.
_SOURCE_PRIORITY = {
    ".td RETIRED NOTE (RVVOps.td)": 0,
    "monolith-retire-whitelist.v1.json retired_ledger": 1,
    "emit-bypass-whitelist.v1.json retired_ledger": 2,
}
_FIELDS = ("op", "retirement_basis", "alternative_path", "restore_ref", "prose")


def merge_entries(raw):
    """Merge raw entries by (axis, format): union sources, first-non-empty per field."""
    by_key = {}
    for e in raw:
        key = (e["axis"], e["format"])
        by_key.setdefault(key, []).append(e)
    merged = []
    for (axis, fmt), group in by_key.items():
        group = sorted(group, key=lambda x: _SOURCE_PRIORITY.get(x["source"], 9))
        out = {"axis": axis, "format": fmt}
        for f in _FIELDS:
            val = ""
            for g in group:
                if g.get(f):
                    val = g[f]
                    break
            if val or f != "prose":
                out[f] = val
        out["sources"] = sorted({g["source"] for g in group})
        merged.append(out)
    merged.sort(key=lambda x: (x["axis"], x["format"]))
    return merged


def build_index(repo=REPO):
    """Assemble the full RETIRED-INDEX document (deterministic, no timestamps)."""
    td_text = open(os.path.join(repo, "include/TianChenRV/Dialect/RVV/IR/RVVOps.td"),
                   encoding="utf-8").read()
    mono = json.load(open(os.path.join(repo, "schema/monolith-retire-whitelist.v1.json"),
                          encoding="utf-8"))
    bypass = json.load(open(os.path.join(repo, "schema/emit-bypass-whitelist.v1.json"),
                            encoding="utf-8"))

    raw = (parse_td_retired_notes(td_text)
           + parse_monolith_retired_ledger(mono)
           + parse_emit_bypass_retired_ledger(bypass))
    entries = merge_entries(raw)

    vec = [e for e in entries if e["axis"] == "vec_dot"]
    gem = [e for e in entries if e["axis"] == "gemm_tile"]
    return {
        "$meta": {
            "schema": "retired-index.generated.v1",
            "GENERATED": "MACHINE-GENERATED by tools/lint/gen_retired_index.py -- DO NOT "
                         "HAND-EDIT. Regenerate after editing ANY source ledger (.td RETIRED "
                         "NOTE markers / monolith-retire-whitelist retired_ledger / "
                         "emit-bypass-whitelist retired_ledger). tools/lint/check_retired_index.py "
                         "(CI job retired-index-gate) fails closed on drift.",
            "ruling": "判定书 §3 轴B 裁决③ -- single read-only query point for RETIRED cells.",
            "sources": [
                ".td RETIRED NOTE markers (include/TianChenRV/Dialect/RVV/IR/RVVOps.td) -- "
                "vec_dot monolith op-def retirements (form B, four-requirement prose)",
                "schema/monolith-retire-whitelist.v1.json retired_ledger -- vec_dot monolith "
                "op-def retirements (form A, structured)",
                "schema/emit-bypass-whitelist.v1.json retired_ledger -- repack gemm_tile "
                "direct-emitter retirements (structured)",
            ],
            "four_requirements": [
                "format (格名)",
                "op (op-def for vec_dot / retired direct-emitter symbols for gemm_tile)",
                "retirement_basis (退役依据 = front door now constructs the typed body)",
                "alternative_path (替代路径 = front-door construction / emit-anchor pointer)",
                "restore_ref (复原指针 = git ref or named flip/campaign)",
            ],
            "counts": {
                "total": len(entries),
                "vec_dot_monolith": len(vec),
                "gemm_tile_repack": len(gem),
            },
        },
        "entries": entries,
    }


def dumps(index):
    return json.dumps(index, indent=2, ensure_ascii=False) + "\n"


def write_index(repo=REPO, out=OUT_DEFAULT):
    index = build_index(repo)
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(dumps(index))
    return index


def main(argv):
    out = OUT_DEFAULT
    if "-o" in argv:
        out = argv[argv.index("-o") + 1]
    index = write_index(REPO, out)
    c = index["$meta"]["counts"]
    print(f"[gen-retired-index] wrote {os.path.relpath(out, REPO)}: "
          f"{c['total']} entries ({c['vec_dot_monolith']} vec_dot monolith + "
          f"{c['gemm_tile_repack']} gemm_tile repack)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
