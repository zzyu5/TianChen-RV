#!/usr/bin/env python3
# tools/lint/_manifest_common.py — shared helpers for the org STAGE2 lints
# (experiments/ + docs/ reorg, 2026-07-06). No side effects on import.
#
# org STAGE1 split the former single-body experiments/MANIFEST.md REGISTRY into
# ONE MANIFEST.md per cell under experiments/{active,sealed,archive}/<campaign>/<cell>/.
# Every STAGE2 gate shares:
#   - the "durable content" lens (git-tracked + untracked-not-ignored = what a
#     fresh clone gets), and
#   - a tolerant per-cell MANIFEST parser (cell / campaign / status / role / durable list).

import os
import re
import subprocess

SELF_DIR = os.path.dirname(os.path.abspath(__file__))

TIERS = ("active", "sealed", "archive")

# --- MANIFEST line grammar (bullet / bold-key / durable-list heading) ---
TITLE_RE = re.compile(r"^\s*#\s+cell MANIFEST\s*[—:-]\s*(.+?)\s*$")
CAMPAIGN_RE = re.compile(r"^\s*[-*]\s*\*\*campaign\*\*\s*:\s*(.+?)\s*$", re.I)
STATUS_RE = re.compile(r"^\s*[-*]\s*\*\*status\*\*\s*:\s*(.+?)\s*$", re.I)
ROLE_RE = re.compile(r"^\s*[-*]\s*\*\*role\*\*\s*:\s*(.+?)\s*$", re.I)
# durable-file registry sections: "## durable files ..." or "## templates ...".
# (the parked-cell "## contents" section lists GITIGNORED files -> NOT durable, not parsed.)
DURABLE_HEADING_RE = re.compile(r"^\s*##\s+(durable files|templates)\b", re.I)
HEADING_RE = re.compile(r"^\s*##\s+")
BULLET_TOKEN_RE = re.compile(r"^\s*[-*]\s+`([^`]+)`")

# code / binary artifact-pointer extensions (allowed in the data cell ONLY when
# registered as a durable evidence pointer in the owning cell MANIFEST).
CODE_ARTIFACT_EXTS = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh",
    ".mlir", ".inc", ".ll", ".s", ".o", ".a", ".so",
}


def repo_root():
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=SELF_DIR, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def durable_paths(root, subdir):
    """tracked + untracked-not-ignored under <subdir> (fresh-clone lens), repo-relative."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", subdir],
        cwd=root, capture_output=True, check=True,
    )
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


def tracked_paths(root, subdir):
    """git-tracked only (index/HEAD) under <subdir>, repo-relative."""
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", subdir],
        cwd=root, capture_output=True, check=True,
    )
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


def classify_status(raw):
    """Normalise a MANIFEST status string to {sealed,active,stale,archive,template,parked}."""
    u = (raw or "").upper()
    if "STALE" in u:
        return "stale"
    if "SEALED" in u or "SEAL" in u:
        return "sealed"
    if "PARKED" in u or "INTERIM" in u:
        return "parked"
    if "TEMPLATE" in u:
        return "template"
    if "ARCHIVE" in u:
        return "archive"
    if "ACTIVE" in u:
        return "active"
    return "unknown"


def parse_manifest(path, root):
    """Parse one per-cell MANIFEST.md. Returns a dict; never raises on shape drift."""
    reldir = os.path.relpath(os.path.dirname(path), root).replace(os.sep, "/")
    info = {
        "path": os.path.relpath(path, root).replace(os.sep, "/"),
        "reldir": reldir,
        "cell": os.path.basename(reldir),
        "campaign": "",
        "status_raw": "",
        "status": "unknown",
        "role": "",
        "durable": [],
    }
    try:
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines()
    except OSError:
        return info
    in_durable = False
    for ln in lines:
        m = TITLE_RE.match(ln)
        if m:
            info["cell"] = m.group(1).strip()
        m = CAMPAIGN_RE.match(ln)
        if m:
            info["campaign"] = m.group(1).strip()
        m = STATUS_RE.match(ln)
        if m:
            info["status_raw"] = m.group(1).strip()
            info["status"] = classify_status(m.group(1))
        m = ROLE_RE.match(ln)
        if m:
            info["role"] = m.group(1).strip()
        if HEADING_RE.match(ln):
            in_durable = bool(DURABLE_HEADING_RE.match(ln))
            continue
        if in_durable:
            bm = BULLET_TOKEN_RE.match(ln)
            if bm:
                rel = os.path.normpath(os.path.join(reldir, bm.group(1))).replace(os.sep, "/")
                info["durable"].append(rel)
    return info


def cell_manifests(root):
    """All per-cell MANIFEST.md under experiments/{active,sealed,archive}/** (durable lens)."""
    res = []
    for p in durable_paths(root, "experiments/"):
        parts = p.split("/")
        if (len(parts) >= 3 and parts[0] == "experiments" and parts[1] in TIERS
                and os.path.basename(p) == "MANIFEST.md"):
            res.append(os.path.join(root, p))
    return sorted(res)


def all_cells(root):
    """Parsed info for every per-cell MANIFEST, sorted by (tier, campaign, cell)."""
    cells = [parse_manifest(mf, root) for mf in cell_manifests(root)]

    def tier_of(reldir):
        p = reldir.split("/")
        return p[1] if len(p) >= 2 else "?"

    for c in cells:
        c["tier"] = tier_of(c["reldir"])
    order = {t: i for i, t in enumerate(TIERS)}
    cells.sort(key=lambda c: (order.get(c["tier"], 9), c["campaign"], c["reldir"]))
    return cells


def registered_pointers(root):
    """Union of durable file paths registered across all per-cell MANIFESTs (repo-rel)."""
    reg = set()
    for mf in cell_manifests(root):
        reg.update(parse_manifest(mf, root)["durable"])
    return reg


def artifact_pointers(durable_rels):
    """Subset of a durable file list whose ext is a code/binary artifact pointer."""
    out = []
    for rel in durable_rels:
        ext = os.path.splitext(rel)[1].lower()
        if ext in CODE_ARTIFACT_EXTS:
            out.append(rel)
    return sorted(out)
