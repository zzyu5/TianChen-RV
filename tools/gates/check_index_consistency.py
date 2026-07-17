#!/usr/bin/env python3
# tools/gates/check_index_consistency.py — org STAGE2 Lint ③ (CI, fail-closed).
#
# Bidirectional consistency between the per-cell MANIFESTs, the tree, and INDEX.md, plus the
# tracking invariants that keep sealed evidence durable and interim scratch out of history.
#
#   (1) INDEX FRESH.      experiments/INDEX.md == the generator's output (every MANIFEST is
#                         represented; no stale/orphan rows). Regenerate to fix.
#   (2) CELL COVERAGE.    every durable file under experiments/{active,sealed,archive}/ belongs
#                         to a cell that has a MANIFEST.md (tier-root docs like archive/MOVES.md
#                         are exempt). No orphan evidence without a manifest.
#   (3) PER-CELL DRIFT.   for each cell, the files its MANIFEST registers == the durable files
#                         physically in the cell (minus the MANIFEST itself). Missing or
#                         unregistered file -> drift (this is the per-cell successor to the old
#                         single-REGISTRY check_manifest.py).
#   (4) TRACKING BIDI.    sealed/ evidence must be git-TRACKED and NOT gitignored (no gitignore
#                         line may cover sealed/ — sealing deletes it); an interim/gitignored
#                         cell under active/ must have ZERO tracked files (never force-added).
#                         (archive/ force-adds are a documented historical exception.)
#
#   check_index_consistency.py             # audit
#   check_index_consistency.py --self-test # exercise the pure helpers

import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _manifest_common as mc  # noqa: E402
import gen_experiments_index as gen  # noqa: E402


# ------------------------------------------------------------ pure helpers
def owning_cell(filerel, manifest_dirs):
    """Nearest ancestor dir (within its tier) that carries a MANIFEST.md, else None.

    Returns None for a file that sits directly at a tier root (experiments/<tier>/x).
    """
    d = os.path.dirname(filerel)
    while True:
        if d in manifest_dirs:
            return d
        parts = d.split("/")
        if len(parts) <= 2:  # reached 'experiments/<tier>' -> no owning cell
            return None
        d = os.path.dirname(d)


def is_tier_root_doc(filerel):
    """True for a documentation file living directly under a tier root (e.g. archive/MOVES.md)."""
    d = os.path.dirname(filerel)
    parts = d.split("/")
    return (len(parts) == 2 and parts[0] == "experiments" and parts[1] in mc.TIERS
            and filerel.lower().endswith(".md"))


# ---------------------------------------------------------------- git helpers
def check_ignored(root, paths):
    """Subset of <paths> that git considers ignored (pattern match, tracking-agnostic)."""
    if not paths:
        return set()
    out = subprocess.run(
        ["git", "check-ignore", "--stdin"],
        cwd=root, input="\n".join(paths), capture_output=True, text=True,
    )
    return {p for p in out.stdout.splitlines() if p}


def gitignore_mentions(root, needle):
    hits = []
    for gi in (".gitignore", "experiments/.gitignore"):
        p = os.path.join(root, gi)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for i, ln in enumerate(f, 1):
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                if needle in s:
                    hits.append(f"{gi}:{i}: {s}")
    return hits


# ------------------------------------------------------------------ self-test
def self_test():
    ok_all = True
    mdirs = {
        "experiments/sealed/repack/cellA",
        "experiments/active/repack",  # campaign-level manifest (parked)
        "experiments/archive/perf-historical/cellB",
    }
    print("-- owning_cell --")
    oc_cases = [
        ("experiments/sealed/repack/cellA/evidence.json", "experiments/sealed/repack/cellA"),
        ("experiments/sealed/repack/cellA/results/r/run.txt", "experiments/sealed/repack/cellA"),
        ("experiments/sealed/repack/cellA/MANIFEST.md", "experiments/sealed/repack/cellA"),
        ("experiments/active/repack/MANIFEST.md", "experiments/active/repack"),
        ("experiments/archive/MOVES.md", None),            # tier-root doc
        ("experiments/sealed/orphan.txt", None),           # tier-root, no cell
    ]
    for f, expect in oc_cases:
        got = owning_cell(f, mdirs)
        mark = "PASS" if got == expect else "FAIL"
        ok_all = ok_all and got == expect
        print(f"  [{mark}] {f} -> {got} (expect {expect})")

    print("-- is_tier_root_doc --")
    tr_cases = [
        ("experiments/archive/MOVES.md", True),
        ("experiments/sealed/note.md", True),
        ("experiments/sealed/repack/cellA/e.json", False),  # nested, not tier root
        ("experiments/archive/blob.o", False),              # not .md
    ]
    for f, expect in tr_cases:
        got = is_tier_root_doc(f)
        mark = "PASS" if got == expect else "FAIL"
        ok_all = ok_all and got == expect
        print(f"  [{mark}] {f} -> {got} (expect {expect})")

    # drift set logic
    print("-- per-cell drift (set logic) --")
    registered = {"a.json", "b.txt"}
    actual = {"a.json", "b.txt"}
    d1 = (registered ^ actual) == set()
    actual2 = {"a.json", "c.txt"}
    d2 = (registered ^ actual2) == {"b.txt", "c.txt"}
    for name, cond in [("identical -> no drift", d1), ("mismatch -> symmetric diff", d2)]:
        mark = "PASS" if cond else "FAIL"
        ok_all = ok_all and cond
        print(f"  [{mark}] {name}")

    print("SELF-TEST:", "GREEN" if ok_all else "RED")
    return 0 if ok_all else 1


# ------------------------------------------------------------------ main
def main(argv):
    if "--self-test" in argv:
        return self_test()

    root = mc.repo_root()
    viol = []

    # (1) INDEX freshness.
    want = gen.render(root)
    index_path = os.path.join(root, gen.INDEX_REL)
    have = ""
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as f:
            have = f.read()
    if have != want:
        viol.append((gen.INDEX_REL, "STALE vs generator — run tools/gates/gen_experiments_index.py"))

    # setup for (2)/(3)
    manifests = [mc.parse_manifest(m, root) for m in mc.cell_manifests(root)]
    mdirs = {c["reldir"] for c in manifests}
    durable = [p for p in mc.durable_paths(root, "experiments/")
               if p.split("/")[1] in mc.TIERS]  # only tiered files (skip top-level README/INDEX/…)

    # (2) cell coverage: every durable tiered file has an owning MANIFEST (or is a tier-root doc).
    actual_by_cell = {d: set() for d in mdirs}
    for p in durable:
        cell = owning_cell(p, mdirs)
        if cell is None:
            if not is_tier_root_doc(p):
                viol.append((p, "orphan durable file: no owning cell MANIFEST"))
            continue
        if os.path.basename(p) != "MANIFEST.md":
            actual_by_cell[cell].add(p)

    # (3) per-cell drift: registered == actual.
    for c in manifests:
        registered = set(c["durable"])
        actual = actual_by_cell.get(c["reldir"], set())
        for miss in sorted(registered - actual):
            viol.append((miss, f"registered in {c['reldir']}/MANIFEST but MISSING from tree"))
        for extra in sorted(actual - registered):
            viol.append((extra, f"present in cell {c['reldir']} but NOT registered in its MANIFEST"))

    # (4) tracking bidirectional.
    #  4a sealed evidence must be tracked (no untracked-not-ignored durable in sealed/).
    sealed_durable = mc.durable_paths(root, "experiments/sealed/")
    sealed_tracked = set(mc.tracked_paths(root, "experiments/sealed/"))
    for p in sealed_durable:
        if p not in sealed_tracked:
            viol.append((p, "sealed evidence is untracked (must be git-tracked when sealed)"))
    #  4b no gitignore line may cover sealed/.
    for hit in gitignore_mentions(root, "sealed"):
        viol.append((hit, "gitignore covers sealed/ (sealing must delete this line)"))
    #  4c under active/ + sealed/, a TRACKED file must not be gitignored (interim force-add /
    #     wrongly-ignored seal). archive/ force-adds are the documented historical exception.
    for tier in ("active", "sealed"):
        tracked = mc.tracked_paths(root, f"experiments/{tier}/")
        for p in check_ignored(root, tracked):
            viol.append((p, f"tracked file under {tier}/ is gitignored "
                            "(interim cell force-added, or a seal wrongly ignored)"))
    #  4d interim/gitignored cells declared in experiments/.gitignore under active/ hold 0 tracked.
    egi = os.path.join(root, "experiments", ".gitignore")
    if os.path.exists(egi):
        with open(egi, encoding="utf-8") as f:
            for ln in f:
                s = ln.strip()
                if not s or s.startswith("#") or not s.startswith("active/"):
                    continue
                celldir = "experiments/" + s.rstrip("/")
                tracked_in = mc.tracked_paths(root, celldir + "/")
                if tracked_in:
                    viol.append((celldir, f"interim/gitignored cell has {len(tracked_in)} "
                                          "tracked file(s) (must not be tracked)"))

    if not viol:
        print(f"OK: INDEX fresh; {len(manifests)} cells fully covered + registered; "
              f"sealed tracked, interim untracked.")
        return 0
    print(f"RED: index/tracking consistency violated ({len(viol)} issue(s)):")
    for rel, reason in viol:
        print(f"  ! {rel}  <-  {reason}")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
