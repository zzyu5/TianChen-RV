#!/usr/bin/env python3
# tools/gates/check_experiments_layout.py — org STAGE2 Lint ① (CI, fail-closed).
#
# Two invariants over experiments/, both fail-closed (exit non-zero = RED):
#
#   (A) DATA-CELL WHITELIST.  experiments/ is a data cell, not a code tree. Its durable
#       content (git-tracked + untracked-not-ignored = fresh-clone lens) may contain ONLY:
#         - MANIFEST + data/evidence : *.md / *.json / *.csv / *.txt / *.objdump / *.log /
#                                      *.err / .gitignore
#         - artifact pointers        : a code/binary artifact (*.kernel.c / *.cpp / *.emitc.mlir /
#                                      sealed *.o / ...) is allowed ONLY when it is REGISTERED as a
#                                      durable file in its owning per-cell MANIFEST.md.
#       ALWAYS RED (harness lives under tools/, never in the data cell):
#         - scripts (*.py/*.sh/*.bash/*.pl/*.zsh) or any file with the +x bit,
#         - harness-named code (run_*/board_*/verify_driver/*_driver.*/*harness*) even if registered,
#         - an UNREGISTERED code/binary artifact (code leaked into the data cell),
#         - an oversized non-evidence, non-registered blob.
#
#   (B) SEALED READ-ONLY.  Files under experiments/sealed/ are immutable evidence. Compared to
#       git HEAD (rename-aware), the ONLY permitted deltas are:
#         - a content-preserving relocation (R100 / C100 — an org move that keeps the bytes),
#         - a MANIFEST.md being added / relocated, or edited as a STALE-flip only.
#       Any content modification (M, or R/C <100%), any newly-added non-MANIFEST evidence, and any
#       deletion under sealed/ -> RED.
#
# SUPERSEDES _attic/tools/lint/check_experiments_data_only.py (judged dead 2026-07-17) (whole whitelist half) and the durable-vs-
# REGISTRY drift half of _attic/tools/lint/check_manifest.py (judged dead 2026-07-17), now that registration is per-cell.
#
#   check_experiments_layout.py             # audit the tree
#   check_experiments_layout.py --self-test # exercise the pure classifiers

import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _manifest_common as mc  # noqa: E402

DATA_EXTS = {".json", ".csv", ".txt", ".md", ".objdump", ".log", ".err"}
DATA_BASENAMES = {".gitignore"}
SCRIPT_EXTS = {".py", ".sh", ".bash", ".pl", ".zsh"}
HARNESS_NAME_RE = re.compile(
    r"(verify_driver|_driver\.|(^|[_/-])harness|^run_|^board_)", re.IGNORECASE
)
OVERSIZE_CAP_BYTES = 512 * 1024


# ---------------------------------------------------------------- (A) whitelist
def classify(relpath, size_bytes, is_exec, registered):
    """Pure whitelist classifier. Returns (ok: bool, reason: str)."""
    base = os.path.basename(relpath)
    ext = os.path.splitext(base)[1].lower()

    if is_exec:
        return False, "executable bit set (harness/binary -> move to tools/)"
    if ext in SCRIPT_EXTS:
        return False, f"script '{ext}' (harness -> move to tools/e2e-harness/)"
    if base in DATA_BASENAMES or ext in DATA_EXTS:
        return True, "data/evidence"
    if ext in mc.CODE_ARTIFACT_EXTS:
        if HARNESS_NAME_RE.search(base):
            return False, "harness code (driver/orchestrator -> move to tools/e2e-harness/)"
        if registered:
            return True, "registered artifact pointer (per-cell MANIFEST durable)"
        return False, "unregistered code/binary artifact (code leaked into data cell)"
    if size_bytes is not None and size_bytes > OVERSIZE_CAP_BYTES and not registered:
        return False, f"oversized non-evidence file ({size_bytes} B > cap)"
    return False, f"unrecognized non-data file (ext '{ext or '<none>'}')"


# ------------------------------------------------------------- (B) sealed R/O
def sealed_classify(code, sim, basename, manifest_stale_only):
    """Pure classifier for one `git diff HEAD` entry whose target is under sealed/.

    code in {A,M,D,R,C}; sim = rename/copy similarity (0..100) or None for A/M/D.
    """
    if basename == "MANIFEST.md":
        if code in ("A", "R", "C"):
            return True, "manifest add/relocate (mutable descriptor)"
        if code == "M":
            return (manifest_stale_only,
                    "manifest STALE-flip" if manifest_stale_only else "manifest edited beyond STALE-flip")
        if code == "D":
            return False, "manifest removed from sealed cell"
        return False, f"manifest unexpected change '{code}'"
    # evidence files: only a content-preserving relocation is allowed.
    if code in ("R", "C"):
        return (sim == 100, f"relocation similarity={sim}%")
    if code == "A":
        return False, "new evidence added to a sealed cell (append forbidden)"
    if code == "M":
        return False, "sealed evidence modified (immutable)"
    if code == "D":
        return False, "sealed evidence deleted (immutable)"
    return False, f"unexpected change '{code}'"


def _manifest_diff_is_stale_only(root, path):
    """True iff the working-tree vs HEAD diff of <path> only touches status/STALE lines."""
    out = subprocess.run(
        ["git", "diff", "HEAD", "--", path],
        cwd=root, capture_output=True, text=True,
    )
    changed = []
    for ln in out.stdout.splitlines():
        if ln.startswith("+++") or ln.startswith("---"):
            continue
        if ln.startswith("+") or ln.startswith("-"):
            changed.append(ln[1:])
    if not changed:
        return True
    for c in changed:
        low = c.lower()
        if "stale" not in low and "status" not in low:
            return False
    return True


def sealed_violations(root):
    viol = []
    # rename/copy-aware diff vs HEAD over ALL of experiments/ (rename detection needs BOTH
    # endpoints visible: an org move relocates old non-sealed paths INTO sealed/, so scoping
    # the pathspec to sealed/ alone would mis-read content-preserving R100 moves as additions).
    # ASCII experiment paths (no tabs) -> safe TSV parse.
    out = subprocess.run(
        ["git", "-c", "core.quotepath=false", "diff", "HEAD", "-M", "-C",
         "--name-status", "--", "experiments/"],
        cwd=root, capture_output=True, text=True,
    )
    for ln in out.stdout.splitlines():
        parts = ln.split("\t")
        if not parts or not parts[0]:
            continue
        code0 = parts[0][0]
        sim = int(parts[0][1:]) if parts[0][1:].isdigit() else None
        target = parts[-1]
        if not target.startswith("experiments/sealed/"):
            continue
        base = os.path.basename(target)
        stale_only = False
        if base == "MANIFEST.md" and code0 == "M":
            stale_only = _manifest_diff_is_stale_only(root, target)
        ok, reason = sealed_classify(code0, sim, base, stale_only)
        if not ok:
            viol.append((target, f"{code0}{sim if sim is not None else ''}: {reason}"))
    # newly dropped untracked (unstaged) files inside a sealed cell are also mutations.
    for p in mc.durable_paths(root, "experiments/sealed/"):
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", "--", p],
            cwd=root, capture_output=True, text=True,
        )
        if tracked.returncode != 0 and os.path.basename(p) != "MANIFEST.md":
            viol.append((p, "A: untracked file dropped into a sealed cell"))
    return viol


# ------------------------------------------------------------------ self-test
def self_test():
    ok_all = True
    wl_cases = [
        # (relpath, size, is_exec, registered, expected_ok)
        ("experiments/active/result-tables/T3.csv", 100, False, False, True),
        ("experiments/sealed/x/evidence.json", 100, False, False, True),
        ("experiments/sealed/x/run_rvv.txt", 100, False, False, True),   # data txt, not harness
        ("experiments/sealed/x/NOTES.md", 100, False, False, True),
        ("experiments/archive/x/a.rv64gcv.objdump", 100, False, False, True),
        ("experiments/.gitignore", 40, False, False, True),
        ("experiments/x/check_manifest.py", 100, False, True, False),    # script even if registered
        ("experiments/x/run_e2e.sh", 100, True, True, False),            # script + exec
        ("experiments/x/board_ab.sh", 100, False, False, False),         # script
        ("experiments/x/verify_driver.c", 100, False, True, False),      # harness name even if registered
        ("experiments/x/kernels/iq1_s.kernel.c", 100, False, True, True),   # registered pointer
        ("experiments/x/kernels/iq1_s.kernel.c", 100, False, False, False), # unregistered -> RED
        ("experiments/x/foo.emitc.mlir", 100, False, True, True),
        ("experiments/x/foo.o", 100, False, True, True),                 # registered sealed .o
        ("experiments/x/foo.o", 100, False, False, False),               # unregistered .o -> RED
        ("experiments/x/blob.bin", 900 * 1024, False, False, False),     # oversized non-evidence
        ("experiments/x/mystery", 10, False, False, False),              # no ext
    ]
    print("-- (A) whitelist classifier --")
    for relpath, size, is_exec, registered, expected in wl_cases:
        got, reason = classify(relpath, size, is_exec, registered)
        mark = "PASS" if got == expected else "FAIL"
        ok_all = ok_all and got == expected
        print(f"  [{mark}] {relpath} exec={int(is_exec)} reg={int(registered)} "
              f"-> ok={got} (expect {expected}) : {reason}")

    print("-- (B) sealed read-only classifier --")
    sc_cases = [
        # (code, sim, basename, stale_only, expected_ok)
        ("R", 100, "evidence.json", False, True),    # content-preserving relocation
        ("R", 74, "evidence.json", False, False),    # renamed WITH edit
        ("C", 100, "evidence.json", False, True),
        ("M", None, "evidence.json", False, False),  # tampered evidence
        ("A", None, "evidence.json", False, False),  # appended evidence
        ("D", None, "evidence.json", False, False),  # removed evidence
        ("A", None, "MANIFEST.md", False, True),     # new per-cell manifest
        ("R", 100, "MANIFEST.md", False, True),      # relocate manifest
        ("M", None, "MANIFEST.md", True, True),      # STALE-flip allowed
        ("M", None, "MANIFEST.md", False, False),    # manifest rewrite -> RED
        ("D", None, "MANIFEST.md", False, False),    # manifest deleted -> RED
    ]
    for code, sim, base, stale_only, expected in sc_cases:
        got, reason = sealed_classify(code, sim, base, stale_only)
        mark = "PASS" if got == expected else "FAIL"
        ok_all = ok_all and got == expected
        print(f"  [{mark}] {code}{sim if sim is not None else ''} {base} stale_only={int(stale_only)} "
              f"-> ok={got} (expect {expected}) : {reason}")

    print("SELF-TEST:", "GREEN" if ok_all else "RED")
    return 0 if ok_all else 1


def main(argv):
    if "--self-test" in argv:
        return self_test()

    root = mc.repo_root()
    durable = mc.durable_paths(root, "experiments/")
    registered = mc.registered_pointers(root)

    wl_viol = []
    code_exempt = 0
    for rel in sorted(durable):
        abspath = os.path.join(root, rel)
        try:
            st = os.stat(abspath)
            size = st.st_size
            is_exec = bool(st.st_mode & 0o111) and not os.path.isdir(abspath)
        except OSError:
            size, is_exec = None, False
        ok, reason = classify(rel, size, is_exec, rel in registered)
        if not ok:
            wl_viol.append((rel, reason))
        elif reason.startswith("registered artifact pointer"):
            code_exempt += 1

    seal_viol = sealed_violations(root)

    rc = 0
    if not wl_viol:
        print(f"OK (A) whitelist: experiments/ is data-only ({len(durable)} durable files; "
              f"{code_exempt} registered artifact pointer(s) exempt).")
    else:
        rc = 1
        print(f"RED (A) whitelist: {len(wl_viol)} non-data file(s) in the data cell:")
        for rel, reason in wl_viol:
            print(f"  ! {rel}  <-  {reason}")

    if not seal_viol:
        print("OK (B) sealed read-only: experiments/sealed/ evidence unchanged vs HEAD "
              "(relocations + MANIFEST descriptors only).")
    else:
        rc = 1
        print(f"RED (B) sealed read-only: {len(seal_viol)} illegal change(s) under sealed/:")
        for rel, reason in seal_viol:
            print(f"  ! {rel}  <-  {reason}")

    if rc:
        print("Fix: move harness/code to tools/ (git mv) or register a genuine evidence pointer in "
              "the cell MANIFEST; never mutate sealed evidence (only a MANIFEST STALE-flip is allowed).")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
