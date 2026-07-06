#!/usr/bin/env python3
# tools/lint/check_experiments_data_only.py -- experiments/ DATA-ONLY gate (CI, fail-closed).
#
# INVARIANT (A2): experiments/ is a DATA CELL, not a code tree. Its durable
# content (= git-tracked + untracked-not-ignored, i.e. what a fresh clone gets,
# the SAME "durable" lens tools/lint/check_manifest.py uses) may contain ONLY:
#
#   DATA / EVIDENCE  : *.json (measurement cells) / *.csv (measurement tables) /
#                      *.txt (raw / objdump / aggregate evidence) / *.md
#                      (MANIFEST / CADENCE-LAW / NOTES / ledger docs) /
#                      *.objdump / *.log / *.err (build-attempt evidence) / .gitignore
#   REFERENCED CODE  : a code-shaped artifact (kernel-under-test .c/.cpp, .emitc.mlir,
#   ARTIFACT           sealed .o, ...) is allowed ONLY when it is a REGISTERED evidence
#                      pointer in experiments/MANIFEST.md's REGISTRY (MANIFEST CLASS 3
#                      "kernel 源与 emitc / objdump seal" -- the evidence-pointer landing).
#                      An UNREGISTERED code artifact = code leaked into the data cell => RED.
#
# ALWAYS RED (regardless of MANIFEST registration -- harness lives in tools/, never here):
#   - scripts:      *.py / *.sh / *.bash / *.pl / *.zsh, or any file with the +x bit.
#   - harness code: basename looks like a driver / orchestrator (verify_driver,
#                   run_*, board_*, *harness*) even if it is a .c and got registered.
#   - oversized non-evidence: a NON-data, NON-registered file over the size cap.
#
# This is complementary to check_manifest.py: that gate pins durable == REGISTRY
# EXACTLY; THIS gate pins durable content to the data-only WHITELIST (+ registered
# evidence-code exception). Exit non-zero = RED. Current tree self-audit is GREEN.

import os
import re
import subprocess
import sys

SELF_DIR = os.path.dirname(os.path.abspath(__file__))

# --- whitelist of pure DATA / EVIDENCE extensions (always allowed) ---
DATA_EXTS = {".json", ".csv", ".txt", ".md", ".objdump", ".log", ".err"}
DATA_BASENAMES = {".gitignore"}

# --- extensions that are CODE ARTIFACTS: allowed ONLY if registered evidence ---
CODE_ARTIFACT_EXTS = {
    ".c", ".cc", ".cpp", ".cxx", ".h", ".hpp", ".hh",
    ".mlir", ".inc", ".ll", ".s", ".o", ".a", ".so",
}

# --- extensions/marks that are ALWAYS harness/script => never allowed here ---
SCRIPT_EXTS = {".py", ".sh", ".bash", ".pl", ".zsh"}

# --- harness-by-name: code shaped like a driver/orchestrator is RED even if
#     someone registers it as "evidence" (harness belongs in tools/e2e-harness/). ---
HARNESS_NAME_RE = re.compile(
    r"(verify_driver|_driver\.|(^|[_/-])harness|^run_|^board_)", re.IGNORECASE
)

# non-data, non-registered files bigger than this are flagged (evidence + registered
# code are exempt; nothing legitimate in the data cell approaches this today).
OVERSIZE_CAP_BYTES = 512 * 1024


def classify(relpath, size_bytes, is_exec, registered):
    """Pure classifier. Returns (ok: bool, reason: str)."""
    base = os.path.basename(relpath)
    ext = os.path.splitext(base)[1].lower()

    # 1) executables + scripts are ALWAYS harness -> RED.
    if is_exec:
        return False, "executable bit set (harness/binary -> move to tools/)"
    if ext in SCRIPT_EXTS:
        return False, f"script '{ext}' (harness -> move to tools/e2e-harness/)"

    # 2) pure data / evidence -> GREEN.
    if base in DATA_BASENAMES or ext in DATA_EXTS:
        return True, "data/evidence"

    # 3) code artifacts: allowed ONLY as a REGISTERED evidence pointer, and never
    #    when the name looks like a driver/orchestrator.
    if ext in CODE_ARTIFACT_EXTS:
        if HARNESS_NAME_RE.search(base):
            return False, "harness code (driver/orchestrator -> move to tools/e2e-harness/)"
        if registered:
            return True, "registered evidence-code (MANIFEST CLASS 3 pointer)"
        return False, "unregistered code artifact (code leaked into data cell)"

    # 4) oversized non-evidence backstop.
    if size_bytes is not None and size_bytes > OVERSIZE_CAP_BYTES and not registered:
        return False, f"oversized non-evidence file ({size_bytes} B > cap)"

    # 5) unknown extension, not data, not registered code -> RED.
    return False, f"unrecognized non-data file (ext '{ext or '<none>'}')"


def repo_root():
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=SELF_DIR, capture_output=True, text=True, check=True,
    )
    return out.stdout.strip()


def durable_set(root):
    """tracked + untracked-not-ignored under experiments/ (fresh-clone lens)."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z",
         "--", "experiments/"],
        cwd=root, capture_output=True, check=True,
    )
    return [p for p in out.stdout.decode("utf-8").split("\0") if p]


def registered_set(root):
    """Registered evidence paths parsed from experiments/MANIFEST.md REGISTRY."""
    manifest = os.path.join(root, "experiments", "MANIFEST.md")
    reg = set()
    if not os.path.exists(manifest):
        return reg
    with open(manifest, encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"<!-- REGISTRY:BEGIN -->(.*?)<!-- REGISTRY:END -->", text, re.S)
    if not m:
        return reg
    for line in m.group(1).splitlines():
        for tok in re.findall(r"`([^`]+)`", line):
            if tok.startswith("experiments/"):
                reg.add(tok)
                break
    return reg


def self_test():
    """Exercise the pure classifier on synthetic cases (no tree access)."""
    cases = [
        # (relpath, size, is_exec, registered, expected_ok)
        ("experiments/T3.csv", 100, False, False, True),
        ("experiments/results/evidence.json", 100, False, False, True),
        ("experiments/results/run_rvv.txt", 100, False, False, True),
        ("experiments/NOTES.md", 100, False, False, True),
        ("experiments/x.rv64gcv.objdump", 100, False, False, True),
        ("experiments/.gitignore", 40, False, False, True),
        ("experiments/aggregate_summary.txt", 100, False, False, True),  # data, not harness
        ("experiments/check_manifest.py", 100, False, True, False),      # script even if registered
        ("experiments/run_e2e.sh", 100, True, True, False),              # script + exec
        ("experiments/board_ab.sh", 100, False, False, False),           # script
        ("experiments/verify_driver.c", 100, False, True, False),        # harness-name even if registered
        ("experiments/kernels/iq1_s.kernel.c", 100, False, True, True),  # registered evidence-code
        ("experiments/kernels/iq1_s.kernel.c", 100, False, False, False),# same, unregistered -> RED
        ("experiments/foo.emitc.mlir", 100, False, True, True),          # registered emitc
        ("experiments/foo.o", 100, False, True, True),                   # registered sealed .o
        ("experiments/foo.o", 100, False, False, False),                 # unregistered .o -> RED
        ("experiments/blob.bin", 900 * 1024, False, False, False),       # oversized non-evidence
        ("experiments/mystery", 10, False, False, False),                # no ext, unknown
    ]
    ok_all = True
    for relpath, size, is_exec, registered, expected in cases:
        got, reason = classify(relpath, size, is_exec, registered)
        mark = "PASS" if got == expected else "FAIL"
        if got != expected:
            ok_all = False
        print(f"  [{mark}] {relpath} exec={int(is_exec)} reg={int(registered)} "
              f"-> ok={got} (expect {expected}) : {reason}")
    print("SELF-TEST:", "GREEN" if ok_all else "RED")
    return 0 if ok_all else 1


def main(argv):
    if "--self-test" in argv:
        return self_test()

    root = repo_root()
    durable = durable_set(root)
    registered = registered_set(root)

    violations = []
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
            violations.append((rel, reason))
        elif reason.startswith("registered evidence-code"):
            code_exempt += 1

    if not violations:
        print(f"OK: experiments/ is data-only ({len(durable)} durable files; "
              f"{code_exempt} registered evidence-code pointer(s) exempt).")
        return 0

    print(f"RED: experiments/ contains {len(violations)} non-data file(s) "
          f"(code/harness/executable leaked into the data cell):")
    for rel, reason in violations:
        print(f"  ! {rel}  <-  {reason}")
    print("Fix: move harness/code to tools/ (git mv), or -- for a genuine evidence "
          "pointer -- register it in experiments/MANIFEST.md REGISTRY (CLASS 3).")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
