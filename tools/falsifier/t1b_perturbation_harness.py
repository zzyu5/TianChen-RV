#!/usr/bin/env python3
"""tools/falsifier/t1b_perturbation_harness.py -- T1b capability-FACT perturbation harness (E3).

WHAT: perturb ONE valid capability-fact plan into the four T1b perturbation classes,
feed each to the real `weft-opt --weft-check-capability-requires`, and assert each is
FAIL-CLOSED rejected (diagnostic + non-zero exit + no crash + no silent-pass):

  (1) delete_fact          -- delete a declared capability a variant `requires`
  (2) forge_conflict       -- declare a conflict relation that HITS an available capability
  (3) inject_unknown_fact  -- `requires` an undeclared capability symbol (unknown = false)
  (4) vector_absent        -- the vector capability instance is present but NOT available

WHY THIS IS NOT A DUPLICATE OF F-5: `tools/fuzz/f5_failclosed_fuzz.sh` fuzzes the
*typed loop-body op verifier* (attr / posguard / abi / arity / region / bypass). This
harness attacks a DIFFERENT layer: the *capability-fact + requires/conflicts* surface
([D-1]/[I7] "unknown = false" default-deny). Disjoint corpora, disjoint ops; this
harness is NOT registered in schema/f5-failclosed-baseline.v1.json and does not run
through the f5 fingerprint ratchet.

ENFORCEMENT-LAYER ATTRIBUTION (why --no-pass is probed for every scenario):
a scenario can be rejected either by the `weft.exec.variant` OP VERIFIER (fires on
parse+verify, pass or no pass) or ONLY by the pass. The harness records which, by
re-running each input with NO pass. `pass-conditional=yes` means: omit the pass from
the pipeline and this malformed plan is ACCEPTED. That is a real scheduling-dependency
fact about where fail-closed actually lives -- it is recorded, not smoothed over.

ANTI-VACUOUS-GREEN GUARDS (a fail-closed harness must be fail-closed about ITSELF):
  G1 seed precondition : the unmutated seed MUST be accepted. If the seed were already
                         invalid every mutation would "reject" and the run would go
                         green for the wrong reason. Bad seed => FATAL, not green.
  G2 no-op mutation    : a mutation whose output == seed => FATAL. A find/replace that
                         silently fails to match must never be graded as a pass.
  G3 reverse test      : `--self-test` feeds the KNOWN-ACCEPTED seed under an inverted
                         `expected=reject` declaration. The harness MUST grade it RED.
                         Self-test exits 0 iff RED was produced. This proves the RED
                         path actually fires and the harness is not a green stub.

The harness is READ-ONLY over the compiler: it never edits lib/ include/ schema/.
The seed is embedded below (self-contained), so it cannot be broken by another line
editing a shared test file.

Usage:
  t1b_perturbation_harness.py [--opt <weft-opt>] [--csv <out.csv>] [--append-t1b]
                              [--self-test] [--head <sha>] [-v]
  WEFT_OPT=<path> t1b_perturbation_harness.py

Exit 0 iff every scenario is fail-closed. Non-zero on any fail-open / crash / FATAL
guard trip -- the harness is itself fail-closed.
"""

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T1B_CSV = os.path.join(REPO, "experiments/active/result-tables/T1b_failclosed_runtime.csv")
PASS_FLAG = "--weft-check-capability-requires"
CSV_HEADER = ("perturbation_scenario,expected_behavior,observed_behavior,"
              "load_resolution_record_ptr,verdict,snapshot")
EXPECTED = "fail-closed reject: diagnostic + non-zero exit + no crash + no silent-pass"

# --------------------------------------------------------------------- the seed
# A VALID capability-fact plan: an available vector capability, a vector variant that
# requires it, and a portable scalar variant. Accepted (exit 0) under $PASS_FLAG --
# guard G1 re-proves that on every run rather than trusting this comment.
SEED = '''weft.exec.kernel @t1b_vec_dot attributes {} {
  weft.exec.capability @generic_toolchain {id = "generic.toolchain", kind = "toolchain"}
  weft.exec.capability @rvv_vector {id = "rvv.vector", kind = "isa-vector", status = "available"}
  weft.exec.variant @rvv_path attributes {
    origin = "rvv-plugin",
    requires = [@generic_toolchain, @rvv_vector]
  } {
  }
  weft.exec.variant @scalar_path attributes {
    origin = "portable-plugin",
    requires = [@generic_toolchain]
  } {
  }
}
'''

CAP_RVV = ('  weft.exec.capability @rvv_vector {id = "rvv.vector", kind = "isa-vector", '
           'status = "available"}\n')

# conflict payload: @rvv_vector conflicts with an id that an AVAILABLE capability holds.
CAP_RVV_CONFLICT = (
    '  weft.exec.capability @rvv_vector {id = "rvv.vector", kind = "isa-vector", '
    'relations = #weft.capability_relations<conflicts = ["build.policy.no_vector"]>, '
    'status = "available"}\n'
    '  weft.exec.capability @no_vector {id = "build.policy.no_vector", '
    'kind = "build-policy", status = "available"}\n'
)


def _sub(old, new):
    """Build a mutation fn doing one exact string replacement (G2 catches no-ops)."""
    return lambda s: s.replace(old, new)


# id | class | description (NO commas -- CSV is split-on-comma) | mutation
SCENARIOS = [
    ("delete_fact", "capfact-delete",
     "delete the declared @rvv_vector capability fact a variant requires",
     _sub(CAP_RVV, "")),
    ("forge_conflict", "capfact-conflict",
     "forge a conflicts relation that HITS available @no_vector build policy",
     _sub(CAP_RVV, CAP_RVV_CONFLICT)),
    ("inject_unknown_fact", "capfact-unknown",
     "inject undeclared @phantom_ime into requires (unknown = false)",
     _sub("requires = [@generic_toolchain, @rvv_vector]",
          "requires = [@generic_toolchain, @rvv_vector, @phantom_ime]")),
    ("vector_absent", "capfact-vector-absent",
     "vector capability instance present but status=unavailable (absent vector)",
     _sub('kind = "isa-vector", status = "available"',
          'kind = "isa-vector", status = "unavailable"')),
]


def sha12(text):
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def csv_clean(text, width=110):
    """Match the F-5 convention: commas -> ';' so every data row stays 6 fields."""
    out = " ".join(text.replace(",", ";").split())
    return out[:width]


class Runner:
    def __init__(self, opt, workdir):
        self.opt = opt
        self.work = workdir

    def run(self, mlir_text, name, with_pass=True):
        path = os.path.join(self.work, name + ".mlir")
        with open(path, "w") as f:
            f.write(mlir_text)
        cmd = [self.opt, path] + ([PASS_FLAG] if with_pass else [])
        proc = subprocess.run(cmd, capture_output=True, text=True)
        err = proc.stderr
        crash = proc.returncode < 0 or proc.returncode >= 128 or bool(
            re.search(r"PLEASE submit a bug report|Stack dump|Segmentation fault|"
                      r"LLVM ERROR|UNREACHABLE", err))
        m = re.search(r"error: (.*)", err)
        diag = m.group(1).strip() if m else ""
        return proc.returncode, diag, crash


def classify(exit_code, diag, crash):
    """Grade one observation against the fail-closed contract. RED unless truly closed."""
    if crash:
        return "FAIL(crash)", "CRASH exit=%d | %s" % (exit_code, csv_clean(diag))
    if exit_code == 0:
        return ("FAIL(fail-open:silent-pass)",
                "SILENT-PASS exit=0 (no diagnostic; malformed capability plan accepted)")
    if not diag:
        return ("FAIL(non-diagnostic-reject)",
                "REJECTED exit=%d but NO error: diagnostic emitted" % exit_code)
    return "PASS(fail-closed)", None  # observed string is built by the caller (needs layer)


def layer_of(diag, rejected_without_pass):
    """Attribute WHERE fail-closed lives. pass-conditional => the pass is load-bearing."""
    if rejected_without_pass:
        return "verifier-op(weft.exec.variant)", "no"
    return "pass:weft-check-capability-requires", "yes"


def build_rows(runner, head, verbose):
    seed_sha = sha12(SEED)

    # ---- G1: seed precondition. A rejected seed makes every mutation vacuously "pass".
    ec, diag, crash = runner.run(SEED, "seed")
    if crash or ec != 0:
        print("[t1b] FATAL(G1): the unmutated seed is NOT accepted (exit=%d) -- every "
              "mutation would reject vacuously. Refusing to report green.\n  %s"
              % (ec, diag), file=sys.stderr)
        return None, None
    if verbose:
        print("[t1b] G1 ok: seed accepted (exit=0) seed-sha=%s" % seed_sha)

    rows, n_pass, n_fail = [], 0, 0
    print("T1b capability-fact perturbation -- opt=%s  pass=%s  HEAD=%s  seed-sha=%s"
          % (runner.opt, PASS_FLAG, head, seed_sha))
    print("%-22s %-22s %-6s %-34s %s" % ("SCENARIO", "CLASS", "EXIT", "LAYER", "VERDICT"))
    print("-" * 118)

    for sid, cls, desc, mutate in SCENARIOS:
        mutated = mutate(SEED)
        # ---- G2: a mutation that changed nothing must never be graded a pass.
        if mutated == SEED:
            print("[t1b] FATAL(G2): scenario %r produced NO change vs seed (stale "
                  "find/replace). Refusing to report green." % sid, file=sys.stderr)
            return None, None

        ec, diag, crash = runner.run(mutated, sid)
        verdict, observed = classify(ec, diag, crash)

        # layer attribution: does it still reject with the pass omitted?
        ec_np, _, _ = runner.run(mutated, sid + "_nopass", with_pass=False)
        layer, pass_cond = layer_of(diag, rejected_without_pass=(ec_np != 0))

        if observed is None:  # fail-closed: build the observed string with attribution
            observed = ("REJECTED exit=%d layer=%s pass-conditional=%s | %s"
                        % (ec, layer, pass_cond, csv_clean(diag, 60)))
            n_pass += 1
        else:
            n_fail += 1

        print("%-22s %-22s %-6s %-34s %s" % (sid, cls, ec, layer, verdict))
        in_sha = sha12(mutated)
        ptr = ("tools/falsifier/t1b_perturbation_harness.py#%s; in-sha256=%s"
               % (sid, in_sha))
        snap = ("HEAD=%s; seed-sha256=%s; board=n/a(host-static-verify)"
                % (head, seed_sha))
        rows.append("%s:%s -- %s,%s,%s,%s,%s,%s"
                    % (sid, cls, desc, EXPECTED, observed, ptr, verdict, snap))

    print("-" * 118)
    print("fail-closed rate: %d/%d  (fail-open/crash: %d)"
          % (n_pass, len(SCENARIOS), n_fail))
    return rows, n_fail


def self_test(runner, verbose):
    """G3 reverse test: prove the harness grades RED when it should.

    Feed the KNOWN-ACCEPTED seed under the inverted declaration expected=reject.
    The fail-closed contract says 'accepted' must grade RED. If classify() returns
    anything starting with PASS here, the harness is a green stub and self-test FAILS.
    """
    print("=== G3 reverse test: valid seed declared expected=reject -- harness MUST say RED")
    ec, diag, crash = runner.run(SEED, "reverse_seed")
    verdict, observed = classify(ec, diag, crash)
    print("  observed: exit=%d verdict=%s" % (ec, verdict))
    print("  detail  : %s" % (observed or "(none)"))
    if verdict.startswith("PASS"):
        print("  RESULT  : SELF-TEST FAILED -- harness graded an ACCEPTED input as "
              "fail-closed. It cannot detect fail-open. Findings are worthless.",
              file=sys.stderr)
        return 1
    if verdict != "FAIL(fail-open:silent-pass)":
        print("  RESULT  : SELF-TEST FAILED -- expected FAIL(fail-open:silent-pass), "
              "got %r." % verdict, file=sys.stderr)
        return 1
    print("  RESULT  : SELF-TEST PASSED -- harness correctly reported RED "
          "(%s) on a should-be-caught input." % verdict)
    return 0


def write_csv(path, rows):
    with open(path, "w") as f:
        f.write(CSV_HEADER + "\n")
        for r in rows:
            f.write(r + "\n")
    print("[t1b] wrote %d rows -> %s" % (len(rows), path))


def append_t1b(rows):
    """Idempotently append our rows to the T1b table.

    The 20 pre-existing F-5 rows are a SEALED f96f767a evidence snapshot: they are
    copied through byte-identically and never rewritten. Only rows owned by THIS
    harness (matched by scenario id) are refreshed, so re-running is idempotent
    instead of duplicating.
    """
    ours = {sid for sid, _, _, _ in SCENARIOS}
    with open(T1B_CSV) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    header, body = lines[0], lines[1:]
    if header != CSV_HEADER:
        print("[t1b] FATAL: T1b header drifted; refusing to write.\n  want: %s\n  got : %s"
              % (CSV_HEADER, header), file=sys.stderr)
        return 1
    kept = [ln for ln in body if ln.split(":", 1)[0].strip() not in ours]
    dropped = len(body) - len(kept)
    with open(T1B_CSV, "w") as f:
        f.write(header + "\n")
        for ln in kept + rows:
            f.write(ln + "\n")
    print("[t1b] appended %d rows to %s (kept %d pre-existing rows byte-identical; "
          "refreshed %d prior rows of ours)" % (len(rows), T1B_CSV, len(kept), dropped))
    return 0


def locate_opt(override):
    if override:
        return override if os.path.isfile(override) else None
    env = os.environ.get("WEFT_BUILD")
    cands = ([os.path.join(env, "bin", "weft-opt")] if env else []) + [
        os.path.join(REPO, "build-weft/bin/weft-opt"),
        os.path.join(REPO, "build/bin/weft-opt"),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    return shutil.which("weft-opt")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--opt", default=os.environ.get("WEFT_OPT"))
    ap.add_argument("--csv", help="write our rows to this standalone CSV")
    ap.add_argument("--append-t1b", action="store_true",
                    help="idempotently append our rows into the T1b table")
    ap.add_argument("--self-test", action="store_true",
                    help="G3 reverse test only: prove the harness can report RED")
    ap.add_argument("--head")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    opt = locate_opt(args.opt)
    if not opt:
        # No SKIP-green: a harness that cannot run the compiler reports BLOCKED, loudly.
        print("[t1b] BLOCKED: weft-opt not found (pass --opt <path>, set WEFT_OPT, or "
              "set $WEFT_BUILD). NOT reporting green.", file=sys.stderr)
        return 2

    head = args.head or subprocess.run(
        ["git", "-C", REPO, "rev-parse", "--short=8", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "unknown"

    work = tempfile.mkdtemp(prefix="t1b-")
    try:
        runner = Runner(opt, work)
        if args.self_test:
            return self_test(runner, args.verbose)

        rows, n_fail = build_rows(runner, head, args.verbose)
        if rows is None:
            return 3  # a guard tripped: FATAL, never green
        if args.csv:
            write_csv(args.csv, rows)
        if args.append_t1b:
            rc = append_t1b(rows)
            if rc:
                return rc
        return 0 if n_fail == 0 else 1
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
