#!/usr/bin/env python3
"""check_f5_failclosed_fingerprint.py -- [F-5] fail-closed fuzz CI gate (M1 evidence line).

Makes the [F-5] fail-closed fuzzer (tools/fuzz/f5_failclosed_fuzz.sh) CI-executable as a
NON-REGRESSION ratchet. The fuzzer mutates ONE verified-valid typed-region program into a
batch of illegal variants, feeds each to `weft-opt --weft-rvv-lower-to-emitc`, and asserts
each is fail-closed rejected (graceful diagnostic + non-zero exit + no crash + no silent
pass). This gate compares the per-scenario verdicts against a COMMITTED fingerprint baseline
(schema/f5-failclosed-baseline.v1.json) and fails closed on:

  (R1) REGRESSION      : a scenario the baseline expects 'fail-closed' is observed FAIL
                         (fail-open silent-pass / crash / non-diagnostic reject).
  (R2) CORPUS DRIFT    : a baseline-pinned scenario is missing from the run, OR a run
                         scenario is undeclared in the baseline. This pins the fuzz corpus
                         so nobody deletes a scenario to green the gate.

The gate is a '指纹不退化 / fail-closed 覆盖不退化' ratchet -- it is NOT a hard '20/20'
requirement. A scenario legitimately KNOWN to fail-open may be declared
expected='known-fail-open' in the baseline; it is then TOLERATED (not RED) and only surfaces
an ADVISORY (not RED) if it later starts passing (baseline should tighten).

HONEST NOTE: T1b (recorded @ f96f767a) reported 17/20 with three fail-OPEN
(neg_qk / neg_weight_stride / neg_activ_stride). Those three are CLOSED @ HEAD 1bb06375
(re-verified, identical seed): the verifier now rejects non-positive i64 block facts. The
committed baseline records the true 20/20 fail-closed HEAD state and ratchets against
regression from there.

Two modes (sibling idiom of the falsifier-gate.yml gates):
  --self-test : hermetic classifier discrimination on synthetic verdict tables. Proves the
     classifier FIRES (regression -> RED, corpus drift -> RED) and TOLERATES declared gaps
     (known-fail-open observed FAIL -> GREEN; observed PASS -> GREEN + advisory) before it
     judges the real tree. No compiler needed.
  (default)  : locates weft-opt (build/bin, $WEFT_BUILD/bin, or --opt), runs the fuzz
     harness with --csv, parses the per-scenario verdicts, classifies against the baseline.
     If weft-opt is absent the build-free lane SKIPs (exit 0) so binary-free CI stays green;
     pass --require-binaries to make the absence itself RED.

Stdlib-only.
Usage:  python3 tools/lint/check_f5_failclosed_fingerprint.py [--self-test] [-v]
                 [--opt PATH] [--require-binaries]
Exit:   0 GREEN ; 1 RED (regression / corpus drift) ; 2 setup error.
"""
import json
import os
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HARNESS = os.path.join(REPO, "tools/fuzz/f5_failclosed_fuzz.sh")
BASELINE = os.path.join(REPO, "schema/f5-failclosed-baseline.v1.json")


# ---------------------------------------------------------------------------
# Pure classifier core (fed synthetic inputs by --self-test; real run at runtime).
# ---------------------------------------------------------------------------
def classify(baseline_scenarios, observed):
    """(baseline dict {id: {expected: ...}}, observed dict {id: 'PASS'|'FAIL'})
    -> (violations, advisories).

    A verdict is a PASS iff it fail-closes. 'expected' is 'fail-closed' or
    'known-fail-open'. Empty violations == GREEN.
    """
    violations = []
    advisories = []

    base_ids = set(baseline_scenarios)
    obs_ids = set(observed)

    # (R2) corpus drift -- pin the scenario set both ways.
    for missing in sorted(base_ids - obs_ids):
        violations.append(f"corpus-drift(missing): baseline scenario {missing!r} "
                          "not present in fuzz run")
    for extra in sorted(obs_ids - base_ids):
        violations.append(f"corpus-drift(undeclared): fuzz scenario {extra!r} "
                          "not declared in baseline")

    # (R1) per-scenario verdict vs expectation.
    for sid in sorted(base_ids & obs_ids):
        expected = baseline_scenarios[sid].get("expected", "fail-closed")
        passed = observed[sid] == "PASS"
        if expected == "fail-closed":
            if not passed:
                violations.append(
                    f"regression: {sid!r} expected fail-closed but observed FAIL "
                    "(fail-open / crash / non-diagnostic)")
        elif expected == "known-fail-open":
            if passed:
                advisories.append(
                    f"tightening-available: {sid!r} declared known-fail-open now "
                    "fail-closes -- promote baseline to 'fail-closed'")
            # observed FAIL for a known gap is tolerated (not a regression).
        else:
            violations.append(
                f"bad-baseline: {sid!r} has unknown expected={expected!r}")

    return violations, advisories


# ---------------------------------------------------------------------------
def load_baseline():
    with open(BASELINE) as f:
        data = json.load(f)
    scenarios = data.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios:
        raise RuntimeError(f"baseline has no scenarios: {BASELINE}")
    return scenarios


def locate_opt(override):
    if override:
        return override if (os.path.isfile(override) or _on_path(override)) else None
    candidates = []
    env = os.environ.get("WEFT_BUILD")
    if env:
        candidates.append(os.path.join(env, "bin", "weft-opt"))
    candidates.append(os.path.join(REPO, "build", "bin", "weft-opt"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def _on_path(name):
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if d and os.path.isfile(os.path.join(d, name)):
            return True
    return False


def parse_csv_verdicts(csv_path):
    """Parse the harness CSV into {scenario_id: 'PASS'|'FAIL'}.

    Row 0 col is `id:class -- desc`; the verdict is column index 4 and starts with
    'PASS(' iff the scenario fail-closed. Harness csv_clean() maps commas in the free
    text fields to ';', so every data row is exactly six comma-separated fields.
    """
    observed = {}
    with open(csv_path) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError("harness produced an empty CSV")
    for ln in lines[1:]:  # skip header
        fields = ln.split(",")
        if len(fields) < 5:
            raise RuntimeError(f"malformed CSV row (want >=5 fields): {ln!r}")
        sid = fields[0].split(":", 1)[0].strip()
        verdict = fields[4].strip()
        observed[sid] = "PASS" if verdict.startswith("PASS") else "FAIL"
    return observed


def run_real(verbose, opt_override, require_binaries):
    if not os.path.isfile(HARNESS):
        print(f"[f5-fingerprint] setup error: harness missing: {HARNESS}")
        return 2
    if not os.path.isfile(BASELINE):
        print(f"[f5-fingerprint] setup error: baseline missing: {BASELINE}")
        return 2

    opt = locate_opt(opt_override)
    if not opt:
        msg = ("[f5-fingerprint] weft-opt not built "
               "(looked under $WEFT_BUILD/bin and build/bin)")
        if require_binaries:
            print(msg + " -- RED (--require-binaries)")
            return 2
        print(msg + " -- SKIP (build-free lane; pass --require-binaries to gate)")
        return 0

    try:
        baseline = load_baseline()
    except Exception as e:  # noqa: BLE001
        print(f"[f5-fingerprint] setup error: {e}")
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "f5.csv")
        proc = subprocess.run(
            ["bash", HARNESS, "--opt", opt, "--csv", csv_path],
            capture_output=True, text=True)
        if verbose:
            print(proc.stdout)
        # Harness exits: 0 all fail-closed; 1 a scenario fail-opened/crashed (still
        # writes the CSV -- we classify it); 2/3 setup/fatal (no usable CSV).
        if proc.returncode not in (0, 1) or not os.path.isfile(csv_path):
            print("[f5-fingerprint] setup error: fuzz harness failed to produce a CSV "
                  f"(exit={proc.returncode})")
            sys.stderr.write(proc.stderr)
            return 2
        try:
            observed = parse_csv_verdicts(csv_path)
        except Exception as e:  # noqa: BLE001
            print(f"[f5-fingerprint] setup error: {e}")
            return 2

    violations, advisories = classify(baseline, observed)

    n_pass = sum(1 for v in observed.values() if v == "PASS")
    if verbose:
        print(f"  fail-closed: {n_pass}/{len(observed)}  "
              f"(baseline pins {len(baseline)} scenarios)")

    for a in advisories:
        print(f"[f5-fingerprint] ADVISORY: {a}")

    if violations:
        print(f"[f5-fingerprint] RED: {len(violations)} fail-closed coverage "
              "regression / corpus drift")
        for v in violations:
            print(f"  - {v}")
        return 1

    print(f"[f5-fingerprint] GREEN: fail-closed coverage did not regress "
          f"({n_pass}/{len(observed)} fail-closed; corpus pinned to baseline)")
    return 0


# ---------------------------------------------------------------------------
def run_self_test():
    """Prove the classifier DISCRIMINATES before it judges the real fuzz run."""
    fails = []

    def check(label, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            fails.append(label)

    base = {
        "a": {"expected": "fail-closed"},
        "b": {"expected": "fail-closed"},
        "c": {"expected": "fail-closed"},
    }

    # compliant: every fail-closed scenario passes -> GREEN
    v, adv = classify(base, {"a": "PASS", "b": "PASS", "c": "PASS"})
    check("all-fail-closed observed PASS -> GREEN", v == [] and adv == [])

    # (R1) regression: a fail-closed scenario fail-opened -> RED
    v, _ = classify(base, {"a": "PASS", "b": "FAIL", "c": "PASS"})
    check("fail-closed scenario observed FAIL -> RED",
          any(x.startswith("regression") for x in v))

    # (R2) corpus drift -- missing pinned scenario -> RED
    v, _ = classify(base, {"a": "PASS", "b": "PASS"})
    check("baseline scenario missing from run -> RED",
          any("corpus-drift(missing)" in x for x in v))

    # (R2) corpus drift -- undeclared scenario -> RED
    v, _ = classify(base, {"a": "PASS", "b": "PASS", "c": "PASS", "z": "PASS"})
    check("undeclared fuzz scenario -> RED",
          any("corpus-drift(undeclared)" in x for x in v))

    # known-fail-open TOLERATED: declared gap observed FAIL -> GREEN (not a regression)
    base_gap = dict(base, d={"expected": "known-fail-open"})
    v, adv = classify(base_gap,
                      {"a": "PASS", "b": "PASS", "c": "PASS", "d": "FAIL"})
    check("declared known-fail-open observed FAIL -> GREEN (tolerated)",
          v == [] and adv == [])

    # known-fail-open IMPROVED: declared gap now passes -> GREEN + advisory
    v, adv = classify(base_gap,
                      {"a": "PASS", "b": "PASS", "c": "PASS", "d": "PASS"})
    check("declared known-fail-open now fail-closes -> GREEN + advisory",
          v == [] and any(x.startswith("tightening-available") for x in adv))

    # CSV parse: id extraction + verdict class from a realistic harness row.
    import io
    csv_text = (
        "perturbation_scenario,expected_behavior,observed_behavior,ptr,verdict,snapshot\n"
        "neg_qk:posguard -- qk=-32 (NEGATIVE block element count),want reject,"
        "REJECTED exit=1 layer=verify | 'op' invalid,ptr#neg_qk,PASS(fail-closed),snap\n"
        "bad_kind:attr -- kind=plain,want reject,"
        "SILENT-PASS exit=0,ptr#bad_kind,FAIL(fail-open:silent-pass),snap\n")
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as tf:
        tf.write(csv_text)
        tf_path = tf.name
    try:
        parsed = parse_csv_verdicts(tf_path)
    finally:
        os.unlink(tf_path)
    check("CSV parse: neg_qk->PASS, bad_kind->FAIL",
          parsed == {"neg_qk": "PASS", "bad_kind": "FAIL"})

    # The committed baseline must load and pin a non-empty corpus.
    try:
        real = load_baseline()
        check("committed baseline loads with a non-empty corpus", bool(real))
    except Exception as e:  # noqa: BLE001
        check(f"committed baseline loads ({e})", False)

    if fails:
        print(f"[f5-fingerprint --self-test] RED: {len(fails)} discrimination(s) failed")
        return 2
    print("[f5-fingerprint --self-test] GREEN: classifier discriminates all cases")
    return 0


def main(argv):
    verbose = "-v" in argv or "--verbose" in argv
    if "--self-test" in argv:
        return run_self_test()

    def opt_arg(flag):
        if flag in argv:
            i = argv.index(flag)
            if i + 1 < len(argv):
                return argv[i + 1]
        return None

    return run_real(
        verbose,
        opt_override=opt_arg("--opt"),
        require_binaries="--require-binaries" in argv,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
