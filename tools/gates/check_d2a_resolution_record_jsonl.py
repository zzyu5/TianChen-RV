#!/usr/bin/env python3
"""check_d2a_resolution_record_jsonl.py -- [D-2a] loading-time resolution-record CI gate.

Makes the [D-2a] LOADING-TIME resolution record ([D-4] stage (2), [C1-1] head-claim second
half "starter" 起步) CI-executable. A deployment process resolves its capability instance ONCE
at startup and drops ONE per-process record of the documented shape
{declared_instance_hash, resolved_variant_set, ts}. The sibling unittest
(weft-load-time-resolution-test) is the load-time resolver's producer; run with
`--emit-jsonl=<path>` it drops exactly that one per-process record. This gate validates -- WITHOUT
touching the resolver logic (read-only artifact check) -- that:

  (V1) SHAPE       : every record carries the three required keys; declared_instance_hash is a
                     lowercase-hex digest; resolved_variant_set is a SORTED, DEDUPLICATED list of
                     strings (an I4 order-invariant MIRROR, not a route authority).
  (V2) PER-PROCESS : [D-4] stage (2) hard gate = EXACTLY ONE record per process. More than one
                     record from a single process == a per-dispatch drop leak (a [NG-3] violation:
                     per-dispatch enforcement is FOREVER forbidden -- the resolver must compute
                     once and drop one record, the hot path reading the cache). Zero records ==
                     no resolution dropped.

The [I3] zero-family-branch and [I7] fail-closed守法 (a variant keyed on family/symbol NAME rather
than capability FACT; a route synthesized from the mirror record) are machine-checked by the
sibling unittest itself: its assertions return non-zero on any such regression, and this gate's
real mode runs that binary FIRST -- a non-zero exit is RED before any record is even validated. So
a resolver that (a) introduced a per-dispatch check, (b) branched on a family name, or (c) stopped
failing closed makes this gate RED.

The gate changes NO resolver judgement -- it is an artifact-shape + per-process-count falsifier
only. It does NOT gate on ISSUE-105 deployment (that is 判据级·待裁, D single-track); it neither
reads nor mutates GEN_SEAL / master.

Two modes (sibling idiom of check_f4_attribution_jsonl.py):
  --self-test : hermetic classifier discrimination on synthetic records. Proves the validator FIRES
     (missing key / bad hash / unsorted-or-dup resolved set / non-string entry / != 1 record per
     process -> RED) and STAYS GREEN on a compliant one-record process. No binary needed.
  (default)   : locates weft-load-time-resolution-test (build/weft/bin, $WEFT_BUILD/bin, or --bin),
     runs it with --emit-jsonl, parses the emitted per-process JSONL, classifies. If the binary is
     absent the build-free lane SKIPs (exit 0); pass --require-binaries to make the absence RED.

Stdlib-only.
Usage:  python3 tools/gates/check_d2a_resolution_record_jsonl.py [--self-test] [-v] [--bin PATH]
                 [--require-binaries]
Exit:   0 GREEN ; 1 RED (validation failure) ; 2 setup error.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The [D-2a] per-process resolution record shape {declared_instance_hash, resolved_variant_set, ts}.
REQUIRED_KEYS = {"declared_instance_hash", "resolved_variant_set", "ts"}
_HEX_RE = re.compile(r"^[0-9a-f]+$")

BIN_NAME = "weft-load-time-resolution-test"


# ---------------------------------------------------------------------------
# Pure validator core (fed synthetic records by --self-test; real JSONL at runtime).
# ---------------------------------------------------------------------------
def validate_record(rec):
    """One resolution record -> list of violation strings (empty == valid)."""
    v = []
    if not isinstance(rec, dict):
        return [f"record is not a JSON object: {rec!r}"]

    missing = REQUIRED_KEYS - set(rec)
    if missing:
        v.append(f"missing required key(s): {sorted(missing)}")
        return v

    dih = rec["declared_instance_hash"]
    if not (isinstance(dih, str) and _HEX_RE.match(dih)):
        v.append(f"declared_instance_hash not a lowercase-hex digest: {dih!r}")

    rvs = rec["resolved_variant_set"]
    if not isinstance(rvs, list):
        v.append(f"resolved_variant_set is not a list: {rvs!r}")
    else:
        if not all(isinstance(x, str) for x in rvs):
            v.append(f"resolved_variant_set has non-string entry: {rvs!r}")
        elif rvs != sorted(rvs):
            v.append(f"resolved_variant_set not sorted (I4 mirror must be order-invariant): {rvs!r}")
        elif len(set(rvs)) != len(rvs):
            v.append(f"resolved_variant_set has duplicate entry: {rvs!r}")

    if not isinstance(rec["ts"], str):
        v.append(f"ts is not a string: {rec['ts']!r}")
    return v


def classify_process(name, records):
    """One process file (name, records) -> list of violation strings (empty == GREEN)."""
    violations = []
    # (V2) [D-4] stage (2) hard gate: EXACTLY ONE record per process.
    if len(records) != 1:
        violations.append(
            f"{name}: emitted {len(records)} record(s) for one process "
            "([D-4] stage (2) hard gate = per-process EXACTLY 1; >1 == per-dispatch drop leak "
            "([NG-3]), 0 == no resolution dropped)")
    for i, rec in enumerate(records):
        for msg in validate_record(rec):
            violations.append(f"{name}[record {i}]: {msg}")
    return violations


# ---------------------------------------------------------------------------
def _on_path(name):
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if d and os.path.isfile(os.path.join(d, name)):
            return os.path.join(d, name)
    return None


def locate_bin(override):
    if override:
        return override if os.path.isfile(override) else _on_path(override)
    candidates = []
    env = os.environ.get("WEFT_BUILD")
    if env:
        candidates.append(os.path.join(env, "bin", BIN_NAME))
    candidates.append(os.path.join(REPO, "build", "weft", "bin", BIN_NAME))
    candidates.append(os.path.join(REPO, "build", "bin", BIN_NAME))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def run_real(verbose, bin_override, require_binaries):
    binary = locate_bin(bin_override)
    if not binary:
        msg = (f"[d2a-resolution] {BIN_NAME} not built "
               "(looked under $WEFT_BUILD/bin and build/weft/bin)")
        if require_binaries:
            print(msg + " -- RED (--require-binaries)")
            return 2
        print(msg + " -- SKIP (build-free lane; pass --require-binaries to gate)")
        return 0

    with tempfile.TemporaryDirectory() as tmp:
        jsonl_path = os.path.join(tmp, "resolution.jsonl")
        cmd = [binary, f"--emit-jsonl={jsonl_path}"]
        if verbose:
            print(f"  $ {' '.join(cmd)}")
        # The producer runs the [I3]/[I7]/[NG-3] machine-checks FIRST; a non-zero exit is a
        # 守法 regression (family-name branch / non-fail-closed / per-dispatch) -> RED here.
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"[d2a-resolution] RED: {BIN_NAME} exited {proc.returncode} "
                  "(a load-time resolver 守法 machine-check failed [I3]/[I7]/[NG-3]):")
            print(proc.stdout + proc.stderr)
            return 1
        if not os.path.isfile(jsonl_path):
            print(f"[d2a-resolution] setup error: producer emitted no JSONL at {jsonl_path}")
            return 2
        records = []
        with open(jsonl_path) as jf:
            for ln in jf.read().splitlines():
                ln = ln.strip()
                if ln:
                    records.append(json.loads(ln))

    violations = classify_process(BIN_NAME, records)
    if verbose:
        print(f"  {BIN_NAME}: {len(records)} per-process record(s) "
              f"resolved_variant_set={[r.get('resolved_variant_set') for r in records]}")
    if violations:
        print(f"[d2a-resolution] RED: {len(violations)} resolution-record violation(s)")
        for v in violations:
            print(f"  - {v}")
        return 1
    print("[d2a-resolution] GREEN: per-process EXACTLY 1 record, shape "
          "{declared_instance_hash, resolved_variant_set, ts} intact "
          "([D-4] stage (2) hard gate + [NG-3] no per-dispatch leak)")
    return 0


# ---------------------------------------------------------------------------
def run_self_test():
    """Prove the validator DISCRIMINATES before it judges the real JSONL."""
    fails = []

    def check(label, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            fails.append(label)

    def rec(dih="d40ed4ed", rvs=None, ts="7", **over):
        base = {"declared_instance_hash": dih,
                "resolved_variant_set": [] if rvs is None else rvs, "ts": ts}
        base.update(over)
        return base

    # ---- per-record validator ----
    check("valid record (hex hash + sorted uniq rvs + str ts) -> valid",
          validate_record(rec(rvs=["a_var", "rvv_wide"])) == [])
    check("empty resolved_variant_set (fail-closed no-cap instance) -> valid",
          validate_record(rec(rvs=[])) == [])

    check("missing required key -> RED",
          any("missing required key" in m for m in validate_record({"ts": "0"})))
    check("non-hex declared_instance_hash -> RED",
          any("declared_instance_hash" in m for m in validate_record(rec(dih="NOTHEX!"))))
    check("resolved_variant_set not sorted -> RED (I4 mirror order-invariant)",
          any("not sorted" in m for m in validate_record(rec(rvs=["z", "a"]))))
    check("resolved_variant_set duplicate -> RED",
          any("duplicate" in m for m in validate_record(rec(rvs=["a", "a"]))))
    check("resolved_variant_set non-string entry -> RED",
          any("non-string" in m for m in validate_record(rec(rvs=[1, 2]))))
    check("resolved_variant_set not a list -> RED",
          any("not a list" in m for m in validate_record(rec(rvs="rvv_wide"))))
    check("ts not a string -> RED",
          any("ts is not a string" in m for m in validate_record(rec(ts=7))))

    # ---- process-level: EXACTLY ONE record per process ([D-4]② / [NG-3]) ----
    check("one-record process -> GREEN",
          classify_process("p", [rec(rvs=["rvv_wide"])]) == [])
    check("two records (per-dispatch leak, [NG-3]) -> RED",
          any("per-dispatch drop leak" in m
              for m in classify_process("p", [rec(), rec()])))
    check("zero records (no resolution dropped) -> RED",
          any("hard gate = per-process EXACTLY 1" in m
              for m in classify_process("p", [])))

    if fails:
        print(f"[d2a-resolution --self-test] RED: {len(fails)} discrimination(s) failed")
        return 2
    print("[d2a-resolution --self-test] GREEN: validator discriminates all cases")
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
        bin_override=opt_arg("--bin"),
        require_binaries="--require-binaries" in argv,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
