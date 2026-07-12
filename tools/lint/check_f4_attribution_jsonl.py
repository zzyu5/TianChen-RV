#!/usr/bin/env python3
"""check_f4_attribution_jsonl.py -- [F-4] attribution-completeness CI gate (M1 evidence line).

Makes the [F-4] COMPILE-TIME SELECTION attribution JSONL export ([D-4] stage (1)) CI-executable.
The `--tcrv-select-variants=attribution-jsonl=<path>` sink writes ONE canonical-JSON object per
planned kernel: {candidates[], chosen, declared_instance_hash, kernel, keys_evaluated{}, reason,
ts}. This gate runs tcrv-opt over the COMMITTED attribution lit corpus, parses the emitted JSONL,
and validates -- WITHOUT touching the selection logic (read-only export check) -- that:

  (V1) SHAPE       : every record carries the seven required top-level keys; declared_instance_hash
                     is a lowercase-hex digest; candidates is a list of {feasible, ...} objects.
  (V2) ENUM        : reason is null OR one of {only_feasible, static_order, prior, measured}
                     ([D-4] (1) primary key). No other token may appear.
  (V3) CONSISTENCY : the reason is DERIVABLE from the record -- only_feasible <=> exactly one
                     feasible candidate + chosen; static_order <=> >=2 feasible + chosen;
                     null <=> nothing chosen (NoViableVariant). This catches a sink that emits a
                     reason the candidate set does not support.
  (V4) COVERAGE    : [D-4] M1 hard gate = 100% -- every kernel op in the corpus produces exactly
                     one record (a silently-dropped kernel = an incomplete C_attr^CT denominator).
  (V5) M1 ENUM     : the export EXERCISES both M1 reason tokens {only_feasible, static_order}
                     across the corpus (查①②). A regression that stops emitting either branch is
                     RED -- the export-completeness claim is what F-4 is about.

The gate changes NO selection judgement / NO reason-enum semantics -- it is a serialization-output
falsifier only. It does NOT gate on prior/measured appearing (those are valid enum members reserved
for [SEL-1]/[SEL-3]; whether static_order goes to zero is a burn-down signal, not a gate --
canon attribution-reason=static_order).

Two modes (sibling idiom of the falsifier-gate.yml gates):
  --self-test : hermetic classifier discrimination on synthetic JSONL records. Proves the validator
     FIRES (missing key / bad reason token / reason-not-supported-by-candidates / dropped kernel /
     M1 branch never emitted -> RED) and STAYS GREEN on a compliant corpus. No compiler needed.
  (default)   : locates tcrv-opt (build/bin, $TCRV_BUILD/bin, or --opt), runs the pass pipeline
     parsed from each corpus file's own `// RUN:` line (single source of truth), parses the emitted
     JSONL, classifies. If tcrv-opt is absent the build-free lane SKIPs (exit 0) so binary-free CI
     stays green; pass --require-binaries to make the absence itself RED.

Stdlib-only.
Usage:  python3 tools/lint/check_f4_attribution_jsonl.py [--self-test] [-v] [--opt PATH]
                 [--require-binaries]
Exit:   0 GREEN ; 1 RED (validation failure) ; 2 setup error.
"""
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The COMMITTED compile-time selection attribution lit corpus ([D-4] stage (1)). Each file's own
# `// RUN:` line (the one that writes attribution-jsonl=) is the single source of truth for the
# pass pipeline -- deleting a file is a setup error (RED), so the corpus cannot be shrunk to green.
CORPUS = [
    "test/Transforms/VariantSelection/attribution-jsonl.mlir",
    "test/Transforms/VariantSelection/attribution-jsonl-instance-hash.mlir",
]

REQUIRED_KEYS = {
    "candidates", "chosen", "declared_instance_hash", "kernel", "keys_evaluated", "reason", "ts",
}
# [D-4] (1) reason primary key. null (nothing chosen) is also legal.
VALID_REASONS = {"only_feasible", "static_order", "prior", "measured"}
# [D-4] M1 查①②: the two reason tokens stage (1) emits today and must keep exercising.
M1_REQUIRED_REASONS = {"only_feasible", "static_order"}
_HEX_RE = re.compile(r"^[0-9a-f]+$")
_KERNEL_RE = re.compile(r"(?m)^\s*tcrv\.exec\.kernel\s+@")


# ---------------------------------------------------------------------------
# Pure validator core (fed synthetic records by --self-test; real JSONL at runtime).
# ---------------------------------------------------------------------------
def validate_record(rec):
    """One attribution record -> list of violation strings (empty == valid)."""
    v = []
    if not isinstance(rec, dict):
        return [f"record is not a JSON object: {rec!r}"]

    missing = REQUIRED_KEYS - set(rec)
    if missing:
        v.append(f"missing required key(s): {sorted(missing)}")
        # Without the shape we cannot check the rest coherently.
        return v

    reason = rec["reason"]
    if reason is not None and reason not in VALID_REASONS:
        v.append(f"reason {reason!r} not in {sorted(VALID_REASONS)} | null")

    cands = rec["candidates"]
    if not isinstance(cands, list):
        v.append(f"candidates is not a list: {cands!r}")
        return v
    for i, c in enumerate(cands):
        if not isinstance(c, dict) or "feasible" not in c:
            v.append(f"candidate[{i}] missing 'feasible' flag: {c!r}")
    feasible_count = sum(1 for c in cands if isinstance(c, dict) and c.get("feasible") is True)

    chosen = rec["chosen"]
    dih = rec["declared_instance_hash"]
    if not (isinstance(dih, str) and _HEX_RE.match(dih)):
        v.append(f"declared_instance_hash not a lowercase-hex digest: {dih!r}")

    # (V3) reason must be DERIVABLE from the candidate set.
    if reason is None:
        if chosen is not None:
            v.append(f"reason=null but chosen={chosen!r} (NoViableVariant must have chosen=null)")
    elif reason == "only_feasible":
        if chosen is None:
            v.append("reason=only_feasible but chosen=null")
        if feasible_count != 1:
            v.append(f"reason=only_feasible but feasible candidate count={feasible_count} (want 1)")
    elif reason == "static_order":
        if chosen is None:
            v.append("reason=static_order but chosen=null")
        if feasible_count < 2:
            v.append(f"reason=static_order but feasible candidate count={feasible_count} (want >=2)")
    else:  # prior / measured -- reserved but valid enum members
        if chosen is None:
            v.append(f"reason={reason!r} but chosen=null")
    return v


def classify_corpus(files):
    """files: [{name, records, expected_kernels}] -> list of violation strings (empty == GREEN)."""
    violations = []
    reasons_seen = set()
    for f in files:
        name, recs, expected = f["name"], f["records"], f["expected_kernels"]
        # (V4) 100% coverage: one record per kernel op, none silently dropped.
        if len(recs) != expected:
            violations.append(
                f"{name}: emitted {len(recs)} attribution record(s) for {expected} kernel op(s) "
                "([D-4] M1 hard gate = 100%: every planned kernel must produce exactly one record)")
        for i, rec in enumerate(recs):
            for msg in validate_record(rec):
                violations.append(f"{name}[record {i}]: {msg}")
            if isinstance(rec, dict):
                r = rec.get("reason")
                if r is not None:
                    reasons_seen.add(r)
    # (V5) [D-4] M1 查①②: both branches must be exercised somewhere in the corpus.
    for missing in sorted(M1_REQUIRED_REASONS - reasons_seen):
        violations.append(
            f"[D-4] M1 reason-enum incomplete: {missing!r} is never emitted across the corpus "
            "(the compile-time selection attribution export must exercise both only_feasible and "
            "static_order)")
    return violations


# ---------------------------------------------------------------------------
def _on_path(name):
    for d in os.environ.get("PATH", "").split(os.pathsep):
        if d and os.path.isfile(os.path.join(d, name)):
            return True
    return False


def locate_opt(override):
    if override:
        return override if (os.path.isfile(override) or _on_path(override)) else None
    candidates = []
    env = os.environ.get("TCRV_BUILD")
    if env:
        candidates.append(os.path.join(env, "bin", "tcrv-opt"))
    candidates.append(os.path.join(REPO, "build", "bin", "tcrv-opt"))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def parse_run_command(mlir_path):
    """Extract the tcrv-opt `// RUN:` line that writes the attribution JSONL.

    Returns the shlex-tokenized argv (with %s / %t still symbolic), or raises.
    """
    with open(mlir_path) as f:
        for line in f:
            s = line.strip()
            if s.startswith("// RUN:") and "attribution-jsonl=" in s and "tcrv-opt" in s:
                return shlex.split(s[len("// RUN:"):].strip())
    raise RuntimeError(f"no tcrv-opt attribution-jsonl RUN line found in {mlir_path}")


def run_corpus_file(opt, mlir_path, tmpdir, verbose):
    """Run the pass pipeline for one corpus file; return (records, expected_kernels)."""
    with open(mlir_path) as f:
        content = f.read()
    expected_kernels = len(_KERNEL_RE.findall(content))

    argv = parse_run_command(mlir_path)
    tbase = os.path.join(tmpdir, os.path.splitext(os.path.basename(mlir_path))[0])
    jsonl_path = tbase + ".jsonl"
    cmd = []
    for tok in argv:
        if tok == "tcrv-opt" or tok.endswith("/tcrv-opt"):
            cmd.append(opt)
        else:
            cmd.append(tok.replace("%s", mlir_path).replace("%t", tbase))
    if verbose:
        print(f"  $ {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"tcrv-opt exited {proc.returncode} on {mlir_path}:\n{proc.stderr}")
    if not os.path.isfile(jsonl_path):
        raise RuntimeError(f"attribution sink produced no JSONL at {jsonl_path}")
    records = []
    with open(jsonl_path) as jf:
        for ln in jf.read().splitlines():
            ln = ln.strip()
            if ln:
                records.append(json.loads(ln))
    return records, expected_kernels


def run_real(verbose, opt_override, require_binaries):
    for rel in CORPUS:
        if not os.path.isfile(os.path.join(REPO, rel)):
            print(f"[f4-attribution] setup error: corpus file missing: {rel}")
            return 2

    opt = locate_opt(opt_override)
    if not opt:
        msg = ("[f4-attribution] tcrv-opt not built "
               "(looked under $TCRV_BUILD/bin and build/bin)")
        if require_binaries:
            print(msg + " -- RED (--require-binaries)")
            return 2
        print(msg + " -- SKIP (build-free lane; pass --require-binaries to gate)")
        return 0

    files = []
    with tempfile.TemporaryDirectory() as tmp:
        for rel in CORPUS:
            mlir_path = os.path.join(REPO, rel)
            try:
                records, expected = run_corpus_file(opt, mlir_path, tmp, verbose)
            except Exception as e:  # noqa: BLE001
                print(f"[f4-attribution] setup error: {e}")
                return 2
            files.append({"name": rel, "records": records, "expected_kernels": expected})

    violations = classify_corpus(files)

    total = sum(len(f["records"]) for f in files)
    kernels = sum(f["expected_kernels"] for f in files)
    if verbose:
        for f in files:
            reasons = [r.get("reason") for r in f["records"]]
            print(f"  {f['name']}: {len(f['records'])}/{f['expected_kernels']} records "
                  f"reasons={reasons}")

    if violations:
        print(f"[f4-attribution] RED: {len(violations)} attribution-JSONL violation(s)")
        for v in violations:
            print(f"  - {v}")
        return 1

    print(f"[f4-attribution] GREEN: {total}/{kernels} kernels attributed "
          "(shape + reason enum + [D-4] M1 只 only_feasible/static_order coverage intact)")
    return 0


# ---------------------------------------------------------------------------
def run_self_test():
    """Prove the validator DISCRIMINATES before it judges the real JSONL."""
    fails = []

    def check(label, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            fails.append(label)

    def rec(reason, feasible, chosen="k", dih="abc123", **over):
        cands = [{"feasible": True, "variant": f"v{i}"} for i in range(feasible)]
        base = {
            "candidates": cands, "chosen": chosen, "declared_instance_hash": dih,
            "kernel": "kern", "keys_evaluated": {}, "reason": reason, "ts": "0",
        }
        base.update(over)
        return base

    # ---- per-record validator ----
    check("only_feasible + 1 feasible + chosen -> valid",
          validate_record(rec("only_feasible", 1)) == [])
    check("static_order + 2 feasible + chosen -> valid",
          validate_record(rec("static_order", 2)) == [])
    check("null reason + no chosen + 0 feasible -> valid",
          validate_record(rec(None, 0, chosen=None)) == [])

    check("missing required key -> RED",
          any("missing required key" in m
              for m in validate_record({"reason": None})))
    check("bad reason token -> RED",
          any("not in" in m for m in validate_record(rec("prior_guess", 1))))
    check("only_feasible but 2 feasible -> RED (not derivable)",
          any("feasible candidate count=2" in m
              for m in validate_record(rec("only_feasible", 2))))
    check("static_order but 1 feasible -> RED (not derivable)",
          any("feasible candidate count=1" in m
              for m in validate_record(rec("static_order", 1))))
    check("null reason but chosen set -> RED",
          any("chosen=" in m for m in validate_record(rec(None, 0, chosen="k"))))
    check("non-hex declared_instance_hash -> RED",
          any("declared_instance_hash" in m
              for m in validate_record(rec("only_feasible", 1, dih="NOTHEX!"))))
    check("candidate missing 'feasible' -> RED",
          any("missing 'feasible'" in m
              for m in validate_record(rec("only_feasible", 1,
                                           candidates=[{"variant": "x"}]))))

    # ---- corpus-level ----
    good = [
        {"name": "a", "records": [rec("only_feasible", 1), rec("static_order", 2)],
         "expected_kernels": 2},
        {"name": "b", "records": [rec(None, 0, chosen=None)], "expected_kernels": 1},
    ]
    check("compliant corpus (both M1 tokens + null) -> GREEN", classify_corpus(good) == [])

    # dropped kernel: 1 record for 2 kernel ops -> RED
    drop = [{"name": "a", "records": [rec("only_feasible", 1)], "expected_kernels": 2},
            {"name": "b", "records": [rec("static_order", 2)], "expected_kernels": 1}]
    check("dropped kernel (record count != kernel count) -> RED",
          any("hard gate = 100%" in m for m in classify_corpus(drop)))

    # M1 enum incomplete: static_order never emitted -> RED
    only1 = [{"name": "a", "records": [rec("only_feasible", 1)], "expected_kernels": 1}]
    check("static_order never emitted across corpus -> RED",
          any("reason-enum incomplete" in m and "static_order" in m
              for m in classify_corpus(only1)))
    only2 = [{"name": "a", "records": [rec("static_order", 2)], "expected_kernels": 1}]
    check("only_feasible never emitted across corpus -> RED",
          any("reason-enum incomplete" in m and "only_feasible" in m
              for m in classify_corpus(only2)))

    # The committed corpus files must exist (deletion cannot green the gate).
    check("committed corpus files present",
          all(os.path.isfile(os.path.join(REPO, r)) for r in CORPUS))

    if fails:
        print(f"[f4-attribution --self-test] RED: {len(fails)} discrimination(s) failed")
        return 2
    print("[f4-attribution --self-test] GREEN: validator discriminates all cases")
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
