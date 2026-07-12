#!/usr/bin/env python3
"""check_f6_scalar_family_independence.py -- [F-6] independent-family gate (XS-M3).

Machine-checks the two [F-6] conjuncts for the X-SCALAR (scalar.fallback) family on
the COMMITTED vector-absent F-6 instance -- i.e. the N2-boundary 判据④ "向量缺席 ->
标量变体 only_feasible 真实选中". This scriptifies, as a single named CI-runnable gate,
the property that today lives ONLY as embedded assertions inside the 989-LOC
ScalarExtensionPluginTest.cpp + a multi-RUN lit file:

  (C2) only_feasible truly-selected : in the vector-absent instance the scalar family
       variant is the SOLE feasible candidate AND is the one actually CHOSEN by the
       selector -- attribution record `chosen` == the scalar variant AND
       `reason` == "only_feasible" (not a dead metadata shell that never wins).

  (C1) closure INTERSECT rvv.* = EMPTY : that same vector-absent instance evaluates
       ZERO capability keys in the `rvv.*` namespace (`keys_evaluated` INTERSECT rvv.* = EMPTY),
       AND the emitted route carries no vector machinery -- no `__riscv_` intrinsic,
       no `weft_rvv` symbol, no XOR-popcount codebook. The namespace test is a true
       dotted-namespace test, NOT a substring test: `rvv` / `rvv.zvfh` are inside,
       `rvvish` stays independent (mirrors the gtest bare-prefix guard).

This gate does NOT touch the selector, the reason enum, the Scalar legality predicate,
or any numerics. It only RE-VERIFIES an already-wired, already-passing mechanism and
gives the [F-6] independence claim a standalone, greppable home. It is the sibling of
the F-2' / opponent-facts / monolith-retire / front-door gates in falsifier-gate.yml,
which the workflow header flags F-6 independence as a "later round" of.

Two modes (sibling idiom):
  --self-test : hermetic classifier discrimination. Feeds a compliant synthetic F-6
     record (GREEN) + one record per violation class (RED): chosen-not-scalar,
     reason-not-only_feasible, rvv-key-in-closure, plus the emitted-C independence
     classifier (rvv-machinery / xor-popcount). Proves the classifier FIRES before it
     judges the real tree, and that `rvvish` (bare-prefix lookalike) stays independent.
  (default)  : drives the committed F-6 lit instance through weft-opt + weft-translate
     (auto-located under build/bin, or --opt/--translate, or $WEFT_BUILD), classifies
     the produced attribution record + emitted C. If the built binaries are absent it
     SKIPs (exit 0) so build-free lanes stay green; pass --require-binaries to make the
     absence itself RED.

Stdlib-only.
Usage:  python3 tools/lint/check_f6_scalar_family_independence.py [--self-test] [-v]
                 [--opt PATH] [--translate PATH] [--require-binaries]
Exit:   0 GREEN ; 1 RED (independence violated) ; 2 setup error.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
F6_INSTANCE = os.path.join(
    REPO,
    "test/Transforms/VariantSelection/f6-independent-scalar-family-emittable.mlir",
)

# The scalar family's canonical vector-absent variant. The gate asserts THIS is the
# only_feasible chosen variant; kept as a prefix so a family-slice rename that keeps
# the scalar_fallback lineage still classifies as scalar-owned.
SCALAR_VARIANT_PREFIX = "scalar_fallback"

# True dotted-namespace test for the rvv.* closure conjunct: `rvv` (exact) and any
# `rvv.<sub>` are INSIDE the namespace; `rvvish` (bare-prefix lookalike) is OUTSIDE.
RE_RVV_NAMESPACE = re.compile(r"^rvv(\.|$)")

# Emission-independence bans: vector intrinsics, the rvv symbol prefix, and the
# XOR-popcount codebook the [J-3] trap forbids for low-bit scalar math.
EMIT_BANS = [
    ("rvv-machinery-in-emit(__riscv_)", re.compile(r"__riscv_")),
    ("rvv-machinery-in-emit(weft_rvv)", re.compile(r"weft_rvv")),
    ("xor-popcount-codebook-in-emit", re.compile(r"popcount", re.IGNORECASE)),
]


# ---------------------------------------------------------------------------
# Pure classifier core (fed synthetic inputs by --self-test; real output at runtime).
# ---------------------------------------------------------------------------
def classify_attribution(record):
    """A compile-time selection-attribution record (dict) -> list of violation strings.

    Empty list == GREEN (this vector-absent instance truly selects the scalar family
    as the only_feasible candidate and its capability closure avoids the rvv.* namespace).
    """
    violations = []
    chosen = record.get("chosen")
    reason = record.get("reason")
    keys = record.get("keys_evaluated", {}) or {}

    # (C2) only_feasible truly-selected.
    if not isinstance(chosen, str) or not chosen.startswith(SCALAR_VARIANT_PREFIX):
        violations.append(f"chosen-not-scalar: chosen={chosen!r}")
    if reason != "only_feasible":
        violations.append(f"reason-not-only_feasible: reason={reason!r}")

    # (C1) closure INTERSECT rvv.* = EMPTY, at the instance/selection level.
    rvv_keys = sorted(k for k in keys if RE_RVV_NAMESPACE.match(k))
    if rvv_keys:
        violations.append(f"rvv-key-in-closure: {rvv_keys}")

    return violations


def classify_emitted_c(emitted_c):
    """Emitted scalar C text -> list of violation strings (emission-level independence).

    Empty list == GREEN (the selected scalar body lowers to real pure-scalar C with no
    vector machinery and no XOR-popcount codebook)."""
    violations = []
    for label, pattern in EMIT_BANS:
        if pattern.search(emitted_c):
            violations.append(label)
    return violations


# ---------------------------------------------------------------------------
# Binary location + the committed-instance drivers.
# ---------------------------------------------------------------------------
def locate_binary(name, override):
    if override:
        return override if os.path.isfile(override) else None
    candidates = []
    env = os.environ.get("WEFT_BUILD")
    if env:
        candidates.append(os.path.join(env, "bin", name))
    candidates.append(os.path.join(REPO, "build", "bin", name))
    for c in candidates:
        if os.path.isfile(c):
            return c
    return None


def run_selection_attribution(opt):
    """RUN 1 of the F-6 lit: emit the canonical selection-attribution JSONL for the
    committed vector-absent instance. Returns the parsed record dict."""
    with tempfile.TemporaryDirectory() as tmp:
        jsonl = os.path.join(tmp, "f6.jsonl")
        cmd = [
            opt, F6_INSTANCE,
            "--weft-check-capability-requires",
            "--weft-materialize-plugin-variants",
            "--weft-verify-plugin-variant-legality",
            f"--weft-select-variants=attribution-jsonl={jsonl} "
            "attribution-jsonl-no-timestamp",
            "-o", os.devnull,
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        with open(jsonl) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    if len(lines) != 1:
        raise RuntimeError(
            f"expected exactly one attribution record, got {len(lines)}")
    return json.loads(lines[0])


def run_emit_route(opt, translate):
    """RUN 2 of the F-6 lit: lower the selected scalar body to C. Returns the C text."""
    select = subprocess.run(
        [opt, F6_INSTANCE,
         "--weft-check-capability-requires",
         "--weft-materialize-plugin-variants",
         "--weft-verify-plugin-variant-legality",
         "--weft-select-variants"],
        check=True, capture_output=True, text=True)
    emit = subprocess.run(
        [translate, "--weft-scalar-emitc-to-cpp"],
        input=select.stdout, check=True, capture_output=True, text=True)
    return emit.stdout


# ---------------------------------------------------------------------------
def run_real(verbose, opt_override, translate_override, require_binaries):
    if not os.path.isfile(F6_INSTANCE):
        print(f"[f6-independence] setup error: committed instance missing: {F6_INSTANCE}")
        return 2

    opt = locate_binary("weft-opt", opt_override)
    translate = locate_binary("weft-translate", translate_override)
    if not opt or not translate:
        msg = ("[f6-independence] weft-opt/weft-translate not built "
               "(looked under $WEFT_BUILD/bin and build/bin)")
        if require_binaries:
            print(msg + " -- RED (--require-binaries)")
            return 2
        print(msg + " -- SKIP (build-free lane; pass --require-binaries to gate)")
        return 0

    try:
        record = run_selection_attribution(opt)
        emitted = run_emit_route(opt, translate)
    except subprocess.CalledProcessError as e:
        print("[f6-independence] setup error: compiler invocation failed")
        print(e.stderr or e.stdout or str(e))
        return 2
    except Exception as e:  # noqa: BLE001 -- any harness failure is a setup error
        print(f"[f6-independence] setup error: {e}")
        return 2

    violations = classify_attribution(record) + classify_emitted_c(emitted)

    if verbose:
        print(f"  chosen        = {record.get('chosen')!r}")
        print(f"  reason        = {record.get('reason')!r}")
        print(f"  keys_evaluated= {record.get('keys_evaluated')}")
        print(f"  emitted C len = {len(emitted)} bytes")

    if violations:
        print(f"[f6-independence] RED: {len(violations)} independence violation(s)")
        for v in violations:
            print(f"  - {v}")
        return 1

    print("[f6-independence] GREEN: vector-absent instance truly selects the scalar "
          "family (only_feasible), closure INTERSECT rvv.* = EMPTY, emit is pure scalar")
    return 0


# ---------------------------------------------------------------------------
def run_self_test():
    """Prove the classifier DISCRIMINATES before it judges the committed tree:
    a compliant F-6 record -> GREEN, each violation class -> RED, and the
    bare-prefix lookalike `rvvish` stays independent (namespace not substring)."""
    fails = []

    def check(label, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            fails.append(label)

    # --- compliant record -> GREEN (no violations) ---
    compliant = {
        "chosen": "scalar_fallback_first_slice",
        "reason": "only_feasible",
        "keys_evaluated": {"scalar.fallback": "available"},
    }
    check("compliant vector-absent record -> GREEN",
          classify_attribution(compliant) == [])

    # --- (C2) chosen is not the scalar family -> RED ---
    not_scalar = dict(compliant, chosen="rvv_gemm_m1")
    v = classify_attribution(not_scalar)
    check("chosen-not-scalar -> RED",
          any(x.startswith("chosen-not-scalar") for x in v))

    # --- (C2) reason is not only_feasible (e.g. static_order among >=2) -> RED ---
    static_order = dict(compliant, reason="static_order")
    v = classify_attribution(static_order)
    check("reason-not-only_feasible -> RED",
          any(x.startswith("reason-not-only_feasible") for x in v))

    # --- (C1) an rvv.* key entered the closure -> RED (direct) ---
    rvv_key = dict(compliant,
                   keys_evaluated={"scalar.fallback": "available",
                                   "rvv.zvfh": "unavailable"})
    v = classify_attribution(rvv_key)
    check("rvv.zvfh key in closure -> RED",
          any(x.startswith("rvv-key-in-closure") for x in v))

    # --- (C1) bare `rvv` (exact) is INSIDE the namespace -> RED ---
    rvv_exact = dict(compliant,
                     keys_evaluated={"scalar.fallback": "available",
                                     "rvv": "unavailable"})
    check("bare 'rvv' (exact) in closure -> RED",
          any(x.startswith("rvv-key-in-closure")
              for x in classify_attribution(rvv_exact)))

    # --- (C1) namespace guard: `rvvish` is a bare-prefix lookalike, NOT rvv.* -> GREEN ---
    rvvish = dict(compliant,
                  keys_evaluated={"scalar.fallback": "available",
                                  "rvvish": "unavailable"})
    check("'rvvish' lookalike stays independent (namespace not substring) -> GREEN",
          classify_attribution(rvvish) == [])

    # --- emission-independence classifier ---
    check("pure scalar C -> GREEN",
          classify_emitted_c("v3[v4] = v5 & 15; v6 = v7 >> 4;") == [])
    check("__riscv_ intrinsic in emit -> RED",
          any(x.startswith("rvv-machinery-in-emit(__riscv_)")
              for x in classify_emitted_c("__riscv_vle32_v_f32m1(...)")))
    check("weft_rvv symbol in emit -> RED",
          any(x.startswith("rvv-machinery-in-emit(weft_rvv)")
              for x in classify_emitted_c("weft_rvv_block_dot(...)")))
    check("XOR-popcount codebook in emit -> RED",
          classify_emitted_c("acc += __builtin_popcount(x ^ y);") ==
          ["xor-popcount-codebook-in-emit"])

    if fails:
        print(f"[f6-independence --self-test] RED: {len(fails)} discrimination(s) failed")
        return 2
    print("[f6-independence --self-test] GREEN: classifier discriminates all cases")
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
        translate_override=opt_arg("--translate"),
        require_binaries="--require-binaries" in argv,
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
