#!/usr/bin/env python3
"""check_zero_core_family_branch.py -- [F-1] zero-core-branch gate (M1 evidence line).

Machine-checks the I3 invariant "零 family-name 分支": the core / common dispatch path
reaches plugins ONLY through the capability registry + plugin interface, NEVER via an
`if RVV` / `if IME` / `if Sophgo` family-name branch. This scriptifies the [F-1] falsifier
that canon ([F-1], .trellis/spec/canon/能力模型与插件协议.md §二) specifies as a CI gate but which had no manifest / checker
until now (the invariant itself was already green -- this gate LOCKS it).

Mechanism (faithful to the [F-1] judgment protocol at .trellis/spec/canon/能力模型与插件协议.md §二):
  1. Enumerate the CORE control-flow file scope from schema/family-regex.v1.json
     (recursive/depth1/single roots, minus each family's own directory).
  2. For each core line, strip comments (// , /* */ , TableGen // , leading-* doc lines),
     then test whether it is a family-name-keyed control-flow DECISION:
        - a family dispatch-key STRING LITERAL ("rvv" / "rvv.zvfh" / "scalar.fallback")
          used as the operand of a branch operator
          (StringSwitch / .Case / .starts_with / .consume_front / .contains / .equals /
           == / !=), OR
        - a bareword family ENUM in a switch/case/comparison (case RVV: / == Family::IME).
  3. A branch-construct hit not covered by the manifest allow_list = RED (a real I3
     violation). Declared false-positive classes -- comments, mnemonics, dialect-type refs,
     registration tables, identity-by-origin comparisons, and a family key passed as a data
     argument to a family-agnostic function -- are NOT branches and are NOT flagged.

This gate does NOT touch the compiler, the selector, or any numerics -- it greps source and
is build-free (runs on ubuntu-latest with no compilation). At HEAD 1bb06375 it is GREEN:
zero family-keyed branch across the core scope.

Two modes (sibling idiom of the falsifier-gate.yml gates):
  --self-test : hermetic classifier discrimination on synthetic core lines. Proves the
     branch detector FIRES on each RED construct (if == "rvv" / StringSwitch.Case("ime") /
     starts_with("scalar") / case Family::RVV) and STAYS GREEN on each declared
     false-positive class (comment / pass mnemonic / dependentDialect ref / registration
     table / identity-by-origin / namespace-arg). No repo scan needed.
  (default)  : scans the committed core scope, classifies every family-key hit, prints any
     RED branch with file:line, and returns the count.

Stdlib-only.
Usage:  python3 tools/gates/check_zero_core_family_branch.py [--self-test] [-v]
Exit:   0 GREEN (zero family branch) ; 1 RED (real family branch found) ; 2 setup error.
"""
import fnmatch
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(REPO, "schema/family-regex.v1.json")


# ---------------------------------------------------------------------------
# Branch-operator + family-key detection (pure; fed synthetic lines by --self-test).
# ---------------------------------------------------------------------------
# Branch operators that make a family key a control-flow DISCRIMINATOR.
BRANCH_OP_RE = re.compile(
    r"(StringSwitch|\.Cases?\b|\.CaseLower\b|\.starts_with\b|\.startswith\b"
    r"|\.consume_front\b|\.ends_with\b|\.contains\b|\.equals(_insensitive)?\b"
    r"|==|!=|\bcase\b)")


def build_family_key_re(keys):
    """String-literal family key: an opening quote immediately followed by a family
    token that is either exact ("rvv") or a dotted namespace ("rvv.zvfh")."""
    alt = "|".join(re.escape(k) for k in keys)
    return re.compile(r'"(' + alt + r')(\.[A-Za-z0-9_]+)*"')


def build_family_enum_re(keys):
    """Bareword family enum in a switch/comparison, e.g. `case RVV:` / `== Family::IME` /
    `!= weft::Scalar`. Capitalised family token as a whole word (optionally namespace-
    qualified). Kept separate from the string-literal detector."""
    caps = sorted({k[:1].upper() + k[1:] for k in keys} | {k.upper() for k in keys})
    alt = "|".join(re.escape(c) for c in caps)
    return re.compile(r"(\bcase\b|==|!=)\s*(?:[A-Za-z_][A-Za-z0-9_]*::)*(" + alt + r")\b")


def strip_comment(line):
    """Remove // line comments, /* */ inline, and leading TableGen/doc `*` lines.
    Conservative: a `//` or `/*` inside a string literal is rare in this codebase's core
    and would only make the detector MORE strict (never hide a real branch)."""
    s = line
    # whole-line doc/comment forms
    stripped = s.lstrip()
    if stripped.startswith("//") or stripped.startswith("*") or stripped.startswith("/*"):
        return ""
    # trailing // comment
    idx = s.find("//")
    if idx != -1:
        s = s[:idx]
    # inline /* ... */
    s = re.sub(r"/\*.*?\*/", "", s)
    return s


def classify_line(code_line, family_key_re, family_enum_re):
    """A comment-STRIPPED code line -> True iff it is a family-name-keyed branch.

    RED iff a branch operator co-occurs with a family-key string literal OR a bareword
    family enum appears in a switch/comparison. Empty/None => not a branch.
    """
    if not code_line or not code_line.strip():
        return False
    has_branch_op = bool(BRANCH_OP_RE.search(code_line))
    if has_branch_op and family_key_re.search(code_line):
        return True
    if family_enum_re.search(code_line):
        return True
    return False


# ---------------------------------------------------------------------------
def load_manifest():
    with open(MANIFEST) as f:
        data = json.load(f)
    keys = []
    for fam in data.get("families", []):
        keys.extend(fam.get("keys", []))
    if not keys:
        raise RuntimeError("manifest has no family keys")
    scope = data.get("core_scope", {})
    allow = data.get("allow_list", []) or []
    return keys, scope, allow


def enumerate_core_files(scope):
    exts = tuple(scope.get("file_extensions", [".cpp", ".h", ".td"]))
    exclude = set(scope.get("exclude_subdir_names", []))
    files = []

    def add_if_ext(path):
        if path.endswith(exts) and os.path.isfile(path):
            files.append(path)

    for root in scope.get("recursive_roots", []):
        base = os.path.join(REPO, root)
        for dirpath, dirnames, filenames in os.walk(base):
            # prune family-owned subdirs
            dirnames[:] = [d for d in dirnames if d not in exclude]
            for fn in filenames:
                add_if_ext(os.path.join(dirpath, fn))

    for root in scope.get("depth1_roots", []):
        base = os.path.join(REPO, root)
        if os.path.isdir(base):
            for fn in sorted(os.listdir(base)):
                add_if_ext(os.path.join(base, fn))

    for f in scope.get("single_files", []):
        add_if_ext(os.path.join(REPO, f))

    return sorted(set(files))


def allow_listed(relpath, line, allow):
    for entry in allow:
        glob = entry.get("file_glob", "")
        rx = entry.get("regex", "")
        if glob and fnmatch.fnmatch(relpath, glob) and rx and re.search(rx, line):
            return True
    return False


def run_real(verbose):
    if not os.path.isfile(MANIFEST):
        print(f"[f1-zero-branch] setup error: manifest missing: {MANIFEST}")
        return 2
    try:
        keys, scope, allow = load_manifest()
    except Exception as e:  # noqa: BLE001
        print(f"[f1-zero-branch] setup error: {e}")
        return 2

    family_key_re = build_family_key_re(keys)
    family_enum_re = build_family_enum_re(keys)
    files = enumerate_core_files(scope)
    if not files:
        print("[f1-zero-branch] setup error: core scope enumerated ZERO files "
              "(scope roots wrong?)")
        return 2

    reds = []
    for path in files:
        rel = os.path.relpath(path, REPO)
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                for lineno, raw in enumerate(f, 1):
                    code = strip_comment(raw)
                    if classify_line(code, family_key_re, family_enum_re):
                        if allow_listed(rel, raw, allow):
                            continue
                        reds.append((rel, lineno, raw.rstrip()))
        except OSError as e:
            print(f"[f1-zero-branch] setup error: cannot read {rel}: {e}")
            return 2

    if verbose:
        print(f"  scanned {len(files)} core files; family keys = {keys}")

    if reds:
        print(f"[f1-zero-branch] RED: {len(reds)} family-name-keyed branch(es) in core "
              "(I3 zero-branch violated)")
        for rel, lineno, text in reds:
            print(f"  - {rel}:{lineno}: {text.strip()}")
        print("  If a hit is a declared false-positive (comment / mnemonic / registration "
              "/ identity-by-origin), register it in schema/family-regex.v1.json allow_list "
              "(file_glob, regex) per the judgment protocol; do NOT widen the family regex.")
        return 1

    print(f"[f1-zero-branch] GREEN: zero family-name-keyed branch across {len(files)} "
          "core files (I3 holds)")
    return 0


# ---------------------------------------------------------------------------
def run_self_test():
    """Prove the branch detector FIRES on each RED construct and STAYS GREEN on each
    declared false-positive class, before it judges the committed tree."""
    fails = []
    keys = ["rvv", "ime", "scalar", "offload", "sophgo", "ame"]
    fkre = build_family_key_re(keys)
    fere = build_family_enum_re(keys)

    def is_branch(raw):
        return classify_line(strip_comment(raw), fkre, fere)

    def check(label, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            fails.append(label)

    # --- RED: real family-name-keyed branches ---
    check('if (route.origin == "rvv") -> RED',
          is_branch('  if (route.origin == "rvv") return lowerRVV();'))
    check('StringSwitch<int>(id).Case("ime", ...) -> RED',
          is_branch('  int k = StringSwitch<int>(id).Case("ime", 1).Default(0);'))
    check('name.starts_with("scalar") -> RED',
          is_branch('  if (name.starts_with("scalar")) useScalar();'))
    check('id == "rvv.zvfh" (dotted key) -> RED',
          is_branch('  if (id == "rvv.zvfh") { doit(); }'))
    check('cap.contains("offload") -> RED',
          is_branch('  if (caps.contains("offload")) offloadIt();'))
    check('bareword enum: case Family::RVV: -> RED',
          is_branch('    case Family::RVV:'))
    check('bareword enum: fam == IME -> RED',
          is_branch('  if (fam == IME) return true;'))

    # --- GREEN: declared false-positive classes ---
    check('comment "// does not branch on RVV, IME, scalar" -> GREEN',
          not is_branch('    // does not branch on RVV, scalar fallback, IME, offload,'))
    check('leading-* doc line mentioning rvv -> GREEN',
          not is_branch('   * mirrors the --weft-rvv-lower-to-emitc pass'))
    check('pass mnemonic Pass<"weft-rvv-lower-to-emitc"> -> GREEN',
          not is_branch('  : Pass<"weft-rvv-lower-to-emitc", "::mlir::ModuleOp"> {'))
    check('dependentDialect "::weft::rvv::WEFTRVVDialect" -> GREEN',
          not is_branch('    "::weft::rvv::WEFTRVVDialect"'))
    check('registration table entry rvv::registerRVVBackendEmitter, -> GREEN',
          not is_branch('    rvv::registerRVVBackendEmitter,'))
    check('registration entry ::...::ime::registerIMEBackendEmitter, -> GREEN',
          not is_branch('    ::weft::plugin::ime::registerIMEBackendEmitter,'))
    check('identity-by-origin (no family literal) -> GREEN',
          not is_branch('  if (candidate.origin == selectedRoute.originPlugin) keep();'))
    check('namespace-arg impliedClosureAvoidsNamespace(seed, "rvv") -> GREEN',
          not is_branch('  return impliedClosureAvoidsNamespace(seed, "rvv");'))
    check('generic compare id == namespacePrefix (variable, not literal) -> GREEN',
          not is_branch('    if (id == namespacePrefix || id.starts_with(namespacePrefix))'))
    check('role mnemonic return "rhs-scalar-value"; -> GREEN',
          not is_branch('    return "rhs-scalar-value";'))
    check('dialect doc "weft.rvv, weft.ime, weft.offload, scalar" in comment -> GREEN',
          not is_branch('    // by extension dialects such as weft.rvv, weft.ime, scalar'))

    # --- the committed manifest must load + enumerate a non-empty core scope ---
    try:
        mkeys, scope, _ = load_manifest()
        core = enumerate_core_files(scope)
        check(f"committed manifest enumerates a non-empty core scope ({len(core)} files)",
              len(core) > 0)
        check("manifest family keys non-empty", bool(mkeys))
    except Exception as e:  # noqa: BLE001
        check(f"committed manifest loads ({e})", False)

    if fails:
        print(f"[f1-zero-branch --self-test] RED: {len(fails)} discrimination(s) failed")
        return 2
    print("[f1-zero-branch --self-test] GREEN: branch detector discriminates all cases")
    return 0


def main(argv):
    verbose = "-v" in argv or "--verbose" in argv
    if "--self-test" in argv:
        return run_self_test()
    return run_real(verbose)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
