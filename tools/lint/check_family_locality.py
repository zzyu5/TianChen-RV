#!/usr/bin/env python3
"""check_family_locality.py -- [F-3] family change-containment gate (M1 evidence line).

Machine-checks the [F-3] falsifier "变更收容" (canon 科研目标总纲v2:107 ·
.trellis/spec/plugin-protocol/locality-contract.md:76). Per the 2026-07-12 user ruling
「MLIR 分层为主 + family 清单机检」, containment is asserted by a per-family MANIFEST
(schema/family-manifest.v1.json), NOT by a cross-root physical directory move: the repo
keeps the idiomatic MLIR layered layout (lib/{Dialect,Conversion,Plugin,Target}/<Fam>/ +
include/ mirrors), and each family declares its OWNED source territory in the manifest.

The gate asserts, over SOURCE CODE only (tracked *.cpp/*.h/*.td under lib/ and
include/Weft/ -- where the '(no core-file edits)' claim bites and the per-family
directories are clean):

  * [default / whole-repo consistency]
      - every family-territory source file (under a family root subdir that is not a
        declared core_subdir) is claimed by EXACTLY ONE family manifest -> zero-match is an
        ORPHAN (or an under-declared range), a family-shaped subdir with no family entry is
        an UNKNOWN-FAMILY-DIR, two-match is CROSS-FAMILY POLLUTION;
      - NO family source_range matches a file in the authoritative family-AGNOSTIC core
        scope (schema/family-regex.v1.json core_scope) -> a range widened into core is RED
        (the anti-widen '偷偷扩范围骗绿' tripwire);
      - shrink-only ratchet: $meta.baseline_range_count == total declared source_ranges
        (any range add/remove forces a visible baseline bump in the same diff).

  * [CI / base..head diff]  a family-integration PR's diff touches ONLY that one family's
      source territory. The shared allowances -- schema/** (TABLE ROWS '+ 表行'), docs/** /
      *.md / .trellis/** (DOCS '+ 文档'), test/** ([P-2] onboarding-kit tests), CMakeLists /
      tools/** (build+tooling) -- are freely touchable. RED iff the diff touches >=2
      distinct families' source territory (cross-family) OR, while it IS a family PR, also
      touches a core-source code file (a core edit).

This gate touches NO kernel / NO selection logic / NO numerics -- it greps paths and is
build-free (pure filesystem walk + manifest read). Maintained SAME-ADDRESS as the [F-EMIT]
emit-bypass whitelist and the [F-1] family-regex manifest (schema/ + tools/lint/ +
falsifier-gate.yml), same shrink-only ratchet idiom.

Usage:  python3 tools/lint/check_family_locality.py [--self-test] [-v]
        python3 tools/lint/check_family_locality.py --base <sha> --head <sha> [-v]
Exit:   0 GREEN ; 1 RED (containment / orphan / pollution / anti-widen / ratchet) ; 2 setup.
"""
import fnmatch
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MANIFEST = os.path.join(REPO, "schema/family-manifest.v1.json")
FAMILY_REGEX = os.path.join(REPO, "schema/family-regex.v1.json")

SOURCE_EXTS = (".cpp", ".h", ".td")
# The four family roots (+ their include mirrors) under which per-family subdirs live.
FAMILY_ROOTS = [
    "lib/Dialect", "lib/Conversion", "lib/Plugin", "lib/Target",
    "include/Weft/Dialect", "include/Weft/Conversion",
    "include/Weft/Plugin", "include/Weft/Target",
]


# ---------------------------------------------------------------------------
# Pure path helpers (fed synthetic inputs by --self-test).
# ---------------------------------------------------------------------------
def matches_range(rel, ranges):
    """rel (repo-relative POSIX path) is in `ranges` iff it equals / is under a directory
    prefix, or matches a glob range (a range containing * ? [). source_ranges are plain
    directory prefixes; test_ranges may carry name-prefix globs (informational only)."""
    for r in ranges:
        r = r.rstrip("/")
        if any(ch in r for ch in "*?["):
            if fnmatch.fnmatch(rel, r):
                return True
        elif rel == r or rel.startswith(r + "/"):
            return True
    return False


def is_source_code(rel):
    """A tracked lib/ or include/Weft/ code file (the enforced containment scope)."""
    if not rel.endswith(SOURCE_EXTS):
        return False
    return rel.startswith("lib/") or rel.startswith("include/Weft/")


def matched_families(rel, families):
    return [f["family"] for f in families if matches_range(rel, f.get("source_ranges", []))]


def root_subdir(rel):
    """For a file under a FAMILY_ROOT, return the path component right after the root
    (a family/core subdir name, or the FILE name itself when the file sits at depth-1 of
    the root, e.g. lib/Plugin/ExtensionPlugin.cpp -> 'ExtensionPlugin.cpp'). None if not
    under any family root."""
    for root in FAMILY_ROOTS:
        if rel.startswith(root + "/"):
            return rel[len(root) + 1:].split("/", 1)[0]
    return None


# ---------------------------------------------------------------------------
# Pure evaluators.
# ---------------------------------------------------------------------------
def validate_manifest_shape(doc):
    errors = []
    meta = doc.get("$meta") or {}
    if not isinstance(meta.get("baseline_range_count"), int):
        errors.append("manifest: $meta.baseline_range_count missing or not an int")
    if meta.get("shrink_only") is not True:
        errors.append("manifest: $meta.shrink_only must be true")
    core_subdirs = ((meta.get("core_subdirs") or {}).get("names"))
    if not isinstance(core_subdirs, list) or not core_subdirs:
        errors.append("manifest: $meta.core_subdirs.names missing or empty")
        core_subdirs = []
    families = doc.get("families")
    if not isinstance(families, list) or not families:
        errors.append("manifest: 'families' missing or empty")
        return errors, [], set(core_subdirs), meta.get("baseline_range_count")
    seen = set()
    for f in families:
        name = f.get("family")
        if not name:
            errors.append("manifest: a family entry lacks 'family'")
            continue
        if name in seen:
            errors.append(f"manifest: duplicate family '{name}'")
        seen.add(name)
        rngs = f.get("source_ranges")
        if not isinstance(rngs, list) or not rngs:
            errors.append(f"manifest[{name}]: 'source_ranges' missing or empty")
    return errors, families, set(core_subdirs), meta.get("baseline_range_count")


def evaluate_default(family_root_files, core_files, families, core_subdirs, baseline):
    """Whole-repo consistency verdict (pure).
      family_root_files: [rel] every *.cpp/*.h/*.td under a FAMILY_ROOT.
      core_files:        [rel] authoritative family-agnostic core scope (family-regex).
      families:          manifest families list.
      core_subdirs:      set of non-family subdir names under the roots (skip in coverage).
      baseline:          $meta.baseline_range_count.
    Returns (ok, errors)."""
    errors = []
    fam_names = {f["family"] for f in families}

    # --- coverage / orphan / cross-family pollution over family-root source files ---
    for rel in sorted(family_root_files):
        hits = matched_families(rel, families)
        if len(hits) >= 2:
            errors.append(
                f"CROSS-FAMILY-POLLUTION: '{rel}' is claimed by >1 family manifest "
                f"({', '.join(sorted(hits))}). Family territories must be disjoint.")
            continue
        if len(hits) == 1:
            continue  # covered by exactly one family -> OK
        # zero-match: is it a legitimate core subdir/file, an under-declared family file,
        # or an unmanifested (unknown) family-shaped directory?
        sub = root_subdir(rel)
        if sub is None:
            continue
        if sub.endswith(SOURCE_EXTS):
            continue  # a depth-1 core file directly under the root (e.g. ExtensionPlugin.cpp)
        if sub in core_subdirs:
            continue  # declared non-family core subdir (Exec / EmitC / Builtin / ...)
        if sub in fam_names:
            errors.append(
                f"ORPHAN: '{rel}' lives under known family dir '{sub}' but no "
                f"'{sub}' source_range matches it -- the manifest UNDER-DECLARES its "
                f"territory. Widen the range to cover it (visible baseline bump).")
        else:
            errors.append(
                f"UNKNOWN-FAMILY-DIR: '{rel}' lives under family-shaped subdir '{sub}' "
                f"which is neither a declared family nor a declared core_subdir. Add a "
                f"family-manifest entry for '{sub}', or declare it a core_subdir.")

    # --- anti-widen: no family range may claim an authoritative core-scope file ---
    core_set = set(core_files)
    for f in families:
        name = f["family"]
        for rel in sorted(core_set):
            if matches_range(rel, f.get("source_ranges", [])):
                errors.append(
                    f"ANTI-WIDEN: family '{name}' source_range matches CORE file '{rel}' "
                    f"(family-agnostic core scope per schema/family-regex.v1.json). A "
                    f"family may never claim core source (偷偷扩范围骗绿).")

    # --- shrink-only ratchet: baseline == total declared source_ranges ---
    total = sum(len(f.get("source_ranges", [])) for f in families)
    if isinstance(baseline, int) and baseline != total:
        errors.append(
            f"RATCHET: $meta.baseline_range_count={baseline} != total source_ranges="
            f"{total}. Any range add/remove needs a visible baseline_range_count bump "
            f"(shrink-only tamper-evident tripwire).")

    return (len(errors) == 0), errors


def evaluate_diff(changed_files, families):
    """CI verdict for a base..head diff (pure).
      changed_files: [rel] paths touched by the PR.
      families:      manifest families list.
    A family-integration PR must stay contained to ONE family's source territory; shared
    allowances (non-source-code: schema / docs / tests / build) are freely touchable.
    Returns (ok, errors)."""
    errors = []
    touched = {}          # family -> [files]
    core_src = []         # lib/include code files claimed by no family
    for rel in changed_files:
        if not is_source_code(rel):
            continue      # test / schema / docs / cmake / tooling = shared allowance
        hits = matched_families(rel, families)
        if hits:
            for name in hits:
                touched.setdefault(name, []).append(rel)
        else:
            core_src.append(rel)

    if not touched:
        return True, []   # not a family-integration PR -> [F-3] does not constrain it

    if len(touched) >= 2:
        parts = "; ".join(f"{k}: {', '.join(sorted(v))}" for k, v in sorted(touched.items()))
        errors.append(
            f"CROSS-FAMILY-PR: the diff touches >1 family's source territory ({parts}). A "
            f"family-integration PR must be contained to ONE family.")
    if core_src:
        fam = ", ".join(sorted(touched))
        errors.append(
            f"CORE-EDIT: this is a family-integration PR (touches {fam}) but ALSO edits "
            f"core-source file(s) not claimed by any family: {', '.join(sorted(core_src))}. "
            f"[F-3] containment forbids core-file edits in a family PR (route via the "
            f"capability registry + plugin interface, tables, or docs).")
    return (len(errors) == 0), errors


# ---------------------------------------------------------------------------
# Real-tree collectors.
# ---------------------------------------------------------------------------
def enumerate_family_root_files():
    files = []
    for root in FAMILY_ROOTS:
        base = os.path.join(REPO, root)
        for dirpath, _dirnames, filenames in os.walk(base):
            for fn in filenames:
                if fn.endswith(SOURCE_EXTS):
                    files.append(os.path.relpath(os.path.join(dirpath, fn), REPO))
    return sorted(set(files))


def enumerate_core_files():
    """Authoritative family-agnostic core scope, replicated from schema/family-regex.v1.json
    core_scope (recursive/depth1/single roots minus each family's own subdir)."""
    with open(FAMILY_REGEX, encoding="utf-8") as f:
        scope = json.load(f).get("core_scope", {})
    exts = tuple(scope.get("file_extensions", list(SOURCE_EXTS)))
    exclude = set(scope.get("exclude_subdir_names", []))
    files = []

    def add(path):
        if path.endswith(exts) and os.path.isfile(path):
            files.append(os.path.relpath(path, REPO))

    for root in scope.get("recursive_roots", []):
        for dirpath, dirnames, filenames in os.walk(os.path.join(REPO, root)):
            dirnames[:] = [d for d in dirnames if d not in exclude]
            for fn in filenames:
                add(os.path.join(dirpath, fn))
    for root in scope.get("depth1_roots", []):
        base = os.path.join(REPO, root)
        if os.path.isdir(base):
            for fn in sorted(os.listdir(base)):
                add(os.path.join(base, fn))
    for rel in scope.get("single_files", []):
        add(os.path.join(REPO, rel))
    return sorted(set(files))


def git_changed_files(base, head):
    out = subprocess.check_output(
        ["git", "-C", REPO, "diff", "--name-only", f"{base}..{head}"],
        text=True)
    return [ln.strip() for ln in out.splitlines() if ln.strip()]


# ---------------------------------------------------------------------------
# Runners.
# ---------------------------------------------------------------------------
def run_default(verbose):
    for p in (MANIFEST, FAMILY_REGEX):
        if not os.path.isfile(p):
            print(f"[f3-family-locality] setup error: missing {p}", file=sys.stderr)
            return 2
    try:
        doc = json.load(open(MANIFEST, encoding="utf-8"))
    except Exception as ex:  # noqa: BLE001
        print(f"[f3-family-locality] setup error: {ex}", file=sys.stderr)
        return 2

    shape_errs, families, core_subdirs, baseline = validate_manifest_shape(doc)
    if shape_errs:
        print("[f3-family-locality] RED (manifest shape):", file=sys.stderr)
        for e in shape_errs:
            print(f"  ::error:: {e}", file=sys.stderr)
        return 1

    family_root_files = enumerate_family_root_files()
    core_files = enumerate_core_files()
    if not family_root_files or not core_files:
        print("[f3-family-locality] setup error: enumerated ZERO family-root or core files "
              "(roots wrong?)", file=sys.stderr)
        return 2

    ok, errors = evaluate_default(family_root_files, core_files, families,
                                  core_subdirs, baseline)

    if verbose or not ok:
        print(f"[f3-family-locality] families={len(families)} "
              f"source_ranges={sum(len(f.get('source_ranges', [])) for f in families)} "
              f"(baseline {baseline}); scanned {len(family_root_files)} family-root code "
              f"files, {len(core_files)} core-scope files")
        for f in families:
            n = sum(1 for rel in family_root_files if matches_range(rel, f["source_ranges"]))
            print(f"    {f['family']:14s} {len(f['source_ranges'])} ranges -> {n} files")

    if ok:
        print("[f3-family-locality] GREEN: every family-territory source file is claimed by "
              "exactly one manifest; no orphan / no cross-family pollution / no range in "
              "core; ratchet holds.")
        return 0
    print("[f3-family-locality] RED:", file=sys.stderr)
    for e in errors:
        print(f"  ::error:: {e}", file=sys.stderr)
    return 1


def run_ci(base, head, verbose):
    if not os.path.isfile(MANIFEST):
        print(f"[f3-family-locality] setup error: missing {MANIFEST}", file=sys.stderr)
        return 2
    doc = json.load(open(MANIFEST, encoding="utf-8"))
    shape_errs, families, _cs, _b = validate_manifest_shape(doc)
    if shape_errs:
        print("[f3-family-locality] RED (manifest shape):", file=sys.stderr)
        for e in shape_errs:
            print(f"  ::error:: {e}", file=sys.stderr)
        return 1
    try:
        changed = git_changed_files(base, head)
    except Exception as ex:  # noqa: BLE001
        print(f"[f3-family-locality] setup error: git diff {base}..{head}: {ex}",
              file=sys.stderr)
        return 2

    ok, errors = evaluate_diff(changed, families)
    if verbose or not ok:
        srcs = [c for c in changed if is_source_code(c)]
        print(f"[f3-family-locality] CI diff {base}..{head}: {len(changed)} files "
              f"({len(srcs)} lib/include source)")
    if ok:
        print("[f3-family-locality] GREEN: diff stays contained to a single family's "
              "source territory (+ table rows / docs / tests).")
        return 0
    print("[f3-family-locality] RED:", file=sys.stderr)
    for e in errors:
        print(f"  ::error:: {e}", file=sys.stderr)
    return 1


# ---------------------------------------------------------------------------
def run_self_test():
    """Prove BOTH evaluators DISCRIMINATE (matching -> GREEN, each violation class -> RED)
    on synthetic inputs before judging the committed tree."""
    fails = []

    def check(label, cond):
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
        if not cond:
            fails.append(label)

    syn_families = [
        {"family": "RVV", "source_ranges": ["lib/Dialect/RVV", "lib/Plugin/RVV"]},
        {"family": "IME", "source_ranges": ["lib/Dialect/IME", "lib/Plugin/IME"]},
    ]
    core_subdirs = {"Exec", "EmitC", "Builtin", "Construction", "IR"}
    core_files = ["lib/Transforms/VariantSelection.cpp",
                  "lib/Conversion/EmitC/Builtin/BuiltinBackendEmitters.cpp"]

    # ---- DEFAULT evaluator ----
    good_root_files = [
        "lib/Dialect/RVV/IR/RVVOps.td", "lib/Plugin/RVV/RVVExtensionPlugin.cpp",
        "lib/Dialect/IME/IR/IMEOps.td", "lib/Plugin/IME/IMEExtensionPlugin.cpp",
        "lib/Dialect/Exec/ExecOps.cpp",          # core subdir -> skipped
        "lib/Plugin/ExtensionPlugin.cpp",        # depth-1 core file -> skipped
    ]
    ok, _ = evaluate_default(good_root_files, core_files, syn_families, core_subdirs, 4)
    check("default: clean tree (disjoint, covered, ratchet ok) -> GREEN", ok)

    # orphan: a file under known family dir RVV not covered by its ranges
    ok, errs = evaluate_default(good_root_files + ["lib/Target/RVV/RVVExport.cpp"],
                                core_files, syn_families, core_subdirs, 4)
    check("default: family-dir file outside its range -> RED (ORPHAN)",
          (not ok) and any("ORPHAN" in e for e in errs))

    # unknown family-shaped dir with no manifest entry
    ok, errs = evaluate_default(good_root_files + ["lib/Dialect/NewFam/IR/NewOps.td"],
                                core_files, syn_families, core_subdirs, 4)
    check("default: unmanifested family-shaped dir -> RED (UNKNOWN-FAMILY-DIR)",
          (not ok) and any("UNKNOWN-FAMILY-DIR" in e for e in errs))

    # cross-family pollution: a file matched by two families
    poll_families = [
        {"family": "RVV", "source_ranges": ["lib/Dialect/RVV", "lib/Dialect"]},
        {"family": "IME", "source_ranges": ["lib/Dialect/IME"]},
    ]
    ok, errs = evaluate_default(["lib/Dialect/IME/IR/IMEOps.td"], core_files,
                                poll_families, core_subdirs, 3)
    check("default: file claimed by two families -> RED (CROSS-FAMILY-POLLUTION)",
          (not ok) and any("CROSS-FAMILY-POLLUTION" in e for e in errs))

    # anti-widen: a family range swallows a core file
    widen_families = [
        {"family": "RVV", "source_ranges": ["lib/Dialect/RVV", "lib/Transforms"]},
        {"family": "IME", "source_ranges": ["lib/Dialect/IME"]},
    ]
    ok, errs = evaluate_default(["lib/Dialect/RVV/IR/RVVOps.td",
                                 "lib/Dialect/IME/IR/IMEOps.td"],
                                core_files, widen_families, core_subdirs, 3)
    check("default: family range matches a core file -> RED (ANTI-WIDEN)",
          (not ok) and any("ANTI-WIDEN" in e for e in errs))

    # ratchet: baseline != total ranges
    ok, errs = evaluate_default(good_root_files, core_files, syn_families, core_subdirs, 9)
    check("default: baseline != total source_ranges -> RED (RATCHET)",
          (not ok) and any("RATCHET" in e for e in errs))

    # ---- CI diff evaluator ----
    ok, _ = evaluate_diff(
        ["lib/Dialect/RVV/IR/RVVOps.td", "lib/Plugin/RVV/RVVExtensionPlugin.cpp",
         "schema/coverage-sixstate.v1.json", "docs/method/FALSIFIER-INDEX.md",
         "test/Dialect/RVV/foo.mlir", "lib/Dialect/RVV/CMakeLists.txt"],
        syn_families)
    check("CI: single-family PR + table rows + docs + tests -> GREEN", ok)

    ok, errs = evaluate_diff(
        ["lib/Dialect/RVV/IR/RVVOps.td", "lib/Dialect/IME/IR/IMEOps.td"], syn_families)
    check("CI: PR touching two families -> RED (CROSS-FAMILY-PR)",
          (not ok) and any("CROSS-FAMILY-PR" in e for e in errs))

    ok, errs = evaluate_diff(
        ["lib/Plugin/RVV/RVVExtensionPlugin.cpp", "lib/Transforms/VariantSelection.cpp"],
        syn_families)
    check("CI: family PR that also edits a core-source file -> RED (CORE-EDIT)",
          (not ok) and any("CORE-EDIT" in e for e in errs))

    ok, _ = evaluate_diff(["lib/Transforms/VariantSelection.cpp"], syn_families)
    check("CI: pure core refactor (no family touched) -> GREEN (not a family PR)", ok)

    ok, _ = evaluate_diff(
        ["schema/coverage-roster.v1.json", "docs/x.md", "test/Dialect/RVV/foo.mlir"],
        syn_families)
    check("CI: pure table/docs/tests PR -> GREEN (no source touched)", ok)

    if fails:
        print(f"[f3-family-locality --self-test] RED: {len(fails)} discrimination(s) failed")
        return 2
    print("[f3-family-locality --self-test] GREEN: both evaluators discriminate all cases")
    return 0


def main(argv):
    verbose = "-v" in argv or "--verbose" in argv
    if "--self-test" in argv:
        return run_self_test()
    if "--base" in argv and "--head" in argv:
        base = argv[argv.index("--base") + 1]
        head = argv[argv.index("--head") + 1]
        return run_ci(base, head, verbose)
    return run_default(verbose)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
