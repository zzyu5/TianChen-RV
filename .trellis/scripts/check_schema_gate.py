#!/usr/bin/env python3
"""check_schema_gate.py — E1 two-level schema.def governance gate.

Implements [S-6] / [F-2'] for the hand-authored canonical schema.def
(schema/capability.schema.v1.json). This is governance / workflow tooling and
is stdlib-only: Python is tooling here, never the compiler stack (implementation
-stack red line). It does NOT read or touch any C++/ODS.

Subcommands
-----------
report [--check]
    [S-6] report gate. Normalize the schema JSON via
    json.dumps(obj, sort_keys=True, separators=(",",":")) and take its SHA256.
    Prints the digest. With --check, also verifies that the VERSIONLOG entry for
    the schema's own version carries the same digest (fails if the shape hash
    drifted with no matching VERSIONLOG line).

gate --base <ref> --head <ref> [--onboarding]
    [F-2'] operation gate. Computes the changed-file set of a PR
    (git diff --name-only base..head). IF the PR is a family-onboarding PR
    (marked by a `Family-Onboarding:` commit trailer in base..head, or forced
    with --onboarding), the gate FAILS (exit non-zero) when the change set
    intersects the frozen schema.def contract (schema/capability.schema.v1.json +
    schema/VERSIONLOG.md) — NOT the rest of schema/ (family manifests / facts / table
    rows are [F-3] territory). An onboarding PR reaching for even one additive schema
    field IS the falsifier firing — it is never waved through as "additive". A
    non-onboarding diff, or one that does not touch schema.def, passes.

--self-test
    Hermetic checks (no real git refs, no repo writes): report determinism,
    F-2' path-intersection, and additive-vs-breaking classification.

The report gate (per-version shape hash) and the operation gate (per-PR ∅) act on
different PR classes: additive-minor is how a core-author EVOLUTION PR (via RFC) is
graded; it is NOT an onboarding loophole.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

# --- locations (single source of truth) ------------------------------------
# This file lives at <repo>/.trellis/scripts/check_schema_gate.py
REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_JSON = REPO_ROOT / "schema" / "capability.schema.v1.json"
VERSIONLOG = REPO_ROOT / "schema" / "VERSIONLOG.md"

# [F-2'] is `diff ∩ schema.def = ∅` -- schema.def is the [S-5] frozen SHAPE artifact
# (schema/capability.schema.v1.json, the six-item hashed shape) + its VERSIONLOG, NOT the
# whole schema/ directory. The other schema/ artifacts (family-manifest.v1.json,
# family-regex.v1.json, coverage-*.json, emit-bypass-whitelist.v1.json, ...) are capability
# FACTS / TABLE ROWS / family manifests -- explicitly OUT of the [S-5] shape ("具体事实行 /
# 模式注册表条目 不入 shape") and governed by other gates ([F-1] regex, [F-3] containment,
# [F-EMIT] whitelist). They are exactly the "+ 表行" allowance an onboarding PR MUST touch
# (adding its family block + a baseline bump). Gating the whole schema/ prefix would
# false-trigger [F-2'] on that mandatory family-manifest edit -- a direct collision with
# [F-3] (spec architecture/插件协议.md prescribes this narrowing). Gate the EXACT contract
# files only. Derived from the single-source-of-truth SCHEMA_JSON / VERSIONLOG constants.
SCHEMA_DEF_PATHS = (
    SCHEMA_JSON.relative_to(REPO_ROOT).as_posix(),   # schema.def -- the [S-5] shape artifact
    VERSIONLOG.relative_to(REPO_ROOT).as_posix(),    # its version log (report-gate companion)
)

# The commit-trailer that marks a family-onboarding PR (interim identifier until
# E2b lands the plugins/<family>/ directory layout for [F-3] containment).
ONBOARDING_TRAILER = "Family-Onboarding:"


# --- core: normalization + hashing ([S-6]) ---------------------------------
def canonicalize(obj) -> str:
    """Canonical serialization used for the [S-6] shape hash."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def compute_hash(obj) -> str:
    """SHA256 hexdigest of the canonical serialization of a JSON object."""
    return hashlib.sha256(canonicalize(obj).encode("utf-8")).hexdigest()


def load_schema(path: Path = SCHEMA_JSON):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# --- [F-2'] operation gate (pure logic) ------------------------------------
def schema_intersection(changed_files, gate_paths=SCHEMA_DEF_PATHS):
    """Files in the change set that touch the frozen schema.def contract.

    EXACT-path match against the [S-5] shape artifact + its VERSIONLOG only -- NOT the rest
    of schema/ (facts / table rows / family manifests are [F-3] territory, not [F-2']).
    """
    gate = set(gate_paths)
    return [f for f in changed_files if f in gate]


def is_f2prime_violation(changed_files, is_onboarding: bool,
                         gate_paths=SCHEMA_DEF_PATHS) -> bool:
    """[F-2'] falsifier: an onboarding PR intersecting the frozen schema.def contract.

    A non-onboarding PR touching schema.def is NOT a violation here (that is a
    core-author evolution PR, graded by classify_schema_change under RFC).
    """
    if not is_onboarding:
        return False
    return len(schema_intersection(changed_files, gate_paths)) > 0


# --- additive-vs-breaking classifier (structural JSON diff) ----------------
def _jtype(v) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, float):
        return "float"
    if isinstance(v, str):
        return "str"
    if isinstance(v, list):
        return "list"
    if isinstance(v, dict):
        return "dict"
    if v is None:
        return "null"
    return "unknown"


def _list_identity_key(objs):
    """Identity field common to all dict members of a list, or None.

    Tries conventional identity fields in priority order; returns the first that
    is present in every object with a hashable scalar value. None when no such
    field exists (the caller then falls back to a conservative count check).
    """
    for candidate in ("name", "id", "key"):
        if all(candidate in o and isinstance(o[candidate], (str, int, bool))
               for o in objs):
            return candidate
    return None


def _has_breaking(old, new) -> bool:
    """True if new drops/renames/retypes any shape element present in old.

    Only-additions (new keys, new scalar list members, new object list members)
    are NOT breaking. A rename collapses to remove+add and is therefore correctly
    graded breaking. Pure scalar value edits (e.g. a note string) are not shape
    changes -> not breaking.
    """
    if _jtype(old) != _jtype(new):
        return True  # retype of an existing element
    if isinstance(old, dict):
        for key, oval in old.items():
            if key not in new:
                return True  # removed key
            if _has_breaking(oval, new[key]):
                return True
        return False  # extra keys in new are additive
    if isinstance(old, list):
        old_scalars = [x for x in old if not isinstance(x, (dict, list))]
        new_scalars = [x for x in new if not isinstance(x, (dict, list))]
        for member in old_scalars:
            if member not in new_scalars:
                return True  # removed enum/vocabulary member
        # List-of-object members (e.g. item_5.methods[]): key each object by its
        # identity field so a removed object entry, or a key-removal/retype
        # WITHIN a matched object, grades breaking. item ⑤ is P-1 frozen, so a
        # dropped method IS a shape break, never an "additive" no-op.
        old_objs = [x for x in old if isinstance(x, dict)]
        new_objs = [x for x in new if isinstance(x, dict)]
        if old_objs:
            key = _list_identity_key(old_objs)
            if key is None:
                # No stable identity: any decrease in object count is breaking.
                if len(new_objs) < len(old_objs):
                    return True
            else:
                new_by_key = {o[key]: o for o in new_objs if key in o}
                for o in old_objs:
                    if o[key] not in new_by_key:
                        return True  # removed object member
                    if _has_breaking(o, new_by_key[o[key]]):
                        return True
        return False  # extra scalar/object members are additive
    return False  # scalar value change alone is not a shape break


def classify_schema_change(old, new) -> str:
    """'breaking' (major, RFC required) or 'additive' (minor).

    Coverage: dict key add/remove/retype; SCALAR list-member (enum) add/remove;
    and OBJECT list members keyed by identity field (name/id/key) — a removed
    object entry, or a key-removal/type-change within a matched object (e.g. one
    entry of item_5.methods[]), grades 'breaking'.

    Residual limitation (narrowed, must still go through RFC review): a pure
    value-level edit to a name-matched object's scalar field — e.g. a method's
    `arity` 2->1, `pure` true->false, or its `signature` string text — grades
    'additive', because a value edit is indistinguishable from a benign note
    edit without field-level semantics (real scope). item ⑤ is P-1-frozen, so
    such an in-place signature edit is truly breaking and is caught by RFC review,
    not relied upon from this classifier.
    """
    return "breaking" if _has_breaking(old, new) else "additive"


# --- git plumbing (only used by the live `gate` subcommand) -----------------
def _git(args):
    return subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        check=True, capture_output=True, text=True,
    ).stdout


def git_changed_files(base: str, head: str):
    out = _git(["diff", "--name-only", f"{base}..{head}"])
    return [line for line in out.splitlines() if line.strip()]


def git_range_is_onboarding(base: str, head: str) -> bool:
    """True if any commit message in base..head carries the onboarding trailer."""
    out = _git(["log", "--format=%B", f"{base}..{head}"])
    return any(line.strip().startswith(ONBOARDING_TRAILER)
               for line in out.splitlines())


# --- subcommands -----------------------------------------------------------
def cmd_report(args) -> int:
    obj = load_schema()
    version = obj["$meta"]["schema_version"]
    digest = compute_hash(obj)
    print(f"schema:  {SCHEMA_JSON.relative_to(REPO_ROOT)}")
    print(f"version: {version}")
    print(f"sha256:  {digest}")
    if not args.check:
        return 0

    logged = _versionlog_digest_for(version)
    if logged is None:
        print(f"[--check] FAIL: no VERSIONLOG entry for {version}", file=sys.stderr)
        return 1
    if logged != digest:
        print("[--check] FAIL: shape hash drifted without a VERSIONLOG entry",
              file=sys.stderr)
        print(f"  computed: {digest}", file=sys.stderr)
        print(f"  logged:   {logged}", file=sys.stderr)
        return 1
    print(f"[--check] OK: matches VERSIONLOG entry for {version}")
    return 0


def _versionlog_digest_for(version: str):
    """Return the sha256 field of the VERSIONLOG line for `version`, or None.

    Line format: `v<semver> | <RFC-id> | <sha256> | <additive|breaking> | note`
    """
    if not VERSIONLOG.exists():
        return None
    for raw in VERSIONLOG.read_text(encoding="utf-8").splitlines():
        parts = [p.strip() for p in raw.split("|")]
        if len(parts) >= 3 and parts[0] == version:
            return parts[2]
    return None


def cmd_gate(args) -> int:
    changed = git_changed_files(args.base, args.head)
    is_onboarding = args.onboarding or git_range_is_onboarding(args.base, args.head)
    offending = schema_intersection(changed)

    print(f"base..head: {args.base}..{args.head}")
    print(f"onboarding PR: {is_onboarding}")
    print(f"changed files: {len(changed)}")

    if is_f2prime_violation(changed, is_onboarding):
        print("[F-2'] FAIL: family-onboarding PR modifies the schema shape:",
              file=sys.stderr)
        for f in offending:
            print(f"  - {f}", file=sys.stderr)
        print("An onboarding PR touching schema.def — even to ADD a field — is "
              "the falsifier firing; it is not 'additive'.", file=sys.stderr)
        return 1

    if offending and not is_onboarding:
        print("[F-2'] OK: schema.def touched, but this is not an onboarding PR "
              "(evolution PR — grade with classify_schema_change under RFC).")
    else:
        print("[F-2'] OK: no onboarding-PR intersection with the schema shape.")
    return 0


# --- self-test -------------------------------------------------------------
def cmd_self_test(_args) -> int:
    results = []

    def check(name, cond):
        results.append((name, bool(cond)))

    # (a) report determinism -------------------------------------------------
    obj = load_schema()
    h1 = compute_hash(obj)
    h2 = compute_hash(json.loads(canonicalize(obj)))  # reparse round-trip
    check("report: real schema hashes identically twice", h1 == h2)

    shuffled_a = {"b": 1, "a": [3, 2, 1], "c": {"y": 2, "x": 1}}
    shuffled_b = {"c": {"x": 1, "y": 2}, "a": [3, 2, 1], "b": 1}
    check("report: key insertion order does not affect the hash",
          compute_hash(shuffled_a) == compute_hash(shuffled_b))

    # (b) [F-2'] path-intersection ------------------------------------------
    onboarding_touch = ["schema/capability.schema.v1.json", "plugins/foo/x.cpp"]
    onboarding_clean = ["plugins/foo/x.cpp", "docs/foo.md"]
    evolution_touch = ["schema/capability.schema.v1.json"]
    check("F-2': onboarding PR touching schema/ -> violation",
          is_f2prime_violation(onboarding_touch, is_onboarding=True) is True)
    check("F-2': onboarding PR not touching schema/ -> pass",
          is_f2prime_violation(onboarding_clean, is_onboarding=True) is False)
    check("F-2': non-onboarding PR touching schema/ -> pass (evolution PR)",
          is_f2prime_violation(evolution_touch, is_onboarding=False) is False)
    check("F-2': schema_intersection isolates the schema path",
          schema_intersection(onboarding_touch) ==
          ["schema/capability.schema.v1.json"])

    # (b') narrowing: schema.def = the frozen SHAPE contract only (capability.schema.v1.json
    # + VERSIONLOG.md), NOT the whole schema/ dir. An onboarding PR MUST add its family block
    # in family-manifest.v1.json (+ a baseline bump) and its dispatch keys / coverage rows --
    # those are the [F-3] "+ 表行" allowance and must NOT false-trigger [F-2'] (collision).
    onboarding_manifest = ["schema/family-manifest.v1.json", "plugins/foo/x.cpp"]
    onboarding_tablerows = ["schema/family-regex.v1.json",
                            "schema/coverage-sixstate.v1.json"]
    onboarding_versionlog = ["schema/VERSIONLOG.md"]
    check("F-2': onboarding PR touching family-manifest.v1.json -> pass ([F-3] table row, "
          "not schema.def)",
          is_f2prime_violation(onboarding_manifest, is_onboarding=True) is False)
    check("F-2': onboarding PR touching family-regex + coverage rows -> pass (table rows)",
          is_f2prime_violation(onboarding_tablerows, is_onboarding=True) is False)
    check("F-2': onboarding PR touching schema.def (capability.schema) -> still violation",
          is_f2prime_violation(onboarding_touch, is_onboarding=True) is True)
    check("F-2': onboarding PR touching VERSIONLOG.md -> still violation (schema.def companion)",
          is_f2prime_violation(onboarding_versionlog, is_onboarding=True) is True)
    check("F-2': schema_intersection ignores non-contract schema/ files (only schema.def)",
          schema_intersection(["schema/family-manifest.v1.json",
                               "schema/family-regex.v1.json",
                               "schema/capability.schema.v1.json"]) ==
          ["schema/capability.schema.v1.json"])

    # (c) additive-vs-breaking classifier -----------------------------------
    base = {
        "fields": {"id": {"type": "string"}},
        "kind": {"members": ["isa_ext", "sub_ext"]},
    }
    add_field = {
        "fields": {"id": {"type": "string"}, "subclass": {"type": "string"}},
        "kind": {"members": ["isa_ext", "sub_ext"]},
    }
    add_enum_member = {
        "fields": {"id": {"type": "string"}},
        "kind": {"members": ["isa_ext", "sub_ext", "uarch"]},
    }
    remove_field = {
        "fields": {},
        "kind": {"members": ["isa_ext", "sub_ext"]},
    }
    retype_field = {
        "fields": {"id": {"type": 123}},
        "kind": {"members": ["isa_ext", "sub_ext"]},
    }
    remove_enum_member = {
        "fields": {"id": {"type": "string"}},
        "kind": {"members": ["isa_ext"]},
    }
    check("classify: add optional field -> additive",
          classify_schema_change(base, add_field) == "additive")
    check("classify: add enum member -> additive",
          classify_schema_change(base, add_enum_member) == "additive")
    check("classify: remove field -> breaking",
          classify_schema_change(base, remove_field) == "breaking")
    check("classify: retype field -> breaking",
          classify_schema_change(base, retype_field) == "breaking")
    check("classify: remove enum member -> breaking",
          classify_schema_change(base, remove_enum_member) == "breaking")

    # (c') object list members keyed by identity (item_5.methods[]) ----------
    methods_base = {"methods": [
        {"name": "getName", "arity": 0, "pure": True},
        {"name": "getVersion", "arity": 0, "pure": False},
    ]}
    methods_add = {"methods": [
        {"name": "getName", "arity": 0, "pure": True},
        {"name": "getVersion", "arity": 0, "pure": False},
        {"name": "getExtra", "arity": 1, "pure": False},
    ]}
    methods_add_field = {"methods": [
        {"name": "getName", "arity": 0, "pure": True, "const": True},
        {"name": "getVersion", "arity": 0, "pure": False},
    ]}
    methods_remove = {"methods": [
        {"name": "getName", "arity": 0, "pure": True},
    ]}
    methods_retype = {"methods": [
        {"name": "getName", "arity": "zero", "pure": True},  # int -> str
        {"name": "getVersion", "arity": 0, "pure": False},
    ]}
    methods_drop_key = {"methods": [
        {"name": "getName", "arity": 0},  # dropped `pure`
        {"name": "getVersion", "arity": 0, "pure": False},
    ]}
    check("classify: add a method object -> additive",
          classify_schema_change(methods_base, methods_add) == "additive")
    check("classify: add a field to a matched method -> additive",
          classify_schema_change(methods_base, methods_add_field) == "additive")
    check("classify: remove a method object -> breaking",
          classify_schema_change(methods_base, methods_remove) == "breaking")
    check("classify: retype a matched method field -> breaking",
          classify_schema_change(methods_base, methods_retype) == "breaking")
    check("classify: drop a key within a matched method -> breaking",
          classify_schema_change(methods_base, methods_drop_key) == "breaking")

    # report -----------------------------------------------------------------
    ok = True
    for name, passed in results:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
        ok = ok and passed
    print(f"\n{'ALL PASS' if ok else 'FAILURES PRESENT'} "
          f"({sum(p for _, p in results)}/{len(results)})")
    return 0 if ok else 1


# --- entrypoint ------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true",
                        help="run hermetic self-tests and exit")
    sub = parser.add_subparsers(dest="cmd")

    p_report = sub.add_parser("report", help="[S-6] report gate: normalize -> SHA256")
    p_report.add_argument("--check", action="store_true",
                          help="verify the digest against the VERSIONLOG entry")
    p_report.set_defaults(func=cmd_report)

    p_gate = sub.add_parser("gate", help="[F-2'] operation gate: diff ∩ schema.def = ∅")
    p_gate.add_argument("--base", required=True)
    p_gate.add_argument("--head", required=True)
    p_gate.add_argument("--onboarding", action="store_true",
                        help="force onboarding-PR classification (testing)")
    p_gate.set_defaults(func=cmd_gate)

    args = parser.parse_args(argv)
    if args.self_test:
        return cmd_self_test(args)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
