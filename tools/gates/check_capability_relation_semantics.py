#!/usr/bin/env python3
"""check_capability_relation_semantics.py — schema<->code conformance falsifier (C6).

WHAT THIS GATE IS FOR
---------------------
schema/capability.schema.v1.json is declared TARGET-shaped ($meta.declares =
"target"): several of its fields are deliberately ahead of the code, and that
divergence is tracked, not a defect. But three parts of the schema are NOT
target-ahead — they are inventories/annotations OF EXISTING CODE, so a
disagreement there is a plain conformance bug that nothing was catching:

  A. item_3_relation_types  — the per-relation SEMANTIC annotation
     (satisfies-by-alias / transitive-satisfiable / fail-closed-mutual-exclusion).
     Landed in code as the typed table in lib/Support/CapabilityModel.cpp; this
     gate holds schema and that table to each other.
  B. item_6 axis A         — RuntimeABIParameterRole, an inventory of the enum in
     include/Weft/Support/RuntimeABI.h (declared member_count + member list).
  C. item_6 axis B         — role_attributes, an inventory of the role attributes
     actually declared in RVVOps.td + ConstructionProtocol.h.
  D. fail-closed invariant — a relation annotated fail-closed-mutual-exclusion
     must never participate in satisfaction (a conflict is the opposite of a
     satisfier). Text-checked here; the BEHAVIORAL proof is
     test/Support/CapabilityModelTest.cpp, which goes red if the table entry is
     flipped (that is what makes the annotation load-bearing rather than a
     comment).

This gate does NOT touch item_6's two-axis structure itself: Decision #5 keeps
axis A and axis B deliberately separate, and merging them is a canon-level
ruling, not this checker's business. It only asserts each axis faithfully
inventories the code it claims to describe.

SCOPE / NON-GOALS
-----------------
Static text conformance only. It does not execute weft-opt and does not prove
the semantics are implemented correctly — only that code and schema agree on
what they are, and that the annotation table exists in the shape the schema
declares. Behavior is the unit test's job.

--self-test feeds the pure comparison core synthetic COMPLIANT and NON-COMPLIANT
inputs and asserts it discriminates. A gate that cannot go red is worse than no
gate, so the negative control is part of the gate itself.

Usage:  python3 tools/gates/check_capability_relation_semantics.py [--self-test] [-v]
Exit:   0 = GREEN, 1 = RED (conformance violation), 2 = setup error.
"""

import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(os.path.dirname(__file__) + "/../")))
REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

SCHEMA = os.path.join(REPO, "schema", "capability.schema.v1.json")
MODEL_CPP = os.path.join(REPO, "lib", "Support", "CapabilityModel.cpp")
RUNTIME_ABI_H = os.path.join(REPO, "include", "Weft", "Support", "RuntimeABI.h")
RVV_OPS_TD = os.path.join(REPO, "include", "Weft", "Dialect", "RVV", "IR", "RVVOps.td")
CONSTRUCTION_PROTOCOL_H = os.path.join(REPO, "include", "Weft", "Plugin", "ConstructionProtocol.h")


# ---------------------------------------------------------------------------
# Pure comparison core (fed synthetic inputs by --self-test; real inputs at run)
# ---------------------------------------------------------------------------

def normalize_role_spelling(name):
    """Fold a role spelling to a comparison key.

    Schema uses kebab-case ("rhs-secondary-scalar-value"); the C++ enum uses
    CamelCase with acronym runs ("RHSSecondaryScalarValue"). Dropping '-' and
    lowercasing makes the two directly comparable without an acronym table.
    """
    return name.replace("-", "").lower()


def compare_mapping(label, schema_map, code_map):
    """Compare two {key: value} tables. Returns a list of failure strings."""
    failures = []
    for key in sorted(set(schema_map) - set(code_map)):
        failures.append(f"{label}: '{key}' declared in schema but absent from code")
    for key in sorted(set(code_map) - set(schema_map)):
        failures.append(f"{label}: '{key}' present in code but not declared in schema")
    for key in sorted(set(schema_map) & set(code_map)):
        if schema_map[key] != code_map[key]:
            failures.append(
                f"{label}: '{key}' semantics disagree — schema='{schema_map[key]}' "
                f"code='{code_map[key]}'"
            )
    return failures


def compare_role_sets(label, schema_names, code_names):
    """Compare two role-name collections under normalized spelling."""
    failures = []
    schema_keys = {normalize_role_spelling(n): n for n in schema_names}
    code_keys = {normalize_role_spelling(n): n for n in code_names}
    for key in sorted(set(schema_keys) - set(code_keys)):
        failures.append(
            f"{label}: '{schema_keys[key]}' declared in schema but not found in code"
        )
    for key in sorted(set(code_keys) - set(schema_keys)):
        failures.append(
            f"{label}: '{code_keys[key]}' declared in code but missing from schema "
            f"(schema under-lists the code it claims to inventory)"
        )
    return failures


def check_declared_count(label, declared_count, actual_len):
    if declared_count != actual_len:
        return [
            f"{label}: declared member_count={declared_count} but the listed member "
            f"array holds {actual_len} entries (schema self-inconsistent)"
        ]
    return []


def check_fail_closed_invariant(label, participation):
    """`fail-closed-mutual-exclusion` must never participate in satisfaction."""
    failures = []
    for semantics, participates in sorted(participation.items()):
        if semantics == "FailClosedMutualExclusion" and participates:
            failures.append(
                f"{label}: FailClosedMutualExclusion participates in satisfaction — "
                f"a conflict must never satisfy (fail-closed violated)"
            )
    return failures


# ---------------------------------------------------------------------------
# Extractors (impure: read the real tree)
# ---------------------------------------------------------------------------

def read(path):
    if not os.path.isfile(path):
        raise SystemExit(f"[relation-semantics] SETUP RED: missing file {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def slice_function_body(text, signature_fragment, path_label):
    """Return the body text of the function whose definition contains fragment."""
    idx = text.find(signature_fragment)
    if idx < 0:
        raise SystemExit(
            f"[relation-semantics] SETUP RED: {path_label} has no '{signature_fragment}' "
            f"— the code-side annotation table this gate checks is absent or renamed."
        )
    open_brace = text.find("{", idx)
    depth = 0
    for pos in range(open_brace, len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace : pos + 1]
    raise SystemExit(f"[relation-semantics] SETUP RED: unbalanced braces in {path_label}")


def extract_code_relation_table(cpp_text):
    """kind -> schema-facing relation name, and kind -> semantics enum name."""
    name_body = slice_function_body(
        cpp_text, "llvm::StringRef getCapabilityRelationName", "CapabilityModel.cpp"
    )
    kind_to_name = dict(
        re.findall(r'case\s+CapabilityRelationKind::(\w+):\s*return\s+"([^"]+)";', name_body)
    )

    sem_body = slice_function_body(
        cpp_text, "getCapabilityRelationSemantics(CapabilityRelationKind", "CapabilityModel.cpp"
    )
    kind_to_semantics_enum = dict(
        re.findall(
            r"case\s+CapabilityRelationKind::(\w+):\s*return\s+CapabilityRelationSemantics::(\w+);",
            sem_body,
        )
    )

    sem_name_body = slice_function_body(
        cpp_text, "getCapabilityRelationSemanticsName(CapabilityRelationSemantics", "CapabilityModel.cpp"
    )
    semantics_enum_to_name = dict(
        re.findall(
            r'case\s+CapabilityRelationSemantics::(\w+):\s*return\s+"([^"]+)";', sem_name_body
        )
    )

    # Compose: relation name -> semantics spelling, as the schema states it.
    table = {}
    for kind, relation_name in kind_to_name.items():
        semantics_enum = kind_to_semantics_enum.get(kind)
        if semantics_enum is None:
            raise SystemExit(
                f"[relation-semantics] SETUP RED: relation kind {kind} has a name but no "
                f"semantics entry in the code table."
            )
        spelling = semantics_enum_to_name.get(semantics_enum)
        if spelling is None:
            raise SystemExit(
                f"[relation-semantics] SETUP RED: semantics {semantics_enum} has no spelling."
            )
        table[relation_name] = spelling
    return table


def extract_code_participation(cpp_text):
    """semantics enum name -> bool participates-in-satisfaction."""
    body = slice_function_body(
        cpp_text, "bool relationSemanticsParticipateInSatisfaction", "CapabilityModel.cpp"
    )
    # Comment lines may sit between the case label and its return.
    pairs = re.findall(
        r"case\s+CapabilityRelationSemantics::(\w+):\s*(?://[^\n]*\n\s*)*return\s+(true|false);",
        body,
    )
    return {name: value == "true" for name, value in pairs}


def extract_code_abi_roles(header_text):
    """Ordered RuntimeABIParameterRole enum members."""
    body = slice_function_body(
        header_text, "enum class RuntimeABIParameterRole", "RuntimeABI.h"
    )
    return re.findall(r"^\s{2,}([A-Z]\w*),\s*$", body, flags=re.MULTILINE)


def extract_code_body_op_roles(td_text, protocol_text):
    """Role attributes declared in code, across every declaration form.

    Three forms exist and a form-blind grep silently misses one:
      RVVOps.td :  StrAttr:$mask_role
      RVVOps.td :  OptionalAttr<StrAttr>:$selected_path_role
      header    :  llvm::StringRef sourceRoleAttrName = "source_role";
    Matching '$<name>_role' in the .td covers both .td forms regardless of the
    attr wrapper; the quoted form covers the protocol header.
    """
    roles = set(re.findall(r"\$([a-z0-9_]+_roles?)\b", td_text))
    roles |= set(re.findall(r'"([a-z0-9_]+_roles?)"', protocol_text))
    return roles


# ---------------------------------------------------------------------------
# Self-test: negative control
# ---------------------------------------------------------------------------

def run_self_test(verbose=False):
    cases = []

    # --- compare_mapping: compliant vs each way it can diverge ---------------
    good = {"provides": "satisfies-by-alias", "implies": "transitive-satisfiable"}
    cases.append(("mapping/identical", compare_mapping("m", good, dict(good)), False))
    cases.append(
        ("mapping/semantics-flipped",
         compare_mapping("m", good, {"provides": "transitive-satisfiable",
                                     "implies": "transitive-satisfiable"}), True))
    cases.append(
        ("mapping/relation-missing-in-code",
         compare_mapping("m", good, {"provides": "satisfies-by-alias"}), True))
    cases.append(
        ("mapping/extra-relation-in-code",
         compare_mapping("m", good, dict(good, conflicts="fail-closed-mutual-exclusion")), True))

    # --- compare_role_sets: spelling folding + under-listing -----------------
    cases.append(
        ("roles/kebab-vs-camel-equivalent",
         compare_role_sets("r", ["rhs-secondary-scalar-value"], ["RHSSecondaryScalarValue"]), False))
    cases.append(
        ("roles/schema-under-lists-code",
         compare_role_sets("r", ["mask_role"], ["mask_role", "field0_role"]), True))
    cases.append(
        ("roles/schema-over-lists-code",
         compare_role_sets("r", ["mask_role", "ghost_role"], ["mask_role"]), True))

    # --- declared count self-consistency ------------------------------------
    cases.append(("count/consistent", check_declared_count("c", 28, 28), False))
    cases.append(("count/inconsistent", check_declared_count("c", 28, 27), True))

    # --- fail-closed invariant ----------------------------------------------
    cases.append(
        ("failclosed/conflicts-does-not-satisfy",
         check_fail_closed_invariant("f", {"SatisfiesByAlias": True,
                                           "FailClosedMutualExclusion": False}), False))
    cases.append(
        ("failclosed/conflicts-satisfies",
         check_fail_closed_invariant("f", {"SatisfiesByAlias": True,
                                           "FailClosedMutualExclusion": True}), True))

    bad = []
    for label, failures, should_be_red in cases:
        is_red = len(failures) > 0
        ok = is_red == should_be_red
        if verbose or not ok:
            print(f"  [{'ok' if ok else 'BROKEN'}] {label}: "
                  f"{'RED' if is_red else 'GREEN'} (expected {'RED' if should_be_red else 'GREEN'})")
        if not ok:
            bad.append(label)

    if bad:
        print(f"[relation-semantics --self-test] RED: {len(bad)} discrimination(s) failed: {bad}")
        return 1
    print(f"[relation-semantics --self-test] GREEN: core discriminates all "
          f"{len(cases)} cases ({sum(1 for c in cases if c[2])} negative controls fire)")
    return 0


# ---------------------------------------------------------------------------
# Real run
# ---------------------------------------------------------------------------

def main(argv):
    verbose = "-v" in argv
    if "--self-test" in argv:
        return run_self_test(verbose)

    schema = json.loads(read(SCHEMA))
    cpp_text = read(MODEL_CPP)
    failures = []

    # --- A. item_3 relation semantics annotation ----------------------------
    schema_relations = schema["item_3_relation_types"]["relations"]
    schema_table = {name: spec["semantics"] for name, spec in schema_relations.items()}
    code_table = extract_code_relation_table(cpp_text)
    failures += compare_mapping("item_3 relation semantics", schema_table, code_table)

    # --- D. fail-closed invariant -------------------------------------------
    participation = extract_code_participation(cpp_text)
    failures += check_fail_closed_invariant("item_3 fail-closed", participation)

    # --- B. item_6 axis A: RuntimeABIParameterRole inventory -----------------
    axis_a = schema["item_6_operand_role_vocabulary"]["axis_abi_signature_slot"]
    code_abi_roles = extract_code_abi_roles(read(RUNTIME_ABI_H))
    failures += check_declared_count(
        "item_6 axis A", axis_a["member_count"], len(axis_a["members"])
    )
    failures += compare_role_sets("item_6 axis A", axis_a["members"], code_abi_roles)
    if len(code_abi_roles) != axis_a["member_count"]:
        failures.append(
            f"item_6 axis A: declared member_count={axis_a['member_count']} but "
            f"RuntimeABI.h enum holds {len(code_abi_roles)} members"
        )

    # --- C. item_6 axis B: body-op role attribute inventory ------------------
    axis_b = schema["item_6_operand_role_vocabulary"]["axis_body_op_semantic_role"]
    code_body_roles = extract_code_body_op_roles(read(RVV_OPS_TD), read(CONSTRUCTION_PROTOCOL_H))
    failures += compare_role_sets("item_6 axis B", axis_b["role_attributes"], code_body_roles)

    if verbose:
        print(f"  item_3 relations (code): {code_table}")
        print(f"  item_3 participation   : {participation}")
        print(f"  item_6 axis A members  : schema={len(axis_a['members'])} code={len(code_abi_roles)}")
        print(f"  item_6 axis B roles    : schema={len(axis_b['role_attributes'])} code={len(code_body_roles)}")

    if failures:
        print(f"[relation-semantics] RED: {len(failures)} conformance violation(s):")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(
        f"[relation-semantics] GREEN: item_3 semantics annotation "
        f"({len(code_table)} relations) + item_6 axis A ({len(code_abi_roles)} slots) "
        f"+ axis B ({len(code_body_roles)} role attrs) all conform to code"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
