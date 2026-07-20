#!/usr/bin/env python3
"""Validate the A1 formula-authority freeze against the current source tree.

This is deliberately a characterization gate.  It records both working authority
chains and known duplicate/default paths.  Later A-line cutovers must update the
matrix atomically with the code that retires a recorded path.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any


EXPECTED_DECISIONS = {
    "dequant.nibble",
    "dequant.codebook",
    "dequant.kquant",
    "dequant.grid",
    "dequant.ternary",
    "repack.accumulator_lmul",
    "repack.sp4_tiling",
    "repack.loop_order",
}

REQUIRED_DECISION_FIELDS = {
    "id",
    "actual_owner",
    "g",
    "c",
    "omega",
    "provider",
    "legal_set",
    "selection",
    "stamp",
    "consumer",
    "emission",
    "classification",
    "tests",
}


def repo_path(root: Path, relative: str, errors: list[str]) -> Path | None:
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        errors.append(f"path escapes repository root: {relative}")
        return None
    return candidate


def read_text(root: Path, relative: str, errors: list[str]) -> str | None:
    path = repo_path(root, relative, errors)
    if path is None:
        return None
    if not path.is_file():
        errors.append(f"referenced file does not exist: {relative}")
        return None
    return path.read_text(encoding="utf-8")


def validate(root: Path, matrix: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    if matrix.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")
    allowed = set(matrix.get("classifications", []))
    if allowed != {"analytic", "measured", "constant", "honest-null", "fallback"}:
        errors.append("classification vocabulary drifted")

    decisions = matrix.get("decisions")
    if not isinstance(decisions, list):
        return errors + ["decisions must be a list"]
    ids = [entry.get("id") for entry in decisions if isinstance(entry, dict)]
    if set(ids) != EXPECTED_DECISIONS or len(ids) != len(EXPECTED_DECISIONS):
        errors.append(
            "decision set must be exactly: " + ", ".join(sorted(EXPECTED_DECISIONS))
        )
    if matrix.get("snapshot", {}).get("decision_count") != len(decisions):
        errors.append("snapshot.decision_count does not match decisions")

    for index, decision in enumerate(decisions):
        if not isinstance(decision, dict):
            errors.append(f"decision[{index}] is not an object")
            continue
        missing = REQUIRED_DECISION_FIELDS - set(decision)
        if missing:
            errors.append(f"{decision.get('id', index)} missing fields: {sorted(missing)}")
        if not isinstance(decision.get("actual_owner"), str) or not decision.get(
            "actual_owner", ""
        ).strip():
            errors.append(f"{decision.get('id', index)} needs one actual_owner description")
        classes = decision.get("classification", [])
        if not classes or any(item not in allowed for item in classes):
            errors.append(f"{decision.get('id', index)} has invalid classification")
        tests = decision.get("tests", [])
        if not tests:
            errors.append(f"{decision.get('id', index)} has no characterization test")
        for test in tests:
            content = read_text(root, test, errors)
            if content is not None and "RUN:" not in content:
                errors.append(f"characterization test has no RUN line: {test}")

    issues_text = ""
    issues_dir = root / ".trellis/spec/issues"
    if issues_dir.is_dir():
        issues_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in sorted(issues_dir.glob("*.md"))
            if path.name != "index.md"
        )
    else:
        errors.append("issue register directory is missing")

    debts = matrix.get("debts")
    if not isinstance(debts, list) or not debts:
        errors.append("debts must be a non-empty list")
    else:
        names: set[str] = set()
        for index, debt in enumerate(debts):
            if not isinstance(debt, dict):
                errors.append(f"debt[{index}] is not an object")
                continue
            for field in ("name", "issue", "owner_task", "retire", "killing_test"):
                if not debt.get(field):
                    errors.append(f"debt[{index}] missing {field}")
            name = debt.get("name", "")
            if name in names:
                errors.append(f"duplicate debt name: {name}")
            names.add(name)
            issue = debt.get("issue", "")
            if issue and f"### {issue} " not in issues_text:
                errors.append(f"debt issue is not registered: {issue}")
            owner_task = debt.get("owner_task", "")
            task_path = repo_path(root, owner_task, errors) if owner_task else None
            if task_path is not None and not (task_path / "prd.md").is_file():
                errors.append(f"debt owner task has no prd.md: {owner_task}")
            if not isinstance(debt.get("retire"), list) or not debt.get("retire"):
                errors.append(f"debt has no concrete retirement symbols/callers: {name}")

    for contract in matrix.get("test_contracts", []):
        relative = contract.get("path", "")
        content = read_text(root, relative, errors)
        if content is None:
            continue
        for needle in contract.get("needles", []):
            if needle not in content:
                errors.append(f"test contract drift: {relative} lacks {needle!r}")

    for contract in matrix.get("source_contracts", []):
        relative = contract.get("path", "")
        content = read_text(root, relative, errors)
        if content is None:
            continue
        needle = contract.get("needle")
        expected = contract.get("count")
        if not isinstance(needle, str) or not isinstance(expected, int) or expected < 0:
            errors.append(f"invalid source contract: {contract!r}")
            continue
        actual = content.count(needle)
        if actual != expected:
            errors.append(
                f"source contract drift: {relative} has {actual} occurrences of "
                f"{needle!r}; expected {expected}"
            )

    return errors


def run_self_test(root: Path, matrix: dict[str, Any]) -> list[str]:
    failures: list[str] = []

    missing_decision = copy.deepcopy(matrix)
    missing_decision["decisions"] = missing_decision["decisions"][:-1]
    if not validate(root, missing_decision):
        failures.append("decision-removal mutation was not detected")

    stale_source = copy.deepcopy(matrix)
    stale_source["source_contracts"][0]["count"] += 1
    if not validate(root, stale_source):
        failures.append("source-count mutation was not detected")

    orphan_debt = copy.deepcopy(matrix)
    orphan_debt["debts"][0]["owner_task"] = ".trellis/tasks/does-not-exist"
    if not validate(root, orphan_debt):
        failures.append("orphan-task mutation was not detected")

    return failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    root = args.repo_root.resolve()
    matrix_path = args.matrix.resolve()
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    errors = validate(root, matrix)
    if errors:
        for error in errors:
            print(f"error: {error}", file=sys.stderr)
        return 1

    if args.self_test:
        failures = run_self_test(root, matrix)
        if failures:
            for failure in failures:
                print(f"error: {failure}", file=sys.stderr)
            return 1
        print("formula authority matrix self-test ok")
    else:
        print("formula authority matrix ok: 8 decisions, all source/test contracts match")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
