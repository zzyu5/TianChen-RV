#!/usr/bin/env python3
"""Canonical measurement row-key contract shared by bench and recon checks.

This module owns identity validation only.  It does not choose a candidate,
classify a performance result, qualify a run, or write the canonical master.
Missing/blank/unknown components fail closed; there is deliberately no legacy
normalization or empty-regime wildcard.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping, Sequence


ROW_KEY = ("op", "format", "engine", "regime")
ENGINE_VALUES = frozenset({"rvv", "ime", "scalar"})
REGIME_VALUES = frozenset({"micro-fixed", "decode", "prefill"})


class MeasurementKeyError(ValueError):
    """A row identity is missing, ambiguous, or outside the closed schema."""


def key_from_mapping(row: Mapping[str, object], *, source: str) -> tuple[str, str, str, str]:
    values = []
    for field in ROW_KEY:
        raw = row.get(field)
        if not isinstance(raw, str) or not raw:
            raise MeasurementKeyError(f"{source}: row key field {field!r} is missing/blank")
        if raw != raw.strip():
            raise MeasurementKeyError(f"{source}: row key field {field!r} contains surrounding whitespace")
        values.append(raw)
    key = tuple(values)
    if key[2] not in ENGINE_VALUES:
        raise MeasurementKeyError(
            f"{source}: unknown engine {key[2]!r}; allowed={sorted(ENGINE_VALUES)}"
        )
    if key[3] not in REGIME_VALUES:
        raise MeasurementKeyError(
            f"{source}: unknown regime {key[3]!r}; allowed={sorted(REGIME_VALUES)}"
        )
    return key  # type: ignore[return-value]


def _reject_duplicates(keys: Sequence[tuple[str, str, str, str]], *, source: str) -> None:
    duplicates = sorted(key for key, count in Counter(keys).items() if count != 1)
    if duplicates:
        raise MeasurementKeyError(f"{source}: duplicate exact row key(s): {duplicates}")


def validate_roster_contract_document(doc: Mapping[str, object], *, source: str) -> None:
    """Bind the implementation copy to roster's versioned contract declaration."""
    meta = doc.get("$meta")
    if not isinstance(meta, Mapping):
        raise MeasurementKeyError(f"{source}: missing $meta")
    contract = meta.get("measurement_row_key")
    if not isinstance(contract, Mapping):
        raise MeasurementKeyError(f"{source}: missing $meta.measurement_row_key")
    fields = contract.get("fields")
    engines = contract.get("engine_values")
    regimes = contract.get("regime_values")
    if tuple(fields or ()) != ROW_KEY:
        raise MeasurementKeyError(
            f"{source}: row-key fields drift: schema={fields!r}, implementation={ROW_KEY!r}"
        )
    if frozenset(engines or ()) != ENGINE_VALUES:
        raise MeasurementKeyError(
            f"{source}: engine enum drift: schema={engines!r}, implementation={sorted(ENGINE_VALUES)!r}"
        )
    if frozenset(regimes or ()) != REGIME_VALUES:
        raise MeasurementKeyError(
            f"{source}: regime enum drift: schema={regimes!r}, implementation={sorted(REGIME_VALUES)!r}"
        )
    if contract.get("empty_components_forbidden") is not True:
        raise MeasurementKeyError(f"{source}: empty-components prohibition is not active")


def load_roster_keys(path: Path) -> list[tuple[str, str, str, str]]:
    with path.open(encoding="utf-8") as stream:
        doc = json.load(stream)
    validate_roster_contract_document(doc, source=str(path))
    keys = []
    for index, row in enumerate(doc.get("kernels", [])):
        # Class-C entries are explicitly outside the T3 five-group master.  They
        # are not silently canonicalized into measurable cells here.
        if row.get("class") == "C":
            continue
        keys.append(key_from_mapping(row, source=f"{path}:kernels[{index}]"))
    _reject_duplicates(keys, source=str(path))
    return keys


def load_master_keys(path: Path) -> list[tuple[str, str, str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = [key_from_mapping(row, source=f"{path}:row[{index + 2}]") for index, row in enumerate(rows)]
    _reject_duplicates(keys, source=str(path))
    return keys


def validate_registered_key(
    key: tuple[str, str, str, str],
    *,
    roster_keys: Iterable[tuple[str, str, str, str]],
    master_keys: Iterable[tuple[str, str, str, str]],
) -> None:
    """Require exactly one roster declaration and at most one master cell.

    A zero-row master result is legal for a newly declared cell; the roster is
    the declaration authority.  A duplicate anywhere is rejected before this
    function by the loaders, and no missing component is inferred.
    """

    # Re-run the closed-enum checks for direct callers constructing a tuple.
    key_from_mapping(dict(zip(ROW_KEY, key)), source="requested key")
    roster_count = sum(candidate == key for candidate in roster_keys)
    if roster_count != 1:
        raise MeasurementKeyError(
            f"requested key {key!r}: expected exactly one roster declaration, found {roster_count}"
        )
    master_count = sum(candidate == key for candidate in master_keys)
    if master_count > 1:
        raise MeasurementKeyError(
            f"requested key {key!r}: canonical master is ambiguous ({master_count} rows)"
        )


def self_test() -> None:
    good = ("vec_dot", "q4_K", "rvv", "micro-fixed")
    validate_registered_key(good, roster_keys=[good], master_keys=[good])
    validate_registered_key(good, roster_keys=[good], master_keys=[])

    bad_rows = [
        {"op": "vec_dot", "format": "q4_K", "engine": "rvv", "regime": ""},
        {"op": "vec_dot", "format": "q4_K", "engine": "rvv", "regime": "PREFILL"},
        {"op": "vec_dot", "format": "q4_K", "engine": "rvv ", "regime": "micro-fixed"},
        {"op": "vec_dot", "format": "q4_K", "engine": "gpu", "regime": "micro-fixed"},
    ]
    for index, row in enumerate(bad_rows):
        try:
            key_from_mapping(row, source=f"negative[{index}]")
        except MeasurementKeyError:
            pass
        else:
            raise AssertionError(f"negative row unexpectedly accepted: {row}")

    for roster in ([], [good, good]):
        try:
            validate_registered_key(good, roster_keys=roster, master_keys=[])
        except MeasurementKeyError:
            pass
        else:
            raise AssertionError(f"ambiguous/missing roster unexpectedly accepted: {roster}")

    contract = {
        "$meta": {"measurement_row_key": {
            "fields": list(ROW_KEY),
            "engine_values": sorted(ENGINE_VALUES),
            "regime_values": sorted(REGIME_VALUES),
            "empty_components_forbidden": True,
        }}
    }
    validate_roster_contract_document(contract, source="synthetic-good")
    contract["$meta"]["measurement_row_key"]["regime_values"] = ["legacy-empty"]
    try:
        validate_roster_contract_document(contract, source="synthetic-drift")
    except MeasurementKeyError:
        pass
    else:
        raise AssertionError("roster contract drift unexpectedly accepted")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate canonical measurement row identities")
    parser.add_argument("--roster", type=Path)
    parser.add_argument("--master", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        print("measurement-key self-test: PASS")
        return 0
    if not args.roster or not args.master:
        parser.error("--roster and --master are required unless --self-test is used")
    roster = load_roster_keys(args.roster)
    master = load_master_keys(args.master)
    missing = sorted(set(master) - set(roster))
    if missing:
        raise MeasurementKeyError(f"master contains row key(s) absent from roster: {missing}")
    print(f"measurement-key contract: PASS roster={len(roster)} master={len(master)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
