#!/usr/bin/env python3
"""Generate and validate reproducible T-N qualification evidence.

The tool turns pre-registered repeat-to-repeat noise samples and paired effect
samples into a deterministic qualification document.  It does not benchmark,
choose a compiler variant, publish the master, or infer missing provenance.
Evidence is accepted only when:

* both pre-registered sample counts are at least ten and match the arrays;
* ``abs(median(effect)) > 2 * IQR(noise)``;
* a deterministic bootstrap 95% CI of the effect median excludes zero;
* every named source artifact is immutable-run evidence with a matching SHA-256.

The emitted JSON embeds its normalized input and can therefore be recomputed
byte-for-byte by the publication gate.  There is no textual-token shortcut.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "experiments" / "runs"
SCHEMA_VERSION = "weft.tn.qualification.v1"
MIN_N = 10
MIN_BOOTSTRAP_RESAMPLES = 1000
BOARDS = frozenset({"rvv", "k1", "scalar", "rvv07"})


class TNQualificationError(ValueError):
    """T-N input/evidence is incomplete, inconsistent, or not qualified."""


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise TNQualificationError("percentile input is empty")
    if len(ordered) == 1:
        return ordered[0]
    index = percentile / 100.0 * (len(ordered) - 1)
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _finite_samples(raw: object, *, field: str) -> list[float]:
    if not isinstance(raw, list):
        raise TNQualificationError(f"{field} must be an array")
    values = []
    for index, value in enumerate(raw):
        if isinstance(value, bool):
            raise TNQualificationError(f"{field}[{index}] is boolean, not a sample")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise TNQualificationError(f"{field}[{index}] is not numeric") from exc
        if not math.isfinite(number):
            raise TNQualificationError(f"{field}[{index}] is not finite")
        values.append(number)
    return values


def _normalize_input(raw: dict[str, Any]) -> dict[str, Any]:
    board = raw.get("board")
    if board not in BOARDS:
        raise TNQualificationError(f"unknown board {board!r}")
    benchmark_class = raw.get("benchmark_class")
    protocol_id = raw.get("protocol_id")
    effect_run_id = raw.get("effect_run_id")
    for name, value in (("benchmark_class", benchmark_class),
                        ("protocol_id", protocol_id),
                        ("effect_run_id", effect_run_id)):
        if not isinstance(value, str) or not value or value != value.strip():
            raise TNQualificationError(f"{name} must be a nonblank exact string")
    run_pattern = rf"\d{{8}}T\d{{6}}Z-[A-Za-z0-9_.+-]+-{re.escape(board)}-[0-9a-f]{{8}}"
    if not re.fullmatch(run_pattern, effect_run_id):
        raise TNQualificationError("effect_run_id is malformed or board-mismatched")

    n_noise = raw.get("pre_registered_n_noise")
    n_effect = raw.get("pre_registered_n_effect")
    if (not isinstance(n_noise, int) or isinstance(n_noise, bool)
            or not isinstance(n_effect, int) or isinstance(n_effect, bool)):
        raise TNQualificationError("pre-registered N values must be integers")
    if n_noise < MIN_N or n_effect < MIN_N:
        raise TNQualificationError(
            f"pre-registered N below T-N floor: noise={n_noise}, effect={n_effect}, min={MIN_N}"
        )
    noise = _finite_samples(raw.get("noise_repeat_deltas_pct"),
                            field="noise_repeat_deltas_pct")
    effect = _finite_samples(raw.get("effect_deltas_pct"), field="effect_deltas_pct")
    if len(noise) != n_noise or len(effect) != n_effect:
        raise TNQualificationError(
            "sample counts do not equal pre-registration: "
            f"noise={len(noise)}/{n_noise}, effect={len(effect)}/{n_effect}"
        )

    seed = raw.get("bootstrap_seed")
    resamples = raw.get("bootstrap_resamples")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TNQualificationError("bootstrap_seed must be an integer")
    if (not isinstance(resamples, int) or isinstance(resamples, bool)
            or resamples < MIN_BOOTSTRAP_RESAMPLES):
        raise TNQualificationError(
            f"bootstrap_resamples must be >= {MIN_BOOTSTRAP_RESAMPLES}"
        )

    artifacts = raw.get("source_artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise TNQualificationError("source_artifacts must be a nonempty array")
    normalized_artifacts = []
    seen_paths = set()
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict):
            raise TNQualificationError(f"source_artifacts[{index}] must be an object")
        path = artifact.get("path")
        digest = artifact.get("sha256")
        if not isinstance(path, str) or not path.startswith("experiments/runs/"):
            raise TNQualificationError(f"source_artifacts[{index}] is outside immutable runs")
        if path in seen_paths:
            raise TNQualificationError(f"duplicate source artifact {path!r}")
        seen_paths.add(path)
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise TNQualificationError(f"source_artifacts[{index}] has invalid sha256")
        normalized_artifacts.append({"path": path, "sha256": digest})

    return {
        "board": board,
        "benchmark_class": benchmark_class,
        "protocol_id": protocol_id,
        "effect_run_id": effect_run_id,
        "pre_registered_n_noise": n_noise,
        "pre_registered_n_effect": n_effect,
        "noise_repeat_deltas_pct": noise,
        "effect_deltas_pct": effect,
        "bootstrap_seed": seed,
        "bootstrap_resamples": resamples,
        "source_artifacts": normalized_artifacts,
    }


def _bootstrap_ci95_median(samples: list[float], *, seed: int,
                           resamples: int) -> tuple[float, float]:
    rng = random.Random(seed)
    size = len(samples)
    medians = [statistics.median(rng.choices(samples, k=size))
               for _ in range(resamples)]
    return _percentile(medians, 2.5), _percentile(medians, 97.5)


def qualify(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = _normalize_input(raw)
    noise = normalized["noise_repeat_deltas_pct"]
    effect = normalized["effect_deltas_pct"]
    noise_iqr = _percentile(noise, 75) - _percentile(noise, 25)
    effect_median = statistics.median(effect)
    ci_low, ci_high = _bootstrap_ci95_median(
        effect,
        seed=normalized["bootstrap_seed"],
        resamples=normalized["bootstrap_resamples"],
    )
    exceeds_noise = abs(effect_median) > 2.0 * noise_iqr
    excludes_zero = ci_low > 0.0 or ci_high < 0.0
    qualified = exceeds_noise and excludes_zero
    result = {
        "status": "T-N-QUALIFIED" if qualified else "T-N-REJECTED",
        "noise_floor_iqr_pct": round(noise_iqr, 12),
        "effect_median_pct": round(effect_median, 12),
        "twice_noise_floor_pct": round(2.0 * noise_iqr, 12),
        "effect_exceeds_twice_noise_floor": exceeds_noise,
        "bootstrap": {
            "method": "deterministic-resample-median-v1",
            "seed": normalized["bootstrap_seed"],
            "resamples": normalized["bootstrap_resamples"],
            "ci95_low_pct": round(ci_low, 12),
            "ci95_high_pct": round(ci_high, 12),
            "excludes_zero": excludes_zero,
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "input_sha256": hashlib.sha256(_canonical_bytes(normalized)).hexdigest(),
        "input": normalized,
        "result": result,
    }


def validate_qualification_document(document: dict[str, Any], *, repo_root: Path,
                                    expected_board: str,
                                    expected_run_id: str,
                                    require_qualified: bool = True) -> dict[str, Any]:
    if document.get("schema_version") != SCHEMA_VERSION:
        raise TNQualificationError("unknown T-N evidence schema")
    recomputed = qualify(document.get("input") or {})
    if document != recomputed:
        raise TNQualificationError("T-N evidence does not match deterministic recomputation")
    if recomputed["input"]["board"] != expected_board:
        raise TNQualificationError("T-N evidence board does not match publication board")
    if recomputed["input"]["effect_run_id"] != expected_run_id:
        raise TNQualificationError("T-N evidence run-id does not match publication run")
    if require_qualified and recomputed["result"]["status"] != "T-N-QUALIFIED":
        raise TNQualificationError("T-N evidence is a measured rejection, not qualification")

    runs_root = (repo_root / "experiments" / "runs").resolve()
    for artifact in recomputed["input"]["source_artifacts"]:
        path = (repo_root / artifact["path"]).resolve()
        try:
            path.relative_to(runs_root)
        except ValueError as exc:
            raise TNQualificationError("source artifact escapes immutable runs") from exc
        if not path.is_file():
            raise TNQualificationError(f"source artifact is missing: {path}")
        if _sha256(path) != artifact["sha256"]:
            raise TNQualificationError(f"source artifact hash mismatch: {path}")
    return recomputed


def validate_evidence_file(path: Path, *, repo_root: Path, expected_board: str,
                           expected_run_id: str) -> dict[str, Any]:
    runs_root = (repo_root / "experiments" / "runs").resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(runs_root)
    except ValueError as exc:
        raise TNQualificationError("T-N evidence must live under immutable runs") from exc
    if not resolved.is_file():
        raise TNQualificationError(f"T-N evidence is missing: {resolved}")
    with resolved.open(encoding="utf-8") as stream:
        document = json.load(stream)
    return validate_qualification_document(
        document, repo_root=repo_root, expected_board=expected_board,
        expected_run_id=expected_run_id,
    )


def _synthetic_raw(root: Path, *, n: int = MIN_N) -> dict[str, Any]:
    run_id = "20990101T000000Z-q4_K-rvv-deadbeef"
    artifact = root / "experiments" / "runs" / run_id / "measure.stdout.txt"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("synthetic immutable timing source\n", encoding="utf-8")
    noise_base = [-0.20, -0.15, -0.10, -0.05, 0.0, 0.0, 0.05, 0.10, 0.15, 0.20]
    effect_base = [2.0, 2.1, 2.2, 2.1, 2.0, 2.2, 2.1, 2.0, 2.2, 2.1]
    return {
        "board": "rvv",
        "benchmark_class": "vec_dot-micro-fixed",
        "protocol_id": "synthetic-preregistered-protocol",
        "effect_run_id": run_id,
        "pre_registered_n_noise": n,
        "pre_registered_n_effect": n,
        "noise_repeat_deltas_pct": noise_base[:n],
        "effect_deltas_pct": effect_base[:n],
        "bootstrap_seed": 7,
        "bootstrap_resamples": 1000,
        "source_artifacts": [{
            "path": str(artifact.relative_to(root)),
            "sha256": _sha256(artifact),
        }],
    }


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="weft-tn-") as directory:
        root = Path(directory)
        raw = _synthetic_raw(root, n=MIN_N)
        document = qualify(raw)
        assert document["result"]["status"] == "T-N-QUALIFIED"
        validated = validate_qualification_document(
            document, repo_root=root, expected_board="rvv",
            expected_run_id=raw["effect_run_id"],
        )
        assert validated == document

        below = dict(raw)
        below["pre_registered_n_noise"] = MIN_N - 1
        below["pre_registered_n_effect"] = MIN_N - 1
        below["noise_repeat_deltas_pct"] = raw["noise_repeat_deltas_pct"][:-1]
        below["effect_deltas_pct"] = raw["effect_deltas_pct"][:-1]
        try:
            qualify(below)
        except TNQualificationError:
            pass
        else:
            raise AssertionError("N=9 unexpectedly qualified")

        tampered = json.loads(json.dumps(document))
        tampered["result"]["status"] = "T-N-QUALIFIED"
        tampered["result"]["effect_median_pct"] = 999.0
        try:
            validate_qualification_document(
                tampered, repo_root=root, expected_board="rvv",
                expected_run_id=raw["effect_run_id"],
            )
        except TNQualificationError:
            pass
        else:
            raise AssertionError("tampered evidence unexpectedly accepted")

        weak = dict(raw)
        weak["effect_deltas_pct"] = [0.0] * MIN_N
        rejected = qualify(weak)
        assert rejected["result"]["status"] == "T-N-REJECTED"
        try:
            validate_qualification_document(
                rejected, repo_root=root, expected_board="rvv",
                expected_run_id=raw["effect_run_id"],
            )
        except TNQualificationError:
            pass
        else:
            raise AssertionError("T-N rejection unexpectedly accepted as qualified")
    print("T-N qualification self-test: PASS (N=10 accept; N=9/tamper/no-effect reject)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input", type=Path, help="pre-registered T-N sample JSON")
    parser.add_argument("--out", type=Path,
                        help="new immutable qualification JSON under experiments/runs/")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return 0
    if not args.input:
        parser.error("--input is required unless --self-test is used")
    with args.input.open(encoding="utf-8") as stream:
        raw = json.load(stream)
    document = qualify(raw)
    validate_qualification_document(
        document,
        repo_root=ROOT,
        expected_board=document["input"]["board"],
        expected_run_id=document["input"]["effect_run_id"],
        require_qualified=False,
    )
    text = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        out = args.out.resolve()
        try:
            out.relative_to(RUNS.resolve())
        except ValueError:
            parser.error("--out must be under experiments/runs/")
        if not out.parent.is_dir():
            parser.error("--out parent run directory must already exist")
        with out.open("x", encoding="utf-8") as stream:
            stream.write(text)
        print(f"wrote immutable T-N evidence: {out}")
    else:
        print(text, end="")
    return 0 if document["result"]["status"] == "T-N-QUALIFIED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
