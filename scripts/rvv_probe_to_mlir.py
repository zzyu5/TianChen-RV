#!/usr/bin/env python3
"""Replay sanitized RVV probe evidence as tcrv.exec capability MLIR.

This is artifact parsing tooling only. It does not implement TianChen-RV
capability relations, plugin decisions, legality, selection, lowering,
emission, runtime glue, correctness, or performance measurement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


try:
    from rvv_remote_probe import PROBE_NAME, SCHEMA_VERSION
except ImportError:  # pragma: no cover - only used for unusual import setups.
    PROBE_NAME = "tianchenrv-rvv-remote-probe"
    SCHEMA_VERSION = 3

SUPPORTED_SCHEMA_VERSIONS = {2, SCHEMA_VERSION}


MAX_FACT_LENGTH = 512
MAX_KERNEL_NAME_LENGTH = 128

SECRET_LIKE_RE = re.compile(
    r"(?i)(token|secret|password|passwd|api[_-]?key|access[_-]?key|"
    r"private[_ -]?key|authorization\s*:)"
)

BARE_SYMBOL_RE = re.compile(r"^[A-Za-z_$][A-Za-z0-9_.$-]*$")


class ReplayError(ValueError):
    pass


def load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            artifact = json.load(handle)
    except OSError as error:
        raise ReplayError(f"failed to read evidence artifact: {error}") from error
    except json.JSONDecodeError as error:
        raise ReplayError(f"failed to parse evidence JSON: {error}") from error

    if not isinstance(artifact, dict):
        raise ReplayError("evidence artifact must be a JSON object")
    return artifact


def require_type(container: dict[str, Any], key: str, expected: type) -> Any:
    if key not in container:
        raise ReplayError(f"missing required evidence key: {key}")
    value = container[key]
    if not isinstance(value, expected):
        raise ReplayError(f"evidence key {key} has wrong type")
    return value


def optional_str(container: dict[str, Any], key: str) -> str:
    value = container.get(key, "")
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ReplayError(f"capability fact {key} has wrong type")
    return validate_fact_string(key, value, required=False)


def optional_bool(container: dict[str, Any], key: str) -> bool:
    value = container.get(key, False)
    if not isinstance(value, bool):
        raise ReplayError(f"capability fact {key} has wrong type")
    return value


def optional_int(container: dict[str, Any], key: str) -> int:
    value = container.get(key, 0)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReplayError(f"capability fact {key} has wrong type")
    if value < 0:
        raise ReplayError(f"capability fact {key} must be non-negative")
    return value


def validate_fact_string(name: str, value: str, *, required: bool) -> str:
    text = value.strip()
    if not text:
        if required:
            raise ReplayError(f"capability fact {name} is required")
        return ""

    if len(text) > MAX_FACT_LENGTH:
        raise ReplayError(
            f"capability fact {name} must be at most {MAX_FACT_LENGTH} bytes"
        )

    for character in text:
        byte = ord(character)
        if character in ("\n", "\r", "\x00"):
            raise ReplayError(
                f"capability fact {name} must be a bounded single-line fact"
            )
        if byte < 0x20 and character != "\t":
            raise ReplayError(
                f"capability fact {name} contains unsupported control text"
            )

    if SECRET_LIKE_RE.search(text):
        raise ReplayError(
            f"capability fact {name} must not contain secret-like or raw-log text"
        )

    return text


def validate_kernel_name(name: str) -> str:
    text = name.strip()
    if not text:
        raise ReplayError("kernel name must be non-empty")
    if len(text) > MAX_KERNEL_NAME_LENGTH:
        raise ReplayError("kernel name is too long")
    if not BARE_SYMBOL_RE.match(text):
        raise ReplayError(
            "kernel name cannot be represented by the current MLIR symbol subset"
        )
    return text


def mlir_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def attr_line(name: str, value: str, *, trailing_comma: bool = True) -> str:
    comma = "," if trailing_comma else ""
    return f"      {name} = {mlir_string(value)}{comma}"


def status_from_bool(value: bool) -> str:
    return "available" if value else "unavailable"


def append_capability(
    lines: list[str],
    symbol: str,
    attrs: list[tuple[str, str | int | list[str]]],
    provider_symbols: list[str] | None = None,
) -> None:
    if provider_symbols is not None:
        provider_symbols.append(symbol)
    lines.append(f"    tcrv.exec.capability @{symbol} {{")
    for index, (name, value) in enumerate(attrs):
        is_last = index == len(attrs) - 1
        comma = "" if is_last else ","
        if isinstance(value, int):
            lines.append(f"      {name} = {value} : i64{comma}")
        elif isinstance(value, list):
            entries = ", ".join(mlir_string(entry) for entry in value)
            lines.append(f"      {name} = [{entries}]{comma}")
        else:
            lines.append(f"      {name} = {mlir_string(value)}{comma}")
    lines.append("    }")


def validated_capability_facts(artifact: dict[str, Any]) -> dict[str, Any]:
    schema_version = require_type(artifact, "schema_version", int)
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ReplayError("unsupported RVV probe evidence schema version")

    probe_name = require_type(artifact, "probe_name", str)
    if probe_name != PROBE_NAME:
        raise ReplayError("unexpected RVV probe evidence name")

    require_type(artifact, "status", str)
    require_type(artifact, "success", bool)
    capability_facts = require_type(artifact, "capability_facts", dict)

    compile_run_succeeded = optional_bool(
        capability_facts, "minimal_rvv_compile_run_succeeded"
    )
    first_slice_sew_bits = optional_int(capability_facts, "first_slice_sew_bits")
    first_slice_lmul = optional_str(capability_facts, "first_slice_lmul")
    first_slice_tail_policy = optional_str(
        capability_facts, "first_slice_tail_policy"
    )
    first_slice_mask_policy = optional_str(
        capability_facts, "first_slice_mask_policy"
    )
    i32_m2_sew_bits = optional_int(capability_facts, "i32_m2_sew_bits")
    i32_m2_lmul = optional_str(capability_facts, "i32_m2_lmul")
    i32_m2_tail_policy = optional_str(capability_facts, "i32_m2_tail_policy")
    i32_m2_mask_policy = optional_str(capability_facts, "i32_m2_mask_policy")
    if compile_run_succeeded:
        first_slice_sew_bits = first_slice_sew_bits or 32
        first_slice_lmul = first_slice_lmul or "m1"
        first_slice_tail_policy = first_slice_tail_policy or "agnostic"
        first_slice_mask_policy = first_slice_mask_policy or "agnostic"
        i32_m2_sew_bits = i32_m2_sew_bits or 32
        i32_m2_lmul = i32_m2_lmul or "m2"
        i32_m2_tail_policy = i32_m2_tail_policy or "agnostic"
        i32_m2_mask_policy = i32_m2_mask_policy or "agnostic"

    facts = {
        "architecture": optional_str(capability_facts, "architecture"),
        "hart_count": optional_int(capability_facts, "hart_count"),
        "vlenb_bytes": optional_int(capability_facts, "vlenb_bytes"),
        "i32_m1_lane_count": optional_int(capability_facts, "i32_m1_lane_count"),
        "first_slice_sew_bits": first_slice_sew_bits,
        "first_slice_lmul": first_slice_lmul,
        "first_slice_tail_policy": first_slice_tail_policy,
        "first_slice_mask_policy": first_slice_mask_policy,
        "i32_m2_sew_bits": i32_m2_sew_bits,
        "i32_m2_lmul": i32_m2_lmul,
        "i32_m2_tail_policy": i32_m2_tail_policy,
        "i32_m2_mask_policy": i32_m2_mask_policy,
        "i64_m1_sew_bits": optional_int(capability_facts, "i64_m1_sew_bits"),
        "i64_m1_lmul": optional_str(capability_facts, "i64_m1_lmul"),
        "i64_m1_tail_policy": optional_str(
            capability_facts, "i64_m1_tail_policy"
        ),
        "i64_m1_mask_policy": optional_str(
            capability_facts, "i64_m1_mask_policy"
        ),
        "isa_vector_hints": optional_str(capability_facts, "isa_vector_hints"),
        "clang_available": optional_bool(capability_facts, "clang_available"),
        "clang_version": optional_str(capability_facts, "clang_version"),
        "cmake_available": optional_bool(capability_facts, "cmake_available"),
        "cmake_version": optional_str(capability_facts, "cmake_version"),
        "minimal_rvv_compile_run_succeeded": compile_run_succeeded,
        "selected_march": optional_str(capability_facts, "selected_march"),
        "selected_mabi": optional_str(capability_facts, "selected_mabi"),
        "source_sha256": optional_str(capability_facts, "source_sha256"),
        "binary_sha256": optional_str(capability_facts, "binary_sha256"),
    }
    return facts


def default_kernel_name(artifact: dict[str, Any]) -> str:
    run_id = str(artifact.get("run_id", "rvv_probe_replay"))
    sanitized = re.sub(r"[^A-Za-z0-9_.$-]+", "_", run_id).strip("_")
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"rvv_probe_{sanitized or 'replay'}"
    return validate_kernel_name(sanitized)


def append_kernel_header(
    lines: list[str],
    kernel_symbol: str,
    *,
    target_profile: str,
) -> None:
    attrs: list[str] = []
    if target_profile:
        attrs.append(f"target = @{target_profile}")
    if attrs:
        lines.append(
            f"  tcrv.exec.kernel @{kernel_symbol} attributes "
            f"{{{', '.join(attrs)}}} {{"
        )
    else:
        lines.append(f"  tcrv.exec.kernel @{kernel_symbol} {{")


def emit_replay_mlir(
    artifact: dict[str, Any],
    *,
    kernel_name: str | None,
    include_scalar_fallback: bool,
    scalar_fallback_status: str,
    emit_target_profile: bool = False,
) -> str:
    facts = validated_capability_facts(artifact)
    kernel_symbol = validate_kernel_name(kernel_name or default_kernel_name(artifact))
    target_profile_symbol = f"{kernel_symbol}_profile" if emit_target_profile else ""

    rvv_status = (
        "available"
        if facts["architecture"] or facts["isa_vector_hints"]
        else "unavailable"
    )
    compile_run_status = status_from_bool(
        facts["minimal_rvv_compile_run_succeeded"]
    )

    lines: list[str] = ["module {"]
    provider_symbols: list[str] = []
    if not emit_target_profile:
        append_kernel_header(
            lines,
            kernel_symbol,
            target_profile="",
        )
    append_capability(
        lines,
        "rvv",
        [
            ("id", "rvv"),
            ("kind", "isa-vector"),
            ("architecture", facts["architecture"]),
            ("isa_vector_hints", facts["isa_vector_hints"]),
            ("status", rvv_status),
        ],
        provider_symbols if emit_target_profile else None,
    )
    append_capability(
        lines,
        "rvv_hart_count",
        [
            ("id", "rvv.hart_count"),
            ("kind", "uarch"),
            ("provides", ["target.hart_count"]),
            ("count", facts["hart_count"]),
            ("status", status_from_bool(facts["hart_count"] > 0)),
        ],
        provider_symbols if emit_target_profile else None,
    )
    if facts["vlenb_bytes"]:
        append_capability(
            lines,
            "rvv_vlenb_bytes",
            [
                ("id", "rvv.vlenb_bytes"),
                ("kind", "uarch"),
                ("bytes", facts["vlenb_bytes"]),
                ("status", status_from_bool(facts["vlenb_bytes"] > 0)),
            ],
            provider_symbols if emit_target_profile else None,
        )
    if facts["i32_m1_lane_count"]:
        append_capability(
            lines,
            "rvv_i32_m1_lanes",
            [
                ("id", "rvv.i32_m1_lane_count"),
                ("kind", "uarch"),
                ("lanes", facts["i32_m1_lane_count"]),
                ("status", status_from_bool(facts["i32_m1_lane_count"] > 0)),
            ],
            provider_symbols if emit_target_profile else None,
        )
    if facts["first_slice_sew_bits"]:
        append_capability(
            lines,
            "rvv_i32_m1_sew32",
            [
                ("id", "rvv.i32_m1.sew32"),
                ("kind", "isa-vector-config"),
                (
                    "status",
                    status_from_bool(
                        facts["minimal_rvv_compile_run_succeeded"]
                        and facts["first_slice_sew_bits"] == 32
                    ),
                ),
                ("sew_bits", facts["first_slice_sew_bits"]),
            ],
            provider_symbols if emit_target_profile else None,
        )
    if facts["first_slice_lmul"]:
        append_capability(
            lines,
            "rvv_i32_m1_lmul_m1",
            [
                ("id", "rvv.i32_m1.lmul_m1"),
                ("kind", "isa-vector-config"),
                (
                    "status",
                    status_from_bool(
                        facts["minimal_rvv_compile_run_succeeded"]
                        and facts["first_slice_lmul"] == "m1"
                    ),
                ),
                ("lmul", facts["first_slice_lmul"]),
            ],
            provider_symbols if emit_target_profile else None,
        )
    if facts["first_slice_tail_policy"]:
        append_capability(
            lines,
            "rvv_i32_m1_tail_agnostic",
            [
                ("id", "rvv.i32_m1.tail_policy.agnostic"),
                ("kind", "isa-vector-config"),
                (
                    "status",
                    status_from_bool(
                        facts["minimal_rvv_compile_run_succeeded"]
                        and facts["first_slice_tail_policy"] == "agnostic"
                    ),
                ),
                ("tail_policy", facts["first_slice_tail_policy"]),
            ],
            provider_symbols if emit_target_profile else None,
        )
    if facts["first_slice_mask_policy"]:
        append_capability(
            lines,
            "rvv_i32_m1_mask_agnostic",
            [
                ("id", "rvv.i32_m1.mask_policy.agnostic"),
                ("kind", "isa-vector-config"),
                (
                    "status",
                    status_from_bool(
                        facts["minimal_rvv_compile_run_succeeded"]
                        and facts["first_slice_mask_policy"] == "agnostic"
                    ),
                ),
                ("mask_policy", facts["first_slice_mask_policy"]),
            ],
            provider_symbols if emit_target_profile else None,
        )
    if facts["i32_m2_sew_bits"]:
        append_capability(
            lines,
            "rvv_i32_m2_sew32",
            [
                ("id", "rvv.i32_m2.sew32"),
                ("kind", "isa-vector-config"),
                (
                    "status",
                    status_from_bool(
                        facts["minimal_rvv_compile_run_succeeded"]
                        and facts["i32_m2_sew_bits"] == 32
                    ),
                ),
                ("sew_bits", facts["i32_m2_sew_bits"]),
            ],
            provider_symbols if emit_target_profile else None,
        )
    if facts["i32_m2_lmul"]:
        append_capability(
            lines,
            "rvv_i32_m2_lmul_m2",
            [
                ("id", "rvv.i32_m2.lmul_m2"),
                ("kind", "isa-vector-config"),
                (
                    "status",
                    status_from_bool(
                        facts["minimal_rvv_compile_run_succeeded"]
                        and facts["i32_m2_lmul"] == "m2"
                    ),
                ),
                ("lmul", facts["i32_m2_lmul"]),
            ],
            provider_symbols if emit_target_profile else None,
        )
    if facts["i32_m2_tail_policy"]:
        append_capability(
            lines,
            "rvv_i32_m2_tail_agnostic",
            [
                ("id", "rvv.i32_m2.tail_policy.agnostic"),
                ("kind", "isa-vector-config"),
                (
                    "status",
                    status_from_bool(
                        facts["minimal_rvv_compile_run_succeeded"]
                        and facts["i32_m2_tail_policy"] == "agnostic"
                    ),
                ),
                ("tail_policy", facts["i32_m2_tail_policy"]),
            ],
            provider_symbols if emit_target_profile else None,
        )
    if facts["i32_m2_mask_policy"]:
        append_capability(
            lines,
            "rvv_i32_m2_mask_agnostic",
            [
                ("id", "rvv.i32_m2.mask_policy.agnostic"),
                ("kind", "isa-vector-config"),
                (
                    "status",
                    status_from_bool(
                        facts["minimal_rvv_compile_run_succeeded"]
                        and facts["i32_m2_mask_policy"] == "agnostic"
                    ),
                ),
                ("mask_policy", facts["i32_m2_mask_policy"]),
            ],
            provider_symbols if emit_target_profile else None,
        )
    if facts["i64_m1_sew_bits"]:
        append_capability(
            lines,
            "rvv_i64_m1_sew64",
            [
                ("id", "rvv.i64_m1.sew64"),
                ("kind", "isa-vector-config"),
                (
                    "status",
                    status_from_bool(
                        facts["minimal_rvv_compile_run_succeeded"]
                        and facts["i64_m1_sew_bits"] == 64
                    ),
                ),
                ("sew_bits", facts["i64_m1_sew_bits"]),
            ],
            provider_symbols if emit_target_profile else None,
        )
    if facts["i64_m1_lmul"]:
        append_capability(
            lines,
            "rvv_i64_m1_lmul_m1",
            [
                ("id", "rvv.i64_m1.lmul_m1"),
                ("kind", "isa-vector-config"),
                (
                    "status",
                    status_from_bool(
                        facts["minimal_rvv_compile_run_succeeded"]
                        and facts["i64_m1_lmul"] == "m1"
                    ),
                ),
                ("lmul", facts["i64_m1_lmul"]),
            ],
            provider_symbols if emit_target_profile else None,
        )
    if facts["i64_m1_tail_policy"]:
        append_capability(
            lines,
            "rvv_i64_m1_tail_agnostic",
            [
                ("id", "rvv.i64_m1.tail_policy.agnostic"),
                ("kind", "isa-vector-config"),
                (
                    "status",
                    status_from_bool(
                        facts["minimal_rvv_compile_run_succeeded"]
                        and facts["i64_m1_tail_policy"] == "agnostic"
                    ),
                ),
                ("tail_policy", facts["i64_m1_tail_policy"]),
            ],
            provider_symbols if emit_target_profile else None,
        )
    if facts["i64_m1_mask_policy"]:
        append_capability(
            lines,
            "rvv_i64_m1_mask_agnostic",
            [
                ("id", "rvv.i64_m1.mask_policy.agnostic"),
                ("kind", "isa-vector-config"),
                (
                    "status",
                    status_from_bool(
                        facts["minimal_rvv_compile_run_succeeded"]
                        and facts["i64_m1_mask_policy"] == "agnostic"
                    ),
                ),
                ("mask_policy", facts["i64_m1_mask_policy"]),
            ],
            provider_symbols if emit_target_profile else None,
        )
    append_capability(
        lines,
        "rvv_toolchain_clang",
        [
            ("id", "rvv.toolchain.clang"),
            ("kind", "toolchain"),
            ("status", status_from_bool(facts["clang_available"])),
            ("version", facts["clang_version"]),
        ],
        provider_symbols if emit_target_profile else None,
    )
    append_capability(
        lines,
        "rvv_toolchain_cmake",
        [
            ("id", "rvv.toolchain.cmake"),
            ("kind", "toolchain"),
            ("status", status_from_bool(facts["cmake_available"])),
            ("version", facts["cmake_version"]),
        ],
        provider_symbols if emit_target_profile else None,
    )

    compile_run_attrs: list[tuple[str, str | int | list[str]]] = [
        ("id", "rvv.probe.compile_run"),
        ("kind", "toolchain"),
        ("status", compile_run_status),
    ]
    if facts["selected_march"]:
        compile_run_attrs.append(("selected_march", facts["selected_march"]))
    if facts["selected_mabi"]:
        compile_run_attrs.append(("selected_mabi", facts["selected_mabi"]))
    if facts["source_sha256"]:
        compile_run_attrs.append(("source_sha256", facts["source_sha256"]))
    if facts["binary_sha256"]:
        compile_run_attrs.append(("binary_sha256", facts["binary_sha256"]))
    append_capability(lines, "rvv_probe_compile_run", compile_run_attrs)
    if emit_target_profile:
        provider_symbols.append("rvv_probe_compile_run")

    if facts["selected_march"]:
        append_capability(
            lines,
            "rvv_toolchain_march",
            [
                ("id", "rvv.toolchain.march"),
                ("kind", "toolchain"),
                ("status", compile_run_status),
                ("value", facts["selected_march"]),
            ],
            provider_symbols if emit_target_profile else None,
        )
    if facts["selected_mabi"]:
        append_capability(
            lines,
            "rvv_toolchain_mabi",
            [
                ("id", "rvv.toolchain.mabi"),
                ("kind", "toolchain"),
                ("status", compile_run_status),
                ("value", facts["selected_mabi"]),
            ],
            provider_symbols if emit_target_profile else None,
        )

    if include_scalar_fallback:
        if scalar_fallback_status not in {"available", "unavailable", "disabled", "missing"}:
            raise ReplayError("scalar fallback status is not supported")
        append_capability(
            lines,
            "scalar_fallback",
            [
                ("id", "scalar.fallback"),
                ("kind", "fallback"),
                ("status", scalar_fallback_status),
            ],
            provider_symbols if emit_target_profile else None,
        )

    if emit_target_profile:
        provider_refs = ", ".join(f"@{symbol}" for symbol in provider_symbols)
        lines.append(f"  tcrv.exec.target @{target_profile_symbol} {{")
        lines.append("    id = \"rvv.profile.replay\",")
        lines.append("    kind = \"profile\",")
        lines.append("    status = \"available\",")
        lines.append(f"    capability_providers = [{provider_refs}]")
        lines.append("  }")
        append_kernel_header(
            lines,
            kernel_symbol,
            target_profile=target_profile_symbol,
        )

    lines.append("  }")
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_fixture(path: Path, artifact: dict[str, Any]) -> None:
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_artifact(**capability_overrides: Any) -> dict[str, Any]:
    capability_facts: dict[str, Any] = {
        "architecture": "riscv64",
        "hart_count": 64,
        "vlenb_bytes": 16,
        "i32_m1_lane_count": 4,
        "first_slice_sew_bits": 32,
        "first_slice_lmul": "m1",
        "first_slice_tail_policy": "agnostic",
        "first_slice_mask_policy": "agnostic",
        "i32_m2_sew_bits": 32,
        "i32_m2_lmul": "m2",
        "i32_m2_tail_policy": "agnostic",
        "i32_m2_mask_policy": "agnostic",
        "i64_m1_sew_bits": 64,
        "i64_m1_lmul": "m1",
        "i64_m1_tail_policy": "agnostic",
        "i64_m1_mask_policy": "agnostic",
        "isa_vector_hints": "isa: rv64imafdcv_zve64d_zvfh_zvl128b",
        "clang_available": True,
        "clang_version": "clang version 18.1.3",
        "cmake_available": True,
        "cmake_version": "cmake version 3.28.3",
        "minimal_rvv_compile_run_succeeded": True,
        "selected_march": "rv64gcv",
        "selected_mabi": "lp64d",
        "source_sha256": "0" * 64,
        "binary_sha256": "1" * 64,
    }
    capability_facts.update(capability_overrides)
    return {
        "schema_version": SCHEMA_VERSION,
        "probe_name": PROBE_NAME,
        "run_id": "self-test",
        "timestamp_utc": "2026-05-07T00:00:00Z",
        "ssh_target": "rvv",
        "artifact_dir": "artifacts/tmp/rvv_probe/self-test",
        "status": "success",
        "success": True,
        "capability_facts": capability_facts,
        "rvv_compile_run": {"status": "success"},
        "facts": {},
        "commands": [],
    }


def assert_self_test(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def run_self_test() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        fixture = Path(temp_dir) / "rvv_probe_evidence.json"
        write_fixture(fixture, make_artifact())
        artifact = load_json(fixture)
        mlir = emit_replay_mlir(
            artifact,
            kernel_name="rvv_probe_replay",
            include_scalar_fallback=True,
            scalar_fallback_status="available",
        )
        assert_self_test(
            'tcrv.exec.capability @rvv' in mlir and 'architecture = "riscv64"' in mlir,
            "RVV capability was not emitted",
        )
        assert_self_test(
            'tcrv.exec.capability @rvv_probe_compile_run' in mlir
            and 'selected_march = "rv64gcv"' in mlir,
            "compile/run capability was not emitted",
        )
        assert_self_test(
            'tcrv.exec.capability @rvv_hart_count' in mlir
            and 'provides = ["target.hart_count"]' in mlir
            and "count = 64 : i64" in mlir,
            "RVV hart count capability relation was not emitted",
        )
        assert_self_test(
            'tcrv.exec.capability @rvv_vlenb_bytes' in mlir
            and "bytes = 16 : i64" in mlir
            and 'tcrv.exec.capability @rvv_i32_m1_lanes' in mlir
            and "lanes = 4 : i64" in mlir,
            "RVV vector capacity capabilities were not emitted",
        )
        assert_self_test(
            'tcrv.exec.capability @rvv_i32_m1_sew32' in mlir
            and 'id = "rvv.i32_m1.sew32"' in mlir
            and "sew_bits = 32 : i64" in mlir
            and 'tcrv.exec.capability @rvv_i32_m1_lmul_m1' in mlir
            and 'lmul = "m1"' in mlir
            and 'tcrv.exec.capability @rvv_i32_m1_tail_agnostic' in mlir
            and 'tail_policy = "agnostic"' in mlir
            and 'tcrv.exec.capability @rvv_i32_m1_mask_agnostic' in mlir
            and 'mask_policy = "agnostic"' in mlir,
            "RVV first-slice config/policy capabilities were not emitted",
        )
        assert_self_test(
            'tcrv.exec.capability @rvv_i32_m2_sew32' in mlir
            and 'id = "rvv.i32_m2.sew32"' in mlir
            and "sew_bits = 32 : i64" in mlir
            and 'tcrv.exec.capability @rvv_i32_m2_lmul_m2' in mlir
            and 'lmul = "m2"' in mlir
            and 'tcrv.exec.capability @rvv_i32_m2_tail_agnostic' in mlir
            and 'tail_policy = "agnostic"' in mlir
            and 'tcrv.exec.capability @rvv_i32_m2_mask_agnostic' in mlir
            and 'mask_policy = "agnostic"' in mlir,
            "RVV i32m2 config/policy capabilities were not emitted",
        )
        assert_self_test(
            'tcrv.exec.capability @rvv_i64_m1_sew64' in mlir
            and 'id = "rvv.i64_m1.sew64"' in mlir
            and "sew_bits = 64 : i64" in mlir
            and 'tcrv.exec.capability @rvv_i64_m1_lmul_m1' in mlir
            and 'lmul = "m1"' in mlir
            and 'tcrv.exec.capability @rvv_i64_m1_tail_agnostic' in mlir
            and 'tail_policy = "agnostic"' in mlir
            and 'tcrv.exec.capability @rvv_i64_m1_mask_agnostic' in mlir
            and 'mask_policy = "agnostic"' in mlir,
            "RVV i64m1 config/policy capabilities were not emitted",
        )
        assert_self_test(
            'tcrv.exec.capability @scalar_fallback' in mlir,
            "explicit scalar fallback capability was not emitted",
        )

        target_profile_mlir = emit_replay_mlir(
            artifact,
            kernel_name="rvv_probe_i64_replay",
            include_scalar_fallback=False,
            scalar_fallback_status="available",
            emit_target_profile=True,
        )
        assert_self_test(
            "tcrv.exec.target @rvv_probe_i64_replay_profile" in target_profile_mlir
            and "capability_providers = [@rvv, @rvv_hart_count" in target_profile_mlir
            and "target = @rvv_probe_i64_replay_profile" in target_profile_mlir,
            "target-profile replay fixture was not emitted",
        )

        failed_compile = make_artifact(
            minimal_rvv_compile_run_succeeded=False,
            selected_march="",
            selected_mabi="",
        )
        mlir = emit_replay_mlir(
            failed_compile,
            kernel_name="failed_compile",
            include_scalar_fallback=True,
            scalar_fallback_status="available",
        )
        assert_self_test(
            'tcrv.exec.capability @rvv_probe_compile_run' in mlir
            and 'status = "unavailable"' in mlir
            and 'selected_march' not in mlir,
            "failed compile/run evidence was not preserved as unavailable",
        )

        try:
            emit_replay_mlir(
                make_artifact(clang_version="PASSWORD=hunter2"),
                kernel_name="secret",
                include_scalar_fallback=False,
                scalar_fallback_status="available",
            )
        except ReplayError as error:
            assert_self_test("secret-like" in str(error), "secret rejection message changed")
        else:
            raise AssertionError("secret-like capability fact was accepted")

        try:
            emit_replay_mlir(
                make_artifact(),
                kernel_name="1bad",
                include_scalar_fallback=False,
                scalar_fallback_status="available",
            )
        except ReplayError as error:
            assert_self_test("MLIR symbol" in str(error), "symbol rejection message changed")
        else:
            raise AssertionError("invalid MLIR symbol was accepted")

    print("rvv_probe_to_mlir self-test passed")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence_json", nargs="?")
    parser.add_argument("--kernel-name", default="")
    parser.add_argument("--include-scalar-fallback", action="store_true")
    parser.add_argument(
        "--scalar-fallback-status",
        default="available",
        choices=["available", "unavailable", "disabled", "missing"],
    )
    parser.add_argument("--emit-target-profile", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if args.self_test:
        run_self_test()
        return 0

    if not args.evidence_json:
        raise ReplayError("missing evidence JSON path")

    artifact = load_json(Path(args.evidence_json))
    sys.stdout.write(
        emit_replay_mlir(
            artifact,
            kernel_name=args.kernel_name or None,
            include_scalar_fallback=args.include_scalar_fallback,
            scalar_fallback_status=args.scalar_fallback_status,
            emit_target_profile=args.emit_target_profile,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main(sys.argv[1:]))
    except ReplayError as error:
        print(f"rvv_probe_to_mlir: {error}", file=sys.stderr)
        raise SystemExit(1)
